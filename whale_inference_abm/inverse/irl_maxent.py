"""Sequential IRL anchor (A) — primary inverse-recovery via state-conditional policy estimation.

Design ref: docs/02-design/features/whale_inference_abm.design.md Section 3.1 (MVP A).

MVP scope: behavioral-cloning-style policy estimation as starting point.
- State space: 9 = orderbook imbalance (3 bins) × trend regime (3 bins)
- Action space: 4 = {buy_aggressive, buy_passive, sell_passive, sell_aggressive}
  (Skipping HOLD for v1 MVP — derived only from trade tape; no all-decisions log needed.
   Documented deviation from design Section 3.1's "5 actions including hold".)
- Per-agent training: estimate P(action | state) empirically on 80% trajectories
- Per-agent test: predict argmax P(a|state) for 20% held-out, compute accuracy

True MaxEnt IRL (recover reward function via gradient descent on feature expectations)
is v2 path. MVP behavioral cloning is sufficient if it beats the null baselines:
  modal-action, last-action-copy, uniform random (=0.25).

Pass criterion (advisor binding decision #1):
  mean_accuracy >= max(null_baselines) + 0.15
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np

from inverse.null_baselines import last_action_accuracy, modal_action_accuracy
from inverse.trajectory_collector import ActionRecord, filter_by_min_trades


# ----- State + action discretization -----

ACTION_BUY_AGG = "buy_aggressive"
ACTION_BUY_PASS = "buy_passive"
ACTION_SELL_PASS = "sell_passive"
ACTION_SELL_AGG = "sell_aggressive"
ACTIONS = (ACTION_BUY_AGG, ACTION_BUY_PASS, ACTION_SELL_PASS, ACTION_SELL_AGG)
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}
N_ACTIONS = len(ACTIONS)

STATE_TREND_DOWN = 0
STATE_TREND_FLAT = 1
STATE_TREND_UP = 2
STATE_IMB_BID = 0  # bid-heavy (imbalance > 0.2)
STATE_IMB_BAL = 1  # balanced (-0.2 < imb < 0.2)
STATE_IMB_ASK = 2  # ask-heavy (imbalance < -0.2)
N_STATES = 9  # 3 trend × 3 imbalance


def action_from_record(rec: ActionRecord) -> str:
    """Map (side, role) → 4-action discretization."""
    if rec.side == "buy" and rec.role == "taker":
        return ACTION_BUY_AGG
    if rec.side == "buy" and rec.role == "maker":
        return ACTION_BUY_PASS
    if rec.side == "sell" and rec.role == "maker":
        return ACTION_SELL_PASS
    if rec.side == "sell" and rec.role == "taker":
        return ACTION_SELL_AGG
    raise ValueError(f"Cannot discretize: side={rec.side}, role={rec.role}")


def discretize_state(
    imbalance: float,
    mid_change_pct: float,
    thresholds: Optional[dict[str, float]] = None,
) -> int:
    """Map (orderbook imbalance, recent mid change) → 0..8 state index.

    If thresholds is provided (advisor-recommended empirical terciles), use them.
    Otherwise fall back to hard-coded boundaries (legacy behavior).

    State index = trend * 3 + imbalance_bin
    """
    if thresholds is not None:
        imb_p33 = thresholds["imb_p33"]
        imb_p66 = thresholds["imb_p66"]
        trend_p33 = thresholds["trend_p33"]
        trend_p66 = thresholds["trend_p66"]
    else:
        # Legacy hard-coded
        imb_p33, imb_p66 = -0.2, 0.2
        trend_p33, trend_p66 = -0.0005, 0.0005

    if imbalance > imb_p66:
        imb = STATE_IMB_BID
    elif imbalance < imb_p33:
        imb = STATE_IMB_ASK
    else:
        imb = STATE_IMB_BAL

    if mid_change_pct > trend_p66:
        trend = STATE_TREND_UP
    elif mid_change_pct < trend_p33:
        trend = STATE_TREND_DOWN
    else:
        trend = STATE_TREND_FLAT

    return trend * 3 + imb


def fit_state_thresholds(bar_snapshots: list, trend_lookback_bars: int = 5) -> dict[str, float]:
    """Fit empirical 33/66 percentile thresholds for trend and imbalance.

    Per advisor (G1 IRL diagnostic 2026-05-01): hard-coded ±0.05% trend / ±0.2 imbalance
    yielded 84.5% of pairs in single state. Empirical terciles guarantee balanced bins.

    Returns dict with keys: trend_p33, trend_p66, imb_p33, imb_p66
    """
    trend_pcts = []
    imbalances = []
    mids = []
    for snap, _ in bar_snapshots:
        if snap.mid_price is None:
            mids.append(None)
            continue
        mids.append(snap.mid_price)
        if len(mids) > trend_lookback_bars:
            old_mid = mids[-trend_lookback_bars - 1]
            if old_mid is not None and old_mid > 0:
                trend_pcts.append((snap.mid_price - old_mid) / old_mid)
        bid_qty = sum(q for _, q in (snap.bid_depth or []))
        ask_qty = sum(q for _, q in (snap.ask_depth or []))
        denom = bid_qty + ask_qty
        if denom > 0:
            imbalances.append((bid_qty - ask_qty) / denom)

    if not trend_pcts or not imbalances:
        # Fall back to defaults
        return {"trend_p33": -0.0005, "trend_p66": 0.0005, "imb_p33": -0.2, "imb_p66": 0.2}

    return {
        "trend_p33": float(np.quantile(trend_pcts, 0.333)),
        "trend_p66": float(np.quantile(trend_pcts, 0.667)),
        "imb_p33": float(np.quantile(imbalances, 0.333)),
        "imb_p66": float(np.quantile(imbalances, 0.667)),
    }


# ----- State context reconstruction from bar snapshots -----

def build_state_context(
    bar_snapshots: list,
    trend_lookback_bars: int = 5,
    use_empirical_terciles: bool = True,
) -> list:
    """For each bar snapshot, compute (timestamp_ns, state_idx).

    Args:
        bar_snapshots: list of (snapshot, wealth_dist) tuples
        trend_lookback_bars: how many bars back to compute mid_change_pct
        use_empirical_terciles: if True (default), fit thresholds from data
                                (advisor recommended after IRL diagnostic)
    """
    if use_empirical_terciles:
        thresholds = fit_state_thresholds(bar_snapshots, trend_lookback_bars)
    else:
        thresholds = None

    contexts = []
    mids = []
    for i, (snap, _) in enumerate(bar_snapshots):
        if snap.mid_price is None:
            mids.append(None)
            continue
        mids.append(snap.mid_price)

        if len(mids) <= trend_lookback_bars:
            mid_change_pct = 0.0
        else:
            old_mid = mids[-trend_lookback_bars - 1]
            if old_mid is None or old_mid <= 0:
                mid_change_pct = 0.0
            else:
                mid_change_pct = (snap.mid_price - old_mid) / old_mid

        bid_qty = sum(qty for _, qty in (snap.bid_depth or []))
        ask_qty = sum(qty for _, qty in (snap.ask_depth or []))
        denom = bid_qty + ask_qty
        if denom <= 0:
            imbalance = 0.0
        else:
            imbalance = (bid_qty - ask_qty) / denom

        state = discretize_state(imbalance, mid_change_pct, thresholds)
        contexts.append((snap.timestamp_ns, state))

    return contexts


def state_at_timestamp(contexts: list, timestamp_ns: int) -> Optional[int]:
    """Find state at most-recent context AT OR BEFORE timestamp_ns (causal)."""
    if not contexts:
        return None
    # Linear scan; small dataset so OK. Could binary-search if needed.
    last_state = None
    for ts, st in contexts:
        if ts <= timestamp_ns:
            last_state = st
        else:
            break
    return last_state


# ----- Per-agent policy estimation -----

def collect_state_action_pairs(
    actions: list[ActionRecord], state_contexts: list
) -> list[tuple[int, int]]:
    """For each ActionRecord, find concurrent state and produce (state_idx, action_idx) pair."""
    pairs = []
    for rec in actions:
        st = state_at_timestamp(state_contexts, rec.timestamp_ns)
        if st is None:
            continue
        try:
            act = ACTION_TO_IDX[action_from_record(rec)]
        except ValueError:
            continue
        pairs.append((st, act))
    return pairs


def fit_policy(pairs: list[tuple[int, int]], laplace_alpha: float = 1.0) -> np.ndarray:
    """Estimate P(action | state) via histogram + Laplace smoothing.

    Returns:
        policy: shape (N_STATES, N_ACTIONS), each row sums to 1.
    """
    counts = np.full((N_STATES, N_ACTIONS), laplace_alpha, dtype=float)
    for st, act in pairs:
        counts[st, act] += 1.0
    policy = counts / counts.sum(axis=1, keepdims=True)
    return policy


def predict_actions(policy: np.ndarray, states: list[int]) -> list[int]:
    """argmax P(a|s) for each state in states."""
    return [int(np.argmax(policy[s])) for s in states]


def accuracy(predicted: list[int], actual: list[int]) -> float:
    if not actual:
        return 0.0
    correct = sum(1 for p, a in zip(predicted, actual) if p == a)
    return correct / len(actual)


def per_agent_train_test_split(
    pairs: list[tuple[int, int]], train_frac: float = 0.8, seed: int = 42
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Deterministic random split. Pairs sorted before split for cross-process reproducibility."""
    if len(pairs) < 5:
        return [], []  # too few to split meaningfully
    rng = np.random.default_rng(seed)
    indices = np.arange(len(pairs))
    rng.shuffle(indices)
    n_train = int(len(pairs) * train_frac)
    train = [pairs[i] for i in indices[:n_train]]
    test = [pairs[i] for i in indices[n_train:]]
    return train, test


def evaluate_irl_per_agent(
    per_agent: dict[str, list[ActionRecord]],
    state_contexts: list,
    min_trades: int = 5,
    train_frac: float = 0.8,
    seed: int = 42,
) -> dict:
    """Per-agent train/test policy evaluation.

    Returns dict with:
      - per_agent_accuracy: dict[agent_id, accuracy]
      - mean_accuracy
      - n_eligible_agents
      - per_agent_n_test_pairs
    """
    eligible = filter_by_min_trades(per_agent, min_trades)
    per_agent_acc: dict[str, float] = {}
    per_agent_n_test: dict[str, int] = {}

    for aid in sorted(eligible.keys()):
        pairs = collect_state_action_pairs(eligible[aid], state_contexts)
        train, test = per_agent_train_test_split(pairs, train_frac=train_frac, seed=seed)
        if not train or not test:
            continue
        policy = fit_policy(train)
        test_states = [s for s, _ in test]
        test_actions = [a for _, a in test]
        predicted = predict_actions(policy, test_states)
        per_agent_acc[aid] = accuracy(predicted, test_actions)
        per_agent_n_test[aid] = len(test)

    mean_acc = float(np.mean(list(per_agent_acc.values()))) if per_agent_acc else 0.0
    return {
        "per_agent_accuracy": per_agent_acc,
        "mean_accuracy": mean_acc,
        "n_eligible_agents": len(per_agent_acc),
        "per_agent_n_test_pairs": per_agent_n_test,
    }


def compute_null_baselines(
    per_agent: dict[str, list[ActionRecord]],
    state_contexts: list,
    min_trades: int = 5,
    train_frac: float = 0.8,
    seed: int = 42,
) -> dict:
    """Compute modal-action and last-action baselines on the SAME train/test split per agent."""
    eligible = filter_by_min_trades(per_agent, min_trades)
    modal_accs: list[float] = []
    last_accs: list[float] = []

    for aid in sorted(eligible.keys()):
        pairs = collect_state_action_pairs(eligible[aid], state_contexts)
        train, test = per_agent_train_test_split(pairs, train_frac=train_frac, seed=seed)
        if not train or not test:
            continue
        train_actions = [a for _, a in train]
        test_actions = [a for _, a in test]

        # modal: predict most-common training action for all test
        from collections import Counter
        if train_actions:
            modal = Counter(train_actions).most_common(1)[0][0]
            modal_acc = sum(1 for a in test_actions if a == modal) / len(test_actions)
            modal_accs.append(modal_acc)

        # last-action: predict prev test-action; first test predicts last-train
        sequence = train_actions + test_actions
        n_train = len(train_actions)
        correct = 0
        for i in range(len(test_actions)):
            prev = sequence[n_train + i - 1]
            if prev == test_actions[i]:
                correct += 1
        if test_actions:
            last_accs.append(correct / len(test_actions))

    return {
        "modal_mean": float(np.mean(modal_accs)) if modal_accs else 0.0,
        "last_action_mean": float(np.mean(last_accs)) if last_accs else 0.0,
        "uniform_random": 1.0 / N_ACTIONS,
        "n_agents": len(modal_accs),
    }
