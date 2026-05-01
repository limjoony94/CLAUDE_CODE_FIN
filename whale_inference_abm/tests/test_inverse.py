"""Inverse anchor tests (G1).

Coverage:
- trajectory_collector: per-agent action grouping + sorting + filtering
- null_baselines: random ARI distribution stable, modal/last/uniform formulas correct
- signature_clustering: feature extraction shape, K-means runs deterministic,
  ARI on canonical 5-agent ABM beats random+0.4 (G1 pass criterion)
"""

from __future__ import annotations

import numpy as np
import pytest

from inverse.null_baselines import (
    last_action_accuracy,
    modal_action_accuracy,
    random_clustering_ari_baseline,
    uniform_prior_posterior_mass,
    uniform_random_accuracy,
)
from inverse.signature_clustering import (
    FEATURE_DIM,
    cluster_agents,
    compute_features,
    evaluate_ari,
)
from inverse.trajectory_collector import (
    ActionRecord,
    collect_per_agent_actions,
    filter_by_min_trades,
    trade_count_per_agent,
)


# ============= trajectory_collector =============

def _make_trade(buyer: str, seller: str, ts: int = 0, price: float = 100.0, size: float = 0.01,
                buyer_role: str = "taker", seller_role: str = "maker", tid: str = "t1"):
    from abm.types import Role, Trade
    return Trade(
        trade_id=tid,
        timestamp_ns=ts,
        sequence_no=1,
        buyer_agent_id=buyer,
        seller_agent_id=seller,
        buyer_order_id=f"{buyer}_o",
        seller_order_id=f"{seller}_o",
        price=price,
        size=size,
        buyer_role=Role(buyer_role),
        seller_role=Role(seller_role),
    )


def test_collect_groups_trades_by_agent() -> None:
    trades = [_make_trade("a1", "a2"), _make_trade("a1", "a3", ts=100), _make_trade("a2", "a3", ts=200)]
    per = collect_per_agent_actions(trades)
    assert len(per["a1"]) == 2
    assert len(per["a2"]) == 2
    assert len(per["a3"]) == 2


def test_collect_each_trade_produces_two_legs() -> None:
    trades = [_make_trade("a1", "a2")]
    per = collect_per_agent_actions(trades)
    a1_action = per["a1"][0]
    a2_action = per["a2"][0]
    assert a1_action.side == "buy"
    assert a1_action.role == "taker"
    assert a2_action.side == "sell"
    assert a2_action.role == "maker"


def test_collect_sorts_by_timestamp() -> None:
    trades = [
        _make_trade("a1", "x", ts=300, tid="t1"),
        _make_trade("a1", "x", ts=100, tid="t2"),
        _make_trade("a1", "x", ts=200, tid="t3"),
    ]
    per = collect_per_agent_actions(trades)
    timestamps = [a.timestamp_ns for a in per["a1"]]
    assert timestamps == [100, 200, 300]


def test_filter_by_min_trades() -> None:
    per = {"a1": [None] * 5, "a2": [None] * 2, "a3": [None] * 10}  # type: ignore
    filtered = filter_by_min_trades(per, min_trades=3)
    assert set(filtered.keys()) == {"a1", "a3"}


# ============= null_baselines =============

def test_modal_action_majority() -> None:
    assert modal_action_accuracy(["A", "A", "A", "B"]) == 0.75


def test_modal_action_empty() -> None:
    assert modal_action_accuracy([]) == 0.0


def test_last_action_persistence() -> None:
    # AABBA: pairs (A,A), (A,B), (B,B), (B,A) -> matches: 2/4 = 0.5
    assert last_action_accuracy(["A", "A", "B", "B", "A"]) == 0.5


def test_last_action_too_short() -> None:
    assert last_action_accuracy(["A"]) == 0.0


def test_uniform_random_accuracy() -> None:
    assert uniform_random_accuracy(5) == 0.2
    assert uniform_random_accuracy(0) == 0.0


def test_uniform_prior_posterior_mass() -> None:
    assert uniform_prior_posterior_mass(5) == 0.2


def test_random_ari_baseline_near_zero() -> None:
    """Random clustering of true 5-class labels should give ARI close to 0."""
    rng = np.random.default_rng(0)
    true_labels = list(rng.integers(0, 5, size=20))
    null = random_clustering_ari_baseline(true_labels, K=5, n_trials=200, seed=42)
    assert -0.1 <= null["mean"] <= 0.1, f"Random ARI mean unexpected: {null['mean']}"


def test_random_ari_baseline_deterministic() -> None:
    true = ["A", "B", "C", "A", "B", "C"] * 3
    n1 = random_clustering_ari_baseline(true, K=3, n_trials=50, seed=42)
    n2 = random_clustering_ari_baseline(true, K=3, n_trials=50, seed=42)
    assert n1["samples"] == n2["samples"]


# ============= signature_clustering — feature extraction =============

def test_feature_vector_dim() -> None:
    actions = [
        ActionRecord(timestamp_ns=0, side="buy", role="taker", size=0.01, price=100.0),
        ActionRecord(timestamp_ns=100, side="sell", role="maker", size=0.02, price=101.0),
        ActionRecord(timestamp_ns=200, side="buy", role="taker", size=0.015, price=100.5),
    ]
    f = compute_features(actions)
    assert f.shape == (FEATURE_DIM,)


def test_feature_taker_fraction_correct() -> None:
    actions = [
        ActionRecord(0, "buy", "taker", 0.01, 100.0),
        ActionRecord(100, "buy", "taker", 0.01, 100.0),
        ActionRecord(200, "buy", "maker", 0.01, 100.0),
        ActionRecord(300, "buy", "maker", 0.01, 100.0),
    ]
    f = compute_features(actions)
    # 6th index (after 5 quantiles + mean_inter): taker_frac = 2/4 = 0.5
    assert f[6] == pytest.approx(0.5)


def test_feature_lag1_corr_alternating() -> None:
    # Alternating buy/sell should produce strong negative autocorr ~ -1
    actions = [
        ActionRecord(i * 100, side, "taker", 0.01, 100.0)
        for i, side in enumerate(["buy", "sell"] * 5)
    ]
    f = compute_features(actions)
    corr = f[7]  # last feature
    assert corr < -0.5


def test_feature_lag1_corr_persistent() -> None:
    # All same direction = nan corr handled as 0
    actions = [ActionRecord(i * 100, "buy", "taker", 0.01, 100.0) for i in range(5)]
    f = compute_features(actions)
    corr = f[7]
    assert corr == 0.0


# ============= signature_clustering — clustering =============

def test_cluster_agents_returns_correct_shapes() -> None:
    rng = np.random.default_rng(0)
    per_agent = {
        f"agent_{i:02d}": [
            ActionRecord(t * 1000, "buy" if rng.random() > 0.5 else "sell",
                         "taker" if rng.random() > 0.5 else "maker",
                         float(rng.uniform(0.01, 0.1)), 100.0)
            for t in range(20)
        ]
        for i in range(10)
    }
    aids, labels, features = cluster_agents(per_agent, K=3, min_trades=3)
    assert len(aids) == 10
    assert labels.shape == (10,)
    assert features.shape == (10, FEATURE_DIM)


def test_cluster_too_few_eligible_raises() -> None:
    per_agent = {"a1": [ActionRecord(0, "buy", "taker", 0.01, 100.0)]}
    with pytest.raises(ValueError, match="at least K=5"):
        cluster_agents(per_agent, K=5, min_trades=3)


def test_cluster_deterministic_same_seed() -> None:
    rng = np.random.default_rng(0)
    per_agent = {
        f"a{i:02d}": [
            ActionRecord(t * 1000, "buy" if t % 2 == 0 else "sell",
                         "taker", float(rng.uniform(0.01, 0.1)), 100.0)
            for t in range(20)
        ]
        for i in range(10)
    }
    _, labels1, _ = cluster_agents(per_agent, K=3, random_state=42)
    _, labels2, _ = cluster_agents(per_agent, K=3, random_state=42)
    assert (labels1 == labels2).all()


# ============= G1 acceptance gate: ARI vs canonical ABM =============

def test_signature_recovers_canonical_families_above_null_baseline() -> None:
    """G1 PASS CRITERION: signature ARI >= random_baseline + 0.4 on canonical 5-agent ABM."""
    from tests.test_simulation_smoke import _RecordingLogger, _build_smoke_sim

    logger = _RecordingLogger()
    sim = _build_smoke_sim(seed=42, terminal_bars=1000, logger=logger)
    sim.run()

    per_agent = collect_per_agent_actions(logger.trades)
    family_lookup = {aid: sim.registry.get(aid).family for aid in sim.registry.alive_ids()}

    aids, predicted, _ = cluster_agents(per_agent, K=5, min_trades=3)
    ari, true_labels = evaluate_ari(aids, predicted, family_lookup)

    # Random baseline ARI
    null = random_clustering_ari_baseline(true_labels, K=5, n_trials=200, seed=42)
    threshold = null["mean"] + 0.4

    print(f"\nSignature G1 evaluation:")
    print(f"  Eligible agents: {len(aids)}")
    print(f"  ARI: {ari:.3f}")
    print(f"  Random null mean: {null['mean']:.3f}, p95: {null['p95']:.3f}")
    print(f"  Threshold (null + 0.4): {threshold:.3f}")
    print(f"  Verdict: {'PASS' if ari >= threshold else 'FAIL'}")

    # Per-family breakdown
    from collections import Counter
    family_in_clusters: dict[str, Counter] = {}
    for aid, lbl in zip(aids, predicted):
        if aid not in family_lookup:
            continue
        fam = family_lookup[aid]
        family_in_clusters.setdefault(fam, Counter())[int(lbl)] += 1
    print(f"  Per-family cluster distribution:")
    for fam, counts in sorted(family_in_clusters.items()):
        print(f"    {fam}: {dict(counts)}")

    assert ari >= threshold, (
        f"Signature ARI {ari:.3f} below null+0.4 = {threshold:.3f}. "
        f"5-family canonical recovery insufficient."
    )
