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


# ============= Anchor C (Parametric Bayes) =============

from inverse.parametric_bayes import (
    build_feature_matrix,
    evaluate_per_family_best_representative,
    evaluate_posteriors,
    loo_posteriors,
)


def test_parametric_bayes_recovers_canonical_families() -> None:
    """G1 PASS CRITERION (Anchor C, advisor amendment):
    Per-family-best-rep evaluation — at least 4/5 families have their best-trades
    representative agent receive posterior > 50% on correct family.
    Aggregate threshold also reported as secondary metric.
    """
    from tests.test_simulation_smoke import _RecordingLogger, _build_smoke_sim

    logger = _RecordingLogger()
    sim = _build_smoke_sim(seed=42, terminal_bars=1000, logger=logger)
    sim.run()

    per_agent = collect_per_agent_actions(logger.trades)
    family_lookup = {aid: sim.registry.get(aid).family for aid in sim.registry.alive_ids()}

    aids, features, families = build_feature_matrix(per_agent, family_lookup, min_trades=3)
    posteriors = loo_posteriors(features, families, classifier="logreg")

    # Aggregate evaluation
    agg = evaluate_posteriors(families, posteriors, threshold=0.5)
    # Per-family best-rep (advisor amendment)
    by_family = evaluate_per_family_best_representative(
        aids, families, posteriors, per_agent, threshold=0.5
    )

    print(f"\nParametric Bayes G1 evaluation (LogisticRegression + balanced):")
    print(f"  Aggregate: {agg['n_correct_above_threshold']}/{agg['n_agents']} "
          f"= {agg['fraction_above_threshold']:.1%} (lenient secondary)")
    print(f"  Per-family (correct/total):")
    for fam, (c, t) in agg["per_family_correct"].items():
        print(f"    {fam}: {c}/{t}")

    print(f"  Per-family-best-rep (PRIMARY criterion):")
    for fam, info in by_family["per_family_best"].items():
        verdict = "PASS" if info["passed"] else "FAIL"
        print(f"    {fam}: rep={info['agent_id']} (n={info['n_trades']}) "
              f"post={info['posterior_on_true']:.3f} → {verdict}")
    print(f"  {by_family['families_passed']}/{by_family['families_total']} families pass; "
          f"4/5 threshold: {'PASS' if by_family['passes_4_of_5'] else 'FAIL'}")

    # PRIMARY: 4/5 families per advisor amendment
    assert by_family["passes_4_of_5"], (
        f"Per-family-best-rep: only {by_family['families_passed']}/{by_family['families_total']} "
        f"families have rep agent with posterior > 50% on correct family (need >= 4/5)"
    )


def test_parametric_bayes_uniform_posterior_structure() -> None:
    """LOO posteriors should sum to ~1 per agent."""
    rng = np.random.default_rng(0)
    features = rng.standard_normal((20, 8))
    families = ["A", "A", "B", "B", "C"] * 4
    posteriors = loo_posteriors(features, families)
    sums = posteriors.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-9), f"Posterior sums: {sums}"


def test_build_feature_matrix_filters_correctly() -> None:
    per_agent = {
        "a1": [ActionRecord(0, "buy", "taker", 0.01, 100.0)] * 5,
        "a2": [ActionRecord(0, "buy", "taker", 0.01, 100.0)] * 2,  # filtered (< 3)
        "a3": [ActionRecord(0, "buy", "taker", 0.01, 100.0)] * 4,
        "a4": [ActionRecord(0, "buy", "taker", 0.01, 100.0)] * 5,  # not in family_lookup
    }
    family_lookup = {"a1": "fam_x", "a3": "fam_y"}
    aids, features, families = build_feature_matrix(per_agent, family_lookup, min_trades=3)
    assert aids == ["a1", "a3"]
    assert features.shape == (2, 8)
    assert families == ["fam_x", "fam_y"]


# ============= Anchor A (Sequential IRL) =============

from inverse.irl_maxent import (
    ACTIONS,
    N_ACTIONS,
    N_STATES,
    action_from_record,
    build_state_context,
    compute_null_baselines,
    discretize_state,
    evaluate_irl_per_agent,
)


def test_action_from_record_4_categories() -> None:
    assert action_from_record(ActionRecord(0, "buy", "taker", 0.01, 100.0)) == "buy_aggressive"
    assert action_from_record(ActionRecord(0, "buy", "maker", 0.01, 100.0)) == "buy_passive"
    assert action_from_record(ActionRecord(0, "sell", "maker", 0.01, 100.0)) == "sell_passive"
    assert action_from_record(ActionRecord(0, "sell", "taker", 0.01, 100.0)) == "sell_aggressive"


def test_discretize_state_corners() -> None:
    # imb high + trend up
    assert discretize_state(0.5, 0.001) == 2 * 3 + 0  # trend_up * 3 + imb_bid
    # imb low + trend down
    assert discretize_state(-0.5, -0.001) == 0 * 3 + 2
    # balanced + flat
    assert discretize_state(0.0, 0.0) == 1 * 3 + 1


def test_n_states_n_actions() -> None:
    assert N_STATES == 9
    assert N_ACTIONS == 4
    assert len(ACTIONS) == 4


def test_irl_recovers_canonical_policy_above_null() -> None:
    """G1 PASS CRITERION (Anchor A): IRL accuracy >= max(null) + 15pp on held-out trajectories."""
    from tests.test_simulation_smoke import _RecordingLogger, _build_smoke_sim

    logger = _RecordingLogger()
    sim = _build_smoke_sim(seed=42, terminal_bars=1000, logger=logger)
    sim.run()

    per_agent = collect_per_agent_actions(logger.trades)
    state_contexts = build_state_context(logger.bar_snapshots, trend_lookback_bars=5)

    # IRL evaluation
    irl_result = evaluate_irl_per_agent(per_agent, state_contexts, min_trades=5, train_frac=0.8, seed=42)
    # Null baselines on SAME train/test split
    null = compute_null_baselines(per_agent, state_contexts, min_trades=5, train_frac=0.8, seed=42)

    max_null = max(null["modal_mean"], null["last_action_mean"], null["uniform_random"])
    threshold = max_null + 0.15

    print(f"\nIRL G1 evaluation (behavioral cloning baseline):")
    print(f"  Eligible agents: {irl_result['n_eligible_agents']}")
    print(f"  IRL mean accuracy: {irl_result['mean_accuracy']:.3f}")
    print(f"  Null baselines:")
    print(f"    modal-action mean: {null['modal_mean']:.3f}")
    print(f"    last-action mean:  {null['last_action_mean']:.3f}")
    print(f"    uniform random:    {null['uniform_random']:.3f}")
    print(f"  max(null) + 0.15: {threshold:.3f}")
    print(f"  Verdict: {'PASS' if irl_result['mean_accuracy'] >= threshold else 'FAIL'}")

    assert irl_result["mean_accuracy"] >= threshold, (
        f"IRL mean accuracy {irl_result['mean_accuracy']:.3f} below "
        f"max(null)+0.15 = {threshold:.3f}"
    )
