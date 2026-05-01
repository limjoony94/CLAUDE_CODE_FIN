"""Wealth-concentration metric tests + G2 integration."""

from __future__ import annotations

import numpy as np
import pytest

from abm.metrics import evaluate_concentration, gini, top_k_overlap, top_k_share


# ----- gini -----

def test_gini_perfect_equality_zero() -> None:
    assert gini([100, 100, 100, 100, 100]) == pytest.approx(0.0, abs=1e-9)


def test_gini_max_inequality_approaches_one() -> None:
    # One agent owns everything, others 0
    assert gini([0, 0, 0, 0, 100]) == pytest.approx(0.8, abs=0.01)
    # As n grows, perfect-inequality gini → 1
    n = 1000
    arr = [0] * (n - 1) + [1000]
    assert gini(arr) > 0.99


def test_gini_empty_zero() -> None:
    assert gini([]) == 0.0


def test_gini_all_zero_zero() -> None:
    assert gini([0, 0, 0]) == 0.0


def test_gini_negative_clipped() -> None:
    # Negative wealth (bankrupt MTM) clipped to 0
    assert gini([100, 100, 100, -50]) == gini([100, 100, 100, 0])


def test_gini_known_value() -> None:
    # Known: [1,2,3,4,5] has gini 0.2667
    assert gini([1, 2, 3, 4, 5]) == pytest.approx(0.267, abs=0.01)


# ----- top_k_share -----

def test_top_k_share_uniform() -> None:
    # 100 agents each with 100 wealth, top 5% = 5 agents = 5% of total
    arr = [100] * 100
    assert top_k_share(arr, k_pct=0.05) == pytest.approx(0.05)


def test_top_k_share_concentrated() -> None:
    # Top 5 agents have 1000 each, others 1 each — top-5% (= 5 agents) holds nearly all
    arr = [1000] * 5 + [1] * 95
    share = top_k_share(arr, k_pct=0.05)
    assert share > 0.9


def test_top_k_share_empty() -> None:
    assert top_k_share([], k_pct=0.05) == 0.0


def test_top_k_share_min_one_agent() -> None:
    # k_pct=0.05 of 5 agents = 0.25 → max(1, int(0.25)) = 1
    arr = [10, 5, 3, 2, 1]
    # Top 1 of 5 = 10/21 ≈ 0.476
    assert top_k_share(arr, k_pct=0.05) == pytest.approx(10 / 21, abs=0.01)


# ----- top_k_overlap -----

def test_top_k_overlap_identical_perfect() -> None:
    snap = {f"a{i}": float(i) for i in range(100)}
    assert top_k_overlap(snap, snap, k_pct=0.05) == 1.0


def test_top_k_overlap_disjoint_zero() -> None:
    snap_old = {f"a{i}": float(i) for i in range(100)}
    snap_new = {f"b{i}": float(i) for i in range(100)}  # different agents
    assert top_k_overlap(snap_old, snap_new, k_pct=0.05) == 0.0


def test_top_k_overlap_partial() -> None:
    # Old: top-5% are a0..a4 (richest)
    snap_old = {**{f"a{i}": float(100 + i) for i in range(5)},
                **{f"b{i}": float(i) for i in range(95)}}
    # New: top-5% are a0..a2 + new agents
    snap_new = {**{f"a{i}": float(100 + i) for i in range(3)},
                **{f"c{i}": float(200 + i) for i in range(2)},
                **{f"b{i}": float(i) for i in range(95)}}
    overlap = top_k_overlap(snap_old, snap_new, k_pct=0.05)
    # 3 of 5 overlap (a0..a2)
    assert overlap == pytest.approx(0.6, abs=0.01)


def test_top_k_overlap_empty_returns_zero() -> None:
    assert top_k_overlap({}, {"a1": 100}, k_pct=0.05) == 0.0


# ----- evaluate_concentration -----

def test_evaluate_concentration_basic() -> None:
    bar_ns = 60 * 10**9
    history = [
        (i * bar_ns, {f"a{j}": 100.0 + j * (i / 100) for j in range(20)})
        for i in range(100)
    ]
    # By bar 99, wealth disparity grows linearly
    result = evaluate_concentration(
        history,
        bar_indices_to_compare=(50, 99),
        k_pct=0.10,
        bar_duration_ns=bar_ns,
    )
    assert result["n_agents_late"] == 20
    assert 0.0 <= result["gini_at_late"] <= 1.0
    assert 0.0 <= result["top_k_share_at_late"] <= 1.0
    assert 0.0 <= result["top_k_overlap_early_late"] <= 1.0


# ============= G2 integration: 10k bar wealth concentration validity =============

@pytest.mark.slow
def test_g2_wealth_concentration_at_10k_bars() -> None:
    """G2 PASS CRITERION: gini > 0.5, top-5% rank stability >= 50% between T=5k and T=10k.

    Uses 10k-bar smoke (post-perf-cache, ~6.6 min runtime).
    Marked @slow to exclude from default suite.
    """
    from tests.test_simulation_smoke import _build_smoke_sim

    sim = _build_smoke_sim(seed=42, terminal_bars=10000)
    sim.run()

    history = sim.wealth_tracker._history  # list of (ts, dict[agent_id, wealth])
    result = evaluate_concentration(
        history,
        bar_indices_to_compare=(5000, 10000),
        k_pct=0.05,
    )

    print(f"\nG2 wealth concentration at 10k bars:")
    print(f"  Agents alive at T=10k: {result['n_agents_late']}")
    print(f"  Gini at T=10k: {result['gini_at_late']:.3f}")
    print(f"  Top-5% share at T=10k: {result['top_k_share_at_late']:.3f}")
    print(f"  Top-5% overlap T=5k vs T=10k: {result['top_k_overlap_early_late']:.3f}")
    print(f"  Pass criteria:")
    print(f"    gini > 0.5: {'PASS' if result['gini_at_late'] > 0.5 else 'FAIL'}")
    print(f"    top-5% overlap >= 0.5: {'PASS' if result['top_k_overlap_early_late'] >= 0.5 else 'FAIL'}")

    # Abandon trigger check
    assert result["gini_at_late"] >= 0.3, (
        f"Gini {result['gini_at_late']:.3f} < 0.3 — market mechanics fail to concentrate, "
        f"abandon trigger fired"
    )
    # Pass criteria
    assert result["gini_at_late"] > 0.5, (
        f"Gini {result['gini_at_late']:.3f} <= 0.5 — concentration insufficient for G2"
    )
    assert result["top_k_overlap_early_late"] >= 0.5, (
        f"Top-5% rank stability {result['top_k_overlap_early_late']:.3f} < 50% — no stable whales"
    )
