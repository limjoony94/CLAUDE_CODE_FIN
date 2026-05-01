"""Wealth-concentration metrics for G2 evaluation.

Design ref: docs/02-design/features/whale_inference_abm.design.md Section 2 (Gate G2).

Metrics:
  - gini(wealths): standard Gini coefficient (0=equal, 1=max inequality)
  - top_k_share(wealths, k_pct): fraction of total wealth held by top k_pct of agents
  - top_k_overlap(snap_old, snap_new, k_pct): rank stability across snapshots

G2 pass criteria:
  - gini > max(0.5, empirical_target - 0.1) at T=10000 sim-bars
  - top-5% rank stability >= 50% between T=5000 and T=10000
  - top-5% emerges from initially uniform distribution

G2 abandon trigger:
  - gini < 0.3 at T=10000 (market mechanics fail to concentrate)
  - top-5% turnover > 80% (no stable whales = inverse problem ill-defined)
"""

from __future__ import annotations

import numpy as np


def gini(wealths) -> float:
    """Standard Gini coefficient of wealth distribution.

    Returns 0 for perfectly equal, approaches 1 for maximum inequality (one agent owns all).
    Negative-wealth agents are clipped to 0 (per design: bankrupts stay in MTM but don't
    contribute negative inequality).
    """
    arr = np.asarray(wealths, dtype=float)
    if len(arr) == 0:
        return 0.0
    arr = np.clip(arr, 0.0, None)
    if arr.sum() <= 0:
        return 0.0
    sorted_w = np.sort(arr)
    n = len(sorted_w)
    # Standard formula: (2 * sum(i * x_i) - (n+1) * sum(x)) / (n * sum(x))
    indices = np.arange(1, n + 1)
    return float((2 * np.sum(indices * sorted_w) - (n + 1) * sorted_w.sum()) / (n * sorted_w.sum()))


def top_k_share(wealths, k_pct: float = 0.05) -> float:
    """Fraction of total wealth held by top k_pct of agents.

    For k_pct=0.05 with 100 agents, takes top 5 agents by wealth and returns their
    cumulative share. Negative wealth clipped to 0.
    """
    arr = np.asarray(wealths, dtype=float)
    if len(arr) == 0:
        return 0.0
    arr = np.clip(arr, 0.0, None)
    total = arr.sum()
    if total <= 0:
        return 0.0
    k = max(1, int(len(arr) * k_pct))
    sorted_desc = np.sort(arr)[::-1]
    return float(sorted_desc[:k].sum() / total)


def top_k_overlap(
    snap_old: dict[str, float],
    snap_new: dict[str, float],
    k_pct: float = 0.05,
) -> float:
    """Fraction overlap between top-k_pct agents in snap_old and snap_new.

    Returns intersection_size / max(|top_k_old|, |top_k_new|). Higher = more rank-stable.
    Bankrupts (missing from snap_new) implicitly out of top-k_new.
    """
    if not snap_old or not snap_new:
        return 0.0
    sorted_old = sorted(snap_old.items(), key=lambda x: x[1], reverse=True)
    sorted_new = sorted(snap_new.items(), key=lambda x: x[1], reverse=True)
    k_old = max(1, int(len(sorted_old) * k_pct))
    k_new = max(1, int(len(sorted_new) * k_pct))
    top_old = {aid for aid, _ in sorted_old[:k_old]}
    top_new = {aid for aid, _ in sorted_new[:k_new]}
    intersection = top_old & top_new
    return len(intersection) / max(len(top_old), len(top_new))


def evaluate_concentration(
    wealth_history: list[tuple[int, dict[str, float]]],
    bar_indices_to_compare: tuple[int, int] = (5000, 10000),
    k_pct: float = 0.05,
    bar_duration_ns: int = 60 * 10**9,
    min_wealth_to_include: float = 0.0,
) -> dict:
    """Evaluate wealth concentration at two checkpoints.

    Args:
        wealth_history: list of (timestamp_ns, dict[agent_id, wealth]) per bar snapshot
        bar_indices_to_compare: (early_bar_idx, late_bar_idx)
        k_pct: top-k fraction for stability check
        bar_duration_ns: ns per bar
        min_wealth_to_include: filter out agents with wealth <= this

    Returns dict with: gini_at_late, top_k_share_at_late, top_k_overlap_5k_10k,
                       n_agents_late, snap_indices_actual.
    """
    early_target_ns = bar_indices_to_compare[0] * bar_duration_ns
    late_target_ns = bar_indices_to_compare[1] * bar_duration_ns

    # Find nearest-or-before snapshot to each target
    snap_early = None
    snap_late = None
    actual_early_idx = None
    actual_late_idx = None
    for i, (ts, snap) in enumerate(wealth_history):
        if ts <= early_target_ns:
            snap_early = snap
            actual_early_idx = i
        if ts <= late_target_ns:
            snap_late = snap
            actual_late_idx = i
        if ts > late_target_ns:
            break

    if snap_late is None:
        snap_late = wealth_history[-1][1] if wealth_history else {}
        actual_late_idx = len(wealth_history) - 1
    if snap_early is None:
        snap_early = wealth_history[0][1] if wealth_history else {}
        actual_early_idx = 0

    if min_wealth_to_include > 0:
        snap_late = {a: w for a, w in snap_late.items() if w > min_wealth_to_include}

    wealths_late = list(snap_late.values())
    return {
        "gini_at_late": gini(wealths_late),
        "top_k_share_at_late": top_k_share(wealths_late, k_pct),
        "top_k_overlap_early_late": top_k_overlap(snap_early, snap_late, k_pct),
        "n_agents_late": len(wealths_late),
        "snap_actual_indices": (actual_early_idx, actual_late_idx),
        "k_pct": k_pct,
    }
