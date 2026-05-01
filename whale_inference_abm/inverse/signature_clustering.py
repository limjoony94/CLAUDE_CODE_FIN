"""Statistical Signature inverse anchor (B) — K-means clustering on per-agent feature vector.

Design ref: docs/02-design/features/whale_inference_abm.design.md Section 3.2 (MVP B).

Per-agent features (8-dim):
  - Trade-size distribution: 5 quantile values (q20, q40, q60, q80, q100)
  - Mean inter-trade arrival time (sim-ns)
  - Taker fraction (count(role=='taker') / count(all))
  - Direction lag-1 autocorrelation (Pearson corr of buy/sell sequence)

Pipeline:
  trades -> per_agent ActionRecord lists -> filter min_trades -> features per agent
  -> StandardScaler -> K-means (K=5) -> labels

Pass criterion (advisor binding decision #1): ARI >= random_baseline_ari + 0.4
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from inverse.trajectory_collector import ActionRecord, filter_by_min_trades


FEATURE_DIM = 8
QUANTILES = [0.2, 0.4, 0.6, 0.8, 1.0]


def compute_features(actions: list[ActionRecord]) -> np.ndarray:
    """Extract 8-dim feature vector for one agent.

    Returns zero-vector with NaN markers if insufficient data; caller should filter.
    """
    if not actions:
        return np.zeros(FEATURE_DIM)

    sizes = np.array([a.size for a in actions])
    timestamps = np.array([a.timestamp_ns for a in actions])

    # Size quantiles (5 features)
    size_quantiles = np.quantile(sizes, QUANTILES)

    # Inter-trade arrival time (1 feature) — mean of differences
    if len(timestamps) >= 2:
        inter = np.diff(timestamps)
        mean_inter_ns = float(np.mean(inter))
    else:
        mean_inter_ns = 0.0

    # Taker fraction (1 feature)
    taker_count = sum(1 for a in actions if a.role == "taker")
    taker_frac = taker_count / len(actions)

    # Direction lag-1 autocorrelation (1 feature)
    dir_int = np.array([1.0 if a.side == "buy" else -1.0 for a in actions])
    if len(dir_int) >= 2 and np.std(dir_int) > 0:
        # Pearson corr of (x[1:], x[:-1])
        corr_matrix = np.corrcoef(dir_int[1:], dir_int[:-1])
        corr = float(corr_matrix[0, 1])
        if np.isnan(corr):
            corr = 0.0
    else:
        corr = 0.0

    return np.concatenate([size_quantiles, [mean_inter_ns, taker_frac, corr]])


def cluster_agents(
    per_agent: dict[str, list[ActionRecord]],
    K: int = 5,
    min_trades: int = 3,
    random_state: int = 42,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """K-means cluster agents based on signature features.

    Returns:
        (agent_ids_sorted, predicted_labels, raw_features) — sorted for determinism
    """
    eligible = filter_by_min_trades(per_agent, min_trades)
    if len(eligible) < K:
        raise ValueError(
            f"Need at least K={K} eligible agents (>= {min_trades} trades), got {len(eligible)}"
        )

    aids = sorted(eligible.keys())
    features = np.array([compute_features(eligible[aid]) for aid in aids])

    # Standardize so different feature scales (size << inter_time_ns) don't dominate
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    km = KMeans(n_clusters=K, random_state=random_state, n_init=10)
    predicted = km.fit_predict(features_scaled)

    return aids, predicted, features


def evaluate_ari(
    aids: list[str],
    predicted_labels: np.ndarray,
    family_lookup: dict[str, str],
) -> tuple[float, list[str]]:
    """Compute ARI between predicted clusters and true family labels.

    Returns:
        (ari_score, aligned_true_labels)
    """
    from sklearn.metrics import adjusted_rand_score

    # Filter to agents present in family_lookup (skip orphan / bankrupt agents not in registry)
    pairs = [(i, aid) for i, aid in enumerate(aids) if aid in family_lookup]
    if not pairs:
        return 0.0, []
    indices = [i for i, _ in pairs]
    aids_filt = [aid for _, aid in pairs]

    true_labels = [family_lookup[aid] for aid in aids_filt]
    predicted_filt = predicted_labels[indices]

    ari = float(adjusted_rand_score(true_labels, predicted_filt))
    return ari, true_labels
