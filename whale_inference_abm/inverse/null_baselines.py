"""Null baseline computations for relative-criterion pass evaluation.

Per advisor binding decision #1 (architecture v1.1 patch 1):
  Pass criteria are RELATIVE to null, not raw thresholds. Raw threshold of "0.6 ARI"
  is meaningless without knowing what random clustering achieves on the same labels.

Null baselines per anchor:
  - Signature (K-means): random uniform clustering ARI distribution
  - IRL (MaxEnt): "always predict modal action" + "copy last action" + uniform random
  - Parametric (Bayesian): uniform prior posterior (1/K)
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np
from sklearn.metrics import adjusted_rand_score


def random_clustering_ari_baseline(
    true_labels: Sequence,
    K: int,
    n_trials: int = 200,
    seed: int = 42,
) -> dict[str, float]:
    """Random uniform clustering ARI distribution.

    Returns:
        dict with keys 'mean', 'std', 'p95', 'samples' (last for inspection)
    """
    rng = np.random.default_rng(seed)
    n = len(true_labels)
    aris: list[float] = []
    for _ in range(n_trials):
        random_labels = rng.integers(0, K, size=n)
        aris.append(float(adjusted_rand_score(true_labels, random_labels)))
    return {
        "mean": float(np.mean(aris)),
        "std": float(np.std(aris)),
        "p95": float(np.quantile(aris, 0.95)),
        "samples": aris,
    }


def modal_action_accuracy(actions: Sequence) -> float:
    """Predict-modal-action baseline accuracy. Trivial baseline for IRL evaluation."""
    if len(actions) == 0:
        return 0.0
    counts = Counter(actions)
    modal_count = counts.most_common(1)[0][1]
    return modal_count / len(actions)


def last_action_accuracy(actions: Sequence) -> float:
    """Predict-previous-action baseline. Captures simple persistence."""
    if len(actions) < 2:
        return 0.0
    correct = sum(1 for i in range(1, len(actions)) if actions[i] == actions[i - 1])
    return correct / (len(actions) - 1)


def uniform_random_accuracy(action_space_size: int) -> float:
    """Theoretical: 1 / |action_space|."""
    if action_space_size <= 0:
        return 0.0
    return 1.0 / action_space_size


def uniform_prior_posterior_mass(num_families: int) -> float:
    """Bayesian uniform prior posterior probability per family."""
    if num_families <= 0:
        return 0.0
    return 1.0 / num_families
