"""Parametric Prior inverse anchor (C) — Bayesian classifier over 5 strategy families.

Design ref: docs/02-design/features/whale_inference_abm.design.md Section 3.3 (MVP C).

Approach: Naive Bayes with Gaussian likelihoods over the 8-dim signature feature space.
- "Parametric" = each family has Gaussian likelihood over each feature dimension
  (mean + variance estimated from observed agents of that family)
- "Bayesian" = uniform prior, posterior = likelihood × prior, normalized
- Uses sklearn.naive_bayes.GaussianNB which implements exactly this

Per design v0.4 patch: this is MVP. PyMC over true parametric strategy forms (momentum N
parameter, mean-rev threshold, etc.) is the v2 path if MVP shows insufficient differentiation.

Pass criterion (advisor binding decision #1):
  posterior on correct family > 50% for >= 4/5 agents (>= 80% per-agent accuracy)
  AND posterior > uniform_prior + 30pp = 50% (uniform K=5 prior)
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

from inverse.signature_clustering import compute_features
from inverse.trajectory_collector import ActionRecord, filter_by_min_trades


def build_feature_matrix(
    per_agent: dict[str, list[ActionRecord]],
    family_lookup: dict[str, str],
    min_trades: int = 3,
) -> tuple[list[str], np.ndarray, list[str]]:
    """Build (agent_ids, feature_matrix, family_labels) aligned arrays.

    Filters: agents with >= min_trades AND present in family_lookup.
    """
    eligible = filter_by_min_trades(per_agent, min_trades)
    aids = sorted(aid for aid in eligible if aid in family_lookup)
    if not aids:
        raise ValueError("No eligible agents with both min_trades and family_lookup entry")

    features = np.array([compute_features(eligible[aid]) for aid in aids])
    families = [family_lookup[aid] for aid in aids]
    return aids, features, families


def loo_posteriors(
    features: np.ndarray,
    families: list[str],
    classifier: str = "logreg",
) -> np.ndarray:
    """Leave-one-out cross-validated posterior probabilities.

    Args:
        classifier: 'logreg' (default per advisor — multinomial + balanced),
                    'gnb' (GaussianNB, original), 'lda' (LinearDiscriminantAnalysis)

    Returns:
        posteriors: shape (n_agents, n_families) with each row summing to 1.
        Column order matches sorted(unique(families)).
    """
    unique_families = sorted(set(families))
    family_to_idx = {f: i for i, f in enumerate(unique_families)}
    family_indices = np.array([family_to_idx[f] for f in families])

    n = len(features)
    n_classes = len(unique_families)
    posteriors = np.zeros((n, n_classes))

    # Standardize features (logreg + lda need scale-invariance to play nicely)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(features_scaled):
        x_train = features_scaled[train_idx]
        y_train = family_indices[train_idx]
        x_test = features_scaled[test_idx]

        if len(set(y_train)) < 2:
            posteriors[test_idx] = 1.0 / n_classes
            continue

        if classifier == "logreg":
            # multi_class default is multinomial in sklearn >= 1.7 (per deprecation notice)
            clf = LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
            )
        elif classifier == "lda":
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            clf = LinearDiscriminantAnalysis()
        elif classifier == "gnb":
            clf = GaussianNB()
        else:
            raise ValueError(f"Unknown classifier: {classifier}")

        try:
            clf.fit(x_train, y_train)
        except ValueError:
            posteriors[test_idx] = 1.0 / n_classes
            continue

        proba = clf.predict_proba(x_test)
        full_proba = np.zeros((1, n_classes))
        for j, cls_idx in enumerate(clf.classes_):
            full_proba[0, cls_idx] = proba[0, j]
        posteriors[test_idx] = full_proba

    return posteriors


def evaluate_per_family_best_representative(
    aids: list[str],
    families: list[str],
    posteriors: np.ndarray,
    per_agent_actions: dict,
    threshold: float = 0.5,
) -> dict:
    """Per-family-best-rep evaluation (advisor amendment to design Section 3.3).

    Selects the agent with most trades per family as that family's representative,
    then checks whether posterior on its true family > threshold.
    Pass criterion: at least 4 of 5 families pass.
    """
    unique_families = sorted(set(families))
    family_to_idx = {f: i for i, f in enumerate(unique_families)}

    family_to_best: dict[str, tuple[str, int]] = {}
    for aid, fam in zip(aids, families):
        n_trades = len(per_agent_actions.get(aid, []))
        cur_best = family_to_best.get(fam)
        if cur_best is None or n_trades > cur_best[1]:
            family_to_best[fam] = (aid, n_trades)

    pass_per_family: dict[str, dict] = {}
    families_passed = 0
    for fam, (best_aid, best_n) in family_to_best.items():
        i = aids.index(best_aid)
        post_on_correct = float(posteriors[i, family_to_idx[fam]])
        passed = post_on_correct > threshold
        pass_per_family[fam] = {
            "agent_id": best_aid,
            "n_trades": best_n,
            "posterior_on_true": post_on_correct,
            "passed": passed,
        }
        if passed:
            families_passed += 1

    return {
        "per_family_best": pass_per_family,
        "families_passed": families_passed,
        "families_total": len(unique_families),
        "passes_4_of_5": families_passed >= 4,
    }


def evaluate_posteriors(
    families: list[str],
    posteriors: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Evaluate posterior assignment quality.

    Returns dict with:
      - n_agents
      - n_correct_above_threshold: agents where posterior on TRUE family > threshold
      - fraction_above_threshold: ratio
      - per_family_correct: dict[family, (correct, total)]
      - mean_correct_posterior: mean posterior on true family across all agents
    """
    unique_families = sorted(set(families))
    family_to_idx = {f: i for i, f in enumerate(unique_families)}

    n = len(families)
    correct_count = 0
    per_family_correct: dict[str, list[int]] = {f: [0, 0] for f in unique_families}
    correct_posteriors: list[float] = []

    for i, fam in enumerate(families):
        true_idx = family_to_idx[fam]
        post_on_true = float(posteriors[i, true_idx])
        correct_posteriors.append(post_on_true)
        per_family_correct[fam][1] += 1
        if post_on_true > threshold:
            correct_count += 1
            per_family_correct[fam][0] += 1

    return {
        "n_agents": n,
        "n_correct_above_threshold": correct_count,
        "fraction_above_threshold": correct_count / n if n > 0 else 0.0,
        "per_family_correct": {
            f: tuple(v) for f, v in per_family_correct.items()
        },
        "mean_correct_posterior": float(np.mean(correct_posteriors)),
        "threshold": threshold,
    }
