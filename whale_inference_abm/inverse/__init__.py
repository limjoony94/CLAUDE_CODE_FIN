"""Inverse-recovery anchors for whale strategy reconstruction.

3 parallel MVPs per design Section 3 (G1):
  - signature_clustering.py: K-means on per-agent feature vector (Anchor B, simplest)
  - irl_maxent.py: MaxEnt IRL on (state, action) trajectories (Anchor A, primary)
  - parametric_bayes.py: PyMC posterior over 5 strategy families (Anchor C, tertiary)

All three pass criteria are RELATIVE to null baselines (advisor binding decision #1):
  - Signature: ARI >= random_baseline + 0.4
  - IRL: action-prediction accuracy >= max(null_baselines) + 15pp
  - Parametric: posterior on correct family >= uniform_prior + 30pp on >= 4/5 agents
"""
