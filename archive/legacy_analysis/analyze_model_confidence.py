"""
Model Confidence Analysis

결정적 실험: 모델이 실제 기회(opportunity)에 부여하는 확률 직접 측정

가설:
- LONG 모델 + LONG opportunities → 평균 확률 X
- SHORT 모델 + SHORT opportunities → 평균 확률 Y
- 만약 X >> Y → 모델 confidence 불균형 (root cause!)
- 만약 X ≈ Y → 전체 데이터 분포 문제
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures

# Load data
DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"

df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_max.csv")

# Calculate features
df = calculate_features(df)
adv_features = AdvancedTechnicalFeatures()
df = adv_features.calculate_all_features(df)
df = df.ffill().dropna()

print("=" * 80)
print("MODEL CONFIDENCE ANALYSIS")
print("=" * 80)
print(f"Data: {len(df)} candles\n")

# Define opportunities
lookahead = 3
df['future_max'] = df['close'].shift(-1).rolling(window=lookahead).apply(lambda x: x.max())
df['future_min'] = df['close'].shift(-1).rolling(window=lookahead).apply(lambda x: x.min())
df['return_to_max'] = (df['future_max'] - df['close']) / df['close']
df['return_to_min'] = (df['close'] - df['future_min']) / df['close']

threshold = 0.003  # 0.3%
df['is_long_opp'] = (df['return_to_max'] >= threshold).astype(int)
df['is_short_opp'] = (df['return_to_min'] >= threshold).astype(int)

df = df.dropna()

print(f"LONG opportunities: {df['is_long_opp'].sum()}")
print(f"SHORT opportunities: {df['is_short_opp'].sum()}")
print()

# Load models and scalers
with open(MODELS_DIR / 'xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl', 'rb') as f:
    long_model = pickle.load(f)

with open(MODELS_DIR / 'xgboost_v4_phase4_advanced_lookahead3_thresh0_scaler.pkl', 'rb') as f:
    long_scaler = pickle.load(f)

with open(MODELS_DIR / 'xgboost_short_model_lookahead3_thresh0.3.pkl', 'rb') as f:
    short_model = pickle.load(f)

with open(MODELS_DIR / 'xgboost_short_model_lookahead3_thresh0.3_scaler.pkl', 'rb') as f:
    short_scaler = pickle.load(f)

with open(MODELS_DIR / 'xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt', 'r') as f:
    feature_columns = [line.strip() for line in f]

# Get features
X = df[feature_columns].values

# Scale
X_long_scaled = long_scaler.transform(X)
X_short_scaled = short_scaler.transform(X)

# Predict probabilities
prob_long = long_model.predict_proba(X_long_scaled)[:, 1]
prob_short = short_model.predict_proba(X_short_scaled)[:, 1]

# Extract opportunity samples
long_opp_indices = df[df['is_long_opp'] == 1].index
short_opp_indices = df[df['is_short_opp'] == 1].index

# Model confidence on actual opportunities
long_model_on_long_opps = prob_long[df['is_long_opp'] == 1]
short_model_on_short_opps = prob_short[df['is_short_opp'] == 1]

print("=" * 80)
print("CRITICAL EXPERIMENT: Model Confidence on Actual Opportunities")
print("=" * 80)
print()

print("LONG Model on LONG Opportunities:")
print(f"  Mean probability: {long_model_on_long_opps.mean():.4f}")
print(f"  Median probability: {np.median(long_model_on_long_opps):.4f}")
print(f"  Std: {long_model_on_long_opps.std():.4f}")
print(f"  Prob > 0.5: {(long_model_on_long_opps > 0.5).sum()} ({(long_model_on_long_opps > 0.5).sum() / len(long_model_on_long_opps) * 100:.1f}%)")
print(f"  Prob > 0.8: {(long_model_on_long_opps > 0.8).sum()} ({(long_model_on_long_opps > 0.8).sum() / len(long_model_on_long_opps) * 100:.1f}%)")
print()

print("SHORT Model on SHORT Opportunities:")
print(f"  Mean probability: {short_model_on_short_opps.mean():.4f}")
print(f"  Median probability: {np.median(short_model_on_short_opps):.4f}")
print(f"  Std: {short_model_on_short_opps.std():.4f}")
print(f"  Prob > 0.5: {(short_model_on_short_opps > 0.5).sum()} ({(short_model_on_short_opps > 0.5).sum() / len(short_model_on_short_opps) * 100:.1f}%)")
print(f"  Prob > 0.8: {(short_model_on_short_opps > 0.8).sum()} ({(short_model_on_short_opps > 0.8).sum() / len(short_model_on_short_opps) * 100:.1f}%)")
print()

# Calculate ratio
ratio = long_model_on_long_opps.mean() / short_model_on_short_opps.mean()

print("=" * 80)
print(f"CONFIDENCE RATIO: {ratio:.2f}x")
print("=" * 80)
print()

if ratio > 1.5:
    print("🔴 ROOT CAUSE FOUND: Model Confidence Imbalance!")
    print()
    print("LONG 모델은 LONG opportunities에 높은 확률을 부여하지만,")
    print("SHORT 모델은 SHORT opportunities에 낮은 확률을 부여합니다.")
    print()
    print("Possible Reasons:")
    print("  1. LONG opportunities have clearer feature patterns")
    print("  2. SHORT opportunities are more subtle/noisy")
    print("  3. Model calibration issue (Platt scaling needed)")
    print("  4. Class imbalance handling (SMOTE ratio different)")
    print()
    print("Solutions:")
    print("  A. Increase SMOTE ratio for SHORT model")
    print("  B. Apply probability calibration (Isotonic/Platt)")
    print("  C. Use different hyperparameters for SHORT (higher learning rate)")
    print("  D. Lower SHORT threshold (0.80 → 0.50-0.60)")
elif ratio < 0.7:
    print("⚠️ Unexpected: SHORT model MORE confident than LONG!")
else:
    print("✅ Models have similar confidence on their opportunities")
    print()
    print("Root cause is NOT model confidence imbalance.")
    print("Problem is likely in overall data distribution or threshold selection.")

# Additional analysis: Overall distribution
print()
print("=" * 80)
print("OVERALL PROBABILITY DISTRIBUTION (All Data)")
print("=" * 80)
print()
print(f"{'Metric':<20} {'LONG Model':>15} {'SHORT Model':>15} {'Ratio':>10}")
print("-" * 80)
print(f"{'Mean':<20} {prob_long.mean():>15.4f} {prob_short.mean():>15.4f} {prob_long.mean() / prob_short.mean():>10.2f}x")
print(f"{'Median':<20} {np.median(prob_long):>15.4f} {np.median(prob_short):>15.4f} {np.median(prob_long) / np.median(prob_short):>10.2f}x")
print(f"{'Std':<20} {prob_long.std():>15.4f} {prob_short.std():>15.4f} {prob_long.std() / prob_short.std():>10.2f}x")
print()
print("Overall distribution shows LONG probabilities are {:.2f}x higher.".format(prob_long.mean() / prob_short.mean()))
print("This affects threshold crossing rates → trade frequency imbalance.")
