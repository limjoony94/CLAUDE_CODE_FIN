"""
Quick Analysis: Check XGBoost Probability Distribution

Compare historical backtest probabilities vs current live probabilities
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys
from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"

logger.info("=" * 80)
logger.info("XGBoost Probability Distribution Analysis")
logger.info("=" * 80)

# Load model
model_file = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl"
with open(model_file, 'rb') as f:
    model = pickle.load(f)
logger.success(f"✅ Model loaded")

# Load features
feature_file = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt"
with open(feature_file, 'r') as f:
    feature_columns = [line.strip() for line in f.readlines()]
logger.success(f"✅ Features: {len(feature_columns)}")

# Load data
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df = pd.read_csv(data_file)
logger.success(f"✅ Data loaded: {len(df)} candles")

# Calculate features
logger.info("Calculating features...")
df = calculate_features(df)

adv_features = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)
df = adv_features.calculate_all_features(df)

df = df.ffill().dropna()
logger.info(f"After cleaning: {len(df)} rows")

# Get predictions on all data
logger.info("\nCalculating probabilities...")
X = df[feature_columns].values
probabilities = model.predict_proba(X)[:, 1]

logger.info(f"\n{'=' * 80}")
logger.info("PROBABILITY DISTRIBUTION ANALYSIS")
logger.info(f"{'=' * 80}")

logger.info(f"\nTotal samples: {len(probabilities)}")
logger.info(f"\nBasic Statistics:")
logger.info(f"  Mean: {np.mean(probabilities):.3f}")
logger.info(f"  Median: {np.median(probabilities):.3f}")
logger.info(f"  Std: {np.std(probabilities):.3f}")
logger.info(f"  Min: {np.min(probabilities):.3f}")
logger.info(f"  Max: {np.max(probabilities):.3f}")

logger.info(f"\nPercentiles:")
for p in [25, 50, 75, 90, 95, 99]:
    value = np.percentile(probabilities, p)
    logger.info(f"  {p:>2}th: {value:.3f}")

logger.info(f"\nThreshold Analysis:")
for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    count = sum(probabilities > threshold)
    pct = count / len(probabilities) * 100
    logger.info(f"  > {threshold:.1f}: {count:>6} samples ({pct:>5.2f}%)")

# Calculate expected entry frequency
logger.info(f"\n{'=' * 80}")
logger.info("EXPECTED TRADING FREQUENCY")
logger.info(f"{'=' * 80}")

window_size = 576  # 2 days
num_windows = len(probabilities) // window_size

for threshold in [0.5, 0.6, 0.7]:
    entries_per_window = []
    for i in range(num_windows):
        start_idx = i * window_size
        end_idx = start_idx + window_size
        window_probs = probabilities[start_idx:end_idx]
        entries = sum(window_probs > threshold)
        entries_per_window.append(entries)

    avg_entries = np.mean(entries_per_window) if entries_per_window else 0
    logger.info(f"\nThreshold {threshold:.1f}:")
    logger.info(f"  Avg entries per 2-day window: {avg_entries:.1f}")
    logger.info(f"  Expected per hour: {avg_entries / 48:.2f}")
    logger.info(f"  Expected per 76 minutes: {avg_entries * (76 / 2880):.2f}")

# Recent data (last 500 candles, ~42 hours)
logger.info(f"\n{'=' * 80}")
logger.info("RECENT DATA ANALYSIS (Last 500 candles ~ 42 hours)")
logger.info(f"{'=' * 80}")

recent_probs = probabilities[-500:]
logger.info(f"\nRecent Probabilities:")
logger.info(f"  Mean: {np.mean(recent_probs):.3f}")
logger.info(f"  Median: {np.median(recent_probs):.3f}")
logger.info(f"  Max: {np.max(recent_probs):.3f}")

for threshold in [0.5, 0.6, 0.7]:
    count = sum(recent_probs > threshold)
    pct = count / len(recent_probs) * 100
    logger.info(f"  > {threshold:.1f}: {count} ({pct:.1f}%)")

logger.info(f"\n{'=' * 80}")
logger.success("✅ Probability Analysis Complete!")
logger.info(f"{'=' * 80}")

# Critical assessment
logger.info(f"\n🎯 CRITICAL FINDINGS:")

mean_prob = np.mean(probabilities)
pct_above_07 = sum(probabilities > 0.7) / len(probabilities) * 100

if pct_above_07 < 1.0:
    logger.warning(f"⚠️ Only {pct_above_07:.2f}% of data has probability > 0.7")
    logger.warning(f"⚠️ Threshold 0.7 may be TOO HIGH for reliable trading")
    logger.warning(f"⚠️ Consider lower threshold (0.5-0.6) for more frequent trades")
elif pct_above_07 < 5.0:
    logger.info(f"📊 {pct_above_07:.2f}% of data has probability > 0.7")
    logger.info(f"📊 This is selective but may result in long waiting periods")
else:
    logger.success(f"✅ {pct_above_07:.2f}% of data has probability > 0.7")
    logger.success(f"✅ Threshold 0.7 is reasonable")
