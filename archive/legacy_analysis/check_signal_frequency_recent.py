"""
Check Signal Frequency on Recent Data (Oct 15-20)
=================================================
Calculate LONG/SHORT signal frequency using production models
"""
import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.calculate_all_features import calculate_all_features

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"

# Thresholds
LONG_THRESHOLD = 0.65
SHORT_THRESHOLD = 0.70

print("=" * 80)
print("📊 Signal Frequency Analysis - Recent Data (Oct 15-20)")
print("=" * 80)

# Load models (PRODUCTION MODELS)
print("\nLoading production models...")
long_model_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146.pkl"
with open(long_model_path, 'rb') as f:
    long_model = pickle.load(f)

long_scaler_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146_scaler.pkl"
long_scaler = joblib.load(long_scaler_path)

long_features_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146_features.txt"
with open(long_features_path, 'r') as f:
    long_feature_columns = [line.strip() for line in f.readlines()]

short_model_path = MODELS_DIR / "xgboost_short_trade_outcome_full_20251018_233146.pkl"
with open(short_model_path, 'rb') as f:
    short_model = pickle.load(f)

short_scaler_path = MODELS_DIR / "xgboost_short_trade_outcome_full_20251018_233146_scaler.pkl"
short_scaler = joblib.load(short_scaler_path)

short_features_path = MODELS_DIR / "xgboost_short_trade_outcome_full_20251018_233146_features.txt"
with open(short_features_path, 'r') as f:
    short_feature_columns = [line.strip() for line in f.readlines()]

print(f"✅ Models loaded: LONG({len(long_feature_columns)}), SHORT({len(short_feature_columns)})")

# Load recent data
print("\nLoading recent data...")
data_path = DATA_DIR / "BTCUSDT_5m_recent.csv"
df = pd.read_csv(data_path)
df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f"✅ Loaded {len(df)} candles")
print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

# Calculate features
print("\nCalculating features...")
df = calculate_all_features(df)
print(f"✅ Features calculated")

# Calculate probabilities
print("\nCalculating signal probabilities...")
long_probs = []
short_probs = []

for i in range(len(df)):
    try:
        # LONG
        long_feat = df[long_feature_columns].iloc[i:i+1].values
        if not np.isnan(long_feat).any():
            long_scaled = long_scaler.transform(long_feat)
            long_prob = long_model.predict_proba(long_scaled)[0][1]
            long_probs.append(long_prob)
        else:
            long_probs.append(np.nan)

        # SHORT
        short_feat = df[short_feature_columns].iloc[i:i+1].values
        if not np.isnan(short_feat).any():
            short_scaled = short_scaler.transform(short_feat)
            short_prob = short_model.predict_proba(short_scaled)[0][1]
            short_probs.append(short_prob)
        else:
            short_probs.append(np.nan)
    except:
        long_probs.append(np.nan)
        short_probs.append(np.nan)

df['long_prob'] = long_probs
df['short_prob'] = short_probs

# Remove NaN
df_clean = df.dropna(subset=['long_prob', 'short_prob']).copy()

print(f"✅ Probabilities calculated for {len(df_clean)} candles")

# Analyze signal frequency by date
print("\n" + "=" * 80)
print("📊 SIGNAL FREQUENCY ANALYSIS BY DATE")
print("=" * 80)

df_clean['date'] = df_clean['timestamp'].dt.date

for date in sorted(df_clean['date'].unique())[-5:]:  # Last 5 days
    df_day = df_clean[df_clean['date'] == date]

    long_signals = (df_day['long_prob'] >= LONG_THRESHOLD).sum()
    short_signals = (df_day['short_prob'] >= SHORT_THRESHOLD).sum()
    total = len(df_day)

    print(f"\n📅 {date}:")
    print(f"   Total candles: {total}")
    print(f"   LONG signals (>= {LONG_THRESHOLD}): {long_signals} ({long_signals/total*100:.1f}%)")
    print(f"   SHORT signals (>= {SHORT_THRESHOLD}): {short_signals} ({short_signals/total*100:.1f}%)")
    print(f"   Avg LONG prob: {df_day['long_prob'].mean():.4f}")
    print(f"   Avg SHORT prob: {df_day['short_prob'].mean():.4f}")

# Overall statistics
print("\n" + "=" * 80)
print("📊 OVERALL STATISTICS")
print("=" * 80)

long_signals_total = (df_clean['long_prob'] >= LONG_THRESHOLD).sum()
short_signals_total = (df_clean['short_prob'] >= SHORT_THRESHOLD).sum()
total_candles = len(df_clean)

print(f"\nTotal candles analyzed: {total_candles}")
print(f"LONG signals: {long_signals_total} ({long_signals_total/total_candles*100:.1f}%)")
print(f"SHORT signals: {short_signals_total} ({short_signals_total/total_candles*100:.1f}%)")
print(f"\nAverage probabilities:")
print(f"  LONG: {df_clean['long_prob'].mean():.4f}")
print(f"  SHORT: {df_clean['short_prob'].mean():.4f}")

# Probability distribution
print("\n" + "=" * 80)
print("📊 PROBABILITY DISTRIBUTION")
print("=" * 80)

print("\nLONG probability ranges:")
for lower in [0.0, 0.3, 0.5, 0.65, 0.75, 0.85]:
    upper = lower + 0.10 if lower < 0.85 else 1.0
    count = ((df_clean['long_prob'] >= lower) & (df_clean['long_prob'] < upper)).sum()
    pct = count / len(df_clean) * 100
    print(f"  {lower:.2f} - {upper:.2f}: {count:5d} ({pct:5.1f}%)")

print("\n" + "=" * 80)
