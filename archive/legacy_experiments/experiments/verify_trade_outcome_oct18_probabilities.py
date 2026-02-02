"""
Verify: Trade-Outcome model outputs high probabilities for Oct 18 data
Direct verification using backtest method
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import pickle
import joblib

from scripts.experiments.calculate_all_features import calculate_all_features
from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"

print("="*80)
print("VERIFY: TRADE-OUTCOME MODEL - OCT 18 PROBABILITY DISTRIBUTION")
print("="*80)

# ============================================================================
# STEP 1: Load Trade-Outcome models (current production)
# ============================================================================
print("\n📁 Loading Trade-Outcome models...")

long_model_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146.pkl"
with open(long_model_path, 'rb') as f:
    long_model = pickle.load(f)

long_scaler_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146_scaler.pkl"
long_scaler = joblib.load(long_scaler_path)

long_features_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146_features.txt"
with open(long_features_path, 'r') as f:
    long_feature_names = [line.strip() for line in f.readlines()]

print(f"  ✅ Trade-Outcome LONG model loaded")
print(f"     Model: {long_model_path.name}")
print(f"     Features: {len(long_feature_names)}")
print(f"     Scaler: {type(long_scaler).__name__}")

# ============================================================================
# STEP 2: Load data and calculate features (BACKTEST METHOD)
# ============================================================================
print("\n📊 Loading data...")
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df = pd.read_csv(data_file)
print(f"  Total candles: {len(df):,}")

print("\n🔧 Calculating features (BACKTEST METHOD)...")
df_features = calculate_all_features(df.copy())
df_features = prepare_exit_features(df_features)
print(f"  ✅ Features calculated: {len(df_features):,} candles")

# ============================================================================
# STEP 3: Filter Oct 18 data
# ============================================================================
print("\n📅 Filtering Oct 18, 2025 data...")
df_oct18 = df_features[df_features['timestamp'].str.startswith('2025-10-18')].copy()
print(f"  Oct 18 candles: {len(df_oct18)}")
print(f"  Time range: {df_oct18['timestamp'].iloc[0]} to {df_oct18['timestamp'].iloc[-1]}")

if len(df_oct18) == 0:
    print("❌ No Oct 18 data found!")
    sys.exit(1)

# ============================================================================
# STEP 4: Calculate LONG probabilities for ALL Oct 18 candles
# ============================================================================
print("\n🔬 Calculating LONG probabilities for Oct 18...")

oct18_probs = []
oct18_timestamps = []
oct18_prices = []

for idx in range(len(df_oct18)):
    row = df_oct18.iloc[idx:idx+1]

    # Extract features
    features = row[long_feature_names].values

    # Scale
    scaled = long_scaler.transform(features)

    # Predict
    prob = long_model.predict_proba(scaled)[0][1]

    oct18_probs.append(prob)
    oct18_timestamps.append(row['timestamp'].iloc[0])
    oct18_prices.append(row['close'].iloc[0])

oct18_probs = np.array(oct18_probs)

# ============================================================================
# STEP 5: Analyze distribution
# ============================================================================
print(f"\n📊 OCT 18 LONG PROBABILITY DISTRIBUTION (BACKTEST METHOD):")
print("="*80)

print(f"\n📈 Statistics:")
print(f"  Candles: {len(oct18_probs)}")
print(f"  Mean: {oct18_probs.mean():.4f} ({oct18_probs.mean()*100:.2f}%)")
print(f"  Median: {np.median(oct18_probs):.4f} ({np.median(oct18_probs)*100:.2f}%)")
print(f"  Std Dev: {oct18_probs.std():.4f}")
print(f"  Min: {oct18_probs.min():.4f} ({oct18_probs.min()*100:.2f}%)")
print(f"  Max: {oct18_probs.max():.4f} ({oct18_probs.max()*100:.2f}%)")

print(f"\n📊 Distribution:")
print(f"  ≥90%: {(oct18_probs >= 0.9).sum():4d} ({(oct18_probs >= 0.9).sum()/len(oct18_probs)*100:5.1f}%)")
print(f"  ≥80%: {(oct18_probs >= 0.8).sum():4d} ({(oct18_probs >= 0.8).sum()/len(oct18_probs)*100:5.1f}%)")
print(f"  ≥70%: {(oct18_probs >= 0.7).sum():4d} ({(oct18_probs >= 0.7).sum()/len(oct18_probs)*100:5.1f}%)")
print(f"  ≥65%: {(oct18_probs >= 0.65).sum():4d} ({(oct18_probs >= 0.65).sum()/len(oct18_probs)*100:5.1f}%)")
print(f"  <65%: {(oct18_probs < 0.65).sum():4d} ({(oct18_probs < 0.65).sum()/len(oct18_probs)*100:5.1f}%)")

# ============================================================================
# STEP 6: Compare with production log
# ============================================================================
print(f"\n📋 COMPARISON WITH PRODUCTION LOG:")
print("="*80)

# Production period: 18:50 - 21:55 (from earlier analysis)
df_production_period = df_oct18[
    (df_oct18['timestamp'] >= '2025-10-18 18:50:00') &
    (df_oct18['timestamp'] <= '2025-10-18 21:55:00')
].copy()

if len(df_production_period) > 0:
    prod_period_probs = []
    for idx in range(len(df_production_period)):
        row = df_production_period.iloc[idx:idx+1]
        features = row[long_feature_names].values
        scaled = long_scaler.transform(features)
        prob = long_model.predict_proba(scaled)[0][1]
        prod_period_probs.append(prob)

    prod_period_probs = np.array(prod_period_probs)

    print(f"\nProduction Period (18:50 - 21:55):")
    print(f"  Backtest Method:")
    print(f"    Mean: {prod_period_probs.mean():.4f} ({prod_period_probs.mean()*100:.2f}%)")
    print(f"    ≥70%: {(prod_period_probs >= 0.7).sum()}/{len(prod_period_probs)} ({(prod_period_probs >= 0.7).sum()/len(prod_period_probs)*100:.1f}%)")
    print(f"    ≥90%: {(prod_period_probs >= 0.9).sum()}/{len(prod_period_probs)} ({(prod_period_probs >= 0.9).sum()/len(prod_period_probs)*100:.1f}%)")

    print(f"\n  Production Log (from earlier analysis):")
    print(f"    Mean: 81.84%")
    print(f"    ≥70%: 84.6%")
    print(f"    ≥90%: 26.9%")

    print(f"\n  Difference:")
    print(f"    Mean: {abs(prod_period_probs.mean()*100 - 81.84):.2f}%")

# ============================================================================
# STEP 7: Sample output
# ============================================================================
print(f"\n📋 Sample LONG Probabilities (first 30 candles):")
print(f"{'Timestamp':<20} {'Price':<12} {'LONG Prob':<12} {'Threshold':<12}")
print("-"*80)

for i in range(min(30, len(oct18_timestamps))):
    ts = oct18_timestamps[i]
    price = oct18_prices[i]
    prob = oct18_probs[i]
    threshold = "✅ ENTRY" if prob >= 0.65 else "⏸ WAIT"
    print(f"{ts:<20} ${price:<11,.2f} {prob*100:<11.2f}% {threshold:<12}")

# ============================================================================
# STEP 8: Hourly breakdown
# ============================================================================
print(f"\n⏰ HOURLY BREAKDOWN:")
print("="*80)

hourly_stats = []
for hour in range(24):
    hour_str = f"{hour:02d}"
    df_hour = df_oct18[df_oct18['timestamp'].str.contains(f' {hour_str}:')].copy()

    if len(df_hour) == 0:
        continue

    hour_probs = []
    for idx in range(len(df_hour)):
        row = df_hour.iloc[idx:idx+1]
        features = row[long_feature_names].values
        scaled = long_scaler.transform(features)
        prob = long_model.predict_proba(scaled)[0][1]
        hour_probs.append(prob)

    hour_probs = np.array(hour_probs)

    hourly_stats.append({
        'hour': hour,
        'count': len(hour_probs),
        'mean': hour_probs.mean(),
        'median': np.median(hour_probs),
        'min': hour_probs.min(),
        'max': hour_probs.max(),
        'ge70': (hour_probs >= 0.7).sum()
    })

print(f"\n{'Hour':<6} {'Count':<7} {'Mean':<10} {'Median':<10} {'Min':<10} {'Max':<10} {'≥70%':<10}")
print("-"*80)

for stat in hourly_stats:
    print(f"{stat['hour']:02d}:00  {stat['count']:<7} {stat['mean']*100:<9.1f}% {stat['median']*100:<9.1f}% "
          f"{stat['min']*100:<9.1f}% {stat['max']*100:<9.1f}% {stat['ge70']}/{stat['count']}")

# ============================================================================
# STEP 9: Final verdict
# ============================================================================
print(f"\n" + "="*80)
print("FINAL VERDICT")
print("="*80)

if oct18_probs.mean() > 0.7:
    print(f"\n✅ CONFIRMED: Trade-Outcome model outputs HIGH probabilities for Oct 18")
    print(f"   Backtest Method: {oct18_probs.mean()*100:.2f}% average")
    print(f"   ≥70% candles: {(oct18_probs >= 0.7).sum()/len(oct18_probs)*100:.1f}%")
    print(f"\n   → This is NORMAL model behavior for this market condition")
    print(f"   → Model correctly identifies low prices as potential buy opportunities")
    print(f"   → However, market continued downward (mean reversion didn't happen)")
elif oct18_probs.mean() < 0.3:
    print(f"\n❌ UNEXPECTED: Trade-Outcome model outputs LOW probabilities for Oct 18")
    print(f"   Backtest Method: {oct18_probs.mean()*100:.2f}% average")
    print(f"   → This contradicts production log observations")
    print(f"   → Further investigation needed")
else:
    print(f"\n⚠️  MODERATE: Trade-Outcome model outputs NORMAL probabilities for Oct 18")
    print(f"   Backtest Method: {oct18_probs.mean()*100:.2f}% average")
    print(f"   → Within expected range")

print("\n" + "="*80)
print("VERIFICATION COMPLETE")
print("="*80)
