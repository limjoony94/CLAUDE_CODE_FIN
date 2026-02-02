"""
Verify: Trade-Outcome model probabilities for Oct 3 data
Compare with Sep 23, Oct 18, and Oct 19
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
print("VERIFY: TRADE-OUTCOME MODEL - OCT 3 PROBABILITY DISTRIBUTION")
print("="*80)

# ============================================================================
# STEP 1: Load Trade-Outcome models
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

# ============================================================================
# STEP 2: Load data and calculate features
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
# STEP 3: Filter dates
# ============================================================================
print("\n📅 Filtering dates...")

df_sep23 = df_features[df_features['timestamp'].str.startswith('2025-09-23')].copy()
df_oct03 = df_features[df_features['timestamp'].str.startswith('2025-10-03')].copy()
df_oct18 = df_features[df_features['timestamp'].str.startswith('2025-10-18')].copy()

# Calculate overall stats
overall_prices = df_features['close'].values
overall_mean_price = overall_prices.mean()

print(f"\n  Sep 23 candles: {len(df_sep23)}")
if len(df_sep23) > 0:
    print(f"    Time range: {df_sep23['timestamp'].iloc[0]} to {df_sep23['timestamp'].iloc[-1]}")
    print(f"    Price avg: ${df_sep23['close'].mean():,.2f} (vs overall: {(df_sep23['close'].mean() - overall_mean_price)/overall_mean_price*100:+.2f}%)")

print(f"\n  Oct 3 candles: {len(df_oct03)}")
if len(df_oct03) > 0:
    print(f"    Time range: {df_oct03['timestamp'].iloc[0]} to {df_oct03['timestamp'].iloc[-1]}")
    print(f"    Price avg: ${df_oct03['close'].mean():,.2f} (vs overall: {(df_oct03['close'].mean() - overall_mean_price)/overall_mean_price*100:+.2f}%)")

print(f"\n  Oct 18 candles: {len(df_oct18)}")
if len(df_oct18) > 0:
    print(f"    Time range: {df_oct18['timestamp'].iloc[0]} to {df_oct18['timestamp'].iloc[-1]}")
    print(f"    Price avg: ${df_oct18['close'].mean():,.2f} (vs overall: {(df_oct18['close'].mean() - overall_mean_price)/overall_mean_price*100:+.2f}%)")

if len(df_oct03) == 0:
    print("\n❌ No Oct 3 data found!")
    sys.exit(1)

# ============================================================================
# STEP 4: Calculate probabilities
# ============================================================================
def calculate_probs(df_date, date_name):
    """Calculate LONG probabilities for a given date"""
    probs = []
    timestamps = []
    prices = []
    macd_values = []
    macd_diff_values = []

    for idx in range(len(df_date)):
        row = df_date.iloc[idx:idx+1]
        features = row[long_feature_names].values
        scaled = long_scaler.transform(features)
        prob = long_model.predict_proba(scaled)[0][1]
        probs.append(prob)
        timestamps.append(row['timestamp'].iloc[0])
        prices.append(row['close'].iloc[0])
        macd_values.append(row['macd'].iloc[0])
        macd_diff_values.append(row['macd_diff'].iloc[0])

    return np.array(probs), timestamps, prices, macd_values, macd_diff_values

print("\n🔬 Calculating LONG probabilities...")

sep23_probs, sep23_timestamps, sep23_prices, sep23_macd, sep23_macd_diff = calculate_probs(df_sep23, "Sep 23")
oct03_probs, oct03_timestamps, oct03_prices, oct03_macd, oct03_macd_diff = calculate_probs(df_oct03, "Oct 3")
oct18_probs, oct18_timestamps, oct18_prices, oct18_macd, oct18_macd_diff = calculate_probs(df_oct18, "Oct 18")

# ============================================================================
# STEP 5: Compare distributions
# ============================================================================
print("\n" + "="*80)
print("PROBABILITY DISTRIBUTION COMPARISON")
print("="*80)

for date_name, probs, prices, macd, macd_diff in [
    ("SEP 23, 2025", sep23_probs, sep23_prices, sep23_macd, sep23_macd_diff),
    ("OCT 3, 2025", oct03_probs, oct03_prices, oct03_macd, oct03_macd_diff),
    ("OCT 18, 2025", oct18_probs, oct18_prices, oct18_macd, oct18_macd_diff)
]:
    print(f"\n📊 {date_name}:")
    print(f"  Candles: {len(probs)}")
    print(f"  Mean: {probs.mean():.4f} ({probs.mean()*100:.2f}%)")
    print(f"  Median: {np.median(probs):.4f} ({np.median(probs)*100:.2f}%)")
    print(f"  Std Dev: {probs.std():.4f}")
    print(f"  Min: {probs.min():.4f} ({probs.min()*100:.2f}%)")
    print(f"  Max: {probs.max():.4f} ({probs.max()*100:.2f}%)")

    print(f"\n  Distribution:")
    print(f"    ≥90%: {(probs >= 0.9).sum():4d} ({(probs >= 0.9).sum()/len(probs)*100:5.1f}%)")
    print(f"    ≥80%: {(probs >= 0.8).sum():4d} ({(probs >= 0.8).sum()/len(probs)*100:5.1f}%)")
    print(f"    ≥70%: {(probs >= 0.7).sum():4d} ({(probs >= 0.7).sum()/len(probs)*100:5.1f}%)")
    print(f"    ≥65%: {(probs >= 0.65).sum():4d} ({(probs >= 0.65).sum()/len(probs)*100:5.1f}%)")
    print(f"    <65%: {(probs < 0.65).sum():4d} ({(probs < 0.65).sum()/len(probs)*100:5.1f}%)")

    mean_price = np.mean(prices)
    print(f"\n  Price: ${mean_price:,.2f} (vs overall: {(mean_price - overall_mean_price)/overall_mean_price*100:+.2f}%)")
    print(f"  MACD: {np.mean(macd):.2f}, MACD_diff: {np.mean(macd_diff):.2f}")

# ============================================================================
# STEP 6: Check if Oct 3 is in Train or Test
# ============================================================================
print("\n" + "="*80)
print("TRAIN/TEST SPLIT ANALYSIS")
print("="*80)

split_idx = int(len(df_features) * 0.8)
train_end = df_features.iloc[split_idx-1]['timestamp']
test_start = df_features.iloc[split_idx]['timestamp']

print(f"\n📊 Split Point (80/20):")
print(f"  Train ends: {train_end}")
print(f"  Test starts: {test_start}")

# Check which dataset each date belongs to
dates_to_check = [
    ("Sep 23", df_sep23),
    ("Oct 3", df_oct03),
    ("Oct 18", df_oct18)
]

for date_name, df_date in dates_to_check:
    if len(df_date) > 0:
        date_idx = df_features[df_features['timestamp'] == df_date.iloc[0]['timestamp']].index[0]
        if date_idx < split_idx:
            print(f"  {date_name}: TRAIN 데이터 (index {date_idx} < {split_idx})")
        else:
            print(f"  {date_name}: TEST 데이터 (index {date_idx} >= {split_idx})")

# ============================================================================
# STEP 7: Sample output (Oct 3, first 30 candles)
# ============================================================================
print(f"\n📋 Sample LONG Probabilities (Oct 3, first 30 candles):")
print(f"{'Timestamp':<20} {'Price':<12} {'LONG Prob':<12} {'MACD':<10} {'MACD_diff':<12} {'Threshold':<12}")
print("-"*100)

for i in range(min(30, len(oct03_timestamps))):
    ts = oct03_timestamps[i]
    price = oct03_prices[i]
    prob = oct03_probs[i]
    macd_val = oct03_macd[i]
    macd_diff_val = oct03_macd_diff[i]
    threshold = "✅ ENTRY" if prob >= 0.65 else "⏸ WAIT"
    print(f"{ts:<20} ${price:<11,.2f} {prob*100:<11.2f}% {macd_val:<9.2f} {macd_diff_val:<11.2f} {threshold:<12}")

# ============================================================================
# STEP 8: Final summary
# ============================================================================
print(f"\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

print(f"\n📊 Average LONG Probability:")
print(f"  Sep 23 (TRAIN): {sep23_probs.mean()*100:6.2f}%")
print(f"  Oct 3  (TEST):  {oct03_probs.mean()*100:6.2f}%")
print(f"  Oct 18 (TEST):  {oct18_probs.mean()*100:6.2f}%")

print(f"\n📈 MACD Analysis:")
print(f"  Sep 23: MACD {np.mean(sep23_macd):8.2f}, MACD_diff {np.mean(sep23_macd_diff):8.2f}")
print(f"  Oct 3:  MACD {np.mean(oct03_macd):8.2f}, MACD_diff {np.mean(oct03_macd_diff):8.2f}")
print(f"  Oct 18: MACD {np.mean(oct18_macd):8.2f}, MACD_diff {np.mean(oct18_macd_diff):8.2f}")

# Pattern analysis
oct03_mean = oct03_probs.mean() * 100

if oct03_mean > 75:
    print(f"\n⚠️  OCT 3 shows HIGH probabilities (like Oct 18):")
    print(f"   Average: {oct03_mean:.2f}%")
    print(f"   Similar pattern to Oct 18")
elif oct03_mean > 50:
    print(f"\n⚖️  OCT 3 shows MODERATE probabilities:")
    print(f"   Average: {oct03_mean:.2f}%")
    print(f"   Between Sep 23 and Oct 18")
else:
    print(f"\n✅ OCT 3 shows NORMAL probabilities:")
    print(f"   Average: {oct03_mean:.2f}%")
    print(f"   Similar to Sep 23 baseline")

print("\n" + "="*80)
print("VERIFICATION COMPLETE")
print("="*80)
