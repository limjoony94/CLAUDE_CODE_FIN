"""
Verify: Trade-Outcome model probabilities for Oct 19 data (TODAY)
Compare with Oct 18 and Sep 23
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
print("VERIFY: TRADE-OUTCOME MODEL - OCT 19 (TODAY) PROBABILITY DISTRIBUTION")
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
print(f"     Features: {len(long_feature_names)}")

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
df_oct18 = df_features[df_features['timestamp'].str.startswith('2025-10-18')].copy()
df_oct19 = df_features[df_features['timestamp'].str.startswith('2025-10-19')].copy()

print(f"\n  Sep 23 candles: {len(df_sep23)}")
if len(df_sep23) > 0:
    print(f"    Time range: {df_sep23['timestamp'].iloc[0]} to {df_sep23['timestamp'].iloc[-1]}")
    print(f"    Price range: ${df_sep23['close'].min():,.2f} - ${df_sep23['close'].max():,.2f}")
    print(f"    Avg price: ${df_sep23['close'].mean():,.2f}")

print(f"\n  Oct 18 candles: {len(df_oct18)}")
if len(df_oct18) > 0:
    print(f"    Time range: {df_oct18['timestamp'].iloc[0]} to {df_oct18['timestamp'].iloc[-1]}")
    print(f"    Price range: ${df_oct18['close'].min():,.2f} - ${df_oct18['close'].max():,.2f}")
    print(f"    Avg price: ${df_oct18['close'].mean():,.2f}")

print(f"\n  Oct 19 candles: {len(df_oct19)}")
if len(df_oct19) > 0:
    print(f"    Time range: {df_oct19['timestamp'].iloc[0]} to {df_oct19['timestamp'].iloc[-1]}")
    print(f"    Price range: ${df_oct19['close'].min():,.2f} - ${df_oct19['close'].max():,.2f}")
    print(f"    Avg price: ${df_oct19['close'].mean():,.2f}")

if len(df_oct19) == 0:
    print("\n❌ No Oct 19 data found!")
    sys.exit(1)

# ============================================================================
# STEP 4: Calculate probabilities for all three dates
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
oct18_probs, oct18_timestamps, oct18_prices, oct18_macd, oct18_macd_diff = calculate_probs(df_oct18, "Oct 18")
oct19_probs, oct19_timestamps, oct19_prices, oct19_macd, oct19_macd_diff = calculate_probs(df_oct19, "Oct 19")

# ============================================================================
# STEP 5: Compare distributions
# ============================================================================
print("\n" + "="*80)
print("PROBABILITY DISTRIBUTION COMPARISON")
print("="*80)

# Overall dataset stats
overall_prices = df_features['close'].values
overall_mean_price = overall_prices.mean()

for date_name, probs, prices, macd, macd_diff in [
    ("SEP 23, 2025", sep23_probs, sep23_prices, sep23_macd, sep23_macd_diff),
    ("OCT 18, 2025", oct18_probs, oct18_prices, oct18_macd, oct18_macd_diff),
    ("OCT 19, 2025 (TODAY)", oct19_probs, oct19_prices, oct19_macd, oct19_macd_diff)
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
    print(f"\n  Price Analysis:")
    print(f"    Mean: ${mean_price:,.2f}")
    print(f"    vs Overall Avg: {(mean_price - overall_mean_price)/overall_mean_price*100:+.2f}%")

    print(f"\n  MACD Analysis:")
    print(f"    MACD Mean: {np.mean(macd):.2f}")
    print(f"    MACD_diff Mean: {np.mean(macd_diff):.2f}")

# ============================================================================
# STEP 6: Sample outputs (first 30 candles of Oct 19)
# ============================================================================
print(f"\n📋 Sample LONG Probabilities (first 30 candles of Oct 19):")
print(f"{'Timestamp':<20} {'Price':<12} {'LONG Prob':<12} {'MACD':<10} {'MACD_diff':<12} {'Threshold':<12}")
print("-"*100)

for i in range(min(30, len(oct19_timestamps))):
    ts = oct19_timestamps[i]
    price = oct19_prices[i]
    prob = oct19_probs[i]
    macd_val = oct19_macd[i]
    macd_diff_val = oct19_macd_diff[i]
    threshold = "✅ ENTRY" if prob >= 0.65 else "⏸ WAIT"
    print(f"{ts:<20} ${price:<11,.2f} {prob*100:<11.2f}% {macd_val:<9.2f} {macd_diff_val:<11.2f} {threshold:<12}")

# ============================================================================
# STEP 7: Hourly breakdown for Oct 19
# ============================================================================
print(f"\n⏰ HOURLY BREAKDOWN (OCT 19):")
print("="*80)

hourly_stats = []
for hour in range(24):
    hour_str = f"{hour:02d}"
    df_hour = df_oct19[df_oct19['timestamp'].str.contains(f' {hour_str}:')].copy()

    if len(df_hour) == 0:
        continue

    hour_probs = []
    hour_macd = []
    hour_macd_diff = []
    for idx in range(len(df_hour)):
        row = df_hour.iloc[idx:idx+1]
        features = row[long_feature_names].values
        scaled = long_scaler.transform(features)
        prob = long_model.predict_proba(scaled)[0][1]
        hour_probs.append(prob)
        hour_macd.append(row['macd'].iloc[0])
        hour_macd_diff.append(row['macd_diff'].iloc[0])

    hour_probs = np.array(hour_probs)

    hourly_stats.append({
        'hour': hour,
        'count': len(hour_probs),
        'mean_prob': hour_probs.mean(),
        'median_prob': np.median(hour_probs),
        'min_prob': hour_probs.min(),
        'max_prob': hour_probs.max(),
        'ge70': (hour_probs >= 0.7).sum(),
        'mean_macd': np.mean(hour_macd),
        'mean_macd_diff': np.mean(hour_macd_diff)
    })

print(f"\n{'Hour':<6} {'Count':<7} {'Mean%':<10} {'Median%':<10} {'Min%':<10} {'Max%':<10} {'≥70%':<10} {'MACD':<10} {'MACD_diff':<12}")
print("-"*110)

for stat in hourly_stats:
    print(f"{stat['hour']:02d}:00  {stat['count']:<7} {stat['mean_prob']*100:<9.1f}% {stat['median_prob']*100:<9.1f}% "
          f"{stat['min_prob']*100:<9.1f}% {stat['max_prob']*100:<9.1f}% {stat['ge70']}/{stat['count']:<9} "
          f"{stat['mean_macd']:<9.1f} {stat['mean_macd_diff']:<11.1f}")

# ============================================================================
# STEP 8: Comparison Summary
# ============================================================================
print(f"\n" + "="*80)
print("COMPARISON SUMMARY")
print("="*80)

print(f"\n📊 Average LONG Probability:")
print(f"  Sep 23: {sep23_probs.mean()*100:6.2f}% (NORMAL)")
print(f"  Oct 18: {oct18_probs.mean()*100:6.2f}% (HIGH)")
print(f"  Oct 19: {oct19_probs.mean()*100:6.2f}% ← TODAY")

print(f"\n💰 Price vs Overall Average:")
print(f"  Overall Avg: ${overall_mean_price:,.2f}")
print(f"  Sep 23: {(np.mean(sep23_prices) - overall_mean_price)/overall_mean_price*100:+.2f}%")
print(f"  Oct 18: {(np.mean(oct18_prices) - overall_mean_price)/overall_mean_price*100:+.2f}%")
print(f"  Oct 19: {(np.mean(oct19_prices) - overall_mean_price)/overall_mean_price*100:+.2f}%")

print(f"\n📈 MACD Analysis:")
print(f"  Sep 23 MACD: {np.mean(sep23_macd):8.2f}, MACD_diff: {np.mean(sep23_macd_diff):8.2f}")
print(f"  Oct 18 MACD: {np.mean(oct18_macd):8.2f}, MACD_diff: {np.mean(oct18_macd_diff):8.2f}")
print(f"  Oct 19 MACD: {np.mean(oct19_macd):8.2f}, MACD_diff: {np.mean(oct19_macd_diff):8.2f}")

# ============================================================================
# STEP 9: Final verdict
# ============================================================================
print(f"\n" + "="*80)
print("FINAL VERDICT - OCT 19")
print("="*80)

oct19_high = oct19_probs.mean() > 0.65
oct19_mean_prob = oct19_probs.mean() * 100

if oct19_mean_prob > 75:
    print(f"\n⚠️  OCT 19 shows HIGH probabilities:")
    print(f"   Average: {oct19_mean_prob:.2f}%")
    print(f"   ≥70% candles: {(oct19_probs >= 0.7).sum()/len(oct19_probs)*100:.1f}%")
    print(f"\n   → Similar to Oct 18 pattern")
    print(f"   → MACD indicators likely showing golden cross")
    print(f"   → Price below overall average = model sees mean reversion opportunity")
    print(f"\n   ⚠️  CAUTION: Oct 18 showed similar pattern but market continued down")
    print(f"   → Monitor if mean reversion actually happens")
elif oct19_mean_prob > 50:
    print(f"\n⚖️  OCT 19 shows MODERATE probabilities:")
    print(f"   Average: {oct19_mean_prob:.2f}%")
    print(f"   ≥65% candles: {(oct19_probs >= 0.65).sum()/len(oct19_probs)*100:.1f}%")
    print(f"\n   → Normal entry opportunities")
    print(f"   → Model showing balanced signals")
else:
    print(f"\n✅ OCT 19 shows NORMAL probabilities:")
    print(f"   Average: {oct19_mean_prob:.2f}%")
    print(f"   Similar to Sep 23 baseline")
    print(f"\n   → Model operating in normal range")

print("\n" + "="*80)
print("VERIFICATION COMPLETE")
print("="*80)
