"""
Verify: Trade-Outcome model probabilities for Sep 23 data
Compare with Oct 18 to see if high probabilities are date-specific
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
print("VERIFY: TRADE-OUTCOME MODEL - SEP 23 vs OCT 18 COMPARISON")
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
# STEP 3: Filter Sep 23 and Oct 18 data
# ============================================================================
print("\n📅 Filtering dates...")

df_sep23 = df_features[df_features['timestamp'].str.startswith('2025-09-23')].copy()
df_oct18 = df_features[df_features['timestamp'].str.startswith('2025-10-18')].copy()

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

if len(df_sep23) == 0:
    print("\n❌ No Sep 23 data found!")
    sys.exit(1)

# ============================================================================
# STEP 4: Calculate probabilities for both dates
# ============================================================================
def calculate_probs(df_date, date_name):
    """Calculate LONG probabilities for a given date"""
    probs = []
    timestamps = []
    prices = []

    for idx in range(len(df_date)):
        row = df_date.iloc[idx:idx+1]
        features = row[long_feature_names].values
        scaled = long_scaler.transform(features)
        prob = long_model.predict_proba(scaled)[0][1]
        probs.append(prob)
        timestamps.append(row['timestamp'].iloc[0])
        prices.append(row['close'].iloc[0])

    return np.array(probs), timestamps, prices

print("\n🔬 Calculating LONG probabilities...")

sep23_probs, sep23_timestamps, sep23_prices = calculate_probs(df_sep23, "Sep 23")
oct18_probs, oct18_timestamps, oct18_prices = calculate_probs(df_oct18, "Oct 18")

# ============================================================================
# STEP 5: Compare distributions
# ============================================================================
print("\n" + "="*80)
print("PROBABILITY DISTRIBUTION COMPARISON")
print("="*80)

print(f"\n📊 SEP 23, 2025:")
print(f"  Candles: {len(sep23_probs)}")
print(f"  Mean: {sep23_probs.mean():.4f} ({sep23_probs.mean()*100:.2f}%)")
print(f"  Median: {np.median(sep23_probs):.4f} ({np.median(sep23_probs)*100:.2f}%)")
print(f"  Std Dev: {sep23_probs.std():.4f}")
print(f"  Min: {sep23_probs.min():.4f} ({sep23_probs.min()*100:.2f}%)")
print(f"  Max: {sep23_probs.max():.4f} ({sep23_probs.max()*100:.2f}%)")

print(f"\n  Distribution:")
print(f"    ≥90%: {(sep23_probs >= 0.9).sum():4d} ({(sep23_probs >= 0.9).sum()/len(sep23_probs)*100:5.1f}%)")
print(f"    ≥80%: {(sep23_probs >= 0.8).sum():4d} ({(sep23_probs >= 0.8).sum()/len(sep23_probs)*100:5.1f}%)")
print(f"    ≥70%: {(sep23_probs >= 0.7).sum():4d} ({(sep23_probs >= 0.7).sum()/len(sep23_probs)*100:5.1f}%)")
print(f"    ≥65%: {(sep23_probs >= 0.65).sum():4d} ({(sep23_probs >= 0.65).sum()/len(sep23_probs)*100:5.1f}%)")
print(f"    <65%: {(sep23_probs < 0.65).sum():4d} ({(sep23_probs < 0.65).sum()/len(sep23_probs)*100:5.1f}%)")

print(f"\n📊 OCT 18, 2025:")
print(f"  Candles: {len(oct18_probs)}")
print(f"  Mean: {oct18_probs.mean():.4f} ({oct18_probs.mean()*100:.2f}%)")
print(f"  Median: {np.median(oct18_probs):.4f} ({np.median(oct18_probs)*100:.2f}%)")
print(f"  Std Dev: {oct18_probs.std():.4f}")
print(f"  Min: {oct18_probs.min():.4f} ({oct18_probs.min()*100:.2f}%)")
print(f"  Max: {oct18_probs.max():.4f} ({oct18_probs.max()*100:.2f}%)")

print(f"\n  Distribution:")
print(f"    ≥90%: {(oct18_probs >= 0.9).sum():4d} ({(oct18_probs >= 0.9).sum()/len(oct18_probs)*100:5.1f}%)")
print(f"    ≥80%: {(oct18_probs >= 0.8).sum():4d} ({(oct18_probs >= 0.8).sum()/len(oct18_probs)*100:5.1f}%)")
print(f"    ≥70%: {(oct18_probs >= 0.7).sum():4d} ({(oct18_probs >= 0.7).sum()/len(oct18_probs)*100:5.1f}%)")
print(f"    ≥65%: {(oct18_probs >= 0.65).sum():4d} ({(oct18_probs >= 0.65).sum()/len(oct18_probs)*100:5.1f}%)")
print(f"    <65%: {(oct18_probs < 0.65).sum():4d} ({(oct18_probs < 0.65).sum()/len(oct18_probs)*100:5.1f}%)")

print(f"\n🔍 DIFFERENCE:")
print(f"  Mean difference: {abs(sep23_probs.mean() - oct18_probs.mean()):.4f} ({abs(sep23_probs.mean() - oct18_probs.mean())*100:.2f}%)")
print(f"  Median difference: {abs(np.median(sep23_probs) - np.median(oct18_probs)):.4f}")

# ============================================================================
# STEP 6: Sample outputs
# ============================================================================
print(f"\n📋 Sample LONG Probabilities (first 20 candles):")
print("\n  SEP 23:")
print(f"  {'Timestamp':<20} {'Price':<12} {'LONG Prob':<12} {'Threshold':<12}")
print("  " + "-"*70)

for i in range(min(20, len(sep23_timestamps))):
    ts = sep23_timestamps[i]
    price = sep23_prices[i]
    prob = sep23_probs[i]
    threshold = "✅ ENTRY" if prob >= 0.65 else "⏸ WAIT"
    print(f"  {ts:<20} ${price:<11,.2f} {prob*100:<11.2f}% {threshold:<12}")

print(f"\n  OCT 18:")
print(f"  {'Timestamp':<20} {'Price':<12} {'LONG Prob':<12} {'Threshold':<12}")
print("  " + "-"*70)

for i in range(min(20, len(oct18_timestamps))):
    ts = oct18_timestamps[i]
    price = oct18_prices[i]
    prob = oct18_probs[i]
    threshold = "✅ ENTRY" if prob >= 0.65 else "⏸ WAIT"
    print(f"  {ts:<20} ${price:<11,.2f} {prob*100:<11.2f}% {threshold:<12}")

# ============================================================================
# STEP 7: Price correlation analysis
# ============================================================================
print(f"\n💰 PRICE ANALYSIS:")
print("="*80)

overall_prices = df_features['close'].values
overall_mean = overall_prices.mean()

sep23_mean_price = df_sep23['close'].mean()
oct18_mean_price = df_oct18['close'].mean()

print(f"\n  Overall dataset:")
print(f"    Mean price: ${overall_mean:,.2f}")

print(f"\n  Sep 23:")
print(f"    Mean price: ${sep23_mean_price:,.2f}")
print(f"    vs Overall: {(sep23_mean_price - overall_mean)/overall_mean*100:+.2f}%")
print(f"    LONG prob mean: {sep23_probs.mean()*100:.2f}%")

print(f"\n  Oct 18:")
print(f"    Mean price: ${oct18_mean_price:,.2f}")
print(f"    vs Overall: {(oct18_mean_price - overall_mean)/overall_mean*100:+.2f}%")
print(f"    LONG prob mean: {oct18_probs.mean()*100:.2f}%")

# ============================================================================
# STEP 8: Final verdict
# ============================================================================
print(f"\n" + "="*80)
print("FINAL VERDICT")
print("="*80)

sep23_high = sep23_probs.mean() > 0.5
oct18_high = oct18_probs.mean() > 0.5

if sep23_high and oct18_high:
    print(f"\n✅ BOTH DATES show HIGH probabilities:")
    print(f"   Sep 23: {sep23_probs.mean()*100:.2f}% average")
    print(f"   Oct 18: {oct18_probs.mean()*100:.2f}% average")
    print(f"\n   → Model consistently outputs high probabilities for low-price periods")
    print(f"   → This is normal model behavior across different dates")
elif not sep23_high and oct18_high:
    print(f"\n⚠️  ONLY OCT 18 shows HIGH probabilities:")
    print(f"   Sep 23: {sep23_probs.mean()*100:.2f}% average (NORMAL)")
    print(f"   Oct 18: {oct18_probs.mean()*100:.2f}% average (HIGH)")
    print(f"\n   → Oct 18 is an anomaly")
    print(f"   → Model behavior may be date-specific")
elif sep23_high and not oct18_high:
    print(f"\n⚠️  ONLY SEP 23 shows HIGH probabilities:")
    print(f"   Sep 23: {sep23_probs.mean()*100:.2f}% average (HIGH)")
    print(f"   Oct 18: {oct18_probs.mean()*100:.2f}% average (NORMAL)")
    print(f"\n   → Sep 23 is an anomaly")
else:
    print(f"\n✅ BOTH DATES show NORMAL probabilities:")
    print(f"   Sep 23: {sep23_probs.mean()*100:.2f}% average")
    print(f"   Oct 18: {oct18_probs.mean()*100:.2f}% average")
    print(f"\n   → Model outputs normal probability range")

print("\n" + "="*80)
print("VERIFICATION COMPLETE")
print("="*80)
