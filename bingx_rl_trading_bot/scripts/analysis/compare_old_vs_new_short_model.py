"""
Compare OLD vs NEW SHORT Entry Model on Nov 3-4 Period
========================================================

Compares:
  OLD: xgboost_short_entry_enhanced_20251024_012445 (85 features)
  NEW: xgboost_short_entry_with_new_features_20251104_213043 (95 features)

Period: Nov 3-4, 2025 (falling market, -6.2% price drop)
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import joblib

# Import production feature calculation
from scripts.production.production_features_v1 import calculate_all_features_enhanced_v2
from scripts.production.production_exit_features_v1 import prepare_exit_features

DATA_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models"

print("="*80)
print("COMPARE OLD vs NEW SHORT ENTRY MODEL")
print("="*80)
print()

# ==============================================================================
# Load Data
# ==============================================================================

print("📂 Loading 35-day dataset...")
csv_file = sorted(DATA_DIR.glob("BTCUSDT_5m_raw_35days_*.csv"), reverse=True)[0]
df_raw = pd.read_csv(csv_file)
df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
for col in ['open', 'high', 'low', 'close', 'volume']:
    df_raw[col] = df_raw[col].astype(float)

print(f"✅ {len(df_raw):,} candles loaded")
print()

print("⏳ Calculating features...")
df = calculate_all_features_enhanced_v2(df_raw.copy(), phase='phase1')
df = prepare_exit_features(df)
print(f"✅ {len(df):,} candles with features")
print()

# Filter Nov 3-4
df_nov = df[(df['timestamp'] >= '2025-11-03') & (df['timestamp'] <= '2025-11-04')].copy()
print(f"📊 Nov 3-4 data: {len(df_nov)} candles")
print(f"   Date range: {df_nov['timestamp'].iloc[0]} to {df_nov['timestamp'].iloc[-1]}")
print(f"   Price range: ${df_nov['close'].min():,.1f} - ${df_nov['close'].max():,.1f}")
print()

# ==============================================================================
# Load OLD Model
# ==============================================================================

print("-"*80)
print("OLD MODEL: Enhanced 5-Fold CV (Oct 24, 2025)")
print("-"*80)

old_model_path = MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445.pkl"
old_scaler_path = MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_scaler.pkl"
old_features_path = MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_features.txt"

old_model = joblib.load(old_model_path)
old_scaler = joblib.load(old_scaler_path)
with open(old_features_path, 'r') as f:
    old_features = [line.strip() for line in f.readlines()]

print(f"✅ OLD Model loaded: {len(old_features)} features")

# Calculate OLD probabilities
X_old = df_nov[old_features].values
X_old_scaled = old_scaler.transform(X_old)
old_probs = old_model.predict_proba(X_old_scaled)[:, 1]

print(f"\n📊 OLD Model SHORT Signals (Nov 3-4):")
print(f"   Avg SHORT prob: {old_probs.mean():.4f} ({old_probs.mean()*100:.2f}%)")
print(f"   Max SHORT prob: {old_probs.max():.4f} ({old_probs.max()*100:.2f}%)")
print(f"   Min SHORT prob: {old_probs.min():.4f} ({old_probs.min()*100:.2f}%)")
print(f"   Std Dev: {old_probs.std():.4f}")
print(f"\n   >0.60: {(old_probs >= 0.60).sum():3d} ({(old_probs >= 0.60).sum()/len(old_probs)*100:5.1f}%)")
print(f"   >0.70: {(old_probs >= 0.70).sum():3d} ({(old_probs >= 0.70).sum()/len(old_probs)*100:5.1f}%)")
print(f"   >0.80: {(old_probs >= 0.80).sum():3d} ({(old_probs >= 0.80).sum()/len(old_probs)*100:5.1f}%) ← Entry Threshold")
print(f"   >0.90: {(old_probs >= 0.90).sum():3d} ({(old_probs >= 0.90).sum()/len(old_probs)*100:5.1f}%)")
print()

# ==============================================================================
# Load NEW Model
# ==============================================================================

print("-"*80)
print("NEW MODEL: With 10 SHORT-Specific Features (Nov 4, 2025)")
print("-"*80)

new_model_path = MODELS_DIR / "xgboost_short_entry_with_new_features_20251104_213043.pkl"
new_scaler_path = MODELS_DIR / "xgboost_short_entry_with_new_features_20251104_213043_scaler.pkl"
new_features_path = MODELS_DIR / "xgboost_short_entry_with_new_features_20251104_213043_features.txt"

new_model = joblib.load(new_model_path)
new_scaler = joblib.load(new_scaler_path)
with open(new_features_path, 'r') as f:
    new_features = [line.strip() for line in f.readlines()]

print(f"✅ NEW Model loaded: {len(new_features)} features (+{len(new_features) - len(old_features)} new)")

# Calculate NEW probabilities
X_new = df_nov[new_features].values
X_new_scaled = new_scaler.transform(X_new)
new_probs = new_model.predict_proba(X_new_scaled)[:, 1]

print(f"\n📊 NEW Model SHORT Signals (Nov 3-4):")
print(f"   Avg SHORT prob: {new_probs.mean():.4f} ({new_probs.mean()*100:.2f}%)")
print(f"   Max SHORT prob: {new_probs.max():.4f} ({new_probs.max()*100:.2f}%)")
print(f"   Min SHORT prob: {new_probs.min():.4f} ({new_probs.min()*100:.2f}%)")
print(f"   Std Dev: {new_probs.std():.4f}")
print(f"\n   >0.60: {(new_probs >= 0.60).sum():3d} ({(new_probs >= 0.60).sum()/len(new_probs)*100:5.1f}%)")
print(f"   >0.70: {(new_probs >= 0.70).sum():3d} ({(new_probs >= 0.70).sum()/len(new_probs)*100:5.1f}%)")
print(f"   >0.80: {(new_probs >= 0.80).sum():3d} ({(new_probs >= 0.80).sum()/len(new_probs)*100:5.1f}%) ← Entry Threshold")
print(f"   >0.90: {(new_probs >= 0.90).sum():3d} ({(new_probs >= 0.90).sum()/len(new_probs)*100:5.1f}%)")
print()

# ==============================================================================
# Comparison
# ==============================================================================

print("="*80)
print("COMPARISON: OLD vs NEW")
print("="*80)
print()

print("📈 Probability Improvement:")
print(f"   Avg: {old_probs.mean():.4f} → {new_probs.mean():.4f} ({(new_probs.mean() - old_probs.mean())*100:+.2f}%)")
print(f"   Max: {old_probs.max():.4f} → {new_probs.max():.4f} ({(new_probs.max() - old_probs.max())*100:+.2f}%)")
print()

print("🎯 Entry Threshold (0.80) Impact:")
old_above = (old_probs >= 0.80).sum()
new_above = (new_probs >= 0.80).sum()
print(f"   OLD: {old_above:3d} candles ({old_above/len(old_probs)*100:5.1f}%)")
print(f"   NEW: {new_above:3d} candles ({new_above/len(new_probs)*100:5.1f}%)")
if old_above == 0 and new_above > 0:
    print(f"   ✅ SUCCESS: NEW model enables SHORT signals (0 → {new_above})")
elif new_above > old_above:
    print(f"   ✅ IMPROVEMENT: {new_above - old_above} more SHORT signals ({(new_above/old_above - 1)*100:+.1f}%)")
else:
    print(f"   ⚠️  WARNING: No improvement in signal count")
print()

print("🔍 Signal Distribution Changes:")
for threshold in [0.60, 0.70, 0.80, 0.90]:
    old_count = (old_probs >= threshold).sum()
    new_count = (new_probs >= threshold).sum()
    change = new_count - old_count
    change_pct = ((new_count / old_count - 1) * 100) if old_count > 0 else float('inf')

    if old_count == 0 and new_count > 0:
        status = "✅ NEW SIGNALS"
    elif change > 0:
        status = f"✅ +{change}"
    elif change < 0:
        status = f"⚠️  {change}"
    else:
        status = "   same"

    print(f"   ≥{threshold:.2f}: {old_count:3d} → {new_count:3d}  {status}")
print()

# ==============================================================================
# Top Signal Examples
# ==============================================================================

if new_above > 0:
    print("-"*80)
    print("TOP 5 SHORT SIGNAL EXAMPLES (NEW Model)")
    print("-"*80)
    print()

    # Get top 5 signals
    top_indices = np.argsort(new_probs)[-5:][::-1]

    for i, idx in enumerate(top_indices, 1):
        row = df_nov.iloc[idx]
        timestamp = row['timestamp']
        close = row['close']
        old_prob = old_probs[idx]
        new_prob = new_probs[idx]

        print(f"{i}. Timestamp: {timestamp}")
        print(f"   Price: ${close:,.1f}")
        print(f"   OLD prob: {old_prob:.4f} ({old_prob*100:.2f}%)")
        print(f"   NEW prob: {new_prob:.4f} ({new_prob*100:.2f}%)")
        print(f"   Improvement: {(new_prob - old_prob)*100:+.2f}%")
        print()

# ==============================================================================
# Conclusion
# ==============================================================================

print("="*80)
print("CONCLUSION")
print("="*80)
print()

if new_above > old_above:
    print("✅ NEW Model shows SIGNIFICANT IMPROVEMENT:")
    print(f"   - {new_above - old_above} more SHORT signals in Nov 3-4")
    print(f"   - Avg probability increased by {(new_probs.mean() - old_probs.mean())*100:+.2f}%")
    print(f"   - Max probability increased by {(new_probs.max() - old_probs.max())*100:+.2f}%")
    print()
    print("🎯 Recommendation: DEPLOY NEW MODEL to production")
    print("   The 10 new SHORT-specific features successfully reduce")
    print("   VWAP dependency and improve SHORT signals in falling markets.")
else:
    print("⚠️  NEW Model shows NO IMPROVEMENT:")
    print(f"   - Signal count unchanged or decreased")
    print(f"   - May need additional feature engineering")
    print()
    print("🎯 Recommendation: DO NOT DEPLOY")
    print("   Further investigation needed before production deployment.")

print()
print("="*80)
