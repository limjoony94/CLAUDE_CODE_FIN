"""
Validate Production Code - Probability Calculation Test
Directly execute production code logic to verify it matches backtest
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys
import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import PRODUCTION functions (same as bot uses)
from scripts.production.production_features_v1 import calculate_all_features_enhanced_v2
from scripts.production.production_exit_features_v1 import prepare_exit_features
from src.api.bingx_client import BingXClient

print("=" * 80)
print("PRODUCTION CODE VALIDATION - PROBABILITY CALCULATION")
print("=" * 80)

# ====================
# 1. Load config (same as production)
# ====================
config_path = PROJECT_ROOT / "config" / "api_keys.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

client = BingXClient(
    api_key=config['bingx']['mainnet']['api_key'],
    secret_key=config['bingx']['mainnet']['secret_key']
)

# ====================
# 2. Fetch candles (PRODUCTION METHOD)
# ====================
SYMBOL = "BTC-USDT"
INTERVAL = "5m"
MAX_DATA_CANDLES = 1000  # PRODUCTION VALUE (line 106)

print(f"\n📥 Fetching {MAX_DATA_CANDLES} candles (PRODUCTION METHOD)...")
klines = client.get_klines(SYMBOL, INTERVAL, limit=MAX_DATA_CANDLES)
df = pd.DataFrame(klines)
df['timestamp'] = pd.to_datetime(df['time'], unit='ms', utc=True).dt.tz_localize(None)

# Convert types (same as production)
for col in ['open', 'high', 'low', 'close', 'volume']:
    df[col] = df[col].astype(float)

print(f"✅ Fetched: {len(df)} candles")
print(f"   Range: {df['timestamp'].min()} to {df['timestamp'].max()}")

# ====================
# 3. Calculate features (PRODUCTION METHOD)
# ====================
print("\n🔧 Calculating features (PRODUCTION METHOD)...")
print("   Using: calculate_all_features_enhanced_v2(phase='phase1')")
df_features = calculate_all_features_enhanced_v2(df.copy(), phase='phase1')

print("   Using: prepare_exit_features()")
df_features = prepare_exit_features(df_features)

print(f"✅ Features calculated: {len(df_features)} rows")
print(f"   Lost {len(df) - len(df_features)} rows to lookback")

# ====================
# 4. Load models (PRODUCTION METHOD)
# ====================
print("\n📦 Loading models (PRODUCTION METHOD)...")
MODELS_DIR = PROJECT_ROOT / "models"

# Entry models (same as production lines 184-201)
long_entry_model_path = MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445.pkl"
long_entry_scaler_path = MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_scaler.pkl"
long_entry_features_path = MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_features.txt"

long_entry_model = joblib.load(long_entry_model_path)
long_entry_scaler = joblib.load(long_entry_scaler_path)
with open(long_entry_features_path, 'r') as f:
    long_entry_features = [line.strip() for line in f.readlines()]

print(f"✅ LONG Entry model loaded: {len(long_entry_features)} features")

# ====================
# 5. Find Nov 1 19:40 (if exists)
# ====================
target_time = pd.to_datetime('2025-11-01 19:40:00')

print("\n" + "=" * 80)
print("TESTING NOV 1 19:40:00 (if available in current data)")
print("=" * 80)

if target_time in df_features['timestamp'].values:
    row = df_features[df_features['timestamp'] == target_time].iloc[0]
    row_idx = df_features[df_features['timestamp'] == target_time].index[0]

    print(f"\n✅ Nov 1 19:40 found at row {row_idx}")
    print(f"   Close price: ${row['close']:,.2f}")

    # ====================
    # 6. Calculate probability (PRODUCTION METHOD)
    # ====================
    # Extract features (same as production line 1196-1203)
    long_features_dict = {}
    for feat in long_entry_features:
        if feat in row.index:
            long_features_dict[feat] = row[feat]
        else:
            print(f"   ⚠️  Missing feature: {feat}")
            long_features_dict[feat] = 0.0

    long_feat = [[long_features_dict[f] for f in long_entry_features]]

    # Scale (PRODUCTION line 1204)
    long_scaled = long_entry_scaler.transform(long_feat)

    # Predict (PRODUCTION line 1205)
    long_prob = long_entry_model.predict_proba(long_scaled)[0][1]

    print(f"\n🎯 PRODUCTION PROBABILITY CALCULATION:")
    print(f"   LONG probability: {long_prob:.6f} ({long_prob*100:.2f}%)")
    print(f"   Entry threshold: 0.80 (80%)")
    print(f"   Decision: {'✅ ENTER' if long_prob >= 0.80 else '❌ NOT ENTER'}")

else:
    print(f"\n❌ Nov 1 19:40 NOT in current data")
    print(f"   Data range: {df_features['timestamp'].min()} to {df_features['timestamp'].max()}")
    print(f"   (Data is from recent API fetch, Nov 1 19:40 may be too old)")

# ====================
# 7. Test with most recent candle
# ====================
print("\n" + "=" * 80)
print("TESTING MOST RECENT CANDLE")
print("=" * 80)

row = df_features.iloc[-1]
timestamp = row['timestamp']
close = row['close']

print(f"\n📊 Most recent candle:")
print(f"   Timestamp: {timestamp}")
print(f"   Close: ${close:,.2f}")

# Extract features
long_features_dict = {}
for feat in long_entry_features:
    if feat in row.index:
        long_features_dict[feat] = row[feat]
    else:
        long_features_dict[feat] = 0.0

long_feat = [[long_features_dict[f] for f in long_entry_features]]

# Scale and predict (PRODUCTION METHOD)
long_scaled = long_entry_scaler.transform(long_feat)
long_prob = long_entry_model.predict_proba(long_scaled)[0][1]

print(f"\n🎯 PRODUCTION PROBABILITY CALCULATION:")
print(f"   LONG probability: {long_prob:.6f} ({long_prob*100:.2f}%)")
print(f"   Entry threshold: 0.80 (80%)")
print(f"   Decision: {'✅ ENTER' if long_prob >= 0.80 else '❌ NOT ENTER'}")

# ====================
# 8. Compare with CSV-based calculation (if Nov 1 19:40 available)
# ====================
if target_time in df_features['timestamp'].values:
    print("\n" + "=" * 80)
    print("COMPARING WITH CSV-BASED BACKTEST")
    print("=" * 80)

    print("\n📂 Loading 28-day CSV data...")
    csv_file = PROJECT_ROOT / "data" / "features" / "BTCUSDT_5m_raw_latest4weeks_20251104_014102.csv"
    df_csv = pd.read_csv(csv_file)
    df_csv['timestamp'] = pd.to_datetime(df_csv['timestamp'])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df_csv[col] = df_csv[col].astype(float)

    print("   Calculating features from CSV...")
    df_csv_features = calculate_all_features_enhanced_v2(df_csv.copy(), phase='phase1')
    df_csv_features = prepare_exit_features(df_csv_features)

    if target_time in df_csv_features['timestamp'].values:
        row_csv = df_csv_features[df_csv_features['timestamp'] == target_time].iloc[0]

        # Calculate probability
        long_features_dict_csv = {}
        for feat in long_entry_features:
            if feat in row_csv.index:
                long_features_dict_csv[feat] = row_csv[feat]
            else:
                long_features_dict_csv[feat] = 0.0

        long_feat_csv = [[long_features_dict_csv[f] for f in long_entry_features]]
        long_scaled_csv = long_entry_scaler.transform(long_feat_csv)
        long_prob_csv = long_entry_model.predict_proba(long_scaled_csv)[0][1]

        print(f"\n📊 COMPARISON:")
        print(f"   API (Production): {long_prob:.6f} ({long_prob*100:.2f}%)")
        print(f"   CSV (Backtest):   {long_prob_csv:.6f} ({long_prob_csv*100:.2f}%)")
        print(f"   Difference:       {abs(long_prob - long_prob_csv):.6f} ({abs(long_prob - long_prob_csv)*100:.2f}%)")

        if abs(long_prob - long_prob_csv) < 0.01:
            print(f"\n✅ VALIDATION PASSED: Probabilities match (<1% difference)")
        else:
            print(f"\n⚠️  WARNING: Probabilities differ by >1%")
    else:
        print(f"   Nov 1 19:40 not in CSV data")

print("\n" + "=" * 80)
print("VALIDATION COMPLETE")
print("=" * 80)
