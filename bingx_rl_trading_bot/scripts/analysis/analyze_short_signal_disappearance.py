"""
Analyze why SHORT signals disappeared in recent market
Compare Oct mid-month (when SHORT worked) vs Nov 3-4 (no SHORT)
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.production_features_v1 import calculate_all_features_enhanced_v2
from scripts.production.production_exit_features_v1 import prepare_exit_features

print("=" * 80)
print("ANALYZING SHORT SIGNAL DISAPPEARANCE")
print("=" * 80)

# Load models
print("\n📦 Loading SHORT entry model...")
model_path = PROJECT_ROOT / 'models' / 'xgboost_short_entry_enhanced_20251024_012445.pkl'
features_path = PROJECT_ROOT / 'models' / 'xgboost_short_entry_enhanced_20251024_012445_features.txt'
scaler_path = PROJECT_ROOT / 'models' / 'xgboost_short_entry_enhanced_20251024_012445_scaler.pkl'

model = joblib.load(model_path)
with open(features_path) as f:
    features_list = [line.strip() for line in f.readlines()]
scaler = joblib.load(scaler_path)

print(f"✅ Model loaded: {len(features_list)} features")

# Load backtest data
print("\n📂 Loading backtest trades...")
backtest_df = pd.read_csv(PROJECT_ROOT / 'results' / 'backtest_28days_full_20251104_0142.csv')
backtest_df['entry_time'] = pd.to_datetime(backtest_df['entry_time'])

# Find periods
oct_short_period = backtest_df[
    (backtest_df['entry_time'] >= '2025-10-20') &
    (backtest_df['entry_time'] < '2025-10-25') &
    (backtest_df['side'] == 'SHORT')
]

nov_period = backtest_df[backtest_df['entry_time'] >= '2025-11-03']

print(f"\n📊 Trade Distribution:")
print(f"   Oct 20-25: {len(oct_short_period)} SHORT trades")
print(f"   Nov 3-4: {len(nov_period)} total trades ({len(nov_period[nov_period['side']=='SHORT'])} SHORT)")

# Load raw candle data
print("\n📂 Loading raw candle data...")
csv_file = PROJECT_ROOT / 'data' / 'features' / 'BTCUSDT_5m_raw_latest4weeks_20251104_014102.csv'

try:
    # Try to load with error handling
    df_raw = pd.read_csv(csv_file, on_bad_lines='skip')
    df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df_raw.columns:
            df_raw[col] = df_raw[col].astype(float)

    print(f"✅ Loaded: {len(df_raw)} candles")

    # Calculate features
    print("\n⏳ Calculating features...")
    df_features = calculate_all_features_enhanced_v2(df_raw.copy(), phase='phase1')
    df_features = prepare_exit_features(df_features)

    print(f"✅ Features calculated: {len(df_features)} rows")

    # Analyze Oct period (SHORT worked)
    print("\n" + "=" * 80)
    print("PERIOD 1: OCT 20-25 (SHORT WORKED)")
    print("=" * 80)

    oct_data = df_features[(df_features['timestamp'] >= '2025-10-20') & (df_features['timestamp'] < '2025-10-25')]

    if len(oct_data) > 0:
        # Calculate SHORT probabilities
        X_oct = oct_data[features_list].values
        X_oct_scaled = scaler.transform(X_oct)
        short_probs_oct = model.predict_proba(X_oct_scaled)[:, 1]

        oct_data_with_probs = oct_data.copy()
        oct_data_with_probs['short_prob'] = short_probs_oct

        # Statistics
        print(f"\n📊 SHORT Signal Statistics:")
        print(f"   Total candles: {len(oct_data)}")
        print(f"   Avg SHORT prob: {short_probs_oct.mean():.4f} ({short_probs_oct.mean()*100:.2f}%)")
        print(f"   Max SHORT prob: {short_probs_oct.max():.4f} ({short_probs_oct.max()*100:.2f}%)")
        print(f"   Min SHORT prob: {short_probs_oct.min():.4f} ({short_probs_oct.min()*100:.2f}%)")
        print(f"   >0.80 threshold: {(short_probs_oct >= 0.80).sum()} ({(short_probs_oct >= 0.80).sum()/len(oct_data)*100:.1f}%)")

        # Market conditions
        print(f"\n📈 Market Conditions:")
        print(f"   Price range: ${oct_data['close'].min():,.1f} - ${oct_data['close'].max():,.1f}")
        print(f"   Avg price: ${oct_data['close'].mean():,.1f}")
        print(f"   Price change: {(oct_data['close'].iloc[-1] - oct_data['close'].iloc[0]) / oct_data['close'].iloc[0] * 100:+.2f}%")

        # Top features for highest SHORT signal
        if short_probs_oct.max() >= 0.80:
            best_idx = short_probs_oct.argmax()
            best_row = oct_data.iloc[best_idx]
            best_features = X_oct[best_idx]

            print(f"\n🎯 Highest SHORT Signal Example:")
            print(f"   Timestamp: {best_row['timestamp']}")
            print(f"   Close: ${best_row['close']:,.1f}")
            print(f"   SHORT prob: {short_probs_oct[best_idx]:.4f} ({short_probs_oct[best_idx]*100:.2f}%)")

            # Get feature importance
            feature_importance = model.feature_importances_
            top_features_idx = np.argsort(feature_importance)[-10:][::-1]

            print(f"\n   Top 10 Feature Values:")
            for idx in top_features_idx:
                feat_name = features_list[idx]
                feat_value = best_features[idx]
                importance = feature_importance[idx]
                print(f"      {feat_name:30s} = {feat_value:>10.4f} (imp: {importance:.4f})")

    # Analyze Nov period (SHORT disappeared)
    print("\n" + "=" * 80)
    print("PERIOD 2: NOV 3-4 (SHORT DISAPPEARED)")
    print("=" * 80)

    nov_data = df_features[df_features['timestamp'] >= '2025-11-03']

    if len(nov_data) > 0:
        # Calculate SHORT probabilities
        X_nov = nov_data[features_list].values
        X_nov_scaled = scaler.transform(X_nov)
        short_probs_nov = model.predict_proba(X_nov_scaled)[:, 1]

        nov_data_with_probs = nov_data.copy()
        nov_data_with_probs['short_prob'] = short_probs_nov

        # Statistics
        print(f"\n📊 SHORT Signal Statistics:")
        print(f"   Total candles: {len(nov_data)}")
        print(f"   Avg SHORT prob: {short_probs_nov.mean():.4f} ({short_probs_nov.mean()*100:.2f}%)")
        print(f"   Max SHORT prob: {short_probs_nov.max():.4f} ({short_probs_nov.max()*100:.2f}%)")
        print(f"   Min SHORT prob: {short_probs_nov.min():.4f} ({short_probs_nov.min()*100:.2f}%)")
        print(f"   >0.80 threshold: {(short_probs_nov >= 0.80).sum()} ({(short_probs_nov >= 0.80).sum()/len(nov_data)*100:.1f}%)")

        # Market conditions
        print(f"\n📈 Market Conditions:")
        print(f"   Price range: ${nov_data['close'].min():,.1f} - ${nov_data['close'].max():,.1f}")
        print(f"   Avg price: ${nov_data['close'].mean():,.1f}")
        print(f"   Price change: {(nov_data['close'].iloc[-1] - nov_data['close'].iloc[0]) / nov_data['close'].iloc[0] * 100:+.2f}%")

        # Example when price was lowest (~$103k)
        low_price_idx = nov_data['close'].idxmin()
        low_price_row = nov_data.loc[low_price_idx]
        low_price_features = X_nov[nov_data.index.get_loc(low_price_idx)]
        low_price_short_prob = short_probs_nov[nov_data.index.get_loc(low_price_idx)]

        print(f"\n❌ Lowest Price Example (should be SHORT opportunity!):")
        print(f"   Timestamp: {low_price_row['timestamp']}")
        print(f"   Close: ${low_price_row['close']:,.1f}")
        print(f"   SHORT prob: {low_price_short_prob:.4f} ({low_price_short_prob*100:.2f}%) ← TOO LOW!")

        # Get feature importance
        feature_importance = model.feature_importances_
        top_features_idx = np.argsort(feature_importance)[-10:][::-1]

        print(f"\n   Top 10 Feature Values:")
        for idx in top_features_idx:
            feat_name = features_list[idx]
            feat_value = low_price_features[idx]
            importance = feature_importance[idx]
            print(f"      {feat_name:30s} = {feat_value:>10.4f} (imp: {importance:.4f})")

    # Comparison
    if len(oct_data) > 0 and len(nov_data) > 0:
        print("\n" + "=" * 80)
        print("COMPARISON: WHY SHORT DISAPPEARED")
        print("=" * 80)

        print(f"\n📊 SHORT Probability Change:")
        print(f"   Oct avg: {short_probs_oct.mean():.4f} → Nov avg: {short_probs_nov.mean():.4f}")
        print(f"   Difference: {short_probs_nov.mean() - short_probs_oct.mean():.4f} ({(short_probs_nov.mean() - short_probs_oct.mean())*100:.2f}%)")

        print(f"\n📈 Market Condition Change:")
        print(f"   Oct avg price: ${oct_data['close'].mean():,.1f}")
        print(f"   Nov avg price: ${nov_data['close'].mean():,.1f}")
        print(f"   Difference: ${nov_data['close'].mean() - oct_data['close'].mean():,.1f} ({(nov_data['close'].mean() - oct_data['close'].mean()) / oct_data['close'].mean() * 100:+.2f}%)")

        # Feature comparison
        print(f"\n🔍 Key Feature Changes (Top 10 features):")
        feature_importance = model.feature_importances_
        top_features_idx = np.argsort(feature_importance)[-10:][::-1]

        for idx in top_features_idx:
            feat_name = features_list[idx]
            oct_mean = X_oct[:, idx].mean()
            nov_mean = X_nov[:, idx].mean()
            diff = nov_mean - oct_mean
            pct_change = (diff / oct_mean * 100) if oct_mean != 0 else 0

            print(f"   {feat_name:30s}: {oct_mean:>8.2f} → {nov_mean:>8.2f} ({pct_change:+6.1f}%)")

except Exception as e:
    print(f"\n❌ Error loading/processing data: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
