"""
Deep Analysis of LONG Probability Behavior
- Probability trend over time
- Performance in high vs low probability periods
- Price relationship with probabilities
- Mean reversion bias check
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import pickle
import joblib
from datetime import datetime

from scripts.experiments.calculate_all_features import calculate_all_features
from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"

print("="*80)
print("DEEP PROBABILITY ANALYSIS")
print("="*80)

# Load data
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df = pd.read_csv(data_file)
print(f"\n✅ Loaded {len(df):,} candles")
print(f"   Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

# Calculate features
print("\nCalculating features...")
df = calculate_all_features(df)
df = prepare_exit_features(df)

# Load models
long_model_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146.pkl"
long_scaler_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146_scaler.pkl"
long_features_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146_features.txt"

with open(long_model_path, 'rb') as f:
    long_model = pickle.load(f)
long_scaler = joblib.load(long_scaler_path)
with open(long_features_path, 'r') as f:
    long_features = [line.strip() for line in f.readlines() if line.strip()]

# Calculate probabilities
print("\nCalculating probabilities...")
probabilities = []
for i in range(len(df)):
    try:
        features = df[long_features].iloc[i:i+1].values
        scaled = long_scaler.transform(features)
        prob = long_model.predict_proba(scaled)[0][1]
        probabilities.append(prob)
    except:
        probabilities.append(np.nan)

df['long_prob'] = probabilities
df = df.dropna(subset=['long_prob'])
print(f"✅ Probabilities calculated for {len(df):,} points")

# ============================================================================
# ANALYSIS 1: Probability Trend Over Time
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS 1: PROBABILITY TREND OVER TIME")
print("="*80)

# Monthly breakdown
df['month'] = pd.to_datetime(df['timestamp']).dt.to_period('M')
monthly = df.groupby('month').agg({
    'long_prob': ['mean', 'median', 'std', lambda x: (x >= 0.65).sum(), 'count'],
    'close': ['mean', 'min', 'max']
}).round(4)

print("\n📅 Monthly Probability Statistics:")
print(monthly.to_string())

# Weekly trend (last 12 weeks)
df['week'] = pd.to_datetime(df['timestamp']).dt.to_period('W')
weekly = df.groupby('week').agg({
    'long_prob': ['mean', 'median', lambda x: (x >= 0.70).sum() / len(x) * 100],
    'close': 'mean'
}).round(4)
weekly.columns = ['_'.join(col).strip() for col in weekly.columns.values]
print("\n📊 Weekly Trend (Last 12 weeks):")
print(weekly.tail(12).to_string())

# ============================================================================
# ANALYSIS 2: Price vs Probability Relationship
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS 2: PRICE vs PROBABILITY RELATIONSHIP")
print("="*80)

# Bin prices into ranges
df['price_bin'] = pd.cut(df['close'], bins=10)
price_prob = df.groupby('price_bin').agg({
    'long_prob': ['mean', 'median', 'count'],
    'close': 'mean'
}).round(4)

print("\n💰 Probability by Price Range:")
print(price_prob.to_string())

# Check correlation
correlation = df['close'].corr(df['long_prob'])
print(f"\n📈 Correlation (Price vs LONG Prob): {correlation:.4f}")

# ============================================================================
# ANALYSIS 3: Mean Reversion Pattern Check
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS 3: MEAN REVERSION PATTERN CHECK")
print("="*80)

# Calculate price deviation from rolling mean
df['price_ma_20'] = df['close'].rolling(20).mean()
df['price_dev'] = (df['close'] - df['price_ma_20']) / df['price_ma_20']

# Group by price deviation
df['dev_bin'] = pd.cut(df['price_dev'].dropna(), bins=[-1, -0.05, -0.03, -0.01, 0.01, 0.03, 0.05, 1],
                       labels=['<<-5%', '-5~-3%', '-3~-1%', '-1~1%', '1~3%', '3~5%', '>>5%'])

dev_prob = df.groupby('dev_bin').agg({
    'long_prob': ['mean', 'median', 'count'],
    'price_dev': 'mean'
}).round(4)

print("\n📉 Probability by Price Deviation from MA(20):")
print(dev_prob.to_string())

dev_correlation = df['price_dev'].corr(df['long_prob'])
print(f"\n📊 Correlation (Price Deviation vs LONG Prob): {dev_correlation:.4f}")

# ============================================================================
# ANALYSIS 4: High Probability Performance
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS 4: PERFORMANCE BY PROBABILITY LEVEL")
print("="*80)

# Simulate simple strategy: Enter LONG when prob >= threshold
def simulate_entries(threshold):
    entries = df[df['long_prob'] >= threshold].copy()

    if len(entries) == 0:
        return None

    # Calculate next 24 candles return (2 hours)
    returns = []
    for idx in entries.index:
        if idx + 24 < len(df):
            entry_price = df.loc[idx, 'close']
            future_prices = df.loc[idx+1:idx+24, 'close']
            max_return = (future_prices.max() - entry_price) / entry_price
            final_return = (df.loc[idx+24, 'close'] - entry_price) / entry_price
            returns.append({
                'max': max_return,
                'final': final_return,
                'win': final_return > 0
            })

    if not returns:
        return None

    return {
        'count': len(returns),
        'win_rate': sum(r['win'] for r in returns) / len(returns),
        'avg_return': np.mean([r['final'] for r in returns]),
        'avg_max': np.mean([r['max'] for r in returns])
    }

thresholds = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
print("\n🎯 Performance by Entry Threshold (24 candle = 2 hour hold):")
print(f"{'Threshold':<12} {'Entries':<10} {'Win Rate':<12} {'Avg Return':<12} {'Avg Max':<12}")
print("-" * 60)

for threshold in thresholds:
    result = simulate_entries(threshold)
    if result:
        print(f"{threshold:<12.2f} {result['count']:<10} {result['win_rate']*100:<12.1f}% "
              f"{result['avg_return']*100:<12.2f}% {result['avg_max']*100:<12.2f}%")

# ============================================================================
# ANALYSIS 5: Recent High Probability Period Analysis
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS 5: RECENT HIGH PROBABILITY PERIOD")
print("="*80)

# Last 1000 candles
recent = df.tail(1000)
high_prob_recent = recent[recent['long_prob'] >= 0.70]

print(f"\n📅 Recent Period (Last 1000 candles):")
print(f"  Date: {recent['timestamp'].iloc[0]} to {recent['timestamp'].iloc[-1]}")
print(f"  Price Range: ${recent['close'].min():,.0f} - ${recent['close'].max():,.0f}")
print(f"  Avg Price: ${recent['close'].mean():,.0f}")
print(f"  Price Trend: {(recent['close'].iloc[-1] - recent['close'].iloc[0]) / recent['close'].iloc[0] * 100:+.2f}%")

print(f"\n🔴 High Probability Points (>= 70%):")
print(f"  Count: {len(high_prob_recent)} ({len(high_prob_recent)/len(recent)*100:.1f}%)")
print(f"  Avg Prob: {high_prob_recent['long_prob'].mean()*100:.1f}%")
print(f"  Price when high prob: ${high_prob_recent['close'].mean():,.0f}")

# Compare with overall average
overall_avg_price = df['close'].mean()
print(f"\n💡 Comparison:")
print(f"  Overall avg price: ${overall_avg_price:,.0f}")
print(f"  Recent avg price: ${recent['close'].mean():,.0f} ({(recent['close'].mean() - overall_avg_price) / overall_avg_price * 100:+.2f}%)")
print(f"  Recent high-prob avg price: ${high_prob_recent['close'].mean():,.0f} ({(high_prob_recent['close'].mean() - overall_avg_price) / overall_avg_price * 100:+.2f}%)")

# ============================================================================
# ANALYSIS 6: Model Feature Importance
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS 6: TOP FEATURES DRIVING PREDICTIONS")
print("="*80)

# Get feature importance
importance = long_model.feature_importances_
feature_imp = pd.DataFrame({
    'feature': long_features,
    'importance': importance
}).sort_values('importance', ascending=False)

print("\n🎯 Top 15 Most Important Features:")
print(feature_imp.head(15).to_string(index=False))

# Check if price-related features dominate
price_features = [f for f in long_features if 'close' in f.lower() or 'price' in f.lower()]
price_importance = feature_imp[feature_imp['feature'].isin(price_features)]['importance'].sum()
print(f"\n💰 Price-related features importance: {price_importance:.4f} ({price_importance/importance.sum()*100:.1f}%)")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
