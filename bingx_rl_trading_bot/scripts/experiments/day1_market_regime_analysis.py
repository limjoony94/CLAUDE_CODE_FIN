"""
Day 1 Morning - Market Regime Analysis
======================================

Analyze market regimes in training and test sets to identify:
1. Volatility regimes (low/medium/high)
2. Trend regimes (strong_bear to strong_bull)
3. Volume regimes (low/medium/high)
4. Market phases (accumulation/markup/distribution/markdown)

Goal: Understand why SHORT model fails on test set
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.calculate_all_features import calculate_all_features

# Paths
DATA_DIR = PROJECT_ROOT / "data" / "historical"
RESULTS_DIR = PROJECT_ROOT / "results" / "regime_analysis"
RESULTS_DIR.mkdir(exist_ok=True)

print("="*80)
print("DAY 1 MORNING - MARKET REGIME ANALYSIS")
print("="*80)

# ============================================================================
# STEP 1: Load Data
# ============================================================================

print("\n" + "="*80)
print("STEP 1: Loading Data")
print("="*80)

data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df = pd.read_csv(data_file)
print(f"\n✅ Loaded {len(df):,} candles")
print(f"   Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

# Calculate basic features
print("\nCalculating features...")
df = calculate_all_features(df)
df = df.dropna().reset_index(drop=True)

# Train/Test split
split_idx = int(len(df) * 0.8)
df['split'] = 'train'
df.loc[split_idx:, 'split'] = 'test'

train_df = df[:split_idx].copy()
test_df = df[split_idx:].copy()

print(f"✅ Features calculated")
print(f"\n  Train: {len(train_df):,} candles ({df['timestamp'].iloc[0]} → {df['timestamp'].iloc[split_idx-1]})")
print(f"  Test:  {len(test_df):,} candles ({df['timestamp'].iloc[split_idx]} → {df['timestamp'].iloc[-1]})")

# ============================================================================
# STEP 2: Volatility Regime Classification
# ============================================================================

print("\n" + "="*80)
print("STEP 2: Volatility Regime Classification")
print("="*80)

# Calculate ATR percentile
df['atr_percentile'] = df['atr'].rolling(window=288, min_periods=50).apply(
    lambda x: stats.percentileofscore(x, x.iloc[-1]) if len(x) > 0 else np.nan
)

# Classify volatility regimes
def classify_volatility(atr_pct):
    if pd.isna(atr_pct):
        return 'unknown'
    elif atr_pct < 33:
        return 'low'
    elif atr_pct < 67:
        return 'medium'
    else:
        return 'high'

df['volatility_regime'] = df['atr_percentile'].apply(classify_volatility)

# Redefine split
train_df = df[df['split'] == 'train'].copy()
test_df = df[df['split'] == 'test'].copy()

# Analyze by split
print("\n📊 Volatility Regime Distribution:")
vol_train = train_df['volatility_regime'].value_counts(normalize=True) * 100
vol_test = test_df['volatility_regime'].value_counts(normalize=True) * 100

print("\n  Training Set:")
for regime in ['low', 'medium', 'high', 'unknown']:
    if regime in vol_train.index:
        print(f"    {regime:10s}: {vol_train[regime]:5.1f}%")

print("\n  Test Set:")
for regime in ['low', 'medium', 'high', 'unknown']:
    if regime in vol_test.index:
        print(f"    {regime:10s}: {vol_test[regime]:5.1f}%")

# Chi-square test skipped
    print(f"    (Statistical test skipped)")
    except Exception as e:
        print(f"    ⚠️  Could not perform chi-square test: {e}")

# ============================================================================
# STEP 3: Trend Regime Classification
# ============================================================================

print("\n" + "="*80)
print("STEP 3: Trend Regime Classification")
print("="*80)

# Multi-timeframe MA alignment
df['ma_20'] = df['close'].rolling(window=20).mean()
df['ma_50'] = df['close'].rolling(window=50).mean()
df['ma_100'] = df['close'].rolling(window=100).mean()
df['ma_200'] = df['close'].rolling(window=200).mean()

# Trend strength: how many MAs are aligned
def classify_trend(row):
    if pd.isna(row['ma_200']):
        return 'unknown'

    close = row['close']
    ma20, ma50, ma100, ma200 = row['ma_20'], row['ma_50'], row['ma_100'], row['ma_200']

    # Bullish alignment score
    bull_score = 0
    if close > ma20: bull_score += 1
    if close > ma50: bull_score += 1
    if close > ma100: bull_score += 1
    if close > ma200: bull_score += 1
    if ma20 > ma50: bull_score += 0.5
    if ma50 > ma100: bull_score += 0.5
    if ma100 > ma200: bull_score += 0.5

    # Classify
    if bull_score >= 5.5:
        return 'strong_bull'
    elif bull_score >= 4:
        return 'bull'
    elif bull_score >= 3:
        return 'neutral'
    elif bull_score >= 1.5:
        return 'bear'
    else:
        return 'strong_bear'

df['trend_regime'] = df.apply(classify_trend, axis=1)

# Redefine split
train_df = df[df['split'] == 'train'].copy()
test_df = df[df['split'] == 'test'].copy()

# Analyze by split
print("\n📊 Trend Regime Distribution:")
trend_train = train_df['trend_regime'].value_counts(normalize=True) * 100
trend_test = test_df['trend_regime'].value_counts(normalize=True) * 100

print("\n  Training Set:")
for regime in ['strong_bull', 'bull', 'neutral', 'bear', 'strong_bear', 'unknown']:
    if regime in trend_train.index:
        print(f"    {regime:15s}: {trend_train[regime]:5.1f}%")

print("\n  Test Set:")
for regime in ['strong_bull', 'bull', 'neutral', 'bear', 'strong_bear', 'unknown']:
    if regime in trend_test.index:
        print(f"    {regime:15s}: {trend_test[regime]:5.1f}%")

# Chi-square test skipped
    print(f"    (Statistical test skipped)")
    except Exception as e:
        print(f"    ⚠️  Could not perform chi-square test: {e}")

# ============================================================================
# STEP 4: Volume Regime Classification
# ============================================================================

print("\n" + "="*80)
print("STEP 4: Volume Regime Classification")
print("="*80)

# Calculate volume percentile
df['volume_percentile'] = df['volume'].rolling(window=288, min_periods=50).apply(
    lambda x: stats.percentileofscore(x, x.iloc[-1]) if len(x) > 0 else np.nan
)

def classify_volume(vol_pct):
    if pd.isna(vol_pct):
        return 'unknown'
    elif vol_pct < 33:
        return 'low'
    elif vol_pct < 67:
        return 'medium'
    else:
        return 'high'

df['volume_regime'] = df['volume_percentile'].apply(classify_volume)

# Redefine split
train_df = df[df['split'] == 'train'].copy()
test_df = df[df['split'] == 'test'].copy()

# Analyze by split
print("\n📊 Volume Regime Distribution:")
vol_regime_train = train_df['volume_regime'].value_counts(normalize=True) * 100
vol_regime_test = test_df['volume_regime'].value_counts(normalize=True) * 100

print("\n  Training Set:")
for regime in ['low', 'medium', 'high', 'unknown']:
    if regime in vol_regime_train.index:
        print(f"    {regime:10s}: {vol_regime_train[regime]:5.1f}%")

print("\n  Test Set:")
for regime in ['low', 'medium', 'high', 'unknown']:
    if regime in vol_regime_test.index:
        print(f"    {regime:10s}: {vol_regime_test[regime]:5.1f}%")

# ============================================================================
# STEP 5: Combined Regime Analysis
# ============================================================================

print("\n" + "="*80)
print("STEP 5: Combined Regime Analysis")
print("="*80)

# Create combined regime label
df['combined_regime'] = (
    df['volatility_regime'] + '_' +
    df['trend_regime']
)

# Top combined regimes
print("\n📊 Top 10 Combined Regimes:")
print("\n  Training Set:")
combined_train = train_df['combined_regime'].value_counts(normalize=True) * 100
for regime, pct in combined_train.head(10).items():
    print(f"    {regime:30s}: {pct:5.2f}%")

print("\n  Test Set:")
combined_test = test_df['combined_regime'].value_counts(normalize=True) * 100
for regime, pct in combined_test.head(10).items():
    print(f"    {regime:30s}: {pct:5.2f}%")

# ============================================================================
# STEP 6: Key Feature Statistics by Regime
# ============================================================================

print("\n" + "="*80)
print("STEP 6: Key Feature Statistics by Regime")
print("="*80)

# SHORT model top features
short_features = ['atr', 'ema_12', 'upside_volatility', 'downside_volatility',
                  'atr_pct', 'trend_strength', 'volatility']

print("\n📊 SHORT Model Features by Volatility Regime (Test Set):")
for feature in short_features[:4]:  # Top 4 features
    print(f"\n  {feature}:")
    for regime in ['low', 'medium', 'high']:
        regime_data = test_df[test_df['volatility_regime'] == regime][feature]
        if len(regime_data) > 0:
            print(f"    {regime:10s}: mean={regime_data.mean():8.4f}, std={regime_data.std():8.4f}")

# Compare train vs test for each regime
print("\n📊 Train vs Test Comparison (ATR by Volatility Regime):")
for regime in ['low', 'medium', 'high']:
    train_atr = train_df[train_df['volatility_regime'] == regime]['atr']
    test_atr = test_df[test_df['volatility_regime'] == regime]['atr']

    if len(train_atr) > 0 and len(test_atr) > 0:
        print(f"\n  {regime} volatility:")
        print(f"    Train ATR: mean={train_atr.mean():.2f}, std={train_atr.std():.2f}")
        print(f"    Test ATR:  mean={test_atr.mean():.2f}, std={test_atr.std():.2f}")

        # K-S test
        ks_stat, ks_p = stats.ks_2samp(train_atr, test_atr)
        print(f"    K-S test: statistic={ks_stat:.4f}, p-value={ks_p:.4f}")
        if ks_p < 0.05:
            print(f"    ⚠️  SIGNIFICANT DIFFERENCE")

# ============================================================================
# STEP 7: Save Results
# ============================================================================

print("\n" + "="*80)
print("STEP 7: Saving Results")
print("="*80)

# Save regime-classified data
output_file = RESULTS_DIR / "regime_classified_data.csv"
df[['timestamp', 'close', 'volatility_regime', 'trend_regime', 'volume_regime',
    'combined_regime', 'split', 'atr', 'atr_percentile']].to_csv(output_file, index=False)
print(f"\n✅ Regime data saved: {output_file.name}")

# Save summary statistics
summary = {
    'volatility_regime': {
        'train': vol_train.to_dict(),
        'test': vol_test.to_dict()
    },
    'trend_regime': {
        'train': trend_train.to_dict(),
        'test': trend_test.to_dict()
    },
    'volume_regime': {
        'train': vol_regime_train.to_dict(),
        'test': vol_regime_test.to_dict()
    }
}

import json
summary_file = RESULTS_DIR / "regime_summary.json"
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"✅ Summary saved: {summary_file.name}")

# ============================================================================
# STEP 8: Diagnostic Summary
# ============================================================================

print("\n" + "="*80)
print("DIAGNOSTIC SUMMARY")
print("="*80)

findings = []

# Check volatility distribution
if 'high' in vol_test.index and 'high' in vol_train.index:
    if vol_test['high'] > vol_train['high'] * 1.5:
        findings.append(f"Test set has {vol_test['high']:.1f}% high volatility vs {vol_train['high']:.1f}% in training")

# Check trend distribution
if 'bear' in trend_test.index or 'strong_bear' in trend_test.index:
    bear_test = trend_test.get('bear', 0) + trend_test.get('strong_bear', 0)
    bear_train = trend_train.get('bear', 0) + trend_train.get('strong_bear', 0)
    if bear_test > bear_train * 1.3:
        findings.append(f"Test set has more bearish periods: {bear_test:.1f}% vs {bear_train:.1f}%")

# Check bullish periods
if 'bull' in trend_test.index or 'strong_bull' in trend_test.index:
    bull_test = trend_test.get('bull', 0) + trend_test.get('strong_bull', 0)
    bull_train = trend_train.get('bull', 0) + trend_train.get('strong_bull', 0)
    if bull_test > bull_train * 1.3:
        findings.append(f"Test set has more bullish periods: {bull_test:.1f}% vs {bull_train:.1f}%")

if len(findings) > 0:
    print(f"\n⚠️  {len(findings)} Significant Findings:")
    for i, finding in enumerate(findings, 1):
        print(f"  {i}. {finding}")

    print("\n💡 Implications for SHORT Model:")
    print("  - SHORT features (ATR, volatility) behave differently in test set")
    print("  - Model trained on different market conditions")
    print("  - Regime-aware features may help")
else:
    print("\n✅ Train and test sets have similar regime distributions")

print("\n" + "="*80)
print("MARKET REGIME ANALYSIS COMPLETE")
print("="*80)
print(f"\nResults saved to: {RESULTS_DIR}")
print("\nNext: Feature Distribution Analysis")
