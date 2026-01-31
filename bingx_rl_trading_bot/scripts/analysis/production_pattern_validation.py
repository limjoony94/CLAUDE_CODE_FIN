"""
Production Pattern Validation - Comprehensive Testing for v1.16 Patterns
Validates all 19 patterns with Walk-Forward, Regime Analysis, and Statistical Tests.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats

print('=' * 70)
print('Production Pattern Validation v1.16')
print('=' * 70)

# ============================================================
# PRODUCTION PATTERNS (v1.16)
# ============================================================

LONG_PATTERNS = {
    # Existing
    'U-BU-U': (1.5, 2.0),
    'ST-BD-DN': (2.0, 3.0),
    # NEW v1.16
    'DN-DN-DN': (1.0, 3.0),
    'DN-U-U': (1.0, 3.0),
    'DN-DN-U': (1.0, 3.0),
    'DN-ST-U': (1.0, 3.0),
    'U-ST-U': (1.0, 3.0),
    'U-U-U': (1.5, 3.0),
}

SHORT_PATTERNS = {
    # Existing
    'BD-BD-BD': (3.0, 2.5),
    'DN-DN-BD': (1.5, 3.0),
    'MU-ST-DN': (1.0, 2.5),
    'IH-DN-DN': (1.0, 3.0),
    'BD-ST-DN': (1.5, 3.0),
    'BU-U-DN': (1.5, 2.5),
    'D-DN-BD': (2.5, 2.0),
    # NEW v1.16
    'U-DN-DN': (1.0, 3.0),
    'DN-U-DN': (2.0, 3.0),
    'DN-DN-ST': (1.5, 3.0),
    'U-U-DN': (2.0, 3.0),
}

# Parameters
LEVERAGE = 3
FEE_PCT = 0.10
MAX_BARS = 100

# Load data
df = pd.read_csv('data/btc_5m_90days_validation.csv')
print(f'Loaded {len(df)} bars')

# ============================================================
# CANDLE CLASSIFICATION (Vectorized)
# ============================================================

df['body'] = df['close'] - df['open']
df['body_abs'] = df['body'].abs()
df['range'] = df['high'] - df['low']
df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
df['avg_body'] = df['body_abs'].rolling(20).mean()  # NaN for bars 0-19, matching production
df['norm_body'] = df['body_abs'] / df['avg_body'].replace(0, 1)
df['body_ratio'] = df['body_abs'] / df['range'].replace(0, 1)
df['upper_ratio'] = df['upper_wick'] / df['range'].replace(0, 1)
df['lower_ratio'] = df['lower_wick'] / df['range'].replace(0, 1)
df['wick_ratio'] = (df['upper_wick'] + df['lower_wick']) / df['range'].replace(0, 1)

is_bullish = df['body'] > 0
small_range = df['range'] < 1e-10

df['candle_type'] = np.where(is_bullish, 'U', 'DN')

doji_mask = df['body_ratio'] < 0.10
df.loc[doji_mask & (df['lower_ratio'] > 0.70), 'candle_type'] = 'DF'
df.loc[doji_mask & (df['upper_ratio'] > 0.70) & (df['lower_ratio'] <= 0.70), 'candle_type'] = 'GS'
df.loc[doji_mask & (df['lower_ratio'] <= 0.70) & (df['upper_ratio'] <= 0.70), 'candle_type'] = 'D'

marubozu_mask = (~doji_mask) & (df['wick_ratio'] < 0.15)
df.loc[marubozu_mask & is_bullish, 'candle_type'] = 'MU'
df.loc[marubozu_mask & ~is_bullish, 'candle_type'] = 'MD'

hammer_mask = (~doji_mask) & (~marubozu_mask) & (df['body_abs'] > 0)
h_mask = hammer_mask & (df['lower_wick'] > 2.0 * df['body_abs']) & (df['upper_wick'] < 0.3 * df['body_abs'])
ih_mask = hammer_mask & (df['upper_wick'] > 2.0 * df['body_abs']) & (df['lower_wick'] < 0.3 * df['body_abs'])
df.loc[h_mask, 'candle_type'] = 'H'
df.loc[ih_mask, 'candle_type'] = 'IH'

st_mask = (~doji_mask) & (~marubozu_mask) & (~h_mask) & (~ih_mask) & (df['norm_body'] < 0.5)
st_mask = st_mask & (df['upper_wick'] > 0.5 * df['body_abs']) & (df['lower_wick'] > 0.5 * df['body_abs'])
df.loc[st_mask, 'candle_type'] = 'ST'

big_mask = (~doji_mask) & (~marubozu_mask) & (~h_mask) & (~ih_mask) & (~st_mask) & (df['norm_body'] > 1.5)
df.loc[big_mask & is_bullish, 'candle_type'] = 'BU'
df.loc[big_mask & ~is_bullish, 'candle_type'] = 'BD'

df.loc[small_range, 'candle_type'] = 'D'

# Create pattern
df['pattern'] = df['candle_type'].shift(2) + '-' + df['candle_type'].shift(1) + '-' + df['candle_type']
df['next_open'] = df['open'].shift(-1)

# Regime classification (20-bar price change)
df['regime_change'] = (df['close'] - df['close'].shift(20)) / df['close'].shift(20) * 100
df['regime'] = 'SIDE'
df.loc[df['regime_change'] > 3, 'regime'] = 'BULL'
df.loc[df['regime_change'] < -3, 'regime'] = 'BEAR'

print('Candle classification done')
print(f"Regime distribution: BULL {(df['regime']=='BULL').sum()}, BEAR {(df['regime']=='BEAR').sum()}, SIDE {(df['regime']=='SIDE').sum()}")

# ============================================================
# BACKTEST FUNCTION
# ============================================================

def backtest_pattern(pattern_df, direction, tp_pct, sl_pct):
    """Backtest a pattern with given TP/SL."""
    results = []
    pnls = []

    for idx in range(len(pattern_df)):
        row = pattern_df.iloc[idx]
        entry = row['next_open']

        if pd.isna(entry):
            continue

        pos = pattern_df.index.get_loc(idx) if isinstance(pattern_df.index, pd.RangeIndex) else idx
        original_pos = df.index.get_loc(pattern_df.iloc[idx].name)

        if direction == 'LONG':
            tp_price = entry * (1 + tp_pct / 100)
            sl_price = entry * (1 - sl_pct / 100)
        else:
            tp_price = entry * (1 - tp_pct / 100)
            sl_price = entry * (1 + sl_pct / 100)

        # Check future bars
        for i in range(1, min(MAX_BARS, len(df) - original_pos)):
            future_bar = df.iloc[original_pos + i]

            if direction == 'LONG':
                if future_bar['high'] >= tp_price:
                    results.append('WIN')
                    pnls.append(tp_pct * LEVERAGE - FEE_PCT)
                    break
                elif future_bar['low'] <= sl_price:
                    results.append('LOSS')
                    pnls.append(-sl_pct * LEVERAGE - FEE_PCT)
                    break
            else:
                if future_bar['low'] <= tp_price:
                    results.append('WIN')
                    pnls.append(tp_pct * LEVERAGE - FEE_PCT)
                    break
                elif future_bar['high'] >= sl_price:
                    results.append('LOSS')
                    pnls.append(-sl_pct * LEVERAGE - FEE_PCT)
                    break

    if not results:
        return {'trades': 0, 'wr': 0, 'pnl': 0, 'edge': 0, 'pnls': []}

    trades = len(results)
    wins = sum(1 for r in results if r == 'WIN')
    wr = wins / trades * 100
    total_pnl = sum(pnls)
    edge = total_pnl / trades

    return {'trades': trades, 'wr': wr, 'pnl': total_pnl, 'edge': edge, 'pnls': pnls}

# ============================================================
# WALK-FORWARD VALIDATION
# ============================================================

def walk_forward_validation(pattern, direction, tp, sl, n_folds=5):
    """5-fold time-series walk-forward validation."""
    pattern_df = df[df['pattern'] == pattern].copy()

    if len(pattern_df) < 10:
        return {'folds': 0, 'profitable': 0, 'fold_results': []}

    fold_size = len(pattern_df) // n_folds
    fold_results = []
    profitable = 0

    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else len(pattern_df)
        fold_df = pattern_df.iloc[start:end]

        result = backtest_pattern(fold_df, direction, tp, sl)
        fold_results.append(result)

        if result['pnl'] > 0:
            profitable += 1

    return {'folds': n_folds, 'profitable': profitable, 'fold_results': fold_results}

# ============================================================
# REGIME ANALYSIS
# ============================================================

def regime_analysis(pattern, direction, tp, sl):
    """Test pattern performance in different market regimes."""
    results = {}

    for regime in ['BULL', 'BEAR', 'SIDE']:
        regime_df = df[(df['pattern'] == pattern) & (df['regime'] == regime)]
        if len(regime_df) >= 5:
            result = backtest_pattern(regime_df, direction, tp, sl)
            results[regime] = result
        else:
            results[regime] = {'trades': len(regime_df), 'wr': 0, 'pnl': 0, 'edge': 0}

    return results

# ============================================================
# STATISTICAL TESTS
# ============================================================

def statistical_analysis(pattern, direction, tp, sl):
    """Perform statistical significance tests."""
    pattern_df = df[df['pattern'] == pattern]
    result = backtest_pattern(pattern_df, direction, tp, sl)

    if result['trades'] < 10:
        return {'significant': False, 'p_value': 1.0, 'confidence_interval': (0, 0)}

    pnls = result['pnls']

    # T-test: Is mean PnL significantly > 0?
    if len(pnls) > 1 and np.std(pnls) > 0:
        t_stat, p_value = stats.ttest_1samp(pnls, 0)
        p_value = p_value / 2  # One-tailed test
    else:
        p_value = 1.0

    # Confidence interval
    if len(pnls) > 1:
        ci = stats.t.interval(0.95, len(pnls)-1, loc=np.mean(pnls), scale=stats.sem(pnls))
    else:
        ci = (0, 0)

    # Binomial test for WR > 50%
    wins = sum(1 for p in pnls if p > 0)
    try:
        binom_result = stats.binomtest(wins, len(pnls), 0.5, alternative='greater') if len(pnls) > 0 else None
        binom_p = binom_result.pvalue if binom_result else 1.0
    except:
        binom_p = 1.0

    return {
        'significant': p_value < 0.05,
        'p_value': p_value,
        'binom_p': binom_p,
        'confidence_interval': ci,
        'mean_pnl': np.mean(pnls) if pnls else 0,
        'std_pnl': np.std(pnls) if len(pnls) > 1 else 0
    }

# ============================================================
# COUNTER-TREND ANALYSIS
# ============================================================

def counter_trend_analysis(pattern, direction, tp, sl):
    """Test if pattern has alpha beyond market direction."""
    # For LONG patterns, test in BEAR market (counter-trend)
    # For SHORT patterns, test in BULL market (counter-trend)

    if direction == 'LONG':
        counter_regime = 'BEAR'
        with_regime = 'BULL'
    else:
        counter_regime = 'BULL'
        with_regime = 'BEAR'

    counter_df = df[(df['pattern'] == pattern) & (df['regime'] == counter_regime)]
    with_df = df[(df['pattern'] == pattern) & (df['regime'] == with_regime)]

    counter_result = backtest_pattern(counter_df, direction, tp, sl) if len(counter_df) >= 3 else None
    with_result = backtest_pattern(with_df, direction, tp, sl) if len(with_df) >= 3 else None

    return {
        'counter_trend': counter_result,
        'with_trend': with_result,
        'counter_regime': counter_regime,
        'with_regime': with_regime
    }

# ============================================================
# MAIN VALIDATION
# ============================================================

print('\n' + '=' * 70)
print('VALIDATING ALL PRODUCTION PATTERNS')
print('=' * 70)

all_results = []

# Process LONG patterns
print('\n### LONG PATTERNS ###')
for pattern, (tp, sl) in LONG_PATTERNS.items():
    print(f'\n--- {pattern} (LONG) ---')

    pattern_df = df[df['pattern'] == pattern]

    # Basic backtest
    basic = backtest_pattern(pattern_df, 'LONG', tp, sl)
    print(f'  Basic: {basic["trades"]} trades, WR {basic["wr"]:.1f}%, Edge {basic["edge"]:.2f}')

    # Walk-forward
    wf = walk_forward_validation(pattern, 'LONG', tp, sl)
    print(f'  Walk-Forward: {wf["profitable"]}/{wf["folds"]} folds profitable')

    # Regime analysis
    regime = regime_analysis(pattern, 'LONG', tp, sl)
    for r, data in regime.items():
        if data['trades'] > 0:
            print(f'  {r}: {data["trades"]} trades, WR {data["wr"]:.1f}%')

    # Statistical analysis
    stat = statistical_analysis(pattern, 'LONG', tp, sl)
    print(f'  Statistical: p={stat["p_value"]:.4f}, significant={stat["significant"]}')

    # Counter-trend
    ct = counter_trend_analysis(pattern, 'LONG', tp, sl)
    if ct['counter_trend'] and ct['counter_trend']['trades'] > 0:
        print(f'  Counter-trend (BEAR): {ct["counter_trend"]["trades"]} trades, WR {ct["counter_trend"]["wr"]:.1f}%')

    all_results.append({
        'pattern': pattern,
        'direction': 'LONG',
        'tp': tp,
        'sl': sl,
        'trades': basic['trades'],
        'wr': basic['wr'],
        'edge': basic['edge'],
        'pnl': basic['pnl'],
        'wf_pass': wf['profitable'],
        'wf_total': wf['folds'],
        'p_value': stat['p_value'],
        'significant': stat['significant'],
        'bull_trades': regime.get('BULL', {}).get('trades', 0),
        'bull_wr': regime.get('BULL', {}).get('wr', 0),
        'bear_trades': regime.get('BEAR', {}).get('trades', 0),
        'bear_wr': regime.get('BEAR', {}).get('wr', 0),
        'side_trades': regime.get('SIDE', {}).get('trades', 0),
        'side_wr': regime.get('SIDE', {}).get('wr', 0),
    })

# Process SHORT patterns
print('\n### SHORT PATTERNS ###')
for pattern, (tp, sl) in SHORT_PATTERNS.items():
    print(f'\n--- {pattern} (SHORT) ---')

    pattern_df = df[df['pattern'] == pattern]

    # Basic backtest
    basic = backtest_pattern(pattern_df, 'SHORT', tp, sl)
    print(f'  Basic: {basic["trades"]} trades, WR {basic["wr"]:.1f}%, Edge {basic["edge"]:.2f}')

    # Walk-forward
    wf = walk_forward_validation(pattern, 'SHORT', tp, sl)
    print(f'  Walk-Forward: {wf["profitable"]}/{wf["folds"]} folds profitable')

    # Regime analysis
    regime = regime_analysis(pattern, 'SHORT', tp, sl)
    for r, data in regime.items():
        if data['trades'] > 0:
            print(f'  {r}: {data["trades"]} trades, WR {data["wr"]:.1f}%')

    # Statistical analysis
    stat = statistical_analysis(pattern, 'SHORT', tp, sl)
    print(f'  Statistical: p={stat["p_value"]:.4f}, significant={stat["significant"]}')

    # Counter-trend
    ct = counter_trend_analysis(pattern, 'SHORT', tp, sl)
    if ct['counter_trend'] and ct['counter_trend']['trades'] > 0:
        print(f'  Counter-trend (BULL): {ct["counter_trend"]["trades"]} trades, WR {ct["counter_trend"]["wr"]:.1f}%')

    all_results.append({
        'pattern': pattern,
        'direction': 'SHORT',
        'tp': tp,
        'sl': sl,
        'trades': basic['trades'],
        'wr': basic['wr'],
        'edge': basic['edge'],
        'pnl': basic['pnl'],
        'wf_pass': wf['profitable'],
        'wf_total': wf['folds'],
        'p_value': stat['p_value'],
        'significant': stat['significant'],
        'bull_trades': regime.get('BULL', {}).get('trades', 0),
        'bull_wr': regime.get('BULL', {}).get('wr', 0),
        'bear_trades': regime.get('BEAR', {}).get('trades', 0),
        'bear_wr': regime.get('BEAR', {}).get('wr', 0),
        'side_trades': regime.get('SIDE', {}).get('trades', 0),
        'side_wr': regime.get('SIDE', {}).get('wr', 0),
    })

# ============================================================
# SUMMARY REPORT
# ============================================================

print('\n' + '=' * 70)
print('VALIDATION SUMMARY')
print('=' * 70)

results_df = pd.DataFrame(all_results)

# Categorize patterns
strong = results_df[(results_df['wf_pass'] >= 4) & (results_df['wr'] >= 60) & (results_df['edge'] > 0)]
moderate = results_df[(results_df['wf_pass'] >= 3) & (results_df['wr'] >= 50) & (results_df['edge'] > 0) & ~results_df.index.isin(strong.index)]
weak = results_df[~results_df.index.isin(strong.index) & ~results_df.index.isin(moderate.index)]

print(f'\n✅ STRONG PATTERNS ({len(strong)}):')
print('   WF ≥4/5, WR ≥60%, Edge >0')
for _, row in strong.iterrows():
    print(f'   {row["pattern"]} ({row["direction"]}): WR {row["wr"]:.1f}%, Edge {row["edge"]:.2f}, WF {row["wf_pass"]}/5')

print(f'\n⚠️ MODERATE PATTERNS ({len(moderate)}):')
print('   WF ≥3/5, WR ≥50%, Edge >0')
for _, row in moderate.iterrows():
    print(f'   {row["pattern"]} ({row["direction"]}): WR {row["wr"]:.1f}%, Edge {row["edge"]:.2f}, WF {row["wf_pass"]}/5')

print(f'\n❌ WEAK PATTERNS ({len(weak)}):')
print('   Did not meet criteria')
for _, row in weak.iterrows():
    print(f'   {row["pattern"]} ({row["direction"]}): WR {row["wr"]:.1f}%, Edge {row["edge"]:.2f}, WF {row["wf_pass"]}/5')

# Statistical significance summary
sig_count = results_df['significant'].sum()
print(f'\n📊 STATISTICAL SIGNIFICANCE:')
print(f'   {sig_count}/{len(results_df)} patterns statistically significant (p<0.05)')

# Recommendations
print('\n' + '=' * 70)
print('RECOMMENDATIONS')
print('=' * 70)

if len(weak) > 0:
    print('\n🔴 PATTERNS TO REVIEW/REMOVE:')
    for _, row in weak.iterrows():
        reason = []
        if row['wf_pass'] < 3:
            reason.append(f'WF {row["wf_pass"]}/5')
        if row['wr'] < 50:
            reason.append(f'WR {row["wr"]:.1f}%')
        if row['edge'] <= 0:
            reason.append(f'Edge {row["edge"]:.2f}')
        print(f'   {row["pattern"]} ({row["direction"]}): {", ".join(reason)}')

# Save results
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file = f'results/production_validation_{timestamp}.csv'
results_df.to_csv(output_file, index=False)
print(f'\n✅ Results saved to: {output_file}')

# Print final summary table
print('\n' + '=' * 70)
print('FULL RESULTS TABLE')
print('=' * 70)
print(results_df[['pattern', 'direction', 'trades', 'wr', 'edge', 'wf_pass', 'significant']].to_string(index=False))
