"""
Pattern Discovery - Optimized Search for New Trading Patterns
Uses vectorized operations for speed.
"""

import pandas as pd
import numpy as np
from datetime import datetime

print('=' * 70)
print('Pattern Discovery - Optimized Search')
print('=' * 70)

# Production patterns to exclude
PRODUCTION = {
    'U-BU-U', 'ST-BD-DN',  # LONG
    'BD-BD-BD', 'DN-DN-BD', 'MU-ST-DN', 'IH-DN-DN', 'BD-ST-DN', 'BU-U-DN', 'D-DN-BD'  # SHORT
}

# Parameters
MIN_TRADES = 20
MIN_WR = 55.0
MIN_WF = 3
LEVERAGE = 3
FEE_PCT = 0.10

# Reduced grid for speed
TP_GRID = [1.0, 1.5, 2.0, 2.5, 3.0]
SL_GRID = [1.5, 2.0, 2.5, 3.0]

# Load data
df = pd.read_csv('data/btc_5m_90days_validation.csv')
print(f'Loaded {len(df)} bars')

# Classify candles (vectorized)
df['body'] = df['close'] - df['open']
df['body_abs'] = df['body'].abs()
df['range'] = df['high'] - df['low']
df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
df['avg_body'] = df['body_abs'].rolling(20).mean().fillna(1.0)  # default 1.0 for early bars: preserves range-based classification
df['norm_body'] = df['body_abs'] / df['avg_body'].replace(0, 1)
df['body_ratio'] = df['body_abs'] / df['range'].replace(0, 1)
df['upper_ratio'] = df['upper_wick'] / df['range'].replace(0, 1)
df['lower_ratio'] = df['lower_wick'] / df['range'].replace(0, 1)
df['wick_ratio'] = (df['upper_wick'] + df['lower_wick']) / df['range'].replace(0, 1)

# Vectorized classification
is_bullish = df['body'] > 0
small_range = df['range'] < 1e-10

# Initialize with medium candles
df['candle_type'] = np.where(is_bullish, 'U', 'DN')

# Doji types
doji_mask = df['body_ratio'] < 0.10
df.loc[doji_mask & (df['lower_ratio'] > 0.70), 'candle_type'] = 'DF'
df.loc[doji_mask & (df['upper_ratio'] > 0.70) & (df['lower_ratio'] <= 0.70), 'candle_type'] = 'GS'
df.loc[doji_mask & (df['lower_ratio'] <= 0.70) & (df['upper_ratio'] <= 0.70), 'candle_type'] = 'D'

# Marubozu
marubozu_mask = (~doji_mask) & (df['wick_ratio'] < 0.15)
df.loc[marubozu_mask & is_bullish, 'candle_type'] = 'MU'
df.loc[marubozu_mask & ~is_bullish, 'candle_type'] = 'MD'

# Hammer patterns
hammer_mask = (~doji_mask) & (~marubozu_mask) & (df['body_abs'] > 0)
h_mask = hammer_mask & (df['lower_wick'] > 2.0 * df['body_abs']) & (df['upper_wick'] < 0.3 * df['body_abs'])
ih_mask = hammer_mask & (df['upper_wick'] > 2.0 * df['body_abs']) & (df['lower_wick'] < 0.3 * df['body_abs'])
df.loc[h_mask, 'candle_type'] = 'H'
df.loc[ih_mask, 'candle_type'] = 'IH'

# Spinning top
st_mask = (~doji_mask) & (~marubozu_mask) & (~h_mask) & (~ih_mask) & (df['norm_body'] < 0.5)
st_mask = st_mask & (df['upper_wick'] > 0.5 * df['body_abs']) & (df['lower_wick'] > 0.5 * df['body_abs'])
df.loc[st_mask, 'candle_type'] = 'ST'

# Big candles
big_mask = (~doji_mask) & (~marubozu_mask) & (~h_mask) & (~ih_mask) & (~st_mask) & (df['norm_body'] > 1.5)
df.loc[big_mask & is_bullish, 'candle_type'] = 'BU'
df.loc[big_mask & ~is_bullish, 'candle_type'] = 'BD'

# Small range fallback
df.loc[small_range, 'candle_type'] = 'D'

# Create pattern
df['pattern'] = df['candle_type'].shift(2) + '-' + df['candle_type'].shift(1) + '-' + df['candle_type']
df['next_open'] = df['open'].shift(-1)

# Pre-compute max high and min low for next N bars (for fast exit detection)
MAX_BARS = 50
for i in range(1, MAX_BARS + 1):
    df[f'future_high_{i}'] = df['high'].shift(-i)
    df[f'future_low_{i}'] = df['low'].shift(-i)

print('Candle classification done')

# Find patterns with enough trades
pattern_counts = df['pattern'].value_counts()
candidates = [p for p, c in pattern_counts.items()
              if c >= MIN_TRADES and p not in PRODUCTION and pd.notna(p)]
print(f'Candidate patterns: {len(candidates)}')

def fast_backtest(pattern_df, direction, tp_pct, sl_pct):
    """Fast backtest using pre-computed future prices."""
    if len(pattern_df) < 5:
        return 0, 0, 0

    entry_prices = pattern_df['next_open'].values
    results = []

    for idx, entry in enumerate(entry_prices):
        if pd.isna(entry):
            continue

        row = pattern_df.iloc[idx]

        if direction == 'LONG':
            tp_price = entry * (1 + tp_pct / 100)
            sl_price = entry * (1 - sl_pct / 100)
        else:
            tp_price = entry * (1 - tp_pct / 100)
            sl_price = entry * (1 + sl_pct / 100)

        # Check each future bar
        for i in range(1, MAX_BARS + 1):
            future_high = row.get(f'future_high_{i}', np.nan)
            future_low = row.get(f'future_low_{i}', np.nan)

            if pd.isna(future_high) or pd.isna(future_low):
                break

            if direction == 'LONG':
                if future_high >= tp_price:
                    results.append('WIN')
                    break
                elif future_low <= sl_price:
                    results.append('LOSS')
                    break
            else:
                if future_low <= tp_price:
                    results.append('WIN')
                    break
                elif future_high >= sl_price:
                    results.append('LOSS')
                    break

    if not results:
        return 0, 0, 0

    trades = len(results)
    wins = sum(1 for r in results if r == 'WIN')
    wr = wins / trades * 100
    pnl = wins * (tp_pct * LEVERAGE - FEE_PCT) + (trades - wins) * (-sl_pct * LEVERAGE - FEE_PCT)

    return trades, wr, pnl

def walk_forward_fast(pattern_df, direction, tp, sl, n_folds=5):
    """Fast walk-forward validation."""
    fold_size = len(pattern_df) // n_folds
    if fold_size < 3:
        return 0

    profitable = 0
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else len(pattern_df)
        fold_df = pattern_df.iloc[start:end]

        trades, wr, pnl = fast_backtest(fold_df, direction, tp, sl)
        if pnl > 0:
            profitable += 1

    return profitable

# Search
print('\nSearching for promising patterns...')
results = []
total = len(candidates) * 2

for i, pattern in enumerate(candidates):
    if (i + 1) % 100 == 0:
        print(f'  Progress: {i+1}/{len(candidates)} patterns...')

    pattern_df = df[df['pattern'] == pattern].copy()

    for direction in ['LONG', 'SHORT']:
        best_score = -999
        best_tp, best_sl = 1.5, 2.0
        best_result = None

        # Quick filter: test with default first
        trades, wr, pnl = fast_backtest(pattern_df, direction, 1.5, 2.0)
        if trades < MIN_TRADES * 0.5 or wr < 40:
            continue  # Skip unpromising patterns

        # Grid search
        for tp in TP_GRID:
            for sl in SL_GRID:
                trades, wr, pnl = fast_backtest(pattern_df, direction, tp, sl)
                if trades < 10:
                    continue
                edge = pnl / trades
                score = wr * 0.5 + edge * 50
                if score > best_score:
                    best_score = score
                    best_tp, best_sl = tp, sl
                    best_result = (trades, wr, pnl, edge)

        if best_result and best_result[0] >= MIN_TRADES and best_result[1] >= MIN_WR:
            wf = walk_forward_fast(pattern_df, direction, best_tp, best_sl)
            if wf >= MIN_WF:
                results.append({
                    'pattern': pattern,
                    'direction': direction,
                    'trades': best_result[0],
                    'wr': round(best_result[1], 1),
                    'pnl': round(best_result[2], 1),
                    'edge': round(best_result[3], 2),
                    'tp': best_tp,
                    'sl': best_sl,
                    'wf': wf,
                    'score': round(best_result[0]/100*20 + best_result[1]/100*30 + min(best_result[3], 1.0)*30 + wf/5*20, 1)
                })

print(f'\n✅ Found {len(results)} promising new patterns!')

# Sort and display
results.sort(key=lambda x: x['score'], reverse=True)

print('\n' + '=' * 70)
print('TOP NEW PATTERN CANDIDATES')
print('=' * 70)

for i, r in enumerate(results[:20], 1):
    print(f"\n[{i}] {r['pattern']} ({r['direction']})")
    print(f"    Trades: {r['trades']}, WR: {r['wr']}%, Edge: {r['edge']}, PnL: {r['pnl']}%")
    print(f"    TP/SL: {r['tp']}%/{r['sl']}%, WF: {r['wf']}/5, Score: {r['score']}")

# Separate by direction
longs = [r for r in results if r['direction'] == 'LONG']
shorts = [r for r in results if r['direction'] == 'SHORT']

print('\n' + '=' * 70)
print('RECOMMENDED ADDITIONS TO PRODUCTION')
print('=' * 70)

print('\nNEW LONG PATTERNS:')
if not longs:
    print('  (No new LONG patterns found)')
else:
    for r in longs[:5]:
        print(f"  '{r['pattern']}': ({r['tp']}, {r['sl']}),  # WR {r['wr']}%, {r['trades']} trades, WF {r['wf']}/5")

print('\nNEW SHORT PATTERNS:')
if not shorts:
    print('  (No new SHORT patterns found)')
else:
    for r in shorts[:5]:
        print(f"  '{r['pattern']}': ({r['tp']}, {r['sl']}),  # WR {r['wr']}%, {r['trades']} trades, WF {r['wf']}/5")

# Save results
if results:
    results_df = pd.DataFrame(results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'results/pattern_discovery_{timestamp}.csv'
    results_df.to_csv(output_file, index=False)
    print(f'\n✅ Results saved to: {output_file}')
