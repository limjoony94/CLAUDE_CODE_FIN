"""
Pattern Discovery - Fast Search for New Trading Patterns
Finds promising patterns not currently in production.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys

print('=' * 70, flush=True)
print('Pattern Discovery - Finding New Trading Patterns', flush=True)
print('=' * 70, flush=True)

# Production patterns to exclude
PRODUCTION = {
    'U-BU-U', 'ST-BD-DN',  # LONG
    'BD-BD-BD', 'DN-DN-BD', 'MU-ST-DN', 'IH-DN-DN', 'BD-ST-DN', 'BU-U-DN', 'D-DN-BD'  # SHORT
}

# Parameters
MIN_TRADES = 20
TP_GRID = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
SL_GRID = [1.0, 1.5, 2.0, 2.5, 3.0]
LEVERAGE = 3
FEE_PCT = 0.10

# Load data
df = pd.read_csv('data/btc_5m_90days_validation.csv')
print(f'Loaded {len(df)} bars', flush=True)

# Classify candles
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

def classify_row(row):
    if row['range'] < 1e-10:
        return 'D'
    body_ratio = row['body_ratio']
    upper_ratio = row['upper_ratio']
    lower_ratio = row['lower_ratio']
    wick_ratio = row['wick_ratio']
    norm_body = row['norm_body']
    is_bullish = row['body'] > 0

    if body_ratio < 0.10:
        if lower_ratio > 0.70:
            return 'DF'
        elif upper_ratio > 0.70:
            return 'GS'
        return 'D'
    if wick_ratio < 0.15:
        return 'MU' if is_bullish else 'MD'
    body_abs = row['body_abs']
    if body_abs > 0:
        if row['lower_wick'] > 2.0 * body_abs and row['upper_wick'] < 0.3 * body_abs:
            return 'H'
        if row['upper_wick'] > 2.0 * body_abs and row['lower_wick'] < 0.3 * body_abs:
            return 'IH'
    if norm_body < 0.5 and row['upper_wick'] > 0.5 * body_abs and row['lower_wick'] > 0.5 * body_abs:
        return 'ST'
    if norm_body > 1.5:
        return 'BU' if is_bullish else 'BD'
    return 'U' if is_bullish else 'DN'

df['candle_type'] = df.apply(classify_row, axis=1)
df['pattern'] = df['candle_type'].shift(2) + '-' + df['candle_type'].shift(1) + '-' + df['candle_type']

# Find candidate patterns
pattern_counts = df['pattern'].value_counts()
candidate_patterns = [p for p, c in pattern_counts.items()
                      if c >= MIN_TRADES and p not in PRODUCTION and pd.notna(p)]
print(f'\nPatterns with >= {MIN_TRADES} trades (excluding production): {len(candidate_patterns)}', flush=True)

def backtest(pattern, direction, tp_pct, sl_pct):
    signals = df[df['pattern'] == pattern]
    results = []
    for idx in signals.index:
        pos = df.index.get_loc(idx)
        if pos + 1 >= len(df):
            continue
        entry = df.iloc[pos + 1]['open']
        for i in range(pos + 1, min(pos + 100, len(df))):
            bar = df.iloc[i]
            if direction == 'LONG':
                tp_price = entry * (1 + tp_pct/100)
                sl_price = entry * (1 - sl_pct/100)
                if bar['high'] >= tp_price:
                    results.append('WIN')
                    break
                elif bar['low'] <= sl_price:
                    results.append('LOSS')
                    break
            else:  # SHORT
                tp_price = entry * (1 - tp_pct/100)
                sl_price = entry * (1 + sl_pct/100)
                if bar['low'] <= tp_price:
                    results.append('WIN')
                    break
                elif bar['high'] >= sl_price:
                    results.append('LOSS')
                    break

    if not results:
        return 0, 0, 0

    trades = len(results)
    wins = sum(1 for r in results if r == 'WIN')
    wr = wins / trades * 100
    pnl = wins * (tp_pct * LEVERAGE - FEE_PCT) + (trades - wins) * (-sl_pct * LEVERAGE - FEE_PCT)
    return trades, wr, pnl

def walk_forward(pattern, direction, tp, sl, n_folds=5):
    fold_size = len(df) // n_folds
    profitable = 0

    for i in range(n_folds):
        start = i * fold_size
        end = (i + 1) * fold_size if i < n_folds - 1 else len(df)
        fold_df = df.iloc[start:end].copy().reset_index(drop=True)

        signals = fold_df[fold_df['pattern'] == pattern]
        results = []

        for idx in signals.index:
            pos = fold_df.index.get_loc(idx)
            if pos + 1 >= len(fold_df):
                continue
            entry = fold_df.iloc[pos + 1]['open']

            for j in range(pos + 1, min(pos + 100, len(fold_df))):
                bar = fold_df.iloc[j]
                if direction == 'LONG':
                    if bar['high'] >= entry * (1 + tp/100):
                        results.append('WIN')
                        break
                    elif bar['low'] <= entry * (1 - sl/100):
                        results.append('LOSS')
                        break
                else:
                    if bar['low'] <= entry * (1 - tp/100):
                        results.append('WIN')
                        break
                    elif bar['high'] >= entry * (1 + sl/100):
                        results.append('LOSS')
                        break

        if results:
            wins = sum(1 for r in results if r == 'WIN')
            losses = len(results) - wins
            pnl = wins * (tp * LEVERAGE - FEE_PCT) + losses * (-sl * LEVERAGE - FEE_PCT)
            if pnl > 0:
                profitable += 1

    return profitable

# Search for best patterns
print('\nSearching for promising patterns...', flush=True)
results = []

for i, pattern in enumerate(candidate_patterns):
    if (i + 1) % 50 == 0:
        print(f'  Progress: {i+1}/{len(candidate_patterns)}...', flush=True)

    for direction in ['LONG', 'SHORT']:
        best_score = -999
        best_tp, best_sl = 1.5, 2.0
        best_result = None

        for tp in TP_GRID:
            for sl in SL_GRID:
                trades, wr, pnl = backtest(pattern, direction, tp, sl)
                if trades < 10:
                    continue
                edge = pnl / trades if trades > 0 else 0
                score = wr * 0.5 + edge * 50
                if score > best_score:
                    best_score = score
                    best_tp, best_sl = tp, sl
                    best_result = (trades, wr, pnl, edge)

        if best_result and best_result[0] >= MIN_TRADES and best_result[1] >= 55:
            wf = walk_forward(pattern, direction, best_tp, best_sl)
            if wf >= 3:
                results.append({
                    'pattern': pattern,
                    'direction': direction,
                    'trades': best_result[0],
                    'wr': best_result[1],
                    'pnl': best_result[2],
                    'edge': best_result[3],
                    'tp': best_tp,
                    'sl': best_sl,
                    'wf': wf,
                    'score': best_result[0]/100*20 + best_result[1]/100*30 + min(best_result[3], 1.0)*30 + wf/5*20
                })

print(f'\n✅ Found {len(results)} promising new patterns!', flush=True)

# Sort and display
results.sort(key=lambda x: x['score'], reverse=True)

print('\n' + '=' * 70, flush=True)
print('TOP NEW PATTERN CANDIDATES', flush=True)
print('=' * 70, flush=True)

for i, r in enumerate(results[:30], 1):
    print(f"\n[{i}] {r['pattern']} ({r['direction']})", flush=True)
    print(f"    Trades: {r['trades']}, WR: {r['wr']:.1f}%, Edge: {r['edge']:.2f}, PnL: {r['pnl']:.1f}%", flush=True)
    print(f"    TP/SL: {r['tp']:.1f}%/{r['sl']:.1f}%, WF: {r['wf']}/5, Score: {r['score']:.1f}", flush=True)

# Separate by direction
longs = [r for r in results if r['direction'] == 'LONG']
shorts = [r for r in results if r['direction'] == 'SHORT']

print('\n' + '=' * 70, flush=True)
print('RECOMMENDED ADDITIONS TO PRODUCTION', flush=True)
print('=' * 70, flush=True)

print('\nNEW LONG PATTERNS:', flush=True)
if not longs:
    print('  (No new LONG patterns found)', flush=True)
else:
    for r in longs[:5]:
        print(f"  '{r['pattern']}': ({r['tp']}, {r['sl']}),  # WR {r['wr']:.1f}%, {r['trades']} trades, WF {r['wf']}/5", flush=True)

print('\nNEW SHORT PATTERNS:', flush=True)
if not shorts:
    print('  (No new SHORT patterns found)', flush=True)
else:
    for r in shorts[:5]:
        print(f"  '{r['pattern']}': ({r['tp']}, {r['sl']}),  # WR {r['wr']:.1f}%, {r['trades']} trades, WF {r['wf']}/5", flush=True)

# Save results
results_df = pd.DataFrame(results)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file = f'results/pattern_discovery_{timestamp}.csv'
results_df.to_csv(output_file, index=False)
print(f'\n✅ Results saved to: {output_file}', flush=True)
