"""
Tight TP/SL Full Validation Pipeline
=====================================
Goal: Find regime-independent LONG + SHORT balanced strategy
using tight TP/SL (0.3-0.5%) where fee impact and regime
dependency are carefully analyzed.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from collections import defaultdict
import sys

# Import canonical classify_candle from production
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.production.pattern_5m.indicators import classify_candle

MAX_BARS = 500
FEE_PCT = 0.10
LEVERAGE = 3

df = pd.read_csv(Path(__file__).parent / '../../data/btc_5m_270days.csv')
print(f"Loaded: {len(df)} bars")

highs = df['high'].values
lows = df['low'].values
opens = df['open'].values
closes = df['close'].values
n_bars = len(df)

df['timestamp'] = pd.to_datetime(df['timestamp'])
dates = df['timestamp'].values

period_masks = {
    'H1': (dates >= np.datetime64('2025-05-01')) & (dates < np.datetime64('2025-08-01')),
    'H2': (dates >= np.datetime64('2025-08-01')) & (dates < np.datetime64('2025-11-01')),
    'H3': (dates >= np.datetime64('2025-11-01')) & (dates < np.datetime64('2026-02-01')),
}

print("Classifying candles...")
body_abs_arr = np.abs(closes - opens)
avg_body_20_arr = pd.Series(body_abs_arr).rolling(20).mean().values  # NaN for bars 0-19
# For early bars: default avg_body_20=1.0 preserves range-based classification
types = [classify_candle(opens[i], highs[i], lows[i], closes[i],
                         avg_body_20_arr[i] if not pd.isna(avg_body_20_arr[i]) else 1.0)
         for i in range(n_bars)]

pattern_arr = [''] * n_bars
for i in range(2, n_bars):
    pattern_arr[i] = f"{types[i-2]}-{types[i-1]}-{types[i]}"

pattern_indices = {}
for i in range(2, n_bars):
    p = pattern_arr[i]
    if p not in pattern_indices:
        pattern_indices[p] = []
    pattern_indices[p].append(i)
for k in pattern_indices:
    pattern_indices[k] = np.array(pattern_indices[k])

def fast_backtest(indices, direction, tp_pct, sl_pct):
    pnls = []
    for idx in indices:
        if idx + 1 >= n_bars: continue
        entry = opens[idx + 1]
        if entry <= 0: continue
        if direction == 'LONG':
            tp_price = entry * (1 + tp_pct / 100)
            sl_price = entry * (1 - sl_pct / 100)
        else:
            tp_price = entry * (1 - tp_pct / 100)
            sl_price = entry * (1 + sl_pct / 100)
        for i in range(idx + 2, min(idx + 2 + MAX_BARS, n_bars)):
            if direction == 'LONG':
                hit_tp = highs[i] >= tp_price
                hit_sl = lows[i] <= sl_price
            else:
                hit_tp = lows[i] <= tp_price
                hit_sl = highs[i] >= sl_price
            if hit_tp and hit_sl:
                tp_d = abs(tp_price - entry)
                sl_d = abs(sl_price - entry)
                if tp_d <= sl_d:
                    pnls.append(tp_pct * LEVERAGE - FEE_PCT)
                else:
                    pnls.append(-sl_pct * LEVERAGE - FEE_PCT)
                break
            elif hit_tp:
                pnls.append(tp_pct * LEVERAGE - FEE_PCT)
                break
            elif hit_sl:
                pnls.append(-sl_pct * LEVERAGE - FEE_PCT)
                break
    return pnls

def monte_carlo_test(indices, direction, tp_pct, sl_pct, n_sims=3000):
    real_pnls = fast_backtest(indices, direction, tp_pct, sl_pct)
    if not real_pnls:
        return 1.0, 0, 0
    real_total = sum(real_pnls)
    valid_entries = np.arange(2, n_bars - 2)
    count_better = 0
    for _ in range(n_sims):
        rand_idx = np.random.choice(valid_entries, size=len(indices), replace=False)
        rand_pnls = fast_backtest(rand_idx, direction, tp_pct, sl_pct)
        if sum(rand_pnls) >= real_total:
            count_better += 1
    return count_better / n_sims, real_total, len(real_pnls)

def walk_forward(indices, direction, tp_pct, sl_pct, n_folds=5):
    sorted_idx = np.sort(indices)
    fold_size = len(sorted_idx) // n_folds
    if fold_size < 3: return 0
    profitable = 0
    for f in range(n_folds):
        s = f * fold_size
        e = s + fold_size if f < n_folds - 1 else len(sorted_idx)
        pnls = fast_backtest(sorted_idx[s:e], direction, tp_pct, sl_pct)
        if pnls and sum(pnls) > 0:
            profitable += 1
    return profitable

# ==========================================
# PHASE 1: Fee Impact + Random Baselines
# ==========================================
print("\n" + "="*80)
print("PHASE 1: FEE IMPACT & RANDOM BASELINES")
print("="*80)

tp_sl_grid = [(0.3, 0.3), (0.3, 0.5), (0.5, 0.5), (0.5, 0.3),
              (0.5, 1.0), (0.7, 0.7), (0.7, 1.0), (1.0, 1.0)]

print(f"\nFee impact (fee={FEE_PCT}%, leverage={LEVERAGE}x):")
print(f"{'TP/SL':>8} {'Gross TP':>10} {'Net TP':>10} {'Fee%ofTP':>10} {'BE WR':>8}")
for tp, sl in tp_sl_grid:
    gross_tp = tp * LEVERAGE
    net_tp = gross_tp - FEE_PCT
    net_sl = sl * LEVERAGE + FEE_PCT
    fee_ratio = FEE_PCT / gross_tp * 100
    be_wr = net_sl / (net_tp + net_sl) * 100
    print(f"  {tp}/{sl:>3}   {gross_tp:>8.1f}%  {net_tp:>8.2f}%  {fee_ratio:>8.0f}%  {be_wr:>6.1f}%")

np.random.seed(42)
valid_entries = np.arange(2, n_bars - 2)
rand_idx = np.random.choice(valid_entries, size=600, replace=False)

print(f"\nRandom baselines:")
print(f"{'TP/SL':>8} {'LONG WR':>9} {'SHORT WR':>10} {'L Edge':>8} {'S Edge':>8}")
baselines = {}
for tp, sl in tp_sl_grid:
    for d in ['LONG', 'SHORT']:
        pnls = fast_backtest(rand_idx, d, tp, sl)
        wr = sum(1 for x in pnls if x > 0) / len(pnls) * 100 if pnls else 0
        edge = sum(pnls) / len(pnls) if pnls else 0
        baselines[(tp, sl, d)] = {'wr': wr, 'edge': edge}
    lw = baselines[(tp, sl, 'LONG')]['wr']
    sw = baselines[(tp, sl, 'SHORT')]['wr']
    le = baselines[(tp, sl, 'LONG')]['edge']
    se = baselines[(tp, sl, 'SHORT')]['edge']
    print(f"  {tp}/{sl:>3}   {lw:>7.1f}%  {sw:>8.1f}%  {le:>+7.3f}%  {se:>+7.3f}%")

# ==========================================
# PHASE 2: Pattern Discovery
# ==========================================
print("\n" + "="*80)
print("PHASE 2: PATTERN DISCOVERY")
print("="*80)

candidates = []
n_checked = 0
for pat, idx in pattern_indices.items():
    if len(idx) < 15:
        continue
    for tp, sl in tp_sl_grid:
        for d in ['LONG', 'SHORT']:
            pnls = fast_backtest(idx, d, tp, sl)
            if len(pnls) < 15:
                continue
            wr = sum(1 for x in pnls if x > 0) / len(pnls) * 100
            edge = sum(pnls) / len(pnls)
            base_wr = baselines.get((tp, sl, d), {}).get('wr', 50)
            excess_wr = wr - base_wr
            total_pnl = sum(pnls)
            if excess_wr > 3 and edge > 0:
                candidates.append({
                    'pattern': pat, 'direction': d,
                    'tp': tp, 'sl': sl,
                    'trades': len(pnls), 'wr': wr,
                    'edge': edge, 'pnl': total_pnl,
                    'excess_wr': excess_wr, 'base_wr': base_wr,
                })
    n_checked += 1
    if n_checked % 400 == 0:
        print(f"  {n_checked} patterns, {len(candidates)} candidates")

print(f"\nPhase 2: {len(candidates)} candidates (LONG: {sum(1 for c in candidates if c['direction']=='LONG')}, SHORT: {sum(1 for c in candidates if c['direction']=='SHORT')})")

# Remove bidirectional (same pattern+TP/SL has both LONG and SHORT)
pat_dirs = defaultdict(set)
for c in candidates:
    pat_dirs[(c['pattern'], c['tp'], c['sl'])].add(c['direction'])
both_dirs = {k for k, v in pat_dirs.items() if len(v) == 2}
before = len(candidates)
candidates = [c for c in candidates if (c['pattern'], c['tp'], c['sl']) not in both_dirs]
print(f"Removed {before - len(candidates)} bidirectional, remaining: {len(candidates)}")

# Keep best TP/SL per pattern+direction
best = {}
for c in candidates:
    key = (c['pattern'], c['direction'])
    if key not in best or c['edge'] > best[key]['edge']:
        best[key] = c
candidates = sorted(best.values(), key=lambda x: -x['excess_wr'])
print(f"Deduplicated to best TP/SL per pattern: {len(candidates)}")

# ==========================================
# PHASE 3: Full Validation (MC + WF + Period)
# ==========================================
print("\n" + "="*80)
print("PHASE 3: FULL VALIDATION (top 80)")
print("="*80)

test_set = candidates[:80]
validated = []

for i, c in enumerate(test_set):
    pat = c['pattern']
    d = c['direction']
    tp, sl = c['tp'], c['sl']
    idx = pattern_indices[pat]

    p_mc, total_pnl, n_trades = monte_carlo_test(idx, d, tp, sl, n_sims=3000)

    wf = walk_forward(idx, d, tp, sl)

    period_pnls = {}
    for pname, mask in period_masks.items():
        pidx = idx[mask[idx]]
        if len(pidx) >= 3:
            pp = fast_backtest(pidx, d, tp, sl)
            period_pnls[pname] = {'pnl': sum(pp) if pp else 0, 'trades': len(pp)}
        else:
            period_pnls[pname] = {'pnl': 0, 'trades': 0}

    profitable_periods = sum(1 for v in period_pnls.values() if v['pnl'] > 0 and v['trades'] >= 3)

    c['p_mc'] = p_mc
    c['wf'] = wf
    c['h1_pnl'] = period_pnls['H1']['pnl']
    c['h2_pnl'] = period_pnls['H2']['pnl']
    c['h3_pnl'] = period_pnls['H3']['pnl']
    c['profitable_periods'] = profitable_periods

    # Strict: MC<0.05, WF>=3, 2/3 periods profitable
    if p_mc < 0.05 and wf >= 3 and profitable_periods >= 2:
        validated.append(c)

    if (i + 1) % 20 == 0:
        print(f"  {i+1}/80 checked, {len(validated)} validated")

print(f"\nPhase 3: {len(validated)} fully validated")

# ==========================================
# RESULTS
# ==========================================
print("\n" + "="*80)
print("VALIDATED PATTERNS")
print("="*80)

validated.sort(key=lambda x: -x['excess_wr'])
n_long = sum(1 for v in validated if v['direction'] == 'LONG')
n_short = sum(1 for v in validated if v['direction'] == 'SHORT')
print(f"Total: {len(validated)} (LONG: {n_long}, SHORT: {n_short})")

print(f"\n{'Pattern':<15} {'Dir':<6} {'TP/SL':>6} {'N':>4} {'WR':>6} {'ExcWR':>7} {'Edge':>7} {'PnL':>7} {'MC':>7} {'WF':>3} {'H1':>7} {'H2':>7} {'H3':>7}")
print("-" * 115)
for v in validated:
    print(f"  {v['pattern']:<13} {v['direction']:<6} {v['tp']}/{v['sl']:>3} {v['trades']:>4} {v['wr']:>5.1f}% {v['excess_wr']:>+6.1f}% {v['edge']:>+6.3f} {v['pnl']:>+6.1f}% {v['p_mc']:>6.4f} {v['wf']:>2}/5 {v['h1_pnl']:>+6.1f}% {v['h2_pnl']:>+6.1f}% {v['h3_pnl']:>+6.1f}%")

# ==========================================
# PORTFOLIO SIMULATION
# ==========================================
print("\n" + "="*80)
print("PORTFOLIO SIMULATION (compound, 270 days)")
print("="*80)

# Tight portfolio
tight_trades_list = []
for v in validated:
    idx = pattern_indices[v['pattern']]
    for ii in np.sort(idx):
        pnls = fast_backtest(np.array([ii]), v['direction'], v['tp'], v['sl'])
        if pnls:
            tight_trades_list.append((int(ii), pnls[0], v['direction']))

tight_trades_list.sort(key=lambda x: x[0])

equity = 100.0
peak = 100.0
mdd = 0
wins = 0
for _, pnl, _ in tight_trades_list:
    equity *= (1 + pnl / 100)
    peak = max(peak, equity)
    dd = (peak - equity) / peak * 100
    mdd = max(mdd, dd)
    if pnl > 0: wins += 1

n_trades = len(tight_trades_list)
total_pnl = equity - 100
wr = wins / n_trades * 100 if n_trades else 0

print(f"  Trades: {n_trades}")
print(f"  Win Rate: {wr:.1f}%")
print(f"  Total PnL: {total_pnl:+.1f}%")
print(f"  Max DD: {mdd:.1f}%")
print(f"  PnL/MDD: {total_pnl/mdd:.2f}" if mdd > 0 else "  PnL/MDD: N/A")
print(f"  LONG trades: {sum(1 for _,_,d in tight_trades_list if d=='LONG')}")
print(f"  SHORT trades: {sum(1 for _,_,d in tight_trades_list if d=='SHORT')}")

# Period breakdown
for pname, mask in period_masks.items():
    p_trades = [(idx, pnl, d) for idx, pnl, d in tight_trades_list if mask[idx]]
    p_pnl = sum(pnl for _, pnl, _ in p_trades)
    p_n = len(p_trades)
    p_long = sum(1 for _,_,d in p_trades if d=='LONG')
    p_short = sum(1 for _,_,d in p_trades if d=='SHORT')
    print(f"  {pname}: {p_n} trades (L:{p_long} S:{p_short}), PnL={p_pnl:+.1f}%")

# Save
output = {
    'timestamp': pd.Timestamp.now().isoformat(),
    'tp_sl_grid': [list(x) for x in tp_sl_grid],
    'n_validated': len(validated),
    'n_long': n_long,
    'n_short': n_short,
    'validated': validated,
    'portfolio': {
        'trades': n_trades, 'wr': wr, 'pnl': total_pnl, 'mdd': mdd,
    }
}
out_path = Path(__file__).parent / '../../results/tight_tpsl_validation.json'
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nSaved to {out_path}")
print("Done.")
