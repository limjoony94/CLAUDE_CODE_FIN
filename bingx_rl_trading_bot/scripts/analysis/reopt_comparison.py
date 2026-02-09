"""
v1.26.3 Re-optimization Comparison: Old (v1.26.2) vs New (v1.26.3) TP/SL
5 patterns side-by-side + portfolio impact analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'production'))
from pattern_5m.indicators import classify_candle
from pattern_5m.constants import (
    PATTERN_OPTIMAL_TPSL, VALIDATED_LONG_PATTERNS, VALIDATED_SHORT_PATTERNS,
)

MAX_BARS = 500
FEE_PCT = 0.10
LEVERAGE = 3
MC_SIMS = 10000

# Old (v1.26.2) vs New (v1.26.3) TP/SL
COMPARE = {
    'D-MU-U':    {'dir': 'LONG',  'old': (1.5, 2.0), 'new': (3.0, 2.0)},
    'GS-ST-ST':  {'dir': 'LONG',  'old': (1.5, 2.0), 'new': (2.0, 2.0)},
    'MU-BD-ST':  {'dir': 'LONG',  'old': (2.5, 3.0), 'new': (3.0, 3.0)},
    'D-BD-ST':   {'dir': 'SHORT', 'old': (2.5, 3.0), 'new': (3.0, 3.0)},
    'ST-DN-BU':  {'dir': 'SHORT', 'old': (2.0, 2.5), 'new': (3.0, 3.0)},
}

# Load data
data_path = Path(__file__).resolve().parent / '../../data/btc_5m_270days_reclassified.csv'
df = pd.read_csv(data_path)
highs = df['high'].values
lows = df['low'].values
opens = df['open'].values
closes = df['close'].values
n_bars = len(df)
df['timestamp'] = pd.to_datetime(df['timestamp'])
dates = df['timestamp'].values

period_masks = {
    'P1_May_Jul': (dates >= np.datetime64('2025-05-01')) & (dates < np.datetime64('2025-08-01')),
    'P2_Aug_Oct': (dates >= np.datetime64('2025-08-01')) & (dates < np.datetime64('2025-11-01')),
    'P3_Nov_Jan': (dates >= np.datetime64('2025-11-01')) & (dates < np.datetime64('2026-02-01')),
}

print("Classifying candles...")
body_abs_arr = np.abs(closes - opens)
avg_body_20_arr = pd.Series(body_abs_arr).rolling(20).mean().values
types = []
for i in range(n_bars):
    row = pd.Series({'open': opens[i], 'high': highs[i], 'low': lows[i], 'close': closes[i]})
    avg_b = avg_body_20_arr[i] if not pd.isna(avg_body_20_arr[i]) else 1.0
    types.append(classify_candle(row, avg_b).value)

pattern_indices = defaultdict(list)
for i in range(2, n_bars):
    pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
    pattern_indices[pat].append(i)
for k in pattern_indices:
    pattern_indices[k] = np.array(pattern_indices[k])


def bt_fixed(indices, direction, tp_pct, sl_pct):
    pnls = []
    for idx in indices:
        if idx + 1 >= n_bars:
            continue
        entry = opens[idx + 1]
        if entry <= 0:
            continue
        if direction == 'LONG':
            tpp = entry * (1 + tp_pct / 100)
            slp = entry * (1 - sl_pct / 100)
        else:
            tpp = entry * (1 - tp_pct / 100)
            slp = entry * (1 + sl_pct / 100)
        for j in range(idx + 2, min(idx + 2 + MAX_BARS, n_bars)):
            ht = (highs[j] >= tpp) if direction == 'LONG' else (lows[j] <= tpp)
            hs = (lows[j] <= slp) if direction == 'LONG' else (highs[j] >= slp)
            if ht and hs:
                pnls.append((tp_pct if abs(tpp - entry) <= abs(slp - entry) else -sl_pct) * LEVERAGE - FEE_PCT)
                break
            elif ht:
                pnls.append(tp_pct * LEVERAGE - FEE_PCT)
                break
            elif hs:
                pnls.append(-sl_pct * LEVERAGE - FEE_PCT)
                break
    return pnls


def mc_test(pnl_list, n_sims=MC_SIMS):
    if len(pnl_list) < 5:
        return 1.0
    pa = np.array(pnl_list)
    real = np.sum(pa)
    signs = np.random.choice([-1, 1], size=(n_sims, len(pa)))
    return float(np.mean(np.sum(pa * signs, axis=1) >= real))


def walk_forward_test(indices, direction, tp_pct, sl_pct, n_folds=5):
    si = np.sort(indices)
    fs = len(si) // n_folds
    if fs < 3:
        return 0, []
    fold_pnls = []
    profitable = 0
    for f in range(n_folds):
        s = f * fs
        e = s + fs if f < n_folds - 1 else len(si)
        pnls = bt_fixed(si[s:e], direction, tp_pct, sl_pct)
        fpnl = sum(pnls) if pnls else 0
        fold_pnls.append(fpnl)
        if pnls and sum(pnls) > 0:
            profitable += 1
    return profitable, fold_pnls


def period_test(indices, direction, tp_pct, sl_pct):
    results = []
    for name, mask in period_masks.items():
        pidx = indices[mask[indices]]
        if len(pidx) < 3:
            results.append((name, 0, 0))
            continue
        pnls = bt_fixed(pidx, direction, tp_pct, sl_pct)
        fpnl = sum(pnls) if pnls else 0
        results.append((name, len(pnls), fpnl))
    return results


def analyze(pat, direction, tp, sl, indices):
    pnls = bt_fixed(indices, direction, tp, sl)
    n = len(pnls)
    if n == 0:
        return None
    wr = sum(1 for x in pnls if x > 0) / n * 100
    total_pnl = sum(pnls)
    avg_w = np.mean([x for x in pnls if x > 0]) if any(x > 0 for x in pnls) else 0
    avg_l = np.mean([abs(x) for x in pnls if x <= 0]) if any(x <= 0 for x in pnls) else 0
    exp = (wr / 100) * avg_w - (1 - wr / 100) * avg_l
    mc = mc_test(pnls)
    wf, wf_folds = walk_forward_test(indices, direction, tp, sl)
    periods = period_test(indices, direction, tp, sl)
    rr = tp / sl if sl > 0 else 999
    return {
        'tp': tp, 'sl': sl, 'rr': rr,
        'trades': n, 'wr': wr, 'total_pnl': total_pnl,
        'avg_win': avg_w, 'avg_loss': avg_l, 'exp': exp,
        'mc': mc, 'wf': wf, 'wf_folds': wf_folds, 'periods': periods
    }


# ============================================================
# Individual pattern comparison
# ============================================================
print("=" * 90)
print("v1.26.3 Re-optimized 5 Patterns: Old (v1.26.2) vs New (v1.26.3) TP/SL")
print("=" * 90)

for pat, info in COMPARE.items():
    d = info['dir']
    indices = pattern_indices.get(pat, np.array([]))
    old = analyze(pat, d, info['old'][0], info['old'][1], indices)
    new = analyze(pat, d, info['new'][0], info['new'][1], indices)

    print(f"\n--- {pat} ({d}) ---")
    print(f"  Occurrences: {len(indices)}")

    print(f"  {'':>12} {'TP/SL':>7} {'R:R':>5} {'Tr':>4} {'WR':>6} {'PnL':>9} {'Exp':>8} {'MC':>8} {'WF':>4}")
    print(f"  OLD v1.26.2 {old['tp']}/{old['sl']:>5} {old['rr']:>5.2f} {old['trades']:>4} {old['wr']:>5.1f}% {old['total_pnl']:>+8.1f}% {old['exp']:>+7.3f}% {old['mc']:>8.4f} {old['wf']:>2}/5")
    print(f"  NEW v1.26.3 {new['tp']}/{new['sl']:>5} {new['rr']:>5.2f} {new['trades']:>4} {new['wr']:>5.1f}% {new['total_pnl']:>+8.1f}% {new['exp']:>+7.3f}% {new['mc']:>8.4f} {new['wf']:>2}/5")

    exp_delta = new['exp'] - old['exp']
    pnl_delta = new['total_pnl'] - old['total_pnl']
    mc_delta = new['mc'] - old['mc']
    print(f"  {'DELTA':>12} {'':>7} {'':>5} {new['trades']-old['trades']:>+4} {new['wr']-old['wr']:>+5.1f}% {pnl_delta:>+8.1f}% {exp_delta:>+7.3f}% {mc_delta:>+8.4f} {new['wf']-old['wf']:>+2}")

    # WF fold detail
    print(f"  WF folds (OLD): ", end='')
    for i, fp in enumerate(old['wf_folds']):
        s = 'OK' if fp > 0 else 'X'
        print(f"F{i+1}:{fp:+.1f}%({s}) ", end='')
    print()
    print(f"  WF folds (NEW): ", end='')
    for i, fp in enumerate(new['wf_folds']):
        s = 'OK' if fp > 0 else 'X'
        print(f"F{i+1}:{fp:+.1f}%({s}) ", end='')
    print()

    # Period detail
    print(f"  Periods (OLD): ", end='')
    for name, nt, fp in old['periods']:
        s = 'OK' if fp > 0 else ('X' if nt > 0 else '-')
        print(f"{name}:{nt}t/{fp:+.1f}%({s}) ", end='')
    print()
    print(f"  Periods (NEW): ", end='')
    for name, nt, fp in new['periods']:
        s = 'OK' if fp > 0 else ('X' if nt > 0 else '-')
        print(f"{name}:{nt}t/{fp:+.1f}%({s}) ", end='')
    print()

    # MC pass/fail
    old_mc_pass = old['mc'] < 0.01
    new_mc_pass = new['mc'] < 0.01
    print(f"  MC verdict:  OLD={'PASS' if old_mc_pass else 'FAIL'} ({old['mc']:.4f}),  NEW={'PASS' if new_mc_pass else 'FAIL'} ({new['mc']:.4f})")

    # Recommendation
    if old_mc_pass and not new_mc_pass:
        if new['exp'] > old['exp'] * 1.2:
            print(f"  >> KEEP NEW — MC fails but Exp {exp_delta:+.3f}% substantial, PnL {pnl_delta:+.1f}%")
        else:
            print(f"  >> ROLLBACK — MC fails and Exp gain ({exp_delta:+.3f}%) marginal")
    elif old_mc_pass and new_mc_pass:
        print(f"  >> KEEP NEW — both pass MC")
    elif not old_mc_pass and not new_mc_pass:
        if new['exp'] > old['exp']:
            print(f"  >> KEEP NEW — both fail MC, NEW has better Exp")
        else:
            print(f"  >> ROLLBACK — both fail MC, OLD has better Exp")
    else:
        print(f"  >> KEEP NEW — NEW passes MC where OLD did not")


# ============================================================
# Portfolio impact comparison
# ============================================================
print(f"\n{'='*90}")
print("Portfolio Impact: Full 52-pattern 1-pos-at-a-time simulation")
print(f"{'='*90}")


def collect_trades_1pos(pattern_map):
    raw_trades = []
    for i in range(2, n_bars):
        pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
        if pat not in pattern_map:
            continue
        direction, tp, sl = pattern_map[pat]
        if i + 1 >= n_bars:
            continue
        entry = opens[i + 1]
        if entry <= 0:
            continue
        entry_bar = i + 1
        if direction == 'LONG':
            tpp = entry * (1 + tp / 100)
            slp = entry * (1 - sl / 100)
        else:
            tpp = entry * (1 - tp / 100)
            slp = entry * (1 + sl / 100)
        for j in range(i + 2, min(i + 2 + MAX_BARS, n_bars)):
            ht = (highs[j] >= tpp) if direction == 'LONG' else (lows[j] <= tpp)
            hs = (lows[j] <= slp) if direction == 'LONG' else (highs[j] >= slp)
            if ht and hs:
                pnl = (tp if abs(tpp - entry) <= abs(slp - entry) else -sl) * LEVERAGE - FEE_PCT
                raw_trades.append((entry_bar, j, pnl))
                break
            elif ht:
                raw_trades.append((entry_bar, j, tp * LEVERAGE - FEE_PCT))
                break
            elif hs:
                raw_trades.append((entry_bar, j, -sl * LEVERAGE - FEE_PCT))
                break
    raw_trades.sort(key=lambda x: x[0])
    filtered = []
    last_exit = -1
    for eb, xb, pnl in raw_trades:
        if eb > last_exit:
            filtered.append((eb, xb, pnl))
            last_exit = xb
    return filtered


def eval_simple(trades):
    if not trades:
        return {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0, 'exp': 0}
    pnl_list = [t[2] for t in trades]
    cum = 0
    peak = 0
    mdd = 0
    wins = 0
    wps = []
    lps = []
    for p in pnl_list:
        cum += p
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > mdd:
            mdd = dd
        if p > 0:
            wins += 1
            wps.append(p)
        else:
            lps.append(abs(p))
    aw = np.mean(wps) if wps else 0
    al = np.mean(lps) if lps else 0
    tw = sum(wps)
    tl = sum(lps)
    wr = wins / len(pnl_list) * 100
    exp = (wr / 100) * aw - (1 - wr / 100) * al
    return {'pnl': cum, 'trades': len(pnl_list), 'wr': wr, 'mdd': mdd,
            'pf': tw / tl if tl > 0 else 999, 'exp': exp}


# Build both maps
map_new = {}
for p in VALIDATED_LONG_PATTERNS:
    tp, sl = PATTERN_OPTIMAL_TPSL[p]
    map_new[p] = ('LONG', tp, sl)
for p in VALIDATED_SHORT_PATTERNS:
    tp, sl = PATTERN_OPTIMAL_TPSL[p]
    map_new[p] = ('SHORT', tp, sl)

map_old = dict(map_new)
for pat, info in COMPARE.items():
    map_old[pat] = (info['dir'], info['old'][0], info['old'][1])

t_new = collect_trades_1pos(map_new)
t_old = collect_trades_1pos(map_old)
r_new = eval_simple(t_new)
r_old = eval_simple(t_old)

print(f"\n  {'':>16} {'v1.26.2 (old)':>14} {'v1.26.3 (new)':>14} {'Delta':>10}")
print(f"  {'-'*58}")
for k, label in [('trades', 'Trades'), ('wr', 'WR (%)'), ('pnl', 'PnL (%)'),
                  ('mdd', 'MDD (%)'), ('pf', 'PF'), ('exp', 'Exp/trade (%)')]:
    o = r_old[k]
    n = r_new[k]
    d = n - o
    if k in ('wr', 'pnl', 'mdd', 'exp'):
        print(f"  {label:>16} {o:>13.1f}% {n:>13.1f}% {d:>+9.1f}%")
    elif k == 'pf':
        print(f"  {label:>16} {o:>14.2f} {n:>14.2f} {d:>+10.2f}")
    else:
        print(f"  {label:>16} {o:>14} {n:>14} {d:>+10}")

pnl_mdd_old = r_old['pnl'] / r_old['mdd'] if r_old['mdd'] > 0 else 999
pnl_mdd_new = r_new['pnl'] / r_new['mdd'] if r_new['mdd'] > 0 else 999
print(f"  {'PnL/MDD':>16} {pnl_mdd_old:>13.2f}x {pnl_mdd_new:>13.2f}x {pnl_mdd_new - pnl_mdd_old:>+9.2f}x")

# Portfolio WF
print(f"\n  Portfolio WF comparison:")
fold_size = n_bars // 5
for label, pmap in [('v1.26.2', map_old), ('v1.26.3', map_new)]:
    wf_cnt = 0
    print(f"  {label}: ", end='')
    for f in range(5):
        s = f * fold_size
        e = s + fold_size if f < 4 else n_bars
        ft = []
        last_exit = -1
        for i in range(max(s, 2), e):
            pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
            if pat not in pmap:
                continue
            dr, tp, sl = pmap[pat]
            if i + 1 >= n_bars:
                continue
            eb = i + 1
            if eb <= last_exit:
                continue
            entry = opens[eb]
            if entry <= 0:
                continue
            if dr == 'LONG':
                tpp = entry * (1 + tp / 100)
                slp = entry * (1 - sl / 100)
            else:
                tpp = entry * (1 - tp / 100)
                slp = entry * (1 + sl / 100)
            for j in range(eb + 1, min(eb + MAX_BARS, n_bars)):
                ht = (highs[j] >= tpp) if dr == 'LONG' else (lows[j] <= tpp)
                hs = (lows[j] <= slp) if dr == 'LONG' else (highs[j] >= slp)
                if ht and hs:
                    pnl = (tp if abs(tpp - entry) <= abs(slp - entry) else -sl) * LEVERAGE - FEE_PCT
                    ft.append((eb, j, pnl))
                    last_exit = j
                    break
                elif ht:
                    ft.append((eb, j, tp * LEVERAGE - FEE_PCT))
                    last_exit = j
                    break
                elif hs:
                    ft.append((eb, j, -sl * LEVERAGE - FEE_PCT))
                    last_exit = j
                    break
        fpnl = sum(t[2] for t in ft)
        ok = fpnl > 0
        if ok:
            wf_cnt += 1
        print(f"F{f+1}:{fpnl:+.1f}%({('OK' if ok else 'X')}) ", end='')
    print(f"  => {wf_cnt}/5")

print("\nDone.")
