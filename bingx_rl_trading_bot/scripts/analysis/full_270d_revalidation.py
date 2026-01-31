"""
Full 270-Day Revalidation Pipeline (v4 - correct MAX_BARS=500)
Uses numpy global arrays for speed. Timeout trades excluded from PnL (standard).
"""

import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats
import json, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

print('=' * 70)
print('FULL 270-DAY REVALIDATION PIPELINE v4')
print(f'Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
print('=' * 70)

LEVERAGE = 3
FEE_PCT = 0.10
MAX_BARS = 500  # Production standard
N_MC = 5000

DISC_MIN_TRADES = 20
DISC_MIN_WR = 55.0
DISC_MIN_WF = 3
VAL_MIN_TRADES = 15
VAL_MIN_WR = 55.0
VAL_MIN_WF = 4

TP_GRID = [1.0, 1.5, 2.0, 2.5, 3.0]
SL_GRID = [1.0, 1.5, 2.0, 2.5, 3.0]

CURRENT_LONG = {
    'U-BU-U': (1.5, 2.0), 'ST-BD-DN': (2.0, 3.0),
    'DN-DN-DN': (1.0, 3.0), 'DN-U-U': (1.0, 3.0),
    'DN-DN-U': (1.0, 3.0), 'DN-ST-U': (1.0, 3.0),
    'U-ST-U': (1.0, 3.0), 'U-U-U': (1.5, 3.0),
}
CURRENT_SHORT = {
    'BD-BD-BD': (3.0, 2.5), 'DN-DN-BD': (1.5, 3.0),
    'MU-ST-DN': (1.0, 2.5), 'IH-DN-DN': (1.0, 3.0),
    'BD-ST-DN': (1.5, 3.0), 'BU-U-DN': (1.5, 2.5),
    'U-DN-DN': (1.0, 3.0), 'DN-U-DN': (2.0, 3.0),
    'DN-DN-ST': (1.5, 3.0), 'U-U-DN': (2.0, 3.0),
}

# Load
data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'btc_5m_270days.csv')
df = pd.read_csv(data_path)
print(f'Loaded: {len(df)} bars, {df["timestamp"].iloc[0]} ~ {df["timestamp"].iloc[-1]}')

# Candle classification
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

df['pattern'] = df['candle_type'].shift(2) + '-' + df['candle_type'].shift(1) + '-' + df['candle_type']

df['price_change_100'] = (df['close'] - df['close'].shift(100)) / df['close'].shift(100) * 100
df['regime'] = 'SIDEWAYS'
df.loc[df['price_change_100'] > 1.5, 'regime'] = 'BULL'
df.loc[df['price_change_100'] < -1.5, 'regime'] = 'BEAR'

print(f'Regime: BULL={int((df["regime"]=="BULL").sum())}, BEAR={int((df["regime"]=="BEAR").sum())}, SW={int((df["regime"]=="SIDEWAYS").sum())}')

# Global arrays
highs = df['high'].values
lows = df['low'].values
opens = df['open'].values
n_bars = len(df)
regimes = df['regime'].values
patterns = df['pattern'].values

# Build index
print('Building pattern index...')
pattern_indices = {}
for i in range(len(df)):
    p = patterns[i]
    if isinstance(p, str):
        if p not in pattern_indices:
            pattern_indices[p] = []
        pattern_indices[p].append(i)
for k in pattern_indices:
    pattern_indices[k] = np.array(pattern_indices[k])
print(f'Unique patterns: {len(pattern_indices)}')


def fast_backtest(indices, direction, tp_pct, sl_pct):
    """Backtest using global arrays with MAX_BARS=500."""
    pnls = []
    for idx in indices:
        if idx + 1 >= n_bars:
            continue
        entry = opens[idx + 1]
        if direction == 'LONG':
            tp_price = entry * (1 + tp_pct / 100)
            sl_price = entry * (1 - sl_pct / 100)
            for i in range(idx + 2, min(idx + 2 + MAX_BARS, n_bars)):
                if highs[i] >= tp_price:
                    pnls.append(tp_pct * LEVERAGE - FEE_PCT); break
                elif lows[i] <= sl_price:
                    pnls.append(-sl_pct * LEVERAGE - FEE_PCT); break
        else:
            tp_price = entry * (1 - tp_pct / 100)
            sl_price = entry * (1 + sl_pct / 100)
            for i in range(idx + 2, min(idx + 2 + MAX_BARS, n_bars)):
                if lows[i] <= tp_price:
                    pnls.append(tp_pct * LEVERAGE - FEE_PCT); break
                elif highs[i] >= sl_price:
                    pnls.append(-sl_pct * LEVERAGE - FEE_PCT); break
    if not pnls:
        return {'trades': 0, 'wr': 0, 'pnl': 0, 'edge': 0, 'pnls': []}
    t = len(pnls)
    w = sum(1 for p in pnls if p > 0)
    return {'trades': t, 'wr': w/t*100, 'pnl': sum(pnls), 'edge': sum(pnls)/t, 'pnls': pnls}


def walk_forward(indices, direction, tp, sl, n_folds=5):
    if len(indices) < 10:
        return 0
    fold_size = len(indices) // n_folds
    profitable = 0
    for i in range(n_folds):
        s = i * fold_size
        e = s + fold_size if i < n_folds - 1 else len(indices)
        r = fast_backtest(indices[s:e], direction, tp, sl)
        if r['pnl'] > 0:
            profitable += 1
    return profitable


def mc_test(pnls, n_sim=N_MC):
    if len(pnls) < 5:
        return 1.0
    actual = sum(pnls)
    arr = np.array(pnls)
    c = sum(1 for _ in range(n_sim) if np.sum(arr * np.random.choice([-1, 1], size=len(arr))) >= actual)
    return c / n_sim


def stat_tests(pnls):
    if len(pnls) < 5:
        return 1.0, 1.0
    if np.std(pnls) > 0:
        _, p_t = stats.ttest_1samp(pnls, 0)
        p_t = p_t / 2
    else:
        p_t = 0.0 if np.mean(pnls) > 0 else 1.0
    w = sum(1 for p in pnls if p > 0)
    try:
        p_b = stats.binomtest(w, len(pnls), 0.5, alternative='greater').pvalue
    except:
        p_b = 1.0
    return p_t, p_b


# Baseline check
print('\nBaseline (random entries, 1.5/1.5):')
rng_idxs = np.arange(500, n_bars - 600, 100)
for d in ['LONG', 'SHORT']:
    r = fast_backtest(rng_idxs, d, 1.5, 1.5)
    print(f'  {d}: {r["trades"]} trades, WR {r["wr"]:.1f}%')

# ============================================================
# PHASE 1: DISCOVERY (only patterns with high-count; skip grid, use default TP/SL first)
# ============================================================
print('\n' + '=' * 70)
print('PHASE 1: PATTERN DISCOVERY (MAX_BARS=500)')
print('=' * 70)

CANDLE_TYPES = ['D', 'DF', 'GS', 'H', 'IH', 'ST', 'MU', 'MD', 'BU', 'BD', 'U', 'DN']
discovered = []
checked = 0
t0 = time.time()

for c1 in CANDLE_TYPES:
    for c2 in CANDLE_TYPES:
        for c3 in CANDLE_TYPES:
            checked += 1
            pat = f'{c1}-{c2}-{c3}'
            if pat not in pattern_indices or len(pattern_indices[pat]) < DISC_MIN_TRADES:
                continue
            idxs = pattern_indices[pat]

            for direction in ['LONG', 'SHORT']:
                # Quick screen with default TP/SL
                r0 = fast_backtest(idxs, direction, 1.5, 2.0)
                if r0['trades'] < 10 or r0['wr'] < 45:
                    continue

                best_score, best_tp, best_sl, best_r = -999, 1.5, 2.0, None
                for tp in TP_GRID:
                    for sl in SL_GRID:
                        r = fast_backtest(idxs, direction, tp, sl)
                        if r['trades'] < 10:
                            continue
                        score = r['wr'] * 0.5 + min(r['edge'], 3.0) * 50
                        if score > best_score:
                            best_score, best_tp, best_sl, best_r = score, tp, sl, r

                if best_r and best_r['trades'] >= DISC_MIN_TRADES and best_r['wr'] >= DISC_MIN_WR:
                    wf = walk_forward(idxs, direction, best_tp, best_sl)
                    if wf >= DISC_MIN_WF:
                        discovered.append({
                            'pattern': pat, 'direction': direction,
                            'trades': best_r['trades'], 'wr': round(best_r['wr'], 1),
                            'edge': round(best_r['edge'], 2), 'pnl': round(best_r['pnl'], 1),
                            'tp': best_tp, 'sl': best_sl, 'wf': wf,
                        })

            if checked % 400 == 0:
                print(f'  {checked}/1728 ({time.time()-t0:.0f}s) found={len(discovered)}')

print(f'\nPhase 1: {len(discovered)} candidates in {time.time()-t0:.0f}s')

# ============================================================
# PHASE 2: STRICT VALIDATION
# ============================================================
print('\n' + '=' * 70)
print('PHASE 2: STRICT VALIDATION (WR>=55%, WF>=4/5, Edge>0)')
print('=' * 70)

validated = []
for d in discovered:
    pat, direction = d['pattern'], d['direction']
    tp, sl = d['tp'], d['sl']
    idxs = pattern_indices[pat]
    r = fast_backtest(idxs, direction, tp, sl)
    if r['trades'] < VAL_MIN_TRADES or r['wr'] < VAL_MIN_WR or r['edge'] <= 0:
        continue
    wf = walk_forward(idxs, direction, tp, sl)
    if wf < VAL_MIN_WF:
        continue
    p_t, p_b = stat_tests(r['pnls'])
    mc_p = mc_test(r['pnls'])
    rr = {}
    for regime in ['BULL', 'BEAR', 'SIDEWAYS']:
        r_idxs = idxs[np.array([regimes[i] == regime for i in idxs])]
        if len(r_idxs) >= 3:
            res = fast_backtest(r_idxs, direction, tp, sl)
            rr[regime] = (res['trades'], round(res['wr'], 1))
        else:
            rr[regime] = (0, 0)
    validated.append({
        'pattern': pat, 'direction': direction,
        'trades': r['trades'], 'wr': round(r['wr'], 1),
        'edge': round(r['edge'], 2), 'pnl': round(r['pnl'], 1),
        'tp': tp, 'sl': sl, 'wf': wf,
        'p_ttest': round(p_t, 4), 'p_binom': round(p_b, 4), 'p_mc': round(mc_p, 4),
        'bull': rr['BULL'], 'bear': rr['BEAR'], 'sw': rr['SIDEWAYS'],
        'pnls': r['pnls'],
    })

validated.sort(key=lambda x: (x['wf'], x['wr'], x['edge']), reverse=True)
longs_v = [v for v in validated if v['direction'] == 'LONG']
shorts_v = [v for v in validated if v['direction'] == 'SHORT']

print(f'Validated: {len(validated)} (LONG: {len(longs_v)}, SHORT: {len(shorts_v)})')

print('\n--- VALIDATED LONG ---')
for v in longs_v:
    sig = '*' if v['p_ttest'] < 0.05 else ''
    print(f"  {v['pattern']:12s} | {v['trades']:3d}t | WR {v['wr']:5.1f}% | E {v['edge']:+.2f} | WF {v['wf']}/5 | {v['tp']}/{v['sl']} | p={v['p_ttest']:.3f}{sig} MC={v['p_mc']:.3f}")

print('\n--- VALIDATED SHORT ---')
for v in shorts_v:
    sig = '*' if v['p_ttest'] < 0.05 else ''
    print(f"  {v['pattern']:12s} | {v['trades']:3d}t | WR {v['wr']:5.1f}% | E {v['edge']:+.2f} | WF {v['wf']}/5 | {v['tp']}/{v['sl']} | p={v['p_ttest']:.3f}{sig} MC={v['p_mc']:.3f}")

# ============================================================
# PHASE 3: TP/SL RE-OPTIMIZATION
# ============================================================
print('\n' + '=' * 70)
print('PHASE 3: TP/SL RE-OPTIMIZATION')
print('=' * 70)

optimized = []
for v in validated:
    pat, direction = v['pattern'], v['direction']
    idxs = pattern_indices[pat]
    best_score, best_tp, best_sl, best_r, best_wf = -999, v['tp'], v['sl'], None, v['wf']
    for tp in TP_GRID:
        for sl in SL_GRID:
            r = fast_backtest(idxs, direction, tp, sl)
            if r['trades'] < VAL_MIN_TRADES or r['wr'] < VAL_MIN_WR or r['edge'] <= 0:
                continue
            wf = walk_forward(idxs, direction, tp, sl)
            if wf < VAL_MIN_WF:
                continue
            score = r['wr'] * 0.3 + min(r['edge'], 3.0) * 40 + wf * 10
            if score > best_score:
                best_score, best_tp, best_sl, best_r, best_wf = score, tp, sl, r, wf
    if best_r and best_r['wr'] >= VAL_MIN_WR and best_r['edge'] > 0:
        p_t, _ = stat_tests(best_r['pnls'])
        mc_p = mc_test(best_r['pnls'])
        changed = (best_tp != v['tp'] or best_sl != v['sl'])
        optimized.append({
            'pattern': pat, 'direction': direction,
            'trades': best_r['trades'], 'wr': round(best_r['wr'], 1),
            'edge': round(best_r['edge'], 2), 'pnl': round(best_r['pnl'], 1),
            'tp': best_tp, 'sl': best_sl, 'wf': best_wf,
            'p_ttest': round(p_t, 4), 'p_mc': round(mc_p, 4),
            'old_tp': v['tp'], 'old_sl': v['sl'], 'tp_changed': changed,
        })

optimized.sort(key=lambda x: (x['direction'], -x['trades']))
for o in optimized:
    mark = ' <<CHANGED>>' if o['tp_changed'] else ''
    print(f"  {o['pattern']:12s} ({o['direction']:5s}) | {o['tp']}/{o['sl']} (was {o['old_tp']}/{o['old_sl']}) | WR {o['wr']:.1f}% | E {o['edge']:+.2f} | WF {o['wf']}/5{mark}")

# ============================================================
# PHASE 4: COMPARISON
# ============================================================
print('\n' + '=' * 70)
print('PHASE 4: 90-DAY vs 270-DAY COMPARISON')
print('=' * 70)

current_all = {**{k: ('LONG', *v) for k, v in CURRENT_LONG.items()},
               **{k: ('SHORT', *v) for k, v in CURRENT_SHORT.items()}}
opt_lookup = {(o['pattern'], o['direction']): o for o in optimized}
retained, removed, new_pats = [], [], []

print(f'\n{"Pattern":12s} | {"Dir":5s} | {"90d":5s} | {"270d":6s} | {"Trades":6s} | {"WR":7s} | {"Edge":7s} | {"WF":4s} | Status')
print('-' * 90)

for pat, (direction, tp90, sl90) in current_all.items():
    key = (pat, direction)
    if key in opt_lookup:
        o = opt_lookup[key]
        status = 'TP/SL CHANGED' if o['tp_changed'] else 'OK'
        retained.append(o)
        print(f"  {pat:12s} | {direction:5s} | {tp90}/{sl90} | {o['tp']}/{o['sl']}  | {o['trades']:6d} | {o['wr']:5.1f}% | {o['edge']:+5.2f} | {o['wf']}/5 | {status}")
    else:
        idxs = pattern_indices.get(pat, np.array([]))
        r = fast_backtest(idxs, direction, tp90, sl90) if len(idxs) > 0 else {'trades':0,'wr':0,'edge':0,'pnl':0,'pnls':[]}
        wf = walk_forward(idxs, direction, tp90, sl90) if len(idxs) >= 10 else 0
        reasons = []
        if r['trades'] < VAL_MIN_TRADES: reasons.append(f't={r["trades"]}')
        if r['wr'] < VAL_MIN_WR: reasons.append(f'WR={r["wr"]:.1f}%')
        if r['edge'] <= 0: reasons.append(f'E={r["edge"]:.2f}')
        if wf < VAL_MIN_WF: reasons.append(f'WF={wf}')
        removed.append({'pattern': pat, 'direction': direction, 'reason': ', '.join(reasons), **r, 'wf': wf})
        print(f"  {pat:12s} | {direction:5s} | {tp90}/{sl90} | ---    | {r['trades']:6d} | {r['wr']:5.1f}% | {r['edge']:+5.2f} | {wf}/5 | FAIL: {', '.join(reasons)}")

for o in optimized:
    if (o['pattern'], o['direction']) not in current_all:
        new_pats.append(o)

if new_pats:
    print(f'\n--- NEW PATTERNS (270-day) ---')
    for n in new_pats:
        print(f"  {n['pattern']:12s} ({n['direction']:5s}) | {n['tp']}/{n['sl']} | {n['trades']}t | WR {n['wr']:.1f}% | E {n['edge']:+.2f} | WF {n['wf']}/5 | p={n['p_ttest']:.3f}")

# Summary
print('\n' + '=' * 70)
print('FINAL SUMMARY')
print('=' * 70)
print(f'Current (90d): {len(current_all)} patterns')
print(f'270d: Retained={len(retained)}, Failed={len(removed)}, New={len(new_pats)}')

if removed:
    print(f'\nFAILED:')
    for r in removed:
        print(f"  {r['pattern']} ({r['direction']}): {r['reason']}")

tp_changed = [o for o in retained if o['tp_changed']]
if tp_changed:
    print(f'\nTP/SL CHANGES:')
    for o in tp_changed:
        print(f"  {o['pattern']} ({o['direction']}): {o['old_tp']}/{o['old_sl']} -> {o['tp']}/{o['sl']}")

if new_pats:
    print(f'\nNEW ({len(new_pats)}):')
    for n in new_pats[:20]:
        print(f"  {n['pattern']} ({n['direction']}): WR {n['wr']}%, E {n['edge']}, WF {n['wf']}/5")
    if len(new_pats) > 20:
        print(f'  ... and {len(new_pats)-20} more')

# Config
print('\n' + '=' * 70)
print('RECOMMENDED CONFIG')
print('=' * 70)
fl = sorted([o for o in optimized if o['direction'] == 'LONG'], key=lambda x: -x['trades'])
fs = sorted([o for o in optimized if o['direction'] == 'SHORT'], key=lambda x: -x['trades'])
print(f'\nLONG: {len(fl)}, SHORT: {len(fs)}')
print(f'\nPATTERN_OPTIMAL_TPSL = {{')
for o in sorted(optimized, key=lambda x: x['pattern']):
    print(f"    '{o['pattern']}': ({o['tp']}, {o['sl']}),  # {o['direction']}, WR {o['wr']}%, {o['trades']}t, WF {o['wf']}/5")
print('}')

# Save
ts = datetime.now().strftime('%Y%m%d_%H%M%S')
save = {
    'timestamp': ts, 'data_bars': len(df), 'max_bars': MAX_BARS,
    'retained': [{k: v for k, v in r.items() if k != 'pnls'} for r in retained],
    'removed': removed,
    'new_patterns': [{k: v for k, v in n.items() if k != 'pnls'} for n in new_pats],
    'tp_changed': [{k: v for k, v in o.items() if k != 'pnls'} for o in tp_changed],
    'final_long': [{k: v for k, v in o.items() if k != 'pnls'} for o in fl],
    'final_short': [{k: v for k, v in o.items() if k != 'pnls'} for o in fs],
}
out = os.path.join(os.path.dirname(__file__), '..', '..', 'results', f'revalidation_270d_{ts}.json')
with open(out, 'w') as f:
    json.dump(save, f, indent=2)
print(f'\nSaved: {out}')
print(f'Completed: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
