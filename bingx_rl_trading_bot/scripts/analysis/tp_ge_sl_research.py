"""
TP >= SL Constraint Pattern Discovery Research
================================================
현재 전략의 모든 패턴이 TP < SL (R:R 0.25~0.5)이므로
실거래에서 WR 하락 시 수익이 크게 감소하는 문제가 있음.

이 연구는 TP >= SL 제약 하에서 유효한 패턴을 발굴하여
실거래 성능 개선 가능성을 평가함.

검증 기준 (기존과 동일):
  - MC p-value < 0.05 (sign randomization, 10k sims)
  - WF >= 4/5 (walk-forward OOS)
  - 기간별 안정성 >= 2/3 (H1/H2/H3)
  - MIN_TRADES >= 15
  - WR >= 85% (quality filter)

변경사항:
  - TP/SL 그리드에서 TP >= SL 조합만 허용
  - 현재 포트폴리오(TP<SL)와 직접 비교
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import json
from datetime import datetime
import time
import sys

# Import canonical classify_candle from production
sys.path.insert(0, str(Path(__file__).parent.parent / 'production'))
from pattern_5m.indicators import classify_candle

# ============================================================
# Parameters
# ============================================================
MAX_BARS = 500
FEE_PCT = 0.10       # 0.05% x 2
SLIPPAGE = 0.02      # 0.02% buffer
LEVERAGE = 3
MIN_TRADES = 15
MC_SIMS = 10000
WR_THRESHOLD = 85.0  # Quality filter
TP_SL_GRID = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]  # Extended grid

# Current production portfolio for comparison
CURRENT_PORTFOLIO = {
    # LONG patterns
    "MD-BU-U":  ("LONG",  0.5, 2.0),
    "MU-MU-U":  ("LONG",  0.7, 2.0),
    "MU-U-MU":  ("LONG",  0.5, 2.0),
    "BU-BU-BD": ("LONG",  0.7, 2.0),
    "ST-MU-U":  ("LONG",  0.7, 2.0),
    "IH-DN-DN": ("LONG",  0.7, 2.0),
    # SHORT patterns
    "MD-ST-ST": ("SHORT", 0.5, 2.0),
    "U-MU-BU":  ("SHORT", 0.5, 2.0),
    "MU-BU-DN": ("SHORT", 0.7, 2.0),
    "ST-H-U":   ("SHORT", 0.5, 2.0),
    "ST-DN-H":  ("SHORT", 0.7, 2.0),
    "MD-MU-U":  ("SHORT", 1.0, 2.0),
    "BU-U-ST":  ("SHORT", 0.7, 2.0),
    "H-DN-ST":  ("SHORT", 0.7, 2.0),
    "DN-BD-BU": ("SHORT", 0.7, 2.0),
    "DN-BU-U":  ("SHORT", 1.0, 2.0),
    "DN-U-U":   ("SHORT", 0.5, 2.0),
    "ST-DN-U":  ("SHORT", 0.7, 2.0),
    "ST-DN-DN": ("SHORT", 0.7, 2.0),
    "U-ST-DN":  ("SHORT", 0.7, 2.0),
    "U-U-ST":   ("SHORT", 0.7, 2.0),
    "ST-U-DN":  ("SHORT", 0.7, 2.0),
    "ST-ST-U":  ("SHORT", 0.7, 2.0),
}

print("=" * 70)
print("TP >= SL Constraint Pattern Discovery Research")
print("=" * 70)
print(f"Constraint: TP >= SL (favorable risk-reward only)")
print(f"Grid: {TP_SL_GRID}")
print(f"Validation: MC<0.05, WF>=4/5, Period>=2/3, WR>={WR_THRESHOLD}%")
print()

# ============================================================
# Load data
# ============================================================
data_path = Path(__file__).resolve().parent / '../../data/btc_5m_270days_reclassified.csv'
df = pd.read_csv(data_path)
print(f"Loaded: {len(df)} bars from {data_path.name}")

highs = df['high'].values
lows = df['low'].values
opens = df['open'].values
closes = df['close'].values
n_bars = len(df)

df['timestamp'] = pd.to_datetime(df['timestamp'])
dates = df['timestamp'].values

period_masks = {
    'H1_May_Jul': (dates >= np.datetime64('2025-05-01')) & (dates < np.datetime64('2025-08-01')),
    'H2_Aug_Oct': (dates >= np.datetime64('2025-08-01')) & (dates < np.datetime64('2025-11-01')),
    'H3_Nov_Jan': (dates >= np.datetime64('2025-11-01')) & (dates < np.datetime64('2026-02-01')),
}

# ============================================================
# Candle Classification - PRODUCTION IDENTICAL
# ============================================================
print("Classifying candles (production-identical)...")
body_abs_arr = np.abs(closes - opens)
avg_body_20_arr = pd.Series(body_abs_arr).rolling(20).mean().values

types = []
for i in range(n_bars):
    row = pd.Series({
        'open': opens[i], 'high': highs[i], 'low': lows[i], 'close': closes[i]
    })
    avg_b = avg_body_20_arr[i] if not pd.isna(avg_body_20_arr[i]) else 1.0
    candle_type = classify_candle(row, avg_b)
    types.append(candle_type.value)  # Use short code (MD, BU, etc.)

# Verify against pre-classified data
if 'type_code' in df.columns:
    match_count = sum(1 for i in range(20, min(100, n_bars)) if types[i] == df['type_code'].iloc[i])
    print(f"  Classification verification: {match_count}/{min(80, n_bars-20)} match with pre-classified data")

# Build pattern index
pattern_indices = defaultdict(list)
for i in range(2, n_bars):
    pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
    pattern_indices[pat].append(i)
for k in pattern_indices:
    pattern_indices[k] = np.array(pattern_indices[k])

total_patterns = len(pattern_indices)
eligible = sum(1 for v in pattern_indices.values() if len(v) >= MIN_TRADES)
print(f"Total unique 3-candle patterns: {total_patterns}")
print(f"Patterns with >= {MIN_TRADES} trades: {eligible}")

# ============================================================
# Backtest engine
# ============================================================
def bt_fixed(indices, direction, tp_pct, sl_pct):
    """Returns list of per-trade PnL percentages."""
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
        return 0
    profitable = 0
    for f in range(n_folds):
        s = f * fs
        e = s + fs if f < n_folds - 1 else len(si)
        pnls = bt_fixed(si[s:e], direction, tp_pct, sl_pct)
        if pnls and sum(pnls) > 0:
            profitable += 1
    return profitable


def period_test(indices, direction, tp_pct, sl_pct):
    profitable = 0
    for mask in period_masks.values():
        pidx = indices[mask[indices]]
        if len(pidx) < 3:
            continue
        pnls = bt_fixed(pidx, direction, tp_pct, sl_pct)
        if pnls and sum(pnls) > 0:
            profitable += 1
    return profitable


# ============================================================
# Baselines (random entry benchmark)
# ============================================================
print("\nComputing baselines...")
sample_idx = np.arange(2, n_bars, 20)

baselines = {}
for tp in TP_SL_GRID:
    for sl in TP_SL_GRID:
        if tp < sl:
            continue  # Only compute baselines for TP >= SL
        wins_l, wins_s, total_l, total_s = 0, 0, 0, 0
        for idx in sample_idx:
            if idx + 1 >= n_bars:
                continue
            entry = opens[idx + 1]
            if entry <= 0:
                continue
            for d in ['LONG', 'SHORT']:
                if d == 'LONG':
                    tpp = entry * (1 + tp / 100)
                    slp = entry * (1 - sl / 100)
                else:
                    tpp = entry * (1 - tp / 100)
                    slp = entry * (1 + sl / 100)
                for j in range(idx + 2, min(idx + 2 + MAX_BARS, n_bars)):
                    ht = (highs[j] >= tpp) if d == 'LONG' else (lows[j] <= tpp)
                    hs = (lows[j] <= slp) if d == 'LONG' else (highs[j] >= slp)
                    if ht and hs:
                        if abs(tpp - entry) <= abs(slp - entry):
                            if d == 'LONG': wins_l += 1
                            else: wins_s += 1
                        if d == 'LONG': total_l += 1
                        else: total_s += 1
                        break
                    elif ht:
                        if d == 'LONG': wins_l += 1; total_l += 1
                        else: wins_s += 1; total_s += 1
                        break
                    elif hs:
                        if d == 'LONG': total_l += 1
                        else: total_s += 1
                        break
        baselines[(tp, sl)] = {
            'LONG': wins_l / total_l * 100 if total_l else 50,
            'SHORT': wins_s / total_s * 100 if total_s else 50,
        }
print(f"  Baselines computed for {len(baselines)} TP>=SL combos.")

# ============================================================
# Phase 1: TP >= SL Constrained Pattern Scan
# ============================================================
tp_ge_sl_combos = [(tp, sl) for tp in TP_SL_GRID for sl in TP_SL_GRID if tp >= sl]
print(f"\n{'='*70}")
print(f"Phase 1: TP >= SL Pattern Scan")
print(f"  TP/SL combos (TP>=SL only): {len(tp_ge_sl_combos)}")
print(f"  vs all combos: {len(TP_SL_GRID)**2}")
print(f"{'='*70}")

t0 = time.time()
valid_patterns = {}

for pat, indices in pattern_indices.items():
    if len(indices) < MIN_TRADES:
        continue
    for direction in ['LONG', 'SHORT']:
        best = None
        for tp, sl in tp_ge_sl_combos:
            pnls = bt_fixed(indices, direction, tp, sl)
            if len(pnls) < MIN_TRADES:
                continue
            wr = sum(1 for x in pnls if x > 0) / len(pnls) * 100

            # Quality filter: WR >= 85%
            if wr < WR_THRESHOLD:
                continue

            base = baselines.get((tp, sl), {}).get(direction, 50)
            excess = wr - base
            if excess < 5:  # Slightly relaxed since TP>=SL is harder
                continue
            total_pnl = sum(pnls)
            if total_pnl <= 0:
                continue

            # Quick MC
            mc = mc_test(pnls, 5000)
            if mc >= 0.05:
                continue

            # Walk-forward
            wf = walk_forward_test(indices, direction, tp, sl)
            if wf < 4:  # Strict: WF >= 4/5
                continue

            # Period stability
            pp = period_test(indices, direction, tp, sl)
            if pp < 2:
                continue

            # Full MC
            mc_full = mc_test(pnls, MC_SIMS)
            # Score: PnL * stability * statistical significance
            score = total_pnl * (wf / 5) * (1 - mc_full) * (tp / sl)  # Bonus for better R:R
            if best is None or score > best[-1]:
                best = (direction, tp, sl, len(pnls), wr, excess, mc_full, wf, pp, total_pnl, score)

        if best:
            key = f"{pat}_{best[0]}"
            valid_patterns[key] = (pat,) + best[:-1]  # drop score

elapsed = time.time() - t0
print(f"  Scan complete in {elapsed:.1f}s")
print(f"  Raw valid patterns (TP>=SL): {len(valid_patterns)}")

# Remove bidirectional (keep best direction per pattern)
pat_names = [v[0] for v in valid_patterns.values()]
bidir = {p for p, c in Counter(pat_names).items() if c > 1}
if bidir:
    to_remove = []
    for p in bidir:
        keys = [k for k, v in valid_patterns.items() if v[0] == p]
        keys.sort(key=lambda k: valid_patterns[k][10], reverse=True)
        to_remove.extend(keys[1:])
    for k in to_remove:
        del valid_patterns[k]
    print(f"  After removing bidirectional: {len(valid_patterns)}")

# ============================================================
# Phase 2: Results Table
# ============================================================
print(f"\n{'='*70}")
print("Phase 2: Validated Patterns (TP >= SL only)")
print(f"{'='*70}\n")

sorted_pats = sorted(valid_patterns.items(), key=lambda x: -x[1][10])

longs = [(k, v) for k, v in sorted_pats if v[1] == 'LONG']
shorts = [(k, v) for k, v in sorted_pats if v[1] == 'SHORT']

print(f"LONG patterns: {len(longs)}")
print(f"{'Pattern':<15} {'Dir':>5} {'TP':>4} {'SL':>4} {'R:R':>5} {'Trades':>6} {'WR':>6} {'Excess':>7} {'MC':>8} {'WF':>4} {'Per':>4} {'TotPnL':>8}")
print('-' * 95)
for k, v in longs:
    pat, d, tp, sl, nt, wr, exc, mc, wf, pp, tpnl = v
    rr = tp / sl
    print(f"{pat:<15} {d:>5} {tp:>4.1f} {sl:>4.1f} {rr:>5.2f} {nt:>6} {wr:>5.1f}% {exc:>+6.1f}% {mc:>8.4f} {wf:>3}/5 {pp:>3}/3 {tpnl:>+7.1f}%")

print(f"\nSHORT patterns: {len(shorts)}")
print(f"{'Pattern':<15} {'Dir':>5} {'TP':>4} {'SL':>4} {'R:R':>5} {'Trades':>6} {'WR':>6} {'Excess':>7} {'MC':>8} {'WF':>4} {'Per':>4} {'TotPnL':>8}")
print('-' * 95)
for k, v in shorts:
    pat, d, tp, sl, nt, wr, exc, mc, wf, pp, tpnl = v
    rr = tp / sl
    print(f"{pat:<15} {d:>5} {tp:>4.1f} {sl:>4.1f} {rr:>5.2f} {nt:>6} {wr:>5.1f}% {exc:>+6.1f}% {mc:>8.4f} {wf:>3}/5 {pp:>3}/3 {tpnl:>+7.1f}%")

# ============================================================
# Phase 3: Portfolio Comparison
# ============================================================
print(f"\n{'='*70}")
print("Phase 3: Portfolio Comparison (TP>=SL vs Current TP<SL)")
print(f"{'='*70}\n")

def collect_trades(pattern_map, bar_range=None):
    trades = []
    start = bar_range[0] if bar_range else 2
    end = bar_range[1] if bar_range else n_bars
    for i in range(max(start, 2), end):
        pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
        if pat not in pattern_map:
            continue
        direction, tp, sl = pattern_map[pat]
        if i + 1 >= n_bars:
            continue
        entry = opens[i + 1]
        if entry <= 0:
            continue
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
                trades.append((i + 1, (tp if abs(tpp - entry) <= abs(slp - entry) else -sl) * LEVERAGE - FEE_PCT))
                break
            elif ht:
                trades.append((i + 1, tp * LEVERAGE - FEE_PCT))
                break
            elif hs:
                trades.append((i + 1, -sl * LEVERAGE - FEE_PCT))
                break
    trades.sort(key=lambda x: x[0])
    return trades


def eval_trades(trades):
    if not trades:
        return {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pnl_mdd': 0, 'pf': 0,
                'avg_win': 0, 'avg_loss': 0, 'expectancy': 0}
    pnl_list = [t[1] for t in trades]
    equity = 100.0
    peak = 100.0
    max_dd = 0
    wins = 0
    total_win = 0
    total_loss = 0
    win_pnls = []
    loss_pnls = []
    for p in pnl_list:
        equity *= (1 + p / 100)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd:
            max_dd = dd
        if p > 0:
            wins += 1
            total_win += p
            win_pnls.append(p)
        else:
            total_loss += abs(p)
            loss_pnls.append(abs(p))
    avg_win = np.mean(win_pnls) if win_pnls else 0
    avg_loss = np.mean(loss_pnls) if loss_pnls else 0
    wr_pct = wins / len(pnl_list) * 100
    expectancy = (wr_pct / 100 * avg_win) - ((1 - wr_pct / 100) * avg_loss)
    return {
        'pnl': (equity / 100 - 1) * 100,
        'trades': len(pnl_list),
        'wr': wr_pct,
        'mdd': max_dd,
        'pnl_mdd': (equity / 100 - 1) * 100 / max_dd if max_dd > 0 else 999,
        'pf': total_win / total_loss if total_loss > 0 else 999,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'expectancy': expectancy,
    }


# Build TP>=SL pattern map
tp_ge_sl_map = {}
for k, v in valid_patterns.items():
    pat, d, tp, sl = v[0], v[1], v[2], v[3]
    tp_ge_sl_map[pat] = (d, tp, sl)

portfolios = {
    'A: Current (TP<SL)': CURRENT_PORTFOLIO,
    'B: TP>=SL (new)': tp_ge_sl_map,
}

for label, pmap in portfolios.items():
    fn = lambda br=None, pm=pmap: collect_trades(pm, br)
    trades = fn()
    res = eval_trades(trades)

    # Walk-Forward
    fold_size = n_bars // 5
    wf_count = 0
    wf_detail = []
    for f in range(5):
        s = f * fold_size
        e = s + fold_size if f < 4 else n_bars
        ft = fn(br=(s, e))
        fr = eval_trades(ft)
        wf_detail.append(fr)
        if fr['pnl'] > 0:
            wf_count += 1

    # Period stability
    period_results = {}
    for pname, mask in period_masks.items():
        bidx = np.where(mask)[0]
        if len(bidx) == 0:
            continue
        pt = fn(br=(bidx[0], bidx[-1] + 1))
        period_results[pname] = eval_trades(pt)
    pp_count = sum(1 for r in period_results.values() if r['pnl'] > 0)

    # MC
    pnl_list = np.array([t[1] for t in trades]) if trades else np.array([])
    mc_val = 1.0
    if len(pnl_list) >= 5:
        real_eq = np.prod(1 + pnl_list / 100)
        count = sum(1 for _ in range(MC_SIMS)
                    if np.prod(1 + pnl_list * np.random.choice([-1, 1], len(pnl_list)) / 100) >= real_eq)
        mc_val = count / MC_SIMS

    # R:R stats
    tp_vals = [v[1] for v in pmap.values()]
    sl_vals = [v[2] for v in pmap.values()]
    avg_tp = np.mean(tp_vals) if tp_vals else 0
    avg_sl = np.mean(sl_vals) if sl_vals else 0
    avg_rr = avg_tp / avg_sl if avg_sl > 0 else 0

    print(f"--- {label} ({len(pmap)} patterns) ---")
    print(f"  Avg TP: {avg_tp:.2f}%, Avg SL: {avg_sl:.2f}%, Avg R:R: {avg_rr:.2f}")
    print(f"  Trades: {res['trades']}, WR: {res['wr']:.1f}%, PnL: {res['pnl']:+.1f}%")
    print(f"  Avg Win: {res['avg_win']:.2f}%, Avg Loss: {res['avg_loss']:.2f}%")
    print(f"  Expectancy: {res['expectancy']:.3f}% per trade")
    print(f"  MDD: {res['mdd']:.1f}%, PnL/MDD: {res['pnl_mdd']:.1f}, PF: {res['pf']:.2f}")
    print(f"  MC p-value: {mc_val:.4f}")
    print(f"  Walk-Forward: {wf_count}/5")
    for i, fr in enumerate(wf_detail):
        s = "OK" if fr['pnl'] > 0 else "FAIL"
        print(f"    Fold {i+1}: {fr['trades']:>4}t, {fr['wr']:>5.1f}% WR, {fr['pnl']:>+10.1f}% PnL, MDD {fr['mdd']:>5.1f}% [{s}]")
    print(f"  Period stability: {pp_count}/3")
    for pn, pr in period_results.items():
        s = "OK" if pr['pnl'] > 0 else "FAIL"
        print(f"    {pn}: {pr['trades']:>4}t, {pr['wr']:>5.1f}% WR, {pr['pnl']:>+10.1f}% PnL [{s}]")
    print()

# ============================================================
# Phase 4: Overlap Analysis
# ============================================================
print(f"{'='*70}")
print("Phase 4: Overlap Analysis")
print(f"{'='*70}\n")

current_pats = set(CURRENT_PORTFOLIO.keys())
new_pats = set(tp_ge_sl_map.keys())
overlap = current_pats & new_pats
only_current = current_pats - new_pats
only_new = new_pats - current_pats

print(f"Current portfolio: {len(current_pats)} patterns")
print(f"TP>=SL portfolio: {len(new_pats)} patterns")
print(f"Overlap: {len(overlap)} patterns")
print(f"Only in current (TP<SL): {len(only_current)}")
print(f"Only in TP>=SL: {len(only_new)}")

if overlap:
    print(f"\nOverlapping patterns (direction change?):")
    for pat in sorted(overlap):
        cur = CURRENT_PORTFOLIO[pat]
        new = tp_ge_sl_map[pat]
        print(f"  {pat}: Current=({cur[0]}, TP={cur[1]}, SL={cur[2]}) → New=({new[0]}, TP={new[1]}, SL={new[2]})")

if only_new:
    print(f"\nNew patterns (TP>=SL only):")
    for pat in sorted(only_new):
        v = tp_ge_sl_map[pat]
        print(f"  {pat}: {v[0]}, TP={v[1]}, SL={v[2]}, R:R={v[1]/v[2]:.2f}")

# ============================================================
# Phase 5: Sensitivity Analysis - What if WR drops?
# ============================================================
print(f"\n{'='*70}")
print("Phase 5: WR Drop Sensitivity Analysis")
print(f"{'='*70}\n")

print("Simulated impact when WR drops by X%:")
print(f"{'Portfolio':<25} {'Actual WR':>10} {'WR-5%':>10} {'WR-10%':>10} {'WR-15%':>10}")
print('-' * 70)

for label, pmap in portfolios.items():
    trades = collect_trades(pmap)
    if not trades:
        continue
    pnl_list = [t[1] for t in trades]
    actual_wr = sum(1 for p in pnl_list if p > 0) / len(pnl_list) * 100
    avg_win = np.mean([p for p in pnl_list if p > 0]) if any(p > 0 for p in pnl_list) else 0
    avg_loss = np.mean([abs(p) for p in pnl_list if p <= 0]) if any(p <= 0 for p in pnl_list) else 0

    results_row = [f"{actual_wr:.1f}%"]
    for wr_drop in [5, 10, 15]:
        sim_wr = (actual_wr - wr_drop) / 100
        exp = sim_wr * avg_win - (1 - sim_wr) * avg_loss
        results_row.append(f"{exp:+.3f}%")

    print(f"{label:<25} {'  '.join(results_row)}")

print()
print("Interpretation: TP>=SL portfolios are more robust to WR drops")
print("because losses are smaller relative to wins.")

# ============================================================
# Phase 6: Combined Portfolio (TP>=SL + best of current)
# ============================================================
print(f"\n{'='*70}")
print("Phase 6: Hybrid Analysis (current patterns retested with TP>=SL)")
print(f"{'='*70}\n")

# For each current pattern, find best TP>=SL setting
hybrid_map = {}
print(f"{'Pattern':<15} {'Dir':>5} {'Cur TP/SL':>10} {'Best TP>=SL':>12} {'Cur WR':>7} {'New WR':>7} {'Cur PnL':>8} {'New PnL':>8}")
print('-' * 90)

for pat, (direction, cur_tp, cur_sl) in CURRENT_PORTFOLIO.items():
    indices = pattern_indices.get(pat, np.array([]))
    if len(indices) < MIN_TRADES:
        continue

    # Current performance
    cur_pnls = bt_fixed(indices, direction, cur_tp, cur_sl)
    cur_wr = sum(1 for x in cur_pnls if x > 0) / len(cur_pnls) * 100 if cur_pnls else 0
    cur_total = sum(cur_pnls) if cur_pnls else 0

    # Find best TP>=SL
    best_tp_ge_sl = None
    best_score = -999
    for tp, sl in tp_ge_sl_combos:
        pnls = bt_fixed(indices, direction, tp, sl)
        if len(pnls) < MIN_TRADES:
            continue
        wr = sum(1 for x in pnls if x > 0) / len(pnls) * 100
        total_pnl = sum(pnls)
        if total_pnl <= 0:
            continue
        score = total_pnl * (tp / sl)  # Favor higher R:R
        if score > best_score:
            best_score = score
            best_tp_ge_sl = (tp, sl, wr, total_pnl, len(pnls))

    if best_tp_ge_sl:
        btp, bsl, bwr, bpnl, btrades = best_tp_ge_sl
        tag = "OK" if bpnl > 0 else "FAIL"
        print(f"{pat:<15} {direction:>5} {f'{cur_tp}/{cur_sl}':>10} {f'{btp}/{bsl}':>12} {cur_wr:>6.1f}% {bwr:>6.1f}% {cur_total:>+7.1f}% {bpnl:>+7.1f}% [{tag}]")
        if bpnl > 0:
            hybrid_map[pat] = (direction, btp, bsl)
    else:
        print(f"{pat:<15} {direction:>5} {f'{cur_tp}/{cur_sl}':>10} {'N/A':>12} {cur_wr:>6.1f}% {'N/A':>7} {cur_total:>+7.1f}% {'N/A':>8} [NO VALID TP>=SL]")

# Evaluate hybrid portfolio
if hybrid_map:
    print(f"\nHybrid Portfolio: {len(hybrid_map)} patterns (current patterns with TP>=SL)")
    trades = collect_trades(hybrid_map)
    res = eval_trades(trades)

    tp_vals = [v[1] for v in hybrid_map.values()]
    sl_vals = [v[2] for v in hybrid_map.values()]
    avg_rr = np.mean(tp_vals) / np.mean(sl_vals) if sl_vals else 0

    print(f"  Avg R:R: {avg_rr:.2f}")
    print(f"  Trades: {res['trades']}, WR: {res['wr']:.1f}%, PnL: {res['pnl']:+.1f}%")
    print(f"  Avg Win: {res['avg_win']:.2f}%, Avg Loss: {res['avg_loss']:.2f}%")
    print(f"  Expectancy: {res['expectancy']:.3f}% per trade")
    print(f"  MDD: {res['mdd']:.1f}%, PF: {res['pf']:.2f}")

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}\n")

print("Current Portfolio (TP<SL):")
cur_trades = collect_trades(CURRENT_PORTFOLIO)
cur_res = eval_trades(cur_trades)
tp_vals = [v[1] for v in CURRENT_PORTFOLIO.values()]
sl_vals = [v[2] for v in CURRENT_PORTFOLIO.values()]
print(f"  {len(CURRENT_PORTFOLIO)} patterns, Avg R:R={np.mean(tp_vals)/np.mean(sl_vals):.2f}")
print(f"  {cur_res['trades']}t, WR {cur_res['wr']:.1f}%, PnL {cur_res['pnl']:+.1f}%, MDD {cur_res['mdd']:.1f}%")
print(f"  Avg Win: {cur_res['avg_win']:.2f}%, Avg Loss: {cur_res['avg_loss']:.2f}%")
if cur_res['avg_win'] > 0:
    print(f"  1 loss = {cur_res['avg_loss']/cur_res['avg_win']:.1f}x wins (wipeout ratio)")

print(f"\nTP>=SL Portfolio:")
if tp_ge_sl_map:
    new_trades = collect_trades(tp_ge_sl_map)
    new_res = eval_trades(new_trades)
    tp_vals = [v[1] for v in tp_ge_sl_map.values()]
    sl_vals = [v[2] for v in tp_ge_sl_map.values()]
    print(f"  {len(tp_ge_sl_map)} patterns, Avg R:R={np.mean(tp_vals)/np.mean(sl_vals):.2f}")
    print(f"  {new_res['trades']}t, WR {new_res['wr']:.1f}%, PnL {new_res['pnl']:+.1f}%, MDD {new_res['mdd']:.1f}%")
    print(f"  Avg Win: {new_res['avg_win']:.2f}%, Avg Loss: {new_res['avg_loss']:.2f}%")
    if new_res['avg_win'] > 0:
        print(f"  1 loss = {new_res['avg_loss']/new_res['avg_win']:.1f}x wins (wipeout ratio)")
else:
    print("  No valid patterns found with TP>=SL constraint!")

# Save results
output = {
    'timestamp': datetime.now().isoformat(),
    'constraint': 'TP >= SL',
    'tp_ge_sl_patterns': [],
    'current_comparison': {
        'current_pnl': round(cur_res['pnl'], 1),
        'current_wr': round(cur_res['wr'], 1),
        'current_trades': cur_res['trades'],
    },
}

for k, v in sorted_pats:
    pat, d, tp, sl, nt, wr, exc, mc, wf, pp, tpnl = v
    output['tp_ge_sl_patterns'].append({
        'pattern': pat, 'direction': d, 'tp': tp, 'sl': sl,
        'rr': round(tp/sl, 2), 'trades': nt, 'wr': round(wr, 1),
        'mc': round(mc, 4), 'wf': wf, 'periods': pp, 'total_pnl': round(tpnl, 1),
    })

if tp_ge_sl_map:
    output['tp_ge_sl_comparison'] = {
        'pnl': round(new_res['pnl'], 1),
        'wr': round(new_res['wr'], 1),
        'trades': new_res['trades'],
    }

out_path = Path(__file__).resolve().parent / '../../results/tp_ge_sl_research.json'
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nResults saved to {out_path}")
print("\nDone.")
