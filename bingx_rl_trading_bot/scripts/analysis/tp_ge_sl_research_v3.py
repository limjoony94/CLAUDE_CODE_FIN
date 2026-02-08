"""
TP/SL Research v3 — R:R 개선 연구
===================================
v2 문제점 수정:
  1) PnL: compound → simple returns (sum of % PnL)
  2) MDD: 1-position-at-a-time 시뮬레이션
  3) 패턴: 69+ → Top 15-25 엄선 (MC<0.005, WF=5/5)
  4) 신호 충돌: 시간순 정렬, 포지션 중복 방지
  5) WR 기준: BE+10% → BE+15% safety margin

연구 목표:
  Phase 1: 기존 23 패턴에 R:R>=0.75 TP/SL 재테스트
  Phase 2: v2 combined에서 엄선된 신규 R:R>=1.0 패턴
  Phase 3: 1-position-at-a-time 현실적 포트폴리오 시뮬레이션
  Phase 4: WR 하락 내성 분석
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import json
from datetime import datetime
import time
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'production'))
from pattern_5m.indicators import classify_candle

# ============================================================
# Parameters
# ============================================================
MAX_BARS = 500
FEE_PCT = 0.10          # 0.05% * 2 (round trip)
SLIPPAGE = 0.02
LEVERAGE = 3
MIN_TRADES = 20          # v3: stricter than v2 (15)
MC_SIMS = 10000
TP_SL_GRID = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]
SAFETY_MARGIN = 15       # v3: BE + 15% (v2: BE + 10%)

# Current portfolio (v1.25.4)
CURRENT_PORTFOLIO = {
    "MD-BU-U": ("LONG", 0.5, 2.0), "MU-MU-U": ("LONG", 0.7, 2.0),
    "MU-U-MU": ("LONG", 0.5, 2.0), "BU-BU-BD": ("LONG", 0.7, 2.0),
    "ST-MU-U": ("LONG", 0.7, 2.0), "IH-DN-DN": ("LONG", 0.7, 2.0),
    "MD-ST-ST": ("SHORT", 0.5, 2.0), "U-MU-BU": ("SHORT", 0.5, 2.0),
    "MU-BU-DN": ("SHORT", 0.7, 2.0), "ST-H-U": ("SHORT", 0.5, 2.0),
    "ST-DN-H": ("SHORT", 0.7, 2.0), "MD-MU-U": ("SHORT", 1.0, 2.0),
    "BU-U-ST": ("SHORT", 0.7, 2.0), "H-DN-ST": ("SHORT", 0.7, 2.0),
    "DN-BD-BU": ("SHORT", 0.7, 2.0), "DN-BU-U": ("SHORT", 1.0, 2.0),
    "DN-U-U": ("SHORT", 0.5, 2.0), "ST-DN-U": ("SHORT", 0.7, 2.0),
    "ST-DN-DN": ("SHORT", 0.7, 2.0), "U-ST-DN": ("SHORT", 0.7, 2.0),
    "U-U-ST": ("SHORT", 0.7, 2.0), "ST-U-DN": ("SHORT", 0.7, 2.0),
    "ST-ST-U": ("SHORT", 0.7, 2.0),
}

print("=" * 70)
print("TP/SL Research v3 — R:R Improvement Study")
print("  Fixes: simple returns, 1-pos-at-a-time, strict filters, BE+15%")
print("=" * 70)

# ============================================================
# Load and classify data
# ============================================================
data_path = Path(__file__).resolve().parent / '../../data/btc_5m_270days_reclassified.csv'
df = pd.read_csv(data_path)
print(f"\nLoaded: {len(df)} bars")

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

eligible = sum(1 for v in pattern_indices.values() if len(v) >= MIN_TRADES)
print(f"Patterns with >= {MIN_TRADES} trades: {eligible}")


# ============================================================
# Backtest engine (same as v2 — per-trade PnL list)
# ============================================================
def bt_fixed(indices, direction, tp_pct, sl_pct):
    """Backtest: returns list of per-trade PnL (leveraged, fee-adjusted)."""
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
                # Both hit in same bar: TP wins if target is closer
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
    """Monte Carlo sign randomization test."""
    if len(pnl_list) < 5:
        return 1.0
    pa = np.array(pnl_list)
    real = np.sum(pa)
    signs = np.random.choice([-1, 1], size=(n_sims, len(pa)))
    return float(np.mean(np.sum(pa * signs, axis=1) >= real))


def walk_forward_test(indices, direction, tp_pct, sl_pct, n_folds=5):
    """Walk-forward: count folds with positive PnL."""
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
    """Period stability: count periods with positive PnL."""
    profitable = 0
    for mask in period_masks.values():
        pidx = indices[mask[indices]]
        if len(pidx) < 3:
            continue
        pnls = bt_fixed(pidx, direction, tp_pct, sl_pct)
        if pnls and sum(pnls) > 0:
            profitable += 1
    return profitable


def collect_trades_1pos(pattern_map):
    """
    Collect trades with 1-position-at-a-time constraint.
    Returns list of (entry_bar, exit_bar, pnl_pct) tuples, time-sorted.
    """
    # First collect all possible trade signals with their exit bars
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

    # Sort by entry bar
    raw_trades.sort(key=lambda x: x[0])

    # Apply 1-position-at-a-time filter
    filtered = []
    last_exit = -1
    for entry_bar, exit_bar, pnl in raw_trades:
        if entry_bar > last_exit:
            filtered.append((entry_bar, exit_bar, pnl))
            last_exit = exit_bar

    return filtered


def eval_trades_simple(trades):
    """
    Evaluate trades using SIMPLE returns (sum of PnL %).
    v3 fix: no compound overflow, realistic MDD.
    """
    if not trades:
        return {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0,
                'avg_win': 0, 'avg_loss': 0, 'expectancy': 0, 'pnl_mdd': 0}

    pnl_list = [t[2] for t in trades]
    cum_pnl = 0
    peak_pnl = 0
    max_dd = 0
    wins = 0
    win_pnls = []
    loss_pnls = []

    for p in pnl_list:
        cum_pnl += p
        if cum_pnl > peak_pnl:
            peak_pnl = cum_pnl
        dd = peak_pnl - cum_pnl
        if dd > max_dd:
            max_dd = dd
        if p > 0:
            wins += 1
            win_pnls.append(p)
        else:
            loss_pnls.append(abs(p))

    avg_win = np.mean(win_pnls) if win_pnls else 0
    avg_loss = np.mean(loss_pnls) if loss_pnls else 0
    total_win = sum(win_pnls)
    total_loss = sum(loss_pnls)
    wr_pct = wins / len(pnl_list) * 100
    expectancy = (wr_pct / 100 * avg_win) - ((1 - wr_pct / 100) * avg_loss)

    return {
        'pnl': cum_pnl,
        'trades': len(pnl_list),
        'wr': wr_pct,
        'mdd': max_dd,
        'pf': total_win / total_loss if total_loss > 0 else 999,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'expectancy': expectancy,
        'pnl_mdd': cum_pnl / max_dd if max_dd > 0 else 999,
    }


# ============================================================
# Break-even WR table
# ============================================================
print(f"\n--- Break-Even WR by R:R (safety margin = +{SAFETY_MARGIN}%) ---")
print(f"{'R:R':>6} {'TP%':>6} {'SL%':>6} {'Win PnL':>10} {'Loss PnL':>10} {'BE WR':>8} {'Safe WR':>8}")
print("-" * 65)

be_table = {}
for tp in TP_SL_GRID:
    for sl in TP_SL_GRID:
        rr = tp / sl
        win_pnl = tp * LEVERAGE - FEE_PCT - SLIPPAGE
        loss_pnl = sl * LEVERAGE + FEE_PCT + SLIPPAGE
        be_wr = loss_pnl / (win_pnl + loss_pnl) * 100
        safe_wr = be_wr + SAFETY_MARGIN
        be_table[(tp, sl)] = (rr, win_pnl, loss_pnl, be_wr, safe_wr)

seen_rr = set()
for (tp, sl), (rr, wp, lp, be, safe) in sorted(be_table.items(), key=lambda x: x[1][0]):
    rr_key = f"{rr:.2f}"
    if rr_key not in seen_rr and rr >= 0.5:
        seen_rr.add(rr_key)
        print(f"{rr:>6.2f} {tp:>5.1f}% {sl:>5.1f}% {wp:>+9.2f}% {lp:>9.2f}% {be:>7.1f}% {safe:>7.1f}%")


# ============================================================
# Baseline WR computation
# ============================================================
print("\nComputing baselines...")
sample_idx = np.arange(2, n_bars, 20)

baselines = {}
for tp in TP_SL_GRID:
    for sl in TP_SL_GRID:
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
print("  Done.")


# ============================================================
# Phase 1: Existing 23 Patterns — R:R >= 0.75 Re-test
# ============================================================
print(f"\n{'='*70}")
print("Phase 1: Existing 23 Patterns — R:R >= 0.75 TP/SL Re-test")
print(f"{'='*70}")

t0 = time.time()

phase1_results = {}  # pattern -> {current: {...}, best_rr: {...}}

for pat, (direction, cur_tp, cur_sl) in CURRENT_PORTFOLIO.items():
    indices = pattern_indices.get(pat, np.array([]))
    if len(indices) < MIN_TRADES:
        continue

    # Current performance
    cur_pnls = bt_fixed(indices, direction, cur_tp, cur_sl)
    cur_wr = sum(1 for x in cur_pnls if x > 0) / len(cur_pnls) * 100 if cur_pnls else 0
    cur_total = sum(cur_pnls)
    cur_rr = cur_tp / cur_sl

    # Scan R:R >= 0.75 combos
    best = None
    best_score = -999

    for tp in TP_SL_GRID:
        for sl in TP_SL_GRID:
            rr = tp / sl
            if rr < 0.75:
                continue

            _, _, _, _, safe_wr = be_table[(tp, sl)]
            pnls = bt_fixed(indices, direction, tp, sl)
            if len(pnls) < MIN_TRADES:
                continue

            wr = sum(1 for x in pnls if x > 0) / len(pnls) * 100
            if wr < safe_wr:
                continue

            base = baselines.get((tp, sl), {}).get(direction, 50)
            excess = wr - base
            if excess < 5:
                continue

            total_pnl = sum(pnls)
            if total_pnl <= 0:
                continue

            mc = mc_test(pnls, 3000)
            if mc >= 0.05:
                continue

            wf = walk_forward_test(indices, direction, tp, sl)
            if wf < 4:
                continue

            pp = period_test(indices, direction, tp, sl)
            if pp < 2:
                continue

            # Full MC
            mc_full = mc_test(pnls, MC_SIMS)
            if mc_full >= 0.05:
                continue

            # Score: weighted by expectancy and R:R
            avg_win = np.mean([x for x in pnls if x > 0]) if any(x > 0 for x in pnls) else 0
            avg_loss = np.mean([abs(x) for x in pnls if x <= 0]) if any(x <= 0 for x in pnls) else 0
            exp = (wr / 100) * avg_win - (1 - wr / 100) * avg_loss
            score = exp * len(pnls) * (wf / 5)

            if score > best_score:
                best_score = score
                best = {
                    'tp': tp, 'sl': sl, 'rr': round(rr, 2),
                    'trades': len(pnls), 'wr': round(wr, 1),
                    'mc': round(mc_full, 4), 'wf': wf, 'pp': pp,
                    'total_pnl': round(total_pnl, 1),
                    'avg_win': round(avg_win, 2), 'avg_loss': round(avg_loss, 2),
                    'expectancy': round(exp, 3),
                }

    phase1_results[pat] = {
        'direction': direction,
        'current': {
            'tp': cur_tp, 'sl': cur_sl, 'rr': round(cur_rr, 2),
            'trades': len(cur_pnls), 'wr': round(cur_wr, 1),
            'total_pnl': round(cur_total, 1),
        },
        'best_rr': best,
    }

elapsed = time.time() - t0
print(f"  Scan complete in {elapsed:.1f}s")

# Print comparison table
print(f"\n  {'Pattern':<12} {'Dir':>5} | {'Cur TP/SL':>9} {'R:R':>5} {'WR':>6} {'PnL':>8} | {'New TP/SL':>9} {'R:R':>5} {'WR':>6} {'Exp':>7} {'PnL':>8} {'WF':>3}")
print("  " + "-" * 100)

improved = 0
for pat in sorted(phase1_results.keys()):
    r = phase1_results[pat]
    c = r['current']
    b = r['best_rr']
    d = r['direction']

    cur_str = f"{c['tp']}/{c['sl']}"
    if b:
        new_str = f"{b['tp']}/{b['sl']}"
        mark = " *" if b['expectancy'] > 0 else ""
        print(f"  {pat:<12} {d:>5} | {cur_str:>9} {c['rr']:>5.2f} {c['wr']:>5.1f}% {c['total_pnl']:>+7.1f}% | {new_str:>9} {b['rr']:>5.2f} {b['wr']:>5.1f}% {b['expectancy']:>+6.3f}% {b['total_pnl']:>+7.1f}% {b['wf']:>2}/5{mark}")
        if b['expectancy'] > 0:
            improved += 1
    else:
        print(f"  {pat:<12} {d:>5} | {cur_str:>9} {c['rr']:>5.2f} {c['wr']:>5.1f}% {c['total_pnl']:>+7.1f}% | {'---':>9} {'---':>5} {'---':>6} {'---':>7} {'---':>8} {'---':>3}")

print(f"\n  Patterns with valid R:R>=0.75 alternative: {improved}/23")


# ============================================================
# Phase 2: New R:R >= 1.0 Pattern Discovery (strict filters)
# ============================================================
print(f"\n{'='*70}")
print("Phase 2: New R:R >= 1.0 Pattern Discovery")
print(f"  Filters: MC < 0.005, WF = 5/5, Period >= 3/3, Trades >= {MIN_TRADES}")
print(f"{'='*70}")

t0 = time.time()

# Scan all patterns for R:R >= 1.0
new_rr_candidates = {}  # key -> entry

for pat, indices in pattern_indices.items():
    if len(indices) < MIN_TRADES:
        continue
    # Skip patterns already in current portfolio
    if pat in CURRENT_PORTFOLIO:
        continue

    for direction in ['LONG', 'SHORT']:
        for tp in TP_SL_GRID:
            for sl in TP_SL_GRID:
                rr = tp / sl
                if rr < 1.0:
                    continue

                _, _, _, _, safe_wr = be_table[(tp, sl)]
                pnls = bt_fixed(indices, direction, tp, sl)
                if len(pnls) < MIN_TRADES:
                    continue

                wr = sum(1 for x in pnls if x > 0) / len(pnls) * 100
                if wr < safe_wr:
                    continue

                base = baselines.get((tp, sl), {}).get(direction, 50)
                excess = wr - base
                if excess < 5:
                    continue

                total_pnl = sum(pnls)
                if total_pnl <= 0:
                    continue

                # Quick MC screen
                mc = mc_test(pnls, 3000)
                if mc >= 0.01:
                    continue

                # Strict WF = 5/5
                wf = walk_forward_test(indices, direction, tp, sl)
                if wf < 5:
                    continue

                # Strict Period = 3/3
                pp = period_test(indices, direction, tp, sl)
                if pp < 3:
                    continue

                # Full MC with strict threshold
                mc_full = mc_test(pnls, MC_SIMS)
                if mc_full >= 0.005:
                    continue

                avg_win = np.mean([x for x in pnls if x > 0]) if any(x > 0 for x in pnls) else 0
                avg_loss = np.mean([abs(x) for x in pnls if x <= 0]) if any(x <= 0 for x in pnls) else 0
                exp = (wr / 100) * avg_win - (1 - wr / 100) * avg_loss

                key = f"{pat}_{direction}_{tp}_{sl}"
                new_rr_candidates[key] = {
                    'pattern': pat, 'direction': direction,
                    'tp': tp, 'sl': sl, 'rr': round(rr, 2),
                    'trades': len(pnls), 'wr': round(wr, 1),
                    'mc': round(mc_full, 4), 'wf': wf, 'pp': pp,
                    'total_pnl': round(total_pnl, 1),
                    'avg_win': round(avg_win, 2), 'avg_loss': round(avg_loss, 2),
                    'expectancy': round(exp, 3),
                }

elapsed = time.time() - t0
print(f"  Scan complete in {elapsed:.1f}s")
print(f"  Total valid entries: {len(new_rr_candidates)}")

# Best per pattern-direction (by expectancy * trades)
best_per_pat = {}
for k, v in new_rr_candidates.items():
    pat_key = f"{v['pattern']}_{v['direction']}"
    score = v['expectancy'] * v['trades'] * (v['wf'] / 5)
    if pat_key not in best_per_pat or score > best_per_pat[pat_key][1]:
        best_per_pat[pat_key] = (v, score)

# Remove bidirectional conflicts
pat_names = [v[0]['pattern'] for v in best_per_pat.values()]
bidir = {p for p, c in Counter(pat_names).items() if c > 1}
if bidir:
    to_remove = []
    for p in bidir:
        keys = [k for k, v in best_per_pat.items() if v[0]['pattern'] == p]
        keys.sort(key=lambda k: best_per_pat[k][1], reverse=True)
        to_remove.extend(keys[1:])
    for k in to_remove:
        del best_per_pat[k]

# Sort by total_pnl
sorted_new = sorted(best_per_pat.values(), key=lambda x: -x[0]['total_pnl'])

print(f"  Unique patterns (best per direction): {len(sorted_new)}")

if sorted_new:
    longs = [v for v, _ in sorted_new if v['direction'] == 'LONG']
    shorts = [v for v, _ in sorted_new if v['direction'] == 'SHORT']

    print(f"\n  LONG: {len(longs)}")
    print(f"  {'Pattern':<12} {'TP':>4} {'SL':>4} {'R:R':>5} {'Trades':>6} {'WR':>6} {'MC':>8} {'WF':>4} {'Per':>4} {'Exp':>7} {'PnL':>8}")
    print("  " + "-" * 80)
    for v in longs:
        print(f"  {v['pattern']:<12} {v['tp']:>4.1f} {v['sl']:>4.1f} {v['rr']:>5.2f} {v['trades']:>6} {v['wr']:>5.1f}% {v['mc']:>8.4f} {v['wf']:>3}/5 {v['pp']:>3}/3 {v['expectancy']:>+6.3f}% {v['total_pnl']:>+7.1f}%")

    print(f"\n  SHORT: {len(shorts)}")
    print(f"  {'Pattern':<12} {'TP':>4} {'SL':>4} {'R:R':>5} {'Trades':>6} {'WR':>6} {'MC':>8} {'WF':>4} {'Per':>4} {'Exp':>7} {'PnL':>8}")
    print("  " + "-" * 80)
    for v in shorts:
        print(f"  {v['pattern']:<12} {v['tp']:>4.1f} {v['sl']:>4.1f} {v['rr']:>5.2f} {v['trades']:>6} {v['wr']:>5.1f}% {v['mc']:>8.4f} {v['wf']:>3}/5 {v['pp']:>3}/3 {v['expectancy']:>+6.3f}% {v['total_pnl']:>+7.1f}%")


# ============================================================
# Phase 3: Realistic Portfolio Simulation (1-pos-at-a-time)
# ============================================================
print(f"\n{'='*70}")
print("Phase 3: Realistic Portfolio Simulation (1-position-at-a-time, simple returns)")
print(f"{'='*70}\n")

# Build portfolio maps
portfolio_current = CURRENT_PORTFOLIO.copy()

# Current-RR: replace TP/SL where valid R:R>=0.75 alternative exists
portfolio_current_rr = {}
for pat, r in phase1_results.items():
    d = r['direction']
    if r['best_rr'] and r['best_rr']['expectancy'] > 0:
        portfolio_current_rr[pat] = (d, r['best_rr']['tp'], r['best_rr']['sl'])
    else:
        portfolio_current_rr[pat] = (d, r['current']['tp'], r['current']['sl'])

# New-RR: only new patterns
portfolio_new_rr = {}
for v, _ in sorted_new:
    portfolio_new_rr[v['pattern']] = (v['direction'], v['tp'], v['sl'])

# Hybrid: Current-RR + New-RR
portfolio_hybrid = {}
portfolio_hybrid.update(portfolio_current_rr)
for pat, val in portfolio_new_rr.items():
    if pat not in portfolio_hybrid:
        portfolio_hybrid[pat] = val

portfolios = {
    'Current': portfolio_current,
    'Current-RR': portfolio_current_rr,
    'New-RR': portfolio_new_rr,
    'Hybrid': portfolio_hybrid,
}

# Evaluate each
print(f"{'Portfolio':<14} {'Pats':>4} {'Trades':>6} {'Tr/day':>6} {'WR':>6} {'Exp/tr':>8} {'PnL':>10} {'MDD':>8} {'PnL/MDD':>8} {'PF':>6} {'AvgW':>6} {'AvgL':>6} {'W/L':>5}")
print("-" * 110)

portfolio_results = {}

for label, pmap in portfolios.items():
    if not pmap:
        print(f"{label:<14} {'(empty)':>4}")
        continue

    trades = collect_trades_1pos(pmap)
    res = eval_trades_simple(trades)
    tr_per_day = res['trades'] / 270 if res['trades'] > 0 else 0

    wipeout = res['avg_loss'] / res['avg_win'] if res['avg_win'] > 0 else 0

    print(f"{label:<14} {len(pmap):>4} {res['trades']:>6} {tr_per_day:>6.1f} {res['wr']:>5.1f}% {res['expectancy']:>+7.3f}% {res['pnl']:>+9.1f}% {res['mdd']:>7.1f}% {res['pnl_mdd']:>7.2f}x {res['pf']:>5.2f} {res['avg_win']:>5.2f}% {res['avg_loss']:>5.2f}% {wipeout:>4.1f}x")

    # Walk-forward at portfolio level
    fold_size = n_bars // 5
    wf_count = 0
    wf_details = []
    for f in range(5):
        s = f * fold_size
        e = s + fold_size if f < 4 else n_bars
        # Build filtered pmap for this fold
        ft = []
        last_exit = -1
        for i in range(max(s, 2), e):
            pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
            if pat not in pmap:
                continue
            direction, tp, sl = pmap[pat]
            if i + 1 >= n_bars:
                continue
            entry_bar = i + 1
            if entry_bar <= last_exit:
                continue
            entry = opens[entry_bar]
            if entry <= 0:
                continue
            if direction == 'LONG':
                tpp = entry * (1 + tp / 100)
                slp = entry * (1 - sl / 100)
            else:
                tpp = entry * (1 - tp / 100)
                slp = entry * (1 + sl / 100)
            for j in range(entry_bar + 1, min(entry_bar + MAX_BARS, n_bars)):
                ht = (highs[j] >= tpp) if direction == 'LONG' else (lows[j] <= tpp)
                hs = (lows[j] <= slp) if direction == 'LONG' else (highs[j] >= slp)
                if ht and hs:
                    pnl = (tp if abs(tpp - entry) <= abs(slp - entry) else -sl) * LEVERAGE - FEE_PCT
                    ft.append((entry_bar, j, pnl))
                    last_exit = j
                    break
                elif ht:
                    ft.append((entry_bar, j, tp * LEVERAGE - FEE_PCT))
                    last_exit = j
                    break
                elif hs:
                    ft.append((entry_bar, j, -sl * LEVERAGE - FEE_PCT))
                    last_exit = j
                    break

        fr = eval_trades_simple(ft)
        wf_details.append(fr)
        if fr['pnl'] > 0:
            wf_count += 1

    portfolio_results[label] = {
        'patterns': len(pmap),
        'result': res,
        'wf': wf_count,
        'wf_details': wf_details,
    }

    print(f"  WF: {wf_count}/5  ", end="")
    for i, fr in enumerate(wf_details):
        s = "OK" if fr['pnl'] > 0 else "FAIL"
        print(f"[F{i+1}: {fr['trades']}t {fr['pnl']:+.1f}% {s}]  ", end="")
    print()


# ============================================================
# Phase 4: WR Drop Resilience Analysis
# ============================================================
print(f"\n{'='*70}")
print("Phase 4: WR Drop Resilience Analysis")
print(f"{'='*70}\n")

print(f"{'Portfolio':<14} {'WR':>6} | {'WR-5%':>10} {'WR-10%':>10} {'WR-15%':>10} {'WR-20%':>10} | {'Max Drop':>10}")
print("-" * 90)

wr_resilience = {}

for label, pmap in portfolios.items():
    if not pmap:
        continue

    trades = collect_trades_1pos(pmap)
    if not trades:
        continue

    pnl_list = [t[2] for t in trades]
    actual_wr = sum(1 for p in pnl_list if p > 0) / len(pnl_list) * 100
    avg_win = np.mean([p for p in pnl_list if p > 0]) if any(p > 0 for p in pnl_list) else 0
    avg_loss = np.mean([abs(p) for p in pnl_list if p <= 0]) if any(p <= 0 for p in pnl_list) else 0

    results_str = f"{actual_wr:>5.1f}% |"
    max_drop = 0

    exps = {}
    for wr_drop in [5, 10, 15, 20]:
        sim_wr = (actual_wr - wr_drop) / 100
        if sim_wr < 0:
            sim_wr = 0
        exp = sim_wr * avg_win - (1 - sim_wr) * avg_loss
        exps[wr_drop] = exp
        status = "OK" if exp > 0 else "LOSS"
        results_str += f" {exp:>+7.3f}% {status:>4}"
        if exp > 0:
            max_drop = wr_drop

    # Find exact break-even WR drop
    # WR_sim * avg_win = (1 - WR_sim) * avg_loss
    # WR_sim = avg_loss / (avg_win + avg_loss)
    be_wr = avg_loss / (avg_win + avg_loss) * 100 if (avg_win + avg_loss) > 0 else 0
    max_tolerable = actual_wr - be_wr

    print(f"{label:<14} {results_str} | {max_tolerable:>+8.1f}pp")

    wr_resilience[label] = {
        'actual_wr': round(actual_wr, 1),
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'max_tolerable_wr_drop': round(max_tolerable, 1),
        'be_wr': round(be_wr, 1),
        'sensitivities': {str(d): round(e, 3) for d, e in exps.items()},
    }


# ============================================================
# Final Summary & Recommendations
# ============================================================
print(f"\n{'='*70}")
print("FINAL SUMMARY & RECOMMENDATIONS")
print(f"{'='*70}\n")

# Side-by-side comparison of key portfolios
print("Key Portfolio Comparison (1-position-at-a-time, simple returns):")
print(f"{'':>25} {'Current':>12} {'Current-RR':>12} {'New-RR':>12} {'Hybrid':>12}")
print("-" * 75)

metrics_order = ['patterns', 'trades', 'wr', 'expectancy', 'pnl', 'mdd', 'pnl_mdd', 'pf', 'avg_win', 'avg_loss']
metric_labels = {
    'patterns': 'Patterns',
    'trades': 'Trades',
    'wr': 'WR (%)',
    'expectancy': 'Exp/trade (%)',
    'pnl': 'Total PnL (%)',
    'mdd': 'Max DD (%)',
    'pnl_mdd': 'PnL/MDD',
    'pf': 'Profit Factor',
    'avg_win': 'Avg Win (%)',
    'avg_loss': 'Avg Loss (%)',
}

for m in metrics_order:
    label_str = metric_labels[m]
    vals = []
    for pname in ['Current', 'Current-RR', 'New-RR', 'Hybrid']:
        pr = portfolio_results.get(pname, {})
        if not pr:
            vals.append('---')
            continue
        if m == 'patterns':
            vals.append(f"{pr['patterns']}")
        else:
            v = pr['result'].get(m, 0)
            if m in ('wr', 'pnl', 'mdd', 'expectancy', 'avg_win', 'avg_loss'):
                vals.append(f"{v:+.1f}" if m in ('pnl', 'expectancy') else f"{v:.1f}")
            elif m == 'pnl_mdd':
                vals.append(f"{v:.2f}x")
            elif m == 'pf':
                vals.append(f"{v:.2f}")
            else:
                vals.append(f"{v}")
    print(f"{label_str:>25} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12} {vals[3]:>12}")

# WR resilience summary
print(f"\nWR Drop Resilience (max tolerable WR drop before break-even):")
for pname in ['Current', 'Current-RR', 'New-RR', 'Hybrid']:
    wr = wr_resilience.get(pname, {})
    if wr:
        print(f"  {pname:<14}: WR {wr['actual_wr']}% → BE at {wr['be_wr']}% → max drop {wr['max_tolerable_wr_drop']:+.1f}pp")

# WF summary
print(f"\nWalk-Forward (portfolio level):")
for pname in ['Current', 'Current-RR', 'New-RR', 'Hybrid']:
    pr = portfolio_results.get(pname, {})
    if pr:
        print(f"  {pname:<14}: {pr['wf']}/5")

# Recommendations
print(f"\n--- Actionable Insights ---")
cur = portfolio_results.get('Current', {}).get('result', {})
cur_rr = portfolio_results.get('Current-RR', {}).get('result', {})
new_rr = portfolio_results.get('New-RR', {}).get('result', {})
hybrid = portfolio_results.get('Hybrid', {}).get('result', {})

if cur_rr and cur:
    if cur_rr.get('expectancy', 0) > cur.get('expectancy', 0):
        print(f"  [1] Current-RR IMPROVES expectancy: {cur.get('expectancy',0):+.3f}% → {cur_rr.get('expectancy',0):+.3f}%")
        print(f"      → Recommend: Update existing patterns' TP/SL")
    else:
        print(f"  [1] Current-RR does NOT improve expectancy.")
        print(f"      → Recommend: Keep current TP/SL settings")

if new_rr:
    print(f"  [2] New-RR patterns: {new_rr.get('trades', 0)} trades, Exp {new_rr.get('expectancy',0):+.3f}%")
    if new_rr.get('trades', 0) >= 100 and new_rr.get('expectancy', 0) > 0:
        print(f"      → Recommend: Consider portfolio expansion")
    else:
        print(f"      → Insufficient volume or negative expectancy")

wr_cur = wr_resilience.get('Current', {}).get('max_tolerable_wr_drop', 0)
wr_hyb = wr_resilience.get('Hybrid', {}).get('max_tolerable_wr_drop', 0)
if wr_hyb > wr_cur:
    print(f"  [3] Hybrid has BETTER WR resilience: {wr_cur:+.1f}pp → {wr_hyb:+.1f}pp")
else:
    print(f"  [3] Hybrid does NOT improve WR resilience: {wr_cur:+.1f}pp → {wr_hyb:+.1f}pp")


# ============================================================
# Save results
# ============================================================
output = {
    'timestamp': datetime.now().isoformat(),
    'research': 'TP/SL Research v3 — R:R Improvement Study',
    'methodology': {
        'returns': 'simple (sum of % PnL)',
        'position_limit': '1-position-at-a-time',
        'safety_margin': f'BE + {SAFETY_MARGIN}%',
        'new_pattern_filters': 'MC<0.005, WF=5/5, Period=3/3, Trades>=20, R:R>=1.0',
    },
    'phase1_current_rr_retest': {},
    'phase2_new_rr_patterns': [],
    'phase3_portfolio_comparison': {},
    'phase4_wr_resilience': wr_resilience,
}

# Phase 1 detail
for pat, r in phase1_results.items():
    output['phase1_current_rr_retest'][pat] = r

# Phase 2 detail
for v, score in sorted_new:
    output['phase2_new_rr_patterns'].append(v)

# Phase 3 detail
for label in ['Current', 'Current-RR', 'New-RR', 'Hybrid']:
    pr = portfolio_results.get(label, {})
    if pr:
        res = pr['result']
        output['phase3_portfolio_comparison'][label] = {
            'patterns': pr['patterns'],
            'trades': res['trades'],
            'trades_per_day': round(res['trades'] / 270, 1),
            'wr': round(res['wr'], 1),
            'expectancy': round(res['expectancy'], 3),
            'pnl': round(res['pnl'], 1),
            'mdd': round(res['mdd'], 1),
            'pnl_mdd': round(res['pnl_mdd'], 2),
            'pf': round(res['pf'], 2),
            'avg_win': round(res['avg_win'], 2),
            'avg_loss': round(res['avg_loss'], 2),
            'wf': pr['wf'],
        }

out_path = Path(__file__).resolve().parent / '../../results/tp_ge_sl_research_v3.json'
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nResults saved to {out_path}")
print("\nDone.")
