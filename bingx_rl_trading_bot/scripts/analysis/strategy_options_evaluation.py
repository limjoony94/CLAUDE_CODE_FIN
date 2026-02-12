"""
Strategy Options Evaluation
============================
Pattern-Rediscovery WF 결과(90% look-ahead) 기반 전략 옵션 종합 평가.

Options:
  A. Status Quo: 51 patterns, current TP/SL (look-ahead baseline)
  B. Robust-4: 50%+ rediscovery (U-H-MU, IH-MD-MD, MU-DF-U, BU-ST-GS)
  C. Top-17: 25%+ rediscovery
  D. Dynamic: per-fold rediscovered from our 51, train TP/SL (genuine forward)
  E. Fresh-50: per-fold top 50 from full universe scan, train TP/SL ("restart every period")
  F. Fresh-20: per-fold top 20 from full universe scan, train TP/SL ("focused restart")
  G. Expanding Adaptive: expanding window re-scan, use all MC-passing from our 51 + current TP/SL

Methodology:
  - Same expanding-window WF as pattern_rediscovery_wf.py
  - 4 OOS folds of 54 days each
  - 1-position-at-a-time portfolio simulation
  - All options evaluated on identical OOS periods
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import json
from datetime import datetime
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent / 'production'))
from pattern_5m.indicators import classify_candle
from pattern_5m.constants import (
    PATTERN_OPTIMAL_TPSL, VALIDATED_LONG_PATTERNS, VALIDATED_SHORT_PATTERNS,
)

# ============================================================
# Parameters
# ============================================================
MAX_BARS = 500
FEE_PCT = 0.10
LEVERAGE = 3
MC_SIMS = 10000
MIN_TRADES = 10
MIN_EDGE_PP = 5.0
MC_THRESHOLD = 0.01

TP_GRID = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]
SL_GRID = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]
CANDLE_TYPES = ['D', 'DF', 'GS', 'H', 'IH', 'ST', 'MU', 'MD', 'BU', 'BD', 'U', 'DN']

# Pattern groups from rediscovery analysis
ROBUST_4 = ['U-H-MU', 'IH-MD-MD', 'MU-DF-U', 'BU-ST-GS']  # 50%+ rediscovery
TOP_17_EXTRA = [  # 25% rediscovery (1/4 folds)
    'DN-DF-ST', 'DN-DN-H', 'GS-ST-ST', 'H-BU-BU', 'MD-MD-ST',
    'MD-ST-BD', 'MU-IH-DN', 'U-MD-GS', 'U-MD-MD', 'U-MU-H',
    'D-BD-ST', 'DN-DF-DN', 'ST-MU-ST',
]
TOP_17 = ROBUST_4 + TOP_17_EXTRA

print("=" * 80)
print("Strategy Options Evaluation")
print("  Pattern-Rediscovery WF 기반 전략 옵션 종합 비교")
print("=" * 80)

# ============================================================
# Load and classify data
# ============================================================
data_path = Path(__file__).resolve().parent / '../../data/btc_5m_270days_reclassified.csv'
df = pd.read_csv(data_path)
n_bars = len(df)
bars_per_day = 288
total_days = n_bars // bars_per_day
print(f"\nLoaded: {n_bars} bars ({total_days} days)")

highs = df['high'].values
lows = df['low'].values
opens = df['open'].values
closes = df['close'].values

print("Classifying candles...")
body_abs_arr = np.abs(closes - opens)
avg_body_20_arr = pd.Series(body_abs_arr).rolling(20).mean().values

types = []
for i in range(n_bars):
    row = pd.Series({'open': opens[i], 'high': highs[i], 'low': lows[i], 'close': closes[i]})
    avg_b = avg_body_20_arr[i] if not pd.isna(avg_body_20_arr[i]) else 1.0
    types.append(classify_candle(row, avg_b).value)
print("Classification complete.")

# ============================================================
# Build pattern maps
# ============================================================
current_patterns = {}
for p in VALIDATED_LONG_PATTERNS:
    tp, sl = PATTERN_OPTIMAL_TPSL[p]
    current_patterns[p] = ('LONG', tp, sl)
for p in VALIDATED_SHORT_PATTERNS:
    tp, sl = PATTERN_OPTIMAL_TPSL[p]
    current_patterns[p] = ('SHORT', tp, sl)

current_pat_directions = {p: v[0] for p, v in current_patterns.items()}
print(f"Current patterns: {len(current_patterns)}")

# Pre-compute all pattern instances
all_instances = defaultdict(list)
for i in range(2, n_bars):
    pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
    all_instances[pat].append(i)
print(f"Unique patterns: {len(all_instances)}, Total instances: {sum(len(v) for v in all_instances.values())}")


# ============================================================
# Engine functions (same as pattern_rediscovery_wf.py)
# ============================================================
def evaluate_pattern_grid(instances, direction, start_bar, end_bar):
    valid = [i for i in instances if start_bar <= i < end_bar and i + 1 < n_bars]
    if len(valid) < MIN_TRADES:
        return None

    grid_results = {(tp, sl): {'pnls': [], 'exit_bars': []} for tp in TP_GRID for sl in SL_GRID}

    for idx in valid:
        entry = opens[idx + 1]
        if entry <= 0:
            continue
        end = min(idx + 2 + MAX_BARS, n_bars)
        if idx + 2 >= end:
            continue

        bar_range = slice(idx + 2, end)
        if direction == 'LONG':
            fav = (highs[bar_range] - entry) / entry * 100
            adv = (entry - lows[bar_range]) / entry * 100
        else:
            fav = (entry - lows[bar_range]) / entry * 100
            adv = (highs[bar_range] - entry) / entry * 100

        tp_bars = {}
        for tp in TP_GRID:
            mask = fav >= tp
            tp_bars[tp] = int(np.argmax(mask)) if np.any(mask) else MAX_BARS + 1
        sl_bars = {}
        for sl in SL_GRID:
            mask = adv >= sl
            sl_bars[sl] = int(np.argmax(mask)) if np.any(mask) else MAX_BARS + 1

        for tp in TP_GRID:
            tb = tp_bars[tp]
            for sl in SL_GRID:
                sb = sl_bars[sl]
                if tb < sb:
                    pnl = tp * LEVERAGE - FEE_PCT
                    exit_bar = idx + 2 + tb
                elif sb < tb:
                    pnl = -sl * LEVERAGE - FEE_PCT
                    exit_bar = idx + 2 + sb
                elif tb <= MAX_BARS:
                    pnl = (tp * LEVERAGE - FEE_PCT) if tp <= sl else (-sl * LEVERAGE - FEE_PCT)
                    exit_bar = idx + 2 + tb
                else:
                    pnl = 0
                    exit_bar = idx + 2 + MAX_BARS
                grid_results[(tp, sl)]['pnls'].append(pnl)
                grid_results[(tp, sl)]['exit_bars'].append(exit_bar)

    best_pnl = -999
    best_config = None
    for (tp, sl), data in grid_results.items():
        if len(data['pnls']) < MIN_TRADES:
            continue
        total = sum(data['pnls'])
        if total > best_pnl:
            best_pnl = total
            best_config = (tp, sl, data['pnls'], data['exit_bars'])

    if best_config is None:
        return None

    tp, sl, pnls, exit_bars = best_config
    wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
    rw_wr = sl / (tp + sl) * 100
    excess_wr = wr - rw_wr

    return {
        'tp': tp, 'sl': sl, 'pnl': round(best_pnl, 2),
        'trades': len(pnls), 'wr': round(wr, 1),
        'excess_wr': round(excess_wr, 1),
    }


def mc_test(pnls, n_sims=MC_SIMS):
    pnls = np.array(pnls)
    real = np.sum(pnls)
    if real <= 0:
        return 1.0
    signs = np.random.choice([-1, 1], size=(n_sims, len(pnls)))
    sims = np.sum(pnls * signs, axis=1)
    return float(np.mean(sims >= real))


def collect_trades_1pos(pattern_map, start_bar=0, end_bar=None):
    if end_bar is None:
        end_bar = n_bars
    raw_trades = []
    for i in range(max(2, start_bar), min(end_bar, n_bars)):
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
                pnl_val = (tp if abs(tpp - entry) <= abs(slp - entry) else -sl) * LEVERAGE - FEE_PCT
                raw_trades.append((i + 1, j, pnl_val, pat))
                break
            elif ht:
                raw_trades.append((i + 1, j, tp * LEVERAGE - FEE_PCT, pat))
                break
            elif hs:
                raw_trades.append((i + 1, j, -sl * LEVERAGE - FEE_PCT, pat))
                break
    raw_trades.sort(key=lambda x: x[0])
    filtered = []
    last_exit = -1
    for eb, xb, pnl, pat in raw_trades:
        if eb > last_exit:
            filtered.append((eb, xb, pnl, pat))
            last_exit = xb
    return filtered


def eval_portfolio(trades):
    if not trades:
        return {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0, 'pnl_mdd': 0}
    pnl_list = [t[2] for t in trades]
    cum = 0; peak = 0; mdd = 0; wins = 0; wps = []; lps = []
    for p in pnl_list:
        cum += p
        if cum > peak: peak = cum
        dd = peak - cum
        if dd > mdd: mdd = dd
        if p > 0: wins += 1; wps.append(p)
        else: lps.append(abs(p))
    tw = sum(wps); tl = sum(lps)
    return {
        'pnl': round(cum, 2), 'trades': len(pnl_list),
        'wr': round(wins / len(pnl_list) * 100, 2) if pnl_list else 0,
        'mdd': round(mdd, 2),
        'pf': round(tw / tl, 2) if tl > 0 else 999,
        'pnl_mdd': round(cum / mdd, 2) if mdd > 0 else 999,
    }


def mc_portfolio_test(trades, n_sims=MC_SIMS):
    if not trades:
        return 1.0, 0
    pnls = np.array([t[2] for t in trades])
    real = np.sum(pnls)
    signs = np.random.choice([-1, 1], size=(n_sims, len(pnls)))
    sims = np.sum(pnls * signs, axis=1)
    p = float(np.mean(sims >= real))
    cum_sims = np.cumsum(pnls * signs, axis=1)
    peak_sims = np.maximum.accumulate(cum_sims, axis=1)
    dd_sims = peak_sims - cum_sims
    mdd_p99 = float(np.percentile(np.max(dd_sims, axis=1), 99))
    return round(p, 6), round(mdd_p99, 1)


# ============================================================
# Phase 1: Per-Fold Universe Scan
# ============================================================
print("\n" + "=" * 80)
print("PHASE 1: Per-Fold Universe Scan (3456 patterns)")
print("=" * 80)

np.random.seed(42)
fold_days = total_days // 5
n_folds = 4

universe = []
for t1 in CANDLE_TYPES:
    for t2 in CANDLE_TYPES:
        for t3 in CANDLE_TYPES:
            pat = f"{t1}-{t2}-{t3}"
            for direction in ['LONG', 'SHORT']:
                universe.append((pat, direction))

# Store per-fold scan results
fold_discovered = []  # [{key: {pattern, direction, tp, sl, pnl, trades, ...}}, ...]
fold_rediscovered = []  # [set of our-51 patterns rediscovered]

for fold_i in range(n_folds):
    train_end_bar = (fold_i + 1) * fold_days * bars_per_day
    train_days = (fold_i + 1) * fold_days
    t0 = time.time()

    discovered = {}
    rediscovered = set()

    for pat, direction in universe:
        instances = all_instances.get(pat, [])
        if not instances:
            continue

        result = evaluate_pattern_grid(instances, direction, 0, train_end_bar)
        if result is None:
            continue
        if result['excess_wr'] < MIN_EDGE_PP:
            continue

        # Quick MC: use stored PnLs from grid search
        # Re-evaluate best TP/SL to get individual trade PnLs for MC
        best_tp, best_sl = result['tp'], result['sl']
        valid = [i for i in instances if 0 <= i < train_end_bar and i + 1 < n_bars]
        trade_pnls = []
        for idx in valid:
            entry = opens[idx + 1]
            if entry <= 0:
                continue
            end = min(idx + 2 + MAX_BARS, n_bars)
            if idx + 2 >= end:
                continue
            if direction == 'LONG':
                fav = (highs[idx+2:end] - entry) / entry * 100
                adv = (entry - lows[idx+2:end]) / entry * 100
            else:
                fav = (entry - lows[idx+2:end]) / entry * 100
                adv = (highs[idx+2:end] - entry) / entry * 100

            tp_mask = fav >= best_tp
            sl_mask = adv >= best_sl
            tb = int(np.argmax(tp_mask)) if np.any(tp_mask) else MAX_BARS + 1
            sb = int(np.argmax(sl_mask)) if np.any(sl_mask) else MAX_BARS + 1

            if tb < sb:
                trade_pnls.append(best_tp * LEVERAGE - FEE_PCT)
            elif sb < tb:
                trade_pnls.append(-best_sl * LEVERAGE - FEE_PCT)
            elif tb <= MAX_BARS:
                trade_pnls.append((best_tp * LEVERAGE - FEE_PCT) if best_tp <= best_sl
                                  else (-best_sl * LEVERAGE - FEE_PCT))
            else:
                trade_pnls.append(0)

        if len(trade_pnls) < MIN_TRADES:
            continue

        mc_p = mc_test(trade_pnls)
        if mc_p >= MC_THRESHOLD:
            continue

        key = f"{pat}_{direction}"
        discovered[key] = {
            'pattern': pat, 'direction': direction,
            'tp': best_tp, 'sl': best_sl,
            'pnl': sum(trade_pnls), 'trades': len(trade_pnls),
            'wr': sum(1 for p in trade_pnls if p > 0) / len(trade_pnls) * 100,
            'mc_p': mc_p,
        }

        if pat in current_pat_directions and current_pat_directions[pat] == direction:
            rediscovered.add(pat)

    fold_discovered.append(discovered)
    fold_rediscovered.append(rediscovered)

    # Build current pattern key set
    current_keys = set(f"{p}_{d}" for p, (d, tp, sl) in current_patterns.items())
    n_new = sum(1 for k in discovered if k not in current_keys)

    elapsed = time.time() - t0
    print(f"  Fold {fold_i+1} (Train {train_days}d): "
          f"{len(discovered)} discovered ({len(rediscovered)} from 51, {n_new} new) [{elapsed:.1f}s]")


# ============================================================
# Phase 2: Evaluate All Options
# ============================================================
print("\n" + "=" * 80)
print("PHASE 2: Portfolio WF — All Options")
print("=" * 80)

options_results = {}

for fold_i in range(n_folds):
    train_end_bar = (fold_i + 1) * fold_days * bars_per_day
    oos_start_bar = train_end_bar
    oos_end_bar = min((fold_i + 2) * fold_days * bars_per_day, n_bars)

    # --- Option A: Status Quo (51, current TP/SL) ---
    map_A = {p: (d, tp, sl) for p, (d, tp, sl) in current_patterns.items()}
    trades_A = collect_trades_1pos(map_A, oos_start_bar, oos_end_bar)
    eval_A = eval_portfolio(trades_A)

    # --- Option B: Robust-4 (current TP/SL) ---
    map_B = {p: (d, tp, sl) for p, (d, tp, sl) in current_patterns.items() if p in ROBUST_4}
    trades_B = collect_trades_1pos(map_B, oos_start_bar, oos_end_bar)
    eval_B = eval_portfolio(trades_B)

    # --- Option C: Top-17 (current TP/SL) ---
    map_C = {p: (d, tp, sl) for p, (d, tp, sl) in current_patterns.items() if p in TOP_17}
    trades_C = collect_trades_1pos(map_C, oos_start_bar, oos_end_bar)
    eval_C = eval_portfolio(trades_C)

    # --- Option D: Dynamic Rediscovered from 51 (train TP/SL) ---
    map_D = {}
    for key, data in fold_discovered[fold_i].items():
        pat = data['pattern']
        if pat in current_pat_directions and current_pat_directions[pat] == data['direction']:
            map_D[pat] = (data['direction'], data['tp'], data['sl'])
    trades_D = collect_trades_1pos(map_D, oos_start_bar, oos_end_bar)
    eval_D = eval_portfolio(trades_D)

    # --- Option E: Fresh Top-50 from full universe (train TP/SL) ---
    # Sort all discovered by PnL, take top 50
    sorted_disc = sorted(fold_discovered[fold_i].values(), key=lambda x: -x['pnl'])
    map_E = {}
    for data in sorted_disc[:50]:
        pat = data['pattern']
        if pat not in map_E:  # avoid duplicates (same pattern, different direction shouldn't happen)
            map_E[pat] = (data['direction'], data['tp'], data['sl'])
    trades_E = collect_trades_1pos(map_E, oos_start_bar, oos_end_bar)
    eval_E = eval_portfolio(trades_E)

    # --- Option F: Fresh Top-20 (train TP/SL) ---
    map_F = {}
    for data in sorted_disc[:20]:
        pat = data['pattern']
        if pat not in map_F:
            map_F[pat] = (data['direction'], data['tp'], data['sl'])
    trades_F = collect_trades_1pos(map_F, oos_start_bar, oos_end_bar)
    eval_F = eval_portfolio(trades_F)

    # --- Option G: Rediscovered from 51, current TP/SL ---
    map_G = {}
    for pat in fold_rediscovered[fold_i]:
        map_G[pat] = current_patterns[pat]
    trades_G = collect_trades_1pos(map_G, oos_start_bar, oos_end_bar)
    eval_G = eval_portfolio(trades_G)

    # Store results
    for label, eval_result, n_pat in [
        ('A_StatusQuo_51', eval_A, 51),
        ('B_Robust4', eval_B, len(map_B)),
        ('C_Top17', eval_C, len(map_C)),
        ('D_Dynamic_Train', eval_D, len(map_D)),
        ('E_Fresh50_Train', eval_E, len(map_E)),
        ('F_Fresh20_Train', eval_F, len(map_F)),
        ('G_Dynamic_CurrTPSL', eval_G, len(map_G)),
    ]:
        if label not in options_results:
            options_results[label] = {'folds': [], 'n_patterns': []}
        options_results[label]['folds'].append(eval_result)
        options_results[label]['n_patterns'].append(n_pat)


# ============================================================
# Phase 3: Aggregate and Compare
# ============================================================
print("\n" + "=" * 80)
print("PHASE 3: OOS Comparison (4 folds aggregated)")
print("=" * 80)

print(f"\n  {'Option':<25} {'PnL':>8} {'MDD':>6} {'P/MDD':>7} {'WR':>6} {'Trades':>7} {'Avg Pat':>8} {'LA':>4}")
print(f"  {'-'*25} {'-'*8} {'-'*6} {'-'*7} {'-'*6} {'-'*7} {'-'*8} {'-'*4}")

summary = {}
for label, data in options_results.items():
    total_pnl = sum(f['pnl'] for f in data['folds'])
    total_trades = sum(f['trades'] for f in data['folds'])
    total_wins = sum(f['trades'] * f['wr'] / 100 for f in data['folds'])
    avg_wr = total_wins / total_trades * 100 if total_trades > 0 else 0

    # Concatenate all fold PnLs for MDD calculation
    all_pnls = []
    for fi in range(n_folds):
        fold_start = (fi + 1) * fold_days * bars_per_day
        fold_end = min((fi + 2) * fold_days * bars_per_day, n_bars)
        pat_map = {}
        if label == 'A_StatusQuo_51':
            pat_map = {p: (d, tp, sl) for p, (d, tp, sl) in current_patterns.items()}
        elif label == 'B_Robust4':
            pat_map = {p: (d, tp, sl) for p, (d, tp, sl) in current_patterns.items() if p in ROBUST_4}
        elif label == 'C_Top17':
            pat_map = {p: (d, tp, sl) for p, (d, tp, sl) in current_patterns.items() if p in TOP_17}
        elif label == 'D_Dynamic_Train':
            for key, d in fold_discovered[fi].items():
                pat = d['pattern']
                if pat in current_pat_directions and current_pat_directions[pat] == d['direction']:
                    pat_map[pat] = (d['direction'], d['tp'], d['sl'])
        elif label == 'E_Fresh50_Train':
            sorted_d = sorted(fold_discovered[fi].values(), key=lambda x: -x['pnl'])
            for d in sorted_d[:50]:
                if d['pattern'] not in pat_map:
                    pat_map[d['pattern']] = (d['direction'], d['tp'], d['sl'])
        elif label == 'F_Fresh20_Train':
            sorted_d = sorted(fold_discovered[fi].values(), key=lambda x: -x['pnl'])
            for d in sorted_d[:20]:
                if d['pattern'] not in pat_map:
                    pat_map[d['pattern']] = (d['direction'], d['tp'], d['sl'])
        elif label == 'G_Dynamic_CurrTPSL':
            for pat in fold_rediscovered[fi]:
                pat_map[pat] = current_patterns[pat]

        trades = collect_trades_1pos(pat_map, fold_start, fold_end)
        all_pnls.extend([t[2] for t in trades])

    # Calculate continuous MDD
    cum = 0; peak = 0; mdd = 0
    for p in all_pnls:
        cum += p
        if cum > peak: peak = cum
        dd = peak - cum
        if dd > mdd: mdd = dd

    pnl_mdd = round(cum / mdd, 2) if mdd > 0 else 999
    avg_pat = np.mean(data['n_patterns'])

    # Look-ahead classification
    la_map = {
        'A_StatusQuo_51': 'ALL',
        'B_Robust4': 'TP/SL',
        'C_Top17': 'TP/SL',
        'D_Dynamic_Train': 'NONE',
        'E_Fresh50_Train': 'NONE',
        'F_Fresh20_Train': 'NONE',
        'G_Dynamic_CurrTPSL': 'TP/SL',
    }
    la = la_map[label]

    summary[label] = {
        'pnl': round(total_pnl, 1),
        'mdd': round(mdd, 1),
        'pnl_mdd': pnl_mdd,
        'wr': round(avg_wr, 1),
        'trades': total_trades,
        'avg_patterns': round(avg_pat, 1),
        'look_ahead': la,
    }

    nice_name = label.replace('_', ' ')
    print(f"  {nice_name:<25} {total_pnl:>+8.1f} {mdd:>6.1f} {pnl_mdd:>7.1f} "
          f"{avg_wr:>5.1f}% {total_trades:>7} {avg_pat:>8.1f} {la:>4}")


# ============================================================
# Phase 4: Per-Fold Detail
# ============================================================
print("\n" + "=" * 80)
print("PHASE 4: Per-Fold Detail")
print("=" * 80)

for fold_i in range(n_folds):
    train_days = (fold_i + 1) * fold_days
    print(f"\n  Fold {fold_i+1} (Train {train_days}d):")
    print(f"    {'Option':<25} {'PnL':>8} {'Trades':>7} {'WR':>6} {'MDD':>6} {'Patterns':>9}")
    for label, data in options_results.items():
        f = data['folds'][fold_i]
        np_ = data['n_patterns'][fold_i]
        nice = label.replace('_', ' ')
        print(f"    {nice:<25} {f['pnl']:>+8.1f} {f['trades']:>7} {f['wr']:>5.1f}% {f['mdd']:>6.1f} {np_:>9}")


# ============================================================
# Phase 5: MC Tests for Key Options
# ============================================================
print("\n" + "=" * 80)
print("PHASE 5: Portfolio MC Tests (In-Sample 270d)")
print("=" * 80)

# Test each option on full 270d in-sample
is_mc_results = {}
for label in ['A_StatusQuo_51', 'B_Robust4', 'C_Top17']:
    if label == 'A_StatusQuo_51':
        pm = {p: (d, tp, sl) for p, (d, tp, sl) in current_patterns.items()}
    elif label == 'B_Robust4':
        pm = {p: (d, tp, sl) for p, (d, tp, sl) in current_patterns.items() if p in ROBUST_4}
    elif label == 'C_Top17':
        pm = {p: (d, tp, sl) for p, (d, tp, sl) in current_patterns.items() if p in TOP_17}

    trades = collect_trades_1pos(pm)
    ev = eval_portfolio(trades)
    mc_p, mdd_p99 = mc_portfolio_test(trades)
    is_mc_results[label] = {'is_eval': ev, 'mc_p': mc_p, 'mdd_p99': mdd_p99}

    nice = label.replace('_', ' ')
    print(f"  {nice:<25}: IS PnL {ev['pnl']:>+8.1f}%, WR {ev['wr']:>5.1f}%, "
          f"MDD {ev['mdd']:>5.1f}%, MC p={mc_p:.6f}, MDD_P99={mdd_p99:.1f}%")


# ============================================================
# Phase 6: Practical Assessment
# ============================================================
print("\n" + "=" * 80)
print("PHASE 6: Practical Assessment")
print("=" * 80)

print("""
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ Option              │ Look-Ahead │ Complexity │ Trade Freq │ Edge Genuine? │
  ├─────────────────────┼────────────┼────────────┼────────────┼───────────────┤
  │ A. Status Quo (51)  │ BOTH       │ None       │ High       │ 10% genuine   │
  │ B. Robust-4         │ TP/SL only │ Low        │ Very Low   │ Most genuine  │
  │ C. Top-17           │ TP/SL only │ Low        │ Low-Med    │ Mostly genuine│
  │ D. Dynamic (train)  │ NONE       │ High       │ Very Low   │ Pure genuine  │
  │ E. Fresh-50 (train) │ NONE       │ High       │ Medium     │ Pure genuine  │
  │ F. Fresh-20 (train) │ NONE       │ High       │ Low        │ Pure genuine  │
  │ G. Dynamic (curr)   │ TP/SL only │ Medium     │ Very Low   │ Most genuine  │
  └─────────────────────┴────────────┴────────────┴────────────┴───────────────┘

  Non-Simulation Options:
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ H. Expectation Reset: Keep 51, set EXPECTED_WIN_RATE=67%, daily limit 5%  │
  │ I. Leverage Reduction: 3x → 2x (proportional PnL/MDD reduction)          │
  │ J. Data Collection: Pause optimization, collect 90+ days new data for OOS │
  │ K. Rolling Adaptive Bot: Implement periodic re-scan (every 30-54 days)    │
  │ L. Strategy Pivot: Move to different signal type or timeframe             │
  └─────────────────────────────────────────────────────────────────────────────┘
""")

# ============================================================
# Phase 7: Final Verdict & Ranking
# ============================================================
print("=" * 80)
print("PHASE 7: Ranking by Risk-Adjusted Genuine OOS Performance")
print("=" * 80)

# Rank by PnL/MDD (only OOS, no in-sample)
ranked = sorted(summary.items(), key=lambda x: -x[1]['pnl_mdd'] if x[1]['pnl_mdd'] < 500 else 0)

print(f"\n  {'Rank':<5} {'Option':<25} {'OOS PnL':>8} {'MDD':>6} {'PnL/MDD':>8} {'LA':>5} {'Trades':>7}")
print(f"  {'-'*5} {'-'*25} {'-'*8} {'-'*6} {'-'*8} {'-'*5} {'-'*7}")

for rank, (label, data) in enumerate(ranked, 1):
    nice = label.replace('_', ' ')
    print(f"  {rank:<5} {nice:<25} {data['pnl']:>+8.1f} {data['mdd']:>6.1f} "
          f"{data['pnl_mdd']:>8.1f} {data['look_ahead']:>5} {data['trades']:>7}")

# Identify best no-look-ahead option
best_no_la = None
for label, data in ranked:
    if data['look_ahead'] == 'NONE':
        best_no_la = (label, data)
        break

# Identify best TP/SL-only look-ahead
best_tpsl_la = None
for label, data in ranked:
    if data['look_ahead'] == 'TP/SL':
        best_tpsl_la = (label, data)
        break

print(f"\n  Best no-look-ahead: {best_no_la[0] if best_no_la else 'N/A'}")
print(f"  Best TP/SL-only LA: {best_tpsl_la[0] if best_tpsl_la else 'N/A'}")
print(f"  Status Quo (full LA): A_StatusQuo_51")


# ============================================================
# Save results
# ============================================================
results = {
    'meta': {
        'script': 'strategy_options_evaluation.py',
        'version': 'v1.27.2',
        'timestamp': datetime.now().isoformat(),
        'n_folds': n_folds,
        'fold_days': fold_days,
    },
    'options_summary': summary,
    'per_fold': {
        label: {
            'folds': data['folds'],
            'n_patterns': data['n_patterns'],
        }
        for label, data in options_results.items()
    },
    'in_sample_mc': {
        label: {
            'pnl': d['is_eval']['pnl'],
            'wr': d['is_eval']['wr'],
            'mdd': d['is_eval']['mdd'],
            'mc_p': d['mc_p'],
            'mdd_p99': d['mdd_p99'],
        }
        for label, d in is_mc_results.items()
    },
    'pattern_groups': {
        'robust_4': ROBUST_4,
        'top_17': TOP_17,
    },
    'fold_discovery': [
        {
            'fold': fi + 1,
            'train_days': (fi + 1) * fold_days,
            'total_discovered': len(fold_discovered[fi]),
            'rediscovered_from_51': list(fold_rediscovered[fi]),
        }
        for fi in range(n_folds)
    ],
}

out_path = Path(__file__).resolve().parent / '../../results/strategy_options_evaluation.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved: {out_path}")
print("Done.")
