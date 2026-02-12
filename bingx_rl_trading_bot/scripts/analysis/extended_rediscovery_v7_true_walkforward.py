#!/usr/bin/env python3
"""
Extended Rediscovery v7: True Walk-Forward — Zero Look-Ahead
=============================================================
핵심 질문: 패턴 선택 + TP/SL 최적화를 모두 과거 데이터로만 수행하면
           전략이 여전히 수익성이 있는가?

Look-Ahead 분해 (v4):
  - 패턴 선택: 82.8%   ← 가장 큰 원인
  - TP/SL 최적화: 7.3%
  - Genuine edge: 9.9%

Phase 1: Full Universe Scan
  - 3456 패턴 (12^3 × 2방향) 전수 스캔
  - Universal TP 2.0/SL 3.0
  - Pre-overlap vs In-sample edge 비교
  - 현재 51패턴과 전체 우주 비교

Phase 2: Look-Ahead 정밀 분해
  - A: Current 51 + Current TP/SL       (full LA)
  - B: Current 51 + Universal 2.0/3.0   (pattern selection LA only)
  - C: Universe top + Universal 2.0/3.0  (zero LA: selection on pre-overlap)
  - D: ALL universe + Universal 2.0/3.0  (zero selection, zero TP/SL LA)

Phase 3: True Walk-Forward (Zero Look-Ahead)
  - 4-fold expanding window
  - 각 fold: 전수 3456 스캔 → MC 필터 → 1-pos portfolio OOS
  - Universal TP 2.0/SL 3.0 (TP/SL 최적화 없음)

Phase 4: True WF + Per-Pattern TP/SL Optimization
  - Phase 3와 동일하되 MC pass 패턴에 TP/SL grid search 추가
  - TP/SL 최적화가 OOS에 도움이 되는지 vs 해가 되는지

Phase 5: Selection Threshold Sensitivity
  - Edge threshold, MC threshold, min trades 변화에 따른 OOS 성과
"""

import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

sys.path.insert(0, str(SCRIPT_DIR.parent / 'production'))
from pattern_5m.indicators import classify_candle

FEE_PCT = 0.10
LEVERAGE = 3
MAX_BARS = 500
MC_SIMS = 5000

CANDLE_TYPES = ['D', 'DF', 'GS', 'H', 'IH', 'ST', 'MU', 'MD', 'BU', 'BD', 'U', 'DN']
TP_SL_GRID = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]

UNI_TP = 2.0
UNI_SL = 3.0

PATTERNS_CURRENT = {
    "BD-BD-U": ("LONG", 1.4, 3.0), "BD-MU-BD": ("LONG", 0.7, 0.7),
    "BD-ST-U": ("LONG", 1.4, 2.0), "BU-BU-BD": ("LONG", 2.1, 3.0),
    "D-MU-U": ("LONG", 1.05, 2.0), "DN-BD-BD": ("LONG", 1.4, 1.5),
    "DN-DF-MU": ("LONG", 1.05, 3.0), "DN-DF-ST": ("LONG", 1.4, 3.0),
    "DN-DN-H": ("LONG", 0.7, 2.5), "DN-MD-DN": ("LONG", 1.5, 3.0),
    "GS-ST-ST": ("LONG", 1.05, 2.0), "H-BU-BU": ("LONG", 1.4, 3.0),
    "H-MU-MD": ("LONG", 0.7, 2.0), "IH-MD-MD": ("LONG", 0.7, 2.0),
    "IH-ST-MU": ("LONG", 0.35, 2.5), "MD-BU-MD": ("LONG", 1.4, 2.5),
    "MD-DN-MU": ("LONG", 1.0, 1.0), "MD-H-MD": ("LONG", 0.7, 0.7),
    "MD-MD-ST": ("LONG", 1.05, 2.0), "MD-ST-BD": ("LONG", 1.0, 0.5),
    "MD-ST-MD": ("LONG", 2.0, 2.0), "MU-BD-ST": ("LONG", 1.4, 3.0),
    "MU-DF-U": ("LONG", 1.05, 2.0), "MU-H-MU": ("LONG", 1.5, 0.7),
    "MU-IH-DN": ("LONG", 1.4, 2.0), "MU-U-H": ("LONG", 1.75, 3.0),
    "U-H-MU": ("LONG", 1.5, 2.0), "U-MD-GS": ("LONG", 0.35, 2.5),
    "U-MD-MD": ("LONG", 1.5, 0.7), "U-MU-H": ("LONG", 1.05, 3.0),
    "U-MU-IH": ("LONG", 2.1, 2.5), "U-ST-DF": ("LONG", 1.4, 3.0),
    "BD-BU-DN": ("SHORT", 3.0, 3.0), "BD-D-D": ("SHORT", 1.05, 3.0),
    "BD-U-H": ("SHORT", 2.5, 3.0), "BU-MD-MD": ("SHORT", 3.0, 2.0),
    "BU-ST-GS": ("SHORT", 0.7, 2.5), "D-BD-ST": ("SHORT", 1.75, 3.0),
    "D-DN-DN": ("SHORT", 2.5, 3.0), "DN-BD-BU": ("SHORT", 1.75, 3.0),
    "DN-D-BD": ("SHORT", 1.75, 1.0), "DN-DF-DN": ("SHORT", 2.0, 2.0),
    "H-U-BD": ("SHORT", 2.1, 2.0), "IH-ST-ST": ("SHORT", 1.4, 3.0),
    "MD-MD-MD": ("SHORT", 3.0, 3.0), "ST-BD-BU": ("SHORT", 2.1, 3.0),
    "ST-DN-BU": ("SHORT", 1.4, 3.0), "ST-DN-U": ("SHORT", 2.1, 3.0),
    "ST-MU-ST": ("SHORT", 1.4, 3.0), "U-GS-DN": ("SHORT", 3.0, 3.0),
    "U-ST-DN": ("SHORT", 1.4, 3.0),
}

# ===================================================================
# Core functions
# ===================================================================
def load_and_classify(csv_path):
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    o = df['open'].values; h = df['high'].values
    l = df['low'].values; c = df['close'].values
    n = len(df)
    body_abs = np.abs(c - o)
    avg_b20 = pd.Series(body_abs).rolling(20).mean().values
    types = []
    for i in range(n):
        row = pd.Series({'open': o[i], 'high': h[i], 'low': l[i], 'close': c[i]})
        ab = avg_b20[i] if not pd.isna(avg_b20[i]) else 1.0
        types.append(classify_candle(row, ab).value)
    return o, h, l, c, np.array(types), df['timestamp'].values, n


def build_signal_index(types, n):
    """Build index: pattern_name -> list of signal bars (where 3-candle pattern ends)."""
    idx = {}
    for i in range(2, n):
        pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
        if pat not in idx:
            idx[pat] = []
        idx[pat].append(i)
    return idx


def bt_signals(signal_bars, direction, tp_pct, sl_pct, opens, highs, lows, n_bars):
    """Backtest given signal bars. Returns list of (entry_bar, exit_bar, pnl)."""
    trades = []
    for idx in signal_bars:
        if idx + 1 >= n_bars:
            continue
        entry = opens[idx + 1]
        if entry <= 0:
            continue
        eb = idx + 1
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
                pnl = (tp_pct if abs(tpp - entry) <= abs(slp - entry) else -sl_pct) * LEVERAGE - FEE_PCT
                trades.append((eb, j, pnl))
                break
            elif ht:
                trades.append((eb, j, tp_pct * LEVERAGE - FEE_PCT))
                break
            elif hs:
                trades.append((eb, j, -sl_pct * LEVERAGE - FEE_PCT))
                break
    return trades


def portfolio_1pos(all_trades):
    """1-position-at-a-time: sort by entry, skip overlapping."""
    if not all_trades:
        return []
    all_trades.sort(key=lambda x: x[0])
    filtered = []
    last_exit = -1
    for eb, xb, pnl in all_trades:
        if eb > last_exit:
            filtered.append((eb, xb, pnl))
            last_exit = xb
    return filtered


def stats(trades):
    if not trades:
        return {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0,
                'pnl_mdd': 0, 'avg_win': 0, 'avg_loss': 0, 'safety': 0}
    pnls = [t[2] for t in trades]
    cum = 0; peak = 0; mdd = 0
    wins_list = []; losses_list = []
    for p in pnls:
        cum += p
        if cum > peak: peak = cum
        dd = peak - cum
        if dd > mdd: mdd = dd
        if p > 0: wins_list.append(p)
        else: losses_list.append(p)
    avg_w = float(np.mean(wins_list)) if wins_list else 0
    avg_l = float(np.mean(losses_list)) if losses_list else 0
    be_wr = abs(avg_l) / (avg_w + abs(avg_l)) * 100 if (avg_w + abs(avg_l)) > 0 else 50
    wr = len(wins_list) / len(pnls) * 100 if pnls else 0
    wsum = sum(wins_list); lsum = sum(abs(x) for x in losses_list)
    return {
        'pnl': round(cum, 1), 'trades': len(pnls),
        'wr': round(wr, 1), 'mdd': round(mdd, 1),
        'pf': round(wsum / lsum, 2) if lsum > 0 else 999,
        'pnl_mdd': round(cum / mdd, 1) if mdd > 0 else 999,
        'avg_win': round(avg_w, 3), 'avg_loss': round(avg_l, 3),
        'safety': round(wr - be_wr, 1),
    }


def mc_test(pnls, n_sims=MC_SIMS):
    """Sign randomization MC test."""
    if len(pnls) < 5:
        return 1.0
    pnls_arr = np.array(pnls)
    actual = np.sum(pnls_arr)
    signs = np.random.choice([-1, 1], size=(n_sims, len(pnls_arr)))
    rand_sums = signs @ pnls_arr
    return float(np.mean(rand_sums >= actual))


def mc_portfolio(trades, n_sims=MC_SIMS):
    """Portfolio-level MC: shuffle trade signs."""
    if len(trades) < 5:
        return 1.0
    pnls = np.array([t[2] for t in trades])
    return mc_test(pnls, n_sims)


# ===================================================================
# Phase 1: Full Universe Scan
# ===================================================================
def phase1_universe_scan(types, opens, highs, lows, n, signal_index, pre_end_bar):
    """Scan all 3456 patterns with Universal TP/SL."""
    print("\n=== Phase 1: Full Universe Scan (3456 patterns) ===")
    results = {}
    directions = ['LONG', 'SHORT']

    for pat_name, sig_bars in signal_index.items():
        for direction in directions:
            key = f"{pat_name}_{direction[0]}"

            # Split signals by period
            pre_sigs = [s for s in sig_bars if s < pre_end_bar]
            is_sigs = [s for s in sig_bars if s >= pre_end_bar]

            # Pre-overlap backtest
            pre_trades = bt_signals(pre_sigs, direction, UNI_TP, UNI_SL, opens, highs, lows, n)
            is_trades = bt_signals(is_sigs, direction, UNI_TP, UNI_SL, opens, highs, lows, n)
            full_trades = bt_signals(sig_bars, direction, UNI_TP, UNI_SL, opens, highs, lows, n)

            if len(full_trades) < 5:
                continue

            pre_pnls = [t[2] for t in pre_trades]
            is_pnls = [t[2] for t in is_trades]
            full_pnls = [t[2] for t in full_trades]

            # Random walk baseline
            be_wr = UNI_SL / (UNI_TP + UNI_SL) * 100

            pre_wr = len([p for p in pre_pnls if p > 0]) / len(pre_pnls) * 100 if pre_pnls else 0
            is_wr = len([p for p in is_pnls if p > 0]) / len(is_pnls) * 100 if is_pnls else 0
            full_wr = len([p for p in full_pnls if p > 0]) / len(full_pnls) * 100 if full_pnls else 0

            in_current = pat_name in PATTERNS_CURRENT and PATTERNS_CURRENT[pat_name][0] == direction

            results[key] = {
                'pattern': pat_name,
                'direction': direction,
                'pre_trades': len(pre_trades),
                'pre_wr': round(pre_wr, 1),
                'pre_edge': round(pre_wr - be_wr, 1),
                'pre_pnl': round(sum(pre_pnls), 1),
                'is_trades': len(is_trades),
                'is_wr': round(is_wr, 1),
                'is_edge': round(is_wr - be_wr, 1),
                'is_pnl': round(sum(is_pnls), 1),
                'full_trades': len(full_trades),
                'full_wr': round(full_wr, 1),
                'full_edge': round(full_wr - be_wr, 1),
                'full_pnl': round(sum(full_pnls), 1),
                'in_current_51': in_current,
            }

    # Summary
    all_pats = list(results.values())
    pre_positive = [p for p in all_pats if p['pre_edge'] > 0]
    pre_strong = [p for p in all_pats if p['pre_edge'] >= 5]
    pre_very_strong = [p for p in all_pats if p['pre_edge'] >= 10]
    is_positive = [p for p in all_pats if p['is_edge'] > 0]
    both_positive = [p for p in all_pats if p['pre_edge'] > 0 and p['is_edge'] > 0]

    current_in_pre = [p for p in all_pats if p['in_current_51'] and p['pre_edge'] > 0]

    summary = {
        'total_scanned': len(all_pats),
        'pre_edge_positive': len(pre_positive),
        'pre_edge_5pp': len(pre_strong),
        'pre_edge_10pp': len(pre_very_strong),
        'is_edge_positive': len(is_positive),
        'both_periods_positive': len(both_positive),
        'current_51_with_pre_edge': len(current_in_pre),
        'current_51_total': sum(1 for p in all_pats if p['in_current_51']),
    }

    print(f"  Total patterns with >=5 trades: {len(all_pats)}")
    print(f"  Pre-overlap edge > 0:  {len(pre_positive)}")
    print(f"  Pre-overlap edge >= 5pp: {len(pre_strong)}")
    print(f"  Both periods positive:   {len(both_positive)}")
    print(f"  Current 51 with pre edge: {len(current_in_pre)}/{sum(1 for p in all_pats if p['in_current_51'])}")

    return {'patterns': results, 'summary': summary}


# ===================================================================
# Phase 2: Look-Ahead Decomposition
# ===================================================================
def phase2_decomposition(types, opens, highs, lows, n, signal_index, pre_end_bar):
    """Compare 4 scenarios on pre-overlap period."""
    print("\n=== Phase 2: Look-Ahead Decomposition ===")

    def build_portfolio(pattern_map, start_bar=0, end_bar=None):
        """Build 1-pos portfolio from pattern_map {name: (dir, tp, sl)}."""
        if end_bar is None:
            end_bar = n
        raw = []
        for pat_name, (d, tp, sl) in pattern_map.items():
            if pat_name not in signal_index:
                continue
            sigs = [s for s in signal_index[pat_name] if start_bar <= s < end_bar]
            trades = bt_signals(sigs, d, tp, sl, opens, highs, lows, n)
            raw.extend(trades)
        return portfolio_1pos(raw)

    # A: Current 51 + Current TP/SL (full look-ahead)
    trades_a = build_portfolio(PATTERNS_CURRENT, 0, pre_end_bar)
    stats_a = stats(trades_a)

    # B: Current 51 + Universal 2.0/3.0 (pattern selection LA only)
    uni_map = {p: (d, UNI_TP, UNI_SL) for p, (d, _, _) in PATTERNS_CURRENT.items()}
    trades_b = build_portfolio(uni_map, 0, pre_end_bar)
    stats_b = stats(trades_b)

    # C: Pre-overlap selected + Universal 2.0/3.0
    # Select patterns with pre-overlap edge >= 5pp and >= 10 trades
    pre_selected = {}
    for pat_name in signal_index:
        for d in ['LONG', 'SHORT']:
            sigs = [s for s in signal_index[pat_name] if s < pre_end_bar]
            if len(sigs) < 10:
                continue
            trades = bt_signals(sigs, d, UNI_TP, UNI_SL, opens, highs, lows, n)
            if len(trades) < 5:
                continue
            pnls = [t[2] for t in trades]
            wr = len([p for p in pnls if p > 0]) / len(pnls) * 100
            be = UNI_SL / (UNI_TP + UNI_SL) * 100
            if wr - be >= 5:
                pre_selected[pat_name] = (d, UNI_TP, UNI_SL)
    # C measured on pre-overlap (circular for selection, but for decomposition only)
    trades_c = build_portfolio(pre_selected, 0, pre_end_bar)
    stats_c = stats(trades_c)

    # Also measure C on in-sample (TRUE out-of-sample for C's selection)
    trades_c_oos = build_portfolio(pre_selected, pre_end_bar, n)
    stats_c_oos = stats(trades_c_oos)

    # D: ALL patterns + Universal 2.0/3.0 (zero selection)
    # Use both directions for every pattern
    all_map = {}
    for pat_name in signal_index:
        # Pick direction with higher pre-overlap edge
        best_d = None; best_edge = -999
        for d in ['LONG', 'SHORT']:
            sigs = [s for s in signal_index[pat_name] if s < pre_end_bar]
            trades = bt_signals(sigs, d, UNI_TP, UNI_SL, opens, highs, lows, n)
            if len(trades) < 3:
                continue
            pnls = [t[2] for t in trades]
            wr = len([p for p in pnls if p > 0]) / len(pnls) * 100
            be = UNI_SL / (UNI_TP + UNI_SL) * 100
            edge = wr - be
            if edge > best_edge:
                best_edge = edge
                best_d = d
        if best_d and best_edge > 0:
            all_map[pat_name] = (best_d, UNI_TP, UNI_SL)
    trades_d = build_portfolio(all_map, 0, pre_end_bar)
    stats_d = stats(trades_d)

    result = {
        'A_current_full_LA': stats_a,
        'B_current_universal': stats_b,
        'C_pre_selected_universal': {
            'on_pre_overlap': stats_c,
            'on_insample_OOS': stats_c_oos,
            'n_patterns': len(pre_selected),
            'patterns': list(pre_selected.keys()),
        },
        'D_all_positive_universal': {
            'stats': stats_d,
            'n_patterns': len(all_map),
        },
        'decomposition': {
            'A_pnl_pre': stats_a['pnl'],
            'B_pnl_pre': stats_b['pnl'],
            'tpsl_LA_contribution': round(stats_a['pnl'] - stats_b['pnl'], 1),
            'pattern_selection_LA': round(stats_b['pnl'] - stats_d['pnl'], 1),
            'note': 'A-B = TP/SL optimization LA, B-D = pattern selection LA (approximate)',
        }
    }

    print(f"  A (51 + current TP/SL):   PnL {stats_a['pnl']:+.1f}%, safety {stats_a['safety']:+.1f}pp")
    print(f"  B (51 + universal):        PnL {stats_b['pnl']:+.1f}%, safety {stats_b['safety']:+.1f}pp")
    print(f"  C (pre-selected + uni):    PnL(pre) {stats_c['pnl']:+.1f}%, OOS {stats_c_oos['pnl']:+.1f}% ({len(pre_selected)} pats)")
    print(f"  D (all positive + uni):    PnL {stats_d['pnl']:+.1f}% ({len(all_map)} pats)")

    return result


# ===================================================================
# Phase 3: True Walk-Forward (Zero Look-Ahead) — Universal TP/SL
# ===================================================================
def phase3_true_wf_universal(types, opens, highs, lows, n, signal_index):
    """True walk-forward: select patterns from training only, test on OOS."""
    print("\n=== Phase 3: True Walk-Forward (Universal TP/SL, Zero LA) ===")

    bars_per_day = 288  # 5min bars per day
    total_days = n // bars_per_day

    # 4-fold expanding window: train grows, test fixed ~120d
    folds = []
    test_days = 120
    for i in range(4):
        train_end_day = 180 + i * test_days  # 180, 300, 420, 540
        test_end_day = train_end_day + test_days  # 300, 420, 540, 660
        if test_end_day * bars_per_day > n:
            test_end_day = total_days
        folds.append({
            'train_end': train_end_day * bars_per_day,
            'test_start': train_end_day * bars_per_day,
            'test_end': min(test_end_day * bars_per_day, n),
            'train_days': train_end_day,
            'test_days': test_end_day - train_end_day,
        })

    print(f"  {len(folds)} folds, test={test_days}d each")

    fold_results = []
    all_oos_trades = []

    for fi, fold in enumerate(folds):
        t0 = time.time()
        train_end = fold['train_end']
        test_start = fold['test_start']
        test_end = fold['test_end']

        # Scan all patterns on TRAINING only
        selected = {}
        pattern_stats_train = []

        for pat_name in signal_index:
            for d in ['LONG', 'SHORT']:
                train_sigs = [s for s in signal_index[pat_name] if s < train_end]
                if len(train_sigs) < 8:  # minimum trades in training
                    continue
                trades = bt_signals(train_sigs, d, UNI_TP, UNI_SL, opens, highs, lows, n)
                if len(trades) < 5:
                    continue
                pnls = [t[2] for t in trades]
                wr = len([p for p in pnls if p > 0]) / len(pnls) * 100
                be = UNI_SL / (UNI_TP + UNI_SL) * 100
                edge = wr - be

                if edge < 5:  # minimum edge threshold
                    continue

                # MC test on training data
                p = mc_test(pnls, MC_SIMS)
                if p >= 0.01:
                    continue

                # Selected for this fold
                selected[pat_name] = (d, UNI_TP, UNI_SL)
                pattern_stats_train.append({
                    'pattern': pat_name, 'direction': d,
                    'train_trades': len(trades), 'train_wr': round(wr, 1),
                    'train_edge': round(edge, 1), 'mc_p': round(p, 4),
                })

        # Test selected on OOS
        raw_oos = []
        for pat_name, (d, tp, sl) in selected.items():
            test_sigs = [s for s in signal_index[pat_name]
                         if test_start <= s < test_end]
            trades = bt_signals(test_sigs, d, tp, sl, opens, highs, lows, n)
            raw_oos.extend(trades)

        oos_trades = portfolio_1pos(raw_oos)
        oos_stats = stats(oos_trades)
        all_oos_trades.extend(oos_trades)

        elapsed = time.time() - t0
        fold_results.append({
            'fold': fi + 1,
            'train_days': fold['train_days'],
            'test_days': fold['test_days'],
            'n_selected': len(selected),
            'selected_patterns': list(selected.keys()),
            'oos_stats': oos_stats,
            'elapsed_s': round(elapsed, 1),
        })

        print(f"  Fold {fi+1}: train={fold['train_days']}d, "
              f"selected={len(selected)} patterns, "
              f"OOS PnL={oos_stats['pnl']:+.1f}%, WR={oos_stats['wr']}%, "
              f"trades={oos_stats['trades']} ({elapsed:.1f}s)")

    # Aggregate all OOS
    agg_oos = stats(all_oos_trades) if all_oos_trades else stats([])
    total_oos_pnl = sum(f['oos_stats']['pnl'] for f in fold_results)
    folds_positive = sum(1 for f in fold_results if f['oos_stats']['pnl'] > 0)

    # MC on aggregate OOS
    mc_agg = mc_portfolio(all_oos_trades) if all_oos_trades else 1.0

    result = {
        'folds': fold_results,
        'aggregate_oos': agg_oos,
        'total_oos_pnl': round(total_oos_pnl, 1),
        'folds_positive': folds_positive,
        'mc_aggregate': round(mc_agg, 4),
    }

    print(f"\n  Aggregate OOS: PnL={total_oos_pnl:+.1f}%, "
          f"folds positive={folds_positive}/{len(folds)}, MC={mc_agg:.4f}")

    return result


# ===================================================================
# Phase 4: True Walk-Forward + TP/SL Optimization
# ===================================================================
def phase4_true_wf_with_tpsl(types, opens, highs, lows, n, signal_index):
    """Same as Phase 3 but with per-pattern TP/SL grid search on training."""
    print("\n=== Phase 4: True WF + Per-Pattern TP/SL Optimization ===")

    bars_per_day = 288
    folds = []
    test_days = 120
    for i in range(4):
        train_end_day = 180 + i * test_days
        test_end_day = train_end_day + test_days
        if test_end_day * bars_per_day > n:
            test_end_day = n // bars_per_day
        folds.append({
            'train_end': train_end_day * bars_per_day,
            'test_start': train_end_day * bars_per_day,
            'test_end': min(test_end_day * bars_per_day, n),
            'train_days': train_end_day,
            'test_days': test_end_day - train_end_day,
        })

    fold_results = []
    all_oos_trades = []

    for fi, fold in enumerate(folds):
        t0 = time.time()
        train_end = fold['train_end']
        test_start = fold['test_start']
        test_end = fold['test_end']

        # Step 1: Quick scan with Universal TP/SL to find candidates
        candidates = []
        for pat_name in signal_index:
            for d in ['LONG', 'SHORT']:
                train_sigs = [s for s in signal_index[pat_name] if s < train_end]
                if len(train_sigs) < 8:
                    continue
                trades = bt_signals(train_sigs, d, UNI_TP, UNI_SL, opens, highs, lows, n)
                if len(trades) < 5:
                    continue
                pnls = [t[2] for t in trades]
                wr = len([p for p in pnls if p > 0]) / len(pnls) * 100
                be = UNI_SL / (UNI_TP + UNI_SL) * 100
                if wr - be >= 3:  # looser pre-filter
                    candidates.append((pat_name, d, train_sigs))

        # Step 2: Grid search TP/SL for candidates
        selected = {}
        for pat_name, d, train_sigs in candidates:
            best_pnl = -999
            best_tp = UNI_TP
            best_sl = UNI_SL
            best_mc = 1.0

            for tp, sl in product(TP_SL_GRID, TP_SL_GRID):
                trades = bt_signals(train_sigs, d, tp, sl, opens, highs, lows, n)
                if len(trades) < 5:
                    continue
                pnls = [t[2] for t in trades]
                wr = len([p for p in pnls if p > 0]) / len(pnls) * 100
                be = sl / (tp + sl) * 100
                edge = wr - be
                total_pnl = sum(pnls)

                if edge < 5 or total_pnl <= 0:
                    continue
                if total_pnl > best_pnl:
                    best_pnl = total_pnl
                    best_tp = tp
                    best_sl = sl

            if best_pnl <= 0:
                continue

            # MC test on best TP/SL
            trades = bt_signals(train_sigs, d, best_tp, best_sl, opens, highs, lows, n)
            pnls = [t[2] for t in trades]
            p = mc_test(pnls, MC_SIMS)
            if p < 0.01:
                selected[pat_name] = (d, best_tp, best_sl)

        # Step 3: Test on OOS
        raw_oos = []
        for pat_name, (d, tp, sl) in selected.items():
            test_sigs = [s for s in signal_index[pat_name]
                         if test_start <= s < test_end]
            trades = bt_signals(test_sigs, d, tp, sl, opens, highs, lows, n)
            raw_oos.extend(trades)

        oos_trades = portfolio_1pos(raw_oos)
        oos_stats = stats(oos_trades)
        all_oos_trades.extend(oos_trades)

        elapsed = time.time() - t0
        fold_results.append({
            'fold': fi + 1,
            'train_days': fold['train_days'],
            'test_days': fold['test_days'],
            'n_selected': len(selected),
            'oos_stats': oos_stats,
            'elapsed_s': round(elapsed, 1),
        })

        print(f"  Fold {fi+1}: train={fold['train_days']}d, "
              f"selected={len(selected)} pats (w/ TP/SL opt), "
              f"OOS PnL={oos_stats['pnl']:+.1f}%, WR={oos_stats['wr']}%, "
              f"trades={oos_stats['trades']} ({elapsed:.1f}s)")

    # Aggregate
    total_oos_pnl = sum(f['oos_stats']['pnl'] for f in fold_results)
    folds_positive = sum(1 for f in fold_results if f['oos_stats']['pnl'] > 0)
    mc_agg = mc_portfolio(all_oos_trades) if all_oos_trades else 1.0

    result = {
        'folds': fold_results,
        'total_oos_pnl': round(total_oos_pnl, 1),
        'folds_positive': folds_positive,
        'mc_aggregate': round(mc_agg, 4),
    }

    print(f"\n  Aggregate OOS: PnL={total_oos_pnl:+.1f}%, "
          f"folds positive={folds_positive}/{len(folds)}, MC={mc_agg:.4f}")

    return result


# ===================================================================
# Phase 5: Selection Threshold Sensitivity
# ===================================================================
def phase5_sensitivity(types, opens, highs, lows, n, signal_index):
    """Vary selection thresholds and measure OOS impact."""
    print("\n=== Phase 5: Selection Threshold Sensitivity ===")

    bars_per_day = 288
    # Use single representative fold: train 0-360d, test 360-540d
    train_end = 360 * bars_per_day
    test_start = train_end
    test_end = min(540 * bars_per_day, n)

    edge_thresholds = [0, 3, 5, 7, 10, 15]
    mc_thresholds = [1.0, 0.10, 0.05, 0.01, 0.005]
    min_trades_list = [3, 5, 10, 15]

    results = []

    for edge_th in edge_thresholds:
        for mc_th in mc_thresholds:
            for min_tr in min_trades_list:
                selected = {}
                for pat_name in signal_index:
                    for d in ['LONG', 'SHORT']:
                        train_sigs = [s for s in signal_index[pat_name] if s < train_end]
                        trades = bt_signals(train_sigs, d, UNI_TP, UNI_SL, opens, highs, lows, n)
                        if len(trades) < min_tr:
                            continue
                        pnls = [t[2] for t in trades]
                        wr = len([p for p in pnls if p > 0]) / len(pnls) * 100
                        be = UNI_SL / (UNI_TP + UNI_SL) * 100
                        if wr - be < edge_th:
                            continue
                        if mc_th < 1.0:
                            p = mc_test(pnls, MC_SIMS)
                            if p >= mc_th:
                                continue
                        selected[pat_name] = (d, UNI_TP, UNI_SL)

                # OOS test
                raw_oos = []
                for pat_name, (d, tp, sl) in selected.items():
                    test_sigs = [s for s in signal_index[pat_name]
                                 if test_start <= s < test_end]
                    trades = bt_signals(test_sigs, d, tp, sl, opens, highs, lows, n)
                    raw_oos.extend(trades)
                oos_trades = portfolio_1pos(raw_oos)
                oos_st = stats(oos_trades)

                results.append({
                    'edge_th': edge_th, 'mc_th': mc_th, 'min_trades': min_tr,
                    'n_selected': len(selected),
                    'oos_pnl': oos_st['pnl'], 'oos_wr': oos_st['wr'],
                    'oos_trades': oos_st['trades'], 'oos_safety': oos_st['safety'],
                    'oos_pnl_mdd': oos_st['pnl_mdd'],
                })

    # Find optimal thresholds
    viable = [r for r in results if r['oos_pnl'] > 0 and r['oos_trades'] >= 10]
    if viable:
        best_pnl = max(viable, key=lambda x: x['oos_pnl'])
        best_safety = max(viable, key=lambda x: x['oos_safety'])
        best_pnl_mdd = max(viable, key=lambda x: x['oos_pnl_mdd'])
    else:
        best_pnl = best_safety = best_pnl_mdd = None

    print(f"  Total combos tested: {len(results)}")
    print(f"  Viable (OOS PnL>0, trades>=10): {len(viable)}")
    if best_pnl:
        print(f"  Best PnL: edge>={best_pnl['edge_th']}pp, MC<{best_pnl['mc_th']}, "
              f"min_tr={best_pnl['min_trades']} → {best_pnl['n_selected']} pats, "
              f"OOS PnL={best_pnl['oos_pnl']:+.1f}%")
    if best_safety:
        print(f"  Best Safety: edge>={best_safety['edge_th']}pp, MC<{best_safety['mc_th']}, "
              f"min_tr={best_safety['min_trades']} → {best_safety['n_selected']} pats, "
              f"OOS safety={best_safety['oos_safety']:+.1f}pp")

    return {
        'grid': results,
        'n_viable': len(viable),
        'best_by_pnl': best_pnl,
        'best_by_safety': best_safety,
        'best_by_pnl_mdd': best_pnl_mdd,
    }


# ===================================================================
# Main
# ===================================================================
def main():
    t_start = time.time()
    np.random.seed(42)

    csv_path = DATA_DIR / 'btc_5m_720days_binance.csv'
    print(f"Loading data from {csv_path}")
    opens, highs, lows, closes, types, timestamps, n = load_and_classify(str(csv_path))
    print(f"  Loaded {n} bars, {n // 288} days")

    # Pre-overlap boundary: 437 days before the 270d in-sample
    # In-sample starts at bar (720-270)*288 = 450*288 = 129600
    is_start_bar = (720 - 270) * 288
    pre_end_bar = is_start_bar
    print(f"  Pre-overlap: bars 0-{pre_end_bar} ({pre_end_bar//288}d)")
    print(f"  In-sample: bars {is_start_bar}-{n} ({(n - is_start_bar)//288}d)")

    # Build signal index
    print("Building signal index for all patterns...")
    signal_index = build_signal_index(types, n)
    print(f"  {len(signal_index)} unique 3-candle patterns found")

    # Phase 1
    p1 = phase1_universe_scan(types, opens, highs, lows, n, signal_index, pre_end_bar)

    # Phase 2
    p2 = phase2_decomposition(types, opens, highs, lows, n, signal_index, pre_end_bar)

    # Phase 3
    p3 = phase3_true_wf_universal(types, opens, highs, lows, n, signal_index)

    # Phase 4
    p4 = phase4_true_wf_with_tpsl(types, opens, highs, lows, n, signal_index)

    # Phase 5
    p5 = phase5_sensitivity(types, opens, highs, lows, n, signal_index)

    # Compile results
    results = {
        'phase1_universe': {
            'summary': p1['summary'],
            # Store top patterns only to keep JSON manageable
            'top_pre_edge': sorted(
                [v for v in p1['patterns'].values() if v['pre_edge'] >= 5],
                key=lambda x: -x['pre_edge']
            )[:50],
            'current_51_pre_stats': [
                v for v in p1['patterns'].values() if v['in_current_51']
            ],
        },
        'phase2_decomposition': p2,
        'phase3_true_wf_universal': p3,
        'phase4_true_wf_with_tpsl': p4,
        'phase5_sensitivity': {
            'n_viable': p5['n_viable'],
            'best_by_pnl': p5['best_by_pnl'],
            'best_by_safety': p5['best_by_safety'],
            'best_by_pnl_mdd': p5['best_by_pnl_mdd'],
            # Store only viable results
            'viable_results': [r for r in p5['grid']
                               if r['oos_pnl'] > 0 and r['oos_trades'] >= 10],
        },
        'comparison': {
            'note': 'Phase 3 vs Phase 4: Does TP/SL optimization help or hurt OOS?',
            'phase3_total_oos': p3['total_oos_pnl'],
            'phase4_total_oos': p4['total_oos_pnl'],
            'tpsl_opt_delta': round(p4['total_oos_pnl'] - p3['total_oos_pnl'], 1),
        },
        'metadata': {
            'script': 'extended_rediscovery_v7_true_walkforward.py',
            'data': str(csv_path),
            'n_bars': n,
            'uni_tp': UNI_TP,
            'uni_sl': UNI_SL,
            'mc_sims': MC_SIMS,
            'elapsed_seconds': round(time.time() - t_start, 1),
        }
    }

    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY: True Walk-Forward Results")
    print("=" * 70)
    print(f"\nPhase 3 (Universal TP/SL, zero LA):")
    print(f"  Total OOS PnL: {p3['total_oos_pnl']:+.1f}%")
    print(f"  Folds positive: {p3['folds_positive']}/4")
    print(f"  MC aggregate: {p3['mc_aggregate']:.4f}")

    print(f"\nPhase 4 (WF + TP/SL Optimization):")
    print(f"  Total OOS PnL: {p4['total_oos_pnl']:+.1f}%")
    print(f"  Folds positive: {p4['folds_positive']}/4")
    print(f"  MC aggregate: {p4['mc_aggregate']:.4f}")

    print(f"\nTP/SL optimization OOS delta: {p4['total_oos_pnl'] - p3['total_oos_pnl']:+.1f}%")

    print(f"\nComparison with current system:")
    print(f"  Current 51 + current TP/SL (pre-overlap): {p2['A_current_full_LA']['pnl']:+.1f}%")
    print(f"  Current 51 + universal (pre-overlap):      {p2['B_current_universal']['pnl']:+.1f}%")
    print(f"  True WF zero-LA (total OOS):               {p3['total_oos_pnl']:+.1f}%")

    out = RESULTS_DIR / 'extended_rediscovery_v7_true_walkforward.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out}")
    print(f"Total elapsed: {time.time() - t_start:.1f}s")


if __name__ == '__main__':
    main()
