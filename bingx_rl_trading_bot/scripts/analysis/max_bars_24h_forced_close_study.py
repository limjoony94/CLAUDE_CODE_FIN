#!/usr/bin/env python3
"""
MAX_BARS 24h Forced Close Study — timeout을 패배(시장가 청산) 처리하는 재발굴

기존 scanner: timeout → DROP (통계 제외)
본 연구:     timeout → 24h 시점의 close price로 강제청산, PnL 반영

목적:
  24시간 강제청산 조건에서 살아남는 패턴 발굴.
  DROP 방식 대비 더 보수적이고 실전에 가까운 결과.

Method:
  1. bt_signals_forced_close(): timeout 시 해당 bar의 close price로 PnL 계산
  2. Scanner의 grid_search 로직을 forced-close 백테스트로 대체
  3. MC test + BH FDR + dedup 적용
  4. 후처리 필터 (Edge + WR + SL)
  5. Portfolio 1-pos + WF 검증

Protocol:
  - Production classify_candle() 사용 (indicators.py)
  - LEVERAGE = 3, FEE = 0.10%
  - Sequential mode for reproducibility
"""

import os
import sys
import json
import time
import numpy as np
from collections import defaultdict

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)

import scripts.scanner.pattern_scanner as scanner

# Constants (match scanner)
LEVERAGE = scanner.LEVERAGE
FEE_PCT = scanner.FEE_PCT
MAX_BARS_24H = 288
MC_SIMS = scanner.MC_SIMS
MC_SEEDS = scanner.MC_SEEDS
DEFAULT_MIN_TRADES = scanner.DEFAULT_MIN_TRADES
DEFAULT_EDGE_THRESHOLD = scanner.DEFAULT_EDGE_THRESHOLD
DEFAULT_MC_THRESHOLD = scanner.DEFAULT_MC_THRESHOLD
MAX_BASELINE_WR = scanner.MAX_BASELINE_WR
TP_GRID = scanner.TP_GRID
SL_GRID = scanner.SL_GRID


# ============================================================
# Core: Forced Close Backtest
# ============================================================

def bt_signals_forced_close(signal_bars, direction, tp_pct, sl_pct,
                             opens, highs, lows, closes, n_bars,
                             max_bars=MAX_BARS_24H):
    """
    Backtest with forced close at timeout.

    - TP hit within max_bars → WIN (PnL = +tp_pct * LEVERAGE - FEE)
    - SL hit within max_bars → LOSS (PnL = -sl_pct * LEVERAGE - FEE)
    - Neither within max_bars → FORCED CLOSE at close[last_bar]
      PnL = (close - entry) / entry * 100 * LEVERAGE * direction_sign - FEE

    Returns: list of (entry_bar, exit_bar, pnl_pct, exit_type)
      exit_type: 'TP', 'SL', 'TIMEOUT'
    """
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

        hit = False
        last_bar = min(idx + 1 + max_bars, n_bars - 1)

        for j in range(idx + 2, min(idx + 2 + max_bars, n_bars)):
            if direction == 'LONG':
                ht = highs[j] >= tpp
                hs = lows[j] <= slp
            else:
                ht = lows[j] <= tpp
                hs = highs[j] >= slp

            if ht and hs:
                bo = opens[j]
                dist_tp = abs(tpp - bo)
                dist_sl = abs(slp - bo)
                pnl = (tp_pct if dist_tp <= dist_sl else -sl_pct) * LEVERAGE - FEE_PCT
                exit_type = 'TP' if dist_tp <= dist_sl else 'SL'
                trades.append((eb, j, pnl, exit_type))
                hit = True
                break
            elif ht:
                trades.append((eb, j, tp_pct * LEVERAGE - FEE_PCT, 'TP'))
                hit = True
                break
            elif hs:
                trades.append((eb, j, -sl_pct * LEVERAGE - FEE_PCT, 'SL'))
                hit = True
                break

        if not hit:
            # Forced close at timeout bar's close
            close_price = closes[last_bar]
            if direction == 'LONG':
                raw_pnl = (close_price - entry) / entry * 100
            else:
                raw_pnl = (entry - close_price) / entry * 100
            pnl = raw_pnl * LEVERAGE - FEE_PCT
            trades.append((eb, last_bar, pnl, 'TIMEOUT'))

    return trades


def grid_search_best_fc(signal_bars, direction, opens, highs, lows, closes, n_bars):
    """Grid search for best TP/SL using forced-close backtest."""
    best_score = -1e9
    best_result = None

    for tp in TP_GRID:
        for sl in SL_GRID:
            trades = bt_signals_forced_close(
                signal_bars, direction, tp, sl,
                opens, highs, lows, closes, n_bars)

            if len(trades) < DEFAULT_MIN_TRADES:
                continue

            pnls = [t[2] for t in trades]
            wins = sum(1 for p in pnls if p > 0)
            wr = wins / len(pnls) * 100

            # Baseline WR check
            baseline_wr = sl / (tp + sl) * 100
            if baseline_wr > MAX_BASELINE_WR * 100:
                continue

            edge = wr - baseline_wr
            if edge < DEFAULT_EDGE_THRESHOLD:
                continue

            cum_pnl = sum(pnls)
            peak = 0
            mdd = 0
            c = 0
            for p in pnls:
                c += p
                if c > peak:
                    peak = c
                dd = peak - c
                if dd > mdd:
                    mdd = dd

            # Score: PnL / MDD (same as scanner)
            score = cum_pnl / mdd if mdd > 0 else cum_pnl
            if score > best_score:
                best_score = score
                # Timeout stats
                n_timeout = sum(1 for t in trades if t[3] == 'TIMEOUT')
                timeout_pct = round(n_timeout / len(trades) * 100, 1)
                timeout_pnls = [t[2] for t in trades if t[3] == 'TIMEOUT']
                avg_timeout_pnl = round(float(np.mean(timeout_pnls)), 2) if timeout_pnls else 0

                best_result = {
                    'tp': tp, 'sl': sl,
                    'trades': len(trades), 'wr': round(wr, 1),
                    'edge': round(edge, 1),
                    'pnl': round(cum_pnl, 1), 'mdd': round(mdd, 1),
                    'score': round(score, 2),
                    'baseline_wr': round(baseline_wr, 1),
                    'n_timeout': n_timeout,
                    'timeout_pct': timeout_pct,
                    'avg_timeout_pnl': avg_timeout_pnl,
                }

    return best_result


# ============================================================
# Discovery Pipeline
# ============================================================

def discover_patterns_fc(df, signal_index):
    """Full pattern discovery with forced-close backtest."""
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    n_bars = len(df)

    results = {}
    n_tested = 0

    print(f"  Scanning {len(signal_index)} patterns x 2 directions x {len(TP_GRID)}x{len(SL_GRID)} grid...")

    from tqdm import tqdm
    combos = []
    for pat, sigs in signal_index.items():
        for direction in ['LONG', 'SHORT']:
            combos.append((pat, direction, sigs))

    for pat, direction, sigs in tqdm(combos, desc="FC Grid Search"):
        n_tested += 1
        best = grid_search_best_fc(sigs, direction, opens, highs, lows, closes, n_bars)
        if best is None:
            continue

        # MC test
        trades = bt_signals_forced_close(
            sigs, direction, best['tp'], best['sl'],
            opens, highs, lows, closes, n_bars)
        pnls = [t[2] for t in trades]
        mc_p = scanner.mc_test(pnls)
        if mc_p >= DEFAULT_MC_THRESHOLD:
            continue

        key = f"{pat}_{direction}"
        results[key] = {
            'key': key,
            'pattern': pat,
            'direction': direction,
            **best,
            'mc_p': round(mc_p, 4),
        }

    print(f"  Patterns passing filters: {len(results)} (tested: {n_tested})")

    # BH FDR correction
    if results:
        selected_for_bh = {k: v for k, v in results.items()}
        corrected, meta = scanner.apply_multiple_testing_correction(
            selected_for_bh, n_tested, method='bh', fdr_q=0.05)
        print(f"  BH FDR: {len(results)} -> {len(corrected)} patterns")
        results = corrected

    # Dedup (same as scanner: keep direction with higher edge)
    pat_best = {}
    for key, v in results.items():
        pat = v['pattern']
        if pat not in pat_best or v['edge'] > pat_best[pat]['edge']:
            pat_best[pat] = v

    n_before = len(results)
    results = {v['key']: v for v in pat_best.values()}
    n_removed = n_before - len(results)
    if n_removed > 0:
        print(f"  Dedup: removed {n_removed} dual-direction duplicates")

    # Portfolio
    all_trades = []
    for key, v in results.items():
        sigs = signal_index[v['pattern']]
        trades = bt_signals_forced_close(
            sigs, v['direction'], v['tp'], v['sl'],
            opens, highs, lows, closes, n_bars)
        all_trades.extend([(t[0], t[1], t[2]) for t in trades])

    port = scanner.portfolio_1pos(all_trades)
    stats = scanner.calc_stats(port)

    long_p = sorted([v['pattern'] for v in results.values() if v['direction'] == 'LONG'])
    short_p = sorted([v['pattern'] for v in results.values() if v['direction'] == 'SHORT'])
    print(f"  Selected: {len(long_p)}L + {len(short_p)}S = {len(results)} patterns")
    print(f"  Portfolio: {stats['trades']} trades, WR {stats['wr']}%, PnL {stats['pnl']}%")

    return {
        'pattern_details': results,
        'long_patterns': long_p,
        'short_patterns': short_p,
        'portfolio_stats': stats,
        'n_tested': n_tested,
    }


# ============================================================
# Walk-Forward with Forced Close
# ============================================================

def wf_forced_close(df, signal_index, n_folds=3):
    """Expanding window WF with forced-close backtest."""
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    n = len(df)
    fold_size = n // (n_folds + 1)

    folds = []
    all_oos_trades = []
    stable_counts = defaultdict(int)

    for fold in range(1, n_folds + 1):
        is_end = fold_size * (fold + 1)
        oos_start = is_end
        oos_end = min(is_end + fold_size, n)

        if oos_start >= n:
            break

        print(f"  WF Fold {fold}/{n_folds}: IS=[0,{is_end}), OOS=[{oos_start},{oos_end})")

        # IS discovery
        is_types = df['candle_type'].tolist()[:is_end]
        is_signal_index = scanner.build_signal_index(is_types, is_end)

        is_results = {}
        for pat, sigs in is_signal_index.items():
            for direction in ['LONG', 'SHORT']:
                best = grid_search_best_fc(
                    sigs, direction,
                    opens[:is_end], highs[:is_end], lows[:is_end], closes[:is_end],
                    is_end)
                if best is None:
                    continue

                trades = bt_signals_forced_close(
                    sigs, direction, best['tp'], best['sl'],
                    opens[:is_end], highs[:is_end], lows[:is_end], closes[:is_end],
                    is_end)
                pnls = [t[2] for t in trades]
                mc_p = scanner.mc_test(pnls)
                if mc_p >= DEFAULT_MC_THRESHOLD:
                    continue

                key = f"{pat}_{direction}"
                is_results[key] = {
                    'key': key, 'pattern': pat, 'direction': direction,
                    **best, 'mc_p': round(mc_p, 4),
                }

        # Dedup
        pat_best = {}
        for key, v in is_results.items():
            pat = v['pattern']
            if pat not in pat_best or v['edge'] > pat_best[pat]['edge']:
                pat_best[pat] = v
        is_results = {v['key']: v for v in pat_best.values()}

        n_long = sum(1 for v in is_results.values() if v['direction'] == 'LONG')
        n_short = len(is_results) - n_long

        # OOS test
        oos_signal_index = scanner.build_signal_index(
            df['candle_type'].tolist()[oos_start:oos_end], oos_end - oos_start)

        oos_trades = []
        for key, v in is_results.items():
            pat = v['pattern']
            if pat not in oos_signal_index:
                continue
            sigs = oos_signal_index[pat]
            trades = bt_signals_forced_close(
                sigs, v['direction'], v['tp'], v['sl'],
                opens[oos_start:oos_end], highs[oos_start:oos_end],
                lows[oos_start:oos_end], closes[oos_start:oos_end],
                oos_end - oos_start)
            oos_trades.extend([(t[0], t[1], t[2]) for t in trades])

            stable_counts[key] += 1

        oos_port = scanner.portfolio_1pos(oos_trades)
        oos_stats = scanner.calc_stats(oos_port)
        all_oos_trades.extend(oos_port)

        folds.append({
            'fold': fold,
            'is_bars': is_end,
            'oos_bars': oos_end - oos_start,
            'is_patterns': len(is_results),
            'is_long': n_long,
            'is_short': n_short,
            'oos_trades': oos_stats['trades'],
            'oos_wr': oos_stats['wr'],
            'oos_pnl': oos_stats['pnl'],
            'oos_mdd': oos_stats['mdd'],
            'oos_pf': oos_stats['pf'],
            'oos_positive': oos_stats['pnl'] > 0,
        })

        print(f"    IS: {len(is_results)} pat ({n_long}L+{n_short}S) | "
              f"OOS: {oos_stats['trades']}tr, WR {oos_stats['wr']}%, "
              f"PnL {oos_stats['pnl']}%")

    positive_folds = sum(1 for f in folds if f['oos_positive'])
    total_oos_pnl = round(sum(f['oos_pnl'] for f in folds), 1)
    total_oos_trades = sum(f['oos_trades'] for f in folds)
    stable = {k: c for k, c in stable_counts.items() if c >= n_folds}

    return {
        'n_folds': n_folds,
        'folds': folds,
        'positive_folds': positive_folds,
        'total_oos_pnl': total_oos_pnl,
        'total_oos_trades': total_oos_trades,
        'stable_pattern_count': len(stable),
        'stable_patterns': sorted(stable.keys()),
    }


# ============================================================
# Post-filter + Comparison
# ============================================================

def apply_post_filters(details, edge_min=21.8, wr_min=60.0, sl_min=1.0):
    """Apply Edge + WR + SL post-filter."""
    filtered = {}
    for key, v in details.items():
        if v['edge'] >= edge_min and v['wr'] >= wr_min and v['sl'] >= sl_min:
            filtered[key] = v
    long_p = sorted([v['pattern'] for v in filtered.values() if v['direction'] == 'LONG'])
    short_p = sorted([v['pattern'] for v in filtered.values() if v['direction'] == 'SHORT'])
    return filtered, long_p, short_p


# ============================================================
# Main
# ============================================================

def main():
    data_file = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
    df = scanner.load_and_classify(data_file)
    types = df['candle_type'].tolist()
    n = len(types)
    signal_index = scanner.build_signal_index(types, n)

    print(f"Data: {n} bars ({n*5/60/24:.0f} days)")
    print(f"Signal patterns: {len(signal_index)} unique 3-candle patterns")
    print(f"MAX_BARS: {MAX_BARS_24H} ({MAX_BARS_24H*5/60:.1f}h)")
    print(f"Timeout handling: FORCED CLOSE at market price")

    t_start = time.time()

    # ============================================================
    # Phase 1: Full Discovery (Forced Close)
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  PHASE 1: Pattern Discovery (Forced Close, 24h)")
    print(f"{'='*70}")

    result = discover_patterns_fc(df, signal_index)

    # ============================================================
    # Phase 2: Post-filter scenarios
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  PHASE 2: Post-Filter Scenarios")
    print(f"{'='*70}")

    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values

    scenarios = {}
    for edge_min in [21.8, 25.0, 30.0]:
        label = f"E{edge_min:.0f}" if edge_min == int(edge_min) else f"E{edge_min}"
        filtered, long_p, short_p = apply_post_filters(
            result['pattern_details'], edge_min=edge_min)

        if filtered:
            all_trades = []
            for key, v in filtered.items():
                sigs = signal_index[v['pattern']]
                trades = bt_signals_forced_close(
                    sigs, v['direction'], v['tp'], v['sl'],
                    opens, highs, lows, closes, n)
                all_trades.extend([(t[0], t[1], t[2]) for t in trades])
            port = scanner.portfolio_1pos(all_trades)
            stats = scanner.calc_stats(port)
            mc_p = scanner.mc_test([t[2] for t in port]) if port else 1.0

            # Timeout breakdown
            all_trades_full = []
            for key, v in filtered.items():
                sigs = signal_index[v['pattern']]
                trades = bt_signals_forced_close(
                    sigs, v['direction'], v['tp'], v['sl'],
                    opens, highs, lows, closes, n)
                all_trades_full.extend(trades)
            n_tp = sum(1 for t in all_trades_full if t[3] == 'TP')
            n_sl = sum(1 for t in all_trades_full if t[3] == 'SL')
            n_to = sum(1 for t in all_trades_full if t[3] == 'TIMEOUT')
            to_pnls = [t[2] for t in all_trades_full if t[3] == 'TIMEOUT']
            to_win = sum(1 for p in to_pnls if p > 0)
        else:
            stats = {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0}
            mc_p = 1.0
            n_tp = n_sl = n_to = to_win = 0
            to_pnls = []

        scenarios[label] = {
            'edge_threshold': edge_min,
            'n_patterns': len(filtered),
            'n_long': len(long_p),
            'n_short': len(short_p),
            'portfolio_stats': stats,
            'mc_p': round(mc_p, 4),
            'exit_breakdown': {
                'TP': n_tp, 'SL': n_sl, 'TIMEOUT': n_to,
                'timeout_win_pct': round(to_win / n_to * 100, 1) if n_to > 0 else 0,
                'timeout_avg_pnl': round(float(np.mean(to_pnls)), 2) if to_pnls else 0,
            },
            'pattern_details': {
                k: {kk: vv for kk, vv in v.items() if kk != 'trades_raw'}
                for k, v in filtered.items()
            },
        }

        print(f"\n  --- {label}: Edge >= {edge_min}pp ---")
        print(f"  Patterns: {len(filtered)} ({len(long_p)}L+{len(short_p)}S)")
        print(f"  PnL: {stats.get('pnl',0):+.1f}% | WR: {stats.get('wr',0):.1f}% | "
              f"MDD: {stats.get('mdd',0):.1f}% | PF: {stats.get('pf',0):.2f}")
        print(f"  Exits: TP={n_tp}, SL={n_sl}, TIMEOUT={n_to} "
              f"(timeout avg PnL: {scenarios[label]['exit_breakdown']['timeout_avg_pnl']:+.2f}%)")

    # ============================================================
    # Phase 3: Walk-Forward (Forced Close)
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  PHASE 3: Walk-Forward Validation (Forced Close)")
    print(f"{'='*70}")

    wf = wf_forced_close(df, signal_index, n_folds=3)
    print(f"\n  WF Result: {wf['positive_folds']}/{wf['n_folds']} positive, "
          f"OOS PnL {wf['total_oos_pnl']:+.1f}%, "
          f"Stable {wf['stable_pattern_count']} patterns")

    # ============================================================
    # Phase 4: Compare with DROP method (from v1 results)
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  PHASE 4: Forced Close vs DROP Comparison")
    print(f"{'='*70}")

    v1_file = os.path.join(_PROJECT_ROOT, 'results', 'max_bars_24h_study.json')
    drop_data = None
    if os.path.exists(v1_file):
        with open(v1_file) as f:
            v1 = json.load(f)
        drop_stats = v1['results']['TEST_288']['post_filter']['portfolio_stats']
        drop_n = v1['results']['TEST_288']['post_filter']['n_patterns']
        drop_wf = v1['walk_forward']['TEST_288']

        fc_stats = scenarios.get('E21.8', {}).get('portfolio_stats', {})
        fc_n = scenarios.get('E21.8', {}).get('n_patterns', 0)

        drop_data = {
            'drop_288': {
                'patterns': drop_n,
                'pnl': drop_stats.get('pnl', 0),
                'wr': drop_stats.get('wr', 0),
                'mdd': drop_stats.get('mdd', 0),
                'pf': drop_stats.get('pf', 0),
                'trades': drop_stats.get('trades', 0),
                'wf_positive': drop_wf['positive_folds'],
                'wf_oos_pnl': drop_wf['total_oos_pnl'],
            },
            'fc_288': {
                'patterns': fc_n,
                'pnl': fc_stats.get('pnl', 0),
                'wr': fc_stats.get('wr', 0),
                'mdd': fc_stats.get('mdd', 0),
                'pf': fc_stats.get('pf', 0),
                'trades': fc_stats.get('trades', 0),
                'wf_positive': wf['positive_folds'],
                'wf_oos_pnl': wf['total_oos_pnl'],
            },
        }

        print(f"\n  {'Method':<20} {'Pat':>5} {'PnL':>10} {'WR':>7} {'MDD':>7} "
              f"{'PF':>7} {'Trades':>7} {'WF':>5} {'WF OOS':>10}")
        print(f"  {'-'*79}")
        d = drop_data['drop_288']
        print(f"  {'DROP (기존)':<20} {d['patterns']:>5} {d['pnl']:>+9.1f}% "
              f"{d['wr']:>6.1f}% {d['mdd']:>6.1f}% {d['pf']:>6.2f} "
              f"{d['trades']:>7} {d['wf_positive']:>2}/3 {d['wf_oos_pnl']:>+9.1f}%")
        c = drop_data['fc_288']
        print(f"  {'FORCED CLOSE':<20} {c['patterns']:>5} {c['pnl']:>+9.1f}% "
              f"{c['wr']:>6.1f}% {c['mdd']:>6.1f}% {c['pf']:>6.2f} "
              f"{c['trades']:>7} {c['wf_positive']:>2}/3 {c['wf_oos_pnl']:>+9.1f}%")

    total_elapsed = time.time() - t_start

    # ============================================================
    # Save Results
    # ============================================================
    output = {
        'study': 'MAX_BARS 24h Forced Close Study',
        'date': '2026-02-18',
        'data': f'{n} bars ({n*5/60/24:.0f} days)',
        'method': 'Forced close at 24h (288 bars) market price',
        'pre_filter': {
            'n_patterns': len(result['pattern_details']),
            'n_long': len(result['long_patterns']),
            'n_short': len(result['short_patterns']),
            'portfolio_stats': result['portfolio_stats'],
        },
        'scenarios': {
            label: {k: v for k, v in sc.items() if k != 'pattern_details'}
            for label, sc in scenarios.items()
        },
        'scenarios_details': {
            label: sc['pattern_details']
            for label, sc in scenarios.items()
        },
        'walk_forward': wf,
        'comparison_with_drop': drop_data,
        'elapsed_seconds': round(total_elapsed, 1),
    }

    out_file = os.path.join(_PROJECT_ROOT, 'results', 'max_bars_24h_forced_close_study.json')
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Total elapsed: {total_elapsed/60:.1f} min")
    print(f"  Results saved: {out_file}")


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)s %(levelname)s %(message)s')
    main()
