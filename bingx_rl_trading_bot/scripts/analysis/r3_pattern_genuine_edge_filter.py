#!/usr/bin/env python3
"""R3: Pattern Filtering by Genuine Edge (v1.53.0 config).
Q1: How many of 131 patterns have genuine edge (cascade-OFF, 1-pos)?
Q2: Does filtering improve WF discrimination? Q3: Lower breakeven WR?
Protocol: production classify_candle, LEV=3, timeout=DROP, fee=0.10%, slip=0.02%.
Usage: cd bingx_rl_trading_bot && python scripts/analysis/r3_pattern_genuine_edge_filter.py
"""

import os, sys, json, logging, time
import numpy as np, pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)

from scripts.production.pattern_5m.indicators import classify_candle
from scripts.production.pattern_5m.constants import AVG_BODY_WINDOW

sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'scripts', 'scanner'))
from pattern_scanner import (
    load_and_classify, find_neutral_window, build_signal_index,
    mc_test, compute_atr_ratio, compute_ema_slope,
    portfolio_npos, calc_stats_compound,
    scan_universe_range, _check_exit_npos,
    bt_signals_atr,
    FEE_PCT, LEVERAGE, MAX_BARS, SLIPPAGE_BUFFER,
    DEFAULT_N_SLOTS, DEFAULT_DIRECTION_CAP,
    DEFAULT_AGG_RISK_COUNTER, DEFAULT_AGG_RISK_WITH,
    DEFAULT_MOMENTUM_LOOKBACK, DEFAULT_MOMENTUM_THRESHOLD,
    DEFAULT_MOMENTUM_COOLDOWN, DEFAULT_CASCADE_TIGHTEN_PCT,
    DEFAULT_ATR_PERIOD, DEFAULT_ATR_WINDOW,
    DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI,
    DEFAULT_EDGE_THRESHOLD, DEFAULT_MC_THRESHOLD, DEFAULT_MIN_TRADES,
    MC_SEEDS, TIMEOUT_BARS,
)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('r3_genuine_edge')

DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
PATTERNS_FILE = os.path.join(_PROJECT_ROOT, 'results', 'dynamic_patterns.json')
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'r3_pattern_genuine_edge_filter.json')

# Genuine edge thresholds (more lenient than production for classification)
GENUINE_MC_P = 0.05
GENUINE_WR = 55.0
MARGINAL_MC_P = 0.10
MARGINAL_WR = 52.0

N_RANDOM_SEEDS = 10
WF_FOLDS = 3


def load_patterns():
    """Load dynamic_patterns.json and extract per-pattern TP/SL."""
    with open(PATTERNS_FILE) as f:
        dp = json.load(f)
    details = dp.get('pattern_details', {})
    patterns = {}
    for key, info in details.items():
        patterns[key] = {
            'pattern': info['pattern'],
            'direction': info['direction'],
            'tp': info['tp'],
            'sl': info['sl'],
        }
    return patterns, dp


def phase1_per_pattern_edge(df, nw_start, nw_end, atr_ratio, pat_configs):
    """Phase 1: Assess genuine edge per pattern (cascade-OFF, 1-pos, ATR-scaled)."""
    logger.info("=== Phase 1: Per-Pattern Genuine Edge Assessment ===")
    opens, highs, lows = df['open'].values, df['high'].values, df['low'].values
    n_bars = len(df)
    signal_index = build_signal_index(df['candle_type'].values, n_bars)
    no_edge = lambda reason: {'wr': 0, 'pnl': 0, 'trades': 0, 'mc_p': 1.0,
                               'avg_win': 0, 'avg_loss': 0, 'rr': 0,
                               'class': 'none', 'reason': reason}
    results = {}
    for pat_key, cfg in pat_configs.items():
        sigs = [s for s in signal_index.get(cfg['pattern'], []) if nw_start <= s < nw_end]
        if len(sigs) < 5:
            results[pat_key] = no_edge('insufficient_signals'); continue
        trades = bt_signals_atr(sigs, cfg['direction'], cfg['tp'], cfg['sl'],
                                opens, highs, lows, nw_end,
                                atr_ratio, DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI)
        if len(trades) < 5:
            results[pat_key] = no_edge('insufficient_trades'); continue
        pnls = [t[2] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        wr = len(wins) / len(trades) * 100
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        rr = abs(avg_win / avg_loss) if avg_loss != 0 else 999
        mc_p = mc_test(pnls)
        if mc_p < GENUINE_MC_P and wr > GENUINE_WR: cls = 'genuine'
        elif mc_p < MARGINAL_MC_P or wr > MARGINAL_WR: cls = 'marginal'
        else: cls = 'none'
        results[pat_key] = {'wr': round(wr, 1), 'pnl': round(sum(pnls), 2),
            'trades': len(trades), 'mc_p': round(mc_p, 4),
            'avg_win': round(avg_win, 3), 'avg_loss': round(avg_loss, 3),
            'rr': round(rr, 3), 'class': cls}

    genuine = [k for k, v in results.items() if v['class'] == 'genuine']
    marginal = [k for k, v in results.items() if v['class'] == 'marginal']
    none_cls = [k for k, v in results.items() if v['class'] == 'none']
    summary = {'genuine': len(genuine), 'marginal': len(marginal), 'none': len(none_cls),
               'genuine_patterns': genuine, 'marginal_patterns': marginal}
    logger.info(f"Phase 1: {len(genuine)} genuine, {len(marginal)} marginal, "
                f"{len(none_cls)} none (of {len(pat_configs)})")
    return results, summary


def build_signal_tuples(signal_index, pat_configs, filter_keys, start, end):
    """Build signal_tuples for portfolio_npos from filtered pattern keys."""
    tuples = []
    for pk in filter_keys:
        c = pat_configs[pk]
        for s in signal_index.get(c['pattern'], []):
            if start <= s < end:
                tuples.append((s, c['pattern'], c['direction'], c['tp'], c['sl']))
    return sorted(tuples, key=lambda x: x[0])


def run_npos(sig_tuples, opens, highs, lows, closes, n_bars,
             atr_ratio, ema_slope, start_bar, end_bar,
             cascade_pct=DEFAULT_CASCADE_TIGHTEN_PCT):
    """Run N-pos portfolio simulation, return (stats, trades)."""
    trades, meta = portfolio_npos(
        sig_tuples, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, start_bar, end_bar,
        n_slots=DEFAULT_N_SLOTS, direction_cap=DEFAULT_DIRECTION_CAP,
        agg_risk_counter=DEFAULT_AGG_RISK_COUNTER, agg_risk_with=DEFAULT_AGG_RISK_WITH,
        momentum_lookback=DEFAULT_MOMENTUM_LOOKBACK, momentum_threshold=DEFAULT_MOMENTUM_THRESHOLD,
        momentum_cooldown=DEFAULT_MOMENTUM_COOLDOWN,
        clamp_lo=DEFAULT_ATR_CLAMP_LO, clamp_hi=DEFAULT_ATR_CLAMP_HI,
        timeout_bars=TIMEOUT_BARS, cascade_tighten_pct=cascade_pct)
    stats = calc_stats_compound(trades)
    wins = [t['pnl_slot'] for t in trades if t['pnl_slot'] > 0]
    losses = [t['pnl_slot'] for t in trades if t['pnl_slot'] <= 0]
    avg_w = np.mean(wins) if wins else 0
    avg_l = abs(np.mean(losses)) if losses else 0
    rr = avg_w / avg_l if avg_l > 0 else 999
    stats['rr'] = round(rr, 3)
    stats['be_wr'] = round(1 / (1 + rr) * 100 if rr > 0 else 100, 1)
    stats['blocked'] = meta.get('blocked', {})
    return stats, trades


def phase2_filtered_backtest(df, nw_start, nw_end, atr_ratio, ema_slope,
                             pat_configs, signal_index, p1_results):
    """Phase 2: N-pos backtest with filtered pattern sets (cascade ON)."""
    logger.info("=== Phase 2: Filtered N-pos Backtest (Cascade ON) ===")
    opens, highs, lows, closes = (df['open'].values, df['high'].values,
                                   df['low'].values, df['close'].values)
    n_bars = len(df)
    genuine_keys = [k for k, v in p1_results.items() if v['class'] == 'genuine']
    gm_keys = genuine_keys + [k for k, v in p1_results.items() if v['class'] == 'marginal']
    empty = {'trades': 0, 'wr': 0, 'pnl': 0, 'mdd': 0, 'pnl_mdd': 0, 'rr': 0, 'be_wr': 100}
    results = {}
    for label, keys in [('genuine_only', genuine_keys),
                         ('genuine_marginal', gm_keys),
                         ('all_131', list(pat_configs.keys()))]:
        if not keys:
            results[label] = empty; continue
        st = build_signal_tuples(signal_index, pat_configs, keys, nw_start, nw_end)
        stats, _ = run_npos(st, opens, highs, lows, closes, n_bars,
                            atr_ratio, ema_slope, nw_start, nw_end)
        results[label] = stats
        logger.info(f"  {label}: {len(keys)}pat {stats['trades']}t WR{stats['wr']}% "
                    f"PnL{stats['pnl']}% MDD{stats['mdd']}% R:R{stats['rr']} BE{stats['be_wr']}%")
    return results


def phase3_random_discrimination(df, nw_start, nw_end, atr_ratio, ema_slope,
                                  pat_configs, signal_index, p1_results):
    """Phase 3: Random signal discrimination test (10 seeds per filter level)."""
    logger.info("=== Phase 3: Random Discrimination Test ===")
    opens, highs, lows, closes = (df['open'].values, df['high'].values,
                                   df['low'].values, df['close'].values)
    n_bars = len(df)
    genuine_keys = [k for k, v in p1_results.items() if v['class'] == 'genuine']
    gm_keys = [k for k, v in p1_results.items() if v['class'] in ('genuine', 'marginal')]
    seg_size = (nw_end - nw_start) // (WF_FOLDS + 1)
    results = {}
    for label, keys in [('genuine_only', genuine_keys), ('genuine_marginal', gm_keys),
                         ('all_131', list(pat_configs.keys()))]:
        if not keys:
            results[label] = {'pass_rate': 0, 'n_seeds': 0}; continue
        real_tuples = build_signal_tuples(signal_index, pat_configs, keys, nw_start, nw_end)
        n_signals = len(real_tuples)
        if n_signals < 10:
            results[label] = {'pass_rate': 0, 'n_seeds': 0, 'note': 'too_few'}; continue
        dir_tp_sl = [(pat_configs[k]['direction'], pat_configs[k]['tp'],
                      pat_configs[k]['sl']) for k in keys]
        wf_passes = 0
        for seed in range(N_RANDOM_SEEDS):
            rng = np.random.RandomState(seed + 1000)
            rand_bars = rng.randint(nw_start + 2, nw_end - MAX_BARS, size=n_signals)
            rand_tuples = sorted([(int(b), f"R{i}", *dir_tp_sl[i % len(dir_tp_sl)])
                                   for i, b in enumerate(rand_bars)], key=lambda x: x[0])
            ok = True
            for fold in range(WF_FOLDS):
                oos_s = nw_start + (fold + 1) * seg_size
                oos_e = min(oos_s + seg_size, nw_end)
                oos_t = [t for t in rand_tuples if oos_s <= t[0] < oos_e]
                is_t = [t for t in rand_tuples if nw_start <= t[0] < oos_s]
                if len(is_t) < DEFAULT_MIN_TRADES or len(oos_t) < 5:
                    ok = False; break
                s, _ = run_npos(oos_t, opens, highs, lows, closes, n_bars,
                                atr_ratio, ema_slope, oos_s, oos_e)
                if s['pnl'] <= 0: ok = False; break
            if ok: wf_passes += 1
        pr = wf_passes / N_RANDOM_SEEDS * 100
        results[label] = {'pass_rate': round(pr, 1), 'n_seeds': N_RANDOM_SEEDS,
                          'wf_passes': wf_passes, 'n_signals': n_signals}
        logger.info(f"  {label}: {wf_passes}/{N_RANDOM_SEEDS} random pass ({pr:.0f}%)")
    return results


def phase4_wf_validation(df, nw_start, nw_end, atr_ratio, ema_slope,
                          pat_configs, signal_index, p1_results):
    """Phase 4: Expanding window WF for genuine-only and genuine+marginal."""
    logger.info("=== Phase 4: WF Validation (3-fold) ===")
    opens, highs, lows, closes = (df['open'].values, df['high'].values,
                                   df['low'].values, df['close'].values)
    n_bars, types = len(df), df['candle_type'].values
    genuine_keys = [k for k, v in p1_results.items() if v['class'] == 'genuine']
    gm_keys = [k for k, v in p1_results.items() if v['class'] in ('genuine', 'marginal')]
    sig_idx = build_signal_index(types, n_bars)
    seg_size = (nw_end - nw_start) // (WF_FOLDS + 1)
    results = {}
    for label, fkeys in [('genuine_only', genuine_keys), ('genuine_marginal', gm_keys)]:
        if len(fkeys) < 3:
            results[label] = {'verdict': 'SKIP', 'reason': 'too_few_patterns'}; continue
        folds, all_pass = [], True
        for fold in range(WF_FOLDS):
            is_end = nw_start + (fold + 1) * seg_size
            oos_s, oos_e = is_end, min(is_end + seg_size, nw_end)
            # IS: re-validate patterns in IS window
            is_sel = []
            for pk in fkeys:
                c = pat_configs[pk]
                is_sigs = [s for s in sig_idx.get(c['pattern'], []) if nw_start <= s < is_end]
                if len(is_sigs) < DEFAULT_MIN_TRADES: continue
                tr = bt_signals_atr(is_sigs, c['direction'], c['tp'], c['sl'],
                                    opens, highs, lows, is_end,
                                    atr_ratio, DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI)
                if len(tr) < DEFAULT_MIN_TRADES: continue
                pnls = [t[2] for t in tr]
                wr = len([p for p in pnls if p > 0]) / len(pnls) * 100
                edge = wr - c['sl'] / (c['tp'] + c['sl']) * 100
                if edge < DEFAULT_EDGE_THRESHOLD: continue
                if mc_test(pnls) >= DEFAULT_MC_THRESHOLD: continue
                is_sel.append(pk)
            if not is_sel:
                folds.append({'fold': fold+1, 'is_patterns': 0, 'oos_pnl': 0, 'pass': False})
                all_pass = False; continue
            oos_t = build_signal_tuples(sig_idx, pat_configs, is_sel, oos_s, oos_e)
            if not oos_t:
                folds.append({'fold': fold+1, 'is_patterns': len(is_sel),
                              'oos_trades': 0, 'oos_pnl': 0, 'pass': False})
                all_pass = False; continue
            st, _ = run_npos(oos_t, opens, highs, lows, closes, n_bars,
                             atr_ratio, ema_slope, oos_s, oos_e)
            fp = st['pnl'] > 0
            if not fp: all_pass = False
            folds.append({'fold': fold+1, 'is_bars': is_end - nw_start,
                          'oos_bars': oos_e - oos_s, 'is_patterns': len(is_sel),
                          'oos_trades': st['trades'], 'oos_wr': st['wr'],
                          'oos_pnl': st['pnl'], 'oos_mdd': st['mdd'], 'pass': fp})
            logger.info(f"  {label} F{fold+1}: {len(is_sel)}pat OOS {st['trades']}t "
                        f"WR{st['wr']}% PnL{st['pnl']}% {'PASS' if fp else 'FAIL'}")
        tot_pnl = sum(f.get('oos_pnl', 0) for f in folds)
        pos_f = sum(1 for f in folds if f.get('pass'))
        results[label] = {'folds': folds, 'positive_folds': pos_f, 'total_folds': WF_FOLDS,
                          'total_oos_pnl': round(tot_pnl, 2), 'all_pass': all_pass,
                          'verdict': 'PASS' if all_pass else 'FAIL'}
        logger.info(f"  {label} WF: {pos_f}/{WF_FOLDS}, OOS PnL {tot_pnl:.1f}%")
    return results


def main():
    t0 = time.time()
    logger.info("R3: Pattern Genuine Edge Filter Study (v1.53.0 config)")
    df = load_and_classify(DATA_FILE)
    n_bars = len(df)
    closes, opens, highs, lows = df['close'].values, df['open'].values, df['high'].values, df['low'].values
    nw = find_neutral_window(closes, tol_pct=1.0)
    if nw is None:
        logger.error("No neutral window found"); return
    nw_start, nw_end = nw
    nw_days = (nw_end - nw_start) / 288
    logger.info(f"Neutral window: {nw_start}-{nw_end} ({nw_days:.0f}d)")
    atr_ratio = compute_atr_ratio(highs, lows, closes, DEFAULT_ATR_PERIOD, DEFAULT_ATR_WINDOW)
    ema_slope = compute_ema_slope(closes)
    pat_configs, _ = load_patterns()
    logger.info(f"Loaded {len(pat_configs)} patterns")
    signal_index = build_signal_index(df['candle_type'].values, n_bars)

    p1_results, p1_summary = phase1_per_pattern_edge(df, nw_start, nw_end, atr_ratio, pat_configs)
    p2_results = phase2_filtered_backtest(df, nw_start, nw_end, atr_ratio, ema_slope,
                                           pat_configs, signal_index, p1_results)
    p3_results = phase3_random_discrimination(df, nw_start, nw_end, atr_ratio, ema_slope,
                                               pat_configs, signal_index, p1_results)
    p4_results = phase4_wf_validation(df, nw_start, nw_end, atr_ratio, ema_slope,
                                       pat_configs, signal_index, p1_results)

    # Build verdicts
    ng = p1_summary['genuine']
    apr = p3_results.get('all_131', {}).get('pass_rate', 100)
    gpr = p3_results.get('genuine_only', {}).get('pass_rate', 100)
    gmpr = p3_results.get('genuine_marginal', {}).get('pass_rate', 100)
    disc_imp = gpr < apr
    all_be = p2_results.get('all_131', {}).get('be_wr', 100)
    gen_be = p2_results.get('genuine_only', {}).get('be_wr', 100)
    gm_be = p2_results.get('genuine_marginal', {}).get('be_wr', 100)
    gen_wf = p4_results.get('genuine_only', {}).get('verdict', 'N/A')
    verdicts = {
        'q1_genuine_count': {'genuine': ng, 'marginal': p1_summary['marginal'],
            'none': p1_summary['none'],
            'verdict': f"{ng} of {len(pat_configs)} genuine (cascade-OFF)"},
        'q2_discrimination': {'all_131_random_pass_rate': apr,
            'genuine_only_random_pass_rate': gpr, 'genuine_marginal_random_pass_rate': gmpr,
            'improved': disc_imp,
            'verdict': f"{'IMPROVED' if disc_imp else 'NO_CHANGE'} — gen {gpr}% vs all {apr}%"},
        'q3_survival': {'all_131_be_wr': all_be, 'genuine_only_be_wr': gen_be,
            'genuine_marginal_be_wr': gm_be,
            'verdict': f"BE_WR gen {gen_be}% vs all {all_be}% "
                       f"({'BETTER' if gen_be < all_be else 'WORSE'})"},
        'overall': {
            'recommendation': 'FILTER' if (disc_imp and gen_wf == 'PASS') else 'KEEP_ALL',
            'reasoning': f"WF {gen_wf}, disc {'improved' if disc_imp else 'unchanged'}, "
                         f"BE_WR {'lower' if gen_be < all_be else 'higher'}"},
    }
    elapsed = time.time() - t0
    output = {
        'metadata': {'study': 'R3 Pattern Genuine Edge Filter',
            'date': pd.Timestamp.now().isoformat(), 'data_bars': n_bars,
            'neutral_window': {'start': int(nw_start), 'end': int(nw_end), 'days': round(nw_days, 1)},
            'config': {'leverage': LEVERAGE, 'fee_pct': FEE_PCT, 'slippage': SLIPPAGE_BUFFER,
                'timeout_bars': TIMEOUT_BARS, 'atr_clamp': [DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI],
                'agg_risk': [DEFAULT_AGG_RISK_COUNTER, DEFAULT_AGG_RISK_WITH],
                'cascade_tighten_pct': DEFAULT_CASCADE_TIGHTEN_PCT,
                'n_slots': DEFAULT_N_SLOTS, 'direction_cap': DEFAULT_DIRECTION_CAP,
                'genuine_thresholds': {'mc_p': GENUINE_MC_P, 'wr': GENUINE_WR},
                'marginal_thresholds': {'mc_p': MARGINAL_MC_P, 'wr': MARGINAL_WR}},
            'elapsed_seconds': round(elapsed, 1)},
        'phase1_per_pattern': p1_results, 'phase1_summary': p1_summary,
        'phase2_filtered': p2_results, 'phase3_random_discrimination': p3_results,
        'phase4_wf': p4_results, 'verdicts': verdicts,
    }
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"Saved to {OUTPUT_FILE} ({elapsed:.1f}s)")

    # Summary
    print(f"\n{'='*70}\nR3 PATTERN GENUINE EDGE FILTER — SUMMARY\n{'='*70}")
    print(f"\nP1: {ng} genuine / {p1_summary['marginal']} marginal / "
          f"{p1_summary['none']} none (of {len(pat_configs)})")
    print("\nP2 (N-pos, Cascade ON):")
    for lb in ['genuine_only', 'genuine_marginal', 'all_131']:
        s = p2_results.get(lb, {})
        print(f"  {lb:20s}: {s.get('trades',0):4d}t WR{s.get('wr',0):5.1f}% "
              f"PnL{s.get('pnl',0):+8.1f}% MDD{s.get('mdd',0):5.2f}% "
              f"R:R{s.get('rr',0):.3f} BE{s.get('be_wr',0):.1f}%")
    print("\nP3 (Random Discrimination):")
    for lb in ['genuine_only', 'genuine_marginal', 'all_131']:
        s = p3_results.get(lb, {})
        print(f"  {lb:20s}: {s.get('pass_rate',0):.0f}% ({s.get('wf_passes',0)}/{s.get('n_seeds',0)})")
    print("\nP4 (WF):")
    for lb in ['genuine_only', 'genuine_marginal']:
        s = p4_results.get(lb, {})
        print(f"  {lb:20s}: {s.get('verdict','N/A')} ({s.get('positive_folds',0)}/{s.get('total_folds',0)}) "
              f"OOS{s.get('total_oos_pnl',0):+.1f}%")
    print(f"\nVerdict: {verdicts['overall']['recommendation']}")
    print(f"  {verdicts['overall']['reasoning']}\nElapsed: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
