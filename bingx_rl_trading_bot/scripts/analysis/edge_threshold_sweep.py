#!/usr/bin/env python3
"""
Edge Threshold Sweep — Quality vs Quantity Analysis
=====================================================
Filters patterns by edge >= {18, 20, 22, 25, 30}pp and measures:
  - Pattern count (total, LONG, SHORT)
  - IS N-pos portfolio performance (PnL, MDD, PnL/MDD, WR)
  - 3-fold expanding-window WF OOS performance

Uses production classify_candle (via scanner), LEVERAGE=3, timeout=DROP.
tp_scale = 0.72 as specified (not 0.5).

Standard Research Protocol enforced.
"""

import sys, os, json, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from scripts.scanner.pattern_scanner import (
    load_and_classify, build_signal_index, portfolio_npos, calc_stats_compound,
    compute_atr_ratio, compute_ema_slope,
    DEFAULT_N_SLOTS, DEFAULT_DIRECTION_CAP, DEFAULT_REGIME_MULT,
    DEFAULT_AGG_RISK_COUNTER, DEFAULT_AGG_RISK_WITH,
    DEFAULT_MOMENTUM_LOOKBACK, DEFAULT_MOMENTUM_THRESHOLD, DEFAULT_MOMENTUM_COOLDOWN,
    DEFAULT_CASCADE_TIGHTEN_PCT, TIMEOUT_BARS,
)

DATA_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'btc_5m_270days_reclassified.csv')
PATTERNS_JSON = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'dynamic_patterns.json')
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'edge_threshold_sweep.json')

EDGE_THRESHOLDS = [18, 20, 22, 25, 30]
TP_SCALE = 0.72
SL_SCALE = 1.0
N_FOLDS = 3


def load_all():
    df = load_and_classify(DATA_FILE)
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    types = df['candle_type'].values
    n = len(df)
    atr_ratio = compute_atr_ratio(highs, lows, closes)
    ema_slope = compute_ema_slope(closes)
    signal_index = build_signal_index(types, n)

    with open(PATTERNS_JSON) as f:
        data = json.load(f)
    json_details = data.get('pattern_details', {})

    return opens, highs, lows, closes, n, atr_ratio, ema_slope, signal_index, json_details


def filter_patterns(json_details, edge_threshold):
    """Filter pattern_details by edge >= threshold. Returns filtered dict."""
    filtered = {}
    for key, det in json_details.items():
        if det.get('edge', 0) >= edge_threshold:
            filtered[key] = det
    return filtered


def build_signal_tuples(filtered_details, signal_index, tp_scale=TP_SCALE, sl_scale=SL_SCALE):
    """Build signal tuples from filtered patterns."""
    tuples = []
    for key, det in filtered_details.items():
        pat = det['pattern']
        direction = det['direction']
        tp = round(max(0.3, det['tp'] * tp_scale), 3)
        sl = round(max(0.5, det['sl'] * sl_scale), 3)
        if pat in signal_index:
            for sig_bar in signal_index[pat]:
                tuples.append((sig_bar, pat, direction, tp, sl))
    return sorted(tuples, key=lambda x: x[0])


def run_npos(signal_tuples, opens, highs, lows, closes, n, atr_ratio, ema_slope,
             start_bar, end_bar):
    """Run N-pos portfolio sim with production defaults."""
    if not signal_tuples:
        return {'trades': 0, 'wr': 0.0, 'pnl': 0.0, 'mdd': 0.0, 'pnl_mdd': 0.0}
    trades, stats = portfolio_npos(
        signal_tuples, opens, highs, lows, closes, n, atr_ratio, ema_slope,
        start_bar, end_bar,
        n_slots=DEFAULT_N_SLOTS,
        direction_cap=DEFAULT_DIRECTION_CAP,
        regime_mult=DEFAULT_REGIME_MULT,
        agg_risk_counter=DEFAULT_AGG_RISK_COUNTER,
        agg_risk_with=DEFAULT_AGG_RISK_WITH,
        momentum_lookback=DEFAULT_MOMENTUM_LOOKBACK,
        momentum_threshold=DEFAULT_MOMENTUM_THRESHOLD,
        momentum_cooldown=DEFAULT_MOMENTUM_COOLDOWN,
        cascade_tighten_pct=DEFAULT_CASCADE_TIGHTEN_PCT,
        timeout_bars=TIMEOUT_BARS,
    )
    active = [t for t in trades if not t.get('drop', False)]
    c = calc_stats_compound(active)
    mdd = stats.get('mdd_mtm', c.get('mdd', 0))
    c['mdd_mtm'] = round(mdd, 2)
    c['pnl_mdd'] = round(c['pnl'] / mdd, 1) if mdd > 0 else round(c['pnl'], 1)
    return c


def expanding_window_folds(n, n_folds=N_FOLDS):
    """Correct expanding window: is_end = n*(fi+1)/(nf+1), last fold OOS goes to end."""
    folds = []
    for fi in range(n_folds):
        is_end = int(n * (fi + 1) / (n_folds + 1))
        oos_end = int(n * (fi + 2) / (n_folds + 1)) if fi < n_folds - 1 else n
        folds.append((is_end, oos_end))
    return folds


def wf_validate(signal_tuples, opens, highs, lows, closes, n, atr_ratio, ema_slope,
                label=""):
    """Run 3-fold expanding window WF. Returns dict with fold details."""
    folds = expanding_window_folds(n)
    fold_results = []
    for fi, (is_end, oos_end) in enumerate(folds):
        oos = run_npos(signal_tuples, opens, highs, lows, closes, n,
                       atr_ratio, ema_slope, is_end, oos_end)
        fold_results.append({
            'fold': fi + 1,
            'is_end': is_end,
            'oos_end': oos_end,
            'oos_bars': oos_end - is_end,
            'trades': oos['trades'],
            'wr': round(oos['wr'], 1),
            'pnl': round(oos['pnl'], 1),
            'mdd_mtm': oos.get('mdd_mtm', 0),
            'pass': oos['pnl'] > 0,
        })
        tag = "PASS" if oos['pnl'] > 0 else "FAIL"
        print(f"  {label:25s} F{fi+1}: OOS={oos['pnl']:+7.1f}% ({oos['trades']}t, WR={oos['wr']:.1f}%) [{tag}]")

    total_oos = sum(f['pnl'] for f in fold_results)
    passes = sum(1 for f in fold_results if f['pass'])
    print(f"  {label:25s} TOTAL={total_oos:+7.1f}% [{passes}/{N_FOLDS}]")
    return {
        'folds': fold_results,
        'total_oos_pnl': round(total_oos, 1),
        'passes': passes,
        'verdict': 'PASS' if passes == N_FOLDS else 'PARTIAL' if passes > 0 else 'FAIL',
    }


def main():
    t0 = time.time()
    opens, highs, lows, closes, n, atr_ratio, ema_slope, signal_index, json_details = load_all()
    total_patterns = len(json_details)
    print(f"Data: {n} bars, {n / 288:.0f} days, {total_patterns} total patterns in JSON")
    print(f"Config: N={DEFAULT_N_SLOTS}, DirCap={DEFAULT_DIRECTION_CAP}, "
          f"Cascade={DEFAULT_CASCADE_TIGHTEN_PCT}%, Timeout={TIMEOUT_BARS}, "
          f"TP_scale={TP_SCALE}")
    print(f"Edge thresholds to sweep: {EDGE_THRESHOLDS}\n")

    results = {}

    for edge_th in EDGE_THRESHOLDS:
        print(f"{'='*60}")
        print(f"Edge >= {edge_th}pp")
        print(f"{'='*60}")

        filtered = filter_patterns(json_details, edge_th)
        n_long = sum(1 for d in filtered.values() if d['direction'] == 'LONG')
        n_short = sum(1 for d in filtered.values() if d['direction'] == 'SHORT')
        n_pat = len(filtered)
        print(f"Patterns: {n_pat} ({n_long}L + {n_short}S) — dropped {total_patterns - n_pat}")

        if n_pat == 0:
            print("  No patterns — skip")
            results[str(edge_th)] = {
                'edge_threshold': edge_th,
                'patterns': 0, 'long': 0, 'short': 0,
                'is': None, 'wf': None,
            }
            continue

        signal_tuples = build_signal_tuples(filtered, signal_index)
        print(f"Signals: {len(signal_tuples)}")

        # Full IS
        print(f"\n  IS (full data):")
        is_stats = run_npos(signal_tuples, opens, highs, lows, closes, n,
                            atr_ratio, ema_slope, 0, n)
        print(f"    Trades={is_stats['trades']}, WR={is_stats['wr']:.1f}%, "
              f"PnL={is_stats['pnl']:+.1f}%, MDD_MTM={is_stats.get('mdd_mtm', 0):.2f}%, "
              f"PnL/MDD={is_stats.get('pnl_mdd', 0):.1f}")

        # WF
        print(f"\n  WF (3-fold expanding):")
        wf = wf_validate(signal_tuples, opens, highs, lows, closes, n,
                         atr_ratio, ema_slope, label=f"edge>={edge_th}")

        results[str(edge_th)] = {
            'edge_threshold': edge_th,
            'patterns': n_pat,
            'long': n_long,
            'short': n_short,
            'is': {
                'trades': is_stats['trades'],
                'wr': round(is_stats['wr'], 1),
                'pnl': round(is_stats['pnl'], 1),
                'mdd_mtm': is_stats.get('mdd_mtm', 0),
                'pnl_mdd': is_stats.get('pnl_mdd', 0),
            },
            'wf': wf,
        }
        print()

    # Summary table
    print(f"\n{'='*80}")
    print(f"SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'Edge':>6} {'Pats':>5} {'L':>3} {'S':>3} {'IS Trades':>10} {'IS WR':>7} "
          f"{'IS PnL':>9} {'IS MDD':>8} {'IS P/M':>8} {'OOS Tot':>9} {'WF':>5}")
    print(f"{'-'*80}")

    for edge_th in EDGE_THRESHOLDS:
        r = results[str(edge_th)]
        if r['is'] is None:
            print(f"{edge_th:>5}pp {'0':>5} {'-':>3} {'-':>3} {'---':>10} {'---':>7} "
                  f"{'---':>9} {'---':>8} {'---':>8} {'---':>9} {'N/A':>5}")
            continue
        i = r['is']
        w = r['wf']
        print(f"{edge_th:>5}pp {r['patterns']:>5} {r['long']:>3} {r['short']:>3} "
              f"{i['trades']:>10} {i['wr']:>6.1f}% {i['pnl']:>+8.1f}% "
              f"{i['mdd_mtm']:>7.2f}% {i['pnl_mdd']:>7.1f} {w['total_oos_pnl']:>+8.1f}% "
              f"{w['verdict']:>5}")

    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.0f}s")

    # Save
    output = {
        'study': 'edge_threshold_sweep',
        'date': '2026-03-12',
        'config': {
            'tp_scale': TP_SCALE,
            'sl_scale': SL_SCALE,
            'n_slots': DEFAULT_N_SLOTS,
            'direction_cap': DEFAULT_DIRECTION_CAP,
            'cascade_tighten_pct': DEFAULT_CASCADE_TIGHTEN_PCT,
            'timeout_bars': TIMEOUT_BARS,
            'agg_risk_counter': DEFAULT_AGG_RISK_COUNTER,
            'agg_risk_with': DEFAULT_AGG_RISK_WITH,
            'momentum': f'{DEFAULT_MOMENTUM_THRESHOLD}%/{DEFAULT_MOMENTUM_LOOKBACK}b/{DEFAULT_MOMENTUM_COOLDOWN}b',
            'n_folds': N_FOLDS,
        },
        'total_patterns_in_json': total_patterns,
        'edge_thresholds': EDGE_THRESHOLDS,
        'results': results,
        'elapsed_s': round(elapsed, 0),
    }
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, cls=NpEncoder)
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
