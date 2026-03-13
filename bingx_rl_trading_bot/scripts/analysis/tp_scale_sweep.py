"""TP Scale Factor N-pos Sweep: 0.4~1.0 range.

Sweeps tp_scale_factor using existing 131 patterns from dynamic_patterns.json.
For each scale factor:
  - Apply max(0.3%, tp * scale) to each pattern's TP (SL unchanged)
  - Run N-pos portfolio simulation on full IS period
  - Run 3-fold expanding window WF (OOS only)
  - Collect IS + OOS metrics

Output: results/tp_scale_sweep.json
"""
import sys
import os
import json
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.scanner.pattern_scanner import (
    load_and_classify, find_neutral_window, build_signal_index,
    portfolio_npos, compute_atr_ratio, compute_ema_slope,
    calc_stats_compound,
)

import numpy as np

# --- Constants (production-aligned) ---
LEVERAGE = 3
FEE_PCT = 0.0005
N_SLOTS = 9
DIRECTION_CAP = 7
MAX_BARS = 288
CASCADE_TIGHTEN_PCT = 85
AGG_RISK_COUNTER = 8.0
AGG_RISK_WITH = 15.0
MOMENTUM_LOOKBACK = 3
MOMENTUM_THRESHOLD = 1.5
MOMENTUM_COOLDOWN = 12
ATR_PERIOD = 14
ATR_WINDOW = 576
ATR_CLAMP_LO = 0.5
ATR_CLAMP_HI = 1.5
REGIME_MULT = 1.0
TP_MIN_FLOOR = 0.3  # minimum TP after scaling

N_FOLDS = 3
DATA_FILE = 'data/btc_5m_270days_reclassified.csv'
PATTERNS_FILE = 'results/dynamic_patterns.json'
OUTPUT_FILE = 'results/tp_scale_sweep.json'

SCALE_FACTORS = [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.72, 0.75, 0.80, 0.90, 1.00]


def load_patterns(patterns_file):
    """Load pattern details from dynamic_patterns.json."""
    with open(patterns_file, 'r') as f:
        data = json.load(f)
    pattern_details = data['pattern_details']
    patterns = []
    for key, info in pattern_details.items():
        patterns.append({
            'pattern': info['pattern'],
            'direction': info['direction'],
            'tp': info['tp'],
            'sl': info['sl'],
        })
    return patterns


def build_signal_tuples(patterns, signal_index, tp_scale, start_bar, end_bar):
    """Build signal tuples with scaled TP for portfolio_npos."""
    tuples = []
    for p in patterns:
        pat_name = p['pattern']
        direction = p['direction']
        tp_scaled = max(TP_MIN_FLOOR, p['tp'] * tp_scale)
        sl = p['sl']
        for s in signal_index.get(pat_name, []):
            if start_bar <= s < end_bar:
                tuples.append((s, pat_name, direction, tp_scaled, sl))
    return tuples


def run_npos(signal_tuples, opens, highs, lows, closes, n_bars,
             atr_ratio, ema_slope, start_bar, end_bar):
    """Run portfolio_npos with production-aligned parameters."""
    trades, stats = portfolio_npos(
        signal_tuples, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, start_bar, end_bar,
        n_slots=N_SLOTS,
        direction_cap=DIRECTION_CAP,
        regime_mult=REGIME_MULT,
        agg_risk_counter=AGG_RISK_COUNTER,
        agg_risk_with=AGG_RISK_WITH,
        momentum_lookback=MOMENTUM_LOOKBACK,
        momentum_threshold=MOMENTUM_THRESHOLD,
        momentum_cooldown=MOMENTUM_COOLDOWN,
        clamp_lo=ATR_CLAMP_LO,
        clamp_hi=ATR_CLAMP_HI,
        timeout_bars=MAX_BARS,
        cascade_tighten_pct=CASCADE_TIGHTEN_PCT,
    )
    return trades, stats


def main():
    print("=" * 70)
    print("TP Scale Factor N-pos Sweep")
    print("=" * 70)

    # Load data
    print("\n[1] Loading data...")
    df = load_and_classify(DATA_FILE)
    closes = df['close'].values
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    n = len(df)
    print(f"    Data: {n} bars ({n / 288:.1f} days)")

    # Neutral window
    nw = find_neutral_window(closes, tol_pct=1.0)
    if nw is None:
        print("    ERROR: No neutral window found")
        return
    ns, ne = nw[0], nw[1]
    print(f"    Neutral window: [{ns}, {ne}] ({(ne - ns) / 288:.0f} days)")

    # Build signal index
    type_codes = df['candle_type'].tolist()
    signal_index = build_signal_index(type_codes, len(type_codes))

    # ATR ratio (full data for IS, per-fold for OOS)
    atr_ratio_full = compute_atr_ratio(highs, lows, closes,
                                       atr_period=ATR_PERIOD, window=ATR_WINDOW)
    ema_slope_full = compute_ema_slope(closes)

    # Load patterns
    print("\n[2] Loading patterns...")
    patterns = load_patterns(PATTERNS_FILE)
    n_long = sum(1 for p in patterns if p['direction'] == 'LONG')
    n_short = sum(1 for p in patterns if p['direction'] == 'SHORT')
    print(f"    Patterns: {len(patterns)} ({n_long}L + {n_short}S)")

    # WF fold boundaries (expanding window, same as scanner)
    seg_size = n // (N_FOLDS + 1)
    folds = []
    for fi in range(N_FOLDS):
        is_end = (fi + 1) * seg_size
        oos_start = is_end
        oos_end = (fi + 2) * seg_size if fi < N_FOLDS - 1 else n
        folds.append((0, is_end, oos_start, oos_end))
    print(f"\n    WF folds ({N_FOLDS}):")
    for i, (is_s, is_e, oos_s, oos_e) in enumerate(folds):
        print(f"      Fold {i+1}: IS=[{is_s}, {is_e}) OOS=[{oos_s}, {oos_e})")

    # Sweep
    print(f"\n[3] Sweeping {len(SCALE_FACTORS)} scale factors...")
    results = {}

    for scale in SCALE_FACTORS:
        t0 = time.time()
        print(f"\n  --- tp_scale = {scale:.2f} ---")

        # IS: full period simulation
        is_tuples = build_signal_tuples(patterns, signal_index, scale, 0, n)
        is_trades, is_stats = run_npos(
            is_tuples, opens, highs, lows, closes, n,
            atr_ratio_full, ema_slope_full, 0, n
        )
        is_compound = calc_stats_compound(is_trades)
        is_mdd_mtm = is_stats.get('mdd_mtm', is_compound['mdd'])
        is_pnl = is_compound['pnl']
        is_wr = is_compound['wr']
        is_n_trades = is_compound['trades']
        is_pnl_mdd = is_pnl / is_mdd_mtm if is_mdd_mtm > 0 else 0

        print(f"    IS: PnL={is_pnl:+.1f}%  MDD={is_mdd_mtm:.2f}%  "
              f"PnL/MDD={is_pnl_mdd:.1f}x  WR={is_wr:.1f}%  trades={is_n_trades}")

        # WF OOS
        oos_pnls = []
        for fi, (_, is_e, oos_s, oos_e) in enumerate(folds):
            # ATR on IS data only (no look-ahead)
            fold_atr = compute_atr_ratio(
                highs[:is_e], lows[:is_e], closes[:is_e],
                atr_period=ATR_PERIOD, window=ATR_WINDOW
            )
            fold_ema = compute_ema_slope(closes[:oos_e])

            # OOS signal tuples (patterns fixed, TP scaled)
            oos_tuples = build_signal_tuples(patterns, signal_index, scale, oos_s, oos_e)

            oos_trades, oos_npos_stats = run_npos(
                oos_tuples, opens, highs, lows, closes, oos_e,
                fold_atr, fold_ema, oos_s, oos_e
            )
            oos_compound = calc_stats_compound(oos_trades)
            oos_pnl = oos_compound['pnl']
            oos_pnls.append(oos_pnl)
            print(f"    Fold {fi+1} OOS: PnL={oos_pnl:+.1f}%  "
                  f"WR={oos_compound['wr']:.1f}%  trades={oos_compound['trades']}")

        oos_total = sum(oos_pnls)
        all_positive = all(p > 0 for p in oos_pnls)
        verdict = "PASS" if all_positive else "FAIL"

        elapsed = time.time() - t0
        print(f"    OOS Total: {oos_total:+.1f}%  Verdict: {verdict}  ({elapsed:.1f}s)")

        results[str(scale)] = {
            'tp_scale': scale,
            'is_pnl': round(is_pnl, 2),
            'is_mdd': round(is_mdd_mtm, 3),
            'is_pnl_mdd': round(is_pnl_mdd, 1),
            'is_wr': round(is_wr, 1),
            'is_trades': is_n_trades,
            'oos_pnls': [round(p, 2) for p in oos_pnls],
            'oos_total': round(oos_total, 2),
            'wf_verdict': verdict,
        }

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Scale':>6} | {'IS PnL':>8} | {'IS MDD':>7} | {'PnL/MDD':>8} | "
          f"{'IS WR':>6} | {'OOS Tot':>8} | {'F1':>7} {'F2':>7} {'F3':>7} | {'Verdict':>7}")
    print("-" * 95)

    best_pnl_mdd_scale = None
    best_pnl_mdd = 0
    best_oos_scale = None
    best_oos = -999999

    for scale_str, r in sorted(results.items(), key=lambda x: float(x[0])):
        oos = r['oos_pnls']
        print(f"{r['tp_scale']:>6.2f} | {r['is_pnl']:>+8.1f} | {r['is_mdd']:>7.2f} | "
              f"{r['is_pnl_mdd']:>8.1f} | {r['is_wr']:>6.1f} | {r['oos_total']:>+8.1f} | "
              f"{oos[0]:>+7.1f} {oos[1]:>+7.1f} {oos[2]:>+7.1f} | {r['wf_verdict']:>7}")

        if r['is_pnl_mdd'] > best_pnl_mdd:
            best_pnl_mdd = r['is_pnl_mdd']
            best_pnl_mdd_scale = r['tp_scale']
        if r['oos_total'] > best_oos and r['wf_verdict'] == 'PASS':
            best_oos = r['oos_total']
            best_oos_scale = r['tp_scale']

    print(f"\n  Best IS PnL/MDD: scale={best_pnl_mdd_scale}  ({best_pnl_mdd:.1f}x)")
    if best_oos_scale is not None:
        print(f"  Best OOS Total (PASS only): scale={best_oos_scale}  ({best_oos:+.1f}%)")
    else:
        print("  No scale factor achieved WF PASS on all folds.")

    # Save results
    output = {
        'study': 'tp_scale_factor_npos_sweep',
        'date': datetime.now().isoformat(),
        'data_file': DATA_FILE,
        'data_bars': n,
        'patterns': len(patterns),
        'scale_factors': SCALE_FACTORS,
        'n_folds': N_FOLDS,
        'config': {
            'n_slots': N_SLOTS,
            'direction_cap': DIRECTION_CAP,
            'leverage': LEVERAGE,
            'fee_pct': FEE_PCT,
            'max_bars': MAX_BARS,
            'cascade_tighten_pct': CASCADE_TIGHTEN_PCT,
            'agg_risk_counter': AGG_RISK_COUNTER,
            'agg_risk_with': AGG_RISK_WITH,
            'momentum_threshold': MOMENTUM_THRESHOLD,
            'momentum_lookback': MOMENTUM_LOOKBACK,
            'momentum_cooldown': MOMENTUM_COOLDOWN,
            'atr_period': ATR_PERIOD,
            'atr_window': ATR_WINDOW,
            'atr_clamp_lo': ATR_CLAMP_LO,
            'atr_clamp_hi': ATR_CLAMP_HI,
            'regime_mult': REGIME_MULT,
            'tp_min_floor': TP_MIN_FLOOR,
        },
        'results': results,
        'best_is_pnl_mdd': {'scale': best_pnl_mdd_scale, 'value': best_pnl_mdd},
        'best_oos_total': {'scale': best_oos_scale, 'value': best_oos} if best_oos_scale else None,
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
