#!/usr/bin/env python3
"""
N-Position Portfolio Simulator Study: Scanner 1-pos vs Production N-pos Comparison
==================================================================================
Scanner backtest uses 1-position additive PnL with no filters,
while production uses N=9 compound equity with dir_cap=7, regime×0.3,
agg risk cap 3/7, momentum guard, and loss burst brake.

Result: WF OOS WR 88.8% vs Live WR 52.7% — 36pp gap.

3 Phases:
  Phase 1: 1-pos vs N-pos IS comparison (130 patterns, same data)
  Phase 2: WF 3-fold OOS comparison — does N-pos OOS WR approach Live?
  Phase 3: Feature sensitivity — baseline N-pos vs +regime vs +agg_risk vs +momentum vs full

GO/STOP: N-pos WF 3/3 PASS + Live WR closer → GO → Scanner default transition.

Standard Research Protocol:
  Entry: signal bar + 1 open
  Exit: Intrabar high/low (distance-based, abs(tp - opens[bar]))
  Fee: 0.10% * LEVERAGE(3) = 0.30% per trade (capital-space)
  Slippage: 0.02%
  ATR-scaled TP/SL: ATR(14)/median(576), clamp [0.6, 1.7]
  WF: 3-fold expanding window, PASS = all folds OOS PnL > 0
"""

import json
import os
import sys
import time
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from scripts.production.pattern_5m.indicators import classify_candle

# Import scanner functions
from scripts.scanner.pattern_scanner import (
    load_and_classify, build_signal_index, compute_atr_ratio,
    compute_ema_slope, portfolio_npos, calc_stats_compound,
    portfolio_1pos, calc_stats, expanding_window_wf,
    scan_patterns_mae_mfe, find_neutral_window,
    LEVERAGE, FEE_PCT, DEFAULT_N_SLOTS, DEFAULT_DIRECTION_CAP,
    DEFAULT_REGIME_MULT, DEFAULT_AGG_RISK_COUNTER, DEFAULT_AGG_RISK_WITH,
    DEFAULT_MOMENTUM_LOOKBACK, DEFAULT_MOMENTUM_THRESHOLD,
    DEFAULT_MOMENTUM_COOLDOWN, TIMEOUT_BARS,
    DEFAULT_ATR_PERIOD, DEFAULT_ATR_WINDOW,
    DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI,
)

# ============================================================
# CONSTANTS
# ============================================================
DATA_FILE = os.path.join(PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
PATTERNS_FILE = os.path.join(PROJECT_ROOT, 'results', 'dynamic_patterns.json')
OUTPUT_FILE = os.path.join(PROJECT_ROOT, 'results', 'npos_portfolio_study.json')
BARS_PER_DAY = 288
WF_FOLDS = 3
EDGE_THRESHOLD = 18.0
MC_THRESHOLD = 0.01
MIN_TRADES = 25
NEUTRAL_TOL = 1.0


def load_patterns():
    """Load current dynamic patterns."""
    with open(PATTERNS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    patterns_tpsl = data.get('patterns_tpsl', {})
    long_pats = data.get('patterns', {}).get('long', [])
    short_pats = data.get('patterns', {}).get('short', [])
    return long_pats, short_pats, patterns_tpsl


def collect_signals(signal_index, long_pats, short_pats, patterns_tpsl,
                    start_bar=0, end_bar=None):
    """Collect signal tuples for portfolio simulation."""
    tuples = []
    default_tp, default_sl = 2.0, 3.0
    for pat in long_pats:
        tpsl = patterns_tpsl.get(pat, [default_tp, default_sl])
        for s in signal_index.get(pat, []):
            if start_bar <= s < (end_bar if end_bar else float('inf')):
                tuples.append((s, pat, 'LONG', tpsl[0], tpsl[1]))
    for pat in short_pats:
        tpsl = patterns_tpsl.get(pat, [default_tp, default_sl])
        for s in signal_index.get(pat, []):
            if start_bar <= s < (end_bar if end_bar else float('inf')):
                tuples.append((s, pat, 'SHORT', tpsl[0], tpsl[1]))
    return tuples


def run_1pos(signal_index, long_pats, short_pats, patterns_tpsl,
             opens, highs, lows, closes, n_bars, atr_ratio,
             clamp_lo, clamp_hi):
    """Run 1-position additive backtest (scanner default)."""
    from scripts.scanner.pattern_scanner import bt_signals_atr
    all_trades = []
    for pat in long_pats:
        tpsl = patterns_tpsl.get(pat, [2.0, 3.0])
        sigs = signal_index.get(pat, [])
        trades = bt_signals_atr(sigs, 'LONG', tpsl[0], tpsl[1],
                                opens, highs, lows, n_bars,
                                atr_ratio, clamp_lo, clamp_hi)
        all_trades.extend(trades)
    for pat in short_pats:
        tpsl = patterns_tpsl.get(pat, [2.0, 3.0])
        sigs = signal_index.get(pat, [])
        trades = bt_signals_atr(sigs, 'SHORT', tpsl[0], tpsl[1],
                                opens, highs, lows, n_bars,
                                atr_ratio, clamp_lo, clamp_hi)
        all_trades.extend(trades)
    portfolio_trades = portfolio_1pos(all_trades)
    stats = calc_stats(portfolio_trades)
    return stats


def run_npos(signal_tuples, opens, highs, lows, closes, n_bars,
             atr_ratio, ema_slope, start_bar, end_bar, **kwargs):
    """Run N-position compound backtest."""
    trades, raw_stats = portfolio_npos(
        signal_tuples, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, start_bar, end_bar, **kwargs,
    )
    stats = calc_stats_compound(trades)
    stats.update(raw_stats)
    return stats


# ============================================================
# NPOS KWARGS BUILDER
# ============================================================
def build_npos_kwargs(regime=True, agg_risk=True, momentum=True,
                      regime_mult=DEFAULT_REGIME_MULT,
                      n_slots=DEFAULT_N_SLOTS,
                      direction_cap=DEFAULT_DIRECTION_CAP):
    """Build npos kwargs with optional feature disabling."""
    return {
        'n_slots': n_slots,
        'direction_cap': direction_cap,
        'regime_mult': regime_mult if regime else 1.0,
        'agg_risk_counter': DEFAULT_AGG_RISK_COUNTER if agg_risk else 999.0,
        'agg_risk_with': DEFAULT_AGG_RISK_WITH if agg_risk else 999.0,
        'momentum_lookback': DEFAULT_MOMENTUM_LOOKBACK if momentum else 0,
        'momentum_threshold': DEFAULT_MOMENTUM_THRESHOLD if momentum else 999.0,
        'momentum_cooldown': DEFAULT_MOMENTUM_COOLDOWN if momentum else 0,
        'clamp_lo': DEFAULT_ATR_CLAMP_LO,
        'clamp_hi': DEFAULT_ATR_CLAMP_HI,
        'timeout_bars': TIMEOUT_BARS,
    }


# ============================================================
# PHASE 1: 1-pos vs N-pos IS Comparison
# ============================================================
def phase1_is_comparison(signal_index, long_pats, short_pats, patterns_tpsl,
                         opens, highs, lows, closes, n_bars, atr_ratio):
    """Compare 1-pos (scanner) vs N-pos (production-aligned) on full IS data."""
    print("\n" + "=" * 70)
    print("PHASE 1: 1-pos vs N-pos IS Comparison")
    print("=" * 70)

    # 1-pos (scanner default)
    stats_1pos = run_1pos(signal_index, long_pats, short_pats, patterns_tpsl,
                          opens, highs, lows, closes, n_bars, atr_ratio,
                          DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI)
    print(f"\n1-pos (scanner): {stats_1pos['trades']} trades, "
          f"WR {stats_1pos['wr']}%, PnL {stats_1pos['pnl']}%, "
          f"MDD {stats_1pos['mdd']}%, PF {stats_1pos['pf']}")

    # N-pos (production-aligned, all features)
    ema_slope = compute_ema_slope(closes)
    signal_tuples = collect_signals(signal_index, long_pats, short_pats,
                                    patterns_tpsl, 0, n_bars)
    kwargs = build_npos_kwargs(regime=True, agg_risk=True, momentum=True)
    stats_npos = run_npos(signal_tuples, opens, highs, lows, closes, n_bars,
                          atr_ratio, ema_slope, 0, n_bars, **kwargs)
    pnl_mdd = round(stats_npos['pnl'] / stats_npos['mdd'], 1) if stats_npos['mdd'] > 0 else 0
    print(f"\nN-pos (production): {stats_npos['trades']} trades, "
          f"WR {stats_npos['wr']}%, PnL {stats_npos['pnl']}%, "
          f"MDD {stats_npos['mdd']}%, PnL/MDD {pnl_mdd}x")
    if stats_npos.get('blocked'):
        blk = stats_npos['blocked']
        print(f"  Blocked: dir_cap={blk.get('dir_cap',0)}, "
              f"momentum={blk.get('momentum',0)}, "
              f"agg_risk={blk.get('agg_risk',0)}, "
              f"dup_pat={blk.get('dup_pat',0)}, "
              f"max_pos={blk.get('max_pos',0)}")

    print(f"\n  WR delta: {stats_npos['wr'] - stats_1pos['wr']:+.1f}pp")
    print(f"  PnL delta: {stats_npos['pnl'] - stats_1pos['pnl']:+.1f}%")
    print(f"  Trades delta: {stats_npos['trades'] - stats_1pos['trades']:+d}")

    return {'1pos': stats_1pos, 'npos': stats_npos}


# ============================================================
# PHASE 2: WF 3-fold OOS Comparison
# ============================================================
def phase2_wf_comparison(signal_index, opens, highs, lows, closes, n_bars,
                         atr_ratio):
    """Compare WF OOS between 1-pos and N-pos modes."""
    print("\n" + "=" * 70)
    print("PHASE 2: WF 3-fold OOS Comparison (1-pos vs N-pos)")
    print("=" * 70)

    common_args = dict(
        n_folds=WF_FOLDS,
        mode='mae_mfe',
        min_trades=MIN_TRADES,
        edge_threshold=EDGE_THRESHOLD,
        mc_threshold=MC_THRESHOLD,
        max_baseline_wr=70.0,
        closes=closes,
        atr_period=DEFAULT_ATR_PERIOD,
        atr_window=DEFAULT_ATR_WINDOW,
        clamp_lo=DEFAULT_ATR_CLAMP_LO,
        clamp_hi=DEFAULT_ATR_CLAMP_HI,
        use_atr=True,
    )

    # 1-pos WF (scanner default)
    print("\n--- 1-pos WF ---")
    t0 = time.time()
    wf_1pos = expanding_window_wf(
        signal_index, opens, highs, lows, n_bars,
        use_npos=False, **common_args,
    )
    print(f"  Time: {time.time()-t0:.1f}s")
    print(f"  {wf_1pos['positive_folds']}/{WF_FOLDS} positive, "
          f"OOS PnL {wf_1pos['total_oos_pnl']}%")
    for fold in wf_1pos.get('folds', []):
        print(f"  Fold {fold['fold']}: OOS WR {fold.get('oos_wr','?')}%, "
              f"PnL {fold.get('oos_pnl','?')}%, "
              f"Trades {fold.get('oos_trades','?')}")

    # N-pos WF (production-aligned)
    print("\n--- N-pos WF ---")
    npos_kwargs = build_npos_kwargs(regime=True, agg_risk=True, momentum=True)
    t0 = time.time()
    wf_npos = expanding_window_wf(
        signal_index, opens, highs, lows, n_bars,
        use_npos=True, npos_kwargs=npos_kwargs, **common_args,
    )
    print(f"  Time: {time.time()-t0:.1f}s")
    print(f"  {wf_npos['positive_folds']}/{WF_FOLDS} positive, "
          f"OOS PnL {wf_npos['total_oos_pnl']}%")
    for fold in wf_npos.get('folds', []):
        print(f"  Fold {fold['fold']}: OOS WR {fold.get('oos_wr','?')}%, "
              f"PnL {fold.get('oos_pnl','?')}%, "
              f"Trades {fold.get('oos_trades','?')}, "
              f"MDD {fold.get('oos_mdd','?')}%"
              f"{', PnL/MDD '+str(fold.get('oos_pnl_mdd','?'))+'x' if fold.get('oos_pnl_mdd') else ''}")
        if fold.get('oos_blocked'):
            blk = fold['oos_blocked']
            print(f"    Blocked: dir_cap={blk.get('dir_cap',0)}, "
                  f"momentum={blk.get('momentum',0)}, "
                  f"agg_risk={blk.get('agg_risk',0)}")

    # Comparison
    print("\n--- Comparison ---")
    avg_wr_1pos = np.mean([f.get('oos_wr', 0) for f in wf_1pos.get('folds', [])])
    avg_wr_npos = np.mean([f.get('oos_wr', 0) for f in wf_npos.get('folds', [])])
    print(f"  Avg OOS WR: 1-pos {avg_wr_1pos:.1f}% vs N-pos {avg_wr_npos:.1f}% "
          f"(delta {avg_wr_npos - avg_wr_1pos:+.1f}pp)")
    print(f"  Total OOS PnL: 1-pos {wf_1pos['total_oos_pnl']}% vs "
          f"N-pos {wf_npos['total_oos_pnl']}%")
    print(f"  Live WR: 52.7% (target)")
    print(f"  N-pos WR gap to live: {avg_wr_npos - 52.7:+.1f}pp "
          f"(vs 1-pos gap: {avg_wr_1pos - 52.7:+.1f}pp)")

    return {'wf_1pos': wf_1pos, 'wf_npos': wf_npos,
            'avg_wr_1pos': round(avg_wr_1pos, 1),
            'avg_wr_npos': round(avg_wr_npos, 1)}


# ============================================================
# PHASE 3: Feature Sensitivity
# ============================================================
def phase3_feature_sensitivity(signal_index, long_pats, short_pats,
                               patterns_tpsl, opens, highs, lows, closes,
                               n_bars, atr_ratio):
    """Test individual feature contributions to N-pos results."""
    print("\n" + "=" * 70)
    print("PHASE 3: Feature Sensitivity (IS)")
    print("=" * 70)

    ema_slope = compute_ema_slope(closes)
    signal_tuples = collect_signals(signal_index, long_pats, short_pats,
                                    patterns_tpsl, 0, n_bars)

    scenarios = {
        'baseline_npos': build_npos_kwargs(regime=False, agg_risk=False, momentum=False),
        '+regime_0.3': build_npos_kwargs(regime=True, agg_risk=False, momentum=False),
        '+agg_risk_3_7': build_npos_kwargs(regime=False, agg_risk=True, momentum=False),
        '+momentum': build_npos_kwargs(regime=False, agg_risk=False, momentum=True),
        '+regime+agg': build_npos_kwargs(regime=True, agg_risk=True, momentum=False),
        '+regime+momentum': build_npos_kwargs(regime=True, agg_risk=False, momentum=True),
        'full (production)': build_npos_kwargs(regime=True, agg_risk=True, momentum=True),
    }

    results = {}
    print(f"\n{'Scenario':<25} {'Trades':>7} {'WR%':>7} {'PnL%':>10} {'MDD%':>8} "
          f"{'PnL/MDD':>8} {'CorrLoss':>10}")
    print("-" * 80)

    for name, kwargs in scenarios.items():
        stats = run_npos(signal_tuples, opens, highs, lows, closes, n_bars,
                         atr_ratio, ema_slope, 0, n_bars, **kwargs)
        pnl_mdd = round(stats['pnl'] / stats['mdd'], 1) if stats['mdd'] > 0 else 0
        corr_loss = stats.get('max_corr_loss', 0)
        print(f"{name:<25} {stats['trades']:>7} {stats['wr']:>6.1f}% "
              f"{stats['pnl']:>9.1f}% {stats['mdd']:>7.1f}% "
              f"{pnl_mdd:>7.1f}x {corr_loss:>9.1f}%")
        results[name] = {**stats, 'pnl_mdd': pnl_mdd}

    return results


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("N-Position Portfolio Simulator Study")
    print("Scanner 1-pos vs Production N-pos Comparison")
    print("=" * 70)

    t_start = time.time()

    # Load data
    print("\nLoading data...")
    df = load_and_classify(DATA_FILE)

    # Apply neutral window
    closes_all = df['close'].values
    nw_result = find_neutral_window(closes_all, tol_pct=NEUTRAL_TOL)
    if nw_result is not None:
        ns, ne = nw_result
        drift = (closes_all[ne] / closes_all[ns] - 1) * 100
        neutral_days = (ne - ns) / BARS_PER_DAY
        df = df.iloc[ns:ne + 1].reset_index(drop=True)
        print(f"Neutral window: {neutral_days:.0f}d (bars {ns}-{ne}), "
              f"drift {drift:+.2f}%")
    else:
        print("No neutral window found, using full data")

    types = df['candle_type'].tolist()
    n = len(types)
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values

    signal_index = build_signal_index(types, n)
    atr_ratio = compute_atr_ratio(highs, lows, closes,
                                   atr_period=DEFAULT_ATR_PERIOD,
                                   window=DEFAULT_ATR_WINDOW)

    # Load current patterns
    long_pats, short_pats, patterns_tpsl = load_patterns()
    n_total = len(long_pats) + len(short_pats)
    print(f"Patterns: {len(long_pats)}L + {len(short_pats)}S = {n_total}")
    print(f"Data: {n} bars ({n/BARS_PER_DAY:.0f} days)")

    # Phase 1
    p1 = phase1_is_comparison(signal_index, long_pats, short_pats, patterns_tpsl,
                              opens, highs, lows, closes, n, atr_ratio)

    # Phase 2
    p2 = phase2_wf_comparison(signal_index, opens, highs, lows, closes, n, atr_ratio)

    # Phase 3
    p3 = phase3_feature_sensitivity(signal_index, long_pats, short_pats, patterns_tpsl,
                                    opens, highs, lows, closes, n, atr_ratio)

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    npos_wr = p2['avg_wr_npos']
    onepos_wr = p2['avg_wr_1pos']
    live_wr = 52.7

    npos_wf_pass = p2['wf_npos']['positive_folds'] == WF_FOLDS
    onepos_wf_pass = p2['wf_1pos']['positive_folds'] == WF_FOLDS

    gap_npos = abs(npos_wr - live_wr)
    gap_1pos = abs(onepos_wr - live_wr)
    gap_improved = gap_npos < gap_1pos

    print(f"\n  1-pos WF: {'PASS' if onepos_wf_pass else 'FAIL'} "
          f"({p2['wf_1pos']['positive_folds']}/{WF_FOLDS}), "
          f"Avg OOS WR {onepos_wr}%")
    print(f"  N-pos WF: {'PASS' if npos_wf_pass else 'FAIL'} "
          f"({p2['wf_npos']['positive_folds']}/{WF_FOLDS}), "
          f"Avg OOS WR {npos_wr}%")
    print(f"  Live WR: {live_wr}%")
    print(f"  Gap to live: 1-pos {gap_1pos:.1f}pp vs N-pos {gap_npos:.1f}pp")
    print(f"  Gap improved: {'YES' if gap_improved else 'NO'} "
          f"({gap_1pos:.1f}pp → {gap_npos:.1f}pp)")

    if npos_wf_pass and gap_improved:
        print("\n  >>> GO: N-pos WF PASS + gap improved → consider scanner default transition")
    else:
        print(f"\n  >>> STOP: N-pos WF {'FAIL' if not npos_wf_pass else 'PASS'}, "
              f"gap {'not improved' if not gap_improved else 'improved'}")

    # Save results
    elapsed = round(time.time() - t_start, 1)
    output = {
        'study': 'npos_portfolio_study',
        'timestamp': pd.Timestamp.now().isoformat(timespec='seconds'),
        'data_bars': n,
        'patterns': n_total,
        'phase1_is': p1,
        'phase2_wf': {
            'wf_1pos_summary': {
                'positive_folds': p2['wf_1pos']['positive_folds'],
                'total_oos_pnl': p2['wf_1pos']['total_oos_pnl'],
                'avg_oos_wr': p2['avg_wr_1pos'],
            },
            'wf_npos_summary': {
                'positive_folds': p2['wf_npos']['positive_folds'],
                'total_oos_pnl': p2['wf_npos']['total_oos_pnl'],
                'avg_oos_wr': p2['avg_wr_npos'],
            },
        },
        'phase3_sensitivity': {k: {kk: vv for kk, vv in v.items()
                                    if kk != 'blocked'}
                                for k, v in p3.items()},
        'verdict': {
            'npos_wf_pass': npos_wf_pass,
            'gap_improved': gap_improved,
            'gap_1pos_pp': round(gap_1pos, 1),
            'gap_npos_pp': round(gap_npos, 1),
            'decision': 'GO' if (npos_wf_pass and gap_improved) else 'STOP',
        },
        'elapsed_sec': elapsed,
    }

    class _NE(json.JSONEncoder):
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

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, cls=_NE)
    print(f"\nResults saved to {OUTPUT_FILE}")
    print(f"Total time: {elapsed}s")


if __name__ == '__main__':
    main()
