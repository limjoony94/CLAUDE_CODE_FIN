#!/usr/bin/env python3
"""
TP Scale Factor Sweep — IS + WF OOS Analysis

Background:
  Current bot uses TP×0.5 (v1.57.0). Live WR margin is thin (+2.1pp over BE WR).
  This study sweeps TP scale factors 0.50-1.00 to find optimal factor
  considering both IS performance and WF OOS robustness.

Sweep: [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]

For each factor:
  - IS full-period N-pos backtest (PnL, MDD, WR, PnL/MDD, trades/day)
  - 3-fold Expanding Window WF (OOS PnL per fold, total, PASS/FAIL)
  - BE WR = avg_loss / (avg_win + avg_loss), WR margin = actual WR - BE WR

Standard Research Protocol: compound, 0.10% fee, 0.02% slippage, 3x leverage.
"""

import json
import numpy as np
from pathlib import Path
import sys
import time
import warnings
warnings.filterwarnings('ignore')

start_time = time.time()

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scanner"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "production" / "pattern_5m"))

from pattern_scanner import (
    load_and_classify, build_signal_index, find_neutral_window,
    portfolio_npos, calc_stats_compound,
    compute_atr_ratio, compute_ema_slope,
    LEVERAGE, FEE_PCT,
    DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI,
    DEFAULT_N_SLOTS, DEFAULT_DIRECTION_CAP, TIMEOUT_BARS,
    DEFAULT_AGG_RISK_COUNTER, DEFAULT_AGG_RISK_WITH,
    DEFAULT_MOMENTUM_LOOKBACK, DEFAULT_MOMENTUM_THRESHOLD,
    DEFAULT_MOMENTUM_COOLDOWN, DEFAULT_CASCADE_TIGHTEN_PCT,
    DEFAULT_REGIME_MULT,
)

DATA_FILE = "data/btc_5m_270days_reclassified.csv"
PATTERNS_FILE = "results/dynamic_patterns.json"
OUTPUT_FILE = "results/tp_factor_sweep.json"

TP_FACTORS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
BARS_PER_DAY = 288

NPOS_DEFAULTS = dict(
    n_slots=DEFAULT_N_SLOTS, direction_cap=DEFAULT_DIRECTION_CAP,
    regime_mult=DEFAULT_REGIME_MULT,
    agg_risk_counter=DEFAULT_AGG_RISK_COUNTER, agg_risk_with=DEFAULT_AGG_RISK_WITH,
    momentum_lookback=DEFAULT_MOMENTUM_LOOKBACK, momentum_threshold=DEFAULT_MOMENTUM_THRESHOLD,
    momentum_cooldown=DEFAULT_MOMENTUM_COOLDOWN,
    clamp_lo=DEFAULT_ATR_CLAMP_LO, clamp_hi=DEFAULT_ATR_CLAMP_HI,
    timeout_bars=TIMEOUT_BARS, cascade_tighten_pct=DEFAULT_CASCADE_TIGHTEN_PCT,
)


def load_patterns(filepath=PATTERNS_FILE):
    with open(filepath) as f:
        data = json.load(f)
    details = data.get('pattern_details') or {}
    result = {}
    for k, v in details.items():
        result[k] = {
            'pattern': v['pattern'], 'direction': v['direction'],
            'tp': v['tp'], 'sl': v['sl'],
        }
    return result


def apply_tp_factor(patterns, tp_factor):
    """Apply TP scale factor with min floor 0.3%."""
    result = {}
    for k, v in patterns.items():
        result[k] = {
            'pattern': v['pattern'], 'direction': v['direction'],
            'tp': round(max(0.3, v['tp'] * tp_factor), 3),
            'sl': v['sl'],  # SL unchanged
        }
    return result


def build_signal_tuples(patterns, sig_idx):
    tuples = []
    for k, v in patterns.items():
        pat_name = v.get('pattern') or k.rsplit('_', 1)[0]
        if pat_name in sig_idx:
            for bar in sig_idx[pat_name]:
                tuples.append((bar, k, v['direction'], v['tp'], v['sl']))
    return tuples


def run_npos(signal_tuples, opens, highs, lows, closes, n_bars,
             atr_ratio, ema_slope, start_bar, end_bar, **extra):
    kwargs = {**NPOS_DEFAULTS}
    kwargs.update(extra)
    trades, raw = portfolio_npos(
        signal_tuples, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, start_bar, end_bar, **kwargs
    )
    stats = calc_stats_compound(trades)
    if raw.get('mdd_mtm', 0) > 0:
        stats['mdd'] = raw['mdd_mtm']
        stats['pnl_mdd'] = round(stats['pnl'] / stats['mdd'], 2) if stats['mdd'] > 0 else 0
    stats.update({k: v for k, v in raw.items() if k not in stats})
    return trades, stats


def compute_trade_metrics(trades):
    """Compute avg win, avg loss, BE WR from trade list."""
    wins = [t['pnl_slot'] for t in trades if t['pnl_slot'] > 0]
    losses = [abs(t['pnl_slot']) for t in trades if t['pnl_slot'] <= 0]

    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0

    # BE WR = avg_loss / (avg_win + avg_loss) — breakeven win rate
    if avg_win + avg_loss > 0:
        be_wr = avg_loss / (avg_win + avg_loss) * 100
    else:
        be_wr = 50.0

    actual_wr = len(wins) / len(trades) * 100 if trades else 0
    wr_margin = actual_wr - be_wr

    return {
        'avg_win': round(avg_win, 3),
        'avg_loss': round(avg_loss, 3),
        'be_wr': round(be_wr, 1),
        'actual_wr': round(actual_wr, 1),
        'wr_margin': round(wr_margin, 1),
        'n_wins': len(wins),
        'n_losses': len(losses),
    }


def run_wf(signal_tuples, opens, highs, lows, closes, n_bars,
           atr_ratio, ema_slope, ns, ne, n_folds=3, **extra):
    total = ne - ns
    min_train = total // 3
    fold_size = total // (n_folds + 1)
    results = []
    for fold in range(n_folds):
        train_end = ns + min_train + fold_size * fold
        test_start = train_end
        test_end = min(train_end + fold_size, ne)
        if test_start >= ne or test_end <= test_start:
            continue
        trades, stats = run_npos(signal_tuples, opens, highs, lows, closes, n_bars,
                                 atr_ratio, ema_slope, test_start, test_end, **extra)
        metrics = compute_trade_metrics(trades)
        results.append({
            'fold': fold + 1, 'pnl': stats.get('pnl', 0),
            'wr': stats.get('wr', 0), 'trades': stats.get('trades', 0),
            'mdd': stats.get('mdd', 0),
            'be_wr': metrics['be_wr'], 'wr_margin': metrics['wr_margin'],
        })
    oos_pnl = sum(r['pnl'] for r in results)
    all_pass = all(r['pnl'] > 0 for r in results) if results else False
    return results, oos_pnl, all_pass


# ─── Load data ───
print("=" * 70)
print("TP SCALE FACTOR SWEEP — IS + WF OOS ANALYSIS")
print("=" * 70)

df = load_and_classify(DATA_FILE)
opens = df['open'].values.astype(np.float64)
highs = df['high'].values.astype(np.float64)
lows = df['low'].values.astype(np.float64)
closes = df['close'].values.astype(np.float64)
n_bars = len(df)
type_codes = df['candle_type'].values

atr_ratio = compute_atr_ratio(highs, lows, closes)
ema_slope = compute_ema_slope(closes)

signal_index = build_signal_index(type_codes, n_bars)
ns, ne = find_neutral_window(closes)
n_days = (ne - ns) / BARS_PER_DAY
print(f"Data: {n_bars} bars, neutral: {ns}-{ne} ({n_days:.0f}d)")

base_patterns = load_patterns()
print(f"Patterns: {len(base_patterns)}")

# ─── Sweep ───
print(f"\nSweeping TP factors: {TP_FACTORS}")
print("-" * 70)

sweep_results = []

for factor in TP_FACTORS:
    t0 = time.time()

    # Apply TP factor
    scaled_patterns = apply_tp_factor(base_patterns, factor)
    signal_tuples = build_signal_tuples(scaled_patterns, signal_index)

    # IS full period
    is_trades, is_stats = run_npos(
        signal_tuples, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, ns, ne
    )
    is_metrics = compute_trade_metrics(is_trades)
    is_trades_per_day = is_stats.get('trades', 0) / n_days if n_days > 0 else 0

    # TP range for this factor
    tp_vals = [v['tp'] for v in scaled_patterns.values()]
    tp_min, tp_max = min(tp_vals), max(tp_vals)

    # WF OOS
    wf_folds, oos_pnl, wf_pass = run_wf(
        signal_tuples, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, ns, ne
    )

    elapsed = time.time() - t0

    entry = {
        'tp_factor': factor,
        'is': {
            'pnl': is_stats.get('pnl', 0),
            'mdd': is_stats.get('mdd', 0),
            'pnl_mdd': is_stats.get('pnl_mdd', 0),
            'wr': is_stats.get('wr', 0),
            'trades': is_stats.get('trades', 0),
            'trades_per_day': round(is_trades_per_day, 2),
            'avg_win': is_metrics['avg_win'],
            'avg_loss': is_metrics['avg_loss'],
            'be_wr': is_metrics['be_wr'],
            'wr_margin': is_metrics['wr_margin'],
        },
        'tp_range': f"{tp_min:.2f}-{tp_max:.2f}%",
        'wf': {
            'folds': wf_folds,
            'oos_pnl': round(oos_pnl, 1),
            'all_pass': wf_pass,
        },
    }
    sweep_results.append(entry)

    # Console progress
    wf_str = ', '.join(f"F{r['fold']}:{r['pnl']:+.1f}%" for r in wf_folds)
    pass_str = "PASS" if wf_pass else "FAIL"
    print(f"TP×{factor:.2f} | IS: PnL {is_stats.get('pnl', 0):+.1f}%, "
          f"MDD {is_stats.get('mdd', 0):.1f}%, "
          f"PnL/MDD {is_stats.get('pnl_mdd', 0):.1f}x, "
          f"WR {is_stats.get('wr', 0):.1f}%, "
          f"BE_WR {is_metrics['be_wr']:.1f}%, "
          f"margin {is_metrics['wr_margin']:+.1f}pp | "
          f"WF: {wf_str} = {oos_pnl:+.1f}% [{pass_str}] | "
          f"{elapsed:.1f}s")

# ─── Summary table ───
print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)
print(f"{'Factor':>7} | {'IS PnL':>9} | {'IS MDD':>7} | {'PnL/MDD':>8} | "
      f"{'IS WR':>6} | {'BE WR':>6} | {'Margin':>7} | {'T/day':>6} | "
      f"{'OOS PnL':>9} | {'WF':>5} | {'TP range':>14}")
print("-" * 110)

for r in sweep_results:
    i = r['is']
    w = r['wf']
    print(f"  {r['tp_factor']:.2f}  | {i['pnl']:>+8.1f}% | {i['mdd']:>6.1f}% | "
          f"{i['pnl_mdd']:>7.1f}x | {i['wr']:>5.1f}% | {i['be_wr']:>5.1f}% | "
          f"{i['wr_margin']:>+6.1f}pp | {i['trades_per_day']:>5.1f} | "
          f"{w['oos_pnl']:>+8.1f}% | {'PASS' if w['all_pass'] else 'FAIL':>5} | "
          f"{r['tp_range']:>14}")

# ─── Find optimal ───
# Primary criterion: WF all_pass. Among passing, maximize PnL/MDD.
passing = [r for r in sweep_results if r['wf']['all_pass']]
if passing:
    best = max(passing, key=lambda x: x['is']['pnl_mdd'])
    print(f"\nBest (WF PASS, max PnL/MDD): TP×{best['tp_factor']:.2f}")
    print(f"  IS: PnL {best['is']['pnl']:+.1f}%, MDD {best['is']['mdd']:.1f}%, "
          f"PnL/MDD {best['is']['pnl_mdd']:.1f}x, WR {best['is']['wr']:.1f}%")
    print(f"  WR margin: {best['is']['wr_margin']:+.1f}pp (WR {best['is']['wr']:.1f}% - BE {best['is']['be_wr']:.1f}%)")
    print(f"  OOS: {best['wf']['oos_pnl']:+.1f}%")

    # Compare current (0.5) vs best
    current = next((r for r in sweep_results if r['tp_factor'] == 0.50), None)
    if current and best['tp_factor'] != 0.50:
        print(f"\n  vs Current (TP×0.50):")
        print(f"    IS PnL/MDD: {current['is']['pnl_mdd']:.1f}x -> {best['is']['pnl_mdd']:.1f}x "
              f"({best['is']['pnl_mdd'] - current['is']['pnl_mdd']:+.1f}x)")
        print(f"    OOS PnL:    {current['wf']['oos_pnl']:+.1f}% -> {best['wf']['oos_pnl']:+.1f}% "
              f"({best['wf']['oos_pnl'] - current['wf']['oos_pnl']:+.1f}%)")
        print(f"    WR margin:  {current['is']['wr_margin']:+.1f}pp -> {best['is']['wr_margin']:+.1f}pp "
              f"({best['is']['wr_margin'] - current['is']['wr_margin']:+.1f}pp)")
else:
    print("\nNo TP factor achieved WF 3/3 PASS.")
    best_oos = max(sweep_results, key=lambda x: x['wf']['oos_pnl'])
    print(f"Best OOS: TP×{best_oos['tp_factor']:.2f} = {best_oos['wf']['oos_pnl']:+.1f}%")

# ─── WR margin analysis ───
print("\n" + "=" * 70)
print("WR MARGIN ANALYSIS (Live safety indicator)")
print("=" * 70)
print("Higher WR margin = more buffer against live WR degradation")
print(f"Current live margin: +2.1pp (WR 64.8% - BE 62.6%)")
print()
for r in sweep_results:
    i = r['is']
    bar = '#' * max(0, int(i['wr_margin']))
    print(f"  TP×{r['tp_factor']:.2f}: margin {i['wr_margin']:>+6.1f}pp | "
          f"WR {i['wr']:>5.1f}% - BE {i['be_wr']:>5.1f}% | {bar}")

# ─── R:R analysis ───
print("\n" + "=" * 70)
print("RISK-REWARD ANALYSIS")
print("=" * 70)
for r in sweep_results:
    i = r['is']
    rr = i['avg_win'] / i['avg_loss'] if i['avg_loss'] > 0 else 0
    tp_needed = 1 / rr if rr > 0 else float('inf')
    print(f"  TP×{r['tp_factor']:.2f}: avg_win {i['avg_win']:>6.2f}% / avg_loss {i['avg_loss']:>6.2f}% = "
          f"R:R {rr:.3f} | {tp_needed:.1f} TPs to cover 1 SL")

# ─── Save results ───
output = {
    "study": "tp_factor_sweep",
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "data_bars": n_bars,
    "neutral_window": [ns, ne],
    "neutral_days": round(n_days, 1),
    "patterns": len(base_patterns),
    "tp_factors_tested": TP_FACTORS,
    "config": {
        "n_slots": DEFAULT_N_SLOTS,
        "direction_cap": DEFAULT_DIRECTION_CAP,
        "cascade_tighten_pct": DEFAULT_CASCADE_TIGHTEN_PCT,
        "agg_risk_counter": DEFAULT_AGG_RISK_COUNTER,
        "agg_risk_with": DEFAULT_AGG_RISK_WITH,
        "timeout_bars": TIMEOUT_BARS,
        "regime_mult": DEFAULT_REGIME_MULT,
        "momentum_threshold": DEFAULT_MOMENTUM_THRESHOLD,
    },
    "sweep_results": sweep_results,
    "elapsed_seconds": round(time.time() - start_time, 1),
}

output_path = Path(OUTPUT_FILE)
output_path.parent.mkdir(exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to {OUTPUT_FILE}")
print(f"Total elapsed: {time.time() - start_time:.1f}s")
