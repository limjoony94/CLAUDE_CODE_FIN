#!/usr/bin/env python3
"""
TP Factor Deep Study: Fine-grained sweep + WR degradation robustness + MC discrimination

Phase 1: Fine-grained sweep (0.68-0.92, step 0.02) — IS + WF 3-fold OOS
Phase 2: WR degradation robustness — expected PnL at degraded WR levels
Phase 3: MC discrimination test — sign randomization on top 3 factors

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
OUTPUT_FILE = "results/tp_factor_deep_study.json"

# Phase 1: fine-grained sweep
TP_FACTORS = [round(0.68 + i * 0.02, 2) for i in range(13)]  # 0.68..0.92
BARS_PER_DAY = 288

# Phase 2: WR degradation levels (pp below IS)
WR_DEGRADATION_PP = [0, -5, -10, -15, -20, -25]

# Phase 3: MC seeds
MC_SEEDS = [42, 123, 7]

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
    result = {}
    for k, v in patterns.items():
        result[k] = {
            'pattern': v['pattern'], 'direction': v['direction'],
            'tp': round(max(0.3, v['tp'] * tp_factor), 3),
            'sl': v['sl'],
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
    """Compute avg win, avg loss, BE WR, R:R from trade list."""
    wins = [t['pnl_slot'] for t in trades if t['pnl_slot'] > 0]
    losses = [abs(t['pnl_slot']) for t in trades if t['pnl_slot'] <= 0]

    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0

    if avg_win + avg_loss > 0:
        be_wr = avg_loss / (avg_win + avg_loss) * 100
    else:
        be_wr = 50.0

    actual_wr = len(wins) / len(trades) * 100 if trades else 0
    wr_margin = actual_wr - be_wr
    rr = avg_win / avg_loss if avg_loss > 0 else 0

    return {
        'avg_win': round(avg_win, 4),
        'avg_loss': round(avg_loss, 4),
        'be_wr': round(be_wr, 2),
        'actual_wr': round(actual_wr, 2),
        'wr_margin': round(wr_margin, 2),
        'rr': round(rr, 4),
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


def expected_pnl_at_wr(wr_pct, avg_win, avg_loss, fee_per_trade, trades_count):
    """
    Expected total PnL if win rate were wr_pct%.
    fee_per_trade = FEE_PCT * LEVERAGE * 2 (entry+exit) applied as % of notional.
    Uses simple expectancy * trades (approximation for comparison).
    """
    wr = wr_pct / 100.0
    # Per-trade expectancy (% of slot, already leveraged from avg_win/avg_loss)
    exp = wr * avg_win - (1 - wr) * avg_loss
    return exp * trades_count


def wr_for_breakeven(avg_win, avg_loss):
    """WR at which expected PnL = 0 (already includes fees in avg_win/avg_loss)."""
    if avg_win + avg_loss == 0:
        return 50.0
    return avg_loss / (avg_win + avg_loss) * 100


# ─── Load data ───
print("=" * 80)
print("TP FACTOR DEEP STUDY — Fine Sweep + WR Degradation + MC Discrimination")
print("=" * 80)

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

# ═══════════════════════════════════════════════════════════════════════
# PHASE 1: Fine-grained sweep
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 80}")
print("PHASE 1: Fine-Grained Sweep (0.68-0.92, step 0.02)")
print(f"{'=' * 80}")

phase1_results = []

for factor in TP_FACTORS:
    t0 = time.time()
    scaled_patterns = apply_tp_factor(base_patterns, factor)
    signal_tuples = build_signal_tuples(scaled_patterns, signal_index)

    # IS full period
    is_trades, is_stats = run_npos(
        signal_tuples, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, ns, ne
    )
    is_metrics = compute_trade_metrics(is_trades)
    is_tpd = is_stats.get('trades', 0) / n_days if n_days > 0 else 0

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
            'pnl': round(is_stats.get('pnl', 0), 2),
            'mdd': round(is_stats.get('mdd', 0), 3),
            'pnl_mdd': round(is_stats.get('pnl_mdd', 0), 2),
            'wr': round(is_stats.get('wr', 0), 2),
            'trades': is_stats.get('trades', 0),
            'trades_per_day': round(is_tpd, 2),
            'avg_win': is_metrics['avg_win'],
            'avg_loss': is_metrics['avg_loss'],
            'be_wr': is_metrics['be_wr'],
            'wr_margin': is_metrics['wr_margin'],
            'rr': is_metrics['rr'],
        },
        'tp_range': f"{tp_min:.3f}-{tp_max:.3f}%",
        'wf': {
            'folds': wf_folds,
            'oos_pnl': round(oos_pnl, 2),
            'all_pass': wf_pass,
        },
        'is_trades_raw': is_trades,  # keep for Phase 2
    }
    phase1_results.append(entry)

    wf_str = ', '.join(f"F{r['fold']}:{r['pnl']:+.1f}%" for r in wf_folds)
    pass_str = "PASS" if wf_pass else "FAIL"
    print(f"TP x{factor:.2f} | IS: PnL {is_stats.get('pnl', 0):+.1f}%, "
          f"MDD {is_stats.get('mdd', 0):.2f}%, "
          f"PnL/MDD {is_stats.get('pnl_mdd', 0):.1f}x, "
          f"WR {is_stats.get('wr', 0):.1f}%, "
          f"R:R {is_metrics['rr']:.3f}, "
          f"margin {is_metrics['wr_margin']:+.1f}pp | "
          f"WF: {wf_str} = {oos_pnl:+.1f}% [{pass_str}] | {elapsed:.1f}s")

# Phase 1 summary table
print(f"\n{'=' * 80}")
print("PHASE 1 SUMMARY TABLE")
print(f"{'=' * 80}")
print(f"{'Factor':>7} | {'IS PnL':>9} | {'IS MDD':>7} | {'PnL/MDD':>8} | "
      f"{'IS WR':>6} | {'R:R':>6} | {'Margin':>7} | {'T/day':>6} | "
      f"{'OOS PnL':>9} | {'WF':>5}")
print("-" * 100)

for r in phase1_results:
    i = r['is']
    w = r['wf']
    print(f"  {r['tp_factor']:.2f}  | {i['pnl']:>+8.1f}% | {i['mdd']:>6.2f}% | "
          f"{i['pnl_mdd']:>7.1f}x | {i['wr']:>5.1f}% | {i['rr']:>5.3f} | "
          f"{i['wr_margin']:>+6.1f}pp | {i['trades_per_day']:>5.1f} | "
          f"{w['oos_pnl']:>+8.1f}% | {'PASS' if w['all_pass'] else 'FAIL':>5}")


# ═══════════════════════════════════════════════════════════════════════
# PHASE 2: WR Degradation Robustness
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 80}")
print("PHASE 2: WR Degradation Robustness Test")
print(f"{'=' * 80}")
print("Question: At live WR ~65%, which factor gives best expected PnL?")
print(f"Degradation levels (from IS WR): {WR_DEGRADATION_PP}")
print()

phase2_results = []

for r in phase1_results:
    i = r['is']
    factor = r['tp_factor']
    avg_win = i['avg_win']
    avg_loss = i['avg_loss']
    is_wr = i['wr']
    n_trades = i['trades']
    be_wr = i['be_wr']

    degradation_data = {}
    for deg in WR_DEGRADATION_PP:
        scenario_wr = is_wr + deg
        if scenario_wr < 0:
            scenario_wr = 0
        exp_pnl = expected_pnl_at_wr(scenario_wr, avg_win, avg_loss, 0, n_trades)
        degradation_data[f"wr_{deg:+d}pp"] = {
            'wr': round(scenario_wr, 2),
            'exp_pnl': round(exp_pnl, 2),
        }

    # Also compute expected PnL at fixed WR = 65% (live proxy)
    exp_at_65 = expected_pnl_at_wr(65.0, avg_win, avg_loss, 0, n_trades)
    # Expected PnL at fixed WR = 60%
    exp_at_60 = expected_pnl_at_wr(60.0, avg_win, avg_loss, 0, n_trades)
    # WR where PnL = 0 (fees already in avg_win/avg_loss)
    be_wr_val = wr_for_breakeven(avg_win, avg_loss)

    p2_entry = {
        'tp_factor': factor,
        'is_wr': is_wr,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'rr': i['rr'],
        'be_wr': round(be_wr_val, 2),
        'is_margin': round(is_wr - be_wr_val, 2),
        'exp_pnl_at_65': round(exp_at_65, 2),
        'exp_pnl_at_60': round(exp_at_60, 2),
        'n_trades': n_trades,
        'degradation': degradation_data,
    }
    phase2_results.append(p2_entry)

# Phase 2 summary table
print(f"{'Factor':>7} | {'IS WR':>6} | {'BE WR':>6} | {'Margin':>7} | {'R:R':>6} | "
      f"{'ExpPnL@65%':>11} | {'ExpPnL@60%':>11} | {'avg_W':>7} | {'avg_L':>7}")
print("-" * 100)

for p in phase2_results:
    print(f"  {p['tp_factor']:.2f}  | {p['is_wr']:>5.1f}% | {p['be_wr']:>5.1f}% | "
          f"{p['is_margin']:>+6.1f}pp | {p['rr']:>5.3f} | "
          f"{p['exp_pnl_at_65']:>+10.1f}% | {p['exp_pnl_at_60']:>+10.1f}% | "
          f"{p['avg_win']:>6.3f}% | {p['avg_loss']:>6.3f}%")

# Degradation table
print(f"\n{'Factor':>7}", end="")
for deg in WR_DEGRADATION_PP:
    print(f" | {'WR'+str(deg)+'pp':>10}", end="")
print()
print("-" * (8 + 13 * len(WR_DEGRADATION_PP)))

for p in phase2_results:
    print(f"  {p['tp_factor']:.2f} ", end="")
    for deg in WR_DEGRADATION_PP:
        key = f"wr_{deg:+d}pp"
        val = p['degradation'][key]
        print(f" | {val['exp_pnl']:>+9.1f}%", end="")
    print()

# Find which factor is best at live WR 65%
best_at_65 = max(phase2_results, key=lambda x: x['exp_pnl_at_65'])
best_at_60 = max(phase2_results, key=lambda x: x['exp_pnl_at_60'])
print(f"\nBest at live WR 65%: TP x{best_at_65['tp_factor']:.2f} "
      f"(ExpPnL {best_at_65['exp_pnl_at_65']:+.1f}%, "
      f"R:R {best_at_65['rr']:.3f}, margin {best_at_65['is_margin']:+.1f}pp)")
print(f"Best at live WR 60%: TP x{best_at_60['tp_factor']:.2f} "
      f"(ExpPnL {best_at_60['exp_pnl_at_60']:+.1f}%, "
      f"R:R {best_at_60['rr']:.3f}, margin {best_at_60['is_margin']:+.1f}pp)")

# Rank factors by robustness: which factor stays positive longest as WR degrades?
print(f"\nWR Robustness Ranking (highest WR margin = last to go negative):")
sorted_by_margin = sorted(phase2_results, key=lambda x: x['is_margin'], reverse=True)
for rank, p in enumerate(sorted_by_margin, 1):
    status = "SAFE@65%" if p['exp_pnl_at_65'] > 0 else "NEG@65%"
    print(f"  {rank}. TP x{p['tp_factor']:.2f}: margin {p['is_margin']:+.1f}pp, "
          f"BE WR {p['be_wr']:.1f}%, {status}")


# ═══════════════════════════════════════════════════════════════════════
# PHASE 3: MC Discrimination Test (top 3 factors)
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 80}")
print("PHASE 3: MC Discrimination Test (Sign Randomization)")
print(f"{'=' * 80}")

# Select top 3 by composite score: WF PASS + high PnL/MDD + high margin
passing = [r for r in phase1_results if r['wf']['all_pass']]
if len(passing) >= 3:
    # Sort by PnL/MDD among passing
    top3_candidates = sorted(passing, key=lambda x: x['is']['pnl_mdd'], reverse=True)[:3]
else:
    # If <3 passing, use top 3 by OOS PnL
    top3_candidates = sorted(phase1_results, key=lambda x: x['wf']['oos_pnl'], reverse=True)[:3]

top3_factors = [r['tp_factor'] for r in top3_candidates]
print(f"Top 3 factors for MC test: {top3_factors}")
print(f"MC seeds: {MC_SEEDS}, 10k simulations each")
print()

phase3_results = []

for factor in top3_factors:
    t0 = time.time()
    scaled_patterns = apply_tp_factor(base_patterns, factor)
    signal_tuples = build_signal_tuples(scaled_patterns, signal_index)

    # Real IS PnL (from Phase 1)
    real_entry = next(r for r in phase1_results if r['tp_factor'] == factor)
    real_is_pnl = real_entry['is']['pnl']

    # MC: randomize trade directions, run WF
    mc_results = []
    for seed in MC_SEEDS:
        rng = np.random.RandomState(seed)

        # Randomize direction in signal_tuples
        random_tuples = []
        for (bar, pat, dirn, tp, sl) in signal_tuples:
            rand_dir = rng.choice(['LONG', 'SHORT'])
            random_tuples.append((bar, pat, rand_dir, tp, sl))

        # WF on random signals
        wf_folds, oos_pnl, wf_pass = run_wf(
            random_tuples, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, ns, ne
        )

        # IS on random signals
        rand_trades, rand_stats = run_npos(
            random_tuples, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, ns, ne
        )
        rand_pnl = rand_stats.get('pnl', 0)

        mc_results.append({
            'seed': seed,
            'random_is_pnl': round(rand_pnl, 2),
            'random_oos_pnl': round(oos_pnl, 2),
            'random_wf_pass': wf_pass,
        })

    # Discrimination: real should beat all random
    all_random_fail = all(not m['random_wf_pass'] for m in mc_results)
    real_beats_random = all(real_is_pnl > m['random_is_pnl'] for m in mc_results)
    # p-value approximation: fraction of random that pass WF
    n_random_pass = sum(1 for m in mc_results if m['random_wf_pass'])
    p_value = n_random_pass / len(MC_SEEDS)

    discriminating = all_random_fail and real_beats_random

    elapsed = time.time() - t0
    mc_entry = {
        'tp_factor': factor,
        'real_is_pnl': round(real_is_pnl, 2),
        'real_oos_pnl': real_entry['wf']['oos_pnl'],
        'mc_results': mc_results,
        'all_random_wf_fail': all_random_fail,
        'real_beats_all_random': real_beats_random,
        'p_value': round(p_value, 4),
        'discriminating': discriminating,
    }
    phase3_results.append(mc_entry)

    rand_str = ', '.join(f"s{m['seed']}:{m['random_is_pnl']:+.1f}%({'PASS' if m['random_wf_pass'] else 'FAIL'})"
                         for m in mc_results)
    disc_str = "DISCRIMINATING" if discriminating else "NON-DISC"
    print(f"TP x{factor:.2f} | Real IS: {real_is_pnl:+.1f}% | "
          f"Random: [{rand_str}] | p={p_value:.3f} | {disc_str} | {elapsed:.1f}s")


# ═══════════════════════════════════════════════════════════════════════
# FINAL SYNTHESIS
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 80}")
print("FINAL SYNTHESIS AND RECOMMENDATION")
print(f"{'=' * 80}")

# Composite scoring
print("\nComposite Score = WF_PASS(30) + PnL/MDD_rank(20) + OOS_rank(15) + "
      "Margin_rank(15) + ExpPnL@65_rank(10) + R:R_rank(10)")
print()

n = len(phase1_results)
# Compute ranks (higher is better)
pnl_mdd_sorted = sorted(range(n), key=lambda i: phase1_results[i]['is']['pnl_mdd'], reverse=True)
oos_sorted = sorted(range(n), key=lambda i: phase1_results[i]['wf']['oos_pnl'], reverse=True)
margin_sorted = sorted(range(n), key=lambda i: phase1_results[i]['is']['wr_margin'], reverse=True)
rr_sorted = sorted(range(n), key=lambda i: phase1_results[i]['is']['rr'], reverse=True)
exp65_sorted = sorted(range(n), key=lambda i: phase2_results[i]['exp_pnl_at_65'], reverse=True)

rank_of = lambda idx, sorted_list: sorted_list.index(idx) + 1

composite_scores = []
for i in range(n):
    r = phase1_results[i]
    wf_score = 30 if r['wf']['all_pass'] else 0
    pnl_mdd_score = 20 * (1 - (rank_of(i, pnl_mdd_sorted) - 1) / max(n - 1, 1))
    oos_score = 15 * (1 - (rank_of(i, oos_sorted) - 1) / max(n - 1, 1))
    margin_score = 15 * (1 - (rank_of(i, margin_sorted) - 1) / max(n - 1, 1))
    exp65_score = 10 * (1 - (rank_of(i, exp65_sorted) - 1) / max(n - 1, 1))
    rr_score = 10 * (1 - (rank_of(i, rr_sorted) - 1) / max(n - 1, 1))
    total = wf_score + pnl_mdd_score + oos_score + margin_score + exp65_score + rr_score
    composite_scores.append({
        'tp_factor': r['tp_factor'],
        'composite': round(total, 1),
        'wf_pass': r['wf']['all_pass'],
        'pnl_mdd': r['is']['pnl_mdd'],
        'oos_pnl': r['wf']['oos_pnl'],
        'wr_margin': r['is']['wr_margin'],
        'rr': r['is']['rr'],
        'exp_pnl_65': phase2_results[i]['exp_pnl_at_65'],
    })

composite_scores.sort(key=lambda x: x['composite'], reverse=True)

print(f"{'Rank':>5} | {'Factor':>7} | {'Score':>6} | {'WF':>5} | {'PnL/MDD':>8} | "
      f"{'OOS PnL':>9} | {'Margin':>7} | {'R:R':>6} | {'ExpPnL@65':>10}")
print("-" * 90)
for rank, c in enumerate(composite_scores, 1):
    wf_str = "PASS" if c['wf_pass'] else "FAIL"
    print(f"  {rank:>3} |   {c['tp_factor']:.2f}  | {c['composite']:>5.1f} | {wf_str:>5} | "
          f"{c['pnl_mdd']:>7.1f}x | {c['oos_pnl']:>+8.1f}% | "
          f"{c['wr_margin']:>+6.1f}pp | {c['rr']:>5.3f} | {c['exp_pnl_65']:>+9.1f}%")

# Check if top factor is discriminating
top_factor = composite_scores[0]['tp_factor']
disc_entry = next((p for p in phase3_results if p['tp_factor'] == top_factor), None)
if disc_entry:
    disc_str = "DISCRIMINATING" if disc_entry['discriminating'] else "NON-DISC"
    print(f"\nTop factor TP x{top_factor:.2f} MC status: {disc_str}")
else:
    print(f"\nTop factor TP x{top_factor:.2f} was not in MC test top 3")

# Final recommendation
print(f"\n{'=' * 80}")
print("RECOMMENDATION")
print(f"{'=' * 80}")
best = composite_scores[0]
p2_best = next(p for p in phase2_results if p['tp_factor'] == best['tp_factor'])
print(f"Recommended factor: TP x{best['tp_factor']:.2f}")
print(f"  IS: PnL/MDD {best['pnl_mdd']:.1f}x, WR margin {best['wr_margin']:+.1f}pp, R:R {best['rr']:.3f}")
print(f"  OOS: {best['oos_pnl']:+.1f}%")
print(f"  WR degradation: BE WR {p2_best['be_wr']:.1f}%, "
      f"ExpPnL@65% {p2_best['exp_pnl_at_65']:+.1f}%, "
      f"ExpPnL@60% {p2_best['exp_pnl_at_60']:+.1f}%")
print(f"  vs Current (TP x0.50): check tp_factor_sweep.json for comparison")

# Compare with current 0.50 — need to get it from sweep or run it
print(f"\nKey insight: Higher R:R ({best['rr']:.3f}) means fewer TPs needed to cover 1 SL.")
print(f"At live WR 65%, expected PnL per {p2_best['n_trades']} trades: {p2_best['exp_pnl_at_65']:+.1f}%")


# ─── Save results (strip raw trades before saving) ───
phase1_save = []
for r in phase1_results:
    r_copy = {k: v for k, v in r.items() if k != 'is_trades_raw'}
    phase1_save.append(r_copy)

output = {
    "study": "tp_factor_deep_study",
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "data_bars": n_bars,
    "neutral_window": [ns, ne],
    "neutral_days": round(n_days, 1),
    "patterns": len(base_patterns),
    "phases": {
        "phase1": {
            "description": "Fine-grained sweep 0.68-0.92 step 0.02",
            "tp_factors": TP_FACTORS,
            "results": phase1_save,
        },
        "phase2": {
            "description": "WR degradation robustness test",
            "degradation_levels_pp": WR_DEGRADATION_PP,
            "results": phase2_results,
            "best_at_65pct": best_at_65['tp_factor'],
            "best_at_60pct": best_at_60['tp_factor'],
        },
        "phase3": {
            "description": "MC discrimination test (sign randomization)",
            "seeds": MC_SEEDS,
            "top3_factors": top3_factors,
            "results": phase3_results,
        },
    },
    "composite_ranking": composite_scores,
    "recommendation": {
        "tp_factor": best['tp_factor'],
        "composite_score": best['composite'],
        "pnl_mdd": best['pnl_mdd'],
        "oos_pnl": best['oos_pnl'],
        "wr_margin": best['wr_margin'],
        "rr": best['rr'],
        "exp_pnl_at_65": best['exp_pnl_65'],
    },
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
    "elapsed_seconds": round(time.time() - start_time, 1),
}

output_path = Path(OUTPUT_FILE)
output_path.parent.mkdir(exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to {OUTPUT_FILE}")
print(f"Total elapsed: {time.time() - start_time:.1f}s")
