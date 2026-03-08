#!/usr/bin/env python3
"""
R1: Survival Probability Analysis at Realistic Live WR

Context: Backtest IS WR=70.8% but Live WR=65.9%. rr_holdtime_study tested all
variants (SL tightening, breakeven, partial exits) -- ALL passed WF 3/3.
Random discrimination only covered tp0.9/sl1.0.

Phase 1: MC Survival Simulation (1000 seeds x 200 trades) at WR 50-70%
Phase 2: Random Signal Discrimination (20 seeds) for top 3 variants
Phase 3: Live WR Crossover Analysis (safety margin = live WR - crossover WR)

Protocol: LEVERAGE=3, Fee=0.10%, Slippage=0.02%, Timeout=DROP, Compound sizing.
Output: results/r1_survival_probability_study.json
"""

import os, sys, json, time
from datetime import datetime
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

# ============================================================
# Paths & Config
# ============================================================
DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
RR_STUDY_FILE = os.path.join(_PROJECT_ROOT, 'results', 'rr_holdtime_study.json')
PATTERNS_FILE = os.path.join(_PROJECT_ROOT, 'results', 'dynamic_patterns.json')
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'r1_survival_probability_study.json')

MC_SIM_SEEDS = 1000
TRADE_SEQ_LEN = 200
WR_LEVELS = [50, 55, 60, 65, 70]
N_RANDOM_SEEDS = 20
WF_FOLDS = 3
LIVE_WR = 65.9

VARIANT_KEYS = {
    'baseline':         ('baseline', None),
    'sl_mult_0.7':      ('H1_tighter_sl', 'sl_mult_0.7'),
    'sl_mult_0.9':      ('H1_tighter_sl', 'sl_mult_0.9'),
    'breakeven_12h':    ('H5_breakeven_stop', 'breakeven_after_12h'),
    'partial_exit_12h': ('H6_partial_exit', 'partial_exit_12h'),
}

NPOS_KW = dict(
    n_slots=DEFAULT_N_SLOTS, direction_cap=DEFAULT_DIRECTION_CAP,
    regime_mult=0.3,
    agg_risk_counter=DEFAULT_AGG_RISK_COUNTER,
    agg_risk_with=DEFAULT_AGG_RISK_WITH,
    momentum_lookback=DEFAULT_MOMENTUM_LOOKBACK,
    momentum_threshold=DEFAULT_MOMENTUM_THRESHOLD,
    momentum_cooldown=DEFAULT_MOMENTUM_COOLDOWN,
    clamp_lo=DEFAULT_ATR_CLAMP_LO, clamp_hi=DEFAULT_ATR_CLAMP_HI,
    cascade_tighten_pct=DEFAULT_CASCADE_TIGHTEN_PCT,
    timeout_bars=TIMEOUT_BARS,
)


def _get_variant(rr_data, group, subkey):
    return rr_data[group] if subkey is None else rr_data[group][subkey]


# ============================================================
# Phase 1: Monte Carlo Survival Simulation
# ============================================================
def run_phase1(rr_data):
    print("\n" + "=" * 70)
    print("PHASE 1: Monte Carlo Survival Simulation")
    print(f"  {MC_SIM_SEEDS} sims x {TRADE_SEQ_LEN} trades per WR level")
    print("=" * 70)

    results = {}
    for vname, (group, subkey) in VARIANT_KEYS.items():
        v = _get_variant(rr_data, group, subkey)
        avg_win, avg_loss = v['avg_win'], v['avg_loss']
        rr, be_wr = v['rr'], v['breakeven_wr']

        print(f"\n--- {vname}: WR={v['wr']}%, R:R={rr}, "
              f"avg_win={avg_win}%, avg_loss={avg_loss}%, BE_WR={be_wr}% ---")

        wr_results = {}
        for wr_target in WR_LEVELS:
            rng = np.random.RandomState(42)
            wr_f = wr_target / 100.0
            # Size per slot as fraction
            slot_frac = 1.0 / DEFAULT_N_SLOTS

            terminals = np.empty(MC_SIM_SEEDS)
            max_dds = np.empty(MC_SIM_SEEDS)

            for sim in range(MC_SIM_SEEDS):
                eq = 100.0
                peak = 100.0
                mdd = 0.0
                for _ in range(TRADE_SEQ_LEN):
                    if rng.random() < wr_f:
                        pnl_pct = avg_win / 100.0 * slot_frac
                    else:
                        pnl_pct = -avg_loss / 100.0 * slot_frac
                    eq *= (1 + pnl_pct)
                    if eq > peak:
                        peak = eq
                    dd = (peak - eq) / peak * 100 if peak > 0 else 0
                    if dd > mdd:
                        mdd = dd
                terminals[sim] = eq - 100.0
                max_dds[sim] = mdd

            # Analytical E[PnL/trade] per slot
            e_pnl = (wr_f * avg_win - (1 - wr_f) * avg_loss) * slot_frac

            wr_results[f'wr_{wr_target}'] = {
                'median_terminal_pnl': round(float(np.median(terminals)), 2),
                'p5_terminal_pnl': round(float(np.percentile(terminals, 5)), 2),
                'p95_terminal_pnl': round(float(np.percentile(terminals, 95)), 2),
                'prob_mdd_gt_25': round(float(np.mean(max_dds > 25) * 100), 1),
                'prob_mdd_gt_50': round(float(np.mean(max_dds > 50) * 100), 1),
                'expected_pnl_per_trade': round(float(e_pnl), 4),
                'mean_max_dd': round(float(np.mean(max_dds)), 2),
            }
            print(f"  WR {wr_target}%: E[PnL/t]={e_pnl:+.4f}% | "
                  f"Median={np.median(terminals):+.1f}% | "
                  f"P5={np.percentile(terminals, 5):+.1f}% | "
                  f"P(MDD>25%)={np.mean(max_dds > 25)*100:.1f}%")

        results[vname] = {
            'actual_wr': v['wr'], 'avg_win': avg_win, 'avg_loss': avg_loss,
            'rr': rr, 'breakeven_wr': be_wr, 'wr_levels': wr_results,
        }
    return results


# ============================================================
# Phase 2: Random Signal Discrimination
# ============================================================
def generate_random_signals(rng, neutral_start, neutral_end, target_count, pattern_details):
    available = np.arange(neutral_start + 2, neutral_end - 1)
    n = min(len(available), target_count)
    sig_bars = rng.choice(available, size=n, replace=False)
    sig_bars.sort()
    pats = list(pattern_details.values())
    signals = []
    for sb in sig_bars:
        d = rng.choice(['LONG', 'SHORT'])
        p = pats[rng.randint(0, len(pats))]
        signals.append((int(sb), f"RND_{sb}", d, p['tp'], p['sl']))
    return signals


def wf_random(signals, opens, highs, lows, closes, n_bars,
              atr_ratio, ema_slope, ns, ne):
    data_len = ne - ns
    seg = data_len // (WF_FOLDS + 1)
    folds = []
    for fold in range(WF_FOLDS):
        oos_s = ns + (fold + 1) * seg
        oos_e = ns + (fold + 2) * seg if fold < WF_FOLDS - 1 else ne
        oos_sig = [(s, p, d, tp, sl) for s, p, d, tp, sl in signals
                   if oos_s <= s < oos_e]
        if not oos_sig:
            folds.append({'fold': fold+1, 'oos_pnl': 0.0, 'oos_trades': 0,
                          'oos_wr': 0.0, 'oos_positive': False})
            continue
        trades, _stats = portfolio_npos(oos_sig, opens, highs, lows, closes, n_bars,
                                atr_ratio, ema_slope, oos_s, oos_e, **NPOS_KW)
        st = calc_stats_compound(trades)
        folds.append({'fold': fold+1, 'oos_pnl': st['pnl'], 'oos_trades': st['trades'],
                      'oos_wr': st['wr'], 'oos_positive': st['pnl'] > 0})

    pos_folds = sum(1 for f in folds if f['oos_positive'])
    return {
        'folds': folds, 'positive_folds': pos_folds,
        'pass_3_3': pos_folds == WF_FOLDS,
        'total_oos_pnl': round(sum(f['oos_pnl'] for f in folds), 2),
    }


def run_phase2(pat_details, ns, ne, opens, highs, lows, closes, n_bars, atr, ema):
    print("\n" + "=" * 70)
    print("PHASE 2: Random Signal Discrimination (20 seeds)")
    print("=" * 70)

    target_signals = 3500
    test_variants = ['baseline', 'breakeven_12h', 'partial_exit_12h']
    results = {}

    for vname in test_variants:
        print(f"\n--- {vname} ---")
        pass_count = 0
        seed_details = []
        for seed in range(1, N_RANDOM_SEEDS + 1):
            rng = np.random.RandomState(seed)
            sigs = generate_random_signals(rng, ns, ne, target_signals, pat_details)
            wf = wf_random(sigs, opens, highs, lows, closes, n_bars, atr, ema, ns, ne)
            if wf['pass_3_3']:
                pass_count += 1
            fp = [f['oos_pnl'] for f in wf['folds']]
            seed_details.append({'seed': seed, 'pass_3_3': wf['pass_3_3'],
                                 'total_oos_pnl': wf['total_oos_pnl']})
            s = "PASS" if wf['pass_3_3'] else "FAIL"
            print(f"  Seed {seed:2d}: {s} | OOS: "
                  f"{fp[0]:+6.1f} / {fp[1]:+6.1f} / {fp[2]:+6.1f}")

        pp = pass_count / N_RANDOM_SEEDS * 100
        results[vname] = {'n_seeds': N_RANDOM_SEEDS, 'pass_count': pass_count,
                          'pass_pct': round(pp, 1), 'seed_details': seed_details}
        print(f"  Result: {pass_count}/{N_RANDOM_SEEDS} PASS ({pp:.1f}%)")

    return results


# ============================================================
# Phase 3: Crossover Analysis
# ============================================================
def run_phase3(rr_data):
    print("\n" + "=" * 70)
    print(f"PHASE 3: Live WR Crossover Analysis (Live WR = {LIVE_WR}%)")
    print("=" * 70)

    results = {}
    for vname, (group, subkey) in VARIANT_KEYS.items():
        v = _get_variant(rr_data, group, subkey)
        aw, al = v['avg_win'], v['avg_loss']
        # Crossover: wr*aw = (1-wr)*al => wr = al/(aw+al)
        cross_wr = al / (aw + al) * 100 if (aw + al) > 0 else 100
        margin = LIVE_WR - cross_wr
        results[vname] = {
            'avg_win': aw, 'avg_loss': al,
            'crossover_wr': round(cross_wr, 1),
            'breakeven_wr_reported': v['breakeven_wr'],
            'safety_margin': round(margin, 1),
        }
        print(f"  {vname:20s}: crossover={cross_wr:5.1f}%, margin={margin:+5.1f}pp "
              f"(win={aw:.3f}%, loss={al:.3f}%)")
    return results


# ============================================================
# Main
# ============================================================
def main():
    t0 = time.time()
    print("=" * 70)
    print("R1: SURVIVAL PROBABILITY ANALYSIS AT REALISTIC LIVE WR")
    print(f"  Date: {datetime.now().isoformat()}")
    print("=" * 70)

    # Load prior study results
    print("\n[1] Loading rr_holdtime_study results...")
    with open(RR_STUDY_FILE) as f:
        rr_data = json.load(f)

    # Phase 1: MC simulation from distributions
    phase1 = run_phase1(rr_data)

    # Load data for Phase 2
    print("\n[2] Loading data for Phase 2 backtests...")
    df = pd.read_csv(DATA_FILE)
    df['body'] = abs(df['close'] - df['open'])
    df['avg_body'] = df['body'].rolling(AVG_BODY_WINDOW).mean()
    types = []
    for i, row in df.iterrows():
        ab = df.at[i, 'avg_body']
        if pd.isna(ab):
            ab = 1.0
        types.append(classify_candle(row, ab).value)
    df['candle_type'] = types
    n_bars = len(df)
    opens, highs = df['open'].values, df['high'].values
    lows, closes = df['low'].values, df['close'].values
    print(f"  {n_bars} bars")

    with open(PATTERNS_FILE) as f:
        dp = json.load(f)
    pat_details = dp['pattern_details']
    nw = dp.get('neutral_window', {})
    ns, ne = nw.get('start_bar', 0), nw.get('end_bar', n_bars)
    print(f"  {len(pat_details)} patterns, neutral [{ns},{ne}]")

    atr = compute_atr_ratio(highs, lows, closes)
    ema = compute_ema_slope(closes)

    # Phase 2: Random discrimination
    phase2 = run_phase2(pat_details, ns, ne, opens, highs, lows, closes, n_bars, atr, ema)

    # Phase 3: Crossover
    phase3 = run_phase3(rr_data)

    # ============================================================
    # Summary Table
    # ============================================================
    print("\n" + "=" * 70)
    print("FINAL COMPARISON TABLE")
    print("=" * 70)

    hdr = (f"{'Variant':<20} {'BE_WR':>6} {'Cross':>6} {'Margin':>7} "
           f"{'E@60':>8} {'E@65':>8} {'P(DD>25)@60':>12} {'RndPass':>8}")
    print(hdr)
    print("-" * len(hdr))

    for vname in VARIANT_KEYS:
        p1 = phase1[vname]
        p3 = phase3[vname]
        e60 = p1['wr_levels']['wr_60']['expected_pnl_per_trade']
        e65 = p1['wr_levels']['wr_65']['expected_pnl_per_trade']
        mdd60 = p1['wr_levels']['wr_60']['prob_mdd_gt_25']
        margin = p3['safety_margin']
        rnd = phase2.get(vname, {}).get('pass_pct', '-')
        rs = f"{rnd}%" if isinstance(rnd, (int, float)) else str(rnd)
        print(f"{vname:<20} {p1['breakeven_wr']:>5.1f}% {p3['crossover_wr']:>5.1f}% "
              f"{margin:>+6.1f}pp {e60:>+7.4f}% {e65:>+7.4f}% "
              f"{mdd60:>11.1f}% {rs:>8}")

    # Verdicts
    print("\n" + "=" * 70)
    print("VERDICTS")
    print("=" * 70)

    verdicts = {}

    best_margin = max(phase3.items(), key=lambda x: x[1]['safety_margin'])
    verdicts['best_safety_margin'] = {
        'variant': best_margin[0], 'margin_pp': best_margin[1]['safety_margin']}
    print(f"  Best safety margin: {best_margin[0]} "
          f"({best_margin[1]['safety_margin']:+.1f}pp)")

    best_e65 = max(phase1.items(),
                   key=lambda x: x[1]['wr_levels']['wr_65']['expected_pnl_per_trade'])
    verdicts['best_epnl_at_65'] = {
        'variant': best_e65[0],
        'epnl': best_e65[1]['wr_levels']['wr_65']['expected_pnl_per_trade']}
    print(f"  Best E[PnL/t] at WR=65%: {best_e65[0]} "
          f"({best_e65[1]['wr_levels']['wr_65']['expected_pnl_per_trade']:+.4f}%)")

    best_ruin = min(phase1.items(),
                    key=lambda x: x[1]['wr_levels']['wr_60']['prob_mdd_gt_25'])
    verdicts['lowest_ruin_at_60'] = {
        'variant': best_ruin[0],
        'prob_mdd_gt_25': best_ruin[1]['wr_levels']['wr_60']['prob_mdd_gt_25']}
    print(f"  Lowest P(MDD>25%) at WR=60%: {best_ruin[0]} "
          f"({best_ruin[1]['wr_levels']['wr_60']['prob_mdd_gt_25']:.1f}%)")

    for vn in ['baseline', 'breakeven_12h', 'partial_exit_12h']:
        pp = phase2[vn]['pass_pct']
        disc = "DISCRIMINATING" if pp < 10 else ("WEAK" if pp < 50 else "NON-DISCRIMINATING")
        verdicts[f'random_disc_{vn}'] = {'pass_pct': pp, 'verdict': disc}
        print(f"  Random disc ({vn}): {phase2[vn]['pass_count']}/{N_RANDOM_SEEDS} "
              f"PASS ({pp:.1f}%) -> {disc}")

    print(f"\n  LIVE WR = {LIVE_WR}%")
    print(f"  Recommendation: Prefer variant with highest safety margin + discriminating.")

    elapsed = time.time() - t0
    output = {
        'metadata': {
            'script': 'r1_survival_probability_study.py',
            'date': datetime.now().isoformat(),
            'live_wr': LIVE_WR,
            'mc_sims': MC_SIM_SEEDS,
            'trade_sequence_len': TRADE_SEQ_LEN,
            'wr_levels': WR_LEVELS,
            'n_random_seeds': N_RANDOM_SEEDS,
            'protocol': {
                'leverage': LEVERAGE, 'fee_pct': FEE_PCT,
                'slippage': SLIPPAGE_BUFFER,
                'n_slots': DEFAULT_N_SLOTS,
                'direction_cap': DEFAULT_DIRECTION_CAP,
                'cascade_tighten_pct': DEFAULT_CASCADE_TIGHTEN_PCT,
                'timeout_bars': TIMEOUT_BARS,
            },
            'timing_sec': round(elapsed, 1),
        },
        'phase1_survival': phase1,
        'phase2_random_discrimination': phase2,
        'phase3_crossover': phase3,
        'verdicts': verdicts,
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_FILE}")
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f}min)")


if __name__ == '__main__':
    main()
