#!/usr/bin/env python3
"""
Production Stack Audit — v1.42.0
==================================

Comprehensive statistical audit of every active mechanism AND the core
pattern strategy itself. Tests for significance, consistency, and overfitting.

AUDIT 1: Individual Mechanism Ablation with Bootstrap CI
AUDIT 2: Core Pattern Strategy vs Random Baseline
AUDIT 3: Pattern Edge Temporal Decay (H1/H2 train/test)
AUDIT 4: Pattern-Level Consistency (binomial test per pattern)
AUDIT 5: Mechanism Interaction — Full Stack vs Bare
AUDIT 6: Temporal Consistency of Full Stack (6-period split)

Standard Research Protocol:
  - Production classify_candle import
  - LEVERAGE = 3 FIXED (v1.42.0: adaptive disabled)
  - FEE = FEE_PCT * LEVERAGE (notional basis)
  - Timeout = DROP, Same-bar = abs(tp - bar_open)
  - Entry = next-bar open, ATR-scaled TP/SL
  - Compound (multiplicative) sizing

Author: Research Agent
Date: 2026-03-04
"""

import os
import sys
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)

from scripts.production.pattern_5m.indicators import classify_candle
from scripts.production.pattern_5m.constants import AVG_BODY_WINDOW

warnings.filterwarnings('ignore')

# ============================================================
# Constants — matching production v1.42.0 exactly
# ============================================================
LEVERAGE = 3          # v1.42.0: FIXED (adaptive leverage DISABLED)
FEE_PCT = 0.10
SLIPPAGE_BUFFER = 0.02
TIMEOUT_BARS = 864
N_SLOTS = 9
DIRECTION_CAP = 7
AGG_RISK_COUNTER = 3.0
AGG_RISK_WITH = 7.0
MOMENTUM_LOOKBACK = 6
MOMENTUM_THRESHOLD = 1.0
MOMENTUM_COOLDOWN = 6
ATR_PERIOD = 14
ATR_WINDOW = 576
ATR_CLAMP_LO = 0.6
ATR_CLAMP_HI = 1.7
EMA_PERIOD = 20
EMA_LOOKBACK = 5
BARS_PER_DAY = 288

# MDD sizing params (production v1.35.2)
MDD_FULL_BELOW = 3.0
MDD_MIN_ABOVE = 15.0
MDD_MIN_SCALE = 0.25

# Early exit params
EARLY_CONFIRM = 3
EARLY_MIN_PROFIT = 0.3

DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
PATTERNS_FILE = os.path.join(_PROJECT_ROOT, 'results', 'dynamic_patterns.json')
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'production_stack_audit.json')

# Bootstrap params
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 42
BOOTSTRAP_CI = 95  # percent

# Random baseline params
RANDOM_SEEDS = 100  # seeds 0-99


# ============================================================
# Data Loading & Preprocessing
# ============================================================

def load_and_classify(data_file):
    df = pd.read_csv(data_file)
    if 'open' not in df.columns:
        df.columns = [c.lower() for c in df.columns]
    if 'type_code' in df.columns:
        df['rctype'] = df['type_code']
    elif 'rctype' not in df.columns:
        if 'avg_body' not in df.columns:
            df['avg_body'] = (abs(df['close'] - df['open'])).rolling(AVG_BODY_WINDOW).mean()
        types = []
        for i in range(len(df)):
            row = df.iloc[i]
            ab = row.get('avg_body', 1.0)
            if pd.isna(ab):
                ab = 1.0
            types.append(classify_candle(row, ab))
        df['rctype'] = types
    print(f"  Loaded {len(df)} bars, {df['rctype'].nunique()} candle types")
    return df


def compute_atr_ratio(df):
    h, l, c = df['high'].values, df['low'].values, df['close'].values
    tr = np.maximum(h - l, np.maximum(abs(h - np.roll(c, 1)), abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    atr = pd.Series(tr).ewm(span=ATR_PERIOD, adjust=False).mean().values
    med = pd.Series(atr).rolling(ATR_WINDOW, min_periods=1).median().values
    return np.where(med > 0, atr / med, 1.0)


def compute_ema_slope(closes):
    ema = pd.Series(closes).ewm(span=EMA_PERIOD, adjust=False).mean().values
    slope = np.full(len(closes), 0.0)
    for i in range(EMA_LOOKBACK, len(closes)):
        slope[i] = ema[i] - ema[i - EMA_LOOKBACK]
    return slope


def find_neutral_window(closes, tol_pct=1.0):
    n = len(closes)
    best_start, best_end, best_len = 0, n - 1, 0
    for i in range(n):
        for j in range(n - 1, i + best_len - 1, -1):
            if closes[i] > 0 and abs(closes[j] / closes[i] - 1) * 100 <= tol_pct:
                length = j - i
                if length > best_len:
                    best_start, best_end, best_len = i, j, length
                break
    return best_start, best_end


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


# ============================================================
# Portfolio Simulator — all 8 active mechanisms toggleable
# ============================================================

def portfolio_sim(
    signal_tuples, opens, highs, lows, closes, type_codes, n_bars,
    atr_ratio, ema_slope, start_bar, end_bar,
    direction_cap_enabled=True,
    momentum_enabled=True,
    agg_risk_enabled=True,
    mdd_sizing_enabled=True,
    atr_scaling_enabled=True,
    timeout_enabled=True,
    early_exit_enabled=True,
    cascade_enabled=True,
):
    """N-pos portfolio simulator with all 8 ACTIVE v1.42.0 mechanisms toggleable.

    v1.42.0 settings: M2 Regime sizing DISABLED, M4 Adaptive leverage DISABLED.
    These are not included as toggles since they are already OFF in production.

    Returns: (trades_list, per_trade_pnl_portfolio_array)
    """
    fee = FEE_PCT * LEVERAGE
    size_pct = 100.0 / N_SLOTS

    dir_cap = DIRECTION_CAP if direction_cap_enabled else 999
    timeout_bars = TIMEOUT_BARS if timeout_enabled else 999999

    # State
    positions = []
    trades = []
    equity = 100.0
    peak_equity = 100.0
    max_dd = 0.0
    mom_pause_until = {'LONG': -1, 'SHORT': -1}

    # Index signals by bar for O(1) lookup
    sig_by_bar = {}
    for s_bar, pat, direction, tp, sl in signal_tuples:
        if start_bar <= s_bar < end_bar:
            sig_by_bar.setdefault(s_bar, []).append((pat, direction, tp, sl))

    for bar in range(start_bar, end_bar):
        if bar >= n_bars - 1:
            break

        # ====================================================
        # 1. CHECK EXITS
        # ====================================================
        closed_slots = set()
        sl_exits = []
        bar_pnl_sum = 0.0

        h_bar = highs[bar] if bar < n_bars else 0
        l_bar = lows[bar] if bar < n_bars else 0
        o_bar = opens[bar] if bar < n_bars else 0

        for pos in positions:
            entry_bar = pos['entry_bar']
            if bar < entry_bar:
                continue
            entry_p = pos['entry_price']
            if entry_p <= 0:
                continue

            direction = pos['direction']
            eff_tp = pos['eff_tp_pct']
            eff_sl = pos['eff_sl_pct']
            sm = pos['size_mult']
            hold = bar - entry_bar

            exit_price = None
            reason = None

            # 1a. Timeout
            if timeout_bars > 0 and hold >= timeout_bars:
                closed_slots.add(pos['slot'])
                continue  # DROP

            # 1b. Early exit
            if early_exit_enabled and hold >= EARLY_CONFIRM:
                exit_types = ['BD'] if direction == 'LONG' else ['BU']
                candle_ok = True
                for k in range(EARLY_CONFIRM):
                    cb = bar - 1 - k
                    if cb < 0 or cb >= len(type_codes) or type_codes[cb] not in exit_types:
                        candle_ok = False
                        break
                if candle_ok:
                    cp = closes[bar - 1] if bar - 1 < n_bars else entry_p
                    if direction == 'LONG':
                        unr = (cp / entry_p - 1) * 100 * LEVERAGE
                    else:
                        unr = (1 - cp / entry_p) * 100 * LEVERAGE
                    if unr >= EARLY_MIN_PROFIT:
                        exit_price = cp
                        reason = 'EARLY'

            # 1c. TP/SL intrabar
            if reason is None:
                if direction == 'LONG':
                    tp_p = entry_p * (1 + eff_tp / 100)
                    sl_p = entry_p * (1 - eff_sl / 100)
                    hit_tp = h_bar >= tp_p
                    hit_sl = l_bar <= sl_p
                else:
                    tp_p = entry_p * (1 - eff_tp / 100)
                    sl_p = entry_p * (1 + eff_sl / 100)
                    hit_tp = l_bar <= tp_p
                    hit_sl = h_bar >= sl_p

                if hit_tp and hit_sl:
                    if abs(tp_p - o_bar) <= abs(sl_p - o_bar):
                        exit_price, reason = tp_p, 'TP'
                    else:
                        exit_price, reason = sl_p, 'SL'
                elif hit_tp:
                    exit_price, reason = tp_p, 'TP'
                elif hit_sl:
                    exit_price, reason = sl_p, 'SL'

            if reason is None:
                continue

            # Calculate PnL
            if direction == 'LONG':
                pnl = (exit_price / entry_p - 1) * 100 * LEVERAGE
            else:
                pnl = (1 - exit_price / entry_p) * 100 * LEVERAGE
            pnl -= fee

            pnl_portfolio = pnl * (size_pct / 100) * sm
            trades.append({
                'entry_bar': entry_bar, 'exit_bar': bar, 'pnl_slot': pnl,
                'reason': reason, 'pattern': pos['pattern'],
                'direction': direction, 'size_mult': sm,
                'pnl_portfolio': pnl_portfolio, 'leverage': LEVERAGE,
            })
            closed_slots.add(pos['slot'])
            bar_pnl_sum += pnl_portfolio

            if reason == 'SL':
                sl_exits.append(pos)

        # Remove closed positions
        positions = [p for p in positions if p['slot'] not in closed_slots]

        # 1d. Cascade SL tightening
        if cascade_enabled and sl_exits:
            cascade_keep = 0.25  # 75% tighten => keep 25%
            for sl_pos in sl_exits:
                sl_dir = sl_pos['direction']
                for pos in positions:
                    if pos['direction'] == sl_dir:
                        pos['eff_sl_pct'] *= cascade_keep

        # Update equity (multiplicative)
        equity *= (1 + bar_pnl_sum / 100)
        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0
        if dd > max_dd:
            max_dd = dd

        # ====================================================
        # 2. MOMENTUM GUARD state update
        # ====================================================
        if momentum_enabled and bar >= MOMENTUM_LOOKBACK:
            pa = closes[bar - MOMENTUM_LOOKBACK]
            if pa > 0:
                pct = (closes[bar] / pa - 1) * 100
                if pct > MOMENTUM_THRESHOLD:
                    mom_pause_until['SHORT'] = max(mom_pause_until['SHORT'],
                                                   bar + MOMENTUM_COOLDOWN)
                elif pct < -MOMENTUM_THRESHOLD:
                    mom_pause_until['LONG'] = max(mom_pause_until['LONG'],
                                                  bar + MOMENTUM_COOLDOWN)

        # ====================================================
        # 3. PROCESS NEW ENTRIES
        # ====================================================
        if bar not in sig_by_bar:
            continue

        for pat, direction, tp_pct, sl_pct in sig_by_bar[bar]:
            # Max positions
            if len(positions) >= N_SLOTS:
                continue

            # Direction cap
            dir_count = sum(1 for p in positions if p['direction'] == direction)
            if dir_count >= dir_cap:
                continue

            # Duplicate pattern
            if any(p['pattern'] == pat for p in positions):
                continue

            entry_bar = bar + 1
            if entry_bar >= n_bars:
                continue
            entry_price = opens[entry_bar]
            if entry_price <= 0:
                continue

            # Momentum guard
            if momentum_enabled and bar < mom_pause_until.get(direction, -1):
                continue

            # Size multiplier
            sm = 1.0

            # MDD sizing
            if mdd_sizing_enabled and peak_equity > 0:
                dd_pct = (peak_equity - equity) / peak_equity * 100
                if dd_pct <= MDD_FULL_BELOW:
                    mdd_scale = 1.0
                elif dd_pct >= MDD_MIN_ABOVE:
                    mdd_scale = MDD_MIN_SCALE
                else:
                    mdd_scale = 1.0 - (1.0 - MDD_MIN_SCALE) * (
                        dd_pct - MDD_FULL_BELOW) / (MDD_MIN_ABOVE - MDD_FULL_BELOW)
                sm *= mdd_scale

            # ATR scaling
            if (atr_scaling_enabled and bar < len(atr_ratio)
                    and not np.isnan(atr_ratio[bar])):
                r = clamp(atr_ratio[bar], ATR_CLAMP_LO, ATR_CLAMP_HI)
            else:
                r = 1.0

            eff_tp = tp_pct * r + SLIPPAGE_BUFFER
            eff_sl = max(0.1, sl_pct * r - SLIPPAGE_BUFFER)

            # Aggregate risk cap
            if agg_risk_enabled:
                slope = ema_slope[bar] if bar < len(ema_slope) else 0
                is_uptrend = slope > 0
                is_counter = ((direction == 'SHORT' and is_uptrend) or
                              (direction == 'LONG' and not is_uptrend))
                cap_pct = AGG_RISK_COUNTER if is_counter else AGG_RISK_WITH

                existing = sum(
                    p['eff_sl_pct'] * (1.0 / N_SLOTS) * LEVERAGE * p['size_mult']
                    for p in positions if p['direction'] == direction
                )
                new_exp = eff_sl * (1.0 / N_SLOTS) * LEVERAGE * sm
                if existing + new_exp > cap_pct:
                    continue

            positions.append({
                'slot': f"{pat}_{bar}",
                'entry_bar': entry_bar,
                'entry_price': entry_price,
                'direction': direction,
                'pattern': pat,
                'tp_pct': tp_pct,
                'sl_pct': sl_pct,
                'eff_tp_pct': eff_tp,
                'eff_sl_pct': eff_sl,
                'size_mult': sm,
            })

    # Force-close remaining (as END, not DROP)
    for pos in positions:
        eb = pos['entry_bar']
        if eb >= n_bars:
            continue
        ep = pos['entry_price']
        if ep <= 0:
            continue
        exit_bar = min(end_bar - 1, n_bars - 1)
        exit_price = opens[exit_bar]
        if pos['direction'] == 'LONG':
            pnl = (exit_price / ep - 1) * 100 * LEVERAGE
        else:
            pnl = (1 - exit_price / ep) * 100 * LEVERAGE
        pnl -= fee
        sm = pos['size_mult']
        trades.append({
            'entry_bar': eb, 'exit_bar': exit_bar, 'pnl_slot': pnl,
            'reason': 'END', 'pattern': pos['pattern'],
            'direction': pos['direction'], 'size_mult': sm,
            'pnl_portfolio': pnl * (size_pct / 100) * sm, 'leverage': LEVERAGE,
        })

    # Collect per-trade portfolio PnL array (for bootstrap)
    pnl_array = np.array([t['pnl_portfolio'] for t in trades]) if trades else np.array([])

    return trades, pnl_array


# ============================================================
# Stats helpers
# ============================================================

def calc_stats(trades):
    """Compute PnL, MDD, WR from trades list."""
    if not trades:
        return {'trades': 0, 'wr': 0.0, 'pnl': 0.0, 'mdd': 0.0, 'pnl_mdd': 0.0}

    wins = sum(1 for t in trades if t['pnl_slot'] > 0)
    sorted_t = sorted(trades, key=lambda x: x['entry_bar'])
    eq = 100.0
    pk = eq
    mdd = 0.0
    for t in sorted_t:
        eq *= (1 + t['pnl_portfolio'] / 100)
        if eq > pk:
            pk = eq
        d = (pk - eq) / pk * 100 if pk > 0 else 0
        if d > mdd:
            mdd = d

    total_pnl = eq - 100.0
    wr = wins / len(trades) * 100 if trades else 0

    return {
        'trades': len(trades),
        'wr': round(wr, 2),
        'pnl': round(total_pnl, 2),
        'mdd': round(mdd, 2),
        'pnl_mdd': round(total_pnl / mdd, 2) if mdd > 0 else 0.0,
    }


def bootstrap_ci_delta(pnl_full, pnl_ablated, n_boot=BOOTSTRAP_N,
                        seed=BOOTSTRAP_SEED, ci_pct=BOOTSTRAP_CI):
    """Bootstrap CI on the PnL delta (full - ablated).

    Resamples per-trade PnL arrays independently, computes sum PnL for each,
    reports CI on the difference.
    """
    rng = np.random.RandomState(seed)
    n_full = len(pnl_full)
    n_abl = len(pnl_ablated)
    if n_full == 0 or n_abl == 0:
        return 0.0, 0.0, False

    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx_f = rng.randint(0, n_full, size=n_full)
        idx_a = rng.randint(0, n_abl, size=n_abl)
        deltas[i] = pnl_full[idx_f].sum() - pnl_ablated[idx_a].sum()

    alpha = (100 - ci_pct) / 2
    ci_lo = np.percentile(deltas, alpha)
    ci_hi = np.percentile(deltas, 100 - alpha)
    significant = (ci_lo > 0) or (ci_hi < 0)  # CI doesn't include 0
    return round(ci_lo, 2), round(ci_hi, 2), significant


# ============================================================
# AUDIT 1: Mechanism Ablation with Bootstrap CI
# ============================================================

def run_audit1(signal_tuples, opens, highs, lows, closes, type_codes,
               n_bars, atr_ratio, ema_slope, start_bar, end_bar):
    """Run full_stack baseline then ablate each of 8 mechanisms."""
    print("\n[AUDIT 1] Mechanism Ablation with Bootstrap CI")
    print("-" * 78)

    mechanisms = [
        ('G1 Direction Cap',   'direction_cap_enabled'),
        ('G2 Momentum Guard',  'momentum_enabled'),
        ('G5 Aggregate Risk',  'agg_risk_enabled'),
        ('M1 MDD Sizing',      'mdd_sizing_enabled'),
        ('M5 ATR Scaling',     'atr_scaling_enabled'),
        ('P1 Timeout',         'timeout_enabled'),
        ('P2 Early Exit',      'early_exit_enabled'),
        ('P3 Cascade SL',      'cascade_enabled'),
    ]

    common_args = dict(
        signal_tuples=signal_tuples, opens=opens, highs=highs, lows=lows,
        closes=closes, type_codes=type_codes, n_bars=n_bars,
        atr_ratio=atr_ratio, ema_slope=ema_slope,
        start_bar=start_bar, end_bar=end_bar,
    )

    # Full stack baseline
    print("  Running full_stack (all 8 ON)...")
    full_trades, full_pnl_arr = portfolio_sim(**common_args)
    full_stats = calc_stats(full_trades)
    print(f"    PnL={full_stats['pnl']:+.2f}%, MDD={full_stats['mdd']:.2f}%, "
          f"WR={full_stats['wr']:.1f}%, Trades={full_stats['trades']}")

    results = {
        'full_stack': full_stats,
        'ablations': {},
    }

    print(f"\n  {'Mechanism':<22} {'FullPnL':>8} {'AblPnL':>8} {'Delta':>8} "
          f"{'CI_Lo':>8} {'CI_Hi':>8} {'Sig?':>6}")
    print("  " + "-" * 76)

    for name, flag_key in mechanisms:
        # Ablate: set this mechanism OFF, rest ON
        kwargs = {flag: True for _, flag in mechanisms}
        kwargs[flag_key] = False
        kwargs.update(common_args)

        print(f"  Ablating {name}...", end='', flush=True)
        abl_trades, abl_pnl_arr = portfolio_sim(**kwargs)
        abl_stats = calc_stats(abl_trades)

        delta = full_stats['pnl'] - abl_stats['pnl']
        ci_lo, ci_hi, sig = bootstrap_ci_delta(full_pnl_arr, abl_pnl_arr)
        sig_str = "YES" if sig else "NO"

        results['ablations'][name] = {
            'ablated_stats': abl_stats,
            'delta_pnl': round(delta, 2),
            'ci_lo': ci_lo,
            'ci_hi': ci_hi,
            'significant': sig,
        }

        print(f"\r  {name:<22} {full_stats['pnl']:>+8.2f} {abl_stats['pnl']:>+8.2f} "
              f"{delta:>+8.2f} {ci_lo:>+8.2f} {ci_hi:>+8.2f} {sig_str:>6}")

    return results


# ============================================================
# AUDIT 2: Core Pattern Strategy vs Random Baseline
# ============================================================

def run_audit2(signal_tuples, opens, highs, lows, closes, type_codes,
               n_bars, atr_ratio, ema_slope, start_bar, end_bar):
    """Compare real patterns vs random entry baselines."""
    print("\n[AUDIT 2] Pattern Strategy vs Random Baseline")
    print("-" * 78)

    common_args = dict(
        opens=opens, highs=highs, lows=lows, closes=closes,
        type_codes=type_codes, n_bars=n_bars,
        atr_ratio=atr_ratio, ema_slope=ema_slope,
        start_bar=start_bar, end_bar=end_bar,
    )

    # Real patterns PnL
    real_trades, _ = portfolio_sim(signal_tuples=signal_tuples, **common_args)
    real_stats = calc_stats(real_trades)
    real_pnl = real_stats['pnl']
    print(f"  Real patterns PnL: {real_pnl:+.2f}%  (trades={real_stats['trades']})")

    # Gather real signal stats for building random baselines
    neutral_signals = [(s, p, d, tp, sl) for s, p, d, tp, sl in signal_tuples
                       if start_bar <= s < end_bar]
    n_signals = len(neutral_signals)
    n_long = sum(1 for _, _, d, _, _ in neutral_signals if d == 'LONG')
    n_short = n_signals - n_long
    long_frac = n_long / max(1, n_signals)

    # Average TP/SL across all real patterns
    avg_tp = np.mean([tp for _, _, _, tp, _ in neutral_signals])
    avg_sl = np.mean([sl for _, _, _, _, sl in neutral_signals])
    print(f"  Real signals: {n_signals} ({n_long}L + {n_short}S), "
          f"avg TP={avg_tp:.2f}%, avg SL={avg_sl:.2f}%")

    results = {'real_pnl': real_pnl, 'real_trades': real_stats['trades']}

    # --- Baseline 1: random_same_frequency ---
    print("\n  [2a] random_same_frequency (100 seeds)...", end='', flush=True)
    random_pnls_freq = []
    window_bars = list(range(start_bar, end_bar))

    for seed in range(RANDOM_SEEDS):
        rng = np.random.RandomState(seed)
        # Generate n_signals random entry bars, uniformly distributed
        rand_bars = sorted(rng.choice(window_bars, size=n_signals, replace=True))
        # Assign direction with same L/S ratio
        rand_dirs = rng.choice(['LONG', 'SHORT'], size=n_signals,
                               p=[long_frac, 1 - long_frac])
        rand_signals = []
        for i in range(n_signals):
            rand_signals.append((rand_bars[i], f"RAND_{i}", rand_dirs[i], avg_tp, avg_sl))
        trades_r, _ = portfolio_sim(signal_tuples=rand_signals, **common_args)
        stats_r = calc_stats(trades_r)
        random_pnls_freq.append(stats_r['pnl'])

    mean_freq = np.mean(random_pnls_freq)
    p_freq = np.mean([1 for p in random_pnls_freq if p >= real_pnl]) / RANDOM_SEEDS
    results['random_same_frequency'] = {
        'mean_pnl': round(mean_freq, 2),
        'std_pnl': round(np.std(random_pnls_freq), 2),
        'p_value': round(p_freq, 4),
        'distribution': [round(p, 2) for p in sorted(random_pnls_freq)],
    }
    print(f" mean={mean_freq:+.2f}%, p={p_freq:.4f}")

    # --- Baseline 2: random_shuffled_direction ---
    print("  [2b] random_shuffled_direction (100 seeds)...", end='', flush=True)
    random_pnls_dir = []
    base_bars = [s[0] for s in neutral_signals]
    base_pats = [s[1] for s in neutral_signals]
    base_tps = [s[3] for s in neutral_signals]
    base_sls = [s[4] for s in neutral_signals]

    for seed in range(RANDOM_SEEDS):
        rng = np.random.RandomState(seed)
        # Shuffle direction, keep same timing and TP/SL
        shuffled_dirs = rng.choice(['LONG', 'SHORT'], size=n_signals,
                                   p=[long_frac, 1 - long_frac])
        rand_signals = []
        for i in range(n_signals):
            rand_signals.append((base_bars[i], f"SHUF_{i}", shuffled_dirs[i],
                                 base_tps[i], base_sls[i]))
        trades_r, _ = portfolio_sim(signal_tuples=rand_signals, **common_args)
        stats_r = calc_stats(trades_r)
        random_pnls_dir.append(stats_r['pnl'])

    mean_dir = np.mean(random_pnls_dir)
    p_dir = np.mean([1 for p in random_pnls_dir if p >= real_pnl]) / RANDOM_SEEDS
    results['random_shuffled_direction'] = {
        'mean_pnl': round(mean_dir, 2),
        'std_pnl': round(np.std(random_pnls_dir), 2),
        'p_value': round(p_dir, 4),
    }
    print(f" mean={mean_dir:+.2f}%, p={p_dir:.4f}")

    # --- Baseline 3: random_shuffled_tpsl ---
    print("  [2c] random_shuffled_tpsl (100 seeds)...", end='', flush=True)
    random_pnls_tpsl = []
    base_dirs = [s[2] for s in neutral_signals]

    for seed in range(RANDOM_SEEDS):
        rng = np.random.RandomState(seed)
        # Shuffle TP/SL assignments across signals (keep timing + direction)
        indices = rng.permutation(n_signals)
        rand_signals = []
        for i in range(n_signals):
            rand_signals.append((base_bars[i], f"TPSL_{i}", base_dirs[i],
                                 base_tps[indices[i]], base_sls[indices[i]]))
        trades_r, _ = portfolio_sim(signal_tuples=rand_signals, **common_args)
        stats_r = calc_stats(trades_r)
        random_pnls_tpsl.append(stats_r['pnl'])

    mean_tpsl = np.mean(random_pnls_tpsl)
    p_tpsl = np.mean([1 for p in random_pnls_tpsl if p >= real_pnl]) / RANDOM_SEEDS
    results['random_shuffled_tpsl'] = {
        'mean_pnl': round(mean_tpsl, 2),
        'std_pnl': round(np.std(random_pnls_tpsl), 2),
        'p_value': round(p_tpsl, 4),
    }
    print(f" mean={mean_tpsl:+.2f}%, p={p_tpsl:.4f}")

    # Print summary
    print(f"\n  {'Baseline':<25} {'Mean PnL':>10} {'p-value':>10}")
    print(f"  {'-'*50}")
    print(f"  {'Real patterns':<25} {real_pnl:>+10.2f} {'---':>10}")
    print(f"  {'Random same-freq':<25} {mean_freq:>+10.2f} {p_freq:>10.4f}")
    print(f"  {'Random shuffle-dir':<25} {mean_dir:>+10.2f} {p_dir:>10.4f}")
    print(f"  {'Random shuffle-tpsl':<25} {mean_tpsl:>+10.2f} {p_tpsl:>10.4f}")

    return results


# ============================================================
# AUDIT 3: Temporal Decay (H1/H2 train/test)
# ============================================================

def run_audit3(signal_tuples, opens, highs, lows, closes, type_codes,
               n_bars, atr_ratio, ema_slope, start_bar, end_bar):
    """Split neutral window in half. Test IS vs OOS performance."""
    print("\n[AUDIT 3] Temporal Decay — H1/H2 Train/Test")
    print("-" * 78)

    mid_bar = (start_bar + end_bar) // 2
    h1_start, h1_end = start_bar, mid_bar
    h2_start, h2_end = mid_bar, end_bar

    n_h1 = h1_end - h1_start
    n_h2 = h2_end - h2_start
    print(f"  H1: bars {h1_start}-{h1_end} ({n_h1} bars, {n_h1/BARS_PER_DAY:.0f}d)")
    print(f"  H2: bars {h2_start}-{h2_end} ({n_h2} bars, {n_h2/BARS_PER_DAY:.0f}d)")

    common_args = dict(
        signal_tuples=signal_tuples, opens=opens, highs=highs, lows=lows,
        closes=closes, type_codes=type_codes, n_bars=n_bars,
        atr_ratio=atr_ratio, ema_slope=ema_slope,
    )

    # H1 as IS, H2 as OOS
    print("  Running H1 (IS)...")
    trades_h1, _ = portfolio_sim(start_bar=h1_start, end_bar=h1_end, **common_args)
    stats_h1 = calc_stats(trades_h1)

    print("  Running H2 (OOS for H1-discovered patterns)...")
    trades_h2, _ = portfolio_sim(start_bar=h2_start, end_bar=h2_end, **common_args)
    stats_h2 = calc_stats(trades_h2)

    # Ratio
    ratio_h1_h2 = stats_h2['pnl'] / stats_h1['pnl'] if stats_h1['pnl'] != 0 else 0

    print(f"\n  H1->H2:")
    print(f"    IS (H1):  PnL={stats_h1['pnl']:+.2f}%, WR={stats_h1['wr']:.1f}%, "
          f"MDD={stats_h1['mdd']:.2f}%, Trades={stats_h1['trades']}")
    print(f"    OOS (H2): PnL={stats_h2['pnl']:+.2f}%, WR={stats_h2['wr']:.1f}%, "
          f"MDD={stats_h2['mdd']:.2f}%, Trades={stats_h2['trades']}")
    print(f"    OOS/IS ratio: {ratio_h1_h2:.3f}")

    results = {
        'h1_stats': stats_h1,
        'h2_stats': stats_h2,
        'h1_to_h2_ratio': round(ratio_h1_h2, 4),
    }

    return results


# ============================================================
# AUDIT 4: Pattern-Level Consistency (binomial test)
# ============================================================

def run_audit4(signal_tuples, opens, highs, lows, closes, type_codes,
               n_bars, atr_ratio, ema_slope, start_bar, end_bar,
               pat_lookup):
    """Test each of 130 patterns individually for WR > 50%."""
    print("\n[AUDIT 4] Pattern-Level Significance (Binomial Test)")
    print("-" * 78)

    # Run full stack sim to get per-trade results
    common_args = dict(
        signal_tuples=signal_tuples, opens=opens, highs=highs, lows=lows,
        closes=closes, type_codes=type_codes, n_bars=n_bars,
        atr_ratio=atr_ratio, ema_slope=ema_slope,
        start_bar=start_bar, end_bar=end_bar,
    )
    trades, _ = portfolio_sim(**common_args)

    # For individual pattern stats we run a simpler approach:
    # backtest each pattern individually (1-pos, no portfolio effects)
    # to get clean per-pattern WR without portfolio interaction artifacts.
    fee = FEE_PCT * LEVERAGE

    neutral_signals = [(s, p, d, tp, sl) for s, p, d, tp, sl in signal_tuples
                       if start_bar <= s < end_bar]

    pattern_stats = {}
    for pat_name, pat_info in pat_lookup.items():
        direction = pat_info['direction']
        tp_base = pat_info['tp']
        sl_base = pat_info['sl']

        # Gather signals for this pattern in neutral window
        pat_signals = [(s, p, d, tp, sl) for s, p, d, tp, sl in neutral_signals
                       if p == pat_name]
        if not pat_signals:
            continue

        wins = 0
        losses = 0
        pnl_sum = 0.0

        for sig_bar, _, _, tp_pct, sl_pct in pat_signals:
            entry_bar = sig_bar + 1
            if entry_bar >= n_bars:
                continue
            entry_p = opens[entry_bar]
            if entry_p <= 0:
                continue

            # ATR scaling
            if sig_bar < len(atr_ratio) and not np.isnan(atr_ratio[sig_bar]):
                r = clamp(atr_ratio[sig_bar], ATR_CLAMP_LO, ATR_CLAMP_HI)
            else:
                r = 1.0
            eff_tp = tp_pct * r + SLIPPAGE_BUFFER
            eff_sl = max(0.1, sl_pct * r - SLIPPAGE_BUFFER)

            if direction == 'LONG':
                tp_price = entry_p * (1 + eff_tp / 100)
                sl_price = entry_p * (1 - eff_sl / 100)
            else:
                tp_price = entry_p * (1 - eff_tp / 100)
                sl_price = entry_p * (1 + eff_sl / 100)

            # Walk forward from entry bar
            resolved = False
            for b in range(entry_bar, min(entry_bar + TIMEOUT_BARS, n_bars)):
                h, l = highs[b], lows[b]
                o = opens[b]
                if direction == 'LONG':
                    hit_tp = h >= tp_price
                    hit_sl = l <= sl_price
                else:
                    hit_tp = l <= tp_price
                    hit_sl = h >= sl_price

                if hit_tp and hit_sl:
                    if abs(tp_price - o) <= abs(sl_price - o):
                        pnl = ((tp_price / entry_p - 1) * 100 * LEVERAGE
                               if direction == 'LONG' else
                               (1 - tp_price / entry_p) * 100 * LEVERAGE)
                        pnl -= fee
                        wins += 1
                        pnl_sum += pnl
                    else:
                        pnl = ((sl_price / entry_p - 1) * 100 * LEVERAGE
                               if direction == 'LONG' else
                               (1 - sl_price / entry_p) * 100 * LEVERAGE)
                        pnl -= fee
                        losses += 1
                        pnl_sum += pnl
                    resolved = True
                    break
                elif hit_tp:
                    pnl = ((tp_price / entry_p - 1) * 100 * LEVERAGE
                           if direction == 'LONG' else
                           (1 - tp_price / entry_p) * 100 * LEVERAGE)
                    pnl -= fee
                    wins += 1
                    pnl_sum += pnl
                    resolved = True
                    break
                elif hit_sl:
                    pnl = ((sl_price / entry_p - 1) * 100 * LEVERAGE
                           if direction == 'LONG' else
                           (1 - sl_price / entry_p) * 100 * LEVERAGE)
                    pnl -= fee
                    losses += 1
                    pnl_sum += pnl
                    resolved = True
                    break
            # If not resolved = timeout = DROP (excluded)

        total = wins + losses
        if total == 0:
            continue

        wr = wins / total * 100
        avg_pnl = pnl_sum / total

        # Binomial test: is WR significantly > 50%?
        binom_result = scipy_stats.binomtest(wins, total, 0.5, alternative='greater')
        p_val = binom_result.pvalue

        pattern_stats[pat_name] = {
            'direction': direction,
            'total_trades': total,
            'wins': wins,
            'losses': losses,
            'wr': round(wr, 2),
            'avg_pnl': round(avg_pnl, 3),
            'p_value': round(p_val, 6),
        }

    # Summary
    n_total = len(pattern_stats)
    n_sig_05 = sum(1 for s in pattern_stats.values() if s['p_value'] < 0.05)
    n_sig_01 = sum(1 for s in pattern_stats.values() if s['p_value'] < 0.01)
    n_not_sig = n_total - n_sig_05
    n_wr_below_50 = sum(1 for s in pattern_stats.values() if s['wr'] < 50)

    print(f"  Total patterns tested: {n_total}")
    print(f"  Significant at p<0.05: {n_sig_05} ({n_sig_05/max(1,n_total)*100:.1f}%)")
    print(f"  Significant at p<0.01: {n_sig_01} ({n_sig_01/max(1,n_total)*100:.1f}%)")
    print(f"  Not significant:       {n_not_sig} ({n_not_sig/max(1,n_total)*100:.1f}%)")
    print(f"  Patterns with WR < 50%: {n_wr_below_50}")

    # Show worst 10 patterns by p-value
    worst = sorted(pattern_stats.items(), key=lambda x: -x[1]['p_value'])[:10]
    print(f"\n  Top 10 weakest patterns (highest p-value):")
    print(f"  {'Pattern':<18} {'Dir':>5} {'Trades':>7} {'WR%':>6} {'AvgPnL':>8} {'p-value':>9}")
    for name, s in worst:
        print(f"  {name:<18} {s['direction']:>5} {s['total_trades']:>7} "
              f"{s['wr']:>6.1f} {s['avg_pnl']:>+8.3f} {s['p_value']:>9.5f}")

    results = {
        'n_patterns_tested': n_total,
        'significant_p05': n_sig_05,
        'significant_p01': n_sig_01,
        'not_significant': n_not_sig,
        'wr_below_50': n_wr_below_50,
        'pattern_details': pattern_stats,
    }

    return results


# ============================================================
# AUDIT 5: Full Stack vs Bare (no mechanisms)
# ============================================================

def run_audit5(signal_tuples, opens, highs, lows, closes, type_codes,
               n_bars, atr_ratio, ema_slope, start_bar, end_bar):
    """Compare full stack (all 8 ON) vs bare (all 8 OFF)."""
    print("\n[AUDIT 5] Full Stack vs Bare Strategy")
    print("-" * 78)

    common_args = dict(
        signal_tuples=signal_tuples, opens=opens, highs=highs, lows=lows,
        closes=closes, type_codes=type_codes, n_bars=n_bars,
        atr_ratio=atr_ratio, ema_slope=ema_slope,
        start_bar=start_bar, end_bar=end_bar,
    )

    # Full stack
    print("  Running full_stack (all 8 ON)...")
    full_trades, full_pnl = portfolio_sim(**common_args)
    full_stats = calc_stats(full_trades)

    # Bare: all 8 mechanisms OFF
    print("  Running bare (all 8 OFF — just N=9 compound + patterns)...")
    bare_trades, bare_pnl = portfolio_sim(
        direction_cap_enabled=False,
        momentum_enabled=False,
        agg_risk_enabled=False,
        mdd_sizing_enabled=False,
        atr_scaling_enabled=False,
        timeout_enabled=False,
        early_exit_enabled=False,
        cascade_enabled=False,
        **common_args,
    )
    bare_stats = calc_stats(bare_trades)

    delta = full_stats['pnl'] - bare_stats['pnl']
    ci_lo, ci_hi, sig = bootstrap_ci_delta(full_pnl, bare_pnl)

    print(f"\n  {'Metric':<15} {'Full Stack':>12} {'Bare':>12} {'Delta':>10}")
    print(f"  {'-'*55}")
    for key in ['pnl', 'mdd', 'wr', 'trades', 'pnl_mdd']:
        fv = full_stats[key]
        bv = bare_stats[key]
        d = fv - bv
        if isinstance(fv, float):
            print(f"  {key:<15} {fv:>+12.2f} {bv:>+12.2f} {d:>+10.2f}")
        else:
            print(f"  {key:<15} {fv:>12} {bv:>12} {d:>+10}")

    print(f"\n  Bootstrap 95% CI on PnL delta: [{ci_lo:+.2f}%, {ci_hi:+.2f}%]")
    print(f"  Significant: {'YES' if sig else 'NO'}")

    results = {
        'full_stack': full_stats,
        'bare': bare_stats,
        'delta_pnl': round(delta, 2),
        'ci_lo': ci_lo,
        'ci_hi': ci_hi,
        'significant': sig,
    }
    return results


# ============================================================
# AUDIT 6: Temporal Consistency (6-period split)
# ============================================================

def run_audit6(signal_tuples, opens, highs, lows, closes, type_codes,
               n_bars, atr_ratio, ema_slope, start_bar, end_bar):
    """Split neutral window into 6 periods. Run full stack on each."""
    print("\n[AUDIT 6] Temporal Consistency — 6-Period Split")
    print("-" * 78)

    n_periods = 6
    total_bars = end_bar - start_bar
    seg_size = total_bars // n_periods

    common_args = dict(
        signal_tuples=signal_tuples, opens=opens, highs=highs, lows=lows,
        closes=closes, type_codes=type_codes, n_bars=n_bars,
        atr_ratio=atr_ratio, ema_slope=ema_slope,
    )

    period_results = []

    print(f"  {'Period':<8} {'Days':>6} {'PnL%':>8} {'MDD%':>7} {'WR%':>6} {'Trades':>7}")
    print(f"  {'-'*50}")

    for p in range(n_periods):
        p_start = start_bar + p * seg_size
        p_end = start_bar + (p + 1) * seg_size if p < n_periods - 1 else end_bar
        days = (p_end - p_start) / BARS_PER_DAY

        trades_p, _ = portfolio_sim(start_bar=p_start, end_bar=p_end, **common_args)
        stats_p = calc_stats(trades_p)
        stats_p['period'] = p + 1
        stats_p['days'] = round(days, 1)
        period_results.append(stats_p)

        print(f"  P{p+1:<6} {days:>6.1f} {stats_p['pnl']:>+8.2f} {stats_p['mdd']:>7.2f} "
              f"{stats_p['wr']:>6.1f} {stats_p['trades']:>7}")

    # Summary
    profitable = sum(1 for s in period_results if s['pnl'] > 0)
    pnl_std = np.std([s['pnl'] for s in period_results])
    pnl_mean = np.mean([s['pnl'] for s in period_results])

    print(f"\n  Profitable periods: {profitable}/{n_periods}")
    print(f"  PnL mean: {pnl_mean:+.2f}%, std: {pnl_std:.2f}%")
    print(f"  Consistency ratio (mean/std): {abs(pnl_mean)/max(0.01, pnl_std):.2f}")

    results = {
        'periods': period_results,
        'profitable_count': profitable,
        'total_periods': n_periods,
        'pnl_mean': round(pnl_mean, 2),
        'pnl_std': round(pnl_std, 2),
    }
    return results


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 78)
    print("PRODUCTION STACK AUDIT -- v1.42.0")
    print("=" * 78)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Settings: LEVERAGE={LEVERAGE} (fixed), N={N_SLOTS}, "
          f"DirCap={DIRECTION_CAP}, Timeout={TIMEOUT_BARS}")
    print(f"Bootstrap: {BOOTSTRAP_N} resamples, {BOOTSTRAP_CI}% CI, seed={BOOTSTRAP_SEED}")
    print(f"Random baselines: {RANDOM_SEEDS} seeds")

    # ---- Load Data ----
    print("\n[0] Loading data...")
    df = load_and_classify(DATA_FILE)
    n_bars = len(df)
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    type_codes = df['rctype'].values

    print("  Computing indicators...")
    atr_ratio = compute_atr_ratio(df)
    ema_slope = compute_ema_slope(closes)

    neutral_start, neutral_end = find_neutral_window(closes)
    n_neutral = neutral_end - neutral_start
    print(f"  Neutral window: bars {neutral_start}-{neutral_end} "
          f"({n_neutral} bars, {n_neutral / BARS_PER_DAY:.0f}d)")

    # ---- Load Patterns ----
    print("\n  Loading patterns...")
    with open(PATTERNS_FILE) as f:
        pat_data = json.load(f)
    pats_raw = pat_data['patterns']
    tpsl = pat_data.get('patterns_tpsl', {})

    pat_lookup = {}
    for pat_name in pats_raw.get('long', []):
        tp_sl = tpsl.get(pat_name, [2.0, 3.0])
        pat_lookup[pat_name] = {'direction': 'LONG', 'tp': tp_sl[0], 'sl': tp_sl[1]}
    for pat_name in pats_raw.get('short', []):
        tp_sl = tpsl.get(pat_name, [2.0, 3.0])
        pat_lookup[pat_name] = {'direction': 'SHORT', 'tp': tp_sl[0], 'sl': tp_sl[1]}
    n_long = len(pats_raw.get('long', []))
    n_short = len(pats_raw.get('short', []))
    print(f"  {len(pat_lookup)} patterns ({n_long}L + {n_short}S)")

    # ---- Build Signals ----
    print("  Building signal index...")
    rctypes = df['rctype'].values
    signal_tuples = []
    for i in range(2, n_bars):
        tri = f"{rctypes[i-2]}-{rctypes[i-1]}-{rctypes[i]}"
        if tri in pat_lookup:
            p = pat_lookup[tri]
            signal_tuples.append((i, tri, p['direction'], p['tp'], p['sl']))

    n_in_neutral = sum(1 for s in signal_tuples if neutral_start <= s[0] < neutral_end)
    print(f"  {len(signal_tuples)} total signals, {n_in_neutral} in neutral window")

    all_results = {
        'meta': {
            'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'version': 'v1.42.0',
            'data_bars': n_bars,
            'neutral_start': int(neutral_start),
            'neutral_end': int(neutral_end),
            'neutral_days': round(n_neutral / BARS_PER_DAY, 1),
            'patterns': len(pat_lookup),
            'patterns_long': n_long,
            'patterns_short': n_short,
            'signals_in_neutral': n_in_neutral,
            'leverage': LEVERAGE,
            'n_slots': N_SLOTS,
            'direction_cap': DIRECTION_CAP,
        }
    }

    # ================================================================
    # AUDIT 1: Mechanism Ablation
    # ================================================================
    audit1 = run_audit1(signal_tuples, opens, highs, lows, closes, type_codes,
                        n_bars, atr_ratio, ema_slope, neutral_start, neutral_end)
    all_results['audit1_ablation'] = audit1

    # ================================================================
    # AUDIT 2: Pattern Strategy vs Random
    # ================================================================
    audit2 = run_audit2(signal_tuples, opens, highs, lows, closes, type_codes,
                        n_bars, atr_ratio, ema_slope, neutral_start, neutral_end)
    all_results['audit2_random'] = audit2

    # ================================================================
    # AUDIT 3: Temporal Decay
    # ================================================================
    audit3 = run_audit3(signal_tuples, opens, highs, lows, closes, type_codes,
                        n_bars, atr_ratio, ema_slope, neutral_start, neutral_end)
    all_results['audit3_decay'] = audit3

    # ================================================================
    # AUDIT 4: Pattern-Level Significance
    # ================================================================
    audit4 = run_audit4(signal_tuples, opens, highs, lows, closes, type_codes,
                        n_bars, atr_ratio, ema_slope, neutral_start, neutral_end,
                        pat_lookup)
    all_results['audit4_patterns'] = {
        'n_patterns_tested': audit4['n_patterns_tested'],
        'significant_p05': audit4['significant_p05'],
        'significant_p01': audit4['significant_p01'],
        'not_significant': audit4['not_significant'],
        'wr_below_50': audit4['wr_below_50'],
        # Store per-pattern detail separately (large)
        'pattern_details': audit4['pattern_details'],
    }

    # ================================================================
    # AUDIT 5: Full Stack vs Bare
    # ================================================================
    audit5 = run_audit5(signal_tuples, opens, highs, lows, closes, type_codes,
                        n_bars, atr_ratio, ema_slope, neutral_start, neutral_end)
    all_results['audit5_stack_vs_bare'] = audit5

    # ================================================================
    # AUDIT 6: Temporal Consistency
    # ================================================================
    audit6 = run_audit6(signal_tuples, opens, highs, lows, closes, type_codes,
                        n_bars, atr_ratio, ema_slope, neutral_start, neutral_end)
    all_results['audit6_temporal'] = audit6

    # ================================================================
    # OVERALL VERDICT
    # ================================================================
    print("\n" + "=" * 78)
    print("OVERALL AUDIT VERDICT")
    print("=" * 78)

    # Summarize key findings
    full_pnl = audit1['full_stack']['pnl']
    full_mdd = audit1['full_stack']['mdd']
    full_wr = audit1['full_stack']['wr']
    full_trades = audit1['full_stack']['trades']

    print(f"\n  Full Stack Performance: PnL={full_pnl:+.2f}%, MDD={full_mdd:.2f}%, "
          f"WR={full_wr:.1f}%, Trades={full_trades}")

    # Ablation summary
    n_sig_ablations = sum(1 for v in audit1['ablations'].values() if v['significant'])
    n_total_ablations = len(audit1['ablations'])
    print(f"\n  [A1] Mechanism ablation: {n_sig_ablations}/{n_total_ablations} "
          f"mechanisms have statistically significant impact")
    for name, data in sorted(audit1['ablations'].items(),
                             key=lambda x: -abs(x[1]['delta_pnl'])):
        sig_mark = '*' if data['significant'] else ' '
        print(f"       {sig_mark} {name:<22} delta={data['delta_pnl']:+.2f}%  "
              f"CI=[{data['ci_lo']:+.2f}, {data['ci_hi']:+.2f}]")

    # Random baseline summary
    print(f"\n  [A2] Random baseline comparison:")
    for key in ['random_same_frequency', 'random_shuffled_direction', 'random_shuffled_tpsl']:
        p = audit2[key]['p_value']
        label = key.replace('random_', '')
        verdict = "PASS (p<0.05)" if p < 0.05 else "FAIL (p>=0.05)"
        print(f"       {label:<25} p={p:.4f}  {verdict}")

    # Temporal decay
    ratio = audit3['h1_to_h2_ratio']
    decay_verdict = ("HEALTHY" if ratio >= 0.5 else
                     "MODERATE DECAY" if ratio >= 0.2 else "SEVERE DECAY")
    if ratio < 0:
        decay_verdict = "INVERTED (H2 negative)"
    print(f"\n  [A3] Temporal decay: OOS/IS ratio = {ratio:.3f} ({decay_verdict})")

    # Pattern significance
    pct_sig = audit4['significant_p05'] / max(1, audit4['n_patterns_tested']) * 100
    pat_verdict = ("STRONG" if pct_sig >= 70 else
                   "MODERATE" if pct_sig >= 40 else "WEAK")
    print(f"\n  [A4] Pattern significance: {audit4['significant_p05']}/"
          f"{audit4['n_patterns_tested']} ({pct_sig:.1f}%) at p<0.05 ({pat_verdict})")
    print(f"       WR < 50%: {audit4['wr_below_50']} patterns")

    # Stack vs bare
    stack_sig = "SIGNIFICANT" if audit5['significant'] else "NOT SIGNIFICANT"
    print(f"\n  [A5] Full stack vs bare: delta={audit5['delta_pnl']:+.2f}%, "
          f"CI=[{audit5['ci_lo']:+.2f}, {audit5['ci_hi']:+.2f}] ({stack_sig})")

    # Temporal consistency
    prof = audit6['profitable_count']
    tot = audit6['total_periods']
    print(f"\n  [A6] Temporal consistency: {prof}/{tot} periods profitable, "
          f"PnL std={audit6['pnl_std']:.2f}%")

    # Overall verdict
    issues = []
    if n_sig_ablations < 3:
        issues.append(f"Only {n_sig_ablations}/8 mechanisms are statistically significant")
    if audit2['random_same_frequency']['p_value'] >= 0.05:
        issues.append("Strategy does NOT beat random same-frequency entries")
    if audit2['random_shuffled_direction']['p_value'] >= 0.05:
        issues.append("Strategy does NOT beat random direction shuffling")
    if ratio < 0.2:
        issues.append(f"Severe temporal decay (OOS/IS = {ratio:.3f})")
    if pct_sig < 40:
        issues.append(f"Only {pct_sig:.0f}% of patterns individually significant")
    if prof < 4:
        issues.append(f"Only {prof}/{tot} periods profitable")

    if not issues:
        verdict = "PASS -- Strategy and mechanism stack appear statistically robust"
    elif len(issues) <= 2:
        verdict = "CAUTION -- Some concerns detected"
    else:
        verdict = "FAIL -- Multiple statistical concerns"

    print(f"\n  VERDICT: {verdict}")
    if issues:
        print("  Issues:")
        for iss in issues:
            print(f"    - {iss}")

    all_results['verdict'] = {
        'overall': verdict,
        'issues': issues,
        'n_significant_mechanisms': n_sig_ablations,
        'random_beats_real': any(
            audit2[k]['p_value'] >= 0.05
            for k in ['random_same_frequency', 'random_shuffled_direction',
                       'random_shuffled_tpsl']
        ),
        'temporal_decay_ratio': round(ratio, 4),
        'pct_patterns_significant': round(pct_sig, 1),
        'profitable_periods': f"{prof}/{tot}",
    }

    # ---- Save Results ----
    print(f"\n  Saving results to {OUTPUT_FILE}...")

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            return super().default(obj)

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    print("  Done.")


if __name__ == '__main__':
    main()
