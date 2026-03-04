#!/usr/bin/env python3
"""
Stack Resolution Study
========================

Resolve ALL issues found in the production_stack_audit. This is an ACTION study.

Phase 1: Random Baseline Deep Verification (1000 seeds)
  - 3 random baselines with proper p-values (not ceiling-limited by 100 seeds)
Phase 2: Negative Expected PnL Pattern Identification
  - Find and remove harmful patterns (avg_pnl < 0 in full portfolio sim)
Phase 3: Minimal Mechanism Stack
  - Test progressively simplified mechanism stacks
Phase 4: WF 3-fold Validation of Top Candidates
Phase 5: Combined Resolution — Best Config with Clean Patterns

Standard Research Protocol:
  - Production classify_candle import
  - LEVERAGE = 3 FIXED (v1.42.0: adaptive leverage DISABLED)
  - FEE = FEE_PCT * LEVERAGE (notional basis)
  - Timeout = DROP, Same-bar = abs(tp - bar_open)
  - Entry = next-bar open, ATR-scaled TP/SL
  - Compound (multiplicative) sizing
  - WF: 3-fold expanding window

Author: Research Agent
Date: 2026-03-04
"""

import os
import sys
import json
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

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
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'stack_resolution_study.json')


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
    track_per_pattern=False,
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


def calc_compound_pnl_only(trades):
    """Fast: compute only the compound PnL (no MDD tracking)."""
    if not trades:
        return 0.0
    eq = 100.0
    for t in sorted(trades, key=lambda x: x['entry_bar']):
        eq *= (1 + t['pnl_portfolio'] / 100)
    return eq - 100.0


def bootstrap_ci_delta(pnl_full, pnl_ablated, n_boot=500, seed=42, ci_pct=95):
    """Bootstrap CI on compound PnL delta (full - ablated).

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
# PHASE 1: Random Baseline Deep Verification (1000 seeds)
# ============================================================

def run_phase1(signal_tuples, opens, highs, lows, closes, type_codes,
               n_bars, atr_ratio, ema_slope, start_bar, end_bar):
    """Run 1000-seed random baseline comparison for proper p-values."""
    print("\n[PHASE 1] Random Baseline Deep Verification (1000 seeds)")
    print("=" * 78)

    common_args = dict(
        opens=opens, highs=highs, lows=lows, closes=closes,
        type_codes=type_codes, n_bars=n_bars,
        atr_ratio=atr_ratio, ema_slope=ema_slope,
        start_bar=start_bar, end_bar=end_bar,
    )

    # Real patterns PnL (full stack)
    real_trades, _ = portfolio_sim(signal_tuples=signal_tuples, **common_args)
    real_stats = calc_stats(real_trades)
    real_pnl = real_stats['pnl']
    print(f"  Real patterns PnL: {real_pnl:+.2f}%  (trades={real_stats['trades']}, "
          f"WR={real_stats['wr']:.1f}%, MDD={real_stats['mdd']:.2f}%)")

    # Gather real signal stats for building random baselines
    neutral_signals = [(s, p, d, tp, sl) for s, p, d, tp, sl in signal_tuples
                       if start_bar <= s < end_bar]
    n_signals = len(neutral_signals)
    n_long = sum(1 for _, _, d, _, _ in neutral_signals if d == 'LONG')
    n_short = n_signals - n_long
    long_frac = n_long / max(1, n_signals)

    avg_tp = np.mean([tp for _, _, _, tp, _ in neutral_signals])
    avg_sl = np.mean([sl for _, _, _, _, sl in neutral_signals])
    print(f"  Real signals: {n_signals} ({n_long}L + {n_short}S), "
          f"avg TP={avg_tp:.2f}%, avg SL={avg_sl:.2f}%")

    n_seeds = 1000
    window_bars = list(range(start_bar, end_bar))
    results = {'real_pnl': real_pnl, 'real_trades': real_stats['trades'],
               'real_wr': real_stats['wr'], 'real_mdd': real_stats['mdd']}

    # Pre-extract base signal components for shuffled baselines
    base_bars = [s[0] for s in neutral_signals]
    base_pats = [s[1] for s in neutral_signals]
    base_dirs = [s[2] for s in neutral_signals]
    base_tps = [s[3] for s in neutral_signals]
    base_sls = [s[4] for s in neutral_signals]

    # --- Baseline 1: random_same_frequency ---
    print(f"\n  [1a] random_same_frequency ({n_seeds} seeds)...", flush=True)
    t0 = time.time()
    random_pnls_freq = []

    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        rand_bars = sorted(rng.choice(window_bars, size=n_signals, replace=True))
        rand_dirs = rng.choice(['LONG', 'SHORT'], size=n_signals,
                               p=[long_frac, 1 - long_frac])
        rand_signals = []
        for i in range(n_signals):
            rand_signals.append((rand_bars[i], f"RAND_{i}", rand_dirs[i], avg_tp, avg_sl))
        trades_r, _ = portfolio_sim(signal_tuples=rand_signals, **common_args)
        random_pnls_freq.append(calc_compound_pnl_only(trades_r))

    t1 = time.time()
    mean_freq = np.mean(random_pnls_freq)
    std_freq = np.std(random_pnls_freq)
    p_freq = np.sum(np.array(random_pnls_freq) >= real_pnl) / n_seeds
    d_freq = (real_pnl - mean_freq) / max(0.01, std_freq)  # Cohen's d
    results['random_same_frequency'] = {
        'mean_pnl': round(mean_freq, 2), 'std_pnl': round(std_freq, 2),
        'p_value': round(p_freq, 6), 'cohens_d': round(d_freq, 2),
    }
    print(f"    Done in {t1-t0:.1f}s. mean={mean_freq:+.2f}%, p={p_freq:.6f}, d={d_freq:.2f}")

    # --- Baseline 2: random_shuffled_direction ---
    print(f"  [1b] random_shuffled_direction ({n_seeds} seeds)...", flush=True)
    t0 = time.time()
    random_pnls_dir = []

    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        shuffled_dirs = rng.choice(['LONG', 'SHORT'], size=n_signals,
                                   p=[long_frac, 1 - long_frac])
        rand_signals = []
        for i in range(n_signals):
            rand_signals.append((base_bars[i], f"SHUF_{i}", shuffled_dirs[i],
                                 base_tps[i], base_sls[i]))
        trades_r, _ = portfolio_sim(signal_tuples=rand_signals, **common_args)
        random_pnls_dir.append(calc_compound_pnl_only(trades_r))

    t1 = time.time()
    mean_dir = np.mean(random_pnls_dir)
    std_dir = np.std(random_pnls_dir)
    p_dir = np.sum(np.array(random_pnls_dir) >= real_pnl) / n_seeds
    d_dir = (real_pnl - mean_dir) / max(0.01, std_dir)
    results['random_shuffled_direction'] = {
        'mean_pnl': round(mean_dir, 2), 'std_pnl': round(std_dir, 2),
        'p_value': round(p_dir, 6), 'cohens_d': round(d_dir, 2),
    }
    print(f"    Done in {t1-t0:.1f}s. mean={mean_dir:+.2f}%, p={p_dir:.6f}, d={d_dir:.2f}")

    # --- Baseline 3: random_shuffled_tpsl ---
    print(f"  [1c] random_shuffled_tpsl ({n_seeds} seeds)...", flush=True)
    t0 = time.time()
    random_pnls_tpsl = []

    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        indices = rng.permutation(n_signals)
        rand_signals = []
        for i in range(n_signals):
            rand_signals.append((base_bars[i], f"TPSL_{i}", base_dirs[i],
                                 base_tps[indices[i]], base_sls[indices[i]]))
        trades_r, _ = portfolio_sim(signal_tuples=rand_signals, **common_args)
        random_pnls_tpsl.append(calc_compound_pnl_only(trades_r))

    t1 = time.time()
    mean_tpsl = np.mean(random_pnls_tpsl)
    std_tpsl = np.std(random_pnls_tpsl)
    p_tpsl = np.sum(np.array(random_pnls_tpsl) >= real_pnl) / n_seeds
    d_tpsl = (real_pnl - mean_tpsl) / max(0.01, std_tpsl)
    results['random_shuffled_tpsl'] = {
        'mean_pnl': round(mean_tpsl, 2), 'std_pnl': round(std_tpsl, 2),
        'p_value': round(p_tpsl, 6), 'cohens_d': round(d_tpsl, 2),
    }
    print(f"    Done in {t1-t0:.1f}s. mean={mean_tpsl:+.2f}%, p={p_tpsl:.6f}, d={d_tpsl:.2f}")

    # Summary table
    print(f"\n  {'Baseline':<25} {'RealPnL':>9} {'RandMean':>10} {'RandStd':>9} "
          f"{'p-value':>10} {'Cohen d':>9}")
    print(f"  {'-'*72}")
    print(f"  {'random_same_freq':<25} {real_pnl:>+9.1f} {mean_freq:>+10.1f} "
          f"{std_freq:>9.1f} {p_freq:>10.6f} {d_freq:>9.2f}")
    print(f"  {'random_shuffle_dir':<25} {real_pnl:>+9.1f} {mean_dir:>+10.1f} "
          f"{std_dir:>9.1f} {p_dir:>10.6f} {d_dir:>9.2f}")
    print(f"  {'random_shuffle_tpsl':<25} {real_pnl:>+9.1f} {mean_tpsl:>+10.1f} "
          f"{std_tpsl:>9.1f} {p_tpsl:>10.6f} {d_tpsl:>9.2f}")

    return results


# ============================================================
# PHASE 2: Negative Expected PnL Pattern Identification
# ============================================================

def run_phase2(signal_tuples, opens, highs, lows, closes, type_codes,
               n_bars, atr_ratio, ema_slope, start_bar, end_bar, pat_lookup):
    """Identify patterns with negative avg PnL and test removal."""
    print("\n[PHASE 2] Negative Expected PnL Pattern Identification")
    print("=" * 78)

    common_args = dict(
        opens=opens, highs=highs, lows=lows, closes=closes,
        type_codes=type_codes, n_bars=n_bars,
        atr_ratio=atr_ratio, ema_slope=ema_slope,
        start_bar=start_bar, end_bar=end_bar,
    )

    # Run full stack and collect per-pattern stats
    print("  Running full_stack to collect per-pattern trade stats...")
    trades_full, _ = portfolio_sim(signal_tuples=signal_tuples, **common_args)
    full_stats = calc_stats(trades_full)

    # Aggregate per-pattern stats from the portfolio sim trades
    pat_trade_stats = {}
    for t in trades_full:
        pat = t['pattern']
        if pat not in pat_trade_stats:
            pat_trade_stats[pat] = {'trades': 0, 'wins': 0, 'pnl_sum': 0.0,
                                    'direction': t['direction']}
        pat_trade_stats[pat]['trades'] += 1
        if t['pnl_slot'] > 0:
            pat_trade_stats[pat]['wins'] += 1
        pat_trade_stats[pat]['pnl_sum'] += t['pnl_slot']

    # Compute avg_pnl for each pattern
    pattern_results = []
    for pat, stats in pat_trade_stats.items():
        n_trades = stats['trades']
        avg_pnl = stats['pnl_sum'] / max(1, n_trades)
        wr = stats['wins'] / max(1, n_trades) * 100
        pattern_results.append({
            'pattern': pat,
            'direction': stats['direction'],
            'trades': n_trades,
            'wr': round(wr, 1),
            'avg_pnl': round(avg_pnl, 3),
            'total_pnl': round(stats['pnl_sum'], 2),
        })

    # Sort by avg_pnl ascending
    pattern_results.sort(key=lambda x: x['avg_pnl'])

    # Identify harmful patterns
    harmful = [p for p in pattern_results if p['avg_pnl'] < 0]
    harmful_strict = [p for p in pattern_results if p['avg_pnl'] < -0.3]

    print(f"\n  Total patterns with trades: {len(pattern_results)}")
    print(f"  Harmful (avg_pnl < 0): {len(harmful)}")
    print(f"  Harmful strict (avg_pnl < -0.3%): {len(harmful_strict)}")

    if harmful:
        print(f"\n  Harmful patterns (avg_pnl < 0):")
        print(f"  {'Pattern':<18} {'Dir':>5} {'Trades':>7} {'WR%':>6} "
              f"{'AvgPnL':>8} {'TotalPnL':>10}")
        print(f"  {'-'*60}")
        for p in harmful:
            print(f"  {p['pattern']:<18} {p['direction']:>5} {p['trades']:>7} "
                  f"{p['wr']:>6.1f} {p['avg_pnl']:>+8.3f} {p['total_pnl']:>+10.2f}")

    # Build filtered signal sets
    harmful_set = {p['pattern'] for p in harmful}
    harmful_strict_set = {p['pattern'] for p in harmful_strict}

    sigs_full = signal_tuples
    sigs_no_neg = [(s, p, d, tp, sl) for s, p, d, tp, sl in signal_tuples
                   if p not in harmful_set]
    sigs_no_neg_strict = [(s, p, d, tp, sl) for s, p, d, tp, sl in signal_tuples
                          if p not in harmful_strict_set]

    configs = [
        ('full_130', sigs_full, len(pat_lookup)),
        ('remove_neg_pnl', sigs_no_neg, len(pat_lookup) - len(harmful_set)),
        ('remove_neg_strict', sigs_no_neg_strict,
         len(pat_lookup) - len(harmful_strict_set)),
    ]

    print(f"\n  Testing removal configurations:")
    print(f"  {'Config':<20} {'PnL%':>9} {'MDD%':>7} {'P/M':>7} {'WR%':>6} "
          f"{'Trades':>7} {'Patterns':>9}")
    print(f"  {'-'*70}")

    config_results = {}
    for name, sigs, n_pats in configs:
        trades_c, _ = portfolio_sim(signal_tuples=sigs, **common_args)
        stats_c = calc_stats(trades_c)
        config_results[name] = {**stats_c, 'n_patterns': n_pats}
        print(f"  {name:<20} {stats_c['pnl']:>+9.2f} {stats_c['mdd']:>7.2f} "
              f"{stats_c['pnl_mdd']:>7.2f} {stats_c['wr']:>6.1f} "
              f"{stats_c['trades']:>7} {n_pats:>9}")

    results = {
        'full_stats': full_stats,
        'pattern_details': pattern_results,
        'n_harmful': len(harmful),
        'n_harmful_strict': len(harmful_strict),
        'harmful_patterns': [p['pattern'] for p in harmful],
        'harmful_strict_patterns': [p['pattern'] for p in harmful_strict],
        'config_results': config_results,
    }

    return results


# ============================================================
# PHASE 3: Minimal Mechanism Stack
# ============================================================

def run_phase3(signal_tuples, opens, highs, lows, closes, type_codes,
               n_bars, atr_ratio, ema_slope, start_bar, end_bar):
    """Test progressively simplified mechanism stacks."""
    print("\n[PHASE 3] Minimal Mechanism Stack")
    print("=" * 78)

    common_args = dict(
        signal_tuples=signal_tuples, opens=opens, highs=highs, lows=lows,
        closes=closes, type_codes=type_codes, n_bars=n_bars,
        atr_ratio=atr_ratio, ema_slope=ema_slope,
        start_bar=start_bar, end_bar=end_bar,
    )

    # Mechanism toggle configurations
    # Keys: direction_cap, momentum, agg_risk, mdd_sizing, atr_scaling,
    #        timeout, early_exit, cascade
    configs = {
        'full_8': {
            'direction_cap_enabled': True, 'momentum_enabled': True,
            'agg_risk_enabled': True, 'mdd_sizing_enabled': True,
            'atr_scaling_enabled': True, 'timeout_enabled': True,
            'early_exit_enabled': True, 'cascade_enabled': True,
        },
        'only_significant': {
            'direction_cap_enabled': False, 'momentum_enabled': False,
            'agg_risk_enabled': True, 'mdd_sizing_enabled': False,
            'atr_scaling_enabled': False, 'timeout_enabled': False,
            'early_exit_enabled': False, 'cascade_enabled': True,
        },
        'g5_only': {
            'direction_cap_enabled': False, 'momentum_enabled': False,
            'agg_risk_enabled': True, 'mdd_sizing_enabled': False,
            'atr_scaling_enabled': False, 'timeout_enabled': False,
            'early_exit_enabled': False, 'cascade_enabled': False,
        },
        'g5_timeout': {
            'direction_cap_enabled': False, 'momentum_enabled': False,
            'agg_risk_enabled': True, 'mdd_sizing_enabled': False,
            'atr_scaling_enabled': False, 'timeout_enabled': True,
            'early_exit_enabled': False, 'cascade_enabled': False,
        },
        'g5_atr': {
            'direction_cap_enabled': False, 'momentum_enabled': False,
            'agg_risk_enabled': True, 'mdd_sizing_enabled': False,
            'atr_scaling_enabled': True, 'timeout_enabled': False,
            'early_exit_enabled': False, 'cascade_enabled': False,
        },
        'g5_atr_timeout': {
            'direction_cap_enabled': False, 'momentum_enabled': False,
            'agg_risk_enabled': True, 'mdd_sizing_enabled': False,
            'atr_scaling_enabled': True, 'timeout_enabled': True,
            'early_exit_enabled': False, 'cascade_enabled': False,
        },
        'g5_cascade_atr': {
            'direction_cap_enabled': False, 'momentum_enabled': False,
            'agg_risk_enabled': True, 'mdd_sizing_enabled': False,
            'atr_scaling_enabled': True, 'timeout_enabled': False,
            'early_exit_enabled': False, 'cascade_enabled': True,
        },
        'essential_4': {
            'direction_cap_enabled': False, 'momentum_enabled': False,
            'agg_risk_enabled': True, 'mdd_sizing_enabled': False,
            'atr_scaling_enabled': True, 'timeout_enabled': True,
            'early_exit_enabled': False, 'cascade_enabled': True,
        },
        'bare': {
            'direction_cap_enabled': False, 'momentum_enabled': False,
            'agg_risk_enabled': False, 'mdd_sizing_enabled': False,
            'atr_scaling_enabled': False, 'timeout_enabled': False,
            'early_exit_enabled': False, 'cascade_enabled': False,
        },
    }

    # Run all configurations
    all_trades = {}
    all_pnl_arrays = {}
    all_stats = {}

    print(f"  Running {len(configs)} configurations...", flush=True)
    for name, toggles in configs.items():
        print(f"    {name}...", end='', flush=True)
        t0 = time.time()
        trades_c, pnl_arr = portfolio_sim(**toggles, **common_args)
        stats_c = calc_stats(trades_c)
        all_trades[name] = trades_c
        all_pnl_arrays[name] = pnl_arr
        all_stats[name] = stats_c
        print(f" PnL={stats_c['pnl']:+.1f}%, MDD={stats_c['mdd']:.1f}%, "
              f"P/M={stats_c['pnl_mdd']:.1f} ({time.time()-t0:.1f}s)")

    # Bootstrap CI comparing each to bare
    print(f"\n  Bootstrap CI (500 resamples) comparing each config vs bare...")
    bare_pnl_arr = all_pnl_arrays['bare']
    ci_results = {}
    for name in configs:
        if name == 'bare':
            ci_results[name] = {'ci_lo': 0.0, 'ci_hi': 0.0, 'significant': False}
            continue
        ci_lo, ci_hi, sig = bootstrap_ci_delta(
            all_pnl_arrays[name], bare_pnl_arr, n_boot=500, seed=42)
        ci_results[name] = {'ci_lo': ci_lo, 'ci_hi': ci_hi, 'significant': sig}

    # Print summary table
    print(f"\n  {'Config':<20} {'PnL%':>9} {'MDD%':>7} {'P/M':>7} {'WR%':>6} "
          f"{'Trades':>7} {'vs-bare CI':>18} {'Sig?':>5}")
    print(f"  {'-'*82}")

    for name in configs:
        s = all_stats[name]
        ci = ci_results[name]
        ci_str = f"[{ci['ci_lo']:+.1f}, {ci['ci_hi']:+.1f}]" if name != 'bare' else "---"
        sig_str = "YES" if ci['significant'] else ("NO" if name != 'bare' else "---")
        print(f"  {name:<20} {s['pnl']:>+9.2f} {s['mdd']:>7.2f} {s['pnl_mdd']:>7.2f} "
              f"{s['wr']:>6.1f} {s['trades']:>7} {ci_str:>18} {sig_str:>5}")

    results = {
        'configs': {},
        'ranking': [],
    }
    ranked = sorted(all_stats.items(), key=lambda x: x[1]['pnl_mdd'], reverse=True)
    for name, stats in ranked:
        results['configs'][name] = {
            **stats,
            **ci_results[name],
        }
        results['ranking'].append(name)

    print(f"\n  Ranking by PnL/MDD:")
    for i, (name, stats) in enumerate(ranked):
        print(f"    {i+1}. {name:<20} PnL/MDD={stats['pnl_mdd']:.2f}")

    return results, all_stats, all_pnl_arrays, configs


# ============================================================
# PHASE 4: WF 3-fold Validation
# ============================================================

def run_phase4(signal_tuples, opens, highs, lows, closes, type_codes,
               n_bars, atr_ratio, ema_slope, neutral_start, neutral_end,
               phase3_results, phase3_stats, phase3_configs):
    """WF validation of top candidates from Phase 3."""
    print("\n[PHASE 4] WF 3-fold Validation of Top Candidates")
    print("=" * 78)

    # Select top 3 by PnL/MDD + always include full_8 and bare
    ranking = phase3_results['ranking']
    wf_configs = []
    seen = set()
    # Top 3
    for name in ranking[:3]:
        if name not in seen:
            wf_configs.append(name)
            seen.add(name)
    # Always include full_8 and bare
    for must in ['full_8', 'bare']:
        if must not in seen:
            wf_configs.append(must)
            seen.add(must)

    print(f"  WF configs: {wf_configs}")

    n_folds = 3
    total = neutral_end - neutral_start
    seg_size = total // (n_folds + 1)

    common_args = dict(
        opens=opens, highs=highs, lows=lows, closes=closes,
        type_codes=type_codes, n_bars=n_bars,
        atr_ratio=atr_ratio, ema_slope=ema_slope,
    )

    results = {}
    for config_name in wf_configs:
        toggles = phase3_configs[config_name]
        fold_results = []

        for fold in range(n_folds):
            oos_start = neutral_start + seg_size * (fold + 1)
            oos_end = (neutral_start + seg_size * (fold + 2)
                       if fold < n_folds - 1 else neutral_end)

            trades_f, _ = portfolio_sim(
                signal_tuples=signal_tuples,
                start_bar=oos_start, end_bar=oos_end,
                **toggles, **common_args)
            stats_f = calc_stats(trades_f)
            stats_f['fold'] = fold + 1
            stats_f['oos_bars'] = int(oos_end - oos_start)
            fold_results.append(stats_f)

        all_pass = all(f['pnl'] > 0 for f in fold_results)
        avg_pnl = np.mean([f['pnl'] for f in fold_results])
        min_pnl = min(f['pnl'] for f in fold_results)
        results[config_name] = {
            'folds': fold_results,
            'all_pass': all_pass,
            'n_pass': sum(1 for f in fold_results if f['pnl'] > 0),
            'avg_oos_pnl': round(avg_pnl, 2),
            'min_oos_pnl': round(min_pnl, 2),
        }

    # Print results
    print(f"\n  {'Config':<20} {'F1 PnL%':>9} {'F2 PnL%':>9} {'F3 PnL%':>9} "
          f"{'AvgOOS':>8} {'MinOOS':>8} {'PASS':>6}")
    print(f"  {'-'*75}")
    for name in wf_configs:
        r = results[name]
        folds = r['folds']
        pass_str = f"{r['n_pass']}/3 {'PASS' if r['all_pass'] else 'FAIL'}"
        print(f"  {name:<20} {folds[0]['pnl']:>+9.2f} {folds[1]['pnl']:>+9.2f} "
              f"{folds[2]['pnl']:>+9.2f} {r['avg_oos_pnl']:>+8.2f} "
              f"{r['min_oos_pnl']:>+8.2f} {pass_str:>6}")

    # Also print WR and MDD per fold
    print(f"\n  Detailed fold stats:")
    for name in wf_configs:
        r = results[name]
        print(f"\n  {name}:")
        for f in r['folds']:
            print(f"    Fold {f['fold']}: PnL={f['pnl']:+.2f}%, WR={f['wr']:.1f}%, "
                  f"MDD={f['mdd']:.2f}%, Trades={f['trades']}")

    return results


# ============================================================
# PHASE 5: Combined Resolution
# ============================================================

def run_phase5(signal_tuples, opens, highs, lows, closes, type_codes,
               n_bars, atr_ratio, ema_slope, neutral_start, neutral_end,
               phase2_results, phase3_results, phase3_stats, phase3_configs,
               phase4_results, pat_lookup):
    """Combine best mechanism stack with clean pattern set."""
    print("\n[PHASE 5] Combined Resolution -- Best Config + Clean Patterns")
    print("=" * 78)

    common_args = dict(
        opens=opens, highs=highs, lows=lows, closes=closes,
        type_codes=type_codes, n_bars=n_bars,
        atr_ratio=atr_ratio, ema_slope=ema_slope,
    )

    # Determine best mechanism stack from Phase 3/4
    # Criteria: must be WF 3/3 PASS, then highest PnL/MDD
    best_config_name = None
    best_pnl_mdd = -999

    for name, wf_data in phase4_results.items():
        if not wf_data['all_pass']:
            continue
        is_pnl_mdd = phase3_stats[name]['pnl_mdd']
        if is_pnl_mdd > best_pnl_mdd:
            best_pnl_mdd = is_pnl_mdd
            best_config_name = name

    if best_config_name is None:
        # Fallback: if no WF 3/3 pass, use best by PnL/MDD regardless
        best_config_name = phase3_results['ranking'][0]
        print(f"  WARNING: No config achieved WF 3/3 PASS. "
              f"Using best IS PnL/MDD: {best_config_name}")
    else:
        print(f"  Best WF-validated mechanism stack: {best_config_name} "
              f"(IS PnL/MDD={best_pnl_mdd:.2f})")

    best_toggles = phase3_configs[best_config_name]

    # Determine best pattern set from Phase 2
    # Choose the pattern removal config with best PnL/MDD
    p2_configs = phase2_results['config_results']
    best_pat_config = max(p2_configs.items(), key=lambda x: x[1].get('pnl_mdd', 0))
    best_pat_name = best_pat_config[0]
    print(f"  Best pattern config: {best_pat_name} "
          f"(PnL/MDD={best_pat_config[1].get('pnl_mdd', 0):.2f})")

    # Build the filtered signal set
    if best_pat_name == 'remove_neg_pnl':
        exclude_set = set(phase2_results['harmful_patterns'])
    elif best_pat_name == 'remove_neg_strict':
        exclude_set = set(phase2_results['harmful_strict_patterns'])
    else:
        exclude_set = set()

    filtered_signals = [(s, p, d, tp, sl) for s, p, d, tp, sl in signal_tuples
                        if p not in exclude_set]
    n_remaining = len(pat_lookup) - len(exclude_set)

    print(f"  Patterns: {len(pat_lookup)} -> {n_remaining} "
          f"(removed {len(exclude_set)} harmful)")

    # --- Current production baseline ---
    print(f"\n  Running current production (full_8 + 130 patterns)...")
    trades_current, _ = portfolio_sim(
        signal_tuples=signal_tuples,
        start_bar=neutral_start, end_bar=neutral_end,
        **common_args)
    stats_current = calc_stats(trades_current)

    # --- Proposed combination ---
    print(f"  Running proposed ({best_config_name} + {best_pat_name})...")
    trades_proposed, _ = portfolio_sim(
        signal_tuples=filtered_signals,
        start_bar=neutral_start, end_bar=neutral_end,
        **best_toggles, **common_args)
    stats_proposed = calc_stats(trades_proposed)

    print(f"\n  {'Metric':<12} {'Current':>12} {'Proposed':>12} {'Delta':>10}")
    print(f"  {'-'*50}")
    for key in ['pnl', 'mdd', 'pnl_mdd', 'wr', 'trades']:
        cv = stats_current[key]
        pv = stats_proposed[key]
        d = pv - cv
        if isinstance(cv, float):
            print(f"  {key:<12} {cv:>+12.2f} {pv:>+12.2f} {d:>+10.2f}")
        else:
            print(f"  {key:<12} {cv:>12} {pv:>12} {d:>+10}")

    # --- WF 3-fold for proposed ---
    print(f"\n  WF 3-fold for proposed config...")
    n_folds = 3
    total = neutral_end - neutral_start
    seg_size = total // (n_folds + 1)

    wf_folds = []
    for fold in range(n_folds):
        oos_start = neutral_start + seg_size * (fold + 1)
        oos_end = (neutral_start + seg_size * (fold + 2)
                   if fold < n_folds - 1 else neutral_end)

        trades_f, _ = portfolio_sim(
            signal_tuples=filtered_signals,
            start_bar=oos_start, end_bar=oos_end,
            **best_toggles, **common_args)
        stats_f = calc_stats(trades_f)
        stats_f['fold'] = fold + 1
        wf_folds.append(stats_f)

    all_pass = all(f['pnl'] > 0 for f in wf_folds)
    n_pass = sum(1 for f in wf_folds if f['pnl'] > 0)

    print(f"\n  WF Results for proposed ({best_config_name} + {best_pat_name}):")
    for f in wf_folds:
        status = "PASS" if f['pnl'] > 0 else "FAIL"
        print(f"    Fold {f['fold']}: PnL={f['pnl']:+.2f}%, WR={f['wr']:.1f}%, "
              f"MDD={f['mdd']:.2f}%, Trades={f['trades']} [{status}]")
    print(f"  WF Verdict: {n_pass}/3 {'PASS' if all_pass else 'FAIL'}")

    results = {
        'best_mechanism_stack': best_config_name,
        'best_mechanism_toggles': {k: v for k, v in best_toggles.items()},
        'best_pattern_config': best_pat_name,
        'patterns_removed': list(exclude_set),
        'n_patterns_remaining': n_remaining,
        'current_stats': stats_current,
        'proposed_stats': stats_proposed,
        'wf_folds': wf_folds,
        'wf_all_pass': all_pass,
        'wf_n_pass': n_pass,
    }

    return results


# ============================================================
# Main
# ============================================================

def main():
    t_start = time.time()

    print("=" * 78)
    print("STACK RESOLUTION STUDY")
    print("Resolve ALL issues found in the production_stack_audit")
    print("=" * 78)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Settings: LEVERAGE={LEVERAGE} (fixed), N={N_SLOTS}, "
          f"DirCap={DIRECTION_CAP}, Timeout={TIMEOUT_BARS}")

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
    # PHASE 1: Random Baseline Deep Verification
    # ================================================================
    phase1 = run_phase1(signal_tuples, opens, highs, lows, closes, type_codes,
                        n_bars, atr_ratio, ema_slope, neutral_start, neutral_end)
    all_results['phase1_random_baseline'] = phase1

    # ================================================================
    # PHASE 2: Negative PnL Pattern Identification
    # ================================================================
    phase2 = run_phase2(signal_tuples, opens, highs, lows, closes, type_codes,
                        n_bars, atr_ratio, ema_slope, neutral_start, neutral_end,
                        pat_lookup)
    all_results['phase2_pattern_removal'] = phase2

    # ================================================================
    # PHASE 3: Minimal Mechanism Stack
    # ================================================================
    phase3, phase3_stats, phase3_pnl_arrays, phase3_configs = run_phase3(
        signal_tuples, opens, highs, lows, closes, type_codes,
        n_bars, atr_ratio, ema_slope, neutral_start, neutral_end)
    all_results['phase3_minimal_stack'] = phase3

    # ================================================================
    # PHASE 4: WF Validation
    # ================================================================
    phase4 = run_phase4(signal_tuples, opens, highs, lows, closes, type_codes,
                        n_bars, atr_ratio, ema_slope, neutral_start, neutral_end,
                        phase3, phase3_stats, phase3_configs)
    all_results['phase4_wf_validation'] = phase4

    # ================================================================
    # PHASE 5: Combined Resolution
    # ================================================================
    phase5 = run_phase5(signal_tuples, opens, highs, lows, closes, type_codes,
                        n_bars, atr_ratio, ema_slope, neutral_start, neutral_end,
                        phase2, phase3, phase3_stats, phase3_configs, phase4,
                        pat_lookup)
    all_results['phase5_combined'] = phase5

    # ================================================================
    # RECOMMENDATION
    # ================================================================
    print("\n" + "=" * 78)
    print("RECOMMENDATION")
    print("=" * 78)

    # Phase 1 summary
    print("\n  [Phase 1] Random Baseline Verification:")
    for key in ['random_same_frequency', 'random_shuffled_direction', 'random_shuffled_tpsl']:
        p = phase1[key]['p_value']
        d = phase1[key]['cohens_d']
        verdict = "PASS (p<0.01)" if p < 0.01 else (
            "PASS (p<0.05)" if p < 0.05 else "FAIL")
        short_key = key.replace('random_', '')
        print(f"    {short_key:<25} p={p:.6f}  d={d:.2f}  {verdict}")

    # Phase 2 summary
    print(f"\n  [Phase 2] Pattern Cleanup:")
    p2_best = max(phase2['config_results'].items(),
                  key=lambda x: x[1].get('pnl_mdd', 0))
    print(f"    Best pattern config: {p2_best[0]} "
          f"(PnL/MDD={p2_best[1].get('pnl_mdd', 0):.2f})")
    print(f"    Harmful patterns identified: {phase2['n_harmful']}")

    # Phase 3 summary
    print(f"\n  [Phase 3] Minimal Stack:")
    print(f"    Top 3 by IS PnL/MDD: {phase3['ranking'][:3]}")

    # Phase 4 summary
    print(f"\n  [Phase 4] WF Validation:")
    for name, wf_data in phase4.items():
        pass_str = f"{wf_data['n_pass']}/3 {'PASS' if wf_data['all_pass'] else 'FAIL'}"
        print(f"    {name:<20} {pass_str}  avg_OOS={wf_data['avg_oos_pnl']:+.2f}%  "
              f"min_OOS={wf_data['min_oos_pnl']:+.2f}%")

    # Phase 5 summary
    print(f"\n  [Phase 5] Combined Resolution:")
    c = phase5['current_stats']
    p = phase5['proposed_stats']
    print(f"    Current:   full_8 + 130pat -> PnL {c['pnl']:+.1f}%, "
          f"MDD {c['mdd']:.1f}%, PnL/MDD {c['pnl_mdd']:.1f}")
    print(f"    Proposed:  {phase5['best_mechanism_stack']} + "
          f"{phase5['best_pattern_config']} ({phase5['n_patterns_remaining']}pat) -> "
          f"PnL {p['pnl']:+.1f}%, MDD {p['mdd']:.1f}%, PnL/MDD {p['pnl_mdd']:.1f}")
    print(f"    WF: {phase5['wf_n_pass']}/3 {'PASS' if phase5['wf_all_pass'] else 'FAIL'}")

    # Final recommendation
    print(f"\n  FINAL RECOMMENDATION:")
    if phase5['wf_all_pass'] and p['pnl_mdd'] > c['pnl_mdd']:
        print(f"    ADOPT proposed config:")
        print(f"      Mechanism stack: {phase5['best_mechanism_stack']}")
        print(f"      Pattern set: {phase5['best_pattern_config']} "
              f"({phase5['n_patterns_remaining']} patterns)")
        if phase5['patterns_removed']:
            print(f"      Remove patterns: {phase5['patterns_removed']}")
        print(f"      Expected improvement: PnL/MDD {c['pnl_mdd']:.1f} -> {p['pnl_mdd']:.1f} "
              f"({(p['pnl_mdd']/max(0.01,c['pnl_mdd'])-1)*100:+.0f}%)")
    elif phase5['wf_all_pass']:
        print(f"    MAINTAIN current config (proposed WF PASS but PnL/MDD not better)")
        print(f"    Current PnL/MDD: {c['pnl_mdd']:.1f} vs Proposed: {p['pnl_mdd']:.1f}")
    else:
        print(f"    MAINTAIN current config (proposed WF FAIL: "
              f"{phase5['wf_n_pass']}/3)")
        print(f"    Consider individual mechanism adjustments from Phase 3 results")

    # All random baselines should beat at p<0.05
    all_random_pass = all(
        phase1[k]['p_value'] < 0.05
        for k in ['random_same_frequency', 'random_shuffled_direction',
                   'random_shuffled_tpsl']
    )
    if not all_random_pass:
        print(f"\n    WARNING: Not all random baselines beaten at p<0.05!")
        print(f"    Strategy may not have genuine edge. Review Phase 1 details.")

    all_results['recommendation'] = {
        'action': 'ADOPT' if (phase5['wf_all_pass'] and p['pnl_mdd'] > c['pnl_mdd'])
                  else 'MAINTAIN',
        'mechanism_stack': phase5['best_mechanism_stack'],
        'pattern_config': phase5['best_pattern_config'],
        'patterns_removed': phase5['patterns_removed'],
        'current_pnl_mdd': c['pnl_mdd'],
        'proposed_pnl_mdd': p['pnl_mdd'],
        'wf_pass': phase5['wf_all_pass'],
        'random_baselines_all_pass': all_random_pass,
    }

    # ---- Save Results ----
    print(f"\n  Saving results to {OUTPUT_FILE}...")

    class NumpyEncoder(json.JSONEncoder):
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
        json.dump(all_results, f, cls=NumpyEncoder, indent=2)

    t_total = time.time() - t_start
    print(f"\n  Total runtime: {t_total:.1f}s ({t_total/60:.1f}min)")
    print("=" * 78)
    print("STUDY COMPLETE")
    print("=" * 78)


if __name__ == '__main__':
    main()
