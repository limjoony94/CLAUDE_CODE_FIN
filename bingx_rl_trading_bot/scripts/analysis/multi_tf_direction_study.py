#!/usr/bin/env python3
"""
Multi-Timeframe Direction Filter Study
=======================================
Investigate whether higher-timeframe (15m/1h/4h) trend signals improve
the 5m pattern trading bot's performance.

Context:
  - 51 patterns (16L+35S), SHORT-heavy
  - Same-TF (5m) trend FILTERS all FAILED WF
  - Same-TF regime SIZING (counter x0.3) PASSED 19/19 WF (v1.35.3)
  - Question: does HIGHER TF direction signal add value?

Hypotheses:
  H1: 15m EMA(20) direction filter — block counter-trend 5m trades
  H2: 1h EMA(20) direction filter  — block counter-trend 5m trades
  H3: 15m regime sizing — counter x0.3, with-trend x1.0
  H4: 1h regime sizing  — counter x0.3, with-trend x1.0
  H5: 15m boost sizing  — counter x0.3, with-trend x1.5
  H6: 1h boost sizing   — counter x0.3, with-trend x1.5
  H7: 4h regime sizing  — counter x0.3, with-trend x1.0

Standard Research Protocol:
  Entry: signal bar + 1 open
  Exit: Intrabar high/low (distance-based)
  Same-bar resolution: abs(tp - opens[j]) vs abs(sl - opens[j])
  Fee: 0.10% * LEVERAGE(3) = 0.30% per trade (capital-space)
  Slippage buffer: 0.02%
  ATR-scaled TP/SL: ATR(14)/median(576), clamp [0.6, 1.7]
  Timeout: 864 bars -> DROP
  N=9 multi-position (virtual slots), direction_cap=8, Hedge mode
  MC: sign randomization 5k sims, seeds [42,123,7], max p < 0.01
  WF: 3-fold expanding window
  Quality: edge>=21.8pp, WR>=60%, SL>=1.0%, min_trades>=25

Output:
  - results/multi_tf_direction_study.json
  - Console report with per-hypothesis GO/STOP
"""

import json
import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from scripts.production.pattern_5m.indicators import classify_candle
from scripts.production.pattern_5m.constants import AVG_BODY_WINDOW

# ============================================================
# CONSTANTS
# ============================================================
LEVERAGE = 3
FEE_PCT = 0.10          # roundtrip fee in price-space
SLIPPAGE_BUFFER = 0.02  # 0.02% slippage buffer for ATR
MAX_BARS = 288           # 24h timeout for scanner-level trades
TIMEOUT_BARS = 864       # 72h portfolio timeout
MC_SIMS = 5000
MC_SEEDS = [42, 123, 7]
MAX_POSITIONS = 9
DIRECTION_CAP = 8        # v1.35.1 current
ATR_PERIOD = 14
ATR_WINDOW = 576
CLAMP_LO = 0.6
CLAMP_HI = 1.7
ATR_WARMUP = ATR_PERIOD + ATR_WINDOW  # 590 bars
BARS_PER_DAY = 288

# Higher-TF EMA parameters
EMA_PERIOD = 20
SLOPE_LOOKBACK = 5  # same as production regime_sizing

# Higher-TF resampling: how many 5m bars per higher TF bar
BARS_15M = 3     # 15m = 3 x 5m
BARS_1H = 12     # 1h  = 12 x 5m
BARS_4H = 48     # 4h  = 48 x 5m

DATA_FILE = os.path.join(PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
PATTERNS_FILE = os.path.join(PROJECT_ROOT, 'results', 'dynamic_patterns.json')
OUTPUT_FILE = os.path.join(PROJECT_ROOT, 'results', 'multi_tf_direction_study.json')


# ============================================================
# DATA LOADING
# ============================================================

def load_data():
    """Load 270d 5m CSV. Use pre-classified type_code column."""
    df = pd.read_csv(DATA_FILE)
    types = df['type_code'].values.tolist()
    opens = df['open'].values.astype(np.float64)
    highs = df['high'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)
    closes = df['close'].values.astype(np.float64)
    n_bars = len(df)
    print(f"Loaded {n_bars} bars ({n_bars / BARS_PER_DAY:.0f}d)")
    print(f"Period: {df['timestamp'].iloc[0]} ~ {df['timestamp'].iloc[-1]}")
    return df, types, opens, highs, lows, closes, n_bars


def load_patterns():
    """Load dynamic_patterns.json -> pattern sets + TP/SL map."""
    with open(PATTERNS_FILE) as f:
        pat_data = json.load(f)
    pL = set(pat_data['patterns']['long'])
    pS = set(pat_data['patterns']['short'])
    tpsl = {}
    for pat_name, tp_sl_list in pat_data['patterns_tpsl'].items():
        tpsl[pat_name] = (tp_sl_list[0], tp_sl_list[1])
    print(f"Patterns: {len(pL)}L + {len(pS)}S = {len(pL) + len(pS)} total")
    print(f"TP/SL map: {len(tpsl)} entries")
    return pL, pS, tpsl


# ============================================================
# ATR RATIO (from scanner protocol)
# ============================================================

def compute_atr_ratio(highs, lows, closes):
    """ATR / rolling_median(ATR) ratio (Wilder EMA). Same as scanner."""
    n = len(closes)
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i - 1]),
                     abs(lows[i] - closes[i - 1]))
    atr = np.full(n, np.nan)
    if n >= ATR_PERIOD:
        atr[ATR_PERIOD - 1] = tr[:ATR_PERIOD].mean()
        for i in range(ATR_PERIOD, n):
            atr[i] = (atr[i - 1] * (ATR_PERIOD - 1) + tr[i]) / ATR_PERIOD
    med = pd.Series(atr).rolling(ATR_WINDOW, min_periods=ATR_WINDOW).median().values
    ratio = np.full(n, np.nan)
    valid = (~np.isnan(atr)) & (~np.isnan(med)) & (med > 0)
    ratio[valid] = atr[valid] / med[valid]
    return ratio


# ============================================================
# HIGHER-TF RESAMPLING + EMA DIRECTION
# ============================================================

def resample_to_higher_tf(opens_5m, highs_5m, lows_5m, closes_5m, n_bars, tf_bars):
    """Resample 5m bars to higher TF using OHLC aggregation.

    Aligns to boundaries: bar i belongs to group i // tf_bars.
    For each group: O=first_open, H=max_high, L=min_low, C=last_close.
    Returns (htf_opens, htf_highs, htf_lows, htf_closes, group_ids) where
    group_ids[i] = the higher-TF bar index that 5m bar i belongs to.
    """
    n_groups = (n_bars + tf_bars - 1) // tf_bars
    htf_o = np.full(n_groups, np.nan)
    htf_h = np.full(n_groups, np.nan)
    htf_l = np.full(n_groups, np.nan)
    htf_c = np.full(n_groups, np.nan)

    for g in range(n_groups):
        start = g * tf_bars
        end = min(start + tf_bars, n_bars)
        htf_o[g] = opens_5m[start]
        htf_h[g] = np.max(highs_5m[start:end])
        htf_l[g] = np.min(lows_5m[start:end])
        htf_c[g] = closes_5m[end - 1]

    # Map each 5m bar to its higher-TF group
    group_ids = np.arange(n_bars) // tf_bars
    return htf_o, htf_h, htf_l, htf_c, group_ids


def compute_htf_direction(closes_htf, n_htf, group_ids, n_bars_5m,
                          ema_period=EMA_PERIOD, lookback=SLOPE_LOOKBACK):
    """Compute EMA direction on higher-TF and map back to 5m bars.

    For each 5m bar, the direction comes from the COMPLETED higher-TF bar
    (the one before the current group), to avoid look-ahead bias.

    Returns numpy array of length n_bars_5m:
      +1 = UP trend (EMA rising)
      -1 = DOWN trend (EMA falling/flat)
       0 = not enough data
    """
    # EMA on higher-TF closes
    ema = pd.Series(closes_htf).ewm(span=ema_period, adjust=False).mean().values

    # Compute slope of EMA: difference between current and lookback bars ago
    htf_direction = np.zeros(n_htf, dtype=np.int8)
    for i in range(lookback, n_htf):
        if ema[i] > ema[i - lookback]:
            htf_direction[i] = 1
        else:
            htf_direction[i] = -1

    # Map to 5m: use the PREVIOUS completed higher-TF bar's direction
    # (if 5m bar is in group g, use direction from group g-1)
    direction_5m = np.zeros(n_bars_5m, dtype=np.int8)
    for i in range(n_bars_5m):
        g = group_ids[i]
        prev_g = g - 1  # previous completed HTF bar
        if prev_g >= lookback and prev_g < n_htf:
            direction_5m[i] = htf_direction[prev_g]
        # else: 0 (insufficient data)

    return direction_5m


def build_all_htf_directions(opens, highs, lows, closes, n_bars):
    """Build higher-TF direction arrays for 15m, 1h, and 4h.

    Returns dict: {'15m': direction_array, '1h': ..., '4h': ...}
    """
    result = {}
    for tf_label, tf_bars in [('15m', BARS_15M), ('1h', BARS_1H), ('4h', BARS_4H)]:
        htf_o, htf_h, htf_l, htf_c, gids = resample_to_higher_tf(
            opens, highs, lows, closes, n_bars, tf_bars)
        direction = compute_htf_direction(htf_c, len(htf_c), gids, n_bars)
        result[tf_label] = direction

        # Stats for reporting
        up_pct = np.sum(direction == 1) / np.sum(direction != 0) * 100 if np.sum(direction != 0) > 0 else 0
        print(f"  {tf_label} direction: {np.sum(direction == 1)} UP, "
              f"{np.sum(direction == -1)} DOWN, "
              f"{np.sum(direction == 0)} N/A  "
              f"({up_pct:.1f}% UP of valid)")
    return result


# ============================================================
# SIGNAL GENERATION
# ============================================================

def generate_signals(types, patterns_long, patterns_short, n_bars, start_idx=0):
    """Generate signals from type codes. Returns list of (bar_idx, pattern, direction)."""
    signals = []
    for i in range(max(2, start_idx), n_bars):
        pat = f"{types[i - 2]}-{types[i - 1]}-{types[i]}"
        if pat in patterns_long:
            signals.append((i, pat, 'LONG'))
        if pat in patterns_short:
            signals.append((i, pat, 'SHORT'))
    return signals


# ============================================================
# PORTFOLIO SIMULATION (N=9, direction_cap, Hedge, ATR, timeout)
# ============================================================

def _check_exit(pos, bar, opens, highs, lows, n_bars, atr_ratio, fee):
    """Check if position should exit at given bar."""
    entry_bar = pos['entry_bar']
    if bar < entry_bar:
        return None
    entry = opens[entry_bar]
    if entry <= 0:
        return None

    tp_pct = pos['tp_pct']
    sl_pct = pos['sl_pct']
    direction = pos['direction']
    sig_bar = pos['signal_bar']

    if atr_ratio is not None and sig_bar < len(atr_ratio) and not np.isnan(atr_ratio[sig_bar]):
        r = max(CLAMP_LO, min(CLAMP_HI, atr_ratio[sig_bar]))
    else:
        r = 1.0

    eff_tp = tp_pct * r + SLIPPAGE_BUFFER
    eff_sl = max(0.1, sl_pct * r - SLIPPAGE_BUFFER)

    if direction == 'LONG':
        tp_price = entry * (1 + eff_tp / 100)
        sl_price = entry * (1 - eff_sl / 100)
    else:
        tp_price = entry * (1 - eff_tp / 100)
        sl_price = entry * (1 + eff_sl / 100)

    hold = bar - entry_bar
    if hold >= TIMEOUT_BARS:
        return {'entry_bar': entry_bar, 'exit_bar': bar, 'pnl_slot': 0,
                'reason': 'TIMEOUT', 'drop': True}

    h, l = highs[bar], lows[bar]
    if direction == 'LONG':
        hit_tp = h >= tp_price
        hit_sl = l <= sl_price
    else:
        hit_tp = l <= tp_price
        hit_sl = h >= sl_price

    if not hit_tp and not hit_sl:
        return None

    if hit_tp and hit_sl:
        tp_dist = abs(tp_price - opens[bar])
        sl_dist = abs(sl_price - opens[bar])
        if tp_dist <= sl_dist:
            exit_price, reason = tp_price, 'TP'
        else:
            exit_price, reason = sl_price, 'SL'
    elif hit_tp:
        exit_price, reason = tp_price, 'TP'
    else:
        exit_price, reason = sl_price, 'SL'

    if direction == 'LONG':
        pnl = (exit_price / entry - 1) * 100 * LEVERAGE
    else:
        pnl = (1 - exit_price / entry) * 100 * LEVERAGE
    pnl -= fee

    return {'entry_bar': entry_bar, 'exit_bar': bar, 'pnl_slot': pnl,
            'reason': reason, 'drop': False}


def simulate_portfolio(signals, tpsl_map, opens, highs, lows, n_bars,
                       atr_ratio, direction_cap, oos_start, oos_end,
                       htf_direction=None, mode='baseline',
                       counter_mult=0.3, with_mult=1.0):
    """N=9 multi-position portfolio simulation.

    Modes:
      'baseline': no higher-TF filter, uniform sizing
      'filter':   block counter-trend trades (signal removed if counter-trend)
      'sizing':   scale counter-trend down (counter_mult), with-trend up (with_mult)

    htf_direction: numpy array indexed by 5m bar, values {-1, 0, +1}.
      +1 = UP trend, -1 = DOWN trend, 0 = no data.

    Timeout trades are DROPPED (not counted).
    """
    positions = []
    trades = []
    SIZE_PCT = 100.0 / MAX_POSITIONS
    fee = FEE_PCT * LEVERAGE

    signals_in_range = [(s, p, d) for s, p, d in signals if oos_start <= s < oos_end]
    signals_sorted = sorted(signals_in_range, key=lambda x: x[0])
    sig_idx = 0

    blocked_count = 0

    for bar in range(oos_start, oos_end):
        # Check exits
        closed_slots = []
        for pos in positions:
            result = _check_exit(pos, bar, opens, highs, lows, n_bars, atr_ratio, fee)
            if result is not None:
                if result.get('drop', False):
                    closed_slots.append(pos['slot'])
                    continue
                result['pattern'] = pos['pattern']
                result['direction'] = pos['direction']
                result['size_mult'] = pos.get('size_mult', 1.0)
                result['pnl_portfolio'] = result['pnl_slot'] * SIZE_PCT / 100 * result['size_mult']
                trades.append(result)
                closed_slots.append(pos['slot'])
        positions = [p for p in positions if p['slot'] not in closed_slots]

        # Process signals
        while sig_idx < len(signals_sorted) and signals_sorted[sig_idx][0] == bar:
            sig_bar, pat, direction = signals_sorted[sig_idx]
            sig_idx += 1

            # --- Higher-TF filter/sizing logic ---
            sm = 1.0
            if htf_direction is not None and mode != 'baseline':
                htf_dir = htf_direction[sig_bar] if sig_bar < len(htf_direction) else 0

                # Determine if counter-trend:
                # UP trend (+1) and SHORT = counter
                # DOWN trend (-1) and LONG = counter
                is_counter = (htf_dir == 1 and direction == 'SHORT') or \
                             (htf_dir == -1 and direction == 'LONG')
                is_with = (htf_dir == 1 and direction == 'LONG') or \
                          (htf_dir == -1 and direction == 'SHORT')

                if mode == 'filter' and is_counter:
                    blocked_count += 1
                    continue  # Skip counter-trend trade entirely
                elif mode == 'sizing':
                    if is_counter:
                        sm = counter_mult
                    elif is_with:
                        sm = with_mult
                    # htf_dir == 0 -> sm stays 1.0

            if len(positions) >= MAX_POSITIONS:
                continue
            dir_count = sum(1 for p in positions if p['direction'] == direction)
            if dir_count >= direction_cap:
                continue
            if any(p['pattern'] == pat for p in positions):
                continue

            entry_bar = sig_bar + 1
            if entry_bar >= n_bars:
                continue
            tp_sl = tpsl_map.get(pat)
            if tp_sl is None:
                continue

            positions.append({
                'slot': f"{pat}_{sig_bar}",
                'signal_bar': sig_bar,
                'entry_bar': entry_bar,
                'direction': direction,
                'pattern': pat,
                'tp_pct': tp_sl[0],
                'sl_pct': tp_sl[1],
                'size_mult': sm,
            })

    # Force-close remaining at OOS end
    for pos in positions:
        entry_bar = pos['entry_bar']
        if entry_bar >= n_bars:
            continue
        entry = opens[entry_bar]
        if entry <= 0:
            continue
        exit_bar = min(oos_end - 1, n_bars - 1)
        exit_price = opens[exit_bar]
        direction = pos['direction']
        if direction == 'LONG':
            pnl = (exit_price / entry - 1) * 100 * LEVERAGE
        else:
            pnl = (1 - exit_price / entry) * 100 * LEVERAGE
        pnl -= fee
        sm = pos.get('size_mult', 1.0)
        trades.append({
            'entry_bar': entry_bar, 'exit_bar': exit_bar,
            'pnl_slot': pnl, 'reason': 'OOS_END',
            'pattern': pos['pattern'], 'direction': direction,
            'size_mult': sm,
            'pnl_portfolio': pnl * SIZE_PCT / 100 * sm,
        })

    return trades, blocked_count


def compute_portfolio_metrics(trades):
    """Compute portfolio-level metrics from trade list."""
    if not trades:
        return {'trades': 0, 'wr': 0.0, 'pnl': 0.0, 'mdd': 0.0,
                'long_trades': 0, 'short_trades': 0,
                'long_pnl': 0.0, 'short_pnl': 0.0}

    wins = [t for t in trades if t['pnl_slot'] > 0]
    sorted_trades = sorted(trades, key=lambda x: x['entry_bar'])
    equity = 100.0
    peak = equity
    max_dd = 0.0
    for t in sorted_trades:
        equity += t['pnl_portfolio']
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    long_t = [t for t in trades if t['direction'] == 'LONG']
    short_t = [t for t in trades if t['direction'] == 'SHORT']
    return {
        'trades': len(trades),
        'wr': round(len(wins) / len(trades) * 100, 1) if trades else 0,
        'pnl': round(sum(t['pnl_portfolio'] for t in trades), 2),
        'mdd': round(max_dd, 2),
        'long_trades': len(long_t),
        'short_trades': len(short_t),
        'long_pnl': round(sum(t['pnl_portfolio'] for t in long_t), 2),
        'short_pnl': round(sum(t['pnl_portfolio'] for t in short_t), 2),
    }


def mc_test_portfolio(trades, n_sims=MC_SIMS):
    """MC sign randomization on portfolio trade PnLs."""
    if len(trades) < 5:
        return 1.0
    pnls = np.array([t['pnl_portfolio'] for t in trades])
    actual = np.sum(pnls)
    p_vals = []
    for seed in MC_SEEDS:
        rng = np.random.RandomState(seed)
        signs = rng.choice([-1, 1], size=(n_sims, len(pnls)))
        rand_sums = signs @ pnls
        p_vals.append(float(np.mean(rand_sums >= actual)))
    return max(p_vals)


# ============================================================
# WALK-FORWARD VALIDATION
# ============================================================

def run_wf(all_signals, tpsl_map, opens, highs, lows, n_bars, atr_ratio,
           htf_direction=None, mode='baseline',
           counter_mult=0.3, with_mult=1.0, label=""):
    """3-fold expanding window WF on fixed pattern set.

    Fold structure (expanding IS, equal OOS):
      4 segments of equal size (n_folds + 1 = 4).
      Fold f: IS = [0, (f+1)*seg), OOS = [(f+1)*seg, (f+2)*seg)
    """
    n_folds = 3
    seg_size = n_bars // (n_folds + 1)

    fold_results = []
    total_blocked = 0

    for fold in range(n_folds):
        oos_start = (fold + 1) * seg_size
        oos_end = (fold + 2) * seg_size if fold < n_folds - 1 else n_bars

        trades, blocked = simulate_portfolio(
            all_signals, tpsl_map, opens, highs, lows, n_bars,
            atr_ratio, DIRECTION_CAP, oos_start, oos_end,
            htf_direction=htf_direction, mode=mode,
            counter_mult=counter_mult, with_mult=with_mult,
        )
        m = compute_portfolio_metrics(trades)
        total_blocked += blocked
        fold_results.append({
            'fold': fold + 1,
            'oos_start': oos_start,
            'oos_end': oos_end,
            'oos_bars': oos_end - oos_start,
            'blocked': blocked,
            **m,
        })

    total_pnl = sum(f['pnl'] for f in fold_results)
    max_mdd = max(f['mdd'] for f in fold_results) if fold_results else 0
    all_positive = all(f['pnl'] > 0 for f in fold_results if f['trades'] > 0)
    n_positive = sum(1 for f in fold_results if f['pnl'] > 0)
    pnl_mdd = total_pnl / max_mdd if max_mdd > 0 else 0
    total_trades = sum(f['trades'] for f in fold_results)

    return {
        'label': label,
        'mode': mode,
        'n_folds': n_folds,
        'folds': fold_results,
        'total_pnl': round(total_pnl, 2),
        'total_trades': total_trades,
        'max_mdd': round(max_mdd, 2),
        'pnl_mdd': round(pnl_mdd, 2),
        'pass_3_3': all_positive,
        'n_positive': n_positive,
        'total_blocked': total_blocked,
    }


# ============================================================
# FULL-PERIOD (IS) METRICS
# ============================================================

def run_full_period(all_signals, tpsl_map, opens, highs, lows, n_bars, atr_ratio,
                    htf_direction=None, mode='baseline',
                    counter_mult=0.3, with_mult=1.0, label=""):
    """Run on full 270d period for IS metrics."""
    trades, blocked = simulate_portfolio(
        all_signals, tpsl_map, opens, highs, lows, n_bars,
        atr_ratio, DIRECTION_CAP, ATR_WARMUP, n_bars,
        htf_direction=htf_direction, mode=mode,
        counter_mult=counter_mult, with_mult=with_mult,
    )
    m = compute_portfolio_metrics(trades)
    mc_p = mc_test_portfolio(trades) if trades else 1.0
    return {
        'label': label,
        'mode': mode,
        **m,
        'mc_p': round(mc_p, 6),
        'blocked': blocked,
    }


# ============================================================
# MAIN STUDY
# ============================================================

def main():
    t0 = time.time()
    print("=" * 90)
    print("MULTI-TIMEFRAME DIRECTION FILTER STUDY")
    print("=" * 90)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # ---- Load data ----
    print("--- Loading Data ---")
    df, types, opens, highs, lows, closes, n_bars = load_data()
    pL, pS, tpsl = load_patterns()
    print()

    # ---- ATR ratio ----
    print("--- Computing ATR Ratio ---")
    atr_ratio = compute_atr_ratio(highs, lows, closes)
    valid_atr = np.sum(~np.isnan(atr_ratio))
    print(f"  ATR ratio valid: {valid_atr}/{n_bars} bars "
          f"(first valid at bar {np.argmax(~np.isnan(atr_ratio))})")
    print()

    # ---- Higher-TF directions ----
    print("--- Computing Higher-TF Directions ---")
    htf_dirs = build_all_htf_directions(opens, highs, lows, closes, n_bars)
    print()

    # ---- Generate all 5m signals ----
    print("--- Generating 5m Signals ---")
    all_signals = generate_signals(types, pL, pS, n_bars, start_idx=ATR_WARMUP)
    n_long_sig = sum(1 for _, _, d in all_signals if d == 'LONG')
    n_short_sig = sum(1 for _, _, d in all_signals if d == 'SHORT')
    print(f"  Total signals: {len(all_signals)} ({n_long_sig}L + {n_short_sig}S)")
    print()

    # ---- Define scenarios ----
    scenarios = [
        # (label, htf_direction, mode, counter_mult, with_mult, hypothesis)
        ('Baseline (no filter)', None, 'baseline', 0.3, 1.0, 'Baseline'),
        ('H1: 15m filter',       htf_dirs['15m'], 'filter', 0.3, 1.0, 'H1'),
        ('H2: 1h filter',        htf_dirs['1h'],  'filter', 0.3, 1.0, 'H2'),
        ('H3: 15m sizing 0.3/1.0', htf_dirs['15m'], 'sizing', 0.3, 1.0, 'H3'),
        ('H4: 1h sizing 0.3/1.0',  htf_dirs['1h'],  'sizing', 0.3, 1.0, 'H4'),
        ('H5: 15m boost 0.3/1.5',  htf_dirs['15m'], 'sizing', 0.3, 1.5, 'H5'),
        ('H6: 1h boost 0.3/1.5',   htf_dirs['1h'],  'sizing', 0.3, 1.5, 'H6'),
        ('H7: 4h sizing 0.3/1.0',  htf_dirs['4h'],  'sizing', 0.3, 1.0, 'H7'),
    ]

    # ---- Run full-period (IS) metrics ----
    print("=" * 90)
    print("PHASE 1: FULL-PERIOD (IS) METRICS")
    print("=" * 90)

    is_results = {}
    for label, htf_dir, mode, c_mult, w_mult, hyp in scenarios:
        print(f"  Running {label}...", end=' ', flush=True)
        r = run_full_period(
            all_signals, tpsl, opens, highs, lows, n_bars, atr_ratio,
            htf_direction=htf_dir, mode=mode,
            counter_mult=c_mult, with_mult=w_mult, label=label,
        )
        is_results[hyp] = r
        print(f"trades={r['trades']}, WR={r['wr']:.1f}%, "
              f"PnL={r['pnl']:+.2f}%, MDD={r['mdd']:.2f}%, "
              f"MC={r['mc_p']:.4f}, blocked={r['blocked']}")

    # Print IS comparison table
    print(f"\n{'Scenario':<28} {'Trades':>7} {'WR':>6} {'PnL':>10} {'MDD':>8} "
          f"{'PnL/MDD':>8} {'L_Tr':>5} {'S_Tr':>5} {'L_PnL':>8} {'S_PnL':>8} {'MC':>7} {'Blkd':>5}")
    print("-" * 120)
    for label, _, mode, _, _, hyp in scenarios:
        r = is_results[hyp]
        pnl_mdd = r['pnl'] / r['mdd'] if r['mdd'] > 0 else 0
        print(f"{label:<28} {r['trades']:>7} {r['wr']:>5.1f}% {r['pnl']:>+9.2f}% "
              f"{r['mdd']:>7.2f}% {pnl_mdd:>7.2f}x {r['long_trades']:>5} {r['short_trades']:>5} "
              f"{r['long_pnl']:>+7.2f}% {r['short_pnl']:>+7.2f}% {r['mc_p']:>6.4f} {r['blocked']:>5}")

    # ---- Run WF validation ----
    print(f"\n{'=' * 90}")
    print("PHASE 2: WALK-FORWARD VALIDATION (3-fold expanding window)")
    print("=" * 90)

    wf_results = {}
    for label, htf_dir, mode, c_mult, w_mult, hyp in scenarios:
        print(f"\n  --- {label} ---")
        r = run_wf(
            all_signals, tpsl, opens, highs, lows, n_bars, atr_ratio,
            htf_direction=htf_dir, mode=mode,
            counter_mult=c_mult, with_mult=w_mult, label=label,
        )
        wf_results[hyp] = r
        for f in r['folds']:
            status = 'OK' if f['pnl'] > 0 else 'NEG'
            print(f"    Fold {f['fold']}: OOS [{f['oos_start']}-{f['oos_end']}] "
                  f"bars={f['oos_bars']}, trades={f['trades']}, "
                  f"WR={f['wr']:.1f}%, PnL={f['pnl']:+.2f}%, "
                  f"MDD={f['mdd']:.2f}%, blkd={f['blocked']} [{status}]")
        status = "PASS" if r['pass_3_3'] else "FAIL"
        print(f"    Summary: Total PnL={r['total_pnl']:+.2f}%, Max MDD={r['max_mdd']:.2f}%, "
              f"PnL/MDD={r['pnl_mdd']:.2f}x, {r['n_positive']}/3 [{status}], "
              f"total_blocked={r['total_blocked']}")

    # Print WF comparison table
    print(f"\n{'=' * 90}")
    print("PHASE 3: COMPARISON TABLE")
    print("=" * 90)

    baseline_wf = wf_results['Baseline']

    print(f"\n{'Scenario':<28} {'OOS_Tr':>7} {'OOS_PnL':>10} {'OOS_MDD':>9} "
          f"{'PnL/MDD':>8} {'3/3':>5} {'Status':>7} {'Folds':>30}")
    print("-" * 110)
    for label, _, mode, _, _, hyp in scenarios:
        r = wf_results[hyp]
        status = "PASS" if r['pass_3_3'] else "FAIL"
        fold_str = " | ".join(f"{f['pnl']:+.1f}" for f in r['folds'])
        print(f"{label:<28} {r['total_trades']:>7} {r['total_pnl']:>+9.2f}% "
              f"{r['max_mdd']:>8.2f}% {r['pnl_mdd']:>7.2f}x "
              f"{r['n_positive']}/3   {status:<6} [{fold_str}]")

    # Delta analysis
    print(f"\n--- Delta vs Baseline ---")
    for label, _, mode, _, _, hyp in scenarios:
        if hyp == 'Baseline':
            continue
        delta_pnl = wf_results[hyp]['total_pnl'] - baseline_wf['total_pnl']
        delta_mdd = wf_results[hyp]['max_mdd'] - baseline_wf['max_mdd']
        delta_pnl_mdd = wf_results[hyp]['pnl_mdd'] - baseline_wf['pnl_mdd']
        status = "PASS" if wf_results[hyp]['pass_3_3'] else "FAIL"
        print(f"  {label:<28} dPnL={delta_pnl:+.2f}%  dMDD={delta_mdd:+.2f}%  "
              f"dPnL/MDD={delta_pnl_mdd:+.2f}x  {status}")

    # ---- Verdicts ----
    print(f"\n{'=' * 90}")
    print("PHASE 4: VERDICTS")
    print("=" * 90)

    verdicts = {}

    # Verdict logic: PASS WF 3/3 AND improvement over baseline in PnL/MDD
    for label, _, mode, _, _, hyp in scenarios:
        if hyp == 'Baseline':
            verdicts[hyp] = 'REFERENCE'
            continue

        r = wf_results[hyp]
        passes_wf = r['pass_3_3']
        improves_pnl_mdd = r['pnl_mdd'] > baseline_wf['pnl_mdd']
        improves_pnl = r['total_pnl'] > baseline_wf['total_pnl']

        if passes_wf and improves_pnl_mdd:
            verdict = 'GO'
        elif passes_wf and improves_pnl:
            verdict = 'MARGINAL'
        elif passes_wf:
            verdict = 'PASS_BUT_WORSE'
        else:
            verdict = 'STOP'

        verdicts[hyp] = verdict
        delta_pnl = r['total_pnl'] - baseline_wf['total_pnl']
        delta_pnl_mdd = r['pnl_mdd'] - baseline_wf['pnl_mdd']
        print(f"  {hyp}: {label:<28} => {verdict:<16} "
              f"(WF {'3/3' if passes_wf else 'FAIL'}, "
              f"dPnL={delta_pnl:+.2f}%, dPnL/MDD={delta_pnl_mdd:+.2f}x)")

    # Best scenario
    go_scenarios = [hyp for hyp, v in verdicts.items() if v == 'GO']
    if go_scenarios:
        best = max(go_scenarios, key=lambda h: wf_results[h]['pnl_mdd'])
        print(f"\n  BEST: {best} — {wf_results[best]['label']} "
              f"(PnL/MDD={wf_results[best]['pnl_mdd']:.2f}x vs "
              f"baseline {baseline_wf['pnl_mdd']:.2f}x)")
    else:
        print("\n  No scenario achieved GO status. Higher-TF filters do not add value.")

    # ---- Save results ----
    elapsed = time.time() - t0
    output = {
        'study': 'multi_tf_direction_study',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'elapsed_seconds': round(elapsed, 1),
        'data_bars': n_bars,
        'data_days': round(n_bars / BARS_PER_DAY, 1),
        'patterns': {'long': len(pL), 'short': len(pS), 'total': len(pL) + len(pS)},
        'parameters': {
            'leverage': LEVERAGE,
            'fee_pct': FEE_PCT,
            'slippage_buffer': SLIPPAGE_BUFFER,
            'timeout_bars': TIMEOUT_BARS,
            'max_positions': MAX_POSITIONS,
            'direction_cap': DIRECTION_CAP,
            'atr_period': ATR_PERIOD,
            'atr_window': ATR_WINDOW,
            'clamp': [CLAMP_LO, CLAMP_HI],
            'htf_ema_period': EMA_PERIOD,
            'htf_slope_lookback': SLOPE_LOOKBACK,
        },
        'htf_direction_stats': {
            tf: {
                'up_bars': int(np.sum(d == 1)),
                'down_bars': int(np.sum(d == -1)),
                'na_bars': int(np.sum(d == 0)),
                'up_pct': round(float(np.sum(d == 1) / max(1, np.sum(d != 0)) * 100), 1),
            }
            for tf, d in htf_dirs.items()
        },
        'is_results': {hyp: r for hyp, r in is_results.items()},
        'wf_results': {hyp: r for hyp, r in wf_results.items()},
        'verdicts': verdicts,
        'signals_total': len(all_signals),
        'signals_long': n_long_sig,
        'signals_short': n_short_sig,
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_FILE}")
    print(f"Total elapsed: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
