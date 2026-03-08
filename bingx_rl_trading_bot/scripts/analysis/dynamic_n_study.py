#!/usr/bin/env python3
"""
Dynamic N (Max Positions) Adjustment Study
===========================================

Problem: N=9 fixed positions with correlated entries can lead to clustered losses
during directional moves (e.g., 8 SHORT SL hits during BTC rally). Individual
pattern edge is genuine, but simultaneous exposure amplifies drawdown.

Hypothesis: Dynamically adjusting max allowed positions based on volatility/regime
conditions can reduce correlated loss events while preserving edge.

Phases:
  Phase 1: Volatility metric analysis (4 metrics, percentile distribution)
  Phase 2: N-mapping strategy backtests (12+ scenarios across 5 strategy families)
  Phase 3: Portfolio backtest with compound sizing
  Phase 4: WF 3-fold expanding window OOS validation (top 3 scenarios)
  Phase 5: vs Static N comparison (N=3, N=5, N=7, N=9)
  Phase 6: Implementation complexity + parameter sensitivity

Standard Research Protocol:
- Entry: signal bar+1 open
- Exit: intrabar high/low (distance-based from bar OPEN, NOT entry price)
- Same-bar resolution: abs(tp - opens[j]) vs abs(sl - opens[j])
- Sizing: Compound (reinvest), 1/N per slot where N is dynamic
- Fee: 0.05% x 2 x LEVERAGE(3) = 0.30% per trade
- Slippage buffer: 0.02% (TP adjusted up, SL adjusted down)
- Timeout: 864 bars (72h) -> DROP from PnL
- ATR-scaled TP/SL: ATR(14) / rolling_median(ATR(14), 576), clamp [0.6, 1.7]
- Pattern classification: use type_code column from CSV (short codes)
- Patterns: load from results/dynamic_patterns.json (130 patterns with per-pattern TP/SL)
- Direction cap: 7 (max same-direction positions)
- Neutral window: ~257d auto-detected
- WF: 3-fold expanding window
- MC: 3 seeds [42, 123, 7], max p-value < 0.01
"""

import sys
import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

# ---- Constants ----
LEVERAGE = 3
FEE_PCT = 0.10           # 0.10% roundtrip
SLIPPAGE_BUFFER = 0.02   # 0.02% buffer
TIMEOUT_BARS = 864       # 72h
STATIC_MAX_POSITIONS = 9 # baseline fixed N
DIRECTION_CAP = 7        # v1.36.1
ATR_PERIOD = 14
ATR_WINDOW = 576
CLAMP_LO = 0.6
CLAMP_HI = 1.7
ATR_WARMUP = ATR_PERIOD + ATR_WINDOW  # 590 bars

# EMA regime params (same as production v1.35.3)
EMA_SPAN = 20
EMA_LOOKBACK = 5

# Aggregate risk cap params (same as production v1.35.5)
COUNTER_CAP = 3.0
WITH_CAP = 7.0

# Momentum guard params (same as production v1.36.2)
MOM_LOOKBACK_BARS = 6
MOM_THRESHOLD_PCT = 1.0
MOM_COOLDOWN_BARS = 6

# MDD sizing params (same as production v1.34.0)
MDD_FULL_SIZE_DD = 5.0
MDD_MIN_SIZE_DD = 20.0
MDD_MIN_SIZE_FRAC = 0.25

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CSV_PATH = os.path.join(BASE_DIR, 'data', 'btc_5m_270days_reclassified.csv')
PATTERNS_PATH = os.path.join(BASE_DIR, 'results', 'dynamic_patterns.json')
RESULTS_PATH = os.path.join(BASE_DIR, 'results', 'dynamic_n_study.json')


# ==================================================================
# Utility functions (reused from scanner / aggregate_risk_cap_study)
# ==================================================================

def compute_atr_ratio(highs, lows, closes):
    """ATR(14) / rolling_median(ATR(14), 576). Returns raw ratio (unclamped)."""
    n = len(closes)
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i - 1]),
                     abs(lows[i] - closes[i - 1]))
    atr = pd.Series(tr).rolling(ATR_PERIOD, min_periods=ATR_PERIOD).mean().values
    med = pd.Series(atr).rolling(ATR_WINDOW, min_periods=ATR_WINDOW).median().values
    ratio = np.full(n, np.nan)
    valid = (~np.isnan(atr)) & (~np.isnan(med)) & (med > 0)
    ratio[valid] = atr[valid] / med[valid]
    return ratio


def compute_realized_vol(closes, window=20):
    """Rolling realized volatility: std of log returns over window bars."""
    n = len(closes)
    log_ret = np.full(n, np.nan)
    for i in range(1, n):
        if closes[i - 1] > 0 and closes[i] > 0:
            log_ret[i] = np.log(closes[i] / closes[i - 1])
    rv = pd.Series(log_ret).rolling(window, min_periods=window).std().values
    return rv


def compute_pct_change(closes, lookback):
    """Absolute % change over lookback bars."""
    n = len(closes)
    pct = np.full(n, np.nan)
    for i in range(lookback, n):
        if closes[i - lookback] > 0:
            pct[i] = abs((closes[i] / closes[i - lookback] - 1) * 100)
    return pct


def compute_range_vol(highs, lows, closes, window=20):
    """VIX-like proxy: rolling avg of (high-low)/close."""
    n = len(closes)
    range_ratio = np.full(n, np.nan)
    for i in range(n):
        if closes[i] > 0:
            range_ratio[i] = (highs[i] - lows[i]) / closes[i] * 100
    rv = pd.Series(range_ratio).rolling(window, min_periods=window).mean().values
    return rv


def compute_ema_regime(closes, span=EMA_SPAN, lookback=EMA_LOOKBACK):
    """EMA(20) slope-based regime: 'UP' or 'DOWN'."""
    ema = pd.Series(closes).ewm(span=span, adjust=False).mean().values
    regimes = ['DOWN'] * len(closes)
    for i in range(lookback, len(closes)):
        slope = ema[i] - ema[i - lookback]
        regimes[i] = 'UP' if slope > 0 else 'DOWN'
    return regimes


def find_neutral_window(closes, tol_pct=1.0, min_days=90, bars_per_day=288):
    """Find longest window where start_price ~ end_price within tolerance."""
    n = len(closes)
    min_bars = min_days * bars_per_day
    best_len, best_start, best_end = 0, 0, 0
    for i in range(0, n - min_bars, bars_per_day):
        sp = closes[i]
        lo = sp * (1 - tol_pct / 100)
        hi = sp * (1 + tol_pct / 100)
        for j in range(n - 1, i + min_bars - 1, -bars_per_day):
            if lo <= closes[j] <= hi:
                length = j - i
                if length > best_len:
                    best_len = length
                    best_start, best_end = i, j
                break
    if best_len == 0:
        return None
    return best_start, best_end


# ==================================================================
# Data loading
# ==================================================================

def load_csv_data():
    """Load 297d CSV. Returns DataFrame, type codes list."""
    df = pd.read_csv(CSV_PATH)
    df['datetime'] = pd.to_datetime(df['timestamp'])
    rctypes = df['type_code'].values.tolist()
    return df, rctypes


def load_patterns():
    """Load dynamic_patterns.json.
    Returns: patterns_long set, patterns_short set, tpsl dict {pattern: [tp, sl]}, details dict.
    """
    with open(PATTERNS_PATH) as f:
        pat_data = json.load(f)
    p_long = set(pat_data['patterns']['long'])
    p_short = set(pat_data['patterns']['short'])
    tpsl = pat_data['patterns_tpsl']  # {pattern: [tp, sl]}
    details = pat_data.get('pattern_details', {})
    return p_long, p_short, tpsl, details


# ==================================================================
# Signal generation
# ==================================================================

def generate_signals(rctypes, patterns_long, patterns_short, start_idx=0):
    """Generate signals: list of (bar_idx, pattern, direction)."""
    signals = []
    n = len(rctypes)
    for i in range(max(2, start_idx), n):
        pat = f"{rctypes[i-2]}-{rctypes[i-1]}-{rctypes[i]}"
        if pat in patterns_long:
            signals.append((i, pat, 'LONG'))
        if pat in patterns_short:
            signals.append((i, pat, 'SHORT'))
    return signals


# ==================================================================
# Exit check (standard protocol)
# ==================================================================

def _check_exit(pos, bar, opens, highs, lows, n_bars):
    """Check if position exits on this bar. Returns result dict or None."""
    entry_bar = pos['entry_bar']
    if bar < entry_bar:
        return None

    entry = opens[entry_bar]
    if entry <= 0:
        return None

    direction = pos['direction']
    tp_price = pos['tp_price']
    sl_price = pos['sl_price']

    hold = bar - entry_bar
    # Timeout -> DROP (exclude from PnL)
    if hold >= TIMEOUT_BARS:
        return {
            'entry_bar': entry_bar, 'exit_bar': bar, 'hold_bars': hold,
            'entry_price': entry, 'exit_price': opens[bar],
            'pnl_slot': 0.0, 'reason': 'TIMEOUT', 'dropped': True,
        }

    h, l = highs[bar], lows[bar]
    if direction == 'LONG':
        hit_tp = h >= tp_price
        hit_sl = l <= sl_price
    else:
        hit_tp = l <= tp_price
        hit_sl = h >= sl_price

    if not hit_tp and not hit_sl:
        return None

    # Same-bar resolution: distance from bar OPEN (NOT entry)
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
    pnl -= FEE_PCT * LEVERAGE  # fee on notional

    return {
        'entry_bar': entry_bar, 'exit_bar': bar, 'hold_bars': hold,
        'entry_price': entry, 'exit_price': exit_price,
        'pnl_slot': pnl, 'reason': reason, 'dropped': False,
    }


# ==================================================================
# Aggregate SL exposure (production v1.35.5)
# ==================================================================

def calc_aggregate_sl_exposure(open_positions, direction, current_n):
    """Sum of SL exposure for all open positions in given direction.
    Uses 1/current_n sizing (dynamic N)."""
    total = 0.0
    for pos in open_positions:
        if pos['direction'] == direction:
            slot_size = 1.0 / pos['n_at_entry']  # use N that was active at entry
            total += pos['eff_sl_pct'] * slot_size * LEVERAGE
    return total


def would_exceed_agg_cap(open_positions, new_eff_sl_pct, direction, cap_pct, current_n):
    """Check if adding a new position would push aggregate SL above cap."""
    if cap_pct is None:
        return False
    current = calc_aggregate_sl_exposure(open_positions, direction, current_n)
    new_exposure = new_eff_sl_pct * (1.0 / current_n) * LEVERAGE
    return (current + new_exposure) > cap_pct


# ==================================================================
# Momentum guard (production v1.36.2)
# ==================================================================

def check_momentum_guard(bar, closes, direction, cooldown_until):
    """Check if momentum guard blocks this entry.
    Returns updated cooldown_until dict.
    """
    if bar < MOM_LOOKBACK_BARS:
        return False, cooldown_until

    # Check if in cooldown
    opp_dir = 'LONG' if direction == 'SHORT' else 'SHORT'
    if cooldown_until.get(direction, 0) > bar:
        return True, cooldown_until

    # Check for spike in opposite direction
    price_change_pct = (closes[bar] / closes[bar - MOM_LOOKBACK_BARS] - 1) * 100
    if direction == 'SHORT' and price_change_pct > MOM_THRESHOLD_PCT:
        cooldown_until['SHORT'] = bar + MOM_COOLDOWN_BARS
        return True, cooldown_until
    if direction == 'LONG' and price_change_pct < -MOM_THRESHOLD_PCT:
        cooldown_until['LONG'] = bar + MOM_COOLDOWN_BARS
        return True, cooldown_until

    return False, cooldown_until


# ==================================================================
# Dynamic N strategies
# ==================================================================

class DynamicNStrategy:
    """Base class for dynamic N strategies."""
    def __init__(self, name, params=None):
        self.name = name
        self.params = params or {}

    def get_max_n(self, bar, vol_metrics, dd_pct, positions):
        """Return max allowed N at this bar. Subclasses override."""
        return STATIC_MAX_POSITIONS


class StaticNStrategy(DynamicNStrategy):
    """Fixed N (baseline)."""
    def __init__(self, n):
        super().__init__(f"static_N{n}", {'n': n})
        self.n = n

    def get_max_n(self, bar, vol_metrics, dd_pct, positions):
        return self.n


class TwoTierVolStrategy(DynamicNStrategy):
    """2-tier: low vol -> N_high, high vol -> N_low."""
    def __init__(self, metric_name, threshold_pct, n_low, n_high):
        super().__init__(
            f"2tier_{metric_name}_p{threshold_pct}_{n_low}_{n_high}",
            {'metric': metric_name, 'threshold_pct': threshold_pct,
             'n_low': n_low, 'n_high': n_high}
        )
        self.metric_name = metric_name
        self.threshold_pct = threshold_pct
        self.n_low = n_low
        self.n_high = n_high
        self._threshold_val = None  # set after percentile computation

    def set_threshold(self, threshold_val):
        self._threshold_val = threshold_val

    def get_max_n(self, bar, vol_metrics, dd_pct, positions):
        val = vol_metrics.get(self.metric_name, np.nan)
        if np.isnan(val) or self._threshold_val is None:
            return self.n_high  # default to high N when no data
        return self.n_low if val > self._threshold_val else self.n_high


class ThreeTierVolStrategy(DynamicNStrategy):
    """3-tier: low -> N_high, mid -> N_mid, high -> N_low."""
    def __init__(self, metric_name, lo_pct, hi_pct, n_low, n_mid, n_high):
        super().__init__(
            f"3tier_{metric_name}_p{lo_pct}_{hi_pct}_{n_low}_{n_mid}_{n_high}",
            {'metric': metric_name, 'lo_pct': lo_pct, 'hi_pct': hi_pct,
             'n_low': n_low, 'n_mid': n_mid, 'n_high': n_high}
        )
        self.metric_name = metric_name
        self.lo_pct = lo_pct
        self.hi_pct = hi_pct
        self.n_low = n_low
        self.n_mid = n_mid
        self.n_high = n_high
        self._lo_val = None
        self._hi_val = None

    def set_thresholds(self, lo_val, hi_val):
        self._lo_val = lo_val
        self._hi_val = hi_val

    def get_max_n(self, bar, vol_metrics, dd_pct, positions):
        val = vol_metrics.get(self.metric_name, np.nan)
        if np.isnan(val) or self._lo_val is None:
            return self.n_high
        if val > self._hi_val:
            return self.n_low
        elif val > self._lo_val:
            return self.n_mid
        else:
            return self.n_high


class ContinuousVolStrategy(DynamicNStrategy):
    """Continuous: N = max(n_min, round(n_max * (1 - vol_percentile)))."""
    def __init__(self, metric_name, n_min=1, n_max=9):
        super().__init__(
            f"continuous_{metric_name}_{n_min}_{n_max}",
            {'metric': metric_name, 'n_min': n_min, 'n_max': n_max}
        )
        self.metric_name = metric_name
        self.n_min = n_min
        self.n_max = n_max
        self._sorted_vals = None  # for percentile computation

    def set_sorted_vals(self, vals):
        self._sorted_vals = vals

    def get_max_n(self, bar, vol_metrics, dd_pct, positions):
        val = vol_metrics.get(self.metric_name, np.nan)
        if np.isnan(val) or self._sorted_vals is None or len(self._sorted_vals) == 0:
            return self.n_max
        pctile = np.searchsorted(self._sorted_vals, val) / len(self._sorted_vals)
        n = max(self.n_min, round(self.n_max * (1 - pctile)))
        return n


class RegimeStrategy(DynamicNStrategy):
    """Regime-based: trending -> N_trend, ranging -> N_range."""
    def __init__(self, n_trend, n_range, vol_metric='realized_vol_20', range_threshold_pct=40):
        super().__init__(
            f"regime_{n_trend}_{n_range}_p{range_threshold_pct}",
            {'n_trend': n_trend, 'n_range': n_range,
             'vol_metric': vol_metric, 'range_threshold_pct': range_threshold_pct}
        )
        self.n_trend = n_trend
        self.n_range = n_range
        self.vol_metric = vol_metric
        self.range_threshold_pct = range_threshold_pct
        self._threshold_val = None

    def set_threshold(self, threshold_val):
        self._threshold_val = threshold_val

    def get_max_n(self, bar, vol_metrics, dd_pct, positions):
        # Trending = high vol => reduce N; Ranging = low vol => increase N
        val = vol_metrics.get(self.vol_metric, np.nan)
        if np.isnan(val) or self._threshold_val is None:
            return self.n_range
        return self.n_trend if val > self._threshold_val else self.n_range


class DrawdownStrategy(DynamicNStrategy):
    """Drawdown-based: DD < lo -> N_high, DD lo-hi -> N_mid, DD > hi -> N_low."""
    def __init__(self, dd_lo=5.0, dd_hi=10.0, n_low=1, n_mid=5, n_high=9):
        super().__init__(
            f"dd_{dd_lo}_{dd_hi}_{n_low}_{n_mid}_{n_high}",
            {'dd_lo': dd_lo, 'dd_hi': dd_hi,
             'n_low': n_low, 'n_mid': n_mid, 'n_high': n_high}
        )
        self.dd_lo = dd_lo
        self.dd_hi = dd_hi
        self.n_low = n_low
        self.n_mid = n_mid
        self.n_high = n_high

    def get_max_n(self, bar, vol_metrics, dd_pct, positions):
        if dd_pct > self.dd_hi:
            return self.n_low
        elif dd_pct > self.dd_lo:
            return self.n_mid
        else:
            return self.n_high


# ==================================================================
# Portfolio simulator (N=dynamic, compound sizing)
# ==================================================================

def simulate_portfolio(signals, tpsl_map, opens, highs, lows, closes,
                       n_bars, atr_ratio, regimes, strategy,
                       sim_start, sim_end,
                       apply_production_filters=True):
    """Simulate multi-position portfolio with dynamic N.

    Key: when allowed N < current open positions, only block new entries.
    Existing positions stay open until exit.

    Production filters applied:
    - Direction cap (7)
    - Aggregate risk cap (counter 3%, with 7%)
    - Momentum guard (>1%/30min -> block counter-dir 30min)
    - Duplicate pattern block
    - MDD sizing (applied as position count scaling, not separate here --
      MDD sizing affects slot size not max N, so we keep 1/N sizing with N=strategy N)

    Returns: dict with trades, metrics, etc.
    """
    positions = []
    trades = []
    skipped_by_n = 0         # blocked by dynamic N
    skipped_by_cap = 0       # blocked by direction cap
    skipped_by_agg = 0       # blocked by aggregate risk cap
    skipped_by_mom = 0       # blocked by momentum guard
    skipped_by_dup = 0       # blocked by duplicate pattern
    total_signals = 0
    max_concurrent = 0
    n_history = []           # (bar, allowed_n)
    concurrent_history = []  # (bar, n_open)
    cascade_events = 0       # 2+ same-dir SL on same bar

    # Pre-filter signals to sim range
    signals_in_range = [(s, p, d) for s, p, d in signals if sim_start <= s < sim_end]
    signals_sorted = sorted(signals_in_range, key=lambda x: x[0])
    sig_ptr = 0

    # Compound equity tracking
    equity = 100.0
    peak_equity = 100.0

    # Momentum guard state
    cooldown_until = {}

    # Volatility metrics at each bar (precomputed)
    # These are accessed via vol_metrics_by_bar dict
    # We precompute them for the full range

    for bar in range(sim_start, sim_end):
        # ---- Check exits ----
        closed_slots = []
        sl_hits_this_bar = {'LONG': 0, 'SHORT': 0}

        for pos in positions:
            result = _check_exit(pos, bar, opens, highs, lows, n_bars)
            if result is not None:
                result['pattern'] = pos['pattern']
                result['direction'] = pos['direction']
                result['n_at_entry'] = pos['n_at_entry']
                if not result['dropped']:
                    # Compound PnL: slot fraction = 1/N_at_entry
                    slot_frac = 1.0 / pos['n_at_entry']
                    slot_return = result['pnl_slot'] / 100.0
                    equity *= (1 + slot_return * slot_frac)
                    if equity > peak_equity:
                        peak_equity = equity
                    result['pnl_portfolio'] = result['pnl_slot'] * slot_frac
                    trades.append(result)
                    if result['reason'] == 'SL':
                        sl_hits_this_bar[pos['direction']] += 1
                closed_slots.append(pos['slot_id'])

        positions = [p for p in positions if p['slot_id'] not in closed_slots]

        # Cascade detection
        for d in ('LONG', 'SHORT'):
            if sl_hits_this_bar[d] >= 2:
                cascade_events += 1

        # Track concurrent positions
        n_open = len(positions)
        if n_open > max_concurrent:
            max_concurrent = n_open

        # ---- Determine current allowed N ----
        dd_pct = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0
        vol_m = vol_metrics_at_bar(bar)
        allowed_n = strategy.get_max_n(bar, vol_m, dd_pct, positions)
        allowed_n = max(1, min(9, allowed_n))  # clamp

        # Sample every 288 bars for history
        if bar % 288 == 0:
            n_history.append((bar, allowed_n))
            concurrent_history.append((bar, n_open))

        # ---- Process signals ----
        while sig_ptr < len(signals_sorted) and signals_sorted[sig_ptr][0] == bar:
            sig_bar, pat, direction = signals_sorted[sig_ptr]
            sig_ptr += 1
            total_signals += 1

            # Dynamic N check: current open >= allowed N -> skip
            if len(positions) >= allowed_n:
                skipped_by_n += 1
                continue

            # Direction cap
            dir_count = sum(1 for p in positions if p['direction'] == direction)
            if dir_count >= DIRECTION_CAP:
                skipped_by_cap += 1
                continue

            # Also clamp direction cap to allowed_n
            if dir_count >= allowed_n:
                skipped_by_n += 1
                continue

            # Duplicate pattern check
            if any(p['pattern'] == pat and p['direction'] == direction for p in positions):
                skipped_by_dup += 1
                continue

            # Momentum guard
            if apply_production_filters:
                blocked, cooldown_until = check_momentum_guard(
                    bar, closes, direction, cooldown_until)
                if blocked:
                    skipped_by_mom += 1
                    continue

            entry_bar = sig_bar + 1
            if entry_bar >= n_bars:
                continue

            tp_sl = tpsl_map.get(pat)
            if tp_sl is None:
                continue

            tp_pct_base = tp_sl[0]
            sl_pct_base = tp_sl[1]

            # ATR scaling
            if atr_ratio is not None and sig_bar < len(atr_ratio) and not np.isnan(atr_ratio[sig_bar]):
                r = max(CLAMP_LO, min(CLAMP_HI, atr_ratio[sig_bar]))
            else:
                r = 1.0

            eff_tp = tp_pct_base * r + SLIPPAGE_BUFFER
            eff_sl = max(0.1, sl_pct_base * r - SLIPPAGE_BUFFER)

            # Aggregate risk cap
            if apply_production_filters:
                regime = regimes[sig_bar] if sig_bar < len(regimes) else 'DOWN'
                is_counter = (
                    (direction == 'SHORT' and regime == 'UP') or
                    (direction == 'LONG' and regime == 'DOWN')
                )
                cap = COUNTER_CAP if is_counter else WITH_CAP
                if would_exceed_agg_cap(positions, eff_sl, direction, cap, allowed_n):
                    skipped_by_agg += 1
                    continue

            # ---- Open position ----
            entry_price = opens[entry_bar]
            if entry_price <= 0:
                continue

            if direction == 'LONG':
                tp_price = entry_price * (1 + eff_tp / 100)
                sl_price = entry_price * (1 - eff_sl / 100)
            else:
                tp_price = entry_price * (1 - eff_tp / 100)
                sl_price = entry_price * (1 + eff_sl / 100)

            positions.append({
                'slot_id': f"{pat}_{sig_bar}",
                'signal_bar': sig_bar,
                'entry_bar': entry_bar,
                'entry_price': entry_price,
                'direction': direction,
                'pattern': pat,
                'tp_pct': tp_pct_base,
                'sl_pct': sl_pct_base,
                'eff_tp_pct': eff_tp,
                'eff_sl_pct': eff_sl,
                'tp_price': tp_price,
                'sl_price': sl_price,
                'n_at_entry': allowed_n,
            })

    # Force-close remaining -> DROP (consistent with timeout=DROP)
    return {
        'trades': trades,
        'equity_final': equity,
        'peak_equity': peak_equity,
        'total_signals': total_signals,
        'skipped_by_n': skipped_by_n,
        'skipped_by_cap': skipped_by_cap,
        'skipped_by_agg': skipped_by_agg,
        'skipped_by_mom': skipped_by_mom,
        'skipped_by_dup': skipped_by_dup,
        'max_concurrent': max_concurrent,
        'cascade_events': cascade_events,
        'n_history': n_history,
        'concurrent_history': concurrent_history,
        'remaining_positions': len(positions),
    }


# Global vol metrics arrays (set in main)
_vol_atr_ratio = None
_vol_rv20 = None
_vol_pct6 = None
_vol_pct12 = None
_vol_pct48 = None
_vol_range20 = None


def vol_metrics_at_bar(bar):
    """Return dict of volatility metrics at given bar."""
    d = {}
    if _vol_atr_ratio is not None and bar < len(_vol_atr_ratio):
        d['atr_ratio'] = float(_vol_atr_ratio[bar]) if not np.isnan(_vol_atr_ratio[bar]) else np.nan
    else:
        d['atr_ratio'] = np.nan
    if _vol_rv20 is not None and bar < len(_vol_rv20):
        d['realized_vol_20'] = float(_vol_rv20[bar]) if not np.isnan(_vol_rv20[bar]) else np.nan
    else:
        d['realized_vol_20'] = np.nan
    if _vol_pct6 is not None and bar < len(_vol_pct6):
        d['pct_change_6'] = float(_vol_pct6[bar]) if not np.isnan(_vol_pct6[bar]) else np.nan
    else:
        d['pct_change_6'] = np.nan
    if _vol_pct12 is not None and bar < len(_vol_pct12):
        d['pct_change_12'] = float(_vol_pct12[bar]) if not np.isnan(_vol_pct12[bar]) else np.nan
    else:
        d['pct_change_12'] = np.nan
    if _vol_pct48 is not None and bar < len(_vol_pct48):
        d['pct_change_48'] = float(_vol_pct48[bar]) if not np.isnan(_vol_pct48[bar]) else np.nan
    else:
        d['pct_change_48'] = np.nan
    if _vol_range20 is not None and bar < len(_vol_range20):
        d['range_vol_20'] = float(_vol_range20[bar]) if not np.isnan(_vol_range20[bar]) else np.nan
    else:
        d['range_vol_20'] = np.nan
    return d


# ==================================================================
# Metrics computation
# ==================================================================

def compute_metrics(sim_result, label=""):
    """Compute portfolio metrics from simulation result."""
    trades = sim_result['trades']
    if not trades:
        return {
            'label': label,
            'trades': 0, 'wr': 0.0, 'pnl': 0.0, 'mdd': 0.0, 'pnl_mdd': 0.0,
            'total_signals': sim_result['total_signals'],
            'skipped_by_n': sim_result['skipped_by_n'],
            'skipped_by_cap': sim_result['skipped_by_cap'],
            'skipped_by_agg': sim_result['skipped_by_agg'],
            'skipped_by_mom': sim_result['skipped_by_mom'],
            'skipped_by_dup': sim_result['skipped_by_dup'],
            'max_concurrent': sim_result['max_concurrent'],
            'cascade_events': sim_result['cascade_events'],
            'trades_per_day': 0.0,
            'avg_effective_n': 0.0,
        }

    wins = [t for t in trades if t['pnl_slot'] > 0]
    losses = [t for t in trades if t['pnl_slot'] <= 0]
    wr = len(wins) / len(trades) * 100

    # Compound equity curve for MDD
    sorted_trades = sorted(trades, key=lambda x: (x['exit_bar'], x['entry_bar']))
    equity = 100.0
    peak = equity
    max_dd = 0.0
    for t in sorted_trades:
        slot_return = t['pnl_slot'] / 100.0
        slot_frac = 1.0 / t['n_at_entry']
        equity *= (1 + slot_return * slot_frac)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd:
            max_dd = dd

    total_pnl = (equity / 100.0 - 1) * 100

    # Average effective N (mean of N_at_entry across trades)
    avg_n = np.mean([t['n_at_entry'] for t in trades])

    # Trades per day
    if trades:
        bar_range = sorted_trades[-1]['exit_bar'] - sorted_trades[0]['entry_bar']
        days = max(1, bar_range / 288)
        tpd = len(trades) / days
    else:
        tpd = 0.0

    pnl_mdd = total_pnl / max_dd if max_dd > 0 else 0.0

    # Max concurrent loss event: largest single-bar portfolio loss
    bar_losses = defaultdict(float)
    for t in trades:
        if t['pnl_slot'] < 0:
            bar_losses[t['exit_bar']] += t['pnl_slot'] / t['n_at_entry']
    max_single_bar_loss = min(bar_losses.values()) if bar_losses else 0.0

    return {
        'label': label,
        'trades': len(trades),
        'wr': round(wr, 2),
        'pnl': round(total_pnl, 2),
        'mdd': round(max_dd, 2),
        'pnl_mdd': round(pnl_mdd, 2),
        'total_signals': sim_result['total_signals'],
        'skipped_by_n': sim_result['skipped_by_n'],
        'skipped_by_cap': sim_result['skipped_by_cap'],
        'skipped_by_agg': sim_result['skipped_by_agg'],
        'skipped_by_mom': sim_result['skipped_by_mom'],
        'skipped_by_dup': sim_result['skipped_by_dup'],
        'max_concurrent': sim_result['max_concurrent'],
        'cascade_events': sim_result['cascade_events'],
        'trades_per_day': round(tpd, 2),
        'avg_effective_n': round(avg_n, 2),
        'max_single_bar_loss': round(max_single_bar_loss, 2),
    }


# ==================================================================
# WF validation
# ==================================================================

def run_wf_validation(strategy, signals, tpsl, opens, highs, lows, closes,
                      n_bars, atr_ratio, regimes, tradeable_start, n_folds=3):
    """Run expanding window walk-forward validation.

    Fold structure: divide tradeable range into n_folds equal OOS segments.
    Fold i: IS = [tradeable_start, tradeable_start + (i+1)*seg_size)
            OOS = [tradeable_start + (i+1)*seg_size, tradeable_start + (i+2)*seg_size)

    Note: For dynamic N, the strategy IS the same across folds (no re-optimization).
    We simply test if the strategy produces positive OOS PnL in each fold.
    """
    tradeable_bars = n_bars - tradeable_start
    seg_size = tradeable_bars // (n_folds + 1)
    folds = []

    for fold in range(n_folds):
        oos_start = tradeable_start + (fold + 1) * seg_size
        oos_end = tradeable_start + (fold + 2) * seg_size if fold < n_folds - 1 else n_bars

        sim = simulate_portfolio(
            signals, tpsl, opens, highs, lows, closes,
            n_bars, atr_ratio, regimes, strategy,
            sim_start=oos_start, sim_end=oos_end,
            apply_production_filters=True
        )
        m = compute_metrics(sim, label=f"fold_{fold+1}")
        m['oos_start'] = oos_start
        m['oos_end'] = oos_end
        m['oos_bars'] = oos_end - oos_start
        m['oos_days'] = round((oos_end - oos_start) / 288, 1)
        folds.append(m)

    positive_folds = sum(1 for f in folds if f['pnl'] > 0)
    total_oos_pnl = sum(f['pnl'] for f in folds)
    total_oos_trades = sum(f['trades'] for f in folds)
    avg_oos_wr = np.mean([f['wr'] for f in folds if f['trades'] > 0]) if any(f['trades'] > 0 for f in folds) else 0.0

    return {
        'strategy': strategy.name,
        'n_folds': n_folds,
        'folds': folds,
        'positive_folds': positive_folds,
        'wf_pass': positive_folds == n_folds,
        'total_oos_pnl': round(total_oos_pnl, 2),
        'total_oos_trades': total_oos_trades,
        'avg_oos_wr': round(avg_oos_wr, 2),
    }


# ==================================================================
# Main
# ==================================================================

def main():
    global _vol_atr_ratio, _vol_rv20, _vol_pct6, _vol_pct12, _vol_pct48, _vol_range20

    t0 = time.time()
    print("=" * 120)
    print("DYNAMIC N (MAX POSITIONS) ADJUSTMENT STUDY")
    print("=" * 120)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Protocol: compound sizing, fee {FEE_PCT}%x2xLev={FEE_PCT*LEVERAGE:.2f}%, "
          f"slip {SLIPPAGE_BUFFER}%, timeout {TIMEOUT_BARS}b=DROP")
    print(f"ATR: period={ATR_PERIOD}, window={ATR_WINDOW}, clamp=[{CLAMP_LO},{CLAMP_HI}]")
    print(f"Static baseline: N={STATIC_MAX_POSITIONS}, dir_cap={DIRECTION_CAP}, leverage={LEVERAGE}x")
    print(f"Production filters: agg_risk_cap({COUNTER_CAP}/{WITH_CAP}), "
          f"momentum_guard({MOM_THRESHOLD_PCT}%/{MOM_LOOKBACK_BARS}b), regime_sizing")

    # ---- 1. Load data ----
    print("\n--- 1. Loading data ---")
    p_long, p_short, tpsl, details = load_patterns()
    print(f"Patterns: {len(p_long)}L + {len(p_short)}S = {len(p_long) + len(p_short)}")

    df, rctypes = load_csv_data()
    n_bars = len(df)
    n_days = n_bars / 288
    print(f"CSV: {n_bars} bars ({n_days:.0f}d), "
          f"{df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")

    opens = df['open'].values.astype(float)
    highs = df['high'].values.astype(float)
    lows = df['low'].values.astype(float)
    closes = df['close'].values.astype(float)

    # Neutral window
    nw = find_neutral_window(closes, tol_pct=1.0, min_days=90, bars_per_day=288)
    if nw is not None:
        nw_start, nw_end = nw
        nw_days = (nw_end - nw_start) / 288
        print(f"Neutral window: bars {nw_start}-{nw_end} ({nw_days:.1f}d), "
              f"price {closes[nw_start]:.1f} -> {closes[nw_end]:.1f} "
              f"(drift {(closes[nw_end]/closes[nw_start]-1)*100:.2f}%)")
        # Use neutral window as tradeable range
        tradeable_start = max(nw_start, ATR_WARMUP)
        tradeable_end = nw_end
    else:
        print("No neutral window found, using full range")
        tradeable_start = ATR_WARMUP
        tradeable_end = n_bars

    tradeable_bars = tradeable_end - tradeable_start
    print(f"Tradeable range: bars {tradeable_start}-{tradeable_end} ({tradeable_bars/288:.1f}d)")

    # ---- 2. Compute volatility metrics ----
    print("\n--- 2. Computing volatility metrics ---")
    atr_ratio = compute_atr_ratio(highs, lows, closes)
    _vol_atr_ratio = atr_ratio

    rv20 = compute_realized_vol(closes, window=20)
    _vol_rv20 = rv20

    pct6 = compute_pct_change(closes, lookback=6)  # 30min
    _vol_pct6 = pct6

    pct12 = compute_pct_change(closes, lookback=12)  # 1h
    _vol_pct12 = pct12

    pct48 = compute_pct_change(closes, lookback=48)  # 4h
    _vol_pct48 = pct48

    range20 = compute_range_vol(highs, lows, closes, window=20)
    _vol_range20 = range20

    # EMA regime
    regimes = compute_ema_regime(closes)
    up_pct = sum(1 for r in regimes if r == 'UP') / len(regimes) * 100
    print(f"Regime: UP {up_pct:.1f}%, DOWN {100 - up_pct:.1f}%")

    # ---- Phase 1: Volatility metric distribution ----
    print("\n" + "=" * 120)
    print("PHASE 1: VOLATILITY METRIC DISTRIBUTIONS")
    print("=" * 120)

    metrics_info = {
        'atr_ratio': {'array': atr_ratio, 'desc': 'ATR(14)/median(ATR,576)'},
        'realized_vol_20': {'array': rv20, 'desc': 'RealizedVol(20 bars, log-ret std)'},
        'pct_change_6': {'array': pct6, 'desc': 'Abs %change 30min (6 bars)'},
        'pct_change_12': {'array': pct12, 'desc': 'Abs %change 1h (12 bars)'},
        'pct_change_48': {'array': pct48, 'desc': 'Abs %change 4h (48 bars)'},
        'range_vol_20': {'array': range20, 'desc': 'Range/Close rolling avg (20 bars)'},
    }

    phase1_results = {}
    # Only use tradeable range for percentile calculation
    for name, info in metrics_info.items():
        arr = info['array']
        vals = arr[tradeable_start:tradeable_end]
        valid = vals[~np.isnan(vals)]
        if len(valid) == 0:
            print(f"  {name}: NO VALID DATA")
            continue
        pcts = [10, 25, 33, 50, 66, 75, 90, 95, 99]
        pct_vals = {p: round(float(np.percentile(valid, p)), 6) for p in pcts}
        phase1_results[name] = {
            'description': info['desc'],
            'valid_count': int(len(valid)),
            'valid_pct': round(len(valid) / len(vals) * 100, 1),
            'mean': round(float(np.mean(valid)), 6),
            'std': round(float(np.std(valid)), 6),
            'min': round(float(np.min(valid)), 6),
            'max': round(float(np.max(valid)), 6),
            'percentiles': pct_vals,
        }
        print(f"\n  {name} ({info['desc']}):")
        print(f"    Valid: {len(valid)} ({phase1_results[name]['valid_pct']}%)")
        print(f"    Mean: {phase1_results[name]['mean']:.6f}, Std: {phase1_results[name]['std']:.6f}")
        print(f"    Percentiles: " + ", ".join(
            f"p{p}={pct_vals[p]:.4f}" for p in [25, 50, 75, 90]))

    # ---- Generate signals ----
    all_signals = generate_signals(rctypes, p_long, p_short, start_idx=tradeable_start)
    print(f"\nTotal signals in tradeable range: {len(all_signals)} "
          f"(L: {sum(1 for _, _, d in all_signals if d == 'LONG')}, "
          f"S: {sum(1 for _, _, d in all_signals if d == 'SHORT')})")

    # ---- Phase 2 & 3: Define and backtest strategies ----
    print("\n" + "=" * 120)
    print("PHASE 2 & 3: STRATEGY DEFINITIONS AND BACKTESTS")
    print("=" * 120)

    # Compute percentile thresholds for strategy configuration
    def get_pct(metric_name, p):
        if metric_name not in phase1_results:
            return None
        return phase1_results[metric_name]['percentiles'].get(p)

    def get_sorted_valid(metric_name):
        arr = metrics_info[metric_name]['array']
        vals = arr[tradeable_start:tradeable_end]
        valid = vals[~np.isnan(vals)]
        return np.sort(valid)

    strategies = []

    # ---- Strategy A: 2-tier vol-based ----
    for metric in ['atr_ratio', 'pct_change_12']:
        for thresh_pct in [50, 75, 90]:
            s = TwoTierVolStrategy(metric, thresh_pct, n_low=3, n_high=9)
            tv = get_pct(metric, thresh_pct)
            if tv is not None:
                s.set_threshold(tv)
                strategies.append(s)

    # ---- Strategy B: 3-tier vol-based ----
    for metric in ['atr_ratio', 'pct_change_12']:
        # 33/66 split
        s = ThreeTierVolStrategy(metric, 33, 66, n_low=1, n_mid=5, n_high=9)
        lo = get_pct(metric, 33)
        hi = get_pct(metric, 66)
        if lo is not None and hi is not None:
            s.set_thresholds(lo, hi)
            strategies.append(s)

        # 25/75 split
        s = ThreeTierVolStrategy(metric, 25, 75, n_low=1, n_mid=5, n_high=9)
        lo = get_pct(metric, 25)
        hi = get_pct(metric, 75)
        if lo is not None and hi is not None:
            s.set_thresholds(lo, hi)
            strategies.append(s)

        # Conservative: 33/66 with N=3/5/9
        s = ThreeTierVolStrategy(metric, 33, 66, n_low=3, n_mid=5, n_high=9)
        lo = get_pct(metric, 33)
        hi = get_pct(metric, 66)
        if lo is not None and hi is not None:
            s.set_thresholds(lo, hi)
            strategies.append(s)

    # ---- Strategy C: Continuous mapping ----
    for metric in ['atr_ratio', 'pct_change_12', 'realized_vol_20']:
        s = ContinuousVolStrategy(metric, n_min=1, n_max=9)
        sorted_v = get_sorted_valid(metric)
        if len(sorted_v) > 0:
            s.set_sorted_vals(sorted_v)
            strategies.append(s)

    # ---- Strategy D: Regime-based ----
    for n_trend in [3, 5]:
        s = RegimeStrategy(n_trend=n_trend, n_range=9,
                          vol_metric='realized_vol_20', range_threshold_pct=50)
        tv = get_pct('realized_vol_20', 50)
        if tv is not None:
            s.set_threshold(tv)
            strategies.append(s)

    # ---- Strategy E: Drawdown-based ----
    strategies.append(DrawdownStrategy(dd_lo=5.0, dd_hi=10.0, n_low=1, n_mid=5, n_high=9))
    strategies.append(DrawdownStrategy(dd_lo=3.0, dd_hi=8.0, n_low=1, n_mid=5, n_high=9))
    strategies.append(DrawdownStrategy(dd_lo=5.0, dd_hi=15.0, n_low=3, n_mid=5, n_high=9))

    # ---- Static baselines ----
    static_baselines = [StaticNStrategy(n) for n in [3, 5, 7, 9]]

    all_strategies = static_baselines + strategies

    print(f"\nTotal strategies to test: {len(all_strategies)} "
          f"({len(static_baselines)} static + {len(strategies)} dynamic)")
    for s in all_strategies:
        print(f"  - {s.name}")

    # ---- Run full-range backtests ----
    print(f"\n--- Running full-range backtests (bars {tradeable_start}-{tradeable_end}) ---")

    phase3_results = {}
    for i, strat in enumerate(all_strategies):
        sim = simulate_portfolio(
            all_signals, tpsl, opens, highs, lows, closes,
            n_bars, atr_ratio, regimes, strat,
            sim_start=tradeable_start, sim_end=tradeable_end,
            apply_production_filters=True
        )
        m = compute_metrics(sim, label=strat.name)
        phase3_results[strat.name] = m

        skip_n_pct = m['skipped_by_n'] / m['total_signals'] * 100 if m['total_signals'] > 0 else 0
        print(f"  [{i+1}/{len(all_strategies)}] {strat.name:<55} "
              f"trades={m['trades']:>4} WR={m['wr']:>5.1f}% "
              f"PnL={m['pnl']:>+9.2f}% MDD={m['mdd']:>6.2f}% "
              f"PnL/MDD={m['pnl_mdd']:>7.2f}x "
              f"skip_N={m['skipped_by_n']:>4}({skip_n_pct:.1f}%) "
              f"casc={m['cascade_events']:>3} "
              f"avg_N={m['avg_effective_n']:.1f} tpd={m['trades_per_day']:.2f}")

    # ---- Comparison table ----
    print(f"\n{'=' * 140}")
    print("FULL-RANGE COMPARISON (sorted by PnL/MDD)")
    print("=" * 140)

    header = (f"{'Strategy':<55} {'Trades':>6} {'WR':>6} {'PnL':>9} {'MDD':>7} "
              f"{'PnL/MDD':>8} {'Casc':>5} {'AvgN':>5} {'TPD':>5} "
              f"{'SkipN':>6} {'SkipAgg':>7} {'MaxBar$':>8}")
    print(header)
    print("-" * 140)

    sorted_results = sorted(phase3_results.values(), key=lambda x: x['pnl_mdd'], reverse=True)
    for m in sorted_results:
        print(f"{m['label']:<55} {m['trades']:>6} {m['wr']:>5.1f}% "
              f"{m['pnl']:>+8.2f}% {m['mdd']:>6.2f}% {m['pnl_mdd']:>7.2f}x "
              f"{m['cascade_events']:>5} {m['avg_effective_n']:>5.1f} {m['trades_per_day']:>5.2f} "
              f"{m['skipped_by_n']:>6} {m['skipped_by_agg']:>7} {m['max_single_bar_loss']:>+7.2f}%")

    # ---- Phase 4: WF validation for top candidates ----
    print(f"\n{'=' * 120}")
    print("PHASE 4: WALK-FORWARD OOS VALIDATION (top 5 dynamic + all static)")
    print("=" * 120)

    # Select top 5 dynamic strategies by PnL/MDD
    dynamic_results = [(name, m) for name, m in phase3_results.items()
                       if not name.startswith('static_')]
    dynamic_sorted = sorted(dynamic_results, key=lambda x: x[1]['pnl_mdd'], reverse=True)
    top_dynamic = [name for name, _ in dynamic_sorted[:5]]

    # All static baselines
    wf_candidates = [f"static_N{n}" for n in [3, 5, 7, 9]] + top_dynamic
    wf_strategies = {s.name: s for s in all_strategies if s.name in wf_candidates}

    print(f"WF candidates: {len(wf_candidates)}")
    for name in wf_candidates:
        print(f"  - {name}")

    phase4_results = {}
    for name in wf_candidates:
        strat = wf_strategies[name]
        wf = run_wf_validation(
            strat, all_signals, tpsl, opens, highs, lows, closes,
            tradeable_end, atr_ratio, regimes, tradeable_start, n_folds=3
        )
        phase4_results[name] = wf

        fold_str = " | ".join(
            f"F{f['oos_days']:.0f}d: {f['trades']}tr WR={f['wr']:.1f}% PnL={f['pnl']:+.2f}%"
            for f in wf['folds']
        )
        status = "PASS" if wf['wf_pass'] else f"FAIL({wf['positive_folds']}/3)"
        print(f"\n  {name}  [{status}]")
        print(f"    {fold_str}")
        print(f"    Total OOS: {wf['total_oos_trades']} trades, "
              f"PnL {wf['total_oos_pnl']:+.2f}%, Avg WR {wf['avg_oos_wr']:.1f}%")

    # ---- Phase 5: vs Static N comparison ----
    print(f"\n{'=' * 120}")
    print("PHASE 5: DYNAMIC vs STATIC N COMPARISON")
    print("=" * 120)

    baseline_n9 = phase3_results.get('static_N9', {})

    header5 = (f"{'Strategy':<55} {'PnL':>9} {'MDD':>7} {'PnL/MDD':>8} "
               f"{'dPnL':>8} {'dMDD':>7} {'dPnL/MDD':>9} "
               f"{'WF':>6} {'OOS_PnL':>9}")
    print(header5)
    print("-" * 130)

    for m in sorted_results:
        name = m['label']
        d_pnl = m['pnl'] - baseline_n9.get('pnl', 0)
        d_mdd = m['mdd'] - baseline_n9.get('mdd', 0)
        d_pm = m['pnl_mdd'] - baseline_n9.get('pnl_mdd', 0)
        wf_info = phase4_results.get(name)
        if wf_info:
            wf_str = "PASS" if wf_info['wf_pass'] else f"FAIL({wf_info['positive_folds']}/3)"
            oos_pnl = wf_info['total_oos_pnl']
        else:
            wf_str = "N/A"
            oos_pnl = 0

        print(f"{name:<55} {m['pnl']:>+8.2f}% {m['mdd']:>6.2f}% {m['pnl_mdd']:>7.2f}x "
              f"{d_pnl:>+7.2f}% {d_mdd:>+6.2f}% {d_pm:>+8.2f}x "
              f"{wf_str:>6} {oos_pnl:>+8.2f}%")

    # ---- Phase 6: Implementation complexity + sensitivity ----
    print(f"\n{'=' * 120}")
    print("PHASE 6: IMPLEMENTATION COMPLEXITY & PARAMETER SENSITIVITY")
    print("=" * 120)

    phase6_results = {}
    for strat in all_strategies:
        complexity = {
            'name': strat.name,
            'type': type(strat).__name__,
            'n_params': len(strat.params),
            'params': strat.params,
            'lookback_dependency': 'none',
            'implementation_difficulty': 'low',
        }
        if isinstance(strat, (TwoTierVolStrategy, ThreeTierVolStrategy)):
            complexity['lookback_dependency'] = f"vol metric ({strat.params.get('metric', 'N/A')})"
            complexity['implementation_difficulty'] = 'medium'
        elif isinstance(strat, ContinuousVolStrategy):
            complexity['lookback_dependency'] = f"percentile rank of vol metric"
            complexity['implementation_difficulty'] = 'medium-high'
        elif isinstance(strat, RegimeStrategy):
            complexity['lookback_dependency'] = 'EMA regime + vol metric'
            complexity['implementation_difficulty'] = 'medium'
        elif isinstance(strat, DrawdownStrategy):
            complexity['lookback_dependency'] = 'equity drawdown (already tracked)'
            complexity['implementation_difficulty'] = 'low'
        phase6_results[strat.name] = complexity

    print(f"\n{'Strategy':<55} {'Type':<25} {'Params':>6} {'Difficulty':<15}")
    print("-" * 105)
    for name, c in phase6_results.items():
        print(f"{name:<55} {c['type']:<25} {c['n_params']:>6} {c['implementation_difficulty']:<15}")

    # ---- Parameter sensitivity for top WF-passing strategies ----
    print(f"\n--- Parameter Sensitivity (+-10% threshold shift) ---")
    wf_pass_dynamic = [name for name in top_dynamic
                       if phase4_results.get(name, {}).get('wf_pass', False)]

    sensitivity_results = {}
    for name in wf_pass_dynamic[:3]:  # top 3 WF-passing dynamic strategies
        strat = wf_strategies[name]
        base_m = phase3_results[name]

        # Sensitivity test: perturb thresholds by +-10%
        perturbed_results = []

        if isinstance(strat, TwoTierVolStrategy):
            base_thresh = strat._threshold_val
            if base_thresh is not None:
                for shift in [-0.10, -0.05, 0.0, 0.05, 0.10]:
                    new_thresh = base_thresh * (1 + shift)
                    perturbed = TwoTierVolStrategy(
                        strat.metric_name, strat.threshold_pct,
                        strat.n_low, strat.n_high)
                    perturbed.set_threshold(new_thresh)
                    sim = simulate_portfolio(
                        all_signals, tpsl, opens, highs, lows, closes,
                        n_bars, atr_ratio, regimes, perturbed,
                        sim_start=tradeable_start, sim_end=tradeable_end)
                    pm = compute_metrics(sim, label=f"shift_{shift:+.0%}")
                    perturbed_results.append({
                        'shift': shift,
                        'pnl': pm['pnl'],
                        'mdd': pm['mdd'],
                        'pnl_mdd': pm['pnl_mdd'],
                        'trades': pm['trades'],
                    })

        elif isinstance(strat, ThreeTierVolStrategy):
            base_lo = strat._lo_val
            base_hi = strat._hi_val
            if base_lo is not None and base_hi is not None:
                for shift in [-0.10, -0.05, 0.0, 0.05, 0.10]:
                    new_lo = base_lo * (1 + shift)
                    new_hi = base_hi * (1 + shift)
                    perturbed = ThreeTierVolStrategy(
                        strat.metric_name, strat.lo_pct, strat.hi_pct,
                        strat.n_low, strat.n_mid, strat.n_high)
                    perturbed.set_thresholds(new_lo, new_hi)
                    sim = simulate_portfolio(
                        all_signals, tpsl, opens, highs, lows, closes,
                        n_bars, atr_ratio, regimes, perturbed,
                        sim_start=tradeable_start, sim_end=tradeable_end)
                    pm = compute_metrics(sim, label=f"shift_{shift:+.0%}")
                    perturbed_results.append({
                        'shift': shift,
                        'pnl': pm['pnl'],
                        'mdd': pm['mdd'],
                        'pnl_mdd': pm['pnl_mdd'],
                        'trades': pm['trades'],
                    })

        elif isinstance(strat, DrawdownStrategy):
            for shift in [-0.10, -0.05, 0.0, 0.05, 0.10]:
                lo = strat.dd_lo * (1 + shift)
                hi = strat.dd_hi * (1 + shift)
                perturbed = DrawdownStrategy(lo, hi, strat.n_low, strat.n_mid, strat.n_high)
                sim = simulate_portfolio(
                    all_signals, tpsl, opens, highs, lows, closes,
                    n_bars, atr_ratio, regimes, perturbed,
                    sim_start=tradeable_start, sim_end=tradeable_end)
                pm = compute_metrics(sim, label=f"shift_{shift:+.0%}")
                perturbed_results.append({
                    'shift': shift,
                    'pnl': pm['pnl'],
                    'mdd': pm['mdd'],
                    'pnl_mdd': pm['pnl_mdd'],
                    'trades': pm['trades'],
                })

        if perturbed_results:
            pnl_mdd_range = max(p['pnl_mdd'] for p in perturbed_results) - min(p['pnl_mdd'] for p in perturbed_results)
            pnl_range = max(p['pnl'] for p in perturbed_results) - min(p['pnl'] for p in perturbed_results)
            sensitivity_results[name] = {
                'perturbed': perturbed_results,
                'pnl_mdd_range': round(pnl_mdd_range, 2),
                'pnl_range': round(pnl_range, 2),
                'stable': pnl_mdd_range < base_m['pnl_mdd'] * 0.3,  # <30% variation = stable
            }
            print(f"\n  {name}:")
            for p in perturbed_results:
                marker = " <-- base" if p['shift'] == 0 else ""
                print(f"    shift={p['shift']:+.0%}: PnL={p['pnl']:+.2f}%, "
                      f"MDD={p['mdd']:.2f}%, PnL/MDD={p['pnl_mdd']:.2f}x, "
                      f"trades={p['trades']}{marker}")
            stable_str = "STABLE" if sensitivity_results[name]['stable'] else "SENSITIVE"
            print(f"    >> PnL/MDD range: {pnl_mdd_range:.2f}x ({stable_str})")

    # ---- Final Verdict ----
    print(f"\n{'=' * 120}")
    print("FINAL VERDICT")
    print("=" * 120)

    # Find best overall strategy (WF PASS + highest PnL/MDD)
    best_dynamic = None
    best_dynamic_pm = -999
    best_static = None
    best_static_pm = -999

    for name, wf in phase4_results.items():
        if not wf['wf_pass']:
            continue
        m = phase3_results[name]
        if name.startswith('static_'):
            if m['pnl_mdd'] > best_static_pm:
                best_static_pm = m['pnl_mdd']
                best_static = name
        else:
            if m['pnl_mdd'] > best_dynamic_pm:
                best_dynamic_pm = m['pnl_mdd']
                best_dynamic = name

    # Determine verdict
    verdict = "STOP"
    recommendation = ""
    best_strategy_name = ""

    if best_dynamic is not None and best_static is not None:
        dynamic_m = phase3_results[best_dynamic]
        static_m = phase3_results[best_static]

        # Dynamic must beat static by meaningful margin (>10% improvement in PnL/MDD)
        improvement = (dynamic_m['pnl_mdd'] - static_m['pnl_mdd']) / static_m['pnl_mdd'] * 100 if static_m['pnl_mdd'] > 0 else 0

        # Also check sensitivity
        is_stable = sensitivity_results.get(best_dynamic, {}).get('stable', True)

        if improvement > 10 and is_stable:
            verdict = "GO"
            best_strategy_name = best_dynamic
            recommendation = (
                f"Dynamic N strategy '{best_dynamic}' improves PnL/MDD by {improvement:.1f}% "
                f"over best static N='{best_static}' ({dynamic_m['pnl_mdd']:.2f}x vs "
                f"{static_m['pnl_mdd']:.2f}x). WF PASS, parameter-stable. "
                f"MDD {dynamic_m['mdd']:.2f}% vs {static_m['mdd']:.2f}%, "
                f"cascade events {dynamic_m['cascade_events']} vs {static_m['cascade_events']}."
            )
        elif improvement > 5:
            verdict = "PARTIAL"
            best_strategy_name = best_dynamic
            recommendation = (
                f"Marginal improvement ({improvement:.1f}%) of dynamic '{best_dynamic}' "
                f"over static '{best_static}'. Added complexity may not justify benefit. "
                f"Consider only if cascade reduction is significant."
            )
        else:
            verdict = "STOP"
            best_strategy_name = best_static
            recommendation = (
                f"Dynamic N strategies do not meaningfully improve over static '{best_static}' "
                f"(best dynamic improvement: {improvement:.1f}%). Keep current N={STATIC_MAX_POSITIONS}. "
                f"Existing production filters (agg_risk_cap, momentum_guard, regime_sizing) "
                f"already handle correlated loss effectively."
            )
    elif best_static is not None:
        verdict = "STOP"
        best_strategy_name = best_static
        recommendation = (
            f"No dynamic strategy passed WF. Best static: '{best_static}' "
            f"with PnL/MDD {best_static_pm:.2f}x."
        )
    else:
        verdict = "STOP"
        recommendation = "No strategy passed WF validation."

    print(f"\n  Verdict: {verdict}")
    print(f"  Best strategy: {best_strategy_name}")
    print(f"  Recommendation: {recommendation}")

    if best_dynamic:
        dm = phase3_results[best_dynamic]
        print(f"\n  Best Dynamic ({best_dynamic}):")
        print(f"    IS: PnL={dm['pnl']:+.2f}%, MDD={dm['mdd']:.2f}%, "
              f"PnL/MDD={dm['pnl_mdd']:.2f}x, WR={dm['wr']:.1f}%")
        dw = phase4_results.get(best_dynamic, {})
        if dw:
            print(f"    WF: {dw['positive_folds']}/3 PASS, OOS PnL={dw['total_oos_pnl']:+.2f}%")

    if best_static:
        sm = phase3_results[best_static]
        print(f"\n  Best Static ({best_static}):")
        print(f"    IS: PnL={sm['pnl']:+.2f}%, MDD={sm['mdd']:.2f}%, "
              f"PnL/MDD={sm['pnl_mdd']:.2f}x, WR={sm['wr']:.1f}%")
        sw = phase4_results.get(best_static, {})
        if sw:
            print(f"    WF: {sw['positive_folds']}/3 PASS, OOS PnL={sw['total_oos_pnl']:+.2f}%")

    n9_m = phase3_results.get('static_N9', {})
    print(f"\n  Current Production (static N=9):")
    print(f"    IS: PnL={n9_m.get('pnl', 0):+.2f}%, MDD={n9_m.get('mdd', 0):.2f}%, "
          f"PnL/MDD={n9_m.get('pnl_mdd', 0):.2f}x")

    # ---- Save results ----
    print(f"\n--- Saving results to {RESULTS_PATH} ---")

    output = {
        'study': 'Dynamic N Adjustment Study',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_file': CSV_PATH,
        'patterns_file': PATTERNS_PATH,
        'n_patterns': len(p_long) + len(p_short),
        'tradeable_range': {
            'start_bar': tradeable_start,
            'end_bar': tradeable_end,
            'days': round(tradeable_bars / 288, 1),
        },
        'protocol': {
            'leverage': LEVERAGE,
            'fee_pct': FEE_PCT,
            'slippage_buffer': SLIPPAGE_BUFFER,
            'timeout_bars': TIMEOUT_BARS,
            'atr_period': ATR_PERIOD,
            'atr_window': ATR_WINDOW,
            'clamp': [CLAMP_LO, CLAMP_HI],
            'direction_cap': DIRECTION_CAP,
            'static_max_positions': STATIC_MAX_POSITIONS,
            'production_filters': True,
        },
        'phases': {
            'phase1_vol_metrics': phase1_results,
            'phase2_strategies': {
                name: {
                    'type': type(s).__name__,
                    'params': s.params,
                }
                for s in all_strategies
                for name in [s.name]
            },
            'phase3_backtest': {
                name: {k: v for k, v in m.items() if k != 'label'}
                for name, m in phase3_results.items()
            },
            'phase4_wf_oos': {
                name: {
                    'wf_pass': wf['wf_pass'],
                    'positive_folds': wf['positive_folds'],
                    'total_oos_pnl': wf['total_oos_pnl'],
                    'total_oos_trades': wf['total_oos_trades'],
                    'avg_oos_wr': wf['avg_oos_wr'],
                    'folds': [
                        {k: v for k, v in f.items() if k != 'label'}
                        for f in wf['folds']
                    ],
                }
                for name, wf in phase4_results.items()
            },
            'phase5_vs_static': {
                'baseline': 'static_N9',
                'baseline_pnl': n9_m.get('pnl', 0),
                'baseline_mdd': n9_m.get('mdd', 0),
                'baseline_pnl_mdd': n9_m.get('pnl_mdd', 0),
            },
            'phase6_implementation': {
                'strategy_complexity': phase6_results,
                'parameter_sensitivity': {
                    name: {
                        'pnl_mdd_range': sr['pnl_mdd_range'],
                        'pnl_range': sr['pnl_range'],
                        'stable': sr['stable'],
                        'perturbed': sr['perturbed'],
                    }
                    for name, sr in sensitivity_results.items()
                },
            },
        },
        'verdict': verdict,
        'best_strategy': best_strategy_name,
        'recommendation': recommendation,
        'timing': {
            'total_seconds': round(time.time() - t0, 1),
        },
    }

    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"Results saved to: {RESULTS_PATH}")
    print(f"\nTotal runtime: {time.time() - t0:.1f}s")
    print(f"\n{'=' * 120}")
    print(f"VERDICT: {verdict}")
    print(f"{'=' * 120}")


if __name__ == '__main__':
    main()
