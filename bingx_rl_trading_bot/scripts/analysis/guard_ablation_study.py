#!/usr/bin/env python3
"""
Guard Ablation Study — 10 Position Sizing Mechanisms Individual PnL Impact + Redundancy
========================================================================================

Goal: Evaluate each of 10 production guard/sizing mechanisms:
  1. Individual PnL/MDD impact (ablation + additive)
  2. Redundancy/overlap detection between suspected pairs

10 Mechanisms:
  Blockers (True/False):
    G1: Direction Cap (max 7 same-direction)
    G2: Momentum Guard (BTC >1%/30min → block reverse 30min)
    G3: Loss Burst Brake (2 same-dir losses/24h → block 12h)
    G4: Correlation-Aware Entry (dir >=70% + counter-regime → block)
    G5: Aggregate Risk Cap (SL exposure sum > cap → block)

  Multipliers (0.x ~ 1.x):
    M1: MDD Sizing (DD 3%→full, 15%→25% linear)
    M2: Regime Sizing (counter-regime → ×0.3)
    M3: Equity Curve Sizing (per-direction cum_pnl < SMA(30) → ×0.5)
    M4: Adaptive Leverage (WR Confidence → 1-3x)
    M5: Vol Mult ATR (ATR ratio → TP/SL scaling)

Phases:
  1. Ablation (ALL minus ONE) — 11 scenarios
  2. Additive (BARE plus ONE) — 11 scenarios
  3. Overlap Detection — 6 pairs × 3 = 18 scenarios
  4. WF validation for key findings

Standard Research Protocol:
  - Production classify_candle import
  - LEVERAGE adaptive (1-3x, wr_confidence w=12) when M4=on, else fixed 3x
  - FEE = FEE_PCT * leverage (notional basis)
  - Timeout = DROP, Same-bar = abs(tp - bar_open)
  - Entry = next-bar open, ATR-scaled TP/SL
  - Compound (multiplicative) sizing
  - WF: 3-fold expanding window

Author: Research Agent
Date: 2026-03-03
"""

import os
import sys
import json
import warnings
from collections import deque
from dataclasses import dataclass, replace, asdict
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
# Constants — matching production exactly
# ============================================================
MAX_LEVERAGE = 3
MIN_LEVERAGE = 1
FEE_PCT = 0.10
SLIPPAGE_BUFFER = 0.02
TIMEOUT_BARS = 864
N_SLOTS = 9
DIRECTION_CAP = 7
REGIME_MULT = 0.3
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

# Loss burst brake params (v1.37.0)
LBB_THRESHOLD = 2
LBB_WINDOW_BARS = 288
LBB_BLOCK_BARS = 144

# MDD sizing params (v1.35.2)
MDD_FULL_BELOW = 3.0
MDD_MIN_ABOVE = 15.0
MDD_MIN_SCALE = 0.25

# Adaptive leverage (v1.39.0)
ADAPTIVE_LEV_WINDOW = 12
EXPECTED_WR = 0.732
REF_EDGE = 0.00126

# Equity Curve Sizing (v1.40.0)
EC_SMA_TRADES = 30
EC_SIZE_MULT = 0.5
EC_MAX_HISTORY = 100

DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
PATTERNS_FILE = os.path.join(_PROJECT_ROOT, 'results', 'dynamic_patterns.json')
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'guard_ablation_study.json')


# ============================================================
# Guard Configuration
# ============================================================

@dataclass
class GuardConfig:
    """Toggle each of 10 production guard/sizing mechanisms."""
    direction_cap: bool = True       # G1
    momentum_guard: bool = True      # G2
    loss_burst_brake: bool = True    # G3
    correlation_aware: bool = True   # G4
    aggregate_risk_cap: bool = True  # G5
    mdd_sizing: bool = True          # M1
    regime_sizing: bool = True       # M2
    equity_curve_sizing: bool = True # M3
    adaptive_leverage: bool = True   # M4
    vol_mult_atr: bool = True        # M5

    def label(self):
        """Short label for enabled mechanisms."""
        flags = []
        if self.direction_cap: flags.append('G1')
        if self.momentum_guard: flags.append('G2')
        if self.loss_burst_brake: flags.append('G3')
        if self.correlation_aware: flags.append('G4')
        if self.aggregate_risk_cap: flags.append('G5')
        if self.mdd_sizing: flags.append('M1')
        if self.regime_sizing: flags.append('M2')
        if self.equity_curve_sizing: flags.append('M3')
        if self.adaptive_leverage: flags.append('M4')
        if self.vol_mult_atr: flags.append('M5')
        return '+'.join(flags) if flags else 'BARE'

    def count_enabled(self):
        return sum([
            self.direction_cap, self.momentum_guard, self.loss_burst_brake,
            self.correlation_aware, self.aggregate_risk_cap, self.mdd_sizing,
            self.regime_sizing, self.equity_curve_sizing, self.adaptive_leverage,
            self.vol_mult_atr
        ])


ALL_ON = GuardConfig()
ALL_OFF = GuardConfig(
    direction_cap=False, momentum_guard=False, loss_burst_brake=False,
    correlation_aware=False, aggregate_risk_cap=False, mdd_sizing=False,
    regime_sizing=False, equity_curve_sizing=False, adaptive_leverage=False,
    vol_mult_atr=False
)

GUARD_NAMES = {
    'G1_direction_cap': 'direction_cap',
    'G2_momentum_guard': 'momentum_guard',
    'G3_loss_burst_brake': 'loss_burst_brake',
    'G4_correlation_aware': 'correlation_aware',
    'G5_aggregate_risk_cap': 'aggregate_risk_cap',
    'M1_mdd_sizing': 'mdd_sizing',
    'M2_regime_sizing': 'regime_sizing',
    'M3_equity_curve_sizing': 'equity_curve_sizing',
    'M4_adaptive_leverage': 'adaptive_leverage',
    'M5_vol_mult_atr': 'vol_mult_atr',
}


# ============================================================
# Data Loading & Preprocessing (identical to mdd_reduction_study)
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
# Adaptive Leverage (production: wr_confidence, w=12)
# ============================================================

def get_leverage_wr_confidence(rolling_wr_frac, rolling_edge_frac):
    if EXPECTED_WR <= 0:
        return MIN_LEVERAGE
    wr_conf = clamp(rolling_wr_frac / EXPECTED_WR, 0, 1)
    edge_q = clamp(rolling_edge_frac / REF_EDGE, 0, 1.5) if REF_EDGE > 0 else 0.5
    combined = clamp(wr_conf * edge_q, 0, 1)
    return MIN_LEVERAGE + (MAX_LEVERAGE - MIN_LEVERAGE) * combined


# ============================================================
# Portfolio Simulator — ALL mechanisms toggleable via GuardConfig
# ============================================================

def _check_exit(pos, bar, opens, highs, lows, n_bars, atr_ratio, gc):
    """Check exit for a single position."""
    entry_bar = pos['entry_bar']
    if bar < entry_bar:
        return None
    entry = opens[entry_bar]
    if entry <= 0:
        return None

    sig_bar = pos['signal_bar']
    # M5: Vol Mult ATR — when off, ratio=1.0
    if gc.vol_mult_atr and atr_ratio is not None and sig_bar < len(atr_ratio) and not np.isnan(atr_ratio[sig_bar]):
        r = clamp(atr_ratio[sig_bar], ATR_CLAMP_LO, ATR_CLAMP_HI)
    else:
        r = 1.0

    eff_tp = pos['tp_pct'] * r + SLIPPAGE_BUFFER
    eff_sl = max(0.1, pos['sl_pct'] * r - SLIPPAGE_BUFFER)
    direction = pos['direction']

    if direction == 'LONG':
        tp_price = entry * (1 + eff_tp / 100)
        sl_price = entry * (1 - eff_sl / 100)
    else:
        tp_price = entry * (1 - eff_tp / 100)
        sl_price = entry * (1 + eff_sl / 100)

    if (bar - entry_bar) >= TIMEOUT_BARS:
        return {'pnl_slot': 0, 'reason': 'TIMEOUT', 'drop': True,
                'entry_bar': entry_bar, 'exit_bar': bar}

    h, l = highs[bar], lows[bar]
    if direction == 'LONG':
        hit_tp, hit_sl = h >= tp_price, l <= sl_price
    else:
        hit_tp, hit_sl = l <= tp_price, h >= sl_price

    if not hit_tp and not hit_sl:
        return None

    if hit_tp and hit_sl:
        if abs(tp_price - opens[bar]) <= abs(sl_price - opens[bar]):
            exit_price, reason = tp_price, 'TP'
        else:
            exit_price, reason = sl_price, 'SL'
    elif hit_tp:
        exit_price, reason = tp_price, 'TP'
    else:
        exit_price, reason = sl_price, 'SL'

    entry_lev = pos['leverage']
    fee = FEE_PCT * entry_lev
    if direction == 'LONG':
        pnl = (exit_price / entry - 1) * 100 * entry_lev
    else:
        pnl = (1 - exit_price / entry) * 100 * entry_lev
    pnl -= fee

    return {'pnl_slot': pnl, 'reason': reason, 'drop': False,
            'entry_bar': entry_bar, 'exit_bar': bar, 'leverage': entry_lev}


def _get_mdd_size_scale(equity, peak_equity):
    if peak_equity <= 0:
        return 1.0
    dd_pct = (peak_equity - equity) / peak_equity * 100
    if dd_pct <= MDD_FULL_BELOW:
        return 1.0
    if dd_pct >= MDD_MIN_ABOVE:
        return MDD_MIN_SCALE
    return 1.0 - (1.0 - MDD_MIN_SCALE) * (dd_pct - MDD_FULL_BELOW) / (MDD_MIN_ABOVE - MDD_FULL_BELOW)


def portfolio_sim(signal_tuples, opens, highs, lows, closes, n_bars,
                  atr_ratio, ema_slope, start_bar, end_bar, gc=None):
    """N-pos portfolio with all production features, each toggleable via GuardConfig."""
    if gc is None:
        gc = ALL_ON

    size_pct = 100.0 / N_SLOTS
    positions = []
    trades = []
    equity = 100.0
    peak_equity = 100.0
    max_dd = 0.0

    # Rolling WR trackers for adaptive leverage (M4)
    recent_results = deque(maxlen=ADAPTIVE_LEV_WINDOW)
    recent_pnls = deque(maxlen=ADAPTIVE_LEV_WINDOW)

    # Loss burst brake state (G3): per-direction loss timestamps
    lbb_losses = {'LONG': [], 'SHORT': []}
    lbb_blocked_until = {'LONG': -1, 'SHORT': -1}

    # Momentum guard state (G2)
    momentum_pause_until = {'LONG': -1, 'SHORT': -1}

    # Equity curve sizing state (M3): per-direction cum PnL tracking (v1.40.0)
    ec_cum_pnls = {'LONG': [], 'SHORT': []}

    signals_in_range = [(s, p, d, tp, sl) for s, p, d, tp, sl in signal_tuples
                        if start_bar <= s < end_bar]
    signals_sorted = sorted(signals_in_range, key=lambda x: x[0])
    sig_idx = 0

    blocked_counts = {
        'G1_direction_cap': 0, 'G2_momentum_guard': 0, 'G3_loss_burst_brake': 0,
        'G4_correlation_aware': 0, 'G5_aggregate_risk_cap': 0,
    }

    for bar in range(start_bar, end_bar):
        # --- Exits ---
        closed_slots = []
        bar_pnl_sum = 0.0

        for pos in positions:
            result = _check_exit(pos, bar, opens, highs, lows, n_bars, atr_ratio, gc)
            if result is not None:
                if result.get('drop', False):
                    closed_slots.append(pos['slot'])
                    continue
                result['pattern'] = pos['pattern']
                result['direction'] = pos['direction']
                sm = pos.get('size_mult', 1.0)
                pnl_portfolio = result['pnl_slot'] * (size_pct / 100) * sm
                result['pnl_portfolio'] = pnl_portfolio
                result['size_mult'] = sm
                result['leverage'] = pos['leverage']
                trades.append(result)
                closed_slots.append(pos['slot'])
                bar_pnl_sum += pnl_portfolio

                is_win = result['pnl_slot'] > 0
                recent_results.append(1 if is_win else 0)
                recent_pnls.append(result['pnl_slot'])

                # M3: Equity curve per-direction tracking
                d = pos['direction']
                prev_cum = ec_cum_pnls[d][-1] if ec_cum_pnls[d] else 0.0
                ec_cum_pnls[d].append(prev_cum + result['pnl_slot'])
                if len(ec_cum_pnls[d]) > EC_MAX_HISTORY:
                    ec_cum_pnls[d] = ec_cum_pnls[d][-EC_MAX_HISTORY:]

                # G3: Loss burst brake — record losses
                if not is_win and gc.loss_burst_brake:
                    lbb_losses[d].append(bar)
                    lbb_losses[d] = [b for b in lbb_losses[d]
                                     if bar - b <= LBB_WINDOW_BARS]
                    if len(lbb_losses[d]) >= LBB_THRESHOLD:
                        lbb_blocked_until[d] = bar + LBB_BLOCK_BARS
                        lbb_losses[d] = []

        positions = [p for p in positions if p['slot'] not in closed_slots]

        equity += bar_pnl_sum
        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0
        if dd > max_dd:
            max_dd = dd

        # G2: Momentum guard — track price changes
        if gc.momentum_guard and MOMENTUM_LOOKBACK > 0 and bar >= MOMENTUM_LOOKBACK:
            price_now, price_ago = closes[bar], closes[bar - MOMENTUM_LOOKBACK]
            if price_ago > 0:
                pct_change = (price_now / price_ago - 1) * 100
                if pct_change > MOMENTUM_THRESHOLD:
                    momentum_pause_until['SHORT'] = bar + MOMENTUM_COOLDOWN
                elif pct_change < -MOMENTUM_THRESHOLD:
                    momentum_pause_until['LONG'] = bar + MOMENTUM_COOLDOWN

        # --- Entries ---
        while sig_idx < len(signals_sorted) and signals_sorted[sig_idx][0] == bar:
            sig_bar, pat, direction, tp_pct, sl_pct = signals_sorted[sig_idx]
            sig_idx += 1

            if len(positions) >= N_SLOTS:
                continue
            if any(p['pattern'] == pat for p in positions):
                continue
            entry_bar = sig_bar + 1
            if entry_bar >= n_bars:
                continue

            # G1: Direction cap
            if gc.direction_cap:
                dir_count = sum(1 for p in positions if p['direction'] == direction)
                if dir_count >= DIRECTION_CAP:
                    blocked_counts['G1_direction_cap'] += 1
                    continue

            # G2: Momentum guard
            if gc.momentum_guard and bar < momentum_pause_until.get(direction, -1):
                blocked_counts['G2_momentum_guard'] += 1
                continue

            # G3: Loss burst brake
            if gc.loss_burst_brake and bar < lbb_blocked_until.get(direction, -1):
                blocked_counts['G3_loss_burst_brake'] += 1
                continue

            # G4: Correlation-aware entry
            if gc.correlation_aware and len(positions) >= 2:
                same_dir = sum(1 for p in positions if p['direction'] == direction)
                dir_ratio = same_dir / len(positions)
                is_uptrend = ema_slope[bar] > 0 if bar < len(ema_slope) else False
                is_counter = ((is_uptrend and direction == 'SHORT') or
                              (not is_uptrend and direction == 'LONG'))
                if dir_ratio >= 0.70 and is_counter:
                    blocked_counts['G4_correlation_aware'] += 1
                    continue

            # Size multiplier: start at 1.0
            sm = 1.0

            # M2: Regime sizing
            if gc.regime_sizing and REGIME_MULT is not None and bar < len(ema_slope):
                s = ema_slope[bar]
                if (s > 0 and direction == 'SHORT') or (s <= 0 and direction == 'LONG'):
                    sm = REGIME_MULT

            # M1: MDD sizing
            if gc.mdd_sizing:
                mdd_scale = _get_mdd_size_scale(equity, peak_equity)
                sm *= mdd_scale

            # M3: Equity curve sizing (per-direction, v1.40.0)
            if gc.equity_curve_sizing:
                dir_pnls = ec_cum_pnls.get(direction, [])
                if len(dir_pnls) >= EC_SMA_TRADES:
                    current_cum = dir_pnls[-1]
                    sma_val = np.mean(dir_pnls[-EC_SMA_TRADES:])
                    if current_cum < sma_val:
                        sm *= EC_SIZE_MULT

            # M4: Adaptive leverage
            if gc.adaptive_leverage:
                rolling_wr_frac = (sum(recent_results) / len(recent_results)
                                   if recent_results else 0.5)
                rolling_edge_frac = (np.mean(list(recent_pnls)) / 100.0
                                     if recent_pnls else 0.0)
                entry_leverage = get_leverage_wr_confidence(rolling_wr_frac, rolling_edge_frac)
                entry_leverage = clamp(entry_leverage, MIN_LEVERAGE, MAX_LEVERAGE)
            else:
                entry_leverage = MAX_LEVERAGE

            # G5: Aggregate risk cap
            if gc.aggregate_risk_cap:
                is_uptrend = ema_slope[bar] > 0 if bar < len(ema_slope) else False
                is_counter = ((is_uptrend and direction == 'SHORT') or
                              (not is_uptrend and direction == 'LONG'))
                cap_pct = AGG_RISK_COUNTER if is_counter else AGG_RISK_WITH

                existing_exposure = 0.0
                for p in positions:
                    if p['direction'] == direction:
                        p_sig = p['signal_bar']
                        p_r = 1.0
                        if gc.vol_mult_atr and atr_ratio is not None and p_sig < len(atr_ratio) and not np.isnan(atr_ratio[p_sig]):
                            p_r = clamp(atr_ratio[p_sig], ATR_CLAMP_LO, ATR_CLAMP_HI)
                        existing_exposure += (p['sl_pct'] * p_r * (1.0 / N_SLOTS)
                                              * p['leverage'] * p.get('size_mult', 1.0))

                new_r = 1.0
                if gc.vol_mult_atr and atr_ratio is not None and sig_bar < len(atr_ratio) and not np.isnan(atr_ratio[sig_bar]):
                    new_r = clamp(atr_ratio[sig_bar], ATR_CLAMP_LO, ATR_CLAMP_HI)
                new_exposure = sl_pct * new_r * (1.0 / N_SLOTS) * entry_leverage * sm
                if existing_exposure + new_exposure > cap_pct:
                    blocked_counts['G5_aggregate_risk_cap'] += 1
                    continue

            positions.append({
                'slot': f"{pat}_{sig_bar}", 'signal_bar': sig_bar,
                'entry_bar': entry_bar, 'direction': direction,
                'pattern': pat, 'tp_pct': tp_pct, 'sl_pct': sl_pct,
                'size_mult': sm, 'leverage': entry_leverage,
            })

    # Force-close remaining
    for pos in positions:
        if pos['entry_bar'] >= n_bars:
            continue
        entry = opens[pos['entry_bar']]
        if entry <= 0:
            continue
        exit_bar = min(end_bar - 1, n_bars - 1)
        exit_price = opens[exit_bar]
        entry_lev = pos['leverage']
        fee = FEE_PCT * entry_lev
        if pos['direction'] == 'LONG':
            pnl = (exit_price / entry - 1) * 100 * entry_lev
        else:
            pnl = (1 - exit_price / entry) * 100 * entry_lev
        pnl -= fee
        sm = pos.get('size_mult', 1.0)
        trades.append({
            'entry_bar': pos['entry_bar'], 'exit_bar': exit_bar, 'pnl_slot': pnl,
            'reason': 'OOS_END', 'pattern': pos['pattern'],
            'direction': pos['direction'], 'size_mult': sm,
            'pnl_portfolio': pnl * (size_pct / 100) * sm,
            'leverage': entry_lev,
        })

    return trades, equity, blocked_counts


def calc_stats(trades):
    """Calculate standard stats from trades."""
    if not trades:
        return {'trades': 0, 'wr': 0, 'pnl': 0, 'mdd': 0, 'pnl_mdd': 0,
                'avg_leverage': 0, 'max_consec_loss': 0, 'worst_daily_pnl': 0}

    wins = sum(1 for t in trades if t['pnl_slot'] > 0)
    sorted_t = sorted(trades, key=lambda x: x['entry_bar'])
    eq = 100.0
    pk = eq
    mdd = 0.0
    daily_pnl = {}
    for t in sorted_t:
        eq += t['pnl_portfolio']
        if eq > pk:
            pk = eq
        d = (pk - eq) / pk * 100 if pk > 0 else 0
        if d > mdd:
            mdd = d
        day = t['entry_bar'] // BARS_PER_DAY
        daily_pnl[day] = daily_pnl.get(day, 0) + t['pnl_portfolio']

    total_pnl = eq - 100.0
    wr = wins / len(trades) * 100 if trades else 0
    avg_lev = np.mean([t.get('leverage', MAX_LEVERAGE) for t in trades])

    max_consec_loss = 0
    curr_streak = 0
    for t in sorted_t:
        if t['pnl_slot'] <= 0:
            curr_streak += 1
            if curr_streak > max_consec_loss:
                max_consec_loss = curr_streak
        else:
            curr_streak = 0

    worst_daily = min(daily_pnl.values()) if daily_pnl else 0.0

    return {
        'trades': len(trades),
        'wr': round(wr, 1),
        'pnl': round(total_pnl, 2),
        'mdd': round(mdd, 2),
        'pnl_mdd': round(total_pnl / mdd, 2) if mdd > 0 else 0,
        'avg_leverage': round(avg_lev, 2),
        'max_consec_loss': max_consec_loss,
        'worst_daily_pnl': round(worst_daily, 2),
    }


# ============================================================
# WF 3-fold expanding window
# ============================================================

def run_wf(signal_tuples, opens, highs, lows, closes, n_bars,
           atr_ratio, ema_slope, neutral_start, neutral_end, gc, n_folds=3):
    total = neutral_end - neutral_start
    seg_size = total // (n_folds + 1)
    results = []

    for fold in range(n_folds):
        oos_start = neutral_start + seg_size * (fold + 1)
        oos_end = (neutral_start + seg_size * (fold + 2)
                   if fold < n_folds - 1 else neutral_end)
        trades, _, _ = portfolio_sim(
            signal_tuples, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, oos_start, oos_end, gc)
        stats = calc_stats(trades)
        stats['fold'] = fold + 1
        results.append(stats)

    return results


# ============================================================
# Scenario Builders
# ============================================================

def build_ablation_scenarios():
    """Phase 1: ALL minus ONE — 11 scenarios."""
    scenarios = [('ALL_ON (baseline)', ALL_ON)]
    for label, field in GUARD_NAMES.items():
        gc = replace(ALL_ON, **{field: False})
        scenarios.append((f'NO_{label}', gc))
    return scenarios


def build_additive_scenarios():
    """Phase 2: BARE plus ONE — 11 scenarios."""
    scenarios = [('BARE (no guards)', ALL_OFF)]
    for label, field in GUARD_NAMES.items():
        gc = replace(ALL_OFF, **{field: True})
        scenarios.append((f'ONLY_{label}', gc))
    return scenarios


def build_overlap_scenarios():
    """Phase 3: 6 pairs × 3 = 18 scenarios."""
    pairs = [
        ('P1_Regime×AggRisk', 'regime_sizing', 'aggregate_risk_cap'),
        ('P2_Regime×CorrAware', 'regime_sizing', 'correlation_aware'),
        ('P3_LossBurst×EqCurve', 'loss_burst_brake', 'equity_curve_sizing'),
        ('P4_MDD×AdaptLev', 'mdd_sizing', 'adaptive_leverage'),
        ('P5_DirCap×AggRisk', 'direction_cap', 'aggregate_risk_cap'),
        ('P6_Momentum×LossBurst', 'momentum_guard', 'loss_burst_brake'),
    ]
    scenarios = []
    for pair_name, field_a, field_b in pairs:
        gc_a = replace(ALL_OFF, **{field_a: True})
        gc_b = replace(ALL_OFF, **{field_b: True})
        gc_ab = replace(ALL_OFF, **{field_a: True, field_b: True})
        scenarios.append((f'{pair_name}_A_only', gc_a, field_a))
        scenarios.append((f'{pair_name}_B_only', gc_b, field_b))
        scenarios.append((f'{pair_name}_A+B', gc_ab, None))
    return scenarios, pairs


# ============================================================
# Main Study
# ============================================================

def main():
    print("=" * 78)
    print("GUARD ABLATION STUDY — 10 Mechanisms Individual PnL Impact + Redundancy")
    print("=" * 78)
    start_time = datetime.now()

    # ---- Load Data ----
    print("\n[1] Loading data...")
    df = load_and_classify(DATA_FILE)
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    n_bars = len(df)

    atr_ratio = compute_atr_ratio(df)
    ema_slope = compute_ema_slope(closes)

    neutral_start, neutral_end = find_neutral_window(closes, tol_pct=1.0)
    neutral_bars = neutral_end - neutral_start
    print(f"  Neutral window: bar {neutral_start}~{neutral_end} "
          f"({neutral_bars} bars, {neutral_bars / BARS_PER_DAY:.0f} days)")

    # ---- Load Patterns ----
    print("\n[2] Loading patterns...")
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
    print("\n[3] Building signal index...")
    rctypes = df['rctype'].values
    signal_tuples = []
    for i in range(2, n_bars):
        tri = f"{rctypes[i-2]}-{rctypes[i-1]}-{rctypes[i]}"
        if tri in pat_lookup:
            p = pat_lookup[tri]
            signal_tuples.append((i, tri, p['direction'], p['tp'], p['sl']))

    n_in_neutral = sum(1 for s in signal_tuples if neutral_start <= s[0] < neutral_end)
    print(f"  {len(signal_tuples)} total signals, {n_in_neutral} in neutral window")

    all_results = {}

    # ================================================================
    # PHASE 1: Ablation (ALL minus ONE) — 11 scenarios
    # ================================================================
    print("\n" + "=" * 78)
    print("PHASE 1: Ablation (ALL minus ONE)")
    print("=" * 78)

    ablation = build_ablation_scenarios()
    phase1 = {}

    print(f"\n  {'Scenario':<35} {'Trades':>6} {'WR%':>6} {'PnL%':>9} "
          f"{'MDD%':>7} {'P/M':>7} {'AvgLev':>6} {'Blocked':>8}")
    print("  " + "-" * 90)

    baseline_stats = None
    for name, gc in ablation:
        trades, _, blocked = portfolio_sim(
            signal_tuples, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, neutral_start, neutral_end, gc)
        stats = calc_stats(trades)
        stats['blocked'] = blocked
        phase1[name] = stats
        if name == 'ALL_ON (baseline)':
            baseline_stats = stats

        total_blocked = sum(blocked.values())
        print(f"  {name:<35} {stats['trades']:>6} {stats['wr']:>6.1f} "
              f"{stats['pnl']:>9.1f} {stats['mdd']:>7.1f} "
              f"{stats['pnl_mdd']:>7.2f} {stats['avg_leverage']:>6.2f} {total_blocked:>8}")

    # Delta analysis
    print(f"\n  --- Impact Analysis (delta from baseline) ---")
    print(f"  {'Removed Guard':<35} {'dPnL%':>8} {'dMDD%':>8} {'dP/M':>8} {'dTrades':>8} {'Verdict':>10}")
    print("  " + "-" * 80)

    phase1_impact = {}
    for name, stats in phase1.items():
        if name == 'ALL_ON (baseline)':
            continue
        d_pnl = stats['pnl'] - baseline_stats['pnl']
        d_mdd = stats['mdd'] - baseline_stats['mdd']
        d_pm = stats['pnl_mdd'] - baseline_stats['pnl_mdd']
        d_trades = stats['trades'] - baseline_stats['trades']

        # Verdict: if removing makes things worse, guard is VALUABLE
        if d_mdd > 0.5 and d_pm < -0.5:
            verdict = "VALUABLE"
        elif abs(d_mdd) < 0.3 and abs(d_pm) < 0.3:
            verdict = "MARGINAL"
        elif d_pnl > 0 and d_mdd <= 0:
            verdict = "HARMFUL"  # removing it improves things
        else:
            verdict = "MIXED"

        phase1_impact[name] = {
            'd_pnl': round(d_pnl, 2), 'd_mdd': round(d_mdd, 2),
            'd_pnl_mdd': round(d_pm, 2), 'd_trades': d_trades,
            'verdict': verdict,
        }
        print(f"  {name:<35} {d_pnl:>+8.1f} {d_mdd:>+8.1f} "
              f"{d_pm:>+8.2f} {d_trades:>+8} {verdict:>10}")

    all_results['phase1_ablation'] = phase1
    all_results['phase1_impact'] = phase1_impact

    # ================================================================
    # PHASE 2: Additive (BARE plus ONE) — 11 scenarios
    # ================================================================
    print("\n" + "=" * 78)
    print("PHASE 2: Additive (BARE plus ONE)")
    print("=" * 78)

    additive = build_additive_scenarios()
    phase2 = {}

    print(f"\n  {'Scenario':<35} {'Trades':>6} {'WR%':>6} {'PnL%':>9} "
          f"{'MDD%':>7} {'P/M':>7} {'AvgLev':>6}")
    print("  " + "-" * 80)

    bare_stats = None
    for name, gc in additive:
        trades, _, blocked = portfolio_sim(
            signal_tuples, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, neutral_start, neutral_end, gc)
        stats = calc_stats(trades)
        phase2[name] = stats
        if name == 'BARE (no guards)':
            bare_stats = stats

        print(f"  {name:<35} {stats['trades']:>6} {stats['wr']:>6.1f} "
              f"{stats['pnl']:>9.1f} {stats['mdd']:>7.1f} "
              f"{stats['pnl_mdd']:>7.2f} {stats['avg_leverage']:>6.2f}")

    # Individual contribution from bare
    print(f"\n  --- Individual Contribution (delta from BARE) ---")
    print(f"  {'Added Guard':<35} {'dPnL%':>8} {'dMDD%':>8} {'dP/M':>8} {'dTrades':>8}")
    print("  " + "-" * 65)

    phase2_contribution = {}
    for name, stats in phase2.items():
        if name == 'BARE (no guards)':
            continue
        d_pnl = stats['pnl'] - bare_stats['pnl']
        d_mdd = stats['mdd'] - bare_stats['mdd']
        d_pm = stats['pnl_mdd'] - bare_stats['pnl_mdd']
        d_trades = stats['trades'] - bare_stats['trades']
        phase2_contribution[name] = {
            'd_pnl': round(d_pnl, 2), 'd_mdd': round(d_mdd, 2),
            'd_pnl_mdd': round(d_pm, 2), 'd_trades': d_trades,
        }
        print(f"  {name:<35} {d_pnl:>+8.1f} {d_mdd:>+8.1f} "
              f"{d_pm:>+8.2f} {d_trades:>+8}")

    all_results['phase2_additive'] = phase2
    all_results['phase2_contribution'] = phase2_contribution

    # ================================================================
    # PHASE 3: Overlap Detection — 6 pairs
    # ================================================================
    print("\n" + "=" * 78)
    print("PHASE 3: Overlap Detection (6 Suspected Pairs)")
    print("=" * 78)

    overlap_scenarios, pairs = build_overlap_scenarios()
    phase3_raw = {}

    print(f"\n  {'Scenario':<35} {'Trades':>6} {'WR%':>6} {'PnL%':>9} "
          f"{'MDD%':>7} {'P/M':>7}")
    print("  " + "-" * 75)

    for name, gc, _ in overlap_scenarios:
        trades, _, _ = portfolio_sim(
            signal_tuples, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, neutral_start, neutral_end, gc)
        stats = calc_stats(trades)
        phase3_raw[name] = stats
        print(f"  {name:<35} {stats['trades']:>6} {stats['wr']:>6.1f} "
              f"{stats['pnl']:>9.1f} {stats['mdd']:>7.1f} "
              f"{stats['pnl_mdd']:>7.2f}")

    # Calculate redundancy scores
    print(f"\n  --- Redundancy Scores ---")
    print(f"  {'Pair':<30} {'A_dP/M':>8} {'B_dP/M':>8} {'AB_dP/M':>8} "
          f"{'Expected':>8} {'Actual':>8} {'Redund':>8}")
    print("  " + "-" * 80)

    phase3_redundancy = {}
    for pair_name, field_a, field_b in pairs:
        a_name = f'{pair_name}_A_only'
        b_name = f'{pair_name}_B_only'
        ab_name = f'{pair_name}_A+B'

        a_pm = phase3_raw[a_name]['pnl_mdd']
        b_pm = phase3_raw[b_name]['pnl_mdd']
        ab_pm = phase3_raw[ab_name]['pnl_mdd']
        bare_pm = bare_stats['pnl_mdd']

        # Contribution = delta from BARE
        a_delta = a_pm - bare_pm
        b_delta = b_pm - bare_pm
        ab_delta = ab_pm - bare_pm

        expected_independent = a_delta + b_delta
        actual_combined = ab_delta

        if abs(expected_independent) > 0.01:
            redundancy = 1.0 - actual_combined / expected_independent
        else:
            redundancy = 0.0

        # Also check MDD overlap
        a_mdd_d = phase3_raw[a_name]['mdd'] - bare_stats['mdd']
        b_mdd_d = phase3_raw[b_name]['mdd'] - bare_stats['mdd']
        ab_mdd_d = phase3_raw[ab_name]['mdd'] - bare_stats['mdd']
        expected_mdd = a_mdd_d + b_mdd_d
        if abs(expected_mdd) > 0.01:
            mdd_redundancy = 1.0 - ab_mdd_d / expected_mdd
        else:
            mdd_redundancy = 0.0

        phase3_redundancy[pair_name] = {
            'A_field': field_a, 'B_field': field_b,
            'A_pnl_mdd_delta': round(a_delta, 2),
            'B_pnl_mdd_delta': round(b_delta, 2),
            'AB_pnl_mdd_delta': round(ab_delta, 2),
            'expected_independent': round(expected_independent, 2),
            'actual_combined': round(actual_combined, 2),
            'pnl_mdd_redundancy': round(redundancy, 3),
            'mdd_redundancy': round(mdd_redundancy, 3),
        }

        print(f"  {pair_name:<30} {a_delta:>+8.2f} {b_delta:>+8.2f} "
              f"{ab_delta:>+8.2f} {expected_independent:>+8.2f} "
              f"{actual_combined:>+8.2f} {redundancy:>8.3f}")

    all_results['phase3_overlap'] = phase3_raw
    all_results['phase3_redundancy'] = phase3_redundancy

    # ================================================================
    # PHASE 4: WF Validation
    # ================================================================
    print("\n" + "=" * 78)
    print("PHASE 4: WF 3-Fold Validation (Baseline + Key Configs)")
    print("=" * 78)

    phase4 = {}

    # Always validate baseline
    wf_configs = [('ALL_ON (baseline)', ALL_ON), ('BARE (no guards)', ALL_OFF)]

    # Add configs where removing a guard improved things (HARMFUL verdict)
    for name, impact in phase1_impact.items():
        if impact['verdict'] == 'HARMFUL':
            field = name.replace('NO_', '').split('_', 1)
            guard_key = name.replace('NO_', '')
            if guard_key in GUARD_NAMES:
                gc = replace(ALL_ON, **{GUARD_NAMES[guard_key]: False})
                wf_configs.append((name, gc))

    # Add configs where removing a guard was MARGINAL
    for name, impact in phase1_impact.items():
        if impact['verdict'] == 'MARGINAL':
            guard_key = name.replace('NO_', '')
            if guard_key in GUARD_NAMES:
                gc = replace(ALL_ON, **{GUARD_NAMES[guard_key]: False})
                wf_configs.append((name, gc))

    # Add high-redundancy pairs (redundancy > 0.3) — test removing one
    for pair_name, info in phase3_redundancy.items():
        if info['pnl_mdd_redundancy'] > 0.3:
            # Remove the one with less individual contribution
            a_d = abs(info['A_pnl_mdd_delta'])
            b_d = abs(info['B_pnl_mdd_delta'])
            remove_field = info['A_field'] if a_d < b_d else info['B_field']
            gc = replace(ALL_ON, **{remove_field: False})
            wf_name = f'REMOVE_weaker_in_{pair_name}'
            wf_configs.append((wf_name, gc))

    print(f"  Validating {len(wf_configs)} configurations...")
    print(f"\n  {'Config':<40} {'F1':>8} {'F2':>8} {'F3':>8} {'Result':>8}")
    print("  " + "-" * 70)

    for name, gc in wf_configs:
        folds = run_wf(signal_tuples, opens, highs, lows, closes, n_bars,
                       atr_ratio, ema_slope, neutral_start, neutral_end, gc)
        all_pass = all(f['pnl'] > 0 for f in folds)
        phase4[name] = {
            'folds': folds,
            'pass': all_pass,
        }
        fold_str = '  '.join([f"{f['pnl']:>+7.1f}%" for f in folds])
        status = "PASS" if all_pass else "FAIL"
        print(f"  {name:<40} {fold_str} {status:>8}")

    all_results['phase4_wf'] = phase4

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)

    # Rank by ablation impact (removing hurts most = most valuable)
    print("\n  --- Guard Value Ranking (most valuable first) ---")
    print(f"  {'Guard':<35} {'dP/M when removed':>18} {'Verdict':>10}")
    print("  " + "-" * 65)

    ranked = sorted(phase1_impact.items(),
                    key=lambda x: x[1]['d_pnl_mdd'])  # most negative first = most valuable
    for name, impact in ranked:
        print(f"  {name:<35} {impact['d_pnl_mdd']:>+18.2f} {impact['verdict']:>10}")

    # Redundancy summary
    print("\n  --- Redundancy Summary ---")
    print(f"  {'Pair':<30} {'Redundancy':>10} {'Interpretation':<30}")
    print("  " + "-" * 72)

    for pair_name, info in sorted(phase3_redundancy.items(),
                                   key=lambda x: x[1]['pnl_mdd_redundancy'],
                                   reverse=True):
        r = info['pnl_mdd_redundancy']
        if r > 0.5:
            interp = "HIGH OVERLAP — review needed"
        elif r > 0.3:
            interp = "MODERATE OVERLAP"
        elif r > 0:
            interp = "LOW OVERLAP — good synergy"
        else:
            interp = "NEGATIVE — super-additive!"
        print(f"  {pair_name:<30} {r:>10.3f} {interp:<30}")

    # Save results
    all_results['metadata'] = {
        'date': datetime.now().isoformat(),
        'data_file': DATA_FILE,
        'patterns': len(pat_lookup),
        'neutral_bars': int(neutral_bars),
        'runtime_sec': (datetime.now() - start_time).total_seconds(),
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n  Results saved to {OUTPUT_FILE}")
    print(f"  Runtime: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
