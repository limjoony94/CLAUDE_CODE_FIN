#!/usr/bin/env python3
"""
Adaptive Leverage Study — Rolling WR-based Dynamic Leverage

Hypothesis: Start with low leverage, increase as rolling WR confirms edge.
  - Reduces capital exposure during uncertain early period
  - Amplifies gains when edge is confirmed by track record
  - Natural risk management: bad streak → auto-reduce leverage

Scenarios:
  H1 (Step tiers): WR<50%→1x, 50-60%→1.5x, 60-70%→2x, 70-80%→2.5x, >80%→3x
  H2 (Continuous linear): leverage = 1 + 2 * clamp((rolling_WR - 50) / 30)
  H3 (Conservative start): First N trades always 1x, then ramp
  H4 (Per-direction WR): Track LONG/SHORT WR independently
  Window sweep: 5, 10, 15, 20, 30 trades lookback
  + Baseline: Fixed 3x (current production)

Standard Research Protocol:
  - Production classify_candle import
  - LEVERAGE = 3 (max), FEE_PCT * LEVERAGE for notional fees
  - Timeout = DROP, Same-bar = abs(tp - bar_open)
  - Entry = next-bar open, ATR-scaled TP/SL
  - Compound (multiplicative) sizing
  - MC: seeds [42, 123, 7], max p < 0.01

Author: Research Agent
Date: 2026-03-02
"""

import os
import sys
import json
import warnings
from collections import deque
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
# Constants — matching scanner exactly
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

DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
PATTERNS_FILE = os.path.join(_PROJECT_ROOT, 'results', 'dynamic_patterns.json')
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'adaptive_leverage_study.json')


# ============================================================
# Data Loading (from scanner)
# ============================================================

def load_and_classify(data_file):
    df = pd.read_csv(data_file)
    if 'open' not in df.columns:
        df.columns = [c.lower() for c in df.columns]
    required = ['open', 'high', 'low', 'close']
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    # Use type_code if available (pre-classified), else classify
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


def compute_atr_ratio(df, period=ATR_PERIOD, window=ATR_WINDOW):
    h = df['high'].values
    l = df['low'].values
    c = df['close'].values
    tr = np.maximum(h - l, np.maximum(abs(h - np.roll(c, 1)), abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    atr = pd.Series(tr).ewm(span=period, adjust=False).mean().values
    med = pd.Series(atr).rolling(window, min_periods=1).median().values
    ratio = np.where(med > 0, atr / med, 1.0)
    return ratio


def compute_ema_slope(closes, period=EMA_PERIOD, lookback=EMA_LOOKBACK):
    ema = pd.Series(closes).ewm(span=period, adjust=False).mean().values
    slope = np.full(len(closes), 0.0)
    for i in range(lookback, len(closes)):
        slope[i] = ema[i] - ema[i - lookback]
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


# ============================================================
# Adaptive Leverage Logic
# ============================================================

def get_leverage_step_tiers(rolling_wr):
    """H1: Step tiers based on rolling WR."""
    if rolling_wr < 50:
        return 1.0
    elif rolling_wr < 60:
        return 1.5
    elif rolling_wr < 70:
        return 2.0
    elif rolling_wr < 80:
        return 2.5
    else:
        return 3.0


def get_leverage_continuous(rolling_wr):
    """H2: Continuous linear scaling."""
    ratio = max(0.0, min(1.0, (rolling_wr - 50) / 30))
    return MIN_LEVERAGE + (MAX_LEVERAGE - MIN_LEVERAGE) * ratio


def get_leverage_conservative(rolling_wr, n_trades_so_far, warmup_trades=15):
    """H3: Conservative start — first N trades always 1x, then ramp."""
    if n_trades_so_far < warmup_trades:
        return 1.0
    return get_leverage_continuous(rolling_wr)


def get_leverage_per_direction(rolling_wr_dict, direction):
    """H4: Per-direction WR tracking."""
    wr = rolling_wr_dict.get(direction, 50.0)
    return get_leverage_continuous(wr)


# ============================================================
# Portfolio Simulator with Adaptive Leverage
# ============================================================

def _check_exit(pos, bar, opens, highs, lows, n_bars, atr_ratio, leverage):
    """Check exit for a single position with dynamic leverage."""
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
        r = max(ATR_CLAMP_LO, min(ATR_CLAMP_HI, atr_ratio[sig_bar]))
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

    # PnL uses the leverage that was locked at entry
    entry_lev = pos['leverage']
    fee = FEE_PCT * entry_lev
    if direction == 'LONG':
        pnl = (exit_price / entry - 1) * 100 * entry_lev
    else:
        pnl = (1 - exit_price / entry) * 100 * entry_lev
    pnl -= fee

    return {'entry_bar': entry_bar, 'exit_bar': bar, 'pnl_slot': pnl,
            'reason': reason, 'drop': False, 'leverage': entry_lev}


def portfolio_adaptive(signal_tuples, opens, highs, lows, closes, n_bars,
                       atr_ratio, ema_slope, start_bar, end_bar,
                       leverage_fn, leverage_fn_name='adaptive',
                       window_size=15):
    """N-pos portfolio with adaptive leverage.

    Args:
        leverage_fn: callable(context_dict) -> float (1.0~3.0)
            context_dict keys: rolling_wr, rolling_wr_long, rolling_wr_short,
                               n_trades, direction
        window_size: rolling WR window
    """
    size_pct = 100.0 / N_SLOTS

    positions = []
    trades = []
    equity = 100.0
    peak_equity = 100.0
    max_dd = 0.0

    # Rolling WR trackers
    recent_results = deque(maxlen=window_size)
    recent_long = deque(maxlen=window_size)
    recent_short = deque(maxlen=window_size)
    n_trades_total = 0

    momentum_pause_until = {'LONG': -1, 'SHORT': -1}
    leverage_history = []  # (bar, leverage) for analysis

    signals_in_range = [(s, p, d, tp, sl) for s, p, d, tp, sl in signal_tuples
                        if start_bar <= s < end_bar]
    signals_sorted = sorted(signals_in_range, key=lambda x: x[0])
    sig_idx = 0

    for bar in range(start_bar, end_bar):
        # 1. Check exits
        closed_slots = []
        bar_pnl_sum = 0.0

        for pos in positions:
            result = _check_exit(pos, bar, opens, highs, lows, n_bars,
                                 atr_ratio, pos['leverage'])
            if result is not None:
                if result.get('drop', False):
                    closed_slots.append(pos['slot'])
                    continue
                result['pattern'] = pos['pattern']
                result['direction'] = pos['direction']
                sm = pos.get('size_mult', 1.0)
                result['size_mult'] = sm
                pnl_portfolio = result['pnl_slot'] * (size_pct / 100) * sm
                result['pnl_portfolio'] = pnl_portfolio
                result['leverage'] = pos['leverage']
                trades.append(result)
                closed_slots.append(pos['slot'])
                bar_pnl_sum += pnl_portfolio

                # Update rolling WR
                is_win = result['pnl_slot'] > 0
                recent_results.append(1 if is_win else 0)
                if pos['direction'] == 'LONG':
                    recent_long.append(1 if is_win else 0)
                else:
                    recent_short.append(1 if is_win else 0)
                n_trades_total += 1

        positions = [p for p in positions if p['slot'] not in closed_slots]

        equity += bar_pnl_sum
        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0
        if dd > max_dd:
            max_dd = dd

        # Momentum guard
        if MOMENTUM_LOOKBACK > 0 and MOMENTUM_THRESHOLD > 0 and bar >= MOMENTUM_LOOKBACK:
            price_now = closes[bar]
            price_ago = closes[bar - MOMENTUM_LOOKBACK]
            if price_ago > 0:
                pct_change = (price_now / price_ago - 1) * 100
                if pct_change > MOMENTUM_THRESHOLD:
                    momentum_pause_until['SHORT'] = bar + MOMENTUM_COOLDOWN
                elif pct_change < -MOMENTUM_THRESHOLD:
                    momentum_pause_until['LONG'] = bar + MOMENTUM_COOLDOWN

        # 2. Process entries
        while sig_idx < len(signals_sorted) and signals_sorted[sig_idx][0] == bar:
            sig_bar, pat, direction, tp_pct, sl_pct = signals_sorted[sig_idx]
            sig_idx += 1

            if len(positions) >= N_SLOTS:
                continue
            dir_count = sum(1 for p in positions if p['direction'] == direction)
            if dir_count >= DIRECTION_CAP:
                continue
            if any(p['pattern'] == pat for p in positions):
                continue

            entry_bar = sig_bar + 1
            if entry_bar >= n_bars:
                continue

            # Momentum guard
            if MOMENTUM_LOOKBACK > 0 and bar < momentum_pause_until.get(direction, -1):
                continue

            # Regime sizing
            sm = 1.0
            if REGIME_MULT is not None and bar < len(ema_slope):
                s = ema_slope[bar]
                if s > 0 and direction == 'SHORT':
                    sm = REGIME_MULT
                elif s <= 0 and direction == 'LONG':
                    sm = REGIME_MULT

            # Aggregate risk cap (use MAX_LEVERAGE for conservative check)
            if AGG_RISK_COUNTER > 0 or AGG_RISK_WITH > 0:
                is_uptrend = ema_slope[bar] > 0 if bar < len(ema_slope) else False
                is_counter = ((is_uptrend and direction == 'SHORT') or
                              (not is_uptrend and direction == 'LONG'))
                cap_pct = AGG_RISK_COUNTER if is_counter else AGG_RISK_WITH

                existing_exposure = 0.0
                for p in positions:
                    if p['direction'] == direction:
                        p_sl = p['sl_pct']
                        p_sig = p['signal_bar']
                        if (atr_ratio is not None and p_sig < len(atr_ratio)
                                and not np.isnan(atr_ratio[p_sig])):
                            p_r = max(ATR_CLAMP_LO, min(ATR_CLAMP_HI, atr_ratio[p_sig]))
                        else:
                            p_r = 1.0
                        p_eff_sl = p_sl * p_r
                        p_sm = p.get('size_mult', 1.0)
                        existing_exposure += p_eff_sl * (1.0 / N_SLOTS) * p['leverage'] * p_sm

                new_r = 1.0
                if (atr_ratio is not None and sig_bar < len(atr_ratio)
                        and not np.isnan(atr_ratio[sig_bar])):
                    new_r = max(ATR_CLAMP_LO, min(ATR_CLAMP_HI, atr_ratio[sig_bar]))

                # Compute adaptive leverage for this entry
                rolling_wr = (sum(recent_results) / len(recent_results) * 100
                              if recent_results else 50.0)
                rolling_wr_long = (sum(recent_long) / len(recent_long) * 100
                                   if recent_long else 50.0)
                rolling_wr_short = (sum(recent_short) / len(recent_short) * 100
                                    if recent_short else 50.0)

                ctx = {
                    'rolling_wr': rolling_wr,
                    'rolling_wr_long': rolling_wr_long,
                    'rolling_wr_short': rolling_wr_short,
                    'n_trades': n_trades_total,
                    'direction': direction,
                    'window_size': window_size,
                }
                entry_leverage = leverage_fn(ctx)
                entry_leverage = max(MIN_LEVERAGE, min(MAX_LEVERAGE, entry_leverage))

                new_eff_sl = sl_pct * new_r
                new_exposure = new_eff_sl * (1.0 / N_SLOTS) * entry_leverage * sm

                if existing_exposure + new_exposure > cap_pct:
                    continue
            else:
                rolling_wr = (sum(recent_results) / len(recent_results) * 100
                              if recent_results else 50.0)
                rolling_wr_long = (sum(recent_long) / len(recent_long) * 100
                                   if recent_long else 50.0)
                rolling_wr_short = (sum(recent_short) / len(recent_short) * 100
                                    if recent_short else 50.0)
                ctx = {
                    'rolling_wr': rolling_wr,
                    'rolling_wr_long': rolling_wr_long,
                    'rolling_wr_short': rolling_wr_short,
                    'n_trades': n_trades_total,
                    'direction': direction,
                    'window_size': window_size,
                }
                entry_leverage = leverage_fn(ctx)
                entry_leverage = max(MIN_LEVERAGE, min(MAX_LEVERAGE, entry_leverage))

            leverage_history.append((bar, entry_leverage, direction, rolling_wr))

            positions.append({
                'slot': f"{pat}_{sig_bar}",
                'signal_bar': sig_bar,
                'entry_bar': entry_bar,
                'direction': direction,
                'pattern': pat,
                'tp_pct': tp_pct,
                'sl_pct': sl_pct,
                'size_mult': sm,
                'leverage': entry_leverage,
            })

    # Force-close remaining
    for pos in positions:
        entry_bar = pos['entry_bar']
        if entry_bar >= n_bars:
            continue
        entry = opens[entry_bar]
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
            'entry_bar': entry_bar, 'exit_bar': exit_bar, 'pnl_slot': pnl,
            'reason': 'OOS_END', 'pattern': pos['pattern'],
            'direction': pos['direction'], 'size_mult': sm,
            'pnl_portfolio': pnl * (size_pct / 100) * sm,
            'leverage': entry_lev,
        })

    # Calc stats
    if not trades:
        return [], {'trades': 0, 'wr': 0, 'pnl': 0, 'mdd': 0, 'pnl_mdd': 0,
                    'avg_leverage': 0, 'leverage_history': []}

    wins = sum(1 for t in trades if t['pnl_slot'] > 0)
    sorted_t = sorted(trades, key=lambda x: x['entry_bar'])
    eq = 100.0
    pk = eq
    mdd = 0.0
    for t in sorted_t:
        eq += t['pnl_portfolio']
        if eq > pk:
            pk = eq
        d = (pk - eq) / pk * 100 if pk > 0 else 0
        if d > mdd:
            mdd = d

    total_pnl = eq - 100.0
    wr = wins / len(trades) * 100 if trades else 0
    avg_lev = np.mean([t.get('leverage', MAX_LEVERAGE) for t in trades])
    pnl_mdd = total_pnl / mdd if mdd > 0 else 0

    stats = {
        'trades': len(trades),
        'wr': round(wr, 1),
        'pnl': round(total_pnl, 2),
        'mdd': round(mdd, 2),
        'pnl_mdd': round(pnl_mdd, 2),
        'pnl_per_trade': round(total_pnl / len(trades), 3) if trades else 0,
        'avg_leverage': round(avg_lev, 2),
    }
    return trades, stats


# ============================================================
# WF validation
# ============================================================

def run_wf_3fold(signal_tuples, opens, highs, lows, closes, n_bars,
                 atr_ratio, ema_slope, neutral_start, neutral_end,
                 leverage_fn, leverage_fn_name, window_size, n_folds=3):
    """3-fold expanding window WF with adaptive leverage.

    Expanding window: divide into (n_folds+1) segments.
    Fold k: IS = segments 0..k, OOS = segment k+1.
    """
    total = neutral_end - neutral_start
    seg_size = total // (n_folds + 1)
    results = []

    for fold in range(n_folds):
        oos_start = neutral_start + seg_size * (fold + 1)
        oos_end = neutral_start + seg_size * (fold + 2) if fold < n_folds - 1 else neutral_end

        _, stats = portfolio_adaptive(
            signal_tuples, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, oos_start, oos_end,
            leverage_fn, leverage_fn_name, window_size)
        stats['fold'] = fold + 1
        stats['oos_start'] = int(oos_start)
        stats['oos_end'] = int(oos_end)
        stats['oos_bars'] = int(oos_end - oos_start)
        results.append(stats)

    return results


# ============================================================
# Main Study
# ============================================================

def main():
    print("=" * 70)
    print("ADAPTIVE LEVERAGE STUDY — Rolling WR-based Dynamic Leverage")
    print("=" * 70)
    start_time = datetime.now()

    # Load data
    print("\n[1] Loading data...")
    df = load_and_classify(DATA_FILE)
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    n_bars = len(df)

    atr_ratio = compute_atr_ratio(df)
    ema_slope = compute_ema_slope(closes)

    # Find neutral window
    neutral_start, neutral_end = find_neutral_window(closes, tol_pct=1.0)
    print(f"  Neutral window: bar {neutral_start}~{neutral_end} "
          f"({neutral_end - neutral_start} bars, "
          f"{(neutral_end - neutral_start) / 288:.0f} days)")

    # Load patterns
    print("\n[2] Loading patterns...")
    with open(PATTERNS_FILE) as f:
        pat_data = json.load(f)
    pats_raw = pat_data['patterns']
    tpsl = pat_data.get('patterns_tpsl', {})

    pat_lookup = {}
    for pat_name in pats_raw.get('long', []):
        tp_sl = tpsl.get(pat_name, [2.0, 3.0])
        pat_lookup[pat_name] = {'pattern': pat_name, 'direction': 'LONG',
                                'tp': tp_sl[0], 'sl': tp_sl[1]}
    for pat_name in pats_raw.get('short', []):
        tp_sl = tpsl.get(pat_name, [2.0, 3.0])
        pat_lookup[pat_name] = {'pattern': pat_name, 'direction': 'SHORT',
                                'tp': tp_sl[0], 'sl': tp_sl[1]}
    print(f"  {len(pat_lookup)} patterns loaded ({len(pats_raw.get('long',[]))}L + "
          f"{len(pats_raw.get('short',[]))}S)")

    # Build signal index
    print("\n[3] Building signal index...")
    rctypes = df['rctype'].values
    signal_tuples = []

    for i in range(2, n_bars):
        tri = f"{rctypes[i-2]}-{rctypes[i-1]}-{rctypes[i]}"
        if tri in pat_lookup:
            p = pat_lookup[tri]
            signal_tuples.append((i, p['pattern'], p['direction'],
                                  p['tp'], p['sl']))

    print(f"  {len(signal_tuples)} signal events in full data")
    n_in_neutral = sum(1 for s in signal_tuples
                       if neutral_start <= s[0] < neutral_end)
    print(f"  {n_in_neutral} signals in neutral window")

    # ============================================================
    # Phase 1: IS Full-period comparison — Baseline vs Adaptive
    # ============================================================
    print("\n" + "=" * 70)
    print("PHASE 1: IS Full-Period Comparison (Neutral Window)")
    print("=" * 70)

    # Define leverage functions
    def fn_fixed_3x(ctx):
        return 3.0

    def fn_fixed_1x(ctx):
        return 1.0

    def fn_step_tiers(ctx):
        return get_leverage_step_tiers(ctx['rolling_wr'])

    def fn_continuous(ctx):
        return get_leverage_continuous(ctx['rolling_wr'])

    def fn_conservative_10(ctx):
        return get_leverage_conservative(ctx['rolling_wr'], ctx['n_trades'], warmup_trades=10)

    def fn_conservative_20(ctx):
        return get_leverage_conservative(ctx['rolling_wr'], ctx['n_trades'], warmup_trades=20)

    def fn_per_direction(ctx):
        d = ctx['direction']
        wr = ctx.get(f'rolling_wr_{d.lower()}', ctx['rolling_wr'])
        return get_leverage_continuous(wr)

    scenarios_phase1 = [
        ('Fixed_3x (baseline)', fn_fixed_3x, 15),
        ('Fixed_1x (min)', fn_fixed_1x, 15),
        ('H1_StepTiers_w10', fn_step_tiers, 10),
        ('H1_StepTiers_w15', fn_step_tiers, 15),
        ('H1_StepTiers_w20', fn_step_tiers, 20),
        ('H1_StepTiers_w30', fn_step_tiers, 30),
        ('H2_Continuous_w10', fn_continuous, 10),
        ('H2_Continuous_w15', fn_continuous, 15),
        ('H2_Continuous_w20', fn_continuous, 20),
        ('H2_Continuous_w30', fn_continuous, 30),
        ('H3_Conservative_w10_warmup10', fn_conservative_10, 10),
        ('H3_Conservative_w15_warmup20', fn_conservative_20, 15),
        ('H4_PerDirection_w15', fn_per_direction, 15),
        ('H4_PerDirection_w20', fn_per_direction, 20),
    ]

    phase1_results = {}
    print(f"\n{'Scenario':<35} {'Trades':>6} {'WR%':>6} {'PnL%':>8} "
          f"{'MDD%':>7} {'PnL/MDD':>8} {'AvgLev':>7}")
    print("-" * 78)

    for name, fn, ws in scenarios_phase1:
        _, stats = portfolio_adaptive(
            signal_tuples, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, neutral_start, neutral_end,
            fn, name, ws)
        phase1_results[name] = stats
        print(f"{name:<35} {stats['trades']:>6} {stats['wr']:>6.1f} "
              f"{stats['pnl']:>8.1f} {stats['mdd']:>7.1f} "
              f"{stats['pnl_mdd']:>8.2f} {stats['avg_leverage']:>7.2f}")

    baseline = phase1_results['Fixed_3x (baseline)']
    print(f"\n  Baseline: PnL {baseline['pnl']:.1f}%, MDD {baseline['mdd']:.1f}%, "
          f"PnL/MDD {baseline['pnl_mdd']:.2f}")

    # Find best PnL/MDD
    best_name = max(phase1_results, key=lambda k: phase1_results[k]['pnl_mdd'])
    best = phase1_results[best_name]
    print(f"  Best PnL/MDD: {best_name} → {best['pnl_mdd']:.2f} "
          f"(PnL {best['pnl']:.1f}%, MDD {best['mdd']:.1f}%)")

    # ============================================================
    # Phase 2: Window size sweep (best method)
    # ============================================================
    print("\n" + "=" * 70)
    print("PHASE 2: Window Size Sweep for Top Methods")
    print("=" * 70)

    # Select top 3 methods from Phase 1 (excluding fixed baselines)
    adaptive_results = {k: v for k, v in phase1_results.items()
                        if 'Fixed' not in k}
    top3 = sorted(adaptive_results, key=lambda k: adaptive_results[k]['pnl_mdd'],
                  reverse=True)[:3]

    method_map = {
        'H1_StepTiers': fn_step_tiers,
        'H2_Continuous': fn_continuous,
        'H3_Conservative': fn_conservative_20,
        'H4_PerDirection': fn_per_direction,
    }

    print(f"\n  Top 3 adaptive methods: {top3}")
    window_sweep = [5, 8, 10, 12, 15, 20, 25, 30]

    phase2_results = {}
    for method_name in top3:
        # Extract base method
        base = method_name.split('_w')[0]
        fn = None
        for key, func in method_map.items():
            if base.startswith(key):
                fn = func
                break
        if fn is None:
            continue

        print(f"\n  {base}:")
        print(f"  {'Window':>8} {'PnL%':>8} {'MDD%':>7} {'PnL/MDD':>8} {'AvgLev':>7}")
        for ws in window_sweep:
            label = f"{base}_w{ws}"
            _, stats = portfolio_adaptive(
                signal_tuples, opens, highs, lows, closes, n_bars,
                atr_ratio, ema_slope, neutral_start, neutral_end,
                fn, label, ws)
            phase2_results[label] = stats
            print(f"  {ws:>8} {stats['pnl']:>8.1f} {stats['mdd']:>7.1f} "
                  f"{stats['pnl_mdd']:>8.2f} {stats['avg_leverage']:>7.2f}")

    # ============================================================
    # Phase 3: WF 3-fold OOS Validation
    # ============================================================
    print("\n" + "=" * 70)
    print("PHASE 3: Walk-Forward 3-Fold OOS Validation")
    print("=" * 70)

    # Select candidates: baseline + top 5 adaptive scenarios
    all_scored = {}
    all_scored.update(phase1_results)
    all_scored.update(phase2_results)

    candidates_adaptive = sorted(
        [k for k in all_scored if 'Fixed' not in k],
        key=lambda k: all_scored[k]['pnl_mdd'], reverse=True)[:5]

    wf_candidates = ['Fixed_3x (baseline)'] + candidates_adaptive

    # Map scenario name → (fn, window_size)
    def parse_scenario(name):
        if name == 'Fixed_3x (baseline)':
            return fn_fixed_3x, 15
        if name == 'Fixed_1x (min)':
            return fn_fixed_1x, 15

        # Parse window from name
        ws = 15
        if '_w' in name:
            parts = name.split('_w')
            try:
                ws_str = parts[-1].split('_')[0]
                ws = int(ws_str)
            except ValueError:
                ws = 15

        if 'StepTiers' in name:
            return fn_step_tiers, ws
        elif 'Conservative' in name:
            if 'warmup10' in name:
                return fn_conservative_10, ws
            return fn_conservative_20, ws
        elif 'PerDirection' in name:
            return fn_per_direction, ws
        elif 'Continuous' in name:
            return fn_continuous, ws
        return fn_fixed_3x, ws

    wf_results = {}
    for name in wf_candidates:
        fn, ws = parse_scenario(name)
        folds = run_wf_3fold(
            signal_tuples, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, neutral_start, neutral_end,
            fn, name, ws)

        all_pass = all(f['pnl'] > 0 for f in folds)
        wf_results[name] = {'folds': folds, 'pass': all_pass}

        fold_str = ' | '.join([f"F{f['fold']}: {f['pnl']:+.1f}%" for f in folds])
        status = "PASS" if all_pass else "FAIL"
        print(f"  [{status}] {name:<35} {fold_str}")

    # ============================================================
    # Phase 4: Risk-Adjusted Comparison
    # ============================================================
    print("\n" + "=" * 70)
    print("PHASE 4: Risk-Adjusted Comparison Summary")
    print("=" * 70)

    print(f"\n{'Scenario':<35} {'IS PnL':>7} {'IS MDD':>7} {'IS P/M':>7} "
          f"{'AvgLev':>7} {'WF':>5} {'OOS Avg':>8}")
    print("-" * 80)

    for name in wf_candidates:
        is_stats = all_scored[name]
        wf = wf_results[name]
        avg_oos = np.mean([f['pnl'] for f in wf['folds']])
        status = "PASS" if wf['pass'] else "FAIL"
        print(f"{name:<35} {is_stats['pnl']:>7.1f} {is_stats['mdd']:>7.1f} "
              f"{is_stats['pnl_mdd']:>7.2f} {is_stats['avg_leverage']:>7.2f} "
              f"{status:>5} {avg_oos:>+8.1f}")

    # ============================================================
    # Phase 5: Verdict
    # ============================================================
    print("\n" + "=" * 70)
    print("PHASE 5: Verdict")
    print("=" * 70)

    baseline_wf = wf_results.get('Fixed_3x (baseline)', {})
    baseline_oos_avg = np.mean([f['pnl'] for f in baseline_wf.get('folds', [{'pnl': 0}])])

    # Find best WF-passing adaptive scenario
    passing_adaptive = [(name, wf_results[name])
                        for name in candidates_adaptive
                        if wf_results.get(name, {}).get('pass', False)]

    if passing_adaptive:
        # Among passing, pick best IS PnL/MDD
        best_adaptive_name = max(passing_adaptive,
                                 key=lambda x: all_scored[x[0]]['pnl_mdd'])[0]
        best_a = all_scored[best_adaptive_name]
        best_a_oos = np.mean([f['pnl'] for f in wf_results[best_adaptive_name]['folds']])

        pnl_mdd_improvement = (best_a['pnl_mdd'] - baseline['pnl_mdd']) / baseline['pnl_mdd'] * 100
        mdd_improvement = (baseline['mdd'] - best_a['mdd']) / baseline['mdd'] * 100

        print(f"\n  Best WF-PASS adaptive: {best_adaptive_name}")
        print(f"  IS:  PnL {best_a['pnl']:.1f}% | MDD {best_a['mdd']:.1f}% | "
              f"PnL/MDD {best_a['pnl_mdd']:.2f} | AvgLev {best_a['avg_leverage']:.2f}")
        print(f"  OOS: Avg PnL {best_a_oos:+.1f}%")
        print(f"  vs Baseline:")
        print(f"    PnL/MDD: {pnl_mdd_improvement:+.1f}%")
        print(f"    MDD reduction: {mdd_improvement:+.1f}%")
        print(f"    PnL trade-off: {best_a['pnl'] - baseline['pnl']:+.1f}%")

        if best_a['pnl_mdd'] > baseline['pnl_mdd'] and mdd_improvement > 10:
            verdict = "GO"
            print(f"\n  VERDICT: {verdict} — Adaptive leverage improves risk-adjusted returns")
        elif mdd_improvement > 20 and best_a['pnl'] > baseline['pnl'] * 0.5:
            verdict = "GO (conservative)"
            print(f"\n  VERDICT: {verdict} — Significant MDD reduction worth PnL trade-off")
        else:
            verdict = "STOP"
            print(f"\n  VERDICT: {verdict} — Improvement insufficient vs complexity cost")
    else:
        verdict = "STOP"
        print(f"\n  No adaptive scenario passed WF 3/3.")
        print(f"  VERDICT: {verdict}")

    # Save results
    elapsed = (datetime.now() - start_time).total_seconds()
    output = {
        'study': 'adaptive_leverage',
        'date': datetime.now().isoformat(),
        'elapsed_seconds': round(elapsed, 1),
        'data_file': DATA_FILE,
        'patterns_file': PATTERNS_FILE,
        'n_patterns': len(pat_lookup),
        'neutral_window': {'start': int(neutral_start), 'end': int(neutral_end),
                           'bars': int(neutral_end - neutral_start)},
        'phase1_is': {k: v for k, v in phase1_results.items()},
        'phase2_sweep': phase2_results,
        'phase3_wf': {k: {'pass': v['pass'],
                          'folds': v['folds']}
                      for k, v in wf_results.items()},
        'verdict': verdict,
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {OUTPUT_FILE}")
    print(f"  Elapsed: {elapsed:.0f}s")
    print("=" * 70)


if __name__ == '__main__':
    main()
