#!/usr/bin/env python3
"""
N Reduction Study: Multi-Position Count Optimization
=====================================================
Currently N=9 virtual slots. Individual pattern edge is genuine (130/130 positive
EV, MC p=0.000) but correlated losses on same-asset (BTC) positions destroy
portfolio WR (IS 86.8% -> Live 52.2%). Break-even WR ~67.3%, Live 52.2% = -15.1pp gap.

Core hypothesis: Reducing N decreases correlation-induced portfolio losses,
making actual portfolio WR converge toward individual pattern WR.

5 Phases:
  Phase 1: N-sweep portfolio backtest (N=1,2,3,5,9) with full filter stack
  Phase 2: Correlation analysis — concurrent position overlap, simultaneous SL events
  Phase 3: 3-fold expanding-window WF OOS validation per N
  Phase 4: Trade-off analysis — PnL/MDD vs frequency composite score
  Phase 5: Live projection — expected monthly/annual returns at optimal N

Standard Research Protocol:
  Entry: signal bar + 1 open
  Exit: Intrabar high/low (distance-based, abs(tp - opens[bar]))
  Fee: 0.10% * LEVERAGE(3) = 0.30% per trade (capital-space)
  Slippage: 0.02%
  ATR-scaled TP/SL: ATR(14)/median(576), clamp [0.6, 1.7]
  Timeout: 864 bars (72h) -> DROP
  WF: 3-fold expanding window, PASS = all folds OOS PnL > 0
  Sizing: compound (multiplicative), 1/N per slot
  Classification: production classify_candle (imported, never self-implemented)
  Neutral window: start~end price +/-1% (~257d)
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

from scripts.production.pattern_5m.indicators import classify_candle  # NEVER self-implement

# ============================================================
# CONSTANTS (Standard Research Protocol)
# ============================================================
LEVERAGE = 3
FEE_PCT = 0.10           # roundtrip fee in price-space (0.05% x 2)
SLIPPAGE_BUFFER = 0.02   # 0.02% per side
TIMEOUT_BARS = 864        # 72h -> DROP
ATR_PERIOD = 14
ATR_WINDOW = 576
CLAMP_LO = 0.6
CLAMP_HI = 1.7
ATR_WARMUP = ATR_PERIOD + ATR_WINDOW  # 590 bars
BARS_PER_DAY = 288
EMA_PERIOD = 20
EMA_LOOKBACK = 5

# MDD Sizing params (v1.34.0)
MDD_FULL_SIZE_BELOW_DD = 5.0
MDD_MIN_SIZE_ABOVE_DD = 15.0
MDD_MIN_SIZE_PCT = 25.0

# Momentum Guard params (v1.36.2)
MOM_LOOKBACK = 6
MOM_THRESHOLD = 1.0
MOM_COOLDOWN = 6

# Aggregate Risk Cap params (v1.35.5)
AGG_CAP_COUNTER = 3.0
AGG_CAP_WITH = 7.0

# N configurations to test
N_VALUES = [1, 2, 3, 5, 9]

# Direction cap mapping: N -> dir_cap (None = no cap)
# N<=3: no cap needed (max same-dir = N), N=5: cap=4, N=9: cap=7
DIR_CAP_MAP = {1: None, 2: None, 3: None, 5: 4, 9: 7}

DATA_FILE = os.path.join(PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
PATTERNS_FILE = os.path.join(PROJECT_ROOT, 'results', 'dynamic_patterns.json')
OUTPUT_FILE = os.path.join(PROJECT_ROOT, 'results', 'n_reduction_study.json')


# ============================================================
# DATA LOADING
# ============================================================

def load_data():
    """Load data and apply neutral window from dynamic_patterns.json."""
    df = pd.read_csv(DATA_FILE)
    n_total = len(df)

    with open(PATTERNS_FILE) as f:
        pat_data = json.load(f)

    neutral = pat_data.get('neutral_window')
    if neutral and neutral.get('enabled'):
        ns = neutral['start_bar']
        ne = neutral['end_bar']
        df = df.iloc[ns:ne + 1].reset_index(drop=True)
        print(f"Neutral window applied: bars {ns}-{ne} ({neutral['window_days']}d), "
              f"drift {neutral['drift_pct']:+.2f}%")
    else:
        target_bars = 270 * BARS_PER_DAY
        if n_total > target_bars:
            start = n_total - target_bars
            df = df.iloc[start:].reset_index(drop=True)
            print(f"No neutral window, using shifted 270d: last {len(df)} of {n_total} bars")

    types = df['type_code'].values.tolist()
    opens = df['open'].values.astype(np.float64)
    highs = df['high'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)
    closes = df['close'].values.astype(np.float64)
    n_bars = len(df)
    print(f"Data: {n_bars} bars ({n_bars / BARS_PER_DAY:.0f}d)")
    return df, types, opens, highs, lows, closes, n_bars


def load_patterns():
    """Load dynamic patterns and build tpsl map."""
    with open(PATTERNS_FILE) as f:
        pat_data = json.load(f)
    pL = set(pat_data['patterns']['long'])
    pS = set(pat_data['patterns']['short'])
    tpsl = {}
    for pat_name, tp_sl_list in pat_data['patterns_tpsl'].items():
        tpsl[pat_name] = (tp_sl_list[0], tp_sl_list[1])
    print(f"Patterns: {len(pL)}L + {len(pS)}S = {len(pL) + len(pS)}")
    return pL, pS, tpsl


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def compute_atr_ratio(highs, lows, closes):
    """Compute ATR(14)/median(ATR(14), 576) ratio."""
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


def compute_ema_slope(closes, ema_period=EMA_PERIOD, lookback=EMA_LOOKBACK):
    """Compute EMA slope for regime detection."""
    n = len(closes)
    ema = pd.Series(closes).ewm(span=ema_period, adjust=False).mean().values
    slope = np.full(n, 0.0)
    for i in range(lookback, n):
        if ema[i - lookback] > 0:
            slope[i] = (ema[i] / ema[i - lookback] - 1) * 100
    return slope


def generate_signals(types, patterns_long, patterns_short, n_bars, start_idx=0):
    """Generate all pattern signals."""
    signals = []
    for i in range(max(2, start_idx), n_bars):
        pat = f"{types[i - 2]}-{types[i - 1]}-{types[i]}"
        if pat in patterns_long:
            signals.append((i, pat, 'LONG'))
        if pat in patterns_short:
            signals.append((i, pat, 'SHORT'))
    return signals


# ============================================================
# EXIT LOGIC
# ============================================================

def _check_exit(pos, bar, opens, highs, lows, n_bars, atr_ratio, fee):
    """Check if position exits on this bar. Timeout -> DROP."""
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

    # ATR scaling
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

    # Timeout -> DROP (exclude from PnL)
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

    # Same-bar resolution: distance from bar OPEN (not entry)
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
    pnl -= fee  # fee = FEE_PCT * LEVERAGE (BingX charges on notional)

    return {'entry_bar': entry_bar, 'exit_bar': bar, 'pnl_slot': pnl,
            'reason': reason, 'drop': False, 'eff_tp': eff_tp, 'eff_sl': eff_sl}


# ============================================================
# PORTFOLIO SIMULATOR (parameterized N)
# ============================================================

def simulate_portfolio(signals, tpsl_map, opens, highs, lows, closes, n_bars,
                       atr_ratio, ema_slope, oos_start, oos_end,
                       max_positions=9, dir_cap=None,
                       use_regime_sizing=True, use_agg_risk=True,
                       use_momentum=True, use_mdd_sizing=True):
    """
    Multi-position portfolio simulator with configurable N.

    Returns:
      trades: list of completed trade dicts
      bar_snapshots: list of per-bar state snapshots (for correlation analysis)
    """
    SIZE_PCT = 100.0 / max_positions
    fee = FEE_PCT * LEVERAGE

    positions = []
    trades = []
    equity = 100.0
    peak_equity = 100.0

    # For correlation analysis
    bar_snapshots = []
    blocked_counts = {'dir_cap': 0, 'slots_full': 0, 'duplicate': 0,
                      'momentum': 0, 'agg_risk': 0}

    # Momentum cooldown tracker
    momentum_pause_until = {'LONG': -1, 'SHORT': -1}

    signals_in_range = [(s, p, d) for s, p, d in signals if oos_start <= s < oos_end]
    signals_sorted = sorted(signals_in_range, key=lambda x: x[0])
    sig_idx = 0

    for bar in range(oos_start, oos_end):
        # 1. Check exits
        closed_slots = []
        bar_pnl_sum = 0.0
        bar_sl_events = []

        for pos in positions:
            result = _check_exit(pos, bar, opens, highs, lows, n_bars, atr_ratio, fee)
            if result is not None:
                if result.get('drop', False):
                    closed_slots.append(pos['slot'])
                    continue
                result['pattern'] = pos['pattern']
                result['direction'] = pos['direction']
                sm = pos.get('size_mult', 1.0)
                result['size_mult'] = sm
                pnl_portfolio = result['pnl_slot'] * (SIZE_PCT / 100) * sm
                result['pnl_portfolio'] = pnl_portfolio
                trades.append(result)
                closed_slots.append(pos['slot'])
                bar_pnl_sum += pnl_portfolio
                if result['reason'] == 'SL':
                    bar_sl_events.append({
                        'direction': pos['direction'],
                        'pattern': pos['pattern'],
                        'pnl_slot': result['pnl_slot'],
                    })

        positions = [p for p in positions if p['slot'] not in closed_slots]

        equity += bar_pnl_sum
        if equity > peak_equity:
            peak_equity = equity

        # Record snapshot for correlation analysis (every bar with positions or SL events)
        if positions or bar_sl_events:
            n_long = sum(1 for p in positions if p['direction'] == 'LONG')
            n_short = sum(1 for p in positions if p['direction'] == 'SHORT')
            bar_snapshots.append({
                'bar': bar,
                'n_positions': len(positions),
                'n_long': n_long,
                'n_short': n_short,
                'equity': equity,
                'sl_events': len(bar_sl_events),
                'sl_directions': [e['direction'] for e in bar_sl_events],
            })

        # Momentum guard: check for cooldown trigger
        if use_momentum and bar >= MOM_LOOKBACK:
            price_now = closes[bar]
            price_ago = closes[bar - MOM_LOOKBACK]
            if price_ago > 0:
                pct_change = (price_now / price_ago - 1) * 100
                if pct_change > MOM_THRESHOLD:
                    momentum_pause_until['SHORT'] = bar + MOM_COOLDOWN
                elif pct_change < -MOM_THRESHOLD:
                    momentum_pause_until['LONG'] = bar + MOM_COOLDOWN

        # 2. Process entries
        while sig_idx < len(signals_sorted) and signals_sorted[sig_idx][0] == bar:
            sig_bar, pat, direction = signals_sorted[sig_idx]
            sig_idx += 1

            if len(positions) >= max_positions:
                blocked_counts['slots_full'] += 1
                continue

            # Direction cap
            if dir_cap is not None:
                dir_count = sum(1 for p in positions if p['direction'] == direction)
                if dir_count >= dir_cap:
                    blocked_counts['dir_cap'] += 1
                    continue

            # Duplicate pattern check (always active)
            if any(p['pattern'] == pat for p in positions):
                blocked_counts['duplicate'] += 1
                continue

            entry_bar = sig_bar + 1
            if entry_bar >= n_bars:
                continue
            tp_sl = tpsl_map.get(pat)
            if tp_sl is None:
                continue

            # Momentum guard
            if use_momentum and bar < momentum_pause_until[direction]:
                blocked_counts['momentum'] += 1
                continue

            # Regime sizing
            sm = 1.0
            if use_regime_sizing and bar < len(ema_slope):
                s = ema_slope[bar]
                if s > 0 and direction == 'SHORT':
                    sm = 0.3
                elif s <= 0 and direction == 'LONG':
                    sm = 0.3

            # MDD sizing
            if use_mdd_sizing and peak_equity > 0:
                dd_pct = (peak_equity - equity) / peak_equity * 100
                if dd_pct > MDD_FULL_SIZE_BELOW_DD:
                    ratio = min(1.0, (dd_pct - MDD_FULL_SIZE_BELOW_DD) /
                                (MDD_MIN_SIZE_ABOVE_DD - MDD_FULL_SIZE_BELOW_DD))
                    mdd_mult = 1.0 - ratio * (1.0 - MDD_MIN_SIZE_PCT / 100)
                    sm *= mdd_mult

            # Aggregate risk cap (only for N > 1)
            if use_agg_risk and max_positions > 1 and bar < len(ema_slope):
                is_uptrend = ema_slope[bar] > 0
                is_counter = (is_uptrend and direction == 'SHORT') or \
                             (not is_uptrend and direction == 'LONG')
                cap_pct = AGG_CAP_COUNTER if is_counter else AGG_CAP_WITH

                existing_exposure = 0.0
                for p in positions:
                    if p['direction'] == direction:
                        p_sl = p['sl_pct']
                        p_sig = p['signal_bar']
                        if atr_ratio is not None and p_sig < len(atr_ratio) and \
                                not np.isnan(atr_ratio[p_sig]):
                            p_r = max(CLAMP_LO, min(CLAMP_HI, atr_ratio[p_sig]))
                        else:
                            p_r = 1.0
                        p_eff_sl = p_sl * p_r
                        p_sm = p.get('size_mult', 1.0)
                        existing_exposure += p_eff_sl * (1.0 / max_positions) * LEVERAGE * p_sm

                new_r = 1.0
                if atr_ratio is not None and sig_bar < len(atr_ratio) and \
                        not np.isnan(atr_ratio[sig_bar]):
                    new_r = max(CLAMP_LO, min(CLAMP_HI, atr_ratio[sig_bar]))
                new_eff_sl = tp_sl[1] * new_r
                new_exposure = new_eff_sl * (1.0 / max_positions) * LEVERAGE * sm

                if existing_exposure + new_exposure > cap_pct:
                    blocked_counts['agg_risk'] += 1
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

    # Force-close remaining (OOS boundary)
    for pos in positions:
        entry_bar = pos['entry_bar']
        if entry_bar >= n_bars:
            continue
        entry = opens[entry_bar]
        if entry <= 0:
            continue
        exit_bar = min(oos_end - 1, n_bars - 1)
        exit_price = opens[exit_bar]
        if pos['direction'] == 'LONG':
            pnl = (exit_price / entry - 1) * 100 * LEVERAGE
        else:
            pnl = (1 - exit_price / entry) * 100 * LEVERAGE
        pnl -= fee
        sm = pos.get('size_mult', 1.0)
        trades.append({
            'entry_bar': entry_bar, 'exit_bar': exit_bar, 'pnl_slot': pnl,
            'reason': 'OOS_END', 'pattern': pos['pattern'],
            'direction': pos['direction'], 'size_mult': sm,
            'pnl_portfolio': pnl * (SIZE_PCT / 100) * sm, 'drop': False,
        })

    return trades, bar_snapshots, blocked_counts


# ============================================================
# METRICS
# ============================================================

def compute_metrics(trades, n_bars_range=0):
    """Compute comprehensive portfolio metrics from trade list."""
    if not trades:
        return {'trades': 0, 'wr': 0.0, 'pnl': 0.0, 'mdd': 0.0,
                'pnl_mdd': 0.0, 'pnl_per_trade': 0.0,
                'long_trades': 0, 'short_trades': 0,
                'trades_per_day': 0.0, 'max_consec_loss': 0,
                'avg_win': 0.0, 'avg_loss': 0.0}

    wins = [t for t in trades if t['pnl_slot'] > 0]
    losses = [t for t in trades if t['pnl_slot'] <= 0]
    sorted_trades = sorted(trades, key=lambda x: x['entry_bar'])

    # Equity curve for MDD
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

    total_pnl = equity - 100.0
    wr = len(wins) / len(trades) * 100 if trades else 0
    pnl_mdd = total_pnl / max_dd if max_dd > 0 else 0
    long_trades = sum(1 for t in trades if t['direction'] == 'LONG')
    short_trades = sum(1 for t in trades if t['direction'] == 'SHORT')

    # Max consecutive losses
    max_consec = 0
    current_consec = 0
    for t in sorted_trades:
        if t['pnl_slot'] <= 0:
            current_consec += 1
            max_consec = max(max_consec, current_consec)
        else:
            current_consec = 0

    # Avg win/loss (slot-level)
    avg_win = np.mean([t['pnl_slot'] for t in wins]) if wins else 0
    avg_loss = np.mean([abs(t['pnl_slot']) for t in losses]) if losses else 0

    # Trades per day
    days = n_bars_range / BARS_PER_DAY if n_bars_range > 0 else 1
    trades_per_day = len(trades) / days if days > 0 else 0

    return {
        'trades': len(trades),
        'wr': round(wr, 1),
        'pnl': round(total_pnl, 2),
        'mdd': round(max_dd, 2),
        'pnl_mdd': round(pnl_mdd, 2),
        'pnl_per_trade': round(total_pnl / len(trades), 3) if trades else 0,
        'long_trades': long_trades,
        'short_trades': short_trades,
        'trades_per_day': round(trades_per_day, 2),
        'max_consec_loss': max_consec,
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
    }


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

def analyze_correlation(bar_snapshots, trades):
    """
    Analyze position correlation and simultaneous SL events.

    Returns dict with:
      - avg_concurrent: average number of concurrent positions
      - max_concurrent: maximum concurrent positions
      - avg_same_dir_ratio: average fraction of positions in majority direction
      - simultaneous_sl_events: list of events where >=2 SL hits on same bar
      - max_simultaneous_sl: maximum SL hits on a single bar
      - correlated_loss_pct: fraction of total losses from simultaneous SL events
    """
    if not bar_snapshots:
        return {
            'avg_concurrent': 0.0, 'max_concurrent': 0,
            'avg_same_dir_ratio': 0.0,
            'simultaneous_sl_events': 0, 'max_simultaneous_sl': 0,
            'correlated_loss_pct': 0.0,
            'total_sl_bars_with_multi_hit': 0,
        }

    # Concurrent position stats
    concurrent = [s['n_positions'] for s in bar_snapshots if s['n_positions'] > 0]
    avg_concurrent = np.mean(concurrent) if concurrent else 0
    max_concurrent = max(concurrent) if concurrent else 0

    # Same-direction ratio (what fraction of positions are in the majority direction)
    same_dir_ratios = []
    for s in bar_snapshots:
        if s['n_positions'] > 1:
            majority = max(s['n_long'], s['n_short'])
            same_dir_ratios.append(majority / s['n_positions'])
    avg_same_dir_ratio = np.mean(same_dir_ratios) if same_dir_ratios else 0

    # Simultaneous SL events (>=2 SL hits on same bar)
    sl_bars = [s for s in bar_snapshots if s['sl_events'] >= 2]
    max_sim_sl = max((s['sl_events'] for s in bar_snapshots), default=0)

    # Correlated loss: fraction of total losses from multi-SL bars
    multi_sl_bars = set(s['bar'] for s in bar_snapshots if s['sl_events'] >= 2)
    total_loss = sum(abs(t['pnl_portfolio']) for t in trades if t['pnl_slot'] <= 0)
    corr_loss = sum(abs(t['pnl_portfolio']) for t in trades
                    if t['pnl_slot'] <= 0 and t.get('exit_bar') in multi_sl_bars)
    corr_loss_pct = corr_loss / total_loss * 100 if total_loss > 0 else 0

    return {
        'avg_concurrent': round(avg_concurrent, 2),
        'max_concurrent': max_concurrent,
        'avg_same_dir_ratio': round(avg_same_dir_ratio, 3),
        'simultaneous_sl_events': len(sl_bars),
        'max_simultaneous_sl': max_sim_sl,
        'correlated_loss_pct': round(corr_loss_pct, 1),
        'total_sl_bars_with_multi_hit': len(multi_sl_bars),
    }


# ============================================================
# WALK-FORWARD (3-fold expanding window)
# ============================================================

def run_wf(signals, tpsl_map, opens, highs, lows, closes, n_bars,
           atr_ratio, ema_slope, max_positions, dir_cap, label=""):
    """Run 3-fold expanding window WF for a given N."""
    n_folds = 3
    seg_size = n_bars // (n_folds + 1)

    fold_results = []
    all_trades = []
    all_snapshots = []

    for fold in range(n_folds):
        oos_start = (fold + 1) * seg_size
        oos_end = (fold + 2) * seg_size if fold < n_folds - 1 else n_bars

        trades, snapshots, blocked = simulate_portfolio(
            signals, tpsl_map, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, oos_start, oos_end,
            max_positions=max_positions, dir_cap=dir_cap,
        )
        m = compute_metrics(trades, oos_end - oos_start)
        corr = analyze_correlation(snapshots, trades)
        fold_results.append({
            'fold': fold + 1,
            'oos_bars': oos_end - oos_start,
            **m,
            'correlation': corr,
        })
        all_trades.extend(trades)
        all_snapshots.extend(snapshots)

    total_pnl = sum(f['pnl'] for f in fold_results)
    max_mdd = max(f['mdd'] for f in fold_results) if fold_results else 0
    n_positive = sum(1 for f in fold_results if f['pnl'] > 0)
    pnl_mdd = total_pnl / max_mdd if max_mdd > 0 else 0
    total_trades = sum(f['trades'] for f in fold_results)
    total_oos_bars = sum(f['oos_bars'] for f in fold_results)
    avg_wr = np.mean([f['wr'] for f in fold_results]) if fold_results else 0

    return {
        'label': label,
        'N': max_positions,
        'dir_cap': dir_cap,
        'folds': fold_results,
        'total_pnl': round(total_pnl, 2),
        'max_mdd': round(max_mdd, 2),
        'pnl_mdd': round(pnl_mdd, 2),
        'total_trades': total_trades,
        'avg_wr': round(avg_wr, 1),
        'trades_per_day': round(total_trades / (total_oos_bars / BARS_PER_DAY), 2) if total_oos_bars > 0 else 0,
        'wf_pass': f"{n_positive}/{n_folds}",
        'wf_passed': n_positive == n_folds,
    }


# ============================================================
# DISPLAY HELPERS
# ============================================================

def _print_n_row(label, m, corr=None):
    pnl_mdd_str = f"{m['pnl_mdd']:>7.2f}" if m['pnl_mdd'] > 0 else "   N/A"
    corr_str = ""
    if corr:
        corr_str = (f" | AvgConc={corr['avg_concurrent']:.1f} "
                    f"MaxSL={corr['max_simultaneous_sl']} "
                    f"CorrLoss={corr['correlated_loss_pct']:.0f}%")
    print(f"  {label:<20s} | "
          f"T={m['trades']:>4d} WR={m['wr']:>5.1f}% "
          f"PnL={m['pnl']:>+8.1f}% MDD={m['mdd']:>5.1f}% "
          f"PnL/MDD={pnl_mdd_str} "
          f"f={m['trades_per_day']:.2f}/d "
          f"ConsecL={m['max_consec_loss']}"
          f"{corr_str}")


def _print_wf_row(r):
    status = "PASS" if r['wf_passed'] else "FAIL"
    print(f"  N={r['N']:<2d} (cap={str(r['dir_cap']):<4s}) | "
          f"WF {r['wf_pass']} {status}  "
          f"T={r['total_trades']:>4d} "
          f"WR={r['avg_wr']:>5.1f}% "
          f"PnL={r['total_pnl']:>+8.1f}% MDD={r['max_mdd']:>5.1f}% "
          f"PnL/MDD={r['pnl_mdd']:>7.2f} "
          f"f={r['trades_per_day']:.2f}/d")


# ============================================================
# MAIN STUDY
# ============================================================

def main():
    t0 = time.time()
    print("=" * 90)
    print("N Reduction Study: Multi-Position Count Optimization")
    print("  130 patterns (61L+69S), Hedge mode, All filters ON")
    print("  N = [1, 2, 3, 5, 9] with proportional direction caps")
    print("=" * 90)

    # Load data
    df, types, opens, highs, lows, closes, n_bars = load_data()
    pL, pS, tpsl = load_patterns()
    atr_ratio = compute_atr_ratio(highs, lows, closes)
    ema_slope = compute_ema_slope(closes)
    signals = generate_signals(types, pL, pS, n_bars)
    n_long_signals = sum(1 for _, _, d in signals if d == 'LONG')
    n_short_signals = sum(1 for _, _, d in signals if d == 'SHORT')
    print(f"Total signals: {len(signals)} (L={n_long_signals}, S={n_short_signals})")
    print()

    results = {
        'metadata': {
            'study': 'N Reduction Study',
            'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'data_bars': n_bars,
            'data_days': round(n_bars / BARS_PER_DAY, 1),
            'total_signals': len(signals),
            'long_signals': n_long_signals,
            'short_signals': n_short_signals,
            'n_values_tested': N_VALUES,
            'dir_cap_map': {str(k): v for k, v in DIR_CAP_MAP.items()},
            'leverage': LEVERAGE,
            'fee_pct': FEE_PCT,
            'timeout_bars': TIMEOUT_BARS,
        }
    }

    # ================================================================
    # PHASE 1: N-sweep Full-Sample Portfolio Backtest
    # ================================================================
    print("=" * 90)
    print("PHASE 1: Full-Sample N-Sweep (All Filters ON)")
    print("  Filters: DirCap + RegimeSz + AggRisk + MomGuard + MDDSz + Timeout")
    print("=" * 90)
    print()

    start_bar = max(ATR_WARMUP, 2)
    phase1_results = {}

    for N in N_VALUES:
        dir_cap = DIR_CAP_MAP[N]
        label = f"N={N} cap={dir_cap}"

        trades, snapshots, blocked = simulate_portfolio(
            signals, tpsl, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, start_bar, n_bars,
            max_positions=N, dir_cap=dir_cap,
        )
        m = compute_metrics(trades, n_bars - start_bar)
        corr = analyze_correlation(snapshots, trades)

        _print_n_row(label, m, corr)
        phase1_results[N] = {
            'N': N, 'dir_cap': dir_cap,
            'metrics': m, 'correlation': corr,
            'blocked': blocked,
        }

    results['phase1_n_comparison'] = phase1_results
    print()

    # Summary table
    print("--- Phase 1 Summary ---")
    print(f"  {'N':>3s} {'Cap':>4s} | {'Trades':>6s} {'WR':>6s} {'PnL':>9s} "
          f"{'MDD':>6s} {'PnL/MDD':>8s} {'f/day':>6s} {'CL':>3s} "
          f"{'AvgConc':>7s} {'MaxSL':>5s} {'CorrL%':>6s}")
    print(f"  {'---':>3s} {'----':>4s} | {'------':>6s} {'------':>6s} {'---------':>9s} "
          f"{'------':>6s} {'--------':>8s} {'------':>6s} {'---':>3s} "
          f"{'-------':>7s} {'-----':>5s} {'------':>6s}")
    for N in N_VALUES:
        d = phase1_results[N]
        m = d['metrics']
        c = d['correlation']
        cap_str = str(d['dir_cap']) if d['dir_cap'] is not None else '-'
        pnl_mdd_str = f"{m['pnl_mdd']:.2f}" if m['pnl_mdd'] > 0 else "N/A"
        print(f"  {N:>3d} {cap_str:>4s} | {m['trades']:>6d} {m['wr']:>5.1f}% "
              f"{m['pnl']:>+8.1f}% {m['mdd']:>5.1f}% {pnl_mdd_str:>8s} "
              f"{m['trades_per_day']:>5.2f} {m['max_consec_loss']:>3d} "
              f"{c['avg_concurrent']:>6.1f} {c['max_simultaneous_sl']:>5d} "
              f"{c['correlated_loss_pct']:>5.1f}%")
    print()

    # ================================================================
    # PHASE 2: Correlation Deep-Dive
    # ================================================================
    print("=" * 90)
    print("PHASE 2: Correlation Analysis")
    print("=" * 90)
    print()

    phase2_results = {}
    for N in N_VALUES:
        dir_cap = DIR_CAP_MAP[N]
        trades, snapshots, blocked = simulate_portfolio(
            signals, tpsl, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, start_bar, n_bars,
            max_positions=N, dir_cap=dir_cap,
        )
        corr = analyze_correlation(snapshots, trades)

        # Additional: direction breakdown of losses
        long_losses = [t for t in trades if t['direction'] == 'LONG' and t['pnl_slot'] <= 0]
        short_losses = [t for t in trades if t['direction'] == 'SHORT' and t['pnl_slot'] <= 0]
        long_wins = [t for t in trades if t['direction'] == 'LONG' and t['pnl_slot'] > 0]
        short_wins = [t for t in trades if t['direction'] == 'SHORT' and t['pnl_slot'] > 0]

        long_wr = len(long_wins) / (len(long_wins) + len(long_losses)) * 100 \
            if (long_wins or long_losses) else 0
        short_wr = len(short_wins) / (len(short_wins) + len(short_losses)) * 100 \
            if (short_wins or short_losses) else 0

        # Count bars where all positions are same direction
        all_same_dir_bars = 0
        total_multi_pos_bars = 0
        for s in snapshots:
            if s['n_positions'] > 1:
                total_multi_pos_bars += 1
                if s['n_long'] == 0 or s['n_short'] == 0:
                    all_same_dir_bars += 1
        all_same_pct = all_same_dir_bars / total_multi_pos_bars * 100 \
            if total_multi_pos_bars > 0 else 0

        phase2_data = {
            'N': N,
            'correlation': corr,
            'long_wr': round(long_wr, 1),
            'short_wr': round(short_wr, 1),
            'long_trades': len(long_wins) + len(long_losses),
            'short_trades': len(short_wins) + len(short_losses),
            'all_same_dir_pct': round(all_same_pct, 1),
            'multi_pos_bars': total_multi_pos_bars,
        }
        phase2_results[N] = phase2_data

        print(f"  N={N}: AvgConc={corr['avg_concurrent']:.1f}, "
              f"MaxConc={corr['max_concurrent']}, "
              f"MaxSimSL={corr['max_simultaneous_sl']}, "
              f"SimSLevents={corr['simultaneous_sl_events']}, "
              f"CorrLoss={corr['correlated_loss_pct']:.1f}%, "
              f"AllSameDir={all_same_pct:.1f}%")
        print(f"         LONG WR={long_wr:.1f}% ({len(long_wins) + len(long_losses)} trades), "
              f"SHORT WR={short_wr:.1f}% ({len(short_wins) + len(short_losses)} trades)")

    results['phase2_correlation'] = phase2_results
    print()

    # ================================================================
    # PHASE 3: Walk-Forward 3-Fold OOS Validation
    # ================================================================
    print("=" * 90)
    print("PHASE 3: Walk-Forward 3-Fold OOS Validation")
    print("=" * 90)
    print()

    phase3_results = {}
    for N in N_VALUES:
        dir_cap = DIR_CAP_MAP[N]
        wf = run_wf(
            signals, tpsl, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, max_positions=N, dir_cap=dir_cap,
            label=f"N={N}",
        )
        _print_wf_row(wf)
        phase3_results[N] = wf

        # Print fold details
        for fold in wf['folds']:
            status_emoji = "+" if fold['pnl'] > 0 else "-"
            print(f"      Fold {fold['fold']}: T={fold['trades']:>4d} "
                  f"WR={fold['wr']:>5.1f}% PnL={fold['pnl']:>+7.1f}% "
                  f"MDD={fold['mdd']:>5.1f}% [{status_emoji}]"
                  f"  MaxSimSL={fold['correlation']['max_simultaneous_sl']} "
                  f"CorrL={fold['correlation']['correlated_loss_pct']:.0f}%")
        print()

    results['phase3_wf_oos'] = phase3_results

    # ================================================================
    # PHASE 4: Trade-off Analysis
    # ================================================================
    print("=" * 90)
    print("PHASE 4: Trade-off Analysis — Risk-Adjusted Return vs Frequency")
    print("=" * 90)
    print()

    phase4_results = {}
    print(f"  {'N':>3s} | {'WF':>5s} | {'OOS PnL':>9s} {'OOS MDD':>8s} {'PnL/MDD':>8s} "
          f"{'f/day':>6s} {'Score':>8s} | Verdict")
    print(f"  {'---':>3s} | {'-----':>5s} | {'---------':>9s} {'--------':>8s} {'--------':>8s} "
          f"{'------':>6s} {'--------':>8s} | -------")

    for N in N_VALUES:
        wf = phase3_results[N]
        is_m = phase1_results[N]['metrics']

        # Composite score: PnL/MDD * sqrt(trades_per_day)
        # sqrt penalizes low frequency less aggressively than linear
        freq = wf['trades_per_day']
        pnl_mdd = wf['pnl_mdd']
        composite = pnl_mdd * (freq ** 0.5) if pnl_mdd > 0 and freq > 0 else 0

        wf_str = wf['wf_pass']
        status = "PASS" if wf['wf_passed'] else "FAIL"

        # Verdict
        if not wf['wf_passed']:
            verdict = "REJECTED (WF FAIL)"
        elif composite <= 0:
            verdict = "REJECTED (negative score)"
        else:
            verdict = f"CANDIDATE (score={composite:.2f})"

        print(f"  {N:>3d} | {wf_str:>5s} | {wf['total_pnl']:>+8.1f}% "
              f"{wf['max_mdd']:>7.1f}% {pnl_mdd:>7.2f} "
              f"{freq:>5.2f} {composite:>7.2f} | {verdict}")

        phase4_results[N] = {
            'N': N,
            'wf_passed': wf['wf_passed'],
            'oos_pnl': wf['total_pnl'],
            'oos_mdd': wf['max_mdd'],
            'oos_pnl_mdd': pnl_mdd,
            'trades_per_day': freq,
            'composite_score': round(composite, 3),
            'is_pnl': is_m['pnl'],
            'is_mdd': is_m['mdd'],
            'is_wr': is_m['wr'],
            'verdict': verdict,
        }

    results['phase4_tradeoff'] = phase4_results
    print()

    # Find optimal N (highest composite among WF-passed)
    wf_passed_ns = [N for N in N_VALUES if phase3_results[N]['wf_passed']]
    if wf_passed_ns:
        optimal_n = max(wf_passed_ns, key=lambda n: phase4_results[n]['composite_score'])
    else:
        # If none pass WF, pick the one with best composite anyway
        optimal_n = max(N_VALUES, key=lambda n: phase4_results[n]['composite_score'])
    print(f"  OPTIMAL N = {optimal_n} (highest composite among WF-passed)")
    print()

    # ================================================================
    # PHASE 5: Live Projection
    # ================================================================
    print("=" * 90)
    print("PHASE 5: Live Projection (Capital $2,261)")
    print("=" * 90)
    print()

    CAPITAL = 2261.0
    DAYS_PER_MONTH = 30
    DAYS_PER_YEAR = 365

    phase5_results = {}
    print(f"  {'N':>3s} | {'WF':>5s} | {'Avg Edge/Trade':>14s} {'f/day':>6s} "
          f"{'Daily EV%':>10s} {'Monthly%':>9s} {'Annual%':>8s} "
          f"{'Mo$':>8s} {'Yr$':>9s}")
    print(f"  {'---':>3s} | {'-----':>5s} | {'--------------':>14s} {'------':>6s} "
          f"{'----------':>10s} {'---------':>9s} {'--------':>8s} "
          f"{'--------':>8s} {'---------':>9s}")

    for N in N_VALUES:
        wf = phase3_results[N]
        freq = wf['trades_per_day']

        # OOS average edge per trade
        oos_total_pnl = wf['total_pnl']
        oos_total_trades = wf['total_trades']
        if oos_total_trades > 0:
            avg_edge = oos_total_pnl / oos_total_trades
        else:
            avg_edge = 0

        daily_ev = avg_edge * freq
        monthly_ev = daily_ev * DAYS_PER_MONTH
        annual_ev = daily_ev * DAYS_PER_YEAR

        # Dollar projections (simple, not compound for conservatism)
        monthly_dollar = CAPITAL * monthly_ev / 100
        annual_dollar = CAPITAL * annual_ev / 100

        wf_str = wf['wf_pass']
        marker = " <-- OPTIMAL" if N == optimal_n else ""

        print(f"  {N:>3d} | {wf_str:>5s} | {avg_edge:>+13.3f}% {freq:>5.2f} "
              f"{daily_ev:>+9.3f}% {monthly_ev:>+8.1f}% {annual_ev:>+7.1f}% "
              f"${monthly_dollar:>+7.0f} ${annual_dollar:>+8.0f}{marker}")

        phase5_results[N] = {
            'N': N,
            'wf_passed': wf['wf_passed'],
            'avg_edge_per_trade': round(avg_edge, 4),
            'trades_per_day': freq,
            'daily_ev_pct': round(daily_ev, 4),
            'monthly_ev_pct': round(monthly_ev, 2),
            'annual_ev_pct': round(annual_ev, 2),
            'capital': CAPITAL,
            'monthly_dollar': round(monthly_dollar, 2),
            'annual_dollar': round(annual_dollar, 2),
        }

    results['phase5_live_projection'] = phase5_results
    print()

    # ================================================================
    # FINAL VERDICT
    # ================================================================
    print("=" * 90)
    print("FINAL VERDICT")
    print("=" * 90)
    print()

    # Determine verdict
    n9_wf = phase3_results[9]
    opt_wf = phase3_results[optimal_n]
    opt_is = phase1_results[optimal_n]

    if optimal_n < 9 and opt_wf['wf_passed']:
        verdict = "GO"
        recommendation = (
            f"Reduce N from 9 to {optimal_n}. "
            f"OOS PnL/MDD improves from {phase3_results[9]['pnl_mdd']:.2f} to "
            f"{opt_wf['pnl_mdd']:.2f}. "
            f"Correlated loss drops from "
            f"{phase2_results[9]['correlation']['correlated_loss_pct']:.0f}% to "
            f"{phase2_results[optimal_n]['correlation']['correlated_loss_pct']:.0f}%. "
            f"Trade frequency: {opt_wf['trades_per_day']:.2f}/day."
        )
    elif optimal_n == 9 and n9_wf['wf_passed']:
        verdict = "STOP"
        recommendation = (
            "N=9 remains optimal. Reducing N does not improve risk-adjusted returns "
            "enough to justify the reduction in trade frequency."
        )
    else:
        verdict = "PARTIAL"
        # Find best WF-passed N
        if wf_passed_ns:
            best_n = max(wf_passed_ns, key=lambda n: phase4_results[n]['composite_score'])
            recommendation = (
                f"Mixed results. Best WF-passed: N={best_n} with composite "
                f"{phase4_results[best_n]['composite_score']:.2f}. "
                f"Review fold details before deploying."
            )
        else:
            recommendation = "No N configuration passes WF. Investigate edge degradation."

    print(f"  Verdict: {verdict}")
    print(f"  Optimal N: {optimal_n}")
    print(f"  Recommendation: {recommendation}")
    print()

    # Side-by-side comparison: N=optimal vs N=9
    if optimal_n != 9:
        print("  --- Head-to-Head: Optimal vs Current ---")
        for metric_name, n_opt_val, n9_val in [
            ("IS WR", phase1_results[optimal_n]['metrics']['wr'],
             phase1_results[9]['metrics']['wr']),
            ("IS PnL", phase1_results[optimal_n]['metrics']['pnl'],
             phase1_results[9]['metrics']['pnl']),
            ("IS MDD", phase1_results[optimal_n]['metrics']['mdd'],
             phase1_results[9]['metrics']['mdd']),
            ("IS PnL/MDD", phase1_results[optimal_n]['metrics']['pnl_mdd'],
             phase1_results[9]['metrics']['pnl_mdd']),
            ("OOS PnL", phase3_results[optimal_n]['total_pnl'],
             phase3_results[9]['total_pnl']),
            ("OOS MDD", phase3_results[optimal_n]['max_mdd'],
             phase3_results[9]['max_mdd']),
            ("OOS PnL/MDD", phase3_results[optimal_n]['pnl_mdd'],
             phase3_results[9]['pnl_mdd']),
            ("Trades/day", phase3_results[optimal_n]['trades_per_day'],
             phase3_results[9]['trades_per_day']),
            ("CorrLoss%", phase2_results[optimal_n]['correlation']['correlated_loss_pct'],
             phase2_results[9]['correlation']['correlated_loss_pct']),
            ("MaxSimSL", phase2_results[optimal_n]['correlation']['max_simultaneous_sl'],
             phase2_results[9]['correlation']['max_simultaneous_sl']),
            ("ConsecLoss", phase1_results[optimal_n]['metrics']['max_consec_loss'],
             phase1_results[9]['metrics']['max_consec_loss']),
        ]:
            delta = n_opt_val - n9_val
            better = "+" if (("MDD" in metric_name or "ConsecLoss" in metric_name or
                              "CorrLoss" in metric_name or "MaxSim" in metric_name)
                             and delta < 0) or \
                            (("PnL" in metric_name or "WR" in metric_name or
                              "Trades" in metric_name) and delta > 0 and
                             "MDD" not in metric_name) else \
                     ("-" if delta != 0 else "=")
            print(f"    {metric_name:<14s}: N={optimal_n}: {n_opt_val:>8.2f}  "
                  f"N=9: {n9_val:>8.2f}  delta={delta:>+8.2f} [{better}]")

    results['verdict'] = verdict
    results['optimal_n'] = optimal_n
    results['recommendation'] = recommendation

    elapsed = time.time() - t0
    results['metadata']['elapsed_sec'] = round(elapsed, 1)
    print(f"\nTotal time: {elapsed:.1f}s")

    # Save results
    class _NumpyEncoder(json.JSONEncoder):
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

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)
    print(f"Results saved: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
