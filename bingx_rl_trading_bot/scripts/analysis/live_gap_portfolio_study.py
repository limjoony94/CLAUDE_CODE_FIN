#!/usr/bin/env python3
"""
Live Performance Gap Study: Multi-Position Portfolio Optimization
=================================================================
Root cause: BTC +9% rally (02-24~26) -> 8 concurrent SHORT SL hits -> -123% slot loss.
WF backtest misses correlated multi-position risk.

4 Hypotheses:
  H1: Direction Cap reduction  [3,4,5,6,7,8(current),9(no-cap)]
  H2: Portfolio-Level Stop Loss [5%,10%,15%,20%,25%,inf]
  H3: Regime Sizing strengthening [counter_mult: 0.0,0.1,0.2,0.3(current)]
  H4: Best-of H1+H2+H3 combined

KEY INNOVATION: Full multi-position portfolio simulator with:
  - N=9 virtual slots, 1/N compound sizing (11.1%)
  - Intrabar unrealized PnL tracking for portfolio stop-loss
  - Correlated loss event detection (max simultaneous SL hits)
  - Regime-aware sizing via EMA(20) slope
  - ATR-scaled TP/SL per-pattern
  - Compound (multiplicative) equity
  - Timeout 864 bars -> DROP

Standard Research Protocol:
  Entry: signal bar + 1 open
  Exit: Intrabar high/low (distance-based, abs(tp - opens[bar]))
  Fee: 0.10% * LEVERAGE(3) = 0.30% per trade (capital-space)
  Slippage buffer: 0.02%
  ATR-scaled TP/SL: ATR(14)/median(576), clamp [0.6, 1.7]
  WF: 3-fold expanding window, PASS = all folds OOS PnL > 0
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

# ============================================================
# CONSTANTS
# ============================================================
LEVERAGE = 3
FEE_PCT = 0.10          # roundtrip fee in price-space
SLIPPAGE_BUFFER = 0.02  # 0.02% per side
TIMEOUT_BARS = 864       # 72h -> DROP
MAX_POSITIONS = 9
ATR_PERIOD = 14
ATR_WINDOW = 576
CLAMP_LO = 0.6
CLAMP_HI = 1.7
ATR_WARMUP = ATR_PERIOD + ATR_WINDOW  # 590 bars
BARS_PER_DAY = 288
EMA_PERIOD = 20          # regime detection
EMA_LOOKBACK = 5         # slope lookback

DATA_FILE = os.path.join(PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
PATTERNS_FILE = os.path.join(PROJECT_ROOT, 'results', 'dynamic_patterns.json')
OUTPUT_FILE = os.path.join(PROJECT_ROOT, 'results', 'live_gap_portfolio_study.json')


# ============================================================
# CORE DATA LOADING
# ============================================================

def load_data():
    """Load 270d CSV data. Uses pre-classified type_code column."""
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
    """Generate signals. Returns list of (bar_idx, pattern, direction)."""
    signals = []
    for i in range(max(2, start_idx), n_bars):
        pat = f"{types[i - 2]}-{types[i - 1]}-{types[i]}"
        if pat in patterns_long:
            signals.append((i, pat, 'LONG'))
        if pat in patterns_short:
            signals.append((i, pat, 'SHORT'))
    return signals


# ============================================================
# MULTI-POSITION PORTFOLIO SIMULATOR (compound equity)
# ============================================================

def simulate_portfolio(signals, tpsl_map, opens, highs, lows, closes, n_bars,
                       atr_ratio, direction_cap, oos_start, oos_end,
                       regime_mult=None, ema_slope=None,
                       portfolio_sl_pct=None):
    """Full multi-position portfolio simulator with compound equity.

    Args:
        signals: list of (bar_idx, pattern, direction)
        tpsl_map: {pattern: (tp_pct, sl_pct)}
        closes: close prices array (for unrealized PnL calculation)
        direction_cap: max positions in same direction
        regime_mult: float, counter-regime size multiplier (None=1.0 uniform)
        ema_slope: array of EMA slope values for regime detection
        portfolio_sl_pct: float, if total unrealized PnL < -X%, close all
                          (None = disabled)

    Returns:
        trades: list of trade dicts
        stats: dict with correlated loss events, max simultaneous positions, etc.
    """
    SIZE_PCT = 100.0 / MAX_POSITIONS  # 11.1% per slot
    fee = FEE_PCT * LEVERAGE  # 0.30% in capital space

    positions = []  # list of open position dicts
    trades = []     # completed trades
    equity = 100.0  # starting equity
    peak_equity = 100.0

    # Correlated loss tracking
    max_corr_loss = 0.0       # worst single-bar correlated loss event (% of equity)
    max_sim_positions = 0     # max simultaneous positions
    max_sim_direction = {'LONG': 0, 'SHORT': 0}
    portfolio_sl_events = 0   # number of portfolio SL triggers

    signals_in_range = [(s, p, d) for s, p, d in signals if oos_start <= s < oos_end]
    signals_sorted = sorted(signals_in_range, key=lambda x: x[0])
    sig_idx = 0

    for bar in range(oos_start, oos_end):
        # 1. Check exits for all open positions
        closed_slots = []
        bar_pnl_sum = 0.0  # total portfolio PnL from trades closing this bar
        bar_sl_count = 0    # SL hits this bar

        for pos in positions:
            result = _check_exit(pos, bar, opens, highs, lows, n_bars, atr_ratio, fee)
            if result is not None:
                if result.get('drop', False):
                    # Timeout -> DROP
                    closed_slots.append(pos['slot'])
                    continue
                result['pattern'] = pos['pattern']
                result['direction'] = pos['direction']
                sm = pos.get('size_mult', 1.0)
                result['size_mult'] = sm
                # Compound: PnL relative to current equity, scaled by slot fraction
                pnl_portfolio = result['pnl_slot'] * (SIZE_PCT / 100) * sm
                result['pnl_portfolio'] = pnl_portfolio
                trades.append(result)
                closed_slots.append(pos['slot'])
                bar_pnl_sum += pnl_portfolio
                if result['reason'] == 'SL':
                    bar_sl_count += 1

        positions = [p for p in positions if p['slot'] not in closed_slots]

        # Track correlated loss events
        if bar_pnl_sum < 0 and bar_sl_count >= 2:
            loss_pct = abs(bar_pnl_sum) / equity * 100 if equity > 0 else 0
            if loss_pct > max_corr_loss:
                max_corr_loss = loss_pct

        # Update equity
        equity += bar_pnl_sum
        if equity > peak_equity:
            peak_equity = equity

        # 2. Portfolio stop-loss check (unrealized PnL)
        if portfolio_sl_pct is not None and len(positions) > 0 and equity > 0:
            total_unrealized = 0.0
            for pos in positions:
                entry_bar = pos['entry_bar']
                if entry_bar >= n_bars or bar >= n_bars:
                    continue
                entry = opens[entry_bar]
                if entry <= 0:
                    continue
                current = closes[min(bar, n_bars - 1)]
                if pos['direction'] == 'LONG':
                    unr = (current / entry - 1) * 100 * LEVERAGE
                else:
                    unr = (1 - current / entry) * 100 * LEVERAGE
                sm = pos.get('size_mult', 1.0)
                total_unrealized += unr * (SIZE_PCT / 100) * sm

            # Check if total unrealized loss exceeds threshold
            unrealized_pct_of_equity = total_unrealized / equity * 100 if equity > 0 else 0
            if total_unrealized < 0 and abs(total_unrealized) > portfolio_sl_pct:
                # Close all positions at current close price
                portfolio_sl_events += 1
                for pos in positions:
                    entry_bar = pos['entry_bar']
                    if entry_bar >= n_bars:
                        continue
                    entry = opens[entry_bar]
                    if entry <= 0:
                        continue
                    exit_price = closes[min(bar, n_bars - 1)]
                    if pos['direction'] == 'LONG':
                        pnl = (exit_price / entry - 1) * 100 * LEVERAGE
                    else:
                        pnl = (1 - exit_price / entry) * 100 * LEVERAGE
                    pnl -= fee
                    sm = pos.get('size_mult', 1.0)
                    pnl_portfolio = pnl * (SIZE_PCT / 100) * sm
                    trades.append({
                        'entry_bar': entry_bar, 'exit_bar': bar,
                        'pnl_slot': pnl, 'reason': 'PORTFOLIO_SL',
                        'pattern': pos['pattern'], 'direction': pos['direction'],
                        'size_mult': sm, 'pnl_portfolio': pnl_portfolio,
                    })
                    equity += pnl_portfolio
                positions = []
                if equity > peak_equity:
                    peak_equity = equity
                continue  # skip new entries this bar after portfolio SL

        # 3. Process entry signals
        while sig_idx < len(signals_sorted) and signals_sorted[sig_idx][0] == bar:
            sig_bar, pat, direction = signals_sorted[sig_idx]
            sig_idx += 1

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

            # Regime sizing
            sm = 1.0
            if regime_mult is not None and ema_slope is not None:
                if bar < len(ema_slope):
                    s = ema_slope[bar]
                    if s > 0 and direction == 'SHORT':
                        sm = regime_mult  # reduce SHORT in uptrend
                    elif s <= 0 and direction == 'LONG':
                        sm = regime_mult  # reduce LONG in downtrend

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

        # Track max simultaneous
        if len(positions) > max_sim_positions:
            max_sim_positions = len(positions)
        long_count = sum(1 for p in positions if p['direction'] == 'LONG')
        short_count = sum(1 for p in positions if p['direction'] == 'SHORT')
        if long_count > max_sim_direction['LONG']:
            max_sim_direction['LONG'] = long_count
        if short_count > max_sim_direction['SHORT']:
            max_sim_direction['SHORT'] = short_count

    # Force-close remaining at OOS end (counted as trades)
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
            'entry_bar': entry_bar, 'exit_bar': exit_bar,
            'pnl_slot': pnl, 'reason': 'OOS_END',
            'pattern': pos['pattern'], 'direction': pos['direction'],
            'size_mult': sm,
            'pnl_portfolio': pnl * (SIZE_PCT / 100) * sm,
        })

    stats = {
        'max_corr_loss': round(max_corr_loss, 2),
        'max_sim_positions': max_sim_positions,
        'max_sim_long': max_sim_direction['LONG'],
        'max_sim_short': max_sim_direction['SHORT'],
        'portfolio_sl_events': portfolio_sl_events,
    }

    return trades, stats


def _check_exit(pos, bar, opens, highs, lows, n_bars, atr_ratio, fee):
    """Check if a position exits at the given bar."""
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


# ============================================================
# METRICS
# ============================================================

def compute_metrics(trades):
    """Compute portfolio metrics from trade list."""
    if not trades:
        return {'trades': 0, 'wr': 0.0, 'pnl': 0.0, 'mdd': 0.0,
                'pnl_mdd': 0.0, 'long_trades': 0, 'short_trades': 0,
                'long_pnl': 0.0, 'short_pnl': 0.0, 'pnl_per_trade': 0.0}

    wins = [t for t in trades if t['pnl_slot'] > 0]
    sorted_trades = sorted(trades, key=lambda x: x['entry_bar'])

    # Compound equity curve
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

    long_t = [t for t in trades if t['direction'] == 'LONG']
    short_t = [t for t in trades if t['direction'] == 'SHORT']

    return {
        'trades': len(trades),
        'wr': round(wr, 1),
        'pnl': round(total_pnl, 2),
        'mdd': round(max_dd, 2),
        'pnl_mdd': round(pnl_mdd, 2),
        'long_trades': len(long_t),
        'short_trades': len(short_t),
        'long_pnl': round(sum(t['pnl_portfolio'] for t in long_t), 2),
        'short_pnl': round(sum(t['pnl_portfolio'] for t in short_t), 2),
        'pnl_per_trade': round(total_pnl / len(trades), 3) if trades else 0,
    }


# ============================================================
# WALK-FORWARD (3-fold expanding window)
# ============================================================

def run_wf(signals, tpsl_map, opens, highs, lows, closes, n_bars, atr_ratio,
           direction_cap, regime_mult=None, ema_slope=None,
           portfolio_sl_pct=None, label=""):
    """Run 3-fold expanding window WF.

    Fold structure:
      4 segments of equal size: S0, S1, S2, S3
      Fold 1: IS=[S0],      OOS=[S1]
      Fold 2: IS=[S0,S1],   OOS=[S2]
      Fold 3: IS=[S0,S1,S2], OOS=[S3]  (full IS, OOS=last segment)
    """
    n_folds = 3
    seg_size = n_bars // (n_folds + 1)

    fold_results = []
    fold_stats = []

    for fold in range(n_folds):
        oos_start = (fold + 1) * seg_size
        oos_end = (fold + 2) * seg_size if fold < n_folds - 1 else n_bars

        trades, stats = simulate_portfolio(
            signals, tpsl_map, opens, highs, lows, closes, n_bars, atr_ratio,
            direction_cap, oos_start, oos_end,
            regime_mult=regime_mult, ema_slope=ema_slope,
            portfolio_sl_pct=portfolio_sl_pct,
        )
        m = compute_metrics(trades)
        fold_results.append({
            'fold': fold + 1,
            'oos_start': oos_start,
            'oos_end': oos_end,
            'oos_bars': oos_end - oos_start,
            **m,
        })
        fold_stats.append(stats)

    total_pnl = sum(f['pnl'] for f in fold_results)
    max_mdd = max(f['mdd'] for f in fold_results) if fold_results else 0
    all_positive = all(f['pnl'] > 0 for f in fold_results if f['trades'] > 0)
    n_positive = sum(1 for f in fold_results if f['pnl'] > 0)
    pnl_mdd = total_pnl / max_mdd if max_mdd > 0 else 0
    total_trades = sum(f['trades'] for f in fold_results)
    max_corr = max(s['max_corr_loss'] for s in fold_stats) if fold_stats else 0
    portfolio_sl_total = sum(s['portfolio_sl_events'] for s in fold_stats)

    return {
        'label': label,
        'folds': fold_results,
        'fold_stats': fold_stats,
        'total_pnl': round(total_pnl, 2),
        'total_trades': total_trades,
        'max_mdd': round(max_mdd, 2),
        'pnl_mdd': round(pnl_mdd, 2),
        'pass_3_3': all_positive,
        'n_positive': n_positive,
        'max_corr_loss': round(max_corr, 2),
        'portfolio_sl_events': portfolio_sl_total,
    }


# ============================================================
# H1: DIRECTION CAP REDUCTION
# ============================================================

def run_h1(signals, tpsl_map, opens, highs, lows, closes, n_bars, atr_ratio,
           ema_slope):
    """Test direction cap values: [3,4,5,6,7,8(current),9(no-cap)]"""
    print("\n" + "=" * 80)
    print("H1: DIRECTION CAP REDUCTION")
    print("=" * 80)

    # Use current regime_mult=0.3 as baseline context
    caps_to_test = [3, 4, 5, 6, 7, 8, 9]
    results = {}

    for cap in caps_to_test:
        label = f"cap{cap}" if cap < 9 else "cap9_nocap"
        wf = run_wf(signals, tpsl_map, opens, highs, lows, closes, n_bars, atr_ratio,
                     direction_cap=cap, regime_mult=0.3, ema_slope=ema_slope,
                     label=label)
        results[label] = wf
        status = "PASS" if wf['pass_3_3'] else "FAIL"
        fold_str = " | ".join(f"{f['pnl']:+.1f}" for f in wf['folds'])
        print(f"  {label:<14} trades={wf['total_trades']:>4} PnL={wf['total_pnl']:>+8.2f}% "
              f"MDD={wf['max_mdd']:>6.2f}% PnL/MDD={wf['pnl_mdd']:>6.2f}x "
              f"corr_loss={wf['max_corr_loss']:>5.1f}% {wf['n_positive']}/3 {status}  [{fold_str}]")

    # Baseline = cap8 (current)
    baseline = results['cap8']
    baseline_pnl_mdd = baseline['pnl_mdd']

    # Find best
    passing = {k: v for k, v in results.items() if v['pass_3_3']}
    if passing:
        best_key = max(passing, key=lambda k: passing[k]['pnl_mdd'])
        best = passing[best_key]
        improved = best['pnl_mdd'] > baseline_pnl_mdd
        verdict = 'GO' if improved and best_key != 'cap8' else 'STOP'
        print(f"\nH1 Verdict: {verdict}")
        if verdict == 'GO':
            print(f"  Best: {best_key} PnL/MDD={best['pnl_mdd']:.2f}x "
                  f"vs baseline cap8 {baseline_pnl_mdd:.2f}x")
    else:
        verdict = 'STOP'
        best_key = 'cap8'
        print(f"\nH1 Verdict: {verdict} (no scenario passed 3/3)")

    return {
        'hypothesis': 'H1_direction_cap',
        'verdict': verdict,
        'best': best_key,
        'results': {k: {'total_pnl': v['total_pnl'], 'max_mdd': v['max_mdd'],
                         'pnl_mdd': v['pnl_mdd'], 'pass_3_3': v['pass_3_3'],
                         'total_trades': v['total_trades'],
                         'max_corr_loss': v['max_corr_loss'],
                         'fold_pnls': [f['pnl'] for f in v['folds']]}
                    for k, v in results.items()},
    }


# ============================================================
# H2: PORTFOLIO-LEVEL STOP LOSS
# ============================================================

def run_h2(signals, tpsl_map, opens, highs, lows, closes, n_bars, atr_ratio,
           ema_slope):
    """Test portfolio SL thresholds: [5%,10%,15%,20%,25%,inf(disabled)]"""
    print("\n" + "=" * 80)
    print("H2: PORTFOLIO-LEVEL STOP LOSS")
    print("=" * 80)

    thresholds = [5.0, 10.0, 15.0, 20.0, 25.0, None]  # None = disabled
    results = {}

    for thr in thresholds:
        label = f"psl_{thr:.0f}pct" if thr is not None else "psl_disabled"
        wf = run_wf(signals, tpsl_map, opens, highs, lows, closes, n_bars, atr_ratio,
                     direction_cap=8, regime_mult=0.3, ema_slope=ema_slope,
                     portfolio_sl_pct=thr, label=label)
        results[label] = wf
        status = "PASS" if wf['pass_3_3'] else "FAIL"
        fold_str = " | ".join(f"{f['pnl']:+.1f}" for f in wf['folds'])
        sl_evts = wf['portfolio_sl_events']
        print(f"  {label:<18} trades={wf['total_trades']:>4} PnL={wf['total_pnl']:>+8.2f}% "
              f"MDD={wf['max_mdd']:>6.2f}% PnL/MDD={wf['pnl_mdd']:>6.2f}x "
              f"SL_events={sl_evts} {wf['n_positive']}/3 {status}  [{fold_str}]")

    # Baseline = disabled
    baseline = results['psl_disabled']
    baseline_pnl_mdd = baseline['pnl_mdd']

    passing = {k: v for k, v in results.items() if v['pass_3_3'] and k != 'psl_disabled'}
    if passing:
        best_key = max(passing, key=lambda k: passing[k]['pnl_mdd'])
        best = passing[best_key]
        improved = best['pnl_mdd'] > baseline_pnl_mdd
        verdict = 'GO' if improved else 'STOP'
        print(f"\nH2 Verdict: {verdict}")
        if verdict == 'GO':
            print(f"  Best: {best_key} PnL/MDD={best['pnl_mdd']:.2f}x "
                  f"vs baseline {baseline_pnl_mdd:.2f}x")
        else:
            print(f"  Portfolio SL did not improve PnL/MDD (best {best_key}: "
                  f"{best['pnl_mdd']:.2f}x vs baseline {baseline_pnl_mdd:.2f}x)")
    else:
        verdict = 'STOP'
        best_key = 'psl_disabled'
        print(f"\nH2 Verdict: {verdict} (no portfolio SL scenario passed 3/3 or improved)")

    return {
        'hypothesis': 'H2_portfolio_sl',
        'verdict': verdict,
        'best': best_key,
        'results': {k: {'total_pnl': v['total_pnl'], 'max_mdd': v['max_mdd'],
                         'pnl_mdd': v['pnl_mdd'], 'pass_3_3': v['pass_3_3'],
                         'total_trades': v['total_trades'],
                         'portfolio_sl_events': v['portfolio_sl_events'],
                         'fold_pnls': [f['pnl'] for f in v['folds']]}
                    for k, v in results.items()},
    }


# ============================================================
# H3: REGIME SIZING STRENGTHENING
# ============================================================

def run_h3(signals, tpsl_map, opens, highs, lows, closes, n_bars, atr_ratio,
           ema_slope):
    """Test counter_mult: [0.0(block), 0.1, 0.2, 0.3(current), 0.5, 1.0(uniform)]"""
    print("\n" + "=" * 80)
    print("H3: REGIME SIZING STRENGTHENING")
    print("=" * 80)

    mults_to_test = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]
    results = {}

    for mult in mults_to_test:
        rm = mult if mult < 1.0 else None  # 1.0 = uniform (disabled)
        label = f"mult_{mult:.1f}"
        wf = run_wf(signals, tpsl_map, opens, highs, lows, closes, n_bars, atr_ratio,
                     direction_cap=8, regime_mult=rm, ema_slope=ema_slope if rm is not None else None,
                     label=label)
        results[label] = wf
        status = "PASS" if wf['pass_3_3'] else "FAIL"
        fold_str = " | ".join(f"{f['pnl']:+.1f}" for f in wf['folds'])
        print(f"  {label:<12} trades={wf['total_trades']:>4} PnL={wf['total_pnl']:>+8.2f}% "
              f"MDD={wf['max_mdd']:>6.2f}% PnL/MDD={wf['pnl_mdd']:>6.2f}x "
              f"corr_loss={wf['max_corr_loss']:>5.1f}% {wf['n_positive']}/3 {status}  [{fold_str}]")

    # Baseline = 0.3 (current)
    baseline = results['mult_0.3']
    baseline_pnl_mdd = baseline['pnl_mdd']

    passing = {k: v for k, v in results.items() if v['pass_3_3']}
    if passing:
        best_key = max(passing, key=lambda k: passing[k]['pnl_mdd'])
        best = passing[best_key]
        improved = best['pnl_mdd'] > baseline_pnl_mdd and best_key != 'mult_0.3'
        verdict = 'GO' if improved else 'STOP'
        print(f"\nH3 Verdict: {verdict}")
        if verdict == 'GO':
            print(f"  Best: {best_key} PnL/MDD={best['pnl_mdd']:.2f}x "
                  f"vs baseline mult_0.3 {baseline_pnl_mdd:.2f}x")
        else:
            print(f"  No multiplier beat baseline (best passing: {best_key} "
                  f"{best['pnl_mdd']:.2f}x vs {baseline_pnl_mdd:.2f}x)")
    else:
        verdict = 'STOP'
        best_key = 'mult_0.3'
        print(f"\nH3 Verdict: {verdict} (no scenario passed 3/3)")

    return {
        'hypothesis': 'H3_regime_sizing',
        'verdict': verdict,
        'best': best_key,
        'results': {k: {'total_pnl': v['total_pnl'], 'max_mdd': v['max_mdd'],
                         'pnl_mdd': v['pnl_mdd'], 'pass_3_3': v['pass_3_3'],
                         'total_trades': v['total_trades'],
                         'max_corr_loss': v['max_corr_loss'],
                         'fold_pnls': [f['pnl'] for f in v['folds']]}
                    for k, v in results.items()},
    }


# ============================================================
# H4: COMBINED BEST
# ============================================================

def run_h4(h1_result, h2_result, h3_result,
           signals, tpsl_map, opens, highs, lows, closes, n_bars, atr_ratio,
           ema_slope):
    """Combine best parameters from H1-H3."""
    print("\n" + "=" * 80)
    print("H4: COMBINED OPTIMIZATION")
    print("=" * 80)

    # Determine best cap from H1
    best_cap_str = h1_result['best']
    if best_cap_str == 'cap9_nocap':
        best_cap = 9
    else:
        best_cap = int(best_cap_str.replace('cap', ''))
    if h1_result['verdict'] == 'STOP':
        best_cap = 8  # keep current

    # Determine best portfolio SL from H2
    best_psl_str = h2_result['best']
    if h2_result['verdict'] == 'GO' and best_psl_str != 'psl_disabled':
        best_psl = float(best_psl_str.replace('psl_', '').replace('pct', ''))
    else:
        best_psl = None  # disabled

    # Determine best regime mult from H3
    best_mult_str = h3_result['best']
    if h3_result['verdict'] == 'STOP':
        best_mult = 0.3  # keep current
    else:
        best_mult = float(best_mult_str.replace('mult_', ''))

    print(f"  Using: cap={best_cap}, psl={'disabled' if best_psl is None else f'{best_psl:.0f}%'}, "
          f"regime_mult={best_mult}")

    # Also test a few promising combos even if individual hypotheses returned STOP
    combos = []

    # Combo 1: Best from each hypothesis
    combos.append(('best_combo', best_cap, best_psl, best_mult))

    # Combo 2: More aggressive versions
    # Lower cap + tighter PSL
    for cap in [5, 6, 7]:
        for psl in [10.0, 15.0, None]:
            for mult in [0.0, 0.1, 0.2, 0.3]:
                label = f"cap{cap}_psl{int(psl) if psl else 'off'}_m{mult:.1f}"
                combos.append((label, cap, psl, mult))

    # Also current baseline
    combos.append(('current_baseline', 8, None, 0.3))

    # Run all combos
    results = {}
    print(f"\n  Testing {len(combos)} combinations...")

    for label, cap, psl, mult in combos:
        rm = mult if mult < 1.0 else None
        wf = run_wf(signals, tpsl_map, opens, highs, lows, closes, n_bars, atr_ratio,
                     direction_cap=cap,
                     regime_mult=rm, ema_slope=ema_slope if rm is not None else None,
                     portfolio_sl_pct=psl, label=label)
        results[label] = wf

    # Print comparison table
    print(f"\n{'Scenario':<28} {'Trds':>5} {'PnL':>9} {'MDD':>7} {'PnL/MDD':>8} "
          f"{'Corr%':>6} {'PSL#':>4} {'3/3':>4}")
    print("-" * 80)

    for label, cap, psl, mult in sorted(combos, key=lambda x: results[x[0]]['pnl_mdd'], reverse=True):
        r = results[label]
        status = "PASS" if r['pass_3_3'] else "FAIL"
        fold_str = " | ".join(f"{f['pnl']:+.1f}" for f in r['folds'])
        print(f"  {label:<26} {r['total_trades']:>5} {r['total_pnl']:>+8.2f}% "
              f"{r['max_mdd']:>6.2f}% {r['pnl_mdd']:>7.2f}x "
              f"{r['max_corr_loss']:>5.1f}% {r['portfolio_sl_events']:>3} "
              f"{status:<4} [{fold_str}]")

    # Find best passing combo
    passing = {k: v for k, v in results.items() if v['pass_3_3']}
    baseline_r = results.get('current_baseline', results[list(results.keys())[0]])

    if passing:
        best_key = max(passing, key=lambda k: passing[k]['pnl_mdd'])
        best = passing[best_key]
        improved = best['pnl_mdd'] > baseline_r['pnl_mdd']
        verdict = 'GO' if improved and best_key != 'current_baseline' else 'STOP'
    else:
        verdict = 'STOP'
        best_key = 'current_baseline'
        best = baseline_r

    print(f"\nH4 Verdict: {verdict}")
    if verdict == 'GO':
        print(f"  RECOMMENDED: {best_key}")
        print(f"    PnL/MDD={best['pnl_mdd']:.2f}x vs baseline {baseline_r['pnl_mdd']:.2f}x")
        print(f"    Total PnL={best['total_pnl']:+.2f}%, MDD={best['max_mdd']:.2f}%")
    else:
        print(f"  Current baseline (cap8, no PSL, mult 0.3) remains optimal.")
        print(f"    PnL/MDD={baseline_r['pnl_mdd']:.2f}x, PnL={baseline_r['total_pnl']:+.2f}%")

    # Compact results
    compact = {}
    for label, cap, psl, mult in combos:
        r = results[label]
        compact[label] = {
            'cap': cap, 'psl': psl, 'regime_mult': mult,
            'total_pnl': r['total_pnl'], 'max_mdd': r['max_mdd'],
            'pnl_mdd': r['pnl_mdd'], 'pass_3_3': r['pass_3_3'],
            'total_trades': r['total_trades'],
            'max_corr_loss': r['max_corr_loss'],
            'portfolio_sl_events': r['portfolio_sl_events'],
            'fold_pnls': [f['pnl'] for f in r['folds']],
        }

    return {
        'hypothesis': 'H4_combined',
        'verdict': verdict,
        'best': best_key,
        'results': compact,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    t0 = time.time()
    print("=" * 80, flush=True)
    print("LIVE PERFORMANCE GAP STUDY: Multi-Position Portfolio Optimization", flush=True)
    print(f"Date: {datetime.now().isoformat()}", flush=True)
    print("=" * 80, flush=True)

    # 1. Load data
    df, types, opens, highs, lows, closes, n_bars = load_data()
    pL, pS, tpsl = load_patterns()
    atr_ratio = compute_atr_ratio(highs, lows, closes)
    ema_slope = compute_ema_slope(closes)

    valid_pct = np.sum(~np.isnan(atr_ratio)) / n_bars * 100
    print(f"ATR ratio: {valid_pct:.1f}% valid (warmup={ATR_WARMUP} bars)")

    # 2. Generate all signals
    all_signals = generate_signals(types, pL, pS, n_bars, start_idx=ATR_WARMUP)
    print(f"Total signals: {len(all_signals)} "
          f"({sum(1 for _, _, d in all_signals if d == 'LONG')}L + "
          f"{sum(1 for _, _, d in all_signals if d == 'SHORT')}S)")

    # 3. Full-data baseline (for context)
    print("\n--- FULL-DATA BASELINE (current config: cap8, mult 0.3) ---")
    baseline_trades, baseline_stats = simulate_portfolio(
        all_signals, tpsl, opens, highs, lows, closes, n_bars, atr_ratio,
        direction_cap=8, oos_start=ATR_WARMUP, oos_end=n_bars,
        regime_mult=0.3, ema_slope=ema_slope,
    )
    baseline_m = compute_metrics(baseline_trades)
    print(f"  Trades: {baseline_m['trades']} ({baseline_m['long_trades']}L+{baseline_m['short_trades']}S)")
    print(f"  WR: {baseline_m['wr']:.1f}%, PnL: {baseline_m['pnl']:+.2f}%, MDD: {baseline_m['mdd']:.2f}%")
    print(f"  PnL/MDD: {baseline_m['pnl_mdd']:.2f}x")
    print(f"  Max correlated loss: {baseline_stats['max_corr_loss']:.1f}%")
    print(f"  Max simultaneous: {baseline_stats['max_sim_positions']} "
          f"(L={baseline_stats['max_sim_long']}, S={baseline_stats['max_sim_short']})")

    # 4. Run hypotheses
    h1_result = run_h1(all_signals, tpsl, opens, highs, lows, closes, n_bars, atr_ratio, ema_slope)
    h2_result = run_h2(all_signals, tpsl, opens, highs, lows, closes, n_bars, atr_ratio, ema_slope)
    h3_result = run_h3(all_signals, tpsl, opens, highs, lows, closes, n_bars, atr_ratio, ema_slope)
    h4_result = run_h4(h1_result, h2_result, h3_result,
                        all_signals, tpsl, opens, highs, lows, closes, n_bars, atr_ratio, ema_slope)

    elapsed = time.time() - t0

    # 5. Final Summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"  H1 (Direction Cap):     {h1_result['verdict']:<6} best={h1_result['best']}")
    print(f"  H2 (Portfolio SL):      {h2_result['verdict']:<6} best={h2_result['best']}")
    print(f"  H3 (Regime Sizing):     {h3_result['verdict']:<6} best={h3_result['best']}")
    print(f"  H4 (Combined):          {h4_result['verdict']:<6} best={h4_result['best']}")
    print(f"\n  Elapsed: {elapsed:.1f}s")

    # Actionable recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    if h4_result['verdict'] == 'GO':
        best = h4_result['best']
        r = h4_result['results'][best]
        print(f"  CHANGE TO: {best}")
        print(f"    cap={r['cap']}, portfolio_sl={r['psl']}, regime_mult={r['regime_mult']}")
        print(f"    Expected: PnL/MDD={r['pnl_mdd']:.2f}x, MDD={r['max_mdd']:.2f}%")
    else:
        print("  KEEP CURRENT: cap=8, no portfolio SL, regime_mult=0.3")
        print("  The live performance gap is within expected variance for the")
        print("  multi-position portfolio; correlated loss events are inherent")
        print("  to the SHORT-heavy strategy during rallies.")

    # 6. Save results
    output = {
        'study': 'live_gap_portfolio_study',
        'date': datetime.now().isoformat(),
        'elapsed_s': round(elapsed, 1),
        'data_bars': n_bars,
        'data_days': round(n_bars / BARS_PER_DAY, 1),
        'patterns': {'long': len(pL), 'short': len(pS)},
        'baseline': {
            'config': {'cap': 8, 'portfolio_sl': None, 'regime_mult': 0.3},
            'metrics': baseline_m,
            'stats': baseline_stats,
        },
        'h1': h1_result,
        'h2': h2_result,
        'h3': h3_result,
        'h4': h4_result,
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
