#!/usr/bin/env python3
"""
Entry Improvement Study: Reducing Correlated SHORT Losses During Uptrends
=========================================================================
Root cause: 2/25 BTC +6.1% rally -> 9 SHORT entries in ~10h -> 8 SL hits -> -69.88% slot PnL.
Current protections (cap7, regime×0.3, agg risk cap 3/7) were NOT active during event.
Question: With protections active, do we need MORE?

3 Hypotheses:
  H1: Entry Spacing — min time gap between same-direction entries [15,30,60,120 min]
  H2: Momentum Guard — pause counter-dir entries during strong price moves
      Lookback [6,12,24 bars] × Threshold [1,2,3%] × Cooldown [6,12 bars]
  H3: Progressive Direction Cap — tighter cap during strong regimes
      Mild trend→cap7, Strong→cap5, Extreme→cap3

Standard Research Protocol:
  Entry: signal bar + 1 open
  Exit: Intrabar high/low (distance-based, abs(tp - opens[bar]))
  Fee: 0.10% * LEVERAGE(3) = 0.30% per trade (capital-space)
  Slippage: 0.02%
  ATR-scaled TP/SL: ATR(14)/median(576), clamp [0.6, 1.7]
  Regime sizing: EMA(20) slope, counter_mult=0.3
  Aggregate risk cap: counter 3%, with 7%
  Direction cap: 7 (baseline)
  WF: 3-fold expanding window, PASS = all folds OOS PnL > 0
"""

import json
import os
import sys
import time
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from scripts.production.pattern_5m.indicators import classify_candle

# ============================================================
# CONSTANTS
# ============================================================
LEVERAGE = 3
FEE_PCT = 0.10
SLIPPAGE_BUFFER = 0.02
TIMEOUT_BARS = 864
MAX_POSITIONS = 9
ATR_PERIOD = 14
ATR_WINDOW = 576
CLAMP_LO = 0.6
CLAMP_HI = 1.7
ATR_WARMUP = ATR_PERIOD + ATR_WINDOW
BARS_PER_DAY = 288
EMA_PERIOD = 20
EMA_LOOKBACK = 5

DATA_FILE = os.path.join(PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
PATTERNS_FILE = os.path.join(PROJECT_ROOT, 'results', 'dynamic_patterns.json')
OUTPUT_FILE = os.path.join(PROJECT_ROOT, 'results', 'entry_improvement_study.json')


# ============================================================
# CORE DATA LOADING (reuse from live_gap_portfolio_study)
# ============================================================

def load_data():
    df = pd.read_csv(DATA_FILE)
    # Use shifted 270d window (last 270d of 297d data)
    n_total = len(df)
    target_bars = 270 * BARS_PER_DAY  # 77,760
    if n_total > target_bars:
        start = n_total - target_bars
        df = df.iloc[start:].reset_index(drop=True)
        print(f"Using shifted 270d window: last {len(df)} of {n_total} bars")
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
    n = len(closes)
    ema = pd.Series(closes).ewm(span=ema_period, adjust=False).mean().values
    slope = np.full(n, 0.0)
    for i in range(lookback, n):
        if ema[i - lookback] > 0:
            slope[i] = (ema[i] / ema[i - lookback] - 1) * 100
    return slope


def generate_signals(types, patterns_long, patterns_short, n_bars, start_idx=0):
    signals = []
    for i in range(max(2, start_idx), n_bars):
        pat = f"{types[i - 2]}-{types[i - 1]}-{types[i]}"
        if pat in patterns_long:
            signals.append((i, pat, 'LONG'))
        if pat in patterns_short:
            signals.append((i, pat, 'SHORT'))
    return signals


# ============================================================
# CHECK EXIT
# ============================================================

def _check_exit(pos, bar, opens, highs, lows, n_bars, atr_ratio, fee):
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
# ENHANCED PORTFOLIO SIMULATOR
# ============================================================

def simulate_portfolio(signals, tpsl_map, opens, highs, lows, closes, n_bars,
                       atr_ratio, ema_slope, oos_start, oos_end,
                       # Baseline params
                       direction_cap=7, regime_mult=0.3,
                       agg_risk_counter=3.0, agg_risk_with=7.0,
                       # H1: Entry Spacing
                       entry_spacing_bars=0,
                       # H2: Momentum Guard
                       momentum_lookback=0, momentum_threshold=0.0,
                       momentum_cooldown=0,
                       # H3: Progressive Direction Cap
                       progressive_cap=None,  # dict: {'mild': 7, 'strong': 5, 'extreme': 3}
                       ):
    """Full multi-position portfolio simulator with all protections + hypotheses."""
    SIZE_PCT = 100.0 / MAX_POSITIONS
    fee = FEE_PCT * LEVERAGE

    positions = []
    trades = []
    equity = 100.0
    peak_equity = 100.0

    # Tracking
    max_corr_loss = 0.0
    max_sim_positions = 0
    total_blocked = {'spacing': 0, 'momentum': 0, 'agg_risk': 0, 'dir_cap': 0}
    corr_events = []  # (bar, n_sl_hits, loss_pct)

    # H1: Last entry time per direction
    last_entry_bar = {'LONG': -999, 'SHORT': -999}

    # H2: Momentum cooldown tracker
    momentum_pause_until = {'LONG': -1, 'SHORT': -1}

    signals_in_range = [(s, p, d) for s, p, d in signals if oos_start <= s < oos_end]
    signals_sorted = sorted(signals_in_range, key=lambda x: x[0])
    sig_idx = 0

    for bar in range(oos_start, oos_end):
        # 1. Check exits
        closed_slots = []
        bar_pnl_sum = 0.0
        bar_sl_count = 0

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
                    bar_sl_count += 1

        positions = [p for p in positions if p['slot'] not in closed_slots]

        # Track correlated loss events
        if bar_pnl_sum < 0 and bar_sl_count >= 2:
            loss_pct = abs(bar_pnl_sum)
            if loss_pct > max_corr_loss:
                max_corr_loss = loss_pct
            corr_events.append((bar, bar_sl_count, loss_pct))

        equity += bar_pnl_sum
        if equity > peak_equity:
            peak_equity = equity

        # H2: Check momentum for cooldown trigger
        if momentum_lookback > 0 and momentum_threshold > 0 and bar >= momentum_lookback:
            price_now = closes[bar]
            price_ago = closes[bar - momentum_lookback]
            if price_ago > 0:
                pct_change = (price_now / price_ago - 1) * 100
                # Strong upward move → pause SHORT
                if pct_change > momentum_threshold:
                    momentum_pause_until['SHORT'] = bar + momentum_cooldown
                # Strong downward move → pause LONG
                elif pct_change < -momentum_threshold:
                    momentum_pause_until['LONG'] = bar + momentum_cooldown

        # 2. Process entries
        while sig_idx < len(signals_sorted) and signals_sorted[sig_idx][0] == bar:
            sig_bar, pat, direction = signals_sorted[sig_idx]
            sig_idx += 1

            if len(positions) >= MAX_POSITIONS:
                continue

            # Direction cap (baseline or progressive)
            dir_count = sum(1 for p in positions if p['direction'] == direction)
            if progressive_cap is not None:
                # Determine regime strength
                slope_val = abs(ema_slope[bar]) if bar < len(ema_slope) else 0
                if slope_val > 0.5:     # extreme trend
                    cap = progressive_cap.get('extreme', 3)
                elif slope_val > 0.2:   # strong trend
                    cap = progressive_cap.get('strong', 5)
                else:
                    cap = progressive_cap.get('mild', 7)
            else:
                cap = direction_cap

            if dir_count >= cap:
                total_blocked['dir_cap'] += 1
                continue

            # Duplicate pattern check
            if any(p['pattern'] == pat for p in positions):
                continue

            entry_bar = sig_bar + 1
            if entry_bar >= n_bars:
                continue
            tp_sl = tpsl_map.get(pat)
            if tp_sl is None:
                continue

            # H1: Entry spacing check
            if entry_spacing_bars > 0:
                if (bar - last_entry_bar[direction]) < entry_spacing_bars:
                    total_blocked['spacing'] += 1
                    continue

            # H2: Momentum guard check
            if momentum_lookback > 0 and bar < momentum_pause_until[direction]:
                total_blocked['momentum'] += 1
                continue

            # Regime sizing
            sm = 1.0
            if regime_mult is not None and bar < len(ema_slope):
                s = ema_slope[bar]
                if s > 0 and direction == 'SHORT':
                    sm = regime_mult
                elif s <= 0 and direction == 'LONG':
                    sm = regime_mult

            # Aggregate risk cap check
            if agg_risk_counter > 0 or agg_risk_with > 0:
                is_uptrend = ema_slope[bar] > 0 if bar < len(ema_slope) else False
                is_counter = (is_uptrend and direction == 'SHORT') or (not is_uptrend and direction == 'LONG')
                cap_pct = agg_risk_counter if is_counter else agg_risk_with

                # Sum existing SL exposure for this direction
                existing_exposure = 0.0
                for p in positions:
                    if p['direction'] == direction:
                        p_sl = p['sl_pct']
                        p_sig = p['signal_bar']
                        if atr_ratio is not None and p_sig < len(atr_ratio) and not np.isnan(atr_ratio[p_sig]):
                            p_r = max(CLAMP_LO, min(CLAMP_HI, atr_ratio[p_sig]))
                        else:
                            p_r = 1.0
                        p_eff_sl = p_sl * p_r
                        p_sm = p.get('size_mult', 1.0)
                        existing_exposure += p_eff_sl * (1.0 / MAX_POSITIONS) * LEVERAGE * p_sm

                # New position's contribution
                new_r = 1.0
                if atr_ratio is not None and sig_bar < len(atr_ratio) and not np.isnan(atr_ratio[sig_bar]):
                    new_r = max(CLAMP_LO, min(CLAMP_HI, atr_ratio[sig_bar]))
                new_eff_sl = tp_sl[1] * new_r
                new_exposure = new_eff_sl * (1.0 / MAX_POSITIONS) * LEVERAGE * sm

                if existing_exposure + new_exposure > cap_pct:
                    total_blocked['agg_risk'] += 1
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
            last_entry_bar[direction] = bar

        # Track max simultaneous
        if len(positions) > max_sim_positions:
            max_sim_positions = len(positions)

    # Force-close remaining
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
            'pnl_portfolio': pnl * (SIZE_PCT / 100) * sm,
        })

    stats = {
        'max_corr_loss': round(max_corr_loss, 2),
        'max_sim_positions': max_sim_positions,
        'corr_events': len(corr_events),
        'blocked': total_blocked,
    }
    return trades, stats


# ============================================================
# METRICS
# ============================================================

def compute_metrics(trades):
    if not trades:
        return {'trades': 0, 'wr': 0.0, 'pnl': 0.0, 'mdd': 0.0,
                'pnl_mdd': 0.0, 'pnl_per_trade': 0.0}

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

    total_pnl = equity - 100.0
    wr = len(wins) / len(trades) * 100 if trades else 0
    pnl_mdd = total_pnl / max_dd if max_dd > 0 else 0

    return {
        'trades': len(trades),
        'wr': round(wr, 1),
        'pnl': round(total_pnl, 2),
        'mdd': round(max_dd, 2),
        'pnl_mdd': round(pnl_mdd, 2),
        'pnl_per_trade': round(total_pnl / len(trades), 3) if trades else 0,
    }


# ============================================================
# WALK-FORWARD (3-fold expanding window)
# ============================================================

def run_wf(signals, tpsl_map, opens, highs, lows, closes, n_bars,
           atr_ratio, ema_slope, label="", **sim_kwargs):
    """3-fold expanding window WF."""
    n_folds = 3
    seg_size = n_bars // (n_folds + 1)

    fold_results = []
    fold_stats = []

    for fold in range(n_folds):
        oos_start = (fold + 1) * seg_size
        oos_end = (fold + 2) * seg_size if fold < n_folds - 1 else n_bars

        trades, stats = simulate_portfolio(
            signals, tpsl_map, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, oos_start, oos_end,
            **sim_kwargs,
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
    n_positive = sum(1 for f in fold_results if f['pnl'] > 0)
    pnl_mdd = total_pnl / max_mdd if max_mdd > 0 else 0
    total_trades = sum(f['trades'] for f in fold_results)
    max_corr = max(s['max_corr_loss'] for s in fold_stats) if fold_stats else 0
    total_blocked = {}
    for s in fold_stats:
        for k, v in s['blocked'].items():
            total_blocked[k] = total_blocked.get(k, 0) + v

    return {
        'label': label,
        'folds': fold_results,
        'total_pnl': round(total_pnl, 2),
        'max_mdd': round(max_mdd, 2),
        'pnl_mdd': round(pnl_mdd, 2),
        'total_trades': total_trades,
        'wf_pass': f"{n_positive}/{n_folds}",
        'wf_passed': n_positive == n_folds,
        'max_corr_loss': round(max_corr, 2),
        'total_blocked': total_blocked,
    }


# ============================================================
# FULL-SAMPLE TEST (for quick parameter screening)
# ============================================================

def run_full_sample(signals, tpsl_map, opens, highs, lows, closes, n_bars,
                    atr_ratio, ema_slope, label="", **sim_kwargs):
    """Run full-sample backtest (IS only, no WF)."""
    # Start after ATR warmup
    start = max(ATR_WARMUP, 2)
    trades, stats = simulate_portfolio(
        signals, tpsl_map, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, start, n_bars,
        **sim_kwargs,
    )
    m = compute_metrics(trades)
    return {
        'label': label,
        **m,
        **stats,
    }


# ============================================================
# MAIN STUDY
# ============================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("Entry Improvement Study: H1-H3")
    print("=" * 70)

    # Load data
    df, types, opens, highs, lows, closes, n_bars = load_data()
    pL, pS, tpsl = load_patterns()
    atr_ratio = compute_atr_ratio(highs, lows, closes)
    ema_slope = compute_ema_slope(closes)
    signals = generate_signals(types, pL, pS, n_bars)
    print(f"Total signals: {len(signals)}")
    print()

    results = {}

    # ========================================
    # BASELINE: Current v1.36.1 protections
    # ========================================
    print("=" * 60)
    print("BASELINE: Current v1.36.1 protections")
    print("  (cap7, regime×0.3, agg risk 3/7)")
    print("=" * 60)

    baseline_wf = run_wf(
        signals, tpsl, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, label="BASELINE",
        direction_cap=7, regime_mult=0.3,
        agg_risk_counter=3.0, agg_risk_with=7.0,
    )
    _print_wf(baseline_wf)
    results['baseline'] = baseline_wf

    # Also run baseline WITHOUT protections for comparison
    print("\n--- NO PROTECTIONS (raw cap9, no regime, no agg risk) ---")
    no_prot_wf = run_wf(
        signals, tpsl, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, label="NO_PROTECTIONS",
        direction_cap=9, regime_mult=None,
        agg_risk_counter=0, agg_risk_with=0,
    )
    _print_wf(no_prot_wf)
    results['no_protections'] = no_prot_wf

    # ========================================
    # H1: Entry Spacing
    # ========================================
    print("\n" + "=" * 60)
    print("H1: Entry Spacing (min bars between same-dir entries)")
    print("=" * 60)

    h1_results = []
    for spacing_min in [15, 30, 60, 120]:
        spacing_bars = spacing_min // 5  # convert to 5m bars: 3, 6, 12, 24
        label = f"H1_spacing_{spacing_min}min"
        r = run_wf(
            signals, tpsl, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, label=label,
            direction_cap=7, regime_mult=0.3,
            agg_risk_counter=3.0, agg_risk_with=7.0,
            entry_spacing_bars=spacing_bars,
        )
        _print_wf(r)
        h1_results.append(r)
    results['H1_entry_spacing'] = h1_results

    # ========================================
    # H2: Momentum Guard
    # ========================================
    print("\n" + "=" * 60)
    print("H2: Momentum Guard (pause counter-dir on strong moves)")
    print("=" * 60)

    h2_results = []
    # Test key combinations
    h2_configs = [
        # (lookback_bars, threshold_pct, cooldown_bars)
        (6, 1.0, 6),     # 30min lookback, 1% move, 30min pause
        (6, 2.0, 6),     # 30min lookback, 2% move, 30min pause
        (12, 1.0, 6),    # 1h lookback, 1% move, 30min pause
        (12, 1.0, 12),   # 1h lookback, 1% move, 1h pause
        (12, 2.0, 6),    # 1h lookback, 2% move, 30min pause
        (12, 2.0, 12),   # 1h lookback, 2% move, 1h pause
        (24, 2.0, 12),   # 2h lookback, 2% move, 1h pause
        (24, 3.0, 12),   # 2h lookback, 3% move, 1h pause
        (24, 3.0, 24),   # 2h lookback, 3% move, 2h pause
    ]
    for lb, thr, cd in h2_configs:
        label = f"H2_lb{lb*5}min_thr{thr}pct_cd{cd*5}min"
        r = run_wf(
            signals, tpsl, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, label=label,
            direction_cap=7, regime_mult=0.3,
            agg_risk_counter=3.0, agg_risk_with=7.0,
            momentum_lookback=lb, momentum_threshold=thr,
            momentum_cooldown=cd,
        )
        _print_wf(r)
        h2_results.append(r)
    results['H2_momentum_guard'] = h2_results

    # ========================================
    # H3: Progressive Direction Cap
    # ========================================
    print("\n" + "=" * 60)
    print("H3: Progressive Direction Cap (tighter in strong trends)")
    print("=" * 60)

    h3_results = []
    h3_configs = [
        {'mild': 7, 'strong': 5, 'extreme': 3},
        {'mild': 7, 'strong': 5, 'extreme': 4},
        {'mild': 7, 'strong': 6, 'extreme': 4},
        {'mild': 7, 'strong': 6, 'extreme': 5},
        {'mild': 6, 'strong': 4, 'extreme': 2},
    ]
    for cfg in h3_configs:
        label = f"H3_cap_{cfg['mild']}/{cfg['strong']}/{cfg['extreme']}"
        r = run_wf(
            signals, tpsl, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, label=label,
            regime_mult=0.3,
            agg_risk_counter=3.0, agg_risk_with=7.0,
            progressive_cap=cfg,
        )
        _print_wf(r)
        h3_results.append(r)
    results['H3_progressive_cap'] = h3_results

    # ========================================
    # PHASE 5: Best Combination
    # ========================================
    print("\n" + "=" * 60)
    print("PHASE 5: Best Combinations")
    print("=" * 60)

    # Find best from each hypothesis (WF PASS + best PnL/MDD)
    best_h1 = _find_best(h1_results)
    best_h2 = _find_best(h2_results)
    best_h3 = _find_best(h3_results)

    print(f"\nBest H1: {best_h1['label'] if best_h1 else 'NONE PASSED'}")
    print(f"Best H2: {best_h2['label'] if best_h2 else 'NONE PASSED'}")
    print(f"Best H3: {best_h3['label'] if best_h3 else 'NONE PASSED'}")

    combo_results = []

    # Test combinations of passing hypotheses
    if best_h1:
        # Extract spacing from label
        spacing = int(best_h1['label'].split('_')[2].replace('min', ''))
        spacing_bars = spacing // 5

        if best_h2:
            # H1 + H2
            # Extract H2 params from label
            h2_params = _parse_h2_label(best_h2['label'])
            label = f"COMBO_H1({spacing}m)+H2(lb{h2_params[0]}thr{h2_params[1]}cd{h2_params[2]})"
            r = run_wf(
                signals, tpsl, opens, highs, lows, closes, n_bars,
                atr_ratio, ema_slope, label=label,
                direction_cap=7, regime_mult=0.3,
                agg_risk_counter=3.0, agg_risk_with=7.0,
                entry_spacing_bars=spacing_bars,
                momentum_lookback=h2_params[0], momentum_threshold=h2_params[1],
                momentum_cooldown=h2_params[2],
            )
            _print_wf(r)
            combo_results.append(r)

    if best_h3:
        h3_cfg = _parse_h3_label(best_h3['label'])
        if best_h1:
            spacing = int(best_h1['label'].split('_')[2].replace('min', ''))
            spacing_bars = spacing // 5
            # H1 + H3
            label = f"COMBO_H1({spacing}m)+H3({h3_cfg['mild']}/{h3_cfg['strong']}/{h3_cfg['extreme']})"
            r = run_wf(
                signals, tpsl, opens, highs, lows, closes, n_bars,
                atr_ratio, ema_slope, label=label,
                regime_mult=0.3,
                agg_risk_counter=3.0, agg_risk_with=7.0,
                entry_spacing_bars=spacing_bars,
                progressive_cap=h3_cfg,
            )
            _print_wf(r)
            combo_results.append(r)

    if best_h1 and best_h2 and best_h3:
        # H1 + H2 + H3
        spacing = int(best_h1['label'].split('_')[2].replace('min', ''))
        spacing_bars = spacing // 5
        h2_params = _parse_h2_label(best_h2['label'])
        h3_cfg = _parse_h3_label(best_h3['label'])
        label = f"COMBO_ALL"
        r = run_wf(
            signals, tpsl, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, label=label,
            regime_mult=0.3,
            agg_risk_counter=3.0, agg_risk_with=7.0,
            entry_spacing_bars=spacing_bars,
            momentum_lookback=h2_params[0], momentum_threshold=h2_params[1],
            momentum_cooldown=h2_params[2],
            progressive_cap=h3_cfg,
        )
        _print_wf(r)
        combo_results.append(r)

    results['combinations'] = combo_results

    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_scenarios = [baseline_wf, no_prot_wf] + h1_results + h2_results + h3_results + combo_results
    print(f"\n{'Label':<55} {'WF':>5} {'PnL':>10} {'MDD':>8} {'P/M':>8} {'Trades':>7} {'CorrL':>7}")
    print("-" * 100)
    for s in all_scenarios:
        wf_mark = "✅" if s['wf_passed'] else "❌"
        print(f"{s['label']:<55} {s['wf_pass']:>5} {s['total_pnl']:>9.1f}% {s['max_mdd']:>7.1f}% "
              f"{s['pnl_mdd']:>7.2f} {s['total_trades']:>7} {s['max_corr_loss']:>6.1f}% {wf_mark}")

    # Recommendation
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    passed = [s for s in all_scenarios if s['wf_passed'] and s['label'] != 'NO_PROTECTIONS']
    if passed:
        best = max(passed, key=lambda x: x['pnl_mdd'])
        print(f"Best WF-PASS scenario (by PnL/MDD): {best['label']}")
        print(f"  PnL: {best['total_pnl']:.1f}%, MDD: {best['max_mdd']:.1f}%, PnL/MDD: {best['pnl_mdd']:.2f}")
        print(f"  vs BASELINE: PnL {best['total_pnl'] - baseline_wf['total_pnl']:+.1f}%, "
              f"MDD {best['max_mdd'] - baseline_wf['max_mdd']:+.1f}%")
        print(f"  Blocked: {best.get('total_blocked', {})}")

        if best['label'] == 'BASELINE':
            print("\n→ VERDICT: Current protections sufficient. No additional measures needed.")
        else:
            print(f"\n→ VERDICT: {best['label']} improves risk-adjusted returns.")
    else:
        print("No scenario passed WF 3/3 except baseline.")

    # Save results
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {OUTPUT_FILE}")
    print(f"Total time: {time.time() - t0:.1f}s")


# ============================================================
# HELPERS
# ============================================================

def _print_wf(r):
    wf_mark = "✅" if r['wf_passed'] else "❌"
    print(f"  {r['label']:<50} WF={r['wf_pass']} {wf_mark} | "
          f"PnL={r['total_pnl']:+.1f}% MDD={r['max_mdd']:.1f}% P/M={r['pnl_mdd']:.2f} "
          f"Trades={r['total_trades']} CorrL={r['max_corr_loss']:.1f}%")
    if r.get('total_blocked'):
        b = r['total_blocked']
        blocked_str = ', '.join(f"{k}={v}" for k, v in b.items() if v > 0)
        if blocked_str:
            print(f"    Blocked: {blocked_str}")
    for fold in r['folds']:
        print(f"    Fold {fold['fold']}: PnL={fold['pnl']:+.1f}% MDD={fold['mdd']:.1f}% "
              f"WR={fold['wr']:.1f}% T={fold['trades']}")


def _find_best(results):
    passed = [r for r in results if r['wf_passed']]
    if not passed:
        return None
    return max(passed, key=lambda x: x['pnl_mdd'])


def _parse_h2_label(label):
    """Parse H2 label to extract params. E.g., 'H2_lb60min_thr2.0pct_cd30min'"""
    parts = label.split('_')
    lb_min = int(parts[1].replace('lb', '').replace('min', ''))
    thr = float(parts[2].replace('thr', '').replace('pct', ''))
    cd_min = int(parts[3].replace('cd', '').replace('min', ''))
    return (lb_min // 5, thr, cd_min // 5)  # convert back to bars


def _parse_h3_label(label):
    """Parse H3 label to extract caps. E.g., 'H3_cap_7/5/3'"""
    caps = label.split('_')[-1].split('/')
    return {'mild': int(caps[0]), 'strong': int(caps[1]), 'extreme': int(caps[2])}


if __name__ == '__main__':
    main()
