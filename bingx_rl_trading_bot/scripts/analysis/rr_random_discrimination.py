#!/usr/bin/env python3
"""
Random Signal Discrimination Test for R:R + Holdtime Study WF Results.

Purpose: The R:R study showed 25/25 GO verdicts. This test checks whether
random signals ALSO pass WF 3/3, which would mean WF is non-discriminating
(same lesson as v1.43.0 Entry Optimization rollback).

Method:
1. Load same data + neutral window as production scanner
2. For 30 random seeds, generate random LONG/SHORT signals matching real trade frequency
3. Assign random real pattern TP/SL (scaled by tp0.9_sl1.0) to each signal
4. Run N-pos simulation (9 slots, dir_cap 7, agg_risk 8/15, cascade SL 85%, momentum guard)
5. Run 3-fold expanding window WF on each random set
6. Count how many random sets get WF 3/3 PASS

Critical rules:
- classify_candle from production (for data loading consistency)
- LEVERAGE = 3, FEE_PCT = 0.10
- Timeout = DROP
- Same-bar resolution: abs(tp_price - bar_OPEN)
- Fee = FEE_PCT * LEVERAGE (BingX notional)
- Expanding window WF only
- Compound sizing
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd

# Project root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)

from scripts.production.pattern_5m.indicators import classify_candle
from scripts.production.pattern_5m.constants import AVG_BODY_WINDOW

# ============================================================
# Constants (production-aligned)
# ============================================================
FEE_PCT = 0.10
LEVERAGE = 3
MAX_BARS = 288
SLIPPAGE_BUFFER = 0.02

# N-pos defaults (v1.53.0 production)
N_SLOTS = 9
DIRECTION_CAP = 7
REGIME_MULT = 0.3
AGG_RISK_COUNTER = 8.0
AGG_RISK_WITH = 15.0
MOMENTUM_LOOKBACK = 3
MOMENTUM_THRESHOLD = 1.5
MOMENTUM_COOLDOWN = 12
CASCADE_TIGHTEN_PCT = 85
TIMEOUT_BARS = 288

# ATR defaults (v1.53.0)
ATR_PERIOD = 14
ATR_WINDOW = 576
ATR_CLAMP_LO = 0.5
ATR_CLAMP_HI = 1.5

# EMA slope for regime sizing
EMA_PERIOD = 20
EMA_LOOKBACK = 5

# WF
WF_FOLDS = 3

# Test config
N_RANDOM_SEEDS = 30
TP_SCALE = 0.9   # tp0.9_sl1.0 scenario (top H3 result)
SL_SCALE = 1.0

DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
PATTERNS_FILE = os.path.join(_PROJECT_ROOT, 'results', 'dynamic_patterns.json')


# ============================================================
# Helpers (from scanner, production-aligned)
# ============================================================

def load_and_classify(data_file):
    df = pd.read_csv(data_file)
    df['body'] = abs(df['close'] - df['open'])
    df['avg_body'] = df['body'].rolling(AVG_BODY_WINDOW).mean()
    types = []
    for i, row in df.iterrows():
        avg_b = df.at[i, 'avg_body']
        if pd.isna(avg_b):
            avg_b = 1.0
        ct = classify_candle(row, avg_b)
        types.append(ct.value)
    df['candle_type'] = types
    return df


def compute_atr_ratio(highs, lows, closes, atr_period=ATR_PERIOD, window=ATR_WINDOW):
    n = len(closes)
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i - 1]),
                     abs(lows[i] - closes[i - 1]))
    atr = np.full(n, np.nan)
    if n >= atr_period:
        atr[atr_period - 1] = tr[:atr_period].mean()
        for i in range(atr_period, n):
            atr[i] = (atr[i - 1] * (atr_period - 1) + tr[i]) / atr_period
    med = pd.Series(atr).rolling(window, min_periods=window).median().values
    ratio = np.full(n, np.nan)
    valid = (~np.isnan(atr)) & (~np.isnan(med)) & (med > 0)
    ratio[valid] = atr[valid] / med[valid]
    return ratio


def compute_ema_slope(closes, period=EMA_PERIOD, lookback=EMA_LOOKBACK):
    n = len(closes)
    ema = pd.Series(closes).ewm(span=period, adjust=False).mean().values
    slope = np.full(n, 0.0)
    for i in range(lookback, n):
        if ema[i - lookback] > 0:
            slope[i] = (ema[i] / ema[i - lookback] - 1) * 100
    return slope


def find_neutral_window(closes, tol_pct=1.0, min_days=90, bars_per_day=288):
    n = len(closes)
    min_bars = min_days * bars_per_day
    best_len, best_start, best_end = 0, 0, 0
    for i in range(0, n - min_bars, bars_per_day):
        start_price = closes[i]
        lo = start_price * (1 - tol_pct / 100)
        hi = start_price * (1 + tol_pct / 100)
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


def _check_exit_npos(pos, bar, opens, highs, lows, n_bars, atr_ratio, fee):
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
    eff_sl_override = pos.get('eff_sl_override')
    if eff_sl_override is not None:
        eff_sl = max(0.1, eff_sl_override - SLIPPAGE_BUFFER)
    else:
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


def portfolio_npos(signal_tuples, opens, highs, lows, closes, n_bars,
                   atr_ratio, ema_slope, start_bar, end_bar):
    """N-pos portfolio simulator with all production protections."""
    size_pct = 100.0 / N_SLOTS
    fee = FEE_PCT * LEVERAGE

    positions = []
    trades = []
    equity = 100.0
    peak_equity = 100.0
    momentum_pause_until = {'LONG': -1, 'SHORT': -1}

    signals_in_range = [(s, p, d, tp, sl) for s, p, d, tp, sl in signal_tuples
                        if start_bar <= s < end_bar]
    signals_sorted = sorted(signals_in_range, key=lambda x: x[0])
    sig_idx = 0

    for bar in range(start_bar, end_bar):
        # 1. Check exits
        closed_slots = []
        bar_sl_count = 0
        bar_pnl_sum = 0.0

        for pos in positions:
            result = _check_exit_npos(pos, bar, opens, highs, lows, n_bars, atr_ratio, fee)
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
                trades.append(result)
                closed_slots.append(pos['slot'])
                bar_pnl_sum += pnl_portfolio
                if result['reason'] == 'SL':
                    bar_sl_count += 1

        # Cascade SL tightening
        if CASCADE_TIGHTEN_PCT > 0 and bar_sl_count > 0:
            keep_ratio = 1.0 - CASCADE_TIGHTEN_PCT / 100.0
            sl_directions = set()
            for t in trades[len(trades) - len(closed_slots):]:
                if t.get('reason') == 'SL':
                    sl_directions.add(t['direction'])
            for sl_dir in sl_directions:
                for pos in positions:
                    if pos['slot'] in closed_slots:
                        continue
                    if pos['direction'] != sl_dir:
                        continue
                    sig = pos['signal_bar']
                    if atr_ratio is not None and sig < len(atr_ratio) and not np.isnan(atr_ratio[sig]):
                        r = max(ATR_CLAMP_LO, min(ATR_CLAMP_HI, atr_ratio[sig]))
                    else:
                        r = 1.0
                    cur_eff_sl = pos.get('eff_sl_override') or (pos['sl_pct'] * r)
                    pos['eff_sl_override'] = cur_eff_sl * keep_ratio

        positions = [p for p in positions if p['slot'] not in closed_slots]
        equity += bar_pnl_sum
        if equity > peak_equity:
            peak_equity = equity

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

            # Momentum guard check
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

            # Aggregate risk cap
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
                        existing_exposure += p_eff_sl * (1.0 / N_SLOTS) * LEVERAGE * p_sm

                new_r = 1.0
                if (atr_ratio is not None and sig_bar < len(atr_ratio)
                        and not np.isnan(atr_ratio[sig_bar])):
                    new_r = max(ATR_CLAMP_LO, min(ATR_CLAMP_HI, atr_ratio[sig_bar]))
                new_eff_sl = sl_pct * new_r
                new_exposure = new_eff_sl * (1.0 / N_SLOTS) * LEVERAGE * sm

                if existing_exposure + new_exposure > cap_pct:
                    continue

            positions.append({
                'slot': f"{pat}_{sig_bar}",
                'signal_bar': sig_bar,
                'entry_bar': entry_bar,
                'direction': direction,
                'pattern': pat,
                'tp_pct': tp_pct,
                'sl_pct': sl_pct,
                'size_mult': sm,
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
            'pnl_portfolio': pnl * (size_pct / 100) * sm,
        })

    return trades


def calc_stats_compound(trades):
    if not trades:
        return {'trades': 0, 'wr': 0.0, 'pnl': 0.0, 'mdd': 0.0, 'pnl_mdd': 0.0}
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
    }


# ============================================================
# Random Signal Generation
# ============================================================

def generate_random_signals(rng, n_bars, neutral_start, neutral_end,
                            target_count, pattern_details):
    """Generate random LONG/SHORT signals within neutral window.

    Each signal gets a randomly assigned real pattern's TP/SL (scaled).
    Returns list of (signal_bar, pattern_name, direction, tp_pct, sl_pct).
    """
    # Available bars for signals (need room for entry at bar+1)
    available_bars = np.arange(neutral_start + 2, neutral_end - 1)
    if len(available_bars) < target_count:
        target_count = len(available_bars)

    # Random signal bars (no replacement to avoid duplicates)
    sig_bars = rng.choice(available_bars, size=target_count, replace=False)
    sig_bars.sort()

    # Get all real pattern TP/SL pairs
    pat_list = list(pattern_details.values())

    signals = []
    for sb in sig_bars:
        # Random direction
        direction = rng.choice(['LONG', 'SHORT'])

        # Random real pattern's TP/SL
        pat_info = pat_list[rng.randint(0, len(pat_list))]
        tp = pat_info['tp'] * TP_SCALE
        sl = pat_info['sl'] * SL_SCALE

        # Use a unique fake pattern name (for dup-pattern blocking)
        fake_pat = f"RND_{sb}"
        signals.append((sb, fake_pat, direction, tp, sl))

    return signals


# ============================================================
# WF for Random Signals
# ============================================================

def wf_random(signal_tuples, opens, highs, lows, closes, n_bars,
              atr_ratio, ema_slope, neutral_start, neutral_end, n_folds=WF_FOLDS):
    """Run expanding window WF on random signals using N-pos simulation.

    IS pattern selection is skipped (we use random signals directly).
    Each fold evaluates OOS PnL with N-pos.
    """
    data_len = neutral_end - neutral_start
    seg_size = data_len // (n_folds + 1)
    folds = []

    for fold in range(n_folds):
        oos_start = neutral_start + (fold + 1) * seg_size
        oos_end = neutral_start + (fold + 2) * seg_size if fold < n_folds - 1 else neutral_end

        # OOS signals only
        oos_signals = [(s, p, d, tp, sl) for s, p, d, tp, sl in signal_tuples
                       if oos_start <= s < oos_end]

        if not oos_signals:
            folds.append({'fold': fold + 1, 'oos_pnl': 0.0, 'oos_trades': 0,
                          'oos_wr': 0.0, 'oos_positive': False})
            continue

        oos_trades = portfolio_npos(
            oos_signals, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, oos_start, oos_end)

        stats = calc_stats_compound(oos_trades)
        folds.append({
            'fold': fold + 1,
            'oos_pnl': stats['pnl'],
            'oos_trades': stats['trades'],
            'oos_wr': stats['wr'],
            'oos_mdd': stats['mdd'],
            'oos_positive': stats['pnl'] > 0,
        })

    positive_folds = sum(1 for f in folds if f['oos_positive'])
    return {
        'folds': folds,
        'positive_folds': positive_folds,
        'pass_3_3': positive_folds == n_folds,
        'total_oos_pnl': round(sum(f['oos_pnl'] for f in folds), 2),
    }


# ============================================================
# Main
# ============================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("RANDOM SIGNAL DISCRIMINATION TEST")
    print("R:R + Holdtime Study WF Validation")
    print("=" * 70)
    print()

    # 1. Load data
    print("[1/5] Loading and classifying data...")
    df = load_and_classify(DATA_FILE)
    n_bars = len(df)
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    print(f"  Loaded {n_bars} bars")

    # 2. Load dynamic patterns for TP/SL and neutral window
    print("[2/5] Loading dynamic patterns...")
    with open(PATTERNS_FILE, 'r') as f:
        dp = json.load(f)

    pattern_details = dp['pattern_details']
    neutral = dp.get('neutral_window', {})
    neutral_start = neutral.get('start_bar', 0)
    neutral_end = neutral.get('end_bar', n_bars)
    print(f"  {len(pattern_details)} patterns loaded")
    print(f"  Neutral window: [{neutral_start}, {neutral_end}] "
          f"({(neutral_end - neutral_start) / 288:.0f} days)")

    # Count real trade frequency for target
    # Real IS had 1162 trades from ~259d neutral window
    # Signal count would be higher (many blocked by N-pos filters)
    # Use ~3x multiplier for signal count to get similar trade count after filtering
    target_signals = 3500  # ~3x of 1162 trades to account for blocking

    # 3. Compute ATR and EMA
    print("[3/5] Computing ATR ratio and EMA slope...")
    atr_ratio = compute_atr_ratio(highs, lows, closes)
    ema_slope = compute_ema_slope(closes)

    # 4. Run random tests
    print(f"[4/5] Running {N_RANDOM_SEEDS} random signal WF tests...")
    print(f"  TP scale: {TP_SCALE}x, SL scale: {SL_SCALE}x (tp0.9_sl1.0)")
    print(f"  N-pos: {N_SLOTS} slots, dir_cap {DIRECTION_CAP}, "
          f"agg_risk {AGG_RISK_COUNTER}/{AGG_RISK_WITH}")
    print(f"  Cascade SL: {CASCADE_TIGHTEN_PCT}%, Momentum guard: "
          f"lb{MOMENTUM_LOOKBACK}/th{MOMENTUM_THRESHOLD}/cd{MOMENTUM_COOLDOWN}")
    print(f"  ATR: period={ATR_PERIOD}, window={ATR_WINDOW}, "
          f"clamp=[{ATR_CLAMP_LO},{ATR_CLAMP_HI}]")
    print(f"  Target signals per seed: ~{target_signals}")
    print()

    results = []
    pass_count = 0

    for seed in range(1, N_RANDOM_SEEDS + 1):
        rng = np.random.RandomState(seed)

        # Generate random signals
        signals = generate_random_signals(
            rng, n_bars, neutral_start, neutral_end,
            target_signals, pattern_details)

        # Run WF
        wf_result = wf_random(
            signals, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, neutral_start, neutral_end)

        is_pass = wf_result['pass_3_3']
        if is_pass:
            pass_count += 1

        fold_pnls = [f['oos_pnl'] for f in wf_result['folds']]
        fold_trades = [f['oos_trades'] for f in wf_result['folds']]
        fold_wrs = [f['oos_wr'] for f in wf_result['folds']]

        status = "PASS" if is_pass else "FAIL"
        neg_folds = sum(1 for p in fold_pnls if p <= 0)

        results.append({
            'seed': seed,
            'pass_3_3': is_pass,
            'positive_folds': wf_result['positive_folds'],
            'total_oos_pnl': wf_result['total_oos_pnl'],
            'fold_pnls': fold_pnls,
            'fold_trades': fold_trades,
            'fold_wrs': fold_wrs,
        })

        print(f"  Seed {seed:2d}: {status} ({wf_result['positive_folds']}/3) | "
              f"OOS PnL: {wf_result['total_oos_pnl']:+8.1f}% | "
              f"Folds: {fold_pnls[0]:+6.1f} / {fold_pnls[1]:+6.1f} / {fold_pnls[2]:+6.1f} | "
              f"Trades: {fold_trades[0]:3d}/{fold_trades[1]:3d}/{fold_trades[2]:3d}")

    # 5. Summary
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()

    pass_pct = pass_count / N_RANDOM_SEEDS * 100
    print(f"Seeds tested:      {N_RANDOM_SEEDS}")
    print(f"WF 3/3 PASS:       {pass_count}/{N_RANDOM_SEEDS} ({pass_pct:.1f}%)")
    print()

    # Distribution of positive folds
    fold_dist = {}
    for r in results:
        pf = r['positive_folds']
        fold_dist[pf] = fold_dist.get(pf, 0) + 1
    print("Positive fold distribution:")
    for k in sorted(fold_dist.keys()):
        print(f"  {k}/3 positive: {fold_dist[k]} seeds ({fold_dist[k]/N_RANDOM_SEEDS*100:.0f}%)")
    print()

    # Avg OOS PnL
    avg_pnl = np.mean([r['total_oos_pnl'] for r in results])
    std_pnl = np.std([r['total_oos_pnl'] for r in results])
    avg_wr = np.mean([np.mean(r['fold_wrs']) for r in results])
    print(f"Avg OOS PnL:       {avg_pnl:+.1f}% (std {std_pnl:.1f}%)")
    print(f"Avg OOS WR:        {avg_wr:.1f}%")
    print()

    if pass_pct < 10:
        verdict = "DISCRIMINATING"
        explanation = ("WF successfully rejects random signals. "
                       "R:R study improvements likely have genuine discriminating power.")
    elif pass_pct < 30:
        verdict = "MODERATELY DISCRIMINATING"
        explanation = ("WF partially rejects random signals. "
                       "R:R study improvements may have some genuine edge but caution advised.")
    elif pass_pct < 50:
        verdict = "WEAKLY DISCRIMINATING"
        explanation = ("WF allows too many random signals through. "
                       "R:R study improvements are likely mechanism-driven, not genuine edge.")
    else:
        verdict = "NON-DISCRIMINATING"
        explanation = ("WF fails to reject random signals (like v1.43.0). "
                       "R:R study improvements are mechanism-driven, not genuine edge improvement.")

    print(f"VERDICT: {verdict}")
    print(f"  {explanation}")
    print()

    elapsed = time.time() - t0
    print(f"Total time: {elapsed:.1f}s")

    # Save results
    output = {
        'test': 'rr_random_discrimination',
        'scenario': 'tp0.9_sl1.0',
        'n_seeds': N_RANDOM_SEEDS,
        'pass_count': pass_count,
        'pass_pct': round(pass_pct, 1),
        'verdict': verdict,
        'avg_oos_pnl': round(avg_pnl, 2),
        'std_oos_pnl': round(std_pnl, 2),
        'avg_oos_wr': round(avg_wr, 1),
        'fold_distribution': fold_dist,
        'config': {
            'tp_scale': TP_SCALE,
            'sl_scale': SL_SCALE,
            'n_slots': N_SLOTS,
            'direction_cap': DIRECTION_CAP,
            'agg_risk': f"{AGG_RISK_COUNTER}/{AGG_RISK_WITH}",
            'cascade_tighten_pct': CASCADE_TIGHTEN_PCT,
            'momentum': f"lb{MOMENTUM_LOOKBACK}/th{MOMENTUM_THRESHOLD}/cd{MOMENTUM_COOLDOWN}",
            'atr_clamp': f"[{ATR_CLAMP_LO},{ATR_CLAMP_HI}]",
            'neutral_window': f"[{neutral_start},{neutral_end}]",
            'target_signals': target_signals,
        },
        'seed_results': results,
        'elapsed_sec': round(elapsed, 1),
    }

    out_file = os.path.join(_PROJECT_ROOT, 'results', 'rr_random_discrimination.json')
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to: {out_file}")


if __name__ == '__main__':
    main()
