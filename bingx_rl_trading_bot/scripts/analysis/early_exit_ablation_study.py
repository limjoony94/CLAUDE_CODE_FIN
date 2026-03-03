#!/usr/bin/env python3
"""
Early Exit Ablation Study — v1.13 Early Exit 로직의 수익 기여 검증

연구 질문: 3연속 BD(LONG)/BU(SHORT) + PnL>=0.3% 조기 청산이 성과를 개선하는가?

방법론:
  - N-pos 포트폴리오 시뮬레이터 (portfolio_npos 기반)에 early exit 로직 추가
  - Scenario A: Baseline (TP/SL/timeout만, early exit OFF)
  - Scenario B: Early Exit ON (production 동일: 3x BD/BU, min_profit 0.3%)
  - Scenario C-F: 파라미터 변형 (confirm=2, min_profit=0/0.5/1.0%)
  - 3-fold Expanding Window WF 검증
  - Neutral window 적용

Standard Research Protocol:
  - Entry: 신호 다음 봉 Open
  - Exit: Intrabar High/Low (distance-based)
  - Sizing: Compound (복리)
  - Fee: 0.05% × 2 = 0.10% (× LEVERAGE = 0.30%)
  - Slippage: 0.02% buffer
  - MC: 생략 (ablation 비교 연구)
  - WF: 3-fold Expanding Window
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)

from scripts.production.pattern_5m.indicators import classify_candle
from scripts.production.pattern_5m.constants import AVG_BODY_WINDOW

# ============================================================
# Constants (from scanner, production-aligned)
# ============================================================
FEE_PCT = 0.10
LEVERAGE = 3
TIMEOUT_BARS = 864
SLIPPAGE_BUFFER = 0.02
DEFAULT_N_SLOTS = 9
DEFAULT_DIRECTION_CAP = 7
DEFAULT_REGIME_MULT = 0.3
DEFAULT_AGG_RISK_COUNTER = 3.0
DEFAULT_AGG_RISK_WITH = 7.0
DEFAULT_MOMENTUM_LOOKBACK = 6
DEFAULT_MOMENTUM_THRESHOLD = 1.0
DEFAULT_MOMENTUM_COOLDOWN = 6
NPOS_EMA_PERIOD = 20
NPOS_EMA_LOOKBACK = 5
DEFAULT_ATR_CLAMP_LO = 0.6
DEFAULT_ATR_CLAMP_HI = 1.7

# Early Exit parameters (production: constants.py)
EARLY_EXIT_BEARISH = ['BD']  # exit LONG
EARLY_EXIT_BULLISH = ['BU']  # exit SHORT
EARLY_EXIT_CONFIRM = 3       # consecutive candles required
EARLY_EXIT_MIN_PROFIT = 0.3  # minimum leveraged PnL %

DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
PATTERNS_FILE = os.path.join(_PROJECT_ROOT, 'results', 'dynamic_patterns.json')
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'early_exit_ablation_study.json')


def load_data():
    """Load and classify data."""
    df = pd.read_csv(DATA_FILE)
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
    return df, types


def load_patterns():
    """Load dynamic patterns → {pat_name: {direction, tp_pct, sl_pct}}."""
    with open(PATTERNS_FILE) as f:
        data = json.load(f)
    directions = data.get('patterns', {})
    tpsl = data.get('patterns_tpsl', {})
    result = {}
    for direction_key, dir_label in [('long', 'LONG'), ('short', 'SHORT')]:
        for pat_name in directions.get(direction_key, []):
            tp, sl = tpsl.get(pat_name, [1.0, 1.0])
            result[pat_name] = {'direction': dir_label, 'tp_pct': tp, 'sl_pct': sl}
    return result


def build_signal_index(types, n):
    """Build pattern -> signal bar indices."""
    idx = {}
    for i in range(2, n):
        pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
        if pat not in idx:
            idx[pat] = []
        idx[pat].append(i)
    return idx


def compute_atr_ratio(highs, lows, closes, atr_period=14, window=576):
    """ATR / rolling_median(ATR) ratio."""
    n = len(closes)
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i-1]),
                     abs(lows[i] - closes[i-1]))
    atr = np.full(n, np.nan)
    if n >= atr_period:
        atr[atr_period-1] = tr[:atr_period].mean()
        for i in range(atr_period, n):
            atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    med = pd.Series(atr).rolling(window, min_periods=window).median().values
    ratio = np.full(n, np.nan)
    valid = (~np.isnan(atr)) & (~np.isnan(med)) & (med > 0)
    ratio[valid] = atr[valid] / med[valid]
    return ratio


def compute_ema_slope(closes, period=NPOS_EMA_PERIOD, lookback=NPOS_EMA_LOOKBACK):
    """EMA slope."""
    n = len(closes)
    ema = pd.Series(closes).ewm(span=period, adjust=False).mean().values
    slope = np.full(n, 0.0)
    for i in range(lookback, n):
        if ema[i - lookback] > 0:
            slope[i] = (ema[i] / ema[i - lookback] - 1) * 100
    return slope


def find_neutral_window(closes, tol_pct=1.0, min_days=90, bars_per_day=288):
    """Find longest neutral price window."""
    n = len(closes)
    min_bars = min_days * bars_per_day
    best_len, best_start, best_end = 0, 0, 0
    for i in range(0, n - min_bars, bars_per_day):
        sp = closes[i]
        lo, hi = sp * (1 - tol_pct/100), sp * (1 + tol_pct/100)
        for j in range(n-1, i + min_bars - 1, -bars_per_day):
            if lo <= closes[j] <= hi:
                length = j - i
                if length > best_len:
                    best_len, best_start, best_end = length, i, j
                break
    if best_len == 0:
        return None
    return best_start, best_end


def portfolio_npos_early_exit(
    signal_tuples, opens, highs, lows, closes, type_codes, n_bars,
    atr_ratio, ema_slope, start_bar, end_bar,
    early_exit_enabled=False,
    early_confirm=EARLY_EXIT_CONFIRM,
    early_min_profit=EARLY_EXIT_MIN_PROFIT,
    n_slots=DEFAULT_N_SLOTS, direction_cap=DEFAULT_DIRECTION_CAP,
    regime_mult=DEFAULT_REGIME_MULT,
    agg_risk_counter=DEFAULT_AGG_RISK_COUNTER,
    agg_risk_with=DEFAULT_AGG_RISK_WITH,
    momentum_lookback=DEFAULT_MOMENTUM_LOOKBACK,
    momentum_threshold=DEFAULT_MOMENTUM_THRESHOLD,
    momentum_cooldown=DEFAULT_MOMENTUM_COOLDOWN,
    clamp_lo=DEFAULT_ATR_CLAMP_LO, clamp_hi=DEFAULT_ATR_CLAMP_HI,
    timeout_bars=TIMEOUT_BARS,
):
    """N-pos portfolio simulator with optional early exit.

    Same as scanner portfolio_npos() + early exit logic from signals.py.
    Early exit: track consecutive reversal candles per position,
    close if count >= early_confirm AND leveraged PnL >= early_min_profit.
    """
    size_pct = 100.0 / n_slots
    fee = FEE_PCT * LEVERAGE

    positions = []
    trades = []
    equity = 100.0
    peak_equity = 100.0
    max_dd = 0.0

    momentum_pause_until = {'LONG': -1, 'SHORT': -1}
    total_blocked = {'momentum': 0, 'agg_risk': 0, 'dir_cap': 0, 'dup_pat': 0, 'max_pos': 0}
    early_exit_count = 0

    signals_in_range = [(s, p, d, tp, sl) for s, p, d, tp, sl in signal_tuples
                        if start_bar <= s < end_bar]
    signals_sorted = sorted(signals_in_range, key=lambda x: x[0])
    sig_idx = 0

    for bar in range(start_bar, end_bar):
        # 1. Check TP/SL/timeout exits
        closed_slots = []
        bar_pnl_sum = 0.0

        for pos in positions:
            result = _check_exit(pos, bar, opens, highs, lows, n_bars,
                                 atr_ratio, fee, clamp_lo, clamp_hi, timeout_bars)
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

        # 2. Early exit check (on last completed candle = bar-1)
        if early_exit_enabled and bar >= 2:
            candle_type = type_codes[bar - 1] if bar - 1 < len(type_codes) else ''

            for pos in positions:
                if pos['slot'] in closed_slots:
                    continue

                entry_bar = pos['entry_bar']
                if entry_bar >= n_bars or bar <= entry_bar:
                    continue

                direction = pos['direction']
                entry_price = opens[entry_bar]
                if entry_price <= 0:
                    continue

                current_price = closes[bar - 1]

                # Calculate leveraged PnL
                if direction == 'LONG':
                    unr_pnl = (current_price / entry_price - 1) * 100 * LEVERAGE
                    reversal_types = EARLY_EXIT_BEARISH
                else:
                    unr_pnl = (entry_price / current_price - 1) * 100 * LEVERAGE
                    reversal_types = EARLY_EXIT_BULLISH

                # Track reversal count
                if candle_type in reversal_types:
                    pos['reversal_count'] = pos.get('reversal_count', 0) + 1
                else:
                    pos['reversal_count'] = 0

                # Check trigger
                if (pos['reversal_count'] >= early_confirm
                        and unr_pnl >= early_min_profit):
                    # Early exit at current close
                    exit_price = current_price
                    if direction == 'LONG':
                        pnl = (exit_price / entry_price - 1) * 100 * LEVERAGE
                    else:
                        pnl = (1 - exit_price / entry_price) * 100 * LEVERAGE
                    pnl -= fee

                    sm = pos.get('size_mult', 1.0)
                    pnl_portfolio = pnl * (size_pct / 100) * sm
                    trades.append({
                        'entry_bar': entry_bar, 'exit_bar': bar, 'pnl_slot': pnl,
                        'reason': 'EARLY_EXIT', 'pattern': pos['pattern'],
                        'direction': direction, 'size_mult': sm,
                        'pnl_portfolio': pnl_portfolio,
                    })
                    closed_slots.append(pos['slot'])
                    bar_pnl_sum += pnl_portfolio
                    early_exit_count += 1

        positions = [p for p in positions if p['slot'] not in closed_slots]

        equity += bar_pnl_sum
        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0
        if dd > max_dd:
            max_dd = dd

        # Momentum guard
        if momentum_lookback > 0 and momentum_threshold > 0 and bar >= momentum_lookback:
            pn = closes[bar]
            pa = closes[bar - momentum_lookback]
            if pa > 0:
                pct = (pn / pa - 1) * 100
                if pct > momentum_threshold:
                    momentum_pause_until['SHORT'] = bar + momentum_cooldown
                elif pct < -momentum_threshold:
                    momentum_pause_until['LONG'] = bar + momentum_cooldown

        # 3. Process entries
        while sig_idx < len(signals_sorted) and signals_sorted[sig_idx][0] == bar:
            sig_bar, pat, direction, tp_pct, sl_pct = signals_sorted[sig_idx]
            sig_idx += 1

            if len(positions) >= n_slots:
                total_blocked['max_pos'] += 1
                continue

            dir_count = sum(1 for p in positions if p['direction'] == direction)
            if dir_count >= direction_cap:
                total_blocked['dir_cap'] += 1
                continue

            if any(p['pattern'] == pat for p in positions):
                total_blocked['dup_pat'] += 1
                continue

            entry_bar = sig_bar + 1
            if entry_bar >= n_bars:
                continue

            if momentum_lookback > 0 and bar < momentum_pause_until.get(direction, -1):
                total_blocked['momentum'] += 1
                continue

            sm = 1.0
            if regime_mult is not None and bar < len(ema_slope):
                s = ema_slope[bar]
                if (s > 0 and direction == 'SHORT') or (s <= 0 and direction == 'LONG'):
                    sm = regime_mult

            if agg_risk_counter > 0 or agg_risk_with > 0:
                is_uptrend = ema_slope[bar] > 0 if bar < len(ema_slope) else False
                is_counter = ((is_uptrend and direction == 'SHORT') or
                              (not is_uptrend and direction == 'LONG'))
                cap_pct = agg_risk_counter if is_counter else agg_risk_with

                existing_exposure = 0.0
                for p in positions:
                    if p['direction'] == direction:
                        p_sl = p['sl_pct']
                        p_sig = p['signal_bar']
                        if (atr_ratio is not None and p_sig < len(atr_ratio)
                                and not np.isnan(atr_ratio[p_sig])):
                            p_r = max(clamp_lo, min(clamp_hi, atr_ratio[p_sig]))
                        else:
                            p_r = 1.0
                        p_eff_sl = p_sl * p_r
                        p_sm = p.get('size_mult', 1.0)
                        existing_exposure += p_eff_sl * (1.0 / n_slots) * LEVERAGE * p_sm

                new_r = 1.0
                if (atr_ratio is not None and sig_bar < len(atr_ratio)
                        and not np.isnan(atr_ratio[sig_bar])):
                    new_r = max(clamp_lo, min(clamp_hi, atr_ratio[sig_bar]))
                new_eff_sl = sl_pct * new_r
                new_exposure = new_eff_sl * (1.0 / n_slots) * LEVERAGE * sm

                if existing_exposure + new_exposure > cap_pct:
                    total_blocked['agg_risk'] += 1
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
                'reversal_count': 0,
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

    return trades, equity, max_dd, early_exit_count, total_blocked


def _check_exit(pos, bar, opens, highs, lows, n_bars, atr_ratio, fee,
                clamp_lo, clamp_hi, timeout_bars):
    """Check TP/SL/timeout exit (same as scanner _check_exit_npos)."""
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
        r = max(clamp_lo, min(clamp_hi, atr_ratio[sig_bar]))
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
    if hold >= timeout_bars:
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


def calc_stats(trades, final_equity, max_dd):
    """Calculate summary statistics."""
    if not trades:
        return {'trades': 0, 'wr': 0, 'pnl': 0, 'mdd': 0, 'pnl_mdd': 0}

    n = len(trades)
    wins = sum(1 for t in trades if t['pnl_portfolio'] > 0)
    wr = wins / n * 100 if n > 0 else 0
    pnl = final_equity - 100
    mdd = max_dd
    pnl_mdd = pnl / mdd if mdd > 0 else float('inf')

    early_exits = sum(1 for t in trades if t['reason'] == 'EARLY_EXIT')
    tp_count = sum(1 for t in trades if t['reason'] == 'TP')
    sl_count = sum(1 for t in trades if t['reason'] == 'SL')

    early_pnl = sum(t['pnl_portfolio'] for t in trades if t['reason'] == 'EARLY_EXIT')

    return {
        'trades': n, 'wins': wins, 'wr': round(wr, 1),
        'pnl': round(pnl, 1), 'mdd': round(mdd, 2),
        'pnl_mdd': round(pnl_mdd, 2),
        'tp': tp_count, 'sl': sl_count,
        'early_exits': early_exits,
        'early_pnl': round(early_pnl, 2),
    }


def run_wf(signal_tuples, opens, highs, lows, closes, type_codes,
           n_bars, atr_ratio, ema_slope, neutral_start, neutral_end,
           scenario_name, early_exit_enabled, early_confirm, early_min_profit,
           n_folds=3):
    """Run expanding window WF with given early exit params."""
    total_bars = neutral_end - neutral_start
    fold_size = total_bars // (n_folds + 1)  # +1 because first fold IS = 1 part, OOS = 1 part

    results = []
    for fold in range(n_folds):
        is_end = neutral_start + fold_size * (fold + 1)
        oos_start = is_end
        oos_end = neutral_start + fold_size * (fold + 2) if fold < n_folds - 1 else neutral_end

        # Run OOS
        trades, equity, max_dd, early_count, blocked = portfolio_npos_early_exit(
            signal_tuples, opens, highs, lows, closes, type_codes, n_bars,
            atr_ratio, ema_slope, oos_start, oos_end,
            early_exit_enabled=early_exit_enabled,
            early_confirm=early_confirm,
            early_min_profit=early_min_profit,
        )

        stats = calc_stats(trades, equity, max_dd)
        stats['fold'] = fold + 1
        stats['oos_bars'] = oos_end - oos_start
        results.append(stats)

    return results


def main():
    print("=" * 70)
    print("Early Exit Ablation Study")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Phase 1: Load data
    print("\n[Phase 1] Loading data and patterns...")
    df, types = load_data()
    n_bars = len(df)
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    type_codes = df['candle_type'].values

    atr_ratio = compute_atr_ratio(highs, lows, closes)
    ema_slope = compute_ema_slope(closes)

    # Neutral window
    nw = find_neutral_window(closes)
    if nw is None:
        print("ERROR: No neutral window found")
        return
    neutral_start, neutral_end = nw
    print(f"  Neutral window: bar {neutral_start} → {neutral_end} ({(neutral_end-neutral_start)/288:.0f} days)")
    print(f"  Total bars: {n_bars}")

    # Load patterns
    patterns = load_patterns()
    print(f"  Patterns loaded: {len(patterns)}")

    # Build signal tuples (neutral-sliced)
    sig_idx = build_signal_index(types[neutral_start:neutral_end+1], neutral_end - neutral_start + 1)

    signal_tuples = []
    for pat_name, info in patterns.items():
        direction = info.get('direction', '')
        tp_pct = info.get('tp_pct', 1.0)
        sl_pct = info.get('sl_pct', 1.0)
        if pat_name in sig_idx:
            for bar_offset in sig_idx[pat_name]:
                abs_bar = neutral_start + bar_offset
                signal_tuples.append((abs_bar, pat_name, direction, tp_pct, sl_pct))

    signal_tuples.sort(key=lambda x: x[0])
    print(f"  Total signals: {len(signal_tuples)}")

    # Phase 2: IS comparison (full neutral window)
    print("\n[Phase 2] In-Sample comparison (full neutral window)...")
    scenarios = [
        ('A_baseline',      False, 3, 0.3),
        ('B_production',    True,  3, 0.3),   # production 동일
        ('C_confirm2',      True,  2, 0.3),   # 더 민감
        ('D_minprofit0',    True,  3, 0.0),   # 이익 조건 제거
        ('E_minprofit0.5',  True,  3, 0.5),   # 더 보수적
        ('F_minprofit1.0',  True,  3, 1.0),   # 매우 보수적
    ]

    is_results = {}
    for name, enabled, confirm, min_profit in scenarios:
        trades, equity, max_dd, early_count, blocked = portfolio_npos_early_exit(
            signal_tuples, opens, highs, lows, closes, type_codes, n_bars,
            atr_ratio, ema_slope, neutral_start, neutral_end,
            early_exit_enabled=enabled,
            early_confirm=confirm,
            early_min_profit=min_profit,
        )
        stats = calc_stats(trades, equity, max_dd)
        is_results[name] = stats
        print(f"  {name:20s} | Trades: {stats['trades']:4d} | WR: {stats['wr']:5.1f}% | "
              f"PnL: {stats['pnl']:+8.1f}% | MDD: {stats['mdd']:5.2f}% | "
              f"PnL/MDD: {stats['pnl_mdd']:7.2f} | "
              f"Early: {stats['early_exits']:3d} ({stats['early_pnl']:+.1f}%)")

    # Phase 3: WF validation
    print("\n[Phase 3] Walk-Forward Validation (3-fold Expanding Window)...")
    wf_results = {}
    for name, enabled, confirm, min_profit in scenarios:
        folds = run_wf(
            signal_tuples, opens, highs, lows, closes, type_codes, n_bars,
            atr_ratio, ema_slope, neutral_start, neutral_end,
            name, enabled, confirm, min_profit,
        )
        wf_results[name] = folds

        fold_pnls = [f['pnl'] for f in folds]
        all_pass = all(p > 0 for p in fold_pnls)
        min_fold = min(fold_pnls)
        total_oos = sum(fold_pnls)
        avg_wr = np.mean([f['wr'] for f in folds])
        total_early = sum(f['early_exits'] for f in folds)

        verdict = "✅ PASS" if all_pass else "❌ FAIL"
        print(f"  {name:20s} | {verdict} | Folds: {[f'+{p:.1f}%' if p > 0 else f'{p:.1f}%' for p in fold_pnls]} | "
              f"Total: {total_oos:+.1f}% | Avg WR: {avg_wr:.1f}% | "
              f"Min fold: {min_fold:+.1f}% | Early exits: {total_early}")

    # Phase 4: Delta analysis (B vs A)
    print("\n[Phase 4] Delta Analysis (Production vs Baseline)...")
    a = is_results['A_baseline']
    b = is_results['B_production']

    print(f"  PnL delta:     {b['pnl'] - a['pnl']:+.1f}% ({a['pnl']:+.1f} → {b['pnl']:+.1f})")
    print(f"  MDD delta:     {b['mdd'] - a['mdd']:+.2f}% ({a['mdd']:.2f} → {b['mdd']:.2f})")
    print(f"  PnL/MDD delta: {b['pnl_mdd'] - a['pnl_mdd']:+.2f} ({a['pnl_mdd']:.2f} → {b['pnl_mdd']:.2f})")
    print(f"  WR delta:      {b['wr'] - a['wr']:+.1f}pp ({a['wr']:.1f} → {b['wr']:.1f})")
    print(f"  Trade count:   {b['trades'] - a['trades']:+d} ({a['trades']} → {b['trades']})")
    print(f"  Early exits:   {b['early_exits']} (contributed {b['early_pnl']:+.2f}% to portfolio)")

    # WF comparison
    a_folds = wf_results['A_baseline']
    b_folds = wf_results['B_production']
    print(f"\n  WF OOS comparison:")
    for i in range(len(a_folds)):
        af, bf = a_folds[i], b_folds[i]
        print(f"    Fold {i+1}: Baseline {af['pnl']:+.1f}% → Production {bf['pnl']:+.1f}% "
              f"(delta {bf['pnl'] - af['pnl']:+.1f}%)")

    # Verdict
    print("\n" + "=" * 70)
    pnl_better = b['pnl'] > a['pnl']
    mdd_better = b['mdd'] <= a['mdd']
    pnl_mdd_better = b['pnl_mdd'] > a['pnl_mdd']

    b_wf_pass = all(f['pnl'] > 0 for f in b_folds)
    a_wf_pass = all(f['pnl'] > 0 for f in a_folds)

    if pnl_mdd_better and b_wf_pass:
        verdict = "GO — Early Exit improves risk-adjusted performance"
    elif not pnl_mdd_better and b_wf_pass:
        verdict = "NEUTRAL — Early Exit WF passes but PnL/MDD not improved"
    else:
        verdict = "STOP — Early Exit degrades performance or WF FAIL"

    print(f"VERDICT: {verdict}")
    print(f"  PnL better: {pnl_better} | MDD better: {mdd_better} | PnL/MDD better: {pnl_mdd_better}")
    print(f"  Baseline WF: {'PASS' if a_wf_pass else 'FAIL'} | Production WF: {'PASS' if b_wf_pass else 'FAIL'}")
    print("=" * 70)

    # Save results
    output = {
        'study': 'early_exit_ablation',
        'timestamp': datetime.now().isoformat(),
        'data_file': DATA_FILE,
        'patterns': len(patterns),
        'signals': len(signal_tuples),
        'neutral_window': {'start': int(neutral_start), 'end': int(neutral_end),
                           'days': round((neutral_end - neutral_start) / 288, 1)},
        'is_results': is_results,
        'wf_results': {k: v for k, v in wf_results.items()},
        'delta': {
            'pnl': round(b['pnl'] - a['pnl'], 1),
            'mdd': round(b['mdd'] - a['mdd'], 2),
            'pnl_mdd': round(b['pnl_mdd'] - a['pnl_mdd'], 2),
            'wr': round(b['wr'] - a['wr'], 1),
        },
        'verdict': verdict,
    }
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
