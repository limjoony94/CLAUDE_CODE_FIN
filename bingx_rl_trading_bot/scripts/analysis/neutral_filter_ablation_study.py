#!/usr/bin/env python3
"""
Neutral Window Filter Ablation Study
======================================
With 51 neutral-window patterns (22L+29S, balanced L/S), the existing filters
— designed for the old 54pat (4L+50S, 93% SHORT) — may behave very differently.

This study tests each filter individually and in combinations to determine
the optimal filter set for the neutral-balanced portfolio.

Filters under test:
  F1: Direction Cap (7)
  F2: Regime Sizing (EMA20, counter_mult=0.3)
  F3: Aggregate Risk Cap (counter 3%, with 7%)
  F4: Momentum Guard (>1%/30min, 30min cooldown)
  F5: MDD Sizing (DD 5%→full, 20%→25%)
  F6: Position Timeout (864 bars = 72h)

Methodology:
  Phase 1: Individual filter tests (6 scenarios + baseline + no-filter)
  Phase 2: All 2^6=64 combinations (full factorial)
  Phase 3: WF validation on top candidates from Phase 2
  Phase 4: Summary and recommendation

Standard Research Protocol:
  Entry: signal bar + 1 open
  Exit: Intrabar high/low (distance-based, abs(tp - opens[bar]))
  Fee: 0.10% * LEVERAGE(3) = 0.30% per trade (capital-space)
  Slippage: 0.02%
  ATR-scaled TP/SL: ATR(14)/median(576), clamp [0.6, 1.7]
  WF: 3-fold expanding window, PASS = all folds OOS PnL > 0
"""

import json
import os
import sys
import time
import itertools
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

# MDD Sizing params (v1.34.0)
MDD_FULL_SIZE_BELOW_DD = 5.0   # below this DD%, full size
MDD_MIN_SIZE_ABOVE_DD = 15.0   # above this DD%, min size (25%)
MDD_MIN_SIZE_PCT = 25.0        # minimum size as % of full

# Momentum Guard params (v1.36.2)
MOM_LOOKBACK = 6
MOM_THRESHOLD = 1.0
MOM_COOLDOWN = 6

DATA_FILE = os.path.join(PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
PATTERNS_FILE = os.path.join(PROJECT_ROOT, 'results', 'dynamic_patterns.json')
OUTPUT_FILE = os.path.join(PROJECT_ROOT, 'results', 'neutral_filter_ablation_study.json')

# Filter names (for display and combinations)
FILTER_NAMES = ['dir_cap', 'regime_sizing', 'agg_risk', 'momentum', 'mdd_sizing', 'timeout']
FILTER_LABELS = {
    'dir_cap': 'F1:DirCap7',
    'regime_sizing': 'F2:RegimeSz',
    'agg_risk': 'F3:AggRisk',
    'momentum': 'F4:MomGuard',
    'mdd_sizing': 'F5:MDDSz',
    'timeout': 'F6:Timeout',
}


# ============================================================
# DATA LOADING
# ============================================================

def load_data():
    """Load data and apply neutral window from dynamic_patterns.json."""
    df = pd.read_csv(DATA_FILE)
    n_total = len(df)

    # Apply neutral window from patterns file metadata
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
        # Fallback: use shifted 270d window
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

def _check_exit(pos, bar, opens, highs, lows, n_bars, atr_ratio, fee, use_timeout=True):
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

    # Timeout check (F6)
    hold = bar - entry_bar
    if use_timeout and hold >= TIMEOUT_BARS:
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
# PORTFOLIO SIMULATOR WITH TOGGLEABLE FILTERS
# ============================================================

def simulate_portfolio(signals, tpsl_map, opens, highs, lows, closes, n_bars,
                       atr_ratio, ema_slope, oos_start, oos_end,
                       # Filter toggles (True = active)
                       use_dir_cap=False,
                       use_regime_sizing=False,
                       use_agg_risk=False,
                       use_momentum=False,
                       use_mdd_sizing=False,
                       use_timeout=False,
                       ):
    """Multi-position portfolio simulator with individually toggleable filters."""
    SIZE_PCT = 100.0 / MAX_POSITIONS
    fee = FEE_PCT * LEVERAGE

    positions = []
    trades = []
    equity = 100.0
    peak_equity = 100.0

    blocked = {k: 0 for k in FILTER_NAMES}

    # H4: Momentum cooldown tracker
    momentum_pause_until = {'LONG': -1, 'SHORT': -1}

    signals_in_range = [(s, p, d) for s, p, d in signals if oos_start <= s < oos_end]
    signals_sorted = sorted(signals_in_range, key=lambda x: x[0])
    sig_idx = 0

    for bar in range(oos_start, oos_end):
        # 1. Check exits
        closed_slots = []
        bar_pnl_sum = 0.0

        for pos in positions:
            result = _check_exit(pos, bar, opens, highs, lows, n_bars,
                                 atr_ratio, fee, use_timeout=use_timeout)
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

        positions = [p for p in positions if p['slot'] not in closed_slots]

        equity += bar_pnl_sum
        if equity > peak_equity:
            peak_equity = equity

        # Momentum guard: check for cooldown trigger (F4)
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

            if len(positions) >= MAX_POSITIONS:
                continue

            # F1: Direction Cap
            if use_dir_cap:
                dir_count = sum(1 for p in positions if p['direction'] == direction)
                if dir_count >= 7:
                    blocked['dir_cap'] += 1
                    continue

            # Duplicate pattern check (always active)
            if any(p['pattern'] == pat for p in positions):
                continue

            entry_bar = sig_bar + 1
            if entry_bar >= n_bars:
                continue
            tp_sl = tpsl_map.get(pat)
            if tp_sl is None:
                continue

            # F4: Momentum guard
            if use_momentum and bar < momentum_pause_until[direction]:
                blocked['momentum'] += 1
                continue

            # F2: Regime sizing
            sm = 1.0
            if use_regime_sizing and bar < len(ema_slope):
                s = ema_slope[bar]
                if s > 0 and direction == 'SHORT':
                    sm = 0.3
                elif s <= 0 and direction == 'LONG':
                    sm = 0.3

            # F5: MDD sizing
            if use_mdd_sizing and peak_equity > 0:
                dd_pct = (peak_equity - equity) / peak_equity * 100
                if dd_pct > MDD_FULL_SIZE_BELOW_DD:
                    # Linear interpolation: 5%→100%, 15%→25%
                    ratio = min(1.0, (dd_pct - MDD_FULL_SIZE_BELOW_DD) /
                                (MDD_MIN_SIZE_ABOVE_DD - MDD_FULL_SIZE_BELOW_DD))
                    mdd_mult = 1.0 - ratio * (1.0 - MDD_MIN_SIZE_PCT / 100)
                    sm *= mdd_mult

            # F3: Aggregate risk cap
            if use_agg_risk and bar < len(ema_slope):
                is_uptrend = ema_slope[bar] > 0
                is_counter = (is_uptrend and direction == 'SHORT') or \
                             (not is_uptrend and direction == 'LONG')
                cap_pct = 3.0 if is_counter else 7.0

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
                        existing_exposure += p_eff_sl * (1.0 / MAX_POSITIONS) * LEVERAGE * p_sm

                new_r = 1.0
                if atr_ratio is not None and sig_bar < len(atr_ratio) and \
                        not np.isnan(atr_ratio[sig_bar]):
                    new_r = max(CLAMP_LO, min(CLAMP_HI, atr_ratio[sig_bar]))
                new_eff_sl = tp_sl[1] * new_r
                new_exposure = new_eff_sl * (1.0 / MAX_POSITIONS) * LEVERAGE * sm

                if existing_exposure + new_exposure > cap_pct:
                    blocked['agg_risk'] += 1
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

    return trades, blocked


# ============================================================
# METRICS
# ============================================================

def compute_metrics(trades):
    if not trades:
        return {'trades': 0, 'wr': 0.0, 'pnl': 0.0, 'mdd': 0.0,
                'pnl_mdd': 0.0, 'pnl_per_trade': 0.0,
                'long_trades': 0, 'short_trades': 0}

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
    long_trades = sum(1 for t in trades if t['direction'] == 'LONG')
    short_trades = sum(1 for t in trades if t['direction'] == 'SHORT')

    return {
        'trades': len(trades),
        'wr': round(wr, 1),
        'pnl': round(total_pnl, 2),
        'mdd': round(max_dd, 2),
        'pnl_mdd': round(pnl_mdd, 2),
        'pnl_per_trade': round(total_pnl / len(trades), 3) if trades else 0,
        'long_trades': long_trades,
        'short_trades': short_trades,
    }


# ============================================================
# WALK-FORWARD (3-fold expanding window)
# ============================================================

def run_wf(signals, tpsl_map, opens, highs, lows, closes, n_bars,
           atr_ratio, ema_slope, label="", **sim_kwargs):
    n_folds = 3
    seg_size = n_bars // (n_folds + 1)

    fold_results = []
    total_blocked = {k: 0 for k in FILTER_NAMES}

    for fold in range(n_folds):
        oos_start = (fold + 1) * seg_size
        oos_end = (fold + 2) * seg_size if fold < n_folds - 1 else n_bars

        trades, blocked = simulate_portfolio(
            signals, tpsl_map, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, oos_start, oos_end,
            **sim_kwargs,
        )
        m = compute_metrics(trades)
        fold_results.append({'fold': fold + 1, **m})
        for k, v in blocked.items():
            total_blocked[k] = total_blocked.get(k, 0) + v

    total_pnl = sum(f['pnl'] for f in fold_results)
    max_mdd = max(f['mdd'] for f in fold_results) if fold_results else 0
    n_positive = sum(1 for f in fold_results if f['pnl'] > 0)
    pnl_mdd = total_pnl / max_mdd if max_mdd > 0 else 0
    total_trades = sum(f['trades'] for f in fold_results)

    return {
        'label': label,
        'folds': fold_results,
        'total_pnl': round(total_pnl, 2),
        'max_mdd': round(max_mdd, 2),
        'pnl_mdd': round(pnl_mdd, 2),
        'total_trades': total_trades,
        'wf_pass': f"{n_positive}/{n_folds}",
        'wf_passed': n_positive == n_folds,
        'total_blocked': total_blocked,
    }


# ============================================================
# FULL SAMPLE TEST
# ============================================================

def run_full_sample(signals, tpsl_map, opens, highs, lows, closes, n_bars,
                    atr_ratio, ema_slope, label="", **sim_kwargs):
    start = max(ATR_WARMUP, 2)
    trades, blocked = simulate_portfolio(
        signals, tpsl_map, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, start, n_bars,
        **sim_kwargs,
    )
    m = compute_metrics(trades)
    return {'label': label, **m, 'blocked': blocked}


# ============================================================
# DISPLAY HELPERS
# ============================================================

def _print_row(r, prefix=""):
    filters_on = r.get('filters_on', '')
    pnl_mdd_str = f"{r['pnl_mdd']:>7.2f}" if r['pnl_mdd'] > 0 else "   N/A"
    print(f"  {prefix}{r['label']:<42s} | "
          f"T={r['trades']:>4d} WR={r['wr']:>5.1f}% "
          f"PnL={r['pnl']:>+8.1f}% MDD={r['mdd']:>5.1f}% "
          f"PnL/MDD={pnl_mdd_str} "
          f"L={r.get('long_trades', 0):>3d} S={r.get('short_trades', 0):>3d}")


def _print_wf_row(r, prefix=""):
    status = "PASS" if r['wf_passed'] else "FAIL"
    print(f"  {prefix}{r['label']:<42s} | "
          f"WF {r['wf_pass']} {status}  "
          f"T={r['total_trades']:>4d} "
          f"PnL={r['total_pnl']:>+8.1f}% MDD={r['max_mdd']:>5.1f}% "
          f"PnL/MDD={r['pnl_mdd']:>7.2f}")


def filter_combo_label(combo):
    """Generate label from filter combination tuple."""
    active = [FILTER_LABELS[FILTER_NAMES[i]] for i, on in enumerate(combo) if on]
    if not active:
        return "NO_FILTERS"
    return "+".join(active)


def combo_to_kwargs(combo):
    """Convert combo tuple (0/1 x6) to simulator kwargs."""
    return {
        'use_dir_cap': bool(combo[0]),
        'use_regime_sizing': bool(combo[1]),
        'use_agg_risk': bool(combo[2]),
        'use_momentum': bool(combo[3]),
        'use_mdd_sizing': bool(combo[4]),
        'use_timeout': bool(combo[5]),
    }


# ============================================================
# MAIN STUDY
# ============================================================

def main():
    t0 = time.time()
    print("=" * 80)
    print("Neutral Window Filter Ablation Study")
    print("  51 patterns (22L+29S), N=9, Hedge mode")
    print("  Systematic individual + combination filter testing")
    print("=" * 80)

    # Load data
    df, types, opens, highs, lows, closes, n_bars = load_data()
    pL, pS, tpsl = load_patterns()
    atr_ratio = compute_atr_ratio(highs, lows, closes)
    ema_slope = compute_ema_slope(closes)
    signals = generate_signals(types, pL, pS, n_bars)
    print(f"Total signals: {len(signals)} "
          f"(L={sum(1 for _, _, d in signals if d == 'LONG')}, "
          f"S={sum(1 for _, _, d in signals if d == 'SHORT')})")
    print()

    all_results = {}

    # ================================================================
    # PHASE 1: Full-sample individual filter tests
    # ================================================================
    print("=" * 80)
    print("PHASE 1: Full-Sample Individual Filter Effects")
    print("=" * 80)
    print()

    # Baseline: no filters
    no_filter = run_full_sample(
        signals, tpsl, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, label="BASELINE (no filters)",
    )
    _print_row(no_filter)
    all_results['baseline_no_filter'] = no_filter

    # All filters ON (current production)
    all_on = run_full_sample(
        signals, tpsl, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, label="ALL FILTERS ON (production)",
        use_dir_cap=True, use_regime_sizing=True, use_agg_risk=True,
        use_momentum=True, use_mdd_sizing=True, use_timeout=True,
    )
    _print_row(all_on)
    all_results['all_filters_on'] = all_on

    print()
    print("--- Individual Filters (each alone) ---")

    individual_results = {}
    for i, fname in enumerate(FILTER_NAMES):
        combo = [False] * 6
        combo[i] = True
        kwargs = combo_to_kwargs(tuple(combo))
        label = f"ONLY {FILTER_LABELS[fname]}"
        r = run_full_sample(
            signals, tpsl, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, label=label, **kwargs,
        )
        _print_row(r)
        individual_results[fname] = r
        all_results[f'individual_{fname}'] = r

    print()
    print("--- Individual Filter Impact vs No-Filter Baseline ---")
    base_pnl = no_filter['pnl']
    base_mdd = no_filter['mdd']
    base_ratio = no_filter['pnl_mdd']
    for fname in FILTER_NAMES:
        r = individual_results[fname]
        dpnl = r['pnl'] - base_pnl
        dmdd = r['mdd'] - base_mdd
        dratio = r['pnl_mdd'] - base_ratio
        verdict = "IMPROVE" if dratio > 0 else "WORSEN"
        print(f"  {FILTER_LABELS[fname]:<14s}: "
              f"dPnL={dpnl:>+7.1f}% dMDD={dmdd:>+5.1f}% "
              f"dPnL/MDD={dratio:>+6.2f} [{verdict}]")

    # ================================================================
    # PHASE 2: Full factorial (2^6 = 64 combinations)
    # ================================================================
    print()
    print("=" * 80)
    print("PHASE 2: Full Factorial — All 64 Combinations (Full-Sample)")
    print("=" * 80)
    print()

    combo_results = []
    total_combos = 2 ** len(FILTER_NAMES)
    print(f"Testing {total_combos} combinations...")

    for bits in range(total_combos):
        combo = tuple((bits >> i) & 1 for i in range(len(FILTER_NAMES)))
        kwargs = combo_to_kwargs(combo)
        label = filter_combo_label(combo)
        n_active = sum(combo)

        r = run_full_sample(
            signals, tpsl, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, label=label, **kwargs,
        )
        r['combo'] = combo
        r['n_filters'] = n_active
        r['filters_on'] = label
        combo_results.append(r)

    # Sort by PnL/MDD (descending)
    combo_results.sort(key=lambda x: x['pnl_mdd'], reverse=True)

    print(f"\n--- TOP 15 by PnL/MDD ---")
    for rank, r in enumerate(combo_results[:15], 1):
        print(f"  #{rank:>2d} [{r['n_filters']}F] ", end="")
        _print_row(r)

    print(f"\n--- BOTTOM 5 by PnL/MDD ---")
    for rank, r in enumerate(combo_results[-5:], total_combos - 4):
        print(f"  #{rank:>2d} [{r['n_filters']}F] ", end="")
        _print_row(r)

    all_results['phase2_top15'] = [
        {'rank': i + 1, 'label': r['label'], 'combo': r['combo'],
         'n_filters': r['n_filters'], 'pnl': r['pnl'], 'mdd': r['mdd'],
         'pnl_mdd': r['pnl_mdd'], 'trades': r['trades'], 'wr': r['wr'],
         'long_trades': r.get('long_trades', 0),
         'short_trades': r.get('short_trades', 0)}
        for i, r in enumerate(combo_results[:15])
    ]

    # ================================================================
    # PHASE 3: WF Validation on Top Candidates
    # ================================================================
    print()
    print("=" * 80)
    print("PHASE 3: Walk-Forward Validation (Top 10 + Current Production)")
    print("=" * 80)
    print()

    wf_candidates = []

    # Always test: no filters, all filters (production), top 10 by PnL/MDD
    seen_combos = set()
    must_test = [
        ((0, 0, 0, 0, 0, 0), "WF: NO_FILTERS"),
        ((1, 1, 1, 1, 1, 1), "WF: ALL_FILTERS (production)"),
    ]
    for combo, label in must_test:
        seen_combos.add(combo)

    for r in combo_results[:10]:
        c = r['combo']
        if c not in seen_combos:
            seen_combos.add(c)
            must_test.append((c, f"WF: {r['label']}"))

    wf_results = []
    for combo, label in must_test:
        kwargs = combo_to_kwargs(combo)
        wf = run_wf(
            signals, tpsl, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, label=label, **kwargs,
        )
        wf['combo'] = combo
        wf['n_filters'] = sum(combo)
        _print_wf_row(wf)
        wf_results.append(wf)

    # Sort WF results: PASS first, then by PnL/MDD
    wf_passed = [w for w in wf_results if w['wf_passed']]
    wf_failed = [w for w in wf_results if not w['wf_passed']]
    wf_passed.sort(key=lambda x: x['pnl_mdd'], reverse=True)
    wf_failed.sort(key=lambda x: x['pnl_mdd'], reverse=True)

    print()
    print("--- WF Results Summary ---")
    print(f"  PASSED: {len(wf_passed)}/{len(wf_results)}")
    if wf_passed:
        print(f"\n  PASS (sorted by PnL/MDD):")
        for i, w in enumerate(wf_passed, 1):
            _print_wf_row(w, prefix=f"#{i} ")
    if wf_failed:
        print(f"\n  FAIL:")
        for w in wf_failed:
            _print_wf_row(w, prefix="   ")

    all_results['phase3_wf'] = [
        {'label': w['label'], 'combo': w['combo'], 'n_filters': w['n_filters'],
         'wf_pass': w['wf_pass'], 'wf_passed': w['wf_passed'],
         'total_pnl': w['total_pnl'], 'max_mdd': w['max_mdd'],
         'pnl_mdd': w['pnl_mdd'], 'total_trades': w['total_trades'],
         'folds': w['folds'], 'total_blocked': w['total_blocked']}
        for w in wf_passed + wf_failed
    ]

    # ================================================================
    # PHASE 4: Summary and Recommendation
    # ================================================================
    print()
    print("=" * 80)
    print("PHASE 4: Summary and Recommendation")
    print("=" * 80)
    print()

    # Best WF-passed combination
    if wf_passed:
        best = wf_passed[0]
        print(f"BEST WF-PASSED: {best['label']}")
        print(f"  PnL/MDD: {best['pnl_mdd']:.2f}")
        print(f"  Total PnL: {best['total_pnl']:+.1f}%")
        print(f"  Max MDD: {best['max_mdd']:.1f}%")
        print(f"  Total Trades: {best['total_trades']}")
        print(f"  WF: {best['wf_pass']}")
        print(f"  Filters: {filter_combo_label(best['combo'])}")
        print(f"  Blocked: {best['total_blocked']}")

        # Compare with production (all filters)
        prod_wf = next((w for w in wf_results if w['combo'] == (1, 1, 1, 1, 1, 1)), None)
        if prod_wf:
            print(f"\n  vs PRODUCTION (all filters):")
            print(f"    PnL/MDD: {best['pnl_mdd']:.2f} vs {prod_wf['pnl_mdd']:.2f} "
                  f"({best['pnl_mdd'] - prod_wf['pnl_mdd']:+.2f})")
            print(f"    PnL: {best['total_pnl']:+.1f}% vs {prod_wf['total_pnl']:+.1f}%")
            print(f"    MDD: {best['max_mdd']:.1f}% vs {prod_wf['max_mdd']:.1f}%")
    else:
        print("WARNING: No combination passed WF!")

    # Filter contribution analysis
    print()
    print("--- Filter Contribution (avg PnL/MDD when ON vs OFF across all 64 combos) ---")
    for i, fname in enumerate(FILTER_NAMES):
        on_ratios = [r['pnl_mdd'] for r in combo_results if r['combo'][i] == 1]
        off_ratios = [r['pnl_mdd'] for r in combo_results if r['combo'][i] == 0]
        avg_on = np.mean(on_ratios) if on_ratios else 0
        avg_off = np.mean(off_ratios) if off_ratios else 0
        delta = avg_on - avg_off
        verdict = "+" if delta > 0 else "-"
        print(f"  {FILTER_LABELS[fname]:<14s}: ON={avg_on:>6.2f}  OFF={avg_off:>6.2f}  "
              f"delta={delta:>+6.2f} [{verdict}]")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")

    # Save results
    all_results['metadata'] = {
        'study': 'neutral_filter_ablation',
        'patterns': f"{len(pL)}L+{len(pS)}S={len(pL)+len(pS)}",
        'data_bars': n_bars,
        'data_days': round(n_bars / BARS_PER_DAY, 1),
        'total_signals': len(signals),
        'elapsed_sec': round(elapsed, 1),
    }

    class _NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            return super().default(obj)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)
    print(f"Results saved: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
