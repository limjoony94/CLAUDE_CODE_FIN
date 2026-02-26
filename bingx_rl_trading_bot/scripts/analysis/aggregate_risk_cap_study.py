#!/usr/bin/env python3
"""
Aggregate Directional Risk Cap Study
=====================================
Problem: During a +5.8% BTC rally (02-24~02-26), 8 SHORT positions hit SL
simultaneously at ~$66,705, causing -3.67% portfolio loss. The correlated risk
from same-direction positions with similar SL levels is the core issue.

Hypotheses:
  H1 (Static Cap): Limit total SL exposure per direction to a fixed % of portfolio.
      When a new entry would push aggregate SL exposure above the cap, skip the entry.
  H2 (Dynamic Cap): Same as H1 but cap varies by EMA(20) regime:
      counter-regime (SHORT in uptrend / LONG in downtrend) -> tighter cap
      with-regime -> looser cap

Validation: 3-fold expanding window WF on 270d CSV data.

Standard Research Protocol:
- Entry: signal bar+1 open
- Exit: intrabar high/low (distance-based from bar open, NOT entry price)
- Same-bar resolution: abs(tp - opens[j]) vs abs(sl - opens[j])
- Sizing: Compound (reinvest), 1/N per slot where N=max_positions=9
- Fee: 0.05% x 2 x LEVERAGE(3) = 0.30% per trade
- Slippage buffer: 0.02% (TP adjusted up, SL adjusted down)
- Timeout: 864 bars (72h) -> DROP from PnL
- ATR-scaled TP/SL: ATR(14) / rolling_median(ATR(14), 576), clamp [0.6, 1.7]
- Pattern classification: use type_code column from CSV (short codes)
- Patterns: load from results/dynamic_patterns.json (51 patterns with per-pattern TP/SL)
- Direction cap: 8 (max same-direction positions)
- WF: 3-fold expanding window (OOS fixed at 1/3 of total)
"""

import sys
import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime

# ─── Constants ───
LEVERAGE = 3
FEE_PCT = 0.10           # 0.10% roundtrip
SLIPPAGE_BUFFER = 0.02   # 0.02% buffer
TIMEOUT_BARS = 864       # 72h
MAX_POSITIONS = 9
DIRECTION_CAP = 8        # existing cap from v1.35.1
ATR_PERIOD = 14
ATR_WINDOW = 576
CLAMP_LO = 0.6
CLAMP_HI = 1.7
ATR_WARMUP = ATR_PERIOD + ATR_WINDOW  # 590 bars

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CSV_PATH = os.path.join(BASE_DIR, 'data', 'btc_5m_270days_reclassified.csv')
PATTERNS_PATH = os.path.join(BASE_DIR, 'results', 'dynamic_patterns.json')
RESULTS_PATH = os.path.join(BASE_DIR, 'results', 'aggregate_risk_cap_study.json')


# ─── ATR ───
def compute_atr_ratio(highs, lows, closes):
    """ATR(14) / rolling_median(ATR(14), 576), clamp [0.6, 1.7]."""
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


# ─── EMA Regime ───
def compute_ema_regime(closes, span=20, lookback=5):
    """EMA(20) slope-based regime: 'UP' or 'DOWN'.
    Returns array of regime strings, same length as closes."""
    ema = pd.Series(closes).ewm(span=span, adjust=False).mean().values
    regimes = ['DOWN'] * len(closes)  # default
    for i in range(lookback, len(closes)):
        slope = ema[i] - ema[i - lookback]
        regimes[i] = 'UP' if slope > 0 else 'DOWN'
    return regimes


# ─── Aggregate SL exposure ───
def calc_aggregate_sl_exposure(open_positions, direction):
    """Sum of SL exposure for all open positions in given direction.
    SL exposure per position = SL_distance_pct * slot_size * leverage
    where slot_size = 1/N (compound: fraction of current equity).

    Note: In compound mode, each slot is 1/N of *current* equity.
    SL distance is the effective (ATR-scaled) SL distance, already computed
    at entry time. We store eff_sl_pct on each position.
    """
    total = 0.0
    for pos in open_positions:
        if pos['direction'] == direction:
            slot_size = 1.0 / MAX_POSITIONS
            total += pos['eff_sl_pct'] * slot_size * LEVERAGE
    return total


def would_exceed_cap(open_positions, new_eff_sl_pct, direction, cap_pct):
    """Check if adding a new position would push aggregate SL above cap."""
    if cap_pct is None:
        return False
    current = calc_aggregate_sl_exposure(open_positions, direction)
    new_exposure = new_eff_sl_pct * (1.0 / MAX_POSITIONS) * LEVERAGE
    return (current + new_exposure) > cap_pct


# ─── Data Loading ───
def load_csv_data():
    """Load 270d CSV. Returns DataFrame, list of type codes."""
    df = pd.read_csv(CSV_PATH)
    df['datetime'] = pd.to_datetime(df['timestamp'])
    rctypes = df['type_code'].values.tolist()
    return df, rctypes


def load_patterns():
    """Load dynamic_patterns.json. Returns sets for long/short, tpsl dict, details dict."""
    with open(PATTERNS_PATH) as f:
        pat_data = json.load(f)
    p_long = set(pat_data['patterns']['long'])
    p_short = set(pat_data['patterns']['short'])
    tpsl = pat_data['patterns_tpsl']  # {pattern: [tp, sl]}
    details = pat_data.get('pattern_details', {})
    return p_long, p_short, tpsl, details


# ─── Signal Generation ───
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


# ─── Exit Check ───
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

    # Same-bar resolution: distance from bar OPEN
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
    pnl -= FEE_PCT * LEVERAGE

    return {
        'entry_bar': entry_bar, 'exit_bar': bar, 'hold_bars': hold,
        'entry_price': entry, 'exit_price': exit_price,
        'pnl_slot': pnl, 'reason': reason, 'dropped': False,
    }


# ─── Portfolio Simulation ───
def simulate_portfolio(signals, tpsl_map, opens, highs, lows, closes,
                       n_bars, atr_ratio, regimes, scenario, oos_start, oos_end):
    """Simulate N=9 multi-position portfolio on OOS period.

    scenario dict keys:
      agg_cap: float or None (static cap, H1)
      counter_cap: float (counter-regime cap, H2)
      with_cap: float (with-regime cap, H2)

    Returns: dict with trades, skip_count, cascade_events, max_agg_exposure, etc.
    """
    positions = []
    trades = []
    skipped = 0
    max_agg_exposure_long = 0.0
    max_agg_exposure_short = 0.0
    max_simultaneous_long = 0
    max_simultaneous_short = 0
    cascade_events = 0  # 2+ same-direction SL hits on same bar
    SIZE_PCT = 100.0 / MAX_POSITIONS

    # Pre-filter signals to OOS range
    signals_in_range = [(s, p, d) for s, p, d in signals if oos_start <= s < oos_end]
    signals_sorted = sorted(signals_in_range, key=lambda x: x[0])
    sig_idx = 0

    # Determine cap mode
    is_static = 'agg_cap' in scenario
    static_cap = scenario.get('agg_cap')
    counter_cap = scenario.get('counter_cap')
    with_cap = scenario.get('with_cap')

    for bar in range(oos_start, oos_end):
        # ── Check exits ──
        closed_slots = []
        sl_hits_this_bar = {'LONG': 0, 'SHORT': 0}

        for pos in positions:
            result = _check_exit(pos, bar, opens, highs, lows, n_bars)
            if result is not None:
                result['pattern'] = pos['pattern']
                result['direction'] = pos['direction']
                if not result['dropped']:
                    result['pnl_portfolio'] = result['pnl_slot'] * SIZE_PCT / 100
                    trades.append(result)
                    if result['reason'] == 'SL':
                        sl_hits_this_bar[pos['direction']] += 1
                # Timeout positions: drop entirely (not added to trades)
                closed_slots.append(pos['slot'])

        positions = [p for p in positions if p['slot'] not in closed_slots]

        # Cascade event: 2+ same-direction SL on same bar
        for d in ('LONG', 'SHORT'):
            if sl_hits_this_bar[d] >= 2:
                cascade_events += 1

        # Track max simultaneous positions
        n_long = sum(1 for p in positions if p['direction'] == 'LONG')
        n_short = sum(1 for p in positions if p['direction'] == 'SHORT')
        max_simultaneous_long = max(max_simultaneous_long, n_long)
        max_simultaneous_short = max(max_simultaneous_short, n_short)

        # Track max aggregate exposure
        agg_long = calc_aggregate_sl_exposure(positions, 'LONG')
        agg_short = calc_aggregate_sl_exposure(positions, 'SHORT')
        max_agg_exposure_long = max(max_agg_exposure_long, agg_long)
        max_agg_exposure_short = max(max_agg_exposure_short, agg_short)

        # ── Process signals ──
        while sig_idx < len(signals_sorted) and signals_sorted[sig_idx][0] == bar:
            sig_bar, pat, direction = signals_sorted[sig_idx]
            sig_idx += 1

            # Standard constraints: max positions, direction cap, no duplicate pattern
            if len(positions) >= MAX_POSITIONS:
                continue
            dir_count = sum(1 for p in positions if p['direction'] == direction)
            if dir_count >= DIRECTION_CAP:
                continue
            if any(p['pattern'] == pat for p in positions):
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
            if atr_ratio is not None and not np.isnan(atr_ratio[sig_bar]):
                r = max(CLAMP_LO, min(CLAMP_HI, atr_ratio[sig_bar]))
            else:
                r = 1.0

            eff_tp = tp_pct_base * r + SLIPPAGE_BUFFER
            eff_sl = max(0.1, sl_pct_base * r - SLIPPAGE_BUFFER)

            # ── Aggregate risk cap check ──
            if is_static:
                cap = static_cap
            else:
                # Dynamic: determine counter-regime vs with-regime
                regime = regimes[sig_bar] if sig_bar < len(regimes) else 'DOWN'
                is_counter = (
                    (direction == 'SHORT' and regime == 'UP') or
                    (direction == 'LONG' and regime == 'DOWN')
                )
                cap = counter_cap if is_counter else with_cap

            if would_exceed_cap(positions, eff_sl, direction, cap):
                skipped += 1
                continue

            # ── Open position ──
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
                'slot': f"{pat}_{sig_bar}",
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
            })

    # Force-close remaining at OOS end -> DROP (do not count in PnL)
    # These are incomplete trades; consistent with timeout=DROP approach

    return {
        'trades': trades,
        'skipped': skipped,
        'cascade_events': cascade_events,
        'max_agg_exposure_long': max_agg_exposure_long,
        'max_agg_exposure_short': max_agg_exposure_short,
        'max_simultaneous_long': max_simultaneous_long,
        'max_simultaneous_short': max_simultaneous_short,
        'remaining_positions': len(positions),
    }


# ─── Metrics ───
def compute_metrics(sim_result):
    """Compute portfolio metrics from simulation result."""
    trades = sim_result['trades']
    if not trades:
        return {
            'trades': 0, 'skipped': sim_result['skipped'],
            'wr': 0.0, 'pnl': 0.0, 'mdd': 0.0, 'pnl_mdd': 0.0,
            'long_trades': 0, 'short_trades': 0,
            'long_pnl': 0.0, 'short_pnl': 0.0,
            'cascade_events': sim_result['cascade_events'],
            'max_agg_exposure_long': sim_result['max_agg_exposure_long'],
            'max_agg_exposure_short': sim_result['max_agg_exposure_short'],
            'max_simul_long': sim_result['max_simultaneous_long'],
            'max_simul_short': sim_result['max_simultaneous_short'],
            'avg_win': 0.0, 'avg_loss': 0.0, 'pnl_per_trade': 0.0,
            'sl_trades': 0, 'tp_trades': 0, 'timeout_trades': 0,
        }

    wins = [t for t in trades if t['pnl_slot'] > 0]
    losses = [t for t in trades if t['pnl_slot'] <= 0]
    wr = len(wins) / len(trades) * 100 if trades else 0

    # Compound MDD: multiplicative equity curve
    sorted_trades = sorted(trades, key=lambda x: (x['exit_bar'], x['entry_bar']))
    equity = 100.0
    peak = equity
    max_dd = 0.0
    for t in sorted_trades:
        # Compound: PnL is on slot_size fraction of equity
        slot_return = t['pnl_slot'] / 100  # e.g., +5% -> 0.05
        slot_frac = 1.0 / MAX_POSITIONS
        # Portfolio equity change from this trade
        equity *= (1 + slot_return * slot_frac)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd:
            max_dd = dd

    total_pnl = (equity / 100.0 - 1) * 100  # compound PnL

    long_t = [t for t in trades if t['direction'] == 'LONG']
    short_t = [t for t in trades if t['direction'] == 'SHORT']

    # Additive PnL breakdown for direction analysis
    long_pnl_add = sum(t['pnl_slot'] / MAX_POSITIONS for t in long_t)
    short_pnl_add = sum(t['pnl_slot'] / MAX_POSITIONS for t in short_t)

    sl_count = sum(1 for t in trades if t['reason'] == 'SL')
    tp_count = sum(1 for t in trades if t['reason'] == 'TP')
    timeout_count = sum(1 for t in trades if t['reason'] == 'TIMEOUT')

    pnl_mdd = total_pnl / max_dd if max_dd > 0 else 0.0

    return {
        'trades': len(trades),
        'skipped': sim_result['skipped'],
        'wr': wr,
        'pnl': total_pnl,
        'mdd': max_dd,
        'pnl_mdd': pnl_mdd,
        'long_trades': len(long_t),
        'short_trades': len(short_t),
        'long_pnl': long_pnl_add,
        'short_pnl': short_pnl_add,
        'cascade_events': sim_result['cascade_events'],
        'max_agg_exposure_long': sim_result['max_agg_exposure_long'],
        'max_agg_exposure_short': sim_result['max_agg_exposure_short'],
        'max_simul_long': sim_result['max_simultaneous_long'],
        'max_simul_short': sim_result['max_simultaneous_short'],
        'avg_win': np.mean([t['pnl_slot'] for t in wins]) if wins else 0.0,
        'avg_loss': np.mean([t['pnl_slot'] for t in losses]) if losses else 0.0,
        'pnl_per_trade': total_pnl / len(trades) if trades else 0.0,
        'sl_trades': sl_count,
        'tp_trades': tp_count,
        'timeout_trades': timeout_count,
    }


# ─── Scenarios ───
SCENARIOS = {
    'baseline_no_cap': {'agg_cap': None},
    'static_3pct': {'agg_cap': 3.0},
    'static_4pct': {'agg_cap': 4.0},
    'static_5pct': {'agg_cap': 5.0},
    'static_6pct': {'agg_cap': 6.0},
    'static_7pct': {'agg_cap': 7.0},
    # Dynamic (regime-aware): counter-regime cap / with-regime cap
    'dynamic_3_7': {'counter_cap': 3.0, 'with_cap': 7.0},
    'dynamic_4_7': {'counter_cap': 4.0, 'with_cap': 7.0},
    'dynamic_3_6': {'counter_cap': 3.0, 'with_cap': 6.0},
}


def main():
    t0 = time.time()
    print("=" * 100)
    print("AGGREGATE DIRECTIONAL RISK CAP STUDY")
    print("=" * 100)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Protocol: compound sizing, fee {FEE_PCT}%x2xLev={FEE_PCT*LEVERAGE:.2f}%, "
          f"slip {SLIPPAGE_BUFFER}%, timeout {TIMEOUT_BARS}b=DROP")
    print(f"ATR: period={ATR_PERIOD}, window={ATR_WINDOW}, clamp=[{CLAMP_LO},{CLAMP_HI}]")
    print(f"Positions: N={MAX_POSITIONS}, dir_cap={DIRECTION_CAP}, leverage={LEVERAGE}x")

    # ── 1. Load patterns ──
    p_long, p_short, tpsl, details = load_patterns()
    print(f"\nPatterns: {len(p_long)}L + {len(p_short)}S = {len(p_long) + len(p_short)}")

    # ── 2. Load CSV ──
    print("\nLoading 270d CSV...")
    df, rctypes = load_csv_data()
    n_bars = len(df)
    n_days = n_bars / 288
    print(f"CSV: {n_bars} bars ({n_days:.0f}d), "
          f"{df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")

    opens = df['open'].values.astype(float)
    highs = df['high'].values.astype(float)
    lows = df['low'].values.astype(float)
    closes = df['close'].values.astype(float)

    # ── 3. Compute ATR ratio ──
    print("\nComputing ATR ratio...")
    atr_ratio = compute_atr_ratio(highs, lows, closes)
    valid_pct = np.sum(~np.isnan(atr_ratio)) / len(atr_ratio) * 100
    print(f"ATR ratio: {valid_pct:.1f}% valid (warmup={ATR_WARMUP} bars)")

    # ── 4. Compute EMA regime ──
    print("Computing EMA(20) regime...")
    regimes = compute_ema_regime(closes, span=20, lookback=5)
    up_pct = sum(1 for r in regimes if r == 'UP') / len(regimes) * 100
    print(f"Regime: UP {up_pct:.1f}%, DOWN {100 - up_pct:.1f}%")

    # ── 5. Generate signals ──
    all_signals = generate_signals(rctypes, p_long, p_short, start_idx=ATR_WARMUP)
    print(f"\nTotal signals: {len(all_signals)} "
          f"(L: {sum(1 for _, _, d in all_signals if d == 'LONG')}, "
          f"S: {sum(1 for _, _, d in all_signals if d == 'SHORT')})")

    # ── 6. Compute theoretical max aggregate exposure ──
    # With dir_cap=8, slot_size=1/9, max SL 4.76%*1.7(clamp_hi)=8.09%, leverage=3
    # Theoretical max per-direction = 8 * (8.09% * 1/9 * 3) = 21.6%
    # In practice with mean SL ~3.0% and typical ATR ratio ~1.0: 8 * (3.0% * 1/9 * 3) = 8.0%
    print(f"\nTheoretical max single-direction SL exposure:")
    sl_values = [tpsl[p][1] for p in tpsl]
    mean_sl = np.mean(sl_values)
    max_sl = max(sl_values)
    print(f"  Mean base SL: {mean_sl:.2f}%, Max base SL: {max_sl:.2f}%")
    print(f"  dir_cap=8: mean agg = 8 * {mean_sl:.2f}% * (1/9) * 3 = {8 * mean_sl / 9 * 3:.2f}%")
    print(f"  dir_cap=8: max agg  = 8 * {max_sl:.2f}% * (1/9) * 3 = {8 * max_sl / 9 * 3:.2f}%")

    # ── 7. Define WF folds: 3-fold expanding window ──
    tradeable_start = ATR_WARMUP
    tradeable_bars = n_bars - tradeable_start
    fold_size = tradeable_bars // 3  # OOS is 1/3 of total each

    folds = []
    for i in range(3):
        oos_start = tradeable_start + i * fold_size
        oos_end = tradeable_start + (i + 1) * fold_size if i < 2 else n_bars
        folds.append((oos_start, oos_end))

    print(f"\nWF Folds: 3 x ~{fold_size / 288:.0f}d")
    for i, (s, e) in enumerate(folds):
        print(f"  Fold {i + 1}: bars {s}-{e} ({(e - s) / 288:.0f}d) "
              f"| {df['datetime'].iloc[s]} ~ {df['datetime'].iloc[min(e - 1, n_bars - 1)]}")

    # ── 8. Run all scenarios across all folds ──
    all_results = {}

    for scenario_name, scenario_cfg in SCENARIOS.items():
        print(f"\n--- Running: {scenario_name} ---")
        fold_metrics = []

        for fold_idx, (oos_start, oos_end) in enumerate(folds):
            sim = simulate_portfolio(
                all_signals, tpsl, opens, highs, lows, closes,
                n_bars, atr_ratio, regimes, scenario_cfg, oos_start, oos_end
            )
            m = compute_metrics(sim)
            fold_metrics.append(m)
            print(f"  Fold {fold_idx + 1}: {m['trades']} trades (skip {m['skipped']}), "
                  f"WR {m['wr']:.1f}%, PnL {m['pnl']:+.2f}%, MDD {m['mdd']:.2f}%, "
                  f"cascades {m['cascade_events']}, "
                  f"max_agg L={m['max_agg_exposure_long']:.2f}%/S={m['max_agg_exposure_short']:.2f}%")

        all_results[scenario_name] = fold_metrics

    # ── 9. Print detailed fold-by-fold results ──
    print(f"\n{'=' * 120}")
    print("FOLD-BY-FOLD RESULTS")
    print('=' * 120)

    for scenario_name in SCENARIOS:
        print(f"\n--- {scenario_name} ---")
        hdr = (f"  {'Fold':<8} {'Trades':>7} {'Skip':>6} {'WR':>7} {'PnL':>9} {'MDD':>7} "
               f"{'PnL/MDD':>8} {'Casc':>5} {'MaxAggL':>8} {'MaxAggS':>8} "
               f"{'MaxSimL':>8} {'MaxSimS':>8} {'TP':>5} {'SL':>5}")
        print(hdr)
        print("  " + "-" * 115)

        for i, m in enumerate(all_results[scenario_name]):
            print(f"  Fold {i + 1}  {m['trades']:>7} {m['skipped']:>6} "
                  f"{m['wr']:>6.1f}% {m['pnl']:>+8.2f}% {m['mdd']:>6.2f}% "
                  f"{m['pnl_mdd']:>7.2f}x {m['cascade_events']:>5} "
                  f"{m['max_agg_exposure_long']:>7.2f}% {m['max_agg_exposure_short']:>7.2f}% "
                  f"{m['max_simul_long']:>8} {m['max_simul_short']:>8} "
                  f"{m['tp_trades']:>5} {m['sl_trades']:>5}")

    # ── 10. Comparison summary table ──
    print(f"\n{'=' * 120}")
    print("COMPARISON SUMMARY (All Folds Aggregated)")
    print('=' * 120)

    header = (f"{'Scenario':<20} {'Trades':>7} {'Skip':>6} {'AvgWR':>7} {'TotPnL':>9} "
              f"{'MaxMDD':>7} {'PnL/MDD':>8} {'Casc':>5} {'MaxAggS':>8} "
              f"{'TP':>5} {'SL':>5} {'WF':>6}")
    print(header)
    print("-" * 115)

    summary_data = {}
    for scenario_name in SCENARIOS:
        folds_m = all_results[scenario_name]
        total_trades = sum(m['trades'] for m in folds_m)
        total_skip = sum(m['skipped'] for m in folds_m)
        avg_wr = np.mean([m['wr'] for m in folds_m if m['trades'] > 0])
        total_pnl = sum(m['pnl'] for m in folds_m)
        max_mdd = max(m['mdd'] for m in folds_m)
        pnl_mdd = total_pnl / max_mdd if max_mdd > 0 else 0
        total_cascade = sum(m['cascade_events'] for m in folds_m)
        max_agg_s = max(m['max_agg_exposure_short'] for m in folds_m)
        total_tp = sum(m['tp_trades'] for m in folds_m)
        total_sl = sum(m['sl_trades'] for m in folds_m)
        positive_folds = sum(1 for m in folds_m if m['pnl'] > 0)
        wf_status = "PASS" if positive_folds == 3 else f"FAIL({positive_folds}/3)"

        print(f"{scenario_name:<20} {total_trades:>7} {total_skip:>6} {avg_wr:>6.1f}% "
              f"{total_pnl:>+8.2f}% {max_mdd:>6.2f}% {pnl_mdd:>7.2f}x "
              f"{total_cascade:>5} {max_agg_s:>7.2f}% "
              f"{total_tp:>5} {total_sl:>5} {wf_status:>6}")

        summary_data[scenario_name] = {
            'total_trades': total_trades,
            'total_skipped': total_skip,
            'avg_wr': round(avg_wr, 2),
            'total_pnl': round(total_pnl, 2),
            'max_mdd': round(max_mdd, 2),
            'pnl_mdd': round(pnl_mdd, 2),
            'total_cascade_events': total_cascade,
            'max_agg_exposure_short': round(max_agg_s, 2),
            'total_tp': total_tp,
            'total_sl': total_sl,
            'wf_pass': positive_folds == 3,
            'positive_folds': positive_folds,
            'folds': [
                {
                    'trades': m['trades'],
                    'skipped': m['skipped'],
                    'wr': round(m['wr'], 2),
                    'pnl': round(m['pnl'], 2),
                    'mdd': round(m['mdd'], 2),
                    'pnl_mdd': round(m['pnl_mdd'], 2),
                    'cascade_events': m['cascade_events'],
                    'max_agg_exposure_long': round(m['max_agg_exposure_long'], 2),
                    'max_agg_exposure_short': round(m['max_agg_exposure_short'], 2),
                    'max_simul_long': m['max_simul_long'],
                    'max_simul_short': m['max_simul_short'],
                    'long_trades': m['long_trades'],
                    'short_trades': m['short_trades'],
                    'long_pnl': round(m['long_pnl'], 2),
                    'short_pnl': round(m['short_pnl'], 2),
                    'tp_trades': m['tp_trades'],
                    'sl_trades': m['sl_trades'],
                }
                for m in folds_m
            ],
        }

    # ── 11. WF Consistency Check ──
    print(f"\n{'=' * 120}")
    print("WF CONSISTENCY CHECK (All 3 Folds Positive PnL?)")
    print('=' * 120)

    for scenario_name in SCENARIOS:
        folds_m = all_results[scenario_name]
        positive = sum(1 for m in folds_m if m['pnl'] > 0)
        status = "PASS" if positive == 3 else "FAIL"
        fold_pnls = " | ".join(f"{m['pnl']:+.2f}%" for m in folds_m)
        cascade_total = sum(m['cascade_events'] for m in folds_m)
        print(f"  {scenario_name:<20} {positive}/3 {status:>4}  [{fold_pnls}]  cascades: {cascade_total}")

    # ── 12. Delta analysis: each scenario vs baseline ──
    print(f"\n{'=' * 120}")
    print("DELTA ANALYSIS vs BASELINE (no_cap)")
    print('=' * 120)

    base_folds = all_results['baseline_no_cap']
    base_total_pnl = sum(m['pnl'] for m in base_folds)
    base_max_mdd = max(m['mdd'] for m in base_folds)
    base_cascades = sum(m['cascade_events'] for m in base_folds)
    base_total_trades = sum(m['trades'] for m in base_folds)

    header_d = (f"{'Scenario':<20} {'dTrades':>8} {'dSkip':>7} {'dPnL':>9} {'dMDD':>8} "
                f"{'dCascade':>8} {'dPnL/MDD':>9}")
    print(header_d)
    print("-" * 80)

    for scenario_name in SCENARIOS:
        if scenario_name == 'baseline_no_cap':
            continue
        folds_m = all_results[scenario_name]
        total_pnl = sum(m['pnl'] for m in folds_m)
        max_mdd = max(m['mdd'] for m in folds_m)
        cascades = sum(m['cascade_events'] for m in folds_m)
        total_trades = sum(m['trades'] for m in folds_m)
        total_skip = sum(m['skipped'] for m in folds_m)

        d_trades = total_trades - base_total_trades
        d_pnl = total_pnl - base_total_pnl
        d_mdd = max_mdd - base_max_mdd
        d_cascade = cascades - base_cascades
        pnl_mdd_new = total_pnl / max_mdd if max_mdd > 0 else 0
        pnl_mdd_base = base_total_pnl / base_max_mdd if base_max_mdd > 0 else 0
        d_pnl_mdd = pnl_mdd_new - pnl_mdd_base

        print(f"{scenario_name:<20} {d_trades:>+8} {total_skip:>7} {d_pnl:>+8.2f}% "
              f"{d_mdd:>+7.2f}% {d_cascade:>+8} {d_pnl_mdd:>+8.2f}x")

    # ── 13. Cascade event deep-dive on baseline ──
    print(f"\n{'=' * 120}")
    print("CASCADE EVENTS ANALYSIS (baseline, showing worst bars)")
    print('=' * 120)

    # Re-run baseline to get detailed trade data for cascade analysis
    all_cascade_bars = []
    for fold_idx, (oos_start, oos_end) in enumerate(folds):
        sim = simulate_portfolio(
            all_signals, tpsl, opens, highs, lows, closes,
            n_bars, atr_ratio, regimes, {'agg_cap': None}, oos_start, oos_end
        )
        # Group SL trades by exit_bar and direction
        sl_trades = [t for t in sim['trades'] if t.get('reason') == 'SL']
        bar_dir_groups = {}
        for t in sl_trades:
            key = (t['exit_bar'], t['direction'])
            if key not in bar_dir_groups:
                bar_dir_groups[key] = []
            bar_dir_groups[key].append(t)

        for (exit_bar, direction), group in bar_dir_groups.items():
            if len(group) >= 2:
                total_loss = sum(t['pnl_slot'] / MAX_POSITIONS for t in group)
                all_cascade_bars.append({
                    'fold': fold_idx + 1,
                    'bar': exit_bar,
                    'date': str(df['datetime'].iloc[exit_bar]) if exit_bar < len(df) else 'N/A',
                    'direction': direction,
                    'n_hits': len(group),
                    'total_loss_pct': total_loss,
                    'patterns': [t['pattern'] for t in group],
                })

    all_cascade_bars.sort(key=lambda x: x['total_loss_pct'])
    print(f"\nTotal cascade events: {len(all_cascade_bars)}")
    print(f"\nWorst 10 cascade bars:")
    print(f"  {'Fold':>4} {'Bar':>6} {'Date':>22} {'Dir':<6} {'Hits':>4} {'Loss':>8} {'Patterns'}")
    print("  " + "-" * 100)
    for ev in all_cascade_bars[:10]:
        pats = ', '.join(ev['patterns'][:4])
        if len(ev['patterns']) > 4:
            pats += f" +{len(ev['patterns']) - 4}"
        print(f"  {ev['fold']:>4} {ev['bar']:>6} {ev['date']:>22} {ev['direction']:<6} "
              f"{ev['n_hits']:>4} {ev['total_loss_pct']:>+7.2f}% {pats}")

    # ── 14. Recommendation ──
    print(f"\n{'=' * 120}")
    print("RECOMMENDATION")
    print('=' * 120)

    # Find best scenario by PnL/MDD among those that pass WF
    best_name = None
    best_pnl_mdd = -999
    for scenario_name in SCENARIOS:
        folds_m = all_results[scenario_name]
        positive_folds = sum(1 for m in folds_m if m['pnl'] > 0)
        if positive_folds < 3:
            continue
        total_pnl = sum(m['pnl'] for m in folds_m)
        max_mdd = max(m['mdd'] for m in folds_m)
        pnl_mdd = total_pnl / max_mdd if max_mdd > 0 else 0
        if pnl_mdd > best_pnl_mdd:
            best_pnl_mdd = pnl_mdd
            best_name = scenario_name

    if best_name:
        folds_m = all_results[best_name]
        total_pnl = sum(m['pnl'] for m in folds_m)
        max_mdd = max(m['mdd'] for m in folds_m)
        total_skip = sum(m['skipped'] for m in folds_m)
        total_cascade = sum(m['cascade_events'] for m in folds_m)
        base_cascade = sum(m['cascade_events'] for m in all_results['baseline_no_cap'])
        print(f"  Best WF 3/3 PASS scenario: {best_name}")
        print(f"  PnL: {total_pnl:+.2f}%, MDD: {max_mdd:.2f}%, PnL/MDD: {best_pnl_mdd:.2f}x")
        print(f"  Skipped entries: {total_skip}")
        print(f"  Cascade events: {total_cascade} (baseline: {base_cascade})")
        d_cascade = total_cascade - base_cascade
        d_pnl = total_pnl - base_total_pnl
        d_mdd = max_mdd - base_max_mdd
        print(f"  Delta vs baseline: PnL {d_pnl:+.2f}%, MDD {d_mdd:+.2f}%, cascades {d_cascade:+d}")
    else:
        print("  No scenario achieved WF 3/3 PASS.")

    # Check if any cap scenario is strictly better than baseline
    print(f"\n  Baseline: PnL {base_total_pnl:+.2f}%, MDD {base_max_mdd:.2f}%, "
          f"PnL/MDD {base_total_pnl / base_max_mdd:.2f}x, cascades {base_cascades}")

    improved = []
    for scenario_name in SCENARIOS:
        if scenario_name == 'baseline_no_cap':
            continue
        folds_m = all_results[scenario_name]
        positive_folds = sum(1 for m in folds_m if m['pnl'] > 0)
        total_pnl = sum(m['pnl'] for m in folds_m)
        max_mdd = max(m['mdd'] for m in folds_m)
        pnl_mdd = total_pnl / max_mdd if max_mdd > 0 else 0
        base_pnl_mdd = base_total_pnl / base_max_mdd if base_max_mdd > 0 else 0
        if positive_folds == 3 and pnl_mdd > base_pnl_mdd:
            improved.append((scenario_name, pnl_mdd, total_pnl, max_mdd))

    if improved:
        improved.sort(key=lambda x: -x[1])
        print(f"\n  Scenarios that PASS WF AND improve PnL/MDD over baseline:")
        for name, pm, pnl, mdd in improved:
            print(f"    {name}: PnL/MDD {pm:.2f}x (PnL {pnl:+.2f}%, MDD {mdd:.2f}%)")
    else:
        print(f"\n  No scenario both passes WF 3/3 AND improves PnL/MDD over baseline.")
        print("  VERDICT: Aggregate risk cap does NOT improve risk-adjusted returns on this data.")
        print("  The existing direction_cap=8 may already be sufficient risk control.")

    # ── 15. Save results ──
    output = {
        'study': 'aggregate_risk_cap_study',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'protocol': {
            'leverage': LEVERAGE,
            'fee_pct': FEE_PCT,
            'slippage_buffer': SLIPPAGE_BUFFER,
            'timeout_bars': TIMEOUT_BARS,
            'max_positions': MAX_POSITIONS,
            'direction_cap': DIRECTION_CAP,
            'atr_period': ATR_PERIOD,
            'atr_window': ATR_WINDOW,
            'clamp': [CLAMP_LO, CLAMP_HI],
            'warmup_bars': ATR_WARMUP,
            'wf_folds': 3,
            'sizing': 'compound',
            'timeout_handling': 'DROP',
        },
        'data': {
            'csv_bars': n_bars,
            'csv_days': round(n_days, 1),
            'total_signals': len(all_signals),
            'patterns': f"{len(p_long)}L+{len(p_short)}S={len(p_long) + len(p_short)}",
        },
        'folds': [
            {'oos_start': s, 'oos_end': e, 'oos_days': round((e - s) / 288, 1)}
            for s, e in folds
        ],
        'scenarios': summary_data,
        'cascade_worst_10': all_cascade_bars[:10],
        'best_wf_pass': best_name,
        'runtime_seconds': round(time.time() - t0, 1),
    }

    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {RESULTS_PATH}")
    print(f"Runtime: {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
