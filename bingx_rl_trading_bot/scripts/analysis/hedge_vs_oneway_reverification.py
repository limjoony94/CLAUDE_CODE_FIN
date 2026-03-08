#!/usr/bin/env python3
"""
Hedge vs One-Way Re-Verification Study (v1.55.0)
=================================================

PURPOSE: Re-verify v1.30.0 Hedge vs FIFO conclusion with CURRENT config:
  - Data: 303d (btc_5m_270days_reclassified.csv)
  - Patterns: 131 (59L+72S) from dynamic_patterns.json
  - N=9, dir_cap=7, cascade SL 85%, agg_risk 8/15, timeout 288
  - ATR clamp [0.5, 1.5], momentum guard (1.5%/3bars)

STRATEGIES TESTED:
  A) Hedge:        Current production mode (LONG+SHORT independent)
  B) FIFO:         Opposite direction → CLOSE_OLDEST
  C) Close-ALL:    Opposite direction → close all → open new
  D) Smart-OneWay: Opposite direction → SKIP (no forced close)
  E) Hedge-NoCap:  Hedge without direction cap (unlimited)

EXPERIMENTS:
  1. IS comparison (full 303d neutral window)
  2. WF 3-fold expanding window
  3. Random discrimination (20 seeds x 5 strategies)

Standard Research Protocol:
  - Entry: next bar open after signal
  - Exit: intrabar high/low (distance-based same-bar resolution)
  - Sizing: compound (1/N per slot)
  - Fee: 0.05% x 2 = 0.10% (capital space: 0.10% x LEVERAGE)
  - Timeout: 288 bars (24h) -> DROP
  - WF: 3-fold expanding window
"""

import os
import sys
import json
import time
import logging
import numpy as np
import pandas as pd
from collections import defaultdict

# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)

from scripts.production.pattern_5m.indicators import classify_candle

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('hedge_reverify')

DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
PATTERNS_FILE = os.path.join(_PROJECT_ROOT, 'results', 'dynamic_patterns.json')
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'hedge_vs_oneway_reverification.json')

# --- Constants (v1.53.0 production aligned) ---
FEE_PCT = 0.0005          # 0.05% per side
LEVERAGE = 3
FEE = FEE_PCT * 2 * LEVERAGE * 100  # 0.30 (percentage points)
MAX_BARS = 288             # 24h timeout
BARS_PER_DAY = 288
ATR_PERIOD = 14
ATR_WINDOW = 576
CLAMP_LO = 0.5
CLAMP_HI = 1.5
SLIPPAGE_BUFFER = 0.02

# --- Mechanism Parameters ---
MAX_POSITIONS = 9
DIRECTION_CAP = 7
CASCADE_TIGHTEN_PCT = 85   # 85% SL tightening on same-dir SL hit
AGG_RISK_COUNTER = 8.0     # Max SL exposure % for counter-regime
AGG_RISK_WITH = 15.0       # Max SL exposure % for with-regime
MOMENTUM_LOOKBACK = 3      # bars (15min)
MOMENTUM_THRESHOLD = 1.5   # % price change threshold
MOMENTUM_COOLDOWN = 12     # bars (1h) cooldown after trigger


# ===================================================================
# DATA & PATTERN LOADING
# ===================================================================

def load_data():
    df = pd.read_csv(DATA_FILE, parse_dates=['timestamp'])
    # Compute avg_body_20 for classify_candle
    df['body'] = (df['close'] - df['open']).abs()
    df['avg_body_20'] = df['body'].rolling(20, min_periods=1).mean()
    # Re-classify using production classify_candle
    codes = []
    for i in range(len(df)):
        row = df.iloc[i]
        code = classify_candle(row, df['avg_body_20'].iloc[i])
        if hasattr(code, 'value'):
            code = code.value
        codes.append(code)
    df['rctype'] = codes

    # Build 3-candle patterns
    patterns = [''] * len(df)
    for i in range(2, len(df)):
        patterns[i] = f"{codes[i-2]}-{codes[i-1]}-{codes[i]}"
    df['pattern_3'] = patterns
    return df


def load_patterns():
    with open(PATTERNS_FILE) as f:
        dp = json.load(f)
    details = dp.get('pattern_details', {})
    patterns = []
    for key, info in details.items():
        parts = key.rsplit('_', 1)
        if len(parts) != 2:
            continue
        pat_name, direction = parts
        patterns.append({
            'pattern': pat_name,
            'direction': direction,
            'tp': info.get('tp_pct', 1.0),
            'sl': info.get('sl_pct', 2.0),
            'edge': info.get('edge_pp', 0),
        })
    return patterns


def build_signal_index(types, n_bars):
    idx = defaultdict(list)
    for i in range(2, n_bars):
        pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
        if pat:
            idx[pat].append(i)
    return idx


def compute_atr_ratio(highs, lows, closes, period=14, window=576):
    tr = np.maximum(highs[1:] - lows[1:],
                    np.maximum(np.abs(highs[1:] - closes[:-1]),
                               np.abs(lows[1:] - closes[:-1])))
    tr = np.concatenate([[highs[0] - lows[0]], tr])
    atr = pd.Series(tr).rolling(period, min_periods=period).mean().values
    atr_ma = pd.Series(atr).rolling(window, min_periods=period).mean().values
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(atr_ma > 0, atr / atr_ma, 1.0)
    return ratio


def find_neutral_window(closes, tol_pct=1.0):
    """Find longest window where start price ≈ end price (within tol_pct)."""
    n = len(closes)
    best_start, best_end = 0, n - 1
    best_len = 0
    for start in range(0, n, BARS_PER_DAY):
        sp = closes[start]
        for end in range(n - 1, start + BARS_PER_DAY, -1):
            ep = closes[end]
            if abs(ep - sp) / sp * 100 <= tol_pct:
                length = end - start
                if length > best_len:
                    best_len = length
                    best_start, best_end = start, end
                break
    return best_start, best_end + 1


# ===================================================================
# SIGNAL SCHEDULE
# ===================================================================

def build_signal_schedule(patterns, sig_index, atr_ratio, opens, n_bars, random_seed=None):
    """Build signal schedule. If random_seed is set, randomize directions."""
    rng = np.random.RandomState(random_seed) if random_seed is not None else None

    schedule = {}
    for p in patterns:
        bars = sig_index.get(p['pattern'])
        if not bars:
            continue
        direction = 1 if p['direction'] == 'LONG' else -1
        base_tp, base_sl = p['tp'], p['sl']

        for sig_bar in bars:
            entry_bar = sig_bar + 1
            if entry_bar >= n_bars:
                continue
            entry_price = opens[entry_bar]
            if entry_price <= 0:
                continue

            # ATR scaling
            if atr_ratio is not None and sig_bar < len(atr_ratio) and not np.isnan(atr_ratio[sig_bar]):
                r = max(CLAMP_LO, min(CLAMP_HI, atr_ratio[sig_bar]))
            else:
                r = 1.0

            tp = base_tp * r + SLIPPAGE_BUFFER
            sl = max(0.1, base_sl * r - SLIPPAGE_BUFFER)

            # Randomize direction if seed provided
            actual_dir = direction
            if rng is not None:
                actual_dir = rng.choice([1, -1])

            if entry_bar not in schedule:
                schedule[entry_bar] = []
            schedule[entry_bar].append((p['pattern'], actual_dir, tp, sl, entry_price, sig_bar))

    return schedule


# ===================================================================
# MULTI-POSITION SIMULATOR WITH MECHANISMS
# ===================================================================

class ActivePosition:
    __slots__ = ('pattern', 'direction', 'entry_price', 'tp_pct', 'sl_pct',
                 'tp_price', 'sl_price', 'entry_bar', 'sig_bar', 'eff_sl_pct')

    def __init__(self, pattern, direction, entry_price, tp_pct, sl_pct, entry_bar, sig_bar):
        self.pattern = pattern
        self.direction = direction
        self.entry_price = entry_price
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.eff_sl_pct = sl_pct  # effective SL (may be tightened by cascade)
        self.entry_bar = entry_bar
        self.sig_bar = sig_bar

        if direction == 1:
            self.tp_price = entry_price * (1 + tp_pct / 100)
            self.sl_price = entry_price * (1 - sl_pct / 100)
        else:
            self.tp_price = entry_price * (1 - tp_pct / 100)
            self.sl_price = entry_price * (1 + sl_pct / 100)

    def update_sl(self, new_sl_pct):
        """Update SL price after cascade tightening."""
        self.eff_sl_pct = new_sl_pct
        if self.direction == 1:
            self.sl_price = self.entry_price * (1 - new_sl_pct / 100)
        else:
            self.sl_price = self.entry_price * (1 + new_sl_pct / 100)


def detect_regime(closes, bar, lookback=100):
    """Simple regime detection: compare SMA vs price."""
    if bar < lookback:
        return 0  # neutral
    sma = np.mean(closes[bar - lookback:bar])
    price = closes[bar]
    if price > sma * 1.005:
        return 1   # uptrend → LONG is with-trend
    elif price < sma * 0.995:
        return -1  # downtrend → SHORT is with-trend
    return 0  # neutral


def check_momentum_guard(closes, bar, direction, cooldown_tracker, lookback=MOMENTUM_LOOKBACK):
    """Check if momentum guard blocks entry."""
    if bar < lookback:
        return False

    # Check cooldown
    if direction in cooldown_tracker and cooldown_tracker[direction] > bar:
        return True  # blocked

    # Check if price moved > threshold in lookback bars
    price_now = closes[bar]
    price_prev = closes[bar - lookback]
    change_pct = (price_now - price_prev) / price_prev * 100

    # Block counter-direction entry
    if abs(change_pct) > MOMENTUM_THRESHOLD:
        if change_pct > 0 and direction == -1:  # big up, block SHORT
            cooldown_tracker[-1] = bar + MOMENTUM_COOLDOWN
            return True
        if change_pct < 0 and direction == 1:   # big down, block LONG
            cooldown_tracker[1] = bar + MOMENTUM_COOLDOWN
            return True

    return False


def check_agg_risk(slots, new_direction, new_sl_pct, closes, bar):
    """Check aggregate risk cap for the new position's direction."""
    regime = detect_regime(closes, bar)

    # Determine if new direction is counter or with-trend
    if regime == 0:
        cap = AGG_RISK_WITH  # neutral → use with (more lenient)
    elif (regime == 1 and new_direction == -1) or (regime == -1 and new_direction == 1):
        cap = AGG_RISK_COUNTER  # counter-regime
    else:
        cap = AGG_RISK_WITH  # with-regime

    # Sum SL exposure for same-direction positions
    total_sl = new_sl_pct
    for pos in slots.values():
        if pos.direction == new_direction:
            total_sl += pos.eff_sl_pct

    return total_sl <= cap


def simulate_strategy(strategy, schedule, opens, highs, lows, closes,
                      lo_bar, hi_bar, max_pos=MAX_POSITIONS, dir_cap=DIRECTION_CAP,
                      use_cascade=True, use_agg_risk=True, use_momentum=True):
    """
    Bar-by-bar multi-position simulation with full mechanism stack.

    strategy: 'hedge' | 'fifo' | 'close_all' | 'smart_oneway' | 'hedge_nocap'
    """
    slots = {}
    slot_counter = 0
    active_direction = None
    results = []
    close_oldest_count = 0
    close_oldest_pnls = []
    momentum_cooldown = {}
    n_bars = len(opens)

    # Effective direction cap
    eff_dir_cap = dir_cap if strategy not in ('hedge_nocap',) else max_pos

    def _calc_pnl(pos, exit_price):
        pm = pos.direction * (exit_price / pos.entry_price - 1) * 100
        return pm * LEVERAGE - FEE

    def _close_slot(sid, exit_bar, exit_price, reason):
        nonlocal active_direction, close_oldest_count
        pos = slots.pop(sid)
        pnl = _calc_pnl(pos, exit_price)
        weight = 1.0 / max_pos
        scaled = pnl * weight
        results.append({
            'entry_bar': pos.entry_bar, 'exit_bar': exit_bar,
            'pnl': scaled, 'reason': reason, 'direction': pos.direction,
            'pattern': pos.pattern, 'slot_pnl': pnl,
        })
        if reason == 'CLOSE_OLDEST':
            close_oldest_count += 1
            close_oldest_pnls.append(pnl)
        if not slots:
            active_direction = None
        return pos.direction, pnl  # for cascade

    def _apply_cascade(direction, exclude_sid=None):
        """Tighten SL for all same-direction positions after SL hit."""
        if not use_cascade:
            return
        tighten_ratio = 1.0 - (CASCADE_TIGHTEN_PCT / 100)  # 0.15 for 85%
        for sid, pos in slots.items():
            if sid == exclude_sid:
                continue
            if pos.direction == direction:
                new_sl = pos.eff_sl_pct * tighten_ratio
                pos.update_sl(max(0.05, new_sl))

    for bar in range(lo_bar, hi_bar):
        if bar >= n_bars:
            break

        # --- 1. Check TP/SL ---
        sids_to_close = []
        for sid, pos in slots.items():
            h, l, o = highs[bar], lows[bar], opens[bar]

            if pos.direction == 1:  # LONG
                tp_hit = h >= pos.tp_price
                sl_hit = l <= pos.sl_price
            else:
                tp_hit = l <= pos.tp_price
                sl_hit = h >= pos.sl_price

            if tp_hit and sl_hit:
                tp_dist = abs(pos.tp_price - o)
                sl_dist = abs(pos.sl_price - o)
                if tp_dist <= sl_dist:
                    tp_hit, sl_hit = True, False
                else:
                    tp_hit, sl_hit = False, True

            if tp_hit:
                sids_to_close.append((sid, bar, pos.tp_price, 'TP'))
            elif sl_hit:
                sids_to_close.append((sid, bar, pos.sl_price, 'SL'))

        for sid, ebar, eprice, reason in sids_to_close:
            if sid in slots:
                closed_dir, _ = _close_slot(sid, ebar, eprice, reason)
                if reason == 'SL':
                    _apply_cascade(closed_dir, exclude_sid=sid)

        # --- 2. Timeouts ---
        for sid in list(slots):
            if sid not in slots:
                continue
            pos = slots[sid]
            if bar - pos.entry_bar >= MAX_BARS:
                exit_price = opens[bar] if bar < n_bars else opens[-1]
                _close_slot(sid, bar, exit_price, 'TIMEOUT')

        # --- 3. Process signals ---
        signals = schedule.get(bar)
        if not signals:
            continue

        active_patterns = {s.pattern for s in slots.values()}
        signals_sorted = sorted(signals, key=lambda s: -s[1])

        for pat_name, sig_dir, tp, sl, entry_price, sig_bar in signals_sorted:
            if pat_name in active_patterns:
                continue

            n_slots = len(slots)

            # Momentum guard
            if use_momentum and check_momentum_guard(closes, bar, sig_dir, momentum_cooldown):
                continue

            # Strategy-specific routing
            action = None

            if strategy in ('hedge', 'hedge_nocap'):
                if n_slots >= max_pos:
                    continue
                same_dir = sum(1 for p in slots.values() if p.direction == sig_dir)
                if same_dir >= eff_dir_cap:
                    continue
                # Aggregate risk check
                if use_agg_risk and not check_agg_risk(slots, sig_dir, sl, closes, bar):
                    continue
                action = 'OPEN'

            elif strategy == 'fifo':
                if n_slots == 0:
                    action = 'OPEN'
                elif active_direction == sig_dir:
                    if n_slots < max_pos:
                        if use_agg_risk and not check_agg_risk(slots, sig_dir, sl, closes, bar):
                            continue
                        action = 'OPEN'
                    else:
                        continue
                else:
                    oldest_sid = min(slots, key=lambda s: slots[s].entry_bar)
                    _close_slot(oldest_sid, bar, opens[bar], 'CLOSE_OLDEST')
                    if not slots:
                        action = 'OPEN'
                    else:
                        continue

            elif strategy == 'close_all':
                if n_slots == 0:
                    action = 'OPEN'
                elif active_direction == sig_dir:
                    if n_slots < max_pos:
                        if use_agg_risk and not check_agg_risk(slots, sig_dir, sl, closes, bar):
                            continue
                        action = 'OPEN'
                    else:
                        continue
                else:
                    for sid in list(slots):
                        _close_slot(sid, bar, opens[bar], 'CLOSE_ALL')
                    action = 'OPEN'

            elif strategy == 'smart_oneway':
                if n_slots == 0:
                    action = 'OPEN'
                elif active_direction == sig_dir:
                    if n_slots < max_pos:
                        if use_agg_risk and not check_agg_risk(slots, sig_dir, sl, closes, bar):
                            continue
                        action = 'OPEN'
                    else:
                        continue
                else:
                    continue  # SKIP opposite direction (no forced close)

            if action == 'OPEN':
                slot_counter += 1
                sid = f's{slot_counter}'
                pos = ActivePosition(pat_name, sig_dir, entry_price, tp, sl, bar, sig_bar)
                slots[sid] = pos
                active_direction = sig_dir
                active_patterns.add(pat_name)

    # Close remaining
    for sid in list(slots):
        exit_price = opens[min(hi_bar - 1, n_bars - 1)]
        _close_slot(sid, hi_bar - 1, exit_price, 'END')

    return results, close_oldest_count, close_oldest_pnls


# ===================================================================
# STATISTICS
# ===================================================================

def calc_stats(results):
    if not results:
        return {'trades': 0, 'wr': 0, 'pnl': 0, 'mdd': 0, 'pnl_mdd': 0}

    pnls = [r['pnl'] for r in results]
    wins = sum(1 for p in pnls if p > 0)
    total = len(pnls)
    wr = wins / total * 100 if total > 0 else 0

    # Compound PnL
    equity = 1.0
    peak = 1.0
    max_dd = 0
    for p in pnls:
        equity *= (1 + p / 100)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd:
            max_dd = dd

    cmp_pnl = (equity - 1) * 100
    pnl_mdd = round(cmp_pnl / max_dd, 2) if max_dd > 0 and cmp_pnl > 0 else 0

    # Exit reasons
    exits = defaultdict(int)
    for r in results:
        exits[r['reason']] += 1

    # Direction breakdown
    long_t = [r for r in results if r['direction'] == 1]
    short_t = [r for r in results if r['direction'] == -1]
    long_w = sum(1 for r in long_t if r['pnl'] > 0)
    short_w = sum(1 for r in short_t if r['pnl'] > 0)

    return {
        'trades': total,
        'wr': round(wr, 1),
        'cmp_pnl': round(cmp_pnl, 1),
        'cmp_mdd': round(max_dd, 1),
        'pnl_mdd': pnl_mdd,
        'long_trades': len(long_t),
        'long_wr': round(long_w / len(long_t) * 100, 1) if long_t else 0,
        'short_trades': len(short_t),
        'short_wr': round(short_w / len(short_t) * 100, 1) if short_t else 0,
        'exits': dict(exits),
        'co_count': sum(1 for r in results if r['reason'] == 'CLOSE_OLDEST'),
        'co_total_pnl': round(sum(r['pnl'] for r in results if r['reason'] == 'CLOSE_OLDEST'), 2),
    }


# ===================================================================
# MAIN
# ===================================================================

def main():
    t0 = time.time()

    logger.info("Loading data...")
    df = load_data()
    n_bars = len(df)
    opens = df['open'].values.astype(np.float64)
    highs = df['high'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)
    closes = df['close'].values.astype(np.float64)
    types = df['rctype'].values

    logger.info("Loading patterns...")
    patterns = load_patterns()
    logger.info(f"Loaded {len(patterns)} patterns")

    sig_index = build_signal_index(types, n_bars)
    atr_ratio = compute_atr_ratio(highs, lows, closes, ATR_PERIOD, ATR_WINDOW)

    # Find neutral window
    nw_start, nw_end = find_neutral_window(closes)
    nw_bars = nw_end - nw_start
    nw_days = nw_bars / BARS_PER_DAY
    logger.info(f"Neutral window: bar {nw_start}-{nw_end} ({nw_bars} bars, {nw_days:.0f} days)")

    output = {
        'meta': {
            'data_file': DATA_FILE,
            'n_bars': n_bars,
            'n_patterns': len(patterns),
            'neutral_window': [int(nw_start), int(nw_end), round(nw_days, 1)],
            'config': {
                'max_positions': MAX_POSITIONS,
                'direction_cap': DIRECTION_CAP,
                'cascade_tighten_pct': CASCADE_TIGHTEN_PCT,
                'agg_risk_counter': AGG_RISK_COUNTER,
                'agg_risk_with': AGG_RISK_WITH,
                'timeout': MAX_BARS,
                'atr_clamp': [CLAMP_LO, CLAMP_HI],
                'leverage': LEVERAGE,
                'fee_pct': FEE_PCT,
            },
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
    }

    strategies = ['hedge', 'fifo', 'close_all', 'smart_oneway', 'hedge_nocap']
    strat_labels = {
        'hedge': 'Hedge (현행)',
        'fifo': 'FIFO',
        'close_all': 'Close-ALL',
        'smart_oneway': 'Smart-OneWay',
        'hedge_nocap': 'Hedge-NoCap',
    }

    # Build signal schedule
    schedule = build_signal_schedule(patterns, sig_index, atr_ratio, opens, n_bars)
    total_signals = sum(len(v) for v in schedule.values())
    logger.info(f"Signal schedule: {len(schedule)} bars with signals, {total_signals} entries")

    # ==================================================================
    # EXPERIMENT 1: IS Comparison (neutral window)
    # ==================================================================
    print("\n" + "=" * 70)
    print("  EXPERIMENT 1: IN-SAMPLE COMPARISON (neutral window)")
    print(f"  Data: {nw_days:.0f} days, N={MAX_POSITIONS}, DirCap={DIRECTION_CAP}")
    print(f"  Mechanisms: Cascade={CASCADE_TIGHTEN_PCT}%, AggRisk={AGG_RISK_COUNTER}/{AGG_RISK_WITH}")
    print("=" * 70)

    exp1 = {}
    for strat in strategies:
        results, co_count, co_pnls = simulate_strategy(
            strat, schedule, opens, highs, lows, closes, nw_start, nw_end
        )
        stats = calc_stats(results)
        exp1[strat] = stats
        logger.info(f"  {strat}: {stats['trades']}t, WR {stats['wr']}%, PnL {stats['cmp_pnl']}%, "
                     f"MDD {stats['cmp_mdd']}%, PnL/MDD {stats['pnl_mdd']}")

    output['experiment1_is'] = exp1

    # Print table
    print(f"\n  {'Metric':<16}", end='')
    for s in strategies:
        print(f" | {strat_labels[s]:>14}", end='')
    print()
    print("  " + "-" * (16 + 17 * len(strategies)))
    for key in ['trades', 'wr', 'cmp_pnl', 'cmp_mdd', 'pnl_mdd',
                'long_trades', 'long_wr', 'short_trades', 'short_wr', 'co_count']:
        print(f"  {key:<16}", end='')
        for s in strategies:
            v = exp1[s].get(key, 0)
            print(f" | {str(v):>14}", end='')
        print()

    # ==================================================================
    # EXPERIMENT 2: Walk-Forward 3-fold
    # ==================================================================
    print("\n" + "=" * 70)
    print("  EXPERIMENT 2: WALK-FORWARD 3-FOLD (expanding window)")
    print("=" * 70)

    quarter = nw_bars // 4
    wf_folds = [
        (nw_start, nw_start + quarter,
         nw_start + quarter, nw_start + 2 * quarter),
        (nw_start, nw_start + 2 * quarter,
         nw_start + 2 * quarter, nw_start + 3 * quarter),
        (nw_start, nw_start + 3 * quarter,
         nw_start + 3 * quarter, nw_end),
    ]

    exp2 = {}
    for strat in strategies:
        strat_folds = []
        for fi, (is_lo, is_hi, oos_lo, oos_hi) in enumerate(wf_folds):
            # IS
            is_results, _, _ = simulate_strategy(
                strat, schedule, opens, highs, lows, closes, is_lo, is_hi
            )
            is_stats = calc_stats(is_results)

            # OOS
            oos_results, _, _ = simulate_strategy(
                strat, schedule, opens, highs, lows, closes, oos_lo, oos_hi
            )
            oos_stats = calc_stats(oos_results)

            fold = {
                'is': is_stats,
                'oos': oos_stats,
                'oos_pass': oos_stats['cmp_pnl'] > 0,
            }
            strat_folds.append(fold)

        pass_count = sum(1 for f in strat_folds if f['oos_pass'])
        exp2[strat] = {
            'folds': strat_folds,
            'pass_count': pass_count,
            'all_pass': pass_count == 3,
            'oos_avg_pnl': round(np.mean([f['oos']['cmp_pnl'] for f in strat_folds]), 1),
            'oos_min_pnl': round(min(f['oos']['cmp_pnl'] for f in strat_folds), 1),
            'oos_avg_wr': round(np.mean([f['oos']['wr'] for f in strat_folds]), 1),
        }

    output['experiment2_wf'] = exp2

    # Print WF results
    print(f"\n  {'Strategy':<16} | {'Pass':>5} | {'OOS Avg PnL':>12} | {'OOS Min':>8} | {'OOS Avg WR':>10}")
    print("  " + "-" * 60)
    for s in strategies:
        e = exp2[s]
        verdict = "PASS" if e['all_pass'] else f"FAIL({e['pass_count']}/3)"
        print(f"  {strat_labels[s]:<16} | {verdict:>5} | {e['oos_avg_pnl']:>+11.1f}% | "
              f"{e['oos_min_pnl']:>+7.1f}% | {e['oos_avg_wr']:>9.1f}%")

    # Per-fold detail
    for s in strategies:
        print(f"\n  {strat_labels[s]}:")
        for fi, f in enumerate(exp2[s]['folds']):
            p = "PASS" if f['oos_pass'] else "FAIL"
            print(f"    F{fi+1}: IS PnL {f['is']['cmp_pnl']:>+8.1f}% WR {f['is']['wr']:>5.1f}% | "
                  f"OOS PnL {f['oos']['cmp_pnl']:>+8.1f}% WR {f['oos']['wr']:>5.1f}% [{p}]")

    # ==================================================================
    # EXPERIMENT 3: Random Discrimination (20 seeds x strategy)
    # ==================================================================
    print("\n" + "=" * 70)
    print("  EXPERIMENT 3: RANDOM DISCRIMINATION (20 seeds)")
    print("=" * 70)

    N_SEEDS = 20
    exp3 = {}

    for strat in strategies:
        random_passes = 0
        random_pnls = []

        for seed in range(N_SEEDS):
            # Build random-direction schedule
            rand_schedule = build_signal_schedule(
                patterns, sig_index, atr_ratio, opens, n_bars, random_seed=seed + 100
            )

            # WF with random signals
            all_pass = True
            fold_pnls = []
            for fi, (is_lo, is_hi, oos_lo, oos_hi) in enumerate(wf_folds):
                oos_results, _, _ = simulate_strategy(
                    strat, rand_schedule, opens, highs, lows, closes, oos_lo, oos_hi
                )
                oos_stats = calc_stats(oos_results)
                fold_pnls.append(oos_stats['cmp_pnl'])
                if oos_stats['cmp_pnl'] <= 0:
                    all_pass = False

            if all_pass:
                random_passes += 1
            random_pnls.append(np.mean(fold_pnls))

        pass_rate = random_passes / N_SEEDS * 100
        exp3[strat] = {
            'n_seeds': N_SEEDS,
            'random_passes': random_passes,
            'pass_rate': round(pass_rate, 1),
            'avg_random_pnl': round(float(np.mean(random_pnls)), 1),
            'discriminating': pass_rate < 50,  # If >50% random pass, non-discriminating
        }

        disc = "DISCRIMINATING" if pass_rate < 50 else "NON-DISCRIMINATING"
        logger.info(f"  {strat}: {random_passes}/{N_SEEDS} random WF pass ({pass_rate:.0f}%) → {disc}")

    output['experiment3_random'] = exp3

    print(f"\n  {'Strategy':<16} | {'Random Pass':>12} | {'Rate':>6} | {'Avg PnL':>8} | {'Verdict':>18}")
    print("  " + "-" * 70)
    for s in strategies:
        e = exp3[s]
        disc = "DISCRIMINATING" if e['discriminating'] else "NON-DISCRIMINATING"
        print(f"  {strat_labels[s]:<16} | {e['random_passes']:>4}/{N_SEEDS:<4}    | "
              f"{e['pass_rate']:>5.0f}% | {e['avg_random_pnl']:>+7.1f}% | {disc:>18}")

    # ==================================================================
    # EXPERIMENT 4: Mechanism Ablation per Strategy
    # ==================================================================
    print("\n" + "=" * 70)
    print("  EXPERIMENT 4: MECHANISM ABLATION (Hedge vs Hedge-bare)")
    print("=" * 70)

    ablation_configs = [
        ('full', True, True, True),
        ('no_cascade', False, True, True),
        ('no_agg', True, False, True),
        ('no_momentum', True, True, False),
        ('bare', False, False, False),
    ]

    exp4 = {}
    for label, cascade, agg, momentum in ablation_configs:
        results, _, _ = simulate_strategy(
            'hedge', schedule, opens, highs, lows, closes, nw_start, nw_end,
            use_cascade=cascade, use_agg_risk=agg, use_momentum=momentum
        )
        stats = calc_stats(results)
        exp4[label] = stats
        logger.info(f"  {label}: {stats['trades']}t, WR {stats['wr']}%, PnL {stats['cmp_pnl']}%, "
                     f"MDD {stats['cmp_mdd']}%, PnL/MDD {stats['pnl_mdd']}")

    output['experiment4_ablation'] = exp4

    print(f"\n  {'Config':<16} | {'Trades':>6} | {'WR':>6} | {'PnL':>10} | {'MDD':>6} | {'PnL/MDD':>8}")
    print("  " + "-" * 65)
    for label, _, _, _ in ablation_configs:
        s = exp4[label]
        print(f"  {label:<16} | {s['trades']:>6} | {s['wr']:>5.1f}% | {s['cmp_pnl']:>+9.1f}% | "
              f"{s['cmp_mdd']:>5.1f}% | {s['pnl_mdd']:>8}")

    # ==================================================================
    # SAVE & SUMMARY
    # ==================================================================
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"Results saved to {OUTPUT_FILE}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")

    # Find best strategy
    best_strat = max(strategies, key=lambda s: exp1[s]['pnl_mdd'])
    print(f"\n  Best IS PnL/MDD: {strat_labels[best_strat]} ({exp1[best_strat]['pnl_mdd']})")

    best_wf = max(strategies, key=lambda s: exp2[s]['oos_avg_pnl'] if exp2[s]['all_pass'] else -999)
    print(f"  Best WF OOS PnL: {strat_labels[best_wf]} ({exp2[best_wf]['oos_avg_pnl']:+.1f}%)")

    # Discrimination check
    for s in strategies:
        if exp3[s]['discriminating']:
            print(f"  DISCRIMINATING: {strat_labels[s]} ({exp3[s]['pass_rate']}% random pass)")
        else:
            print(f"  NON-DISCRIMINATING: {strat_labels[s]} ({exp3[s]['pass_rate']}% random pass)")

    print(f"\n  Hedge vs FIFO ratio: {exp1['hedge']['pnl_mdd']} vs {exp1['fifo']['pnl_mdd']} "
          f"= {exp1['hedge']['pnl_mdd'] / max(exp1['fifo']['pnl_mdd'], 0.01):.1f}x")

    print(f"\n  Elapsed: {elapsed:.0f}s")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
