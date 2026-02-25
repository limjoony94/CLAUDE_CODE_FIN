"""
H1 Bullish LONG WF Validation
==============================
Validates 3 candidate LONG patterns discovered in H1 research:
  BD-U-BD  (LONG, TP 2.14%, SL 4.07%)
  MU-DN-IH (LONG, TP 1.42%, SL 2.56%)
  ST-MU-U  (LONG, TP 2.87%, SL 3.63%)

4-Phase validation:
  Phase 1: Full IS reproduction (270d) — verify IS stats match H1 claims
  Phase 2: Individual 3-fold expanding window WF — per-pattern OOS edge
  Phase 3: Portfolio WF (51pat baseline vs 54pat with H1) — N=9, Hedge, cap8, T864
  Phase 4: Holdout 7d — last 7 days OOS check

Standard Research Protocol:
- Entry: signal bar+1 open
- Exit: intrabar high/low (distance-based, bar open same-bar resolution)
- Fee: 0.10% roundtrip x3 leverage = 0.30% capital-space
- Slippage buffer: 0.02%
- ATR scaling: ATR(14)/median(576), clamp [0.6, 1.7]
- Timeout: 864 bars -> DROP (not counted)
- Compound sizing (portfolio phase)
- classify_candle from production (NOT self-implemented)
"""
import sys
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

# Project root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)

from scripts.production.pattern_5m.indicators import classify_candle
from scripts.production.pattern_5m.constants import AVG_BODY_WINDOW

# ============================================================
# Constants
# ============================================================
LEVERAGE = 3
FEE_PCT = 0.10           # roundtrip fee in price space
SLIPPAGE_BUFFER = 0.02   # 0.02% slippage buffer
TIMEOUT_BARS = 864        # 72h timeout
MAX_POSITIONS = 9
ATR_PERIOD = 14
ATR_WINDOW = 576
CLAMP_LO = 0.6
CLAMP_HI = 1.7
ATR_WARMUP = ATR_PERIOD + ATR_WINDOW  # 590 bars
MC_SIMS = 5000
MC_SEEDS = [42, 123, 7]

# H1 candidate patterns
H1_PATTERNS = {
    'BD-U-BD_LONG': {'pattern': 'BD-U-BD', 'direction': 'LONG', 'tp': 2.14, 'sl': 4.07},
    'MU-DN-IH_LONG': {'pattern': 'MU-DN-IH', 'direction': 'LONG', 'tp': 1.42, 'sl': 2.56},
    'ST-MU-U_LONG': {'pattern': 'ST-MU-U', 'direction': 'LONG', 'tp': 2.87, 'sl': 3.63},
}

DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
PATTERNS_FILE = os.path.join(_PROJECT_ROOT, 'results', 'dynamic_patterns.json')
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'h1_bullish_long_wf_validation.json')


# ============================================================
# Core Functions (from scanner, adapted for local use)
# ============================================================

def compute_atr_ratio(highs, lows, closes, atr_period=ATR_PERIOD, window=ATR_WINDOW):
    """ATR / rolling_median(ATR) ratio. Wilder EMA ATR for production consistency."""
    n = len(closes)
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i - 1]),
                     abs(lows[i] - closes[i - 1]))
    # Wilder EMA ATR
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


def bt_signals_atr(signal_bars, direction, tp_pct, sl_pct,
                    opens, highs, lows, n_bars,
                    atr_ratio, clamp_lo=CLAMP_LO, clamp_hi=CLAMP_HI):
    """Backtest with ATR-scaled TP/SL. Timeout trades are DROPPED."""
    fee = FEE_PCT * LEVERAGE
    is_long = (direction == 'LONG')
    trades = []

    for idx in signal_bars:
        if idx + 1 >= n_bars:
            continue
        entry = opens[idx + 1]
        if entry <= 0:
            continue
        eb = idx + 1

        # ATR scaling at signal bar
        if (atr_ratio is not None and idx < len(atr_ratio)
                and not np.isnan(atr_ratio[idx])):
            r = max(clamp_lo, min(clamp_hi, atr_ratio[idx]))
        else:
            r = 1.0

        eff_tp = tp_pct * r + SLIPPAGE_BUFFER
        eff_sl = max(0.1, sl_pct * r - SLIPPAGE_BUFFER)

        if is_long:
            tpp = entry * (1 + eff_tp / 100)
            slp = entry * (1 - eff_sl / 100)
        else:
            tpp = entry * (1 - eff_tp / 100)
            slp = entry * (1 + eff_sl / 100)

        for j in range(idx + 2, min(idx + 2 + TIMEOUT_BARS, n_bars)):
            if is_long:
                ht = highs[j] >= tpp
                hs = lows[j] <= slp
            else:
                ht = lows[j] <= tpp
                hs = highs[j] >= slp

            if ht and hs:
                bo = opens[j]
                dist_tp = abs(tpp - bo)
                dist_sl = abs(slp - bo)
                pnl = (eff_tp if dist_tp <= dist_sl else -eff_sl) * LEVERAGE - fee
                trades.append((eb, j, pnl))
                break
            elif ht:
                trades.append((eb, j, eff_tp * LEVERAGE - fee))
                break
            elif hs:
                trades.append((eb, j, -eff_sl * LEVERAGE - fee))
                break
        # Timeout -> DROP

    return trades


def mc_test(pnls, n_sims=MC_SIMS):
    """Multi-seed sign randomization MC test. Returns max p-value (conservative)."""
    if len(pnls) < 5:
        return 1.0
    pnls_arr = np.array(pnls)
    actual = np.sum(pnls_arr)
    p_vals = []
    for seed in MC_SEEDS:
        rng = np.random.RandomState(seed)
        signs = rng.choice([-1, 1], size=(n_sims, len(pnls_arr)))
        rand_sums = signs @ pnls_arr
        p_vals.append(float(np.mean(rand_sums >= actual)))
    return max(p_vals)


def calc_stats(pnls):
    """Calculate stats from list of PnL values."""
    if not pnls:
        return {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'edge': 0}
    cum = 0
    peak = 0
    mdd = 0
    wins = 0
    for p in pnls:
        cum += p
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > mdd:
            mdd = dd
        if p > 0:
            wins += 1
    return {
        'pnl': round(cum, 2),
        'trades': len(pnls),
        'wr': round(wins / len(pnls) * 100, 1),
        'mdd': round(mdd, 2),
        'edge': round(cum / len(pnls), 3) if pnls else 0,
    }


# ============================================================
# Data Loading
# ============================================================

def load_data():
    """Load CSV and extract type codes."""
    print(f"Loading data from {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    print(f"  Total bars: {len(df)} ({len(df)/288:.0f} days)")
    # type_code has short codes (BD, DN, U, etc.)
    rctypes = df['type_code'].values.tolist()
    opens = df['open'].values.astype(np.float64)
    highs = df['high'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)
    closes = df['close'].values.astype(np.float64)
    return df, rctypes, opens, highs, lows, closes


def build_signal_index(rctypes):
    """Build pattern -> list of signal bar indices."""
    idx = {}
    for i in range(2, len(rctypes)):
        pat = f"{rctypes[i-2]}-{rctypes[i-1]}-{rctypes[i]}"
        if pat not in idx:
            idx[pat] = []
        idx[pat].append(i)
    return idx


def generate_signals(rctypes, patterns_long, patterns_short, start_idx=0):
    """Generate signals as (bar, pattern, direction) tuples."""
    signals = []
    n = len(rctypes)
    for i in range(max(2, start_idx), n):
        pat = f"{rctypes[i-2]}-{rctypes[i-1]}-{rctypes[i]}"
        if pat in patterns_long:
            signals.append((i, pat, 'LONG'))
        if pat in patterns_short:
            signals.append((i, pat, 'SHORT'))
    return signals


# ============================================================
# Phase 1: Full IS Reproduction
# ============================================================

def phase1_is_reproduction(signal_index, opens, highs, lows, closes, n_bars):
    """Reproduce IS backtest for H1 candidate patterns on full 270d data."""
    print("\n" + "=" * 80)
    print("PHASE 1: Full IS Reproduction (270d)")
    print("=" * 80)

    atr_ratio = compute_atr_ratio(highs, lows, closes)
    results = {}

    for key, info in H1_PATTERNS.items():
        pat = info['pattern']
        direction = info['direction']
        tp = info['tp']
        sl = info['sl']
        baseline_wr = sl / (tp + sl) * 100

        sigs = signal_index.get(pat, [])
        if not sigs:
            print(f"  {key}: NO SIGNALS FOUND")
            results[key] = {'error': 'no signals'}
            continue

        trades = bt_signals_atr(sigs, direction, tp, sl,
                                opens, highs, lows, n_bars,
                                atr_ratio, CLAMP_LO, CLAMP_HI)
        pnls = [t[2] for t in trades]
        wins = sum(1 for p in pnls if p > 0)
        wr = wins / len(pnls) * 100 if pnls else 0
        edge = wr - baseline_wr
        mc_p = mc_test(pnls) if pnls else 1.0
        stats = calc_stats(pnls)

        results[key] = {
            'pattern': pat,
            'direction': direction,
            'tp': tp, 'sl': sl,
            'total_signals': len(sigs),
            'trades': len(trades),
            'wr': round(wr, 1),
            'baseline_wr': round(baseline_wr, 1),
            'edge': round(edge, 1),
            'mc_p': round(mc_p, 4),
            'pnl': stats['pnl'],
            'mdd': stats['mdd'],
        }

        print(f"  {key}: signals={len(sigs)}, trades={len(trades)}, "
              f"WR={wr:.1f}% (baseline {baseline_wr:.1f}%), "
              f"edge={edge:.1f}pp, MC p={mc_p:.4f}, "
              f"PnL={stats['pnl']:.1f}%, MDD={stats['mdd']:.1f}%")

    return results


# ============================================================
# Phase 2: Individual 3-Fold Expanding Window WF
# ============================================================

def phase2_individual_wf(signal_index, opens, highs, lows, closes, n_bars):
    """3-fold expanding window WF for each H1 pattern individually.

    Expanding window (scanner convention):
      total bars split into n_folds+1 = 4 segments
      Fold 1: IS = seg[0:1], OOS = seg[1:2]
      Fold 2: IS = seg[0:2], OOS = seg[2:3]
      Fold 3: IS = seg[0:3], OOS = seg[3:4]
    """
    print("\n" + "=" * 80)
    print("PHASE 2: Individual 3-Fold Expanding Window WF")
    print("=" * 80)

    n_folds = 3
    seg_size = n_bars // (n_folds + 1)
    print(f"  Total bars: {n_bars}, segment size: {seg_size} ({seg_size/288:.0f}d)")

    results = {}

    for key, info in H1_PATTERNS.items():
        pat = info['pattern']
        direction = info['direction']
        tp = info['tp']
        sl = info['sl']
        baseline_wr = sl / (tp + sl) * 100

        sigs = signal_index.get(pat, [])
        if not sigs:
            results[key] = {'error': 'no signals', 'verdict': 'FAIL'}
            continue

        print(f"\n  --- {key} (TP={tp}, SL={sl}, baseline WR={baseline_wr:.1f}%) ---")
        folds = []
        positive_count = 0

        for fold in range(n_folds):
            is_end = (fold + 1) * seg_size
            oos_start = is_end
            oos_end = (fold + 2) * seg_size if fold < n_folds - 1 else n_bars

            # ATR on IS data only (look-ahead bias prevention)
            is_atr = compute_atr_ratio(highs[:is_end], lows[:is_end], closes[:is_end])

            # IS backtest (for reference)
            is_sigs = [s for s in sigs if s < is_end]
            is_trades = bt_signals_atr(is_sigs, direction, tp, sl,
                                        opens, highs, lows, is_end,
                                        is_atr, CLAMP_LO, CLAMP_HI)
            is_pnls = [t[2] for t in is_trades]
            is_stats = calc_stats(is_pnls)

            # OOS backtest with ATR computed up to oos_end
            oos_atr = compute_atr_ratio(highs[:oos_end], lows[:oos_end], closes[:oos_end])
            oos_sigs = [s for s in sigs if oos_start <= s < oos_end]
            oos_trades = bt_signals_atr(oos_sigs, direction, tp, sl,
                                         opens, highs, lows, oos_end,
                                         oos_atr, CLAMP_LO, CLAMP_HI)
            oos_pnls = [t[2] for t in oos_trades]
            oos_stats = calc_stats(oos_pnls)
            oos_wr = oos_stats['wr']
            oos_edge = oos_wr - baseline_wr

            is_positive = oos_stats['pnl'] > 0
            if is_positive:
                positive_count += 1

            fold_result = {
                'fold': fold + 1,
                'is_bars': is_end,
                'oos_bars': oos_end - oos_start,
                'is_trades': is_stats['trades'],
                'is_wr': is_stats['wr'],
                'is_pnl': is_stats['pnl'],
                'oos_trades': oos_stats['trades'],
                'oos_wr': oos_stats['wr'],
                'oos_pnl': oos_stats['pnl'],
                'oos_mdd': oos_stats['mdd'],
                'oos_edge': round(oos_edge, 1),
                'oos_positive': is_positive,
            }
            folds.append(fold_result)

            print(f"    Fold {fold+1}: IS=[0,{is_end}) {is_stats['trades']}tr WR={is_stats['wr']:.1f}% | "
                  f"OOS=[{oos_start},{oos_end}) {oos_stats['trades']}tr WR={oos_stats['wr']:.1f}% "
                  f"edge={oos_edge:.1f}pp PnL={oos_stats['pnl']:.1f}% "
                  f"{'PASS' if is_positive else 'FAIL'}")

        verdict = 'PASS' if positive_count == n_folds else 'FAIL'
        total_oos_pnl = sum(f['oos_pnl'] for f in folds)
        total_oos_trades = sum(f['oos_trades'] for f in folds)
        avg_oos_wr = np.mean([f['oos_wr'] for f in folds if f['oos_trades'] > 0])

        results[key] = {
            'pattern': pat,
            'direction': direction,
            'tp': tp, 'sl': sl,
            'folds': folds,
            'positive_folds': positive_count,
            'total_folds': n_folds,
            'total_oos_pnl': round(total_oos_pnl, 2),
            'total_oos_trades': total_oos_trades,
            'avg_oos_wr': round(avg_oos_wr, 1) if total_oos_trades > 0 else 0,
            'verdict': verdict,
        }

        print(f"    VERDICT: {verdict} ({positive_count}/{n_folds}), "
              f"total OOS PnL={total_oos_pnl:.1f}%, "
              f"total OOS trades={total_oos_trades}, "
              f"avg OOS WR={avg_oos_wr:.1f}%")

    return results


# ============================================================
# Phase 3: Portfolio WF (51 baseline vs 54 with H1)
# ============================================================

def _check_exit(pos, bar, opens, highs, lows, n_bars, atr_ratio):
    """Check if position exits at this bar. Returns result dict or None."""
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
        # Timeout -> DROP (not counted in PnL)
        return {'reason': 'TIMEOUT', 'drop': True}

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
    pnl -= FEE_PCT * LEVERAGE

    return {
        'entry_bar': entry_bar, 'exit_bar': bar, 'hold_bars': hold,
        'pnl_slot': pnl, 'reason': reason, 'drop': False,
    }


def simulate_portfolio(signals, tpsl_map, opens, highs, lows, n_bars,
                        atr_ratio, direction_cap, oos_start, oos_end):
    """Simulate N=9 multi-position portfolio on OOS period.
    Timeout trades are DROPPED (slot freed, no PnL recorded).
    Positions open at OOS end are force-closed (recorded).
    """
    positions = []
    trades = []
    SIZE_PCT = 100.0 / MAX_POSITIONS

    signals_in_range = [(s, p, d) for s, p, d in signals if oos_start <= s < oos_end]
    signals_sorted = sorted(signals_in_range, key=lambda x: x[0])
    sig_idx = 0

    for bar in range(oos_start, oos_end):
        # Check exits
        closed_slots = []
        for pos in positions:
            result = _check_exit(pos, bar, opens, highs, lows, n_bars, atr_ratio)
            if result is not None:
                if result.get('drop'):
                    # Timeout -> DROP (free slot, no PnL)
                    closed_slots.append(pos['slot'])
                else:
                    result['pattern'] = pos['pattern']
                    result['direction'] = pos['direction']
                    result['pnl_portfolio'] = result['pnl_slot'] * SIZE_PCT / 100
                    trades.append(result)
                    closed_slots.append(pos['slot'])
        positions = [p for p in positions if p['slot'] not in closed_slots]

        # Process signals at this bar
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

            positions.append({
                'slot': f"{pat}_{sig_bar}",
                'signal_bar': sig_bar,
                'entry_bar': entry_bar,
                'direction': direction,
                'pattern': pat,
                'tp_pct': tp_sl[0],
                'sl_pct': tp_sl[1],
            })

    # Force-close remaining positions at OOS end
    for pos in positions:
        entry_bar = pos['entry_bar']
        if entry_bar >= n_bars:
            continue
        entry = opens[entry_bar]
        if entry <= 0:
            continue
        exit_bar = min(oos_end - 1, n_bars - 1)
        exit_price = opens[exit_bar] if exit_bar < n_bars else opens[-1]
        direction = pos['direction']
        if direction == 'LONG':
            pnl = (exit_price / entry - 1) * 100 * LEVERAGE
        else:
            pnl = (1 - exit_price / entry) * 100 * LEVERAGE
        pnl -= FEE_PCT * LEVERAGE
        trades.append({
            'entry_bar': entry_bar, 'exit_bar': exit_bar,
            'hold_bars': exit_bar - entry_bar,
            'pnl_slot': pnl, 'reason': 'OOS_END',
            'pattern': pos['pattern'], 'direction': pos['direction'],
            'pnl_portfolio': pnl * SIZE_PCT / 100,
        })

    return trades


def compute_portfolio_metrics(trades):
    """Compute portfolio-level metrics from trade list."""
    if not trades:
        return {'trades': 0, 'wr': 0, 'pnl': 0, 'mdd': 0,
                'long_trades': 0, 'short_trades': 0,
                'long_pnl': 0, 'short_pnl': 0, 'pnl_per_trade': 0,
                'pnl_mdd': 0}

    wins = [t for t in trades if t['pnl_slot'] > 0]
    total_pnl = sum(t['pnl_portfolio'] for t in trades)
    wr = len(wins) / len(trades) * 100

    # MDD (compound equity)
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

    long_t = [t for t in trades if t['direction'] == 'LONG']
    short_t = [t for t in trades if t['direction'] == 'SHORT']

    return {
        'trades': len(trades),
        'wr': round(wr, 1),
        'pnl': round(total_pnl, 2),
        'mdd': round(max_dd, 2),
        'long_trades': len(long_t),
        'short_trades': len(short_t),
        'long_pnl': round(sum(t['pnl_portfolio'] for t in long_t), 2),
        'short_pnl': round(sum(t['pnl_portfolio'] for t in short_t), 2),
        'pnl_per_trade': round(total_pnl / len(trades), 3) if trades else 0,
        'pnl_mdd': round(total_pnl / max_dd, 2) if max_dd > 0 else 0,
    }


def phase3_portfolio_wf(rctypes, opens, highs, lows, closes, n_bars):
    """Compare 51pat baseline vs 54pat (51 + 3 H1) portfolio WF."""
    print("\n" + "=" * 80)
    print("PHASE 3: Portfolio WF — 51pat baseline vs 54pat with H1")
    print("=" * 80)

    # Load existing 51 patterns
    with open(PATTERNS_FILE) as f:
        pat_data = json.load(f)

    pL_51 = set(pat_data['patterns']['long'])
    pS_51 = set(pat_data['patterns']['short'])
    tpsl_51 = pat_data['patterns_tpsl']

    # 54 patterns = 51 + 3 H1 LONGs
    pL_54 = pL_51.copy()
    tpsl_54 = dict(tpsl_51)
    h1_added = []
    for key, info in H1_PATTERNS.items():
        pat = info['pattern']
        if pat not in pL_54:
            pL_54.add(pat)
            h1_added.append(pat)
        tpsl_54[pat] = [info['tp'], info['sl']]

    pS_54 = pS_51.copy()

    print(f"  51pat baseline: {len(pL_51)}L + {len(pS_51)}S = {len(pL_51)+len(pS_51)}")
    print(f"  54pat with H1:  {len(pL_54)}L + {len(pS_54)}S = {len(pL_54)+len(pS_54)}")
    print(f"  H1 patterns added: {h1_added}")

    # Compute ATR
    atr_ratio = compute_atr_ratio(highs, lows, closes)

    # Generate signals for both
    signals_51 = generate_signals(rctypes, pL_51, pS_51, start_idx=ATR_WARMUP)
    signals_54 = generate_signals(rctypes, pL_54, pS_54, start_idx=ATR_WARMUP)
    print(f"  Signals: 51pat={len(signals_51)}, 54pat={len(signals_54)}")

    # WF folds (same as direction_cap_wf_study)
    direction_cap = 8
    tradeable_start = ATR_WARMUP
    tradeable_bars = n_bars - tradeable_start
    n_folds = 3
    fold_size = tradeable_bars // n_folds

    folds_def = []
    for i in range(n_folds):
        oos_start = tradeable_start + i * fold_size
        oos_end = tradeable_start + (i + 1) * fold_size if i < n_folds - 1 else n_bars
        folds_def.append((oos_start, oos_end))

    print(f"\n  WF: {n_folds} folds x ~{fold_size/288:.0f}d, direction_cap={direction_cap}")
    for i, (s, e) in enumerate(folds_def):
        print(f"    Fold {i+1}: bars [{s}, {e}) ({(e-s)/288:.0f}d)")

    # Run portfolio WF for both
    results = {}
    for label, signals, tpsl in [
        ('pat51_baseline', signals_51, tpsl_51),
        ('pat54_with_h1', signals_54, tpsl_54),
    ]:
        fold_metrics = []
        for fold_idx, (oos_start, oos_end) in enumerate(folds_def):
            trades = simulate_portfolio(
                signals, tpsl, opens, highs, lows, n_bars,
                atr_ratio, direction_cap, oos_start, oos_end
            )
            m = compute_portfolio_metrics(trades)
            fold_metrics.append(m)

        results[label] = fold_metrics

    # Print results
    for label in ['pat51_baseline', 'pat54_with_h1']:
        print(f"\n  --- {label} ---")
        print(f"  {'Fold':<8} {'Trades':>7} {'L/S':>8} {'WR':>7} {'PnL':>9} {'MDD':>7} {'PnL/MDD':>8}")
        print("  " + "-" * 60)
        for i, m in enumerate(results[label]):
            pnl_mdd = m['pnl'] / m['mdd'] if m['mdd'] > 0 else 0
            print(f"  Fold {i+1}  {m['trades']:>7} {m['long_trades']:>3}/{m['short_trades']:<4} "
                  f"{m['wr']:>6.1f}% {m['pnl']:>+8.2f}% {m['mdd']:>6.2f}% {pnl_mdd:>7.2f}x")

        all_trades = sum(m['trades'] for m in results[label])
        total_pnl = sum(m['pnl'] for m in results[label])
        max_mdd = max(m['mdd'] for m in results[label])
        pnl_mdd = total_pnl / max_mdd if max_mdd > 0 else 0
        positive_folds = sum(1 for m in results[label] if m['pnl'] > 0)
        print("  " + "-" * 60)
        print(f"  TOTAL  {all_trades:>7} {'':>8} {'':>7} {total_pnl:>+8.2f}% {max_mdd:>6.2f}% "
              f"{pnl_mdd:>7.2f}x  [{positive_folds}/{len(results[label])} {'PASS' if positive_folds == len(results[label]) else 'FAIL'}]")

    # Improvement analysis
    print("\n  --- IMPROVEMENT (54pat - 51pat) ---")
    improvement = {}
    for i in range(n_folds):
        m51 = results['pat51_baseline'][i]
        m54 = results['pat54_with_h1'][i]
        d_trades = m54['trades'] - m51['trades']
        d_pnl = m54['pnl'] - m51['pnl']
        d_mdd = m54['mdd'] - m51['mdd']
        d_long = m54['long_trades'] - m51['long_trades']
        print(f"    Fold {i+1}: trades {d_trades:+d} (L {d_long:+d}), "
              f"PnL {d_pnl:+.2f}%, MDD {d_mdd:+.2f}%")

    total_51_pnl = sum(m['pnl'] for m in results['pat51_baseline'])
    total_54_pnl = sum(m['pnl'] for m in results['pat54_with_h1'])
    max_51_mdd = max(m['mdd'] for m in results['pat51_baseline'])
    max_54_mdd = max(m['mdd'] for m in results['pat54_with_h1'])

    improvement = {
        'pnl_delta': round(total_54_pnl - total_51_pnl, 2),
        'mdd_delta': round(max_54_mdd - max_51_mdd, 2),
        'pnl_mdd_51': round(total_51_pnl / max_51_mdd, 2) if max_51_mdd > 0 else 0,
        'pnl_mdd_54': round(total_54_pnl / max_54_mdd, 2) if max_54_mdd > 0 else 0,
        'long_trades_delta': sum(m54['long_trades'] for m54 in results['pat54_with_h1']) -
                              sum(m51['long_trades'] for m51 in results['pat51_baseline']),
    }
    print(f"\n    TOTAL: PnL delta={improvement['pnl_delta']:+.2f}%, "
          f"MDD delta={improvement['mdd_delta']:+.2f}%, "
          f"PnL/MDD: 51pat={improvement['pnl_mdd_51']:.2f}x -> 54pat={improvement['pnl_mdd_54']:.2f}x")

    # Build serializable results
    output = {}
    for label in ['pat51_baseline', 'pat54_with_h1']:
        folds_out = []
        for i, m in enumerate(results[label]):
            folds_out.append({
                'fold': i + 1,
                'oos_start': folds_def[i][0],
                'oos_end': folds_def[i][1],
                **m,
            })
        total_pnl = sum(m['pnl'] for m in results[label])
        max_mdd = max(m['mdd'] for m in results[label])
        positive_folds = sum(1 for m in results[label] if m['pnl'] > 0)
        output[label] = {
            'folds': folds_out,
            'total_pnl': round(total_pnl, 2),
            'max_mdd': round(max_mdd, 2),
            'pnl_mdd': round(total_pnl / max_mdd, 2) if max_mdd > 0 else 0,
            'positive_folds': positive_folds,
            'total_folds': n_folds,
            'verdict': 'PASS' if positive_folds == n_folds else 'FAIL',
        }

    output['improvement'] = improvement
    return output


# ============================================================
# Phase 4: Holdout Validation (last 7 days)
# ============================================================

def phase4_holdout(signal_index, opens, highs, lows, closes, n_bars):
    """Validate H1 patterns on last 7 days holdout."""
    print("\n" + "=" * 80)
    print("PHASE 4: Holdout Validation (last 7 days)")
    print("=" * 80)

    holdout_bars = 7 * 288  # 2016 bars
    holdout_start = n_bars - holdout_bars
    print(f"  Holdout: bars [{holdout_start}, {n_bars}) = {holdout_bars} bars ({holdout_bars/288:.0f}d)")

    # ATR computed on data up to holdout end (full data)
    atr_ratio = compute_atr_ratio(highs, lows, closes)

    results = {}
    for key, info in H1_PATTERNS.items():
        pat = info['pattern']
        direction = info['direction']
        tp = info['tp']
        sl = info['sl']
        baseline_wr = sl / (tp + sl) * 100

        sigs = signal_index.get(pat, [])
        holdout_sigs = [s for s in sigs if holdout_start <= s < n_bars]

        if not holdout_sigs:
            print(f"  {key}: No signals in holdout")
            results[key] = {
                'holdout_signals': 0, 'holdout_trades': 0,
                'wr': 0, 'wr_excess': 0, 'pnl': 0,
                'verdict': 'NO_DATA',
            }
            continue

        trades = bt_signals_atr(holdout_sigs, direction, tp, sl,
                                opens, highs, lows, n_bars,
                                atr_ratio, CLAMP_LO, CLAMP_HI)
        pnls = [t[2] for t in trades]
        wins = sum(1 for p in pnls if p > 0)
        wr = wins / len(pnls) * 100 if pnls else 0
        wr_excess = wr - baseline_wr
        stats = calc_stats(pnls)

        verdict = 'PASS' if wr_excess > 0 else 'FAIL'
        results[key] = {
            'holdout_signals': len(holdout_sigs),
            'holdout_trades': len(trades),
            'wr': round(wr, 1),
            'baseline_wr': round(baseline_wr, 1),
            'wr_excess': round(wr_excess, 1),
            'pnl': stats['pnl'],
            'verdict': verdict,
        }

        print(f"  {key}: sigs={len(holdout_sigs)}, trades={len(trades)}, "
              f"WR={wr:.1f}% (excess={wr_excess:.1f}pp), PnL={stats['pnl']:.1f}% -> {verdict}")

    return results


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 80)
    print("H1 Bullish LONG WF Validation")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("Patterns: BD-U-BD, MU-DN-IH, ST-MU-U (all LONG)")
    print("=" * 80)

    # Load data
    df, rctypes, opens, highs, lows, closes = load_data()
    n_bars = len(opens)

    # Build signal index
    signal_index = build_signal_index(rctypes)
    print(f"Signal index: {len(signal_index)} unique patterns")

    # Run all phases
    p1 = phase1_is_reproduction(signal_index, opens, highs, lows, closes, n_bars)
    p2 = phase2_individual_wf(signal_index, opens, highs, lows, closes, n_bars)
    p3 = phase3_portfolio_wf(rctypes, opens, highs, lows, closes, n_bars)
    p4 = phase4_holdout(signal_index, opens, highs, lows, closes, n_bars)

    # Final verdict
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)

    adoptable = []
    for key in H1_PATTERNS:
        pat = H1_PATTERNS[key]['pattern']
        wf_verdict = p2[key].get('verdict', 'FAIL')
        holdout_verdict = p4[key].get('verdict', 'FAIL')
        total_oos_pnl = p2[key].get('total_oos_pnl', 0)
        avg_oos_wr = p2[key].get('avg_oos_wr', 0)

        adopt = wf_verdict == 'PASS' and holdout_verdict in ('PASS', 'NO_DATA')
        status = 'ADOPT' if adopt else 'STOP'
        if adopt:
            adoptable.append(key)

        print(f"  {key}: WF={wf_verdict}, Holdout={holdout_verdict}, "
              f"OOS PnL={total_oos_pnl:.1f}%, OOS WR={avg_oos_wr:.1f}% -> {status}")

    # Portfolio verdict
    p3_51_verdict = p3['pat51_baseline']['verdict']
    p3_54_verdict = p3['pat54_with_h1']['verdict']
    pnl_improved = p3['improvement']['pnl_delta'] > 0
    portfolio_adopt = p3_54_verdict == 'PASS' and pnl_improved

    print(f"\n  Portfolio WF: 51pat={p3_51_verdict}, 54pat={p3_54_verdict}")
    print(f"  PnL improvement: {p3['improvement']['pnl_delta']:+.2f}% -> "
          f"{'ADOPT' if portfolio_adopt else 'STOP'}")

    final_verdict = 'ADOPT' if adoptable and portfolio_adopt else 'STOP'
    print(f"\n  FINAL: {final_verdict}")
    if adoptable:
        print(f"  Adoptable patterns: {adoptable}")
    else:
        print(f"  No patterns passed all validation gates")

    # Save results
    output = {
        'metadata': {
            'script': 'h1_bullish_long_wf_validation.py',
            'date': datetime.now().isoformat(),
            'data_file': DATA_FILE,
            'data_bars': n_bars,
            'patterns_tested': list(H1_PATTERNS.keys()),
            'protocol': {
                'leverage': LEVERAGE,
                'fee_pct': FEE_PCT,
                'slippage_buffer': SLIPPAGE_BUFFER,
                'timeout_bars': TIMEOUT_BARS,
                'max_positions': MAX_POSITIONS,
                'direction_cap': 8,
                'atr_period': ATR_PERIOD,
                'atr_window': ATR_WINDOW,
                'clamp_lo': CLAMP_LO,
                'clamp_hi': CLAMP_HI,
                'mc_seeds': MC_SEEDS,
            },
        },
        'phase1_is_reproduction': p1,
        'phase2_individual_wf': p2,
        'phase3_portfolio_wf': p3,
        'phase4_holdout': p4,
        'final_verdict': final_verdict,
        'adoptable_patterns': adoptable,
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
