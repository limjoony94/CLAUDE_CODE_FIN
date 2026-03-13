#!/usr/bin/env python3
"""
Trailing Breakeven SL Study

Reactive mechanism: when a position's unrealized profit reaches X% of TP distance,
move SL to entry price (breakeven). If price reverses, exit at breakeven (0 loss)
instead of original SL.

Sweep activation_pct: [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 999(baseline)]
For each: IS metrics, WF 3-fold OOS, MC discrimination, BE exit analysis, R:R.

Standard Research Protocol: compound, 0.10% fee, 0.02% slippage, 3x leverage.
N-pos: 9 slots, dir_cap 7, cascade 85%, agg_risk 8/15, momentum 1.5%/15min/1h,
timeout 288, tp_scale_factor 0.72, ATR [0.5,1.5].
"""

import json
import numpy as np
from pathlib import Path
import sys
import time
import warnings
warnings.filterwarnings('ignore')

start_time = time.time()

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scanner"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "production" / "pattern_5m"))

from pattern_scanner import (
    load_and_classify, build_signal_index, find_neutral_window,
    portfolio_npos, calc_stats_compound,
    compute_atr_ratio, compute_ema_slope,
    LEVERAGE, FEE_PCT, MAX_BARS, SLIPPAGE_BUFFER, MAX_DAILY_LOSS_PCT,
    DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI,
    DEFAULT_N_SLOTS, DEFAULT_DIRECTION_CAP, TIMEOUT_BARS,
    DEFAULT_AGG_RISK_COUNTER, DEFAULT_AGG_RISK_WITH,
    DEFAULT_MOMENTUM_LOOKBACK, DEFAULT_MOMENTUM_THRESHOLD,
    DEFAULT_MOMENTUM_COOLDOWN, DEFAULT_CASCADE_TIGHTEN_PCT,
    DEFAULT_REGIME_MULT, NPOS_EMA_PERIOD, NPOS_EMA_LOOKBACK,
)

DATA_FILE = "data/btc_5m_270days_reclassified.csv"
PATTERNS_FILE = "results/dynamic_patterns.json"
OUTPUT_FILE = "results/trailing_be_sl_study.json"

TP_SCALE_FACTOR = 0.72
ACTIVATION_PCTS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 999]  # 999 = disabled (baseline)
BARS_PER_DAY = 288
MC_SEEDS = [42, 123, 7]

NPOS_DEFAULTS = dict(
    n_slots=DEFAULT_N_SLOTS, direction_cap=DEFAULT_DIRECTION_CAP,
    regime_mult=DEFAULT_REGIME_MULT,
    agg_risk_counter=DEFAULT_AGG_RISK_COUNTER, agg_risk_with=DEFAULT_AGG_RISK_WITH,
    momentum_lookback=DEFAULT_MOMENTUM_LOOKBACK, momentum_threshold=DEFAULT_MOMENTUM_THRESHOLD,
    momentum_cooldown=DEFAULT_MOMENTUM_COOLDOWN,
    clamp_lo=DEFAULT_ATR_CLAMP_LO, clamp_hi=DEFAULT_ATR_CLAMP_HI,
    timeout_bars=TIMEOUT_BARS, cascade_tighten_pct=DEFAULT_CASCADE_TIGHTEN_PCT,
)


# ============================================================
# Modified N-pos simulator with trailing breakeven SL
# ============================================================

def _check_exit_npos_be(pos, bar, opens, highs, lows, n_bars, atr_ratio, fee,
                        clamp_lo=DEFAULT_ATR_CLAMP_LO, clamp_hi=DEFAULT_ATR_CLAMP_HI,
                        timeout_bars=TIMEOUT_BARS, activation_pct=999):
    """Check exit with trailing breakeven SL.

    Same as production _check_exit_npos but adds:
    - Track per-position MFE (max favorable excursion)
    - When MFE >= activation_pct * TP_distance, set effective SL to entry (breakeven)
    - Exit priority: TP > BE > SL (most favorable first)
    """
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
    # v1.59.5: daily loss cap (production parity)
    if sl_pct > 0:
        r = min(r, MAX_DAILY_LOSS_PCT / LEVERAGE / sl_pct)

    eff_tp = tp_pct * r + SLIPPAGE_BUFFER
    # Cascade SL override
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
    if hold >= timeout_bars:
        return {'entry_bar': entry_bar, 'exit_bar': bar, 'pnl_slot': 0,
                'reason': 'TIMEOUT', 'drop': True}

    h, l = highs[bar], lows[bar]

    # --- Trailing Breakeven SL logic ---
    # Update position's MFE tracking (using highs/lows seen so far INCLUDING this bar)
    if direction == 'LONG':
        # For LONG: track highest high since entry
        prev_max_price = pos.get('mfe_max_price', entry)
        cur_max_price = max(prev_max_price, h)
        pos['mfe_max_price'] = cur_max_price
        mfe_pct = (cur_max_price - entry) / entry * 100  # favorable excursion %
    else:
        # For SHORT: track lowest low since entry
        prev_min_price = pos.get('mfe_min_price', entry)
        cur_min_price = min(prev_min_price, l)
        pos['mfe_min_price'] = cur_min_price
        mfe_pct = (entry - cur_min_price) / entry * 100

    # TP distance in price % (before leverage)
    tp_distance_pct = eff_tp  # already in %

    # Check if breakeven is activated
    be_active = pos.get('be_active', False)
    if not be_active and activation_pct < 999 and mfe_pct >= activation_pct * tp_distance_pct:
        be_active = True
        pos['be_active'] = True

    # Determine exit: TP > BE > SL priority
    if direction == 'LONG':
        hit_tp = h >= tp_price
        hit_sl = l <= sl_price
        hit_be = be_active and l <= entry  # price touches or crosses entry
    else:
        hit_tp = l <= tp_price
        hit_sl = h >= sl_price
        hit_be = be_active and h >= entry

    if not hit_tp and not hit_sl and not hit_be:
        return None

    # Resolve which exit fires
    # Priority: TP first, then BE, then SL (most favorable)
    bo = opens[bar]

    if hit_tp and hit_sl and hit_be:
        # All three hit — use distance from open
        tp_dist = abs(tp_price - bo)
        sl_dist = abs(sl_price - bo)
        be_dist = abs(entry - bo)
        candidates = [('TP', tp_price, tp_dist), ('BE', entry, be_dist), ('SL', sl_price, sl_dist)]
        candidates.sort(key=lambda x: x[2])  # closest to open fires first
        # Among equidistant, prefer TP > BE > SL
        exit_label = candidates[0][0]
        exit_price = candidates[0][1]
    elif hit_tp and hit_be:
        tp_dist = abs(tp_price - bo)
        be_dist = abs(entry - bo)
        if tp_dist <= be_dist:
            exit_label, exit_price = 'TP', tp_price
        else:
            exit_label, exit_price = 'BE', entry
    elif hit_tp and hit_sl:
        tp_dist = abs(tp_price - bo)
        sl_dist = abs(sl_price - bo)
        if tp_dist <= sl_dist:
            exit_label, exit_price = 'TP', tp_price
        else:
            exit_label, exit_price = 'SL', sl_price
    elif hit_be and hit_sl:
        # BE is closer to entry than SL, so BE fires first (protects position)
        # BE is at entry, SL is beyond entry (worse) — BE always fires first
        exit_label, exit_price = 'BE', entry
    elif hit_tp:
        exit_label, exit_price = 'TP', tp_price
    elif hit_be:
        exit_label, exit_price = 'BE', entry
    else:
        exit_label, exit_price = 'SL', sl_price

    if direction == 'LONG':
        pnl = (exit_price / entry - 1) * 100 * LEVERAGE
    else:
        pnl = (1 - exit_price / entry) * 100 * LEVERAGE
    pnl -= fee

    return {'entry_bar': entry_bar, 'exit_bar': bar, 'pnl_slot': pnl,
            'reason': exit_label, 'drop': False}


def portfolio_npos_be(signal_tuples, opens, highs, lows, closes, n_bars,
                      atr_ratio, ema_slope, start_bar, end_bar,
                      activation_pct=999,
                      n_slots=DEFAULT_N_SLOTS, direction_cap=DEFAULT_DIRECTION_CAP,
                      regime_mult=DEFAULT_REGIME_MULT,
                      agg_risk_counter=DEFAULT_AGG_RISK_COUNTER,
                      agg_risk_with=DEFAULT_AGG_RISK_WITH,
                      momentum_lookback=DEFAULT_MOMENTUM_LOOKBACK,
                      momentum_threshold=DEFAULT_MOMENTUM_THRESHOLD,
                      momentum_cooldown=DEFAULT_MOMENTUM_COOLDOWN,
                      clamp_lo=DEFAULT_ATR_CLAMP_LO, clamp_hi=DEFAULT_ATR_CLAMP_HI,
                      timeout_bars=TIMEOUT_BARS,
                      cascade_tighten_pct=DEFAULT_CASCADE_TIGHTEN_PCT):
    """N-position portfolio simulator with trailing breakeven SL.

    Identical to scanner portfolio_npos except uses _check_exit_npos_be
    which adds breakeven SL tracking.
    """
    size_pct = 100.0 / n_slots
    fee = FEE_PCT * LEVERAGE

    positions = []
    trades = []
    equity = 100.0
    peak_equity = 100.0
    max_dd_mtm = 0.0

    max_corr_loss = 0.0
    max_sim_positions = 0
    total_blocked = {'momentum': 0, 'agg_risk': 0, 'dir_cap': 0, 'dup_pat': 0, 'max_pos': 0}
    corr_events = []
    momentum_pause_until = {'LONG': -1, 'SHORT': -1}

    # Exit reason counters
    exit_counts = {'TP': 0, 'SL': 0, 'BE': 0, 'TIMEOUT': 0, 'OOS_END': 0}

    signals_in_range = [(s, p, d, tp, sl) for s, p, d, tp, sl in signal_tuples
                        if start_bar <= s < end_bar]
    signals_sorted = sorted(signals_in_range, key=lambda x: x[0])
    sig_idx = 0

    for bar in range(start_bar, end_bar):
        # 1. Check exits
        closed_slots = []
        bar_pnl_sum = 0.0
        bar_sl_count = 0

        for pos in positions:
            result = _check_exit_npos_be(pos, bar, opens, highs, lows, n_bars,
                                         atr_ratio, fee, clamp_lo, clamp_hi,
                                         timeout_bars, activation_pct)
            if result is not None:
                if result.get('drop', False):
                    closed_slots.append(pos['slot'])
                    exit_counts['TIMEOUT'] = exit_counts.get('TIMEOUT', 0) + 1
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
                reason = result['reason']
                exit_counts[reason] = exit_counts.get(reason, 0) + 1
                if reason == 'SL':
                    bar_sl_count += 1

        # Cascade SL tightening: after SL exits, tighten same-dir remaining SLs
        if cascade_tighten_pct > 0 and bar_sl_count > 0:
            keep_ratio = 1.0 - cascade_tighten_pct / 100.0
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
                    if (atr_ratio is not None and sig < len(atr_ratio)
                            and not np.isnan(atr_ratio[sig])):
                        r_val = max(clamp_lo, min(clamp_hi, atr_ratio[sig]))
                    else:
                        r_val = 1.0
                    p_sl = pos['sl_pct']
                    if p_sl > 0:
                        r_val = min(r_val, MAX_DAILY_LOSS_PCT / LEVERAGE / p_sl)
                    cur_eff_sl = pos.get('eff_sl_override') or (p_sl * r_val)
                    pos['eff_sl_override'] = cur_eff_sl * keep_ratio

        positions = [p for p in positions if p['slot'] not in closed_slots]

        if bar_pnl_sum < 0 and bar_sl_count >= 2:
            loss_pct = abs(bar_pnl_sum)
            if loss_pct > max_corr_loss:
                max_corr_loss = loss_pct
            corr_events.append((bar, bar_sl_count, loss_pct))

        equity += bar_pnl_sum
        if equity > peak_equity:
            peak_equity = equity

        # Momentum guard
        if momentum_lookback > 0 and momentum_threshold > 0 and bar >= momentum_lookback:
            price_now = closes[bar]
            price_ago = closes[bar - momentum_lookback]
            if price_ago > 0:
                pct_change = (price_now / price_ago - 1) * 100
                if pct_change > momentum_threshold:
                    momentum_pause_until['SHORT'] = bar + momentum_cooldown
                elif pct_change < -momentum_threshold:
                    momentum_pause_until['LONG'] = bar + momentum_cooldown

        # 2. Process entries
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
                if s > 0 and direction == 'SHORT':
                    sm = regime_mult
                elif s <= 0 and direction == 'LONG':
                    sm = regime_mult

            # Aggregate risk cap
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
                        if p_sl > 0:
                            p_r = min(p_r, MAX_DAILY_LOSS_PCT / LEVERAGE / p_sl)
                        p_eff_sl = p_sl * p_r
                        p_sm = p.get('size_mult', 1.0)
                        existing_exposure += p_eff_sl * (1.0 / n_slots) * LEVERAGE * p_sm

                new_r = 1.0
                if (atr_ratio is not None and sig_bar < len(atr_ratio)
                        and not np.isnan(atr_ratio[sig_bar])):
                    new_r = max(clamp_lo, min(clamp_hi, atr_ratio[sig_bar]))
                if sl_pct > 0:
                    new_r = min(new_r, MAX_DAILY_LOSS_PCT / LEVERAGE / sl_pct)
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
            })

        if len(positions) > max_sim_positions:
            max_sim_positions = len(positions)

        # Mark-to-market MDD
        if positions and bar < n_bars:
            mtm_equity = equity
            for pos in positions:
                eb = pos['entry_bar']
                if eb >= n_bars or bar < eb:
                    continue
                entry_price = opens[eb]
                if entry_price <= 0:
                    continue
                if pos['direction'] == 'LONG':
                    unr = (closes[bar] / entry_price - 1) * 100 * LEVERAGE
                else:
                    unr = (1 - closes[bar] / entry_price) * 100 * LEVERAGE
                sm_val = pos.get('size_mult', 1.0)
                mtm_equity += unr * (size_pct / 100) * sm_val
            if mtm_equity > peak_equity:
                peak_equity = mtm_equity
            dd = (peak_equity - mtm_equity) / peak_equity * 100 if peak_equity > 0 else 0
            if dd > max_dd_mtm:
                max_dd_mtm = dd
        elif not positions:
            if equity > peak_equity:
                peak_equity = equity
            dd = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0
            if dd > max_dd_mtm:
                max_dd_mtm = dd

    # Force-close remaining
    for pos in positions:
        entry_bar_val = pos['entry_bar']
        if entry_bar_val >= n_bars:
            continue
        entry_val = opens[entry_bar_val]
        if entry_val <= 0:
            continue
        exit_bar = min(end_bar - 1, n_bars - 1)
        exit_price = opens[exit_bar]
        if pos['direction'] == 'LONG':
            pnl = (exit_price / entry_val - 1) * 100 * LEVERAGE
        else:
            pnl = (1 - exit_price / entry_val) * 100 * LEVERAGE
        pnl -= fee
        sm_val = pos.get('size_mult', 1.0)
        trades.append({
            'entry_bar': entry_bar_val, 'exit_bar': exit_bar, 'pnl_slot': pnl,
            'reason': 'OOS_END', 'pattern': pos['pattern'],
            'direction': pos['direction'], 'size_mult': sm_val,
            'pnl_portfolio': pnl * (size_pct / 100) * sm_val,
        })
        exit_counts['OOS_END'] = exit_counts.get('OOS_END', 0) + 1

    stats = {
        'max_corr_loss': round(max_corr_loss, 2),
        'max_sim_positions': max_sim_positions,
        'corr_events': len(corr_events),
        'blocked': total_blocked,
        'mdd_mtm': round(max_dd_mtm, 2),
        'exit_counts': exit_counts,
    }
    return trades, stats


# ============================================================
# Helper functions
# ============================================================

def load_patterns(filepath=PATTERNS_FILE):
    with open(filepath) as f:
        data = json.load(f)
    details = data.get('pattern_details') or {}
    result = {}
    for k, v in details.items():
        result[k] = {
            'pattern': v['pattern'], 'direction': v['direction'],
            'tp': v['tp'], 'sl': v['sl'],
        }
    return result


def apply_tp_factor(patterns, tp_factor):
    result = {}
    for k, v in patterns.items():
        result[k] = {
            'pattern': v['pattern'], 'direction': v['direction'],
            'tp': round(max(0.3, v['tp'] * tp_factor), 3),
            'sl': v['sl'],
        }
    return result


def build_signal_tuples(patterns, sig_idx):
    tuples = []
    for k, v in patterns.items():
        pat_name = v.get('pattern') or k.rsplit('_', 1)[0]
        if pat_name in sig_idx:
            for bar in sig_idx[pat_name]:
                tuples.append((bar, k, v['direction'], v['tp'], v['sl']))
    return tuples


def run_npos_be(signal_tuples, opens, highs, lows, closes, n_bars,
                atr_ratio, ema_slope, start_bar, end_bar,
                activation_pct=999, **extra):
    kwargs = {**NPOS_DEFAULTS}
    kwargs.update(extra)
    trades, raw = portfolio_npos_be(
        signal_tuples, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, start_bar, end_bar,
        activation_pct=activation_pct, **kwargs
    )
    stats = calc_stats_compound(trades)
    if raw.get('mdd_mtm', 0) > 0:
        stats['mdd'] = raw['mdd_mtm']
        stats['pnl_mdd'] = round(stats['pnl'] / stats['mdd'], 2) if stats['mdd'] > 0 else 0
    stats.update({k: v for k, v in raw.items() if k not in stats})
    return trades, stats


def compute_trade_metrics(trades):
    """Compute avg win, avg loss, BE WR, R:R from trade list."""
    wins = [t['pnl_slot'] for t in trades if t['pnl_slot'] > 0]
    losses = [abs(t['pnl_slot']) for t in trades if t['pnl_slot'] < 0]
    neutrals = [t for t in trades if t['pnl_slot'] == 0]  # BE exits (pnl ~ -fee)
    # Actually BE exits have pnl = -fee (small negative), not exactly 0
    # Count by reason instead
    be_exits = sum(1 for t in trades if t.get('reason') == 'BE')

    avg_win = float(np.mean(wins)) if wins else 0
    avg_loss = float(np.mean(losses)) if losses else 1e-9
    rr = avg_win / avg_loss if avg_loss > 0 else float('inf')
    be_wr = 1 / (1 + rr) * 100 if rr > 0 else 50.0

    return {
        'avg_win': round(avg_win, 3),
        'avg_loss': round(avg_loss, 3),
        'rr': round(rr, 3),
        'be_wr': round(be_wr, 1),
        'be_exits': be_exits,
    }


def run_wf(signal_tuples, opens, highs, lows, closes, n_bars,
           atr_ratio, ema_slope, ns, ne, n_folds=3, activation_pct=999, **extra):
    """Expanding window walk-forward. Correct formula: is_end = int(n*(fi+1)/(nf+1))."""
    n = ne - ns
    results = []
    for fi in range(n_folds):
        is_end = ns + int(n * (fi + 1) / (n_folds + 1))
        oos_start = is_end
        oos_end = ns + int(n * (fi + 2) / (n_folds + 1))
        oos_end = min(oos_end, ne)
        if oos_start >= ne or oos_end <= oos_start:
            continue
        _, stats = run_npos_be(signal_tuples, opens, highs, lows, closes, n_bars,
                               atr_ratio, ema_slope, oos_start, oos_end,
                               activation_pct=activation_pct, **extra)
        results.append({
            'fold': fi + 1, 'pnl': stats.get('pnl', 0),
            'wr': stats.get('wr', 0), 'trades': stats.get('trades', 0),
            'mdd': stats.get('mdd', 0),
        })
    oos_pnl = sum(r['pnl'] for r in results)
    all_pass = all(r['pnl'] > 0 for r in results) if results else False
    return results, oos_pnl, all_pass


def mc_discrimination(signal_tuples, opens, highs, lows, closes, n_bars,
                      atr_ratio, ema_slope, ns, ne, real_pnl,
                      activation_pct=999, seeds=None, n_sims=5000, **extra):
    """Monte Carlo sign randomization test."""
    if seeds is None:
        seeds = MC_SEEDS
    p_values = []
    for seed in seeds:
        rng = np.random.RandomState(seed)
        count_ge = 0
        for _ in range(n_sims):
            # Randomize trade signs
            shuffled = []
            for s in signal_tuples:
                if rng.random() < 0.5:
                    # Flip direction
                    new_dir = 'SHORT' if s[2] == 'LONG' else 'LONG'
                    shuffled.append((s[0], s[1], new_dir, s[3], s[4]))
                else:
                    shuffled.append(s)
            _, stats = run_npos_be(shuffled, opens, highs, lows, closes, n_bars,
                                   atr_ratio, ema_slope, ns, ne,
                                   activation_pct=activation_pct, **extra)
            if stats.get('pnl', 0) >= real_pnl:
                count_ge += 1
        p = count_ge / n_sims
        p_values.append(round(p, 4))
    return p_values, max(p_values)


# ============================================================
# Main Study
# ============================================================

print("=" * 70)
print("TRAILING BREAKEVEN SL STUDY")
print("=" * 70)

# Load data
df = load_and_classify(DATA_FILE)
opens = df['open'].values.astype(np.float64)
highs = df['high'].values.astype(np.float64)
lows = df['low'].values.astype(np.float64)
closes = df['close'].values.astype(np.float64)
n_bars = len(df)
type_codes = df['candle_type'].values

atr_ratio = compute_atr_ratio(highs, lows, closes)
ema_slope = compute_ema_slope(closes)

signal_index = build_signal_index(type_codes, n_bars)
ns, ne = find_neutral_window(closes)
data_days = (ne - ns) / BARS_PER_DAY
print(f"Data: {n_bars} bars, neutral: {ns}-{ne} ({data_days:.0f}d)")

# Load patterns and apply TP scale factor
raw_patterns = load_patterns()
patterns = apply_tp_factor(raw_patterns, TP_SCALE_FACTOR)
print(f"Patterns: {len(patterns)} (TP x {TP_SCALE_FACTOR})")

signal_tuples = build_signal_tuples(patterns, signal_index)
print(f"Signal tuples: {len(signal_tuples)}")

# ============================================================
# Phase 1: IS sweep across activation_pct values
# ============================================================
print("\n" + "=" * 70)
print("PHASE 1: IS SWEEP")
print("=" * 70)

phase1_results = {}

for act_pct in ACTIVATION_PCTS:
    label = f"{act_pct}" if act_pct < 999 else "baseline"
    t0 = time.time()

    trades, stats = run_npos_be(
        signal_tuples, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, ns, ne,
        activation_pct=act_pct
    )

    tm = compute_trade_metrics(trades)
    elapsed = time.time() - t0

    n_trades = stats.get('trades', 0)
    wr = stats.get('wr', 0)
    pnl = stats.get('pnl', 0)
    mdd = stats.get('mdd', 0)
    pnl_mdd = stats.get('pnl_mdd', 0)
    trades_per_day = n_trades / data_days if data_days > 0 else 0

    exit_counts = stats.get('exit_counts', {})
    tp_count = exit_counts.get('TP', 0)
    sl_count = exit_counts.get('SL', 0)
    be_count = exit_counts.get('BE', 0)
    to_count = exit_counts.get('TIMEOUT', 0)

    wr_margin = wr - tm['be_wr']

    print(f"\n--- activation_pct={label} ---")
    print(f"  Trades: {n_trades} ({trades_per_day:.1f}/day), WR: {wr:.1f}%, PnL: {pnl:+.1f}%, MDD: {mdd:.2f}%")
    print(f"  PnL/MDD: {pnl_mdd:.1f}x, R:R: {tm['rr']:.3f}, BE_WR: {tm['be_wr']:.1f}%, WR_margin: {wr_margin:+.1f}pp")
    print(f"  Exits: TP={tp_count}, SL={sl_count}, BE={be_count}, TO={to_count}")
    print(f"  Avg win: {tm['avg_win']:.3f}%, Avg loss: {tm['avg_loss']:.3f}%")
    print(f"  [{elapsed:.1f}s]")

    phase1_results[label] = {
        'activation_pct': act_pct,
        'trades': n_trades,
        'trades_per_day': round(trades_per_day, 1),
        'wr': round(wr, 1),
        'pnl': round(pnl, 2),
        'mdd': round(mdd, 2),
        'pnl_mdd': round(pnl_mdd, 1),
        'rr': tm['rr'],
        'be_wr': tm['be_wr'],
        'wr_margin': round(wr_margin, 1),
        'avg_win': tm['avg_win'],
        'avg_loss': tm['avg_loss'],
        'exit_counts': exit_counts,
        'blocked': stats.get('blocked', {}),
        'max_corr_loss': stats.get('max_corr_loss', 0),
    }

# ============================================================
# Phase 2: WF OOS for all activation_pct values
# ============================================================
print("\n" + "=" * 70)
print("PHASE 2: WALK-FORWARD OOS (3-fold expanding window)")
print("=" * 70)

phase2_results = {}

for act_pct in ACTIVATION_PCTS:
    label = f"{act_pct}" if act_pct < 999 else "baseline"
    t0 = time.time()

    wf_folds, oos_pnl, all_pass = run_wf(
        signal_tuples, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, ns, ne, n_folds=3,
        activation_pct=act_pct
    )

    verdict = "PASS" if all_pass else "FAIL"
    fold_str = ', '.join(f"F{r['fold']}:{r['pnl']:+.1f}%" for r in wf_folds)
    elapsed = time.time() - t0

    print(f"\n--- activation_pct={label} ---")
    print(f"  WF: {verdict} | OOS: {oos_pnl:+.1f}% | {fold_str}")
    print(f"  [{elapsed:.1f}s]")

    phase2_results[label] = {
        'activation_pct': act_pct,
        'verdict': verdict,
        'oos_pnl': round(oos_pnl, 1),
        'folds': wf_folds,
    }

# ============================================================
# Phase 3: MC discrimination for top candidates + baseline
# ============================================================
print("\n" + "=" * 70)
print("PHASE 3: MC DISCRIMINATION (3-seed, sign randomization)")
print("=" * 70)

# Select top 2 by PnL/MDD (excluding baseline) + baseline for MC test
non_baseline = [(k, v) for k, v in phase1_results.items() if k != 'baseline']
non_baseline.sort(key=lambda x: x[1].get('pnl_mdd', 0), reverse=True)
mc_candidates = [x[0] for x in non_baseline[:2]] + ['baseline']

phase3_results = {}

# MC discrimination: only run for candidates that improve over baseline
# Since this is a mechanism comparison (not strategy validation),
# we test whether the best BE variant has genuine edge vs random direction.
# Baseline already validated at MC p<0.01 (v1.61.0).
MC_SIMS_STUDY = 100

baseline_pnl_mc = phase1_results.get('baseline', {}).get('pnl', 0)
any_improves = any(phase1_results[k]['pnl'] > baseline_pnl_mc
                   for k in phase1_results if k != 'baseline')

if any_improves:
    for label in mc_candidates:
        act_pct = phase1_results[label]['activation_pct']
        real_pnl = phase1_results[label]['pnl']
        t0 = time.time()

        p_values = []
        for seed in MC_SEEDS:
            rng = np.random.RandomState(seed)
            count_ge = 0
            for _ in range(MC_SIMS_STUDY):
                shuffled = []
                for s in signal_tuples:
                    if rng.random() < 0.5:
                        new_dir = 'SHORT' if s[2] == 'LONG' else 'LONG'
                        shuffled.append((s[0], s[1], new_dir, s[3], s[4]))
                    else:
                        shuffled.append(s)
                rand_trades, rand_raw = portfolio_npos_be(
                    shuffled, opens, highs, lows, closes, n_bars,
                    atr_ratio, ema_slope, ns, ne,
                    activation_pct=act_pct, **NPOS_DEFAULTS)
                rand_stats = calc_stats_compound(rand_trades)
                if rand_stats.get('pnl', 0) >= real_pnl:
                    count_ge += 1
            p = count_ge / MC_SIMS_STUDY
            p_values.append(round(p, 4))

        max_p = max(p_values)
        disc = "DISC" if max_p < 0.01 else "NON-DISC"
        elapsed = time.time() - t0

        print(f"\n--- activation_pct={label} ---")
        print(f"  p-values: {p_values}, max_p: {max_p:.4f} -> {disc}")
        print(f"  [{elapsed:.1f}s]")

        phase3_results[label] = {
            'activation_pct': act_pct,
            'p_values': p_values,
            'max_p': max_p,
            'discriminating': max_p < 0.01,
        }
else:
    print("\n  SKIPPED — no BE variant improves over baseline PnL.")
    print("  Baseline already validated at MC p<0.01 (v1.61.0).")
    print("  MC test is moot when the mechanism REDUCES performance.")
    phase3_results = {'skipped': True, 'reason': 'all_variants_worse_than_baseline'}

# ============================================================
# Phase 4: Net impact analysis — BE saves vs costs
# ============================================================
print("\n" + "=" * 70)
print("PHASE 4: NET IMPACT ANALYSIS")
print("=" * 70)

baseline_data = phase1_results.get('baseline', {})
baseline_pnl = baseline_data.get('pnl', 0)
baseline_mdd = baseline_data.get('mdd', 1)
baseline_wr = baseline_data.get('wr', 0)
baseline_pnl_mdd = baseline_data.get('pnl_mdd', 0)
baseline_sl = baseline_data.get('exit_counts', {}).get('SL', 0)
baseline_tp = baseline_data.get('exit_counts', {}).get('TP', 0)

print(f"\nBaseline: PnL={baseline_pnl:+.1f}%, MDD={baseline_mdd:.2f}%, "
      f"PnL/MDD={baseline_pnl_mdd:.1f}x, WR={baseline_wr:.1f}%, "
      f"TP={baseline_tp}, SL={baseline_sl}")

phase4_results = {}

for act_pct in ACTIVATION_PCTS:
    if act_pct >= 999:
        continue
    label = f"{act_pct}"
    d = phase1_results[label]
    be_count = d.get('exit_counts', {}).get('BE', 0)
    sl_count = d.get('exit_counts', {}).get('SL', 0)
    tp_count = d.get('exit_counts', {}).get('TP', 0)
    n_trades = d.get('trades', 0)

    sl_saved = baseline_sl - sl_count  # SLs avoided (converted to BE)
    tp_lost = baseline_tp - tp_count    # TPs lost (price reversed before TP after BE activation)
    pnl_delta = d['pnl'] - baseline_pnl
    mdd_delta = d['mdd'] - baseline_mdd
    pnl_mdd_delta = d.get('pnl_mdd', 0) - baseline_pnl_mdd
    wr_delta = d['wr'] - baseline_wr

    # WF delta
    wf_oos_delta = phase2_results.get(label, {}).get('oos_pnl', 0) - phase2_results.get('baseline', {}).get('oos_pnl', 0)

    print(f"\n--- activation_pct={act_pct} ---")
    print(f"  BE exits: {be_count} ({be_count/n_trades*100:.1f}% of trades)" if n_trades > 0 else "  BE exits: 0")
    print(f"  SLs saved: {sl_saved} (baseline {baseline_sl} → {sl_count})")
    print(f"  TPs lost:  {tp_lost} (baseline {baseline_tp} → {tp_count})")
    print(f"  PnL delta: {pnl_delta:+.1f}%, MDD delta: {mdd_delta:+.2f}%")
    print(f"  PnL/MDD delta: {pnl_mdd_delta:+.1f}x, WR delta: {wr_delta:+.1f}pp")
    print(f"  WF OOS delta: {wf_oos_delta:+.1f}%")

    phase4_results[label] = {
        'activation_pct': act_pct,
        'be_exits': be_count,
        'be_pct': round(be_count / n_trades * 100, 1) if n_trades > 0 else 0,
        'sl_saved': sl_saved,
        'tp_lost': tp_lost,
        'pnl_delta': round(pnl_delta, 2),
        'mdd_delta': round(mdd_delta, 2),
        'pnl_mdd_delta': round(pnl_mdd_delta, 1),
        'wr_delta': round(wr_delta, 1),
        'wf_oos_delta': round(wf_oos_delta, 1),
    }

# ============================================================
# Summary and Recommendation
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)

header = f"{'Act%':>6} | {'Trades':>6} | {'WR':>6} | {'PnL':>9} | {'MDD':>6} | {'PnL/MDD':>8} | {'R:R':>6} | {'WR_m':>5} | {'BE':>4} | {'WF_OOS':>8} | {'WF':>4}"
print(header)
print("-" * len(header))

for act_pct in ACTIVATION_PCTS:
    label = f"{act_pct}" if act_pct < 999 else "baseline"
    d = phase1_results[label]
    wf = phase2_results.get(label, {})
    be_count = d.get('exit_counts', {}).get('BE', 0)
    act_str = f"{act_pct:.0f}%" if act_pct < 999 else "OFF"

    print(f"{act_str:>6} | {d['trades']:>6} | {d['wr']:>5.1f}% | {d['pnl']:>+8.1f}% | {d['mdd']:>5.2f}% | "
          f"{d.get('pnl_mdd',0):>7.1f}x | {d['rr']:>5.3f} | {d['wr_margin']:>+4.1f} | {be_count:>4} | "
          f"{wf.get('oos_pnl',0):>+7.1f}% | {wf.get('verdict','?'):>4}")

# Find best candidate
best_label = None
best_score = -float('inf')
for act_pct in ACTIVATION_PCTS:
    if act_pct >= 999:
        continue
    label = f"{act_pct}"
    d = phase1_results[label]
    wf = phase2_results.get(label, {})
    # Score: PnL/MDD improvement + WF OOS improvement (both vs baseline)
    pnl_mdd_delta = d.get('pnl_mdd', 0) - baseline_pnl_mdd
    oos_delta = wf.get('oos_pnl', 0) - phase2_results.get('baseline', {}).get('oos_pnl', 0)
    score = pnl_mdd_delta + oos_delta / 10  # Weight WF at 10%
    if wf.get('verdict') == 'FAIL':
        score -= 1000  # Penalize WF failures
    if score > best_score:
        best_score = score
        best_label = label

print(f"\nBest candidate: activation_pct={best_label}")
if best_label:
    bd = phase1_results[best_label]
    bwf = phase2_results.get(best_label, {})
    print(f"  IS: PnL={bd['pnl']:+.1f}%, PnL/MDD={bd.get('pnl_mdd',0):.1f}x, WR={bd['wr']:.1f}%")
    print(f"  WF OOS: {bwf.get('oos_pnl',0):+.1f}% ({bwf.get('verdict','?')})")
    pnl_delta = bd['pnl'] - baseline_pnl
    pnl_mdd_delta = bd.get('pnl_mdd', 0) - baseline_pnl_mdd
    if pnl_delta > 0 and pnl_mdd_delta > 0:
        print(f"  RECOMMENDATION: GO (PnL +{pnl_delta:.1f}%, PnL/MDD +{pnl_mdd_delta:.1f}x)")
    elif pnl_delta < 0 and pnl_mdd_delta <= 0:
        print(f"  RECOMMENDATION: KEEP_BASELINE (no improvement)")
    else:
        print(f"  RECOMMENDATION: MARGINAL (PnL {pnl_delta:+.1f}%, PnL/MDD {pnl_mdd_delta:+.1f}x) — needs judgment")

# ============================================================
# Save results
# ============================================================
elapsed_total = time.time() - start_time

output = {
    'study': 'trailing_be_sl_study',
    'date': time.strftime('%Y-%m-%d %H:%M:%S'),
    'protocol': {
        'fee_pct': FEE_PCT,
        'leverage': LEVERAGE,
        'slippage_buffer': SLIPPAGE_BUFFER,
        'tp_scale_factor': TP_SCALE_FACTOR,
        'data_bars': n_bars,
        'neutral_window': [int(ns), int(ne)],
        'data_days': round(data_days, 1),
        'patterns': len(patterns),
        'mc_sims': MC_SIMS_STUDY,
        'mc_seeds': MC_SEEDS,
        'wf_folds': 3,
    },
    'phase1_is_sweep': phase1_results,
    'phase2_wf_oos': phase2_results,
    'phase3_mc_discrimination': phase3_results,
    'phase4_net_impact': phase4_results,
    'best_candidate': best_label,
    'elapsed_seconds': round(elapsed_total, 1),
}

with open(OUTPUT_FILE, 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nResults saved to {OUTPUT_FILE}")
print(f"Total elapsed: {elapsed_total:.0f}s ({elapsed_total/60:.1f}min)")
