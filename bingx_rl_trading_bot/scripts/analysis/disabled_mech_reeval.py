#!/usr/bin/env python3
"""
Disabled Mechanism Re-evaluation Study

Re-evaluates 5 disabled mechanisms under current conditions (111 patterns,
tp_scale 0.72, vol_mult cap) to determine if environment changes warrant
re-enabling any of them.

Disabled mechanisms:
  1. Regime Sizing (regime_mult): counter-trend sizing. Previously: -121.7% PnL
  2. Adaptive Leverage: volatility-based leverage. Previously: M2+M4 redundancy -46.94
  3. Equity Curve Trading: DD-based sizing reduction. Previously: PnL -11.4%
  4. Correlation-Aware Entry: same-dir ratio blocking. Previously: redundant w/ regime
  5. Loss Burst Brake: consecutive loss → entry block. Previously: redundant w/ momentum

Method:
  - Regime Sizing: direct portfolio_npos param (regime_mult)
  - Others: custom simulation extending portfolio_npos loop with mechanism flags
  - IS + 3-fold expanding WF for top variants
  - MC sign-randomization (3 seeds) for discrimination test

Standard Research Protocol: compound, 0.10% fee, 0.02% slippage, 3x leverage.
"""

import json
import numpy as np
import time
import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

start_time = time.time()

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scanner"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "production" / "pattern_5m"))

from pattern_scanner import (
    load_and_classify, build_signal_index, find_neutral_window,
    portfolio_npos, calc_stats_compound,
    compute_atr_ratio, compute_ema_slope,
    LEVERAGE, FEE_PCT, MAX_BARS,
    DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI,
    DEFAULT_N_SLOTS, DEFAULT_DIRECTION_CAP, TIMEOUT_BARS,
    DEFAULT_AGG_RISK_COUNTER, DEFAULT_AGG_RISK_WITH,
    DEFAULT_MOMENTUM_LOOKBACK, DEFAULT_MOMENTUM_THRESHOLD,
    DEFAULT_MOMENTUM_COOLDOWN, DEFAULT_CASCADE_TIGHTEN_PCT,
    DEFAULT_REGIME_MULT, MAX_DAILY_LOSS_PCT,
    SLIPPAGE_BUFFER, _check_exit_npos,
)

DATA_FILE = "data/btc_5m_270days_reclassified.csv"
PATTERNS_FILE = "results/dynamic_patterns.json"
OUTPUT_FILE = "results/disabled_mech_reeval.json"

TP_SCALE = 0.72  # Current production tp_scale_factor

NPOS_DEFAULTS = dict(
    n_slots=DEFAULT_N_SLOTS, direction_cap=DEFAULT_DIRECTION_CAP,
    regime_mult=DEFAULT_REGIME_MULT,
    agg_risk_counter=DEFAULT_AGG_RISK_COUNTER, agg_risk_with=DEFAULT_AGG_RISK_WITH,
    momentum_lookback=DEFAULT_MOMENTUM_LOOKBACK, momentum_threshold=DEFAULT_MOMENTUM_THRESHOLD,
    momentum_cooldown=DEFAULT_MOMENTUM_COOLDOWN,
    clamp_lo=DEFAULT_ATR_CLAMP_LO, clamp_hi=DEFAULT_ATR_CLAMP_HI,
    timeout_bars=TIMEOUT_BARS, cascade_tighten_pct=DEFAULT_CASCADE_TIGHTEN_PCT,
)

MC_SEEDS = [42, 123, 7]


def load_patterns(filepath=PATTERNS_FILE):
    with open(filepath) as f:
        data = json.load(f)
    details = data.get('pattern_details') or {}
    result = {}
    for k, v in details.items():
        result[k] = {
            'pattern': v['pattern'], 'direction': v['direction'],
            'tp': round(v['tp'] * TP_SCALE, 3),  # Apply tp_scale_factor
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


def run_npos(signal_tuples, opens, highs, lows, closes, n_bars,
             atr_ratio, ema_slope, start_bar, end_bar, **extra):
    kwargs = {**NPOS_DEFAULTS}
    kwargs.update(extra)
    trades, raw = portfolio_npos(
        signal_tuples, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, start_bar, end_bar, **kwargs
    )
    stats = calc_stats_compound(trades)
    if raw.get('mdd_mtm', 0) > 0:
        stats['mdd'] = raw['mdd_mtm']
        stats['pnl_mdd'] = round(stats['pnl'] / stats['mdd'], 2) if stats['mdd'] > 0 else 0
    stats.update({k: v for k, v in raw.items() if k not in stats})
    return trades, stats


def run_wf(signal_tuples, opens, highs, lows, closes, n_bars,
           atr_ratio, ema_slope, ns, ne, n_folds=3, **extra):
    total = ne - ns
    min_train = total // 3
    fold_size = total // (n_folds + 1)
    results = []
    for fold in range(n_folds):
        train_end = ns + min_train + fold_size * fold
        test_start = train_end
        test_end = min(train_end + fold_size, ne)
        if test_start >= ne or test_end <= test_start:
            continue
        _, stats = run_npos(signal_tuples, opens, highs, lows, closes, n_bars,
                            atr_ratio, ema_slope, test_start, test_end, **extra)
        results.append({'fold': fold + 1, 'pnl': stats.get('pnl', 0),
                        'wr': stats.get('wr', 0), 'trades': stats.get('trades', 0),
                        'mdd': stats.get('mdd', 0)})
    oos_pnl = sum(r['pnl'] for r in results)
    all_pass = all(r['pnl'] > 0 for r in results) if results else False
    return results, oos_pnl, all_pass


# ═══════════════════════════════════════════════════════════════
# Extended portfolio simulator with additional mechanisms
# ═══════════════════════════════════════════════════════════════

def portfolio_npos_extended(
    signal_tuples, opens, highs, lows, closes, n_bars,
    atr_ratio, ema_slope, start_bar, end_bar,
    n_slots=DEFAULT_N_SLOTS, direction_cap=DEFAULT_DIRECTION_CAP,
    regime_mult=DEFAULT_REGIME_MULT,
    agg_risk_counter=DEFAULT_AGG_RISK_COUNTER,
    agg_risk_with=DEFAULT_AGG_RISK_WITH,
    momentum_lookback=DEFAULT_MOMENTUM_LOOKBACK,
    momentum_threshold=DEFAULT_MOMENTUM_THRESHOLD,
    momentum_cooldown=DEFAULT_MOMENTUM_COOLDOWN,
    clamp_lo=DEFAULT_ATR_CLAMP_LO, clamp_hi=DEFAULT_ATR_CLAMP_HI,
    timeout_bars=TIMEOUT_BARS,
    cascade_tighten_pct=DEFAULT_CASCADE_TIGHTEN_PCT,
    # --- Extended mechanism params ---
    equity_curve_enabled=False,
    ec_ema_trades=30,
    ec_size_mult=0.5,
    loss_burst_enabled=False,
    lb_threshold=3,
    lb_block_bars=144,
    lb_window_bars=288,
    corr_aware_enabled=False,
    ca_dir_pct_threshold=0.70,
    adaptive_lev_enabled=False,
    al_method='wr_confidence',
    al_min_lev=1.0,
    al_max_lev=3.0,
    al_window=12,
):
    """Extended N-pos simulator with all 5 disabled mechanisms.

    Directly extends portfolio_npos logic to include:
    - Equity Curve Trading: per-direction cumPnL vs SMA → size_mult reduction
    - Loss Burst Brake: consecutive losses → direction block for N bars
    - Correlation-Aware Entry: block counter-regime entries when same-dir ratio > threshold
    - Adaptive Leverage: WR-confidence based leverage adjustment

    Returns: (trades_list, stats_dict) — same format as portfolio_npos
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
    total_blocked = {
        'momentum': 0, 'agg_risk': 0, 'dir_cap': 0, 'dup_pat': 0,
        'max_pos': 0, 'loss_burst': 0, 'corr_aware': 0, 'equity_curve': 0,
    }
    corr_events = []
    momentum_pause_until = {'LONG': -1, 'SHORT': -1}

    # Loss Burst state: track recent losses per direction
    loss_burst_block_until = {'LONG': -1, 'SHORT': -1}
    recent_losses = {'LONG': [], 'SHORT': []}  # list of (bar, loss) tuples

    # Equity Curve state: per-direction trade PnLs for SMA
    dir_trade_pnls = {'LONG': [], 'SHORT': []}

    # Adaptive Leverage state: rolling window of recent trades
    recent_trade_results = []  # list of (bar, win_bool)

    # Filter and sort signals in range
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
            result = _check_exit_npos(pos, bar, opens, highs, lows, n_bars,
                                      atr_ratio, fee, clamp_lo, clamp_hi,
                                      timeout_bars)
            if result is not None:
                if result.get('drop', False):
                    closed_slots.append(pos['slot'])
                    continue
                result['pattern'] = pos['pattern']
                result['direction'] = pos['direction']
                sm = pos.get('size_mult', 1.0)
                lev_mult = pos.get('lev_mult', 1.0)
                result['size_mult'] = sm
                result['lev_mult'] = lev_mult
                pnl_portfolio = result['pnl_slot'] * (size_pct / 100) * sm * lev_mult
                result['pnl_portfolio'] = pnl_portfolio
                trades.append(result)
                closed_slots.append(pos['slot'])
                bar_pnl_sum += pnl_portfolio
                if result['reason'] == 'SL':
                    bar_sl_count += 1

                # Update mechanism states
                d = result['direction']
                is_win = result['pnl_slot'] > 0

                # Loss Burst: track losses
                if loss_burst_enabled and not is_win:
                    recent_losses[d].append(bar)
                    # Clean old losses outside window
                    recent_losses[d] = [b for b in recent_losses[d]
                                        if bar - b <= lb_window_bars]
                    if len(recent_losses[d]) >= lb_threshold:
                        loss_burst_block_until[d] = bar + lb_block_bars
                        recent_losses[d] = []
                elif loss_burst_enabled and is_win:
                    # Reset consecutive count on win
                    recent_losses[d] = []

                # Equity Curve: track per-direction PnLs
                if equity_curve_enabled:
                    dir_trade_pnls[d].append(pnl_portfolio)

                # Adaptive Leverage: track results
                if adaptive_lev_enabled:
                    recent_trade_results.append((bar, is_win))

        # Cascade SL tightening
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
                        r = max(clamp_lo, min(clamp_hi, atr_ratio[sig]))
                    else:
                        r = 1.0
                    p_sl = pos['sl_pct']
                    if p_sl > 0:
                        r = min(r, MAX_DAILY_LOSS_PCT / LEVERAGE / p_sl)
                    cur_eff_sl = pos.get('eff_sl_override') or (p_sl * r)
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

        # Compute adaptive leverage multiplier for this bar
        current_lev_mult = 1.0
        if adaptive_lev_enabled and len(recent_trade_results) >= al_window:
            window_results = recent_trade_results[-al_window:]
            wr = sum(1 for _, w in window_results if w) / len(window_results)
            # WR confidence: scale leverage between min and max based on WR
            # WR=0.5 → min_lev, WR=1.0 → max_lev
            wr_norm = max(0, min(1, (wr - 0.5) / 0.5))
            target_lev = al_min_lev + wr_norm * (al_max_lev - al_min_lev)
            current_lev_mult = target_lev / LEVERAGE  # ratio vs fixed 3x

        # 2. Process entries
        while sig_idx < len(signals_sorted) and signals_sorted[sig_idx][0] == bar:
            sig_bar, pat, direction, tp_pct, sl_pct = signals_sorted[sig_idx]
            sig_idx += 1

            if len(positions) >= n_slots:
                total_blocked['max_pos'] += 1
                continue

            # Direction cap
            dir_count = sum(1 for p in positions if p['direction'] == direction)
            if dir_count >= direction_cap:
                total_blocked['dir_cap'] += 1
                continue

            # Duplicate pattern check
            if any(p['pattern'] == pat for p in positions):
                total_blocked['dup_pat'] += 1
                continue

            entry_bar = sig_bar + 1
            if entry_bar >= n_bars:
                continue

            # Momentum guard check
            if momentum_lookback > 0 and bar < momentum_pause_until.get(direction, -1):
                total_blocked['momentum'] += 1
                continue

            # === Loss Burst Brake ===
            if loss_burst_enabled and bar < loss_burst_block_until.get(direction, -1):
                total_blocked['loss_burst'] += 1
                continue

            # === Correlation-Aware Entry ===
            if corr_aware_enabled and len(positions) >= 2:
                same_dir = sum(1 for p in positions if p['direction'] == direction)
                dir_ratio = same_dir / len(positions) if positions else 0
                # Block if adding would push ratio above threshold
                is_uptrend = ema_slope[bar] > 0 if bar < len(ema_slope) else False
                is_counter = ((is_uptrend and direction == 'SHORT') or
                              (not is_uptrend and direction == 'LONG'))
                if is_counter and dir_ratio >= ca_dir_pct_threshold:
                    total_blocked['corr_aware'] += 1
                    continue

            # Regime sizing
            sm = 1.0
            if regime_mult is not None and bar < len(ema_slope):
                s = ema_slope[bar]
                if s > 0 and direction == 'SHORT':
                    sm = regime_mult
                elif s <= 0 and direction == 'LONG':
                    sm = regime_mult

            # === Equity Curve Trading ===
            ec_mult = 1.0
            if equity_curve_enabled:
                dpnls = dir_trade_pnls.get(direction, [])
                if len(dpnls) >= ec_ema_trades:
                    cum_pnl = sum(dpnls)
                    sma_pnl = sum(dpnls[-ec_ema_trades:])
                    if cum_pnl < sma_pnl:
                        ec_mult = ec_size_mult

            # Aggregate risk cap check
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
                new_exposure = new_eff_sl * (1.0 / n_slots) * LEVERAGE * sm * ec_mult

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
                'size_mult': sm * ec_mult,
                'lev_mult': current_lev_mult,
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
                sm = pos.get('size_mult', 1.0)
                lm = pos.get('lev_mult', 1.0)
                mtm_equity += unr * (size_pct / 100) * sm * lm
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
        lm = pos.get('lev_mult', 1.0)
        trades.append({
            'entry_bar': entry_bar, 'exit_bar': exit_bar, 'pnl_slot': pnl,
            'reason': 'OOS_END', 'pattern': pos['pattern'],
            'direction': pos['direction'], 'size_mult': sm, 'lev_mult': lm,
            'pnl_portfolio': pnl * (size_pct / 100) * sm * lm,
        })

    stats = {
        'max_corr_loss': round(max_corr_loss, 2),
        'max_sim_positions': max_sim_positions,
        'corr_events': len(corr_events),
        'blocked': total_blocked,
        'mdd_mtm': round(max_dd_mtm, 2),
    }
    return trades, stats


def run_npos_ext(signal_tuples, opens, highs, lows, closes, n_bars,
                 atr_ratio, ema_slope, start_bar, end_bar, **extra):
    """Run extended N-pos sim and compute compound stats."""
    trades, raw = portfolio_npos_extended(
        signal_tuples, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, start_bar, end_bar, **extra
    )
    stats = calc_stats_compound(trades)
    if raw.get('mdd_mtm', 0) > 0:
        stats['mdd'] = raw['mdd_mtm']
        stats['pnl_mdd'] = round(stats['pnl'] / stats['mdd'], 2) if stats['mdd'] > 0 else 0
    stats.update({k: v for k, v in raw.items() if k not in stats})
    return trades, stats


def run_wf_ext(signal_tuples, opens, highs, lows, closes, n_bars,
               atr_ratio, ema_slope, ns, ne, n_folds=3, **extra):
    """WF with extended sim."""
    total = ne - ns
    min_train = total // 3
    fold_size = total // (n_folds + 1)
    results = []
    for fold in range(n_folds):
        train_end = ns + min_train + fold_size * fold
        test_start = train_end
        test_end = min(train_end + fold_size, ne)
        if test_start >= ne or test_end <= test_start:
            continue
        _, stats = run_npos_ext(signal_tuples, opens, highs, lows, closes, n_bars,
                                atr_ratio, ema_slope, test_start, test_end, **extra)
        results.append({'fold': fold + 1, 'pnl': stats.get('pnl', 0),
                        'wr': stats.get('wr', 0), 'trades': stats.get('trades', 0),
                        'mdd': stats.get('mdd', 0)})
    oos_pnl = sum(r['pnl'] for r in results)
    all_pass = all(r['pnl'] > 0 for r in results) if results else False
    return results, oos_pnl, all_pass


def mc_discrimination(trades, n_sims=5000, seeds=None):
    """Monte Carlo sign-randomization: is observed PnL better than random signs?"""
    if seeds is None:
        seeds = MC_SEEDS
    if not trades:
        return 1.0
    pnls = [t['pnl_portfolio'] for t in trades]
    observed = sum(pnls)
    max_p = 0.0
    for seed in seeds:
        rng = np.random.RandomState(seed)
        count_ge = 0
        for _ in range(n_sims):
            signs = rng.choice([-1, 1], size=len(pnls))
            rand_pnl = sum(p * s for p, s in zip(pnls, signs))
            if rand_pnl >= observed:
                count_ge += 1
        p_val = count_ge / n_sims
        if p_val > max_p:
            max_p = p_val
    return max_p


def fmt_stats(s):
    return (f"T={s.get('trades',0)} WR={s.get('wr',0):.1f}% "
            f"PnL={s.get('pnl',0):+.1f}% MDD={s.get('mdd',0):.2f}% "
            f"P/M={s.get('pnl_mdd',0):.1f}")


def fmt_wf(wf_folds):
    return ', '.join(f"F{r['fold']}:{r['pnl']:+.1f}%" for r in wf_folds)


# ═══════════════════════════════════════════════════════════
# Load data
# ═══════════════════════════════════════════════════════════
print("=" * 70)
print("DISABLED MECHANISM RE-EVALUATION STUDY")
print("=" * 70)

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
nw = find_neutral_window(closes)
if nw is None:
    print("ERROR: No neutral window found")
    sys.exit(1)
ns, ne = nw
print(f"Data: {n_bars} bars, neutral: {ns}-{ne} ({(ne-ns)//288:.0f}d)")

patterns = load_patterns()
print(f"Patterns: {len(patterns)} (TP scaled x{TP_SCALE})")

base_tuples = build_signal_tuples(patterns, signal_index)
print(f"Total signal events: {len(base_tuples)}")

all_results = {
    "study": "disabled_mech_reeval",
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "tp_scale": TP_SCALE,
    "patterns": len(patterns),
    "data_bars": n_bars,
    "neutral_window": [int(ns), int(ne)],
    "phases": {},
}


# ═══════════════════════════════════════════════════════════
# PHASE 0: Baseline (all 5 mechanisms OFF — current production)
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 0: BASELINE (all disabled mechanisms OFF)")
print("=" * 70)

base_trades, base_stats = run_npos(
    base_tuples, opens, highs, lows, closes, n_bars,
    atr_ratio, ema_slope, ns, ne
)
print(f"IS: {fmt_stats(base_stats)}")

base_wf_folds, base_wf_pnl, base_wf_pass = run_wf(
    base_tuples, opens, highs, lows, closes, n_bars,
    atr_ratio, ema_slope, ns, ne
)
print(f"WF: {fmt_wf(base_wf_folds)} | Total={base_wf_pnl:+.1f}% | {'PASS' if base_wf_pass else 'FAIL'}")

all_results['phases']['P0_baseline'] = {
    'is': base_stats,
    'wf_folds': base_wf_folds,
    'wf_pnl': round(base_wf_pnl, 1),
    'wf_pass': base_wf_pass,
}


# ═══════════════════════════════════════════════════════════
# PHASE 1: Regime Sizing
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 1: REGIME SIZING (counter-trend size reduction)")
print("=" * 70)

regime_variants = [
    ('regime_0.3', 0.3),
    ('regime_0.5', 0.5),
    ('regime_0.7', 0.7),
]

p1_results = {}
for name, mult in regime_variants:
    _, stats = run_npos(
        base_tuples, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, ns, ne, regime_mult=mult
    )
    delta_pnl = stats.get('pnl', 0) - base_stats.get('pnl', 0)
    delta_mdd = stats.get('mdd', 0) - base_stats.get('mdd', 0)
    print(f"  {name}: {fmt_stats(stats)} | dPnL={delta_pnl:+.1f}% dMDD={delta_mdd:+.2f}%")
    p1_results[name] = {
        'regime_mult': mult,
        'is': stats,
        'delta_pnl': round(delta_pnl, 1),
        'delta_mdd': round(delta_mdd, 2),
    }

# WF for best variant
best_regime = max(p1_results.items(), key=lambda x: x[1]['is'].get('pnl_mdd', 0))
best_regime_name = best_regime[0]
best_regime_mult = best_regime[1]['regime_mult']
print(f"\n  Best IS variant: {best_regime_name} (PnL/MDD={best_regime[1]['is'].get('pnl_mdd',0):.1f})")

wf_folds, wf_pnl, wf_pass = run_wf(
    base_tuples, opens, highs, lows, closes, n_bars,
    atr_ratio, ema_slope, ns, ne, regime_mult=best_regime_mult
)
print(f"  WF: {fmt_wf(wf_folds)} | Total={wf_pnl:+.1f}% | {'PASS' if wf_pass else 'FAIL'}")

p1_results[best_regime_name]['wf_folds'] = wf_folds
p1_results[best_regime_name]['wf_pnl'] = round(wf_pnl, 1)
p1_results[best_regime_name]['wf_pass'] = wf_pass
all_results['phases']['P1_regime_sizing'] = p1_results


# ═══════════════════════════════════════════════════════════
# PHASE 2: Equity Curve Trading
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 2: EQUITY CURVE TRADING (DD-based sizing reduction)")
print("=" * 70)

ec_variants = [
    ('ec_ema20_s50', 20, 0.50),
    ('ec_ema30_s50', 30, 0.50),
    ('ec_ema20_s25', 20, 0.25),
    ('ec_ema30_s25', 30, 0.25),
]

p2_results = {}
for name, ema_trades, size_mult in ec_variants:
    _, stats = run_npos_ext(
        base_tuples, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, ns, ne,
        equity_curve_enabled=True, ec_ema_trades=ema_trades, ec_size_mult=size_mult,
    )
    delta_pnl = stats.get('pnl', 0) - base_stats.get('pnl', 0)
    delta_mdd = stats.get('mdd', 0) - base_stats.get('mdd', 0)
    print(f"  {name}: {fmt_stats(stats)} | dPnL={delta_pnl:+.1f}% dMDD={delta_mdd:+.2f}%")
    p2_results[name] = {
        'ema_trades': ema_trades, 'size_mult': size_mult,
        'is': stats,
        'delta_pnl': round(delta_pnl, 1),
        'delta_mdd': round(delta_mdd, 2),
    }

best_ec = max(p2_results.items(), key=lambda x: x[1]['is'].get('pnl_mdd', 0))
best_ec_name = best_ec[0]
print(f"\n  Best IS variant: {best_ec_name} (PnL/MDD={best_ec[1]['is'].get('pnl_mdd',0):.1f})")

wf_folds, wf_pnl, wf_pass = run_wf_ext(
    base_tuples, opens, highs, lows, closes, n_bars,
    atr_ratio, ema_slope, ns, ne,
    equity_curve_enabled=True,
    ec_ema_trades=best_ec[1]['ema_trades'],
    ec_size_mult=best_ec[1]['size_mult'],
)
print(f"  WF: {fmt_wf(wf_folds)} | Total={wf_pnl:+.1f}% | {'PASS' if wf_pass else 'FAIL'}")

p2_results[best_ec_name]['wf_folds'] = wf_folds
p2_results[best_ec_name]['wf_pnl'] = round(wf_pnl, 1)
p2_results[best_ec_name]['wf_pass'] = wf_pass
all_results['phases']['P2_equity_curve'] = p2_results


# ═══════════════════════════════════════════════════════════
# PHASE 3: Loss Burst Brake
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 3: LOSS BURST BRAKE (consecutive loss → entry block)")
print("=" * 70)

lb_variants = [
    ('lb_3c_144b', 3, 144, 288),
    ('lb_3c_288b', 3, 288, 288),
    ('lb_4c_144b', 4, 144, 288),
    ('lb_4c_288b', 4, 288, 288),
    ('lb_2c_72b', 2, 72, 144),
]

p3_results = {}
for name, thresh, block, window in lb_variants:
    _, stats = run_npos_ext(
        base_tuples, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, ns, ne,
        loss_burst_enabled=True, lb_threshold=thresh,
        lb_block_bars=block, lb_window_bars=window,
    )
    delta_pnl = stats.get('pnl', 0) - base_stats.get('pnl', 0)
    delta_mdd = stats.get('mdd', 0) - base_stats.get('mdd', 0)
    blocked = stats.get('blocked', {}).get('loss_burst', 0)
    print(f"  {name}: {fmt_stats(stats)} | dPnL={delta_pnl:+.1f}% dMDD={delta_mdd:+.2f}% blocked={blocked}")
    p3_results[name] = {
        'threshold': thresh, 'block_bars': block, 'window_bars': window,
        'is': stats,
        'delta_pnl': round(delta_pnl, 1),
        'delta_mdd': round(delta_mdd, 2),
        'blocked': blocked,
    }

best_lb = max(p3_results.items(), key=lambda x: x[1]['is'].get('pnl_mdd', 0))
best_lb_name = best_lb[0]
print(f"\n  Best IS variant: {best_lb_name} (PnL/MDD={best_lb[1]['is'].get('pnl_mdd',0):.1f})")

wf_folds, wf_pnl, wf_pass = run_wf_ext(
    base_tuples, opens, highs, lows, closes, n_bars,
    atr_ratio, ema_slope, ns, ne,
    loss_burst_enabled=True,
    lb_threshold=best_lb[1]['threshold'],
    lb_block_bars=best_lb[1]['block_bars'],
    lb_window_bars=best_lb[1]['window_bars'],
)
print(f"  WF: {fmt_wf(wf_folds)} | Total={wf_pnl:+.1f}% | {'PASS' if wf_pass else 'FAIL'}")

p3_results[best_lb_name]['wf_folds'] = wf_folds
p3_results[best_lb_name]['wf_pnl'] = round(wf_pnl, 1)
p3_results[best_lb_name]['wf_pass'] = wf_pass
all_results['phases']['P3_loss_burst_brake'] = p3_results


# ═══════════════════════════════════════════════════════════
# PHASE 4: Correlation-Aware Entry
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 4: CORRELATION-AWARE ENTRY (same-dir ratio blocking)")
print("=" * 70)

ca_variants = [
    ('ca_70pct', 0.70),
    ('ca_60pct', 0.60),
    ('ca_80pct', 0.80),
    ('ca_50pct', 0.50),
]

p4_results = {}
for name, threshold in ca_variants:
    _, stats = run_npos_ext(
        base_tuples, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, ns, ne,
        corr_aware_enabled=True, ca_dir_pct_threshold=threshold,
    )
    delta_pnl = stats.get('pnl', 0) - base_stats.get('pnl', 0)
    delta_mdd = stats.get('mdd', 0) - base_stats.get('mdd', 0)
    blocked = stats.get('blocked', {}).get('corr_aware', 0)
    print(f"  {name}: {fmt_stats(stats)} | dPnL={delta_pnl:+.1f}% dMDD={delta_mdd:+.2f}% blocked={blocked}")
    p4_results[name] = {
        'dir_pct_threshold': threshold,
        'is': stats,
        'delta_pnl': round(delta_pnl, 1),
        'delta_mdd': round(delta_mdd, 2),
        'blocked': blocked,
    }

best_ca = max(p4_results.items(), key=lambda x: x[1]['is'].get('pnl_mdd', 0))
best_ca_name = best_ca[0]
print(f"\n  Best IS variant: {best_ca_name} (PnL/MDD={best_ca[1]['is'].get('pnl_mdd',0):.1f})")

wf_folds, wf_pnl, wf_pass = run_wf_ext(
    base_tuples, opens, highs, lows, closes, n_bars,
    atr_ratio, ema_slope, ns, ne,
    corr_aware_enabled=True,
    ca_dir_pct_threshold=best_ca[1]['dir_pct_threshold'],
)
print(f"  WF: {fmt_wf(wf_folds)} | Total={wf_pnl:+.1f}% | {'PASS' if wf_pass else 'FAIL'}")

p4_results[best_ca_name]['wf_folds'] = wf_folds
p4_results[best_ca_name]['wf_pnl'] = round(wf_pnl, 1)
p4_results[best_ca_name]['wf_pass'] = wf_pass
all_results['phases']['P4_correlation_aware'] = p4_results


# ═══════════════════════════════════════════════════════════
# PHASE 5: Adaptive Leverage
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 5: ADAPTIVE LEVERAGE (WR-confidence based)")
print("=" * 70)

al_variants = [
    ('al_w12_1_3', 12, 1.0, 3.0),
    ('al_w20_1_3', 20, 1.0, 3.0),
    ('al_w12_2_3', 12, 2.0, 3.0),
    ('al_w8_1_3',  8, 1.0, 3.0),
]

p5_results = {}
for name, window, min_lev, max_lev in al_variants:
    _, stats = run_npos_ext(
        base_tuples, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, ns, ne,
        adaptive_lev_enabled=True, al_window=window,
        al_min_lev=min_lev, al_max_lev=max_lev,
    )
    delta_pnl = stats.get('pnl', 0) - base_stats.get('pnl', 0)
    delta_mdd = stats.get('mdd', 0) - base_stats.get('mdd', 0)
    print(f"  {name}: {fmt_stats(stats)} | dPnL={delta_pnl:+.1f}% dMDD={delta_mdd:+.2f}%")
    p5_results[name] = {
        'window': window, 'min_lev': min_lev, 'max_lev': max_lev,
        'is': stats,
        'delta_pnl': round(delta_pnl, 1),
        'delta_mdd': round(delta_mdd, 2),
    }

best_al = max(p5_results.items(), key=lambda x: x[1]['is'].get('pnl_mdd', 0))
best_al_name = best_al[0]
print(f"\n  Best IS variant: {best_al_name} (PnL/MDD={best_al[1]['is'].get('pnl_mdd',0):.1f})")

wf_folds, wf_pnl, wf_pass = run_wf_ext(
    base_tuples, opens, highs, lows, closes, n_bars,
    atr_ratio, ema_slope, ns, ne,
    adaptive_lev_enabled=True,
    al_window=best_al[1]['window'],
    al_min_lev=best_al[1]['min_lev'],
    al_max_lev=best_al[1]['max_lev'],
)
print(f"  WF: {fmt_wf(wf_folds)} | Total={wf_pnl:+.1f}% | {'PASS' if wf_pass else 'FAIL'}")

p5_results[best_al_name]['wf_folds'] = wf_folds
p5_results[best_al_name]['wf_pnl'] = round(wf_pnl, 1)
p5_results[best_al_name]['wf_pass'] = wf_pass
all_results['phases']['P5_adaptive_leverage'] = p5_results


# ═══════════════════════════════════════════════════════════
# PHASE 6: MC Discrimination for any WF-passing variants
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 6: MC DISCRIMINATION TEST (3-seed, p<0.01)")
print("=" * 70)

wf_passing = []
for phase_key, phase_data in all_results['phases'].items():
    if phase_key == 'P0_baseline':
        continue
    if isinstance(phase_data, dict):
        for var_name, var_data in phase_data.items():
            if isinstance(var_data, dict) and var_data.get('wf_pass'):
                wf_passing.append((phase_key, var_name, var_data))

if not wf_passing:
    print("  No WF-passing variants found. All mechanisms remain DISABLED.")
else:
    for phase_key, var_name, var_data in wf_passing:
        print(f"\n  Testing {phase_key}/{var_name}...")

        # Re-run to get trades for MC
        extra = {}
        if 'regime_mult' in var_data:
            extra['regime_mult'] = var_data['regime_mult']
        elif 'ema_trades' in var_data:
            extra = dict(equity_curve_enabled=True,
                         ec_ema_trades=var_data['ema_trades'],
                         ec_size_mult=var_data['size_mult'])
        elif 'threshold' in var_data:
            extra = dict(loss_burst_enabled=True,
                         lb_threshold=var_data['threshold'],
                         lb_block_bars=var_data['block_bars'],
                         lb_window_bars=var_data['window_bars'])
        elif 'dir_pct_threshold' in var_data:
            extra = dict(corr_aware_enabled=True,
                         ca_dir_pct_threshold=var_data['dir_pct_threshold'])
        elif 'window' in var_data:
            extra = dict(adaptive_lev_enabled=True,
                         al_window=var_data['window'],
                         al_min_lev=var_data['min_lev'],
                         al_max_lev=var_data['max_lev'])

        # Use extended sim for non-regime variants
        if 'regime_mult' in var_data:
            trades, _ = run_npos(
                base_tuples, opens, highs, lows, closes, n_bars,
                atr_ratio, ema_slope, ns, ne, **extra
            )
        else:
            trades, _ = run_npos_ext(
                base_tuples, opens, highs, lows, closes, n_bars,
                atr_ratio, ema_slope, ns, ne, **extra
            )

        mc_p = mc_discrimination(trades)
        disc = "DISCRIMINATING" if mc_p < 0.01 else "NON-DISC"
        print(f"    MC p={mc_p:.4f} → {disc}")

        var_data['mc_p'] = round(mc_p, 4)
        var_data['discriminating'] = mc_p < 0.01


# ═══════════════════════════════════════════════════════════
# PHASE 7: Summary
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)
print(f"\n{'Mechanism':<25} {'Best Variant':<20} {'IS PnL':<12} {'IS MDD':<10} "
      f"{'IS P/M':<10} {'WF OOS':<12} {'WF':<8} {'Rec':<12}")
print("-" * 110)

baseline_pmdd = base_stats.get('pnl_mdd', 0)

for phase_key, phase_name, phase_data_key in [
    ('P1_regime_sizing', 'Regime Sizing', 'P1_regime_sizing'),
    ('P2_equity_curve', 'Equity Curve', 'P2_equity_curve'),
    ('P3_loss_burst_brake', 'Loss Burst Brake', 'P3_loss_burst_brake'),
    ('P4_correlation_aware', 'Corr-Aware Entry', 'P4_correlation_aware'),
    ('P5_adaptive_leverage', 'Adaptive Leverage', 'P5_adaptive_leverage'),
]:
    phase_data = all_results['phases'].get(phase_data_key, {})
    # Find best variant
    best = None
    best_name = 'N/A'
    for vn, vd in phase_data.items():
        if not isinstance(vd, dict) or 'is' not in vd:
            continue
        if best is None or vd['is'].get('pnl_mdd', 0) > best['is'].get('pnl_mdd', 0):
            best = vd
            best_name = vn

    if best is None:
        print(f"  {phase_name:<25} {'N/A':<20} {'N/A':<12} {'N/A':<10} {'N/A':<10} {'N/A':<12} {'N/A':<8} {'KEEP OFF':<12}")
        continue

    is_pnl = best['is'].get('pnl', 0)
    is_mdd = best['is'].get('mdd', 0)
    is_pmdd = best['is'].get('pnl_mdd', 0)
    wf_pnl = best.get('wf_pnl', 'N/A')
    wf_p = 'PASS' if best.get('wf_pass') else 'FAIL' if 'wf_pass' in best else 'N/A'
    disc = best.get('discriminating', None)

    # Recommendation logic
    if best.get('wf_pass') and best.get('discriminating') and is_pmdd > baseline_pmdd * 1.05:
        rec = "ENABLE"
    elif best.get('wf_pass') and is_pmdd > baseline_pmdd:
        rec = "MAYBE"
    else:
        rec = "KEEP OFF"

    wf_str = f"{wf_pnl:+.1f}%" if isinstance(wf_pnl, (int, float)) else wf_pnl
    print(f"  {phase_name:<25} {best_name:<20} {is_pnl:+.1f}%{'':<5} {is_mdd:.2f}%{'':<3} "
          f"{is_pmdd:.1f}{'':<5} {wf_str:<12} {wf_p:<8} {rec:<12}")

print(f"\n  Baseline: PnL={base_stats.get('pnl',0):+.1f}% MDD={base_stats.get('mdd',0):.2f}% "
      f"P/M={baseline_pmdd:.1f} WF OOS={base_wf_pnl:+.1f}%")

# Save results
all_results['baseline'] = {
    'is': base_stats,
    'wf_pnl': round(base_wf_pnl, 1),
    'wf_pass': base_wf_pass,
}

elapsed = time.time() - start_time
all_results['elapsed_seconds'] = round(elapsed, 1)
print(f"\nElapsed: {elapsed:.0f}s")

with open(OUTPUT_FILE, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"Results saved to {OUTPUT_FILE}")
