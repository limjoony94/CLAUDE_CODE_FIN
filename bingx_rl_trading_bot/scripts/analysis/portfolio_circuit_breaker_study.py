#!/usr/bin/env python3
"""Portfolio MTM Circuit Breaker Study.

Tests whether closing all positions when mark-to-market drawdown exceeds
a threshold (circuit breaker) improves risk-adjusted returns.

Design:
  - Modifies portfolio_npos to add circuit_breaker_pct and cooldown_bars params
  - When MTM drawdown from peak > circuit_breaker_pct: close ALL open positions
    at current bar close, then pause entries for cooldown_bars
  - Sweeps circuit_breaker_pct x cooldown_bars grid
  - Top-5 by PnL/MDD get 3-fold expanding window WF
  - Top-3 get random discrimination test (3 seeds)

Critical rules:
  - Production classify_candle import
  - LEVERAGE = 3, compound equity, timeout = DROP
  - Fee = FEE_PCT * LEVERAGE (BingX notional)
  - Expanding Window WF only
"""
import json
import sys
import os
import time
import copy
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from scripts.scanner.pattern_scanner import (
    calc_stats_compound,
    compute_atr_ratio, compute_ema_slope,
    load_and_classify, build_signal_index,
    find_neutral_window,
    _check_exit_npos,
    LEVERAGE, FEE_PCT, SLIPPAGE_BUFFER,
    DEFAULT_N_SLOTS, DEFAULT_DIRECTION_CAP,
    DEFAULT_REGIME_MULT,
    DEFAULT_AGG_RISK_COUNTER, DEFAULT_AGG_RISK_WITH,
    DEFAULT_MOMENTUM_LOOKBACK, DEFAULT_MOMENTUM_THRESHOLD,
    DEFAULT_MOMENTUM_COOLDOWN,
    DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI,
    TIMEOUT_BARS, DEFAULT_CASCADE_TIGHTEN_PCT,
    NPOS_EMA_PERIOD, NPOS_EMA_LOOKBACK,
)

DATA_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'btc_5m_270days_reclassified.csv')
PATTERNS_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'dynamic_patterns.json')
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'portfolio_circuit_breaker_study.json')

# Sweep parameters
CB_PCTS = [5, 6, 7, 8, 10, 12, 15, 20, 999]  # 999 = disabled (baseline)
COOLDOWN_BARS_LIST = [36, 72, 144, 288]  # 3h, 6h, 12h, 24h

MC_SEEDS = [42, 123, 7]
WF_FOLDS = 3


def portfolio_npos_cb(signal_tuples, opens, highs, lows, closes, n_bars,
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
                      circuit_breaker_pct=999,
                      cooldown_bars=72):
    """portfolio_npos with circuit breaker.

    When MTM drawdown exceeds circuit_breaker_pct:
      1. Close ALL open positions at current bar close
      2. Pause new entries for cooldown_bars bars

    Returns: (trades, stats) where stats includes cb_triggers count.
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
    total_blocked = {'momentum': 0, 'agg_risk': 0, 'dir_cap': 0, 'dup_pat': 0,
                     'max_pos': 0, 'cb_cooldown': 0}
    corr_events = []

    momentum_pause_until = {'LONG': -1, 'SHORT': -1}

    # Circuit breaker state
    cb_triggers = 0
    cb_trigger_losses = []
    cb_pause_until = -1  # bar until which entries are paused

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
                result['size_mult'] = sm
                pnl_portfolio = result['pnl_slot'] * (size_pct / 100) * sm
                result['pnl_portfolio'] = pnl_portfolio
                trades.append(result)
                closed_slots.append(pos['slot'])
                bar_pnl_sum += pnl_portfolio
                if result['reason'] == 'SL':
                    bar_sl_count += 1

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
                    cur_eff_sl = pos.get('eff_sl_override') or (pos['sl_pct'] * r)
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

        # === CIRCUIT BREAKER CHECK ===
        # After processing exits, check MTM equity including remaining open positions
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
                mtm_equity += unr * (size_pct / 100) * sm
            if mtm_equity > peak_equity:
                peak_equity = mtm_equity
            dd = (peak_equity - mtm_equity) / peak_equity * 100 if peak_equity > 0 else 0
            if dd > max_dd_mtm:
                max_dd_mtm = dd

            # Circuit breaker trigger
            if dd >= circuit_breaker_pct and circuit_breaker_pct < 999:
                cb_triggers += 1
                # Close ALL open positions at current bar close
                cb_loss_total = 0.0
                for pos in positions:
                    eb = pos['entry_bar']
                    if eb >= n_bars:
                        continue
                    entry_price = opens[eb]
                    if entry_price <= 0:
                        continue
                    exit_price = closes[bar]
                    if pos['direction'] == 'LONG':
                        pnl = (exit_price / entry_price - 1) * 100 * LEVERAGE
                    else:
                        pnl = (1 - exit_price / entry_price) * 100 * LEVERAGE
                    pnl -= fee
                    sm = pos.get('size_mult', 1.0)
                    pnl_portfolio = pnl * (size_pct / 100) * sm
                    trades.append({
                        'entry_bar': eb, 'exit_bar': bar, 'pnl_slot': pnl,
                        'reason': 'CIRCUIT_BREAK', 'pattern': pos['pattern'],
                        'direction': pos['direction'], 'size_mult': sm,
                        'pnl_portfolio': pnl_portfolio,
                    })
                    equity += pnl_portfolio
                    cb_loss_total += pnl_portfolio

                cb_trigger_losses.append(round(cb_loss_total, 4))
                positions = []
                cb_pause_until = bar + cooldown_bars

                # Update peak after forced closes
                if equity > peak_equity:
                    peak_equity = equity

        elif not positions:
            if equity > peak_equity:
                peak_equity = equity
            dd = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0
            if dd > max_dd_mtm:
                max_dd_mtm = dd

        # 2. Process entries
        while sig_idx < len(signals_sorted) and signals_sorted[sig_idx][0] == bar:
            sig_bar, pat, direction, tp_pct, sl_pct = signals_sorted[sig_idx]
            sig_idx += 1

            # Circuit breaker cooldown check
            if bar < cb_pause_until:
                total_blocked['cb_cooldown'] += 1
                continue

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
            })

        if len(positions) > max_sim_positions:
            max_sim_positions = len(positions)

        # MTM tracking for bars where no CB triggered (positions still open)
        # Only needed if CB didn't trigger this bar (CB block above already handled MTM)
        # The MTM check above covers both CB and tracking

    # Force-close remaining (OOS end)
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

    stats = {
        'max_corr_loss': round(max_corr_loss, 2),
        'max_sim_positions': max_sim_positions,
        'corr_events': len(corr_events),
        'blocked': total_blocked,
        'mdd_mtm': round(max_dd_mtm, 2),
        'cb_triggers': cb_triggers,
        'cb_trigger_losses': cb_trigger_losses,
    }
    return trades, stats


def build_signal_tuples(patterns, signal_index):
    """Build signal tuples from pattern_details + signal_index."""
    tuples = []
    for pat_key, pat_info in patterns.items():
        pat_name = pat_info['pattern']
        direction = pat_info['direction']
        tp = pat_info['tp']
        sl = pat_info['sl']
        if pat_name in signal_index:
            for sb in signal_index[pat_name]:
                tuples.append((sb, pat_key, direction, tp, sl))
    return tuples


def build_random_signal_tuples(signal_tuples, seed):
    """Shuffle directions of signal tuples for random discrimination test."""
    rng = np.random.RandomState(seed)
    shuffled = []
    for sb, pat, direction, tp, sl in signal_tuples:
        rand_dir = 'LONG' if rng.random() < 0.5 else 'SHORT'
        shuffled.append((sb, pat, rand_dir, tp, sl))
    return shuffled


def run_sim(signal_tuples, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, start_bar, end_bar,
            cb_pct, cd_bars):
    """Run single simulation and return summary dict."""
    trades, stats = portfolio_npos_cb(
        signal_tuples, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, start_bar, end_bar,
        circuit_breaker_pct=cb_pct,
        cooldown_bars=cd_bars,
    )
    cstats = calc_stats_compound(trades)
    return {
        'trades': cstats['trades'],
        'wr': cstats['wr'],
        'pnl': cstats['pnl'],
        'mdd_realized': cstats['mdd'],
        'mdd_mtm': stats['mdd_mtm'],
        'pnl_mdd': cstats['pnl_mdd'],
        'cb_triggers': stats['cb_triggers'],
        'cb_avg_loss': round(np.mean(stats['cb_trigger_losses']), 4) if stats['cb_trigger_losses'] else 0,
        'blocked': stats.get('blocked', {}),
    }


def main():
    t0 = time.time()
    print("=" * 70)
    print("  PORTFOLIO MTM CIRCUIT BREAKER STUDY")
    print("=" * 70)

    # Load data
    df = load_and_classify(DATA_FILE)
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    types = df['candle_type'].values
    n_bars = len(df)
    print(f"\nData: {n_bars} bars ({n_bars / 288:.0f} days)")

    atr_ratio = compute_atr_ratio(highs, lows, closes)
    ema_slope = compute_ema_slope(closes)

    signal_index = build_signal_index(types, n_bars)

    # Load patterns
    with open(PATTERNS_FILE) as f:
        pdata = json.load(f)
    patterns = pdata.get('pattern_details') or {}
    print(f"Patterns: {len(patterns)}")

    signal_tuples = build_signal_tuples(patterns, signal_index)
    print(f"Signal tuples: {len(signal_tuples)}")

    # Neutral window
    nw = pdata.get('neutral_window') or {}
    start_bar = nw.get('start_bar', 0)
    end_bar = nw.get('end_bar', n_bars)
    print(f"Neutral window: [{start_bar}, {end_bar}] ({(end_bar - start_bar) / 288:.1f} days)")

    # ================================================================
    # PHASE 1: IS Grid Sweep
    # ================================================================
    print(f"\n{'=' * 70}")
    print("  PHASE 1: IS Grid Sweep (circuit_breaker_pct x cooldown_bars)")
    print(f"{'=' * 70}")
    print(f"  CB thresholds: {CB_PCTS}")
    print(f"  Cooldown bars:  {COOLDOWN_BARS_LIST}")

    results = []
    for cb_pct in CB_PCTS:
        for cd_bars in COOLDOWN_BARS_LIST:
            label = f"CB={cb_pct}%/CD={cd_bars}b"
            if cb_pct >= 999:
                label = "BASELINE(disabled)"
                # Only run baseline once (cooldown doesn't matter)
                if cd_bars != COOLDOWN_BARS_LIST[0]:
                    continue

            res = run_sim(signal_tuples, opens, highs, lows, closes, n_bars,
                          atr_ratio, ema_slope, start_bar, end_bar,
                          cb_pct, cd_bars)
            res['cb_pct'] = cb_pct
            res['cooldown_bars'] = cd_bars
            res['label'] = label
            results.append(res)
            print(f"  {label:<30} Trades={res['trades']:>5}  WR={res['wr']:>5.1f}%  "
                  f"PnL={res['pnl']:>8.1f}%  MDD_MTM={res['mdd_mtm']:>5.2f}%  "
                  f"PnL/MDD={res['pnl_mdd']:>7.2f}  CB_triggers={res['cb_triggers']}")

    # Sort by PnL/MDD descending
    results_sorted = sorted(results, key=lambda x: x['pnl_mdd'], reverse=True)

    # Find baseline
    baseline = next((r for r in results if r['cb_pct'] >= 999), None)

    print(f"\n{'=' * 70}")
    print("  TOP 10 by PnL/MDD")
    print(f"{'=' * 70}")
    print(f"  {'Rank':<5} {'Config':<30} {'Trades':>6} {'WR':>6} {'PnL':>9} "
          f"{'MDD_MTM':>8} {'PnL/MDD':>8} {'CB_trig':>8}")
    for i, r in enumerate(results_sorted[:10]):
        marker = " <-- BASELINE" if r['cb_pct'] >= 999 else ""
        print(f"  {i + 1:<5} {r['label']:<30} {r['trades']:>6} {r['wr']:>5.1f}% "
              f"{r['pnl']:>8.1f}% {r['mdd_mtm']:>7.2f}% {r['pnl_mdd']:>7.2f}  "
              f"{r['cb_triggers']:>7}{marker}")

    # ================================================================
    # PHASE 2: WF for top 5 (excluding baseline if it's in top 5)
    # ================================================================
    print(f"\n{'=' * 70}")
    print(f"  PHASE 2: Expanding Window WF (top 5 by PnL/MDD)")
    print(f"{'=' * 70}")

    # Top 5 configs that are NOT baseline
    top5 = [r for r in results_sorted if r['cb_pct'] < 999][:5]
    # Always include baseline for comparison
    wf_configs = top5 + ([baseline] if baseline else [])

    wf_results = {}
    fold_size = (end_bar - start_bar) // (WF_FOLDS + 1)

    for cfg in wf_configs:
        cb_pct = cfg['cb_pct']
        cd_bars = cfg['cooldown_bars']
        label = cfg['label']
        print(f"\n  --- WF: {label} ---")

        fold_results = []
        all_pass = True
        for fold in range(WF_FOLDS):
            is_end = start_bar + fold_size * (fold + 2)
            oos_start = is_end
            oos_end = min(is_end + fold_size, end_bar)
            if oos_end <= oos_start:
                continue

            # IS: scan patterns in IS period (reuse existing patterns + TP/SL)
            # OOS: run simulation on OOS with same patterns
            oos_atr = compute_atr_ratio(highs[:oos_end], lows[:oos_end], closes[:oos_end])
            oos_ema = compute_ema_slope(closes[:oos_end])

            oos_sigs = [(s, p, d, tp, sl) for s, p, d, tp, sl in signal_tuples
                        if oos_start <= s < oos_end]
            if not oos_sigs:
                fold_results.append({'fold': fold + 1, 'oos_trades': 0, 'oos_pnl': 0, 'pass': False})
                all_pass = False
                continue

            oos_trades, oos_stats = portfolio_npos_cb(
                signal_tuples, opens, highs, lows, closes, n_bars,
                oos_atr, oos_ema, oos_start, oos_end,
                circuit_breaker_pct=cb_pct,
                cooldown_bars=cd_bars,
            )
            oos_cstats = calc_stats_compound(oos_trades)
            passed = oos_cstats['pnl'] > 0
            if not passed:
                all_pass = False
            fold_results.append({
                'fold': fold + 1,
                'is_bars': is_end - start_bar,
                'oos_bars': oos_end - oos_start,
                'oos_trades': oos_cstats['trades'],
                'oos_wr': oos_cstats['wr'],
                'oos_pnl': oos_cstats['pnl'],
                'oos_mdd': oos_cstats['mdd'],
                'oos_mdd_mtm': oos_stats['mdd_mtm'],
                'oos_pnl_mdd': oos_cstats['pnl_mdd'],
                'cb_triggers': oos_stats['cb_triggers'],
                'pass': passed,
            })
            print(f"    Fold {fold + 1}: OOS trades={oos_cstats['trades']}, "
                  f"WR={oos_cstats['wr']}%, PnL={oos_cstats['pnl']:.1f}%, "
                  f"MDD_MTM={oos_stats['mdd_mtm']:.2f}%, CB_trig={oos_stats['cb_triggers']}  "
                  f"{'PASS' if passed else 'FAIL'}")

        n_pass = sum(1 for f in fold_results if f.get('pass'))
        print(f"    Result: {n_pass}/{WF_FOLDS} PASS {'-- ALL PASS' if all_pass else ''}")
        wf_results[label] = {
            'cb_pct': cb_pct,
            'cooldown_bars': cd_bars,
            'folds': fold_results,
            'n_pass': n_pass,
            'all_pass': all_pass,
        }

    # ================================================================
    # PHASE 3: Random discrimination test (top 3)
    # ================================================================
    print(f"\n{'=' * 70}")
    print(f"  PHASE 3: Random Discrimination Test (top 3)")
    print(f"{'=' * 70}")

    top3 = [r for r in results_sorted if r['cb_pct'] < 999][:3]
    disc_results = {}

    for cfg in top3:
        cb_pct = cfg['cb_pct']
        cd_bars = cfg['cooldown_bars']
        label = cfg['label']
        print(f"\n  --- Discrimination: {label} ---")

        rand_wf_passes = []
        for seed in MC_SEEDS:
            rand_sigs = build_random_signal_tuples(signal_tuples, seed)

            rand_fold_passes = 0
            for fold in range(WF_FOLDS):
                is_end = start_bar + fold_size * (fold + 2)
                oos_start = is_end
                oos_end = min(is_end + fold_size, end_bar)
                if oos_end <= oos_start:
                    continue

                oos_atr = compute_atr_ratio(highs[:oos_end], lows[:oos_end], closes[:oos_end])
                oos_ema = compute_ema_slope(closes[:oos_end])

                oos_trades, _ = portfolio_npos_cb(
                    rand_sigs, opens, highs, lows, closes, n_bars,
                    oos_atr, oos_ema, oos_start, oos_end,
                    circuit_breaker_pct=cb_pct,
                    cooldown_bars=cd_bars,
                )
                oos_cstats = calc_stats_compound(oos_trades)
                if oos_cstats['pnl'] > 0:
                    rand_fold_passes += 1

            rand_wf_passes.append(rand_fold_passes)
            print(f"    Seed {seed}: random WF {rand_fold_passes}/{WF_FOLDS} PASS")

        # Discriminating if random fails (< 3/3 PASS for all seeds)
        is_discriminating = all(p < WF_FOLDS for p in rand_wf_passes)
        max_rand_pass = max(rand_wf_passes)
        print(f"    DISCRIMINATING: {is_discriminating} (max random passes: {max_rand_pass}/{WF_FOLDS})")

        disc_results[label] = {
            'cb_pct': cb_pct,
            'cooldown_bars': cd_bars,
            'random_wf_passes': rand_wf_passes,
            'seeds': MC_SEEDS,
            'is_discriminating': is_discriminating,
            'max_rand_pass': max_rand_pass,
        }

    # ================================================================
    # VERDICT
    # ================================================================
    print(f"\n{'=' * 70}")
    print("  VERDICT")
    print(f"{'=' * 70}")

    if baseline:
        print(f"\n  BASELINE (no CB): Trades={baseline['trades']}, WR={baseline['wr']}%, "
              f"PnL={baseline['pnl']:.1f}%, MDD_MTM={baseline['mdd_mtm']:.2f}%, "
              f"PnL/MDD={baseline['pnl_mdd']:.2f}")

    # Check if any config beats baseline AND passes WF AND is discriminating
    go_configs = []
    for cfg in top3:
        label = cfg['label']
        wf = wf_results.get(label, {})
        disc = disc_results.get(label, {})
        beats_baseline = baseline and cfg['pnl_mdd'] > baseline['pnl_mdd']
        wf_pass = wf.get('all_pass', False)
        discriminating = disc.get('is_discriminating', False)
        if beats_baseline and wf_pass and discriminating:
            go_configs.append(cfg)
            print(f"\n  GO: {label}")
            print(f"    PnL/MDD: {cfg['pnl_mdd']:.2f} (vs baseline {baseline['pnl_mdd']:.2f})")
            print(f"    WF: {wf['n_pass']}/{WF_FOLDS} PASS")
            print(f"    Discrimination: YES")

    if not go_configs:
        print("\n  VERDICT: KEEP_CURRENT (no CB config beats baseline + WF + discrimination)")
        verdict = 'KEEP_CURRENT'
    else:
        best_go = max(go_configs, key=lambda x: x['pnl_mdd'])
        print(f"\n  VERDICT: GO with {best_go['label']}")
        verdict = f"GO_{best_go['label']}"

    elapsed = time.time() - t0
    print(f"\n  Elapsed: {elapsed:.1f}s")

    # Save results
    output = {
        'study': 'portfolio_circuit_breaker',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'data_bars': n_bars,
        'patterns': len(patterns),
        'neutral_window': [start_bar, end_bar],
        'sweep': {
            'cb_pcts': CB_PCTS,
            'cooldown_bars': COOLDOWN_BARS_LIST,
        },
        'is_results': results_sorted,
        'baseline': baseline,
        'wf_results': wf_results,
        'discrimination_results': disc_results,
        'verdict': verdict,
        'elapsed_s': round(elapsed, 1),
    }
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
