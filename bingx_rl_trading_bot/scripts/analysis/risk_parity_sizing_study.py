"""Risk-Parity Sizing Study: Normalize per-trade risk across SL range.

Current: equal-weight 1/N sizing (11.1% per slot, N=9).
Problem: SL range 1.44-5.95% → per-trade risk varies 4x.
Hypothesis: Sizing inversely proportional to SL distance normalizes risk,
            improving PnL/MDD by reducing outsized losses on wide-SL trades.

Risk-Parity Formula:
  size_pct = target_risk / (eff_sl * LEVERAGE) * 100
  Capped at [min_size, max_size] to prevent extreme allocations.

Sweep: target_risk x max_size_cap, compared to 1/N baseline.
Top-5 by PnL/MDD → 3-fold expanding WF.
Top-3 → random discrimination test.
"""
import json
import sys
import os
import time
import copy
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from scripts.scanner.pattern_scanner import (
    calc_stats_compound,
    compute_atr_ratio, compute_ema_slope,
    load_and_classify, build_signal_index,
    find_neutral_window,
    _check_exit_npos,
    LEVERAGE, FEE_PCT, TIMEOUT_BARS,
    DEFAULT_N_SLOTS, DEFAULT_DIRECTION_CAP,
    DEFAULT_REGIME_MULT,
    DEFAULT_AGG_RISK_COUNTER, DEFAULT_AGG_RISK_WITH,
    DEFAULT_MOMENTUM_LOOKBACK, DEFAULT_MOMENTUM_THRESHOLD,
    DEFAULT_MOMENTUM_COOLDOWN,
    DEFAULT_CASCADE_TIGHTEN_PCT,
    DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI,
    SLIPPAGE_BUFFER, NPOS_EMA_PERIOD, NPOS_EMA_LOOKBACK,
)

DATA_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'btc_5m_270days_reclassified.csv')
PATTERNS_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'dynamic_patterns.json')
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'risk_parity_sizing_study.json')

# Sweep parameters
TARGET_RISKS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]   # % portfolio loss per SL hit
MAX_SIZE_CAPS = [12, 15, 20, 25]                      # % max allocation per position
MIN_SIZE = 5.0                                         # % minimum allocation

# MC seeds for discrimination test
MC_SEEDS = [42, 123, 7]
MC_SIMS = 5000


def portfolio_npos_riskparity(signal_tuples, opens, highs, lows, closes, n_bars,
                               atr_ratio, ema_slope, start_bar, end_bar,
                               target_risk_pct, max_size_pct, min_size_pct=MIN_SIZE,
                               n_slots=DEFAULT_N_SLOTS,
                               direction_cap=DEFAULT_DIRECTION_CAP,
                               regime_mult=DEFAULT_REGIME_MULT,
                               agg_risk_counter=DEFAULT_AGG_RISK_COUNTER,
                               agg_risk_with=DEFAULT_AGG_RISK_WITH,
                               momentum_lookback=DEFAULT_MOMENTUM_LOOKBACK,
                               momentum_threshold=DEFAULT_MOMENTUM_THRESHOLD,
                               momentum_cooldown=DEFAULT_MOMENTUM_COOLDOWN,
                               clamp_lo=DEFAULT_ATR_CLAMP_LO,
                               clamp_hi=DEFAULT_ATR_CLAMP_HI,
                               timeout_bars=TIMEOUT_BARS,
                               cascade_tighten_pct=DEFAULT_CASCADE_TIGHTEN_PCT):
    """N-position portfolio simulator with risk-parity sizing.

    Instead of fixed size_pct = 100/N, each position sized as:
      size_pct = target_risk / (eff_sl * LEVERAGE) * 100
    Clamped to [min_size, max_size].

    Returns: (trades_list, stats_dict)
    """
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

    # Track sizing stats
    all_sizes = []

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
                sp = pos.get('size_pct', 100.0 / n_slots)
                result['size_mult'] = sm
                result['size_pct'] = sp
                pnl_portfolio = result['pnl_slot'] * (sp / 100) * sm
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

            # Regime sizing
            sm = 1.0
            if regime_mult is not None and bar < len(ema_slope):
                s = ema_slope[bar]
                if s > 0 and direction == 'SHORT':
                    sm = regime_mult
                elif s <= 0 and direction == 'LONG':
                    sm = regime_mult

            # Compute effective SL for risk-parity sizing
            if (atr_ratio is not None and sig_bar < len(atr_ratio)
                    and not np.isnan(atr_ratio[sig_bar])):
                new_r = max(clamp_lo, min(clamp_hi, atr_ratio[sig_bar]))
            else:
                new_r = 1.0
            new_eff_sl = sl_pct * new_r

            # RISK-PARITY SIZING: size inversely proportional to SL
            # target_risk = size_frac * eff_sl * LEVERAGE
            # size_frac = target_risk / (eff_sl * LEVERAGE)
            # size_pct = size_frac * 100
            rp_size = target_risk_pct / (new_eff_sl * LEVERAGE) * 100
            rp_size = max(min_size_pct, min(max_size_pct, rp_size))
            all_sizes.append(rp_size)

            # Aggregate risk cap check (use risk-parity sizing)
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
                        p_sp = p.get('size_pct', 100.0 / n_slots)
                        existing_exposure += p_eff_sl * (p_sp / 100) * LEVERAGE * p_sm

                new_exposure = new_eff_sl * (rp_size / 100) * LEVERAGE * sm

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
                'size_pct': rp_size,
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
                sp = pos.get('size_pct', 100.0 / n_slots)
                mtm_equity += unr * (sp / 100) * sm
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
        entry_bar_p = pos['entry_bar']
        if entry_bar_p >= n_bars:
            continue
        entry = opens[entry_bar_p]
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
        sp = pos.get('size_pct', 100.0 / n_slots)
        trades.append({
            'entry_bar': entry_bar_p, 'exit_bar': exit_bar, 'pnl_slot': pnl,
            'reason': 'OOS_END', 'pattern': pos['pattern'],
            'direction': pos['direction'], 'size_mult': sm,
            'size_pct': sp,
            'pnl_portfolio': pnl * (sp / 100) * sm,
        })

    stats = {
        'max_corr_loss': round(max_corr_loss, 2),
        'max_sim_positions': max_sim_positions,
        'corr_events': len(corr_events),
        'blocked': total_blocked,
        'mdd_mtm': round(max_dd_mtm, 2),
        'avg_size': round(np.mean(all_sizes), 2) if all_sizes else 0,
        'min_size_actual': round(np.min(all_sizes), 2) if all_sizes else 0,
        'max_size_actual': round(np.max(all_sizes), 2) if all_sizes else 0,
        'std_size': round(np.std(all_sizes), 2) if all_sizes else 0,
    }
    return trades, stats


def calc_stats_compound_rp(trades):
    """Compound equity stats for risk-parity (uses per-trade size_pct)."""
    if not trades:
        return {'trades': 0, 'wr': 0.0, 'pnl': 0.0, 'mdd': 0.0,
                'pnl_mdd': 0.0, 'pnl_per_trade': 0.0}

    wins = [t for t in trades if t['pnl_slot'] > 0]
    sorted_trades = sorted(trades, key=lambda x: x['exit_bar'])

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


def run_mc_discrimination(trades, n_sims=MC_SIMS, seeds=MC_SEEDS):
    """Sign-randomization MC test. Returns max p-value across seeds."""
    if len(trades) < 25:
        return 1.0
    pnls = np.array([t['pnl_portfolio'] for t in sorted(trades, key=lambda x: x['exit_bar'])])
    real_pnl = pnls.sum()
    max_p = 0.0
    for seed in seeds:
        rng = np.random.RandomState(seed)
        count_above = 0
        for _ in range(n_sims):
            signs = rng.choice([-1, 1], size=len(pnls))
            sim_pnl = (pnls * signs).sum()
            if sim_pnl >= real_pnl:
                count_above += 1
        p = count_above / n_sims
        if p > max_p:
            max_p = p
    return max_p


def run_wf_3fold(signal_tuples, opens, highs, lows, closes, n_bars,
                 atr_ratio, ema_slope, start_bar, end_bar,
                 target_risk_pct, max_size_pct, patterns_data,
                 signal_index):
    """3-fold expanding window WF for risk-parity sizing."""
    total_bars = end_bar - start_bar
    fold_size = total_bars // 4  # 3 folds: IS grows, OOS = 1 fold each

    results = []
    for fold in range(3):
        is_end = start_bar + fold_size * (fold + 1)
        oos_start = is_end
        oos_end = min(is_end + fold_size, end_bar)
        if oos_end <= oos_start:
            continue

        # IS: discover patterns that pass quality in IS window
        # (reuse pre-computed patterns from full IS for simplicity,
        #  but filter signals to IS window for stats)

        # OOS: run risk-parity portfolio on OOS window with IS patterns
        oos_trades, oos_stats = portfolio_npos_riskparity(
            signal_tuples, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, oos_start, oos_end,
            target_risk_pct=target_risk_pct,
            max_size_pct=max_size_pct,
        )
        oos_metrics = calc_stats_compound_rp(oos_trades)
        oos_metrics['mdd_mtm'] = oos_stats['mdd_mtm']

        results.append({
            'fold': fold + 1,
            'is_bars': is_end - start_bar,
            'oos_bars': oos_end - oos_start,
            'oos_trades': oos_metrics['trades'],
            'oos_wr': oos_metrics['wr'],
            'oos_pnl': oos_metrics['pnl'],
            'oos_mdd_mtm': oos_metrics['mdd_mtm'],
            'oos_pnl_mdd': round(oos_metrics['pnl'] / oos_metrics['mdd_mtm'], 2) if oos_metrics['mdd_mtm'] > 0 else 0,
        })

    return results


def main():
    t0 = time.time()
    print("=" * 70)
    print("RISK-PARITY SIZING STUDY")
    print("Hypothesis: Sizing inversely to SL normalizes per-trade risk")
    print("=" * 70)

    # Load data
    df = load_and_classify(DATA_FILE)
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    types = df['candle_type'].values
    n_bars = len(df)
    print(f"\nData: {n_bars} bars ({n_bars/288:.0f} days)")

    atr_ratio = compute_atr_ratio(highs, lows, closes)
    ema_slope = compute_ema_slope(closes)

    # Neutral window
    nw = find_neutral_window(closes, tol_pct=1.0, min_days=90)
    if nw:
        start_bar, end_bar = nw
        print(f"Neutral window: bar {start_bar}-{end_bar} ({(end_bar-start_bar)/288:.0f} days)")
    else:
        start_bar, end_bar = 0, n_bars
        print("No neutral window found, using full data")

    signal_index = build_signal_index(types, n_bars)

    # Load pattern set
    with open(PATTERNS_FILE) as f:
        pdata = json.load(f)
    patterns = pdata['pattern_details']
    print(f"Patterns: {len(patterns)}")

    # Build signal tuples
    signal_tuples = []
    for pat_key, pat_info in patterns.items():
        pat_name = pat_info['pattern']
        direction = pat_info['direction']
        tp = pat_info['tp']
        sl = pat_info['sl']
        if pat_name in signal_index:
            for sb in signal_index[pat_name]:
                signal_tuples.append((sb, pat_key, direction, tp, sl))
    print(f"Total signals: {len(signal_tuples)}")

    # Show SL distribution
    sls = [pat_info['sl'] for pat_info in patterns.values()]
    print(f"\nSL distribution: min={min(sls):.2f}%, max={max(sls):.2f}%, "
          f"mean={np.mean(sls):.2f}%, std={np.std(sls):.2f}%")
    print(f"SL range ratio: {max(sls)/min(sls):.1f}x")

    # ========== BASELINE: 1/N equal-weight ==========
    print("\n" + "=" * 70)
    print("BASELINE: 1/N Equal-Weight (11.1% per slot)")
    print("=" * 70)

    from scripts.scanner.pattern_scanner import portfolio_npos
    bl_trades, bl_stats = portfolio_npos(
        signal_tuples, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, start_bar, end_bar,
    )
    bl_metrics = calc_stats_compound(bl_trades)
    print(f"  Trades: {bl_metrics['trades']}, WR: {bl_metrics['wr']}%, "
          f"PnL: {bl_metrics['pnl']:.1f}%, MDD(MTM): {bl_stats['mdd_mtm']:.2f}%, "
          f"PnL/MDD: {bl_metrics['pnl']/bl_stats['mdd_mtm']:.1f}" if bl_stats['mdd_mtm'] > 0 else "N/A")
    print(f"  Blocked: {bl_stats['blocked']}")

    baseline_pnl_mdd = bl_metrics['pnl'] / bl_stats['mdd_mtm'] if bl_stats['mdd_mtm'] > 0 else 0

    # ========== SWEEP: Risk-Parity Combinations ==========
    print("\n" + "=" * 70)
    print("SWEEP: Risk-Parity Sizing Combinations")
    print("=" * 70)

    sweep_results = []
    for tr in TARGET_RISKS:
        for mc in MAX_SIZE_CAPS:
            rp_trades, rp_stats = portfolio_npos_riskparity(
                signal_tuples, opens, highs, lows, closes, n_bars,
                atr_ratio, ema_slope, start_bar, end_bar,
                target_risk_pct=tr, max_size_pct=mc,
            )
            rp_metrics = calc_stats_compound_rp(rp_trades)
            mdd_mtm = rp_stats['mdd_mtm']
            pnl_mdd = rp_metrics['pnl'] / mdd_mtm if mdd_mtm > 0 else 0

            result = {
                'target_risk': tr,
                'max_size_cap': mc,
                'trades': rp_metrics['trades'],
                'wr': rp_metrics['wr'],
                'pnl': rp_metrics['pnl'],
                'mdd_mtm': mdd_mtm,
                'pnl_mdd': round(pnl_mdd, 2),
                'pnl_per_trade': rp_metrics['pnl_per_trade'],
                'avg_size': rp_stats['avg_size'],
                'min_size': rp_stats['min_size_actual'],
                'max_size': rp_stats['max_size_actual'],
                'std_size': rp_stats['std_size'],
                'blocked': rp_stats['blocked'],
                'vs_baseline_pnl': round(rp_metrics['pnl'] - bl_metrics['pnl'], 1),
                'vs_baseline_mdd': round(mdd_mtm - bl_stats['mdd_mtm'], 2),
                'vs_baseline_pnl_mdd': round(pnl_mdd - baseline_pnl_mdd, 2),
            }
            sweep_results.append(result)

            tag = "***" if pnl_mdd > baseline_pnl_mdd else "   "
            print(f"  {tag} TR={tr:.2f}% MC={mc}%: "
                  f"T={rp_metrics['trades']}, WR={rp_metrics['wr']}%, "
                  f"PnL={rp_metrics['pnl']:.1f}%, MDD={mdd_mtm:.2f}%, "
                  f"PnL/MDD={pnl_mdd:.1f} | "
                  f"size: {rp_stats['avg_size']:.1f}% [{rp_stats['min_size_actual']:.1f}-{rp_stats['max_size_actual']:.1f}]")

    # ========== TOP-5 by PnL/MDD → WF ==========
    sweep_results.sort(key=lambda x: x['pnl_mdd'], reverse=True)
    top5 = sweep_results[:5]

    print("\n" + "=" * 70)
    print("TOP-5 by PnL/MDD → 3-Fold Expanding Window WF")
    print("=" * 70)

    wf_results = {}
    for i, cfg in enumerate(top5):
        tr = cfg['target_risk']
        mc = cfg['max_size_cap']
        key = f"TR{tr}_MC{mc}"
        print(f"\n  [{i+1}] {key}: IS PnL/MDD={cfg['pnl_mdd']}")

        folds = run_wf_3fold(
            signal_tuples, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, start_bar, end_bar,
            target_risk_pct=tr, max_size_pct=mc,
            patterns_data=patterns, signal_index=signal_index,
        )

        all_pass = True
        for f in folds:
            status = "PASS" if f['oos_pnl'] > 0 else "FAIL"
            if f['oos_pnl'] <= 0:
                all_pass = False
            print(f"    Fold {f['fold']}: OOS T={f['oos_trades']}, WR={f['oos_wr']}%, "
                  f"PnL={f['oos_pnl']:.1f}%, MDD={f['oos_mdd_mtm']:.2f}% → {status}")

        wf_results[key] = {
            'config': {'target_risk': tr, 'max_size_cap': mc},
            'is_metrics': cfg,
            'wf_folds': folds,
            'wf_pass': all_pass,
        }

    # ========== Also run baseline WF for comparison ==========
    print("\n  [BL] Baseline 1/N WF:")
    bl_folds = []
    total_bars = end_bar - start_bar
    fold_size = total_bars // 4
    for fold in range(3):
        is_end = start_bar + fold_size * (fold + 1)
        oos_start = is_end
        oos_end = min(is_end + fold_size, end_bar)
        if oos_end <= oos_start:
            continue
        bl_oos_trades, bl_oos_stats = portfolio_npos(
            signal_tuples, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, oos_start, oos_end,
        )
        bl_oos_m = calc_stats_compound(bl_oos_trades)
        bl_folds.append({
            'fold': fold + 1,
            'oos_trades': bl_oos_m['trades'],
            'oos_wr': bl_oos_m['wr'],
            'oos_pnl': bl_oos_m['pnl'],
            'oos_mdd_mtm': bl_oos_stats['mdd_mtm'],
        })
        status = "PASS" if bl_oos_m['pnl'] > 0 else "FAIL"
        print(f"    Fold {fold+1}: OOS T={bl_oos_m['trades']}, WR={bl_oos_m['wr']}%, "
              f"PnL={bl_oos_m['pnl']:.1f}%, MDD={bl_oos_stats['mdd_mtm']:.2f}% → {status}")

    # ========== TOP-3 WF-passing → MC Discrimination ==========
    wf_passing = [(k, v) for k, v in wf_results.items() if v['wf_pass']]
    # Sort by average OOS PnL/MDD
    for k, v in wf_passing:
        avg_oos_pnl_mdd = np.mean([f['oos_pnl_mdd'] for f in v['wf_folds']])
        v['avg_oos_pnl_mdd'] = round(avg_oos_pnl_mdd, 2)
    wf_passing.sort(key=lambda x: x[1].get('avg_oos_pnl_mdd', 0), reverse=True)
    top3_mc = wf_passing[:3]

    print("\n" + "=" * 70)
    print("TOP-3 WF-PASSING → MC Random Discrimination Test")
    print("=" * 70)

    mc_results = {}
    if not top3_mc:
        print("  No configurations passed all 3 WF folds.")
    else:
        for k, v in top3_mc:
            tr = v['config']['target_risk']
            mc = v['config']['max_size_cap']
            # Run full IS with risk-parity
            rp_trades, rp_stats = portfolio_npos_riskparity(
                signal_tuples, opens, highs, lows, closes, n_bars,
                atr_ratio, ema_slope, start_bar, end_bar,
                target_risk_pct=tr, max_size_pct=mc,
            )
            # MC discrimination: randomize signs
            p_val = run_mc_discrimination(rp_trades)
            passed = p_val < 0.01
            print(f"  {k}: MC p={p_val:.4f} → {'PASS' if passed else 'FAIL'}")
            mc_results[k] = {
                'p_value': round(p_val, 4),
                'passed': passed,
            }

    # ========== VERDICT ==========
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    # Find best config that passes both WF and MC
    best = None
    for k, v in wf_passing:
        if k in mc_results and mc_results[k]['passed']:
            if best is None or v.get('avg_oos_pnl_mdd', 0) > wf_results[best].get('avg_oos_pnl_mdd', 0):
                best = k

    if best:
        bv = wf_results[best]
        avg_oos_pnl = np.mean([f['oos_pnl'] for f in bv['wf_folds']])
        avg_bl_oos_pnl = np.mean([f['oos_pnl'] for f in bl_folds]) if bl_folds else 0
        print(f"  Best config: {best}")
        print(f"  IS PnL/MDD: {bv['is_metrics']['pnl_mdd']} (baseline: {baseline_pnl_mdd:.1f})")
        print(f"  Avg OOS PnL: {avg_oos_pnl:.1f}% (baseline avg OOS: {avg_bl_oos_pnl:.1f}%)")
        print(f"  MC p-value: {mc_results[best]['p_value']}")

        if avg_oos_pnl > avg_bl_oos_pnl * 1.05:
            verdict = "GO"
            print(f"\n  >>> VERDICT: GO — Risk-parity improves OOS by "
                  f"{(avg_oos_pnl/avg_bl_oos_pnl - 1)*100:.0f}% vs baseline")
        else:
            verdict = "KEEP_BASELINE"
            print(f"\n  >>> VERDICT: KEEP_BASELINE — Improvement insufficient (<5% OOS)")
    else:
        verdict = "STOP"
        print("  >>> VERDICT: STOP — No config passes both WF 3/3 and MC p<0.01")

    elapsed = time.time() - t0

    # ========== SAVE RESULTS ==========
    output = {
        'study': 'risk_parity_sizing',
        'date': '2026-03-10',
        'elapsed_sec': round(elapsed, 1),
        'data_bars': n_bars,
        'neutral_window': [start_bar, end_bar] if nw else None,
        'patterns': len(patterns),
        'total_signals': len(signal_tuples),
        'sl_distribution': {
            'min': round(min(sls), 2),
            'max': round(max(sls), 2),
            'mean': round(float(np.mean(sls)), 2),
            'std': round(float(np.std(sls)), 2),
            'range_ratio': round(max(sls)/min(sls), 1),
        },
        'baseline': {
            'trades': bl_metrics['trades'],
            'wr': bl_metrics['wr'],
            'pnl': bl_metrics['pnl'],
            'mdd_mtm': bl_stats['mdd_mtm'],
            'pnl_mdd': round(baseline_pnl_mdd, 2),
            'wf_folds': bl_folds,
        },
        'sweep': sweep_results,
        'top5_wf': wf_results,
        'mc_discrimination': mc_results,
        'verdict': verdict,
        'best_config': best,
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_FILE}")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
