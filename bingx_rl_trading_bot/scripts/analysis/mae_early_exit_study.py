#!/usr/bin/env python3
"""MAE-Based Early Exit Study.

Research question:
  방향이 틀린 포지션을 24h timeout 전에 MAE(adverse excursion) 기반으로
  조기 청산하면 성과가 개선되는가?

Previous finding (timeout_cause_decomposition.py):
  - MAE r=0.813 > Time r=-0.352 — 방향이 주 원인
  - 24h에 손실 중인 포지션의 88%가 TP 미달
  - "deep loss(< -3%) at 24h" → eventual TP 12%

Approach:
  Phase 1: MAE profile — 어느 시점/깊이에서 포지션이 회복 불가능해지는가
  Phase 2: MAE exit sweep — (check_bar, threshold) 조합 테스트
  Phase 3: Best variant WF 3-fold validation
  Phase 4: Random discrimination (10 seeds)

Implementation:
  portfolio_npos를 복제하되, bar-by-bar unrealized PnL 체크 로직 추가.
  체크 시점에 unrealized < threshold → 시장가 청산 (MAE_EXIT).

Protocol: Entry next-bar open, intrabar exit, Fee 0.10%, Slippage 0.02%, Lev 3x, compound.
"""
import os, sys, json, time, logging, copy
import numpy as np, pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'scripts', 'scanner'))

from pattern_scanner import (
    load_and_classify, find_neutral_window, build_signal_index,
    calc_stats_compound, scan_universe_range,
    compute_atr_ratio, compute_ema_slope,
    _check_exit_npos,
    FEE_PCT, LEVERAGE, MAX_BARS, SLIPPAGE_BUFFER,
    DEFAULT_N_SLOTS, DEFAULT_DIRECTION_CAP,
    DEFAULT_AGG_RISK_COUNTER, DEFAULT_AGG_RISK_WITH,
    DEFAULT_MOMENTUM_LOOKBACK, DEFAULT_MOMENTUM_THRESHOLD,
    DEFAULT_MOMENTUM_COOLDOWN, DEFAULT_CASCADE_TIGHTEN_PCT,
    DEFAULT_ATR_PERIOD, DEFAULT_ATR_WINDOW,
    DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI,
    DEFAULT_REGIME_MULT, TIMEOUT_BARS,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
PATTERNS_FILE = os.path.join(_PROJECT_ROOT, 'results', 'dynamic_patterns.json')
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'mae_early_exit_study.json')


def load_patterns():
    with open(PATTERNS_FILE) as f:
        data = json.load(f)
    details = data.get('pattern_details') or {}
    result = {}
    for k, v in details.items():
        result[k] = {
            'pattern': v['pattern'], 'direction': v['direction'],
            'tp_pct': v['tp'], 'sl_pct': v['sl'],
        }
    return result


def portfolio_npos_mae(signal_tuples, opens, highs, lows, closes, n_bars,
                       atr_ratio, ema_slope, start_bar, end_bar,
                       mae_check_bar=None, mae_threshold=None,
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
    """portfolio_npos with MAE-based early exit.

    Added parameters:
      mae_check_bar: after how many bars to check unrealized PnL (e.g., 72 = 6h)
      mae_threshold: if unrealized PnL (lev %) < this, close at market (e.g., -3.0)
                     None = disabled (baseline behavior)

    Continuous check: once position age >= mae_check_bar, check every bar.
    """
    size_pct = 100.0 / n_slots
    fee = FEE_PCT * LEVERAGE

    positions = []
    trades = []
    equity = 100.0
    peak_equity = 100.0

    max_corr_loss = 0.0
    max_sim_positions = 0
    total_blocked = {'momentum': 0, 'agg_risk': 0, 'dir_cap': 0, 'dup_pat': 0, 'max_pos': 0}
    mae_exits = 0

    momentum_pause_until = {'LONG': -1, 'SHORT': -1}

    signals_in_range = [(s, p, d, tp, sl) for s, p, d, tp, sl in signal_tuples
                        if start_bar <= s < end_bar]
    signals_sorted = sorted(signals_in_range, key=lambda x: x[0])
    sig_idx = 0

    for bar in range(start_bar, end_bar):
        # 1. Check standard exits (TP/SL/timeout)
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

        # 1b. MAE-based early exit check
        if mae_check_bar is not None and mae_threshold is not None:
            for pos in positions:
                if pos['slot'] in closed_slots:
                    continue
                hold = bar - pos['entry_bar']
                if hold < mae_check_bar:
                    continue
                # Compute unrealized PnL at current bar's close
                entry_price = opens[pos['entry_bar']]
                if entry_price <= 0:
                    continue
                current_price = closes[bar]
                if pos['direction'] == 'LONG':
                    unr_pnl = (current_price / entry_price - 1) * 100 * LEVERAGE
                else:
                    unr_pnl = (1 - current_price / entry_price) * 100 * LEVERAGE
                unr_pnl -= fee

                if unr_pnl < mae_threshold:
                    # Market close at current bar's close
                    sm = pos.get('size_mult', 1.0)
                    pnl_portfolio = unr_pnl * (size_pct / 100) * sm
                    trades.append({
                        'entry_bar': pos['entry_bar'], 'exit_bar': bar,
                        'pnl_slot': unr_pnl, 'reason': 'MAE_EXIT',
                        'pattern': pos['pattern'], 'direction': pos['direction'],
                        'size_mult': sm, 'pnl_portfolio': pnl_portfolio,
                    })
                    closed_slots.append(pos['slot'])
                    bar_pnl_sum += pnl_portfolio
                    mae_exits += 1

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

    stats = {
        'max_corr_loss': round(max_corr_loss, 2),
        'max_sim_positions': max_sim_positions,
        'mae_exits': mae_exits,
        'blocked': total_blocked,
    }
    return trades, stats


def build_signal_tuples(patterns, signal_index, bar_start, bar_end):
    tuples = []
    for pk, info in patterns.items():
        pn = info.get('pattern') or pk.rsplit('_', 1)[0]
        if pn in signal_index:
            for sb in signal_index[pn]:
                if bar_start <= sb < bar_end:
                    tuples.append((sb, pk, info['direction'],
                                   info['tp_pct'], info['sl_pct']))
    return sorted(tuples, key=lambda x: x[0])


def run_variant(label, signal_tuples, opens, highs, lows, closes, n_bars,
                atr_ratio, ema_slope, ns, ne,
                mae_check_bar=None, mae_threshold=None):
    """Run a single MAE variant and return stats."""
    trades, meta = portfolio_npos_mae(
        signal_tuples, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, ns, ne,
        mae_check_bar=mae_check_bar, mae_threshold=mae_threshold)
    stats = calc_stats_compound(trades)
    pnl = stats.get('pnl', 0)
    mdd = max(stats.get('mdd', 0), 0.01)

    # Count exit reasons
    exit_counts = {}
    for t in trades:
        r = t.get('reason', 'UNKNOWN')
        exit_counts[r] = exit_counts.get(r, 0) + 1

    return {
        'label': label,
        'trades': stats.get('trades', 0),
        'wr': round(stats.get('wr', 0), 1),
        'pnl': round(pnl, 1),
        'mdd': round(mdd, 2),
        'pnl_mdd': round(pnl / mdd, 1),
        'mae_exits': meta.get('mae_exits', 0),
        'exit_counts': exit_counts,
        'mae_check_bar': mae_check_bar,
        'mae_threshold': mae_threshold,
    }


def phase1_mae_profile(signal_tuples, opens, highs, lows, closes, n_bars,
                       atr_ratio, ema_slope, ns, ne):
    """Phase 1: Profile — at what depth/time do positions become unrecoverable?"""
    logger.info("=== Phase 1: MAE Recovery Profile ===")

    # Run baseline to get trades
    trades, _ = portfolio_npos_mae(
        signal_tuples, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, ns, ne)

    # For each trade, compute unrealized PnL at checkpoints
    checkpoints_bars = [36, 72, 108, 144, 180, 216, 252, 288]
    fee = FEE_PCT * LEVERAGE

    profile = {}
    for cp_bar in checkpoints_bars:
        cp_name = f'{cp_bar * 5 / 60:.0f}h'
        winners_at_cp = 0  # unrealized > 0 at checkpoint
        losers_at_cp = 0
        losers_eventually_won = 0  # was losing at cp, but final reason=TP
        winners_eventually_lost = 0
        loser_avg_unr = []
        winner_avg_unr = []

        for t in trades:
            hold = t['exit_bar'] - t['entry_bar']
            if hold < cp_bar:
                continue  # trade already exited before checkpoint

            entry_price = opens[t['entry_bar']]
            if entry_price <= 0:
                continue

            cp_abs_bar = t['entry_bar'] + cp_bar
            if cp_abs_bar >= n_bars:
                continue

            cp_close = closes[cp_abs_bar]
            if t['direction'] == 'LONG':
                unr = (cp_close / entry_price - 1) * 100 * LEVERAGE
            else:
                unr = (1 - cp_close / entry_price) * 100 * LEVERAGE
            unr -= fee

            if unr > 0:
                winners_at_cp += 1
                winner_avg_unr.append(unr)
                if t['reason'] in ('SL', 'MAE_EXIT', 'TIMEOUT'):
                    winners_eventually_lost += 1
            else:
                losers_at_cp += 1
                loser_avg_unr.append(unr)
                if t['reason'] == 'TP':
                    losers_eventually_won += 1

        total = winners_at_cp + losers_at_cp
        if total == 0:
            continue

        profile[cp_name] = {
            'check_bar': cp_bar,
            'total_alive': total,
            'in_profit': winners_at_cp,
            'in_loss': losers_at_cp,
            'pct_in_loss': round(losers_at_cp / total * 100, 1),
            'losers_recovered_pct': round(losers_eventually_won / max(losers_at_cp, 1) * 100, 1),
            'winners_reversed_pct': round(winners_eventually_lost / max(winners_at_cp, 1) * 100, 1),
            'avg_loser_unr': round(np.mean(loser_avg_unr), 2) if loser_avg_unr else 0,
            'avg_winner_unr': round(np.mean(winner_avg_unr), 2) if winner_avg_unr else 0,
        }

        logger.info(f"  {cp_name}: {total} alive | "
                    f"loss {profile[cp_name]['pct_in_loss']:.0f}% "
                    f"(avg {profile[cp_name]['avg_loser_unr']:+.1f}%) | "
                    f"recovery {profile[cp_name]['losers_recovered_pct']:.0f}% | "
                    f"reversal {profile[cp_name]['winners_reversed_pct']:.0f}%")

    return profile


def phase2_sweep(signal_tuples, opens, highs, lows, closes, n_bars,
                 atr_ratio, ema_slope, ns, ne, baseline_stats):
    """Phase 2: Sweep (check_bar, threshold) combinations."""
    logger.info("=== Phase 2: MAE Exit Sweep ===")

    check_bars = [36, 72, 108, 144]  # 3h, 6h, 9h, 12h
    thresholds = [-1.5, -2.0, -2.5, -3.0, -4.0, -5.0]  # leveraged %

    bl_pm = baseline_stats['pnl_mdd']
    results = {}
    best_key = None
    best_pm = bl_pm

    for cb in check_bars:
        for th in thresholds:
            label = f'cb{cb}_th{th}'
            v = run_variant(label, signal_tuples, opens, highs, lows, closes,
                            n_bars, atr_ratio, ema_slope, ns, ne,
                            mae_check_bar=cb, mae_threshold=th)
            results[label] = v

            delta_pnl = v['pnl'] - baseline_stats['pnl']
            delta_pm = v['pnl_mdd'] - bl_pm

            logger.info(f"  {label}: {v['trades']}t WR{v['wr']:.1f}% "
                        f"PnL{v['pnl']:+.1f}% MDD{v['mdd']:.2f}% "
                        f"PM{v['pnl_mdd']:.1f} MAE_exits={v['mae_exits']} "
                        f"dPnL{delta_pnl:+.1f}% dPM{delta_pm:+.1f}")

            if v['pnl_mdd'] > best_pm:
                best_pm = v['pnl_mdd']
                best_key = label

    if best_key:
        logger.info(f"  Best: {best_key} (PnL/MDD {best_pm:.1f} vs baseline {bl_pm:.1f})")
    else:
        logger.info(f"  No variant beats baseline (PnL/MDD {bl_pm:.1f})")

    return results, best_key


def phase3_wf(signal_tuples_full, opens, highs, lows, closes, n_bars, types,
              atr_ratio, ema_slope, ns, ne,
              mae_check_bar, mae_threshold, label):
    """Phase 3: WF 3-fold expanding window validation."""
    logger.info(f"=== Phase 3: WF — {label} ===")

    total_bars = ne - ns
    n_folds = 3
    fs = total_bars // (n_folds + 1)
    folds = []

    for fold in range(n_folds):
        is_end = ns + (fold + 2) * fs
        oos_s = is_end
        oos_e = min(is_end + fs, ne)
        if oos_s >= ne:
            break

        # IS discovery
        is_si = build_signal_index(types, n_bars)
        is_pats = scan_universe_range(
            is_si, opens, highs, lows, n_bars, ns, is_end, 'per_pattern',
            min_trades=25, edge_threshold=18.0, mc_threshold=0.01,
            atr_ratio=atr_ratio, clamp_lo=DEFAULT_ATR_CLAMP_LO,
            clamp_hi=DEFAULT_ATR_CLAMP_HI)

        # OOS signals
        full_si = build_signal_index(types, n_bars)
        oos_tups = []
        for pi in is_pats:
            pn, d = pi['pattern'], pi['direction']
            pk = f"{pn}_{d}"
            tp = pi.get('tp_pct', pi.get('tp'))
            sl = pi.get('sl_pct', pi.get('sl'))
            if pn in full_si:
                oos_tups.extend(
                    (s, pk, d, tp, sl)
                    for s in full_si[pn] if oos_s <= s < oos_e)

        if not oos_tups:
            folds.append({'fold': fold + 1, 'oos_trades': 0, 'oos_pnl': 0, 'status': 'NO_TRADES'})
            continue

        oos_tups.sort(key=lambda x: x[0])
        trades, meta = portfolio_npos_mae(
            oos_tups, opens, highs, lows, closes, n_bars,
            atr_ratio, ema_slope, oos_s, oos_e,
            mae_check_bar=mae_check_bar, mae_threshold=mae_threshold)
        stats = calc_stats_compound(trades)
        pnl = stats.get('pnl', 0)

        folds.append({
            'fold': fold + 1, 'is_patterns': len(is_pats),
            'oos_trades': stats.get('trades', 0),
            'oos_wr': round(stats.get('wr', 0), 1),
            'oos_pnl': round(pnl, 1),
            'oos_mdd': round(stats.get('mdd', 0), 2),
            'mae_exits': meta.get('mae_exits', 0),
            'status': 'PASS' if pnl > 0 else 'FAIL',
        })
        logger.info(f"  {label} F{fold+1}: OOS {stats.get('trades',0)}t "
                    f"WR{stats.get('wr',0):.1f}% PnL{pnl:+.1f}% "
                    f"MAE_exits={meta.get('mae_exits',0)} "
                    f"{'PASS' if pnl > 0 else 'FAIL'}")

    pc = sum(1 for f in folds if f.get('status') == 'PASS')
    verdict = 'PASS' if pc == len(folds) and len(folds) > 0 else 'FAIL'
    logger.info(f"  WF {label}: {verdict} ({pc}/{len(folds)})")
    return {'label': label, 'folds': folds, 'pass_count': pc,
            'total_folds': len(folds), 'verdict': verdict}


def phase4_random(signal_tuples, opens, highs, lows, closes, n_bars, types,
                  atr_ratio, ema_slope, ns, ne,
                  mae_check_bar, mae_threshold, n_seeds=10):
    """Phase 4: Random discrimination."""
    logger.info(f"=== Phase 4: Random Discrimination (cb={mae_check_bar}, th={mae_threshold}) ===")

    total_bars = ne - ns
    n_folds = 3
    fs = total_bars // (n_folds + 1)
    pass_count = 0

    for seed in range(1, n_seeds + 1):
        rng = np.random.RandomState(seed)
        all_bars = list(range(ns, ne - 1))
        n_sigs = min(len(signal_tuples), len(all_bars))
        rand_bars = sorted(rng.choice(all_bars, size=n_sigs, replace=False))

        pat_keys = list(set(s[1] for s in signal_tuples))
        pat_lookup = {}
        for s in signal_tuples:
            if s[1] not in pat_lookup:
                pat_lookup[s[1]] = s

        rand_tups = []
        for rb in rand_bars:
            pk = rng.choice(pat_keys)
            ref = pat_lookup[pk]
            rand_tups.append((rb, pk, ref[2], ref[3], ref[4]))

        all_pass = True
        for fold in range(n_folds):
            is_end = ns + (fold + 2) * fs
            oos_s = is_end
            oos_e = min(is_end + fs, ne)
            if oos_s >= ne:
                all_pass = False
                break
            oos_tups = sorted(
                [(s, pk, d, tp, sl) for s, pk, d, tp, sl in rand_tups
                 if oos_s <= s < oos_e],
                key=lambda x: x[0])
            if not oos_tups:
                all_pass = False
                break
            trades, _ = portfolio_npos_mae(
                oos_tups, opens, highs, lows, closes, n_bars,
                atr_ratio, ema_slope, oos_s, oos_e,
                mae_check_bar=mae_check_bar, mae_threshold=mae_threshold)
            stats = calc_stats_compound(trades)
            if stats.get('pnl', 0) <= 0:
                all_pass = False
                break
        if all_pass:
            pass_count += 1

    pct = pass_count / n_seeds * 100
    logger.info(f"  Random: {pass_count}/{n_seeds} pass ({pct:.0f}%)")
    return {'n_seeds': n_seeds, 'pass_count': pass_count, 'pass_pct': pct,
            'verdict': 'NON-DISCRIMINATING' if pct >= 80 else 'DISCRIMINATING'}


def main():
    t0 = time.time()
    logger.info("=== MAE-Based Early Exit Study ===")

    df = load_and_classify(DATA_FILE)
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    n_bars = len(df)
    types = df['candle_type'].values
    ns, ne = find_neutral_window(closes)
    logger.info(f"Neutral window: {ns}-{ne} ({(ne-ns)/288:.0f}d)")

    atr_ratio = compute_atr_ratio(highs, lows, closes,
                                   atr_period=DEFAULT_ATR_PERIOD, window=DEFAULT_ATR_WINDOW)
    ema_slope = compute_ema_slope(closes)

    signal_index = build_signal_index(types, n_bars)
    patterns = load_patterns()
    signal_tuples = build_signal_tuples(patterns, signal_index, ns, ne)
    logger.info(f"{len(patterns)} patterns, {len(signal_tuples)} signals")

    results = {'metadata': {
        'script': 'mae_early_exit_study.py',
        'date': pd.Timestamp.now().isoformat(),
        'n_patterns': len(patterns),
        'n_signals': len(signal_tuples),
        'neutral_window': [int(ns), int(ne)],
    }}

    # Baseline
    bl = run_variant('baseline', signal_tuples, opens, highs, lows, closes,
                     n_bars, atr_ratio, ema_slope, ns, ne)
    results['baseline'] = bl
    logger.info(f"Baseline: {bl['trades']}t WR{bl['wr']:.1f}% "
                f"PnL{bl['pnl']:+.1f}% MDD{bl['mdd']:.2f}% PM{bl['pnl_mdd']:.1f}")

    # Phase 1: Recovery profile
    results['phase1'] = phase1_mae_profile(
        signal_tuples, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, ns, ne)

    # Phase 2: Sweep
    sweep_results, best_key = phase2_sweep(
        signal_tuples, opens, highs, lows, closes, n_bars,
        atr_ratio, ema_slope, ns, ne, bl)
    results['phase2'] = sweep_results

    # Phase 3 & 4: WF + Random for baseline and best
    wf_bl = phase3_wf(signal_tuples, opens, highs, lows, closes, n_bars, types,
                      atr_ratio, ema_slope, ns, ne, None, None, 'baseline')
    results['phase3'] = {'baseline': wf_bl}

    rand_bl = phase4_random(signal_tuples, opens, highs, lows, closes, n_bars, types,
                            atr_ratio, ema_slope, ns, ne, None, None)
    results['phase4'] = {'baseline': rand_bl}

    if best_key and best_key in sweep_results:
        best = sweep_results[best_key]
        cb = best['mae_check_bar']
        th = best['mae_threshold']

        wf_best = phase3_wf(signal_tuples, opens, highs, lows, closes, n_bars, types,
                            atr_ratio, ema_slope, ns, ne, cb, th, best_key)
        results['phase3']['best'] = wf_best

        rand_best = phase4_random(signal_tuples, opens, highs, lows, closes, n_bars, types,
                                  atr_ratio, ema_slope, ns, ne, cb, th)
        results['phase4']['best'] = rand_best

        # Also test top-3
        sorted_variants = sorted(
            [(k, v) for k, v in sweep_results.items()],
            key=lambda x: x[1].get('pnl_mdd', 0), reverse=True)

        for rank, (vk, vv) in enumerate(sorted_variants[:3]):
            if vk == best_key:
                continue
            vcb = vv['mae_check_bar']
            vth = vv['mae_threshold']
            if vcb is None:
                continue
            wf_v = phase3_wf(signal_tuples, opens, highs, lows, closes, n_bars, types,
                             atr_ratio, ema_slope, ns, ne, vcb, vth, vk)
            results['phase3'][vk] = wf_v

    elapsed = time.time() - t0
    results['metadata']['runtime_sec'] = round(elapsed, 1)

    # Verdict
    logger.info(f"\n{'='*70}")
    logger.info(f"MAE EARLY EXIT STUDY — VERDICT")
    logger.info(f"{'='*70}")
    logger.info(f"Baseline: {bl['trades']}t PnL{bl['pnl']:+.1f}% PM{bl['pnl_mdd']:.1f}")

    if best_key and best_key in sweep_results:
        best_v = sweep_results[best_key]
        delta_pnl = best_v['pnl'] - bl['pnl']
        delta_pm = best_v['pnl_mdd'] - bl['pnl_mdd']
        wf_v = results['phase3'].get('best', {}).get('verdict', 'N/A')
        rand_v = results['phase4'].get('best', {}).get('verdict', 'N/A')

        logger.info(f"Best: {best_key} | PnL{best_v['pnl']:+.1f}% PM{best_v['pnl_mdd']:.1f} "
                    f"MAE_exits={best_v['mae_exits']}")
        logger.info(f"  dPnL{delta_pnl:+.1f}% dPM{delta_pm:+.1f}")
        logger.info(f"  WF: {wf_v} | Random: {rand_v}")

        if delta_pm > 0 and wf_v == 'PASS':
            if rand_v == 'NON-DISCRIMINATING':
                verdict = 'STOP_NON_DISC'
                reason = 'WF PASS but NON-DISCRIMINATING (random도 pass)'
            else:
                verdict = 'GO'
                reason = f'IS +{delta_pm:.0f} PM, WF PASS, DISCRIMINATING'
        elif delta_pm > 0 and wf_v == 'FAIL':
            verdict = 'STOP_WF_FAIL'
            reason = 'IS 개선되나 WF FAIL (overfitting 가능)'
        else:
            verdict = 'STOP'
            reason = 'baseline 대비 개선 없음'
    else:
        verdict = 'STOP'
        reason = 'No variant beats baseline'

    results['verdict'] = {
        'decision': verdict,
        'reason': reason,
        'best_variant': best_key,
        'baseline_pm': bl['pnl_mdd'],
        'best_pm': sweep_results[best_key]['pnl_mdd'] if best_key else bl['pnl_mdd'],
    }

    logger.info(f"Verdict: {verdict} — {reason}")
    logger.info(f"Runtime: {elapsed:.1f}s")

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved to {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
