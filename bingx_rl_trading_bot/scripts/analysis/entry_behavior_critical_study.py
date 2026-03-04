#!/usr/bin/env python3
"""
Entry Behavior Critical Analysis Study
========================================
v1.42.0 프로덕션 진입 행태의 비판적 심층 분석.

7가지 비판적 관점:
1. 동시 다발 신호 — 같은 bar에 여러 신호 발생 시 선택 순서의 영향
2. 시간대별 성과 — 진입 시간대에 따른 WR/PnL 편차
3. 방향 비대칭 — LONG vs SHORT 성과 갭 + 레짐별 분석
4. Guard 사후 분석 — 각 guard가 차단한 진입의 가상 PnL (opportunity cost)
5. 진입 밀도 — 동시 포지션 수 증가에 따른 한계 수익 체감
6. 패턴 상호작용 — 동시 활성 패턴 조합별 성과
7. 진입 직후 행태 — 진입 후 1-5봉 즉시 손실 vs 유리한 출발 비율

Standard Research Protocol: LEVERAGE=3, fee=0.10%×lev, timeout=DROP, compound sizing.
"""
import sys, os, json, time
import numpy as np
from datetime import datetime
from collections import defaultdict, Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from scripts.analysis.stack_resolution_study import (
    load_and_classify, compute_atr_ratio, compute_ema_slope,
    find_neutral_window, portfolio_sim, calc_stats, clamp,
    DATA_FILE, PATTERNS_FILE,
    N_SLOTS, DIRECTION_CAP, LEVERAGE, FEE_PCT,
    TIMEOUT_BARS, ATR_CLAMP_LO, ATR_CLAMP_HI, SLIPPAGE_BUFFER,
    AGG_RISK_COUNTER, AGG_RISK_WITH, BARS_PER_DAY,
    MDD_FULL_BELOW, MDD_MIN_ABOVE, MDD_MIN_SCALE,
    MOMENTUM_LOOKBACK, MOMENTUM_THRESHOLD, MOMENTUM_COOLDOWN,
    EARLY_CONFIRM, EARLY_MIN_PROFIT,
)

# ============================================================
# CUSTOM SIM: tracks per-entry metadata for analysis
# ============================================================
def sim_with_metadata(
    signal_tuples, opens, highs, lows, closes, type_codes, n_bars,
    atr_ratio, ema_slope, start_bar, end_bar,
):
    """Full v1.42.0 sim that also records entry metadata for analysis."""
    fee = FEE_PCT * LEVERAGE
    size_pct = 100.0 / N_SLOTS

    positions = []
    trades = []
    equity = 100.0
    peak_equity = 100.0
    mom_pause_until = {'LONG': -1, 'SHORT': -1}

    # Track blocked signals for guard analysis
    blocked = {'momentum': [], 'dir_cap': [], 'agg_risk': [], 'slot_full': [], 'dup_pattern': []}

    sig_by_bar = {}
    for s_bar, pat, direction, tp, sl in signal_tuples:
        if start_bar <= s_bar < end_bar:
            sig_by_bar.setdefault(s_bar, []).append((pat, direction, tp, sl))

    # Track multi-signal bars
    multi_signal_bars = {b: sigs for b, sigs in sig_by_bar.items() if len(sigs) > 1}

    for bar in range(start_bar, end_bar):
        if bar >= n_bars - 1:
            break

        # ----- EXITS -----
        closed_slots = set()
        sl_exits = []

        h_bar = highs[bar] if bar < n_bars else 0
        l_bar = lows[bar] if bar < n_bars else 0
        o_bar = opens[bar] if bar < n_bars else 0

        for pos in positions:
            entry_bar = pos['entry_bar']
            if bar < entry_bar:
                continue
            entry_p = pos['entry_price']
            if entry_p <= 0:
                continue

            direction = pos['direction']
            eff_tp = pos['eff_tp_pct']
            eff_sl = pos['eff_sl_pct']
            sm = pos['size_mult']
            hold = bar - entry_bar

            exit_price = None
            reason = None

            # Timeout
            if hold >= TIMEOUT_BARS:
                closed_slots.add(pos['slot'])
                continue

            # Early exit
            if hold >= EARLY_CONFIRM:
                exit_types = ['BD'] if direction == 'LONG' else ['BU']
                candle_ok = True
                for k in range(EARLY_CONFIRM):
                    cb = bar - 1 - k
                    if cb < 0 or cb >= len(type_codes) or type_codes[cb] not in exit_types:
                        candle_ok = False
                        break
                if candle_ok:
                    cp = closes[bar - 1] if bar - 1 < n_bars else entry_p
                    if direction == 'LONG':
                        unr = (cp / entry_p - 1) * 100 * LEVERAGE
                    else:
                        unr = (1 - cp / entry_p) * 100 * LEVERAGE
                    if unr >= EARLY_MIN_PROFIT:
                        exit_price = cp
                        reason = 'EARLY'

            # TP/SL
            if reason is None:
                if direction == 'LONG':
                    tp_p = entry_p * (1 + eff_tp / 100)
                    sl_p = entry_p * (1 - eff_sl / 100)
                    hit_tp = h_bar >= tp_p
                    hit_sl = l_bar <= sl_p
                else:
                    tp_p = entry_p * (1 - eff_tp / 100)
                    sl_p = entry_p * (1 + eff_sl / 100)
                    hit_tp = l_bar <= tp_p
                    hit_sl = h_bar >= sl_p

                if hit_tp and hit_sl:
                    if abs(tp_p - o_bar) <= abs(sl_p - o_bar):
                        exit_price, reason = tp_p, 'TP'
                    else:
                        exit_price, reason = sl_p, 'SL'
                elif hit_tp:
                    exit_price, reason = tp_p, 'TP'
                elif hit_sl:
                    exit_price, reason = sl_p, 'SL'

            if reason is None:
                continue

            if direction == 'LONG':
                pnl = (exit_price / entry_p - 1) * 100 * LEVERAGE
            else:
                pnl = (1 - exit_price / entry_p) * 100 * LEVERAGE
            pnl -= fee

            pnl_portfolio = pnl * (size_pct / 100) * sm
            trades.append({
                'entry_bar': entry_bar, 'exit_bar': bar, 'pnl_slot': pnl,
                'reason': reason, 'pattern': pos['pattern'],
                'direction': direction, 'size_mult': sm,
                'pnl_portfolio': pnl_portfolio,
                'signal_bar': pos['signal_bar'],
                'n_active_at_entry': pos['n_active_at_entry'],
                'n_signals_same_bar': pos['n_signals_same_bar'],
                'hold_bars': bar - entry_bar,
            })
            closed_slots.add(pos['slot'])

            if reason == 'SL':
                sl_exits.append(pos)

        positions = [p for p in positions if p['slot'] not in closed_slots]

        # Cascade
        if sl_exits:
            for sl_pos in sl_exits:
                sl_dir = sl_pos['direction']
                for pos in positions:
                    if pos['direction'] == sl_dir:
                        pos['eff_sl_pct'] *= 0.25

        equity *= (1 + sum(t['pnl_portfolio'] for t in trades if t['exit_bar'] == bar) / 100)
        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0

        # Momentum
        if bar >= MOMENTUM_LOOKBACK:
            pa = closes[bar - MOMENTUM_LOOKBACK]
            if pa > 0:
                pct = (closes[bar] / pa - 1) * 100
                if pct > MOMENTUM_THRESHOLD:
                    mom_pause_until['SHORT'] = max(mom_pause_until['SHORT'],
                                                   bar + MOMENTUM_COOLDOWN)
                elif pct < -MOMENTUM_THRESHOLD:
                    mom_pause_until['LONG'] = max(mom_pause_until['LONG'],
                                                  bar + MOMENTUM_COOLDOWN)

        # ----- ENTRIES -----
        if bar not in sig_by_bar:
            continue

        sigs = sig_by_bar[bar]
        n_sigs = len(sigs)

        for pat, direction, tp_pct, sl_pct in sigs:
            block_reason = None

            if len(positions) >= N_SLOTS:
                block_reason = 'slot_full'
            elif sum(1 for p in positions if p['direction'] == direction) >= DIRECTION_CAP:
                block_reason = 'dir_cap'
            elif any(p['pattern'] == pat for p in positions):
                block_reason = 'dup_pattern'
            elif bar < mom_pause_until.get(direction, -1):
                block_reason = 'momentum'
            else:
                # MDD sizing
                sm = 1.0
                if peak_equity > 0:
                    dd_pct = (peak_equity - equity) / peak_equity * 100
                    if dd_pct <= MDD_FULL_BELOW:
                        mdd_scale = 1.0
                    elif dd_pct >= MDD_MIN_ABOVE:
                        mdd_scale = MDD_MIN_SCALE
                    else:
                        mdd_scale = 1.0 - (1.0 - MDD_MIN_SCALE) * (
                            dd_pct - MDD_FULL_BELOW) / (MDD_MIN_ABOVE - MDD_FULL_BELOW)
                    sm *= mdd_scale

                # ATR
                if bar < len(atr_ratio) and not np.isnan(atr_ratio[bar]):
                    r = clamp(atr_ratio[bar], ATR_CLAMP_LO, ATR_CLAMP_HI)
                else:
                    r = 1.0
                eff_tp = tp_pct * r + SLIPPAGE_BUFFER
                eff_sl = max(0.1, sl_pct * r - SLIPPAGE_BUFFER)

                # Agg risk
                slope = ema_slope[bar] if bar < len(ema_slope) else 0
                is_uptrend = slope > 0
                is_counter = ((direction == 'SHORT' and is_uptrend) or
                              (direction == 'LONG' and not is_uptrend))
                cap_pct = AGG_RISK_COUNTER if is_counter else AGG_RISK_WITH
                existing = sum(
                    p['eff_sl_pct'] * (1.0 / N_SLOTS) * LEVERAGE * p['size_mult']
                    for p in positions if p['direction'] == direction
                )
                new_exp = eff_sl * (1.0 / N_SLOTS) * LEVERAGE * sm
                if existing + new_exp > cap_pct:
                    block_reason = 'agg_risk'

            if block_reason:
                blocked[block_reason].append({
                    'bar': bar, 'pattern': pat, 'direction': direction,
                    'tp': tp_pct, 'sl': sl_pct,
                })
                continue

            entry_bar = bar + 1
            if entry_bar >= n_bars:
                continue
            entry_price = opens[entry_bar]
            if entry_price <= 0:
                continue

            positions.append({
                'slot': f"{pat}_{bar}",
                'entry_bar': entry_bar,
                'entry_price': entry_price,
                'direction': direction,
                'pattern': pat,
                'tp_pct': tp_pct,
                'sl_pct': sl_pct,
                'eff_tp_pct': eff_tp,
                'eff_sl_pct': eff_sl,
                'size_mult': sm,
                'signal_bar': bar,
                'n_active_at_entry': len(positions),
                'n_signals_same_bar': n_sigs,
            })

    return trades, blocked, multi_signal_bars


# ============================================================
# ANALYSIS FUNCTIONS
# ============================================================

def analyze_multi_signal_bars(trades, multi_signal_bars, blocked):
    """Test 1: 동시 다발 신호 분석"""
    print("\n[TEST 1] 동시 다발 신호 분석 (Multi-Signal Bar)")
    print("=" * 78)

    single_sig = [t for t in trades if t['n_signals_same_bar'] == 1]
    multi_sig = [t for t in trades if t['n_signals_same_bar'] > 1]

    print(f"  Single-signal entries: {len(single_sig)}")
    print(f"  Multi-signal entries:  {len(multi_sig)}")
    print(f"  Multi-signal bars:     {len(multi_signal_bars)}")

    if single_sig:
        s_wr = sum(1 for t in single_sig if t['pnl_portfolio'] > 0) / len(single_sig) * 100
        s_pnl = sum(t['pnl_portfolio'] for t in single_sig)
        print(f"  Single: WR={s_wr:.1f}%, PnL={s_pnl:+.1f}%")

    if multi_sig:
        m_wr = sum(1 for t in multi_sig if t['pnl_portfolio'] > 0) / len(multi_sig) * 100
        m_pnl = sum(t['pnl_portfolio'] for t in multi_sig)
        print(f"  Multi:  WR={m_wr:.1f}%, PnL={m_pnl:+.1f}%")

    # Signal count distribution
    count_dist = Counter(len(sigs) for sigs in multi_signal_bars.values())
    print(f"\n  Signals per bar distribution:")
    for n, cnt in sorted(count_dist.items()):
        print(f"    {n} signals: {cnt} bars")

    # Direction mix in multi-signal bars
    mixed = 0
    same = 0
    for bar, sigs in multi_signal_bars.items():
        dirs = set(s[1] for s in sigs)
        if len(dirs) > 1:
            mixed += 1
        else:
            same += 1
    print(f"\n  Mixed direction bars: {mixed}")
    print(f"  Same direction bars:  {same}")

    # First-vs-last signal in multi-bar: does order matter?
    if multi_sig:
        first_in_bar = {}
        last_in_bar = {}
        for t in trades:
            sb = t['signal_bar']
            if sb not in first_in_bar or t['entry_bar'] <= first_in_bar[sb]['entry_bar']:
                first_in_bar[sb] = t
            if sb not in last_in_bar or t['entry_bar'] >= last_in_bar[sb]['entry_bar']:
                last_in_bar[sb] = t

    return {
        'single_count': len(single_sig),
        'multi_count': len(multi_sig),
        'single_wr': round(s_wr, 2) if single_sig else 0,
        'multi_wr': round(m_wr, 2) if multi_sig else 0,
        'single_pnl': round(s_pnl, 2) if single_sig else 0,
        'multi_pnl': round(m_pnl, 2) if multi_sig else 0,
        'mixed_dir_bars': mixed,
        'same_dir_bars': same,
    }


def analyze_time_of_day(trades, opens, bars_per_day=288):
    """Test 2: 시간대별 성과 분석"""
    print("\n[TEST 2] 시간대별 진입 성과")
    print("=" * 78)

    # 5min bars: 288 per day, group by 4h blocks (48 bars)
    block_size = 48  # 4 hours
    n_blocks = bars_per_day // block_size

    by_block = defaultdict(list)
    for t in trades:
        bar_of_day = t['entry_bar'] % bars_per_day
        block = bar_of_day // block_size
        by_block[block].append(t)

    print(f"  {'Block':>6} {'UTC Hours':>12} {'Trades':>8} {'WR%':>8} {'PnL%':>10} {'AvgPnL':>10}")
    print(f"  {'-'*60}")

    block_stats = {}
    for b in range(n_blocks):
        h_start = b * 4
        h_end = h_start + 4
        ts = by_block.get(b, [])
        if not ts:
            continue
        wr = sum(1 for t in ts if t['pnl_portfolio'] > 0) / len(ts) * 100
        pnl = sum(t['pnl_portfolio'] for t in ts)
        avg = np.mean([t['pnl_portfolio'] for t in ts])
        print(f"  {b:>6} {h_start:>02d}:00-{h_end:>02d}:00 {len(ts):>8} {wr:>8.1f} {pnl:>+10.1f} {avg:>+10.3f}")
        block_stats[f"{h_start:02d}-{h_end:02d}"] = {
            'trades': len(ts), 'wr': round(wr, 2),
            'pnl': round(pnl, 2), 'avg_pnl': round(float(avg), 4)
        }

    # Best and worst blocks
    if block_stats:
        best = max(block_stats.items(), key=lambda x: x[1]['avg_pnl'])
        worst = min(block_stats.items(), key=lambda x: x[1]['avg_pnl'])
        print(f"\n  Best:  {best[0]} UTC (avg {best[1]['avg_pnl']:+.3f}%)")
        print(f"  Worst: {worst[0]} UTC (avg {worst[1]['avg_pnl']:+.3f}%)")

    return block_stats


def analyze_direction_asymmetry(trades, ema_slope):
    """Test 3: 방향 비대칭 + 레짐별 분석"""
    print("\n[TEST 3] 방향 비대칭 분석")
    print("=" * 78)

    longs = [t for t in trades if t['direction'] == 'LONG']
    shorts = [t for t in trades if t['direction'] == 'SHORT']

    for label, ts in [('LONG', longs), ('SHORT', shorts)]:
        if not ts:
            print(f"  {label}: no trades")
            continue
        wr = sum(1 for t in ts if t['pnl_portfolio'] > 0) / len(ts) * 100
        pnl = sum(t['pnl_portfolio'] for t in ts)
        avg = np.mean([t['pnl_portfolio'] for t in ts])
        tp_count = sum(1 for t in ts if t['reason'] == 'TP')
        sl_count = sum(1 for t in ts if t['reason'] == 'SL')
        print(f"  {label}: {len(ts)} trades, WR={wr:.1f}%, PnL={pnl:+.1f}%, "
              f"Avg={avg:+.3f}%, TP={tp_count}, SL={sl_count}")

    # Regime analysis
    print(f"\n  Regime analysis (EMA slope):")
    regime_stats = defaultdict(lambda: {'trades': 0, 'wins': 0, 'pnl': 0.0})
    for t in trades:
        eb = t['entry_bar']
        slope = ema_slope[eb] if eb < len(ema_slope) else 0
        regime = 'uptrend' if slope > 0 else 'downtrend'
        is_counter = ((t['direction'] == 'SHORT' and regime == 'uptrend') or
                      (t['direction'] == 'LONG' and regime == 'downtrend'))
        key = f"{t['direction']}_{regime}_{'counter' if is_counter else 'with'}"
        regime_stats[key]['trades'] += 1
        if t['pnl_portfolio'] > 0:
            regime_stats[key]['wins'] += 1
        regime_stats[key]['pnl'] += t['pnl_portfolio']

    print(f"  {'Combo':<35} {'Trades':>8} {'WR%':>8} {'PnL%':>10}")
    print(f"  {'-'*65}")
    for key in sorted(regime_stats.keys()):
        s = regime_stats[key]
        wr = s['wins'] / s['trades'] * 100 if s['trades'] > 0 else 0
        print(f"  {key:<35} {s['trades']:>8} {wr:>8.1f} {s['pnl']:>+10.1f}")

    return {
        'long_trades': len(longs),
        'short_trades': len(shorts),
        'long_wr': round(sum(1 for t in longs if t['pnl_portfolio'] > 0) / max(len(longs), 1) * 100, 2),
        'short_wr': round(sum(1 for t in shorts if t['pnl_portfolio'] > 0) / max(len(shorts), 1) * 100, 2),
        'long_pnl': round(sum(t['pnl_portfolio'] for t in longs), 2),
        'short_pnl': round(sum(t['pnl_portfolio'] for t in shorts), 2),
        'regime_stats': {k: {'trades': v['trades'], 'wr': round(v['wins']/max(v['trades'],1)*100, 2),
                             'pnl': round(v['pnl'], 2)} for k, v in regime_stats.items()},
    }


def analyze_guard_opportunity_cost(blocked, opens, highs, lows, closes, n_bars,
                                    atr_ratio, ema_slope):
    """Test 4: Guard 사후 분석 — 차단된 진입의 가상 PnL"""
    print("\n[TEST 4] Guard Opportunity Cost 분석")
    print("=" * 78)

    fee = FEE_PCT * LEVERAGE

    for guard_name, signals in blocked.items():
        if not signals:
            print(f"  {guard_name}: 0 blocked")
            continue

        # Simulate each blocked signal as if it were entered solo
        wins = 0
        pnls = []
        for sig in signals:
            bar = sig['bar']
            entry_bar = bar + 1
            if entry_bar >= n_bars:
                continue
            entry_price = opens[entry_bar]
            if entry_price <= 0:
                continue

            direction = sig['direction']
            tp_pct = sig['tp']
            sl_pct = sig['sl']

            # ATR scaling
            if bar < len(atr_ratio) and not np.isnan(atr_ratio[bar]):
                r = clamp(atr_ratio[bar], ATR_CLAMP_LO, ATR_CLAMP_HI)
            else:
                r = 1.0
            eff_tp = tp_pct * r + SLIPPAGE_BUFFER
            eff_sl = max(0.1, sl_pct * r - SLIPPAGE_BUFFER)

            # Simulate forward to TP/SL/Timeout
            outcome_pnl = None
            for j in range(entry_bar, min(entry_bar + TIMEOUT_BARS, n_bars)):
                if direction == 'LONG':
                    tp_p = entry_price * (1 + eff_tp / 100)
                    sl_p = entry_price * (1 - eff_sl / 100)
                    if highs[j] >= tp_p:
                        outcome_pnl = eff_tp * LEVERAGE - fee
                        break
                    if lows[j] <= sl_p:
                        outcome_pnl = -eff_sl * LEVERAGE - fee
                        break
                else:
                    tp_p = entry_price * (1 - eff_tp / 100)
                    sl_p = entry_price * (1 + eff_sl / 100)
                    if lows[j] <= tp_p:
                        outcome_pnl = eff_tp * LEVERAGE - fee
                        break
                    if highs[j] >= sl_p:
                        outcome_pnl = -eff_sl * LEVERAGE - fee
                        break

            if outcome_pnl is None:
                outcome_pnl = 0  # timeout = drop

            pnls.append(outcome_pnl)
            if outcome_pnl > 0:
                wins += 1

        wr = wins / len(pnls) * 100 if pnls else 0
        avg = np.mean(pnls) if pnls else 0
        total = sum(pnls)
        print(f"  {guard_name:<15}: {len(signals):>5} blocked, "
              f"Hypothetical WR={wr:.1f}%, AvgPnL={avg:+.2f}%, Total={total:+.1f}%")

    return {guard: len(sigs) for guard, sigs in blocked.items()}


def analyze_entry_density(trades):
    """Test 5: 진입 밀도 — 동시 포지션 수에 따른 한계 수익"""
    print("\n[TEST 5] 진입 밀도 (동시 포지션 수) 분석")
    print("=" * 78)

    by_density = defaultdict(list)
    for t in trades:
        n = t['n_active_at_entry']  # positions active WHEN this entry happened
        by_density[n].append(t)

    print(f"  {'N_active':>10} {'Trades':>8} {'WR%':>8} {'AvgPnL':>10} {'TotalPnL':>10}")
    print(f"  {'-'*50}")

    density_stats = {}
    for n in sorted(by_density.keys()):
        ts = by_density[n]
        wr = sum(1 for t in ts if t['pnl_portfolio'] > 0) / len(ts) * 100
        avg = np.mean([t['pnl_portfolio'] for t in ts])
        total = sum(t['pnl_portfolio'] for t in ts)
        print(f"  {n:>10} {len(ts):>8} {wr:>8.1f} {avg:>+10.3f} {total:>+10.1f}")
        density_stats[n] = {'trades': len(ts), 'wr': round(wr, 2),
                            'avg_pnl': round(float(avg), 4), 'total_pnl': round(total, 2)}

    return density_stats


def analyze_immediate_behavior(trades, opens, highs, lows, closes, n_bars):
    """Test 6: 진입 직후 행태 — 1-5봉 후 가격 움직임"""
    print("\n[TEST 6] 진입 직후 행태 (1-5봉)")
    print("=" * 78)

    lookbacks = [1, 2, 3, 5, 10]
    stats = {}

    for lb in lookbacks:
        favorable = 0
        adverse = 0
        avg_move = []

        for t in trades:
            eb = t['entry_bar']
            check_bar = eb + lb
            if check_bar >= n_bars:
                continue

            entry_p = opens[eb]
            if entry_p <= 0:
                continue

            if t['direction'] == 'LONG':
                move = (closes[check_bar] / entry_p - 1) * 100
            else:
                move = (1 - closes[check_bar] / entry_p) * 100

            avg_move.append(move)
            if move > 0:
                favorable += 1
            else:
                adverse += 1

        total = favorable + adverse
        fav_pct = favorable / total * 100 if total > 0 else 0
        avg_m = np.mean(avg_move) if avg_move else 0
        med_m = np.median(avg_move) if avg_move else 0

        print(f"  After {lb:>2} bars: Favorable={fav_pct:.1f}% ({favorable}/{total}), "
              f"Avg Move={avg_m:+.4f}%, Median={med_m:+.4f}%")
        stats[lb] = {'favorable_pct': round(fav_pct, 2), 'avg_move': round(float(avg_m), 5),
                     'median_move': round(float(med_m), 5)}

    return stats


def analyze_holding_time_distribution(trades):
    """Test 7: 보유 시간 분포 + 수익과의 관계"""
    print("\n[TEST 7] 보유 시간 분포")
    print("=" * 78)

    hold_times = [t['hold_bars'] for t in trades]

    # Percentile analysis
    pcts = [10, 25, 50, 75, 90, 95, 99]
    print("  Holding time percentiles:")
    for p in pcts:
        v = np.percentile(hold_times, p)
        print(f"    P{p:02d}: {v:.0f} bars ({v/12:.1f}h)")

    # Holding time vs outcome
    buckets = [(0, 12), (12, 48), (48, 144), (144, 288), (288, 864)]
    print(f"\n  {'Hold Range':>15} {'Trades':>8} {'WR%':>8} {'AvgPnL':>10} {'AvgHold_h':>10}")
    print(f"  {'-'*55}")

    hold_stats = {}
    for lo, hi in buckets:
        ts = [t for t in trades if lo <= t['hold_bars'] < hi]
        if not ts:
            continue
        wr = sum(1 for t in ts if t['pnl_portfolio'] > 0) / len(ts) * 100
        avg = np.mean([t['pnl_portfolio'] for t in ts])
        avg_h = np.mean([t['hold_bars'] for t in ts]) / 12
        label = f"{lo/12:.0f}-{hi/12:.0f}h"
        print(f"  {label:>15} {len(ts):>8} {wr:>8.1f} {avg:>+10.3f} {avg_h:>10.1f}")
        hold_stats[label] = {'trades': len(ts), 'wr': round(wr, 2),
                             'avg_pnl': round(float(avg), 4)}

    # Exit reason distribution
    reason_counts = Counter(t['reason'] for t in trades)
    print(f"\n  Exit reasons: {dict(reason_counts)}")

    return {
        'percentiles': {f'p{p}': round(float(np.percentile(hold_times, p)), 1) for p in pcts},
        'buckets': hold_stats,
        'exit_reasons': dict(reason_counts),
    }


# ============================================================
# MAIN
# ============================================================
def main():
    t_start = time.time()
    print("=" * 78)
    print("ENTRY BEHAVIOR CRITICAL ANALYSIS")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Settings: LEV={LEVERAGE}, N={N_SLOTS}, DirCap={DIRECTION_CAP}, "
          f"TO={TIMEOUT_BARS}")
    print("=" * 78)

    # Load
    print("\n[0] Loading data...")
    df = load_and_classify(DATA_FILE)
    n_bars = len(df)
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    type_codes = df['rctype'].values

    atr_ratio = compute_atr_ratio(df)
    ema_slope = compute_ema_slope(closes)

    ns, ne = find_neutral_window(closes)
    print(f"  Neutral: {ns}-{ne} ({ne-ns} bars, {(ne-ns)/BARS_PER_DAY:.0f}d)")

    # Load patterns
    with open(PATTERNS_FILE) as f:
        pdata = json.load(f)
    pats_raw = pdata['patterns']
    tpsl = pdata.get('patterns_tpsl', {})

    pat_lookup = {}
    for pat_name in pats_raw.get('long', []):
        tp_sl = tpsl.get(pat_name, [2.0, 3.0])
        pat_lookup[pat_name] = {'direction': 'LONG', 'tp': tp_sl[0], 'sl': tp_sl[1]}
    for pat_name in pats_raw.get('short', []):
        tp_sl = tpsl.get(pat_name, [2.0, 3.0])
        pat_lookup[pat_name] = {'direction': 'SHORT', 'tp': tp_sl[0], 'sl': tp_sl[1]}

    rctypes = df['rctype'].values
    signal_tuples = []
    for i in range(2, n_bars):
        tri = f"{rctypes[i-2]}-{rctypes[i-1]}-{rctypes[i]}"
        if tri in pat_lookup:
            p = pat_lookup[tri]
            signal_tuples.append((i, tri, p['direction'], p['tp'], p['sl']))

    n_in_neutral = sum(1 for s in signal_tuples if ns <= s[0] < ne)
    print(f"  {len(pat_lookup)} patterns, {len(signal_tuples)} total signals, "
          f"{n_in_neutral} in neutral")

    # Run sim with metadata
    print("\n[SIM] Running metadata-tracked simulation...")
    trades, blocked, multi_signal_bars = sim_with_metadata(
        signal_tuples, opens, highs, lows, closes, type_codes, n_bars,
        atr_ratio, ema_slope, ns, ne,
    )
    stats = calc_stats(trades)
    print(f"  {stats['trades']} trades, WR={stats['wr']:.1f}%, PnL={stats['pnl']:+.1f}%, "
          f"MDD={stats['mdd']:.2f}%, PnL/MDD={stats['pnl_mdd']:.1f}")

    results = {'meta': {
        'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'version': 'v1.42.0',
        'n_bars': n_bars,
        'neutral_bars': int(ne - ns),
        'total_signals': len(signal_tuples),
        'baseline_stats': {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                           for k, v in stats.items()},
    }}

    # Run all 7 tests
    results['test1_multi_signal'] = analyze_multi_signal_bars(trades, multi_signal_bars, blocked)
    results['test2_time_of_day'] = analyze_time_of_day(trades, opens)
    results['test3_direction'] = analyze_direction_asymmetry(trades, ema_slope)
    results['test4_guard_cost'] = analyze_guard_opportunity_cost(
        blocked, opens, highs, lows, closes, n_bars, atr_ratio, ema_slope)
    results['test5_density'] = analyze_entry_density(trades)
    results['test6_immediate'] = analyze_immediate_behavior(trades, opens, highs, lows, closes, n_bars)
    results['test7_holding'] = analyze_holding_time_distribution(trades)

    # ============================================================
    # SYNTHESIS
    # ============================================================
    print("\n" + "=" * 78)
    print("SYNTHESIS: 비판적 발견 요약")
    print("=" * 78)

    findings = []

    # Test 1 findings
    t1 = results['test1_multi_signal']
    if t1['multi_count'] > 0:
        wr_gap = t1['multi_wr'] - t1['single_wr']
        findings.append(f"Multi-signal WR gap: {wr_gap:+.1f}pp (multi={t1['multi_wr']:.1f}% vs single={t1['single_wr']:.1f}%)")

    # Test 3 findings
    t3 = results['test3_direction']
    dir_gap = t3['long_wr'] - t3['short_wr']
    findings.append(f"Direction WR gap: LONG={t3['long_wr']:.1f}% vs SHORT={t3['short_wr']:.1f}% (gap={dir_gap:+.1f}pp)")
    findings.append(f"Direction PnL gap: LONG={t3['long_pnl']:+.1f}% vs SHORT={t3['short_pnl']:+.1f}%")

    # Test 4 findings
    t4 = results['test4_guard_cost']
    for guard, count in t4.items():
        if count > 0:
            findings.append(f"Guard '{guard}': {count} signals blocked")

    # Test 6 findings
    t6 = results['test6_immediate']
    for lb, s in t6.items():
        if lb in [1, 5]:
            findings.append(f"After {lb} bars: {s['favorable_pct']:.1f}% favorable, avg={s['avg_move']:+.4f}%")

    print()
    for i, f in enumerate(findings, 1):
        print(f"  [{i}] {f}")

    # Save
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', '..', 'results', 'entry_behavior_critical.json')
    with open(out_path, 'w') as fp:
        json.dump(results, fp, indent=2, default=str)
    print(f"\nSaved: {out_path}")
    print(f"Runtime: {time.time() - t_start:.1f}s")


if __name__ == '__main__':
    main()
