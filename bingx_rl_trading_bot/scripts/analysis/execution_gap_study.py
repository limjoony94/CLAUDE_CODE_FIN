"""
Execution Gap Quantification — Live vs Backtest Fill Analysis

Analyzes 57+ live trades from metrics.json to quantify:
1. Entry slippage (actual fill vs expected)
2. Exit slippage (fill vs TP/SL target)
3. Effective R:R in live vs backtest
4. Direction-specific slippage patterns
5. Exit reason distribution vs backtest expectations
6. Impact quantification on WR and PnL

Uses: results/pattern_5m_metrics.json (trade_history)
      results/pattern_5m_bot_state.json (current positions for additional context)
"""

import os
import sys
import json
import time
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))

METRICS_FILE = os.path.join(_PROJECT_ROOT, 'results', 'pattern_5m_metrics.json')
STATE_FILE = os.path.join(_PROJECT_ROOT, 'results', 'pattern_5m_bot_state.json')
PATTERNS_FILE = os.path.join(_PROJECT_ROOT, 'results', 'dynamic_patterns.json')
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'execution_gap_study.json')

LEVERAGE = 3
FEE_RATE = 0.0005  # 0.05% per side (BingX)


def load_data():
    with open(METRICS_FILE) as f:
        metrics = json.load(f)
    trades = metrics.get('trade_history', [])

    with open(PATTERNS_FILE) as f:
        pat_data = json.load(f)
    patterns_tpsl = {}
    for key, info in pat_data.get('pattern_details', {}).items():
        patterns_tpsl[info['pattern']] = {
            'tp': info['tp'], 'sl': info['sl'], 'direction': info['direction'],
        }

    state = {}
    try:
        with open(STATE_FILE) as f:
            state = json.load(f)
    except Exception:
        pass

    return trades, metrics, patterns_tpsl, state


def main():
    t0 = time.time()
    print("=" * 70)
    print("EXECUTION GAP QUANTIFICATION — Live Trade Analysis")
    print("=" * 70)

    trades, metrics, patterns_tpsl, state = load_data()
    print(f"Live trades: {len(trades)}")
    print(f"Metrics: total_trades={metrics.get('total_trades')}, "
          f"winning={metrics.get('winning_trades')}, "
          f"WR={metrics.get('actual_win_rate', 0):.1f}%")

    output = {'study': 'execution_gap', 'total_trades': len(trades)}

    # ── 1. OVERVIEW ────────────────────────────────────────
    print("\n" + "-" * 70)
    print("1. TRADE OVERVIEW")
    print("-" * 70)
    wins = [t for t in trades if t['pnl_slot'] > 0]
    losses = [t for t in trades if t['pnl_slot'] <= 0]
    pattern_trades = [t for t in trades if t['pattern'] != 'N/A']
    recovery_trades = [t for t in trades if t['pattern'] == 'N/A']

    print(f"  Total: {len(trades)} ({len(wins)}W / {len(losses)}L) = "
          f"{len(wins)/len(trades)*100:.1f}% WR")
    print(f"  Pattern trades: {len(pattern_trades)}")
    print(f"  Recovery (N/A): {len(recovery_trades)}")
    if pattern_trades:
        pw = sum(1 for t in pattern_trades if t['pnl_slot'] > 0)
        print(f"  Pattern WR: {pw}/{len(pattern_trades)} = "
              f"{pw/len(pattern_trades)*100:.1f}%")

    by_dir = defaultdict(list)
    for t in trades:
        by_dir[t['direction']].append(t)
    for d in ['LONG', 'SHORT']:
        dt = by_dir[d]
        if dt:
            dw = sum(1 for t in dt if t['pnl_slot'] > 0)
            print(f"  {d}: {len(dt)} trades ({dw}W/{len(dt)-dw}L) = "
                  f"{dw/len(dt)*100:.1f}% WR")

    # ── 2. EXIT PRICE vs TP/SL TARGET ──────────────────────
    print("\n" + "-" * 70)
    print("2. EXIT SLIPPAGE — Actual Fill vs TP/SL Target")
    print("-" * 70)

    tp_slippages = []
    sl_slippages = []
    for t in pattern_trades:
        entry = t['entry_price']
        exit_p = t['exit_price']
        tp = t['tp_price']
        sl = t['sl_price']
        reason = t['exit_reason']

        if reason == 'TP' and tp > 0:
            # For LONG TP: exit should be near tp (positive = better than expected)
            if t['direction'] == 'LONG':
                slip_bps = (exit_p - tp) / entry * 10000
            else:
                slip_bps = (tp - exit_p) / entry * 10000
            tp_slippages.append({
                'pattern': t['pattern'], 'direction': t['direction'],
                'entry': entry, 'tp_target': tp, 'exit': exit_p,
                'slip_bps': round(slip_bps, 2),
            })

        elif reason == 'SL' and sl > 0:
            # For LONG SL: exit should be near sl (negative = worse than expected)
            if t['direction'] == 'LONG':
                slip_bps = (exit_p - sl) / entry * 10000
            else:
                slip_bps = (sl - exit_p) / entry * 10000
            sl_slippages.append({
                'pattern': t['pattern'], 'direction': t['direction'],
                'entry': entry, 'sl_target': sl, 'exit': exit_p,
                'slip_bps': round(slip_bps, 2),
            })

    if tp_slippages:
        slips = [s['slip_bps'] for s in tp_slippages]
        print(f"\n  TP Fills ({len(tp_slippages)} trades):")
        print(f"    Mean slip: {np.mean(slips):+.1f} bps (positive = better than target)")
        print(f"    Median:    {np.median(slips):+.1f} bps")
        print(f"    Std:       {np.std(slips):.1f} bps")
        print(f"    Range:     [{min(slips):+.1f}, {max(slips):+.1f}] bps")
        # Worst fills
        worst = sorted(tp_slippages, key=lambda x: x['slip_bps'])[:3]
        print(f"    Worst TP fills:")
        for w in worst:
            print(f"      {w['pattern']} {w['direction']}: target ${w['tp_target']:.1f} "
                  f"→ filled ${w['exit']:.1f} ({w['slip_bps']:+.1f} bps)")

    if sl_slippages:
        slips = [s['slip_bps'] for s in sl_slippages]
        print(f"\n  SL Fills ({len(sl_slippages)} trades):")
        print(f"    Mean slip: {np.mean(slips):+.1f} bps (negative = worse than target)")
        print(f"    Median:    {np.median(slips):+.1f} bps")
        print(f"    Std:       {np.std(slips):.1f} bps")
        print(f"    Range:     [{min(slips):+.1f}, {max(slips):+.1f}] bps")
        worst = sorted(sl_slippages, key=lambda x: x['slip_bps'])[:3]
        print(f"    Worst SL fills:")
        for w in worst:
            print(f"      {w['pattern']} {w['direction']}: target ${w.get('sl_target', 0):.1f} "
                  f"→ filled ${w['exit']:.1f} ({w['slip_bps']:+.1f} bps)")

    output['tp_slippage'] = {
        'count': len(tp_slippages),
        'mean_bps': round(np.mean([s['slip_bps'] for s in tp_slippages]), 2) if tp_slippages else 0,
        'median_bps': round(np.median([s['slip_bps'] for s in tp_slippages]), 2) if tp_slippages else 0,
    }
    output['sl_slippage'] = {
        'count': len(sl_slippages),
        'mean_bps': round(np.mean([s['slip_bps'] for s in sl_slippages]), 2) if sl_slippages else 0,
        'median_bps': round(np.median([s['slip_bps'] for s in sl_slippages]), 2) if sl_slippages else 0,
    }

    # ── 3. EFFECTIVE R:R vs BACKTEST R:R ───────────────────
    print("\n" + "-" * 70)
    print("3. EFFECTIVE R:R — Live vs Backtest")
    print("-" * 70)

    live_wins_pct = []
    live_losses_pct = []
    bt_wins_pct = []
    bt_losses_pct = []

    for t in pattern_trades:
        entry = t['entry_price']
        exit_p = t['exit_price']
        tp = t['tp_price']
        sl = t['sl_price']

        if t['direction'] == 'LONG':
            actual_pct = (exit_p - entry) / entry * 100
            expected_tp_pct = (tp - entry) / entry * 100
            expected_sl_pct = (entry - sl) / entry * 100
        else:
            actual_pct = (entry - exit_p) / entry * 100
            expected_tp_pct = (entry - tp) / entry * 100
            expected_sl_pct = (sl - entry) / entry * 100

        if t['pnl_slot'] > 0:
            live_wins_pct.append(actual_pct)
            bt_wins_pct.append(expected_tp_pct)
        else:
            live_losses_pct.append(abs(actual_pct))
            bt_losses_pct.append(expected_sl_pct)

    if live_wins_pct and live_losses_pct:
        live_avg_win = np.mean(live_wins_pct)
        live_avg_loss = np.mean(live_losses_pct)
        live_rr = live_avg_win / live_avg_loss if live_avg_loss > 0 else 0

        bt_avg_win = np.mean(bt_wins_pct)
        bt_avg_loss = np.mean(bt_losses_pct)
        bt_rr = bt_avg_win / bt_avg_loss if bt_avg_loss > 0 else 0

        print(f"                  {'Live':>10} {'Backtest Expected':>18} {'Gap':>10}")
        print(f"  Avg Win (price): {live_avg_win:>9.2f}%  {bt_avg_win:>17.2f}%  "
              f"{live_avg_win - bt_avg_win:>+9.2f}%")
        print(f"  Avg Loss (price):{live_avg_loss:>9.2f}%  {bt_avg_loss:>17.2f}%  "
              f"{live_avg_loss - bt_avg_loss:>+9.2f}%")
        print(f"  R:R ratio:       {live_rr:>9.3f}   {bt_rr:>17.3f}   "
              f"{live_rr - bt_rr:>+9.3f}")

        live_be_wr = live_avg_loss / (live_avg_win + live_avg_loss) * 100
        bt_be_wr = bt_avg_loss / (bt_avg_win + bt_avg_loss) * 100
        print(f"  Breakeven WR:    {live_be_wr:>8.1f}%   {bt_be_wr:>16.1f}%   "
              f"{live_be_wr - bt_be_wr:>+8.1f}%")

        output['rr_comparison'] = {
            'live_avg_win': round(live_avg_win, 3), 'bt_avg_win': round(bt_avg_win, 3),
            'live_avg_loss': round(live_avg_loss, 3), 'bt_avg_loss': round(bt_avg_loss, 3),
            'live_rr': round(live_rr, 4), 'bt_rr': round(bt_rr, 4),
            'live_breakeven_wr': round(live_be_wr, 1), 'bt_breakeven_wr': round(bt_be_wr, 1),
        }

    # ── 4. EXIT REASON DISTRIBUTION ────────────────────────
    print("\n" + "-" * 70)
    print("4. EXIT REASON DISTRIBUTION")
    print("-" * 70)

    by_reason = defaultdict(list)
    for t in pattern_trades:
        by_reason[t['exit_reason']].append(t)

    reason_output = {}
    print(f"  {'Reason':<15} {'Count':>5} {'WR%':>6} {'AvgSlot':>8} {'AvgPort':>8}")
    for reason in sorted(by_reason.keys()):
        rt = by_reason[reason]
        w = sum(1 for t in rt if t['pnl_slot'] > 0)
        wr = w / len(rt) * 100 if rt else 0
        avg_slot = np.mean([t['pnl_slot'] for t in rt])
        avg_port = np.mean([t['pnl_portfolio'] for t in rt])
        print(f"  {reason:<15} {len(rt):>5} {wr:>6.1f} {avg_slot:>+8.2f} {avg_port:>+8.3f}")
        reason_output[reason] = {
            'count': len(rt), 'wr': round(wr, 1),
            'avg_slot_pnl': round(avg_slot, 3), 'avg_port_pnl': round(avg_port, 4),
        }
    output['exit_reasons'] = reason_output

    # ── 5. HOLD TIME ANALYSIS ──────────────────────────────
    print("\n" + "-" * 70)
    print("5. HOLD TIME — Winners vs Losers")
    print("-" * 70)

    win_holds = [t['hold_minutes'] for t in pattern_trades if t['pnl_slot'] > 0 and t.get('hold_minutes')]
    loss_holds = [t['hold_minutes'] for t in pattern_trades if t['pnl_slot'] <= 0 and t.get('hold_minutes')]

    if win_holds:
        print(f"  Winners:  mean {np.mean(win_holds):.0f}min, "
              f"median {np.median(win_holds):.0f}min, "
              f"range [{min(win_holds):.0f}, {max(win_holds):.0f}]min")
    if loss_holds:
        print(f"  Losers:   mean {np.mean(loss_holds):.0f}min, "
              f"median {np.median(loss_holds):.0f}min, "
              f"range [{min(loss_holds):.0f}, {max(loss_holds):.0f}]min")

    # Backtest expected: TP closer → win faster, SL farther → loss takes longer
    if win_holds and loss_holds:
        print(f"\n  Win/Loss hold ratio: {np.mean(win_holds)/np.mean(loss_holds):.2f}x")
        print(f"  (Backtest: wins fill faster when TP < SL distance, "
              f"which matches R:R < 1)")
    output['hold_times'] = {
        'win_mean_min': round(np.mean(win_holds), 1) if win_holds else 0,
        'loss_mean_min': round(np.mean(loss_holds), 1) if loss_holds else 0,
    }

    # ── 6. PATTERN-LEVEL ANALYSIS ──────────────────────────
    print("\n" + "-" * 70)
    print("6. PER-PATTERN LIVE vs EXPECTED")
    print("-" * 70)

    pat_perf = defaultdict(lambda: {'wins': 0, 'total': 0, 'pnls': []})
    for t in pattern_trades:
        key = t['pattern']
        pat_perf[key]['total'] += 1
        if t['pnl_slot'] > 0:
            pat_perf[key]['wins'] += 1
        pat_perf[key]['pnls'].append(t['pnl_slot'])
        pat_perf[key]['direction'] = t['direction']

    sorted_pats = sorted(pat_perf.items(), key=lambda x: x[1]['total'], reverse=True)
    print(f"  {'Pattern':<15} {'Dir':>5} {'Tr':>3} {'W':>2} {'LiveWR':>7} "
          f"{'ExpTP%':>7} {'ExpSL%':>7} {'AvgPnL':>8}")

    pat_analysis = []
    for pat, stats in sorted_pats:
        if stats['total'] < 2:
            continue
        wr = stats['wins'] / stats['total'] * 100
        avg_pnl = np.mean(stats['pnls'])
        bt = patterns_tpsl.get(pat, {})
        tp = bt.get('tp', 0)
        sl = bt.get('sl', 0)
        print(f"  {pat:<15} {stats.get('direction', '?'):>5} {stats['total']:>3} "
              f"{stats['wins']:>2} {wr:>7.1f} {tp:>7.2f} {sl:>7.2f} {avg_pnl:>+8.2f}")
        pat_analysis.append({
            'pattern': pat, 'direction': stats.get('direction', ''),
            'trades': stats['total'], 'wins': stats['wins'],
            'live_wr': round(wr, 1), 'expected_tp': tp, 'expected_sl': sl,
            'avg_pnl': round(avg_pnl, 3),
        })
    output['pattern_performance'] = pat_analysis

    # ── 7. CORRELATED LOSS ANALYSIS ────────────────────────
    print("\n" + "-" * 70)
    print("7. CORRELATED LOSS BURSTS (losses within 10min, same direction)")
    print("-" * 70)

    losses_sorted = sorted(
        [t for t in pattern_trades if t['pnl_slot'] < 0],
        key=lambda t: t['timestamp'])

    bursts = []
    current_burst = []
    for loss in losses_sorted:
        ts = datetime.fromisoformat(loss['timestamp'])
        if not current_burst:
            current_burst = [loss]
        else:
            prev_ts = datetime.fromisoformat(current_burst[-1]['timestamp'])
            if ((ts - prev_ts).total_seconds() < 600
                    and loss['direction'] == current_burst[0]['direction']):
                current_burst.append(loss)
            else:
                if len(current_burst) >= 2:
                    bursts.append(current_burst)
                current_burst = [loss]
    if len(current_burst) >= 2:
        bursts.append(current_burst)

    total_burst_trades = 0
    total_burst_pnl = 0
    for i, burst in enumerate(bursts):
        ts = burst[0]['timestamp'][:16]
        d = burst[0]['direction']
        pnl = sum(t['pnl_portfolio'] for t in burst)
        total_burst_trades += len(burst)
        total_burst_pnl += pnl
        print(f"  Burst #{i+1}: {ts} | {d} x{len(burst)} | "
              f"Portfolio PnL: {pnl:+.2f}%")
        for t in burst:
            print(f"    {t['pattern']}: ${t['entry_price']:.0f}→${t['exit_price']:.0f} "
                  f"| slot {t['pnl_slot']:+.1f}% | {t['exit_reason']}")

    excess_losses = sum(len(b) - 1 for b in bursts)
    print(f"\n  Total burst losses: {total_burst_trades} in {len(bursts)} bursts")
    print(f"  Excess losses (correlated): {excess_losses}")
    print(f"  Burst PnL impact: {total_burst_pnl:+.2f}%")
    output['correlated_bursts'] = {
        'count': len(bursts), 'total_trades': total_burst_trades,
        'excess_losses': excess_losses,
        'pnl_impact': round(total_burst_pnl, 3),
    }

    # ── 8. QUANTIFIED GAP DECOMPOSITION ───────────────────
    print("\n" + "-" * 70)
    print("8. WR GAP DECOMPOSITION")
    print("-" * 70)

    expected_wr = 88.8
    live_wr = len(wins) / len(trades) * 100 if trades else 0
    pattern_wr = (sum(1 for t in pattern_trades if t['pnl_slot'] > 0)
                  / len(pattern_trades) * 100 if pattern_trades else 0)

    gap_total = expected_wr - live_wr
    recovery_drag = 0
    if recovery_trades:
        rec_losses = sum(1 for t in recovery_trades if t['pnl_slot'] <= 0)
        recovery_drag = rec_losses / len(trades) * 100

    burst_drag = excess_losses / len(trades) * 100 if trades else 0

    # Execution: difference between actual PnL outcomes and backtest expectations
    tp_slip_cost = 0
    if tp_slippages:
        # Negative TP slip means worse fill → some TPs might have been missed
        tp_slip_cost = abs(output['tp_slippage']['mean_bps']) / 100 * LEVERAGE  # rough impact
    sl_slip_cost = 0
    if sl_slippages:
        sl_slip_cost = abs(output['sl_slippage']['mean_bps']) / 100 * LEVERAGE

    print(f"  Expected WR (WF OOS):    {expected_wr:.1f}%")
    print(f"  Live WR (all):           {live_wr:.1f}%")
    print(f"  Live WR (pattern only):  {pattern_wr:.1f}%")
    print(f"  Total gap:               {gap_total:+.1f}pp")
    print(f"\n  Gap decomposition:")
    print(f"    Recovery (N/A) drag:   ~{recovery_drag:.1f}pp")
    print(f"    Correlated burst drag: ~{burst_drag:.1f}pp")
    print(f"    TP slip (mean):        {output['tp_slippage']['mean_bps']:+.1f} bps")
    print(f"    SL slip (mean):        {output['sl_slippage']['mean_bps']:+.1f} bps")
    remaining = gap_total - recovery_drag - burst_drag
    print(f"    Remaining unexplained: ~{remaining:.1f}pp")
    print(f"    (= direction prediction + edge decay + market microstructure)")

    output['gap_decomposition'] = {
        'expected_wr': expected_wr, 'live_wr': round(live_wr, 1),
        'pattern_wr': round(pattern_wr, 1),
        'total_gap_pp': round(gap_total, 1),
        'recovery_drag_pp': round(recovery_drag, 1),
        'burst_drag_pp': round(burst_drag, 1),
        'remaining_unexplained_pp': round(remaining, 1),
    }

    # ── 9. TIMELINE ────────────────────────────────────────
    print("\n" + "-" * 70)
    print("9. DAILY PERFORMANCE TIMELINE")
    print("-" * 70)

    by_date = defaultdict(list)
    for t in trades:
        date = t['timestamp'][:10]
        by_date[date].append(t)

    daily_output = []
    print(f"  {'Date':<12} {'Tr':>3} {'W':>2} {'WR%':>6} {'PnL%':>7} {'L':>2} {'S':>2}")
    for date in sorted(by_date.keys()):
        dt = by_date[date]
        w = sum(1 for t in dt if t['pnl_slot'] > 0)
        wr = w / len(dt) * 100
        pnl = sum(t['pnl_portfolio'] for t in dt)
        ln = sum(1 for t in dt if t['direction'] == 'LONG')
        sn = len(dt) - ln
        print(f"  {date:<12} {len(dt):>3} {w:>2} {wr:>6.1f} {pnl:>+7.2f} {ln:>2} {sn:>2}")
        daily_output.append({'date': date, 'trades': len(dt), 'wins': w,
                             'wr': round(wr, 1), 'pnl': round(pnl, 3)})
    output['daily'] = daily_output

    # ── SUMMARY ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Live trades: {len(trades)} ({len(pattern_trades)} pattern + {len(recovery_trades)} recovery)")
    print(f"  WR: {live_wr:.1f}% (pattern {pattern_wr:.1f}%)")
    if tp_slippages:
        print(f"  TP slippage: {output['tp_slippage']['mean_bps']:+.1f} bps mean")
    if sl_slippages:
        print(f"  SL slippage: {output['sl_slippage']['mean_bps']:+.1f} bps mean")
    if live_wins_pct and live_losses_pct:
        print(f"  Live R:R: {live_rr:.3f} (backtest {bt_rr:.3f})")
    print(f"  Correlated bursts: {len(bursts)} ({total_burst_trades} trades, {total_burst_pnl:+.2f}%)")
    print(f"  Unexplained WR gap: ~{remaining:.1f}pp")
    print("=" * 70)

    elapsed = time.time() - t0
    output['elapsed_seconds'] = round(elapsed, 1)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {OUTPUT_FILE} ({elapsed:.1f}s)")


if __name__ == '__main__':
    main()
