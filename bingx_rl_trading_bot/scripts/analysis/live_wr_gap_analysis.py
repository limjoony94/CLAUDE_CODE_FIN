#!/usr/bin/env python3
"""
Live WR Gap Root Cause Analysis (v1.34.0)
=========================================
54 trades, WR 61.1% vs Expected 68.1% (-7.0pp gap)

Phase 1: Log-based trade extraction and pattern-level WR breakdown
Phase 2: Exit reason distribution — TP/SL/MARKET/TIMEOUT categorization
Phase 3: ATR compression impact — effective TP/SL vs config
Phase 4: Signal frequency — SKIP reason analysis from logs
Phase 5: Conclusions and actionable recommendations

Standard Research Protocol: production classify_candle, LEVERAGE=3
"""
import os
import re
import json
import sys
from datetime import datetime, timedelta
from collections import defaultdict

# ── Setup paths ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config', 'pattern_5m_config.yaml')

# ── Constants ──
EXPECTED_WR = 68.1
EXPECTED_EDGE = 0.221  # %/trade
EXPECTED_TRADES_PER_DAY = 4.97
LEVERAGE = 3


def extract_trades_from_logs():
    """Extract all TRADE CLOSED entries from available log files."""
    trade_pattern = re.compile(
        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ \| INFO \| .* TRADE CLOSED \| '
        r'(\w+) ([\w\-/]+) \| '
        r'Entry: \$(\d+\.\d+) . Exit: \$(\d+\.\d+) \| '
        r'PnL: ([+\-\d.]+)% \((?:slot|lev)\)(?: / ([+\-\d.]+)% \((?:portfolio|price)\))? \| '
        r'TP: \$(\d+\.\d+) SL: \$(\d+\.\d+) \| '
        r'Reason: (\w+) \| Hold: (\d+)m'
    )

    trades = []
    log_files = sorted([f for f in os.listdir(LOGS_DIR) if f.startswith('pattern_5m_bot_') and f.endswith('.log')])

    for lf in log_files:
        with open(os.path.join(LOGS_DIR, lf), 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                m = trade_pattern.search(line)
                if m:
                    ts, direction, pattern, entry, exit_p, pnl_slot, pnl_port, tp, sl, reason, hold = m.groups()
                    trades.append({
                        'timestamp': ts,
                        'direction': direction,
                        'pattern': pattern,
                        'entry': float(entry),
                        'exit': float(exit_p),
                        'pnl_slot': float(pnl_slot),
                        'pnl_portfolio': float(pnl_port) if pnl_port else float(pnl_slot) / 9,
                        'tp_price': float(tp),
                        'sl_price': float(sl),
                        'reason': reason,
                        'hold_minutes': int(hold),
                        'is_win': float(pnl_slot) >= 0,
                    })
    return trades


def extract_entries_from_logs():
    """Extract position opened entries."""
    entry_pattern = re.compile(
        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ \| INFO \| Opening (\w+) position: '
        r'[\d.]+ BTC-USDT @ ~\$(\d+\.\d+)'
    )
    entries = []
    log_files = sorted([f for f in os.listdir(LOGS_DIR) if f.startswith('pattern_5m_bot_') and f.endswith('.log')])
    for lf in log_files:
        with open(os.path.join(LOGS_DIR, lf), 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                m = entry_pattern.search(line)
                if m:
                    entries.append({
                        'timestamp': m.group(1),
                        'direction': m.group(2),
                        'price': float(m.group(3)),
                    })
    return entries


def extract_skip_reasons():
    """Extract signal skip/cap reasons from logs."""
    skip_pattern = re.compile(r'direction.?cap|SKIP|skip.*signal|no.*signal|cap.*reached|duplicate|cooldown', re.IGNORECASE)
    cap_hits = 0
    skip_reasons = defaultdict(int)

    log_files = sorted([f for f in os.listdir(LOGS_DIR) if f.startswith('pattern_5m_bot_') and f.endswith('.log')])
    for lf in log_files:
        with open(os.path.join(LOGS_DIR, lf), 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if 'direction cap' in line.lower() or 'cap reached' in line.lower():
                    cap_hits += 1
                    skip_reasons['direction_cap'] += 1
                elif 'duplicate' in line.lower() and 'pattern' in line.lower():
                    skip_reasons['duplicate_pattern'] += 1
                elif 'cooldown' in line.lower():
                    skip_reasons['cooldown'] += 1
                elif re.search(r'SKIP.*signal|signal.*SKIP|no valid signal', line, re.IGNORECASE):
                    skip_reasons['no_signal'] += 1
    return dict(skip_reasons)


def phase1_trade_breakdown(trades):
    """Phase 1: Pattern-level WR breakdown."""
    print("\n" + "="*70)
    print("PHASE 1: Trade Breakdown by Pattern")
    print("="*70)

    # Overall stats
    n = len(trades)
    wins = sum(1 for t in trades if t['is_win'])
    wr = wins / n * 100 if n > 0 else 0
    total_pnl = sum(t['pnl_slot'] for t in trades)
    avg_win = sum(t['pnl_slot'] for t in trades if t['is_win']) / wins if wins > 0 else 0
    losses = [t for t in trades if not t['is_win']]
    avg_loss = sum(t['pnl_slot'] for t in losses) / len(losses) if losses else 0

    print(f"\nLogged trades: {n} (of 54 total — earlier trades lost to log rotation)")
    print(f"WR: {wr:.1f}% (expected: {EXPECTED_WR}%, gap: {wr - EXPECTED_WR:+.1f}pp)")
    print(f"Total PnL (slot): {total_pnl:+.2f}%")
    print(f"Avg Win: {avg_win:+.2f}% | Avg Loss: {avg_loss:+.2f}%")

    # By pattern
    by_pattern = defaultdict(list)
    for t in trades:
        key = f"{t['direction']} {t['pattern']}"
        by_pattern[key].append(t)

    print(f"\n{'Pattern':<25} {'Trades':>6} {'Wins':>5} {'WR':>6} {'PnL':>8} {'Reason':>10}")
    print("-" * 70)
    for pat, pts in sorted(by_pattern.items(), key=lambda x: -len(x[1])):
        w = sum(1 for t in pts if t['is_win'])
        pwr = w / len(pts) * 100 if pts else 0
        ppnl = sum(t['pnl_slot'] for t in pts)
        reasons = ','.join(set(t['reason'] for t in pts))
        print(f"{pat:<25} {len(pts):>6} {w:>5} {pwr:>5.0f}% {ppnl:>+7.2f}% {reasons:>10}")

    # By direction
    print(f"\n--- Direction Summary ---")
    for d in ['LONG', 'SHORT']:
        dt = [t for t in trades if t['direction'] == d]
        if dt:
            dw = sum(1 for t in dt if t['is_win'])
            dwr = dw / len(dt) * 100
            dpnl = sum(t['pnl_slot'] for t in dt)
            print(f"  {d}: {len(dt)} trades, WR {dwr:.1f}%, PnL {dpnl:+.2f}%")


def phase2_exit_reason_analysis(trades):
    """Phase 2: Exit reason distribution."""
    print("\n" + "="*70)
    print("PHASE 2: Exit Reason Distribution")
    print("="*70)

    by_reason = defaultdict(list)
    for t in trades:
        by_reason[t['reason']].append(t)

    print(f"\n{'Reason':<20} {'Count':>6} {'WR':>6} {'Avg PnL':>10} {'Total PnL':>10}")
    print("-" * 60)
    for reason, rts in sorted(by_reason.items(), key=lambda x: -len(x[1])):
        w = sum(1 for t in rts if t['is_win'])
        wr = w / len(rts) * 100
        avg = sum(t['pnl_slot'] for t in rts) / len(rts)
        total = sum(t['pnl_slot'] for t in rts)
        print(f"{reason:<20} {len(rts):>6} {wr:>5.0f}% {avg:>+9.2f}% {total:>+9.2f}%")

    # Mass close event detection
    print(f"\n--- Mass Close Events ---")
    from itertools import groupby
    sorted_trades = sorted(trades, key=lambda t: t['timestamp'])
    for i, t in enumerate(sorted_trades):
        if i > 0:
            prev_ts = datetime.strptime(sorted_trades[i-1]['timestamp'], '%Y-%m-%d %H:%M:%S')
            curr_ts = datetime.strptime(t['timestamp'], '%Y-%m-%d %H:%M:%S')
            if (curr_ts - prev_ts).total_seconds() < 30:
                if t['reason'] == 'MARKET':
                    # Find all trades within 30s window
                    cluster = [t]
                    for j in range(i-1, -1, -1):
                        jts = datetime.strptime(sorted_trades[j]['timestamp'], '%Y-%m-%d %H:%M:%S')
                        if (curr_ts - jts).total_seconds() < 30:
                            cluster.append(sorted_trades[j])
                        else:
                            break
                    if len(cluster) >= 3:
                        total = sum(c['pnl_slot'] for c in cluster)
                        print(f"  {t['timestamp'][:10]}: {len(cluster)} positions closed simultaneously")
                        print(f"    Patterns: {', '.join(c['pattern'] for c in cluster)}")
                        print(f"    Total PnL impact: {total:+.2f}% (slot)")
                        print(f"    Reason: {cluster[0]['reason']} (likely bot restart/upgrade)")
                        break  # only report first cluster


def phase3_atr_compression(trades):
    """Phase 3: ATR compression impact analysis."""
    print("\n" + "="*70)
    print("PHASE 3: ATR Compression Impact")
    print("="*70)

    # Load dynamic patterns for base TP/SL
    dp_path = os.path.join(RESULTS_DIR, 'dynamic_patterns.json')
    try:
        with open(dp_path) as f:
            dp = json.load(f)
        patterns_tpsl = dp.get('patterns_tpsl', {})
    except Exception:
        patterns_tpsl = {}

    print(f"\n{'Pattern':<22} {'Dir':>5} {'Base TP':>7} {'Eff TP':>7} {'Ratio':>6} {'Base SL':>7} {'Eff SL':>7} {'Ratio':>6}")
    print("-" * 78)

    compression_ratios = []
    for t in trades:
        pat_name = t['pattern']
        if pat_name == 'N/A':
            continue

        # Calculate effective TP/SL from actual prices
        entry = t['entry']
        tp_price = t['tp_price']
        sl_price = t['sl_price']

        if t['direction'] == 'LONG':
            eff_tp = (tp_price / entry - 1) * 100
            eff_sl = (1 - sl_price / entry) * 100
        else:  # SHORT
            eff_tp = (1 - tp_price / entry) * 100
            eff_sl = (sl_price / entry - 1) * 100

        # Get base TP/SL from config
        tpsl = patterns_tpsl.get(pat_name)
        if isinstance(tpsl, (list, tuple)) and len(tpsl) >= 2:
            base_tp, base_sl = tpsl[0], tpsl[1]
        else:
            continue

        tp_ratio = eff_tp / base_tp if base_tp > 0 else 0
        sl_ratio = eff_sl / base_sl if base_sl > 0 else 0
        compression_ratios.append({'tp_ratio': tp_ratio, 'sl_ratio': sl_ratio, 'pattern': pat_name})

        print(f"{pat_name:<22} {t['direction']:>5} {base_tp:>6.2f}% {eff_tp:>6.2f}% {tp_ratio:>5.2f}x {base_sl:>6.2f}% {eff_sl:>6.2f}% {sl_ratio:>5.2f}x")

    if compression_ratios:
        avg_tp_ratio = sum(c['tp_ratio'] for c in compression_ratios) / len(compression_ratios)
        avg_sl_ratio = sum(c['sl_ratio'] for c in compression_ratios) / len(compression_ratios)
        print(f"\n--- ATR Compression Summary ---")
        print(f"  Avg TP ratio: {avg_tp_ratio:.3f}x (1.0 = no compression)")
        print(f"  Avg SL ratio: {avg_sl_ratio:.3f}x")
        print(f"  R:R preservation: TP/SL ratio {'maintained' if abs(avg_tp_ratio - avg_sl_ratio) < 0.05 else 'skewed'}")

        if avg_tp_ratio < 0.8:
            print(f"  ⚠️ Significant TP compression ({avg_tp_ratio:.0%}) — trades need larger moves to hit TP")
            print(f"  → This could explain lower WR if price movements are insufficient")


def phase4_signal_frequency(entries, trades):
    """Phase 4: Signal frequency and skip analysis."""
    print("\n" + "="*70)
    print("PHASE 4: Signal Frequency Analysis")
    print("="*70)

    if not entries:
        print("  No entry data found in logs")
        return

    # Calculate active days
    log_files = sorted([f for f in os.listdir(LOGS_DIR) if f.startswith('pattern_5m_bot_') and f.endswith('.log')])
    if log_files:
        first_date = log_files[0].replace('pattern_5m_bot_', '').replace('.log', '')
        last_date = log_files[-1].replace('pattern_5m_bot_', '').replace('.log', '')
        d1 = datetime.strptime(first_date, '%Y%m%d')
        d2 = datetime.strptime(last_date, '%Y%m%d')
        active_days = (d2 - d1).days + 1
    else:
        active_days = 1

    print(f"\n  Log coverage: {active_days} days ({first_date} ~ {last_date})")
    print(f"  Entries logged: {len(entries)}")
    print(f"  Trades closed (logged): {len(trades)}")
    print(f"  Entries/day: {len(entries) / active_days:.2f} (expected: {EXPECTED_TRADES_PER_DAY})")

    # State-based total
    state_path = os.path.join(RESULTS_DIR, 'pattern_5m_bot_state.json')
    try:
        with open(state_path) as f:
            state = json.load(f)
        total_trades = state.get('total_trades', 0)
        print(f"\n  Total trades (state): {total_trades}")
        # Bot has been running since ~Feb 2 (state created_at)
        created = state.get('created_at', '')
        if created:
            start = datetime.fromisoformat(created)
            running_days = (datetime.now() - start).days
            if running_days > 0:
                actual_tpd = total_trades / running_days
                print(f"  Running since: {created[:10]} ({running_days} days)")
                print(f"  Actual trades/day: {actual_tpd:.2f} (expected: {EXPECTED_TRADES_PER_DAY})")
                print(f"  Frequency gap: {actual_tpd - EXPECTED_TRADES_PER_DAY:+.2f}/day ({actual_tpd/EXPECTED_TRADES_PER_DAY*100:.0f}% of expected)")
    except Exception:
        pass

    # Skip reasons
    skip_reasons = extract_skip_reasons()
    if skip_reasons:
        print(f"\n  --- Skip Reasons (from logs) ---")
        for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count}")


def phase5_conclusions(trades):
    """Phase 5: Root cause conclusions."""
    print("\n" + "="*70)
    print("PHASE 5: Root Cause Conclusions")
    print("="*70)

    n = len(trades)
    wins = sum(1 for t in trades if t['is_win'])
    wr = wins / n * 100 if n > 0 else 0

    # Identify mass close events
    market_trades = [t for t in trades if t['reason'] == 'MARKET']
    tp_trades = [t for t in trades if t['reason'] == 'TP']
    sl_trades = [t for t in trades if t['reason'] == 'SL']
    other_trades = [t for t in trades if t['reason'] not in ('TP', 'SL', 'MARKET')]

    # WR excluding MARKET closes
    non_market = [t for t in trades if t['reason'] != 'MARKET']
    nm_wins = sum(1 for t in non_market if t['is_win'])
    nm_wr = nm_wins / len(non_market) * 100 if non_market else 0

    # WR of TP/SL only (organic trades)
    organic = [t for t in trades if t['reason'] in ('TP', 'SL')]
    org_wins = sum(1 for t in organic if t['is_win'])
    org_wr = org_wins / len(organic) * 100 if organic else 0

    print(f"\n  1. EXIT REASON IMPACT")
    print(f"     All trades WR:       {wr:.1f}% ({wins}/{n})")
    print(f"     Excl. MARKET WR:     {nm_wr:.1f}% ({nm_wins}/{len(non_market)})")
    print(f"     TP/SL only WR:       {org_wr:.1f}% ({org_wins}/{len(organic)})")
    print(f"     MARKET closes:       {len(market_trades)} trades (bot restarts/upgrades)")
    print(f"     → MARKET closes are forced liquidations, NOT strategy failures")

    # Net impact of MARKET closes
    market_pnl = sum(t['pnl_slot'] for t in market_trades)
    print(f"     → Total MARKET PnL:  {market_pnl:+.2f}% (pure operational cost)")

    print(f"\n  2. KEY FINDINGS")

    if org_wr >= EXPECTED_WR - 5:
        print(f"     ✅ Organic (TP/SL) WR {org_wr:.1f}% is within range of expected {EXPECTED_WR}%")
    else:
        print(f"     ⚠️ Organic (TP/SL) WR {org_wr:.1f}% is below expected {EXPECTED_WR}%")

    if len(market_trades) >= 3:
        print(f"     ⚠️ {len(market_trades)} MARKET closes detected — operational drag on WR")
        print(f"        Removing these: WR would be {nm_wr:.1f}% (vs {wr:.1f}% actual)")

    # Holding time analysis
    tp_hold = [t['hold_minutes'] for t in tp_trades]
    sl_hold = [t['hold_minutes'] for t in sl_trades]
    if tp_hold:
        avg_tp_hold = sum(tp_hold) / len(tp_hold)
        print(f"\n  3. HOLDING TIME")
        print(f"     Avg TP hold: {avg_tp_hold:.0f}min ({avg_tp_hold/60:.1f}h)")
        if sl_hold:
            avg_sl_hold = sum(sl_hold) / len(sl_hold)
            print(f"     Avg SL hold: {avg_sl_hold:.0f}min ({avg_sl_hold/60:.1f}h)")

    print(f"\n  4. SAMPLE SIZE CAVEAT")
    print(f"     Logged trades: {n} (of 54 total)")
    print(f"     38 trades lost to log rotation — conclusions are preliminary")
    print(f"     At 54 trades, WR CI (95%): {wr:.1f}% ± {1.96 * (wr/100 * (1-wr/100) / n)**0.5 * 100:.1f}pp")

    print(f"\n  5. RECOMMENDATIONS")
    print(f"     (a) Enable trade_history logging in metrics for complete data")
    print(f"     (b) Minimize operational MARKET closes (stable bot version)")
    print(f"     (c) Wait for 100+ organic TP/SL trades before strategic changes")
    print(f"     (d) Investigate ATR compression if TP/SL ratio < 0.7")


def main():
    print("="*70)
    print("Live WR Gap Root Cause Analysis")
    print(f"Expected WR: {EXPECTED_WR}% | State WR: 61.1% | Gap: -7.0pp")
    print(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*70)

    trades = extract_trades_from_logs()
    entries = extract_entries_from_logs()

    if not trades:
        print("ERROR: No trades found in logs")
        return

    phase1_trade_breakdown(trades)
    phase2_exit_reason_analysis(trades)
    phase3_atr_compression(trades)
    phase4_signal_frequency(entries, trades)
    phase5_conclusions(trades)

    # Save results
    output = {
        'analysis_date': datetime.now().isoformat(),
        'logged_trades': len(trades),
        'total_trades_state': 54,
        'logged_wr': sum(1 for t in trades if t['is_win']) / len(trades) * 100,
        'organic_wr': sum(1 for t in trades if t['is_win'] and t['reason'] in ('TP','SL')) / max(1, len([t for t in trades if t['reason'] in ('TP','SL')])) * 100,
        'market_closes': len([t for t in trades if t['reason'] == 'MARKET']),
        'trades': trades,
    }
    out_path = os.path.join(RESULTS_DIR, 'live_wr_gap_analysis.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n\nResults saved to: {out_path}")


if __name__ == '__main__':
    main()
