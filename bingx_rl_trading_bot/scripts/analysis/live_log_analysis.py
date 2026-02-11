"""
Live Bot Log Analysis — Parse all trade history from logs
Matches signal entries with position closes to build complete trade record.
"""
import re
import os
import json
from datetime import datetime
from collections import defaultdict

LOG_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'logs')

def parse_logs():
    """Parse all pattern_5m_bot logs and extract trade records."""

    signals = []  # (timestamp, direction, pattern, entry_price)
    closes = []   # (timestamp, reason, exit_price, pnl)
    metrics = []  # (timestamp, total_trades, wr)

    # Pattern for signal detection
    sig_re = re.compile(
        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ \| INFO \| .*Signal detected: (LONG|SHORT) \| Pattern: (.+?)(?:\s|$)'
    )
    # Pattern for opening
    open_re = re.compile(
        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ \| INFO \| Opening (LONG|SHORT) position: .+ @ ~\$([0-9.]+)'
    )
    # Pattern for old-style close
    close_old_re = re.compile(
        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ \| INFO \| Position closed: (TP|SL|TIMEOUT|UNKNOWN|EARLY_EXIT) \| Exit: \$([0-9.]+) \| PnL: ([+-]?[0-9.]+)%'
    )
    # Pattern for new-style close (v1.27.0+)
    close_new_re = re.compile(
        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ \| INFO \| .+TRADE CLOSED \| (LONG|SHORT) (.+?) \| Entry: \$([0-9.]+) .+ Exit: \$([0-9.]+) \| PnL: ([+-]?[0-9.]+)%.+Reason: (\w+) \| Hold: (\d+)m'
    )
    # Pattern for metrics saved
    metrics_re = re.compile(
        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ \| INFO \| .+Metrics saved: (\d+) trades, ([0-9.]+)% WR'
    )

    log_files = sorted([
        f for f in os.listdir(LOG_DIR)
        if f.startswith('pattern_5m_bot_') and f.endswith('.log')
    ])

    seen_closes = set()  # deduplicate

    for lf in log_files:
        path = os.path.join(LOG_DIR, lf)
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # Signal
                m = sig_re.search(line)
                if m:
                    ts = datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')
                    signals.append({
                        'ts': ts, 'direction': m.group(2),
                        'pattern': m.group(3).strip().rstrip(')')
                    })
                    continue

                # Opening (get entry price)
                m = open_re.search(line)
                if m:
                    ts = datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')
                    price = float(m.group(3))
                    # Attach to most recent signal
                    if signals and abs((ts - signals[-1]['ts']).total_seconds()) < 10:
                        signals[-1]['entry_price'] = price
                    continue

                # New-style close
                m = close_new_re.search(line)
                if m:
                    ts = datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')
                    key = f"{m.group(1)}_{m.group(6)}"
                    if key not in seen_closes:
                        seen_closes.add(key)
                        closes.append({
                            'ts': ts, 'direction': m.group(2),
                            'pattern': m.group(3).strip(),
                            'entry_price': float(m.group(4)),
                            'exit_price': float(m.group(5)),
                            'pnl': float(m.group(6)),
                            'reason': m.group(7),
                            'hold_min': int(m.group(8))
                        })
                    continue

                # Old-style close
                m = close_old_re.search(line)
                if m:
                    ts = datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')
                    key = f"{m.group(1)}_{m.group(4)}"
                    if key not in seen_closes:
                        seen_closes.add(key)
                        closes.append({
                            'ts': ts, 'reason': m.group(2),
                            'exit_price': float(m.group(3)),
                            'pnl': float(m.group(4))
                        })
                    continue

                # Metrics
                m = metrics_re.search(line)
                if m:
                    ts = datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')
                    metrics.append({
                        'ts': ts,
                        'total': int(m.group(2)),
                        'wr': float(m.group(3))
                    })

    return signals, closes, metrics


def match_trades(signals, closes):
    """Match signal entries with closes to build complete trade records."""
    trades = []
    sig_idx = 0

    for close in closes:
        # For new-style closes, pattern info is already in close
        if 'pattern' in close and 'direction' in close:
            trades.append(close)
            continue

        # For old-style closes, find the matching signal
        best_sig = None
        best_dt = None
        for i, sig in enumerate(signals):
            if sig['ts'] < close['ts']:
                dt = (close['ts'] - sig['ts']).total_seconds()
                if best_dt is None or dt < best_dt:
                    # But make sure this signal hasn't been used
                    if not sig.get('used'):
                        best_sig = i
                        best_dt = dt

        trade = dict(close)
        if best_sig is not None:
            sig = signals[best_sig]
            trade['direction'] = sig['direction']
            trade['pattern'] = sig['pattern']
            trade['entry_price'] = sig.get('entry_price')
            trade['signal_ts'] = sig['ts']
            if trade['entry_price'] and trade.get('exit_price'):
                if trade['entry_price'] > 0:
                    dt_sec = (close['ts'] - sig['ts']).total_seconds()
                    trade['hold_min'] = int(dt_sec / 60)
            signals[best_sig]['used'] = True

        trades.append(trade)

    return trades


def analyze(trades, metrics):
    """Comprehensive analysis."""
    results = {}

    # ── 1. Daily Summary ──
    daily = defaultdict(lambda: {'trades': 0, 'wins': 0, 'losses': 0, 'pnl': 0.0, 'details': []})
    for t in trades:
        date = t['ts'].strftime('%m-%d')
        daily[date]['trades'] += 1
        daily[date]['pnl'] += t['pnl']
        if t['pnl'] > 0:
            daily[date]['wins'] += 1
        else:
            daily[date]['losses'] += 1
        daily[date]['details'].append({
            'pattern': t.get('pattern', '?'),
            'direction': t.get('direction', '?'),
            'pnl': t['pnl'],
            'reason': t.get('reason', '?'),
            'hold': t.get('hold_min', '?')
        })

    results['daily'] = {k: {
        'trades': v['trades'], 'wins': v['wins'], 'losses': v['losses'],
        'pnl': round(v['pnl'], 2),
        'wr': round(v['wins'] / v['trades'] * 100, 1) if v['trades'] > 0 else 0,
        'details': v['details']
    } for k, v in sorted(daily.items())}

    # ── 2. Pattern Performance ──
    pattern_stats = defaultdict(lambda: {'trades': 0, 'wins': 0, 'losses': 0, 'pnl': 0.0, 'pnls': []})
    for t in trades:
        pat = t.get('pattern', 'UNKNOWN')
        direction = t.get('direction', '?')
        key = f"{pat} ({direction[0]})" if direction != '?' else pat
        pattern_stats[key]['trades'] += 1
        pattern_stats[key]['pnl'] += t['pnl']
        pattern_stats[key]['pnls'].append(t['pnl'])
        if t['pnl'] > 0:
            pattern_stats[key]['wins'] += 1
        else:
            pattern_stats[key]['losses'] += 1

    results['patterns'] = {k: {
        'trades': v['trades'],
        'wins': v['wins'],
        'losses': v['losses'],
        'wr': round(v['wins'] / v['trades'] * 100, 1),
        'total_pnl': round(v['pnl'], 2),
        'avg_pnl': round(v['pnl'] / v['trades'], 2),
        'best': round(max(v['pnls']), 2),
        'worst': round(min(v['pnls']), 2)
    } for k, v in sorted(pattern_stats.items(), key=lambda x: -x[1]['pnl'])}

    # ── 3. Win/Loss Analysis ──
    wins = [t['pnl'] for t in trades if t['pnl'] > 0]
    losses = [t['pnl'] for t in trades if t['pnl'] <= 0]

    results['win_loss'] = {
        'total_trades': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'wr': round(len(wins) / len(trades) * 100, 1),
        'total_pnl': round(sum(t['pnl'] for t in trades), 2),
        'avg_win': round(sum(wins) / len(wins), 2) if wins else 0,
        'avg_loss': round(sum(losses) / len(losses), 2) if losses else 0,
        'best_trade': round(max(t['pnl'] for t in trades), 2),
        'worst_trade': round(min(t['pnl'] for t in trades), 2),
        'max_consecutive_wins': max_consecutive(trades, True),
        'max_consecutive_losses': max_consecutive(trades, False),
    }

    # Win/Loss by reason
    reason_stats = defaultdict(lambda: {'count': 0, 'pnl': 0.0})
    for t in trades:
        r = t.get('reason', '?')
        reason_stats[r]['count'] += 1
        reason_stats[r]['pnl'] += t['pnl']
    results['by_reason'] = {k: {'count': v['count'], 'pnl': round(v['pnl'], 2)}
                            for k, v in reason_stats.items()}

    # ── 4. Direction Analysis ──
    long_trades = [t for t in trades if t.get('direction') == 'LONG']
    short_trades = [t for t in trades if t.get('direction') == 'SHORT']

    results['by_direction'] = {
        'LONG': {
            'trades': len(long_trades),
            'wins': sum(1 for t in long_trades if t['pnl'] > 0),
            'wr': round(sum(1 for t in long_trades if t['pnl'] > 0) / len(long_trades) * 100, 1) if long_trades else 0,
            'pnl': round(sum(t['pnl'] for t in long_trades), 2)
        },
        'SHORT': {
            'trades': len(short_trades),
            'wins': sum(1 for t in short_trades if t['pnl'] > 0),
            'wr': round(sum(1 for t in short_trades if t['pnl'] > 0) / len(short_trades) * 100, 1) if short_trades else 0,
            'pnl': round(sum(t['pnl'] for t in short_trades), 2)
        }
    }

    # ── 5. Equity Curve (cumulative PnL) ──
    cum_pnl = 0
    equity = []
    for i, t in enumerate(trades):
        cum_pnl += t['pnl']
        equity.append({
            'trade': i + 1,
            'date': t['ts'].strftime('%m-%d %H:%M'),
            'pnl': round(t['pnl'], 2),
            'cum_pnl': round(cum_pnl, 2),
            'pattern': t.get('pattern', '?'),
            'direction': t.get('direction', '?')[0] if t.get('direction') else '?',
            'reason': t.get('reason', '?')
        })
    results['equity_curve'] = equity

    # ── 6. MDD from equity curve ──
    peak = 0
    mdd = 0
    for e in equity:
        if e['cum_pnl'] > peak:
            peak = e['cum_pnl']
        dd = peak - e['cum_pnl']
        if dd > mdd:
            mdd = dd
    results['mdd'] = round(mdd, 2)

    # ── 7. Metrics progression (detect resets) ──
    resets = []
    for i in range(1, len(metrics)):
        if metrics[i]['total'] < metrics[i-1]['total']:
            resets.append({
                'ts': metrics[i]['ts'].strftime('%Y-%m-%d %H:%M'),
                'before': metrics[i-1]['total'],
                'after': metrics[i]['total']
            })
    results['metrics_resets'] = resets

    # ── 8. Hourly distribution ──
    hour_dist = defaultdict(lambda: {'trades': 0, 'wins': 0, 'pnl': 0.0})
    for t in trades:
        h = t['ts'].hour
        hour_dist[h]['trades'] += 1
        hour_dist[h]['pnl'] += t['pnl']
        if t['pnl'] > 0:
            hour_dist[h]['wins'] += 1
    results['hourly'] = {k: {
        'trades': v['trades'],
        'wr': round(v['wins']/v['trades']*100, 1) if v['trades'] > 0 else 0,
        'pnl': round(v['pnl'], 2)
    } for k, v in sorted(hour_dist.items())}

    return results


def max_consecutive(trades, is_win):
    max_c = 0
    cur = 0
    for t in trades:
        if (t['pnl'] > 0) == is_win:
            cur += 1
            max_c = max(max_c, cur)
        else:
            cur = 0
    return max_c


def print_report(results):
    """Print formatted report."""

    print("=" * 70)
    print("  LIVE BOT LOG ANALYSIS — Pattern 5m v1.27.0")
    print("=" * 70)

    # Win/Loss summary
    wl = results['win_loss']
    print(f"\n{'─'*70}")
    print(f"  OVERALL: {wl['total_trades']} trades | {wl['wins']}W/{wl['losses']}L | "
          f"WR {wl['wr']}% | PnL {wl['total_pnl']:+.2f}%")
    print(f"  Avg Win: {wl['avg_win']:+.2f}% | Avg Loss: {wl['avg_loss']:+.2f}% | "
          f"MDD: {results['mdd']:.2f}%")
    print(f"  Best: {wl['best_trade']:+.2f}% | Worst: {wl['worst_trade']:+.2f}%")
    print(f"  Max Consec Win: {wl['max_consecutive_wins']} | "
          f"Max Consec Loss: {wl['max_consecutive_losses']}")

    # Direction
    print(f"\n{'─'*70}")
    print("  BY DIRECTION")
    print(f"{'─'*70}")
    for d, v in results['by_direction'].items():
        print(f"  {d:6s}: {v['trades']:2d} trades | {v['wins']}W | "
              f"WR {v['wr']}% | PnL {v['pnl']:+.2f}%")

    # By reason
    print(f"\n{'─'*70}")
    print("  BY EXIT REASON")
    print(f"{'─'*70}")
    for r, v in sorted(results['by_reason'].items(), key=lambda x: -x[1]['count']):
        print(f"  {r:10s}: {v['count']:2d} trades | PnL {v['pnl']:+.2f}%")

    # Daily summary
    print(f"\n{'─'*70}")
    print("  DAILY P&L")
    print(f"{'─'*70}")
    print(f"  {'Date':>6s} | {'Trades':>6s} | {'W/L':>5s} | {'WR':>5s} | {'PnL':>8s} | {'CumPnL':>8s}")
    print(f"  {'─'*6} | {'─'*6} | {'─'*5} | {'─'*5} | {'─'*8} | {'─'*8}")
    cum = 0
    for date, v in results['daily'].items():
        cum += v['pnl']
        print(f"  {date:>6s} | {v['trades']:>6d} | {v['wins']}W/{v['losses']}L | "
              f"{v['wr']:>4.0f}% | {v['pnl']:>+7.2f}% | {cum:>+7.2f}%")

    # Pattern performance
    print(f"\n{'─'*70}")
    print("  PATTERN PERFORMANCE (sorted by total PnL)")
    print(f"{'─'*70}")
    print(f"  {'Pattern':<20s} | {'#':>2s} | {'W/L':>5s} | {'WR':>5s} | "
          f"{'TotPnL':>8s} | {'AvgPnL':>7s} | {'Best':>7s} | {'Worst':>7s}")
    print(f"  {'─'*20} | {'─'*2} | {'─'*5} | {'─'*5} | {'─'*8} | {'─'*7} | {'─'*7} | {'─'*7}")
    for pat, v in results['patterns'].items():
        print(f"  {pat:<20s} | {v['trades']:>2d} | {v['wins']}W/{v['losses']}L | "
              f"{v['wr']:>4.0f}% | {v['total_pnl']:>+7.2f}% | {v['avg_pnl']:>+6.2f}% | "
              f"{v['best']:>+6.2f}% | {v['worst']:>+6.2f}%")

    # Equity curve
    print(f"\n{'─'*70}")
    print("  EQUITY CURVE")
    print(f"{'─'*70}")
    print(f"  {'#':>3s} | {'Date':>11s} | {'Pattern':<15s} | {'D':>1s} | "
          f"{'PnL':>7s} | {'CumPnL':>8s} | {'Exit':>4s}")
    print(f"  {'─'*3} | {'─'*11} | {'─'*15} | {'─'*1} | {'─'*7} | {'─'*8} | {'─'*4}")
    for e in results['equity_curve']:
        marker = '■' if e['pnl'] > 0 else '□'
        print(f"  {e['trade']:>3d} | {e['date']:>11s} | {e['pattern']:<15s} | "
              f"{e['direction']:>1s} | {e['pnl']:>+6.2f}% | {e['cum_pnl']:>+7.2f}% | "
              f"{e['reason']:>4s} {marker}")

    # Hourly distribution
    print(f"\n{'─'*70}")
    print("  HOURLY DISTRIBUTION (KST)")
    print(f"{'─'*70}")
    for h in range(24):
        if h in results['hourly']:
            v = results['hourly'][h]
            bar = '█' * v['trades']
            print(f"  {h:02d}:00 | {v['trades']:>2d}t | WR {v['wr']:>4.0f}% | "
                  f"PnL {v['pnl']:>+6.2f}% | {bar}")

    # Metrics resets
    if results['metrics_resets']:
        print(f"\n{'─'*70}")
        print("  METRICS RESETS DETECTED")
        print(f"{'─'*70}")
        for r in results['metrics_resets']:
            print(f"  {r['ts']}: {r['before']} → {r['after']} trades")

    print(f"\n{'='*70}")


if __name__ == '__main__':
    print("Parsing logs...")
    signals, closes, metrics = parse_logs()
    print(f"Found {len(signals)} signals, {len(closes)} closes, {len(metrics)} metrics events")

    trades = match_trades(signals, closes)
    print(f"Matched {len(trades)} trades")

    results = analyze(trades, metrics)
    print_report(results)

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'live_log_analysis.json')
    # Convert datetimes for JSON
    def serialize(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return obj

    save_data = {k: v for k, v in results.items() if k != 'equity_curve'}
    save_data['equity_curve_summary'] = {
        'total_trades': len(results['equity_curve']),
        'final_pnl': results['equity_curve'][-1]['cum_pnl'] if results['equity_curve'] else 0,
        'mdd': results['mdd']
    }

    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=serialize)
    print(f"\nResults saved to {out_path}")
