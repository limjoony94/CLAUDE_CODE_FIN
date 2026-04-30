"""R26 Trade Journal Summary — parse JSONL events into pandas-friendly summary.

Usage:
  python scripts/ops/r26_journal_summary.py [--hours 24]

Outputs daily/hourly aggregates of TP fills, force exits, compound updates.
"""
import json
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import Counter, defaultdict

import argparse

project_root = Path(__file__).resolve().parent.parent.parent
os.chdir(project_root)

JOURNAL_PATH = Path('logs/r26_trades.jsonl')


def load_journal(since_dt=None):
    if not JOURNAL_PATH.exists():
        print(f'Journal not found: {JOURNAL_PATH}')
        return []
    events = []
    with open(JOURNAL_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                rec_ts = datetime.fromisoformat(rec['ts_utc'].replace('Z', '+00:00'))
                if since_dt and rec_ts < since_dt:
                    continue
                rec['_ts'] = rec_ts
                events.append(rec)
            except Exception as e:
                print(f'Skip invalid line: {e}')
    return events


def summarize(events):
    if not events:
        return {'total_events': 0}
    by_event = Counter(e['event'] for e in events)
    print(f'Events by type:')
    for ev, count in by_event.most_common():
        print(f'  {ev}: {count}')

    # TP fills
    tp_fills = [e for e in events if e['event'] == 'tp_fill']
    if tp_fills:
        gross_sum = sum(e['gross_pnl_pct'] for e in tp_fills)
        net_sum = sum(e['net_pnl_pct'] for e in tp_fills)
        avg_hold_min = sum(e['hold_seconds'] for e in tp_fills) / len(tp_fills) / 60
        wr = sum(1 for e in tp_fills if e['net_pnl_pct'] > 0) / len(tp_fills)
        long_count = sum(1 for e in tp_fills if e['side'] == 'long')
        short_count = sum(1 for e in tp_fills if e['side'] == 'short')
        print(f'\nTP Fills: {len(tp_fills)} (LONG {long_count}, SHORT {short_count})')
        print(f'  Cumulative gross: {gross_sum:+.4f}%')
        print(f'  Cumulative net (after friction est 0.07%): {net_sum:+.4f}%')
        print(f'  Avg hold time: {avg_hold_min:.1f} min')
        print(f'  WR: {wr*100:.1f}%')

    # Compound updates
    compound = [e for e in events if e['event'] == 'compound_update']
    if compound:
        first_old = compound[0]['old_notional_usd']
        last_new = compound[-1]['new_notional_usd']
        print(f'\nCompound: {len(compound)} updates')
        print(f'  Notional progression: ${first_old:.2f} → ${last_new:.2f} '
              f'({(last_new-first_old)/first_old*100:+.4f}%)')

    # Force closes
    fc = [e for e in events if e['event'] == 'force_close_start']
    if fc:
        print(f'\nForce Closes: {len(fc)}')
        for e in fc:
            print(f'  {e["_ts"]} reason={e["reason"]} '
                  f'orders={e["n_orders_to_cancel"]} positions={e["n_positions_to_close"]}')

    # Grid setups
    setups = [e for e in events if e['event'] == 'grid_setup']
    if setups:
        print(f'\nGrid Setups: {len(setups)}')
        for s in setups[:5]:
            print(f'  {s["_ts"]} mid=${s["init_mid"]:.2f} per_level=${s["per_level_notional_usd"]:.2f}')
        if len(setups) > 5:
            print(f'  ... +{len(setups)-5} more')

    # Hourly distribution of TP fills
    if tp_fills:
        by_hour = defaultdict(lambda: {'count': 0, 'net': 0.0})
        for e in tp_fills:
            hour_key = e['_ts'].strftime('%Y-%m-%d %H:00')
            by_hour[hour_key]['count'] += 1
            by_hour[hour_key]['net'] += e['net_pnl_pct']
        print(f'\nTP fills by hour (last 24):')
        recent = sorted(by_hour.keys())[-24:]
        for h in recent:
            d = by_hour[h]
            print(f'  {h}: {d["count"]} fills, net {d["net"]:+.4f}%')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hours', type=int, default=24,
                         help='Filter events within last N hours (default 24)')
    args = parser.parse_args()

    since = datetime.now(timezone.utc) - timedelta(hours=args.hours)
    print(f'R26 Trade Journal Summary (last {args.hours} hours, since {since.isoformat()})')
    print('=' * 80)

    events = load_journal(since)
    print(f'\nTotal events loaded: {len(events)}\n')
    summarize(events)


if __name__ == '__main__':
    main()
