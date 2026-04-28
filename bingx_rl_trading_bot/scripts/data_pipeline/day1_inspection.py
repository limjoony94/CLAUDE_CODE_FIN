"""Day-1 inspection report — manual gate before continuing 4-week recording.

Per advisor guidance (2026-04-29):
  "the first day of recorded data should be inspected manually before continuing
   for the full month. If BingX's free websocket has degraded resolution or holes,
   the whole plan resets — better to find out at day 1 than week 4."

Outputs single-page markdown with:
  - Event counts per stream
  - Update frequency distribution
  - Gap log summary (count, total duration, longest)
  - Depth resolution check (top-1 spread, level fill)
  - File sizes
  - Verdict: GREEN / YELLOW / RED

Usage:
  python scripts/data_pipeline/day1_inspection.py [YYYYMMDD]
  (defaults to most-recent dated parquet in storage/)
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
STORAGE = ROOT / 'scripts' / 'data_pipeline' / 'storage'


def find_latest_date() -> str | None:
    files = sorted(STORAGE.glob('btc_depth_*.parquet'))
    if not files:
        return None
    return files[-1].stem.split('_')[-1]


def inspect(date_str: str) -> dict:
    depth_p = STORAGE / f'btc_depth_{date_str}.parquet'
    trade_p = STORAGE / f'btc_trades_{date_str}.parquet'
    gaps_p = STORAGE / 'gaps.jsonl'

    report: dict = {'date': date_str, 'verdict_flags': []}

    # --- DEPTH ---
    if not depth_p.exists():
        report['depth'] = {'status': 'MISSING'}
        report['verdict_flags'].append('depth_missing')
    else:
        d = pd.read_parquet(depth_p)
        n = len(d)
        ts = d['event_ts_ms'].sort_values().values
        intervals = pd.Series(np.diff(ts))
        spread = (d['ask_px_0'] - d['bid_px_0']).abs()
        # Level fill: how often are top-20 fully populated?
        bid_levels_filled = (d.filter(like='bid_qty_').notna() & (d.filter(like='bid_qty_') > 0)).sum(axis=1)
        ask_levels_filled = (d.filter(like='ask_qty_').notna() & (d.filter(like='ask_qty_') > 0)).sum(axis=1)
        report['depth'] = {
            'rows': int(n),
            'file_size_mb': round(depth_p.stat().st_size / 1e6, 2),
            'ts_first': pd.to_datetime(ts.min(), unit='ms', utc=True).isoformat(),
            'ts_last': pd.to_datetime(ts.max(), unit='ms', utc=True).isoformat(),
            'interval_ms_median': float(intervals.median()),
            'interval_ms_p95': float(intervals.quantile(0.95)),
            'interval_ms_p99': float(intervals.quantile(0.99)),
            'interval_ms_max': int(intervals.max()),
            'spread_bps_median': float((spread / d['bid_px_0'] * 1e4).median()),
            'spread_bps_p95': float((spread / d['bid_px_0'] * 1e4).quantile(0.95)),
            'bid_levels_median': int(bid_levels_filled.median()),
            'ask_levels_median': int(ask_levels_filled.median()),
            'duration_hours': round((ts.max() - ts.min()) / 3.6e6, 2),
            'effective_hz': round(n / max(1, (ts.max() - ts.min()) / 1000), 3),
        }
        # Sanity flags
        if report['depth']['interval_ms_median'] > 1000:
            report['verdict_flags'].append('depth_slow')
        if report['depth']['bid_levels_median'] < 10:
            report['verdict_flags'].append('depth_shallow')
        if report['depth']['interval_ms_p99'] > 30000:
            report['verdict_flags'].append('depth_p99_gap')
        if report['depth']['duration_hours'] < 23:
            report['verdict_flags'].append('depth_short_duration')

    # --- TRADES ---
    if not trade_p.exists():
        report['trades'] = {'status': 'MISSING'}
        report['verdict_flags'].append('trades_missing')
    else:
        t = pd.read_parquet(trade_p)
        n = len(t)
        ts = t['event_ts_ms'].sort_values().values
        intervals = pd.Series(np.diff(ts))
        report['trades'] = {
            'rows': int(n),
            'file_size_mb': round(trade_p.stat().st_size / 1e6, 2),
            'ts_first': pd.to_datetime(ts.min(), unit='ms', utc=True).isoformat(),
            'ts_last': pd.to_datetime(ts.max(), unit='ms', utc=True).isoformat(),
            'interval_ms_median': float(intervals.median()),
            'interval_ms_p95': float(intervals.quantile(0.95)),
            'interval_ms_p99': float(intervals.quantile(0.99)),
            'interval_ms_max': int(intervals.max()),
            'price_min': float(t['price'].min()),
            'price_max': float(t['price'].max()),
            'qty_median': float(t['qty'].median()),
            'qty_p99': float(t['qty'].quantile(0.99)),
            'maker_share': float(t['is_buyer_maker'].mean()),
            'duration_hours': round((ts.max() - ts.min()) / 3.6e6, 2),
            'effective_hz': round(n / max(1, (ts.max() - ts.min()) / 1000), 3),
        }
        if report['trades']['interval_ms_p99'] > 30000:
            report['verdict_flags'].append('trades_p99_gap')

    # --- GAPS ---
    if gaps_p.exists():
        gaps = []
        with open(gaps_p, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    gaps.append(json.loads(line))
                except Exception:
                    continue
        # Filter to this date
        date_dt = datetime.strptime(date_str, '%Y%m%d').replace(tzinfo=timezone.utc)
        date_start_ms = int(date_dt.timestamp() * 1000)
        date_end_ms = date_start_ms + 86400000
        relevant = [g for g in gaps if date_start_ms <= g['gap_start_ms'] < date_end_ms]
        report['gaps'] = {
            'count': len(relevant),
            'total_duration_sec': round(sum(g['gap_duration_ms'] for g in relevant) / 1000, 1),
            'longest_sec': round(max((g['gap_duration_ms'] for g in relevant), default=0) / 1000, 1),
            'depth_gaps': sum(1 for g in relevant if g['stream'] == 'depth'),
            'trade_gaps': sum(1 for g in relevant if g['stream'] == 'trade'),
        }
        if report['gaps']['total_duration_sec'] > 600:  # >10 minutes total downtime
            report['verdict_flags'].append('high_gap_downtime')
        if report['gaps']['longest_sec'] > 300:  # 5 min single gap
            report['verdict_flags'].append('long_single_gap')
    else:
        report['gaps'] = {'count': 0, 'total_duration_sec': 0, 'longest_sec': 0, 'depth_gaps': 0, 'trade_gaps': 0}

    # --- VERDICT ---
    flags = report['verdict_flags']
    severe = {'depth_missing', 'trades_missing', 'depth_slow', 'depth_shallow', 'long_single_gap'}
    if any(f in severe for f in flags):
        report['verdict'] = 'RED — STOP, surface to user'
    elif flags:
        report['verdict'] = 'YELLOW — continue but flag'
    else:
        report['verdict'] = 'GREEN — proceed with 4-week run'
    return report


def render_md(report: dict) -> str:
    lines: list[str] = []
    lines.append(f'# Day-1 Inspection Report — {report["date"]}')
    lines.append(f'\n**Verdict**: {report["verdict"]}')
    if report['verdict_flags']:
        lines.append(f'\n**Flags**: {", ".join(report["verdict_flags"])}')
    lines.append('\n## Depth Stream')
    d = report.get('depth', {})
    if d.get('status') == 'MISSING':
        lines.append('  ⚠️ MISSING')
    else:
        lines.append(f'  - Rows: {d["rows"]:,}')
        lines.append(f'  - Duration: {d["duration_hours"]:.2f} h')
        lines.append(f'  - Effective Hz: {d["effective_hz"]:.3f}')
        lines.append(f'  - Update interval: median {d["interval_ms_median"]:.0f}ms, p95 {d["interval_ms_p95"]:.0f}ms, p99 {d["interval_ms_p99"]:.0f}ms, max {d["interval_ms_max"]}ms')
        lines.append(f'  - Spread (bps): median {d["spread_bps_median"]:.2f}, p95 {d["spread_bps_p95"]:.2f}')
        lines.append(f'  - Levels filled: bids median {d["bid_levels_median"]}/20, asks median {d["ask_levels_median"]}/20')
        lines.append(f'  - File size: {d["file_size_mb"]} MB')
    lines.append('\n## Trade Stream')
    t = report.get('trades', {})
    if t.get('status') == 'MISSING':
        lines.append('  ⚠️ MISSING')
    else:
        lines.append(f'  - Rows: {t["rows"]:,}')
        lines.append(f'  - Duration: {t["duration_hours"]:.2f} h')
        lines.append(f'  - Effective Hz: {t["effective_hz"]:.3f}')
        lines.append(f'  - Trade interval: median {t["interval_ms_median"]:.0f}ms, p95 {t["interval_ms_p95"]:.0f}ms, p99 {t["interval_ms_p99"]:.0f}ms')
        lines.append(f'  - Price range: ${t["price_min"]:.2f} → ${t["price_max"]:.2f}')
        lines.append(f'  - Qty: median {t["qty_median"]:.4f}, p99 {t["qty_p99"]:.4f}')
        lines.append(f'  - Maker share: {t["maker_share"]:.3f}')
        lines.append(f'  - File size: {t["file_size_mb"]} MB')
    lines.append('\n## Gaps')
    g = report.get('gaps', {})
    lines.append(f'  - Total: {g["count"]} ({g["depth_gaps"]} depth, {g["trade_gaps"]} trades)')
    lines.append(f'  - Total downtime: {g["total_duration_sec"]:.1f} s')
    lines.append(f'  - Longest: {g["longest_sec"]:.1f} s')
    lines.append('\n## Decision criteria')
    lines.append('  - GREEN: no severe flags → proceed with 4-week run')
    lines.append('  - YELLOW: minor flags → continue but log')
    lines.append('  - RED: severe (missing stream / slow / shallow / long single gap) → STOP, surface to user')
    return '\n'.join(lines)


def main() -> None:
    if len(sys.argv) > 1:
        date_str = sys.argv[1]
    else:
        date_str = find_latest_date()
        if date_str is None:
            print('No depth parquet found in storage/. Run collector first.')
            sys.exit(1)

    report = inspect(date_str)
    md = render_md(report)
    print(md)

    out_md = STORAGE / f'day1_inspection_{date_str}.md'
    out_json = STORAGE / f'day1_inspection_{date_str}.json'
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write(md)
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str)
    print(f'\nSaved: {out_md}')
    print(f'Saved: {out_json}')


if __name__ == '__main__':
    main()
