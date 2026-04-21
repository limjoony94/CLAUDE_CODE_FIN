"""
Slippage Raw Data Collector (2026-04-22)
==========================================
Phase 1 of slippage_diagnosis PDCA.

1) 로그 9일치 (.log.2026-04-12 ~ .log.2026-04-20) + 현행 로그 파싱
   - Entry: `Slippage: ±X% (signal=S fill=F)` 패턴
   - Trade events: ENTRY/EXIT reason, prices, timestamps
2) CCXT fetch_my_trades로 BingX 체결 이력 전체 수집 (04-12 이후)
3) 두 소스 cross-reference로 per-trade 4-way 슬리피지 원시값 생성

Output: results/slippage_raw_{date}.json
"""

import sys
import os
import re
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
LOG_DIR = ROOT / 'logs'

# ── Regex patterns ──────────────────────────────────────────────────────
RE_ENTRY_SLIP = re.compile(
    r'^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ \[c1_breakout\] INFO: '
    r'Slippage: (?P<slip>[-+]?\d+\.\d+)% \(signal=(?P<sig>\d+\.?\d*) fill=(?P<fil>\d+\.?\d*)\)'
)
RE_ENTRY = re.compile(
    r'^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ \[c1_breakout\] INFO: '
    r'ENTRY (?P<dir>LONG|SHORT) @ \$(?P<price>\d+\.?\d*) \| SL=\$(?P<sl>\d+\.?\d*)'
)
RE_MARKET = re.compile(
    r'^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ \[c1_breakout\] INFO: '
    r'MARKET (?P<dir>LONG|SHORT) qty=(?P<qty>\d+\.?\d*) fill=\$(?P<fil>\d+\.?\d*)'
)
RE_SL_ORDER = re.compile(
    r'^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ \[c1_breakout\] INFO: '
    r'SL @ \$(?P<price>\d+\.?\d*)'
)
RE_TRAIL_ORDER = re.compile(
    r'^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ \[c1_breakout\] INFO: '
    r'Trail TP: callback=(?P<cb>\d+\.?\d*)% activate=\$(?P<act>\d+\.?\d*)'
)
RE_GHOST = re.compile(
    r'^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ \[c1_breakout\] WARNING: '
    r'GHOST: (?P<dir>LONG|SHORT) @ \$(?P<ep>\d+\.?\d*).*? '
    r'(?P<reason>EXCHANGE_TRAIL|EXCHANGE_SL|TRAIL_TP|EMERGENCY|TIMEOUT).*?exit=\$(?P<xp>\d+\.?\d*) PnL=(?P<pnl>[-+]?\d+\.?\d*)%'
)
RE_EXIT_SLIP = re.compile(
    r'exit_slippage_pct.*?([-+]?\d+\.\d+)'
)
# Additional patterns we may find in v4.7.9+ logs
RE_EXIT = re.compile(
    r'^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ \[c1_breakout\] INFO: '
    r'EXIT (?P<dir>LONG|SHORT).*?reason=(?P<reason>\w+).*?exit=\$(?P<xp>\d+\.?\d*).*?pnl=(?P<pnl>[-+]?\d+\.?\d*)'
)


def parse_logs():
    """Collect all log lines from *.log.* files and current log."""
    log_files = sorted(LOG_DIR.glob('c1_breakout.log.2026-04-*'))
    log_files.append(LOG_DIR / 'c1_breakout.log')
    print(f"Parsing {len(log_files)} log files...")

    events = []  # list of dicts with type and fields
    entry_slips = []  # entries with (ts, slip%, signal, fill)

    for path in log_files:
        if not path.exists():
            continue
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.rstrip('\n')
                m = RE_ENTRY_SLIP.match(line)
                if m:
                    entry_slips.append({
                        'ts': m.group('ts'), 'slip_pct': float(m.group('slip')),
                        'signal': float(m.group('sig')), 'fill': float(m.group('fil')),
                    })
                    continue
                m = RE_ENTRY.match(line)
                if m:
                    events.append({'type': 'entry', 'ts': m.group('ts'),
                                   'dir': m.group('dir'), 'price': float(m.group('price')),
                                   'sl': float(m.group('sl'))})
                    continue
                m = RE_MARKET.match(line)
                if m:
                    events.append({'type': 'market_fill', 'ts': m.group('ts'),
                                   'dir': m.group('dir'), 'qty': float(m.group('qty')),
                                   'fill': float(m.group('fil'))})
                    continue
                m = RE_SL_ORDER.match(line)
                if m:
                    events.append({'type': 'sl_order', 'ts': m.group('ts'),
                                   'price': float(m.group('price'))})
                    continue
                m = RE_TRAIL_ORDER.match(line)
                if m:
                    events.append({'type': 'trail_order', 'ts': m.group('ts'),
                                   'callback_pct': float(m.group('cb')),
                                   'activate': float(m.group('act'))})
                    continue
                m = RE_GHOST.match(line)
                if m:
                    events.append({'type': 'ghost_exit', 'ts': m.group('ts'),
                                   'dir': m.group('dir'), 'entry': float(m.group('ep')),
                                   'reason': m.group('reason'),
                                   'exit': float(m.group('xp')),
                                   'pnl_pct': float(m.group('pnl'))})
                    continue
    return entry_slips, events


def fetch_exchange_trades():
    """Fetch all BTC fills from BingX since 04-12 (same yaml logic as bot.py:322)."""
    import ccxt
    import yaml
    key_file = ROOT / 'config' / 'api_keys.yaml'
    if not key_file.exists():
        print("WARN: api_keys.yaml missing")
        return []
    with open(key_file) as f:
        keys = yaml.safe_load(f) or {}
    bk = keys.get('bingx', keys)
    if isinstance(bk, dict) and 'mainnet' in bk:
        bk = bk['mainnet']
    api_key = bk.get('api_key', '')
    secret = bk.get('secret_key', '')
    if not api_key:
        print("WARN: no api_key in api_keys.yaml")
        return []

    exchange = ccxt.bingx({
        'apiKey': api_key, 'secret': secret,
        'options': {'defaultType': 'swap'},
    })

    since_ts = int(datetime(2026, 4, 12, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
    all_t, since = [], since_ts
    try:
        while True:
            batch = exchange.fetch_my_trades('BTC-USDT', since=since, limit=1000)
            if not batch:
                break
            all_t.extend(batch)
            if batch[-1]['timestamp'] <= since:
                break
            since = batch[-1]['timestamp'] + 1
            if len(all_t) > 5000:
                break
    except Exception as e:
        print(f"WARN: fetch_my_trades failed: {e}")
    return all_t


def fetch_exchange_orders():
    """Fetch order history — trigger prices for STOP/TRAILING orders."""
    import ccxt, yaml
    key_file = ROOT / 'config' / 'api_keys.yaml'
    if not key_file.exists():
        return []
    with open(key_file) as f:
        keys = yaml.safe_load(f) or {}
    bk = keys.get('bingx', keys)
    if isinstance(bk, dict) and 'mainnet' in bk:
        bk = bk['mainnet']
    api_key = bk.get('api_key', '')
    secret = bk.get('secret_key', '')
    if not api_key:
        return []
    exchange = ccxt.bingx({
        'apiKey': api_key, 'secret': secret,
        'options': {'defaultType': 'swap'},
    })
    since_ts = int(datetime(2026, 4, 12, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
    orders = []
    try:
        # Try closed orders endpoint
        since = since_ts
        while True:
            batch = exchange.fetch_closed_orders('BTC-USDT', since=since, limit=500)
            if not batch:
                break
            orders.extend(batch)
            if batch[-1]['timestamp'] <= since:
                break
            since = batch[-1]['timestamp'] + 1
            if len(orders) > 3000:
                break
    except Exception as e:
        print(f"WARN: fetch_closed_orders failed: {e}")
    return orders


def analyze_trades(fills, orders):
    """Match fills to orders, compute exit trigger vs fill slippage.

    Strategy: for each closed STOP_MARKET/TRAILING order with filled status,
    compare its triggerPrice (or stopPrice) to its average fill price.
    """
    # Map order_id → order dict for fast lookup
    order_by_id = {}
    for o in orders:
        oid = o.get('id') or o.get('orderId') or ''
        if oid:
            order_by_id[str(oid)] = o

    # Group fills by order id and classify
    fills_by_order = {}
    for f in fills:
        oid = str(f.get('order') or f.get('orderId') or '')
        fills_by_order.setdefault(oid, []).append(f)

    analyses = []
    for oid, order in order_by_id.items():
        order_type = (order.get('type') or '').upper()
        info = order.get('info') or {}
        order_info_type = (info.get('type') or info.get('origType') or '').upper()
        status = (order.get('status') or '').lower()
        if status not in ('closed', 'filled'):
            continue

        # Trigger / stop price
        trig = (order.get('triggerPrice') or order.get('stopPrice')
                or info.get('triggerPrice') or info.get('stopPrice') or 0)
        try:
            trig = float(trig) if trig else 0.0
        except (TypeError, ValueError):
            trig = 0.0

        avg = order.get('average') or info.get('avgPrice') or 0
        try:
            avg = float(avg) if avg else 0.0
        except (TypeError, ValueError):
            avg = 0.0

        side = (order.get('side') or '').lower()
        amt = order.get('amount') or order.get('filled') or 0

        if trig > 0 and avg > 0:
            # Slippage: positive = adverse for a closing order
            # For SELL (closing LONG): fill < trigger is adverse
            # For BUY (closing SHORT): fill > trigger is adverse
            if side == 'sell':
                slip_pct = (trig - avg) / trig * 100  # positive if adverse
            else:
                slip_pct = (avg - trig) / trig * 100
            analyses.append({
                'order_id': oid, 'type': order_type, 'info_type': order_info_type,
                'side': side, 'amount': amt,
                'trigger': trig, 'fill_avg': avg,
                'slip_pct': round(slip_pct, 4),
                'ts': order.get('timestamp'),
                'reduceOnly': info.get('reduceOnly', False),
            })

    # Summary stats by order type
    by_type = {}
    for a in analyses:
        key = a['info_type'] or a['type']
        by_type.setdefault(key, []).append(a['slip_pct'])
    type_stats = {}
    for k, vals in by_type.items():
        if vals:
            type_stats[k] = {
                'count': len(vals),
                'mean_pct': round(sum(vals)/len(vals), 4),
                'median_pct': round(sorted(vals)[len(vals)//2], 4),
                'min_pct': round(min(vals), 4),
                'max_pct': round(max(vals), 4),
            }

    return {
        'per_order': analyses,
        'by_type_summary': type_stats,
        'total_analyzed': len(analyses),
    }


def summarize_entry_slips(slips):
    if not slips: return {}
    vals = [s['slip_pct'] for s in slips]
    vals.sort()
    mid = len(vals) // 2
    median = vals[mid] if len(vals) % 2 else (vals[mid-1] + vals[mid]) / 2
    mean = sum(vals) / len(vals)
    adv_vals = [abs(v) for v in vals]  # absolute = adverse magnitude
    return {
        'count': len(vals),
        'mean_pct': round(mean, 4),
        'median_pct': round(median, 4),
        'min_pct': round(min(vals), 4),
        'max_pct': round(max(vals), 4),
        'adv_mean_pct': round(sum(adv_vals)/len(adv_vals), 4),
        'adv_max_pct': round(max(adv_vals), 4),
    }


def main():
    print("=" * 70)
    print("Slippage Raw Data Collector — Phase 1")
    print("=" * 70)

    entry_slips, events = parse_logs()
    print(f"\nLog parse: entry_slips={len(entry_slips)} events={len(events)}")

    # Summary
    stats = summarize_entry_slips(entry_slips)
    print(f"\nEntry slippage stats ({stats.get('count', 0)} trades):")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Event breakdown
    event_types = {}
    for e in events:
        event_types[e['type']] = event_types.get(e['type'], 0) + 1
    print(f"\nEvent types: {event_types}")

    # Exchange trades
    print("\nFetching exchange trades...")
    xch = fetch_exchange_trades()
    print(f"Got {len(xch)} exchange fills")

    print("\nFetching exchange orders...")
    ords = fetch_exchange_orders()
    print(f"Got {len(ords)} exchange closed orders")

    # Per-trade slippage analysis using exchange fills
    # A trade has an entry fill and an exit fill (opposite side or reduceOnly)
    # Group by orderId from fills → match to state.json trades
    trade_slip_analysis = analyze_trades(xch, ords)

    out = {
        'generated_at': datetime.now().isoformat(),
        'period': '2026-04-12 to 2026-04-22',
        'entry_slippage_stats': stats,
        'entry_slippage_samples': entry_slips,
        'event_types': event_types,
        'events': events,
        'exchange_trades_count': len(xch),
        'exchange_trades': xch,
        'exchange_orders_count': len(ords),
        'exchange_orders': ords,
        'trade_slip_analysis': trade_slip_analysis,
    }
    path = ROOT / 'results' / f'slippage_raw_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    path.parent.mkdir(exist_ok=True)
    with open(path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {path}")


if __name__ == '__main__':
    main()
