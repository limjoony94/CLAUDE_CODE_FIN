"""30-second smoke test for BingX collector. Connects, subscribes, prints first events, exits."""
import asyncio
import gzip
import io
import json
import sys
import uuid
from pathlib import Path

import websockets

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'scripts' / 'data_pipeline'))

from bingx_l2_collector import WS_URL, DEPTH_DT, TRADE_DT, _parse_message


async def smoke():
    print(f'connecting {WS_URL}')
    n_depth = 0
    n_trade = 0
    n_ping = 0
    n_other = 0
    sample_depth = None
    sample_trade = None
    timeout_sec = 30

    try:
        async with websockets.connect(WS_URL, ping_interval=None, max_size=2**24) as ws:
            await ws.send(json.dumps({'id': str(uuid.uuid4()), 'reqType': 'sub', 'dataType': DEPTH_DT}))
            await ws.send(json.dumps({'id': str(uuid.uuid4()), 'reqType': 'sub', 'dataType': TRADE_DT}))
            print(f'subscribed: {DEPTH_DT}, {TRADE_DT}')

            start = asyncio.get_event_loop().time()
            while asyncio.get_event_loop().time() - start < timeout_sec:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=5)
                except asyncio.TimeoutError:
                    continue
                if isinstance(raw, str):
                    raw = raw.encode('utf-8')
                msg = _parse_message(raw)
                if msg is None:
                    n_other += 1
                    continue
                if msg.get('__ping__'):
                    n_ping += 1
                    await ws.send('Pong')
                    continue
                dt = str(msg.get('dataType', '')).lower()
                if 'depth' in dt:
                    n_depth += 1
                    if sample_depth is None:
                        sample_depth = msg
                elif 'trade' in dt:
                    n_trade += 1
                    if sample_trade is None:
                        sample_trade = msg
                else:
                    n_other += 1
                    if n_other <= 3:
                        print(f'  other: {json.dumps(msg)[:200]}')

    except Exception as e:
        print(f'error: {type(e).__name__}: {e}')

    print(f'\n=== 30s summary ===')
    print(f'depth events: {n_depth}')
    print(f'trade events: {n_trade}')
    print(f'pings: {n_ping}')
    print(f'other: {n_other}')
    if sample_depth:
        print(f'\nsample depth top-level keys: {list(sample_depth.keys())}')
        data = sample_depth.get('data', {})
        if isinstance(data, dict):
            print(f'  data keys: {list(data.keys())[:10]}')
            bids = data.get('bids') or data.get('bidsCoin') or []
            asks = data.get('asks') or data.get('asksCoin') or []
            print(f'  bids[0]: {bids[0] if bids else None}, asks[0]: {asks[0] if asks else None}')
            print(f'  bid levels: {len(bids)}, ask levels: {len(asks)}')
    if sample_trade:
        print(f'\nsample trade top-level keys: {list(sample_trade.keys())}')
        data = sample_trade.get('data')
        print(f'  data type: {type(data).__name__}')
        if isinstance(data, list):
            print(f'  trade list size: {len(data)}, first: {data[0] if data else None}')
        elif isinstance(data, dict):
            print(f'  trade keys: {list(data.keys())}')
            print(f'  values: {data}')


if __name__ == '__main__':
    asyncio.run(smoke())
