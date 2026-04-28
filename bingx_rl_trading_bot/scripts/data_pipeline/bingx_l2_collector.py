"""BingX free WebSocket L2 + trade tape collector for BTC-USDT perpetual.

Phase 1 of M3 closure recommendation (advisor 2026-04-29).
Records depth20 + aggregated trades, daily UTC parquet partitions, no downsampling.
Resilience: exponential backoff reconnect, gap detection, no fill.

Endpoint: wss://open-api-swap.bingx.com/swap-market
Subscription:
  - BTC-USDT@depth20  (top-20 levels orderbook)
  - BTC-USDT@trade    (aggregated trades)
Messages: gzip-compressed JSON. Ping → Pong protocol.

Storage:
  storage/btc_depth_YYYYMMDD.parquet
  storage/btc_trades_YYYYMMDD.parquet
  storage/gaps.jsonl    (gap log, one JSON per line)
  storage/run.log       (operational log)
"""
import asyncio
import gzip
import io
import json
import logging
import sys
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import websockets

ROOT = Path(__file__).resolve().parent.parent.parent
STORAGE = ROOT / 'scripts' / 'data_pipeline' / 'storage'
STORAGE.mkdir(parents=True, exist_ok=True)

WS_URL = 'wss://open-api-swap.bingx.com/swap-market'
SYMBOL = 'BTC-USDT'
DEPTH_DT = f'{SYMBOL}@depth20'
TRADE_DT = f'{SYMBOL}@trade'

# Buffer flush thresholds — flush when buffer hits N rows OR every M seconds
DEPTH_FLUSH_ROWS = 5000
TRADE_FLUSH_ROWS = 5000
FLUSH_INTERVAL_SEC = 30

# Gap detection: timestamp jump > this many ms = gap event
GAP_THRESHOLD_MS = 5000  # 5s gap = anomaly worth logging

# Reconnect backoff
RECONNECT_BASE_SEC = 2
RECONNECT_MAX_SEC = 60


def _utc_now_ms() -> int:
    return int(time.time() * 1000)


def _utc_today_str() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%d')


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger('bingx_collector')
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(STORAGE / 'run.log', encoding='utf-8')
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


class BufferedWriter:
    """Buffers records and flushes to daily parquet files."""

    def __init__(self, name: str, flush_rows: int, logger: logging.Logger):
        self.name = name  # 'depth' or 'trades'
        self.flush_rows = flush_rows
        self.buf: list[dict] = []
        self.last_flush_ts = time.time()
        self.logger = logger

    def add(self, record: dict) -> None:
        self.buf.append(record)
        if len(self.buf) >= self.flush_rows or (time.time() - self.last_flush_ts) >= FLUSH_INTERVAL_SEC:
            self.flush()

    def flush(self) -> None:
        if not self.buf:
            return
        # Group by UTC date in case buffer spans midnight
        df = pd.DataFrame(self.buf)
        if 'event_ts_ms' in df.columns:
            df['utc_date'] = pd.to_datetime(df['event_ts_ms'], unit='ms', utc=True).dt.strftime('%Y%m%d')
        else:
            df['utc_date'] = _utc_today_str()
        for date_str, group in df.groupby('utc_date'):
            file_p = STORAGE / f'btc_{self.name}_{date_str}.parquet'
            group_to_write = group.drop(columns=['utc_date'])
            if file_p.exists():
                # Append: read+concat+write (small daily volume; OK for v1)
                existing = pd.read_parquet(file_p)
                merged = pd.concat([existing, group_to_write], ignore_index=True)
                merged.to_parquet(file_p, index=False)
            else:
                group_to_write.to_parquet(file_p, index=False)
        self.logger.info(f'flush {self.name}: {len(self.buf)} rows → {df["utc_date"].unique().tolist()}')
        self.buf.clear()
        self.last_flush_ts = time.time()


class GapMonitor:
    """Detects timestamp gaps within each stream."""

    def __init__(self, name: str, logger: logging.Logger):
        self.name = name
        self.last_ts_ms: int | None = None
        self.logger = logger

    def observe(self, ts_ms: int) -> None:
        if self.last_ts_ms is not None:
            gap_ms = ts_ms - self.last_ts_ms
            if gap_ms > GAP_THRESHOLD_MS:
                gap_record = {
                    'stream': self.name,
                    'gap_start_ms': self.last_ts_ms,
                    'gap_end_ms': ts_ms,
                    'gap_duration_ms': gap_ms,
                    'gap_start_iso': datetime.fromtimestamp(self.last_ts_ms / 1000, tz=timezone.utc).isoformat(),
                    'gap_end_iso': datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat(),
                }
                with open(STORAGE / 'gaps.jsonl', 'a', encoding='utf-8') as f:
                    f.write(json.dumps(gap_record) + '\n')
                self.logger.warning(f'GAP {self.name}: {gap_ms} ms')
        self.last_ts_ms = ts_ms


def _parse_message(raw: bytes) -> dict | None:
    """Decompress gzip + parse JSON. Returns dict or None on parse failure."""
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as f:
            decompressed = f.read()
    except Exception:
        # Sometimes BingX sends plain text (e.g., Ping)
        decompressed = raw
    text = decompressed.decode('utf-8', errors='replace').strip()
    if not text:
        return None
    if text == 'Ping':
        return {'__ping__': True}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _extract_depth_record(msg: dict) -> dict | None:
    """Map BingX depth20 message to flat record."""
    data = msg.get('data') or msg.get('data_t') or msg
    if not isinstance(data, dict):
        return None
    bids = data.get('bids') or data.get('bidsCoin') or []
    asks = data.get('asks') or data.get('asksCoin') or []
    ts_ms = int(data.get('T') or data.get('ts') or msg.get('ts') or _utc_now_ms())
    rec: dict = {'event_ts_ms': ts_ms, 'symbol': data.get('s', SYMBOL)}
    # Flatten top 20 levels (price, qty)
    for i, (px, qty) in enumerate(bids[:20]):
        rec[f'bid_px_{i}'] = float(px)
        rec[f'bid_qty_{i}'] = float(qty)
    for i, (px, qty) in enumerate(asks[:20]):
        rec[f'ask_px_{i}'] = float(px)
        rec[f'ask_qty_{i}'] = float(qty)
    return rec


def _extract_trade_record(msg: dict) -> dict | None:
    """Map BingX aggregated trade to flat record."""
    data = msg.get('data') or msg
    if isinstance(data, list):
        # Some streams send list of trades
        return None  # Not handled in this single-record helper
    if not isinstance(data, dict):
        return None
    ts_ms = int(data.get('T') or data.get('ts') or msg.get('ts') or _utc_now_ms())
    return {
        'event_ts_ms': ts_ms,
        'symbol': data.get('s', SYMBOL),
        'price': float(data.get('p', 0) or 0),
        'qty': float(data.get('q', 0) or 0),
        'is_buyer_maker': bool(data.get('m', False)),
        'trade_id': str(data.get('t', '')),
    }


async def _run_session(logger: logging.Logger,
                        depth_w: BufferedWriter, trade_w: BufferedWriter,
                        depth_gap: GapMonitor, trade_gap: GapMonitor) -> None:
    logger.info(f'connecting {WS_URL}')
    async with websockets.connect(WS_URL, ping_interval=None, max_size=2**24) as ws:
        # Subscribe
        sub_depth = {'id': str(uuid.uuid4()), 'reqType': 'sub', 'dataType': DEPTH_DT}
        sub_trade = {'id': str(uuid.uuid4()), 'reqType': 'sub', 'dataType': TRADE_DT}
        await ws.send(json.dumps(sub_depth))
        await ws.send(json.dumps(sub_trade))
        logger.info(f'subscribed: {DEPTH_DT}, {TRADE_DT}')

        async for raw in ws:
            if isinstance(raw, str):
                raw = raw.encode('utf-8')
            msg = _parse_message(raw)
            if msg is None:
                continue
            if msg.get('__ping__'):
                await ws.send('Pong')
                continue
            dt = msg.get('dataType') or msg.get('s') or ''
            # BingX returns dataType in response messages
            if 'depth' in str(dt).lower():
                rec = _extract_depth_record(msg)
                if rec is not None and rec.get('event_ts_ms'):
                    depth_gap.observe(rec['event_ts_ms'])
                    depth_w.add(rec)
            elif 'trade' in str(dt).lower():
                # Some BingX trade streams send single dict, others a list
                data = msg.get('data')
                if isinstance(data, list):
                    for item in data:
                        sub_msg = {'data': item, 'ts': msg.get('ts')}
                        rec = _extract_trade_record(sub_msg)
                        if rec is not None and rec.get('event_ts_ms'):
                            trade_gap.observe(rec['event_ts_ms'])
                            trade_w.add(rec)
                else:
                    rec = _extract_trade_record(msg)
                    if rec is not None and rec.get('event_ts_ms'):
                        trade_gap.observe(rec['event_ts_ms'])
                        trade_w.add(rec)
            # Subscription ack messages — log once at info, ignore otherwise
            elif 'code' in msg or msg.get('id'):
                logger.info(f'ack: {msg}')


async def main_loop() -> None:
    logger = _setup_logger()
    logger.info('=' * 80)
    logger.info(f'BingX L2 collector start | symbol={SYMBOL} | streams={DEPTH_DT}, {TRADE_DT}')
    logger.info(f'storage={STORAGE}')

    depth_w = BufferedWriter('depth', DEPTH_FLUSH_ROWS, logger)
    trade_w = BufferedWriter('trades', TRADE_FLUSH_ROWS, logger)
    depth_gap = GapMonitor('depth', logger)
    trade_gap = GapMonitor('trade', logger)

    backoff = RECONNECT_BASE_SEC
    while True:
        try:
            await _run_session(logger, depth_w, trade_w, depth_gap, trade_gap)
            backoff = RECONNECT_BASE_SEC
        except (websockets.ConnectionClosed, OSError, asyncio.TimeoutError) as e:
            logger.warning(f'disconnect: {type(e).__name__}: {e}')
        except Exception as e:
            logger.exception(f'unexpected error: {e}')
        # Flush before reconnect
        depth_w.flush()
        trade_w.flush()
        logger.info(f'reconnecting in {backoff}s')
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, RECONNECT_MAX_SEC)


if __name__ == '__main__':
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        print('\nshutdown requested')
