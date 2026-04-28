"""Binance Vision aggTrades downloader for BTCUSDT perpetual.

Source: https://data.binance.vision/data/futures/um/daily/aggTrades/BTCUSDT/
Schema: agg_trade_id, price, quantity, first_trade_id, last_trade_id, transact_time(ms), is_buyer_maker

Resumable: skips files already present + size > 0.
Idempotent: re-running doesn't redownload.

Usage:
    python scripts/data_pipeline/binance_vision_downloader.py [start_date] [end_date]
    Default range: last 365 days ending T-2 (allowing for upload lag).
"""
import sys
import time
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / 'data' / 'binance_vision_aggtrades'
DATA_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = 'https://data.binance.vision/data/futures/um/daily/aggTrades/BTCUSDT'
SYMBOL = 'BTCUSDT'


def _parse_date(s: str) -> date:
    return datetime.strptime(s, '%Y-%m-%d').date()


def _date_range(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def download_one(d: date, retries: int = 3) -> tuple[bool, str]:
    """Download aggTrades zip for date d. Returns (success, message)."""
    fname = f'{SYMBOL}-aggTrades-{d.isoformat()}.zip'
    url = f'{BASE_URL}/{fname}'
    out = DATA_DIR / fname

    if out.exists() and out.stat().st_size > 0:
        return True, f'cached: {fname}'

    last_err = ''
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'CCFIN-Phase2/1.0'})
            with urllib.request.urlopen(req, timeout=60) as resp:
                if resp.status != 200:
                    last_err = f'status {resp.status}'
                    if resp.status == 404:
                        # Genuinely missing (date too recent or no data) — don't retry
                        return False, f'404: {fname} (likely too recent or data unavailable)'
                    time.sleep(2 ** attempt)
                    continue
                with open(out, 'wb') as f:
                    f.write(resp.read())
            size = out.stat().st_size
            if size <= 0:
                last_err = 'zero size'
                out.unlink(missing_ok=True)
                time.sleep(2 ** attempt)
                continue
            return True, f'downloaded: {fname} ({size/1e6:.1f} MB)'
        except Exception as e:
            last_err = f'{type(e).__name__}: {e}'
            time.sleep(2 ** attempt)
    return False, f'failed after {retries}: {fname} ({last_err})'


def main():
    if len(sys.argv) >= 3:
        start_d = _parse_date(sys.argv[1])
        end_d = _parse_date(sys.argv[2])
    else:
        # Default: last 365 days ending T-2 (allowing for upload lag)
        today = datetime.now(timezone.utc).date()
        end_d = today - timedelta(days=2)
        start_d = end_d - timedelta(days=365)

    print(f'Range: {start_d} → {end_d} ({(end_d - start_d).days + 1} days)')
    print(f'Storage: {DATA_DIR}')

    n_total = 0
    n_cached = 0
    n_dl = 0
    n_fail = 0
    bytes_dl = 0
    failed_dates = []

    for d in _date_range(start_d, end_d):
        n_total += 1
        ok, msg = download_one(d)
        if ok:
            if 'cached' in msg:
                n_cached += 1
            else:
                n_dl += 1
                # parse size from message
                import re
                m = re.search(r'\(([\d.]+) MB\)', msg)
                if m:
                    bytes_dl += float(m.group(1))
        else:
            n_fail += 1
            failed_dates.append((d.isoformat(), msg))
            print(f'  ! {d.isoformat()}: {msg}')

        if n_total % 30 == 0:
            print(f'  progress: {n_total}/{(end_d - start_d).days + 1} | cached={n_cached} downloaded={n_dl} failed={n_fail} | bytes={bytes_dl:.0f} MB')

    print(f'\n=== summary ===')
    print(f'total: {n_total} days')
    print(f'cached (already present): {n_cached}')
    print(f'newly downloaded: {n_dl}')
    print(f'failed: {n_fail}')
    print(f'bytes downloaded: {bytes_dl:.1f} MB')
    if failed_dates:
        print(f'\nfailed dates:')
        for d, msg in failed_dates[:20]:
            print(f'  {d}: {msg}')
        if len(failed_dates) > 20:
            print(f'  ... and {len(failed_dates) - 20} more')


if __name__ == '__main__':
    main()
