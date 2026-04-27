"""
Fetch BingX BTC funding rate history (720 days).
Saves to data/bingx_funding_rates_full.json.
"""
import json, time
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

import ccxt

ex = ccxt.bingx({'options': {'defaultType': 'swap'}})
print("BingX has fetch_funding_rate_history:", ex.has.get('fetchFundingRateHistory'))

end = datetime.now(timezone.utc)
start = end - timedelta(days=720)
since_ms = int(start.timestamp() * 1000)

all_records = []
cur_since = since_ms
limit = 500
empty_count = 0

while cur_since < int(end.timestamp() * 1000):
    try:
        records = ex.fetch_funding_rate_history('BTC/USDT:USDT', since=cur_since, limit=limit)
    except Exception as e:
        print(f"error at since={datetime.fromtimestamp(cur_since/1000, tz=timezone.utc)}: {e}")
        time.sleep(2)
        continue
    if not records:
        empty_count += 1
        if empty_count >= 3:
            print(f"empty 3x at since={datetime.fromtimestamp(cur_since/1000, tz=timezone.utc)}, stop.")
            break
        cur_since += 86400 * 1000  # advance 1 day
        continue
    empty_count = 0
    all_records.extend(records)
    last_ts = records[-1]['timestamp']
    print(f"fetched {len(records)} (total {len(all_records)}); last={datetime.fromtimestamp(last_ts/1000, tz=timezone.utc)}")
    if last_ts <= cur_since:
        cur_since += 8 * 3600 * 1000  # advance 8h
    else:
        cur_since = last_ts + 1
    time.sleep(0.5)

# dedupe
seen = set(); uniq = []
for r in all_records:
    ts = r['timestamp']
    if ts not in seen:
        seen.add(ts); uniq.append(r)
uniq.sort(key=lambda r: r['timestamp'])

print(f"\nTotal unique records: {len(uniq)}")
if uniq:
    print(f"Range: {datetime.fromtimestamp(uniq[0]['timestamp']/1000, tz=timezone.utc)} ~ "
          f"{datetime.fromtimestamp(uniq[-1]['timestamp']/1000, tz=timezone.utc)}")

p = ROOT / 'data' / 'bingx_funding_rates_full.json'
with open(p, 'w') as f:
    json.dump(uniq, f, indent=2, default=str)
print(f"Saved: {p}")
