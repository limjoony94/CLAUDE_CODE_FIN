# Day-1 Inspection Report — 20260429

**Verdict**: YELLOW — continue but flag

**Flags**: high_gap_downtime

## Depth Stream
  - Rows: 128,596
  - Duration: 18.17 h
  - Effective Hz: 1.966
  - Update interval: median 501ms, p95 516ms, p99 548ms, max 9996ms
  - Spread (bps): median 0.04, p95 0.24
  - Levels filled: bids median 20/20, asks median 20/20
  - File size: 25.8 MB

## Trade Stream
  - Rows: 454,475
  - Duration: 18.17 h
  - Effective Hz: 6.946
  - Trade interval: median 173ms, p95 207ms, p99 885ms
  - Price range: $74902.60 → $77865.50
  - Qty: median 0.0044, p99 0.5863
  - Maker share: 0.513
  - File size: 3.85 MB

## Gaps
  - Total: 122 (81 depth, 41 trades)
  - Total downtime: 894.9 s
  - Longest: 10.0 s

## Decision criteria
  - GREEN: no severe flags → proceed with 4-week run
  - YELLOW: minor flags → continue but log
  - RED: severe (missing stream / slow / shallow / long single gap) → STOP, surface to user