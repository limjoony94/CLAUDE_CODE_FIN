# Day-1 Inspection Report — 20260429

**Verdict**: YELLOW — continue but flag

**Flags**: high_gap_downtime

## Depth Stream
  - Rows: 138,853
  - Duration: 19.61 h
  - Effective Hz: 1.967
  - Update interval: median 501ms, p95 516ms, p99 548ms, max 9996ms
  - Spread (bps): median 0.04, p95 0.23
  - Levels filled: bids median 20/20, asks median 20/20
  - File size: 28.14 MB

## Trade Stream
  - Rows: 496,918
  - Duration: 19.61 h
  - Effective Hz: 7.039
  - Trade interval: median 166ms, p95 207ms, p99 852ms
  - Price range: $74902.60 → $77865.50
  - Qty: median 0.0045, p99 0.6052
  - Maker share: 0.506
  - File size: 4.19 MB

## Gaps
  - Total: 122 (81 depth, 41 trades)
  - Total downtime: 894.9 s
  - Longest: 10.0 s

## Decision criteria
  - GREEN: no severe flags → proceed with 4-week run
  - YELLOW: minor flags → continue but log
  - RED: severe (missing stream / slow / shallow / long single gap) → STOP, surface to user