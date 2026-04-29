# Day-1 Inspection Report — 20260428

**Verdict**: YELLOW — continue but flag

**Flags**: depth_short_duration

## Depth Stream
  - Rows: 180
  - Duration: 0.03 h
  - Effective Hz: 1.991
  - Update interval: median 501ms, p95 515ms, p99 564ms, max 607ms
  - Spread (bps): median 0.05, p95 0.76
  - Levels filled: bids median 20/20, asks median 20/20
  - File size: 0.1 MB

## Trade Stream
  - Rows: 480
  - Duration: 0.02 h
  - Effective Hz: 5.341
  - Trade interval: median 199ms, p95 210ms, p99 1008ms
  - Price range: $76036.20 → $76069.70
  - Qty: median 0.0039, p99 0.3610
  - Maker share: 0.465
  - File size: 0.01 MB

## Gaps
  - Total: 0 (0 depth, 0 trades)
  - Total downtime: 0.0 s
  - Longest: 0.0 s

## Decision criteria
  - GREEN: no severe flags → proceed with 4-week run
  - YELLOW: minor flags → continue but log
  - RED: severe (missing stream / slow / shallow / long single gap) → STOP, surface to user