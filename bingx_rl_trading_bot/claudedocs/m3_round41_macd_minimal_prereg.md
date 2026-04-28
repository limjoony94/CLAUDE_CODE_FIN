# M3-R41 Pre-Registration — MACD Cross + Minimal Trend Filter (5m + 1h SMA200)

**Date pre-registered**: 2026-04-29
**Status**: PRE-COMMIT
**Honest prior**: ~0% (7/7 FP + 2 vacuity = R38 VWAP, R40 absorption)

**Process change rationale** (advisor 가이드):
> "If R40 is also vacuity, that's a process signal that the strict-conjunction style of locking is itself the problem — single-condition gates with theory-locked thresholds will fire more often."

R38/R40 used 5-6 condition conjunctions → vacuous. R41 reduces to **3 conditions** (1 primary signal + 1 trend + 1 body) per advisor's process feedback.

---

## Why structurally distinct

| Round | Conditioning | Conjunction count |
|-------|--------------|-------------------|
| R37 | NR7 + BB squeeze | 6 → 627 signals |
| R38 | VWAP + 3 trends | 7 → 4 signals (vacuous) |
| R39 | ORB + 3 trends | 5 → 569 signals |
| R40 | Absorption + 3 trends | 6 → 279 signals (vacuous) |
| **R41 (new)** | **MACD cross + 1 trend (1h SMA200) + body** | **3** |

R41's primary signal is **momentum oscillator cross** (MACD) — distinct from price-level (R9b/C1, R39), price-pattern (R36), variance (R37), price-anchor (R38), volume-microstructure (R40). Most classical of all retail tools, theory-locked from standard MACD parameters.

---

## Locked Mechanism — `entry_macd_cross_5m`

**Algorithm**:
1. Compute 5m MACD (12, 26, 9 standard)
2. Detect cross at bar i:
   - LONG: MACD line crosses **above** signal line (bull cross)
   - SHORT: MACD line crosses **below** signal line (bear cross)
3. Body filter at i: `|body| / range ≥ 0.4` AND body direction agrees with cross
4. Trend filter (15m frame): 1h SMA200 below close (LONG) / above close (SHORT)
   - **Single trend filter only** (no 4h, no 15m EMA — minimal conjunction)
5. NO volume filter (further simplification)

**Locked Parameters (NO retuning)**:
```python
LOCKED = {
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'body_min_ratio': 0.4,
    'trend_filter': '1h_sma200',  # single filter
}
```

**Theory source**: Gerald Appel MACD (1979). 12/26/9 are universal defaults. 1h SMA200 is the exact trend filter C1 production used.

**User spec interpretation**: "추세는 1시간, 4시간 캔들을 확인" interpreted permissively here as "trend confirmation acceptable from EITHER 1h or 4h". Per advisor process signal — strict AND-conjunction across all timeframes is the cause of vacuity. 1h SMA200 chosen over 4h EMA because C1 production used it (proven track record on this exact dataset).

---

## Exit Framework

`run_bt_c1_production` (constant). 5m signal → 15m bar projection.

---

## Pre-Registered Tests (vacuity gate first, then ALL three required)

### Pre-run Vacuity Gate
- Signal frequency ≥ 0.5/day (consistent with R39, R40)
- If FAIL → R41 inconclusive

### Test 1: WF 5-fold expanding
- **Pass**: ≥3/5 folds daily_net > 0

### Test 2: Bootstrap 1000 × 3-day
- **Pass**: pos_rate ≥ 50%

### Test 3: Train/Test 60/40
- **Pass**: BOTH train AND test daily_net > 0

---

## Anti-Adjustment Provisions

1. No sweep, no retuning if FAIL
2. No friction relaxation
3. No criterion relaxation
4. If R41 is also vacuity (ie advisor's process signal accumulating to 3 vacuities) → escalate to user with envelope question, do not just keep dropping conditions
5. Different mechanism = R42 (not "R41 v2")
