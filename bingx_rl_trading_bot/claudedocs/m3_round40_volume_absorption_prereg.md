# M3-R40 Pre-Registration — Volume Absorption + Trend Continuation (5m + MTF)

**Date pre-registered**: 2026-04-29
**Status**: PRE-COMMIT
**Honest prior**: ~0% strict-criterion pass (7/7 prior FP + R38 inconclusive in same envelope)

**Purpose**: 8th rigorous OOS per user explicit instruction "성공적인 전략을 발굴해 낼 때까지 자동적으로 진행". New mechanism class structurally distinct from all 7 prior (volume-microstructure conditioning).

---

## Why structurally distinct from 7 priors

| Round | Conditioning |
|-------|--------------|
| R9b/C1 | Donchian rolling channel |
| R21 | Pattern at extreme |
| R36 | EMA pullback |
| R37 | NR7 + BB squeeze (variance) |
| R38 | VWAP price-anchor (parked) |
| R39 | Daily ORB session-anchor |
| **R40 (new)** | **Volume absorption (high vol + small body = institutional absorption proxy)** |

Volume absorption is a **microstructure proxy** — high volume on a small-body candle indicates institutional absorption (Wyckoff theory, modern order flow analysis). Different from any price-conditional, anchor-conditional, or variance-conditional entry. All 7 prior used price/range/variance signals; R40 uses **volume-to-range relationship**.

---

## Locked Mechanism — `entry_volume_absorption_5m`

**Algorithm**:
1. Detect absorption bar at index i (5m frame):
   - `volume_i ≥ 2.0 × volume_sma20_5m` (high volume)
   - `|body_i| / range_i ≤ 0.3` (small body relative to range = absorption)
2. Confirmation bar at i+1:
   - Body filter: `|body_{i+1}| / range_{i+1} ≥ 0.4`
   - Direction agrees with absorption bar's wick balance:
     - LONG: lower wick of bar i > upper wick (buyers absorbed sellers) AND bar i+1 close > bar i close
     - SHORT: upper wick of bar i > lower wick AND bar i+1 close < bar i close
3. Trend confluence (15m frame at i+1):
   - LONG: 1h SMA200 below close, 4h EMA20>EMA50, 15m EMA20>EMA50
   - SHORT: mirror

**Locked Parameters (NO retuning)**:
```python
LOCKED = {
    'absorption_vol_mult': 2.0,
    'absorption_body_ratio_max': 0.3,
    'confirmation_body_min': 0.4,
    'wick_imbalance_required': True,
}
```

**Theory source**: Wyckoff absorption + modern order flow proxy. High-volume small-body indicates institutional accumulation/distribution; subsequent bar shows trend resumption.

---

## Exit Framework

`run_bt_c1_production` (constant). 5m → 15m projection.

---

## Pre-Registered Tests (ALL three required)

### Pre-run Vacuity Gate (NEW from R39)
- Signal frequency ≥ 0.5/day
- If FAIL → R40 inconclusive, NOT 8th fail

### Test 1: WF 5-fold expanding
- **Pass**: ≥3/5 folds daily_net > 0

### Test 2: Bootstrap 1000 × 3-day
- **Pass**: pos_rate ≥ 50%

### Test 3: Train/Test 60/40
- **Pass**: BOTH train AND test daily_net > 0

---

## Verdict Logic

- Vacuity FAIL: inconclusive
- Vacuity PASS + ALL 3 OOS PASS: call advisor before claim
- Vacuity PASS + ANY OOS FAIL: 8th rigorous negative

---

## Anti-Adjustment Provisions

1. No sweep, no retuning if FAIL
2. No friction relaxation
3. No criterion relaxation
4. No mechanism swap as response to FAIL (different class = R41, not "R40 v2")
5. Theory-locked params before data observation
