# M3-R37 Pre-Registration — Volatility Compression Breakout (NR7 + Bollinger Squeeze)

**Date pre-registered**: 2026-04-29
**Status**: PRE-COMMIT (results not yet observed)
**Honest prior**: ~0% strict-criterion pass (5/5 prior false positives R9b/R15/R19/R30/R35)

**Purpose**: Per user instruction "새 mechanism을 더 시도하라" inside same data envelope. Add 6th rigorous OOS to evidence pile. Pass = scrutinize test design before claiming. Fail = stronger evidence to support envelope-change decision.

---

## Why structurally distinct from 5 priors

| Prior round | Entry trigger | Conditioning |
|-------------|---------------|--------------|
| R9b/C1 family | Donchian channel break | Price level |
| R21 | Pattern reversal at extreme | Price level + body |
| R24 | EMA pullback in trend | Price level + EMA |
| R35-A pullback | Same as R24 with body filter | Price level |
| R35-B retest | Channel break + retest | Price level |
| R35-C momentum | N consecutive same-direction | Price level |
| **R37 (new)** | **NR7 + Bollinger Squeeze break** | **Variance level** |

R37 entry condition fires only when realized variance compression is below the 20th percentile of recent history. None of the 5 priors used variance compression as gating.

---

## Locked Mechanism — `entry_compression_breakout_15m`

**Algorithm**:
1. Compute `range_i = high_i - low_i` for each 15m bar
2. Check `range_i == min(range_{i-6}, ..., range_i)` (NR7: current bar is narrowest of last 7)
3. Compute Bollinger Bandwidth `bw = (upper - lower) / middle` with period 20, std 2.0
4. Check `bw_i ≤ 20th-percentile of bw over [i-19, i]` (squeeze)
5. Body filter: `|close - open| / range ≥ 0.4`
6. Direction match:
   - LONG: `close > prev 7-bar high` AND body > 0
   - SHORT: `close < prev 7-bar low` AND body < 0
7. Volume confirmation: `volume_i ≥ 1.0 × volume_sma20` (consistent with R35 baseline)

**Locked Parameters (NO retuning)**:
```python
LOCKED = {
    'compression_lookback': 7,
    'bandwidth_lookback': 20,
    'bandwidth_pctile_max': 0.20,
    'body_min_ratio': 0.4,
    'volume_mult': 1.0,
    'bb_period': 20,
    'bb_std': 2.0,
}
```

**Theory source**: Crabel (1990) NR7 narrowing pattern + standard Bollinger Bandwidth squeeze. No sweeps, no grid, no per-fold tuning.

---

## Exit Framework (constant across 6 rounds)

`run_bt_c1_production` from `m3_round30_c1_production_exact.py`:
- trail_K=2.5, max_sl_atr=4.5, sl_min_pct=0.15, sl_max_pct=3.0
- emergency_sl_pct=3.0, max_hold_bars=192
- trail_activation_pct=0.05
- progressive_trail enabled (threshold=0.9, K_post=0.5)
- min_bars_between=2

---

## Pre-Registered Tests (ALL three required to pass)

### Test 1: WF 5-fold Expanding
- 5 expanding test windows on locked params
- Friction 0.07
- **Pass**: ≥3/5 folds with daily_net > 0

### Test 2: Bootstrap 1000 × 3-day
- random.seed=42, full data range
- Friction 0.07
- **Pass**: pos_rate ≥ 50%

### Test 3: Train/Test 60/40
- Same 60/40 split as R35
- Friction 0.07
- **Pass**: train AND test both daily_net > 0

---

## Verdict Logic

- **ALL 3 PASS** → call advisor IMMEDIATELY (per advisor instruction). Do NOT claim breakthrough until advisor scrutinizes test design.
- **ANY FAIL** → 6th rigorous negative committed. R37 mechanism dropped permanently. Return to user with sharper evidence supporting envelope-change decision.

---

## Anti-Adjustment Provisions

1. No retuning if FAIL — add to false-positive list (now would be 6/6)
2. No friction relaxation
3. No criterion relaxation
4. No "but n is small" excuses
5. No mechanism swap as "alternative"

---

## What this test cannot establish

Even if all 3 pass:
- Sample size warnings still apply
- BT-LIVE parity gap (C1 demonstrated) unverified
- 1× per-trade still likely below user's strict +0.20%/trade target
- Advisor will scrutinize the test design before any claim

PASS = "earned scrutiny, not breakthrough"
FAIL = decisive 6th evidence point on envelope frontier

---

## Hash anchor

Committed BEFORE results observation. Any deviation = full retraction.
