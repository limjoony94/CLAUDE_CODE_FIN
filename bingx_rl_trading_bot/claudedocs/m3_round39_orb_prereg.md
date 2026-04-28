# M3-R39 Pre-Registration — Opening Range Breakout (ORB) 5m + MTF

**Date pre-registered**: 2026-04-29
**Status**: PRE-COMMIT (results not yet observed)
**Honest prior**: ~0% strict-criterion pass (6/6 pile + R38 inconclusive)

**Purpose**: Different mechanism class per advisor instruction after R38 vacuous. Daily session-anchor range break, theory-locked, with NEW pre-reg frequency gate to prevent another vacuous test.

---

## Why structurally distinct (process improvement after R38)

| Round | Conditioning | Anchor type |
|-------|--------------|-------------|
| R9b/C1 | Donchian channel | Rolling N-bar |
| R36 | EMA pullback | Trailing EMA |
| R37 | NR7 + BB squeeze | Rolling variance |
| R38 (parked) | VWAP deviation | Session-anchored price (volume-weighted) |
| **R39 (new)** | **First-1h Opening Range break** | **Session-anchored range (high/low extremes)** |

ORB is the daily-anchor analog of channel breakout but uses **session high/low extremes from the first hour** — fundamentally different from rolling N-bar Donchian (R9b) and from VWAP price-anchor (R38). Classic strategy from equity markets (Crabel, Toby; ORB is in many institutional playbooks).

---

## Locked Mechanism — `entry_orb_5m`

**Algorithm**:
1. Daily session: UTC 00:00 reset
2. Opening Range = high/low across first 60 minutes (12 5m bars; bars 0-11 of session)
3. After OR complete (bar 12 onwards), monitor for break:
   - LONG: 5m close > OR high AND body > 0
   - SHORT: 5m close < OR low AND body < 0
4. Body filter: `|body| / range ≥ 0.4`
5. Volume confirmation: `volume ≥ 1.0 × volume_sma20_5m`
6. Trend confluence (15m frame at the same bar):
   - LONG: 1h SMA200 below close, 4h EMA20>EMA50, 15m EMA20>EMA50
   - SHORT: mirror
7. Max one entry per day (first valid break wins; no chasing reversals)

**Locked Parameters (NO retuning)**:
```python
LOCKED = {
    'opening_range_minutes': 60,
    'break_buffer_pct': 0.0,
    'body_min_ratio': 0.4,
    'volume_mult': 1.0,
    'max_entries_per_day': 1,
}
```

**Theory source**: Opening Range Breakout (Toby Crabel, equity markets pre-1990; standard institutional playbook). MTF confluence per user's strict spec (1h/4h trend + 15m EMA structure).

---

## Exit Framework (constant across rounds)

`run_bt_c1_production` from `m3_round30_c1_production_exact.py`:
- trail_K=2.5, max_sl_atr=4.5, sl_min_pct=0.15, sl_max_pct=3.0
- emergency_sl_pct=3.0, max_hold_bars=192
- progressive_trail enabled, trail_activation_pct=0.05

5m signals projected to 15m bar index (consistent with R38 design).

---

## Pre-Registered Tests (ALL three required to pass)

### NEW — Pre-run Vacuity Gate
- Before tests, count signals on full dataset.
- **If < 0.5 signals/day** (i.e., < 360 signals over 720 days): R39 declared **inconclusive** (vacuous), NOT failed. R40 with different class.
- This prevents false negatives from over-restrictive locking.

### Test 1: WF 5-fold Expanding
- Locked params, friction 0.07
- **Pass**: ≥3/5 folds daily_net > 0

### Test 2: Bootstrap 1000 × 3-day
- random.seed=42, friction 0.07
- **Pass**: pos_rate ≥ 50%

### Test 3: Train/Test 60/40
- Friction 0.07
- **Pass**: BOTH train AND test daily_net > 0

---

## Verdict Logic

- **Pre-run vacuity gate FAIL**: R39 inconclusive (vacuous), pile unchanged at 6/6, R40 with different class
- **Vacuity passes + ALL 3 OOS PASS**: call advisor before any breakthrough claim
- **Vacuity passes + ANY OOS FAIL**: 7th rigorous negative committed

---

## Anti-Adjustment Provisions

1. No sweep, no retuning if FAIL
2. No friction relaxation
3. No criterion relaxation
4. No mechanism swap **as response to FAIL** (different class would be R40, not "R39 v2")
5. ALL params locked from theory before any data observation

---

## Hash anchor

Committed BEFORE results observation. Frequency gate is binding — applies to R39 and all future rounds.
