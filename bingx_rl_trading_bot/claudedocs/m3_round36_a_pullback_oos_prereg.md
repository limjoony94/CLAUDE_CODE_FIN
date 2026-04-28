# M3-R36 Pre-Registration — A Pullback (15m) Single OOS Verification

**Date pre-registered**: 2026-04-28
**Status**: PRE-COMMIT (results not yet observed)
**Purpose**: Verify whether R35 "A pullback continuation 15m" candidate represents a real edge or selection-after-peek artifact.

---

## Background

R35 explored 3 mechanism classes on 15m timeframe with C1 production exit:
- A: Pullback continuation (4×4=16 grid)
- B: Breakout retest (3×3×3=27 grid)
- C: Multi-bar momentum (4×3×3=36 grid)

Reported: 35/77 OOS test_daily > 0 survivors after 60/40 train/test split.

**Advisor critique on this result**:
1. 35/77 (45%) is **under random expectation** (~50% positive at 0-mean null)
2. Single 60/40 split = **selection-after-peek** if "top by te_daily" is then reported
3. 77 configs across 3 mechanism classes = **multi-comparison** alarm
4. n=82 over 288 days = 0.27 trades/day = previously flagged "ι noise territory"
5. Same shape as R9b/R15/R19/R30 prior false positives

**Therefore**: A single locked candidate must pass strict pre-registered OOS to claim edge.

---

## Locked Candidate (NO retuning permitted after this point)

**Mechanism**: A — Pullback Continuation 15m
**File**: `bingx_rl_trading_bot/scripts/analysis/m3_round35_15m_deep.py::entry_pullback_15m`

**Locked Parameters**:
- `ema_dist_pct = 0.5`
- `volume_mult = 1.0`

**Rationale for selection**: Per advisor — "Pick A pullback (ema=0.5, vol=1.0) — highest avg_gross + R:R, even if n=82 is small."

**Exit framework**: C1 production exact (`run_bt_c1_production` in m3_round30_c1_production_exact.py):
- channel_period: 15 (irrelevant for entry, only for exit framework defaults)
- trail_K: 2.5
- max_sl_atr: 4.5
- sl_min_pct: 0.15, sl_max_pct: 3.0
- emergency_sl_pct: 3.0
- max_hold_bars: 192
- trail_activation_pct: 0.05
- progressive_trail: enabled, threshold=0.9, K_post=0.5
- min_bars_between: 2

**Friction**: 0.07 (mixed maker/taker scenario, consistent with R35 selection criterion)

**Data**: Full BTC 5m → 15m synthesized, ~720 days available.

---

## Pre-Registered Tests (ALL three required to pass)

### Test 1: Walk-Forward 5-fold Expanding
**Setup**:
- 5 expanding windows: train [0, fold_end_i], test [fold_end_i, fold_end_{i+1}]
- Each fold uses LOCKED params (no parameter selection per fold)
- Friction 0.07

**Pass criterion**: ≥3 of 5 folds with `daily_net > 0`

### Test 2: 3-day Random Window Bootstrap
**Setup**:
- 1000 random 3-day windows (random.seed=42)
- Each window: run entry_pullback_15m + run_bt_c1_production with locked params
- Friction 0.07

**Pass criterion**: pos_rate ≥ 50% (≥ 500/1000 windows positive)

### Test 3: Train→Test Parameter Consistency
**Setup**:
- Same 60/40 train/test split as R35
- Compute train daily_net and test daily_net for LOCKED params
- Compute |train - test| / |train| * 100 (percent deviation)

**Pass criterion**:
- Both train and test daily_net > 0
- Train and test agree on direction (both positive)
- Pre-reg note: We do NOT require strict numerical match; only that test does not flip sign

---

## Verdict Logic (Strict)

**ALL THREE TESTS PASS** → Candidate has not been falsified. Promote to deeper validation (full 5-fold WF on multiple frictions, slippage stress, paper trade plan).

**ANY ONE TEST FAILS** → Candidate dropped permanently. R35 "breakthrough" claim retracted. Add to lesson-learned memory alongside R9b/R15/R19.

---

## Anti-Adjustment Provisions

To prevent "anti-fix-impulse" pattern (memory `lessons_fix_impulse_pattern_20260427.md`):

1. **No retuning**: If any test fails, do NOT search nearby parameters.
2. **No mechanism swap**: Do NOT switch to B retest or C momentum candidates as "alternative".
3. **No friction relaxation**: 0.07 is locked. Do not retest at 0.04.
4. **No criterion relaxation**: 50% bootstrap pos_rate, 3/5 WF folds, train+test agree on sign.
5. **No "but the trade count is small" excuses**: Smaller n means MORE skepticism, not less.

---

## What This Test Cannot Establish

Even if all three pass:
- Sample size n=82 over 288 days (0.27 trades/day) is below user's "≥2 trades/day" criterion
- 1× per-trade ~0.21% is just barely above 1× friction band
- BT-LIVE parity gap (C1 demonstrated this gap) remains unverified
- 15m mechanisms have not yet faced advisor-level challenges that 5m mechanisms have

So PASS = "not yet falsified, deserves deeper exam"
NOT PASS = decisive falsification.

**This is intentionally asymmetric**: passing one OOS does not prove edge; failing OOS does prove no edge at this sample size.

---

## Implementation Files (to be created)

- `bingx_rl_trading_bot/scripts/analysis/m3_round36_a_pullback_oos.py`
- `bingx_rl_trading_bot/results/m3_r36_a_pullback_oos_*.json`

---

## Hash anchor

This pre-reg is committed BEFORE results observation. Any deviation from the locked params, friction, or pass criteria after observing results is grounds for full retraction.
