# M3-R38 Pre-Registration — VWAP-anchored Mean Reversion (5m scalping + MTF)

**Date pre-registered**: 2026-04-29
**Status**: PRE-COMMIT (results not yet observed)
**Honest prior**: ~0% strict-criterion pass (6/6 prior false positives R9b/R15/R19/R30/R36/R37 in same envelope)
**User override**: explicit — user restated strict criteria with knowledge of 6/6 fail evidence

**Purpose**: 7th rigorous OOS in same envelope per user instruction. Anchor-relative conditioning structurally distinct from 6 prior. Pass = scrutinize. Fail = 7th evidence point.

---

## Why structurally distinct from 6 priors

| Prior round | Conditioning | Reference frame |
|-------------|--------------|------------------|
| R9b/C1 | Donchian channel break | Absolute price |
| R21 | Pattern at extreme | Absolute price |
| R24/R36 | Pullback to EMA | Trailing EMA |
| R35-B | Channel break + retest | Absolute price |
| R35-C | Multi-bar momentum | Absolute price |
| R37 | NR7 + BB Squeeze | **Variance level** |
| **R38 (new)** | **Distance from session VWAP** | **Anchor-relative (intraday)** |

R38's entry is conditional on **price deviation from a freshly-anchored reference** (UTC 00:00 daily VWAP). The reference resets per session — none of the 6 priors used this conditioning frame.

VWAP reversion is a textbook institutional intraday strategy (HFT, prop firms). Crypto runs 24/7 but UTC 00:00 is the universal session boundary used by most exchanges.

---

## Locked Mechanism — `entry_vwap_reversion_5m`

**Algorithm**:
1. Compute session-anchored VWAP on 5m bars: reset at UTC 00:00 daily
   - `cumvol_d = cumsum(volume from session start)`
   - `cumdv_d = cumsum(close * volume from session start)`
   - `vwap_d = cumdv_d / cumvol_d`
2. Compute deviation: `dev_pct = (close - vwap) / vwap * 100`
3. Mean reversion entry conditions:
   - **LONG**: `dev_pct < -0.5%` (price extended below VWAP)
       AND rejection candle: bullish engulfing or hammer (lower wick > 2× body)
       AND `body > 0` (close > open)
       AND **trend up confluence**: 1h SMA200 below close, 4h EMA20>EMA50, 15m EMA20>EMA50
   - **SHORT**: mirror with `dev_pct > +0.5%`, bearish engulfing or shooting star,
       trends down
4. Body filter: `|close - open| / range ≥ 0.4`
5. Volume confirmation: `volume_5m ≥ 1.0 × volume_sma20_5m`

**Locked Parameters (NO retuning)**:
```python
LOCKED = {
    'vwap_dev_min_pct': 0.5,
    'body_min_ratio': 0.4,
    'volume_mult': 1.0,
    'wick_to_body_min': 2.0,  # for hammer/shooting-star confirmation
    'session_reset_hour_utc': 0,
}
```

**Theory source**: VWAP-anchored mean reversion (institutional intraday standard). Multi-timeframe confluence per user's strict spec.

---

## Exit Framework (constant across rounds)

`run_bt_c1_production` from `m3_round30_c1_production_exact.py`:
- trail_K=2.5, max_sl_atr=4.5, sl_min_pct=0.15, sl_max_pct=3.0
- emergency_sl_pct=3.0, max_hold_bars=192 (15m bars equivalent)
- trail_activation_pct=0.05, progressive_trail enabled

**Note**: R38 uses 5m primary timeframe but exit framework is C1's 15m design. 5m bars are ~3× faster, so max_hold_bars in 5m terms = 192*3 = 576 bars equivalent for time symmetry. **This is intentional — keeping exit framework constant ensures evidence accumulates on the entry mechanism, not the exit.**

For practical purposes, R38 will run on 15m timeframe **with 5m-derived signals** projected to 15m bars (signal at the 15m bar containing the 5m signal). This allows exit framework reuse without re-deriving full intrabar logic.

**Decision**: To preserve the locked exit framework integrity AND honor the "5m scalping" user spec, R38 will use **15m candles for exit** but **VWAP computed on 5m granularity** for entry signal. This keeps signals fresh while exit consistent.

---

## Pre-Registered Tests (ALL three required to pass)

### Test 1: WF 5-fold Expanding
- 5 expanding test windows, locked params, friction 0.07
- **Pass**: ≥3/5 folds daily_net > 0

### Test 2: Bootstrap 1000 × 3-day
- random.seed=42, friction 0.07
- **Pass**: pos_rate ≥ 50%

### Test 3: Train/Test 60/40
- Friction 0.07
- **Pass**: BOTH train AND test daily_net > 0

---

## Verdict Logic

- **ALL 3 PASS** → call advisor IMMEDIATELY before claiming breakthrough
- **ANY FAIL** → 7th rigorous negative committed, R38 mechanism dropped permanently

---

## Anti-Adjustment Provisions

1. No sweep, no retuning if FAIL
2. No friction relaxation
3. No criterion relaxation
4. No mechanism swap as alternative
5. ALL params locked from theory before any data observation

---

## What this test cannot establish (even on PASS)

- BT-LIVE parity unverified (C1 LIVE -12.86% on same envelope is the prior)
- Sample size warnings still apply
- Advisor will scrutinize test design before any claim

---

## Hash anchor

Committed BEFORE results observation. Any deviation = full retraction.
