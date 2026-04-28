# Trade-Tape Track R2 Pre-Registration — Extreme Single-Minute Imbalance Fade

**Date pre-registered**: 2026-04-29
**Status**: PRE-COMMIT
**Honest prior**: R1 (persistent imbalance trend-continuation) failed all 3 OOS with avg_gross 0.029-0.050% < friction 0.07%. Same friction-floor pattern as M3-R41. Advisor: R2 should be **structural opposite** — extreme single-bar mean-reversion fade — to disambiguate friction-floor envelope claim from mechanism-specific failure.

---

## Why R2 (after R1 failed)

Per advisor:
> "Persistent imbalance is just trend-continuation in microstructure clothing. ... Run one mean-reversion test on trade-tape features. Specifically: extreme single-minute imbalance (not persistent — opposite hypothesis) with a fade entry. Theory: when retail piles into one direction at extreme intensity in a single bar, mean reversion in the next 5-15 min is the textbook microstructure result (think VPIN-adjacent, Easley/O'Hara)."

**If R2 also fails**: friction-floor hypothesis hardens to near-certainty for retail BTC perp 1m signals. Both directions of microstructure information (continuation R1, exhaustion R2) eaten by friction.

**If R2 passes**: trade-tape envelope has a regime (mean-reversion at extremes) where edge survives friction.

Either outcome is decisive — no R3 in this envelope per advisor.

---

## Locked Mechanism — `entry_extreme_fade_1m`

**Theory**: extreme single-minute taker imbalance + intensity = exhaustion. Subsequent reversion is the textbook microstructure outcome (VPIN, Easley/O'Hara order-toxicity literature).

**Algorithm** (1-min granularity):
1. **Extreme reading at bar i**:
   - `|vol_imbalance_i| >= 0.85` (≈92.5/7.5 split — extreme aggression)
   - `trade_count_i > 90th-percentile of recent 60-min trade_count` (intensity confirmation)
2. **Fade entry direction**:
   - **LONG** when `vol_imbalance_i <= -0.85` (extreme sell exhaustion)
   - **SHORT** when `vol_imbalance_i >= +0.85` (extreme buy exhaustion)
3. **NO body filter, NO MTF trend filter** (per advisor: extreme reading IS the signal; MTF would dilute the regime detection)
4. Project 1-min entry → 15m bar index for exit framework
5. Max 1 entry per 15m bar (consistent with R36-R41/R1)

**Locked Parameters (NO retuning)**:
```python
LOCKED = {
    'imb_extreme_threshold': 0.85,
    'intensity_pctile': 0.90,
    'intensity_lookback_min': 60,
    'min_bars_between_15m': 2,
}
```

**Theory source**:
- VPIN order toxicity (Easley/O'Hara/de Prado 2012): extreme one-sided flow signals informed-trader exhaustion
- 0.85 threshold = 92.5/7.5 split: standard extreme-tail definition (above 95th percentile of typical readings)
- 60-min intensity lookback: captures recent regime baseline without overlap

**Difference from R1** (advisor's "structural opposite" requirement):
- R1: 5-min smoothed imbalance ≥ 0.30 → trade SAME direction (continuation)
- R2: 1-min single-bar |imbalance| ≥ 0.85 → trade OPPOSITE direction (fade)

---

## Exit Framework

`run_bt_c1_production` (constant). Same trail/SL/timeout as all rounds.

---

## Pre-Registered Tests

### Pre-run Vacuity Gate
- Signal frequency ≥ 0.5/day on full sample
- If FAIL → R2 inconclusive

### Test 1: WF 5-fold Expanding
- Locked params, friction 0.07
- **Pass**: ≥3/5 folds daily_net > 0
- **NEW**: log avg_gross at every fold prominently (advisor instruction — friction-floor pattern is now the most informative number)

### Test 2: Bootstrap 1000 × 3-day
- random.seed=42, friction 0.07
- **Pass**: pos_rate ≥ 50%

### Test 3: Train/Test 60/40
- **Pass**: BOTH train AND test daily_net > 0
- Log avg_gross prominently

---

## Verdict Logic

- **Vacuity FAIL**: R2 inconclusive
- **Vacuity PASS + ALL 3 OOS PASS**: call advisor before any breakthrough claim
- **Vacuity PASS + ANY OOS FAIL**: trade-tape envelope likely friction-bound (combined with R1)

---

## Anti-Adjustment Provisions

1. No sweep, no retuning if FAIL
2. No friction relaxation
3. No criterion relaxation
4. No mechanism swap as response to FAIL — different family = closure update, not "R3 v2"
5. ALL params locked from theory before any data observation
6. **NO R3 in trade-tape envelope per advisor** — closure update if both R1 and R2 fail

---

## Hash anchor

Committed BEFORE results observation. Aggregator output unchanged from R1; no parameter retuning permitted.
