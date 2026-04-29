# Path B R11 — Lo-MacKinlay Extreme-Move Reversal (5m BTC)

**Date pre-registered**: 2026-04-29
**Status**: PRE-COMMIT (locked before strategy code AND before reading BT result)
**Track**: Path B R11 — Round 21 (user delegation D, Lo-MacKinlay 1988 anchor)

---

## DISCLOSURE per advisor mandate

R10 multi-TF breakout produced cum -94% / Sharpe -8.33. This is suggestive
of MR regime in 2024-2026 BTC.

**R11 is NOT a "reverse R10 because R10 lost" strategy.** R11 anchors on
Lo-MacKinlay (1988), a pre-existing theory cited frequently for short-term
mean-reversion. The hypothesis pre-existed R10. R11 tests this hypothesis
on BTC 5m with locked thresholds chosen by academic convention (±1σ),
not by inspecting R10's data distribution.

**Anti-snooping evidence**: this pre-reg locks specific thresholds (0.5% =
~1.5σ for BTC 5m, conventional). Strategy code is written and result
reviewed AFTER this commit. Verifiable via git log timestamps.

---

## What's distinct from prior 20 rounds

| Round | Mechanism | Direction |
|-------|-----------|-----------|
| R1-R10 OHLCV | trend continuation / breakout | with-momentum |
| TT-R1, R7 | continuation / lead-lag | with-direction |
| TT-R2 | extreme single-bar fade | reversal-direction (TESTED) |
| **R11** | **multi-bar reversion at extreme single-bar move** | **opposite direction** |

R11 differs from TT-R2 (which tested 1m extreme fade with continuation
direction). R11 uses 5m timeframe, opposite-direction entry, ATR-based
exits.

---

## Theory Anchor

1. **Lo & MacKinlay (1988) "Stock market prices do not follow random walks:
   Evidence from a simple specification test"** (Rev Fin Stud 1):
   short-term returns have negative autocorrelation in efficient markets,
   driven by overreaction.

2. **Jegadeesh (1990) "Evidence of Predictable Behavior of Security Returns"**:
   monthly stock returns show 1-month reversal followed by momentum.

3. **Crypto extension**: Borri (2019) "Conditional tail-risk in cryptocurrency
   markets" notes BTC daily extreme moves often mean-revert within 1-3 days
   in stable regime.

4. **Mechanism economic story**: BTC 5m extreme moves (≥1.5σ from mean)
   are typically driven by liquidations or stop-cascades. The forced
   selling overshoots equilibrium. Reversion within 30 minutes captures
   the rebound.

---

## Locked Parameters

```python
LOCKED = {
    'asset': 'BTC/USDT',
    'primary_tf': '5m',
    'extreme_move_threshold_pct': 0.5,    # 5m return |≥| 0.5% (~1.5σ)
    'entry_at_bar': 'next',                # enter at next 5m bar open
    'direction': 'reversal',               # opposite of triggering bar's sign
    'atr_period': 14,                      # 5m bars
    'sl_atr_mult': 0.5,                    # tight stop
    'tp_atr_mult': 1.0,                    # aggressive target
    'max_hold_bars': 6,                    # 30 minutes
    'cooldown_bars': 3,                    # 15 min after exit
    'friction_pct': 0.07,                  # taker
    'capital_usd': 1500,
}
```

Logic:
1. At each 5m bar t, compute return = close[t] / close[t-1] - 1.
2. If |return| ≥ 0.5%: signal triggers.
3. Direction = OPPOSITE of return sign (reversal).
4. Cooldown check: if last exit < 3 bars ago, skip.
5. Entry at next 5m bar open.
6. SL = entry ∓ direction × 0.5 × ATR(14, 5m)
7. TP = entry ± direction × 1.0 × ATR(14, 5m)
8. Max hold: 6 bars (30 min).
9. Friction: 0.07% × 2 = 0.14% RT.

---

## Pre-run Gates

### Gate A — Sufficient extreme moves
- |5m return| ≥ 0.5% events / total bars
- **Pass**: ≥ 1,000 events over panel
- **Fail**: too rare

### Gate B — Reversion direction validation (informational)
- Check: avg(return at t+1 to t+6) vs sign of return at t for events
- **Pass**: opposite-sign on average (mean-reversion present)
- This is informational, not gating — strategy proceeds either way per
  pre-reg

### Gate C — Random-baseline (anti-fix-impulse binding)
- 1000 random-entry simulations matching trade frequency + direction
- **Pass**: actual cum_net > 95th percentile of random

---

## Pre-Registered Tests (Korean criteria, A+D+E user trade-offs)

### Test 1: WF 5-fold expanding (≥3/5 positive)

### Test 2: Bootstrap 1000 × 3-day (pos_rate ≥ 50%)

### Test 3: Train/Test 60/40 (BOTH positive)

### Test 4 (HARD): daily ≥ 0.2% at 1×

### Test 5: WR ≥ 30% (RELAXED via A)

### Test 6: R:R ≥ 1.5 (R:R locked at 2.0, gate slightly tighter)

### Test 7 (HARD): trades/day ≥ 2

### Test 8 (HARD): per-trade gross > 0.07% taker

### Test 9: tail worst 5d ≥ -15% (RELAXED via E)

---

## EV Estimate (logged before run)

| Outcome | Probability | Justification |
|---------|------------|---------------|
| All hard PASS | 8-15% | reversion theory + R10 regime evidence raises priors |
| T4 fail (others pass) | 25-30% | edge but small magnitude |
| Mixed regime fragility | 25-30% | reversion may work some periods only |
| Catastrophic fail (mirror of R10) | 5-10% | unlikely if MR regime real |
| All pass + clear ceiling break | 5-10% | dream scenario |

**Realistic prior**: T4 borderline FAIL or PASS. R11 outcome will:
- Confirm MR regime (if positive) → meaningful directional finding
- Reveal alpha is small in BOTH directions (if also negative) → user
  hard goals likely unreachable on retail BTC space

---

## Anti-Adjustment Provisions

1. Threshold 0.5% LOCKED. NOT chosen from R10 distribution analysis.
2. ATR multipliers (0.5/1.0) LOCKED.
3. Hold 6 bars LOCKED.
4. **No retuning post-FAIL.**
5. Direction is REVERSAL (theory-driven, not R10-driven).

---

## Hash Anchor

Committed BEFORE strategy code. R10 result IS visible to me at this point
(R10 was committed earlier). The defense against snooping is:
- Theory anchor predates R10 (Lo-MacKinlay 1988)
- Threshold 0.5% chosen by σ convention, not R10 distribution
- Result reviewed AFTER this commit
- Direction (reversal) is academically standard, not derived from R10
