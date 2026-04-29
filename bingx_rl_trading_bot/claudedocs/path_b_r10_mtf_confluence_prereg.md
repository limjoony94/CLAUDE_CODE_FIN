# Path B R10 — Multi-TF Confluence Breakout (5m + 15m + 1h trend)

**Date pre-registered**: 2026-04-29
**Status**: PRE-COMMIT (locked before strategy code)
**Track**: Path B R10 — Round 20 (user delegation D)
**Authority**: User 2026-04-29: "판단도 위임합니다" (delegated decision)

---

## DISCLOSURE per advisor mandate

19 rounds + 4 round-16-attempts. Frequency-edge frontier observed
(R9 commit f62da33). LIVE-parity prior 0/1 (C1). EV logged below.

R10 is research artifact, not deploy candidate.

---

## What's distinct from prior rounds

| Round | TF | Entry trigger |
|-------|----|---------------| 
| C1 | 15m only | 15m channel breakout |
| R8 | 1h only | 1h Donchian breakout |
| R37 (M3) | 5m only | NR7 + BB squeeze (5m) |
| R36 (M3) | 15m only | EMA pullback (15m) |
| **R10** | **5m + 15m + 1h stacked** | **All three confirm same direction** |

R10 mechanism: stacked confluence requirement is NEW. Prior rounds used
single TF. Theory predicts confluence reduces false breakouts at cost
of frequency.

---

## Theory Anchor

1. **Murphy (1999) "Technical Analysis of the Financial Markets"**:
   multi-timeframe analysis chapter — confluence reduces false positives.
2. **Bulkowski (2008) "Encyclopedia of Chart Patterns"**: empirical
   reliability metrics for breakouts increase with higher-TF confirmation.
3. **Mechanism economic story**: 5m breakout often false (intra-bar noise);
   15m confirms breakout sustained one period; 1h trend ensures larger
   structure agreement. By stacking, signal is rare but stronger.

---

## Locked Parameters

```python
LOCKED = {
    'asset': 'BTC/USDT',
    'primary_tf': '5m',
    'breakout_5m_lookback': 12,           # 12 × 5m = 1h
    'breakout_15m_lookback': 8,           # 8 × 15m = 2h
    'trend_1h_sma_lookback': 24,          # 24h SMA
    'body_min_ratio': 0.40,
    'atr_period': 14,                      # 5m bars
    'sl_atr_mult': 1.0,
    'tp_atr_mult': 2.0,                   # R:R 2.0
    'max_hold_bars': 12,                   # 1 hour max
    'cooldown_bars': 6,                    # 30 min cooldown after exit
    'friction_pct': 0.07,                  # taker
    'capital_usd': 1500,
}
```

Logic:
1. Compute on 5m bars: 12-bar (1h equiv) breakout signal
2. Compute on 15m bars (resampled from 5m): 8-bar (2h equiv) breakout signal
3. Compute on 1h bars (resampled from 5m): close > 24-bar SMA
4. Entry: at 5m bar t, ALL THREE confirm same direction → enter at t+1 5m bar open
5. Body filter on 5m breakout bar: |close - open| / (high - low) ≥ 0.40
6. Cooldown: 6 bars (30 min) after any exit
7. SL = entry ∓ 1.0 × ATR(14, 5m)
8. TP = entry ± 2.0 × ATR(14, 5m)
9. Max hold: 12 bars (1h)
10. Friction: 0.07% × 2 = 0.14% RT

---

## Pre-run Gates

### Gate A — Confluence event frequency
- Bars where ALL three (5m, 15m, 1h) align in same direction
- **Pass**: ≥ 200 events over panel
- **Fail**: too rare to test

### Gate B — Body filter retention
- Post-body filter retention ≥ 30% of confluence events
- **Pass**: meaningful retention

### Gate C — Random-baseline (anti-fix-impulse)
- 1000 random-entry simulations matching frequency + direction
- **Pass**: actual cum_net > 95th percentile of random

---

## Pre-Registered Tests (Korean criteria, A+D+E user trade-offs accepted)

### Test 1: WF 5-fold expanding
- **Pass**: ≥3/5 folds positive

### Test 2: Bootstrap 1000 × 3-day windows
- **Pass**: pos_rate ≥ 50%

### Test 3: Train/Test 60/40
- **Pass**: BOTH positive

### Test 4 (HARD): daily ≥ 0.2%
- **Pass**: avg_daily_net ≥ 0.2%

### Test 5: WR ≥ 30% (RELAXED via A)
- **Pass**: win_rate ≥ 0.30

### Test 6: R:R ≥ 1.5 (R:R is locked at 2.0, gate looser)
- **Pass**: realized rr ≥ 1.5

### Test 7 (HARD): trades/day ≥ 2
- Confluence may make this difficult — anticipated tension

### Test 8 (HARD): per-trade gross > 0.07% taker

### Test 9: tail worst 5d ≥ -15% (RELAXED via E)

---

## EV Estimate (logged before run)

| Outcome | Probability | Justification |
|---------|------------|---------------|
| All hard PASS | 5-10% | confluence theoretically reduces false-positive but historic ceiling |
| T4 fail (others pass) | 30-40% | most likely — gross within friction band |
| T7 fail (frequency too low) | 25-35% | confluence reduces freq |
| Mixed regime | 20-25% | |

Overall: probably **20th data point on alpha-ceiling pile**, not
ceiling-breaker. Pre-logged.

---

## Anti-Adjustment Provisions

1. Lookback periods (12/8/24) LOCKED.
2. ATR multipliers (1.0/2.0) LOCKED.
3. Body filter 0.40 LOCKED.
4. Max hold 12 bars LOCKED.
5. STATIC TP/SL — explicitly avoid C1 TRAILING gap.
6. **NO retuning post-FAIL.**

---

## Hash Anchor

Committed BEFORE strategy code.
