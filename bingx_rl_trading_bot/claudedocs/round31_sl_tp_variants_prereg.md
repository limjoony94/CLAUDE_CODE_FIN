# Round 31 — Two SL/TP Mechanism Variants (Period-Level + Symmetric Candle)

**Date pre-registered**: 2026-04-30
**Status**: PRE-COMMIT (locked before code)
**Track**: User-specific 2 variants pre-specified together

---

## DISCLOSURE

User asked: "기간 내 최저/최고점까지 거리, 또는 진입 시점 캔들 대칭 포지션에 tp sl
넣는 시도는 확인해 보았나?"

These specific TP/SL mechanisms NOT directly tested in 60 prior configs:
- R29 used period_range × multiplier (NOT absolute period extreme level)
- R28 used candle_range × asymmetric multiplier (R:R 2.0, NOT symmetric)
- R30 grid varied within R29's framework, not these alternative methods

R31 = two pre-specified variants on R29 baseline (15m, lookback 16, fade, 50% body).
Both run, both reported, no cherry-pick.

---

## Variant Definitions

### R31a — Period Extreme as Absolute Price Level

Fade SHORT (entry above period_high):
- TP_price = period_low (absolute level, full mean-reversion target)
- SL_price = period_high + 0.5 × ATR(14)

Fade LONG (entry below period_low):
- TP_price = period_high (absolute level)
- SL_price = period_low − 0.5 × ATR(14)

R:R varies by setup (depends on entry overshoot above period_high).

### R31b — Symmetric SL/TP around Entry Candle

Fade SHORT:
- TP_distance = 1.5 × entry_candle_range (down from entry)
- SL_distance = 1.5 × entry_candle_range (up from entry)

Fade LONG: mirror.

R:R = 1.0 fixed.

---

## Locked Parameters

```python
LOCKED = {
    'asset': 'BTC/USDT',
    'tf': '15m',
    'period_lookback_bars': 16,           # 4h
    'body_combined_min_pct_of_range': 0.50,
    'direction': 'fade',
    'max_hold_bars': 96,                  # 24h
    'taker_per_side_pct': 0.05,
    'maker_per_side_pct': 0.02,
    'capital_usd': 1500,

    # Variant A
    'a_sl_atr_buffer_mult': 0.5,
    'a_atr_period': 14,

    # Variant B
    'b_candle_range_multiple': 1.5,       # symmetric TP=SL
}
```

---

## Pre-Registered Tests (User's 4 criteria)

For each variant independently:
- C1 (HARD) Daily ≥ 0.20%
- C2 Per-trade gross > 0.07%
- C3 Trade count ≥ 100
- C4 Bootstrap pos_rate ≥ 50%

Bonferroni-aware with 2 variants: p_per_variant = 0.025 (interpretation aid only).

---

## EV Estimate

| Outcome | R31a | R31b |
|---------|------|------|
| All 4 PASS daily ≥ 0.20% | 5-10% | 3-7% |
| Sub-target positive | 15-20% | 15-20% |
| Catastrophic fail | 25-30% | 25-30% |
| Borderline | 40-50% | 45-55% |

**Realistic prior**:
- R31a wide TPs may struggle to fill within 24h (timeout heavy). High R:R but
  low realized hit rate likely.
- R31b R:R 1.0 requires WR > 50% to be profitable. Fade in 2024-2026 BTC
  showed WR ~35-40% in prior rounds. Likely negative.

---

## Anti-Adjustment

Variants A and B locked. NO post-hoc winner selection. Both reported. If both fail,
the SL/TP mechanism class is empirically tested negative in these two specific
forms. Further variants require separate pre-reg.

---

## Hash Anchor

Committed BEFORE code.
