# Path B R12 — Time-of-Day Calendar Session Momentum

**Date pre-registered**: 2026-04-29
**Status**: PRE-COMMIT (locked before strategy code)
**Track**: Path B R12 — Round 22 (user delegation D continued)

---

## DISCLOSURE

21 rounds + 4 round-16-attempts. Frequency-edge frontier (R9) +
direction-agnostic ceiling (R10+R11) characterize alpha space in 2D.
LIVE-parity prior 0/1 (C1).

R12 is research artifact, not deploy candidate.

---

## What's distinct from prior 21 rounds

R12 uses **time-of-day** as PRIMARY signal — not used in any prior round.
All 21 prior used price-action / order-flow / cross-section / carry / lead-lag
signals. R12 says: trade specific hours regardless of price level.

---

## Theory Anchor

1. **Hansen & Lunde (2005) "A forecast comparison of volatility models"** —
   document day-of-week and time-of-day effects in financial volatility.
2. **Bouri, Lau, Lucey (2019) "Trading volume and market efficiency in
   cryptocurrency markets"** (J Risk Fin) — crypto markets show systematic
   volume patterns by trading session.
3. **Mechanism economic story**: London open (07:00 UTC) and US open
   (13:00 UTC) bring institutional flow. Asian session (00:00-07:00 UTC)
   is retail-dominated, low volume. The transition from Asian → London
   creates predictable liquidity injection that often biases price up
   in BTC bull regimes (or down in bear). US open similarly.

---

## Locked Parameters

```python
LOCKED = {
    'asset': 'BTC/USDT',
    'tf': '1h',
    'session_1_entry_hour_utc': 7,       # London open
    'session_2_entry_hour_utc': 13,      # US open
    'hold_bars': 4,                       # 4 hours
    'trend_filter_sma_periods': 24,       # 24h SMA on 1h bars
    'friction_pct': 0.07,                 # taker
    'capital_usd': 1500,
    'position_size_usd': 1500,            # full position per entry (max 2/day, no overlap)
}
```

Logic:
1. At 07:00 UTC each day: if 1h close > 24-bar SMA → enter LONG, hold 4h, exit at 11:00 UTC.
2. At 13:00 UTC each day: same condition → enter LONG, hold 4h, exit at 17:00 UTC.
3. Symmetrical SHORT entries when close < 24-bar SMA (downtrend).
4. Friction: 0.07% × 2 = 0.14% RT per entry.

---

## Pre-run Gates

### Gate A — Trend filter retention
- At 07:00 and 13:00, fraction of days where trend filter active
- **Pass**: ≥ 50% of days have signal (longs OR shorts)

### Gate B — Random-baseline (anti-fix-impulse)
- 1000 random-hour entries each day (random hour 0-23)
- **Pass**: actual cum_net > 95th percentile of random

---

## Pre-Registered Tests (Korean criteria, A+D+E user trade-offs)

T1 WF 5-fold (≥3/5 positive)
T2 Bootstrap 1000 × 3-day (pos_rate ≥ 50%)
T3 Train/Test 60/40 (BOTH positive)
T4 (HARD) daily ≥ 0.2%
T5 WR ≥ 30%
T6 R:R ≥ 1.0 (no fixed TP/SL — natural realized R:R)
T7 (HARD) trades/day ≥ 2
T8 (HARD) per-trade gross > 0.07% taker
T9 worst 5d ≥ -15%

---

## EV Estimate

- All HARD PASS: 5-8%
- T4 fail (others pass): 30%
- Catastrophic FAIL like R10/R11: 15%
- Mixed: 40-50%

---

## Anti-Adjustment

Hours (07/13), hold (4), trend SMA (24) LOCKED. No retuning post-FAIL.

---

## Hash Anchor

Committed BEFORE code.
