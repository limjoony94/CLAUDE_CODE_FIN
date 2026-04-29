# Round 29 — 15m Fade + Period-Range TP/SL + 2-Bar Body Filter

**Date pre-registered**: 2026-04-30
**Status**: PRE-COMMIT (locked before strategy code)
**Track**: User R28 successor — 5 simultaneous mechanism changes consolidated to single locked variant

---

## DISCLOSURE

R28 (1h C1-fade with breakout-candle range targets) failed: per-trade gross +0.021%,
daily -0.05%. User authorized R29 with 5 specific mechanism changes (each
verbatim per user 2026-04-30 message):

1. **TF 15m** (user: "5m 또는 15m 사용하는 방식이 적절해 보임")
2. **Lookback 16 bars = 4 hours** (user: "4시간")
3. **Body filter — 직전 2개 봉 합산 50%** (user verbatim)
4. **TP — 기간 내 반대방향 거리 × multiplier** (user option A; locked at 2.0×)
5. **SL — 기간 내 최저/최대 위치 × multiplier** (user option A; locked at 1.0× = period extreme)

These are combined into ONE locked variant. NO post-hoc parameter sweep on R29.

LIVE-parity prior 0/1 (C1).

---

## What's distinct from R28

| Aspect | R28 (1h fade) | R29 (15m fade) |
|--------|---------------|----------------|
| Timeframe | 1h | **15m** |
| Lookback | 15 bars | **16 bars (= 4 hours)** |
| Body filter | single bar > 40% range | **2-bar combined sum > 50% range** |
| TP target | breakout candle range × 2 | **period(16-bar) range × 2** |
| SL distance | breakout candle range × 1 | **period(16-bar) range × 1** |
| Direction | fade | fade (same) |

5 of 6 mechanism aspects changed. R29 is genuinely structurally distinct.

---

## Theory Anchor

1. **Donchian (1960s) channel breakout** — original. Fade direction tests
   anti-breakout regime (R10 catastrophic finding).
2. **Period extreme as natural support/resistance**: 4-hour high/low typically
   represents recent price boundary. Fade from these boundaries is classic
   range-fade.
3. **2-bar body filter (Pin bar / engulfing)**: combined two-bar move is more
   reliable signal than single bar (less noise). Bulkowski 2008 "Encyclopedia
   of Chart Patterns" — engulfing patterns documented as reversal signals at
   range extremes.
4. **15m timeframe**: tradeoff — more signals than 1h (4× density), less noise
   than 5m (1/3 of 5m signals false). Common scalping TF in retail.

---

## Locked Parameters

```python
LOCKED = {
    'asset': 'BTC/USDT',
    'tf': '15m',
    'period_lookback_bars': 16,             # 16 × 15m = 4 hours
    'body_combined_min_pct_of_range': 0.50, # 2-bar combined body / 2-bar combined range
    'tp_period_range_multiple': 2.0,
    'sl_period_range_multiple': 1.0,        # SL touches period extreme + 1× range overshoot
    'direction': 'fade',
    'max_hold_bars': 96,                    # 24 hours at 15m (= 24h timeout)
    'taker_per_side_pct': 0.05,
    'maker_per_side_pct': 0.02,
    'capital_usd': 1500,
}
```

Logic:
1. At each bar t, compute period_high = max(high[t-15..t-1]) and
   period_low = min(low[t-15..t-1]).
2. Compute period_range = period_high - period_low.
3. Compute 2-bar combined body filter:
   - bar1_body = abs(close[t-1] - open[t-1])
   - bar0_body = abs(close[t] - open[t])
   - bar1_range = high[t-1] - low[t-1]
   - bar0_range = high[t] - low[t]
   - combined_body_pct = (bar1_body + bar0_body) / (bar1_range + bar0_range)
4. Detection:
   - If close[t] > period_high AND combined_body_pct > 0.50 AND close[t] > open[t]:
     → BULLISH breakout signal → fade SHORT
   - If close[t] < period_low AND combined_body_pct > 0.50 AND close[t] < open[t]:
     → BEARISH breakdown signal → fade LONG
5. Entry: market at next bar open (taker 0.05%)
6. SL: SHORT → entry + 1 × period_range; LONG → entry − 1 × period_range
   Hit at market (taker 0.05%)
7. TP: SHORT → entry − 2 × period_range; LONG → entry + 2 × period_range
   Hit at limit (maker 0.02%)
8. Max hold 96 bars (= 24h), timeout exit at market (taker)
9. Cooldown: no overlap (one position at a time)

R:R = 2.0. Sample friction:
- TP exit: 0.05 + 0.02 = 0.07% RT
- SL exit: 0.05 + 0.05 = 0.10% RT
- Timeout: 0.10% RT

---

## Pre-run Gates

### Gate A — Sufficient signals
- ≥ 100 detections in 720d
- **Pass**: ≥ 100 trades

### Gate B — Random-baseline
- 1000 random-bar simulations matching trade count
- **Pass**: actual cum_net > 95th percentile

---

## Pre-Registered Tests (User's 4 criteria 2026-04-30)

C1 (HARD) **Daily ≥ 0.20% net at 1×**
C2 **Per-trade gross > 0.07% taker RT**
C3 **Trade count ≥ 100** (statistical significance)
C4 **Bootstrap 1000 × 3-day pos_rate ≥ 50%**

---

## EV Estimate

| Outcome | Probability |
|---------|------------|
| All 4 PASS daily ≥ 0.20% | 5-10% |
| C1 fail by < 0.10pp | 20-25% |
| Catastrophic fail (anti-edge persistent) | 25-35% |
| Mixed (1-3 PASS) | 30-40% |

**Honest expectation**: 32 prior rounds + R28 fade-directional all show per-trade
gross [+0.001%, +0.05%] at retail bar-level. R29 changes 5 mechanism aspects
simultaneously. The wider TP/SL (period range vs single candle) increases R:R
margin but also increases hold time and exposure to mean-reversion failure
modes. Realistic prior: per-trade gross [+0.01%, +0.08%], could approach
friction floor.

---

## Anti-Adjustment

5 LOCKED params (TF, lookback, body filter, TP mult, SL mult). NO sweep
post-FAIL. If FAIL, this 5-aspect-change variant is empirically tested negative.
Further specific change requires user explicit re-specification.

---

## Hash Anchor

Committed BEFORE strategy code.
