# Round 25 — R8 1h Donchian + Maker-Only Execution

**Date pre-registered**: 2026-04-30
**Status**: PRE-COMMIT (locked before strategy code)
**Track**: User-specified constraint relaxation (maker entry/TP, taker SL)

---

## DISCLOSURE

29 prior investigations exhausted within taker-friction substrate. R5 only deployable
result (3.28%/yr, fails strict T4/T7).

User specified (2026-04-30): "진입할 때는 maker, exit 할 때 (수익시) limit 주문.
stop loss 시에는 불가피한 taker 주문."

This pre-reg implements that spec on R8 baseline (the highest-EV directional from prior
rounds, +0.04% gross/trade) to measure whether maker-entry filter shifts edge sign.
Per advisor: do NOT pre-design grid trading; do one round at a time.

LIVE-parity prior 0/1 (C1).

---

## What's distinct from R8 (Round 18, 2026-04-)

R8 used **taker market** entry/exit at signal bar close. This R25 changes ONLY the
fill model:

| Aspect | R8 baseline | R25 maker-only |
|--------|-------------|----------------|
| Entry | market (taker 0.05%) | limit (maker 0.02%), 1-bar max wait |
| TP exit | market (taker 0.05%) | limit (maker 0.02%) |
| SL exit | market (taker 0.05%) | market (taker 0.05%) |
| RT mean (40% WR) | 0.10% | 0.058% |
| Strategy logic (entry signal, SL, TP) | unchanged | unchanged |
| Pivot lookback, ATR period, etc. | unchanged | unchanged |

This is fill-model substitution, not new mechanism.

---

## Theory Anchor

1. BingX V2 fee schedule: perp taker 0.05%/side, maker 0.02%/side (Standard tier).
2. Limit-order fill bias literature (Glosten-Milgrom 1985 adverse selection):
   limit orders are filled when price is moving against you, biasing fills toward
   weak setups. **This is the empirical question R25 measures.**
3. Hasbrouck (1991) "Measuring information content of stock trades" — order
   placement information leakage.

---

## Locked Parameters

```python
LOCKED = {
    'asset': 'BTC/USDT',
    'tf': '1h',
    'channel_lookback_bars': 15,           # R8 original
    'body_min_pct_of_range': 0.40,         # R8 original
    'atr_period': 14,                      # R8 original
    'fractal_lookback': 5,                 # R8 original
    'fractal_atr_mult_cap': 3.3,           # R8 original
    'trail_atr_mult': 2.5,                 # R8 original
    'max_hold_bars': 192,                  # R8 original (8 days at 1h)

    # R25 maker semantic (LOCKED)
    'limit_entry_offset_pct': 0.05,        # signal_close - 0.05% (long), + 0.05% (short)
    'limit_max_wait_bars': 1,              # if not filled in 1 bar, no trade
    'maker_friction_per_side_pct': 0.02,
    'taker_friction_per_side_pct': 0.05,

    'capital_usd': 1500,
    'position_size_usd': 1500,
}
```

Logic per signal at bar t:
1. R8 entry signal: close[t] > 15-bar high AND body > 40% of range → bullish breakout
2. Place limit BUY at close[t] − 0.05%·close[t]  (LONG case)
3. Wait at most 1 bar (t+1):
   - If low[t+1] ≤ limit_price → FILLED at limit_price (maker, 0.02% cost)
   - Else → NO TRADE
4. SL: fractal swing point (capped 3.3×ATR), executed as market when hit (taker 0.05%)
5. TP: trailing 2.5×ATR (limit, maker 0.02% when filled)
6. Max hold 192 bars; timeout exit at market (taker 0.05%)

For SHORT: limit SELL at close[t] + 0.05%·close[t], filled if high[t+1] ≥ limit_price.

---

## Pre-run Gates

### Gate A — Sufficient setups
- ≥ 100 R8-style breakouts in 720d → expected from prior R8 result
- **Fill rate** measured: how many of the breakout signals actually get filled at the
  maker offset within 1-bar wait. Reported but not gating.

### Gate B — Random-baseline
- 1000 random-bar simulations using same maker/taker fill semantics
- **Pass**: actual cum_net > 95th percentile of random

---

## Pre-Registered Tests

T1 WF 5-fold (≥3/5 positive)
T2 Bootstrap 1000 × 3-day (pos_rate ≥ 50%)
T3 Train/Test 60/40 (BOTH positive)
T4 (HARD) daily ≥ 0.20%
T5 WR ≥ 30% (relaxed via A)
T6 R:R ≥ 1.0 realized
T7 (HARD) trades/day ≥ 2
T8 (HARD) per-trade gross ≥ 0.07% (taker baseline) — **NOTE**: with maker semantic the
    relevant comparison is per-trade gross > weighted_friction (0.058% at WR=40%).
    Report both.
T9 worst 5d ≥ -15%

---

## EV Estimate (per advisor pattern)

| Outcome | Probability | Justification |
|---------|------------|---------------|
| Maker-entry shifts edge POSITIVE | 15-20% | Limit fills act as price-improvement filter; could elevate per-trade gross |
| Maker-entry shifts edge NEGATIVE | 30-40% | Adverse-selection bias (Glosten-Milgrom): limit fills when adverse |
| Maker-entry neutral but friction reduction enables PASS | 10-15% | If gross stays +0.04%, lower friction net positive |
| Fill rate too low (most signals miss) | 25-30% | Breakout setups have momentum; limit-buy below close often unfilled |
| Mixed / borderline | 10-15% | T7 frequency drops with low fill rate |

**Realistic prior**: T7 fill-rate-driven FAIL most likely. T4 magnitude could improve
slightly from R8's -0.06% baseline but unlikely to clear 0.20%.

---

## Anti-Adjustment

Limit offset 0.05% LOCKED. Max wait 1 bar LOCKED. R8 strategy parameters unchanged.
**No retuning post-FAIL.** If FAIL, infer maker-entry doesn't change envelope and apply
that finding to other rounds without re-running each.

---

## Hash Anchor

Committed BEFORE strategy code.
