# Path B R9 — BTC Funding-Rate Change Momentum (single-asset, distinct from R3-R5)

**Date pre-registered**: 2026-04-29
**Status**: PRE-COMMIT (locked before strategy code)
**Track**: Path B R9 — Round 19 (mechanism distinct from prior 18 rounds)
**Authority**: User 2026-04-29 explicit position: "BingX 1× 일일 ≥0.2% 전략 발굴 해야 함, 아직 찾아내지 못한것임"

---

## DISCLOSURE per advisor mandate

C1 BT methodology produced LIVE -12.86%/14d failure. R8 BT produced
comprehensive negative (-58% over 721d). Same alpha substrate (BTC OHLCV)
tested across 19 mechanisms — all converge below user's hard 0.2%/day
target. Each new round adds incremental data point at saturating returns.

EV estimate logged pre-result:
  P(R9 clears T4 0.2%/day) ≈ 5-8%
  P(R9 produces informational distinct result) ≈ 70%
  Effort: ~1.5h

R9 is research artifact, not deploy candidate without LIVE validation.

---

## What's distinct from R3, R4, R5

| Round | Signal | Mechanism |
|-------|--------|-----------|
| R3 | Cross-sectional 7d funding LEVEL ranking | long bottom-3, short top-3 |
| R4 | Same as R3, expanded universe | universe 27 coins |
| R5 | BTC single-coin funding LEVEL | delta-neutral cash-and-carry hold |
| **R9** | **BTC single-coin funding CHANGE momentum** | **directional bet on positioning unwind** |

R9 mechanism: when funding rate has SHIFTED significantly over recent
window, this reflects positioning unwind → directional pressure on price.

If 24h Δ(7d funding mean) ≥ +threshold → LONGS becoming more crowded →
expect SHORT pressure (counter-trade) OR continuation (if regime is
strengthening). We test CONTINUATION direction (positions are being
established, not exiting). Empirical: Frino-Liu 2021 reports positive
funding change has slight bullish predictive content at next-day horizon.

---

## Theory Anchor

1. **Frino & Liu (2021) "Crypto perpetual swap markets and funding rate
   dynamics"**: documents predictive content of funding rate changes
   for next-period spot/perp returns at daily horizon.

2. **Hu, Lu, Zhang, Zhuang (2024) "Cross-Section of Crypto Carry"**:
   notes funding-rate momentum as adjacent factor to carry level.

3. **Mechanism economic story**: Funding rate change reflects MARGINAL
   demand. Rising funding = new longs entering (or shorts exiting).
   Marginal demand creates short-term price pressure. Falling funding =
   positions being closed, opposite pressure.

---

## Locked Parameters

```python
LOCKED = {
    'asset': 'BTC/USDT',
    'funding_change_lookback_periods': 21,    # 7 days × 3/day = 21 funding periods
    'change_window_periods': 21,              # Δ(funding mean) over 7d → 7d earlier
    'entry_change_threshold_pct_per_8h': 0.005,  # |Δ| ≥ 0.005%/8h trigger
    'hold_periods': 21,                        # hold 7 days
    'friction_per_transaction_pct': 0.07,      # taker
    'capital_usd': 1500,
    'leverage': 1.0,
}
```

Logic:
1. At each 8h funding period t, compute:
   - current_7d_mean = mean of funding[t-21..t]
   - prior_7d_mean = mean of funding[t-42..t-21]
   - Δ = current_7d_mean - prior_7d_mean
2. If Δ ≥ +0.005%/8h: LONG signal (rising funding = bullish flow)
3. If Δ ≤ -0.005%/8h: SHORT signal (falling funding = bearish flow)
4. Hold for 21 periods (7 days). Exit at market.
5. Friction: 0.07% × 2 = 0.14% RT.
6. Cooldown: don't re-enter while position open.

---

## Pre-run Gates

### Gate A — Sufficient signal events
- |Δ| ≥ threshold events / total over panel
- **Pass**: ≥ 100 entry events
- **Fail**: too few

### Gate B — Predictive direction sanity check
- Check: does funding change t correlate positively with BTC return t+1..t+21?
- **Pass**: positive correlation, even if small
- **Fail**: zero/negative correlation = mechanism wrong direction or absent

### Gate C — Random-baseline (anti-fix-impulse binding)
- 1000 random-entry simulations matching trade frequency
- **Pass**: actual cum_net > 95th percentile of random

---

## Pre-Registered Tests (Korean criteria, A+D+E trade-offs accepted)

### Test 1: WF 5-fold expanding
- **Pass**: ≥3/5 folds positive cum_net

### Test 2: Bootstrap 1000 × 3-day windows (USER REQUIREMENT)
- **Pass**: pos_rate ≥ 50%

### Test 3: Train/Test 60/40 sign-agreement
- **Pass**: BOTH positive

### Test 4 (HARD): daily net ≥ 0.2% at 1×
- **Pass**: avg_daily_net ≥ 0.2%

### Test 5: WR ≥ 30% (RELAXED via A)
- **Pass**: win_rate ≥ 0.30

### Test 6: R:R ≥ 1.0 (NOT RELAXED — natural floor)
- **Pass**: avg_win/|avg_loss| ≥ 1.0

### Test 7 (HARD): trades/day ≥ 2
- 7d hold period × ~0.5 entries/period ≈ 0.5 trades/day → likely FAIL T7
- **Pass**: avg_trades_per_day ≥ 2.0
- ANTICIPATED FAIL: discloseure of structural conflict between hold period and frequency gate

### Test 8 (HARD): per-trade gross > 0.07%
- **Pass**: avg_gross_per_trade > 0.07%

### Test 9 (RELAXED via E): worst 5d ≥ -15%

---

## Verdict Logic

| Outcome | Meaning |
|---------|---------|
| Gate A/B/C fail | Mechanism vacuous |
| All hard PASS | First round 16-attempt to clear T4 in 9-mechanism arc — surface |
| T4 FAIL + others PASS | edge exists, magnitude below ceiling (most likely) |
| T7 FAIL only | known structural conflict; strategy is too low-frequency for user gate |

---

## Anti-Adjustment Provisions

1. Threshold 0.005%/8h LOCKED. Sweep is pre-reg violation.
2. Hold 21 periods (7d) LOCKED.
3. Continuation direction LOCKED. Reverse direction is R10.
4. **No retuning post-FAIL.**

---

## Honest Caveats

1. **Hold period 7d**: averages ~0.5 trades/day → T7 anticipated FAIL.
   This is structural, not bug. Documents conflict between low-freq
   funding signal and user's high-freq gate.
2. **C1 LIVE-failure prior**: same risk. R9 result is research artifact.
3. **Funding signal is slow**: 7d average smooths, lag is multi-day.
   Short-term price reactions may not be captured.

---

## Hash Anchor

Committed BEFORE code. EV pre-logged. LIVE-parity disclosure included.
