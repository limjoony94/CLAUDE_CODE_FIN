# Path B R1-maker Pre-Registration — XS Momentum at Maker Friction

**Date pre-registered**: 2026-04-29
**Status**: PRE-COMMIT
**Track**: Path B R1 with friction parameter changed (advisor delegation decision)

---

## What's new

Per advisor delegation (user re-delegated decision after stating "0.019%/day too small"):
1. **New gate**: avg_daily_net ≥ 0.05%/day at 1× leverage (~18% annualized)
2. **Friction change**: 0.07% (taker round-trip) → 0.04% (maker round-trip equivalent, conservative)
3. **Mechanism unchanged**: same Jegadeesh-Titman 1993 + Liu-Tsyvinski 2021 cross-sectional momentum
4. **Universe unchanged**: BTC/ETH/SOL/BNB/XRP/ADA/DOGE/AVAX/TRX/LINK
5. **All other params unchanged**: lookback=30d, long top-3, short bottom-3, weekly rebalance

This is a **single scoped re-evaluation**, NOT a new factor (R3). The only change is friction parameter.

---

## Why this specifically

Per advisor:
> "Path B R1 demonstrated 0.019%/day net at retail taker friction. At maker-only friction (~0.02%/transaction vs 0.07%), that same mechanism would clear roughly 0.04-0.05%/day net. So 0.05% is 'what the evidence already shows is plausibly achievable.'"

> "Just simulate maker friction in the existing R1 backtest by changing the friction parameter — that's a 5-minute test that tells you whether the friction-reduction direction is even worth the engineering investment. Walk before you run."

Engineering reality: BingX/Binance maker fee on perpetual = 0.02% per side = 0.04% round-trip. Used as the friction parameter here. Conservative — actual maker rebates can be lower with tier benefits.

---

## Locked Parameters (only friction changes from R1)

```python
LOCKED = {
    'universe_size': 10,
    'lookback_days': 30,                  # unchanged from R1
    'long_top_n': 3,                       # unchanged
    'short_bottom_n': 3,                   # unchanged
    'rebalance_frequency_days': 7,         # unchanged
    'friction_per_transaction': 0.04,      # CHANGED from 0.07
    'equal_weight': True,
}
```

---

## Pre-Registered Tests (same as R1, NEW gate added)

### Pre-run Dispersion Gate (same)
- Median 30d cross-sectional dispersion ≥ 5%

### Test 1: WF 5-fold Expanding (same)
- **Pass**: ≥3/5 folds avg_weekly_net > 0

### Test 2: Bootstrap 1000 × 30-day NET-return windows (same)
- **Pass**: pos_rate ≥ 50%

### Test 3: Train/Test 60/40 (same)
- **Pass**: BOTH train AND test avg_weekly_net > 0

### Test 4 (NEW — daily-net gate)
- **avg_daily_net ≥ 0.05%** on full sample (1× leverage)
- This is the floor advisor decision implements

---

## Verdict Logic

- **Dispersion FAIL**: inconclusive
- **All 4 PASS**: paper trade candidate at 1× for 4 weeks while L2 collects (advisor instruction)
- **Test 4 alone fails (other 3 pass)**: edge real but smaller than new floor — escalate to advisor (envelope frontier near)
- **Test 4 alone passes (others fail)**: cannot happen logically (Test 4 implies edge → at least Test 3 train)
- **Test 1/2/3 fail with Test 4 PASS**: investigate (probably regime concentration on positive folds)
- **All fail**: friction reduction insufficient to clear new floor → 0.05% likely also not achievable on retail BTC envelope → escalate

---

## Anti-Adjustment Provisions

1. **Friction parameter is the ONLY change** vs R1
2. No other retuning if FAIL
3. No mechanism swap as response to FAIL — different mechanism = R3
4. No further friction reduction post-hoc (e.g., 0.04 → 0.02)

---

## Honest caveats

Even if Test 4 passes:
- Maker-only execution requires limit-order infrastructure (2-4 weeks dev)
- Limit orders can fail to fill in fast markets, eroding the friction advantage
- 0.05%/day at 1× = 18% annual is competitive but small absolute return on $1,500 capital ($270/year)
- 30-day lookback retains regime sensitivity from R1 (folds 2/5 negative)

These are deployment concerns, not validation concerns. Surface in synthesis post-result.

---

## Hash anchor

Committed BEFORE rerun. Friction parameter is the ONE locked change.
