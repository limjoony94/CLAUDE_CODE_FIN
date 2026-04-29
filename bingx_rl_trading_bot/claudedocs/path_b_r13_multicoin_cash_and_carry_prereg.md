# Path B R13 — Multi-Coin Cash-and-Carry Portfolio

**Date pre-registered**: 2026-04-30
**Status**: PRE-COMMIT (locked before strategy code)
**Track**: Path B R13 — Round 23 (user delegation D continued)

---

## DISCLOSURE

22 prior rounds. Frequency-edge frontier (R9), direction-agnostic ceiling (R10+R11),
calendar (R12 outside scope here) characterized 2D alpha space. Only R5 (single-coin
BTC cash-and-carry) reached deployable status: 6/7 PASS, T4 borderline FAIL by 0.72pp.
3.28%/yr net APY ($49/yr on $1500), Sharpe 9.71, MDD 0.16%.

R13 extends R5: parallel multi-coin portfolio. Best-coin selection captures funding
dispersion — single-coin R5 missed this.

LIVE-parity prior 0/1 (C1).

---

## What's distinct from prior 22 rounds

R5 used BTC funding only. R13 uses **funding dispersion across 8 coins**:
- Allocate $1500 across N coins simultaneously, each delta-neutral.
- Best historical APY coins captured (LINK 8.22% > BTC 7.08%).
- Independent funding harvest streams.
- This is NOT R3 (cross-sectional momentum / dispersion betting).
  R13 is N parallel R5s with regime gating.

---

## Theory Anchor

1. **Working (1949) "The Theory of Price of Storage"** — basis convergence at expiry.
2. **Hull (2017) "Options, Futures, and Other Derivatives"** — cash-and-carry parity.
3. **Bianchi, Babiak, Dickerson (2023) "Trading volume and liquidity provision in
   cryptocurrency markets"** — perpetual funding as liquidity premium.
4. **Mechanism**: Each coin's perp trades at premium/discount to spot funded by
   8-hourly settlement. Long spot + short perp = delta-neutral, harvests funding.
   Parallel coins = independent harvests = portfolio diversification.

---

## Locked Parameters

```python
LOCKED = {
    'capital_usd': 1500,
    'coin_universe': ['LINK','DOGE','ADA','ETH','BTC','XRP','SOL','AVAX'],
    # Excluded: BNB (-0.72%/yr), TRX (-0.12%/yr) historically negative
    'allocation': 'equal_weight',                  # $187.50 per pair
    'per_leg_usd': 93.75,                          # $187.50/2 (spot+perp)
    'spot_friction_pct': 0.10,                     # BingX retail spot taker
    'perp_friction_pct': 0.04,                     # BingX retail perp taker
    'regime_filter_apy_pct': 3.0,                  # enter coin if 7d trailing funding APY ≥ 3%
    'regime_exit_apy_pct': 0.0,                    # exit coin when ≤ 0%
    'rebalance_freq_days': 7,                      # weekly per coin, staggered
    'lookback_days': 7,                            # for regime filter
    'data_window_days': 800,                       # full history per R5 spec
}
```

Logic:
1. For each day t, for each coin c in universe:
   a. Compute trailing 7d mean funding APY for c.
   b. If APY ≥ 3% and not in position → enter (long spot, short perp $93.75 each).
   c. If APY ≤ 0% and in position → exit (close both legs).
2. Daily NAV = sum across coins of (spot_pnl + perp_pnl + funding_accrual).
3. Friction logged on entry+exit only (no daily rebalance).
4. Stagger entries: each coin checked daily, no portfolio-level constraint.

---

## Pre-run Gates

### Gate A — Coin universe sufficiency
- ≥ 5 of 8 coins have ≥ 100 days where regime filter active.
- **Pass**: ≥ 5 coins
- **Fail**: insufficient deployment opportunity

### Gate B — Random-baseline (anti-fix-impulse)
- 1000 random-coin/random-entry simulations
- **Pass**: actual cum_net > 95th percentile of random

---

## Pre-Registered Tests (Korean criteria, A+D+E user trade-offs)

T1 WF 5-fold (≥3/5 positive)
T2 Bootstrap 1000 × 3-day (pos_rate ≥ 50%)
T3 Train/Test 60/40 (BOTH positive)
T4 (HARD) daily ≥ 0.2%
T5 WR ≥ 30% (relaxed via A — not strictly applicable to carry; track rebalance hit-rate)
T6 R:R ≥ 1.0 (carry has no fixed TP/SL — natural ratio)
T7 (HARD) trades/day ≥ 2 (entry+exit events count; staggered rebalance = ~2-4/day expected)
T8 (HARD) per-trade gross > 0.07%
T9 worst 5d ≥ -15%

---

## EV Estimate (advisor: ~15-20% T4 PASS — highest of any remaining round)

| Outcome | Probability | Justification |
|---------|------------|---------------|
| All HARD PASS (T4+T7+T8) | 15-20% | best-coin selection raises avg APY ~7-8%, net ~3.5-5.5% |
| T4 fail by < 1pp (deployable) | 35-40% | most likely — gross APY ceiling structural |
| Major regime fail | 15% | 2024-2026 funding regime softens |
| Tail risk realized | 10-15% | basis blowup on single coin |
| Mixed | 15% | partial pass |

**Honest expectation**: T4 borderline FAIL like R5. Multi-coin gives ~50% lift over
single-coin (advisor estimate), pushing to ~5%/yr APY = ~$75/yr — still < daily 0.2%
target by ~25× factor.

---

## Anti-Adjustment

Coin universe (8), regime thresholds (3% in / 0% out), rebalance freq (7d) LOCKED.
No retuning post-FAIL.

---

## Hash Anchor

Committed BEFORE code.
