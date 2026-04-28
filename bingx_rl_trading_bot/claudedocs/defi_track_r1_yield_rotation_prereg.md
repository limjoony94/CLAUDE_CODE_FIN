# DeFi-Track R1 Pre-Registration — L2 Yield Rotation Top-3

**Date pre-registered**: 2026-04-29
**Status**: PRE-COMMIT
**Track**: DeFi-Track Week 2 (productive parallel during L2 collector wait)
**Authority**: Advisor delegation 2026-04-29 — explicit user redirect "L2 4주 그냥 기다리지 말고 무엇이든 해"

---

## Mechanism (locked, single design)

**Top-3 Trailing-APY Yield Rotation on L2 Chains**

Theory: Cross-sectional yield dispersion in DeFi lending/LP pools is sustained
(Week 1 reconnaissance: median 13.81pp top-3 vs bottom-3 spread, 45/51 months
> 10pp). Weekly capital rotation toward the highest-APY pools should harvest
this dispersion *if* friction is low enough.

For $1,500 retail capital constrained to $500 per position, **only L2 chains
are economically viable** (mainnet gas ~$30/swap = 6% friction per swap kills
strategy; L2 gas ~$1-2/swap = 0.2-0.4%).

---

## Locked Universe

**L2-only chains**: `Arbitrum`, `OP Mainnet`, `Base`, `Polygon`

Excludes all Ethereum mainnet pools (gas economic infeasibility) regardless of
APY level. This is a **deployment constraint**, not optimization.

Excludes BSC, Avalanche, Aptos, Mantle, MegaETH, Sonic, Celo, Gnosis, Plasma
(too few pools per chain to maintain rotation universe).

Per Week 1 cohort data: 39 pools across {aave-v3, compound-v3, curve-dex, pendle}
on these chains. Convex absent on L2.

---

## Locked Parameters

```python
LOCKED = {
    'universe': ['Arbitrum', 'OP Mainnet', 'Base', 'Polygon'],
    'lookback_days': 30,                  # trailing APY median
    'top_n': 3,                            # long top-3 by trailing APY
    'equal_weight': True,
    'rebalance_frequency_days': 30,        # monthly (NOT weekly — friction)
    'friction_per_swap_pct': 0.4,          # $2 swap on $500 position = 0.4%
    'min_pool_history_days': 30,           # require lookback bars
    'min_tvl_usd': 1_000_000,              # already filtered, redundant safety
    'capital_usd': 1500,
    'position_size_usd': 500,
}
```

**Friction model**: Each rebalance, count pools that exit top-3 AND new pools
entering. Charge `friction_per_swap_pct` × notional per swap leg. Exits = capital
flowing out. Entries = capital flowing in.

Concretely: if held [A,B,C] and new top-3 [A,B,D], one exit (C) + one entry (D)
= 2 swaps × 0.4% = 0.8% on the $500 position = 0.27% portfolio drag for that
month.

---

## Pre-run Vacuity Gate

- Median monthly eligible pool count ≥ 5 (need universe > top_n for rotation
  to mean anything; avoids "rotation" that's actually fixed selection)
- Mean monthly count of L2 pools with lookback_days ≥ 30 ≥ 8

If either fails → INCONCLUSIVE (vacuous). Park, do not retune.

---

## Pre-Registered Tests (4 OOS gates + 1 tail-risk)

### Test 1: WF 5-fold Expanding (out-of-sample)
- Compute cumulative net return (gross APY − friction) per fold
- **Pass**: ≥3/5 folds with positive cumulative net return

### Test 2: Bootstrap 1000 × 90-day Net-Return Windows
- Sample 90-day rolling windows from full backtest net returns
- **Pass**: pos_rate ≥ 50% (windows with positive cumulative net)

### Test 3: Train/Test 60/40 Sign-Agreement
- First 60% of months = train, last 40% = test
- **Pass**: BOTH train cumulative net AND test cumulative net > 0

### Test 4 (NEW — magnitude gate): avg_daily_net_apy ≥ 0.02%/day full sample
- Rationale: this is the retail BTC envelope ceiling we already established.
  DeFi-Track is interesting *only* if it clears this band, otherwise it's
  redundant evidence at lower volatility.
- Convert: 0.02%/day × 365 = 7.3% net APY annualized
- **Pass**: full-sample net APY ≥ 7.3%

### Test 5 (NEW — tail-risk gate per advisor): worst 5-day net drawdown ≤ 10%
- Identify worst rolling 5-day net return across full sample
- **Pass**: worst_5d_net ≥ -10% (drawdown ≤ 10%)
- Rationale: $150 max drawdown on $1,500 is the real-money ceiling. If a
  catastrophic week (depeg/exploit, Q4 of recon = 0.112%) wipes 20%, strategy
  is undeployable regardless of other metrics.

---

## Verdict Logic

- **Vacuous** (universe too thin) → INCONCLUSIVE, park L2-rotation class
- **All 5 PASS** → R1 candidate, surface to advisor for paper-deploy decision
- **Test 4 fail (magnitude) + others pass** → escalate: "rotation works but
  beneath envelope ceiling — same conclusion as 14-round retail BTC arc"
- **Test 5 fail (tail) + others pass** → escalate: "magnitude there but
  catastrophic, undeployable at $1,500 retail without insurance"
- **All 5 fail** → DeFi-Track envelope falsified at L2 retail, escalate
- **Mixed core 3 fail (1/2/3)** → standard regime fragility, no further
  iteration without advisor

---

## Anti-Adjustment Provisions

1. Universe (L2 chains) locked. Adding Ethereum is a **different strategy**,
   not a tweak.
2. Top_n = 3 locked. Top_5 / Top_2 are different strategies.
3. Lookback 30d locked. 60d / 90d are different strategies.
4. Friction 0.4%/swap locked. Real measured gas can revise post-deployment,
   not pre-OOS.
5. Monthly rebalance locked. Weekly / biweekly are different strategies.
6. No retuning post-FAIL. If R1 fails, that's the result.

---

## Honest Caveats

1. **L2 universe is small** (39 pools). Median monthly eligible count may
   undershoot vacuity gate.
2. **Pendle has high mortality** (33% survives 12m). Top-3 might select
   dying pools that exit in subsequent month.
3. **Catastrophic week base rate 0.112%**: in 4-year backtest with 3 positions
   and 52 weeks, expected catastrophe count = 4×52×3 × 0.00112 = 0.7. So
   tail event is rare but not zero — Test 5 may pass-by-luck on this sample.
4. **Net APY measurement assumes pool APY is realized continuously**.
   Position-level slippage and impermanent loss not in DefiLlama APY data.
5. **Trial regime**: 2022-2026 included Terra collapse (May 2022), FTX (Nov
   2022), USDC depeg (Mar 2023), high-rate environment. Different regime
   risk going forward.

---

## Hash Anchor

Committed BEFORE OOS run. Mechanism, universe, parameters, gates all locked.
Result file timestamps post-commit timestamp = anti-snooping evidence.

---

## What Triggers Advisor Call

1. R1 OOS complete with verdict (PASS / FAIL / VACUOUS / mixed) → advisor call
   for next-step decision.
2. Universe too thin → advisor call for "DeFi tractable for $1,500 retail?"
   architecture question.
3. **No other advisor calls until R1 result.**
