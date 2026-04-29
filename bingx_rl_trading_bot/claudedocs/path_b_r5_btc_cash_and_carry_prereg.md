# Path B R5 — BTC Single-Coin Cash-and-Carry Basis Harvest

**Date pre-registered**: 2026-04-29
**Status**: PRE-COMMIT (locked before backtest code; data already on disk from R3)
**Track**: Path B R5 — Round 16 (genuinely new alpha class, distinct from R3/R4)
**Authority**: User redirect 2026-04-29 ("다음 라운드 진행, 적어도 은행 이자보다는 더 높은 수익, 창의적")

---

## What's distinct from R3/R4

R3 (10-coin) and R4 (27-coin) were **cross-sectional** carry — required
dispersion across coins to harvest the spread. Both vacuum-FAILed (universe
dispersion structurally too thin in retail-tractable perps).

**R5 is single-coin cash-and-carry**: long BTC spot + short BTC perpetual,
equal dollar notional, **delta-neutral**. Harvests BTC's own funding rate
without needing cross-section. The mechanism does not depend on dispersion —
it depends only on BTC's funding being positive on average.

R3/R4 already verified: BTC mean funding rate = +0.0065%/8h = **+7.08%/yr
annualized**. This is structurally above 4% Korean bank deposit baseline.

---

## Theory Anchor

1. **Cash-and-carry arbitrage** (Working 1949 commodities; Hull "Options,
   Futures, and Other Derivatives" Ch. 5): the equilibrium between spot
   price + carry cost and futures/forward price. When perp trades above
   spot (positive funding), longs pay the implied carry; cash-and-carry
   harvests this by holding spot + shorting perp.

2. **Bianchi, Babiak, Dickerson (2023) "Trading volume and liquidity
   provision in cryptocurrency markets"** documents persistent positive
   funding regimes on major perp pairs as compensation for liquidity
   provision.

3. **Mechanism economic story**: positive funding on BTC perpetuals is
   compensation paid by leveraged longs to liquidity providers (shorts).
   By holding spot 1× long + perp 1× short, the position is delta-neutral
   with respect to BTC price but accumulates funding payments equal to
   the rate × notional × periods held.

4. **Why single-coin works where cross-sectional didn't**: We don't need
   dispersion. We need *level*. BTC mean funding +7%/yr is the level.
   The friction is ONE entry + ONE exit, not weekly rebalance.

---

## Locked Parameters

```python
LOCKED = {
    'asset': 'BTC',
    'leg_long_spot': True,       # long BTC spot via BingX
    'leg_short_perp': True,      # short BTC/USDT:USDT perp via BingX
    'notional_balance_pct': 1.0, # equal $ notional both legs (delta-neutral)
    'rebalance_trigger': 'funding_regime_filter',
    'entry_threshold_apy_pct': 3.0,   # enter when 7d trailing funding APY ≥ 3%
    'exit_threshold_apy_pct': 0.0,    # exit when 7d trailing funding APY ≤ 0%
    'spot_long_friction_per_side_pct': 0.10,   # BingX spot taker fee
    'perp_short_friction_per_side_pct': 0.04,  # BingX maker fee on perp short (limit order)
    'capital_usd': 1500,
    'spot_position_usd': 750,    # half on spot long
    'perp_position_usd': 750,    # half on perp short (1× leverage)
}
```

**Note on friction**: spot taker on BingX is 0.10% per side (round-trip 0.20%);
perp maker is 0.02-0.04% per side (round-trip 0.04-0.08%). Combined entry+exit
across 2 legs: ~0.28% one-time setup cost. Amortized over ≥1 month holding
this is 0.28% / 30 = ~0.0093%/day in friction.

---

## Pre-Run Gates

### Gate A — Funding regime existence
- Total days where 7d trailing BTC funding APY ≥ 3% over panel
- **Pass**: ≥ 200 days (~25% of 800-day panel)
- **Fail**: positive carry regime too rare to harvest

### Gate B — Funding sign stability
- Fraction of 7d windows where mean funding rate > 0
- **Pass**: ≥ 70% of windows
- **Fail**: regime too unstable for buy-and-hold harvest

---

## Pre-Registered Tests (5 gates, adapted for low-frequency hold)

### Test 1: WF 5-fold Expanding
- Per fold: cumulative net % from carry harvest
- **Pass**: ≥3/5 folds with cum_net > 0

### Test 2: Bootstrap 1000 × 30-day Net-Return Windows
- Sample 30-day windows from net daily P&L stream
- **Pass**: pos_rate ≥ 50%

### Test 3: Train/Test 60/40
- **Pass**: BOTH train and test cum_net > 0

### Test 4 (Magnitude — RELAXED to user's new bar): annualized net APY ≥ 4%
- Korean bank base deposit ~3-4%, target "at least higher than bank"
- **Pass**: full sample annualized_net_apy ≥ 4.0%

### Test 5 (Tail-risk): worst 5-day net return ≥ -3%
- Cash-and-carry is delta-neutral so tail should be small
- **Pass**: worst_5d_net_pct ≥ -3.0%

---

## Verdict Logic

| Outcome | Interpretation |
|---------|----------------|
| Gate A fail | Positive funding regime too rare; cash-and-carry vacuous |
| Gate B fail | Funding sign too unstable for hold-based harvest |
| All 5 PASS + tail negligible | **First non-trivial deployable strategy in the arc**. Surface for paper-deploy |
| T4 fail (< 4% APY) but others pass | Premium too small after BingX retail friction |
| T5 fail | Hidden tail (e.g., funding regime flip) — investigate before deploy |
| Mixed | Standard regime fragility |

---

## Anti-Adjustment Provisions

1. Asset = BTC only. Adding ETH would be R6, not R5-tweaked.
2. Entry/exit thresholds (3% / 0%) locked. Not a sweep.
3. Friction values (0.10% spot taker, 0.04% perp maker) locked. Use of
   BingX maker rebate (lower) requires R6.
4. **No retuning post-FAIL.** If R5 fails, that's the result.
5. Holding period emerges from funding regime data — not pre-set duration.

---

## Honest Caveats

1. **Spot-perp basis risk**: perp price can deviate from spot during stress.
   Delta-neutrality is approximate; spot-perp spread can widen 0.5-2%
   during BTC sell-offs. Not modeled here. Real deploy requires
   margin-cushion budget.
2. **Funding regime can reverse**: Apr-May 2022 (Terra collapse), Mar 2023
   (USDC depeg), Aug 2023 (BTC -10% wick) — historic episodes of
   negative funding for days. Test 5 should catch these in our 800-day
   sample.
3. **BingX-specific friction**: numbers used are from BingX public fee
   schedule; user has not verified own fee tier.
4. **Liquidation risk**: short perp at 1× leverage on $750 notional
   requires sufficient USDT margin. If BTC rallies 20%, the perp short
   is at -20% on margin, requiring margin top-up. We assume isolated
   1× collateral that auto-deleverages — needs real-world margin
   verification.
5. **Position-size capital efficiency**: $1,500 split into $750 spot +
   $750 perp margin halves the effective capital deployed. Net APY on
   $1,500 will be ~half the funding rate.

---

## Hash Anchor

Committed BEFORE strategy code. Result file timestamps post-this commit
anchor anti-snooping evidence.
