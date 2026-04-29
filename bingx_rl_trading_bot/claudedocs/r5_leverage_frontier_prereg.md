# R5 + Leverage Frontier — Ruin Probability Characterization

**Date pre-registered**: 2026-04-30
**Status**: PRE-COMMIT (locked before simulation code)
**Track**: R5 single-coin cash-and-carry + leverage as amplifier

---

## DISCLOSURE

R5 (BTC cash-and-carry, 1×) is the only structure in 27 prior rounds (R1-R23 +
4 L2 microstructure features) where edge > friction. Result: 6/7 PASS, T4 borderline
FAIL by 0.72pp. Net 3.28%/yr APY at 1× = ~$49/yr on $1500.

User's stated framing (2026-04-30): "build 0.2%/day structure, leverage to 0.5%/day."
At 1× R5 = 0.009%/day. Per-notional friction = capital-friction × L (linear). Per-notional
edge unchanged by leverage. Therefore the only knob that scales R5's daily yield IS
leverage. Question: at what L does tail risk (basis squeeze + perp liquidation)
flip the answer to negative?

This is not R14 in the closed envelope. It is risk characterization of the only
verified edge with leverage as amplifier per user's plan.

---

## Theory Anchor

1. **Hull (2017) "Options, Futures, and Other Derivatives"** — basis convergence + carry.
2. **BingX margin mechanics**: cross-margin = unified collateral pool, perp loss netted
   against spot gain. Maintenance margin typical 0.5-1% for BTC at retail tier.
3. **Liquidation rule (assumed cross-margin)**: position liquidated when
   (collateral + unrealized PnL) / (notional × maintenance_margin_ratio) < 1.
   For delta-neutral hedge: net loss is purely basis squeeze magnitude × L.
   Liquidation threshold ≈ (1 - maintenance_margin) / L for isolated leg.

---

## Locked Parameters

```python
LOCKED = {
    'capital_usd': 1500,
    'leverage_levels': [1, 2, 3, 5, 7, 10, 15, 20, 30],
    'maintenance_margin_pct': 0.50,    # 0.5% conservative for BingX cross-margin BTC
    'taker_friction_per_side_pct': 0.05,
    'maker_friction_per_side_pct': 0.02,
    'spot_friction_per_side_pct': 0.10,
    'perp_friction_per_side_pct': 0.04,
    'entry_threshold_apy_pct': 3.0,
    'exit_threshold_apy_pct': 0.0,
    'lookback_funding_days': 7,
    'min_simulations_per_leverage': 1000,
}
```

Logic per leverage L:
1. Load 800d funding (BTC) + 250d spot/perp daily basis (BingX, overlap subset)
2. For each day: if 7d trailing funding APY ≥ 3% and not in pos → enter
   (long spot $1500 × L/2, short perp $1500 × L/2). If ≤ 0% → exit.
3. Daily P&L on capital = funding_pnl × L − basis_drift × L − friction × L (entry/exit only)
4. Liquidation event: if intraday |basis_swing| > (1 − maintenance_margin) / L,
   position liquidated → capital = 0 (ruin event for that simulation path).
5. Bootstrap 1000 paths sampling 365 consecutive days from history.

---

## Pre-Registered Outputs

### Per-leverage table

| L | mean_annualized_net_pct | mean_daily_pct | ruin_prob_per_year | Sharpe | adjusted_yield = E(yield) × (1 − ruin_prob) |
|---|------------------------|----------------|--------------------|----|--------------------------------------------|

### Decision criteria (LOCKED)

**DEPLOYABLE** if a leverage L exists where ALL hold:
- mean_daily_pct ≥ 0.20% (T4 hard gate per user)
- ruin_prob_per_year ≤ 0.01 (1% per year — tolerable tail)
- adjusted_yield > R5 1× baseline (3.28%/yr)

**SUB-DEPLOYABLE** if a leverage L exists where:
- mean_daily_pct ∈ [0.05%, 0.20%) AND ruin_prob ≤ 0.01

**NOT DEPLOYABLE** if for all L:
- mean_daily_pct < 0.20% OR ruin_prob > 0.01 at all L meeting yield

---

## Anti-Adjustment

Leverage levels {1, 2, 3, 5, 7, 10, 15, 20, 30}, maintenance margin 0.5%, ruin threshold
1%/yr LOCKED. No retuning post-FAIL. If maintenance margin assumption wrong (BingX docs
verification), result reported as conditional on that assumption.

---

## Critical Caveats (logged before run)

1. **Cross-margin assumption**: requires verification via BingX docs. If isolated margin
   only, liquidation behavior is asymmetric (perp side blows up at smaller adverse move).
   Worst-case liquidation threshold under isolated = (1 − maintenance) / L on perp leg
   alone, while spot leg still requires its own collateral.

2. **Daily basis swing data is candle-aligned**: BingX spot vs perp daily candles may not
   align timestamps. True intraday basis std could be smaller (1h check showed 0.0097%
   std, vs daily candle 1.96% std). Worst-case use observed daily candle data; report
   sensitivity to alternative tail estimates.

3. **Liquidation cascade vs single event**: 800-day history likely undersamples tail.
   Bootstrap from observed distribution = backward-looking. Forward tail (e.g., basis
   squeeze beyond observed max) not modeled — report this as risk floor, not ceiling.

---

## Hash Anchor

Committed BEFORE simulation code.
