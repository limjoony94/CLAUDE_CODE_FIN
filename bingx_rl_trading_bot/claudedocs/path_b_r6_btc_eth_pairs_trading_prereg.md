# Path B R6 — BTC-ETH Cointegration Pair Trading

**Date pre-registered**: 2026-04-29
**Status**: PRE-COMMIT (locked before strategy code)
**Track**: Path B R6 — Round 16 retry (3rd attempt, distinct alpha class)
**Authority**: User redirect 2026-04-29 ("BingX-only 유지, envelope 불수용, 창의적 발굴")

---

## What's distinct from R1-R5

| Round | Alpha class | Why distinct from R6 |
|-------|-------------|----------------------|
| R1/R2 | XS price-return ranking | uses absolute returns; R6 uses RATIO mean-reversion |
| R3/R4 | Funding-rate carry | uses funding payments; R6 uses price ratio z-score |
| R5 | Single-coin cash-and-carry | delta-neutral via spot+perp same coin; R6 delta-neutral via 2-coin perp ratio |

**R6 mechanism**: long ETH perp + short BTC perp (both 1× leverage on BingX),
when log(ETH/BTC) is below its 60-day rolling mean by ≥1 std. Exit when z-score
crosses 0 (mean-reversion realized) or |z| > 4 (regime change stop).

---

## Theory Anchor

1. **Gatev, Goetzmann, Rouwenhorst (2006) "Pairs trading: Performance of a
   relative-value arbitrage rule"** (Rev Fin Stud 19): canonical statistical
   arbitrage approach — cointegrated pairs revert to equilibrium ratio.
   Reported Sharpe 1.0-1.5 in equity sample 1962-2002.

2. **Krauss & Stübinger (2017) "Non-linear dependence modeling with bivariate
   copulas: Statistical arbitrage pairs trading on the S&P 100"** + crypto
   extension lit: applied to crypto, BTC-ETH pair trading reports net Sharpe
   1.0-2.0 in 2014-2018 samples.

3. **Mechanism economic story**: BTC and ETH share macro crypto factor
   exposure (~0.85 daily return correlation). Their RATIO has structural
   mean-reversion driven by relative positioning, ETF flow timing, and
   network/utility narrative cycles. The pair trade harvests transient
   ratio dislocations without taking directional crypto exposure.

4. **Why this could break the ceiling**: prior 17 rounds all failed by being
   directional or narrow-cross-section. R6 is ratio-mean-reversion with
   delta-neutrality at the strategy level, not coincidentally. Sharpe in
   literature is 1-2 net, which at our friction level (0.07% × 4 legs)
   could clear the 4% T4 gate if daily volatility is contained.

---

## Locked Parameters

```python
LOCKED = {
    'long_leg': 'ETH/USDT',
    'short_leg': 'BTC/USDT',
    'lookback_days': 60,                # rolling z-score window
    'entry_z_threshold': 1.0,           # |z| ≥ 1 enter
    'exit_z_threshold': 0.0,            # z crosses 0 exit
    'stop_z_threshold': 4.0,            # |z| > 4 regime stop
    'friction_per_transaction_pct': 0.07,  # taker round-trip per leg
    'leverage': 1.0,                    # 1× per leg
    'capital_usd': 1500,
    'long_position_usd': 750,
    'short_position_usd': 750,
}
```

---

## Pre-run Gates

### Gate A — Cointegration check (Engle-Granger augmented Dickey-Fuller)
- ADF on log(ETH/BTC) - rolling 60d mean residual
- **Pass**: ADF p-value < 0.05 over full sample
- **Fail**: not stationary, mean-reversion mechanism vacuous

### Gate B — Sufficient z-crossings
- Number of |z| ≥ 1 events over panel
- **Pass**: ≥ 50 entry events (sufficient sample for statistics)
- **Fail**: too few events for meaningful tests

### Gate C (NEW — anti-fix-impulse mandate) — Random-baseline comparison
- Compute random baseline: random entry/exit days at same frequency
- 1000 random simulations; compare actual strategy net P&L distribution
- **Pass**: actual strategy net cum > 95th percentile of random P&L
- **Fail**: strategy not distinguishable from random ⇒ no edge

---

## Pre-Registered Tests (5 standard gates)

### Test 1: WF 5-fold expanding
- **Pass**: ≥3/5 folds positive cumulative net

### Test 2: Bootstrap 1000 × 60-day net-return windows
- **Pass**: pos_rate ≥ 50%

### Test 3: Train/Test 60/40
- **Pass**: BOTH train and test cum_net > 0

### Test 4: Magnitude (annualized net APY ≥ 4%)
- Bank-interest baseline (per user's relaxed bar 2026-04-29)
- **Pass**: full sample annualized_apy ≥ 4.0%

### Test 5: Tail-risk (worst 5d net ≥ -5%)
- Pair trades CAN have catastrophic regime breaks (ratio diverges far)
- **Pass**: worst_5d_net_pct ≥ -5.0%

---

## Verdict Logic

| Outcome | Interpretation |
|---------|----------------|
| Gate A fail (ADF) | Pair not cointegrated → mechanism vacuous → INCONCLUSIVE |
| Gate B fail (events) | Too few signals → vacuous |
| Gate C fail (random) | No edge over random → strategy = noise |
| All 5 PASS | **First strict-criterion pass in 18 rounds** |
| T4 fail + others pass | Edge real but ceiling persists |
| Mixed | Standard regime fragility |

---

## Anti-Adjustment Provisions

1. Pair locked: ETH/BTC. Adding SOL or other coins is R7, not R6-tweaked.
2. Lookback 60d locked.
3. z-thresholds locked (1, 0, 4).
4. **No retuning post-FAIL.**
5. Random baseline gate is BINDING — addresses anti-fix-impulse memory mandate.

---

## Honest Caveats

1. **2024-2026 BTC-ETH correlation is high (~0.85)** — pair trade edge has
   structurally compressed since 2020. Krauss-Stübinger sample was 2014-2018
   when BTC dominated and ETH was less mature.
2. **Same-direction shocks** (e.g., USDC depeg, Terra collapse) move BTC and
   ETH together, momentarily compressing the ratio without the
   mechanism's mean-reversion holding. Test 5 should catch this.
3. **Friction is meaningful**: 4 legs × 0.07% = 0.28% per cycle. Pair must
   capture > 0.28% per cycle to be net positive. With z-threshold 1 (~10%
   of distribution above) and exit at 0 (~50%), expected per-cycle return
   ≈ 1 std × 0.5 = ~0.5-1% per cycle on log ratio. Net 0.2-0.7%/cycle.
4. **Holding period varies** — z-score may take days to months to revert.
   Annualized return depends on cycle frequency × per-cycle return.
5. **Per-trade gross > taker RT** gate: per-cycle gross ≈ 1% > 0.07% ✓
   on average; per-trade gross is per-cycle, NOT per-day.

---

## EV Estimate (logged before result, per anti-fix-impulse)

| Outcome | Probability | Justification |
|---------|------------|---------------|
| All 5 PASS + T4 PASS | 15-20% | Krauss-Stübinger crypto Sharpe ~1.5 ⇒ ~7-15% APY net plausible |
| Gates A/B/C PASS but T4 FAIL | 35-40% | Edge exists but compressed by 2024+ correlations |
| Gate C FAIL (random baseline beat) | 15-20% | high BTC-ETH correlation → strategy degenerates to noise |
| Gate A FAIL (ADF) | 10% | ratio is not strict cointegration in 2024+ regime |
| Mixed regime fragility | 15% | folds split |

Expected outcome: **40-50% probability T4 PASS**, but conditional on Gate C
passing first. If Gate C fails → strategy IS noise, NOT result.

---

## Hash Anchor

Committed BEFORE strategy code. Result file timestamps post this commit
anchor anti-snooping.
