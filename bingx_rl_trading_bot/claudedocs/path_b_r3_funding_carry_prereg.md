# Path B R3 — Funding-Rate Carry Harvest Pre-Registration

**Date pre-registered**: 2026-04-29
**Status**: PRE-COMMIT (locked before data fetch + code)
**Track**: Path B R3 (Round 16 of envelope research, distinct alpha family from R1/R2)
**Authority**: User redirect 2026-04-29 ("계속 진행, advisor 적극 활용") + advisor delegation

---

## What's distinct from R1/R2 (Path B)

R1 (XS Momentum): signal = trailing 30d **price return**.
R2 (XS Reversal): signal = trailing 7d **price return** (opposite sign).
**R3 (Carry)**: signal = trailing 7d **funding rate** (orthogonal to price return).

**Hypothesis from theory**: funding rate dispersion reflects positioning
imbalance (overcrowded longs/shorts), not directional view. Long low-funding,
short high-funding harvests the carry premium that perpetual longs over-pay.

**Correlation pre-registration**: if Spearman ρ(7d funding rank, 30d momentum
rank) ≥ 0.7 over the in-sample period, declare R3 NOT DISTINCT from R1, abort
without running OOS. This prevents "same family, different label" inflation
of round count.

---

## Theory Anchor (cited pre-data-look)

1. **Koijen, Moskowitz, Pedersen, Vrugt (2018) "Carry"** (J Fin Econ 127):
   General-asset carry framework. Carry = expected return assuming prices
   unchanged. Empirically positive premium across 40+ markets.
2. **Hu, Lu, Zhang, Zhuang (2024) "Cross-Section of Crypto Carry"**:
   Specifically applies KMP 2018 to crypto perpetuals. Long-short portfolios
   on funding rate dispersion produce **gross premium ~10%/yr in their
   sample** before fees, after fees ~3-7% depending on rebalance frequency.
3. **Mechanism economic story**: Funding rate compensates the long side
   (or short side) for holding the perpetual exposure when retail demand
   tilts one way. The cross-section dispersion reflects coin-specific
   demand imbalance. Long-low / short-high captures the **positioning risk
   premium** (pays attention to short-term flow, not long-term value).

---

## Locked Parameters

```python
LOCKED = {
    'universe': ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT',
                 'ADA/USDT', 'DOGE/USDT', 'AVAX/USDT', 'TRX/USDT', 'LINK/USDT'],
    'data_source': 'binance_perp',          # funding rate history via CCXT
    'lookback_funding_periods': 21,          # 21 × 8h = 7 days
    'long_bottom_n': 3,                      # long lowest 3 funding (least over-paid longs)
    'short_top_n': 3,                        # short highest 3 funding (most over-paid longs)
    'rebalance_frequency_days': 7,           # weekly Monday
    'friction_per_transaction': 0.07,        # taker round-trip same as PB-R1
    'equal_weight': True,
    'capture_components': True,              # decompose price-return vs funding-return
}
```

---

## Pre-run Gates

### Gate A — Distinctness (orthogonality from R1)
- Compute Spearman ρ between 7d funding rank and 30d momentum rank
  over full backtest period
- **Pass**: ρ < 0.7 (distinct factor)
- **Fail**: ρ ≥ 0.7 → R3 NOT DISTINCT, abort, declare round count unchanged

### Gate B — Vacuity (sufficient funding signal dispersion)
- Median 7d cross-sectional funding-rate dispersion (std across 10 coins)
  over the panel ≥ **0.05%/8h** (= ~0.15%/day = ~55%/yr funding spread,
  consistent with Hu et al. 2024 reported dispersion floors)
- **Pass**: median dispersion ≥ 0.05%/8h
- **Fail**: dispersion below floor → INCONCLUSIVE (vacuous)

---

## Pre-Registered Tests (5 gates, same architecture as DeFi-R1)

### Test 1: WF 5-fold Expanding
- **Pass**: ≥3/5 folds avg_weekly_net > 0

### Test 2: Bootstrap 1000 × 30-day Net-Return Windows
- **Pass**: pos_rate ≥ 50%

### Test 3: Train/Test 60/40 Sign-Agreement
- **Pass**: BOTH train AND test avg_weekly_net > 0

### Test 4 (Magnitude): avg_daily_net ≥ 0.02%/day full sample
- = 7.3% annualized, the envelope ceiling established at 14 rounds
- **Pass**: full sample avg_daily_net_pct ≥ 0.02

### Test 5 (Tail): worst 5-day net return ≥ -10%
- **Pass**: worst_5d_net_pct ≥ -10

---

## Carry-Specific Decomposition

Output MUST include:
- **price_return_component_pct**: cumulative return from price moves of held legs
- **funding_return_component_pct**: cumulative funding payments collected
- **friction_component_pct**: cumulative friction
- **net_pct**: sum of three above

This decomposition lets us see whether net edge comes from:
- (a) Carry premium itself (funding component dominates net)
- (b) Mean-reversion of price among high-funding coins (price component dominates)
- (c) Both

Per advisor pattern (PB ω* paradigm shift was rejected because 94% of edge
was directional rediscovery, 6% carry — same risk here. If price component
dominates, R3 is structurally PB-R2 reversal in disguise.)

---

## Anti-Adjustment Provisions

1. Universe locked (same 10 coins as R1/R2 — controls for universe selection)
2. Lookback locked (21 funding periods = 7 days)
3. Top/bottom n=3 locked
4. Friction 0.07% taker locked (consistent with R1, fair comparison)
5. Weekly rebalance locked (matches R1)
6. **Correlation gate is BINDING**: if R3 fails Gate A, no retuning. Different
   lookback / different N would be R4, not R3-tweaked.
7. **Decomposition gate is BINDING**: if price_return_component > 70% of net,
   declare R3 = "PB-R2 in disguise", do NOT count as new alpha family.

---

## Verdict Logic

| Outcome | Interpretation |
|---------|----------------|
| Gate A fail (ρ ≥ 0.7) | R3 NOT DISTINCT. Round count remains 15. Funding adds zero info beyond momentum. |
| Gate B fail (dispersion vacuous) | INCONCLUSIVE. Crypto carry signal too thin in our universe. Park. |
| All 5 PASS + carry component dominates | **R3 = first 16th-round candidate that breaks alpha ceiling.** Surface to advisor + user. |
| All 5 PASS + price component dominates | R3 = PB-R2 in disguise. Round 16 not validated as new family. |
| T4 fail + others pass | 16th data point on alpha ceiling — synthesis hardens. |
| Mixed | Standard regime fragility, no further iteration. |

---

## Honest Caveats

1. Hu et al. 2024 reports gross premium ~10%/yr **before fees**. Their net
   number after fees is 3-7%/yr — **likely below T4 7.3% gate.** So Test 4
   FAIL is the prior expectation. Test 4 PASS would be a real find.
2. 10-coin universe is small for cross-section; their paper used 30+ coins.
3. Funding rate history has limited public availability for some altcoins
   pre-2022 — sample may be < 720 days for some coins, causing partial
   eligibility filtering.
4. Funding rate is paid TO position-holders, but to harvest you must HOLD
   the perp position — so you also incur price exposure. Hedging price
   exposure (long perp + short spot) is dollar-cost-neutral but eliminates
   the directional risk; we are NOT doing that here. Long perp = directional
   long exposure plus funding receipt/payment.
5. Test 5 tail-risk is critical for carry: classic carry strategies "pick
   pennies in front of a steamroller" — sudden regime shifts produce
   catastrophic losses (e.g., LUNA collapse May 2022 wiped out crypto carry
   funds).

---

## Hash Anchor

Committed BEFORE data fetch and BEFORE strategy code. Result file
timestamps post-this commit timestamp anchor anti-snooping.
