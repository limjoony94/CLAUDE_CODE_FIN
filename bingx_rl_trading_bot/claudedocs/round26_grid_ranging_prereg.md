# Round 26 — Grid Trading on ATR-Based Ranging Regime

**Date pre-registered**: 2026-04-30
**Status**: PRE-COMMIT (locked before strategy code)
**Track**: Maker-only volatility harvest, advisor-authorized post R25

---

## DISCLOSURE

R25 (R8 + maker-only) failed 0/3 HARD with adverse selection (gross +0.04% taker →
-0.236% maker). Advisor-authorized R26 because grid hedges the directional adverse-
selection by placing both sides simultaneously: in a ranging market, each fill on
one side is balanced by a profit on the other.

User's restated criteria (2026-04-30) removed "≥2 trades/day" — opens grid which has
fewer trades/day. Daily ≥0.20% still required.

LIVE-parity prior 0/1 (C1).

---

## What's distinct from prior 29 rounds

| Class | Prior | R26 |
|-------|-------|-----|
| Directional momentum | R7-R10, R8, R24 | — |
| Mean-reversion | R11 | — |
| Carry | R5, R13 | — |
| Microstructure | L2 F1-F4 | — |
| Pattern detection | R24 | — |
| **Volatility harvest (grid)** | — | **R26** |

R26 is the first volatility-harvest mechanism in the test suite. Profit comes from
price oscillation through pre-placed limit levels, not from directional prediction.

---

## Theory Anchor

1. **Avellaneda & Stoikov (2008) "High-frequency trading in a limit order book"** —
   market-making theory: place symmetric bid/ask limits, profit from spread + noise.
2. **Garman (1976) "Market microstructure"** — bid-ask spread compensation for
   inventory risk. Grid is structurally a multi-level market-making strategy.
3. **TradingView community grid bots** (popular: "Grid Trading Strategy", various):
   common claim 0.5-1.5%/day in range-bound BTC; typical failure mode is trending
   markets where grid accumulates losing positions on one side.

---

## Locked Parameters

```python
LOCKED = {
    'asset': 'BTC/USDT',
    'tf': '1h',
    'capital_usd': 1500,

    # Grid configuration (LOCKED, no tuning)
    'grid_spacing_pct': 0.30,       # 5x friction margin
    'grid_levels_each_side': 5,     # 5 buys + 5 sells = 10 total
    'per_level_usd': 150,           # 1500 / 10
    'init_mid_method': 'close_at_setup',  # mid = close[t] when grid initialized

    # Ranging regime filter (LOCKED)
    'atr_period': 20,
    'atr_pct_median_lookback_days': 30,  # 720 hourly bars
    # 'ranging' if ATR(20)/close < median(ATR(20)/close over last 30d)

    # Trend exit (LOCKED)
    'trend_exit_distance_pct': 1.5,  # if price moves > 1.5% beyond init_mid AND ranging=off
    'trend_exit_taker_friction_pct': 0.05,

    # Maker fills (LOCKED)
    'maker_friction_per_side_pct': 0.02,

    # Grid maintenance (LOCKED)
    'max_grid_lifetime_bars': 168,  # 7 days max per setup
    'reset_after_full_cycle': True,
}
```

Logic:
1. At each bar t, compute ATR(20)/close[t] and 30d trailing median of same.
2. If currently_ranging = (ATR_pct < 30d_median):
   - If no active grid: setup grid centered at close[t] with 5 buys at
     close[t]·(1 − 0.003·k) for k=1..5 and 5 sells at close[t]·(1 + 0.003·k)
3. Per bar evolution:
   - Walk through grid levels checking fills (intrabar high/low):
     - BUY level k filled if low ≤ buy_price[k] AND not already filled this lifetime
     - SELL level k filled if high ≥ sell_price[k]
   - On BUY fill: place SELL limit at +0.30% from fill (TP, maker)
   - On SELL fill: place BUY limit at −0.30% from fill (TP, maker)
   - If TP fills (maker exit): cycle profit = +0.30% − 0.04% friction = +0.26% net
4. Trend-exit: if |close[t] − init_mid| / init_mid > 0.015 AND currently_ranging=False:
   - Close ALL open positions at market (taker 0.05% per side)
   - Reset grid (next bar can re-init if ranging returns)
5. Max grid lifetime 168 bars; force reset.

---

## Pre-run Gates

### Gate A — Sufficient ranging regime
- ≥ 30% of bars must satisfy ranging filter
- **Pass**: ranging fraction ≥ 30%

### Gate B — Random-baseline (anti-fix-impulse)
- 1000 simulations with random fill bars matching same trade count
- **Pass**: actual cum_net > 95th percentile

---

## Pre-Registered Tests (User's restated 4 criteria 2026-04-30)

C1 (HARD) **Daily ≥ 0.20% net at 1× leverage**
C2 **Per-trade gross > 0.07% taker RT** (note: R26 trades have variable friction —
   report per-trade gross AND comparative against weighted realized friction)
C3 **Trade count statistically significant** — N ≥ 100 trades for stable inference
C4 **Bootstrap 1000 × 3-day random window** — pos_rate ≥ 50%

---

## Mandatory Loss Decomposition (advisor requirement)

Report MUST include:
- `cum_harvest_pct`: sum of all (TP-fill) cycle profits (positive contributions)
- `cum_drift_drawdown_pct`: sum of trend-exit losses (negative contributions)
- `cum_friction_pct`: sum of maker + taker fees
- `cum_net_pct = harvest + drawdown − friction`
- `n_full_cycles`: completed buy-fill→sell-fill or sell-fill→buy-fill cycles
- `n_trend_exits`: number of forced exits during trend episodes
- `mean_drawdown_per_trend_exit_pct`: average loss per trend exit
- `ranging_fraction`: fraction of bars where ranging filter active

---

## EV Estimate (advisor)

| Outcome | Probability |
|---------|------------|
| Range harvest > drift drawdown, daily ≥ 0.20% PASS | 15-20% |
| Range harvest > drift drawdown, daily 0.05-0.20% (sub-deployable) | 20-25% |
| Drift drawdown ≈ harvest, daily near zero | 30-35% |
| Drift drawdown > harvest, net negative | 25-30% |

Realistic: borderline outcome with daily 0.05-0.30% range. The test answers
**"can grid harvest exceed trend drift in BTC 2024-2026 1h regime?"** — empirical.

---

## Anti-Adjustment

Grid spacing 0.30%, level count 10, per-level $150, ranging filter (ATR/close vs
30d median), trend-exit threshold 1.5%, max grid lifetime 168h ALL LOCKED. NO
TUNING POST-FAIL. If FAIL: report finding "directional+maker=adverse,
range-grid=drift-bound" and stop further spawn.

---

## Hash Anchor

Committed BEFORE strategy code.
