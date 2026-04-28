# Path B R1 Pre-Registration — Cross-Sectional Momentum (10 Crypto, Daily, Weekly Rebalance)

**Date pre-registered**: 2026-04-29
**Status**: PRE-COMMIT
**Track**: Path B (daily/weekly multi-asset, advisor-blessed parallel to L2 collector)
**Honest prior**: Friction-floor evidence (R41+R1+R2) closed bar-level retail BTC perp envelope. Path B is fundamentally different envelope — friction proportional to trade frequency, ~100x lower transaction count → friction-floor inequality may not bind. **No prior arc evidence in this envelope.**

---

## Why Path B is structurally distinct from closed envelopes

| Track | Envelope | Frequency | Friction impact (annual) | Status |
|-------|----------|-----------|--------------------------|--------|
| OHLCV 5m/15m | bar-level intraday | 5-50 trades/day | 130-1300% | **CLOSED (R41)** |
| Trade-tape 1m | bar-level microstructure | 5-40 trades/day | 130-1100% | **CLOSED (R1+R2)** |
| **Path B daily** | **multi-asset weekly rotation** | **6-12 trades/week** | **22-44%** | **OPEN — this round** |
| L2 forward | future microstructure | unknown | unknown | Collecting |

Path B's friction-to-frequency ratio is 30-60x lower than closed envelopes. Friction-floor evidence does not transfer.

---

## Locked Mechanism — `xs_momentum_weekly_top3`

**Theory** (Jegadeesh-Titman 1993 + Liu-Tsyvinski 2021 crypto extension):
Cross-sectional momentum: relative return ranking has predictive power for next period's returns. In crypto, Liu-Tsyvinski (2021, JFE) documented significant momentum factor across coins.

**Algorithm**:
1. Universe (locked): BTC, ETH, SOL, BNB, XRP, ADA, DOGE, AVAX, TRX, LINK (10 USDT spot pairs)
2. Each Monday at UTC 00:00 (start of week):
   - Compute **trailing 30-day total return** for each coin: `(close_t / close_{t-30}) - 1`
   - Rank coins by this metric
   - **Long top-3**, **Short bottom-3** (equal weight)
3. Hold positions for 7 days (next Monday)
4. Rebalance: close all positions, re-rank, open new positions
5. Friction: 0.07% per transaction × 12 transactions per week (6 close + 6 open, ignoring overlaps)
   - Net friction per week ≤ 0.84% (worst case, all 6 positions rotate)
   - Typically less due to position persistence

**Locked Parameters (NO retuning)**:
```python
LOCKED = {
    'universe_size': 10,
    'lookback_days': 30,
    'long_top_n': 3,
    'short_bottom_n': 3,
    'rebalance_frequency_days': 7,
    'friction_per_transaction': 0.07,  # %
    'equal_weight': True,
}
```

**Theory source**:
- Jegadeesh & Titman (1993): "Returns to Buying Winners and Selling Losers" — momentum factor in equities
- Liu & Tsyvinski (2021): "Risks and Returns of Cryptocurrency" — momentum factor in crypto with t-stat 2.7+
- 30-day lookback: standard "momentum" window in factor literature
- Top-3/Bottom-3 from 10: standard tertile-style portfolio (commonly 30%/30%)
- Weekly rebalance: balances signal decay vs friction

---

## Pre-Registered Tests

### Pre-run Vacuity / Cross-Sectional Dispersion Gate
- Median cross-sectional dispersion (std of trailing 30d returns) must be ≥ 5%
  - Below 5% = coins moving together, no momentum signal worth trading
- If FAIL → R1 inconclusive (insufficient dispersion)

### Test 1: WF 5-fold Expanding
- 5 expanding test windows on locked params
- Friction 0.07% per transaction
- **Pass**: ≥3/5 folds avg_weekly_net > 0

### Test 2: Bootstrap 1000 × 30-day windows
- random.seed=42
- 30-day windows (vs 3-day for intraday — adapted to weekly rebalance horizon)
- **Pass**: pos_rate ≥ 50%

### Test 3: Train/Test 60/40
- **Pass**: BOTH train AND test avg_weekly_net > 0

### Friction-Aware Reporting (advisor mandate)
- Log avg_weekly_gross at every fold (compare to weekly friction estimate ~0.42-0.84%)
- Compute breakeven_avg_gross = friction estimate to make floor visible

---

## Verdict Logic

- **Dispersion gate FAIL**: R1 inconclusive
- **Dispersion PASS + ALL 3 OOS PASS**: call advisor before any breakthrough claim. This is the first non-closed envelope to potentially show edge — needs scrutiny.
- **Dispersion PASS + ANY OOS FAIL**: 1st Path B negative committed. Different from R41/R1/R2 — different envelope. Combined with prior arc evidence, would tighten the "all retail BTC mechanisms infeasible" hypothesis but Path B failure does not imply it (could be momentum factor specifically that doesn't work at this universe size).

---

## Anti-Adjustment Provisions

1. No sweep, no retuning if FAIL
2. No friction relaxation
3. No criterion relaxation
4. No mechanism swap as "R1 v2" — different mechanism = R2 (different factor: e.g., volatility, low-beta, etc.)
5. ALL params locked from theory before any data observation
6. Universe is locked (no removing "underperformers" post-hoc)

---

## What this test cannot establish

Even if all 3 pass:
- Single mechanism (long-short momentum); other factors not yet tested
- Daily-only timeframe; intraday combinations not tested
- Spot only (perpetual basis carry, funding harvest not tested)
- Advisor will scrutinize before any claim

---

## Hash anchor

Committed BEFORE backtest run. Universe + lookback + ranking method all theory-locked.
