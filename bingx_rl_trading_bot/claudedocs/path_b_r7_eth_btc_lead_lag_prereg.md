# Path B R7 — ETH→BTC Lead-Lag Scalping (5m / 15m with 1h trend filter)

**Date pre-registered**: 2026-04-29
**Status**: PRE-COMMIT (locked before strategy code)
**Track**: Path B R7 — Round 17 (genuinely new alpha class — cross-asset lead-lag)
**Authority**: User redirect 2026-04-29 explicit option (i): "C1-style search 진행, criteria 약간 완화 가능"

---

## Disclosure (per advisor mandate)

Methodology that found C1 strategy is the same methodology that produced
C1's LIVE -12.86%/14d failure (`docs/04-report/c1_breakout_postmortem_20260427.md`).
**LIVE-parity prior is 0/1**. BT pass at this round does NOT imply deploy
viability. Result is a research artifact, not a deployment proposal.

---

## What's distinct from R1-R6

R7 is the first round in this codebase to test **cross-asset lead-lag
predictability**. All prior 17 rounds (M3, TT, PB R1-R6, DeFi-R1) used
either:
- Single-asset price action (R1-R8 OHLCV, TT-R1/R2)
- Cross-sectional ranking (PB-R1/R2/maker)
- Same-coin carry (R3/R4/R5)
- Same-pair cointegration (R6)

R7 mechanism: **ETH 15-minute return predicts BTC 30-minute return**.

Theory anchor:
1. **Borri & Shakhnov (2018) "Cryptomarkets, mining, and the economic
   foundations of cryptocurrencies"**: documents lead-lag between BTC
   and altcoins, ETH typically leads BTC by 5-15 minutes during volatile
   regimes.
2. **Yi, Xu, Wang (2018) "Volatility connectedness in the cryptocurrency
   market"** (Int Rev Fin Anal): demonstrates volatility spillover from
   ETH to BTC at intraday horizons.
3. **Mechanism economic story**: ETH liquidity attracts retail flow
   slightly before BTC (informed traders front-run BTC via ETH options
   delta hedging, ETH/BTC ratio plays). Predictable component is small
   but persistent.

---

## Locked Parameters

```python
LOCKED = {
    'asset_trade': 'BTC/USDT',
    'asset_signal': 'ETH/USDT',
    'bar_resolution': '5m',
    'signal_lookback_bars': 3,            # 15 min ETH return
    'signal_threshold_pct': 0.30,         # |ETH 15min return| ≥ 0.30%
    'trend_filter_period': 12,            # 1h SMA20 vs SMA50 (12 5m bars = 1h)
    'sma_short_periods': 240,             # 20h × 12 5m bars
    'sma_long_periods': 600,              # 50h × 12 5m bars
    'atr_period': 14,                     # bars
    'tp_atr_mult': 1.5,
    'sl_atr_mult': 1.0,
    'max_hold_bars': 6,                   # 30 min
    'friction_per_transaction_pct': 0.07,
    'capital_usd': 1500,
    'leverage': 1.0,
}
```

**Logic**:
1. At 5m bar t, compute ETH 15min return = log(ETH[t]/ETH[t-3])
2. If ETH ret ≥ +0.30% AND BTC SMA240 > SMA600 (uptrend) → enter LONG BTC at bar t+1 open
3. If ETH ret ≤ -0.30% AND BTC SMA240 < SMA600 (downtrend) → enter SHORT BTC at bar t+1 open
4. TP = entry ± 1.5 × ATR(14, 5m bars)
5. SL = entry ∓ 1.0 × ATR(14, 5m bars)
6. Max hold: 6 bars (30 min). Exit at market if neither TP nor SL hit.
7. Friction 0.07% × 2 = 0.14% round-trip per trade.

---

## Pre-run Gates

### Gate A — Lead-lag existence (causality direction)
- Granger causality test: ETH 15m return → BTC 30m return at lag 1
- Or simpler: lagged correlation Corr(ETH_ret_t, BTC_ret_{t+1..t+6})
- **Pass**: Corr > 0.05 at any lag 1-6 (positive predictive)
- **Fail**: Corr ≤ 0.05 → lead-lag too weak, mechanism vacuous

### Gate B — Sufficient signal events
- |ETH 15m return| ≥ 0.30% events / total bars over panel
- **Pass**: ≥ 2,000 candidate signal events (= sufficient sample)
- **Fail**: too few events

### Gate C (ANTI-FIX-IMPULSE binding) — Random-baseline
- Run 1000 random-entry simulations matching trade frequency
- **Pass**: actual cum_net > 95th percentile of random distribution

---

## Pre-Registered Tests (Korean criteria — full set)

### Test 1: WF 5-fold expanding
- **Pass**: ≥3/5 folds with positive cumulative net P&L

### Test 2: Bootstrap 1000 × 3-day random windows (USER REQUIREMENT)
- Sample 3-day windows from net daily P&L
- **Pass**: pos_rate ≥ 50%

### Test 3: Train/Test 60/40 sign-agreement
- **Pass**: BOTH train and test cum_net > 0

### Test 4 (HARD goal): annualized net APY ≥ 73% (= 0.2%/day)
- User's stated hard goal
- **Pass**: avg_daily_net ≥ 0.2%

### Test 5 (HARD goal): WR ≥ 40%
- **Pass**: win_rate ≥ 40%

### Test 6 (HARD goal): R:R ≥ 1.0
- **Pass**: avg_win / avg_loss ≥ 1.0 (note: dynamic TP/SL means realized R:R)

### Test 7 (HARD goal): trades/day ≥ 2
- **Pass**: avg_trades_per_day ≥ 2

### Test 8 (HARD goal): per-trade gross > 0.07% taker RT
- **Pass**: avg_gross_per_trade > 0.07%

### Test 9: Tail-risk worst 5d ≥ -10%
- **Pass**: worst_5d_net ≥ -10%

---

## Verdict Logic

| Outcome | Interpretation |
|---------|----------------|
| Gate A fail | lead-lag absent → mechanism vacuous |
| Gate B fail | too few events |
| Gate C fail | random baseline beats strategy |
| All Tests 1-9 PASS | **Round 17 candidate. Surface for advisor review.** |
| T4 fail (others pass) | edge real but doesn't reach 0.2%/day target |
| T5/T6 fail | gates relaxable per user — may still surface |
| T7/T8 fail | structural fail of basic conditions |

Per user 2026-04-29: "성공 조건에서 살짝 완화하더라도 다른 부분에서 trade off 발생해서 성과가 더 뛰어난 경우 채택해볼만." → if T4 strictly hit but T5/T6 borderline, candidate still surfaceable for user judgment.

---

## EV Estimate (logged before result, per anti-fix-impulse)

| Outcome | Probability | Justification |
|---------|------------|---------------|
| All 9 tests PASS | 3-5% | 17 rounds + C1 LIVE failure prior |
| T4 PASS only (≥0.2%/day) | 5-8% | requires lead-lag alpha 5-10× our typical observed |
| Gate C PASS (beats random) | 25-35% | crypto lead-lag has empirical support in lit |
| Gate A PASS (causality) | 50-60% | weak lead-lag should be detectable |
| Mixed regime fragility | 30-40% | most likely outcome |

**Realistic expectation**: edge real but T4 (0.2%/day) FAIL by 3-5×.
Result hardens 17-round ceiling, gives 18th data point.

---

## Anti-Adjustment Provisions

1. Asset pair locked (ETH→BTC). Reversed (BTC→ETH) is R8.
2. Threshold 0.30% locked. Sweep is pre-reg violation.
3. ATR multipliers (1.5 TP, 1.0 SL) locked.
4. Max hold 6 bars locked.
5. 1h trend filter locked.
6. **No post-FAIL retuning.** Result stands.

---

## Honest Caveats

1. **C1 LIVE failure**: same BT methodology produced C1 which failed LIVE.
   BT result here doesn't imply deploy. (Disclosed in opening section.)
2. **Lead-lag may be regime-dependent**: 2024+ (post-ETF, post-Merge)
   ETH-BTC dynamics differ from pre-2022 lit sample.
3. **Friction at 0.07% × ~5 trades/day**: ~0.35%/day friction floor.
   To net 0.2%/day after friction needs ~0.55%/day gross — plausible
   only if multiple trades win consistently.
4. **5m lookback limited**: only ~365 days of overlapping ETH-BTC 5m
   data. Smaller sample than prior rounds.
5. **Slippage**: 5m scalping at $1500 retail size has minimal market
   impact but spread cost is meaningful (~0.02-0.05% on BTC perps).
   Not modeled.

---

## Hash Anchor

Committed BEFORE strategy code. Pre-data-look theory anchor + locked
mechanism + EV estimate. Result file timestamps post this commit.
