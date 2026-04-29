# Path B R4 — Funding-Rate Carry on Expanded 30-Coin Universe

**Date pre-registered**: 2026-04-29
**Status**: PRE-COMMIT (locked before data fetch)
**Track**: Path B R4 (Round 16 retry — addresses R3 vacuity-FAIL via universe expansion)
**Authority**: User redirect "계속 진행" + R3 pre-reg explicit anticipation
  ("Bybit/OKX add 30+ coins might activate this — would require new data source + new pre-reg")

---

## What's distinct from R3

R3 used Binance 10-coin perp universe → Gate B vacuity FAIL (median 7d funding
std = 0.0038%/8h vs floor 0.05%/8h).

R4 = same mechanism, **expanded universe to 30 coins via Bybit linear perpetuals**.
This is the structural redesign R3 pre-reg explicitly anticipated. Gate values
remain identical for honest comparison — if 30-coin universe also vacuums,
that's a real finding about retail-tractable carry crypto crypto.

---

## Locked Universe (committed before any Bybit data fetch)

```
30-coin universe (top market cap perp-listed, expected pre-2024 Bybit listing):
  Original 10:  BTC ETH SOL BNB XRP ADA DOGE AVAX TRX LINK
  Mid-cap 10:   DOT MATIC LTC SHIB BCH ATOM UNI NEAR ICP FIL
  Large-alt 10: APT AAVE ARB OP INJ SUI TIA FTM ALGO SAND
```

If any coin has <600 days Bybit funding history (75% of 800d target), it's
dropped from the universe. Pre-disclosure: this filter is **applied uniformly
without backtest peeking** — drop list will be reported.

Data source: Bybit linear USDT perpetuals via CCXT `fetch_funding_rate_history`.
Free, no auth.

---

## Locked Mechanism (identical to R3 — controlling for non-universe variables)

```python
LOCKED = {
    'data_source': 'bybit_linear_perp',
    'universe_target_size': 30,
    'lookback_funding_periods': 21,            # = 7 days
    'long_bottom_n': 3,
    'short_top_n': 3,
    'rebalance_frequency_days': 7,
    'friction_per_transaction': 0.07,          # %, taker round-trip
    'equal_weight': True,
    'min_history_days_per_coin': 600,
}
```

---

## Pre-run Gates (identical thresholds to R3)

### Gate A — Distinctness from momentum
- Spearman ρ(7d funding rank, 30d momentum rank), panel-wide
- **Pass**: |ρ| < 0.7

### Gate B — Vacuity (universe dispersion sufficient for carry)
- Median 7d cross-sectional funding std ≥ **0.05%/8h** (= 4.93%/yr)
- **Pass**: ≥ 0.05%/8h
- **Note**: This is the same gate that R3 failed at 12× margin. R4 tests
  whether 3× universe size lifts dispersion to floor.

If Gate A fails: NOT DISTINCT, abort.
If Gate B fails: INCONCLUSIVE_VACUOUS — document, no retune.

---

## Pre-Registered Tests (5 standard gates, identical to R3)

### Test 1: WF 5-fold expanding
- **Pass**: ≥3/5 folds avg_weekly_net > 0

### Test 2: Bootstrap 1000 × 30-day net-return windows
- **Pass**: pos_rate ≥ 50%

### Test 3: Train/Test 60/40 sign-agreement
- **Pass**: BOTH train AND test avg_weekly_net > 0

### Test 4: Magnitude (avg_daily_net ≥ 0.02%/day = 7.3%/yr)
- **Pass**: full sample avg_daily_net_pct ≥ 0.02

### Test 5: Tail-risk (worst 5-day net ≥ -10%)
- **Pass**: worst_5d_net_pct ≥ -10

### Decomposition gate
- price_share of net cum < 70% (otherwise R4 = momentum in disguise)
- **Pass**: |price_share/net| < 0.70 OR net_cum < 0 (irrelevant)

---

## Verdict Logic

| Outcome | Interpretation |
|---------|----------------|
| Gate A fail (ρ ≥ 0.7) | NOT DISTINCT, abort |
| Gate B fail (vacuum) | INCONCLUSIVE — universe size ≤ 30 coins not enough; ceiling unaltered |
| All 5 PASS + carry-dominant decomp | **First ceiling break in 16 rounds.** Surface to advisor + user. |
| All 5 PASS + price-dominant decomp | Carry premium is momentum-rediscovery in disguise; not new alpha family |
| T4 fail + others pass | 16th data point on ceiling — synthesis hardens, capital-bound conclusion strengthened |
| Mixed core 1/2/3 fail | Standard regime fragility |

---

## Anti-Adjustment Provisions

1. Universe locked above. Drop list (if any coin < 600d) will be reported.
2. No mechanism parameters changed from R3. Only the universe and the data
   source changed.
3. Gate thresholds unchanged from R3. **No floor adjustment** to fit 30-coin
   universe characteristics.
4. If R4 also vacuum-FAILs: NO R5 with 60-coin universe. The conclusion is
   that retail-tractable carry universes (≤30 coins) are too narrow — moving
   to 60 coins introduces survivorship bias on coin selection that exceeds
   the marginal information value.
5. If R4 PASS Gate B but FAIL Tests 1-3: regime fragility, no further
   universe expansion.

---

## Honest Caveats

1. **Bybit infrastructure beyond BingX**: Deploying this strategy requires
   Bybit account/API. User constraint was "기존 BingX setup 외 인프라는 없음".
   Backtest is informational about factor existence; deployment requires
   user decision on infrastructure expansion.
2. **Drop list risk**: If many of the 20 added coins lack 600d history, R4
   may reduce to 15-20 coins effectively, partially undoing the universe
   expansion benefit.
3. **Bybit listing dates** may be more variable than Binance — some altcoins
   listed perpetuals later. Pre-reg drop filter handles this.
4. **Per-coin friction may differ** on Bybit vs Binance, but we assume same
   0.07% taker for fair comparison. Bybit actual taker is ~0.055% — this
   would slightly favor R4 in absolute terms.
5. **Sample period 2024-02 → 2026-04 includes**: USDC depeg recovery, ETF
   inflows, multiple altseasons. Carry premium may have changed. Hu 2024
   sample was 2018-2023.

---

## Hash Anchor

This pre-reg locks the universe before Bybit data fetch. Result file
timestamps post this commit anchor anti-snooping evidence.
