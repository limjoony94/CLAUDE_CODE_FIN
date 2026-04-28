# Trade-Tape Track R1 Pre-Registration — Persistent Taker Imbalance + 1h Trend

**Date pre-registered**: 2026-04-29
**Status**: PRE-COMMIT (results not yet observed)
**Envelope**: trade-tape-derived 1-min features (Binance Vision aggTrades historical)
**Honest prior**: No prior arc evidence in this envelope. M3 OHLCV envelope falsified separately (R41) but that does NOT directly extend here — different signal type.

---

## Why this envelope is distinct from M3 OHLCV (which was R41-falsified)

| Envelope | Signal type | M3 status |
|----------|-------------|-----------|
| OHLCV 5m/15m | Price/range conditional | **Arithmetically falsified (R41)** |
| Trade-tape 1m | Order flow / microstructure | **Open** — never tested |
| L2 orderbook (forward) | Depth / book imbalance | **Open** — collector running |

R41's arithmetic inequality (avg_gross < friction at n=2,760) was on price-conditional mechanisms. Trade-tape mechanisms condition on **order flow imbalance** — fundamentally different signal, different per-trade economics possible. R41 evidence does not transfer.

---

## Locked Mechanism — `entry_persistent_imbalance_1m`

**Theory**: Sustained taker buy/sell imbalance over a 5-min window indicates persistent institutional aggression. Combined with 1h trend filter to avoid mean-reversion conditions.

**Algorithm** (1-min granularity):
1. Compute rolling 5-min vol_imbalance:
   `roll_imb = sum(vol_buy - vol_sell over last 5 min) / sum(vol_total over last 5 min)`
2. Entry conditions:
   - **LONG**: `roll_imb >= +0.30` (sustained ~65/35 taker buy)
       AND **1h SMA200**: 1h close > 1h SMA200
       AND minute body filter: `|price_last - price_first| / (price_high - price_low) >= 0.4`
       AND minute body direction: `price_last > price_first`
   - **SHORT**: mirror
3. Project 1-min entry timestamp → 15m bar index for exit framework
4. Max 1 entry per 15m bar (consistent with R36-R41 anti-clustering)

**Locked Parameters (NO retuning)**:
```python
LOCKED = {
    'imbalance_window_min': 5,
    'imbalance_threshold': 0.30,
    'body_min_ratio': 0.4,
    'trend_filter': '1h_sma200',
    'min_bars_between_15m': 2,
}
```

**Theory source**:
- Order flow imbalance threshold 30% (~65/35 taker split): standard institutional aggression signal
- 5-min smoothing: standard "persistent flow" window in microstructure literature
- 1h SMA200 trend filter: identical to R41 (consistent across rounds)
- Body filter 0.4: identical to R36-R41

---

## Exit Framework

`run_bt_c1_production` (constant across all rounds). 15m bars.

---

## Pre-Registered Tests

### Pre-run Vacuity Gate
- Signal frequency ≥ 0.5/day on full sample
- If FAIL → R1 inconclusive (vacuous), trade-tape envelope may need different mechanism

### Test 1: WF 5-fold Expanding
- Locked params, friction 0.07
- **Pass**: ≥3/5 folds daily_net > 0

### Test 2: Bootstrap 1000 × 3-day
- random.seed=42, friction 0.07
- **Pass**: pos_rate ≥ 50%

### Test 3: Train/Test 60/40
- **Pass**: BOTH train AND test daily_net > 0

---

## Sample period

Limited by intersection of:
- 15m OHLCV data: 2024-02-23 → 2026-02-12
- Trade-tape data: 2025-04-27 → 2026-04-26 (currently downloading)

**Effective overlap**: ~2025-04-27 to 2026-02-12 = ~291 days

If signals/day = 1, n ≈ 291. Sufficient for OOS but on smaller sample than M3 rounds (720d). Acknowledge in verdict — small-n caveat will apply.

---

## Verdict Logic

- **Vacuity FAIL**: R1 inconclusive
- **Vacuity PASS + ALL 3 OOS PASS**: call advisor before any breakthrough claim
- **Vacuity PASS + ANY OOS FAIL**: 1st rigorous trade-tape negative committed; trade-tape envelope partial evidence

---

## Anti-Adjustment Provisions

1. No sweep, no retuning if FAIL
2. No friction relaxation
3. No criterion relaxation
4. No mechanism swap as response to FAIL — different mechanism = R2
5. ALL params locked from theory before any data observation

---

## Hash anchor

Committed BEFORE any feature analysis or signal observation. Aggregator output (`btc_trade_features_1m.parquet`) was generated but NOT examined for this mechanism's signal counts before this pre-reg commit.
