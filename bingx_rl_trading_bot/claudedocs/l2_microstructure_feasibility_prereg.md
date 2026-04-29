# L2 Microstructure Feasibility EDA — 18h Sample Arithmetic Gate

**Date pre-registered**: 2026-04-30
**Status**: PRE-COMMIT (locked before signal-extraction code)
**Track**: L2 substrate, sample-level feasibility (NOT round 24 in OHLCV/funding envelope)

---

## DISCLOSURE

23 prior rounds × 9 alpha classes (OHLCV + funding) → in-scope envelope empty.
This is NOT R24 in that envelope. This is a **substrate change**: L2 orderbook + trade
prints data. Same arithmetic gate as R41 (avg per-event gross > 0.07% taker friction).

L2 collector running since 2026-04-29 04:19 UTC. As of 2026-04-30 ~03:10 KST,
18.17h captured: 128,596 depth snapshots (~2 Hz) + 454,475 trade prints (~7 Hz).
4-week recording continues — this EDA is feasibility check on collected sample, not
final OOS.

LIVE-parity prior 0/1 (C1).

---

## What's distinct from prior 23 rounds

R1-R23 used bar-level OHLCV (1m/5m/15m/1h/daily) + 8h funding rates. Bar-level data
COMPRESSES intra-bar information. R41 closed bar-level envelope arithmetically:
avg_gross +0.034% < friction 0.07%.

L2 data exposes:
- Top-of-book imbalance (5-step ahead price prediction in Cont-Stoikov 2014)
- Order Flow Imbalance (OFI, Cont-Kukanov-Stoikov 2014)
- Kyle's lambda (price impact per signed volume, Kyle 1985)
- Queue depletion (top-level liquidity exhaustion)

These signals exist in microstructure literature with documented per-trade
edge magnitudes 0.05-0.30%. Whether BingX free websocket retains enough
resolution is the open question.

---

## Theory Anchors (locked before code)

1. **Cont, Stoikov, Talreja (2010) "A stochastic model for order book dynamics"** —
   imbalance-driven mid-price prediction.
2. **Cont, Kukanov, Stoikov (2014) "The price impact of order book events"** —
   OFI as primary price-impact driver.
3. **Kyle (1985) "Continuous auctions and insider trading"** — lambda from
   signed-volume regression.
4. **Lehalle & Laruelle (2018) "Market Microstructure in Practice"** — queue
   depletion at top-of-book signals imminent price level breach.

---

## Locked Features (4)

### F1 — Order Book Imbalance (OBI)
```python
OBI(t) = (Σ bid_qty_0..4) / (Σ bid_qty_0..4 + Σ ask_qty_0..4)  # in [0, 1]
signal = OBI(t) - 0.5    # in [-0.5, +0.5], positive = bid-heavy
```
- Window: per-snapshot
- Signal direction: positive → expect upward mid-move

### F2 — Order Flow Imbalance (OFI, simplified)
```python
For each Δt = 1s window:
  bid_volume_added   = sum of (bid_qty_0[t] - bid_qty_0[t-1]) when bid_px_0 unchanged
  ask_volume_added   = sum of (ask_qty_0[t] - ask_qty_0[t-1]) when ask_px_0 unchanged
  OFI(t) = bid_volume_added - ask_volume_added
  normalized by avg(bid_qty_0 + ask_qty_0) over window
```
- Window: 1-second bucket
- Signal direction: positive → expect upward mid-move next bucket

### F3 — Kyle's Lambda (signed-trade cumulative impact)
```python
For each Δt = 5s window:
  signed_vol = Σ(qty if buyer else -qty)  from trades where is_buyer_maker=False is buyer
  Δmid = mid[t+5s] - mid[t]
  λ̂ = OLS slope of Δmid vs signed_vol over rolling 5min window
  signal(t) = signed_vol(t) × λ̂(t)
```
- Window: 5-second bucket
- Signal direction: positive → expect upward 5s-ahead mid-move

### F4 — Top-Level Queue Depletion
```python
For each snapshot:
  bid_top_qty(t)  vs   bid_top_qty over last 30s mean
  if bid_top_qty[t] / mean_30s < 0.3 AND ask_top_qty[t] / mean_30s > 0.7:
    signal = -1   # bid queue empty → expect downward break
  elif ask_top_qty[t] / mean_30s < 0.3 AND bid_top_qty[t] / mean_30s > 0.7:
    signal = +1   # ask queue empty → expect upward break
  else:
    signal = 0
```
- Window: snapshot vs 30s trailing
- Signal direction: as labeled

---

## Arithmetic Gate (per R41 standard)

For each feature F_i:

```
For each event t with |signal(t)| > threshold_i:
  predict direction = sign(signal(t))
  hold to t + horizon_i (5s for OBI/OFI/λ, 30s for queue depletion)
  realized_gross_pct = (mid[t+h] - mid[t]) / mid[t] × predicted_direction × 100

avg_gross = mean(realized_gross_pct over all events)
hit_rate  = mean(realized_gross_pct > 0)

PASS if (avg_gross × hit_rate − friction_round_trip_pct) > 0
where friction_round_trip_pct = 0.07% (BingX taker)
```

**Thresholds locked**:
- F1: |signal| > 0.10 (i.e., OBI > 60% or < 40%)
- F2: |OFI normalized| > 1.0 (1σ event)
- F3: |signed_vol × λ̂| > 1.0 (top decile)
- F4: as defined (binary +1/-1/0)

**Min sample size**: N ≥ 500 events per feature (R41 had N=2,760 for arithmetic certainty).

---

## Sample-Size Adequacy

18.17h × 1.97 Hz ≈ 128,596 snapshots. F1 events at >0.10 threshold: estimate
~10-20% of snapshots = 12,000-25,000 events. F2 1s buckets: 65,400.
F3 5s buckets: 13,080. F4 events: ~5% = 6,400.

All exceed N≥500 by 10×+. **Sample sufficient for arithmetic gate**.

---

## Pre-Registered Outcomes

### PASS (any 1 of 4 features clears arithmetic gate)
- Proceed to: full OOS post 4-week collection. R14-substrate-shift confirmed.
- L2 collector continues; arithmetic gate result not yet final OOS.

### BORDERLINE (avg_gross × hit_rate ∈ [+0.05%, +0.07%])
- Call advisor for reconcile (relaxing gate threshold = anti-pattern).

### FAIL (all 4 features avg_gross × hit_rate < +0.05%)
- Same envelope falsification as bar-level. Substrate change does not lift edge above
  friction. Tell user: "23 OHLCV/funding rounds + 4 microstructure features = 27/27 negative
  at retail BingX 1× friction floor."
- L2 collector continues for full 4-week observation but mechanism candidacy null.

---

## Anti-Adjustment

Features F1-F4, thresholds, horizons, friction LOCKED. No retuning post-FAIL.

---

## Hash Anchor

Committed BEFORE signal-extraction code.
