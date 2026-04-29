# Round 24 — ICT/SMC Liquidity Sweep + Reversal (TradingView-inspired)

**Date pre-registered**: 2026-04-30
**Status**: PRE-COMMIT (locked before strategy code)
**Track**: TradingView community-inspired (LuxAlgo SMC family), substrate-distinct from R1-R23

---

## DISCLOSURE

27 prior investigations: 23 OHLCV+funding rounds + 4 L2 microstructure features
+ R5+leverage frontier characterization. All negative within retail BingX 1× envelope.
LIVE-parity prior 0/1 (C1).

R24 explicit user authorization (2026-04-30): "TradingView 전략 쉐어하는 곳에서
받은 영감을 토대로 진행할 것". This is substrate-distinct because:

- All 23 prior OHLCV rounds tested **continuous-signal threshold** mechanisms
  (channel breakout, fade, RSI/MACD, momentum, calendar, MTF confluence,
  reversal at extreme threshold).
- L2 features tested **point-in-time microstructure** (OBI, OFI, Kyle, queue).
- **None tested multi-bar structural pattern detection** (pivot identification +
  sweep validation + reversal entry).

R24 = pattern detection class, not signal-threshold class. Different feature
extraction even on same OHLCV substrate.

---

## Theory Anchor

1. **Lehalle & Laruelle (2018) "Market Microstructure in Practice"** — chapter on
   liquidity-driven price action, stop-hunt detection.
2. **ICT (Inner Circle Trader) methodology** — popularized in retail trading
   community via TradingView SMC (Smart Money Concepts) indicators.
3. **LuxAlgo "Smart Money Concepts"** Pine Script — popular open-source TradingView
   indicator implementing liquidity sweeps, order blocks, FVGs.
4. **Mechanism economic story**: stop-loss orders cluster above recent swing highs
   and below recent swing lows. Aggressive participants briefly push price beyond
   level to trigger liquidations (liquidity acquisition), then orderflow reverses
   as smart-money accumulates against panic flow.

---

## Locked Parameters

```python
LOCKED = {
    'asset': 'BTC/USDT',
    'tf': '1h',
    'pivot_lookback_bars': 10,          # bars on each side for pivot detection
    'min_pivot_age_bars': 10,           # must wait full pivot validation
    'max_pivot_age_bars': 72,           # ignore stale pivots (3 days)
    'sweep_min_wick_pct': 0.05,         # wick must exceed pivot by ≥ 0.05%
    'sweep_close_inside_rule': True,    # close must return inside pivot (rejection)
    'atr_period': 14,
    'sl_atr_buffer_mult': 0.1,          # SL = sweep_extreme ± 0.1×ATR
    'rr_target': 2.0,                   # TP = entry ± 2× initial_risk
    'max_hold_bars': 24,                # 24h hard timeout
    'friction_pct': 0.07,               # taker RT = 2 × 0.035
    'capital_usd': 1500,
    'position_size_usd': 1500,          # 1× single-position
}
```

Logic per bar t:
1. Identify pivot high P_H (resp. pivot low P_L) where bar in last
   `pivot_lookback_bars + min_pivot_age_bars` window has highest high (lowest low)
   surrounded by N bars each side with strictly lower (higher) extremes.
2. While pivot active (age ≤ max_pivot_age_bars), watch each new bar:
   - If high[t] > P_H + 0.05%·P_H AND close[t] < P_H → BEARISH sweep at P_H, signal SHORT
   - If low[t] < P_L − 0.05%·P_L AND close[t] > P_L → BULLISH sweep at P_L, signal LONG
3. Entry at next bar open (t+1). Pivot consumed (no re-trigger).
4. SL: SHORT → high[t] + 0.1×ATR(14); LONG → low[t] − 0.1×ATR(14)
5. TP: 2× initial risk distance from entry.
6. Exit at SL/TP touch (intrabar) or max_hold_bars timeout.
7. Friction 0.07% RT applied to gross PnL.

---

## Pre-run Gates

### Gate A — Sufficient setups
- ≥ 100 sweeps across 720 days
- **Pass**: ≥ 100 events
- **Fail**: too rare, vacuous

### Gate B — Random-baseline (anti-fix-impulse)
- 1000 random-entry simulations matching trade count + direction distribution
- **Pass**: actual cum_net > 95th percentile of random

---

## Pre-Registered Tests (Korean criteria, A+D+E user trade-offs)

T1 WF 5-fold (≥3/5 positive)
T2 Bootstrap 1000 × 3-day (pos_rate ≥ 50%)
T3 Train/Test 60/40 (BOTH positive)
T4 (HARD) daily ≥ 0.20%
T5 WR ≥ 30% (relaxed via A — R:R 2.0 means BE WR = 33%)
T6 R:R ≥ 1.0 (locked at 2.0)
T7 (HARD) trades/day ≥ 2
T8 (HARD) per-trade gross > 0.07%
T9 worst 5d ≥ -15%

---

## EV Estimate

- All HARD PASS: **5-10%** (28번째 round, prior 0/27)
- T4 fail by < 0.05%: 25-30% (most-likely borderline)
- Catastrophic fail (R10/R11-style negative): 10-15%
- T7 fail (frequency too low): 30-40% (pattern strategies often <1/day)
- Mixed: 15-20%

**Honest expectation**: T7 frequency or T8 magnitude likely binding. ICT/SMC
community claims 60-80% WR but most lack honest friction accounting.

---

## Anti-Adjustment

Pivot lookback (10), sweep wick threshold (0.05%), R:R (2.0), max hold (24)
LOCKED. No retuning post-FAIL.

---

## Hash Anchor

Committed BEFORE strategy code.
