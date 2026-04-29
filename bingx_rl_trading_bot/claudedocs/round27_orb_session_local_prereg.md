# Round 27 — Opening Range Breakout (Session-Local Formation)

**Date pre-registered**: 2026-04-30
**Status**: PRE-COMMIT (locked before strategy code)
**Track**: User redirection 2026-04-30 — "단기적으로 지표가 형성되는 조건"

---

## DISCLOSURE

R1-R26 all used continuous-rolling-window indicators (200d SMA, 14-bar RSI,
20-bar Donchian, 10-bar ATR, etc.). User's new framing: indicators that **form
fresh on short-term session-local windows**, not all-time rolling.

R27 = Opening Range Breakout (ORB), session-bounded formation:
- UTC day starts at 00:00. First 4 hours = "opening range" (OR).
- After 04:00 UTC: trade breakouts of OR_high (long) or OR_low (short).
- OR resets daily — no all-time history dependency.

LIVE-parity prior 0/1 (C1).

---

## What's distinct from prior 30 rounds

| Round | Indicator type | Reset cycle |
|-------|---------------|-------------|
| R8 Donchian | 15-bar rolling channel | continuous |
| R10 MTF confluence | multi-TF rolling | continuous |
| R12 time-of-day | fixed hour entry, all-time SMA | continuous |
| R24 ICT pivot sweep | rolling pivot lookback | continuous |
| R25 maker-Donchian | rolling channel | continuous |
| R26 grid | rolling ATR percentile | continuous |
| **R27 ORB** | **session-bounded range** | **daily reset** |

ORB is **session-local indicator formation**: today's 00:00-04:00 UTC range is
the only data input for today's entries. Yesterday's data not used. This is the
substrate-distinct mechanism.

---

## Theory Anchor

1. **Toby Crabel (1990) "Day Trading with Short Term Price Patterns and Opening
   Range Breakout"** — original ORB methodology, well-documented in retail.
2. **Crabel + Larry Williams** trading literature: ORB exploits institutional
   participants entering during morning hours, defining day's volatility envelope.
3. **TradingView community ORB strategies** — many public Pine implementations,
   common claim 0.4-1.0%/day on stocks (transferability to crypto open question).
4. **Mechanism economic story**: Asian → London (07:00 UTC) and London → US
   (13:00 UTC) handoffs concentrate institutional flow. First 4 UTC hours
   (00:00-04:00) define quiet pre-Asian / Asian range. Break of this range
   often signals directional regime for remainder of day.

---

## Locked Parameters

```python
LOCKED = {
    'asset': 'BTC/USDT',
    'tf': '1h',
    'opening_range_start_utc': 0,       # 00:00 UTC
    'opening_range_end_utc': 4,         # 04:00 UTC (4-hour range)
    'trade_window_start_utc': 4,        # trade entries from 04:00 UTC
    'trade_window_end_utc': 22,         # exit by 22:00 UTC same day
    'entry_method': 'stop_breakout',    # taker stop order at OR boundary break
    'breakout_buffer_pct': 0.0,         # break exact OR level (no buffer)
    'sl_atr_buffer_mult': 0.1,          # SL = opposite OR side - 0.1 × ATR(14)
    'tp_method': 'or_range_multiple',
    'tp_or_multiple': 1.5,              # TP = 1.5× OR_range from entry
    'max_hold_bars_intraday': 18,       # max 18h hold (day-bounded)
    'max_trades_per_utc_day': 1,        # one shot only
    'friction_taker_per_side_pct': 0.05,
    'friction_maker_per_side_pct': 0.02,
    'tp_exit_method': 'limit_maker',    # TP fills as limit (maker)
    'sl_exit_method': 'market_taker',   # SL hit at market (taker)
    'capital_usd': 1500,
    'position_size_usd': 1500,
}
```

Logic per UTC day d:
1. From 00:00 to 03:59 UTC: collect OR_high (max high) and OR_low (min low).
2. At 04:00 UTC: lock OR. Compute OR_range = OR_high - OR_low.
3. From 04:00 to 22:00 UTC, walk forward 1h bars:
   - If high[t] > OR_high → triggered LONG at OR_high (taker stop entry, 0.05%)
   - If low[t] < OR_low → triggered SHORT at OR_low (taker stop entry, 0.05%)
   - Whichever triggers first; if both in same bar, skip (ambiguous).
4. SL: LONG → OR_low - 0.1×ATR(14, 1h); SHORT → OR_high + 0.1×ATR(14, 1h).
   SL hit = market taker exit (0.05%).
5. TP: 1.5× OR_range from entry. TP hit = limit maker exit (0.02%).
6. Max hold: 18 bars (= until 22:00 UTC same day) → market taker timeout (0.05%).
7. Max 1 trade per UTC day. After exit (whatever reason), no re-entry today.

---

## Pre-run Gates

### Gate A — Sufficient setups
- ≥ 100 daily breakouts in 720 days
- **Pass**: ≥ 100 trades

### Gate B — Random-baseline
- 1000 random-OR-window simulations (random 4-hour window per day instead of 00:00-04:00)
- **Pass**: actual cum_net > 95th percentile

---

## Pre-Registered Tests (User's 4 criteria 2026-04-30)

C1 (HARD) **Daily ≥ 0.20% net at 1×**
C2 **Per-trade gross > 0.07% taker RT**
C3 **Trade count ≥ 100 (statistical significance)**
C4 **Bootstrap 1000 × 3-day window pos_rate ≥ 50%**

---

## EV Estimate

| Outcome | Probability |
|---------|------------|
| All 4 PASS, daily ≥ 0.20% | **5-10%** |
| C1 fail by < 0.10pp (sub-target) | 25-35% |
| Catastrophic fail (negative) | 15-25% |
| Mixed | 35-45% |

**Honest expectation**: BTC trades 24/7, no clear session boundaries like equities.
ORB designed for stocks with definitive open/close. Crypto adaptation has reduced
edge. Likely sub-target but possibly first directional 1× viable round.

---

## Anti-Adjustment

OR window (00:00-04:00 UTC), trade window (04:00-22:00 UTC), TP multiple (1.5),
SL ATR buffer (0.1) ALL LOCKED. **NO PARAMETER SWEEP POST-FAIL**. If FAIL, the
user's "단기적으로 지표가 형성되는 조건" class is empirically tested negative on
ORB representative; different specific named candidate (Pine URL, paper) needed
for further investigation in this class.

---

## Hash Anchor

Committed BEFORE strategy code.
