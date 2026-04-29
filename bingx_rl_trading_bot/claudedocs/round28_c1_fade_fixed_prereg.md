# Round 28 — C1-Fade-Fixed: Reverse Direction + Range-Based Fixed Targets

**Date pre-registered**: 2026-04-30
**Status**: PRE-COMMIT (locked before strategy code)
**Track**: User-specific 2026-04-30 — "C1을 더 가다듬는" 방향, 4가지 modifications combined

---

## DISCLOSURE

User explicitly requested 4 modifications applied together to C1 baseline:
1. Remove trailing stop → use fixed targets
2. Reverse entry direction (short on upward breakout, long on downward) — counter-trend FADE
3. TP = N × breakout candle range (multiplier-based, not ATR)
4. SL = 1 × breakout candle range (opposite-side detected distance)

These 4 are combined into ONE locked variant. "다양한 방식 추가 확인" (post-hoc multi-variant
sweep) is REFUSED per fix-impulse anti-pattern memory.

C1 baseline = 15-bar Donchian channel breakout + body > 40% range, fractal SL,
2.5×ATR trailing TP. C1 LIVE result: -12.86%/14d at n=46, BT-LIVE parity broke,
strategy SHELVED. R28 tests if direction reversal + fixed range-based targets
recover the regime.

LIVE-parity prior 0/1 (C1).

---

## What's distinct from prior 31 rounds

| Round | Detection | Direction | TP method | SL method |
|-------|-----------|-----------|-----------|-----------|
| C1 v2 | 15-bar Donchian + 40% body | with-trend | trailing 2.5×ATR | fractal capped 3.3×ATR |
| R8 | 15-bar Donchian + 40% body | with-trend | trailing | fractal |
| R10 | MTF confluence breakout | with-trend | trail | ATR |
| R11 | extreme single-bar reversal | reverse | ATR-based | tight ATR |
| R24 | sweep + reversal | reverse (opposite of sweep) | R:R 2.0 fixed | sweep-wick + ATR |
| **R28** | **15-bar Donchian + 40% body** | **REVERSE (fade)** | **fixed 2× candle_range** | **fixed 1× candle_range** |

R28 differs from R11 (which used 5m timeframe + ATR-based exits) and R24 (different
detection mechanism — pivot sweep vs Donchian breakout).

---

## Theory Anchor

1. **R10 catastrophic fail finding**: BTC 2024-2026 = anti-breakout regime
   (cum -94%, Sharpe -8.33). Suggests reverse direction may have edge.
2. **R11 reversal extension**: showed reversion exists (+0.008%/6-bar) but per-trade
   too small. R28 differentiates by TF (1h vs 5m) and target method (range-multiple
   vs ATR fixed).
3. **Crabel (1990) volatility expansion**: large body bars often signal exhaustion,
   not continuation. Fade thesis.
4. **Murphy (1999) "Technical Analysis"** chapter on false breakouts: aggressive
   breakouts often fail when not supported by volume/momentum.

---

## Locked Parameters

```python
LOCKED = {
    'asset': 'BTC/USDT',
    'tf': '1h',
    'channel_lookback_bars': 15,        # C1 baseline
    'body_min_pct_of_range': 0.40,      # C1 baseline
    'direction': 'fade',                # REVERSED from C1
    'tp_candle_range_multiple': 2.0,    # TP = 2 × breakout candle's (high-low)
    'sl_candle_range_multiple': 1.0,    # SL = 1 × breakout candle's (high-low)
    'max_hold_bars': 192,               # C1 baseline (8 days at 1h)
    'taker_per_side_pct': 0.05,
    'maker_per_side_pct': 0.02,
    # Entry: market at next bar open (taker stop-style for fade)
    # TP: limit at target (maker)
    # SL: market at level (taker)
    'capital_usd': 1500,
    'position_size_usd': 1500,
}
```

Logic per signal at bar t:
1. C1 detection: close[t] > 15-bar high AND body > 40% range AND close[t] > open[t]
   → BULLISH breakout signal (for fade, SHORT entry)
2. Or: close[t] < 15-bar low AND body > 40% AND close[t] < open[t]
   → BEARISH breakdown signal (for fade, LONG entry)
3. Compute candle_range[t] = high[t] − low[t]
4. Entry at next bar open (t+1), market (taker 0.05%):
   - SHORT (for bullish fade): entry at open[t+1]
   - LONG (for bearish fade): entry at open[t+1]
5. SL: SHORT → entry + 1 × candle_range[t]; LONG → entry − 1 × candle_range[t]
   Hit at market (taker 0.05%)
6. TP: SHORT → entry − 2 × candle_range[t]; LONG → entry + 2 × candle_range[t]
   Hit at limit (maker 0.02%)
7. Max hold 192 bars, timeout exit at market (taker 0.05%)
8. R:R = TP_distance / SL_distance = 2.0 (locked)

Sample friction:
- TP exit: 0.05 + 0.02 = 0.07% RT
- SL exit: 0.05 + 0.05 = 0.10% RT
- Timeout: 0.05 + 0.05 = 0.10% RT

---

## Pre-run Gates

### Gate A — Sufficient signals
- ≥ 100 C1 detections in 720 days
- **Pass**: ≥ 100 trades

### Gate B — Random-baseline
- 1000 random-bar simulations matching trade count + side ratio
- **Pass**: actual cum_net > 95th percentile

---

## Pre-Registered Tests (User's 4 criteria 2026-04-30)

C1 (HARD) **Daily ≥ 0.20% net at 1×**
C2 **Per-trade gross > 0.07% taker RT** (avg entry+exit friction ~0.085% mean)
C3 **Trade count ≥ 100** (statistical significance)
C4 **Bootstrap 1000 × 3-day pos_rate ≥ 50%**

---

## EV Estimate

| Outcome | Probability |
|---------|------------|
| All 4 PASS, daily ≥ 0.20% | **5-10%** |
| C1 fail by < 0.10pp (sub-target but positive) | 15-25% |
| Catastrophic fail like R11 (anti-edge) | 25-35% |
| Mixed (1-3 PASS) | 30-40% |

**Honest expectation**: BTC fade strategies showed mixed results in prior rounds.
R10 catastrophic fail of breakout suggested reverse might work, but R11 showed
reversal magnitude too small (~+0.01%/trade). R28 with **larger TP target
(2× candle range vs ATR-based)** could potentially clear friction IF fade
direction has selectivity. Realistic prior: per-trade gross [+0.02%, +0.10%]
range, daily near zero or slightly negative.

---

## Anti-Adjustment

15-bar lookback, 40% body, 2.0 TP multiple, 1.0 SL multiple ALL LOCKED.
**NO PARAMETER SWEEP POST-FAIL**. If FAIL, the 4-modification combined variant
is empirically tested negative; user "다양한 방식" = autopilot trap, not
authorized work.

---

## Hash Anchor

Committed BEFORE strategy code.
