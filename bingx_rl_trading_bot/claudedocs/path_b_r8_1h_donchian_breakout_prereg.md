# Path B R8 — 1h Donchian Breakout with Static ATR TP/SL

**Date pre-registered**: 2026-04-29
**Status**: PRE-COMMIT (locked before strategy code)
**Track**: Path B R8 — Round 18 (C1-style mechanism on relaxed timeframe per user trade-offs)
**Authority**: User 2026-04-29 explicit acceptance: A (WR↓ R:R↑) + D (timeframe shift) + E (MDD↑)

---

## DISCLOSURE per advisor mandate

C1 strategy was found via the same BT-screen methodology and produced LIVE
**-12.86%/14d failure** (postmortem `c1_breakout_postmortem_20260427.md`).
LIVE-parity prior is 0/1. R8 is research artifact, NOT deploy candidate
without further LIVE-parity validation.

C1's specific LIVE-failure modes (TRAILING_STOP_MARKET intrabar trigger
gap, MARKET slippage cycle, BT-LIVE timing mismatch) are explicitly
addressed in R8 design via STATIC TP/SL (no trailing).

---

## What's distinct from C1 + 18 prior rounds

C1 was 15m Donchian + fractal SL + ATR TRAILING TP (which had structural
LIVE-parity gap). R8 differs in 3 ways:

1. **Timeframe**: 15m → **1h** (D acceptance — lower noise, intrabar gap
   shrinks dramatically)
2. **TP type**: ATR TRAILING → **STATIC ATR multiple** (eliminates main
   C1 LIVE-failure mode — STOP_MARKET only, no TRAILING_STOP_MARKET)
3. **R:R locked**: 3.0 fixed (vs C1's emergent 3.36 from trailing)

R8 vs prior 18 rounds:
- M3 R36-R41: 5m/15m intra-day mechanisms, all FAIL friction-floor
- TT-R1/R2: 1m trade-tape, FAIL
- PB-R1-R7: cross-sectional, carry, cointegration, lead-lag — distinct
  alpha classes
- **R8 uniqueness**: 1h timeframe + Donchian breakout + static dynamic TP/SL.
  No prior round used 1h primary timeframe for directional momentum.

---

## Theory Anchor

1. **Donchian (1960s) original turtle-trader breakout system** — long when
   price exceeds N-bar highest high. Empirically robust across asset
   classes for >50 years.
2. **Wilder (1978) "New Concepts in Technical Trading Systems"** — ATR
   for dynamic stop placement.
3. **Faber (2007) "A Quantitative Approach to Tactical Asset Allocation"** —
   simple breakout systems on monthly data have positive expectancy.
4. **Mechanism economic story**: 1h channel breakouts on BTC capture the
   transition from consolidation to trending phase. Body filter removes
   weak breakouts. Static TP/SL means the trade outcome is decided by
   bar's H/L hitting price levels, not by trailing logic — eliminates
   C1's bar-close-evaluation-vs-intrabar-trigger structural mismatch.

---

## Locked Parameters

```python
LOCKED = {
    'asset': 'BTC/USDT',
    'timeframe': '1h',
    'channel_lookback_bars': 24,             # 24h
    'body_min_ratio': 0.40,                  # body / (high-low) ≥ 0.40 filters weak breakouts
    'atr_period': 14,                        # bars
    'sl_atr_mult': 1.0,
    'tp_atr_mult': 3.0,                      # R:R 3.0 (within user A acceptance)
    'max_hold_bars': 48,                     # 48h max
    'friction_per_transaction_pct': 0.07,    # taker round-trip per leg
    'capital_usd': 1500,
    'leverage': 1.0,
    'cooldown_bars_after_exit': 1,           # 1h cooldown
}
```

Logic:
1. At each closed 1h bar t, check: close > 24-bar high (long signal) OR
   close < 24-bar low (short signal).
2. Body filter: |close - open| / (high - low) ≥ 0.40.
3. Cooldown: if exited last bar, skip.
4. Entry at next 1h bar's open (not current close — bar-close eval in BT
   matches LIVE bar-close detection).
5. SL = entry ∓ 1.0 × ATR(14, 1h)
6. TP = entry ± 3.0 × ATR(14, 1h)
7. Exit if H/L hits SL/TP intrabar (mark price at midpoint of bar's range
   for fair backtesting), else TIMEOUT at 48h max hold.
8. Friction: 0.07% × 2 = 0.14% RT per trade.

---

## Pre-run Gates

### Gate A — Channel breakout frequency
- 24h Donchian breakout events / total bars over 720d
- Must produce ≥ 1,000 candidate breakouts
- **Pass**: ≥1,000 events
- **Fail**: too few events for statistics

### Gate B — Body filter retention
- After body filter, ≥ 40% of breakouts retained
- **Pass**: filter doesn't kill all signal
- **Fail**: body filter too aggressive

### Gate C (anti-fix-impulse) — Random-baseline comparison
- 1000 random-entry simulations matching trade frequency and direction
  distribution
- **Pass**: actual cum_net > 95th percentile of random distribution

---

## Pre-Registered Tests (Korean criteria, A+D+E relaxed)

### Test 1: WF 5-fold expanding
- **Pass**: ≥3/5 folds positive cum_net

### Test 2: Bootstrap 1000 × 3-day windows (USER REQUIREMENT)
- **Pass**: pos_rate ≥ 50%

### Test 3: Train/Test 60/40 sign-agreement
- **Pass**: BOTH positive

### Test 4 (HARD): daily net ≥ 0.2% at 1×
- USER GATE: not relaxed
- **Pass**: avg_daily_net_pct ≥ 0.2%

### Test 5: WR ≥ 30% (RELAXED via A — was 40%)
- **Pass**: win_rate ≥ 0.30

### Test 6: R:R ≥ 1.5 (RELAXED via A — was 1.0; tighter target since R:R is locked at 3.0)
- **Pass**: avg_win/|avg_loss| ≥ 1.5

### Test 7 (HARD): trades/day ≥ 2
- USER GATE: not relaxed
- **Pass**: avg_trades_per_day ≥ 2.0

### Test 8 (HARD): per-trade gross > 0.07% taker RT
- USER GATE: not relaxed
- **Pass**: avg_gross_per_trade_pct > 0.07%

### Test 9: tail worst 5d ≥ -15% (RELAXED via E — was -10%)
- **Pass**: worst_5d_net ≥ -15%

---

## Verdict Logic

| Outcome | Meaning |
|---------|---------|
| Gate A/B/C fail | Mechanism vacuous or no edge |
| Hard gates (T4/T7/T8) PASS + others PASS | **Round 18 candidate. Surface for advisor review.** Live-parity validation REQUIRED before any deploy. |
| T4 fail | daily target missed (most likely outcome) |
| T7 fail | 1h frequency too low for ≥2/day |
| Hard PASS + relaxed (T5/T6/T9) borderline | per user acceptance, surface anyway |

---

## EV Estimate (logged before result)

| Outcome | Probability | Justification |
|---------|------------|---------------|
| All PASS (hard + relaxed) | 8-12% | C1 had similar profile in BT, R8 is C1 timeframe shift |
| Hard PASS only | 15-20% | timeframe shift may compress R:R |
| T4 fail (others PASS) | 30-40% | most likely — aligned with 18-round pattern |
| T7 fail (frequency too low) | 25-35% | 1h breakouts may be < 2/day in low-vol regimes |
| Gate C fail (random beats) | 5-10% | breakouts have weak but real signal |

**Realistic expectation**: T4 borderline FAIL or T7 fail. Result is 19th
data point on alpha ceiling. Gate C PASS would still be informative.

---

## Anti-Adjustment Provisions

1. Channel lookback 24 bars LOCKED. 12/48-bar variants are R9.
2. ATR multipliers (1.0/3.0) LOCKED.
3. Body filter 0.40 LOCKED.
4. Max hold 48h LOCKED.
5. **No retuning post-FAIL.** Result stands.
6. STATIC TP/SL LOCKED — explicitly avoids C1's TRAILING gap.

---

## Honest Caveats (per advisor mandate, expanded)

1. **C1 LIVE failure**: same screen could produce another LIVE-failed BT.
   Even if BT passes, deploy requires walk-forward LIVE validation before
   real money exposure.
2. **Static TP at 3×ATR is conservative vs C1's trailing** — may underperform
   trailing in strong trends but eliminates intrabar timing gap.
3. **1h timeframe lower noise floor** than 5m/15m R36-R41 but breakout signals
   may be slower to enter (1 bar lag), missing fast moves.
4. **Per-trade gross > 0.07% gate** — at static R:R 3.0, single TP hit yields
   3×ATR_pct ≈ 1-3% gross; single SL hit yields 0.5-1% loss. With WR 30%+
   expected positive net per trade if WR ≥ ~1/(1+R:R) = 25%.
5. **C1 LIVE selection drift** (3.1 trades/day BT → 1/day LIVE): R8 STATIC
   TP avoids the TRAILING-trigger drift but cannot eliminate ALL LIVE-parity
   gaps. Real deploy requires shadow/paper period with measured drift.

---

## Hash Anchor

Committed BEFORE strategy code. Pre-data-look theory + locked mechanism +
EV estimate + LIVE-parity disclosure. Result file timestamps post this commit.
