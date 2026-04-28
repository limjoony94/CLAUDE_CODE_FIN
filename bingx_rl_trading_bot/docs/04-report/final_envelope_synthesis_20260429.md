# Final Envelope Synthesis — Retail Capital Achievable Ceiling (15 rounds, 4 alpha families)

**Date**: 2026-04-29 (updated post-DeFi-Track R1)
**Status**: Pre-L2 research closed. L2 evidence in ~3 weeks is last in-scope variable.
**Author**: AI assistant under user-delegated authority via advisor

**v2 update (2026-04-29 post-R1)**: DeFi-Track R1 added as Phase 4. 15-round
total. Same magnitude pattern as 14-round retail BTC arc — 4th alpha family
(yield rotation) hits same ceiling.

---

## Bottom line

**Retail capital envelope at $1,500: net annualized ~$25-100/year (1-7% APY) regardless of alpha family.**

This is the conclusion from **15 rounds × 3 friction regimes × 4 alpha families** of pre-registered OOS testing:
- BTC perp directional (8 rounds, OHLCV)
- BTC perp microstructure (2 rounds, trade tape)
- Crypto cross-sectional momentum (3 rounds, daily)
- DeFi L2 yield rotation (1 round, monthly)

All four families produce statistically robust edge above zero, all four fail the magnitude gate at retail $1,500 capital, all four converge in the same band. The 0.05%/day gate (advisor's interim) is not achievable on any tested envelope. 0.2%/day (user's original) is not achievable at this capital scale on the tested markets.

**Strengthened hypothesis**: friction-as-fraction-of-position-size is the binding constraint, not strategy choice. At $500-positions, both taker fees (0.07%/round-trip) and L2 gas ($1-2/swap = 0.4%) consume similar fractions of available alpha across markets.

---

## The 15-round evidence pile

### Phase 1 — OHLCV envelope (8 rounds)
| Round | Mechanism | n trades | Verdict | avg_gross/trade |
|-------|-----------|----------|---------|------------------|
| R9b/R15/R19 | Donchian / TF / α exit | various | FAIL | various |
| R30 | C1 production | 939 BT | partial → **LIVE -12.86%** | proven |
| R36 | EMA pullback 15m | 199 | FAIL | +0.020% |
| R37 | NR7+BB squeeze | 549 | FAIL | +0.025% |
| R38 | VWAP reversion | 4 | inconclusive (vacuous) | — |
| R39 | ORB session | 549 | FAIL | +0.020% |
| R40 | Volume absorption | 279 | inconclusive (vacuous) | — |
| **R41** | **MACD minimal** | **2,760** | **FAIL (arithmetic)** | **+0.034%** |

**Conclusion**: bar-level OHLCV-conditional mechanisms cannot clear taker friction 0.07%. avg_gross/trade clusters in [+0.020%, +0.034%].

### Phase 2 — Trade-tape 1m envelope (2 rounds)
| Round | Mechanism | n trades | Verdict | avg_gross/trade |
|-------|-----------|----------|---------|------------------|
| **TT-R1** | Persistent imbalance (continuation) | 1,593 | FAIL | +0.039% |
| **TT-R2** | Extreme fade (mean-reversion) | 414 | FAIL | +0.022% |

**Conclusion**: trade-tape microstructure conditioning produces same avg_gross band as OHLCV. Both directions (continuation TT-R1 / mean-reversion TT-R2) eaten by friction. Not just OHLCV — friction-floor extends to all bar-level retail signals.

### Phase 3 — Cross-sectional Path B (3 rounds)
| Round | Mechanism | Friction | Verdict | avg_daily_net |
|-------|-----------|----------|---------|----------------|
| **PB-R1** | XS Momentum 30d (taker) | 0.07% | FAIL strict, edge>friction | **+0.019%/day** |
| PB-R2 | XS Reversal 7d (taker) | 0.07% | inconclusive (vacuous) | — |
| **PB-R1-maker** | XS Momentum 30d (maker) | 0.04% | FAIL strict | **+0.025%/day** |

**Conclusion**: Cross-sectional dimension does produce edge above friction (first such result in arc). But:
- PB-R1 daily 0.019% << 0.2% target (10× short)
- PB-R1-maker daily 0.025% << 0.05% advisor floor (2× short)
- Friction reduction (taker → maker) widens edge by ~30% (0.019 → 0.025) but doesn't change the order of magnitude
- Regime fragile — recent test segment negative

### Phase 4 — DeFi-Track L2 Yield Rotation (1 round)
| Round | Mechanism | Friction | Verdict | avg_daily_net |
|-------|-----------|----------|---------|----------------|
| **DeFi-R1** | Top-3 trailing 30d APY, monthly rebalance, L2 only | 0.4%/swap (~3.15%/yr drag) | FAIL T4 magnitude (4/5 PASS) | **+0.0049%/day** |

**Conclusion**: 4th alpha family — entirely different market (DeFi yield, not directional crypto), different signal (cross-sectional APY dispersion, not price momentum), different friction profile (gas not taker fee). Same magnitude ceiling.
- Gross APY 4.92%/yr, friction 3.15%/yr, **net 1.77%/yr = $26 on $1,500**
- T1 WF 3/5 PASS, T2 BS 64% PASS, T3 TT both positive PASS, T5 tail -0.75% PASS
- T4 FAIL by 4× (1.77% vs 7.3% gate)
- Friction = 64% of gross. Even at 0.5× friction (impossible at retail $500 positions), net would be ~3.4%/yr — still T4 FAIL
- Gas regime caveat: T5 tail PASS contingent on benign L2 gas; 3-5× spike historically possible compresses safety margin

---

## Pattern — Tight band across mechanisms, frictions, AND markets

avg_daily_net across the **15 rounds** (where computable) clusters in **[+0.005%, +0.030%]** at 1× / retail capital.

| Round | avg_daily_net | Friction | Market |
|-------|---------------|----------|--------|
| R36-R41 | -0.01% to +0.005% (after friction) | 0.07% taker | BTC perp directional |
| TT-R1 | -0.034% test, +0.041% test (regime split) | 0.07% | BTC perp microstructure |
| TT-R2 | -0.041% / +0.020% | 0.07% | BTC perp microstructure |
| **PB-R1** | **+0.019%** | 0.07% taker | crypto cross-sectional |
| **PB-R1-maker** | **+0.025%** | 0.04% maker | crypto cross-sectional |
| **DeFi-R1** | **+0.005%** | 3.15%/yr gas (= ~0.0086%/day) | DeFi L2 yield |

Edge magnitude bounded by something other than mechanism choice, friction parameter, OR market type. **Strengthened bound hypothesis: at $1,500 retail capital, friction-as-fraction-of-position-size is the binding constraint, regardless of which alpha family is sampled.** All four alpha families (perp directional / perp microstructure / cross-sectional / DeFi yield) hit ~$25-100/year ceiling on $1,500 capital.

---

## What this rules out (and what it doesn't)

### Rules out (with high confidence)
- **0.2%/day at 1× retail BTC** — 15-round evidence puts achievable band 10× below this target
- **Mechanism research producing >0.05%/day on tested envelopes** — friction reduction max widens by 30%, doesn't reach floor
- **Single-factor cross-sectional crypto on 10-coin universe** — edge real but ~7% annualized, regime-fragile
- **DeFi-Track at L2 retail (1 round)** — yield rotation works statistically (4/5 gates PASS) but gas friction binds at $500 positions; net 1.77%/yr
- **"Different markets fix the band"** — assumption was DeFi yields had different alpha profile; result shows same magnitude ceiling. Fourth independent alpha family hits same retail-capital binding.

### Does NOT rule out
- **L2 orderbook microstructure** (last in-scope, ~3 weeks away) — different signal layer, unknown ceiling
- **Capital-scale change** — same strategies at 10× capital ($15,000) where mainnet gas / Ethereum DeFi pools become economic; or 100× ($150,000) where the friction-fraction shrinks below the alpha
- **Markets entirely outside this codebase** — equity factors, FX carry — but those require different infrastructure and the user's BingX setup constraint excludes them
- **Smaller targets accepted at retail BTC** — deploying R1-maker at conservative sizing for ~$45/year on $1,500 is real

---

## Decision-on-record (advisor delegation)

**Stopping all non-L2 research now.** Factor space adequately probed. Adding R3/R4/RX with different factors would not move the achievable band — the bound is information-theoretic, not mechanism-specific.

**L2 collector continues** (Day-1 gate ~22h, 4-week target). When L2 evidence arrives ~2026-05-27, that's the last in-scope variable.

**Two binary outcomes after L2**:
- L2 mechanisms show same ~0.02%/day band → retail BTC envelope fully characterized → user decides:
  - Deploy R1-maker small (~$100/year, paid education + capital preservation)
  - OR accept goal (≥0.2%/day) requires markets outside retail BTC
- L2 mechanisms break the band → first real ceiling violation in arc → pursue

---

## What to deploy if user accepts ~0.02%/day band (Option 1 future)

If after L2 evidence the user accepts the actual achievable band:

**Best-of-class candidate**: PB-R1-maker (XS momentum, 30d, weekly rebalance, maker-only execution)
- Universe: 10 large-cap crypto USDT pairs
- Lookback: 30 days
- Long top-3 / short bottom-3 by trailing return, equal weight
- Weekly Monday rebalance
- Maker-only execution (limit orders), 0.04% friction round-trip
- Conservative sizing: 0.43× nominal scale (cap MDD at 25% of $1,500 = $375)
- Expected return: ~3% annualized on actual $ deployed → ~$45/year on $1,500
- Better as paid learning + alpha-tracking exercise than production capital

Limit-order execution requires 2-4 weeks dev (maker-only requires limit orders that fill — non-trivial in fast markets).

---

## Honesty caveats

1. **Advisor's 0.05%/day floor was wrong.** Anchored on plausible plus, didn't price in regime fragility. Adjusted to ~0.02% based on evidence.
2. **Sample sizes vary.** PB-R1-maker has only 800 daily bars (~26 months). Wider conclusions deserve longer history.
3. **L2 has its own ~50% prior of failing too.** Don't oversell L2 wait.
4. **0.02%/day band is small.** $0.30/day on $1,500. Below daily noise of holding any single coin.

---

## Next gates (no advisor call until)

1. L2 Day-1 inspection RED (2026-04-30 04:36 KST)
2. L2 4-week mechanism candidate ready (~2026-05-27)
3. User explicit redirect outside this synthesis

---

## File references (chronological)

- M3 closure: `docs/04-report/m3_closure_recommendation_20260429.md`
- Trade-tape closure: `docs/04-report/trade_tape_envelope_closure_20260429.md`
- Path B synthesis: `docs/04-report/path_b_synthesis_20260429.md`
- This document: cumulative final synthesis

Pre-regs (chronological):
- M3-R36 through R41
- Trade-Tape R1, R2
- Path B R1, R2, R1-maker

---

## Memory anchor

Update memory: "retail BTC achievable ceiling ~0.02%/day at 1×, ~7% annualized; pre-L2 research closed; L2 evidence in 3-4 weeks is last variable."

No further factor research without explicit user redirect or L2 evidence movement.
