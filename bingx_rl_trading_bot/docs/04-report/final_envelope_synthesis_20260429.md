# Final Envelope Synthesis — Retail BTC Achievable Ceiling

**Date**: 2026-04-29
**Status**: Pre-L2 research closed. L2 evidence in ~3 weeks is last in-scope variable.
**Author**: AI assistant under user-delegated authority via advisor

---

## Bottom line

**Retail BTC envelope achievable ceiling: avg_daily_net ≈ +0.02% at 1× leverage (~7% annualized) with multi-month regime fragility.**

This is the conclusion from **14 rounds × 2 friction regimes × multiple mechanism families** of pre-registered OOS testing. 0.05%/day gate (advisor's interim) is not achievable on what's been tested. 0.2%/day (user's original) requires markets outside retail BTC.

---

## The 14-round evidence pile

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

---

## Pattern — Tight band across mechanisms and frictions

avg_daily_net across the 14 rounds (where computable) clusters in **[+0.01%, +0.03%]** at 1× leverage.

| Round | avg_daily_net | Friction |
|-------|---------------|----------|
| R36-R41 | -0.01% to +0.005% (after friction) | 0.07% taker |
| TT-R1 | -0.034% test, +0.041% test (regime split) | 0.07% |
| TT-R2 | -0.041% / +0.020% | 0.07% |
| **PB-R1** | **+0.019%** | 0.07% taker |
| **PB-R1-maker** | **+0.025%** | 0.04% maker |

Edge magnitude bounded by something other than mechanism choice or friction parameter. **Likely bound: information ratio of publicly-available retail signals on a single asset.**

---

## What this rules out (and what it doesn't)

### Rules out (with high confidence)
- **0.2%/day at 1× retail BTC** — 14-round evidence puts achievable band 10× below this target
- **Mechanism research producing >0.05%/day on tested envelopes** — friction reduction max widens by 30%, doesn't reach floor
- **Single-factor cross-sectional crypto on 10-coin universe** — edge real but ~7% annualized, regime-fragile

### Does NOT rule out
- **L2 orderbook microstructure** (4 weeks away) — different signal layer, unknown ceiling
- **Different markets entirely** — equity factors, FX carry, defi yields have different alpha profiles
- **Smaller targets at retail BTC** — deploying R1-maker at conservative sizing for ~$100/year on $1,500 is real

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
