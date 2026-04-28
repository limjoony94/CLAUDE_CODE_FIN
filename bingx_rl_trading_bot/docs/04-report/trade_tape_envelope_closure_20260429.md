# Trade-Tape Envelope Closure & Friction-Floor Evidence — 2026-04-29

**Status**: Trade-tape envelope short-horizon (1m) mechanism research closed.
**Default action**: L2 collector continues running on its own schedule (Day-1 gate at +24h fires regardless). No R3 in trade-tape envelope per advisor.

---

## 1. Friction-floor pattern (3 independent rounds, structurally distinct)

| Round | Envelope | Mechanism direction | n trades | avg_gross train | avg_gross test | Friction (0.07%) |
|-------|----------|---------------------|----------|-----------------|----------------|-------------------|
| **R41** | OHLCV 5m | momentum (MACD continuation) | 2,760 | +0.0323% | +0.0342% | **gross < friction** |
| **R1**  | Trade-tape 1m | continuation (persistent imbalance) | 1,593 | +0.0288% | +0.0499% | **gross < friction** |
| **R2**  | Trade-tape 1m | **mean-reversion fade** (extreme exhaustion) | 414 | +0.0098% | +0.0329% | **gross < friction** |

3 independent envelope × direction × mechanism combinations ALL land in the same arithmetic basement. avg_gross clusters in [+0.010%, +0.050%] band, which is **30-70% below taker round-trip friction (0.07%)**.

R2 specifically tested the **structural opposite** of R1 (mean reversion vs continuation, no MTF filter, fade direction). Both directions of microstructure information eaten by friction.

---

## 2. What this establishes

**Strong**: For retail BTC perpetual at taker friction 0.07%, **bar-level (1m, 5m, 15m) directional mechanism families** — whether OHLCV-derived or trade-tape-derived, whether continuation or mean-reversion — produce gross-of-friction returns mechanically below transaction cost.

**Cannot fix by**: switching mechanisms within same friction. R36-R41 (8 OHLCV mechanisms) + R1 (trade-tape continuation) + R2 (trade-tape fade) = **10 distinct mechanism configurations** all in same basement.

**Can ONLY change by**:
- **Friction reduction** (maker-only execution with rebate, exchange tier with sub-bp fees)
- **Different signal layer** (L2 orderbook microstructure — currently waiting on collector, fundamentally different per-trade economics possible)
- **Different timeframe envelope** (daily/weekly — friction proportional to trade frequency)

---

## 3. R2 specifics (mechanism-specific failure modes beyond friction)

Even ignoring friction, R2's results show mechanism-specific issues:
- **WF 0/5** (all folds negative)
- **WR 22-34%** range — extremely low for a mean-reversion mechanism (textbook expects 50%+)
- **Bootstrap pos_rate 28.3%** — anti-edge (random expectation 50%)

Interpretation: extreme single-bar imbalance + intensity in BTC perp may indicate **continuation continuing**, not exhaustion reversal. The contrarian thesis from VPIN/Easley-O'Hara literature appears to fail in this specific market — extreme moves persist rather than reverse on this timeframe.

This is qualitatively different from R1's regime sensitivity (folds 1/4 negative, 3/5 positive). R2 has no positive folds — mean-reversion thesis falsified, not just regime-conditional.

---

## 4. Implications for next steps

Per advisor closure direction:

> "After R2, do not run R3. If both fail, write the closure update for trade-tape envelope and present friction-floor evidence to user with the implication: only friction-reduction paths remain workable for short-horizon BTC mechanisms."

### Active tracks remaining
1. **L2 collector** (forward) — only path that hasn't been falsified by friction-floor evidence yet, because L2 mechanisms can have different per-trade economics (e.g., basis arb, depth imbalance with sub-bar holding). Day-1 gate at 2026-04-30 04:36 KST fires regardless.

### Closed tracks
1. **OHLCV envelope** (M3 closure 2026-04-29 doc) — 8 rounds × strict OOS = 0 production-grade
2. **Trade-tape 1m envelope** (this doc) — 2 rounds R1+R2 = friction-floor confirmed

### User-actionable options after L2 collector evidence in (~4 weeks)
The friction-floor pattern strongly suggests:
- **Path A — Friction reduction**: Switch execution to maker-only with rebate (requires limit-order infrastructure + slippage tolerance) OR move to higher-tier exchange
- **Path B — Lower-frequency**: Daily/weekly multi-asset rotation (friction proportional to trade frequency)
- **Path C — Wait for L2 evidence**: 4-week recording, then test orderbook-imbalance / basis mechanism in L2 envelope
- **Path D — Frontier admission**: Acknowledge retail-friction + 1m/5m/15m + BTC-only is not a profitable envelope; deploy capital differently

Decision deferred until L2 evidence available.

---

## 5. Status of evidence pile (consolidated)

### OHLCV envelope (8 rounds)
| Round | Mechanism | Verdict |
|-------|-----------|---------|
| R9b | Donchian variant | FAIL |
| R15 | Timeframe potential | FAIL |
| R19 | α N=4 fixed exit | FAIL |
| R30 | C1 production exact | partial pass, **LIVE -12.86%** |
| R36 | EMA pullback 15m | FAIL |
| R37 | NR7+BB squeeze | FAIL |
| R38 | VWAP reversion | inconclusive (vacuous) |
| R39 | ORB session | FAIL |
| R40 | Volume absorption | inconclusive (vacuous) |
| R41 | MACD minimal | **FAIL (arithmetic)** |

### Trade-tape 1m envelope (2 rounds)
| Round | Mechanism | Verdict |
|-------|-----------|---------|
| R1 | Persistent imbalance (continuation) | FAIL |
| R2 | Extreme imbalance fade (mean-reversion) | **FAIL (arithmetic)** |

### L2 forward envelope (collecting)
- Collector running PID 14548 since 2026-04-29 04:19 UTC
- Day-1 gate auto-fires at 2026-04-30 04:36 KST
- 4-week recording target before first mechanism candidate

---

## 6. Honesty caveats

### What R1+R2 do NOT establish
- Trade-tape envelope at **higher timeframes** (15m/30m/1h aggregations) — not yet tested
- Trade-tape **conditional regimes** (e.g., low-vol vs high-vol mechanism switch) — not yet tested
- L2 mechanisms — fundamentally different signal type, not yet tested

### What R1+R2 strongly suggest
- BTC perp 1m bar-level mechanisms at retail friction = falsified envelope
- Same friction-floor will likely apply to 5m/15m bar-level trade-tape aggregates
- Friction reduction OR fundamentally different signal layer required for any short-horizon BTC mechanism research to be worthwhile

---

## 7. Next advisor call gates (unchanged from before)

1. L2 Day-1 gate result (2026-04-30) if RED
2. 4 weeks L2 data + first mechanism candidate ready for OOS pre-reg
3. Architectural fork (e.g., decision to pursue Path A/B/D)

Don't call advisor for routine status updates.

---

## References
- R41 result: `bingx_rl_trading_bot/results/m3_r41_macd_oos_20260429_035953.json`
- R1 result: `bingx_rl_trading_bot/results/trade_tape_r1_oos_20260429_053325.json`
- R2 result: `bingx_rl_trading_bot/results/trade_tape_r2_oos_20260429_053648.json`
- M3 closure: `docs/04-report/m3_closure_recommendation_20260429.md`
- Trade-tape R1 pre-reg: `claudedocs/trade_tape_r1_persistent_imbalance_prereg.md`
- Trade-tape R2 pre-reg: `claudedocs/trade_tape_r2_extreme_fade_prereg.md`
