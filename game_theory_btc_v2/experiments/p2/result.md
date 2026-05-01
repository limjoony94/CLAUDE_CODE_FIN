# P2 Closure — Force-Flow Reversal Hypothesis Testing

**Date**: 2026-05-01 (P2 Day 1, single-config evaluation per Option α precommit)
**Status**: ❌ **FAIL** (0/8 PASS, 8/8 FAIL)
**Sealed boundary**: enforced (no leak)
**Raw**: `experiments/p2/results_raw.json`

---

## Verdict (Honest Closure per Mandate § 0.7)

**0/8 PASS** — All 4 mechanisms × 2 friction scenarios (realistic + stress) FAIL.

This is the **expected outcome** per friction-floor prior (>70% FAIL probability):
- 27 prior mechanisms × 5 substrates: 0 deployable except R5 single-coin BTC carry $49/yr
- Mandate § P2 explicit: "FAIL is expected; PASS would be a real find requiring stress validation + paper trade"

P2 envelope (force-flow reversal on free 720d Binance data) **falsified**. Phase A still has 10 untested mechanisms (P3 MAP-Elites scope).

---

## Frequency Scan (Pre-Sweep Vacuous Filter)

Free window: 540.3 days. Min event freq gate: 0.5/day (R38 lesson).

| Mech | Hypothesis | Signals | Freq/day | Vacuous? |
|------|-----------|---------|----------|----------|
| M001 | H3 force-flow long (1h fwd 4h) | 217 | 0.402 | ⚠️ VACUOUS |
| M002 | H5 force-flow short (1h fwd 1h) | 268 | 0.496 | ⚠️ VACUOUS |
| M003 | H4 cascade window long (1m fwd 15m) | 217 | 0.402 | ⚠️ VACUOUS |
| M004 | H7-basis fade perp (1h fwd 4h) | 811 | 1.501 | OK |

**M001-M003 locked threshold (5.5 sum-of-z) too strict** for free 540d sample. Signal frequency 0.40-0.50/day < 0.5/day gate → mark VACUOUS_FAIL per anti-fishing rule (no post-hoc loosening).

---

## 6-Criteria Evaluation (M004 only — M001-M003 vacuous)

### M004_realistic (basis fade ±2σ, 4h forward, 0.16% RT)

| Criterion | Threshold | Value | PASS |
|-----------|-----------|-------|------|
| mean | ≥ 0.10%/d | **-0.167%/d** | ❌ |
| p5 (bs lower CI) | ≥ 0 | -0.299% | ❌ |
| pos_rate | ≥ 0.50 | 0.135 | ❌ |
| p_beats vs B&H | ≥ 0.70 | 0.015 | ❌ |
| max_dd | ≥ -3% | **-96.4%** | ❌ |
| sharpe (annualized) | ≥ 1.5 | -1.76 | ❌ |

**0/6 PASS, status FAIL** (n_signals=811 actually evaluated).

### M004_stress (0.20% RT)

Same direction, worse magnitude:
- mean -0.227%/d, p_beats 0.004, max_dd **-127.8%** (capital ruin), sharpe -2.37
- **0/6 PASS, status FAIL**

---

## Interpretation

### Why M004 Catastrophic Loss
Basis fade at ±2σ z-score:
- 811 signals over 540 days = 1.5/day too frequent
- Each signal pays full RT friction (0.16% or 0.20%)
- Basis is mean-reverting in noise, but 4h forward window introduces drift exposure
- Net: friction × frequency overwhelms any mean reversion edge
- p_beats 0.015 (0.4% confidence beats B&H) = strategy systematically loses to passive
- max_dd -96% to -128% (stress) = full capital wipe

This is exactly the friction-floor pattern (avg_gross > 0.07% friction NEVER achieved on bar-level retail BTC perp).

### Why M001-M003 Vacuous
- Sum-of-z-score threshold 5.5 means proxy_score ≥ 5.5 (each component ~1.83σ avg)
- BTC funding/velocity/volume z-scores correlated negatively under cascade conditions
- Joint extreme (all three above 1.83σ simultaneously) inherently rare → ~0.4/day
- Locked Option α precommit prevents threshold loosening (anti-fishing)

**Mandate § 0.7 compliance**: report VACUOUS as FAIL, no retry with looser threshold.

---

## Sealed Boundary Compliance

`assert_no_sealed_data` 호출 모두 PASS (sealed boundary 위반 없음).
Free 1h: 12,969 rows (540.3d), Free 1m: 267,001 rows (185.4d), Free funding: 1,621 rows.
T_seal_start: 2025-11-02T14:00:00 UTC (immutable).

---

## Anti-Fishing Compliance Audit

| Rule | Compliance |
|------|-----------|
| Single-config Option α (no sweep) | ✅ Locked configs only |
| baseline_pnl mandatory P2+ | ✅ Buy-and-hold same window 사용 |
| Sealed boundary assert | ✅ |
| Stress friction obligatory | ✅ Both realistic + stress reported |
| Vacuous → FAIL counted | ✅ M001-M003 VACUOUS_FAIL |
| No post-hoc threshold tuning | ✅ |
| Honest closure (no rationalization) | ✅ FAIL is FAIL |

---

## P3 Entry Decision

P2 produced **0 candidates** (no PASS or PARTIAL). Mandate § P3 진입 가능:
- P3 = MAP-Elites on full Phase A 12 active mechanisms × 9 regime cells
- P2-tested 4 (M001-M004) are subset
- 8 untested (M005 momentum, M006 XS momentum, M009 Wyckoff spring, M010 distribution, M011 high-vol dip, M012 low-vol carry, M013 ETH outperform, M014 dispersion)
- Plus 2 control_null (M007 channel breakout, M008 compression breakout) for sanity

**Recommendation**: Proceed to P3 entry with **lowered prior expectation** (force-flow proxy already showed friction-floor pattern). MAP-Elites cell-conditional may discover regime-specific niches. If P3 also FAIL → Phase B activation timing decision (60-90d wait or paid plan).

### Phase B Activation Trigger Update
- Original plan: 30-60d forward collection accumulation → activate H1/H6/H7-full
- P2 result strengthens skepticism: OI features may not lift edge above friction
- Monitor forward collector progress; re-evaluate Phase B paid plan after P3 results
- Coinglass paid plan still **NOT recommended** (no Phase A signal of edge)

---

## Lessons Learned

1. **Sum-of-z multiplicative-AND replacement** (proxy v2) still produces sparse events at locked threshold. P3에서 weighted-OR or single-component variants 고려 가치 (P3 mechanism design 시 적용)
2. **Basis fade is anti-edge**: M004 0.4% confidence below B&H demonstrates basis ±2σ is NOT informative direction signal at 4h scale. Spot-perp basis dynamics likely already arb'd at this scale
3. **Friction-floor as binding constraint** confirmed once more: 4 free-window mechanisms × 2 scenarios = 8 evaluations, 0 PASS. Aligns with 27-mechanism-prior

---

## Closure Output

- ✅ `experiments/p2/precommit.md` (lock)
- ✅ `experiments/p2/results_raw.json` (8 eval raw + freq scan)
- ✅ `experiments/p2/result.md` (this — closure summary)
- Status: P2 CLOSED FAIL

**Next**: P3 entry — MAP-Elites design + advisor for P3 interpretation moment.
