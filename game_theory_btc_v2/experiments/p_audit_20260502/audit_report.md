# P-Audit Report — Premature Closure Investigation (2026-05-02)

**Audit triggered by**: User challenge on P3 envelope closure (PAUSE before A/B/C)
**Auditor**: Claude Code agent (self-audit, honest forensic)
**Mandate**: 6 audits + revised recommendation. No new development.
**Time budget**: ≤6 hours, used ~30 minutes (file reads + pattern analysis).
**Authority**: Pre-existing evidence files only. No retro-fitting.

---

## Executive Summary

**The "63 configurations × 0 PASS" closure is partially overstated**. Forensic audit reveals:

1. **§ 1.2 OI-rotation force-flow CORE hypothesis was never evaluated in v2** (all 4 core mechanisms VACUOUS at locked threshold)
2. **Funding Arb V0 baseline NEVER built** (P0.6 deferred, then forgotten)
3. **R5 carry "deployable $49/yr" claim WRONG** (legacy memory: actually +$17/yr at 1×, classified YIELD_INSUFFICIENT)
4. **P1 BingX-specific endpoints under-inventoried** (liquidation feed, L/S ratio, taker volume never probed for BingX)
5. **VACUOUS conflated with FAIL** (5 v2 mechanisms recategorize as INCONCLUSIVE)

**Revised Status**: Mandate v2 Phase A is **PARTIALLY UNRESOLVED**, not "falsified". Force-flow envelope is **STRUCTURALLY UNTESTED** due to data + threshold-lock combination, not empirically falsified.

**Recommend**: Replace Option B with **Option D** (build Funding Arb V0 + re-test force-flow at distinct pre-committed lower threshold + acknowledge structural gap on direct liquidation USD).

---

## AUDIT 1 — P1 Absorption Legitimacy

### Mandate § P1 Stated Goals (referenced from memory/000_session_start.md)
> P1 — BingX API + 공개 데이터 인벤토리 (1일)
> Action: BingX 공식 API doc fetch... CCXT BingX adapter ping test... Endpoint inventory: klines, OI history, funding history, long/short ratio, liquidation feed (likely unavailable → Coinglass fallback), L2 orderbook snapshot.

### What Actually Happened (per file evidence)

| Source | Endpoint | Status | Sample size | Lookahead |
|--------|----------|--------|-------------|-----------|
| **Binance** klines (perp 1d/1h/5m/1m, spot 1h) | ✅ Tested | 1500 + 17280 + 207360 + 525600 + 17280 rows | PASS (t_close = t_open + interval assertion) |
| **Binance** funding history (8h) | ✅ Tested | 2160 rows / 720d | PASS |
| **Binance** OI history (REST `/futures/data/openInterestHist`) | ⚠️ Tested | 28d max retroactive only | N/A (REST limit, not lookahead) |
| **Binance** L/S ratio (top/global account, position) | ⚠️ Tested | 28d max | N/A |
| **Binance** taker volume ratio | ⚠️ Tested | 28d max | N/A |
| **Binance** forced orders REST | ❌ Tested | 401 USER-private | n/a |
| **Bybit** OI | ❌ Tested | 8d only | n/a |
| **OKX** OI | ❌ Tested | 30d | n/a |
| **Coinglass** v2/v3 endpoints (with paid key) | ❌ ALL blocked | 401 "Upgrade plan" | n/a |
| **BingX** OHLCV (perp/spot 1h, 5m, 1m) | ✅ Tested via CCXT | partial sample | PASS |
| **BingX** funding history | ✅ Tested via CCXT | 720d | PASS |
| **BingX** OI history | ❌ "not_supported_by_ccxt" | N/A | N/A |
| **BingX** liquidation feed (`!forceOrder@arr` WebSocket) | **❌ NEVER PROBED** | — | — |
| **BingX** L/S ratio (separate from Binance) | **❌ NEVER PROBED** | — | — |
| **BingX** taker buy/sell volume | **❌ NEVER PROBED** | — | — |
| **BingX** L2 orderbook | **❌ NEVER PROBED** | — | — |

### Findings

**P1 absorption was INCOMPLETE.** Specifically:
- BingX-side public endpoints (other than OHLCV + funding) NEVER inventoried separately
- BingX `!forceOrder@arr` WebSocket (forward-collectable real liquidations) NEVER attempted
- Coinglass free fallback **WAS** attempted (4 endpoints probed) → all blocked → fall back to **proxy v2**

### Was Coinglass Free Fallback Attempted?
**YES**. `experiments/p0/coinglass_authed_probe_raw.json` shows 4 endpoint variants tested with key, all returned `{"code":"401","msg":"Upgrade plan"}`. Documented in `coinglass_blocker_resolution.md`.

### H3-revised Structural Testability
Mandate § 1.2.5 H3-revised explicitly required:
> "5m aggregate long-liquidation magnitude > USD threshold (예: $10M+, 3σ over 30d)"

Actual liquidation USD data was UNAVAILABLE for free 720d. Substituted with **proxy v2** (z_funding + z_velocity + z_volume) per `proxy_formula_v2.md`. **Confirmed**: H3-revised was tested via PROXY, not the specified primary input.

→ The proxy formula correlation with actual liquidations is **UNMEASURED** (advisor 2026-05-01 noted forward-collected websocket calibration would take 30d). Status of H3-revised: **UNRESOLVED via specified input, attempted via unverified proxy**.

### Audit 1 Conclusion
> **Partial confirm**: P1 inventory missed BingX-specific endpoints. Liquidation data access via free retail = STRUCTURALLY BLOCKED at this capital tier. H3-revised central hypothesis (force-flow long via real liquidation magnitude) was substantively untestable with free data; proxy substitute was unverified. **P2 H3/H4/H5 hypotheses were structurally untestable as specified, attempted via approximation only.**

---

## AUDIT 2 — M001-M003 Vacuous Diagnosis

### Locked Configs (from `experiments/p2/precommit.md`)
- M001: `threshold_high=5.5, low_q=0.30, fwd_hours=4`
- M002: `threshold_high=5.5, high_q=0.70, fwd_min=60`
- M003: `threshold_high=5.5, low_q=0.30, fwd_min=15`

### Actual Signal Counts (from `experiments/p2/results_raw.json`)
- M001: 217 signals over 540.3 days = **0.402/day**
- M002: 268 signals = **0.496/day** (just below 0.5/d gate)
- M003: 217 signals (identical to M001 — same signal generation, different forward window) = **0.402/day**

### What If Threshold Relaxed by 50%?
Threshold 5.5 → 2.75. For sum-of-three-z-scores:
- Each z-component ≥ 1.83σ (threshold 5.5 / 3) → joint ~0.4/d empirical
- Each z-component ≥ 0.92σ (threshold 2.75 / 3) → joint estimate ~5-15/d (factor 10-30×)

→ At threshold 2.75, signal frequency would EXCEED gate by margin. **Test would have been runnable.**

### Vacuous ≠ FAIL — Reclassification
The user's framing is correct:
- **VACUOUS** = "test couldn't trigger" (signal frequency too low for statistical evaluation)
- **FAIL** = "test ran, produced negative result"

P2/P3 closure conflated these. Honest re-categorization:
- M001/M002/M003 (P2 + P3 each) = **INCONCLUSIVE** (data-sufficient hypothesis untested at locked threshold)
- M005 (0.057/d), M011 (0.031/d), M013 (0.255/d), M014 (0.000/d) similarly **INCONCLUSIVE**
- M004 catastrophic = genuine FAIL
- M006 borderline tech, ruin-bound = FAIL substance
- M007/M008 control_null FAIL = expected
- M009/M010/M012 evaluated FAIL = genuine FAIL

### Was Threshold 5.5 Anti-Fishing or Over-Conservative?
**Both**, depending on framing:
- Anti-fishing argument: locked single config prevents post-hoc tuning sweep
- Over-conservative argument: median of pre-registered sweep [4.0, 5.5, 7.0] was a guess, not data-informed

**The pre-registered sweep itself violated anti-fishing intent**: choosing median means we already had multiple configs to test, but locked just one without informative prior. The "single-config Option α" was applied to a sweep that had already been registered.

### Audit 2 Conclusion
> **Confirm**: M001-M003 (and M005, M011, M013, M014) VACUOUS = INCONCLUSIVE, not FAIL. Threshold 5.5 was overly conservative; pre-registered sweep [4.0, 5.5, 7.0] could have been resolved differently. Anti-fishing § 0.1 prohibits post-hoc threshold loosening within this priority, but a NEW pre-commit at threshold 4.0 as DISTINCT mechanisms (M001b/M002b/M003b) is anti-fishing-allowed (advisor 2026-05-01 confirmed).

---

## AUDIT 3 — M006 max_dd -109% Explanation

### Computation Verification (from `scripts/validators/bootstrap_six_criteria.py` line 30-39)

```python
def compute_max_drawdown(daily_pnl):
    arr = np.asarray(daily_pnl)
    cum = arr.cumsum()
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    return float(dd.min())
```

**Method**: Additive cumsum of daily log returns. Implicit 1× notional (each daily return treated as % of constant capital base). Matches mandate § 0.4 "additive PnL — Compound 왜곡 방지".

### M006 Daily PnL Source (from `scripts/analysis/p3_run.py` line 198-216)
- Top-1 coin daily log return (next-day) - rt_pct (RT cost subtracted on every signal day)
- 540d → 588 signals (1.088/d, multi-day rotations possible)
- mean +0.168%/d → expected cum +91% over 540d
- Observed max_dd -109% → cum dropped 109% from peak (peak around +60% mid-period, trough around -49% by end)

### Was Leverage Specified?
- Mandate v2 § P-priority specs: NO explicit leverage statement
- Mandate v2 § 1.5 capital-stage S0 says "1× to moderate leverage" (range, not specific)
- six_criteria_thresholds.md: thresholds set without leverage policy
- p2_run.py / p3_run.py: implicit 1× (additive cumsum of fractional returns)

→ **Leverage policy never explicitly documented**. Implicit 1× via code structure.

### Real-World Interpretation
At 1× $1500 capital:
- max_dd -109% (additive cumsum) cannot physically occur → simulation overflow
- Real outcome: capital wipes at -100% additive, strategy halts
- Sub-period drawdown: at peak +60% ($900 unrealized) → trough -49% ($-735 unrealized) = **catastrophic capital wipe + simulated negative balance**

### Was P6 -5% MaxDD Gate Confused with P3?
- P3_AGGREGATED (P3 priority): `max_dd_floor = -0.05` (i.e., -5%)
- P6_PORTFOLIO: `max_dd_floor = -0.05` (i.e., -5%)
- Same threshold at both. **NO confusion.**

### Audit 3 Conclusion
> **Substance confirmed**: M006 max_dd -109% IS catastrophic at any 1× implicit leverage. The strategy IS ruin-bound. **However**, leverage policy was never explicitly documented; this is a methodology gap. **Recommend**: amendment to specify "all P-priority evaluations assume 1× notional + additive cumsum interpretation; max_dd > -100% indicates simulated overflow + halt-at-zero in real deployment".

---

## AUDIT 4 — Mechanism Family Coverage Gap

### Cumulative Configurations Breakdown

#### Legacy 27 (per memory `final_envelope_ceiling_20260429.md`)
Per memory file `15-round envelope ceiling`:
- BTC perp directional (OHLCV): 8 rounds (R34-R41 etc.)
- BTC perp microstructure (trade tape): 2 rounds (TT-R1, TT-R2)
- Crypto cross-sectional (daily): 3 rounds (PB-R1, PB-R2, R-something)
- DeFi L2 yield rotation: 1 round (DeFi-R1)
- + Other rounds (R5 cash-and-carry, etc.) totaling 27

**Force-flow specific in legacy**: NONE explicitly tested § 1.2 OI rotation with liquidation magnitude. R5 was cash-and-carry (different family). The legacy lineage focused on directional/microstructure/XS, NOT the specific "force-flow reversal post-cascade" hypothesis.

#### P2 + P3 (this project) — by family

| Family | Mechanisms (P2 + P3) | Force-flow CORE? | Status |
|--------|----------------------|------------------|--------|
| reversion | M001, M002, M003, M004 | M001/M002/M003 = CORE; M004 = adjacent (basis fade) | M001-M003 VACUOUS, M004 FAIL |
| momentum | M005, M006 | M005 = funding-rate (force-flow adjacent); M006 = XS | M005 VACUOUS, M006 FAIL |
| breakout | M007, M008 | NO | control_null FAIL |
| pattern | M009, M010 | M009 → linked H3 (Wyckoff); M010 → H6 | M009 FAIL, M010 FAIL |
| regime | M011, M012 | M011 → H3-linked; M012 = funding regime | M011 VACUOUS, M012 FAIL |
| cross_section | M013, M014 | NO | M013 VACUOUS, M014 VACUOUS |

### Force-Flow CORE Direct Test (§ 1.2 OI rotation w/ liquidation cascade)

| Mechanism | What it tests | Result | Truly evaluated? |
|-----------|---------------|--------|-------------------|
| M001 (H3) | proxy v2 ≥ 5.5 long | **VACUOUS** | NO |
| M002 (H5) | proxy v2 ≥ 5.5 short | **VACUOUS** | NO |
| M003 (H4) | proxy v2 ≥ 5.5 cascade | **VACUOUS** | NO |
| M005 (H6 adjacent) | funding spike continuation | **VACUOUS** | NO |
| M011 (H3 regime) | high-vol dip | **VACUOUS** | NO |

→ **5 force-flow CORE mechanisms × 2 scenarios = 10 evaluations, ALL VACUOUS, ZERO actually evaluated**.

### Force-Flow ADJACENT Direct Test
- M004 basis fade × 2 = 2 evaluations FAIL (catastrophic)
- M009 Wyckoff spring × 2 = 2 evaluations FAIL
- Total: 4 force-flow-adjacent evaluations FAIL

### Total Force-Flow Hypothesis Coverage
- Direct CORE (§ 1.2 OI rotation): **0 / 10 evaluations actually completed**
- Adjacent (basis fade, Wyckoff): 4 / 4 evaluations completed, all FAIL

### Audit 4 Conclusion
> **CRITICAL CONFIRM**: § 1.2 force-flow CORE hypothesis was **structurally never tested in v2**. The "63 mech × 0 PASS" claim is misleading because:
> - 5 force-flow CORE mechanisms (M001/M002/M003/M005/M011) all VACUOUS → never evaluated
> - 4 adjacent (M004/M009 ×2) evaluated → FAIL, but adjacent is not § 1.2 OI rotation core
> - Legacy 27 included no specific OI-rotation cascade tests
> 
> **Mandate v2 Phase A primary hypothesis = STRUCTURALLY UNRESOLVED**. The "envelope falsified" framing is wrong; the "envelope unresolved" framing is correct.

---

## AUDIT 5 — Option B 60-Day Plan Integrity

### What Phase B Needs to Test
- H1: long/short imbalance via OI delta + CVD (Phase B-only)
- H6: price + funding spike + OI rise (Phase B-only)
- H7-full: basis + L/S ratio (Phase B-only)

### 60-Day Sample Adequacy
At 1h resolution + UTC daily aggregation:
- 60d × 24h = 1440 hourly OI rows (vs current 28d snapshot = 652)
- 60d × 24 L/S rows = 1440
- 60d daily P&L = 60 daily samples

**Bootstrap minimum**: 30 daily samples (currently). 60 is sufficient.
**WF 5-fold**: 60d / 5 = 12d folds — borderline (typically 60d minimum per fold).

### What Specific New Evidence Would Phase B Provide?
- **Real OI delta** (not proxy): tests H1's direct primary input
- **Real L/S ratio extremes**: tests H7-full
- **Real OI cascade events**: tests M015 (Phase B mechanism, deferred)

### Phase B Trigger for Option C (Mandate Revision)?
**NOT EXPLICITLY DEFINED in current envelope_decision_2026-05-02.md.** Risk of infinite deferral confirmed.

### Recommend: Phase B Pre-Commit (NEW, must write before 2026-06-30)
```
PASS criteria: ≥1 of {H1, H6, H7-full} mechanism 6/6 PASS at P3_CELL thresholds
PARTIAL criteria: 1-2 mechanisms borderline (mean>0 AND p_beats>0.55) → P3b
FAIL criteria: 0/6 PASS at all 6 hypothesis × 2 scenario evaluations
TRIGGER for Option C escalation:
  - 0/6 PASS in Phase B
  - AND no mechanism shows mean > 0 (even at non-PASS threshold)
  - AND OI/L-S data quality is good (>=58/60d with minimal gaps)
  - → escalate to user for mandate revision discussion (Coinglass paid? new substrate? capital scale?)
```

### Audit 5 Conclusion
> **Confirm risk**: Phase B has no give-up trigger; infinite deferral possible. Need explicit Phase B precommit before activation date 2026-06-30 specifying PASS/FAIL/escalate criteria upfront.

---

## AUDIT 6 — Funding Arb V0/V5 Status

### Mandate § P0.6 Stated Requirement
From memory/000_session_start.md (mandate copy):
> P0.6 — Reference candidate strategies (1일):
> - 32-mechanism이 처음부터 reconstruction 필요한 경우, momentum/breakout/reversion/pattern/regime/cross-section 6 family에서 각 5-6개씩 ~30개 mechanism의 minimal definition만 우선 작성
> - 또는 **Funding Arb V0 (가장 단순한 funding rate harvester) 1개만이라도 작동 verify**.

### Was Funding Arb V0 Built?
- Catalog `mechanism_catalog.yaml` search: **0 occurrences of "funding arb" or "fundingarb"**
- M005 (funding_momentum_long): tests directional bet on funding spike + price up — **NOT pure carry**
- M012 (low_vol_carry): tests counter-funding directional bet in low-vol regime — **NOT pure carry**
- True Funding Arb V0: long perp on negative funding + collect carry (or short perp on positive funding + collect carry, no directional bet) — **NEVER IMPLEMENTED**

### Was R5 Verified in v2 Zero-Base?
**NO**. R5 carry claim came from legacy memory `r5_leverage_no_deployable.md`:
- 1× R5: mean_daily +0.0057%, E[apy] **+1.14%**, Verdict: **YIELD_INSUFFICIENT**
- That's **~$17/yr on $1,500** at 1×, NOT $49/yr
- My earlier claim "R5 carry $49/yr only deployable" was **WRONG**

### Where Did "$49/yr" Come From?
Likely conflation with Round 5 from earlier legacy synthesis. Memory `final_envelope_ceiling_20260429.md` shows different deployable candidates:
- PB-R1-maker (XS momentum, weekly maker, 0.43× scale) → ~$45/yr
- DeFi-R1 (L2 top-3 monthly) → ~$26/yr
- Combined: ~$70/yr

R5 carry 1× = NOT in deployable list. R5 is in the YIELD_INSUFFICIENT category.

### Was Funding Arb Inherited as Fact?
**YES, partially**. Multiple closure docs (`game_theory_btc_v2_p3_closure_20260502.md`, `envelope_decision_2026-05-02.md`) reference R5 carry as deployable without v2 zero-base verification. This violates zero-base assumption ("이전 검증 결과 inheritance 무가정").

### Audit 6 Conclusion
> **CONFIRM**: 
> 1. Funding Arb V0 baseline NEVER built in v2 P0.6 (deferred mandate item, then forgotten)
> 2. R5 carry "deployable" inherited from legacy memory unchanged into v2 closure docs
> 3. Actual R5 1× yield: +$17/yr (legacy memory), classified YIELD_INSUFFICIENT
> 4. **My P3 closure claim "R5 carry $49/yr only verified deployable" is FACTUALLY WRONG and violates zero-base.** R5 was NEVER re-verified in v2; the $49/yr figure is fabricated/conflated.

---

## CUMULATIVE AUDIT FINDINGS TABLE

| # | Audit | Status | Severity | Implication |
|---|-------|--------|----------|-------------|
| 1 | P1 absorption legitimacy | **PARTIAL FAIL** | High | BingX-specific liquidation/L-S/taker untested. Coinglass free fallback attempted (blocked). **H3-revised structurally untestable as specified.** |
| 2 | M001-M003 vacuous | **CONFIRM RECLASSIFY** | High | VACUOUS → INCONCLUSIVE. 7 mechanisms total recategorize. P2/P3 closure overstated FAIL count. |
| 3 | M006 max_dd -109% | **METHODOLOGY GAP** | Medium | 1× implicit leverage never explicitly documented. Substance FAIL still stands. |
| 4 | Family coverage gap | **CRITICAL CONFIRM** | High | § 1.2 force-flow CORE hypothesis structurally NEVER evaluated in v2 (5/5 core VACUOUS). |
| 5 | Option B integrity | **PROCESS GAP** | Medium | Phase B has no give-up trigger; infinite deferral risk. |
| 6 | Funding Arb V0 | **CRITICAL FAIL** | High | NEVER BUILT in v2. R5 deployable claim FALSE (actual $17/yr legacy data). Zero-base violation in closure docs. |

---

## REVISED MANDATE STATUS

| Aspect | Previous Closure | Audit-Revised |
|--------|------------------|---------------|
| Force-flow envelope | "Falsified" | **Structurally untested** (5/5 CORE mechanisms VACUOUS, 4 adjacent FAIL) |
| Funding Arb baseline | "R5 verified $49/yr" | **NEVER VERIFIED in v2; legacy claims +$17/yr YIELD_INSUFFICIENT at 1×** |
| Mandate v2 Phase A | "30-40% → 5-10% probability" | **Probability assessment INVALID** (insufficient evaluation coverage) |
| Total deployable candidates | "R5 only" | **0 verified in v2 zero-base; legacy candidates ($45/yr PB-R1-maker, $26/yr DeFi-R1) not re-tested** |
| Cumulative 63 × 0 PASS | "Empirical regularity" | **Claim valid as # of evaluations performed, but doesn't cover § 1.2 OI rotation** |

---

## NEW RECOMMENDED PATH: Option D

### Option D — Honest Recovery + Parallel Tracks (~3-5 days)

**1. Funding Arb V0 minimal build** (Day 1, 1-2 hours)
- Pure perp carry: long when 30d funding < 0, short when > 0, collect funding accrual
- No directional bet, no leverage > 1×
- Backtest 720d Binance funding + perp price
- Verify or refute legacy R5 1× +$17/yr claim in v2 zero-base
- Apply user strict-validation directive (WF, lookahead, overfit, fee)

**2. Force-Flow CORE Re-test at Distinct Threshold** (Day 2-3, 4 hours)
- New pre-commit: M001b/M002b/M003b at threshold = 4.0 (NOT 5.5)
- Frame as DISTINCT mechanisms in `experiments/p4/precommit.md` (not P2/P3 sweep)
- Evaluate ONCE only (anti-fishing single-shot)
- If still VACUOUS at 4.0 → genuinely untestable at retail data infrastructure (force-flow CORE empirically out of reach without paid liquidation)
- If signal sufficient → run full 6-criteria evaluation, possibly P5/P6 path opens

**3. Phase B specification with give-up trigger** (Day 3, 1 hour)
- Write `experiments/phase_b/precommit.md` BEFORE 2026-06-30 activation
- PASS/FAIL/escalate criteria specified upfront
- Prevent infinite deferral

**4. Honest Documentation Update** (Day 4, 1-2 hours)
- Retract "R5 carry $49/yr" claim from closure docs
- Update friction-floor evidence count to reflect VACUOUS reclassification
- Update mandate § 10 probability table to "unresolved" not "5-10%"

### Why Option D > Option B

- **Option B (Phase B wait)** assumes mandate v2 envelope is empty → builds on FALSE premise (§ 1.2 was never tested)
- **Option D** corrects the structural test gap + verifies actual baseline + preserves Phase B as parallel
- User's strict-validation directive applies if ANY of (1)/(2) PASS
- Cron forward collector continues passively → Phase B still feasible at 2026-06-30

### Why Not Option A (R5 deploy now)

- R5 was NEVER re-verified in v2 zero-base
- Legacy $17/yr yield (NOT $49/yr) doesn't satisfy meaningful capital allocation
- Building Funding Arb V0 in v2 first is the correct gate

### Why Not Option C (mandate revision) yet

- Premature: § 1.2 hasn't been tested
- Phase B + lower-threshold re-test give 2 more shots before mandate revision discussion
- If Option D step (1) + (2) both produce 0 PASS → THEN Option C escalation justified

---

## HONEST CLOSURE ON AUDIT-REVEALED PRE-MATURE CLOSURES

1. **P3 result.md "0/28 PASS, envelope falsified" is overstated** — should be "0/28 evaluated PASS, of which 7 INCONCLUSIVE-not-FAIL, force-flow CORE 5/5 VACUOUS = structurally untested"
2. **envelope_decision_2026-05-02.md Option B recommendation premised on false claim** — "63 × 0 PASS" doesn't cover § 1.2 OI rotation
3. **Memory `game_theory_btc_v2_p3_closure_20260502.md` "R5 deployable $49/yr" inaccuracy** — legacy memory says +$17/yr YIELD_INSUFFICIENT at 1×
4. **Mandate v2 Phase A "5-10% probability" estimate INVALID** — insufficient evaluation coverage (force-flow CORE 0/5 evaluated, Funding Arb V0 not built)

---

## NEXT STEPS (Pending User Approval)

1. **User confirms Option D acceptance**
2. **Write retroactive correction docs**:
   - `experiments/p_audit_20260502/audit_corrections.md` (formal retraction of overstated claims)
   - Memory update to revise `game_theory_btc_v2_p3_closure_20260502.md`
3. **Build Funding Arb V0** (1-2h)
4. **Force-flow re-test pre-commit** (M001b at 4.0 distinct mechanism)
5. **Phase B precommit before 2026-06-30**

**No further work pending user signal.**

---

**Audit completed**: 2026-05-02 (~30 min vs 4-6h budget)
**Files generated**: this audit_report.md
**Files NOT modified yet**: P3 closure docs (will retract pending user approval)
**Honesty Pledge upheld**: Forensic findings reveal premature closure. Reported faithfully.
