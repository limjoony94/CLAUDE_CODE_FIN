# Envelope Decision — Post-P3a Closure

**Date**: 2026-05-02
**Trigger**: P3a 0/28 PASS, M006 P3b SKIP
**Authority**: Pre-committed framework per `experiments/p3/precommit.md` decision tree

---

## Cumulative Evidence

| Source | Mechanism Configs | PASS | Substrate |
|--------|-------------------|------|-----------|
| Legacy 2025-2026-04 | 27 | 0 (R5 carry $49/yr only) | OHLCV/L2/funding/multi-coin/Wyckoff/SMC |
| P2 (this project) | 4 mech × 2 scenario = 8 | 0 | force-flow proxy + basis |
| P3 (this project) | 14 mech × 2 scenario = 28 | 0 | full Phase A + control_null |
| **Total** | **63 evaluations** | **0 6/6 PASS** | 6+ substrates |

This is no longer "prior probability" — it's **empirical regularity** across project lineages.

---

## Mandate § 10 Probability Update

| Scenario | Daily | Initial Prior | Post-P3 Posterior |
|----------|-------|---------------|--------------------|
| A. Funding Arb only | +0.019% | 99% | 99% (unchanged) |
| B. mandate v2 success | +0.10-0.20% | 30-40% | **5-10%** ⬇⬇ |
| C. Aggressive 0.5%/day | +0.50% | 5-10% | 1-3% ⬇ |
| D. Whale anecdotes | +1%/d | <2% | <2% |

Phase A explicitly tested and falsified. Phase B (OI/L-S) untested but friction-floor likely still binding.

---

## Decision Options

### Option A — Deploy R5 Funding Arb Only (Accept Envelope)
- **What**: Deploy R5 single-coin BTC carry strategy (~$49/yr at $1.5K capital, ~3.28% APY)
- **Pros**: Verified deployable, honest closure, $0 ongoing cost, beats most friction-floor results
- **Cons**: Slow whale-tier path (~100 years), no upside from mandate v2
- **Strict validation suite required** (per user directive): walk-forward TF / lookahead / overfit / fee audit on R5 lineage
- **Decision authority**: User (real money, $1.5K capital deploy)

### Option B — Wait Phase B (60-90d) [⭐ ADVISOR RECOMMENDED]
- **What**: Pause P-priority sequence. Forward collector accumulates OI/L-S data. Re-evaluate 2026-06-30
- **Pros**: Tests mandate's specific Phase B hypotheses (H1/H6/H7-full). Novel data not previously available
- **Cons**: 60d wait, friction-floor may still bind, no immediate revenue
- **Active work during wait**: cron health monitoring, MEMORY.md cleanup, infrastructure_lessons documentation
- **Re-evaluation milestones**:
  - 2026-05-15 (mid-month cron health check)
  - 2026-05-31 (~30d accumulation review)
  - **2026-06-30 (60d Phase B activation ceremony)**
  - 2026-07-30 (90d fallback if 60d insufficient sample)

### Option C — Mandate Revision [DEFERRED, USER-SIGNAL ONLY]
- Fundamental rethink (Coinglass paid? new substrate? different capital tier?)
- Requires user explicit signal — premature now

---

## Recommendation: Option B with R5 Decision Deferred

Per advisor 2026-05-02:
> "Don't deploy R5 immediately. Reason: real-money decision needing explicit user signal. Document recommendation, surface, let user decide."

**Default action (advisor lock)**:
1. Pause P4/P5/P6 sequence (Thompson sampling on 0-PASS arms = meaningless)
2. Forward collector continues accumulating
3. Re-evaluation 2026-06-30 with Phase B activation
4. R5 deploy decision: deferred until Phase B closure (avoid premature deploy)

**User signal required for**:
- Override to Option A immediate R5 deploy
- Override to Option C mandate revision
- Approve Option B wait timeline

---

## Inactive State (during wait)

### Allowed
- ✅ Forward collector cron (auto, hourly via Task Scheduler)
- ✅ Weekly cron health check (manual: `logs/forward_collector_cron.log`)
- ✅ MEMORY.md cleanup (one-time)
- ✅ infrastructure_lessons updates (P5/P6 readiness — strict validation suite design)
- ✅ User-initiated /schedule for 2026-06-30 reminder

### Forbidden (anti-fishing locks during wait)
- ❌ P3 retry with looser thresholds
- ❌ M006 P3b cherry-pick
- ❌ Proxy v2 variants (Choice (a) commitment)
- ❌ New mechanism additions without amendment
- ❌ Premature R5 deploy without user signal
- ❌ Coinglass paid plan without Phase B insufficient signal

---

## Phase B Re-Evaluation Plan (2026-06-30)

When forward collector has ~60d accumulation:
1. Audit `oi_forward.parquet` row count + gap rate
2. Activate H1 (long/short imbalance with OI delta) — `precommit_amendment_004.md` first
3. Activate H6 (price + funding spike + OI rise)
4. Activate H7-full (basis + L/S ratio component)
5. Run 6-criteria evaluations on Phase B hypotheses
6. Apply user's strict-validation directive if any PASS:
   - Walk-forward across timeframes
   - Lookahead bias proof (lag 0 vs lag 1 sensitivity)
   - Overfit check (cross-validation on Phase B subset)
   - Fee application audit (BingX live fees vs modeled)
7. Closure: PASS → live deploy plan / FAIL → escalate to user (Option A or C)

---

## Active Monitoring (during wait)

### Cron Health
- Daily check: `logs/forward_collector_cron.log` last entry timestamp recent (within 2h)
- Weekly review: row count growth in forward parquets
- Anomaly: `logs/forward_collector_cron.log` shows errors → escalate

### Code/Data Hygiene
- No commits to `game_theory_btc_v2/scripts/analysis/p2_*.py` or `p3_*.py` (frozen)
- `experiments/p4/` `p5/` `p6/` empty (skip per advisor)
- `experiments/p3/` final state preserved

---

**Decision pending**: User signal on A vs B vs C.
**Default if no signal**: Option B wait until 2026-06-30.

Recorded: 2026-05-02 02:00 UTC.
