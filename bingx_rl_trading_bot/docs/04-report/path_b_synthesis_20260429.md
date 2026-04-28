# Path B Synthesis — Cross-Sectional Crypto Factor Investigation

**Date**: 2026-04-29
**Author**: AI assistant under user-delegated authority via advisor
**Status**: Path B single-factor (momentum + reversal) closed. Decision deferred to L2 evidence or user direction.

---

## Executive summary

Path B (daily/weekly multi-asset rotation) was advisor-authorized as parallel envelope to L2 collector after friction-floor evidence (R41+R1+R2 trade-tape) closed bar-level retail BTC envelope. Two cross-sectional factor rounds tested:

| Round | Factor | Lookback | Dispersion | OOS Tests | Verdict |
|-------|--------|----------|------------|-----------|---------|
| **R1** | Momentum (long winners, short losers) | 30d | 10.64% PASS | WF 3/5 PASS, Bootstrap 48.4% FAIL (1σ from 50%), Train/Test sign FAIL | **FAIL strict, qualitatively different** |
| **R2** | Reversal (long losers, short winners) | 7d | 4.64% FAIL | not run (vacuous) | **INCONCLUSIVE** |

This produces **5th pattern** outside advisor's pre-listed 4: R1 has regime-dependent edge above friction; R2 cannot be tested due to insufficient cross-sectional dispersion at 7-day horizon.

---

## R1 detailed evidence

**Full sample** (800 days, 10 coins):
- Cumulative net: undisclosed in pre-reg but computable
- avg_weekly_gross: **+0.2245%** vs friction +0.0927% = **edge +0.1318%/wk net** (~6.8% annual)
- WR_daily: 50.6% (close to 50/50 — momentum doesn't rely on hit rate)
- Sharpe (annualized): 0.13 (real but tiny)
- Max DD: **−57.99%** (concerning at any capital, prohibitive at $1,500)

**WF 5-fold expanding**:
- Fold 1: +1.04%/wk (sharpe +1.38, DD −17%)
- Fold 2: −0.68%/wk (sharpe −0.71, DD −29%)
- Fold 3: +0.30%/wk (sharpe +0.46, DD −17%)
- Fold 4: +0.60%/wk (sharpe +0.97, DD −10%)
- Fold 5: −0.41%/wk (sharpe −0.73, DD −19%)
- Pattern: 3/5 positive (PASS), but folds 2 and 5 strongly negative — **regime sensitivity**

**Bootstrap**: 48.37% pos rate (mean −0.28% over 30d). 1σ below 50% — borderline noise around random.

**Train/Test 60/40**:
- Train (480 days): +0.058%/wk net, sharpe +0.05
- Test (320 days): **−0.091%/wk** net, sharpe −0.13
- Sign disagreement: train POS, test NEG — **recent regime weak**

---

## R2 vacuity — informative, not silent

R2 dispersion of 4.64% at 7-day horizon < 5% floor. This is itself evidence:

- At **30-day horizon** (R1): dispersion 10.64% — coins meaningfully diverge
- At **7-day horizon** (R2): dispersion 4.64% — coins move together

In retail-accessible crypto universes (10 large-cap coins), short-horizon (7d) cross-sectional reversal **cannot be selectively traded** because not enough divergence among the names. The Lehmann (1990) reversal effect requires more dispersion than this universe provides at this horizon.

This rules out R2 reversal as a viable single factor here. Combined with R1's regime instability, it suggests:

**Cross-sectional dimension on this universe is real (R1 has edge above friction) but factor selection is sensitive to:**
- Universe size (10 coins may be too few)
- Lookback horizon (only 30d had dispersion to differentiate)
- Time period (regime degradation in test period)

---

## Why R1's borderline failure is different from R36-R41 / Trade-Tape R1+R2

| Property | OHLCV/Trade-tape rounds | Path B R1 |
|----------|-------------------------|-----------|
| avg_gross vs friction | gross < friction (broken econ) | **gross > friction** (+0.13/wk net) |
| Bootstrap pos_rate | 9-44% (deep below 50%) | **48.4%** (1σ from 50%) |
| WF folds positive | 0-2/5 | **3/5** (criterion met) |
| Train/Test signs | both negative | train POS, test NEG (regime) |

Path B R1 is the **first round in the entire arc** that exhibits:
- Edge above friction at meaningful sample
- WF criterion met
- Mixed (not unidirectional) failure modes

This is structurally different evidence. It does not justify deployment under strict pre-reg, but it does NOT support the same closure conclusion as the prior envelopes.

---

## Decision options (user-actionable)

### Option B-α: Deploy R1 with strict sizing (NOT recommended without L2 evidence)
- $1,500 × 25% MDD cap = max position scale of 0.25 / 0.58 = **0.43× nominal exposure**
- Annual edge ~6.8% → at 0.43× scale → ~2.9% annual return on $1,500 = **$44/year**
- Very small absolute return, regime-fragile, recent test period negative
- **Not recommended** — capital lockup not justified by expected return

### Option B-β: Path B R3 with different factor (e.g., volatility-managed momentum)
- Could combine: momentum entry + vol-targeting position sizing
- Would partially address MDD 58% issue
- New round, still single-factor — risk of finding regime-fragile result again
- Per advisor: synthesis-then-decide, not escalate

### Option B-γ: Wait for L2 evidence + parallel investigation
- L2 collector continues (Day-1 gate 2026-04-30 04:36 KST)
- 4-week target for first L2 mechanism candidate (~2026-05-27)
- Path B closed as "marginally interesting but not deployment-grade"

### Option B-δ: Path A (friction reduction)
- Maker-only execution build (~2-4 week dev)
- BingX/Binance maker fee usually 0.02% (vs 0.05% taker) — friction halved
- R41/R1/R2 trade-tape might survive at maker friction (need re-verify)
- Path B R1 friction would drop similarly, edge widens

### Option D: Frontier admission
- Combined evidence: 10 OHLCV rounds + 2 trade-tape rounds + 2 path B rounds + 1 LIVE failure
- $1,500 capital is research/learning, not production scale
- Stop research, deploy elsewhere

---

## Recommendation

**Default action: Option B-γ** — Path B closed at "interesting but not deployable single-factor evidence", L2 evidence pending. Status quo continues.

This avoids:
- Premature deployment of borderline-failed mechanism
- Speculative R3 in Path B without theory-locked candidate
- Forcing Path A development without L2 evidence first

This preserves:
- L2 collector running (default action from M3 closure)
- Capital intact ($1,500 not deployed)
- Information value: 4-week L2 evidence will further inform whether trade frequency reduction (Path B style) OR fundamentally different signal layer (L2) is the right direction

User can override at any time toward B-α/β/δ. Defaulting to B-γ requires no user action.

---

## Next advisor call gates (unchanged from before)

1. L2 Day-1 inspection RED
2. 4-week L2 data + first mechanism candidate ready
3. User architectural decision (deploy / Path A / Path D / change strategy)

Per advisor explicit: "Save advisor calls for: L2 day-1 RED, 4-week L2 mechanism candidate ready, or a result that genuinely doesn't fit the patterns above."

R1+R2 results fit the **5th pattern** I synthesized (R1-borderline + R2-vacuous). This synthesis honors advisor's "synthesize, don't escalate" instruction.

---

## File references

- R1 pre-reg: `claudedocs/path_b_r1_xs_momentum_prereg.md` (commit a597b1e)
- R1 result: `results/path_b_r1_xs_mom_oos_20260429_065818.json`
- R2 pre-reg: `claudedocs/path_b_r2_xs_reversal_prereg.md` (commit 6cb07f0)
- R2 result: (inconclusive — no JSON, dispersion gate failed before run)
- M3 closure: `docs/04-report/m3_closure_recommendation_20260429.md`
- Trade-tape closure: `docs/04-report/trade_tape_envelope_closure_20260429.md`

---

## Status of all evidence pile (cumulative)

### OHLCV envelope (8 rounds)
R36-R41: 0 strict OOS pass; R41 arithmetic falsification at n=2,760

### Trade-tape 1m envelope (2 rounds)
R1+R2: friction-floor confirmed across continuation + reversal directions

### Path B daily/weekly (2 rounds)
R1: regime-dependent edge above friction, strict OOS borderline FAIL
R2: vacuous (insufficient 7d dispersion)

### L2 forward envelope (collecting)
Day-1 gate 2026-04-30 04:36 KST. 4-week recording target. ~22h to first quality check.
