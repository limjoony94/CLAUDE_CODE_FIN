# 6-Criteria Gate Thresholds — Per-Priority Lock (v2 post-advisor 2026-05-01)

**Date**: 2026-05-01 (P0.4 closure, advisor reflection)
**Authority**: Pre-committed before P0.5/P0.6 yaml registry. **Mutability**: changes require new amendment.
**Mandate basis**: § 0.5 6-criteria + per-priority `target_daily` from § P2 / P3 / P6.
**Revision**: v2 fixes from advisor (p5 interpretation + min_p_beats raised + baseline mandatory)

---

## Threshold Table

| Priority | target_daily | min_pos_rate | min_p_beats_baseline | max_dd_floor | min_sharpe (annualized) | baseline_required | Comment |
|----------|-------------|--------------|----------------------|--------------|------------------------|-------------------|---------|
| **P0_BASELINE** | 0.000% | 0.50 | 0.00 | -100% | 0.0 | No | Reference points; validator end-to-end smoke |
| **P2** | 0.10% | 0.50 | **0.70** | -3% | 1.5 | **Yes** | Force-flow reversal (mandate § P2) |
| **P3_CELL** | 0.10% | 0.50 | **0.70** | -5% | 1.5 | **Yes** | MAP-Elites cell-conditional |
| **P3_AGGREGATED** | 0.10% | 0.50 | **0.70** | -5% | 2.0 | **Yes** | Dynamic portfolio across cells |
| **P6_PORTFOLIO** | 0.073% | 0.50 | **0.70** | -5% | 2.0 | **Yes** | annualized 20%/yr + safety margin |

`target_daily` 표기: fraction (0.001 = 0.1%/day). MaxDD: negative fraction (-0.03 = -3%).

---

## Critical Interpretation Locks

### `p5` 해석 (advisor 2026-05-01)
- **Definition**: bootstrap 5-percentile of MEANS = 95% one-sided lower CI of mean
- **Computation**: `np.percentile(bs_means, 5)` where `bs_means` = mean of each B bootstrap resample
- **Anti-fishing intent**: "95% confident the strategy has positive edge"
- **Not** raw daily 5-percentile of returns (that would be "95% of days non-negative" — physically impossible for volatile asset)
- Rationale: Mandate § 0.5 prefixes 6-criteria with "(bootstrap 3-day random window)" → all stats from bootstrap distribution

### `min_p_beats_baseline` raised 0.55 → 0.70 for P2+
- **Rationale**: Buy-and-hold drift +0.110%/day on the 540d free window > P2 target_daily 0.10%/day
- Without raising threshold: strategy with mean=0.10%/day at p_beats=0.55 may LOSE to passive holding
- Raised to 0.70 → strategy must beat B&H at 70%+ confidence
- **Anti-fishing**: prevents PASS cases that underperform passive baseline

### `baseline_required` for P2+
- P2-P6 evaluation MUST provide `baseline_pnl` (typically buy-and-hold same-window)
- `bootstrap_six_criteria(strategy, baseline_pnl=None, priority="P2")` → ValueError
- Methodology lock: every active strategy result includes "did this beat doing nothing?" answer
- P0_BASELINE allows None (reference baseline 자체)

---

## Threshold Rationale (per priority)

### P0_BASELINE
Reference points only. validator end-to-end test. baseline 자체 → None allowed.
Mandatory FAIL probability ≈ 100% on all 6 (BTC daily volatility makes p5 confidence wide). Used to verify validator distinguishes mean direction.

### P2 (force-flow reversal)
Mandate § P2: target_daily=0.10%/day, MaxDD≥-3%, Sharpe≥1.5.
**Prior PASS probability**: <30% per friction-floor 27 mechanisms 0 deployable evidence. Closure rule: "FAIL is expected; PASS requires stress validation + paper trade".

### P3 (MAP-Elites)
Cell-conditional same as P2 but MaxDD floor relaxed (-3% → -5%) recognizing diversification.
Aggregated portfolio level Sharpe stricter (2.0).

### P6 (LIVE-readiness)
Mandate § P6: ≥+20%/yr, MaxDD≥-5%, Sharpe≥2.0.
target_daily 0.073%/day = 20%/yr / 365 (additive) + safety margin.

---

## Validator Behavior (post-fix)

```python
from validators.bootstrap_six_criteria import bootstrap_six_criteria

# P0_BASELINE: baseline optional
result = bootstrap_six_criteria(daily, baseline_pnl=None, priority="P0_BASELINE", B=10000)

# P2+: baseline mandatory
result = bootstrap_six_criteria(strategy_daily, baseline_pnl=buy_and_hold_daily,
                                  priority="P2", B=10000, block_size=3, seed=42)
# raises ValueError if baseline_pnl=None

assert isinstance(result["all_pass"], bool)
```

각 criterion (post-fix):
- `mean_pass = point_mean >= target_daily`
- `p5_pass = bootstrap_5pct_of_means >= 0`
- `pos_rate_pass = (returns > 0).mean() >= min_pos_rate`
- `p_beats_pass = bootstrap_p_beats_baseline >= 0.70`
- `max_dd_pass = additive_max_dd >= max_dd_floor`
- `sharpe_pass = annualized_sharpe >= min_sharpe`

`all_pass = all 6 PASS`. Stress friction은 별도 evaluation.

---

## Anti-Fishing Locks (consolidated)

1. ❌ Threshold lower 변경 금지 (post-hoc relaxation)
2. ❌ "거의 통과" rationalization 금지 (5/6 pass = FAIL)
3. ❌ Additional priority 정의 변경 시 새 amendment
4. ❌ baseline_pnl을 weak proxy로 대체 금지 (must be honest passive comparator)
5. ✅ Threshold tighter 변경은 가능 (anti-fishing 강화 방향)

---

## Closure Reporting Convention

각 priority result file (`experiments/p{N}/result.md`)에 의무 포함:
```markdown
## 6-Criteria Result (P{N})

| Criterion | Threshold | Value | PASS/FAIL |
|-----------|-----------|-------|-----------|
| mean | ≥ X% | Y% | ✅/❌ |
| p5 (bs 5-pct of means) | ≥ 0 | Y | ✅/❌ |
| pos_rate | ≥ 0.5 | Y | ✅/❌ |
| p_beats vs baseline | ≥ 0.70 | Y | ✅/❌ |
| max_dd | ≥ -X% | Y% | ✅/❌ |
| sharpe (annualized) | ≥ X | Y | ✅/❌ |

**all_pass**: True/False
**baseline used**: <name + window>
**friction stress (0.20% RT)**: separate evaluation
```

---

**Pre-commit signed (v2 revision)**: Claude Code agent, 2026-05-01.
v1 (raw p5 interpretation, min_p_beats=0.55) deprecated. v2 supersedes.
