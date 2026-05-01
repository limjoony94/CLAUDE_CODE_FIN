# 6-Criteria Gate Thresholds — Per-Priority Lock

**Date**: 2026-05-01 (P0.3 entry)
**Authority**: Pre-committed before validator code lands. **Mutability**: changes require new amendment.
**Mandate basis**: § 0.5 6-criteria + per-priority `target_daily` from § P2 / P3 / P6.

---

## Threshold Table

| Priority | target_daily | min_pos_rate | min_p_beats_baseline | max_dd_floor | min_sharpe (annualized) | Comment |
|----------|-------------|--------------|----------------------|--------------|------------------------|---------|
| **P0_BASELINE** | 0.000% | 0.50 | 0.00 | -∞ | 0.0 | Reference points (buy-and-hold etc.). Validator end-to-end smoke test |
| **P2** | 0.10% | 0.50 | 0.55 | -3% | 1.5 | Force-flow reversal (mandate § P2) |
| **P3_CELL** | 0.10% | 0.50 | 0.55 | -5% | 1.5 | MAP-Elites cell-conditional |
| **P3_AGGREGATED** | 0.10% | 0.50 | 0.55 | -5% | 2.0 | Dynamic portfolio across cells |
| **P6_PORTFOLIO** | 0.073% | 0.50 | 0.55 | -5% | 2.0 | annualized 20%/yr ≈ 0.073%/day |

Note: `target_daily` 표기는 fraction (예: 0.001 = 0.1%/day). MaxDD 표기는 negative fraction (예: -0.03 = -3%).

---

## Threshold Rationale

### P0_BASELINE (loose reference)
Validator end-to-end test 용도. Buy-and-hold에 PASS, random entry에 FAIL이 기대값. Strategy 평가 아님.

### P2 (force-flow reversal)
Mandate § P2가 명시적으로 사용:
- target = 0.10%/day → annualized ~44%/yr (0.001 × 365)
- MaxDD ≥ -3% (mandate § P2 리스크 제약)
- Sharpe ≥ 1.5 (annualized) — mandate § 0.5

**Prior probability of PASS**: <30% per friction-floor empirical evidence (27 mechanisms × 5 substrates 0 deployable). Closure rule: "FAIL is expected outcome under friction-floor prior; PASS requires stress validation + paper trade."

### P3 (MAP-Elites)
Cell-conditional 동일 P2 + MaxDD floor 완화 (-3% → -5%, multi-mechanism diversification 인정).
Aggregated portfolio level은 Sharpe 더 엄격 (2.0).

### P6 (LIVE-readiness portfolio)
Mandate § P6: "annualized ≥ +20%/yr, MaxDD ≥ -5%, Sharpe ≥ 2.0"
- 20%/yr / 365 ≈ 0.0548%/day. 그러나 친화 보수적 0.073%/day로 약간 상향 (mandate ≥20%/yr 만족 + safety margin).
- 정확히는 `target_daily = (1.20)^(1/365) - 1 = 0.0501%` (compound) 또는 0.0548% (additive 20%/365). 0.073% 사용은 보수.

---

## Validator Behavior

```python
from validators.bootstrap_six_criteria import bootstrap_six_criteria

result = bootstrap_six_criteria(
    daily_pnl=strategy_daily,
    baseline_pnl=buy_and_hold_daily,
    priority="P2",
    B=10000,
    block_size=3,
    seed=42,
)
assert result["all_pass"] is bool
```

각 criterion:
- `mean_pass = point_mean >= target_daily`
- `p5_pass = point_p5 >= 0` (5-percentile of daily returns ≥ 0)
- `pos_rate_pass = (returns > 0).mean() >= min_pos_rate`
- `p_beats_pass = bootstrap_pbeats >= min_p_beats_baseline` (block bootstrap diff-of-means)
- `max_dd_pass = point_max_dd >= max_dd_floor` (additive cumsum drawdown)
- `sharpe_pass = annualized_sharpe >= min_sharpe`

`all_pass = all 6 PASS`. Stress friction은 별도 evaluation (precommit_amendment_001).

---

## Anti-Fishing Locks

1. ❌ Threshold lower 변경 금지 (post-hoc relaxation)
2. ❌ "거의 통과" rationalization 금지 (5/6 pass = FAIL)
3. ❌ 추가 priority 정의 변경 시 새 amendment 의무
4. ✅ Threshold tighter 변경은 가능 (anti-fishing 강화 방향)

---

## Closure Reporting Convention

각 priority result file (`experiments/p{N}/result.md`)에 의무 포함:
```markdown
## 6-Criteria Result (P{N})

| Criterion | Threshold | Value | PASS/FAIL |
|-----------|-----------|-------|-----------|
| mean | ≥ X% | Y% | ✅/❌ |
| p5 | ≥ 0 | Y | ✅/❌ |
| pos_rate | ≥ 0.5 | Y | ✅/❌ |
| p_beats | ≥ 0.55 | Y | ✅/❌ |
| max_dd | ≥ -X% | Y% | ✅/❌ |
| sharpe | ≥ X | Y | ✅/❌ |

**all_pass**: True/False
**friction stress (0.20% RT)**: separate evaluation included
```

---

**Pre-commit signed**: Claude Code agent, 2026-05-01.
