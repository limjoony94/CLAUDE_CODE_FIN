# M3-R17 — Comprehensive Strategy Potential (사전 등록)

> **Date**: 2026-04-28
> **Authority**: 사용자 명시 3-phase methodology 강화 (2번째 반복 directive)
> **Origin**: R14 grid 너무 narrow + strict pass/fail. R17은 8 families × rich grids + relative ranking.

---

## 1. R14 한계 (재평가)

R14는 6 families × 60-400 configs = 작은 sweep. Pass/fail thresholds (p_both_pos ≥ 5%, corr > 0) 너무 strict. 사용자 의도: "**경향**" 측정 + **highest** potential 식별 (절대값 아닌 상대값).

R17 차이:
- 8 families (β, γ 추가)
- Rich grids per family (~125-960 configs)
- All distribution metrics 저장
- **Relative ranking** by composite score (절대 threshold 아닌 highest)
- TOP-1 → R18 deep optimization

## 2. Strategy Families (8개)

| Family | Mechanism | Grid dims | Configs |
|--------|-----------|-----------|---------|
| α | ETH-lag steady-state | et × bl × ap × N | ~500 |
| ι | α + ETH 24-bar break | et × bl × ap × lb × N | ~960 |
| κ | α + mid-vol regime | et × bl × lb × N | ~320 |
| σ | Counter-trend at break | rsi × lb × N | ~125 |
| υ | Volume × cross-asset | vol_mult × et × N | ~125 |
| ζ | ETH return acceleration | thresh × window × N | ~80 |
| β | BTC-ETH spread mean-rev | z × corr × N | ~100 |
| γ | Funding × cross-asset | fsum × rsi × N | ~80 |
| **Total** | | | **~2,290** |

## 3. Composite Potential Score (사전 등록 공식)

```
potential = max_test_daily × 100    # best-case extractable
          + median_test_daily × 50  # typical config
          + p_both_pos × 0.5        # cross-stable density (%)
          + corr_tt × 30            # parameter signal generalizes
          - std_test_daily × 30     # robustness penalty
```

This is a **relative ranking**, not absolute threshold. Even if all families negative, identifies LEAST-bad with most extractable optimum.

## 4. Methodology

- Train/test split: 60/40
- Friction: 0.04% RT (maker-tier assumption)
- Exit: fixed N timeout (no trail/SL — R9b/R12 finding: trail kills alpha)
- Min sample: train_n ≥ 30 AND test_n ≥ 30
- All configs' raw metrics saved (later trend analysis 용)

## 5. Phase 2 진입 조건

**TOP-1 (highest composite potential)** → R18 deep optimization (refined grid in best region).
**Pre-registered**: TOP-1 무조건 R18 진행. 절대 threshold 안 둠 (사용자 mandate).

단, TOP-1 max_test_daily가 명백히 음수 (< -0.1%/day) 시 사용자에게 surface — 이 경우 even LEAST-bad가 unprofitable.

## 6. Trend Analysis (R17 출력의 일부)

각 TOP family에 대해 추가 분석:
- 각 parameter axis별 marginal effect (mean test_daily by axis value)
- 2D heatmap of param × param vs test_daily (top 2 axes)
- Robustness gradient (sensitivity to small param changes)

이 trend analysis가 R18 refinement region 식별에 사용됨.

## 7. Predictions

| Family | Predicted relative rank | Confidence |
|--------|------------------------|-----------|
| ι | possibly 1-2 (strongest C1 magnitude) | LOW |
| α | top 3 (largest sample) | LOW |
| κ | mid (mid-vol untested at scale) | LOW |
| σ | low (RR < 1) | MED |
| υ | low (single-axis fail at C3) | MED |
| β, γ, ζ | likely bottom | MED |

**Most likely outcome**: ι 또는 α가 TOP-1, max_test_daily marginal positive (+0.01~0.05%/day).

## 8. Anti-fix-impulse commitment

- 본 R17 grid 정의대로 한 번 실행. 결과 후 grid 추가 안 함.
- TOP-1 식별 후 R18 detail optimization 자동 진행.
- R18 결과 기반으로 Phase 3 (실제 거래) 진입 여부 결정.
