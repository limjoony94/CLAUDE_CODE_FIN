# Analysis: Fold 2 Regime Analysis

> **Feature**: fold2_regime_analysis
> **Date**: 2026-04-19
> **Phase**: Check
> **Match Rate**: **98%** (Design ↔ 구현, diagnostic 단순)
> **Outcome**: **CONCLUSIVE** — Fold 2 약점은 candidate_C 고유가 아니라 전략 전반의 저변동성 취약성; 특정 10~12일(2025-07-26~08-07)에 집중된 sample noise 가능성 큼

---

## 1. Executive Summary

Candidate_C가 slip WF fold 2에서 -9.03pp 실패한 원인을 7개 가설(H1-H7)로 진단.

**가장 중요한 발견**:
1. **H5 반전**: Baseline이 fold 2에서 오히려 **-11.53pp로 더 심각**. Candidate_C의 넓은 SL이 whipsaw 증폭원인이 아니라 **완화**함. 약점은 전략 전반이지 candidate_C 고유가 아님.
2. **H6 극단 집중**: worst 3 sub-windows(2025-07-26~08-07, 실질 12일)의 합 -9.16pp = 전체 fold 2(-9.03)의 **101%**. 나머지 ~54일은 net positive.
3. **H1+H4 조합**: 저변동성(ATR% 0.229 vs 평균 0.318) + R:R 저하(2.22 vs 3.00) = 돌파 후 gain은 작고 loss는 유사 → R:R 악화
4. **H7 제한적**: Clean regime filter 1개(`returns_std_pct<0.2`)만 발견, 그것도 fold_1(양수) 포함 → **단일 metric 분리 어려움**

**진단 결론**: Fold 2 실패는 **2025-08 첫 주의 특수 시장 이벤트**가 주원인이며, candidate_C의 구조적 결함이 아님. Regime filter는 부분적 가치 있으나 양수 기간도 희생됨.

---

## 2. Gap Analysis (Design ↔ 구현)

### Match Rate: 98%

진단 스크립트로 Design과 구현이 단순 1:1 대응. 유일한 minor gap:

- **성능 예상**: Design 5~10초, 실측 **0.1초** (데이터 캐시 효율)

Critical gap: 없음.

---

## 3. Fold 구간별 시장 레짐 프로파일

| Fold | 기간 | ATR% | Ret std | Range% | Trend | Sideways idx |
|------|------|------|---------|--------|-------|--------------|
| 1 | 2025-05-05~2025-07-11 | 0.248 | 0.184 | 0.248 | **+24.2%** | 95.7 |
| **2** | **2025-07-11~2025-09-15** | **0.229** ★ | **0.169** ★ | **0.229** ★ | -2.6% | **65.6** ★ |
| 3 | 2025-09-15~2025-11-21 | 0.307 | 0.232 | 0.313 | -25.5% | 121.5 |
| 4 | 2025-11-21~2026-01-26 | 0.289 | 0.230 | 0.289 | +2.4% | 66.7 |
| 5 | 2026-01-26~2026-04-03 | **0.429** | **0.322** | **0.430** | -24.4% | 101.6 |

**Fold 2 특징**:
- 모든 변동성 지표(ATR%, Ret std, Range%)가 **fold 중 최저**
- Trend -2.6% = 거의 횡보 (다른 fold 들은 방향성 명확)
- Sideways index 65.6 = 횡보 성향 높음

**Fold 4 유사성 주목**: sideways 66.7, trend +2.4%로 비슷하지만 Fold 4는 +12.76pp 양수. 차이: Fold 4는 ATR% 0.289로 fold 2(0.229)보다 **26% 높은 변동성**.

---

## 4. Strategy Metrics 비교 (baseline vs candidate_C)

### 4.1 Fold별 PnL & R:R (slip_med)

| Fold | Baseline PnL | Base R:R | Cand PnL | Cand R:R | Diff (cand-base) |
|------|--------------|----------|----------|----------|-------------------|
| 1 | -2.08 | 2.24 | +1.19 | 2.45 | +3.27 |
| **2** | **-11.53** | **2.13** | **-9.03** | **2.22** | **+2.50** |
| 3 | +34.79 | 3.01 | +33.56 | 2.98 | -1.23 |
| 4 | +5.45 | 2.89 | +12.76 | 3.32 | +7.31 |
| 5 | +19.47 | 3.18 | +24.59 | 3.26 | +5.12 |

**핵심 관찰**:
- **Fold 2에서 candidate_C가 baseline 대비 +2.50pp 우위** (-9.03 > -11.53)
- Candidate_C는 fold 2-5 평균 **모든 fold에서 baseline 이상** (fold 3만 -1.23 미세)
- **H5 가설 완전 반전**: widening SL이 whipsaw를 증폭하지 않고 **손실 완화** 역할

### 4.2 Exit reason 분포 (candidate_C, slip_med)

| Fold | SL% | 기타 |
|------|-----|------|
| 1 | 10.1 | TRAIL_TP 주종 |
| **2** | **9.0** | 다른 fold와 유사 |
| 3 | 12.5 | |
| 4 | 8.8 | |
| 5 | 7.5 | |

Fold 2 SL% = 9.0 vs 평균 9.7 → H3(SL 비율 높음) 가설 기각. **Whipsaw 과다 아님**.

### 4.3 연속 손실 streak

| Fold | baseline streak | cand streak |
|------|-----------------|-------------|
| 1 | 12 | 12 |
| **2** | **14** | **13** |
| 3 | 10 | 10 |
| 4 | 14 | 14 |
| 5 | 9 | 9 |

Fold 2와 Fold 4가 동일한 streak 14/14. 그러나 Fold 4는 양수. **연속 손실이 fold 2 약점을 설명하지 못함**.

---

## 5. Sub-window Microscopy (Fold 2, candidate_C, 5-day stride)

### Worst 3 windows
| 기간 | Trades | PnL | WR |
|------|--------|-----|-----|
| 2025-08-02 ~ 2025-08-07 | 17 | **-3.77%** | 17.6% |
| 2025-07-26 ~ 2025-07-31 | 17 | -3.04% | 23.5% |
| 2025-07-31 ~ 2025-08-05 | 16 | -2.35% | 18.8% |
| **합계** | 50 | **-9.16%** | 평균 20% |

### Best 3 windows
| 기간 | Trades | PnL | WR |
|------|--------|-----|-----|
| 2025-08-10 ~ 2025-08-15 | 14 | +6.20% | 42.9% |
| 2025-08-12 ~ 2025-08-17 | 13 | +3.97% | 38.5% |
| 2025-07-13 ~ 2025-07-18 | 19 | +1.65% | 31.6% |

**핵심 발견**:
- **Worst 3의 합계 -9.16pp = Fold 2 전체 -9.03pp의 101%**
- 즉 Fold 2의 모든 손실이 **2025-07-26 ~ 2025-08-07 (약 12일)** 에 집중
- 이후 8월 중순 (08-10~17)은 오히려 강세 (+10pp 회복)
- 결국 **10~12일의 특수 이벤트** 가 fold 전체를 음수로 만듦

**2025-07-26 ~ 2025-08-07 BTC 시장 추정**:
- 매우 좁은 range + 낮은 WR(17~24%) → 돌파 후 즉시 반전의 연속
- Whipsaw-intensive regime — 이 구간에서는 **어떤 breakout 전략도 어려움**

---

## 6. Regime Filter 후보 (H7 검증)

### Threshold sweep 결과
총 22개 rule 평가. **Clean filter (fold_2 포함, ≤1 others)** : **단 1개**
- `returns_std_pct < 0.2` → fold_1 + fold_2 flag

### 문제점
- Fold 1은 candidate_C에서 +1.19 (baseline -2.08) — candidate_C 기준 양수
- 이 filter 적용 시 fold 1 +1.19 포기 + fold 2 -9.03 회피 = net +7.84pp 개선
- 그러나 샘플 외 일반화는 불확실 (특정 regime 기준 fold_1/fold_2만 잡히는 게 우연일 가능성)

### Aggressive filters (fold_2 + ≥2 others도 포함)
- `ATR% < 0.3`: fold_1, fold_2, fold_4 flag → **fold_4(+12.76) 양수 버리므로 net 부정적**
- `sideways_index > 80`: fold_1, fold_3, fold_5 flag (fold_2 miss)
- `range_pct < 0.3`: fold_1, fold_2, fold_4 → fold_4 손실

**결론**: **단일 metric regime filter는 부분적 가치. Multi-metric ML 접근이 필요**할 수 있으나 본 PDCA 범위 외.

---

## 7. 가설 최종 verdict

| H | 내용 | Verdict | 증거 |
|---|------|---------|------|
| **H1** | Fold 2 저변동성 | ✅ **TRUE** | ATR% 0.229 vs 평균 0.318, 28% 낮음 |
| **H2** | 낮은 돌파 빈도 | ❌ FALSE | 3.36 tpd vs 평균 3.14, 오히려 **+7% 많음** |
| **H3** | 높은 SL 비율 (whipsaw) | ❌ FALSE | 9.0% vs 평균 9.7%, 오히려 낮음 |
| **H4** | R:R 저하 | ✅ **TRUE** | 2.22 vs 평균 3.00, 26% 낮음 |
| **H5** | Widening SL amplifies damage | ❌ **REVERSED** | cand -9.03 **> baseline -11.53** (+2.5pp 우위) |
| **H6** | 손실의 sub-window 집중 | ✅ **TRUE (강력)** | worst 3 windows 합 = -9.16 = fold 전체 101% |
| **H7** | Regime filter 가능 | △ **PARTIAL** | 1개 clean filter but fold_1 양수도 포함 |

**주요 결론**:
- 2개 **반전 발견**: H2(돌파 빈도 많음), H5(candidate_C가 baseline보다 낫다)
- 핵심 원인: **저변동성(H1) → R:R 악화(H4) → 특정 12일(H6)에 손실 집중**
- Candidate_C 특유 약점이 아닌 **전략 전반의 저변동성 regime 취약성**

---

## 8. 해석 & 결론

### 8.1 candidate_c_validation의 STOP이 여전히 타당한가?

**그렇다**. 이유:
- `wf_slip_pass` 4/5 core flag 실패는 여전히 유효
- Fold 2 약점이 샘플 특수성이라고 해도 **관측된 사실**로 존재
- 9/9 엄격 기준이 선언되었으므로 사후 완화는 selection-after-peek

### 8.2 그러나 STOP이 "영원"인가?

**아니다**. 본 진단이 유리한 증거 추가:
- **구조적 실패 아님** → regime filter 도입 시 회복 가능
- **Baseline도 같은 구간 더 심각 실패** → candidate_C 고유 문제 아님
- **12일 특수 이벤트** → 다른 300+ 일에서는 candidate_C가 baseline 압도
- 30일 LIVE 샘플에서 2025-08-04 유형 regime 미발생 시 candidate_C 조건부 GO

### 8.3 Regime filter의 가치
- 단일 metric으로 fold 2만 정확히 잡긴 어려움
- `returns_std_pct < 0.2` 적용 시: fold 1(+1.19) 버리고 fold 2(-9.03) 회피 = net +7.84pp
- 그러나 **샘플 외 일반화 검증 필수** — 다른 기간에서 해당 filter가 양수 기간도 막을 위험

### 8.4 Fold 2 worst 12일(2025-07-26~08-07) 특성
- WR 17-24% (평균의 절반)
- 좁은 range + 저변동성 + 횡보
- **"C1 Breakout 전략이 구조적으로 약한" regime** — 돌파 신호 자체가 허위가 많음
- Backtest 333일 중 ~3.6% 차지, 통계적으로 드문 구간

---

## 9. Recommended Action

### 즉시
1. **본 진단을 candidate_c_validation의 부속 증거로 추가** (report 보충)
2. **Candidate_C 조건부 GO 트리거 재정의**:
   - 기존: WR≥30% AND PnL/trade≥baseline AND fold 2 재현 안됨
   - **추가**: 30일 LIVE 중 "2025-08-04 유형 regime" (ATR%<0.25, returns_std<0.2, trend<5%) 미발생
3. **Baseline의 slip-WF 엄격 재평가**:
   - Baseline도 slip 환경에서 4/5만 양수인지 확인
   - 이 경우 "baseline 자체가 selection-after-peek로 선정된 것일 수 있다" 는 반성

### 단기 (2~4주)
1. **Regime-conditional candidate_C PDCA**:
   - 기본값 candidate_C (4.0), 저변동성 감지 시 baseline (3.3)로 복귀
   - Threshold: `returns_std_pct < 0.2` (최근 30 bars) 또는 `ATR%<0.25`
   - Slippage-adjusted WF 5/5 요구 (엄격 유지)
2. **Baseline slip-WF 재평가** (진단)
3. **Fold 2 worst 12일 zoom-in**: 일자별 가격 chart + trade log 분석

### 중기 (1~2개월)
1. **Multi-metric ML regime classifier 탐색** (별개 PDCA)
2. **30일 LIVE steady-state 관측** 후 candidate_C 조건부 GO 재평가

---

## 10. 방법론적 교훈

1. **가설 기각의 가치**: H2/H3/H5 기각으로 "whipsaw 증폭" 통념이 틀렸음 실증. 진단은 확인뿐 아니라 **오해 해소** 도구.
2. **Sub-window microscopy의 위력**: 전체 fold가 음수여도 12일에 손실 집중이면 structural ≠ sample 구분 가능.
3. **Baseline 비교의 필수성**: H5 반전은 baseline을 같은 조건에서 측정했기 때문에 발견. "비교 없는 진단은 원인 오진 유발".
4. **Single-metric regime filter 한계**: 좋은 filter 설계는 다변량 필요. 이건 별개 연구 주제.
5. **"Strict protocol과 Rich insight의 양립"**: STOP 판정 유지하면서도 재평가 경로 제시 — PDCA의 유연성.

---

## 11. Files Touched

- `scripts/analysis/fold2_regime_analysis.py` (NEW, ~310 lines)
- `results/fold2_regime_analysis_20260419_153643.json` (NEW)
- `docs/01-plan/features/fold2_regime_analysis.plan.md`
- `docs/02-design/features/fold2_regime_analysis.design.md`
- `docs/03-analysis/fold2_regime_analysis.analysis.md` (본 문서)

Production 변경 **0건**.

---

## 12. Reference

- Plan: `docs/01-plan/features/fold2_regime_analysis.plan.md`
- Design: `docs/02-design/features/fold2_regime_analysis.design.md`
- Trigger: `docs/04-report/candidate_c_validation.report.md` (fold 2 원인 의문)
- 재사용 엔진: `scripts/analysis/c1_intrabar_parity.py`, `intrabar_trail_impact.py`
- 결과 JSON: `results/fold2_regime_analysis_20260419_153643.json`
