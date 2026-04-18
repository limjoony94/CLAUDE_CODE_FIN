# Refined Variants 배포 결정 — 최종: BASELINE 유지

**Date**: 2026-04-18
**Decision**: **Hold C1 v2.6 baseline** (교체 안 함)
**Source**: Refined A/B/C/D 변종 비교 + Advisor 교정 4차례

---

## 연구 요약

### 변종 목록 (332일 15m BTC, additive 1x)

| 전략 | FullPnL | MDD(obs) | MDD(mean)¹ | MDD CI95¹ | WF | 3-way(60/20/20) | 분할점 4종² |
|------|---------|----------|------------|-----------|-----|------------------|--------------|
| **BASELINE (v2.6)** | **+170.5** | **5.38** | 6.09 | [4.25, 8.99] | 5/5 | ALL+ | 4/4 ALL+ |
| REFINED pure (CD=0) | +262.3 | 8.21 | 8.22 | [5.51, 12.43] | 5/5 | ALL+ | — |
| A: CD=2 restored | +212.4 | 10.42 | 9.76 | [6.34, 16.07] | 4/5 | ALL+ | — |
| B: Hard 2×ATR SL | +60.7 | 30.22 | — | — | 4/5 | FAIL | — |
| **C: Channel soft** | **+226.1** | 6.91 | 7.39 | [5.14, 11.01] | 5/5 | ALL+ | **4/4 ALL+** |
| D: Body 50% strict | +233.2 | 6.90 | 8.74 | [5.96, 13.31] | 5/5 | ALL+ | 4/4 ALL+ |

¹ Stationary block bootstrap (mean_block=20, n_boot=1000) — path dependence 보존
² 3-way split at 40/20/40, 50/20/30, 60/20/20, 70/15/15

---

## 결정 과정 (Advisor 교정 반영)

### 오류 1: IID bootstrap
**초기 결과**: BASELINE MDD 5.38이 IID bootstrap CI[5.39, 15.41]의 2.5%ile → "샘플 행운"으로 해석
**교정**: IID는 시계열 의존성(drawdown clustering) 파괴. Stationary bootstrap 적용 시 BASELINE mean은 6.09, obs 5.38은 mean 근처 — **이상치 아님**. 이전 "근본적으로 바꿈" 주장 철회.

### 오류 2: 순환 논리
BASELINE을 "낮은 관측 MDD 5.38"로 선택했으면서 그걸 다시 "운"이라 부르는 건 tautology. 어떤 전략을 MDD 기준으로 고르든 관측값은 자기 분포 하위에 위치.

### 오류 3: "Tie = 교체 근거"
같은 기댓값 하에서도 **교체 비용**(구현 갭 + regret downside)이 존재. Prior는 "기존 유지".

---

## 변종별 평가

| 변종 | 배제 사유 |
|------|-----------|
| A (CD=2 restored) | obs MDD 10.42 > BASELINE CI_HI 8.99 — **통계적으로 유의미하게 worse** |
| B (Hard 2×ATR SL) | 1843 trades + MDD 30% retrigger 루프, WF 4/5 |
| D (Body 50%) | obs 6.90는 행운 (mean 8.74 > BASELINE mean 6.09) |
| REFINED pure (CD=0) | bootstrap mean 8.22 > BASELINE 6.09 — 확인된 열위 |
| **C (Channel soft)** | **유일한 진짜 후보** — CI가 BASELINE과 오버랩, MDD +1.3pp 기댓값 악화 가능성 vs +56pp PnL 개선 |

---

## 최종 배제: C variant

### 정량 근거
- **검정력**: C의 +56pp/332일 = +0.17pp/일 우위. Daily σ ≈ 1-1.5% → 95% 검출에 **190일 (6개월) 필요**
- 2-4주 paper trading은 catastrophic failure만 감지, efficiency 우위는 **noise floor 이하**

### 정성 근거
- **라이브 거래 증거 0**: backtest는 slippage, partial fill (BUG#55), API 이탈 (BUG#48~61) 담지 못함
- BASELINE은 이미 이 시나리오 통과 완료 (live)
- **구현 갭**: `entry_refined_C`는 연구 스크립트(`c1_refined_variants.py`)에만 존재, production `signals.py`에 없음
- 배포 = signals.py 포팅 + BUG#48~61 재검증 + 백테스트 parity 증명 → 추가 2-3 cycle 리스크

### 유효 경로 (참조용)
1. **Hold baseline** ← 선택
2. Bounded capital (80/20 split + kill switch at 상대 -10% 또는 절대 MDD >12%) — signals.py 포팅 선행 필수
3. Extended paper ≥3개월 — regime 변동 리스크

---

## 부산물 (Research 유지)

`scripts/analysis/`:
- `c1_refined_validation.py` — 3개 전략 (baseline/no-fractal/refined) full validation
- `c1_refined_variants.py` — A/B/C/D 비교
- `c1_refined_dmining_check.py` — P1 (4 split points) + P3 (CD sensitivity)
- `c1_refined_bootstrap_mdd.py` — stationary bootstrap MDD CI

`results/`:
- `c1_refined_validation.json`, `c1_refined_variants.json`
- `c1_refined_dmining_check.json`, `c1_refined_bootstrap_mdd.json`

이후 전략 연구에서 재사용 가능.

---

## Lesson Learned

1. **IID bootstrap은 path-dependent 메트릭(MDD)에 부적절**. Stationary / circular block bootstrap 사용.
2. **"Observed = outlier" 논리는 selection bias와 구별해야 함**. 어떤 기준으로 선택한 표본의 관측값은 자기 분포 극단에 위치.
3. **백테스트 우위가 live evidence를 overwhelm 하려면 큰 효과 크기가 필요**. +56pp/년은 unrealistic sample로는 작은 효과.
4. **Research↔Production 구현 갭 자체가 배포 리스크**. 변경된 signal path는 모든 기존 BUG 시나리오 재검증 필요.
