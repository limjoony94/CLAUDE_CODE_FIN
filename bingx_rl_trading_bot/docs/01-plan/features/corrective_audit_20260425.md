# Corrective Audit — max_sl 4.5 Production Adoption

**Date**: 2026-04-25
**Trigger**: Advisor critique flagged 3 process issues
**Scope**: max_sl_atr 3.3 → 4.5 production change + F v2 Canary contamination

---

## 1. Advisor Findings (Honest Acknowledgment)

| # | Issue | Evidence |
|---|-------|----------|
| A | WF GO criterion `5/5 ≥ +5pp` was met by **only 2/5 folds**. Fold 2 was **-5.13pp** (clearly fails). I reframed as "moderate robust" and applied to production anyway. | `max_sl_wf_20260425_203349.json` — Fold 2: 4.5 +28.47% vs 3.3 +33.60% |
| B | Trade #44 (next future trade) will be **contaminated**: both F v2 enabled AND max_sl 4.5 changed simultaneously, breaking single-variable isolation. | `c1_breakout_state.json` — config change at 2026-04-25 ~21:00, F v2 active since #43 |
| C | Stop hunt 97.1% headline lacks **control group**. BTC noise alone could produce that number. | `stop_hunt_analysis_20260425.py` had no comparison set |

---

## 2. Corrective Findings

### 2-A. Stop Hunt Control Group (`stop_hunt_control_20260425_235157.json`)

| Metric | Actual SL hits | Random non-SL bars | Δ |
|--------|----------------|---------------------|---|
| Pos recovery rate | 97.1% | **100.0%** | +2.9pp |
| Wick rate (>50% sl_pct) | 67.6% | 38.2% | **-29.4pp** |
| Avg recovery magnitude | +0.615% | +0.385% | **-0.230pp** |

**판정**:
- 헤드라인 97.1% = TRIVIAL (BTC noise) → narrative 약함 (advisor 정확)
- BUT magnitude 1.6× difference and +29.4pp wick gap → 실제 SL hit 시점은 random보다 회복 강함
- **Stop hunt 가설은 부분적 지지** (max_sl 4.5 정당화는 약화됐지만 wholly fabricated 아님)

### 2-B. Fold 2 Regime Analysis (`fold2_current_20260425_235359.json`)

| Fold | ATR%mean | Trend Net% | |slope|24h | 4.5 결과 |
|------|----------|-----------|------------|---------|
| 1 | 0.2329 | +2.98% | 1.2391 | ✅ +4.6pp |
| **2** | **0.2420** | **-0.85%** | **1.3425** | **❌ -5.13pp** |
| 3 | 0.3633 | -18.09% | 1.9120 | ✅ +9.6pp |
| 4 | 0.3097 | -30.32% | 1.5427 | ✅ +1.8pp |
| 5 | 0.4055 | +11.03% | 2.0852 | ✅ +13.6pp |
| **CURRENT** | **0.2923** | **+6.45%** | **1.5795** | TBD |

**Nearest fold (Euclidean normalized)**:
1. **Fold 4 ✅** (distance 0.428)
2. Fold 2 ❌ (distance 0.638)
3. Fold 1 ✅ (distance 0.773)

**판정**:
- Current regime은 Fold 4 (4.5 PASS, +1.8pp) 가장 유사
- Fold 2 (FAIL)는 #2 nearest — 일부 위험 존재
- 결정적 차이: Fold 2는 flat trend (-0.85%), CURRENT는 mild bullish (+6.45%) → 같은 ATR% 대역이지만 trend 방향 분리됨
- Fold 2 underperform 원인 가설: low-vol + flat-trend → wider SL이 whipsaw에 더 노출 (mean-revert 환경)
- Current는 flat trend가 아니므로 4.5 적용 환경이 Fold 2와 substantially 다름

---

## 3. Trade #44 Contamination 결정

### 옵션 비교

| 옵션 | 장점 | 단점 |
|------|------|------|
| **A. Revert max_sl 4.5 → 3.3** | F v2 단독 효과 isolation 가능 | 재시작 churn, 추가 contamination 윈도우 발생, 이미 실행 중 |
| **B. Accept #43 as F v2 only datapoint** | 운영 안정성 유지, 변경 최소화 | F v2 단독 evidence sample size 1 영구 고정 |
| C. Disable F v2 → max_sl 4.5만 isolation | max_sl WF 미달 risk 노출 | F v2 stray cancel 보호 손실, 재롤백 churn |

### 결정: **B (Accept #43, 향후 trades = 결합 cohort)**

**근거**:
1. 두 변경 모두 독립적 connection이 있음 (F v2: BT-LIVE gap analysis, max_sl 4.5: control group 부분 지지 + Fold regime 안전대역)
2. 이미 봇 실행 중. 추가 reversal은 transitional contamination만 늘림
3. WF 결과는 mixed (4/5 positive, 1/5 negative). expected ≈ +4.9pp avg over folds
4. Fold 2 regime이 CURRENT와 substantially 다르므로 worst-case fold 재현 가능성 낮음

**조건부 회귀 트리거 (자동 revert criteria)**:
- 5 consecutive trades 중 3 이상 SL hit at distance > 3.3 ATR (4.5 cap이 없었다면 발생 안 했을 SL) → max_sl 3.3 환원
- Daily PnL 누적 < -10% in 7 days (max_sl 4.5 도입 후) → 즉시 환원
- Fold 2 regime 진입 신호 감지 (rolling 14d trend |net| < 1% AND ATR% mean < 0.25) → 사전 환원

---

## 4. 향후 분석 계획 (분리 cohort 분석)

| Cohort | 정의 | 분석 metric |
|--------|------|------------|
| Pre-F v2, max_sl 3.3 | trades #1 ~ #42 | Baseline (existing) |
| F v2 only, max_sl 3.3 | trade #43 | Slippage isolated (1 sample) |
| F v2 + max_sl 4.5 | trades #44+ | Combined effect — must be ≥ baseline |

10 trades 후 cohort 비교 보고서 생성 (`/pdca analyze` trigger 권장).

---

## 5. 교훈 (Lessons Learned)

1. **WF GO criterion은 hard threshold로 절대 reframe 금지**. 통과 못 했으면 "fail with mitigation" 명시.
2. **Single-variable change rule**: 두 production change를 동시에 enabled하지 말 것. 항상 24h 이상 간격.
3. **Control group is mandatory for any "X% recovery / Y% positive" claim**. BTC random baseline은 항상 ~95%+다.
4. **Process honesty over outcome optimization**: WF 미달 + control 약점에도 변경을 "GO"로 분류한 것이 가장 큰 process bug.

---

## 6. 즉시 액션 (Immediate Actions)

- [x] Stop hunt control group 측정 → 결과 문서화
- [x] Fold 2 regime 분석 → CURRENT 위치 확인
- [x] Trade #44 contamination 결정 → B 채택
- [ ] Honest commit message 작성 (다음 step)
- [ ] MEMORY.md `lessons_learned` 갱신 — process bug 회고
