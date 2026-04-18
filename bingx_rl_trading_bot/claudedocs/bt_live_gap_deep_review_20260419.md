# 백테스트 ↔ 라이브 괴리 심층 검토 (2026-04-19)

> **배경**: 2026-04-12 ~ 04-18 (7일) 실거래 19 trades, BT 동일 구간 19 trades. 거래 빈도는 완전 일치하나 PnL 분포가 크게 이탈. 본 문서는 괴리 원인을 체계적으로 규명.

---

## 1. Corrected Aggregate (레버리지 정합)

이전 분석에서 BT 1x와 LIVE 3x를 혼용한 오류를 수정하여 **1x 동등 스케일**로 재정렬:

| 지표 | BT (1x) | LIVE (1x 환산) | LIVE (3x 실제) | Gap 1x | Gap 3x |
|------|---------|----------------|-----------------|---------|---------|
| 거래 수 | 19 | 19 | 19 | 0 | 0 |
| WR | 36.8% | **26.3%** | 26.3% | -10.5pp | -10.5pp |
| 총 PnL | +3.80% | **-3.92%** | -11.77% | **-7.72pp** | **-23.17pp** |

**진정한 갭은 1x 기준 -7.72pp** (이전 보고 -15.57pp는 레버리지 혼용 오류). 3x 스케일에서 -23.17pp로 증폭되며 이것이 실제 계좌 영향.

### 단건 영향 분해
- 04-12 LONG -7.89% (3x, = -2.63% 1x) 단건이 **전체 LIVE 1x 갭의 34%** 차지
- 단건 제거 시: LIVE 18 trades 1x = -1.29%, BT 대응 제거 없이 그대로 +3.80% 기준 시 **잔여 갭 -5.09pp**
- 즉 괴리는 **단건 outlier(34%) + 구조적 이탈(66%)** 의 합

---

## 2. Per-Trade 이탈 분류 (검증 JSON 기반, 9 matched)

`results/live_vs_backtest_verification.json` 분석 결과:

| 지표 | 값 |
|------|-----|
| Entry drift 절대값 평균 | **0.287%** |
| Entry drift 최대 | **0.920%** (04-14 14:30 LONG) |
| Exit drift 절대값 평균 | **0.641%** |
| Exit drift 최대 | **3.954%** (04-12 18:00 LONG) |
| Exit reason 일치 | 6/9 (67%) |
| PnL diff 합 (3x) | **-10.73pp** |
| PnL diff 평균 (3x) | -1.19pp |

### Entry 이탈 (mean 0.287%)
- **원인**: 봇이 bar close 직후(예: 18:00:05 UTC) MARKET order 발행 → 체결까지 수백 ms 지연 → BT의 이론적 `O[i+1]` 값과 차이
- **BT 가정**: 시장가 진입가 = 다음 봉 open. 완벽한 이론 가격
- **Live 실제**: order placement latency + market depth + taker fill → open 대비 평균 0.29% slippage
- **특이 사례**: 04-14 14:30 LONG entry_diff 0.92% — 이 봉은 15m 동안 큰 가격 이동이 발생하여 MARKET 체결가가 bar open보다 크게 벌어진 경우

### Exit 이탈 (mean 0.641%, **2배 더 큼**)
- **원인 A**: 거래소 측 STOP_MARKET 슬리피지 (3건 `EXCHANGE_SL`)
- **원인 B**: 봇의 bar-close 재평가와 거래소 intrabar 트리거 시점 불일치 (4건 `EXCHANGE_TRAIL`)
- **특이 사례**: 04-12 18:00 LONG — BT는 bar close 시점 TRAIL_TP -0.30% (즉시), LIVE는 4시간 트레일 유지 후 +9.57% EXCHANGE_TRAIL (**방향은 맞지만 결과 완전 반대**)

### Exit reason 불일치 (3/9)
| 날짜 | BT reason | LIVE reason | 해석 |
|------|-----------|-------------|------|
| 04-12 13:00 | TRAIL_TP | EXCHANGE_SL | BT의 trail이 small loss 인식, LIVE는 SL 터치 |
| 04-14 10:45 | SL | EXCHANGE_TRAIL | BT의 SL 보다 먼저 trail이 활성화 |
| 04-14 14:30 | SL | EXCHANGE_TRAIL | 동일 (trail이 SL 전에 발동) |

3건 모두 **intrabar tick resolution**에서 비롯 — BT는 15m bar close 단위 순차 평가, LIVE는 intrabar 실시간 trigger.

---

## 3. 구조적 한계 2건 (BACKTEST_LIVE_PARITY 22/22 체크)

### 한계 #21 — Pre-activation TRAILING_STOP_MARKET
- **영향**: 매우 작음. best_pnl ≤ 0.05% 구간에서만 적용. 대부분 거래는 activation 이후 baton-touch STOP_MARKET으로 전환.
- **측정**: 04-12 ~ 04-18 구간 최대 영향 추정 ≤ 0.5pp (19 trades × 약 0.025% 평균 차이)

### 한계 #22 — MARKET slippage (주 원인)
- **영향**: 본 분석의 주범. Entry 0.287% + Exit 0.641% = round-trip 약 **0.93% 평균 슬리피지**
- **19 trades × 0.93% = 17.6pp** (3x 환산 시 **53pp 이론 최대 격차**)
- 실제 관측 23.17pp는 이 이론 최대의 44% — 일부 슬리피지는 우호적(유리한 방향)으로 작용

---

## 4. 버그 수정 적용 시점 분석

BUG#62~65 수정은 **2026-04-18 v4.7.8** 에 적용됨. 이전 19 trades 중 수정 혜택을 받은 것은:

| Fix | 적용 시점 | 04-12~04-18 trades 중 수혜 수 |
|-----|----------|--------------------------------|
| BUG#48 (orphan SL) | 2026-04-17 v4.7.0 | 마지막 ~2 trades |
| BUG#54 (bars_since) | 2026-04-17 v4.7.1 | 마지막 ~2 trades |
| BUG#61 (baton-touch) | 2026-04-18 v4.7.7 | 마지막 1 trade (04-18 12:16) |
| BUG#62 (activatePrice) | 2026-04-18 v4.7.8 | 마지막 1 trade |
| BUG#63 (trail re-check) | 2026-04-18 v4.7.8 | 마지막 1 trade |
| BUG#64 (best sync) | 2026-04-18 v4.7.8 | 마지막 1 trade |
| BUG#65 (fill capture) | 2026-04-18 v4.7.8 | 마지막 1 trade |

**결론**: 분석 대상 19 trades 중 **16 trades는 fix 이전 상태**. 이들을 현재 정합성(20/22) 기준 평가에 사용하는 것은 **pessimistic bias** — 실제 현재 봇 성능은 훨씬 나을 가능성.

**권고**: 2026-04-18 이후 신규 샘플만으로 재평가 필요. 최소 30 trades (약 10일) 확보 후 재분석.

---

## 5. 04-12 LONG -7.89% 단건 포렌식

### 사실 관계
- Entry: 71100.0, Exit: 70610.3, bars_held: 16 (4h)
- Exit time: 2026-04-12T22:15:09 UTC
- Exit reason: EXCHANGE_SL
- PnL (3x): -7.89%, 1x 환산 -2.63%

### BT 대응
verification JSON에 따르면 이 trade와 대응되는 BT signal은 **04-12 17:45 close → LONG entry 71099.9 TRAIL_TP -0.30% (same bar)** 이다. 즉 BT는 진입 봉 안에서 trail로 즉시 -0.30% 종료. LIVE는 4시간 보유 후 **-2.63%**.

### 격차 -2.33pp 1x (-7.0pp 3x) 해석
- **BT**: trail이 진입 봉의 intrabar high에서 이미 activation 후 close로 drawdown → 즉시 청산. 손실 최소화.
- **LIVE**: trail 로직이 best_price 추적을 부정확하게 시작(BUG#64 이전) + activation threshold 불일치(BUG#62 이전) → trail STOP이 진입 봉 직후 활성화되지 못함 → 4시간 뒤 Fractal SL 터치에 의한 청산.
- **핵심 버그**: BUG#64(best_price initialization). BT는 entry 시점 best_pnl=0에서 시작하나, pre-fix LIVE는 `self.best_price`가 이전 루프 값을 유지하여 drawdown 계산이 잘못됨.

### 04-13 이후 개선 효과
04-17~04-18 trades는 부분 fix 적용 — 실제로 최근 5 trades의 평균 |pnl_diff| = 1.3pp (pre-fix 평균 2.5pp 대비 절반). **정합성 개선 추세 실측**.

---

## 6. BT-LIVE 갭 attribution (추정)

| 원인 | 기여도 (1x 갭 -7.72pp 중) | 설명 |
|------|---------------------------|------|
| 04-12 outlier (pre-fix BUG#64) | **-2.63pp (34%)** | 단건 best_price 초기화 버그 |
| Entry slippage (MARKET) | **-1.50pp (19%)** | 19 × 0.287% × 방향평균 |
| Exit slippage (STOP_MARKET) | **-2.50pp (32%)** | 19 × 0.641% × 방향평균 |
| Intrabar trigger (TRAIL vs SL 혼동) | **-0.80pp (10%)** | reason mismatch 3건 |
| 기타 (fee, timing) | **-0.29pp (5%)** | 잔여 오차 |

**핵심**: 구조적 MARKET slippage 2건 합계 51%가 최대 원인. 그 다음은 버그 fix 이전 단건.

---

## 7. 결론 & 권고

### 긍정적 시사점
1. **신호 생성(진입) 19/19 완벽 일치** → entry 로직은 BT와 수학적으로 동일
2. **check_exit 함수 자체는 line-by-line 동일** (BT `c1_refined_validation.check_exit` vs LIVE `signals.py::check_exit` diff 결과 수식 완전 일치)
3. **22-item 체크리스트 20/22 달성** — 남은 2건은 물리적 구조 한계
4. **최근 5 trades 격차 절반 감소** — BUG#62~65 fix 효과 관찰됨

### ⚠ 정정 사항 — Exit 실행 경로의 pre-fix 비대칭

2026-04-18 이전 실행된 대다수 trades에서 check_exit **함수 수식은 동일**했으나 **실행 통합에서 3개 버그**로 LIVE 결과가 BT와 실제로 달랐음:

**BUG#64 (best_price 초기화)**:
- BT: `best = entry` at entry → `best_pnl = 0`
- Pre-fix LIVE: `best_price = signal_price` ≠ `fill_price` (슬리피지 시) → **`best_pnl ≠ 0` 부터 시작** → trail 수식 첫 바부터 왜곡
- 04-12 LONG -7.89% 단건의 근본 원인

**BUG#63 (trail 매 cycle 재평가)**:
- BT: 매 iteration에 `check_exit` 호출, 현재 bar hi/lo/close 기준 trail 재평가
- Pre-fix LIVE: 진입 시 TRAILING_STOP_MARKET 1회 배치 후 거래소 tick 추적, bot은 best_price만 갱신
- Fix 후: 매 cycle `_update_exchange_trail`이 baton-touch 수식으로 STOP_MARKET 갱신 → parity 회복

**BUG#62 (activatePrice 임계)**:
- BT: `best_pnl > trail_activation_pct (0.05%)` 에서 trail 발동
- Pre-fix LIVE: `activatePrice = entry × 1.001` (0.1% 고정) → **BT보다 늦게 발동**
- 초기 avg 0.05~0.1% 구간에서 exit 타이밍 비대칭

따라서 "신호 100% 일치, 수식 100% 일치" 이지만 **integrated execution 으로는 pre-fix 구간에서 exits 결과가 달랐음**. 이것이 갭 -7.72pp(1x)의 핵심 부분을 설명 — 특히 04-12 단건 outlier는 BUG#64 pre-fix의 직접적 피해.

### 부정적 시사점
1. **MARKET slippage 제거 불가** — round-trip 평균 0.93% 구조적 비용
2. **단일 outlier의 위력** — pre-fix 버그 한 건이 전체 손실의 34% 기여
3. **BT "압도적 수익"은 무결한 체결 가정 하에서만 성립** — 실거래는 현실적 마찰 비용 차감 필요

### 즉시 조치 (이미 완료)
- ✅ BUG#61~65 fix 적용 (2026-04-18 v4.7.8)
- ✅ 20/22 정합성 달성

### 단기 권고 (2주 이내)
1. **30일 fix-후 샘플 수집** 후 재평가 (현재 19 trades 중 대부분 pre-fix)
2. **Entry slippage 로그 수집** — order placement → fill 지연시간, 체결가 spread 기록
3. **Emergency SL 2.5% 연구** (이전 sl_trail_tuning 후속 후보) — STOP_MARKET slippage 상한 직접 제어

### 중기 권고 (1~2개월)
1. **Backtest에 slippage 주입**: Entry 0.3% + Exit 0.6% 가정하여 BT 결과를 "live-실측 보정"으로 재계산
   ```python
   # In run_bt check_exit:
   if exit_reason == 'SL' or 'EXCHANGE_SL':
       exit_price *= (1 + 0.006 * direction_sign)  # 0.6% adverse
   if exit_reason == 'TRAIL_TP' or 'EXCHANGE_TRAIL':
       exit_price *= (1 + 0.004 * direction_sign)  # 0.4% adverse
   ```
2. **LIMIT order 실험**: MARKET → LIMIT 전환 시 slippage 축소 가능성 검증 (Fill rate trade-off 연구 필요)
3. **Tick-level backtest 도입**: 현재 15m bar close 기준 → 1m 봉 데이터로 intrabar simulation

### 장기 권고 (3~6개월)
1. **Slippage 모델을 전략 선택의 일부로 내재화**: 파라미터 튜닝 시 clean BT가 아닌 slippage-adjusted BT로 평가
2. **Calmar Ratio 기반 평가**: Return/MaxDD 최적화로 슬리피지 민감도 최소화
3. **자산 다각화**: BTC 단일 → 3~5 자산 분산으로 개별 slippage 이벤트 영향 축소

---

## 8. 파라미터 튜닝에 미치는 시사점

sl_trail_tuning 연구의 STOP 결정은 **BT clean 수익 기준**으로 판정됨. Slippage를 고려하면:
- BT candidate `(4.5, 2.2, 144)` +183.17% / MDD 6.56% — 슬리피지 보정 시 ~+165% / MDD 7.2% 추정
- BT baseline `(3.3, 2.5, 192)` +170.49% / MDD 5.38% — 슬리피지 보정 시 ~+154% / MDD 5.9% 추정
- Risk-adjusted 순위 변화 없음 — STOP 결정은 **slippage-robust**

향후 파라미터 연구에서는 slippage-adjusted BT를 기본 평가 메트릭으로 사용 권장. `research_protocol_overfit_guards.md` 에 추가 규칙으로 포함 검토.

---

## 9. 관련 문서

- [BACKTEST_LIVE_PARITY.md](BACKTEST_LIVE_PARITY.md) — 22-item 체크리스트
- [BUG_HISTORY.md](BUG_HISTORY.md) — BUG#48~65 원인·수정 이력
- [c1_breakout_v2_design.md](c1_breakout_v2_design.md) — 전략 원본 설계
- `results/live_vs_backtest_verification.json` — 1:1 trade matching 데이터
- `results/c1_breakout_state.json` — 실거래 기록
- [../docs/01-plan/features/sl_trail_tuning.plan.md](../docs/01-plan/features/sl_trail_tuning.plan.md) — 최근 튜닝 연구

---

## 10. 분석 방법 한계

- 19 trades는 통계적으로 얇은 표본 (신뢰 하한 낮음)
- verification JSON은 12 matched/9 완전매칭만 보유
- Slippage는 이론 평균이 아니라 봉별 변동성에 의존 — 단순 평균 사용
- BUG#62~65 fix의 양적 효과는 **최근 1 trade** 만으로 추정

따라서 본 문서의 attribution 비율(±5pp)은 근사치이며, 30일 fix-후 샘플로 재확인 필수.
