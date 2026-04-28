# M3-R10 — Multi-dim Parameter Grid Search (사전 등록)

> **Date**: 2026-04-28
> **Authority**: 사용자 명시 — "한두 파라미터로 찾아내려고 하면 찾지 못할 가능성이 매우 큽니다. 다양한 파라미터 조합으로 연구를 진행해야 합니다."
> **Scope**: α/ι family에 대해 multi-dim parameter grid search. Single-axis sensitivity가 잡지 못하는 sweet spot 탐색.

---

## 1. 동기 (사용자 지적)

R1~R9까지 sensitivity probe는 각 parameter ±20% **univariate**. Joint multi-dim sweet spot은 미탐색. 예시:
- α @ (eth_thresh=0.5, atr_pctile=85, timeout_N=4) — 강한 ETH 신호 + 극단 vol regime + 짧은 hold 조합 미테스트
- ι @ (eth_thresh=0.4, btc_lag=0.05, eth_break_lookback=18, timeout_N=6) — 4-dim joint 미테스트

**Multi-dim sweet spot이 production criterion 충족할 가능성 — 17개 mechanism 표면 탐색에서는 보이지 않음**.

## 2. Multiple-comparison risk 명시

Grid search는 본질적으로 multiple comparisons:
- 6 × 5 × 5 × 6 = **900 combos** (α family) — 무작위 양수 기대치 ~45 (p=0.05 baseline)
- ι family는 추가 dim → 더 큼

**모든 grid 결과는 OOS 검증 없이 의미 없음**. Pre-register로 selection-after-peek 방지.

## 3. Methodology (사전 등록, 결과 보기 전)

### Step 1: Train/Test split
- Total: ~720 days
- **Train**: First 60% (~432 days)
- **Test**: Last 40% (~288 days, OOS holdout)
- Train과 Test는 **완전 분리** — Test에서 grid 안 함.

### Step 2: Grid 정의

#### α family (continuous parameters)
| Param | Grid values | Total |
|-------|-------------|-------|
| eth_thresh | 0.10, 0.20, 0.30, 0.40, 0.50, 0.60 | 6 |
| btc_lag_thresh | 0.00, 0.05, 0.10, 0.15, 0.20 | 5 |
| atr_pctile | 50, 60, 70, 80, 90 | 5 |
| timeout_N | 2, 4, 6, 8, 12, 16 | 6 |
| **Total combos** | | **900** |

Fixed:
- Exit: use_sl=False, use_trail=False, only emergency 1.5% + N timeout (R9b best framework)
- Friction: 0.04% (maker-tier assumption)
- Direction: trend-aligned (1h+4h)

#### ι family (adds eth_break_lookback dim)
| Param | Grid values | Total |
|-------|-------------|-------|
| eth_thresh | 0.20, 0.30, 0.40 | 3 (narrower) |
| btc_lag_thresh | 0.05, 0.10, 0.15 | 3 |
| atr_pctile | 60, 70, 80 | 3 |
| eth_break_lookback | 12, 18, 24, 30, 36 | 5 |
| timeout_N | 4, 6, 8, 12, 16 | 5 |
| **Total combos** | | **675** |

### Step 3: Train search
각 combo를 train 구간에 BT 실행. 다음 measure:
- daily_net @ 0.04
- n_train (trades in train)
- WR_train, RR_train

**Train pass criteria** (모두 충족):
- daily_net_train > 0
- n_train ≥ 50
- WR_train ≥ 40
- RR_train ≥ 1.0

### Step 4: Top-K selection
Train pass 통과한 combos 중 **daily_net 상위 10개** 선정.

### Step 5: Test verification (OOS)
선정된 10개 각각을 test 구간에 BT. **Test 합격** = daily_net_test > 0 AND n_test ≥ 30.

### Step 6: Random baseline control
**같은 grid를 random entry에 적용** (control). Random에서 OOS survivors가 expected 우연 카운트.

### Step 7: 합격 조건 (사전 등록)

**M3-R10 PASS** = 다음 모두 충족:
1. Train pass count ≥ 5 (충분 sample 다양성)
2. Top-10 중 **OOS test 양수 daily ≥ 5** (Bonferroni 류 — 50% 이상 survival)
3. Random baseline OOS survivors ≤ 2 (실제 alpha vs noise)
4. Best OOS combo의 bootstrap pos_rate ≥ 30% (R9c는 9% — 현저히 개선)
5. Best OOS combo의 train↔test daily 차이 < 50% (regime stability)

### 안전장치
- **결과 후 grid 확장 금지**. 본 grid 정의 그대로 한 번 실행.
- Test 결과에 기반한 추가 pre-reg 금지 (selection-after-peek 회피)
- 5 조건 중 1+ fail → "drop the claim" — 사용자 옵션 A/C로 redirect

## 4. Predictions

| 조건 | Predicted | Confidence |
|------|-----------|-----------|
| (1) Train pass ≥5 | likely PASS | HIGH (900 combos × 4 mech features 충분) |
| (2) OOS top-10 ≥5 survive | borderline | LOW — 다중 비교 noise |
| (3) Random control ≤2 | likely PASS | HIGH (random은 noise floor) |
| (4) bootstrap pos_rate ≥30% | likely FAIL | HIGH — R9c 9% 패턴 |
| (5) train↔test stability | borderline | LOW |

**Most likely outcome**: 조건 1, 3 PASS. 조건 2, 4 FAIL → drop claim.

**Most likely surprise**: ALL 5 PASS → 진짜 multi-dim sweet spot 발견. PDCA Plan으로 진행.

## 5. Anti-fix-impulse commitment

본 R10은 multi-dim grid 한 번 시도. 결과 무관하게:
- 추가 grid 확장 안 함
- Parameter range 확대 안 함
- 다른 family에 같은 search 반복 안 함
- 사용자 명시 redirect 후에만 다음 단계
