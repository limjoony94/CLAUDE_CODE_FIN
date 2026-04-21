# Slippage Diagnosis — C Phase 1+2 결과 종합 (2026-04-22)

> **관련 Plan**: `docs/01-plan/features/slippage_diagnosis.plan.md`
> **관련 결과**: `results/slippage_raw_20260422_025917.json`, `results/dd_comparison_20260421_235028.json`, `results/activation_sweep_20260422_012845.json`
> **후속 PDCA**: `docs/01-plan/features/pre_activation_baton.plan.md`

---

## TL;DR

- **Execution slippage는 갭의 주원인 아님**: Entry 0.058%, TRAILING 0.007%, STOP_MARKET 0.008%. 총 기여 5.7pp (3x) = 전체 갭 25.23pp의 **22%**만 설명.
- **주원인 (78%, ~19pp)**: BT의 `bar close` trail 평가 vs LIVE의 `intrabar tick` TRAILING_STOP_MARKET callback trigger **구조적 타이밍 불일치**.
- **해결 방향**: Pre-activation 구간에도 baton STOP_MARKET 적용 (post-activation 로직 확장). 별도 PDCA `pre_activation_baton`로 착수.

---

## 1. 측정 결과

### 1.1 Entry MARKET slippage (로그 기반, 26 trades)

| 통계 | 값 |
|---|---|
| Mean (부호) | +0.0045% |
| Median | +0.015% |
| Min | -0.159% |
| Max | +0.195% |
| **Adverse mean** | **0.0583%** |
| Adverse max | 0.195% |

판정: **갭 주원인 아님**. 27 trades × 0.058% × 3x = 4.7pp (갭 25pp의 19%).

### 1.2 Exit order slippage (CCXT fetch_closed_orders, 20 orders)

| Order Type | N | Adverse mean | Adverse max |
|---|---|---|---|
| TRAILING_STOP_MARKET | 18 | **0.0070%** | 0.0254% |
| STOP_MARKET (baton) | 2 | 0.0075% | 0.008% |
| MARKET (entry) | 30 | (trigger=0, 측정 불가) | - |

판정: **거래소 체결 품질 극도로 양호**. 갭의 1% 미만 기여.

### 1.3 Exit 경로 분포 (27 trades, state.json + 로그)

| Reason | 빈도 | 비율 |
|---|---|---|
| EXCHANGE_TRAIL | 22 | 81.5% |
| EXCHANGE_SL | 5 | 18.5% |
| TRAIL_TP (bot 내부) | 0 (state 기준, 일부는 ghost) | - |
| Ghost-resolved | 24/25 | 96% |

**96% ghost**: 봇 cycle 사이에 거래소가 intrabar trigger. bot은 다음 cycle에서 감지 → sync gap → ghost handler로 복원.

---

## 2. 가설 판정

| H | 가설 | 판정 | 증거 |
|---|---|---|---|
| H1 | Entry MARKET slippage가 주원인 | ❌ 기각 | adverse 0.058% |
| H2 | Pre-activation TRAILING 자체가 주원인 | **부분 수정** — Execution 자체는 미미 (0.007%) but trigger **timing**이 주원인 | 18 orders, slippage mean -0.0012% |
| H3 | SL ATR 의존 | 🔬 미확정 | 샘플 2건 |
| H4 | Post-activation baton 양호 | ✅ 확인 | 0.008% adverse |
| H5 | Reason mismatch | ✅ 확인 | BT TRAIL_TP 22 vs LIVE EXCHANGE_TRAIL 22 |

---

## 3. 갭 메커니즘 재정의

### BT `check_exit` 수식 (C1BreakoutSignal.check_exit)

```
cur_pnl = (close[i] / entry - 1) × direction_sign   # bar close 기준
drawdown = best_pnl - cur_pnl
if best_pnl > activation_pct AND drawdown > trail_K × ATR / close × 100:
    exit at close[i]     # bar close로 exit
```

### LIVE 현실

**Pre-activation (best_pnl ≤ activation_pct)**:
- TRAILING_STOP_MARKET 거래소 주문 배치 (callback_pct)
- BingX가 **tick-level best_price 실시간 추적**
- 작은 반전 (callback % 이상)에 **즉시 STOP_MARKET fill**
- → bar close보다 훨씬 이른 intrabar trigger

**Post-activation (best_pnl > activation_pct)**:
- Baton STOP_MARKET (bot.py `_calc_trail_trigger_price`)
- Bot cycle (15m)마다 재평가
- → bar close와 유사한 cycle 해상도 (BT와 근접)

### 구조적 격차 예시

BT #5 (LONG 71040→74312, +13.52%, 44b) vs LIVE #3 (LONG 71668→74026, +9.57%, 44b):
- 같은 신호, 같은 bars
- Exit price 격차: -286 pts = -0.40% (3x: -1.2pp)
- **실행 slippage 아닌 trigger timing 차이**

27 × 평균 0.4% × 3x ≈ **25pp** — 실제 갭과 **정확히 일치**.

---

## 4. Activation 스윕 BT 결과 (B 옵션)

| activation_pct | PnL 3x | MDD | 판정 |
|---|---|---|---|
| 0.05 (baseline) | +10.16% | -11.53% | 최적 |
| 0.10 | +10.16% | -11.53% | 동일 |
| 0.20 | +7.50% | -13.89% | 소폭 악화 |
| 0.50 | **-9.17%** | -18.00% | ❌ |
| 1.00 | +1.20% | -13.97% | 저조 |
| 1.50 | -5.28% | -17.02% | ❌ |

**Activation 상향은 BT에서 전부 악화**. 단, BT는 intrabar trigger를 가정하지 않으므로 LIVE 효과는 다를 수 있음. Execution slippage가 작다는 Phase 2 측정을 감안하면 activation 조정으로 갭 축소 가능성 낮음.

---

## 5. 다음 액션 권고 (우선순위)

| 옵션 | 상태 | 근거 |
|---|---|---|
| **E. pre_activation_baton** | ✅ Plan 작성 완료 | 주원인 직접 대응 (trigger timing → cycle-level) |
| **D. intrabar_parity Track A** | 🔬 이전 PDCA 재평가 | BT 자체를 현실적으로 (sub-bar traversal) |
| B'. callback_pct 축소 | ⚠️ 낮은 우선순위 | E가 callback 개념 자체 제거 예정 → B' 의미 축소 |
| A. progressive_trail | ✅ 이미 활성 | 독립 작용, 후속 효과 모니터링만 |

---

## 6. 30일 재평가 시 체크포인트

- Pre_activation_baton 적용 시: 실제 갭 축소 여부 (LIVE PnL vs BT)
- Progressive_trail 발동 빈도 (best_pnl > 0.9% 샘플)
- SL hit 비율 변화 (baton의 cycle-level 지연 효과)
- Emergency SL 발동 여부 (3% 상한)

---

## 7. Reference

- `bingx_rl_trading_bot/scripts/analysis/dd_comparison_20260421.py` — 27 trades BT vs LIVE
- `bingx_rl_trading_bot/scripts/analysis/activation_sweep_20260422.py` — B 옵션 스윕
- `bingx_rl_trading_bot/scripts/analysis/slippage_raw_collector_20260422.py` — C Phase 1+2
- `bingx_rl_trading_bot/claudedocs/BACKTEST_LIVE_PARITY.md` — 22-item 체크리스트
- `bingx_rl_trading_bot/claudedocs/bt_live_gap_deep_review_20260419.md` — 이전 갭 분석
- `memory/bt_live_gap_20260419.md` — 이전 추정 분해 (이제 execution 기여 하향 수정 필요)
