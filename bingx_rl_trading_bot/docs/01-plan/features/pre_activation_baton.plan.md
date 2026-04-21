# Plan: Pre-activation Baton STOP_MARKET 전면 적용

> **Feature**: pre_activation_baton
> **Date**: 2026-04-22
> **Phase**: Plan
> **Scope**: Pre-activation 구간에서도 TRAILING_STOP_MARKET 대신 baton STOP_MARKET 사용 (post-activation과 동일 로직 확장)
> **기반**: `slippage_diagnosis` Phase 1+2 (execution slip 기각, intrabar trigger timing이 주원인 확정)

---

## 1. Background

### Slippage_diagnosis PDCA Phase 2 결정적 측정

| 항목 | 측정값 | 이전 가정 |
|---|---|---|
| Entry MARKET adverse slip | **0.058%** (26 trades) | ~0.287% (추정) |
| TRAILING_STOP_MARKET adverse slip | **0.007%** (18 orders) | ~0.4~0.8% (추정) |
| STOP_MARKET (baton) adverse slip | **0.008%** (2 orders) | ~0.3% (추정) |

합산 시 execution slip 기여: **1.89pp (1x) / 5.7pp (3x)**. 실제 갭 **25.23pp (3x)** 의 **22%**만 설명.

**→ 나머지 78% (~19pp, 3x)는 execution이 아닌 다른 원인.**

### 갭 주원인 재정의: Trigger Timing 구조적 불일치

```
BT check_exit 수식:
  cur_pnl = (close[i] / entry - 1) * direction_sign   ← bar close 기준
  drawdown = best_pnl - cur_pnl
  if best_pnl > activation_pct AND drawdown > trail_K * ATR / close * 100:
      exit at close[i]

LIVE TRAILING_STOP_MARKET (BingX):
  tick-level best_price 실시간 추적
  callback_pct 반전 시 즉시 STOP_MARKET fill
  → bar close보다 훨씬 이른 시점에 trigger

LIVE Baton STOP_MARKET (post-activation, bot.py:1066):
  bot cycle (15m)마다 best_price 재평가
  _calc_trail_trigger_price() → 정확한 2차방정식 해
  STOP_MARKET 갱신 (LOOSEN-only)
  → bar close와 유사한 cycle 해상도
```

### 정량 예시 (BT #5 vs LIVE #3 비교)

같은 LONG 신호(71040 bar open), 같은 44 bars hold:
- BT exit: 74312 (bar close에서 trail 평가, 늦은 exit)
- LIVE exit: 74026 (intrabar callback trigger, 빠른 exit)
- Gap: **-286 pts = -0.40% (3x: -1.2pp)** — execution 아닌 timing

27 trades 중 TRAIL exit가 22건(81.5%). 각 0.4%씩 adverse timing이면:
- 22 × 0.4% × 3x = **~26pp 기여** ← 실제 갭 25pp와 일치

### Pre vs Post Activation

현재 C1 Breakout 설정:
- `trail_activation_pct: 0.05` (극히 낮음) — best_pnl > 0.05% 시 post-activation 전환
- Pre-activation 구간: best_pnl ∈ [0, 0.05%] → 매우 짧음 (이론상)

하지만 실제 order history에선 **TRAILING_STOP_MARKET이 여전히 18건 발생**. 이유:
- Trail activation은 `cur_pnl + draw_estimate > activation_pct` 충족 시점부터 적용
- 그 이전엔 pre-activation TRAILING_STOP_MARKET (callback 0.4~0.7%)이 활성
- 작은 pullback에도 trigger → TRAIL_TP-like 손실 확대

**E 옵션의 핵심**: Pre-activation 구간에도 baton STOP_MARKET (post-activation 로직)을 적용하여 trigger를 cycle-level로 만듦.

---

## 2. Goal

**BT-LIVE trigger timing 격차 축소로 갭 3x 기준 -25pp → -10pp 이내**.

Trade-off 수용:
- 큰 이익 상황에서 15m 지연으로 인한 반전 시 손실 확대 가능 (baton 특성)
- 단, LIVE의 BT 근접 → **LIVE가 BT의 검증된 edge (+9.37%, MDD -11.5%)를 현실화** 가능성

---

## 3. Hypotheses

| 가설 | 내용 | 검증 방법 |
|---|---|---|
| **H1** | Pre-activation baton 적용 시 LIVE trigger timing이 BT cycle과 일치 → 갭 축소 | 신규 BT (baton-only mode) vs 현재 LIVE 비교 |
| **H2** | Pre-activation baton은 큰 이익 국면에서 반전 시 drawdown 확대 (15m 지연) — net 효과는 positive | 과거 27 trades에 baton-only 시뮬 |
| **H3** | 본 변경은 progressive_trail과 독립적 (progressive는 activation 후 K 변경만, 본 변경은 activation 전 order type 변경) | 코드 레벨 독립성 검증 |
| **H4** | `_calc_trail_trigger_price` 수식이 best_pnl < activation_pct에서도 수학적으로 유효 | 수식 미분/경계 조건 테스트 |

---

## 4. Success Criteria (GO 조건 6개)

1. **gap_reduction**: LIVE-BT 갭 축소 (신규 BT 기준 예상 +3~6pp per 10일)
2. **baseline_preservation**: BT baseline (3.3, 2.5, 192) PnL이 +155% 이상 (clean +170%의 91%+)
3. **wf_pass**: baton-only BT WF 5-fold 전부 양수
4. **formula_validity**: `_calc_trail_trigger_price`가 best_pnl < activation_pct에서도 수학적 유효 (no NaN/negative distance)
5. **rollback_ready**: `config.yaml` 토글 `pre_activation_baton.enabled=false`로 즉시 원복 가능
6. **live_ab_ready**: 운영 반영 시 state.json/orphan-resolver와 충돌 없음 (TRAILING → STOP_MARKET 전환 smooth)

5/6 이상 GO. H4(formula_validity) 실패 시 무조건 STOP.

---

## 5. Design

### 5.1 현재 로직 (bot.py)

```python
# Entry 시점 (bot.py:_exchange_open 근방)
if best_pnl <= activation_pct:
    order_type = TRAILING_STOP_MARKET
    params = {'callbackRate': callback_pct}
else:
    order_type = STOP_MARKET
    stop_price = _calc_trail_trigger_price(pos, cur_atr)

# Cycle update (bot.py:_update_exchange_trail 근방)
if best_pnl > activation_pct AND still_loosen:
    baton-touch STOP_MARKET 재배치
```

### 5.2 변경 로직

```python
# Entry 시점
if config.get('pre_activation_baton', {}).get('enabled', False):
    # Pre-activation에도 baton: 초기 best_pnl=0이므로
    # _calc_trail_trigger_price가 best_pnl ≈ 0에서도 유효한지 먼저 검증
    stop_price = _calc_trail_trigger_price_safe(pos, cur_atr, best_pnl=0)
    if stop_price and stop_price > 0:
        order_type = STOP_MARKET
    else:
        # Fallback to legacy TRAILING for edge cases
        order_type = TRAILING_STOP_MARKET

# Cycle update
# best_pnl <= activation_pct 구간에서도 baton-touch 갱신
if config.get('pre_activation_baton', {}).get('enabled', False) or best_pnl > activation_pct:
    baton_update_logic()
```

### 5.3 `_calc_trail_trigger_price` at best_pnl=0 경계

기존 수식 (bot.py:1066):
```
cur² - best·cur + trail_K·ATR·entry = 0
→ cur = (best ± √(best² - 4·trail_K·ATR·entry)) / 2
```

At best = entry (best_pnl=0):
```
cur² - entry·cur + trail_K·ATR·entry = 0
Discriminant = entry² - 4·trail_K·ATR·entry = entry·(entry - 4·K·ATR)

Given entry ~= 75000, K=2.5, ATR ~= 150:
4·K·ATR = 1500 << 75000 → discriminant > 0 ✓

cur = (entry ± √(...)) / 2
     LONG case: lower root (cur < entry) = STOP below entry
     → SL-like behavior before any profit accrues
```

**수식 유효**. 단, best = entry 시 계산된 cur는 기존 fractal SL보다 tight하거나 loose할 수 있음. SL과 baton 중 **더 가까운 쪽** 사용 (max of SL_price, baton_price for SHORT; min for LONG) 필요.

### 5.4 Config

```yaml
strategy:
  # ... existing ...
  pre_activation_baton:
    enabled: false       # 기본값 false (opt-in)
    sl_priority: true    # baton vs fractal SL 중 tighter 선택
```

---

## 6. Implementation Plan

### Phase 1: BT simulation (baton-only mode)
1. `scripts/analysis/baton_only_backtest.py` 신규
2. 27 trades 기간에서 TRAILING 대신 baton만 사용한 BT 실행
3. LIVE 27 결과와 비교 → gap 축소 정량
4. 과거 333일 full BT도 돌려 baseline preservation 검증

### Phase 2: 수식 boundary 테스트
1. `_calc_trail_trigger_price`에 best_pnl=0, 소량 positive 입력 단위 테스트
2. NaN/negative distance 방지 가드 추가 (필요 시)

### Phase 3: 구현 + Pytest
1. `bot.py`에 `pre_activation_baton` config 로드
2. `_exchange_open`, `_update_exchange_trail` 분기 추가
3. 신규 test 케이스 (test_pre_activation_baton.py):
   - Pre-activation baton order type = STOP_MARKET
   - best_pnl=0에서 trigger price 수학적 유효
   - Fractal SL과 baton 공존 시 tighter 선택
   - TRAILING → STOP_MARKET 전환 시 orphan-resolver 호환

### Phase 4: LIVE A/B 배포 (enabled=false → true)
1. Config 기본값 false로 merge
2. Canary 기간 (7일): 하루 1-2 positions만 enabled=true
3. 전면 enabled=true 전환 후 30일 평가
4. Rollback: config flip + 봇 재시작

**총 5~7일 예상 (Phase 3 구현 난이도 중간).**

---

## 7. Non-Goals

- Intrabar BT 모델 변경 (별도 `intrabar_parity` PDCA)
- LIMIT order 도입 (별도 후속)
- Post-activation logic 변경 (현행 baton 유지)
- BingX API 변경 요청

---

## 8. Rollback

- Config `pre_activation_baton.enabled: false` → 1줄 변경으로 즉시 원복
- 봇 재시작 필요 (config 로드 시점)
- 기존 TRAILING_STOP_MARKET orders는 거래소에 유지, 다음 cycle에서 정상 갱신

---

## 9. Risks

| 리스크 | 완화 |
|---|---|
| best_pnl=0 수식 발산 | Phase 2 boundary 테스트 + NaN 가드 |
| Pre-activation baton이 fractal SL보다 loose → 실제 SL 보호 약화 | `sl_priority: true` 옵션으로 tighter 선택 |
| 15m cycle 지연으로 큰 반전 시 drawdown 확대 | Emergency SL 3% 상한 유지 |
| TRAILING → STOP_MARKET 전환 중 cycle이 짧게 끊어지면 포지션 protection 공백 | 기존 BUG#59 streak 감지 재활용 |
| Orphan/ghost resolver가 새 order type 대응 못함 | Phase 3 테스트에 flow 추가 (_resolve_orphan_sl 포함) |

---

## 10. Reference

- `results/slippage_raw_20260422_025917.json` — C Phase 2 실측 (TRAILING 0.007%, baton 0.008%)
- `results/dd_comparison_20260421_235028.json` — 갭 기준 (BT +9.37% vs LIVE -15.86%)
- `bingx_rl_trading_bot/scripts/production/c1_breakout/bot.py:1066` — `_calc_trail_trigger_price`
- `bingx_rl_trading_bot/scripts/production/c1_breakout/bot.py:1223` — 현 baton 적용 구간
- `claudedocs/BACKTEST_LIVE_PARITY.md` — #21 (pre-activation TRAILING 구조적 한계)
- `memory/progressive_trail_20260421.md` — 본 변경과 독립 실행 가능

---

## 11. Relationship with Other PDCAs

| PDCA | 관계 |
|---|---|
| `slippage_diagnosis` | 본 PDCA의 직접 상위 근거 (주원인 확정 → 대응 방안) |
| `intrabar_parity` | 본 PDCA 적용 후에도 BT 자체는 bar-close 가정 유지 → 상호 보완 |
| `progressive_trail` | 본 PDCA와 독립 (activation 후 K 조정 vs activation 전 order type 변경) |
| `callback_tightening` (B', 신규) | 본 PDCA가 성공하면 callback 개념 자체가 제거되어 B' 폐기 |
