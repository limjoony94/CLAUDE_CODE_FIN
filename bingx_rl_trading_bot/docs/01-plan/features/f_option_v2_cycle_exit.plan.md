# Plan: F v2 — Cycle-level check_exit + MARKET close

> **Feature**: f_option_v2_cycle_exit
> **Date**: 2026-04-24
> **Phase**: Plan
> **Scope**: F v1 실패 원인 해소 — BT `check_exit`의 "soft break-even" 트레일을 봇 cycle에서 직접 호출하여 MARKET close
> **기반**: F v1 (activation_gated_trail) Canary 결과 (Gap -10.56pp / 2일)

---

## 1. Background

### F v1 Canary 결과 (2026-04-22 ~ 2026-04-24, 8 trades)

| Metric | BT (semantic) | LIVE Canary | Gap |
|---|---|---|---|
| PnL3x | +3.60% | **-6.96%** | **-10.56pp** |
| WR | 28.6% | 37.5% | - |
| MDD | -0.90% | -8.75% | -7.85pp |
| **Exit reasons** | **100% TRAIL_TP** | **87.5% EXCHANGE_SL** | 완전 역전 |

### 근본 원인: BT vs LIVE의 Trail Semantic 차이

**BT `signals.py check_exit` trail 분기**:
```python
if best_pnl > activation_pct:
    drawdown = best_pnl - cur_pnl        # cur_pnl = (close/entry - 1)*100
    trail_dist = trail_K * atr / close * 100
    if drawdown > trail_dist:
        return {'exit_price': close[i], 'reason': 'TRAIL_TP'}
```

→ 매 bar close마다 drawdown을 **동적 재계산**. 가격이 entry 근처로 회귀 시 drawdown = best_pnl − 0 = 상승된 best_pnl → trail_dist 쉽게 초과 → **break-even 수준에서 exit**.

**LIVE baton STOP_MARKET** (F v1, 현 bot.py):
```python
baton_trigger = (best + sqrt(best² - 4*K*ATR*entry)) / 2
# STOP_MARKET @ baton_trigger (고정)
```

→ Fixed trigger price. Entry 근처로 회귀해도 STOP은 **best − K*ATR 수준**에 있어서 trigger 안 됨. 가격이 더 내려가 SL에 hit.

### 구체 예시 (BT #1 vs LIVE #35, 같은 LONG 신호)

- Signal: 04-22 LONG, entry ~79000
- BT: entry 79014 → best ~79500 → close 회귀 79014 시 drawdown ≈ 0.6% > trail_dist (0.4%) → TRAIL_TP at 79014 (**-0.30% fee only**)
- LIVE: entry 79250 → best ~79300 → **baton 배치 안 함 (F v1에서 activation 미도달 구간)** → 가격 추락 → SL 78727 hit (**-2.28%**)
- 같은 신호, **2.0pp per-trade gap**

### F v1은 왜 실패했나

F v1 "activation-gated placement"는 **pre-activation에 trail을 안 배치**하는 것. 하지만 BT의 `check_exit` trail은 activation 도달 **후** 활성화되는 "soft break-even" 역할을 수행. 이 메커니즘이 LIVE에 복원되지 않아:
- Pre-activation SL hit (F v1에서 trail 배치 안 됨)
- Post-activation에도 baton STOP_MARKET의 고정 trigger가 "entry 회귀 exit"을 복원하지 못함

**결론**: LIVE에 BT의 trail semantic을 복원하려면 **baton STOP_MARKET이 아닌 봇이 매 cycle에서 check_exit trail 로직을 평가 + MARKET close** 방식이 필요.

---

## 2. Goal

**BT의 soft break-even trail semantic을 LIVE에 cycle-level로 복원**하여 LIVE PnL을 classic BT와 일치시킴.

- 4주 구간 LIVE PnL3x이 BT와 ±5pp 이내
- EXCHANGE_SL 비율 ≤ 20% (BT 기준, pre-Canary 27-trade에서 18.5%)
- 333일 baseline +170% 보존

---

## 3. Hypotheses

| H | 가설 | 검증 방법 |
|---|---|---|
| **H1** | 봇 cycle마다 check_exit 호출 + MARKET close 시 BT의 TRAIL_TP가 LIVE에 재현됨 | Canary 구간에서 F v2 적용 paper 시뮬 or Phase 3 A/B |
| **H2** | 15m cycle 해상도로도 intrabar TRAIL_TP 타이밍과 ±0.3% 이내 근접 | BT close exit vs LIVE cycle exit 가격 비교 |
| **H3** | MARKET close의 slippage가 0.1% 이내 (기존 측정 0.058%) | fetch_my_trades로 측정 |
| **H4** | Fractal SL STOP_MARKET을 거래소에 유지하면 cycle 사이 crash 시에도 보호 | 봇 의도적 kill → SL hit 검증 |
| **H5** | F v2 + activation_gated (F v1 조합)은 중복 불필요. F v2만으로 충분 | BT 비교 |

---

## 4. Success Criteria (GO 조건 7개)

1. **gap_reduction**: LIVE 4주 샘플 PnL3x가 BT와 ±5pp 이내
2. **baseline_preservation**: 333일 full BT (F v2 logic) +160% 이상 (classic +170% 94%+)
3. **exit_distribution_match**: EXCHANGE_SL 비율이 BT와 ±15% 이내
4. **slippage_bound**: MARKET close slippage 평균 ≤ 0.15%
5. **crash_safety**: 봇 다운 시 fractal SL이 exchange에서 intrabar 발동
6. **rollback_ready**: config flag `f_v2_cycle_exit.enabled=false`로 즉시 legacy 복귀
7. **cycle_timing**: check_exit trigger 후 MARKET fill까지 평균 < 2초

6/7 이상 GO. #1, #3 모두 실패 시 STOP.

---

## 5. Design

### 5.1 핵심 변경: bot main loop에 check_exit 호출

**현재 (legacy bot)**:
- 봇은 매 cycle에서 signals.py `check_exit`를 **호출하지 않음**
- 거래소의 SL(STOP_MARKET) + TRAILING_STOP_MARKET에만 의존
- 가격 모니터링은 거래소 native

**F v2 (제안)**:
- 매 cycle에서 현재 close, best, ATR로 check_exit 호출
- Trail trigger 시 봇이 **MARKET close** (exchange.create_order(type='market', reduceOnly=True))
- SL(fractal)은 여전히 거래소 STOP_MARKET으로 유지 (crash safety)
- Pre-activation 구간: trail 평가 안 함 (check_exit 내부 gate)

### 5.2 구현 위치

**bot.py main loop (`run` 또는 `process_candles` 메서드)**:
```python
# 현재: _update_exchange_trail만 호출
# 변경: check_exit 먼저 호출 → trail_exit 감지 시 MARKET close

for pos in list(self.positions):
    # 1. best_price 업데이트 (intrabar high/low from recent candles)
    self._update_best_price(pos, candles)
    
    # 2. check_exit 호출 (BT-identical logic)
    cfg = self.config['strategy']
    f_v2 = cfg.get('f_v2_cycle_exit', {}).get('enabled', False)
    if f_v2:
        exit_result = self.signal.check_exit(
            direction=pos['direction'],
            entry_price=pos['entry_price'],
            best_price=pos['best_price'],
            current_high=candles[-1]['high'],
            current_low=candles[-1]['low'],
            current_close=candles[-1]['close'],
            sl_price=pos['sl_price'],
            atr_val=current_atr,
            bars_held=pos['bars_held'],
        )
        if exit_result and exit_result['reason'] == 'TRAIL_TP':
            # Bot-initiated MARKET close
            self._do_close(pos, reason='TRAIL_TP_V2', price=exit_result['exit_price'])
            continue  # skip other cycle logic for this pos
    
    # 3. 기존 거래소 trail 관리 (f_v2=False면)
    if not f_v2:
        self._update_exchange_trail(pos, cur_price, cur_atr)
```

### 5.3 SL은 check_exit에 맡기지 않음 (중요)

`check_exit`의 SL branch는 `low <= sl_price`로 intrabar touch 검사. 하지만:
- LIVE cycle은 15m 해상도 → intrabar low가 SL에 도달해도 다음 cycle까지 bot 모름
- 그 사이 거래소 STOP_MARKET이 자동 실행 → 안전
- 봇이 `check_exit` SL로 감지해도 그때는 이미 거래소 SL 체결됨 → 중복 close 위험

**→ F v2에서도 SL은 거래소 STOP_MARKET 유지**. `check_exit` 중 trail branch만 봇이 실행.

### 5.4 Activation Gate (F v1과 호환)

`check_exit` 내부에 `if best_pnl > activation_pct` 가드 이미 있음. F v2에서는:
- **Pre-activation**: check_exit 호출해도 trail branch 실행 안 됨 → 봇은 아무 action 없음
- **Post-activation**: trail 평가 → 필요시 MARKET close

즉 F v2는 F v1의 "activation gate" 기능을 자동으로 포함.

### 5.5 Config

```yaml
strategy:
  # ... existing ...
  f_v2_cycle_exit:
    enabled: false              # 구현 + paper test 후 true 전환
    # Activation 후 매 cycle check_exit 호출 + trail trigger 시 MARKET close
    # SL은 거래소 STOP_MARKET 유지 (crash protection)
    # TRAILING_STOP_MARKET은 배치 안 함 (F v1과 동일 policy)
```

### 5.6 F v1과의 관계

F v2 enabled → **자동으로 TRAILING_STOP_MARKET 배치 생략** (F v1과 같은 동작). 코드 재사용:
```python
# _exchange_open의 TRAILING 배치 조건:
skip_trailing = (
    cfg.get('activation_gated_trail', {}).get('enabled', False)
    or cfg.get('f_v2_cycle_exit', {}).get('enabled', False)
)
```

---

## 6. Implementation Plan

### Phase 1: BT 검증 (이미 classic BT로 완료)
- Classic BT = F v2 semantic의 정확한 시뮬 (같은 check_exit 호출)
- 결과: +170% (333일), +3.60% (Canary 2일 구간), +10.16% (27-trade 10일 구간)
- **별도 Phase 1 BT 실행 불필요**

### Phase 2: 코드 구현 (2~3일)
1. `bot.py` config load (`f_v2_cycle_exit`)
2. `_exchange_open` 조건 확장 (F v1 OR F v2 시 TRAILING skip)
3. Main loop (`process_candles` 또는 `_do_cycle`)에서 check_exit 호출 추가
4. 신규 helper: `self._update_best_price_from_candles(pos, candles)` — intrabar high/low 추적
5. `_do_close(pos, reason, price=None)` 메서드 확인/수정 — MARKET close 지원
6. Pytest 신규 (test_f_v2_cycle_exit.py):
   - Pre-activation: check_exit 호출하지만 trail 비활성 → no MARKET close
   - Post-activation + drawdown 발생: check_exit trail → MARKET close
   - Best_price 업데이트: intrabar high/low 반영
   - SL은 여전히 거래소 STOP_MARKET에서 trigger
   - Rollback: enabled=false 시 legacy 동작

### Phase 3: Paper trading 검증 (1일)
- Canary 같은 기간(04-22~24) 8 trades에 F v2 적용 시뮬 (post-facto)
- Gap이 ≤ ±2pp인지 확인
- 통과 시 Phase 4 진행

### Phase 4: LIVE Canary (14일)
- config enabled=true + 봇 재시작
- 2 positions 후 동일 비교 분석 (이번에 사용한 canary_comparison_*.py 재활용)
- GO 조건 평가 → 전면 전환 or 롤백

---

## 7. Non-Goals

- BingX API 변경 요청
- Trail 로직 자체 변경 (check_exit 의도 100% 유지)
- Emergency SL / Timeout 로직 변경

---

## 8. Rollback

- Config `f_v2_cycle_exit.enabled: false` + 봇 재시작 → legacy TRAILING_STOP_MARKET 복귀
- 현 포지션 영향 없음 (다음 cycle에서 legacy 관리)

---

## 9. Risks

| 리스크 | 완화 |
|---|---|
| 15m cycle 해상도 → intrabar TRAIL 타이밍 지연 | Acceptable (BT도 bar close 해상도) |
| MARKET close slippage | H3 검증 + BUG#65 측정 정확도 활용 |
| 봇 다운 시 trail 보호 없음 | Fractal SL 거래소 유지 (crash safety) |
| cycle 간 가격 급변 → MARKET 체결 예상보다 벗어남 | slippage cap 알림 (> 0.3% 시 warning) |
| check_exit SL branch 오작동 → 중복 close | check_exit 호출 시 SL branch 결과 무시 (trail만 처리) |

---

## 10. Reference

- `results/canary_comparison_20260424_*.json` — F v1 실패 근거 (8 trades)
- `bingx_rl_trading_bot/scripts/production/c1_breakout/signals.py:159` — BT check_exit trail
- `bingx_rl_trading_bot/scripts/production/c1_breakout/bot.py:1066` — _calc_trail_trigger_price (F v2에서 사용 안 함, baton 폐기)
- `docs/01-plan/features/activation_gated_trail.plan.md` — F v1 plan (STOP)
- `memory/slippage_diagnosis_20260422.md` — 갭 원인 분석 이력

---

## 11. Relationship with Other PDCAs

| PDCA | 관계 |
|---|---|
| `activation_gated_trail` (F v1) | **Superseded** — F v2가 pre-activation skip + trail 모두 포함 |
| `pre_activation_baton` (E) | **기각 확정** — 반대 방향 |
| `progressive_trail` | 독립. check_exit가 progressive K 이미 사용 → 자동 호환 |
| `slippage_diagnosis` | 근거. Exit slip 측정값(0.007~0.058%)이 F v2 slippage_bound 목표의 baseline |
| `intrabar_parity` | 미래 보완. BT를 5m sub-bar로 개선하면 F v2와 추가 일치 |

---

## 12. 주요 변경 요약 (F v1 → F v2)

| 항목 | F v1 (STOP) | F v2 |
|---|---|---|
| Pre-activation TRAILING 배치 | 제거 | 제거 (동일) |
| Post-activation trail | baton STOP_MARKET (fixed trigger) | **봇 cycle check_exit + MARKET close** |
| 가격 회귀 시 exit | ❌ (STOP이 entry보다 아래) | ✅ (drawdown > trail_dist 감지) |
| 15m cycle 해상도 | Acceptable | Acceptable (BT도 bar close) |
| BT parity | 구조적 불가 | **100% 일치 기대** |
