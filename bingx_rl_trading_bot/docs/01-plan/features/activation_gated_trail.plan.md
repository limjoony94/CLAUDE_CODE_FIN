# Plan: Activation-Gated Trail Placement (F 옵션)

> **Feature**: activation_gated_trail
> **Date**: 2026-04-22
> **Phase**: Plan
> **Scope**: Pre-activation 구간에 거래소 TRAILING/baton 배치 자체를 제거. best_pnl ≥ activation_pct 도달 시점에서만 baton STOP_MARKET 최초 배치.
> **기반**: E option (`pre_activation_baton`) Phase 1 BT 결과로 방향 재조정

---

## 1. Background

### E 옵션 Phase 1 결과 요약

`baton_only_backtest_20260422.py` + `baton_k_pre_sweep_20260422.py` 결과:

| K_pre | 27-trade gap vs LIVE | 333일 full PnL1x |
|---|---|---|
| 2.5 (default) | -4.85pp (근접) | **-190% (파산)** |
| 5.0+ | -1.49pp (완벽 매칭) | -181% (파산) |

**결론**: Pre-activation 구간에 baton STOP_MARKET를 "항상" 배치하면, K를 어떻게 조정해도 333일 전략 기반 PnL이 완전 소멸.

### 구조적 발견

**Classic BT (+170%)**가 실제로 어떻게 작동하는지:
- `cur_pnl < activation_pct` 구간에서는 **trail 평가 자체를 건너뜀** (`signals.py check_exit`)
- 즉 pre-activation 시기에는 trail trigger 없음 — **fractal SL + emergency SL만 보호**
- 이익이 `activation_pct` 이상 돌파하면 trail 활성화

**Current LIVE bot**:
- Entry 직후 즉시 TRAILING_STOP_MARKET 배치 (`bot.py:1020~1040`)
- BingX의 `activatePrice`가 설정되지만 (BUG#62 fix), 실제 BingX 동작상 pre-activation 구간에도 tick 단위 감시가 일어나는 듯
- 결과: 작은 반전에도 pre-activation 중 trigger → LIVE PnL -16%

**E 단순 적용 (baton 전면)**: BT와 LIVE 동기화되지만 둘 다 망함 (pre-activation trail이 원래 나쁜 행동).

**F 옵션**: LIVE도 BT처럼 **pre-activation 구간엔 trail 없이** 운영. Classic BT +170% 로직 1:1 재현.

---

## 2. Goal

**LIVE 결과를 classic BT 수준으로 복원**:
- 27-trade 구간 LIVE PnL3x: -16.09% → **≥ +10% 근접**
- 333일 baseline 보존: +170% additive 유지
- BT-LIVE 갭 3x 기준 -25pp → ≤ -5pp

---

## 3. Hypotheses

| H | 가설 | 검증 방법 |
|---|---|---|
| **H1** | Entry 직후 TRAILING 배치 제거 시 BingX pre-activation tick trigger 완전 해소 | LIVE 새 sample의 exit 분포 변화 |
| **H2** | `activation_gated` 적용 후 LIVE PnL이 classic BT와 ±5pp 이내 근접 | 30일 LIVE 결과 vs BT |
| **H3** | Pre-activation 구간의 fractal SL만 보호로도 emergency SL (3%) 발동 빈도 변화 미미 | SL hit 분포 비교 |
| **H4** | Activation 도달 시점의 baton STOP_MARKET 배치가 TRAILING_STOP_MARKET과 거의 동일 효과 | Exit 체결가 비교 (BUG#65 slip 측정) |

---

## 4. Success Criteria (GO 조건 7개)

1. **gap_reduction**: LIVE 30일 샘플 PnL3x가 BT와 ±5pp 이내
2. **baseline_preservation**: 333일 full BT (F logic 적용) +170% 이상 (classic과 동일)
3. **wf_pass**: WF 5-fold 전부 양수 (classic과 동일해야 정상)
4. **sl_hit_stability**: SL hit 비율이 classic BT 대비 ±20% 이내
5. **emergency_rate**: Emergency SL 발동 빈도 ≤ 1% (sample 기반)
6. **rollback_ready**: Config flag `activation_gated_trail.enabled=false`로 즉시 원복
7. **crash_safety**: 봇 다운 상황에서도 fractal SL STOP_MARKET이 거래소 active → 보호 유효

6/7 이상 GO. #2 (baseline) 실패 시 무조건 STOP.

---

## 5. Design

### 5.1 현재 로직 (bot.py:_exchange_open line 1020~1046)

```python
# Entry 성공 후 즉시:
# 1. SL (STOP_MARKET) 배치 — 유지
# 2. TRAILING_STOP_MARKET 배치 ← 제거/조건부 대상
self.exchange.create_order(
    symbol, 'TRAILING_STOP_MARKET', tp_side, filled_qty,
    params={'activatePrice': activate,
            'trailingPercent': callback,
            'reduceOnly': True})
```

### 5.2 F 옵션 변경

```python
# Entry 성공 후:
# 1. SL (STOP_MARKET) 배치 — 유지
# 2. TRAILING_STOP_MARKET 배치 ← 삭제 (if activation_gated_trail.enabled)
if not config['strategy'].get('activation_gated_trail', {}).get('enabled', False):
    # 기존 legacy TRAILING 배치 (fallback)
    create_order(TRAILING_STOP_MARKET, ...)
# else: do nothing — fractal SL만 보호
```

### 5.3 Activation 도달 시 배치 (`_update_exchange_trail` 수정)

현재 `_update_exchange_trail`은 cycle마다 best_pnl 재평가, activation 도달 후 baton 갱신. 변경:

```python
# best_pnl > activation_pct AND trail_order_id == '':
#   → 최초 baton STOP_MARKET 배치 (TRAILING_STOP_MARKET 대신)
if best_pnl > activation_pct and not pos.get('trail_order_id'):
    exact_trigger = _calc_trail_trigger_price(pos, cur_atr)
    if exact_trigger:
        order = exchange.create_order(
            symbol, 'STOP_MARKET', tp_side, qty,
            params={'stopPrice': exact_trigger, 'reduceOnly': True}
        )
        pos['trail_order_id'] = order['id']
        pos['trail_type'] = 'STOP_MARKET'

# 이후 cycle: 기존 baton 갱신 로직 재사용
```

### 5.4 Config

```yaml
strategy:
  # ... existing ...
  activation_gated_trail:
    enabled: false       # 기본 false (opt-in)
    # 활성화 시 Entry 시점 TRAILING 배치 생략
    # Activation 도달 cycle에 baton STOP_MARKET 최초 배치
```

### 5.5 State 복원 (restart 시)

- 기존 orphan/ghost resolver는 trail_type으로 TRAILING vs STOP_MARKET 구분 (BUG#50)
- F 옵션 적용 후 재시작: pos['trail_order_id']가 비어있으면 아직 pre-activation, 그대로 유지
- `_resolve_orphan_sl`: fractal SL만 찾으면 정상 (trail은 optional)

---

## 6. Implementation Plan

### Phase 1: BT 확인 (이미 완료)
- Classic BT가 F 옵션의 정확한 BT 시뮬 — 추가 BT 불필요
- 결과: +170% (333일), +15.72% (27 구간) 이미 확보

### Phase 2: 코드 구현 (2~3일)
1. `bot.py` config load (`activation_gated_trail`)
2. `_exchange_open`에서 TRAILING 배치 조건부 건너뛰기
3. `_update_exchange_trail`에 activation 도달 시 baton 최초 배치 추가
4. Pytest 신규 (test_activation_gated.py):
   - Entry 직후 trail order 없음 검증
   - Activation 도달 시 baton 배치 검증
   - Re-start 시 state 복원 검증
   - Orphan adoption 호환 검증

### Phase 3: Paper trading (1~2일, selective)
- Config `enabled=true` 후 1~2 positions만 dry-run (작은 qty)
- 로그 검증: pre-activation 구간 TRAILING 없음 확인
- Activation 도달 시 STOP_MARKET 배치 확인

### Phase 4: LIVE A/B 배포 (30일)
1. Canary: 7일간 enabled=true, 하루 ≤ 2 positions
2. 전면 적용: enabled=true 고정
3. 30일 후 성과 비교 (vs legacy TRAILING 기간)
4. Rollback 조건: PnL3x 주간 ≤ -5pp 3주 연속 시

**총 5~7일 (Phase 2 구현), 30일 평가 기간.**

---

## 7. Non-Goals

- BingX TRAILING_STOP_MARKET API 변경 요청
- Pre-activation 구간의 다른 exit 메커니즘 추가 (time-based, volatility-based 등)
- Classic BT 로직 변경

---

## 8. Rollback

- `config.yaml` → `activation_gated_trail.enabled: false`
- 봇 재시작 → entry 직후 TRAILING 배치 복귀 (legacy 동작)
- 기존 positions는 다음 cycle에서 정상 관리

---

## 9. Risks

| 리스크 | 완화 |
|---|---|
| Pre-activation 구간에서 큰 반전 발생 시 fractal SL까지 기다림 | Emergency SL 3% 상한 유지 (intrabar touch로 즉시 발동) |
| Fractal SL이 max_sl_atr=3.3 cap이어서 실제 반전 시 3%+ 손실 가능 | `sl_max_pct=3.0` 상한도 적용됨 (이미 config에 있음) |
| BingX가 TRAILING 없어도 우리가 빼먹은 보호 존재? | 검증: SL + Emergency + max_hold만 있으면 충분 (classic BT 결과) |
| Activation 도달 cycle 직전에 반전 발생 → baton 배치 전 손실 | Cycle 15m 격차. 약간의 LIVE-BT timing 차이 가능 but 미미 |
| 봇 다운 중 activation 도달 + 반전 | 봇 복원 시 orphan adoption + fractal SL 유지 |

---

## 10. Reference

- `results/baton_only_backtest_20260422_031115.json` — E 단순 적용 파산 확인
- `results/baton_k_pre_sweep_20260422_*.json` — K_pre 전 범위 파산
- `results/dd_comparison_20260421_235028.json` — classic BT +10.16% vs LIVE -16.09%
- `bingx_rl_trading_bot/scripts/production/c1_breakout/bot.py:1020` — 현 TRAILING 배치 코드
- `bingx_rl_trading_bot/scripts/production/c1_breakout/signals.py:159` — check_exit trail 로직 (activation gate 존재)
- `claudedocs/BACKTEST_LIVE_PARITY.md` — #21 (pre-activation TRAILING 구조적 한계)

---

## 11. Relationship with Other PDCAs

| PDCA | 관계 |
|---|---|
| `slippage_diagnosis` | F의 근거 (execution 아닌 timing이 주원인 확정) |
| `pre_activation_baton` (E) | **Superseded** — E는 baton 방향, F는 반대로 "배치 제거" 방향 |
| `progressive_trail` | 독립. Activation 이후 로직 변경 → F 도달 후 적용 가능 |
| `intrabar_parity` (Track A) | BT 측 개선, F와 상호 보완 가능 |
| B' (callback 축소) | F가 callback을 제거 → B' 완전 폐기 |
