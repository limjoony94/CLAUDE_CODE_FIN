# Plan: F v4 — SL도 Cycle check_exit으로 이동 (DRAFT)

> **Feature**: f_v4_cycle_sl
> **Date**: 2026-04-25
> **Phase**: Plan (DRAFT — 실구현 보류, F v2 결과 + 추가 검증 후 활성화 판단)
> **Scope**: 거래소 STOP_MARKET (fractal SL) → 봇 cycle bar-close 평가 + MARKET close. 거래소엔 emergency 3% backup만.
> **기반**: stop_hunt_analysis_20260425 — SL hit 후 97.1% 회복, 회복/SL 비율 0.82

---

## 1. Background

### Stop Hunt 검증 (2026-04-25, 34 SL/Trail trades)

| 지표 | 값 |
|---|---|
| Wick exits (회복 > 50% sl_pct) | **23/34 (67.6%)** |
| Positive recovery (any) | **33/34 (97.1%)** |
| 평균 회복폭 | +0.615% |
| 평균 sl_pct (1x) | 0.748% |
| **회복/SL 비율** | **0.82** |

**SL/Trail hit 후 평균적으로 SL 거리의 82%만큼 가격이 favorable 방향으로 회복**.

### 메커니즘

```
BT 로직:
  매 bar close에서 check_exit (low ≤ sl_price)
  → 정확한 bar close 시점에 평가
  → wick recovery 시 다음 cycle에서 회복된 close 평가

LIVE 현재 로직:
  거래소 STOP_MARKET = intrabar tick trigger
  → wick이 sl_price에 잠시 닿으면 즉시 fire
  → bar close에 가격이 회복했어도 이미 exit
```

**핵심**: BT 데이터(15m bar low)와 거래소 internal tick 사이 ε-차이가 누적 → 같은 wick이 LIVE에선 trigger, BT에선 hold.

### F v2와 F v4 관계

- F v2: TRAIL_TP만 cycle check_exit으로 이동 (구현 완료, Canary 진행)
- **F v4: SL/Emergency도 cycle check_exit으로 이동** (제안)
- F v4 + F v2 조합 시 모든 exit이 cycle 기반 → BT 100% identity

---

## 2. Goal

LIVE의 SL/Trail wick exit을 회피하여 BT의 trail TP까지 자연스럽게 가는 trade 비율 증가.

**정량 목표**:
- LIVE EXCHANGE_SL/TRAIL 비율 **30% → 18% 이내** (BT 17% 수준)
- LIVE per-trade PnL3x **-0.71% → +0.3%** (BT +0.68% 대비 50% 회복)

---

## 3. Hypotheses

| H | 가설 | 검증 방법 |
|---|---|---|
| **H1** | 봇 cycle bar-close SL 평가 시 wick exit 회피 → SL trigger 빈도 감소 | Canary trades에서 SL hit 비율 측정 (BT 17% 목표) |
| **H2** | Cycle 사이 (15m) 가격 급락 시 emergency 3%로 손실 cap → 평균 큰 손실 빈도 증가 | Emergency SL trigger 빈도 측정 (목표 ≤ 5%) |
| **H3** | SL latency 15m 동안 가격 회복 시 차이 trade의 win 전환 | SL이 cycle에 평가될 때까지 가격이 회복한 trade 수 |
| **H4** | F v2 + F v4 조합으로 LIVE 갭 -1.40pp/trade → -0.3pp 이내 | A/B 비교 (F v2-only vs F v2+F v4) |

---

## 4. Success Criteria (GO 조건 8개)

1. **sl_hit_reduction**: LIVE EXCHANGE_SL 비율 ≤ 20% (현재 30%)
2. **emergency_rate**: Emergency SL 발동 ≤ 5% (cycle 동안 큰 급락 보호 검증)
3. **per_trade_pnl**: LIVE per-trade PnL3x ≥ +0.3% (BT의 50%+)
4. **catastrophic_safety**: 최대 손실 trade ≤ -3.5% (emergency cap + slippage 0.5%)
5. **rollback_ready**: config flag로 즉시 거래소 fractal SL 복귀
6. **pytest_coverage**: SL bar-close 평가 + emergency backup pytest
7. **BT_identity**: BT 변화 없음 (LIVE-only execution)
8. **canary_validation**: 4-6 trades 후 GO 조건 평가

7/8 이상 GO. #2 (emergency rate) 또는 #4 (catastrophic) 실패 시 무조건 STOP.

---

## 5. Design

### 5.1 현재 SL 흐름

```python
# _exchange_open
self.exchange.create_order('STOP_MARKET', sl_side, qty,
    stopPrice=fractal_sl_price, reduceOnly=True)
```
거래소가 intrabar tick으로 trigger → 즉시 MARKET fill.

### 5.2 F v4 변경안

```python
# config
strategy:
  f_v4_cycle_sl:
    enabled: false
    emergency_pct: 3.0     # 거래소 backup STOP (fractal보다 멀리)

# _exchange_open
if cfg['f_v4_cycle_sl'].get('enabled'):
    # 거래소엔 emergency 3% backup STOP만
    emg_pct = cfg['f_v4_cycle_sl']['emergency_pct'] / 100
    emg_sl = entry × (1 - emg_pct) if direction == 'LONG' else entry × (1 + emg_pct)
    self.exchange.create_order('STOP_MARKET', sl_side, qty,
        stopPrice=emg_sl, reduceOnly=True)
    pos['fractal_sl'] = sl_price  # 봇이 internal로 추적
else:
    # legacy: 거래소가 fractal SL trigger
    self.exchange.create_order('STOP_MARKET', sl_side, qty,
        stopPrice=sl_price, reduceOnly=True)

# process_candles 내부 (이미 check_exit 호출 중)
# check_exit가 SL branch에서 'SL' return → _do_close MARKET
# F v4에서는 fractal_sl로 평가 (이미 sl_price를 받음)
# 차이: 거래소엔 fractal_sl이 없으므로 봇이 MARKET close 시 거래소 STOP과 충돌 없음
```

### 5.3 충돌 방지

봇이 cycle에서 SL trigger → MARKET close 발주 → 거래소 backup STOP은 trigger 안 됨 (가격이 emergency 3%까지 안 가서).

만약 가격이 emergency 3%까지 급락 → 거래소 STOP fire → 봇 다음 cycle ghost detection.

### 5.4 봇 다운 안전성

| 시나리오 | F v4 동작 |
|---|---|
| 봇 정상 + 가격 fractal SL touch | 봇이 cycle에서 SL → MARKET close (BT identity) |
| 봇 정상 + 가격 fractal SL 회복 | 봇이 hold → trail까지 자연 진행 (✅ wick 회피) |
| **봇 다운 + 가격 fractal SL touch** | 거래소 STOP은 emergency 3%, 안 fire → **3%까지 손실 위험** |
| 봇 다운 + 가격 emergency 3% touch | 거래소 STOP fire (catastrophic loss cap) |

**리스크**: 봇 다운 + 작은 가격 하락 (fractal SL ~ 1% 수준) → emergency 3%까지 보호 없음.

**완화**:
- 봇 health monitor 강화 (uptime watchdog)
- Cycle stale > 30min 시 emergency SL 자동 활성화 (fractal level로 이동)

### 5.5 Config

```yaml
strategy:
  f_v4_cycle_sl:
    enabled: false
    emergency_pct: 3.0           # 거래소 backup STOP
    bot_down_fallback_min: 30    # 봇 stale 시 fractal로 STOP 이동
```

---

## 6. Implementation Plan

### Phase 1: 추가 검증 (이미 부분 완료)
- ✅ Stop hunt 데이터 수집 (97.1% recovery)
- 🔬 BT 5m sub-bar 모델 (intrabar_parity 진행 중) — F v4 효과 BT-side 검증 보완

### Phase 2: 코드 구현 (3~4일)
1. bot.py `_exchange_open`: F v4 분기 추가 (emergency STOP만)
2. `process_candles`: SL branch 평가는 이미 호출 중, F v4에서 _do_close 자연 발동
3. `_update_exchange_trail`: emergency STOP 점검 추가
4. Bot health watchdog (별도 thread? 또는 cycle 내 검사) — fallback 활성화 로직
5. Pytest (test_f_v4_cycle_sl.py, ~12 케이스)

### Phase 3: Paper test (1~2일)
- testnet 또는 dry-run에서 가상 SL trigger
- Bot down → emergency STOP fire 검증
- Cycle stall → fractal STOP 자동 이동 검증

### Phase 4: Canary (10~14일)
- F v2 Canary 결과 종합 후 결정
- Config enable + 봇 재시작
- 4-6 trades 후 GO 평가

---

## 7. Non-Goals

- F v2 (cycle trail) 변경 — F v4와 독립
- BingX API 변경 요청
- 거래소 internal trigger 메커니즘 변경 (불가)

---

## 8. Risks

| 리스크 | 완화 |
|---|---|
| 봇 다운 시 emergency 3%까지 손실 가능 | uptime watchdog + fallback STOP 이동 |
| Cycle 사이 (15m) 큰 급락 → fractal SL bypass | emergency 3% 상한 (catastrophic cap) |
| Cycle stall (stale candle 등) 시 SL 평가 늦음 | Stall detect → emergency activate |
| F v4 + F v2 + 거래소 동시 close 충돌 | reduceOnly 검증 + retry 로직 |
| 거래소 emergency STOP의 idempotency | sl_order_id 추적, 중복 cancel 안전 |

---

## 9. Rollback

- config `f_v4_cycle_sl.enabled: false` → 거래소 fractal SL 복귀
- 봇 재시작 시 모든 position의 SL을 fractal level로 redepoy

---

## 10. Reference

- `results/stop_hunt_20260425_*.json` (97.1% recovery 데이터)
- `bingx_rl_trading_bot/scripts/production/c1_breakout/signals.py:117` check_exit SL branch
- `bingx_rl_trading_bot/scripts/production/c1_breakout/bot.py:_exchange_open`
- `memory/post_fix_legacy_gap_20260424.md` — sequencing floor concept
- `intrabar_parity` PDCA — BT side 보완

---

## 11. F v2 / F v3 / F v4 연계

| Feature | Scope | Status |
|---|---|---|
| F v1 (activation_gated) | TRAILING entry skip + baton | ❌ 기각 |
| **F v2 (cycle_exit)** | Trail → cycle check_exit + MARKET | 🔬 Canary (1/2) |
| **F v3 (limit_close)** | Trail MARKET → LIMIT + timeout | 📋 Plan draft |
| **F v4 (cycle_sl)** | SL → cycle check_exit + MARKET | 📋 본 plan |

전체 통합 시: 모든 exit (Trail/SL/Emergency)가 봇 cycle 기반 → **BT 100% LIVE 재현 가능 이론**.
