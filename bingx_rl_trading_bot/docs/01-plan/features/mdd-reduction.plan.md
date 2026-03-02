# Plan: MDD Reduction — Equity Curve Trading + Correlation-Aware Entry (v1.40.0)

> **Feature**: mdd-reduction
> **Date**: 2026-03-02
> **Phase**: Plan
> **Verdict**: GO (Combo2 — MDD -47.8%, PnL -3.9%, PnL/MDD +84.1%)

---

## 1. Background

v1.39.0 (Adaptive Leverage) 이후 MDD 추가 감소 기법 탐색.
`mdd_reduction_study.py` — 7가설 45시나리오 6-Phase 연구 완료.

### Baseline (v1.39.0 production)
| 지표 | 값 |
|------|-----|
| Trades | 826 |
| WR | 74.7% |
| PnL | +106.8% |
| MDD | 6.5% |
| PnL/MDD | 16.45 |

### 연구 결과 요약
| 가설 | Verdict | 근거 |
|------|---------|------|
| H1: Equity Curve Trading | **GO** | MDD -32.8%, PnL -2.5%, WF 3/3 PASS |
| H2: Dynamic N | GO | MDD -10.6%, PnL -0.5%, WF 3/3 PASS |
| H3: Volatility Pause | STOP | PnL 손실 > MDD 감소 |
| H4: Intraday Loss Limit | STOP | 현재 수준에서 트리거 안됨 |
| H5: Trailing Equity Stop | STOP | 거래량 -57%, 과도한 제한 |
| H6: Correlation-Aware Entry | **GO** | MDD -10.6%, PnL +9.8%, WF 3/3 PASS |
| H7: Tighter MDD Sizing | GO | MDD -10.6%, PnL +8.4%, WF 3/3 PASS |

### Combo2 선택 (H1 + H6)
| 지표 | Baseline | Combo2 | 변화 |
|------|----------|--------|------|
| PnL | 106.8% | 102.6% | -3.9% |
| MDD | 6.5% | **3.4%** | **-47.8%** |
| PnL/MDD | 16.45 | **30.29** | **+84.1%** |
| WF OOS | +11.4/+37.8/+37.2 | +23.0/+33.6/+23.7 | 3/3 PASS |

선택 이유:
- MDD 감소 폭(47.8%) >> PnL 감소 폭(3.9%) — 사용자 기준 충족
- PnL/MDD 30.29 — baseline 대비 **1.84배**
- WF OOS 안정적 (std 4.6 vs baseline 12.3)
- 두 기법이 독립적 메커니즘으로 동작 (상호 보완)

---

## 2. Goal

Combo2 (H1_EqCurve_half_ema30 + H6_CorrAware_dir70)를 production에 적용하여:
- MDD를 ~3.4% 수준으로 감소 (현 6.5%에서 -48%)
- PnL 손실을 -4% 이내로 제한
- config-driven enable/disable로 즉시 rollback 가능

---

## 3. Technique Details

### H1: Equity Curve Trading (half_ema30)
**원리**: 에퀴티 커브가 자체 EMA(30 trades) 아래일 때, 전략이 "기대 이하" 상태 → 포지션 사이즈를 절반으로 축소.

```
equity < EMA(equity, 30 trades) → size_mult × 0.5
equity >= EMA → size_mult × 1.0 (정상)
```

- **EMA 대상**: 포트폴리오 에퀴티 (거래 완료 시점 기록)
- **Period**: 30 trades (최근 30거래의 에퀴티 평균)
- **Action**: reduce (half), skip 아님 — 거래는 유지하되 리스크 축소
- **효과**: DD 시 자동으로 노출 감소, 회복 시 자동 복귀

### H6: Correlation-Aware Entry (dir70)
**원리**: 현재 포지션 중 같은 방향 비율이 70% 이상이고, 새 진입이 counter-regime이면 스킵.

```
if len(positions) >= 2:
    same_dir_ratio = count(same_dir) / total_positions
    if same_dir_ratio >= 0.70 AND is_counter_regime:
        SKIP entry
```

- **방향 집중도 기준**: 70% (예: 7/9 LONG일 때 counter-regime SHORT 차단)
- **Counter-regime 판정**: EMA(20) slope > 0 → SHORT은 counter, LONG은 with
- **효과**: 이미 한 방향에 쏠린 상태에서 역추세 진입 방지 → correlated loss 감소
- **기존 direction_cap=7과 차이**: dir_cap은 절대 수, H6는 비율+regime 조건

---

## 4. Implementation Plan

### Change 1: `config/pattern_5m_config.yaml`
```yaml
risk:
  equity_curve_trading:          # v1.40.0: Equity Curve Trading (Combo2-H1)
    enabled: true
    ema_trades: 30               # EMA lookback in completed trades
    size_mult: 0.5               # size multiplier when equity < EMA
  correlation_aware_entry:       # v1.40.0: Correlation-Aware Entry (Combo2-H6)
    enabled: true
    dir_pct_threshold: 0.70      # same-direction ratio threshold
```

### Change 2: `scripts/production/pattern_5m/state.py`
`_create_default_state()`에 추가:
```python
'equity_curve_tracker': {
    'trade_equities': [],        # equity after each trade close
}
```

### Change 3: `scripts/production/pattern_5m/bot.py` (핵심)

**3a**: `_check_equity_curve_sizing()` — 새 함수 (~20줄)
```python
def _check_equity_curve_sizing(state, config) -> float:
    """Return size multiplier based on equity curve vs EMA.
    Returns 1.0 (normal) or config size_mult (e.g., 0.5) when equity < EMA.
    """
    ec_cfg = config.get('risk', {}).get('equity_curve_trading', {})
    if not ec_cfg.get('enabled', False):
        return 1.0

    ema_trades = ec_cfg.get('ema_trades', 30)
    size_mult = ec_cfg.get('size_mult', 0.5)

    tracker = state.get('equity_curve_tracker', {})
    equities = tracker.get('trade_equities', [])

    if len(equities) < ema_trades:
        return 1.0  # not enough data yet

    ema = sum(equities[-ema_trades:]) / ema_trades
    current_equity = equities[-1] if equities else 0

    if current_equity < ema:
        return size_mult
    return 1.0
```

**3b**: `_check_correlation_aware_entry()` — 새 함수 (~20줄)
```python
def _check_correlation_aware_entry(state, config, signal_result, df) -> bool:
    """Return True if entry should be BLOCKED (correlation guard).
    Block when: same-dir ratio >= threshold AND new entry is counter-regime.
    """
    ca_cfg = config.get('risk', {}).get('correlation_aware_entry', {})
    if not ca_cfg.get('enabled', False):
        return False

    positions = state.get('positions') or {}
    if len(positions) < 2:
        return False

    direction = signal_result  # 'LONG' or 'SHORT'
    dir_pct = ca_cfg.get('dir_pct_threshold', 0.70)

    same_dir = sum(1 for s in positions.values() if s.get('direction') == direction)
    ratio = same_dir / len(positions)

    if ratio < dir_pct:
        return False

    # Check if counter-regime
    regime_cfg = config.get('risk', {}).get('regime_sizing', {})
    ema_period = regime_cfg.get('ema_period', 20)
    lookback = regime_cfg.get('lookback', 5)

    if df is not None and len(df) >= ema_period + lookback:
        ema = df['close'].ewm(span=ema_period, adjust=False).mean().values
        slope = ema[-1] - ema[-1 - lookback]
        is_uptrend = slope > 0
        is_counter = (is_uptrend and direction == 'SHORT') or (not is_uptrend and direction == 'LONG')
        if is_counter:
            logger.info(f"Correlation guard: dir_ratio={ratio:.0%}, {direction} is counter → BLOCK")
            return True

    return False
```

**3c**: `_update_equity_curve_tracker()` — 거래 종료 시 에퀴티 기록
```python
def _update_equity_curve_tracker(state, total_equity):
    """Record current equity after trade close for equity curve trading."""
    tracker = state.setdefault('equity_curve_tracker', {'trade_equities': []})
    tracker['trade_equities'].append(total_equity)
    # Keep last 100 entries (3x+ of max EMA period)
    if len(tracker['trade_equities']) > 100:
        tracker['trade_equities'] = tracker['trade_equities'][-100:]
```

**3d**: `_process_entry_signal()` 통합 — 기존 guard 체인에 추가
```python
# After loss_burst_brake, before aggregate_risk_cap:
# v1.40.0: Correlation-aware entry guard
if _check_correlation_aware_entry(state, config, signal_result, df):
    return False
```

**3e**: `open_position()` 호출에 equity_curve sizing 반영
```python
# Before open_position call:
ec_mult = _check_equity_curve_sizing(state, config)
# ec_mult is passed as additional size multiplier (see Change 4)
```

### Change 4: `scripts/production/pattern_5m/position_open.py`

`get_position_size()`에 `equity_curve_scale` 파라미터 추가:
```python
def get_position_size(..., equity_curve_scale: float = 1.0):
    ...
    per_slot_equity = total_equity * size_pct / max_positions * mdd_scale * regime_scale * equity_curve_scale
```

`open_position()`에 `equity_curve_scale` 전달 + 슬롯에 저장:
```python
slot['equity_curve_scale'] = equity_curve_scale
```

### Change 5: `scripts/production/pattern_5m/position_close.py`

거래 종료 시 `_update_equity_curve_tracker()` 호출:
```python
# After recording trade in metrics:
_update_equity_curve_tracker(state, total_equity)
```
(`total_equity`는 이미 balance 조회 시 사용 가능)

### Change 6: `scripts/production/pattern_5m/models.py`
BotState 타입에 `equity_curve_tracker` 추가 (문서 목적).

---

## 5. Non-Changes

1. `enabled: false` → 기존 동작 100% 보존
2. Scanner / pattern selection: 변경 없음
3. Adaptive leverage: 변경 없음 (직교)
4. MDD sizing / regime sizing / aggregate risk cap: 변경 없음
5. Momentum guard / loss burst brake: 변경 없음
6. Per-pattern TP/SL: 변경 없음

---

## 6. Implementation Order

1. `state.py` — `equity_curve_tracker` 기본값
2. `models.py` — BotState 타입 확장
3. `bot.py` — `_check_equity_curve_sizing()` + `_check_correlation_aware_entry()` + `_update_equity_curve_tracker()` + `_process_entry_signal()` 통합
4. `position_open.py` — `equity_curve_scale` 파라미터
5. `position_close.py` — 거래 종료 시 tracker 업데이트
6. `config.yaml` — `equity_curve_trading` + `correlation_aware_entry` 섹션
7. Tests 실행 (1061+ 확인)

---

## 7. Rollback

```yaml
# 개별 비활성화
risk:
  equity_curve_trading:
    enabled: false
  correlation_aware_entry:
    enabled: false
```

개별적으로 비활성화 가능 — H1만, H6만, 또는 둘 다.

---

## 8. Verification

```bash
# 1. 기존 테스트
python -m pytest scripts/production/pattern_5m/tests/ -x -q

# 2. enabled: false 상태에서 regression 없음 확인
# 3. enabled: true 후 봇 재시작 → 정상 동작 확인
```

---

## 9. Research Reference

- **스크립트**: `scripts/analysis/mdd_reduction_study.py`
- **결과 JSON**: `results/mdd_reduction_study.json`
- **Combo2 IS**: PnL 102.6%, MDD 3.4%, PnL/MDD 30.29
- **Combo2 WF OOS**: F1 +23.0%, F2 +33.6%, F3 +23.7% (3/3 PASS)
- **MDD Duration**: 4,913 bars (baseline 12,981 — **-62%**)
- **Worst Daily**: -2.9% (baseline -4.2%)
