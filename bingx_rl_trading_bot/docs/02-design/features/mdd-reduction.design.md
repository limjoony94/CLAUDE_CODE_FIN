# Design: MDD Reduction v1.40.0 — Combo2 (Equity Curve + Correlation-Aware)

> **Feature**: mdd-reduction | **Version**: v1.40.0 | **Date**: 2026-03-02

---

## 변경 파일 (6개)

| # | 파일 | 변경 내용 |
|---|------|----------|
| 1 | `state.py` | `equity_curve_tracker` 기본값 추가 |
| 2 | `models.py` | BotState 타입 `equity_curve_tracker` 추가 |
| 3 | `bot.py` | 3개 함수 + `_process_entry_signal` 통합 |
| 4 | `position_open.py` | `equity_curve_scale` 파라미터 |
| 5 | `position_close.py` | 거래 종료 시 equity tracker 업데이트 |
| 6 | `config.yaml` | `equity_curve_trading` + `correlation_aware_entry` 섹션 |

---

## 상세 설계

### 1. state.py — `_create_default_state()`
```python
'equity_curve_tracker': {
    'long_cum_pnls': [],   # cumulative PnL sequence after each LONG trade close
    'short_cum_pnls': [],  # cumulative PnL sequence after each SHORT trade close
}
```
**Per-direction 추적**: LONG/SHORT 각각 독립적으로 에퀴티 커브 평가.
LONG이 부진해도 SHORT은 정상 사이즈 유지 (역도 마찬가지).

### 2. models.py — BotState
`rolling_wr_tracker` 아래에 `equity_curve_tracker` dict 타입 추가.

### 3. bot.py — 3개 새 함수

**`_check_equity_curve_sizing(state, config, direction) -> float`**
- `risk.equity_curve_trading.enabled` false → 1.0 반환
- direction별 cum_pnls 조회 (long_cum_pnls / short_cum_pnls)
- len(cum_pnls) < ema_trades → 1.0 (데이터 부족)
- SMA(cum_pnls[-ema_trades:]) 계산
- current cum_pnl (cum_pnls[-1]) < SMA → size_mult (0.5), 아니면 1.0
- **LONG/SHORT 독립 평가**: LONG 진입 시 LONG 커브만, SHORT 진입 시 SHORT 커브만 확인

**`_check_correlation_aware_entry(state, config, signal_result, df) -> bool`**
- `risk.correlation_aware_entry.enabled` false → False
- positions < 2 → False
- same_dir_count / len(positions) >= dir_pct_threshold AND is_counter_regime → True (BLOCK)
- regime 판정: 기존 regime_sizing의 EMA(20)/lookback(5) 재사용

**`_update_equity_curve_tracker(state, total_equity)`**
- `state['equity_curve_tracker']['trade_equities'].append(total_equity)`
- 최대 100개 유지

**`_process_entry_signal()` 통합 위치**:
```
momentum_guard → loss_burst_brake → correlation_aware_entry(NEW) → aggregate_risk_cap → adaptive_leverage → equity_curve_sizing(NEW) → open_position
```
- correlation guard는 aggregate_risk_cap 전에 (빠른 filter)
- equity_curve sizing은 open_position 직전 (size multiplier)

### 4. position_open.py
`get_position_size()`에 `equity_curve_scale: float = 1.0` 파라미터:
```python
per_slot_equity = total_equity * size_pct / max_positions * mdd_scale * regime_scale * equity_curve_scale
```
`open_position()`에 `equity_curve_scale` 전달.

### 5. position_close.py
`record_closed_position()` 끝에서 equity tracker 업데이트:
- `fetch_balance_cached` → `total_equity` → `_update_equity_curve_tracker(state, total_equity)`
- bot.py에서 import하여 호출 (circular import 방지: bot→position_close 방향만)

실제로는 bot.py의 maintenance window에서 이미 equity를 조회하고 있으므로,
**bot.py에서 거래 종료 감지 후 직접 호출**하는 것이 더 깔끔.
→ `_check_and_record_trade_exit()` 또는 기존 `_record_loss_for_burst_brake` 호출부 근처에서 equity tracker 업데이트.

### 6. config.yaml
```yaml
risk:
  equity_curve_trading:
    enabled: true
    ema_trades: 30
    size_mult: 0.5
  correlation_aware_entry:
    enabled: true
    dir_pct_threshold: 0.70
```
