# v1.34.0 — 주기적 재스캔 + 7일 Holdout + MDD 동적 사이징

## Context

레짐 감지가 사실상 불가능함이 입증됨 (whipsaw 95.2%, 다음24h 예측 49.9%). 대안으로 3가지 방어 메커니즘을 **프로덕션 코드에 직접 구현**:

1. **Scanner 7일 Holdout**: 스캔 시 마지막 7일을 IS에서 제외, OOS 검증용으로 보유
2. **Bot 스캔 Staleness 경고**: `dynamic_patterns.json`의 `generated_at`이 90일 초과 시 WARNING
3. **MDD 기반 동적 사이징**: 드로다운 시 포지션 크기 자동 축소 (5%→100%, 20%→25%)

---

## Change 1: Scanner `--holdout-days` (7일 Holdout 검증)

### 파일: `scripts/scanner/pattern_scanner.py`

#### 1a. CLI 인자 추가
```python
parser.add_argument('--holdout-days', type=int, default=7,
                    help='마지막 N일을 holdout OOS로 보유 (default: 7, 0=비활성)')
```

#### 1b. `holdout_validate()` 함수 신규
```python
def holdout_validate(df_holdout, pattern_schedule, leverage, fee_pct):
    """holdout 기간에서 각 패턴의 WR Excess > 0 검증"""
    results = {}
    for pat_key, info in pattern_schedule.items():
        # backtest on holdout data
        trades = bt_signals(df_holdout, pat_key, info, ...)
        if len(trades) < 3:  # holdout 기간 거래 부족 → SKIP (제거하지 않음)
            results[pat_key] = {'status': 'SKIP', 'reason': 'insufficient_trades'}
            continue
        wr = sum(1 for t in trades if t > 0) / len(trades)
        rw_wr = info['sl_pct'] / (info['tp_pct'] + info['sl_pct'])
        wr_excess = wr - rw_wr
        results[pat_key] = {
            'status': 'PASS' if wr_excess > 0 else 'FAIL',
            'wr': wr, 'rw_wr': rw_wr, 'wr_excess': wr_excess,
            'trades': len(trades)
        }
    return results
```

#### 1c. `main()` 흐름 수정
```
기존: df → [optional --is-days slice] → scan → WF → output
수정: df → [optional --is-days slice] → holdout split → scan(df_is) → holdout_validate(df_holdout) → WF → output
```
- `holdout_bars = holdout_days * 288` (288 = 24h/5m)
- `df_is = df[:-holdout_bars]`, `df_holdout = df[-holdout_bars:]`
- Holdout FAIL 패턴은 출력 JSON에서 제거 + 로그 WARNING
- SKIP(거래 부족)은 유지 (7일에 3거래 미만은 자연스러움)
- 출력 JSON에 `holdout_validation` 섹션 추가 (각 패턴 결과)

---

## Change 2: Bot 스캔 Staleness 체크

### 파일: `scripts/production/pattern_5m/bot.py`

#### 2a. `_check_scan_staleness()` 함수 추가
```python
def _check_scan_staleness(config: dict) -> None:
    """dynamic_patterns.json의 generated_at 확인, 90일 초과 시 WARNING"""
    patterns_file = os.path.join(RESULTS_DIR, 'dynamic_patterns.json')
    try:
        with open(patterns_file) as f:
            data = json.load(f)
        generated_at = datetime.fromisoformat(data.get('generated_at', ''))
        age_days = (datetime.now() - generated_at).days
        interval = config.get('strategy', {}).get('rescan_interval_days', 90)
        if age_days > interval:
            logger.warning(
                f"⚠️ Pattern scan is {age_days} days old (threshold: {interval}d). "
                f"Consider re-scanning: python scripts/scanner/pattern_scanner.py"
            )
        else:
            logger.info(f"Pattern scan age: {age_days}d (threshold: {interval}d) ✓")
    except Exception as e:
        logger.warning(f"Could not check scan staleness: {e}")
```

#### 2b. 봇 시작 시 호출
- `run()` 메서드 초기화 단계 (verify_position_mode 근처)에서 `_check_scan_staleness(self.config)` 호출
- 경고만 출력, 봇 실행은 차단하지 않음

### 파일: `config/pattern_5m_config.yaml`
```yaml
strategy:
  rescan_interval_days: 90  # v1.34.0: scan staleness warning threshold
```

---

## Change 3: MDD 기반 동적 사이징

### 파일: `scripts/production/pattern_5m/state.py`

#### 3a. `_create_default_state()` 수정
```python
'peak_equity': 0.0,  # v1.34.0: MDD sizing high watermark
```

#### 3b. `update_peak_equity()` 함수 추가
```python
def update_peak_equity(state: dict, current_equity: float) -> dict:
    """peak equity 갱신 (high watermark)"""
    if current_equity > state.get('peak_equity', 0):
        state['peak_equity'] = current_equity
    return state
```

### 파일: `scripts/production/pattern_5m/position_open.py`

#### 3c. `get_position_size()` 수정
- 기존 시그니처에 `state` 파라미터 추가: `get_position_size(exchange, config, cache, circuit_breaker=None, metrics=None, state=None)`
- MDD 스케일 계산 삽입:

```python
# MDD dynamic sizing (v1.34.0)
mdd_scale = 1.0
mdd_cfg = config.get('risk', {}).get('mdd_sizing', {})
if mdd_cfg.get('enabled', False) and state:
    peak = state.get('peak_equity', 0)
    if peak > 0 and total_equity < peak:
        dd_pct = (peak - total_equity) / peak * 100
        full_below = mdd_cfg.get('full_size_below_dd', 5.0)
        min_above = mdd_cfg.get('min_size_above_dd', 20.0)
        min_scale = mdd_cfg.get('min_scale', 0.25)
        if dd_pct <= full_below:
            mdd_scale = 1.0
        elif dd_pct >= min_above:
            mdd_scale = min_scale
        else:
            mdd_scale = 1.0 - (1.0 - min_scale) * (dd_pct - full_below) / (min_above - full_below)
        if mdd_scale < 1.0:
            logger.info(f"MDD sizing: DD={dd_pct:.1f}%, scale={mdd_scale:.2f}")

per_slot_equity = total_equity * size_pct / max_positions * mdd_scale
```

#### 3d. 호출부 수정 (`position_open.py` 내부)
- `open_position()` → `get_position_size(..., state=state)` (state는 이미 파라미터로 전달됨)
- `refill_position()` → 동일

### 파일: `scripts/production/pattern_5m/bot.py`

#### 3e. Peak equity 갱신 호출
- `_handle_trading_window()` 또는 메인 루프에서 `update_peak_equity(state, current_equity)` 호출
- `current_equity`는 `get_balance()` 결과 사용

### 파일: `config/pattern_5m_config.yaml`
```yaml
risk:
  mdd_sizing:
    enabled: true
    full_size_below_dd: 5.0   # DD < 5% → full size (scale=1.0)
    min_size_above_dd: 20.0   # DD >= 20% → minimum size (scale=0.25)
    min_scale: 0.25           # minimum position scale factor
```

---

## 수정 대상 파일 요약

| 파일 | 변경 내용 |
|------|----------|
| `scripts/scanner/pattern_scanner.py` | `--holdout-days` 인자 + `holdout_validate()` + main() 흐름 |
| `scripts/production/pattern_5m/bot.py` | `_check_scan_staleness()` + peak equity 갱신 호출 |
| `scripts/production/pattern_5m/state.py` | `peak_equity` 기본값 + `update_peak_equity()` |
| `scripts/production/pattern_5m/position_open.py` | `get_position_size()` MDD scale 추가 |
| `config/pattern_5m_config.yaml` | `rescan_interval_days` + `mdd_sizing` 섹션 |
| `scripts/production/pattern_5m/constants.py` | `BOT_VERSION = "1.34.0"` |

---

## 검증

```bash
cd bingx_rl_trading_bot

# 1. 기존 테스트 통과 확인
python -m pytest scripts/tests/ -x -q

# 2. Scanner holdout 테스트 (7일 holdout)
python scripts/scanner/pattern_scanner.py --discovery-method mae_mfe --edge-threshold 21.8 --holdout-days 7

# 3. Staleness 체크 (generated_at 확인)
python -c "import json; d=json.load(open('results/dynamic_patterns.json')); print(d.get('generated_at'))"
```

핵심 확인:
- Scanner: holdout FAIL 패턴 제거 + SKIP 패턴 유지
- Bot: staleness WARNING이 90일 초과 시에만 출력
- MDD sizing: `peak_equity` 갱신 + DD에 따른 scale 적용
- 기존 테스트 전체 통과 (production 코드 변경이므로 필수)
