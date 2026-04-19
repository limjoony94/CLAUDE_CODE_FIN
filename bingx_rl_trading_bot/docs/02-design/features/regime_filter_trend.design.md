# Design: Regime Filter Trend — Production Implementation

> **Feature**: regime_filter_trend (9/9 GO 달성)
> **Date**: 2026-04-19
> **Status**: **Draft — 30일 LIVE 관찰 후 적용**
> **Robustness warning**: Neighborhood 2/25 sharp peak → overfit 의심, 적용 보수적

---

## 1. Config 변경 (`config/c1_breakout_config.yaml`)

```yaml
strategy:
  # 변경
  channel_period: 15
  body_min_ratio: 0.60           # ← 0.40에서
  atr_period: 14
  trail_K: 2.5
  max_sl_atr: 4.0                # ← 3.3에서
  emergency_sl_pct: 3.0
  max_hold_bars: 192
  sl_min_pct: 0.15
  sl_max_pct: 3.0
  min_bars_between: 2
  trail_activation_pct: 0.05
  max_positions: 1
  backstop_tp_atr: 0

  # 신규
  trend_filter:
    enabled: true                # 즉시 false로 rollback 가능
    lookback_bars: 192           # 2일 rolling (15m × 192 = 48시간)
    min_abs_trend_pct: 1.0       # |(close[-1] / close[-lookback] - 1) × 100| 기준
```

---

## 2. `scripts/production/c1_breakout/bot.py` 변경

### 2.1 신규 메서드 — Trend 계산

```python
def _compute_trend_pct(self, candles, bar):
    """Rolling trend % over lookback_bars (causal, past-only).

    Returns |trend_pct| (absolute value), or None if warmup 부족.
    """
    cfg = self.config['strategy'].get('trend_filter', {})
    if not cfg.get('enabled', False):
        return None  # filter disabled → bypass

    lookback = cfg.get('lookback_bars', 192)
    if bar < lookback:
        return None  # warmup 부족 → 진입 허용 (보수적)

    c_past = candles['close'][bar - lookback]
    c_now = candles['close'][bar]
    if c_past <= 0:
        return None
    return abs((c_now / c_past - 1) * 100)
```

### 2.2 Entry gate 추가 (line 811~819)

```python
# Entries — enforce min_bars_between + regime filter
min_bars = cfg.get('min_bars_between', 2)
if (len(self.positions) < self.max_positions
        and self.bars_since_last_exit >= min_bars):

    # NEW: Trend regime filter
    trend_pct = self._compute_trend_pct(candles, bar)
    trend_cfg = cfg.get('trend_filter', {})
    min_trend = trend_cfg.get('min_abs_trend_pct', 0.0)
    if (trend_cfg.get('enabled', False)
            and trend_pct is not None
            and trend_pct < min_trend):
        logger.info(f"Trend filter skip: |trend|={trend_pct:.2f}% < {min_trend}%")
        self._save_state()
        return  # 진입 skip, 다음 cycle 대기

    sig = self.signal.check_entry(
        candles['open'][bar], candles['high'][bar], candles['low'][bar],
        candles['close'][bar], ch_h[bar], ch_l[bar], cur_atr,
        sw_l[bar], sw_h[bar])
    if sig:
        self._do_open(sig, candles['close'][bar], cur_atr)
self._save_state()
```

---

## 3. Test 추가 (`scripts/tests/test_trend_filter.py`)

```python
def test_trend_filter_bypass_when_disabled():
    cfg = {'strategy': {'trend_filter': {'enabled': False}}}
    # ... mock bot, verify _compute_trend_pct returns None → no skip

def test_trend_filter_skip_when_below_threshold():
    cfg = {'trend_filter': {'enabled': True, 'lookback_bars': 10,
                              'min_abs_trend_pct': 1.0}}
    candles = {'close': [100]*11}  # flat → trend = 0
    # ... verify entry skip logged

def test_trend_filter_pass_when_trending():
    cfg = {'trend_filter': {'enabled': True, 'lookback_bars': 10,
                              'min_abs_trend_pct': 1.0}}
    candles = {'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]}
    # trend = 10% → pass filter

def test_trend_filter_warmup_bypass():
    # bar < lookback → None → no skip
```

---

## 4. Rollback Protocol

### 즉시 비활성화
```yaml
strategy:
  trend_filter:
    enabled: false    # 나머지 파라미터 유지
```

### 완전 복원 (baseline)
```yaml
strategy:
  body_min_ratio: 0.40    # 복원
  max_sl_atr: 3.3         # 복원
  trend_filter:
    enabled: false
```

봇 재시작만으로 완전 rollback.

---

## 5. Deployment 체크리스트 (30일 LIVE 후)

### Pre-deployment
- [ ] 30일 live steady-state 샘플 수집 (≥50 trades)
- [ ] Live slippage median이 slip_med 범위(~0.05~0.30%) 이내 확인
- [ ] 30일 기간에 2025-07~08 유형 regime (저trend) 실재 여부 확인
- [ ] **Neighborhood re-validation** — 30일 data로 thr=0.8/1.0/1.2, lb=144/192/256 재확인

### Deployment
1. `config/c1_breakout_config.yaml` 변경
2. `bot.py` 코드 변경 + 테스트 통과
3. 포지션 0개 확인 (state.json)
4. 봇 중지 → 재시작
5. `BOT START` 로그 + trend filter 활성 메시지 확인

### Post-deployment Monitor (첫 2주)
- Trend filter skip 빈도 ~26% 예상 vs 실측
- 진입 trade 감소 (기존의 ~74%)
- Fold 2 유형 regime 마주침 시 동작 확인

---

## 6. Risk Matrix

| Risk | 완화 |
|------|------|
| Overfit (neighborhood 2/25) | 30일 LIVE 재검증 + 부분 rollout |
| Live slippage > slip_med | 실측 모니터, 초과 시 즉시 rollback |
| Trend filter skip 과다 (>50%) | Skip 빈도 대시보드 경보 |
| Bot downtime 중 regime 오판 | trend 계산은 bar close 시점만, 무결 |
| Fold 2 유형 regime 발생 시 여전 drawdown | 단지 완화지 근본 해결 아님 |

---

## 7. 세션 통합 결과 요약

13 PDCAs 결과:
- Trail/SL/Emergency: 최적 도달, 변경 금지
- Entry body_min_ratio: 0.40 → 0.60 (selectivity↑)
- Entry trend_filter: 신규 (regime 차단)
- Exit/SL 파라미터: max_sl_atr 3.3 → 4.0

세 개 변경점의 **combined effect**가 9/9 GO.
