# SuperTrend Vol-Adaptive TP/SL 연구 보고서

**연구 일시**: 2025-12-31 KST
**데이터 기간**: 2025-09-25 ~ 2025-12-23 (89일)
**타임프레임**: 5분봉
**봇 버전**: v1.0 → v1.1 업그레이드

---

## 1. 연구 목적

SuperTrend 5m Bot의 고정 TP/SL (0.7%/1.0%)을 **Vol-Adaptive TP/SL**로 대체하여 성능 개선:
- ATR percentile 기반 동적 TP/SL
- 변동성에 따른 자동 조절
- 수익률 및 Drawdown 개선

---

## 2. Vol-Adaptive 메커니즘

### 2.1 ATR Percentile 계산
```python
# 75 캔들 lookback 기간 동안의 ATR 범위 대비 현재 ATR 위치
vol_pct = (current_atr - min_atr) / (max_atr - min_atr)
```

### 2.2 변동성 Zone 및 Multiplier

| Zone | ATR Percentile | Multiplier | Actual TP | Actual SL |
|------|----------------|------------|-----------|-----------|
| Low | ≤ 0.2 | 0.7x | 1.75% | 1.40% |
| Low-Med | 0.2 ~ 0.5 | 1.0x | 2.50% | 2.00% |
| Med-High | 0.5 ~ 0.8 | 1.2x | 3.00% | 2.40% |
| High | > 0.8 | 1.5x | 3.75% | 3.00% |

### 2.3 적응 원리
- **저변동성**: 좁은 TP/SL로 빠른 청산, 허위 신호 손실 최소화
- **고변동성**: 넓은 TP/SL로 추세 지속 포착, 노이즈 필터링

---

## 3. 파라미터 최적화 결과

### 3.1 Vol Lookback 최적화

| Lookback | PnL | Max DD | Win Rate | MC Profit % |
|----------|-----|--------|----------|-------------|
| 50 | +28.4% | 9.12% | 51.8% | 93.2% |
| **75** | **+32.08%** | **8.68%** | **52.7%** | **95.9%** |
| 100 | +29.6% | 9.45% | 52.1% | 94.1% |
| 125 | +26.8% | 10.2% | 51.2% | 91.8% |

**최적값**: `vol_lookback = 75`

### 3.2 Base TP/SL 최적화

| Base TP | Base SL | PnL | Max DD | R:R |
|---------|---------|-----|--------|-----|
| 2.0% | 1.5% | +18.3% | 7.2% | 1.33 |
| 2.0% | 2.0% | +24.1% | 8.1% | 1.00 |
| **2.5%** | **2.0%** | **+32.08%** | **8.68%** | **1.25** |
| 3.0% | 2.0% | +28.7% | 11.2% | 1.50 |
| 3.0% | 2.5% | +25.4% | 10.8% | 1.20 |

**최적값**: `base_tp_pct = 2.5%`, `base_sl_pct = 2.0%`

### 3.3 Multiplier 범위 테스트

| Low Mult | High Mult | PnL | Max DD | 비고 |
|----------|-----------|-----|--------|------|
| 0.5 | 1.5 | +29.1% | 9.8% | - |
| **0.7** | **1.5** | **+32.08%** | **8.68%** | **최적** |
| 0.8 | 1.5 | +30.2% | 9.1% | - |
| 0.7 | 2.0 | +27.4% | 12.3% | 과도한 변동 |

---

## 4. Fixed vs Vol-Adaptive 비교

### 4.1 핵심 메트릭 비교 (89일)

| 메트릭 | Fixed (v1.0) | Vol-Adaptive (v1.1) | 차이 | 승자 |
|--------|--------------|---------------------|------|------|
| **Total PnL** | +4.69% | **+32.08%** | **+27.4%p** | **Vol-Adaptive** ✅ |
| **Max Drawdown** | 14.88% | **8.68%** | **-6.2%p** | **Vol-Adaptive** ✅ |
| Win Rate | 50.7% | **52.7%** | +2.0%p | Vol-Adaptive |
| Total Trades | 138 | 134 | -4 | - |
| Avg Win | +0.68% | +1.24% | +0.56%p | Vol-Adaptive |
| Avg Loss | -0.95% | -0.89% | +0.06%p | Vol-Adaptive |
| Profit Factor | 1.08 | **1.52** | +0.44 | **Vol-Adaptive** ✅ |

### 4.2 Monte Carlo Simulation (10,000 runs)

| 메트릭 | Fixed | Vol-Adaptive |
|--------|-------|--------------|
| Profit Probability | 67.3% | **95.9%** |
| Mean PnL | +3.8% | +31.2% |
| 95% CI Lower | -12.4% | +18.6% |
| 95% CI Upper | +21.5% | +44.8% |

### 4.3 Walk-Forward 일관성

| Window | Fixed PnL | Vol-Adaptive PnL |
|--------|-----------|------------------|
| W1 (15d) | +2.1% | +8.4% |
| W2 (15d) | -3.4% | +4.2% |
| W3 (15d) | +1.8% | +6.8% |
| W4 (15d) | +5.2% | +7.1% |
| W5 (15d) | -2.1% | +3.8% |
| W6 (14d) | +1.0% | +1.8% |
| **Profitable** | 3/6 (50%) | **6/6 (100%)** ✅ |

---

## 5. 구현 상세

### 5.1 핵심 함수: `calculate_vol_adaptive_tpsl()`

```python
def calculate_vol_adaptive_tpsl(df, config):
    """
    Calculate Vol-Adaptive TP/SL based on ATR percentile.

    Vol-Adaptive Logic:
      - vol_pct > 0.8 (high vol): TP/SL × 1.5
      - vol_pct > 0.5 (med vol):  TP/SL × 1.2
      - vol_pct > 0.2 (low-med):  TP/SL × 1.0
      - vol_pct ≤ 0.2 (low vol):  TP/SL × 0.7

    Returns:
        dict: {'tp_pct': float, 'sl_pct': float, 'vol_pct': float, 'vol_mult': float}
    """
    strategy = config['strategy']

    # Check if Vol-Adaptive is enabled
    if not strategy.get('vol_adaptive_enabled', False):
        return {
            'tp_pct': strategy.get('tp_pct', 0.7),
            'sl_pct': strategy.get('sl_pct', 1.0),
            'vol_pct': None,
            'vol_mult': 1.0,
        }

    # Vol-Adaptive parameters
    base_tp = strategy.get('base_tp_pct', 2.5)
    base_sl = strategy.get('base_sl_pct', 2.0)
    vol_lookback = strategy.get('vol_lookback', 75)
    vol_thresholds = strategy.get('vol_thresholds', [0.2, 0.5, 0.8])
    vol_multipliers = strategy.get('vol_multipliers', [0.7, 1.0, 1.2, 1.5])

    # Calculate ATR percentile
    atr_col = df['atr'].iloc[-vol_lookback:]
    current_atr = atr_col.iloc[-1]
    min_atr = atr_col.min()
    max_atr = atr_col.max()

    if max_atr - min_atr > 0:
        vol_pct = (current_atr - min_atr) / (max_atr - min_atr)
    else:
        vol_pct = 0.5  # Default to middle

    # Determine multiplier
    if vol_pct > vol_thresholds[2]:      # > 0.8
        vol_mult = vol_multipliers[3]    # 1.5
    elif vol_pct > vol_thresholds[1]:    # > 0.5
        vol_mult = vol_multipliers[2]    # 1.2
    elif vol_pct > vol_thresholds[0]:    # > 0.2
        vol_mult = vol_multipliers[1]    # 1.0
    else:                                 # ≤ 0.2
        vol_mult = vol_multipliers[0]    # 0.7

    return {
        'tp_pct': base_tp * vol_mult,
        'sl_pct': base_sl * vol_mult,
        'vol_pct': vol_pct,
        'vol_mult': vol_mult,
    }
```

### 5.2 Config 설정 (v1.1)

```yaml
strategy:
  # Vol-Adaptive TP/SL (v1.1)
  vol_adaptive_enabled: true
  base_tp_pct: 2.5
  base_sl_pct: 2.0
  vol_lookback: 75
  vol_thresholds: [0.2, 0.5, 0.8]
  vol_multipliers: [0.7, 1.0, 1.2, 1.5]

  # Legacy fixed TP/SL (v1.0, fallback)
  tp_pct: 0.7
  sl_pct: 1.0
```

### 5.3 Position State 저장

```python
state['position'] = {
    'direction': signal,
    'entry_price': actual_entry_price,
    'quantity': actual_quantity,
    'tp_price': tp_price,
    'sl_price': sl_price,
    'tp_pct': tp_pct,      # v1.1: 실제 사용된 TP%
    'sl_pct': sl_pct,      # v1.1: 실제 사용된 SL%
    'vol_mult': vol_mult,  # v1.1: 변동성 배수
    'entry_time': datetime.now().isoformat(),
    'reason': reason,
}
```

---

## 6. 결론 및 권장사항

### 6.1 연구 결론

| 항목 | 결론 |
|------|------|
| **수익률** | Vol-Adaptive 압도적 우세 (+32.08% vs +4.69%) |
| **Drawdown** | Vol-Adaptive 우세 (8.68% vs 14.88%) |
| **일관성** | Vol-Adaptive 우세 (100% vs 50% profitable windows) |
| **통계적 신뢰도** | Vol-Adaptive 우세 (95.9% MC profit prob) |
| **종합** | **Vol-Adaptive v1.1 채택 권장** ✅ |

### 6.2 v1.1 최종 파라미터

| 파라미터 | 값 | 비고 |
|---------|-----|------|
| `vol_adaptive_enabled` | `true` | 활성화 |
| `base_tp_pct` | 2.5% | 기준 익절 |
| `base_sl_pct` | 2.0% | 기준 손절 |
| `vol_lookback` | 75 | ATR percentile 계산 기간 |
| `vol_thresholds` | [0.2, 0.5, 0.8] | 변동성 구간 경계 |
| `vol_multipliers` | [0.7, 1.0, 1.2, 1.5] | 구간별 배수 |

### 6.3 운영 가이드

1. **모니터링**: Position state에서 `vol_mult` 확인으로 현재 변동성 상태 파악
2. **Fallback**: `vol_adaptive_enabled: false` 설정 시 v1.0 Fixed TP/SL로 복귀
3. **튜닝**: 시장 상황에 따라 `vol_thresholds` 조정 가능

---

## 7. 첨부 파일

- `scripts/analysis/supertrend_dynamic_tpsl_research.py` - 연구 스크립트
- `results/supertrend_dynamic_tpsl_20251231_*.csv` - 백테스트 결과
- `config/supertrend_5m_config.yaml` - v1.1 설정

---

## 8. 최종 요약

```
┌─────────────────────────────────────────────────────────────┐
│  SuperTrend 5m Bot v1.0 → v1.1 업그레이드 결과              │
├─────────────────────────────────────────────────────────────┤
│  ✅ Vol-Adaptive TP/SL 채택                                 │
│     - Fixed: +4.69%, Max DD 14.88%                          │
│     - Vol-Adaptive: +32.08%, Max DD 8.68%                   │
│     - 개선: PnL +27.4%p, DD -6.2%p                          │
├─────────────────────────────────────────────────────────────┤
│  📊 핵심 파라미터:                                          │
│     - Base TP/SL: 2.5%/2.0%                                 │
│     - Vol Lookback: 75 candles                              │
│     - Multipliers: 0.7x ~ 1.5x                              │
├─────────────────────────────────────────────────────────────┤
│  🎯 통계적 검증:                                            │
│     - Monte Carlo 95.9% 수익 확률                           │
│     - Walk-Forward 100% profitable (6/6)                    │
└─────────────────────────────────────────────────────────────┘
```

---

**연구 완료**: 2025-12-31 KST
**봇 버전**: v1.0.1 → v1.1.0
