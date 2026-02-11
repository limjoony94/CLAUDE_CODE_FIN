# Production Trading Logic Analysis Report

> **Date**: 2026-02-04 | **Version**: v1.25.0 → **현재 프로덕션: v1.27.0** | **Analyst**: Claude
>
> **Note**: 이 문서는 v1.25.0 시점(2026-02-04)의 코드 로직 분석 스냅샷입니다.
> 코드 아키텍처(캔들 분류, 신호 탐지 흐름, Confidence 계산, 주문 관리, Early Exit 등)는
> v1.27.0에서도 동일하게 유지됩니다. 아래 항목들이 변경되었으므로 참고하세요:
>
> | 항목 | v1.25.0 (본 문서) | v1.27.0 (현재) |
> |------|------------------|----------------|
> | 패턴 수 | 20 (10L+10S) | **52 (32L+20S)** |
> | TP/SL 전략 | Per-pattern 원본값 | **Uniform TP 70%** (TP×0.7, SL 유지) |
> | Daily Loss Limit | 10% | **7%** |
> | 연속 손실 Pause | 없음 | **3연속 → 600초 일시정지** |
> | EXPECTED_WIN_RATE | ~77% | **84.0** |
> | 포트폴리오 성과 | WR 77.2%, PnL +1,346% | **WR 83.7%, PnL +911.1%, MDD 16.2%** |
>
> 최신 패턴 목록과 TP/SL 값은 `constants.py` 또는 `CLAUDE.md`를 참조하세요.

---

## 1. Executive Summary

본 보고서는 프로덕션에 배포된 Pattern 5m Bot v1.25.0의 패턴 기반 거래 로직에 대한 상세 분석입니다.

### 핵심 구성요소

| 모듈 | 파일 | 역할 |
|------|------|------|
| 캔들 분류 | `indicators.py` | 12-type 캔들 분류 시스템 |
| 신호 탐지 | `signals.py` | 3-candle 패턴 매칭 및 Context Filter |
| 진입 실행 | `position_open.py` | 시장가 주문 및 TP/SL 계산 |
| 주문 관리 | `orders.py` | TP/SL 주문 배치 및 검증 |
| 상수 정의 | `constants.py` | 패턴 목록, 임계값, 파라미터 |

---

## 2. Candle Classification System (12-Type)

### 2.1 Classification Hierarchy

`indicators.py:classify_candle()` 함수의 분류 우선순위:

```
1. Marubozu (MU/MD)  - 윅이 거의 없는 강한 방향성
2. Hammer (H/IH)     - 극단적 윅 비율
3. Doji Family (D/DF/GS) - 작은 몸통
4. Spinning Top (ST) - 양쪽 윅 균형
5. Big/Medium (BU/BD/U/DN) - 정규화된 몸통 크기
```

### 2.2 Classification Thresholds

| 타입 | 임계값 | 조건 |
|------|--------|------|
| DOJI | `body_ratio < 0.10` | 몸통 < 범위의 10% |
| DRAGONFLY | `lower_wick > 0.70` | 하단 윅 > 범위의 70% |
| GRAVESTONE | `upper_wick > 0.70` | 상단 윅 > 범위의 70% |
| MARUBOZU | `total_wick < 0.15` | 총 윅 < 범위의 15% |
| HAMMER | `wick/body > 2.0` | 윅이 몸통의 2배 이상 |
| SPINNING_TOP | `norm_body < 0.5` + 양쪽 윅 | 정규화 몸통 < 0.5 |
| BIG | `norm_body > 1.5` | 정규화 몸통 > 1.5 |

### 2.3 Normalization

```python
avg_body_20 = df['body_abs'].rolling(20).mean()
norm_body = body_abs / avg_body_20
```

- 초기 20봉 (avg_body_20 = NaN): `avg_body_20 = 1.0` 기본값 사용
- 이는 range 기반 분류 (DOJI, HAMMER, MARUBOZU)를 보존

---

## 3. Pattern Detection Logic

### 3.1 3-Candle Pattern Construction

`signals.py:add_candle_classification()`

```python
pattern = f"{df.iloc[i-2]['type_code']}-{df.iloc[i-1]['type_code']}-{df.iloc[i]['type_code']}"
```

예시: `"MD-BU-U"` = MARUBOZU_DOWN → BIG_UP → MED_UP

### 3.2 Signal Detection Flow

`signals.py:check_entry_signal()`

```
1. 캔들 분류 완료 확인
2. 마지막 완성된 캔들(iloc[-2])의 패턴 추출
3. 중복 신호 방지 (last_signal_candle_timestamp)
4. 패턴 매칭:
   - Regime Detection (v1.18, 현재 비활성화)
   - VALIDATED_LONG_PATTERNS 확인 → LONG
   - VALIDATED_SHORT_PATTERNS 확인 → SHORT
5. Context Filter 적용 (RSI/Vol/Trend)
6. Confidence 계산 및 로깅
```

### 3.3 Validated Patterns (v1.25.0)

**LONG Patterns (10)**

| Pattern | WR | MC | WF | Trades |
|---------|-----|------|-----|--------|
| MD-BU-U | 94.4% | 0.0005 | 5/5 | 54 |
| MU-MU-U | 90.3% | 0.0050 | 4/5 | 31 |
| MU-U-MU | 89.5% | 0.0140 | 4/5 | 19 |
| BU-BU-BD | 84.4% | 0.0123 | 5/5 | 45 |
| ST-H-DN | 82.6% | 0.0172 | 5/5 | 69 |
| ST-MU-U | 76.9% | 0.0026 | 4/5 | 121 |
| DN-IH-ST | 76.3% | 0.0118 | 5/5 | 59 |
| IH-DN-DN | 71.6% | 0.0045 | 4/5 | 67 |
| MD-DN-MU | 59.2% | 0.0069 | 4/5 | 49 |
| BD-ST-U | 57.6% | 0.0030 | 5/5 | 59 |

**SHORT Patterns (10)**

| Pattern | WR | MC | WF | Trades |
|---------|-----|------|-----|--------|
| MD-ST-ST | 98.5% | 0.0012 | 5/5 | 65 |
| U-MU-BU | 98.1% | 0.0020 | 5/5 | 53 |
| MU-BU-DN | 97.7% | 0.0002 | 5/5 | 44 |
| ST-H-U | 97.1% | 0.0048 | 5/5 | 34 |
| ST-DN-H | 93.6% | 0.0079 | 5/5 | 47 |
| MD-MU-U | 90.5% | 0.0020 | 5/5 | 42 |
| BU-U-ST | 90.2% | 0.0001 | 5/5 | 92 |
| H-DN-ST | 88.2% | 0.0176 | 4/5 | 51 |
| DN-BD-BU | 85.9% | 0.0068 | 4/5 | 78 |
| DN-BU-U | 63.0% | 0.0180 | 4/5 | 119 |

---

## 4. Context Filter System (v1.7+)

### 4.1 Context Calculation

`signals.py:calculate_context()`

| Feature | 계산 방식 | 분류 |
|---------|----------|------|
| RSI Zone | 14-period RSI | OS (<30), N (30-70), OB (>70) |
| Vol | ATR% quantile | L (<33%), M (33-66%), H (>66%) |
| Trend | close vs close[-20] | UP / DN |

### 4.2 Filter Logic

```python
# Required: 반드시 통과해야 함
for ctx_key, allowed_values in required.items():
    if context[ctx_key] not in allowed_values:
        return False  # 필터 실패

# Excluded: 특정 조건에서 거래 금지
for ctx_key, excluded_values in excluded.items():
    if context[ctx_key] in excluded_values:
        return False  # 필터 실패

# Preferred: 선호 조건 시 보너스
if context[ctx_key] in preferred_values:
    confidence_bonus += 0.10
```

### 4.3 현재 상태 (v1.25.0)

- `PATTERN_CONTEXT_FILTERS = {}` (빈 딕셔너리)
- v1.19.0에서 패턴 전면 교체로 기존 필터 제거
- 새 패턴에 대한 Context Filter 연구 미완료

---

## 5. Confidence Scoring System

### 5.1 Confidence Components

`signals.py:calculate_pattern_confidence()`

| Component | Weight | 계산 방식 |
|-----------|--------|----------|
| Clarity | 40% | 캔들 분류 명확도 평균 |
| Historical | 30% | 패턴 WR 정규화 |
| Regime | 30% | 플레이스홀더 (0.6 고정) |

### 5.2 Clarity Calculation

```python
def calculate_candle_clarity(row, avg_body_20):
    # 캔들 타입별 "교과서적" 예시 정도 측정
    # DOJI: body_ratio가 0에 가까울수록 높음
    # MARUBOZU: 윅 비율이 낮을수록 높음
    # HAMMER: wick/body 비율이 높을수록 높음
    return score  # 0.0 ~ 1.0
```

### 5.3 Historical WR Normalization

```python
# 50% = 0.0, 70% = 1.0
historical = (wr - 0.50) / 0.20
```

---

## 6. Position Opening Logic

### 6.1 Entry Flow

`position_open.py:open_position()`

```
1. 기존 포지션 확인 (거래소 조회)
2. 레버리지 설정 (exchange_leverage: 10x)
3. 포지션 사이즈 계산:
   - available * position_size_pct (95%)
   - max_position_size_usd 제한
   - quantity = (value * leverage) / price
4. 시장가 주문 실행:
   - params={'positionSide': 'BOTH'} (One-Way Mode)
5. 실제 체결가 조회
6. TP/SL 계산
7. State 업데이트
8. TP/SL 주문 배치
```

### 6.2 TP/SL Calculation Priority

`position_open.py:_calculate_tp_sl()`

```
우선순위:
1. regime_tp_sl (Regime-Adaptive, 현재 비활성화)
2. PATTERN_OPTIMAL_TPSL[pattern] (Per-Pattern 최적화)
3. strategy defaults (1.0%/1.0%)
```

### 6.3 Per-Pattern TP/SL (v1.25.0)

```python
PATTERN_OPTIMAL_TPSL = {
    # LONG patterns
    "MD-BU-U": {"tp": 0.5, "sl": 2.0},
    "MU-MU-U": {"tp": 0.7, "sl": 1.5},
    "MU-U-MU": {"tp": 1.5, "sl": 2.0},
    # ... (각 패턴별 최적화된 TP/SL)

    # SHORT patterns
    "MD-ST-ST": {"tp": 0.5, "sl": 2.0},
    "U-MU-BU": {"tp": 0.5, "sl": 2.0},
    # ...
}
```

### 6.4 Slippage Adjustment

```python
tp_pct_adjusted = base_tp_pct + SLIPPAGE_BUFFER_PCT  # 0.02% 추가
sl_pct_adjusted = base_sl_pct - SLIPPAGE_BUFFER_PCT  # 0.02% 감소

tp_price = entry * (1 + direction * tp_pct_adjusted / 100)
sl_price = entry * (1 - direction * sl_pct_adjusted / 100)
```

---

## 7. Order Management

### 7.1 TP/SL Order Types

`orders.py`

| 주문 유형 | Type | Purpose |
|----------|------|---------|
| Take Profit | `TAKE_PROFIT_MARKET` | 이익 실현 |
| Stop Loss | `STOP_MARKET` | 손실 제한 |

### 7.2 Order Parameters

```python
# TP Order
exchange.create_order(
    symbol=symbol,
    type='TAKE_PROFIT_MARKET',
    side=close_side,
    amount=quantity,
    params={
        'positionSide': 'BOTH',
        'stopPrice': tp_price,
        'closePosition': True,
    }
)

# SL Order
exchange.create_order(
    symbol=symbol,
    type='STOP_MARKET',
    side=close_side,
    amount=quantity,
    params={
        'positionSide': 'BOTH',
        'stopPrice': sl_price,
        'closePosition': True,
    }
)
```

### 7.3 TP/SL Auto-Adjustment (v1.17)

`orders.py:adjust_tpsl_to_config()`

- 봇 시작 시 기존 포지션의 TP/SL이 현재 `PATTERN_OPTIMAL_TPSL`과 다르면 자동 조정
- 허용 오차: $1

### 7.4 Order Verification

`orders.py:verify_tp_sl_orders()`

- 주기적으로 TP/SL 주문 존재 확인
- 누락된 주문 자동 재배치

---

## 8. Early Exit Signal (v1.13)

### 8.1 조건

`signals.py:check_early_exit_signal()`

| 조건 | 값 |
|------|-----|
| LONG 청산 | 3연속 BD (Big Down) |
| SHORT 청산 | 3연속 BU (Big Up) |
| 최소 이익 | ≥ 0.3% |

### 8.2 Logic Flow

```python
if type_code in reversal_types:
    reversal_count += 1
    if (reversal_count >= 3 and unrealized_pnl >= 0.3%):
        return True  # Early exit
else:
    reversal_count = 0  # Reset
```

### 8.3 Double-Counting Prevention (v1.14.1)

- `last_counted_candle_ts`로 동일 캔들 중복 카운트 방지
- 5분봉 완성 시 1회만 카운트

---

## 9. Risk Management Features

### 9.1 Circuit Breaker

```python
CIRCUIT_BREAKER_THRESHOLD = 5   # 연속 실패 횟수
CIRCUIT_BREAKER_TIMEOUT = 60    # 초기 타임아웃
CB_BACKOFF_MULTIPLIER = 2.0     # 지수 백오프
CB_MAX_TIMEOUT = 600            # 최대 타임아웃 (10분)
```

### 9.2 Daily Loss Limit

```python
max_daily_loss_pct = 7.0  # v1.27.0: -7% 도달 시 거래 중단 (v1.25.0: 10%)
```

### 9.3 Consecutive Loss Pause (v1.27.0 추가)

```python
CONSECUTIVE_LOSS_PAUSE_THRESHOLD = 3   # 3연속 손실 시
CONSECUTIVE_LOSS_COOLDOWN = 600        # 600초(10분) 일시정지
```

### 9.4 Cooldown Period

```python
cooldown_candles = 0  # 현재 비활성화 (즉시 재진입 가능)
```

### 9.5 Duplicate Signal Prevention

```python
if last_signal_candle == current_timestamp:
    return None, None  # 동일 캔들에서 중복 신호 방지
```

---

## 10. Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Main Loop (bot.py)                       │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  1. Fetch OHLCV Data (exchange.py)                          │
│     └── 150 candles, 5m timeframe                           │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  2. Calculate Indicators (indicators.py)                    │
│     ├── Body calculations (body, body_abs, avg_body_20)     │
│     ├── Candle classification (12-type)                     │
│     └── Pattern construction (3-candle)                     │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  3. Check Entry Signal (signals.py)                         │
│     ├── Pattern matching (LONG/SHORT patterns)              │
│     ├── Context filter (RSI/Vol/Trend)                      │
│     ├── Confidence calculation                              │
│     └── Duplicate prevention                                │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  4. Open Position (position_open.py)                        │
│     ├── Position size calculation                           │
│     ├── Market order execution                              │
│     ├── TP/SL calculation (per-pattern optimized)           │
│     └── State update                                        │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  5. Place TP/SL Orders (orders.py)                          │
│     ├── TAKE_PROFIT_MARKET order                            │
│     └── STOP_MARKET order                                   │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  6. Monitor Position (position_monitor.py)                  │
│     ├── Early exit signal check                             │
│     ├── TP/SL order verification                            │
│     └── Exchange sync                                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 11. Key Configuration Parameters

### 11.1 Trading Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Leverage | 3x | 계좌 내 레버리지 |
| Exchange Leverage | 10x | 거래소 설정 레버리지 |
| Position Size | 95% | 가용 잔고의 95% |
| Max Position | $10,000 | 최대 포지션 사이즈 |
| Default TP | 1.0% | 기본 Take Profit |
| Default SL | 1.0% | 기본 Stop Loss |

### 11.2 Classification Thresholds

| Threshold | Value | Purpose |
|-----------|-------|---------|
| DOJI_BODY_RATIO | 0.10 | Doji 판별 |
| WICK_DOMINANCE | 0.70 | Dragonfly/Gravestone |
| MARUBOZU_WICK | 0.15 | Marubozu 판별 |
| HAMMER_WICK_RATIO | 2.0 | Hammer 판별 |
| SPINNING_TOP_NORM | 0.5 | Spinning Top 판별 |
| BIG_CANDLE_NORM | 1.5 | Big candle 판별 |

### 11.3 System Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| CANDLE_POLL_INTERVAL | 60s | 메인 루프 주기 |
| CACHE_TTL | 5s | API 캐시 TTL |
| MAX_OHLCV_CANDLES | 150 | OHLCV 조회 개수 |
| METRICS_SAVE_INTERVAL | 10 | 메트릭 저장 주기 |

---

## 12. Conclusion

### 12.1 Strengths

1. **통계적 검증**: 모든 패턴이 MC < 0.02, WF ≥ 4/5 통과
2. **Per-Pattern Optimization**: 각 패턴별 최적화된 TP/SL
3. **Multi-Layer Validation**: Context Filter + Confidence Score
4. **Robust Error Handling**: Circuit Breaker + Auto-Recovery

### 12.2 Areas for Improvement (v1.25.0 시점)

> **v1.27.0 업데이트**: 아래 3번 항목 외에는 후속 버전에서 진전 없음.
> v1.26.x~v1.27.0에서는 패턴 포트폴리오 최적화와 TP/SL 연구에 집중.

1. **Context Filter 연구 필요**: 52패턴에 대한 Context Filter 미정의 상태 유지
2. **Regime Detection 비활성화**: v1.19.0 이후 비활성화 상태 유지
3. ~~Confidence → Trade Outcome 상관관계~~ → v1.27.0 microstructure research에서 부분 분석

### 12.3 Recommendations

1. 52패턴에 대한 Context Filter 연구 수행
2. Confidence Score와 실제 거래 결과 상관관계 분석
3. Walk-Forward 지속 모니터링 (월간 검증)

---

**Report Generated**: 2026-02-04 02:00 KST
**Updated**: 2026-02-11 (v1.27.0 아카이브 노트, 리스크 관리 섹션 추가)
**Author**: Claude Code Analysis System
