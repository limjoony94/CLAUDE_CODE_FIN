# Donchian Channel Strategy Deep Analysis
**Date**: 2025-11-30
**Status**: Strategy Review & Optimization Proposal

---

## Executive Summary

현재 구현된 Donchian 전략을 분석한 결과, **핵심 로직에 근본적인 설계 문제**가 발견되었습니다. 40년 이상 검증된 Turtle Trading 원칙과 비교했을 때, 현재 전략은 "최적 진입점"을 놓치고 있습니다.

| 항목 | 현재 전략 | 검증된 최적 전략 | 문제점 |
|------|----------|-----------------|--------|
| **진입 로직** | 중간 존 진입 (0.42-0.58) | 브레이크아웃 후 풀백 | ❌ 브레이크아웃 없이 진입 |
| **청산 로직** | 고정 TP/SL (1.5%/0.8%) | 반대 밴드 터치 + ATR 기반 | ❌ 변동성 미반영 |
| **승률** | 50.2% | 45-55% (정상) | ⚠️ 정상 범위 |
| **수익 팩터** | 1.26× | 1.5-2.5× | ❌ 낮음 |

---

## Part 1: Current Strategy Analysis (현재 전략 분석)

### 1.1 Current Entry Logic
```python
# 현재 진입 조건
zone_low = 0.5 - DONCHIAN_ZONE   # 0.42
zone_high = 0.5 + DONCHIAN_ZONE  # 0.58

# LONG: price > EMA50 and in middle zone
if price > ema50 and zone_low < dc_position < zone_high:
    signal = 1  # LONG

# SHORT: price < EMA50 and in middle zone
elif price < ema50 and zone_low < dc_position < zone_high:
    signal = -1  # SHORT
```

### 1.2 Critical Issues with Current Entry

#### Issue 1: "Middle Zone" Entry의 근본 문제
```
현재 로직: 가격이 Donchian 채널의 중간 16% (0.42-0.58)에 있을 때 진입

문제점:
┌─────────────────────────────────────────────────────────────┐
│  Donchian Upper Band (1.0) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│                                                             │
│  dc_position = 0.70                    ← 브레이크아웃 영역    │
│  dc_position = 0.58 ─────────────────  ← 현재 존 상단        │
│  ▲▲▲ 현재 진입 영역 (Middle Zone) ▲▲▲                       │
│  dc_position = 0.42 ─────────────────  ← 현재 존 하단        │
│  dc_position = 0.30                    ← 브레이크다운 영역    │
│                                                             │
│  Donchian Lower Band (0.0) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
└─────────────────────────────────────────────────────────────┘

실제 발생 상황:
- 가격이 중간 영역에 "항상" 있을 수 있음 (횡보장)
- 브레이크아웃이 없었는데 진입 → 방향성 약함
- "어느 방향으로 갈지 모르는" 위치에서 진입
```

#### Issue 2: EMA50 필터의 한계
```
현재: price > ema50 → LONG, price < ema50 → SHORT

문제점:
- EMA50 기울기/강도 미확인
- 횡보장에서 EMA50 ≈ 현재가 → 무의미한 신호
- 트렌드 "존재 여부"만 확인, 트렌드 "강도" 미확인
```

#### Issue 3: 브레이크아웃 확인 없음
```
Turtle Trading 원칙:
1. 브레이크아웃 발생 (20일 최고/최저 돌파)
2. 풀백 대기 (중간 밴드로 복귀)
3. 풀백에서 진입

현재 전략:
1. (브레이크아웃 확인 X)
2. 가격이 중간에 있으면 바로 진입  ← 핵심 문제!
3. 방향성 없이 진입

결과: 횡보장에서 무의미한 신호 다수 발생
```

### 1.3 Current Exit Logic
```python
# 현재 청산 조건
TAKE_PROFIT_PCT = 1.5    # 1.5% TP
STOP_LOSS_PCT = 0.8      # 0.8% SL
MAX_HOLD_CANDLES = 9999  # 비활성화

# 추가 메커니즘
USE_TRAILING_STOP = True
TRAILING_ACTIVATION_PCT = 0.8   # +0.8% 도달 시 활성화
TRAILING_DISTANCE_PCT = 0.4     # 0.4% 거리

USE_BREAKEVEN_STOP = True
BREAKEVEN_ACTIVATION_PCT = 0.5  # +0.5% 도달 시 BE
```

### 1.4 Critical Issues with Current Exit

#### Issue 1: 고정 TP/SL의 문제
```
현재: TP 1.5%, SL 0.8% (고정)

문제점:
- 변동성이 높을 때: 0.8% SL이 너무 타이트 → 노이즈에 손절
- 변동성이 낮을 때: 1.5% TP가 너무 멀어 → 수익 실현 어려움
- ATR 기반 동적 조정 없음

예시 (BTC):
- 고변동성: ATR = 2% → 0.8% SL은 노이즈
- 저변동성: ATR = 0.3% → 0.8% SL은 너무 넓음
```

#### Issue 2: Donchian 밴드 기반 청산 미사용
```
검증된 Turtle 청산:
- LONG: 10일 최저점 이탈 시 청산
- SHORT: 10일 최고점 이탈 시 청산

현재:
- Donchian 밴드 완전 무시
- 고정 % 기반 청산만 사용
- 시장 구조 무시한 청산
```

#### Issue 3: 리스크-리워드 비율
```
현재 R:R = 1.5% / 0.8% = 1.875:1 (양호)
BUT 승률 50.2% → 기대값 = (0.502 × 1.875) - (0.498 × 1) = 0.44

최적 R:R (ATR 기반):
- TP = 2-3× ATR
- SL = 1-1.5× ATR
- R:R = 2:1 ~ 3:1 목표
```

---

## Part 2: Optimal Donchian Strategies (검증된 최적 전략)

### 2.1 Strategy Option A: Classic Turtle (Low Frequency, High Quality)

```
진입 조건:
┌─────────────────────────────────────────────────────────────┐
│ System 1 (단기):                                            │
│  - LONG: 20일 최고가 돌파 시 (dc_position > 1.0)            │
│  - SHORT: 20일 최저가 이탈 시 (dc_position < 0.0)           │
│  - 이전 신호가 수익이었으면 스킵 (휩소 방지)                  │
│                                                             │
│ System 2 (장기, 백업):                                      │
│  - LONG: 55일 최고가 돌파 시                                 │
│  - SHORT: 55일 최저가 이탈 시                                │
│  - 항상 진입 (스킵 없음)                                     │
└─────────────────────────────────────────────────────────────┘

청산 조건:
┌─────────────────────────────────────────────────────────────┐
│ System 1:                                                   │
│  - LONG: 10일 최저가 이탈 시 청산                           │
│  - SHORT: 10일 최고가 돌파 시 청산                          │
│                                                             │
│ System 2:                                                   │
│  - LONG: 20일 최저가 이탈 시 청산                           │
│  - SHORT: 20일 최고가 돌파 시 청산                          │
└─────────────────────────────────────────────────────────────┘

예상 성과:
- 거래 빈도: 0.5-1.5회/일
- 승률: 45-50%
- 수익 팩터: 1.5-2.5×
- 홀드 타임: 5-50 캔들 (5분봉 기준 25분-4시간)
```

### 2.2 Strategy Option B: Pullback Entry (Medium Frequency, Best Risk-Reward)

```
진입 조건 (2단계):
┌─────────────────────────────────────────────────────────────┐
│ 1단계: 브레이크아웃 확인                                     │
│  - breakout_long = 최근 N캔들 내 dc_position > 0.95         │
│  - breakout_short = 최근 N캔들 내 dc_position < 0.05        │
│                                                             │
│ 2단계: 풀백 진입                                            │
│  - LONG: breakout_long AND 현재 dc_position 0.45-0.55      │
│          AND price > EMA50 (트렌드 유지 확인)               │
│  - SHORT: breakout_short AND 현재 dc_position 0.45-0.55    │
│          AND price < EMA50 (트렌드 유지 확인)               │
└─────────────────────────────────────────────────────────────┘

청산 조건 (하이브리드):
┌─────────────────────────────────────────────────────────────┐
│ TP 단계별 청산:                                             │
│  - 50% 물량: 반대 밴드 50% 지점 (dc_position 0.25/0.75)     │
│  - 30% 물량: 반대 밴드 터치 (dc_position 0.0/1.0)           │
│  - 20% 물량: 트레일링 스탑 (1.5× ATR)                       │
│                                                             │
│ SL:                                                         │
│  - 2× ATR 반대 밴드 너머                                    │
│  - 또는 dc_position 반대 밴드 돌파                          │
└─────────────────────────────────────────────────────────────┘

예상 성과:
- 거래 빈도: 2-4회/일
- 승률: 55-60%
- 수익 팩터: 1.8-2.5×
- 홀드 타임: 20-60 캔들 (1-5시간)
```

### 2.3 Strategy Option C: Mean Reversion (High Frequency, For Range Markets)

```
진입 조건 (역발상):
┌─────────────────────────────────────────────────────────────┐
│ LONG:                                                       │
│  - dc_position < 0.10 (하단 밴드 근처 과매도)               │
│  - RSI < 35 (과매도 확인)                                   │
│  - ADX < 25 (트렌드 약함 = 반전 가능)                       │
│                                                             │
│ SHORT:                                                      │
│  - dc_position > 0.90 (상단 밴드 근처 과매수)               │
│  - RSI > 65 (과매수 확인)                                   │
│  - ADX < 25 (트렌드 약함 = 반전 가능)                       │
└─────────────────────────────────────────────────────────────┘

청산 조건:
┌─────────────────────────────────────────────────────────────┐
│ TP: 중간 밴드 복귀 (dc_position 0.45-0.55)                  │
│ SL: 반대 밴드 돌파 + 1× ATR                                 │
│ MAX HOLD: 40 캔들 (시간 제한)                               │
└─────────────────────────────────────────────────────────────┘

예상 성과:
- 거래 빈도: 5-10회/일
- 승률: 50-55%
- 수익 팩터: 1.3-1.6×
- 홀드 타임: 5-20 캔들 (25분-1.5시간)
```

---

## Part 3: Comparison Table (현재 vs 최적)

| 항목 | 현재 전략 | Option A (Turtle) | Option B (Pullback) | Option C (Mean Rev) |
|------|----------|-------------------|---------------------|---------------------|
| **진입 트리거** | 중간존 진입 | 브레이크아웃 | 브레이크아웃→풀백 | 극단값 반전 |
| **트렌드 필터** | EMA50 가격비교 | 없음 (모멘텀) | EMA50+기울기 | ADX<25 |
| **청산 방식** | 고정 % | 반대밴드 터치 | 단계별+트레일링 | 중간밴드 복귀 |
| **SL 계산** | 고정 0.8% | 반대밴드 | 2×ATR | 반대밴드+1×ATR |
| **TP 계산** | 고정 1.5% | 반대밴드 | 단계별 50/30/20% | 중간밴드 |
| **거래빈도** | ~5회/일 | 0.5-1.5회/일 | 2-4회/일 | 5-10회/일 |
| **예상 승률** | 50.2% | 45-50% | 55-60% | 50-55% |
| **예상 PF** | 1.26× | 1.5-2.5× | 1.8-2.5× | 1.3-1.6× |
| **최적 시장** | - | 강한 트렌드 | 중간 트렌드 | 횡보/레인지 |

---

## Part 4: Recommended Implementation (권장 구현)

### 4.1 Hybrid Strategy: Option B + C 조합

현재 BTC 시장 특성상 **트렌드와 횡보가 번갈아** 나타나므로, 시장 상황에 따라 전환하는 하이브리드 전략을 권장합니다.

```python
# 시장 레짐 판단
def detect_market_regime(df):
    """
    트렌드 vs 횡보 판단
    """
    adx = calculate_adx(df, 14)
    dc_range_pct = df['dc_range'].iloc[-1] / df['close'].iloc[-1] * 100

    if adx > 25 and dc_range_pct > 2.0:
        return 'TRENDING'  # Option B 사용
    else:
        return 'RANGING'   # Option C 사용


# 트렌드 시장용 진입 (Option B)
def get_signal_trending(df, state):
    """
    브레이크아웃 후 풀백 진입
    """
    # 최근 10캔들 내 브레이크아웃 확인
    recent_breakout_long = (df['dc_position'].tail(10) > 0.95).any()
    recent_breakout_short = (df['dc_position'].tail(10) < 0.05).any()

    current_dc = df['dc_position'].iloc[-1]
    price = df['close'].iloc[-1]
    ema50 = df['ema50'].iloc[-1]

    # LONG: 브레이크아웃 후 풀백
    if recent_breakout_long and 0.40 < current_dc < 0.60:
        if price > ema50:  # 트렌드 유지 확인
            return 1, f"PULLBACK_LONG: breakout confirmed, pullback to {current_dc:.2f}"

    # SHORT: 브레이크다운 후 풀백
    if recent_breakout_short and 0.40 < current_dc < 0.60:
        if price < ema50:  # 트렌드 유지 확인
            return -1, f"PULLBACK_SHORT: breakdown confirmed, pullback to {current_dc:.2f}"

    return 0, "No signal"


# 횡보 시장용 진입 (Option C)
def get_signal_ranging(df, state):
    """
    극단값 반전 진입
    """
    current_dc = df['dc_position'].iloc[-1]
    rsi = df['rsi'].iloc[-1]

    # LONG: 하단에서 반전
    if current_dc < 0.10 and rsi < 35:
        return 1, f"REVERSAL_LONG: oversold at DC={current_dc:.2f}, RSI={rsi:.1f}"

    # SHORT: 상단에서 반전
    if current_dc > 0.90 and rsi > 65:
        return -1, f"REVERSAL_SHORT: overbought at DC={current_dc:.2f}, RSI={rsi:.1f}"

    return 0, "No signal"
```

### 4.2 ATR-Based Exit (변동성 기반 청산)

```python
def calculate_dynamic_tpsl(df, side, entry_price):
    """
    ATR 기반 동적 TP/SL 계산
    """
    atr = df['atr'].iloc[-1]

    # R:R = 2:1 목표
    sl_distance = 1.5 * atr
    tp_distance = 3.0 * atr  # 2:1 R:R

    if side == 'LONG':
        sl_price = entry_price - sl_distance
        tp_price = entry_price + tp_distance
    else:
        sl_price = entry_price + sl_distance
        tp_price = entry_price - tp_distance

    return {
        'tp_price': round(tp_price, 1),
        'sl_price': round(sl_price, 1),
        'atr': atr,
        'r_multiple': tp_distance / sl_distance
    }


def check_donchian_exit(df, position):
    """
    Donchian 밴드 기반 청산 확인
    """
    side = position['side']
    current_dc = df['dc_position'].iloc[-1]

    # LONG: 반대 밴드(하단) 터치
    if side == 'LONG' and current_dc < 0.05:
        return True, "DC_LOWER_BAND_EXIT"

    # SHORT: 반대 밴드(상단) 터치
    if side == 'SHORT' and current_dc > 0.95:
        return True, "DC_UPPER_BAND_EXIT"

    # 중간 밴드 복귀 (부분 청산용)
    if 0.45 < current_dc < 0.55:
        return False, "DC_MIDDLE_BAND_PARTIAL"  # 부분 청산 트리거

    return False, None
```

### 4.3 Staged Exit (단계별 청산)

```python
def execute_staged_exit(client, position, df, reason):
    """
    단계별 청산 실행

    - 50%: 중간 밴드 복귀 시
    - 30%: 반대 밴드 터치 시
    - 20%: 트레일링 스탑
    """
    current_dc = df['dc_position'].iloc[-1]
    side = position['side']

    # 단계 1: 중간 밴드 복귀 (50% 청산)
    if not position.get('stage1_closed'):
        if 0.45 < current_dc < 0.55:
            close_qty = position['quantity'] * 0.5
            close_position_partial(client, position, close_qty, "STAGE1_MIDDLE_BAND")
            position['stage1_closed'] = True
            position['quantity'] -= close_qty

    # 단계 2: 반대 밴드 터치 (추가 30% 청산)
    if not position.get('stage2_closed') and position.get('stage1_closed'):
        opposite_touch = (side == 'LONG' and current_dc < 0.10) or \
                        (side == 'SHORT' and current_dc > 0.90)
        if opposite_touch:
            close_qty = position['quantity'] * 0.6  # 남은 것의 60% = 전체의 30%
            close_position_partial(client, position, close_qty, "STAGE2_OPPOSITE_BAND")
            position['stage2_closed'] = True
            position['quantity'] -= close_qty

    # 단계 3: 나머지는 트레일링 스탑으로 관리
    # (기존 트레일링 로직 유지)

    return position
```

---

## Part 5: Implementation Roadmap (구현 로드맵)

### Phase 1: Quick Win (즉시 적용 가능)
```yaml
Changes:
  1. ATR 기반 SL 계산 (고정 0.8% → 1.5×ATR)
  2. ATR 기반 TP 계산 (고정 1.5% → 3×ATR)
  3. 트레일링 스탑 거리 동적화 (고정 0.4% → 0.5×ATR)

Expected Improvement:
  - 노이즈 손절 감소 (-30-50%)
  - 수익 팩터 개선 (1.26× → 1.5×+)
  - 변동성 적응력 향상

Implementation Time: 1-2시간
Risk: 낮음 (기존 로직 유지, 파라미터만 동적화)
```

### Phase 2: Entry Logic Improvement (진입 로직 개선)
```yaml
Changes:
  1. 브레이크아웃 확인 로직 추가
  2. 풀백 진입 조건 구현
  3. 시장 레짐 감지 추가

Expected Improvement:
  - 진입 정확도 향상 (+10-15% 승률)
  - 무의미한 신호 필터링 (-50% 허위 신호)
  - 거래 품질 향상

Implementation Time: 2-3시간
Risk: 중간 (로직 변경, 백테스트 필요)
```

### Phase 3: Exit Logic Overhaul (청산 로직 전면 개선)
```yaml
Changes:
  1. Donchian 밴드 기반 청산 추가
  2. 단계별 청산 구현
  3. 시장 레짐별 청산 전략 분리

Expected Improvement:
  - 수익 극대화 (트렌드 최대 활용)
  - 손실 최소화 (빠른 손절)
  - 수익 팩터 1.8-2.5× 목표

Implementation Time: 3-4시간
Risk: 높음 (전면 개편, 철저한 백테스트 필수)
```

---

## Part 6: Conclusion (결론)

### 현재 전략의 핵심 문제
1. **진입 시점 오류**: 브레이크아웃 없이 중간 영역에서 진입
2. **청산 방식 한계**: 시장 구조 무시한 고정 % 청산
3. **변동성 미반영**: ATR 기반 동적 조정 부재

### 권장 개선 방향
1. **Option B (Pullback)**: 브레이크아웃 확인 후 풀백 진입
2. **ATR 기반 TP/SL**: 변동성 적응형 청산
3. **단계별 청산**: 수익 일부 확보 + 추세 추종 병행

### 예상 성과 개선
| 지표 | 현재 | 개선 후 (예상) |
|------|------|---------------|
| 승률 | 50.2% | 55-60% |
| 수익 팩터 | 1.26× | 1.8-2.5× |
| 월간 수익률 | +62.99% (백테스트) | +80-120% (목표) |
| 드로다운 | 미확인 | 감소 예상 |

---

**Next Steps**:
1. Phase 1 (ATR 동적화) 즉시 구현
2. 백테스트 결과 확인
3. Phase 2-3 순차 적용

**Files**:
- Current: `scripts/production/donchian_scalping_bot.py`
- New: `scripts/production/donchian_pullback_bot_v9.py` (예정)
