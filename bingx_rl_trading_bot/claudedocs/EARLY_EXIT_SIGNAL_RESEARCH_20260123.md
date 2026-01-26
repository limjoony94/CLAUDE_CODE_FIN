# Early Exit Signal Research Report

**Date**: 2026-01-23
**Data**: 25,920 candles (90 days, 5m timeframe)
**Result**: **IMPLEMENTED - v1.3 ultra_conservative mode**
**Status**: ✅ Production deployed

---

## Executive Summary

TP/SL 조정 대신 반전 신호 감지 시 조기 청산하는 전략을 연구한 결과, **ultra_conservative** 모드가 베이스라인 대비 **+93% 수익률 향상**을 보여 프로덕션 적용을 권장합니다.

---

## 1. Early Exit Signal Concept

### Logic
- **LONG position**: 강한 베어리시 캔들(BD, MD) 출현 시 조기 청산
- **SHORT position**: 강한 불리시 캔들(BU, MU) 출현 시 조기 청산

### Exit Priority
1. TP/SL hit → Standard exit (highest priority)
2. Early exit signal → Close at next candle open
3. Max hold time → Forced exit (safety net)

### Reversal Signal Types

| Code | Type | Used for |
|------|------|----------|
| BD | BIG_DOWN | Exit LONG (bearish reversal) |
| MD | MARUBOZU_DOWN | Exit LONG (bearish reversal) |
| BU | BIG_UP | Exit SHORT (bullish reversal) |
| MU | MARUBOZU_UP | Exit SHORT (bullish reversal) |

---

## 2. Sensitivity Configurations

| Mode | Bearish Exit | Bullish Exit | Confirm | Min Profit |
|------|--------------|--------------|---------|------------|
| aggressive | BD, MD, GS, IH | BU, MU, DF, H | 1 candle | 0% |
| moderate | BD, MD | BU, MU | 1 candle | 0% |
| conservative | BD | BU | 1 candle | 0% |
| profit_only | BD, MD | BU, MU | 1 candle | 0.5% |
| **ultra_conservative** | BD | BU | **2 candles** | **0.3%** |

---

## 3. Backtest Results

### Performance Comparison

| Config | Trades | WR | Return | MaxDD | Sharpe | Early Exit % |
|--------|--------|-----|--------|-------|--------|--------------|
| **Baseline** | 97 | 51.5% | +49.9% | 45.8% | 2.91 | N/A |
| aggressive | 163 | 47.2% | -40.9% | 46.6% | -2.87 | 85.3% |
| moderate | 159 | 57.2% | -14.0% | 36.8% | -0.59 | 84.9% |
| conservative | 150 | 62.0% | -3.3% | 32.4% | 0.13 | 79.3% |
| profit_only | 138 | 77.5% | +10.7% | 35.9% | 0.92 | 74.6% |
| **ultra_conservative** | **108** | **62.0%** | **+143.0%** | **36.6%** | **5.26** | **23.1%** |

### Key Insight

- **aggressive/moderate**: 너무 많은 조기 청산 (85%+) → 손실
- **conservative**: 조기 청산 과다 (79%) → 중립적 결과
- **profit_only**: 이익 시에만 청산 → 약간의 개선
- **ultra_conservative**: **2개 연속 반전 캔들 + 최소 0.3% 이익** → **최적**

---

## 4. Exit Reason Breakdown (ultra_conservative)

| Exit Reason | Count | % | Description |
|-------------|-------|---|-------------|
| MAX_HOLD | 39 | 36.1% | 최대 보유 시간 도달 |
| TP | 26 | 24.1% | 전체 TP 도달 |
| SL | 18 | 16.7% | 손절 |
| EARLY_BU | 15 | 13.9% | 불리시 반전으로 숏 청산 |
| EARLY_BD | 10 | 9.3% | 베어리시 반전으로 롱 청산 |

**분석**: 23.1%만 조기 청산, 나머지는 정상적인 TP/SL/MAX_HOLD

---

## 5. Walk-Forward Validation (6 Folds)

| Fold | Baseline | ultra_conservative | Delta |
|------|----------|-------------------|-------|
| 1 | -16.6% | **-5.4%** | +11.2% |
| 2 | -13.9% | -14.2% | -0.3% |
| 3 | -1.8% | **+23.2%** | +25.0% |
| 4 | +88.1% | **+94.4%** | +6.3% |
| 5 | +2.7% | **+3.6%** | +0.9% |
| 6 | +21.7% | **+33.5%** | +11.8% |

**WF Result**: Baseline 3/6 → ultra_conservative **4/6** profitable folds

---

## 6. Improvement Summary

| Metric | Baseline | ultra_conservative | Change |
|--------|----------|-------------------|--------|
| Return | +49.9% | +143.0% | **+93.1%** |
| Win Rate | 51.5% | 62.0% | **+10.5%** |
| Max DD | 45.8% | 36.6% | **-9.2%** |
| Sharpe | 2.91 | 5.26 | **+81%** |
| WF Profitable | 3/6 | 4/6 | **+1 fold** |

---

## 7. Recommendation

### Verdict: APPLY

**ultra_conservative** 모드를 Pattern 5m 봇에 적용합니다.

### Implementation Logic

```python
# Early exit conditions
EARLY_EXIT_CONFIG = {
    'enabled': True,
    'bearish_types': ['BD'],  # Big Down only
    'bullish_types': ['BU'],  # Big Up only
    'confirm_candles': 2,     # Need 2 consecutive signals
    'min_profit_pct': 0.3,    # Must be at least 0.3% in profit
}

def check_early_exit(position, current_candle, reversal_count):
    """Check if early exit should trigger."""
    direction = position['direction']
    entry_price = position['entry_price']
    current_price = current_candle['close']

    # Calculate unrealized PnL
    if direction == 'LONG':
        pnl = (current_price / entry_price - 1) * 100 * LEVERAGE
        reversal_types = EARLY_EXIT_CONFIG['bearish_types']
    else:  # SHORT
        pnl = (entry_price / current_price - 1) * 100 * LEVERAGE
        reversal_types = EARLY_EXIT_CONFIG['bullish_types']

    # Check reversal signal
    type_code = current_candle['type_code']
    if type_code in reversal_types:
        reversal_count += 1
        if (reversal_count >= EARLY_EXIT_CONFIG['confirm_candles']
            and pnl >= EARLY_EXIT_CONFIG['min_profit_pct']):
            return True, reversal_count
    else:
        reversal_count = 0

    return False, reversal_count
```

### Why It Works

1. **2-candle confirmation**: 노이즈 필터링, 진짜 반전만 감지
2. **0.3% min profit**: 손실 상태에서 조기 청산 방지
3. **BD/BU only**: 가장 강한 반전 신호만 사용
4. **23% exit rate**: 적절한 빈도, 과도한 거래 방지

---

## 8. Implementation (v1.3)

### ✅ 구현 완료 (2026-01-23)

| 파일 | 변경 내용 |
|------|----------|
| `constants.py` | `EARLY_EXIT_CONFIG` 추가, `BOT_VERSION = "1.3"` |
| `signals.py` | `check_early_exit_signal()` 함수 추가 |
| `position_close.py` | `close_position_market()` 함수 추가 |
| `position.py` | `close_position_market` 재export |
| `bot.py` | `_process_existing_position()` 로직 수정 |

### 처리 흐름 (v1.3)

```
포지션 보유 중
    ↓
check_position_status() → TP/SL hit 확인
    ↓
캔들 데이터 fetch + 분류
    ↓
check_early_exit_signal()
    ├── type_code in reversal_types?
    │   ├── YES → reversal_count++
    │   └── NO → reversal_count = 0
    ↓
reversal_count >= 2 AND PnL >= 0.3%?
    ├── YES → close_position_market(EARLY_BD/EARLY_BU)
    └── NO → 계속 보유
```

### Position State 추가 필드

```python
position = {
    ...
    'reversal_count': 0,  # v1.3 신규: 연속 반전 캔들 카운터
}
```

---

## 9. Files Reference

### 연구 스크립트
| File | Description |
|------|-------------|
| `scripts/analysis/early_exit_signal_research.py` | 메인 연구 스크립트 |
| `scripts/analysis/early_exit_validation.py` | 추가 검증 (Monte Carlo, 파라미터 민감도) |
| `scripts/analysis/early_exit_direction_test.py` | LONG/SHORT 방향별 분석 |

### 프로덕션 코드 (v1.3)
| File | Description |
|------|-------------|
| `scripts/production/pattern_5m/constants.py` | EARLY_EXIT_CONFIG 정의 |
| `scripts/production/pattern_5m/signals.py` | check_early_exit_signal() |
| `scripts/production/pattern_5m/position_close.py` | close_position_market() |
| `scripts/production/pattern_5m/bot.py` | 메인 루프 통합 |

---

## 10. v1.0 → v1.3 변경 요약

| 항목 | v1.0 | v1.3 |
|------|------|------|
| 청산 조건 | TP/SL만 | TP/SL + Early Exit |
| 반전 감지 | 없음 | BD/BU 캔들 감지 |
| Exit Reason | TP, SL, MAX_HOLD | + EARLY_BD, EARLY_BU |
| 예상 승률 | 59% | 62% |
| 예상 수익률 | +50% | +100%+ |

---

## Conclusion

Early Exit Signal 전략은 **ultra_conservative** 설정으로 적용 시 **+93% 수익률 향상**을 제공합니다.

핵심 성공 요인:
1. 보수적인 2-candle 확인
2. 최소 이익 조건 (0.3%)
3. 강한 반전 신호만 사용 (BD/BU)

**상태**: ✅ v1.3으로 프로덕션 배포 완료
