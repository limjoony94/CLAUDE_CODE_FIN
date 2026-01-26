# Backtest Verification Methodology (Standard)

**작성일**: 2025-12-31
**목적**: 모든 전략 연구에서 필수로 수행해야 하는 검증 프레임워크

---

## Executive Summary

전략 연구 시 **반드시 두 가지 검증**을 통과해야 프로덕션 배포 가능:

| 검증 유형 | 설명 | 필요 조건 |
|----------|------|----------|
| **Type 1: Signal Quality** | 신호 발생 시점에서 즉시 진입 (포지션 상태 무시) | 승률 ≥ 50%, 기대값 > 0 |
| **Type 2: Actual Trading** | 실제 거래 시뮬레이션 (포지션 있으면 진입 불가) | PnL > 0, WF 일관성 ≥ 50% |

**핵심 원칙**: 두 검증 모두 우수해야만 유효한 전략으로 인정

---

## Type 1: Signal Quality Verification (신호 품질 검증)

### 목적
**신호 자체의 예측력** 검증 - 포지션 상태와 무관하게 모든 신호의 품질 평가

### 방법론
```
1. 데이터 전체에서 모든 신호 발생 지점 탐지
2. 각 신호에 대해:
   - Entry: 신호 발생 다음 봉 Open
   - Exit: TP/SL 도달 또는 N봉 후 강제 청산
3. 모든 신호의 승률, 기대값 계산
```

### 구현 예시
```python
def verify_signal_quality(df, signal_func, tp_pct, sl_pct, max_bars=100):
    """
    Type 1: Signal Quality Verification
    - 모든 신호에 대해 독립적으로 평가 (포지션 상태 무시)
    - Entry: 신호 다음 봉 Open
    - Exit: High/Low로 TP/SL 체크
    """
    signals = []

    for i in range(1, len(df) - 1):
        signal = signal_func(df, i)  # 신호 감지
        if signal is None:
            continue

        # Entry: 다음 봉 Open (Look-Ahead Bias 방지!)
        entry_idx = i + 1
        entry_price = df.iloc[entry_idx]['open']

        # TP/SL 계산
        if signal == 'LONG':
            tp_price = entry_price * (1 + tp_pct / 100)
            sl_price = entry_price * (1 - sl_pct / 100)
        else:
            tp_price = entry_price * (1 - tp_pct / 100)
            sl_price = entry_price * (1 + sl_pct / 100)

        # Exit 탐색 (entry_idx + 1부터!)
        result = None
        for j in range(entry_idx + 1, min(entry_idx + max_bars, len(df))):
            bar = df.iloc[j]

            if signal == 'LONG':
                if bar['high'] >= tp_price:
                    result = {'outcome': 'TP', 'pnl_pct': tp_pct}
                    break
                elif bar['low'] <= sl_price:
                    result = {'outcome': 'SL', 'pnl_pct': -sl_pct}
                    break
            else:  # SHORT
                if bar['low'] <= tp_price:
                    result = {'outcome': 'TP', 'pnl_pct': tp_pct}
                    break
                elif bar['high'] >= sl_price:
                    result = {'outcome': 'SL', 'pnl_pct': -sl_pct}
                    break

        if result:
            signals.append({
                'signal': signal,
                'entry_idx': entry_idx,
                **result
            })

    # 통계 계산
    wins = sum(1 for s in signals if s['outcome'] == 'TP')
    win_rate = wins / len(signals) * 100 if signals else 0
    expected_value = sum(s['pnl_pct'] for s in signals) / len(signals) if signals else 0

    return {
        'total_signals': len(signals),
        'win_rate': win_rate,
        'expected_value': expected_value,
        'signals': signals
    }
```

### 통과 기준
| 메트릭 | 최소 기준 | 권장 |
|--------|----------|------|
| **총 신호 수** | ≥ 100 | ≥ 300 |
| **승률** | ≥ 50% | ≥ 55% |
| **기대값** | > 0 | > 0.5% |
| **양방향 수익** | LONG ≥ 0, SHORT ≥ 0 | 둘 다 > 0 |

---

## Type 2: Actual Trading Verification (실제 거래 시뮬레이션)

### 목적
**실제 거래 조건**에서의 성과 검증 - 한 번에 하나의 포지션만 보유

### 방법론
```
1. 초기 잔고로 시작
2. 신호 발생 시:
   - 포지션 없으면: 진입 (Entry@next_open)
   - 포지션 있으면: 무시 (진입 불가)
3. 포지션 있을 때:
   - TP/SL 도달 시 청산
   - 역방향 신호는 청산 후 진입 (선택적)
4. 복리 효과 반영 (잔고 기반 포지션 사이징)
```

### 구현 예시
```python
def verify_actual_trading(df, signal_func, tp_pct, sl_pct,
                          leverage=4, position_pct=0.25, fee_pct=0.05):
    """
    Type 2: Actual Trading Verification
    - 한 번에 하나의 포지션만 보유
    - 포지션 있으면 신호 무시
    - 복리 효과 반영
    """
    balance = 100  # 초기 잔고
    position = None
    trades = []

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]

        # 포지션 청산 체크 (매 봉마다)
        if position is not None:
            entry_price = position['entry_price']
            direction = position['direction']
            tp_price = position['tp_price']
            sl_price = position['sl_price']

            exit_price = None
            exit_reason = None

            # High/Low로 TP/SL 체크
            if direction == 'LONG':
                if row['high'] >= tp_price:
                    exit_price = tp_price
                    exit_reason = 'TP'
                elif row['low'] <= sl_price:
                    exit_price = sl_price
                    exit_reason = 'SL'
            else:  # SHORT
                if row['low'] <= tp_price:
                    exit_price = tp_price
                    exit_reason = 'TP'
                elif row['high'] >= sl_price:
                    exit_price = sl_price
                    exit_reason = 'SL'

            if exit_price:
                # PnL 계산 (수수료 포함)
                if direction == 'LONG':
                    pnl_pct = (exit_price - entry_price) / entry_price * 100
                else:
                    pnl_pct = (entry_price - exit_price) / entry_price * 100

                pnl_pct -= fee_pct * 2  # 양방향 수수료

                # 잔고 반영
                position_value = balance * position_pct * leverage
                pnl_dollar = position_value * (pnl_pct / 100)
                balance += pnl_dollar

                trades.append({
                    'direction': direction,
                    'pnl_pct': pnl_pct,
                    'pnl_dollar': pnl_dollar,
                    'exit_reason': exit_reason,
                    'balance': balance
                })
                position = None

        # 신호 체크 (포지션 없을 때만!)
        if position is None:
            signal = signal_func(df, i-1)  # 이전 봉에서 신호 확인

            if signal:
                entry_price = row['open']  # 현재 봉 Open에서 진입

                if signal == 'LONG':
                    tp_price = entry_price * (1 + tp_pct / 100)
                    sl_price = entry_price * (1 - sl_pct / 100)
                else:
                    tp_price = entry_price * (1 - tp_pct / 100)
                    sl_price = entry_price * (1 + sl_pct / 100)

                position = {
                    'direction': signal,
                    'entry_price': entry_price,
                    'tp_price': tp_price,
                    'sl_price': sl_price
                }

    # 통계 계산
    total_pnl = (balance - 100) / 100 * 100  # %
    wins = sum(1 for t in trades if t['pnl_pct'] > 0)
    win_rate = wins / len(trades) * 100 if trades else 0

    return {
        'total_pnl_pct': total_pnl,
        'num_trades': len(trades),
        'win_rate': win_rate,
        'final_balance': balance,
        'trades': trades
    }
```

### 통과 기준
| 메트릭 | 최소 기준 | 권장 |
|--------|----------|------|
| **Total PnL** | > 0% | > 30% |
| **Walk-Forward 일관성** | ≥ 50% (4/8) | ≥ 75% (6/8) |
| **Monte Carlo 수익 확률** | ≥ 80% | ≥ 95% |
| **Max Drawdown** | < 50% | < 30% |
| **양방향 수익** | 둘 다 ≥ 0 | 둘 다 > 0 |

---

## 두 검증의 관계

### 왜 두 가지 모두 필요한가?

| 시나리오 | Type 1 결과 | Type 2 결과 | 결론 |
|----------|------------|------------|------|
| **A: 이상적** | 높은 승률 | 높은 PnL | ✅ 배포 가능 |
| **B: 신호는 좋지만 거래 불가** | 높은 승률 | 낮은 PnL | ⚠️ 포지션 관리 개선 필요 |
| **C: 신호는 나쁘지만 운으로 수익** | 낮은 승률 | 높은 PnL | ❌ 과적합 가능성 |
| **D: 모두 나쁨** | 낮은 승률 | 낮은 PnL | ❌ 폐기 |

### 시나리오 B 해석 (높은 승률, 낮은 PnL)
- 신호 자체는 예측력이 있음
- 하지만 실제 거래에서 손실 발생
- 원인: 포지션 보유 중 좋은 신호를 놓침, 타이밍 불일치 등
- 해결: Exit 로직 개선, 부분 청산 등

### 시나리오 C 해석 (낮은 승률, 높은 PnL)
- 전체 신호의 승률은 낮음
- 하지만 실제 거래에서는 수익
- 원인: 운 좋게 좋은 신호만 잡음, 테스트 기간 특수성
- 해결: 신호 필터 추가, 더 긴 기간 테스트

---

## 필수 검증 체크리스트

### 연구 완료 전 확인 사항

```markdown
## Type 1: Signal Quality
- [ ] 총 신호 수: ___ (≥100 필수)
- [ ] 전체 승률: ___% (≥50% 필수)
- [ ] 전체 기대값: ___% (>0 필수)
- [ ] LONG 승률: ___%, 기대값: ___%
- [ ] SHORT 승률: ___%, 기대값: ___%
- [ ] Entry: 신호 다음 봉 Open 사용 확인

## Type 2: Actual Trading
- [ ] Total PnL: ___% (>0% 필수)
- [ ] Walk-Forward: ___/8 (≥4/8 필수)
- [ ] Monte Carlo 수익 확률: ___% (≥80% 필수)
- [ ] Max Drawdown: ___% (<50% 필수)
- [ ] LONG PnL: $___ (≥0 필수)
- [ ] SHORT PnL: $___ (≥0 필수)
- [ ] 복리 효과 반영 확인
- [ ] 수수료 포함 확인 (0.05% × 2)

## Entry/Exit 로직
- [ ] Entry: 신호 봉 다음 봉 Open (Look-Ahead Bias 없음)
- [ ] TP Check: bar.high/low 사용 (close 아님)
- [ ] SL Check: bar.high/low 사용 (close 아님)
- [ ] 동일 봉 Exit 체크 금지 (entry_idx + 1부터)
```

---

## Look-Ahead Bias 방지 가이드

### 금지 패턴
```python
# ❌ 신호 봉 Close에서 진입 (Look-Ahead!)
entry_price = row['close']  # row = 신호 발생 봉

# ❌ Close로 TP/SL 체크 (덜 정확함)
if current_price >= tp_price:  # current_price = close
    exit()

# ❌ 동일 봉에서 Entry + Exit 체크
for i in range(len(df)):
    if signal_at(i):
        entry_price = df.iloc[i]['open']  # Entry
        if df.iloc[i]['high'] >= tp_price:  # ❌ 동일 봉!
            exit()
```

### 올바른 패턴
```python
# ✅ 신호 다음 봉 Open에서 진입
signal_idx = i  # 신호 발생
entry_idx = i + 1  # 진입은 다음 봉
entry_price = df.iloc[entry_idx]['open']

# ✅ High/Low로 TP/SL 체크
if direction == 'LONG':
    if bar['high'] >= tp_price:
        exit_price = tp_price  # TP Hit
    elif bar['low'] <= sl_price:
        exit_price = sl_price  # SL Hit

# ✅ Entry 봉 이후부터 Exit 체크
for j in range(entry_idx + 1, len(df)):  # entry_idx + 1 부터!
    if check_exit(j):
        exit()
```

---

## 검증 실패 시 조치

### Type 1 실패 (낮은 신호 품질)
1. 신호 조건 강화 (더 엄격한 필터)
2. 다른 인디케이터 추가
3. 타임프레임 변경
4. 전략 폐기 검토

### Type 2 실패 (실제 거래 손실)
1. Exit 로직 개선 (BE, Trail 등)
2. 포지션 사이징 조정
3. TP/SL 최적화
4. 역방향 신호 처리 방식 변경

### 양쪽 모두 실패
→ **전략 폐기** - 다른 접근 방식 필요

---

## 관련 문서

| 문서 | 내용 |
|------|------|
| `LOOK_AHEAD_BIAS_AUDIT_20251224.md` | Look-Ahead Bias 감사 결과 |
| `ADX_FILTER_FINAL_VERIFICATION_20251231.md` | ADX 필터 검증 사례 |
| `RSI_MARTINGALE_DISCREPANCY_ANALYSIS_20251225.md` | 연구-프로덕션 불일치 사례 |

---

**작성자**: Claude AI Assistant
**검토 상태**: 사용자 확인 필요
**버전**: 1.0
