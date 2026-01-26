# MS_ChoCH 연구 결과 vs 실제 백테스트 차이 분석

**날짜**: 2025-12-24
**상태**: 🔴 CRITICAL - 전략 폐기 권고

---

## 요약

| 구분 | 연구 결과 | 실제 백테스트 | 차이 |
|------|----------|--------------|------|
| **Full Period PnL** | +609.1% | -6.50% | **-615.6%** |
| **Win Rate** | 58.0% | 35.5% | **-22.5%p** |
| **Walk-Forward** | 70% 수익 | 0% 수익 | **-70%p** |

---

## 근본 원인 (5가지)

### 1. 🔴 Look-Ahead Bias (가장 치명적)

**연구 스크립트의 Swing Detection**:
```python
# shift(-1), shift(-2) = 미래 데이터 참조
df['swing_high'] = (df['high'] > df['high'].shift(1)) & \
                   (df['high'] > df['high'].shift(-1)) &  # ← 미래!
                   (df['high'] > df['high'].shift(2)) & \
                   (df['high'] > df['high'].shift(-2))    # ← 미래!
```

**월별 백테스트의 Swing Detection**:
```python
# center=True = 미래 데이터 포함
df['swing_high'] = df['high'].rolling(11, center=True).max() == df['high']  # ← 미래!
```

**영향**: Swing High/Low 감지 시 미래 가격을 "본" 상태에서 신호 생성
- 실시간에는 불가능한 신호
- 연구에서는 "완벽한" 고저점 포착
- 실제 트레이딩에서는 고저점 확정이 5봉 뒤에야 가능

### 2. 🟠 ChoCH 신호 로직 차이

**연구 스크립트** (단순 벡터화):
```python
long_base = df['bos_up'] & df['lower_low'].shift(5)
```

**월별 백테스트** (상태 머신):
```python
for i in range(len(df)):
    if df.iloc[i]['swing_high'] and prev_swing_high:
        if df.iloc[i]['high'] > prev_swing_high:
            if current_trend == -1:
                choch = 1
```

**영향**: 완전히 다른 신호 발생 패턴

### 3. 🟠 Entry 시점 차이

| 항목 | 연구 | 월별 백테스트 |
|------|------|--------------|
| Entry 시점 | 시그널+1봉 Open | 시그널봉 Close |
| 레버리지 | 4x 적용 | 미적용 |
| 수수료 | 0.04% 적용 | 미적용 |

### 4. 🟡 데이터 기간 불일치

| 구분 | 연구 데이터 | 월별 백테스트 |
|------|------------|--------------|
| 기간 | 2025-08-09 ~ 11-22 | 2025-09-25 ~ 12-24 |
| 일수 | 105일 | 90일 |
| 겹침 | - | ~58일만 겹침 |

### 5. 🟡 과적합 (Overfitting)

- Walk-Forward 검증을 동일 데이터셋에서 수행
- True Out-of-Sample (완전히 새로운 기간) 테스트 없음
- 파라미터 최적화와 검증이 같은 데이터 풀에서 진행

---

## 왜 연구에서 +609.1%가 나왔나?

1. **완벽한 Swing Point 감지**: 미래 데이터를 참조하여 정확한 고저점 파악
2. **이상적인 Entry**: 반전 신호를 "사후적으로" 확인
3. **낙관적 PnL 계산**: 4x 레버리지 복리 효과
4. **특정 기간 최적화**: BTC가 11만→8만→10만 이동한 기간

---

## 다른 전략에 미치는 영향

**동일한 Look-Ahead Bias 가능성 있는 전략들**:

1. **Swing High/Low 기반 전략**: 모든 전략이 영향 받음
2. **Price Action 패턴 전략**: `shift(-1)` 사용 시 영향
3. **Bollinger Squeeze**: `center=True` 사용 가능성

**검증 필요 전략 목록**:
- RSI Trend Filter (현재 Active)
- SuperTrend 5m
- 기타 CLAUDE.md에 언급된 모든 전략

---

## 권장 조치

### 즉각 조치 (완료)
- [x] MS_ChoCH Bot 중지
- [x] 모든 봇 중지
- [x] 근본 원인 분석

### 필요 조치
1. **모든 연구 스크립트 감사**: `shift(-`, `center=True` 검색
2. **RSI Trend Filter 검증**: Look-Ahead 없는지 확인
3. **SuperTrend 5m 검증**: Look-Ahead 없는지 확인
4. **새로운 백테스트 프레임워크 구축**:
   - Look-Ahead Bias 방지 설계
   - True Out-of-Sample 테스트 의무화
   - 프로덕션 로직과 동일한 백테스트 사용

---

## 기술적 교훈

### Look-Ahead Bias 방지 규칙
```python
# ❌ 금지
df['indicator'].shift(-1)  # 미래 참조
df.rolling(n, center=True)  # 중앙 정렬

# ✅ 허용
df['indicator'].shift(1)   # 과거 참조
df.rolling(n).xxx()        # 과거만 사용 (기본값)
```

### 검증 필수 체크리스트
1. 모든 indicator 계산에서 shift(-n) 검색
2. 모든 rolling()에서 center=True 검색
3. Entry 시점이 시그널 발생 "후"인지 확인
4. 완전히 새로운 기간에서 Out-of-Sample 테스트
5. 프로덕션 봇과 백테스트 로직 완전 일치 검증

---

## 결론

**MS_ChoCH 전략의 +609.1% 성과는 Look-Ahead Bias로 인한 허위 결과입니다.**

미래 데이터를 참조하여 "완벽한" 스윙 포인트를 감지했기 때문에
실제 트레이딩에서는 절대 재현 불가능한 결과였습니다.

모든 기존 전략에 대해 동일한 분석을 수행해야 합니다.

---

**작성자**: Claude
**검토 필요**: 다른 전략 Look-Ahead 감사
