# RSI Martingale 연구 vs 프로덕션 백테스트 불일치 분석

**분석일**: 2025-12-25
**분석자**: Claude Code
**중요도**: 🔴 CRITICAL

---

## 1. 불일치 요약

| 출처 | Daily Return | Total PnL (30d) | Max DD | Win Rate |
|------|-------------|-----------------|--------|----------|
| **이전 세션 주장** | +1.37% | ~+41% | ~25% | ~35% |
| **프로덕션 백테스트** | **-3.33%** | **-100%** | **100%** | **27.3%** |
| **Gap** | **-4.70%p** | **-141%p** | **+75%p** | **-7.7%p** |

---

## 2. 핵심 불일치 원인

### 2.1 RSI 계산 방식 차이 (CRITICAL)

**연구 스크립트 (SMA 기반)**:
```python
# martingale_optimization.py, highfreq_aggressive_research.py
delta = df['close'].diff()
gain = delta.where(delta > 0, 0).rolling(window=7).mean()  # ← SMA
loss = (-delta.where(delta < 0, 0)).rolling(window=7).mean()  # ← SMA
rs = gain / (loss + 1e-10)
rsi = 100 - (100 / (1 + rs))
```

**프로덕션 봇 (EWM 기반)**:
```python
# rsi_martingale_bot.py
delta = close.diff()
gain = delta.where(delta > 0, 0.0)
loss = (-delta).where(delta < 0, 0.0)
avg_gain = gain.ewm(com=period-1, min_periods=period).mean()  # ← EWM
avg_loss = loss.ewm(com=period-1, min_periods=period).mean()  # ← EWM
```

**영향**:
- **SMA RSI**: 모든 기간에 동일 가중치 → 더 부드러운 RSI 곡선
- **EWM RSI**: 최근 데이터에 높은 가중치 → 더 반응적인 RSI 곡선
- **신호 타이밍**: 동일 조건에서 다른 시점에 신호 발생
- **결과**: 완전히 다른 거래 집합 생성

### 2.2 파라미터 불일치

| 파라미터 | 연구 (best result) | 프로덕션 | 차이 |
|----------|-------------------|----------|------|
| **TP** | 1.5% | 2.0% | +0.5% |
| **SL** | 1.0% | 0.7% | -0.3% |
| **R:R** | 1.5:1 | 2.86:1 | +1.36 |
| **Base Position** | 8% | 10% | +2% |
| **Leverage** | 5x | 10x | +5x |
| **Balance Cap** | 60% | 100% (10x) | +40% |
| **Max Martingale** | 8x | 8x | 동일 |

### 2.3 검증되지 않은 이전 세션 주장

이전 세션에서 언급된 성과:
- ✅ "Walk-Forward 5/6 (83%) profitable windows"
- ✅ "+1.37% daily return"
- ✅ "27 strategies meeting all conditions"

**검증 시도 결과**:
- 해당 연구 스크립트 발견 불가
- `martingale_optimization.py` 최고 성과: **+0.134% daily** (목표의 27%)
- 실제 목표 달성 전략 수: **0개**

**추정 원인**:
1. 연구 스크립트가 저장되지 않음
2. Look-Ahead Bias가 포함된 연구 결과였을 가능성
3. 다른 파라미터 조합이었을 가능성

---

## 3. 상세 비교

### 3.1 신호 생성 로직

| 항목 | 연구 | 프로덕션 |
|------|------|----------|
| **RSI 함수** | `rolling().mean()` | `ewm().mean()` |
| **LONG 조건** | `rsi > 25 & prev_rsi <= 25` | `prev_rsi < 25 & rsi >= 25` |
| **SHORT 조건** | `rsi < 75 & prev_rsi >= 75` | `prev_rsi > 75 & rsi <= 75` |

**미묘한 차이**:
- 연구: `rsi > 25` (strictly greater)
- 프로덕션: `rsi >= 25` (greater or equal)

### 3.2 백테스트 규칙

| 규칙 | 연구 | 프로덕션 | 일치 |
|------|------|----------|------|
| Entry 시점 | 다음 봉 OPEN | 다음 봉 OPEN | ✅ |
| Exit 감지 | HIGH/LOW | HIGH/LOW | ✅ |
| 같은 봉 청산 | bars_held >= 1 | 확인 필요 | ⚠️ |
| 수수료 | 0.05% × 2 | 0.05% × 2 | ✅ |
| 포지션 사이징 | 잔고 기반 | 잔고 기반 | ✅ |

### 3.3 실제 거래 결과 비교

**7일 백테스트 (프로덕션 로직)**:
- 신호 수: 85 (LONG 35, SHORT 50)
- 실제 거래: 19
- Win Rate: 21.1%
- Daily PnL: -6.10%

**문제점**:
- 낮은 Win Rate (21.1% vs 예상 ~35%)
- 높은 손실률
- 마틴게일 복구 실패 (연속 손실 누적)

---

## 4. 근본 원인

### 4.1 RSI 계산 방식이 가장 중요

```
SMA RSI vs EWM RSI 영향:
├── 서로 다른 RSI 값 생성
├── 서로 다른 crossover 시점
├── 서로 다른 신호 발생
└── 완전히 다른 거래 결과
```

**예시**:
같은 가격 데이터에서:
- SMA RSI(7) = 27.5 → 신호 발생
- EWM RSI(7) = 23.8 → 신호 미발생

### 4.2 레버리지/포지션 크기 증가

```
연구: 5x leverage × 8% = 40% notional
프로덕션: 10x leverage × 10% = 100% notional

손실 영향:
- 연구: -1% 가격 변동 → -40% × 1% = -0.4% 잔고 손실
- 프로덕션: -1% 가격 변동 → -100% × 1% = -1% 잔고 손실

→ 프로덕션이 2.5배 더 민감
```

### 4.3 마틴게일 + 높은 레버리지 = 파산 위험

```
연속 손실 시나리오 (SL 0.7%):

Loss 1: 100% × 1x × 0.7% = 0.7% 잔고 손실
Loss 2: 100% × 2x × 0.7% = 1.4% 잔고 손실 (누적 2.1%)
Loss 3: 100% × 4x × 0.7% = 2.8% 잔고 손실 (누적 4.9%)
Loss 4: 100% × 8x × 0.7% = 5.6% 잔고 손실 (누적 10.5%)
Loss 5: 100% × 8x × 0.7% = 5.6% 잔고 손실 (누적 16.1%)
...

→ 10연속 손실 시 ~50% 잔고 손실
→ 20연속 손실 시 파산
```

---

## 5. 교훈 및 방지 가이드

### 5.1 RSI 계산 표준화 (CRITICAL)

```python
# ✅ 표준 RSI (Wilder's Smoothing = EWM)
def calculate_rsi_standard(df, period=14):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    # Wilder's smoothing = EWM with com=period-1
    avg_gain = gain.ewm(com=period-1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period-1, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ❌ 단순 SMA RSI (결과가 다름)
def calculate_rsi_sma(df, period=14):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    ...
```

### 5.2 연구-프로덕션 일치 검증 체크리스트

새 전략 연구 시 **반드시** 확인:

| 항목 | 검증 방법 |
|------|----------|
| **인디케이터 계산** | 연구/프로덕션 코드 diff |
| **신호 로직** | 동일 데이터에서 신호 비교 |
| **백테스트 규칙** | CLAUDE.md 규칙 준수 확인 |
| **파라미터** | 연구 결과 파라미터 = 프로덕션 설정 |
| **Look-Ahead Bias** | shift(-n), center=True 검색 |
| **수수료/슬리피지** | 포함 여부 확인 |

### 5.3 백테스트 결과 검증 프로세스

```
1. 연구 완료
   ↓
2. 프로덕션 로직으로 백테스트 재실행 (필수!)
   ↓
3. 결과 비교 (10% 이상 차이 시 원인 분석)
   ↓
4. 원인 파악 및 수정
   ↓
5. 최종 검증 후 배포
```

### 5.4 안전한 연구 스크립트 템플릿

```python
"""
안전한 백테스트 연구 템플릿
==========================
✅ 프로덕션과 동일한 인디케이터 계산
✅ Look-Ahead Bias 없음
✅ 올바른 백테스트 규칙
"""

# 1. 프로덕션과 동일한 RSI 계산
def calculate_rsi(df, period=7):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period-1, min_periods=period).mean()  # EWM!
    avg_loss = loss.ewm(com=period-1, min_periods=period).mean()  # EWM!
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# 2. Look-Ahead Bias 검증
def validate_no_lookahead(df):
    """금지 패턴 검출"""
    code = inspect.getsource(generate_signals)
    if 'shift(-' in code:
        raise ValueError("Look-Ahead Bias detected: shift(-n)")
    if 'center=True' in code:
        raise ValueError("Look-Ahead Bias detected: center=True")

# 3. 백테스트 규칙 준수
# - Entry: 다음 봉 OPEN
# - Exit: HIGH/LOW로 감지
# - bars_held >= 1
# - 수수료 0.05% × 2
```

---

## 6. 결론

### 6.1 RSI Martingale 전략 상태

| 상태 | 설명 |
|------|------|
| **연구 결과** | 검증 불가 (스크립트 미발견, Look-Ahead 의심) |
| **프로덕션 백테스트** | **손실 전략** (-100% in 30 days) |
| **권장 조치** | ❌ **배포 금지** |

### 6.2 핵심 교훈

1. **연구와 프로덕션의 인디케이터 계산은 반드시 동일해야 함**
2. **RSI는 EWM (Wilder's Smoothing)이 표준** - SMA 사용 금지
3. **모든 연구 결과는 프로덕션 로직으로 재검증 필수**
4. **높은 레버리지 + 마틴게일 = 파산 위험**

### 6.3 향후 조치

1. ✅ RSI Martingale 봇 **배포 금지**
2. ⚠️ 모든 연구 스크립트 RSI 계산 방식 표준화
3. ⚠️ 기존 활성 봇 인디케이터 일치 검증
4. 📝 CLAUDE.md에 방지 가이드 추가

---

## 관련 파일

| 파일 | 역할 |
|------|------|
| `scripts/production/rsi_martingale_bot.py` | 프로덕션 봇 (EWM RSI) |
| `scripts/analysis/rsi_martingale_production_backtest.py` | 프로덕션 로직 백테스트 |
| `scripts/analysis/martingale_optimization.py` | 연구 스크립트 (SMA RSI) |
| `scripts/analysis/highfreq_aggressive_research.py` | 연구 스크립트 (SMA RSI) |

---

**문서 작성**: Claude Code (2025-12-25)
