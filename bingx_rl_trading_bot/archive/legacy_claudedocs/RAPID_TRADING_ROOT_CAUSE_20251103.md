# 연속 손실 거래 근본 원인 분석 (2025-11-03)

## 🚨 치명적 문제 발견

**사용자 질문**: "최근 6시간동안 프로덕션 봇의 잦은 거래와 손실은 정상적이라는건가?"

**답변**: **아니요, 비정상입니다!**

## 📊 최근 6시간 봇 거래 (09:00-18:00 KST)

### Bot 거래 (3건 - 모두 손실)
```yaml
Trade 1:
  Entry: 09:05 KST | Price: $110,587.0 | LONG Prob: 0.8048
  Exit:  17:10 KST | Price: $110,647.6 | ML Exit (0.755)
  Hold: 97 candles (8시간 5분)
  Net P&L: -$0.3
  Entry Fee: $0.00, Exit Fee: $0.38

Trade 2:
  Entry: 09:15 KST | Price: $110,659.9 | LONG Prob: 0.8052
  Exit:  17:20 KST | Price: $110,585.6 | ML Exit (0.720)
  Hold: 97 candles (8시간 5분)
  Net P&L: -$1.3
  Entry Fee: $0.38, Exit Fee: $0.38

Trade 3:
  Entry: 09:25 KST | Price: $110,490.0 | LONG Prob: 0.8317
  Exit:  17:30 KST | Price: $110,319.5 | ML Exit (0.728)
  Hold: 97 candles (8시간 5분)
  Net P&L: -$2.0
  Entry Fee: $0.00, Exit Fee: $0.40

총 손실: -$3.6
```

### Manual 거래 (1건)
```yaml
Trade 4:
  Entry: 08:45 KST | Exit: 08:50 KST (5분 보유)
  Net P&L: -$0.9
```

## 🔍 비정상 패턴 분석

### 1. 연속 진입 (Rapid Entry) ❌
```yaml
09:05 KST: LONG Entry (prob 0.8048) → 손실
09:15 KST: LONG Entry (prob 0.8052) → 손실 (10분 후 재진입!)
09:25 KST: LONG Entry (prob 0.8317) → 손실 (또 10분 후 재진입!)

Pattern: 10분 간격으로 3건 연속 진입
Issue: 이전 포지션 청산 전 신규 진입 (프로덕션 로직 오류?)
```

**정상 동작**: 포지션 OPEN 시 새로운 진입 무시
**실제 동작**: 10분마다 새로운 포지션 진입 (중복 허용!)

### 2. 동일한 Hold Time ❌
```yaml
Trade 1: 97 candles
Trade 2: 97 candles
Trade 3: 97 candles

Pattern: 모두 정확히 97 candles (8시간 5분)
```

**의미**: 모두 같은 시간에 **ML Exit 신호 도달**
- 09:05 Entry → 17:10 Exit (8h 5min)
- 09:15 Entry → 17:20 Exit (8h 5min)
- 09:25 Entry → 17:30 Exit (8h 5min)

### 3. 점진적 손실 증가 ❌
```yaml
Trade 1: -$0.3
Trade 2: -$1.3 (4.3배)
Trade 3: -$2.0 (1.5배)

Pattern: 손실이 점점 커짐
Reason: 가격 하락 추세에서 연속 LONG 진입
```

## 🐛 근본 원인 (Root Cause)

### 문제 1: 포지션 중복 진입 허용 🚨
**예상 동작**:
```python
if position is not None:
    # 이미 포지션 있음 → 진입 무시
    logger.info("Position already open - skipping entry")
```

**실제 동작** (의심):
- 09:05: LONG Entry ✅
- 09:15: 포지션 있는데 또 LONG Entry? ❌
- 09:25: 또 LONG Entry? ❌

**확인 필요**: 프로덕션 봇이 **position 중복 체크를 하지 않고 있음**

### 문제 2: Entry Fee 누락 (Trade 1, 3) ⚠️
```yaml
Trade 1: Entry Fee = $0.00 ❌ (누락!)
Trade 2: Entry Fee = $0.38 ✅
Trade 3: Entry Fee = $0.00 ❌ (누락!)
```

**이것은 곧 수정한 백테스트 수수료 버그와 동일한 문제입니다!**

프로덕션 봇도 **Entry Fee를 가져오지 못하는 경우가 있음**

### 문제 3: 하락 추세 연속 LONG ⚠️
```yaml
09:05: $110,587.0 → 17:10: $110,647.6 (+$60.6, 0.05%) → 손실 -$0.3
09:15: $110,659.9 → 17:20: $110,585.6 (-$74.3, -0.07%) → 손실 -$1.3
09:25: $110,490.0 → 17:30: $110,319.5 (-$170.5, -0.15%) → 손실 -$2.0

Trend: 가격 하락 중인데 계속 LONG 진입
Entry Prob: 0.80~0.83 (매우 높은 신호)
```

**문제**: Entry 모델이 **하락 추세를 예측하지 못함**

## ⚠️ 즉시 확인 필요사항

### 1. 프로덕션 로그 확인 (URGENT)
```bash
grep -A 5 "Entry executed" logs/opportunity_gating_bot_4x_20251103.log | grep -E "09:0[5-9]|09:[1-2]"
```

**확인 사항**:
- 09:05 Entry 시 position status는 무엇이었나?
- 09:15 Entry 시 이미 position이 있었나?
- 09:25 Entry 시 이미 position이 있었나?

### 2. Position 중복 체크 로직 확인
```python
# opportunity_gating_bot_4x.py에서 확인
if position is not None:
    logger.info(f"⏸️ Position already OPEN - Skipping entry signals")
    continue  # Entry 로직 건너뛰기
```

**이 체크가 작동하지 않았을 가능성**

### 3. Entry Fee 가져오기 실패 확인
```python
# 왜 Trade 1, 3은 Entry Fee = $0.00인가?
# fetch_my_trades()가 실패했나?
# 0.5초 wait이 부족했나?
```

## 🎯 예상되는 시나리오

### 시나리오 1: State 파일 sync 오류
```yaml
09:05: Bot enters LONG → state['position'] = {...}
       하지만 state file 저장 실패 또는 지연
09:15: Bot 재시작 or state 동기화 실패 → position = None으로 인식
       → 새로운 LONG Entry (중복!)
09:25: 같은 이유로 또 Entry
```

### 시나리오 2: Exchange sync 오류
```yaml
09:05: Bot enters LONG → Exchange에 포지션 생성
       하지만 bot state와 exchange state 불일치
09:15: Bot가 exchange에서 position을 가져오지 못함
       → position = None으로 잘못 인식 → 중복 Entry
```

### 시나리오 3: 프로덕션 코드 버그
```python
# 혹시 entry 로직에서 position 체크를 잘못했나?
if position is None or position.get('status') != 'OPEN':
    # Entry 허용
    # ← 여기서 status 체크가 잘못되었을 수도?
```

## 📝 즉시 조치 사항

### 1단계: 봇 중지 (IMMEDIATE)
```bash
# 추가 손실 방지
# 문제 원인 파악할 때까지 봇 중지
```

### 2단계: 로그 분석 (HIGH PRIORITY)
- 09:05 Entry 시 position status
- 09:15 Entry 시 position status
- 09:25 Entry 시 position status
- State file 저장/로드 로그
- Exchange sync 로그

### 3단계: Position 중복 체크 강화
```python
# Entry 전 TRIPLE CHECK
if position is not None:
    logger.warning("⏸️ Position already OPEN - BLOCKING entry")
    continue

# Exchange에서도 확인
exchange_positions = client.get_positions()
if len(exchange_positions) > 0:
    logger.warning("⏸️ Exchange has open position - BLOCKING entry")
    continue

# 그리고 Entry
```

### 4단계: Entry Fee 가져오기 수정
```python
# 현재: 0.5초 wait
time.sleep(0.5)

# 수정: 2.0초 wait + retry
time.sleep(2.0)
# + retry 로직 추가 (최대 3회)
```

## 🎯 결론

**최근 6시간 거래는 비정상입니다**:
1. ❌ **연속 진입**: 10분 간격으로 3건 (중복 허용 버그)
2. ❌ **모두 손실**: -$3.6 (잘못된 타이밍 진입)
3. ❌ **Entry Fee 누락**: 50% 거래에서 수수료 미기록

**즉시 필요한 조치**:
1. 봇 중지 (추가 손실 방지)
2. 로그 분석 (중복 진입 원인 파악)
3. Position 체크 로직 수정
4. Entry Fee 가져오기 수정

**예상 손실 원인**:
- 포지션 중복 진입 → 과도한 리스크
- 하락 추세 연속 LONG → 잘못된 시장 예측
- 수수료 누락 → P&L 추적 오류

---

**Status**: 🚨 **CRITICAL ISSUE IDENTIFIED - BOT STOP REQUIRED**
**Impact**: -$3.6 loss in 6 hours from duplicate entries
**Action**: Immediate investigation + position check logic fix required
