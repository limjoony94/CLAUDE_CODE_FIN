# 모니터링 시스템 전면 개편 제안 (2025-11-15)

## 📋 Executive Summary

**문제**: 모니터링 시스템이 Entry/Exit 4-모델 구조 기준으로 설정되어 있으나, 실제 배포는 Buy/Sell 2-모델 구조
**영향**: Expected Values 불일치 → 잘못된 알림 트리거 → 성과 오판
**제안**: 모니터링 시스템 전면 개편 (Expected Values, Alert Thresholds, 추적 로직)

---

## 🚨 주요 불일치 사항

### 1. Expected Performance Values (Lines 73-79)

**현재 설정** (Entry/Exit 구조 기준):
```python
EXPECTED_RETURN_5D = 0.2521      # 25.21% per 5-day window
EXPECTED_WIN_RATE = 0.723        # 72.3%
EXPECTED_TRADES_PER_DAY = 4.6    # 4.64 trades/day
EXPECTED_LONG_PCT = 0.618        # 61.8% LONG
EXPECTED_SHORT_PCT = 0.382       # 38.2% SHORT
EXPECTED_SHARPE = 6.610          # Annualized Sharpe
```

**실제 배포 모델 성과** (Buy/Sell 구조):
```yaml
Validation Period: Oct 9 - Nov 14, 2025 (36 days)
Total Return: +3.82%
Monthly Equivalent: ~3.2%
Win Rate: 66.11%
Trades/Day: 8.3
LONG/SHORT: 52.5% / 47.5%
Profit Factor: 1.12×
```

**불일치 영향**:
- ❌ Win Rate: 72.3% 기대 vs 66.11% 실제 → 잘못된 "Low Win Rate" 알림
- ❌ Trades/Day: 4.6 기대 vs 8.3 실제 → 잘못된 "High Frequency" 알림
- ❌ LONG%: 61.8% 기대 vs 52.5% 실제 → 잘못된 "LONG Bias" 알림
- ❌ Return: 25.21%/5d 기대 vs 3.82%/36d 실제 → 성과 오판

---

### 2. Strategy Description (Line 931)

**현재**:
```python
print(f"│ Strategy: Opportunity Gating (SHORT gated by Expected Value)│")
```

**실제**:
- Buy/Sell 2-모델 구조
- Exit: Opposite Signal (반대 신호로 청산)
- No EV Gating (Opportunity Gating 없음)

**수정 필요**:
```python
print(f"│ Strategy: Buy/Sell Signal Structure (Opposite Signal Exit)│")
```

---

### 3. Exit Strategy Description (Line 934)

**현재**:
```python
print(f"│ Exit Strategy: ML Exit + Emergency (ML: 0.75/0.75, SL: -3%, MaxHold: 10h)│")
```

**실제**:
- ML Exit = Opposite Signal (Sell >= 0.60 closes LONG, Buy >= 0.60 closes SHORT)
- Exit Threshold: 0.60 (not 0.75)
- Exit Model: Opposite direction model (Buy = SHORT exit, Sell = LONG exit)

**수정 필요**:
```python
print(f"│ Exit Strategy: Opposite Signal (Buy: 0.60, Sell: 0.60, SL: -3%, MaxHold: 10h)│")
```

---

### 4. Model Configuration (Lines 66-69)

**현재**:
```python
# Entry Models: xgboost_*_entry_enhanced_20251024_012445.pkl (85/79 features)
# Exit Models: xgboost_*_exit_oppgating_improved_20251024_04XXXX.pkl (27 features)
# Configuration: Entry 0.80/0.80, Exit 0.75/0.75
```

**실제**:
```yaml
Models:
  - xgboost_buy_model_171features_20251114_143119.pkl (171 features)
  - xgboost_sell_model_171features_20251114_143119.pkl (171 features)

Configuration:
  - Entry: Buy >= 0.60 (LONG), Sell >= 0.60 (SHORT)
  - Exit: Opposite signal >= 0.60
  - No separate Exit models (Buy/Sell handle both Entry and Exit)
```

---

### 5. Alert Thresholds (Lines 82-90)

**현재 설정**:
```python
ALERT_MIN_WIN_RATE = 0.60        # Min 60% (expected 72.3%)
ALERT_SHORT_RATIO_MIN = 0.25     # SHORT < 25% (expected 38.2%)
ALERT_SHORT_RATIO_MAX = 0.55     # SHORT > 55% (expected 38.2%)
ALERT_TRADES_PER_DAY_MIN = 3.0   # < 3.0 trades/day (expected 4.6)
```

**권장 조정** (Buy/Sell 구조 기준):
```python
ALERT_MIN_WIN_RATE = 0.60        # Min 60% (expected 66.11%, allow -9% degradation) ✅ OK
ALERT_SHORT_RATIO_MIN = 0.35     # SHORT < 35% (expected 47.5%, -26% degradation)
ALERT_SHORT_RATIO_MAX = 0.60     # SHORT > 60% (expected 47.5%, +26% degradation)
ALERT_TRADES_PER_DAY_MIN = 6.0   # < 6.0 trades/day (expected 8.3, -28% degradation)
ALERT_TRADES_PER_DAY_MAX = 12.0  # > 12.0 trades/day (expected 8.3, +45% increase)
```

---

## 📊 필수 개편 항목

### Priority 1: Expected Values (CRITICAL)

**현재 → 권장**:
```python
# Returns
EXPECTED_RETURN_5D = 0.2521 → 0.053  # 5.3% per 5-day (3.82% / 36d * 5d)
EXPECTED_RETURN_MONTHLY = None → 0.032  # 3.2% monthly

# Win Rate
EXPECTED_WIN_RATE = 0.723 → 0.6611  # 66.11%

# Trade Frequency
EXPECTED_TRADES_PER_DAY = 4.6 → 8.3  # 8.3 trades/day

# Direction Mix
EXPECTED_LONG_PCT = 0.618 → 0.525  # 52.5% LONG
EXPECTED_SHORT_PCT = 0.382 → 0.475  # 47.5% SHORT

# Risk Metrics (재계산 필요)
EXPECTED_SHARPE = 6.610 → TBD  # Backtest 기반 재계산
EXPECTED_PROFIT_FACTOR = None → 1.12  # 1.12×

# Gate Metrics (제거 필요)
EXPECTED_GATE_BLOCK_RATE = 0.382 → REMOVE  # No gating in Buy/Sell structure
```

---

### Priority 2: 청산 메커니즘 추적 (HIGH)

**현재 추적**:
- ML Exit (별도 Exit 모델 확률)
- Stop Loss
- Max Hold

**필요 추적** (Buy/Sell 구조):
```python
Exit Mechanisms:
  1. Opposite Signal Exit:
     - LONG: Sell >= 0.60
     - SHORT: Buy >= 0.60
     - Expected: ~70-80% of exits

  2. Stop Loss:
     - Balance-based -3%
     - Expected: ~15-20% of exits

  3. Max Hold:
     - 120 candles (10 hours)
     - Expected: ~5-10% of exits
```

**추가 추적 필요**:
- Buy/Sell 확률 분포 (Entry threshold 0.60 대비)
- Opposite signal 청산 성공률
- Buy >= 0.60 AND Sell >= 0.60 동시 발생 빈도
- 청산 후 재진입까지 대기 시간

---

### Priority 3: 신호 품질 추적 (MEDIUM)

**현재 추적**:
- LONG/SHORT Entry probabilities
- Signal threshold exceedance

**추가 필요**:
```python
Buy/Sell Signal Tracking:
  1. Buy Signal Quality:
     - Buy >= 0.60 빈도
     - Buy 확률 분포 (0.60-0.70, 0.70-0.80, 0.80+)
     - Buy signal → LONG entry 전환율
     - Buy signal → SHORT exit 전환율

  2. Sell Signal Quality:
     - Sell >= 0.60 빈도
     - Sell 확률 분포 (0.60-0.70, 0.70-0.80, 0.80+)
     - Sell signal → SHORT entry 전환율
     - Sell signal → LONG exit 전환율

  3. Signal Conflicts:
     - Buy >= 0.60 AND Sell >= 0.60 동시 발생
     - Conflicting signal 처리 (둘 다 진입 불가)
```

---

### Priority 4: Alert Thresholds (MEDIUM)

**권장 조정**:
```python
# Win Rate
ALERT_MIN_WIN_RATE = 0.60  # Expected 66.11%, allow -9% degradation ✅ KEEP

# Direction Mix
ALERT_SHORT_RATIO_MIN = 0.35  # Expected 47.5%, allow -26% degradation (was 0.25)
ALERT_SHORT_RATIO_MAX = 0.60  # Expected 47.5%, allow +26% increase (was 0.55)

# Trade Frequency
ALERT_TRADES_PER_DAY_MIN = 6.0   # Expected 8.3, allow -28% degradation (was 3.0)
ALERT_TRADES_PER_DAY_MAX = 12.0  # Expected 8.3, allow +45% increase (NEW)

# Gating Metrics (REMOVE)
ALERT_GATE_BLOCK_MIN = REMOVE  # No gating in Buy/Sell structure
ALERT_GATE_BLOCK_MAX = REMOVE  # No gating in Buy/Sell structure

# Exit Mechanisms (NEW)
ALERT_OPPOSITE_SIGNAL_EXIT_MIN = 0.60  # Expected ~70-80%, alert if < 60%
ALERT_STOP_LOSS_RATE_MAX = 0.30        # Expected ~15-20%, alert if > 30%
ALERT_MAX_HOLD_RATE_MAX = 0.15         # Expected ~5-10%, alert if > 15%
```

---

## 🛠️ 구현 로드맵

### Phase 1: Critical Updates (1-2 hours)
1. ✅ Expected Values 전면 업데이트
2. ✅ Strategy Description 수정
3. ✅ Alert Thresholds 재조정
4. ✅ Gating Metrics 제거

### Phase 2: 청산 메커니즘 추적 (2-3 hours)
1. Opposite Signal Exit 추적 로직 추가
2. Buy/Sell 확률 분포 추적
3. Exit mechanism 분포 계산
4. 청산 성공률 추적

### Phase 3: 신호 품질 추적 (2-3 hours)
1. Buy/Sell signal quality metrics
2. Signal conflict detection
3. Entry/Exit 전환율 추적
4. 확률 분포 분석

### Phase 4: 문서화 및 검증 (1 hour)
1. 변경 사항 문서화
2. 모니터링 가이드 업데이트
3. 알림 규칙 검증
4. 백테스트 vs 실제 비교 로직

**총 소요 시간**: 6-9 hours

---

## 📝 변경 필요 파일

### quant_monitor.py
**Lines 66-90**: Expected Values + Alert Thresholds 전면 수정
**Lines 931-937**: Strategy Description 수정
**Lines 500-550**: Signal tracking 로직 확장 (Buy/Sell)
**Lines 700-800**: Exit mechanism tracking 로직 추가
**Lines 900-950**: Display logic 업데이트

### config_sync.py (Optional)
- Buy/Sell 구조 validation 추가
- Model file 경로 검증 (171 features)
- Threshold 검증 (0.60/0.60)

---

## ✅ 권장 조치

### Immediate (지금 당장)
1. **Expected Values 긴급 패치**:
   - Win Rate: 72.3% → 66.11%
   - Trades/Day: 4.6 → 8.3
   - LONG/SHORT: 61.8/38.2 → 52.5/47.5

2. **잘못된 알림 억제**:
   - ALERT_TRADES_PER_DAY_MIN: 3.0 → 6.0
   - ALERT_SHORT_RATIO_MIN: 0.25 → 0.35

### Short-term (1-2일 내)
1. **청산 메커니즘 추적 추가**:
   - Opposite Signal Exit 비율
   - Buy/Sell signal exit 전환율

2. **Strategy Description 정확화**:
   - "Opportunity Gating" → "Buy/Sell Structure"
   - "EV Gating" 언급 제거

### Medium-term (1주일 내)
1. **신호 품질 대시보드**:
   - Buy/Sell 확률 분포
   - Signal conflict 발생 빈도
   - Entry/Exit 전환율

2. **성과 비교 자동화**:
   - Backtest (+3.82%, 66.11% WR) vs Production
   - 편차 알림 (±20% 초과 시)

---

## 🎯 기대 효과

### Before (현재)
- ❌ 잘못된 Expected Values → 부정확한 알림
- ❌ Entry/Exit 구조 가정 → Buy/Sell 구조 미추적
- ❌ Gating metrics 추적 → 사용하지 않는 기능
- ❌ 청산 메커니즘 미구분 → Opposite Signal 미추적

### After (개편 후)
- ✅ 정확한 Expected Values → 신뢰할 수 있는 알림
- ✅ Buy/Sell 구조 추적 → Opposite Signal exit 가시성
- ✅ 불필요 metrics 제거 → 깔끔한 대시보드
- ✅ 청산 메커니즘 세분화 → ML Exit (70-80%) 검증

---

## 💬 질문

**우선순위 확인**:
1. Phase 1 (Critical Updates) 먼저 배포? (1-2시간 작업)
2. 전체 Phase 1-4 한번에? (6-9시간 작업)

**추가 요구사항**:
1. 80 Features 모델 (Phase 1 우수 성과) 성능 추적 필요?
2. Ensemble 모델 비교 대시보드 필요?

---

**생성**: 2025-11-15 02:30 KST
**상태**: 제안 대기 중
**우선순위**: 🔴 CRITICAL - Expected Values 불일치로 알림 시스템 신뢰도 저하
