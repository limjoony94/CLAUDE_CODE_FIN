# Automated Model Retraining System - Usage Guide

**날짜**: 2025-10-14
**버전**: 1.0
**작성자**: Claude Code

---

## 📋 목차

1. [시스템 개요](#시스템-개요)
2. [생성된 스크립트](#생성된-스크립트)
3. [사용 방법](#사용-방법)
4. [주기적 재훈련 설정](#주기적-재훈련-설정)
5. [모니터링 및 검증](#모니터링-및-검증)

---

## 시스템 개요

### 목적
- 최신 시장 데이터로 4개 모델을 주기적으로 재훈련
- 데이터 다운로드 자동화
- 봇 실행 중 안전한 재훈련

### 구성 요소
```
scripts/production/
├── download_historical_data.py      # 데이터 다운로드
├── train_all_models.py             # 4개 모델 통합 훈련
└── periodic_retraining_scheduler.py # 주기적 재훈련 스케줄러
```

### 4개 모델
1. **LONG Entry Model** - 상승 예측 진입 (37 features)
2. **SHORT Entry Model** - 하락 예측 진입 (37 features)
3. **LONG Exit Model** - 롱 포지션 청산 (44 features)
4. **SHORT Exit Model** - 숏 포지션 청산 (44 features)

---

## 생성된 스크립트

### 1. `download_historical_data.py`

**기능**: BingX API로 최신 5분 캔들 데이터 다운로드

**사용법**:
```bash
# 기본 다운로드 (60일)
python scripts/production/download_historical_data.py

# 90일 다운로드
python scripts/production/download_historical_data.py --days 90

# 강제 전체 다운로드 (기존 데이터 무시)
python scripts/production/download_historical_data.py --force
```

**특징**:
- ✅ 기존 데이터가 있으면 마지막 시점부터 이어받기
- ✅ 데이터 검증 (NaN, 시간 갭 체크)
- ✅ 자동 백업 (기존 파일을 .backup으로 저장)
- ✅ Rate limiting (API 속도 제한 준수)

**출력**:
```
data/historical/BTCUSDT_5m_max.csv
```

---

### 2. `train_all_models.py`

**기능**: 4개 모델을 순차적으로 일관된 데이터로 훈련

**사용법**:
```bash
# 기본 훈련 (기존 데이터 사용)
python scripts/production/train_all_models.py

# 데이터 다운로드 + 훈련
python scripts/production/train_all_models.py --download-data

# 특정 모델만 훈련
python scripts/production/train_all_models.py --skip-long-entry
python scripts/production/train_all_models.py --skip-short-entry
python scripts/production/train_all_models.py --skip-exit-models
```

**훈련 순서**:
1. (Optional) 데이터 다운로드
2. LONG Entry Model 훈련 (37 features)
3. SHORT Entry Model 훈련 (37 features)
4. Exit Models 훈련 (LONG + SHORT, 44 features each)

**출력**:
```
models/
├── xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl       # LONG Entry
├── xgboost_short_model_lookahead3_thresh0.3.pkl            # SHORT Entry
├── xgboost_v4_long_exit.pkl                                # LONG Exit
└── xgboost_v4_short_exit.pkl                               # SHORT Exit

results/
└── training_report_YYYYMMDD_HHMMSS.json                     # 훈련 보고서
```

**예상 소요 시간**:
- 데이터 다운로드: ~5분
- LONG Entry: ~3분
- SHORT Entry: ~3분
- Exit Models: ~10분
- **Total: ~20분**

---

### 3. `periodic_retraining_scheduler.py`

**기능**: 주기적 자동 재훈련 시스템

**사용법**:

#### 설정 확인
```bash
python scripts/production/periodic_retraining_scheduler.py --check
```

출력:
```
RETRAINING CONFIGURATION
================================================================================
enabled: True
interval_days: 7
retraining_time: 02:00
wait_for_position_close: True
max_wait_hours: 4
download_data_before_training: True
validate_after_training: True
last_training: 2025-10-14T19:07:15
next_training: 2025-10-21T02:00:00
================================================================================

Next training: 2025-10-21 02:00:00
```

#### 즉시 재훈련 (테스트용)
```bash
python scripts/production/periodic_retraining_scheduler.py --now
```

#### 스케줄러 시작 (백그라운드)
```bash
# Linux/Mac
nohup python scripts/production/periodic_retraining_scheduler.py --start > retraining.log 2>&1 &

# Windows (PowerShell)
Start-Process python -ArgumentList "scripts/production/periodic_retraining_scheduler.py","--start" -WindowStyle Hidden
```

#### Windows Task Scheduler 설정
```bash
# XML 파일 생성
python scripts/production/periodic_retraining_scheduler.py --create-task

# 그 후:
# 1. 작업 스케줄러 열기
# 2. "작업 가져오기..." 클릭
# 3. retraining_task.xml 선택
# 4. 설정 조정 후 저장
```

---

## 주기적 재훈련 설정

### 기본 설정

**파일**: `config/retraining_config.json`

```json
{
  "enabled": true,
  "interval_days": 7,
  "retraining_time": "02:00",
  "wait_for_position_close": true,
  "max_wait_hours": 4,
  "download_data_before_training": true,
  "validate_after_training": true,
  "last_training": "2025-10-14T19:07:15",
  "next_training": "2025-10-21T02:00:00"
}
```

### 설정 항목 설명

| 항목 | 설명 | 권장값 |
|------|------|--------|
| `enabled` | 자동 재훈련 활성화 | `true` |
| `interval_days` | 재훈련 간격 (일) | `7` (매주) |
| `retraining_time` | 재훈련 시각 (HH:MM) | `"02:00"` (새벽 2시, 낮은 활동) |
| `wait_for_position_close` | 포지션 청산 대기 | `true` (안전) |
| `max_wait_hours` | 최대 대기 시간 (시간) | `4` |
| `download_data_before_training` | 훈련 전 데이터 다운로드 | `true` |
| `validate_after_training` | 훈련 후 검증 | `true` |

### 재훈련 주기 권장사항

| 시장 변동성 | 권장 주기 | 이유 |
|------------|----------|------|
| **높음** (>5% 일 변동) | 3-5일 | 빠른 시장 변화 반영 |
| **보통** (2-5% 일 변동) | 7일 | 균형잡힌 업데이트 |
| **낮음** (<2% 일 변동) | 14일 | 안정적 패턴 유지 |

---

## 모니터링 및 검증

### 훈련 로그 확인

```bash
# 최신 훈련 로그
tail -f logs/phase4_dynamic_testnet_trading_YYYYMMDD.log

# 훈련 보고서
cat results/training_report_YYYYMMDD_HHMMSS.json
```

### 모델 성능 검증

#### 1. 훈련 메트릭 확인
```bash
# LONG Entry Model
cat models/xgboost_v4_phase4_advanced_lookahead3_thresh0_metadata.json

# SHORT Entry Model
cat models/xgboost_short_model_lookahead3_thresh0.3_metadata.txt

# Exit Models
cat models/xgboost_v4_long_exit_metadata.json
cat models/xgboost_v4_short_exit_metadata.json
```

**예상 메트릭**:
- LONG Entry: Accuracy ~93.8%, Recall ~9.6%
- SHORT Entry: Accuracy ~97.1%, Recall ~2.5%
- LONG Exit: Accuracy ~89.4%, Recall ~95.6%
- SHORT Exit: Accuracy ~89.6%, Recall ~94.5%

#### 2. 백테스트 검증
```bash
# Dual Model 백테스트
python scripts/experiments/backtest_dual_model.py
```

**예상 성능**:
- 평균 수익률: ~+14.98% per 5일
- 승률: ~66.2%
- LONG/SHORT 비중: 87.6% / 12.4%

#### 3. 실전 성능 모니터링
```bash
# 봇 상태 확인
cat results/phase4_testnet_trading_state.json

# 통계 확인
python scripts/analyze_bot_performance.py
```

---

## 일반적인 사용 시나리오

### 시나리오 1: 주간 자동 재훈련 (권장)

**설정**:
```json
{
  "enabled": true,
  "interval_days": 7,
  "retraining_time": "02:00"
}
```

**실행**:
```bash
# Windows Task Scheduler로 설정
python scripts/production/periodic_retraining_scheduler.py --create-task

# 또는 백그라운드 프로세스로 실행
nohup python scripts/production/periodic_retraining_scheduler.py --start &
```

**장점**:
- ✅ 자동화된 재훈련
- ✅ 최신 시장 데이터 반영
- ✅ 포지션 청산 대기로 안전

---

### 시나리오 2: 수동 재훈련 (필요시)

**사용 경우**:
- 큰 시장 이벤트 발생 (예: 큰 가격 변동)
- 봇 성능 저하 감지
- 새로운 전략 테스트

**실행**:
```bash
# 즉시 재훈련
python scripts/production/train_all_models.py --download-data

# 또는 스케줄러로
python scripts/production/periodic_retraining_scheduler.py --now
```

---

### 시나리오 3: 데이터만 업데이트

**사용 경우**:
- 데이터 갭 발견
- 정기 데이터 백업

**실행**:
```bash
# 최신 데이터 다운로드 (기존 데이터에 추가)
python scripts/production/download_historical_data.py

# 전체 재다운로드
python scripts/production/download_historical_data.py --force --days 90
```

---

## 문제 해결

### 문제 1: API Rate Limit 에러

**증상**:
```
Error: BingX API RATE LIMIT EXCEEDED!
```

**해결책**:
```python
# download_historical_data.py의 대기 시간 증가
time.sleep(0.5)  # → time.sleep(1.0)
```

---

### 문제 2: 포지션이 닫히지 않음

**증상**:
```
Timeout: Positions still open after 4h
```

**해결책**:
1. `max_wait_hours` 증가 (4 → 8)
2. 수동으로 포지션 청산 후 재훈련
3. `wait_for_position_close: false` 설정 (위험!)

---

### 문제 3: 훈련 실패

**증상**:
```
Retraining failed (return code: 1)
```

**해결책**:
1. 로그 확인: `logs/training_*.log`
2. 데이터 검증: `python scripts/production/download_historical_data.py`
3. 디스크 공간 확인
4. Python 패키지 업데이트

---

## 고급 사용법

### 사용자 정의 스케줄

**월요일 & 목요일 재훈련**:
```python
# periodic_retraining_scheduler.py 수정
if datetime.now().weekday() in [0, 3]:  # Monday=0, Thursday=3
    perform_retraining_cycle(config)
```

**시장 변동성 기반 재훈련**:
```python
# 변동성 계산
volatility = calculate_recent_volatility()

# 변동성 높으면 더 자주 재훈련
if volatility > 0.05:
    interval_days = 3
elif volatility > 0.03:
    interval_days = 5
else:
    interval_days = 7
```

---

## 참고 자료

- **봇 상태**: `bingx_rl_trading_bot/SYSTEM_STATUS.md`
- **프로젝트 개요**: `bingx_rl_trading_bot/README.md`
- **모델 분석**: `claudedocs/DUAL_MODEL_DEPLOYMENT_20251014.md`
- **Exit 모델**: `claudedocs/EXIT_MODEL_DEPLOYMENT_DECISION.md`

---

## 요약

✅ **3개 스크립트 생성 완료**:
1. `download_historical_data.py` - 데이터 다운로드
2. `train_all_models.py` - 4개 모델 통합 훈련
3. `periodic_retraining_scheduler.py` - 주기적 재훈련

✅ **핵심 기능**:
- 자동 데이터 다운로드
- 일관된 4개 모델 훈련
- 주기적 재훈련 스케줄링
- 안전한 포지션 관리

✅ **권장 설정**:
- 주기: 매주 (7일)
- 시각: 새벽 2시
- 포지션 청산 대기: 활성화

---

**문의 및 개선 사항**: claudedocs/AUTOMATED_RETRAINING_GUIDE.md 참조
