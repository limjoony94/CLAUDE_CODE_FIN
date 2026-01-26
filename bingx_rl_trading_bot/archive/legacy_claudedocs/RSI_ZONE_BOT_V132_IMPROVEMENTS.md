# RSI Zone Bot v1.3.2 개선사항

**날짜**: 2025-12-11
**버전**: v1.3.2 (Fine-Tuned + Enhanced Resilience)

---

## 1. 주요 개선사항 요약

### v1.3.1 → v1.3.2 변경사항

| 기능 | 설명 |
|------|------|
| **API 재시도 로직** | 지수 백오프 + 지터를 사용한 자동 재시도 |
| **헬스체크 시스템** | 주기적 시스템 상태 점검 (API, 잔고, 포지션, 주문) |
| **에러 분류 시스템** | 복구 가능/불가능 에러 자동 분류 및 처리 |
| **봇 일시정지 기능** | 크리티컬 에러 시 자동 일시정지 후 복구 |

### 전체 버전 히스토리

| 버전 | 날짜 | 주요 변경사항 |
|------|------|--------------|
| v1.3.2 | 2025-12-11 | API 재시도, 헬스체크, 에러 분류 |
| v1.3.1 | 2025-12-11 | YAML 설정, 자동 백업, Deep Sync |
| v1.3 | 2025-12-11 | TP 2.4%, SL 1.4%, BE_SL 1.2% (891조합 최적화) |

---

## 2. API 재시도 로직 (Exponential Backoff)

### 개요

네트워크 오류, 타임아웃, 서버 오류 등 일시적 문제 발생 시 자동으로 재시도합니다.

### 설정 파라미터

```yaml
# config/rsi_zone_bot_config.yaml
api_retry:
  max_retries: 3                    # 최대 재시도 횟수
  base_delay_seconds: 1.0           # 기본 대기 시간 (초)
  max_delay_seconds: 30.0           # 최대 대기 시간 (초)
  exponential_base: 2.0             # 지수 백오프 배수
  retryable_errors:                 # 재시도 가능 에러 목록
    - "TIMEOUT"
    - "CONNECTION_ERROR"
    - "RATE_LIMIT"
    - "SERVER_ERROR"
    - "NETWORK_ERROR"
```

### 작동 방식

1. **1차 시도**: 즉시 실행
2. **실패 시**: 에러 유형 분류
3. **재시도 가능한 에러**: 지수 백오프 대기 후 재시도
   - 1차 재시도: ~1초 대기
   - 2차 재시도: ~2초 대기
   - 3차 재시도: ~4초 대기 (max 30초)
4. **지터 추가**: ±25% 랜덤 지연 (동시 재시도 방지)

### 로그 예시

```
2025-12-11 15:30:00 [WARNING] Retryable error in fetch_candles: TIMEOUT. Retrying in 1.2s (attempt 1/4)
2025-12-11 15:30:02 [WARNING] Retryable error in fetch_candles: TIMEOUT. Retrying in 2.5s (attempt 2/4)
2025-12-11 15:30:05 [INFO] API call successful after retry
```

---

## 3. 헬스체크 시스템

### 개요

주기적으로 시스템 상태를 점검하여 문제를 조기에 감지합니다.

### 설정 파라미터

```yaml
# config/rsi_zone_bot_config.yaml
health_check:
  enabled: true                     # 헬스체크 활성화
  interval_minutes: 10              # 헬스체크 주기 (분)
  api_timeout_seconds: 10           # API 타임아웃 (초)
  max_consecutive_failures: 5       # 연속 실패 허용 횟수
  checks:                           # 체크 항목
    api_connection: true            # API 연결 상태
    balance_available: true         # 잔고 조회 가능 여부
    position_sync: true             # 포지션 동기화 상태
    order_integrity: true           # 주문 무결성
```

### 체크 항목

| 체크 | 설명 | 상태 |
|------|------|------|
| **api_connection** | API 연결 및 응답 시간 확인 | ok/warning/error |
| **balance** | 잔고 조회 및 최소 잔고 확인 | ok/warning/error |
| **position_sync** | State와 거래소 포지션 일치 여부 | ok/warning/error |
| **order_integrity** | 열린 포지션의 TP/SL 주문 존재 여부 | ok/warning/error |
| **failure_count** | 연속 실패 횟수 확인 | ok/warning/error |

### 상태 레벨

- **healthy** ✅: 모든 체크 통과
- **degraded** ⚠️: 경고 있지만 운영 가능
- **unhealthy** ❌: 에러 발생, 주의 필요

### 로그 예시

```
2025-12-11 15:40:00 🏥 Running Health Check...
2025-12-11 15:40:00 🏥 Health Check: ✅ HEALTHY
2025-12-11 15:40:00    [✓] api_connection: ok
2025-12-11 15:40:00    [✓] balance: ok
2025-12-11 15:40:00    [✓] position_sync: ok
2025-12-11 15:40:00    [✓] order_integrity: ok
```

---

## 4. 에러 분류 및 처리

### 개요

발생한 에러를 자동으로 분류하여 적절한 조치를 취합니다.

### 설정 파라미터

```yaml
# config/rsi_zone_bot_config.yaml
error_handling:
  # 복구 가능 에러 (자동 재시도)
  recoverable_errors:
    - "TIMEOUT"
    - "CONNECTION_RESET"
    - "TEMPORARY_UNAVAILABLE"
    - "RATE_LIMIT_EXCEEDED"
    - "INTERNAL_SERVER_ERROR"
  # 복구 불가능 에러 (즉시 중단)
  critical_errors:
    - "INVALID_API_KEY"
    - "INSUFFICIENT_BALANCE"
    - "POSITION_NOT_FOUND"
    - "ORDER_REJECTED"
  # 에러 발생 시 동작
  on_critical_error: "pause"        # "pause" | "stop" | "alert_only"
  pause_duration_minutes: 5         # pause 시 대기 시간 (분)
  alert_on_error: true              # 에러 발생 시 알림
```

### 에러 분류 체계

| 분류 | 에러 유형 | 처리 방식 |
|------|----------|----------|
| **Recoverable** | TIMEOUT, CONNECTION_ERROR, RATE_LIMIT, SERVER_ERROR | 자동 재시도 |
| **Critical** | INVALID_API_KEY, INSUFFICIENT_BALANCE | 봇 일시정지/중단 |

### Critical 에러 처리 옵션

| 옵션 | 동작 |
|------|------|
| `pause` | 설정된 시간 동안 일시정지 후 자동 재개 |
| `stop` | 봇 완전 중지 (수동 재시작 필요) |
| `alert_only` | 로그만 기록하고 계속 실행 |

### 로그 예시 (Critical Error)

```
2025-12-11 16:00:00 ============================================================
2025-12-11 16:00:00 🚨 CRITICAL ERROR: INSUFFICIENT_BALANCE
2025-12-11 16:00:00    Balance too low to open position
2025-12-11 16:00:00    Bot PAUSED for 5 minutes
2025-12-11 16:00:00 ============================================================
```

---

## 5. Claude Code 활용 가이드 (v1.3.2 추가)

### 에러 처리 설정 변경

```
# 에러 발생 시 봇 중지로 변경
"에러 발생 시 봇을 멈추게 해줘"
→ config/rsi_zone_bot_config.yaml에서 error_handling.on_critical_error: "stop"

# 일시정지 시간 변경
"일시정지 시간을 10분으로 늘려줘"
→ config/rsi_zone_bot_config.yaml에서 error_handling.pause_duration_minutes: 10

# 연속 실패 허용 횟수 변경
"5번 연속 실패해도 계속 실행하게 해줘"
→ config/rsi_zone_bot_config.yaml에서 health_check.max_consecutive_failures: 10
```

### 헬스체크 설정 변경

```
# 헬스체크 주기 변경
"헬스체크를 5분마다 실행해줘"
→ config/rsi_zone_bot_config.yaml에서 health_check.interval_minutes: 5

# 헬스체크 비활성화
"헬스체크 기능 끄고 싶어"
→ config/rsi_zone_bot_config.yaml에서 health_check.enabled: false

# 특정 체크 비활성화
"주문 무결성 체크만 끄고 싶어"
→ config/rsi_zone_bot_config.yaml에서 health_check.checks.order_integrity: false
```

### API 재시도 설정 변경

```
# 재시도 횟수 변경
"재시도를 5번까지 해줘"
→ config/rsi_zone_bot_config.yaml에서 api_retry.max_retries: 5

# 최대 대기 시간 변경
"재시도 대기 시간 최대 1분으로 늘려줘"
→ config/rsi_zone_bot_config.yaml에서 api_retry.max_delay_seconds: 60
```

---

## 6. 시스템 아키텍처

### 에러 처리 흐름

```
API 호출
   │
   ├─> 성공 → 정상 처리, 연속 실패 카운터 리셋
   │
   └─> 실패 → 에러 분류
              │
              ├─> Recoverable → 재시도 (지수 백오프)
              │     │
              │     ├─> 성공 → 정상 처리
              │     └─> 실패 (max_retries 도달) → 연속 실패 카운터 증가
              │
              └─> Critical → handle_critical_error()
                    │
                    ├─> pause → 일시정지 후 자동 재개
                    ├─> stop → 봇 완전 중지
                    └─> alert_only → 로그만 기록
```

### 헬스체크 흐름

```
Main Loop
   │
   ├─> check_bot_paused() → True → sleep → continue
   │
   └─> run_health_check() → interval 도달 시
         │
         ├─> API Connection Check
         ├─> Balance Check
         ├─> Position Sync Check
         ├─> Order Integrity Check
         └─> Consecutive Failures Check
              │
              └─> max_consecutive_failures 초과 시 → Critical Error 처리
```

---

## 7. 주의사항

### API 재시도 관련

1. 재시도는 일시적 오류에만 적용됩니다.
2. API 키 오류, 잔고 부족 등은 재시도하지 않습니다.
3. Rate Limit 에러 발생 시 대기 시간이 길어질 수 있습니다.

### 헬스체크 관련

1. 헬스체크는 추가 API 호출을 발생시킵니다.
2. 너무 짧은 interval은 Rate Limit을 유발할 수 있습니다.
3. 포지션이 없을 때는 order_integrity 체크가 스킵됩니다.

### 에러 처리 관련

1. `pause` 모드에서는 설정된 시간 후 자동으로 재개됩니다.
2. `stop` 모드에서는 수동으로 재시작해야 합니다.
3. 연속 실패 카운터는 성공 시 자동으로 리셋됩니다.

---

## 8. 파일 구조

```
bingx_rl_trading_bot/
├── config/
│   └── rsi_zone_bot_config.yaml     ← 설정 파일 (v1.3.2 추가 섹션)
│
├── scripts/
│   └── production/
│       └── rsi_zone_bot.py          ← v1.3.2 (Enhanced Resilience)
│
├── results/
│   ├── rsi_zone_bot_state.json      ← 현재 상태
│   └── backups/                     ← 백업 디렉토리
│       └── rsi_zone_bot_state_*.json
│
└── claudedocs/
    ├── RSI_ZONE_BOT_V131_IMPROVEMENTS.md  ← v1.3.1 문서
    └── RSI_ZONE_BOT_V132_IMPROVEMENTS.md  ← 이 문서 (v1.3.2)
```

---

## 9. 관련 코드 위치

| 기능 | 함수/클래스 | 파일 위치 |
|------|------------|----------|
| 에러 분류 | `classify_error()` | rsi_zone_bot.py:306 |
| 재시도 데코레이터 | `api_retry()` | rsi_zone_bot.py:369 |
| 재시도 딜레이 계산 | `calculate_retry_delay()` | rsi_zone_bot.py:346 |
| 헬스체크 | `run_health_check()` | rsi_zone_bot.py:485 |
| Critical 에러 처리 | `handle_critical_error()` | rsi_zone_bot.py:422 |
| 일시정지 체크 | `check_bot_paused()` | rsi_zone_bot.py:454 |
| BotError 클래스 | `BotError` | rsi_zone_bot.py:298 |

---

## 10. 변경 이력

| 버전 | 날짜 | 변경사항 |
|------|------|---------|
| v1.3.2 | 2025-12-11 | API 재시도 (지수 백오프), 헬스체크, 에러 분류/처리 |
| v1.3.1 | 2025-12-11 | YAML 설정, 자동 백업, Deep Sync 추가 |
| v1.3 | 2025-12-11 | TP 2.4%, SL 1.4%, BE_SL 1.2% (891조합 최적화) |
| v1.2 | 2025-12-10 | TP 2.5%, SL 1.5%, BE_SL 1.5% |
| v1.1 | 2025-12-10 | TP 2.0%, SL 1.5%, BE_SL 1.0% |
