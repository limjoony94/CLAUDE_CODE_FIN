# Supervisor - 완전 자동화 관리 시스템

**최종 업데이트**: 2025-10-12 13:35

---

## 🎯 개요

Supervisor는 V2 봇을 24/7 자동으로 관리하는 시스템입니다.

**기능**:
- ✅ 봇 자동 재시작 (크래시 시)
- ✅ 매일 자동 리포트 (아침 9시)
- ✅ 실시간 알림 (경고 발생 시)
- ✅ 성능 추적 및 로그

---

## 🚀 Quick Start

### 방법 1: 포그라운드 실행 (테스트용)

```bash
cd C:/Users/J/OneDrive/CLAUDE_CODE_FIN/bingx_rl_trading_bot
python scripts/production/supervisor.py
```

**언제 사용**:
- Supervisor 테스트할 때
- 로그를 직접 보고 싶을 때
- 디버깅할 때

**종료**: Ctrl+C

### 방법 2: 백그라운드 실행 (프로덕션 권장)

```bash
cd C:/Users/J/OneDrive/CLAUDE_CODE_FIN/bingx_rl_trading_bot
nohup python scripts/production/supervisor.py > logs/supervisor.log 2>&1 &
```

**언제 사용**:
- 24/7 완전 자동화 원할 때
- 터미널 닫아도 실행 유지
- 프로덕션 배포

**종료**:
```bash
ps aux | grep supervisor.py
kill [PID]
```

---

## 📊 Supervisor가 하는 일

### 1분마다 (자동)
- V2 봇 상태 확인
- 봇 죽었으면 자동 재시작
- 알림 시스템 실행 (경고 감지)

### 매일 아침 9시 (자동)
- 종합 대시보드 리포트
- 성능 요약
- 경고 내역

### 크래시 발생 시 (자동)
- 10초 대기
- 자동 재시작
- 최대 3회/시간 (무한 재시작 방지)

---

## 🔍 모니터링 방법

### Supervisor 로그 확인

```bash
# 실시간 모니터링
tail -f logs/supervisor.log

# 최근 100줄
tail -100 logs/supervisor.log

# 오류만 확인
grep "ERROR\|❌\|🚨" logs/supervisor.log
```

### V2 봇 상태 확인

Supervisor가 자동으로 관리하므로 수동 확인 불필요하지만, 원한다면:

```bash
# Dashboard 실행
python scripts/production/dashboard.py

# Alert 확인
python scripts/production/auto_alert_system.py
```

---

## ⚙️ 설정 커스터마이징

`scripts/production/supervisor.py` 파일 수정:

```python
# 체크 주기 (기본: 60초)
CHECK_INTERVAL = 60

# 일일 리포트 시간 (기본: 9시)
DAILY_REPORT_HOUR = 9

# 재시작 대기 시간 (기본: 10초)
RESTART_DELAY = 10

# 최대 재시작 횟수 (기본: 3회/시간)
MAX_RESTART_ATTEMPTS = 3
```

---

## 🚨 문제 해결

### Q1. Supervisor가 안 켜져요

**확인**:
```bash
# 이미 실행 중인지 확인
ps aux | grep supervisor.py

# 파이썬 경로 확인
which python
```

**해결**:
```bash
# 기존 프로세스 종료
kill [PID]

# 재시작
python scripts/production/supervisor.py
```

### Q2. 봇이 자동 재시작 안 돼요

**확인**:
```bash
# Supervisor 로그 확인
tail -50 logs/supervisor.log
```

**원인**:
- 1시간에 3회 재시작 제한 도달
- 봇 스크립트 에러
- 파이썬 환경 문제

**해결**:
- 1시간 대기 또는 Supervisor 재시작
- 봇 로그 확인: `tail -50 logs/combined_v2_realistic_*.log`

### Q3. 일일 리포트가 안 와요

**확인**:
```bash
# 마지막 리포트 시간 확인
grep "DAILY REPORT" logs/supervisor.log
```

**원인**:
- Supervisor가 9시 이전에 시작됨
- Supervisor가 재시작됨

**해결**:
- 다음 날 9시에 자동 발송됨
- 수동 리포트: `python scripts/production/dashboard.py`

---

## 📈 예상 Supervisor 로그

### 정상 작동

```
[SUPERVISOR 2025-10-12 13:35:00] ================================================================================
[SUPERVISOR 2025-10-12 13:35:00] 🎯 V2 BOT SUPERVISOR - STARTED
[SUPERVISOR 2025-10-12 13:35:00] ================================================================================
[SUPERVISOR 2025-10-12 13:35:00] Check interval: 60 seconds
[SUPERVISOR 2025-10-12 13:35:00] Daily report time: 9:00
[SUPERVISOR 2025-10-12 13:35:00] Max restarts: 3/hour
[SUPERVISOR 2025-10-12 13:35:00] ================================================================================
[SUPERVISOR 2025-10-12 13:35:00] ✅ V2 bot already running
[SUPERVISOR 2025-10-12 09:00:15] ================================================================================
[SUPERVISOR 2025-10-12 09:00:15] 📊 DAILY REPORT
[SUPERVISOR 2025-10-12 09:00:15] ================================================================================
[SUPERVISOR 2025-10-12 09:00:15]   🎯 V2 BOT COMPREHENSIVE DASHBOARD
[SUPERVISOR 2025-10-12 09:00:15]   ... [dashboard output] ...
[SUPERVISOR 2025-10-12 09:00:15] ================================================================================
[SUPERVISOR 2025-10-12 09:00:15] ✅ Daily report completed
```

### 봇 크래시 및 재시작

```
[SUPERVISOR 2025-10-12 14:30:00] 🚨 ALERT: V2 bot stopped!
[SUPERVISOR 2025-10-12 14:30:00] 🔄 Restarting bot (attempt 1/3)...
[SUPERVISOR 2025-10-12 14:30:00] 🚀 Starting V2 bot...
[SUPERVISOR 2025-10-12 14:30:05] ✅ V2 bot started successfully
```

### 재시작 제한 도달

```
[SUPERVISOR 2025-10-12 15:45:00] 🚨 CRITICAL: Reached max restart attempts (3/hour)
[SUPERVISOR 2025-10-12 15:45:00] ⏸️  Pausing auto-restart for 1 hour...
```

---

## 💡 Best Practices

### ✅ 권장 사항

1. **백그라운드 실행**: 24/7 자동화를 위해
   ```bash
   nohup python scripts/production/supervisor.py > logs/supervisor.log 2>&1 &
   ```

2. **정기적 로그 확인**: 일주일에 1회
   ```bash
   tail -200 logs/supervisor.log
   ```

3. **Supervisor 업그레이드**: 시스템 부팅 시 자동 시작 설정
   - Windows: Task Scheduler
   - Linux: systemd service
   - macOS: launchd

### ❌ 피해야 할 것

1. **여러 Supervisor 동시 실행**: 중복 재시작 방지
2. **Supervisor 없이 봇만 실행**: 크래시 시 복구 불가
3. **로그 무시**: 정기적으로 확인 필요

---

## 🎯 완전 자동화 달성

### Before Supervisor

**수동 작업 필요**:
- 매일 dashboard.py 실행
- 봇 죽으면 수동 재시작
- 알림 수동 확인

**시간 소요**: 일일 5-10분

### After Supervisor

**완전 자동**:
- ✅ 봇 24/7 자동 관리
- ✅ 자동 재시작
- ✅ 일일 리포트 자동
- ✅ 알림 자동 추적

**시간 소요**: 주간 2분 (로그 확인만)

---

## 📊 Status Summary

```yaml
System: V2 Bot + Supervisor
Status: ✅ 완전 자동화
Manual Work: 거의 없음 (주 1회 로그 확인)

Supervisor:
  File: scripts/production/supervisor.py
  Status: 준비 완료
  Features: 자동 재시작, 일일 리포트, 알림 추적

Integration:
  Dashboard: ✅ 자동 실행
  Alert System: ✅ 자동 실행
  Bot Management: ✅ 완전 자동
```

---

## 🚀 Next Level

### 더 고급 자동화 원한다면:

1. **Telegram/Email 알림**
   - Supervisor에 알림 통합
   - 경고 발생 시 메시지 전송

2. **Performance Analytics**
   - 일주일 성과 자동 분석
   - 최적화 제안 자동 생성

3. **Adaptive Parameters**
   - 성능 기반 threshold 자동 조정
   - 시장 상황 기반 설정 변경

---

**Bottom Line**: Supervisor = 완전한 hands-free 자동화 ✅

**사용법**: `nohup python scripts/production/supervisor.py > logs/supervisor.log 2>&1 &`

**체크**: `tail -f logs/supervisor.log`

---
