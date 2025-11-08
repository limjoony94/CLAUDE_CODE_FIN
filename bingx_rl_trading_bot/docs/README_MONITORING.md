# ML Exit Bot 모니터링 가이드

**Bot 상태**: ✅ RUNNING
**System**: Phase 4 Dual Entry + Dual Exit Model
**Exit Strategy**: ML-based (LONG/SHORT specialized)

---

## 🎯 빠른 시작

### 1. 대시보드 열기 (권장)

**Windows 탐색기에서**:
```
C:\Users\J\OneDrive\CLAUDE_CODE_FIN\bingx_rl_trading_bot\
```

**파일 더블클릭**:
- `monitor_dashboard.bat` ⭐ 메인 대시보드 (시작하기 좋음!)

또는 **명령 프롬프트에서**:
```bash
cd C:\Users\J\OneDrive\CLAUDE_CODE_FIN\bingx_rl_trading_bot
monitor_dashboard.bat
```

---

## 📊 모니터링 도구 (8가지) ⭐ 강화됨!

### PERFORMANCE (성과 분석)

#### 1. monitor_performance.bat ⭐ 신규!
**성과 요약 모니터**
- 총 거래 수 (POSITION CLOSED 카운트)
- Win/Loss 분석 (최근 거래 결과)
- ML Exit vs Max Hold 비율
- 실시간 성과 통계

**사용법**: 대시보드에서 [1] 선택

**보는 내용**:
```
Total Trades Closed: 15
ML Exits: 13 (86.7%) | Max Hold: 2
Session P&L: +$372.10 (+0.37%)
```

#### 2. monitor_trades.bat ⭐ 신규!
**거래 히스토리 뷰어**
- 최근 거래 진입/청산 (Last 10)
- 거래 결과 및 P&L
- 보유 시간 분석
- 포지션 사이즈 분석

**사용법**: 대시보드에서 [2] 선택

**보는 내용**:
```
Opening LONG position: 0.5 BTC @ $111,000
Exit Reason: ML Exit (LONG model, prob=0.823)
Return: +1.2% ($600 USDT)
Holding: 1.5 hours
```

### SIGNALS (신호 분석)

#### 3. monitor_signals.bat ⭐ 신규!
**신호 강도 모니터**
- Entry 신호 확률 (LONG/SHORT)
- Exit 신호 확률 (ML Exit)
- 고강도 신호 탐지 (>0.80)
- 신호 통계 및 분포
- Auto-refresh 모드

**사용법**: 대시보드에서 [3] 선택

**보는 내용**:
```
LONG signal: 0.756 (threshold: 0.70)
Exit Model Signal: 0.823 (threshold: 0.75)
Total LONG Signals: 45
Total ML Exit Signals: 13
```

#### 4. monitor_ml_exit_signals.bat
**ML Exit 신호 전용**
- Exit Model Signal (LONG/SHORT)
- Exit 확률 및 결정
- ML Exit vs Max Hold 비율

**사용법**: 대시보드에서 [4] 선택

### SYSTEM (시스템 모니터링)

#### 5. monitor_positions.bat
**포지션 및 P&L 추적**
- 포지션 진입 (LONG/SHORT)
- 실시간 P&L 업데이트
- 보유 시간
- Exit 신호 및 결정

**사용법**: 대시보드에서 [5] 선택

#### 6. monitor_ml_exit.bat
**전체 로그 모니터링**
- 실시간 모든 로그 표시
- Bot 전체 활동 추적
- 디버깅에 유용

**사용법**: 대시보드에서 [6] 선택

#### 7. monitor_errors.bat ⭐ 신규!
**에러/경고 전용 모니터**
- Critical Errors (ERROR, Exception)
- Warnings (WARNING)
- 연결 문제 감지
- 데이터 문제 감지
- 거래 실행 에러
- 에러 통계 및 Health Check

**사용법**: 대시보드에서 [7] 선택

**보는 내용**:
```
Total Errors: 3
Total Warnings: 5
Last Error Time: 2025-10-14 02:25:08
Status: No critical errors in last 2 hours
```

### CONTROL (제어)

#### 8. monitor_dashboard.bat ⭐ 메인!
**통합 대시보드**
- Bot 상태 확인 (실행 중 / 중지)
- 최근 활동 요약 (최근 15 로그)
- 모든 모니터링 도구 실행
- Enhanced 9-option menu

**사용법**: 더블클릭 또는 명령 프롬프트에서 실행

---

## 🔍 Bot 현재 상태

### Bot 실행 확인
```
Status: RUNNING ✅
Initial Balance: $101,858.63 USDT
ML Exit Models: LOADED ✅
  - LONG Exit Model: 44 features
  - SHORT Exit Model: 44 features
Exit Threshold: 0.75 (75% probability)
```

### 현재 활동
```
Status: Waiting for sufficient data
Data: 500 candles collected (need 1440 for full analysis)
Next Update: Every 5 minutes

Note: Bot needs ~5 days of data (1440 candles) before trading
      This is normal - bot is collecting historical data
```

---

## 📈 기대 성능 (ML Exit 백테스트)

| 지표 | Rule-based | ML Exit | 개선도 |
|------|-----------|---------|-------|
| 수익률 | 2.04% | 2.85% | **+39.2%** |
| 승률 | 89.7% | 94.7% | **+5.0%** |
| 평균 보유 | 4.00h | 2.36h | **-41%** |
| ML Exit 비율 | 0% | 87.6% | **+87.6%** |

---

## 🎯 모니터링 체크리스트

### 매 5분 확인 (실시간 모니터링)
- [ ] Bot 실행 중
- [ ] 데이터 수신 (500 candles)
- [ ] 로그 업데이트

### 첫 거래 발생 시 확인
- [ ] Entry Signal (LONG/SHORT probability)
- [ ] Position size (20-95% dynamic)
- [ ] Entry price
- [ ] Entry reason logged

### 포지션 보유 중 확인
- [ ] Exit Model Signal 매 5분 업데이트
- [ ] Exit probability (threshold: 0.75)
- [ ] Current P&L
- [ ] Holding time

### 포지션 청산 시 확인
- [ ] Exit reason (ML Exit vs Max Hold)
- [ ] Exit probability at decision
- [ ] Final P&L
- [ ] Total holding time

---

## ⚠️ 정상 동작 vs 문제

### ✅ 정상 동작

**"Insufficient market data" 경고**:
```
WARNING: Insufficient market data
```
- **원인**: 1440 candles 필요, 현재 500개만 수집
- **해결**: 기다리면 자동 해결 (~5일 데이터 수집)
- **상태**: 정상 - Bot이 데이터 수집 중

**"Next update in 300s"**:
```
⏳ Next update in 300s (at :30:05)
```
- **의미**: 5분마다 업데이트
- **상태**: 정상 - Bot이 대기 중

### 🚨 문제 신호

**Bot 중지**:
```
Status: STOPPED ❌
```
- **조치**: Bot 재시작 필요

**Lock file 없음**:
```
Lock File: MISSING ❌
```
- **조치**: Bot이 실행 중이 아님

**로그 파일 없음**:
```
Log File: NOT FOUND ❌
```
- **조치**: Bot 재시작 또는 몇 분 대기

---

## 🛠️ 문제 해결

### 배치파일이 로그를 찾지 못함

**증상**:
```
[ERROR] Log file not found
```

**해결책**:
1. 배치파일이 자동으로 최신 로그 찾음 (수정 완료!)
2. 대시보드에서 [4] Refresh 선택
3. Bot 실행 중인지 확인

### Bot이 거래하지 않음

**원인**:
- 데이터 수집 중 (1440 candles 필요)
- Entry 신호 없음 (threshold 0.7)

**확인**:
```bash
# 신호 확인
grep "Signal Check" logs/phase4_dynamic_testnet_trading_20251014.log | tail -10
```

### Exit Model이 작동하지 않음

**확인**:
```bash
# Exit 신호 확인
grep "Exit Model Signal" logs/phase4_dynamic_testnet_trading_20251014.log
```

**정상 출력**:
```
Exit Model Signal (LONG): 0.652 (threshold: 0.75)
```

---

## 📊 1주일 모니터링 목표

### Success Criteria
- ✅ ML Exit rate ≥ 80% (목표: 87.6%)
- ✅ Win rate ≥ 90% (목표: 94.7%)
- ✅ Avg return ~2.85% per trade
- ✅ Avg holding ~2.4 hours

### Warning Signs
- 🚨 ML Exit rate < 70%
- 🚨 Win rate < 85%
- 🚨 Avg holding > 3.5h
- 🚨 Returns < 1.5% per trade

---

## 📁 로그 파일 위치

**Today's Log**:
```
logs/phase4_dynamic_testnet_trading_20251014.log
```

**수동 로그 확인** (Git Bash):
```bash
# 실시간 로그
tail -f logs/phase4_dynamic_testnet_trading_20251014.log

# Exit 신호만
tail -f logs/phase4_dynamic_testnet_trading_20251014.log | grep "Exit"

# 포지션 업데이트만
tail -f logs/phase4_dynamic_testnet_trading_20251014.log | grep "Position"
```

---

## 🎉 추가 팁

### 여러 모니터링 창 동시 실행 (권장 조합)

**조합 1: 성과 중심 모니터링**
1. `monitor_dashboard.bat` 실행 (대시보드)
2. [1] 성과 요약 모니터 시작
3. [2] 거래 히스토리 뷰어 시작
4. [7] 에러 모니터 시작

→ 4개 창으로 성과 및 문제 추적!

**조합 2: 신호 중심 모니터링**
1. `monitor_dashboard.bat` 실행 (대시보드)
2. [3] 신호 강도 모니터 시작
3. [5] 포지션 모니터 시작
4. [6] 전체 로그 모니터 시작

→ 4개 창으로 신호 및 실시간 활동 추적!

### Ctrl+C로 중지

모든 모니터링 스크립트는 `Ctrl+C`로 중지 가능

### 대시보드에서 Refresh

대시보드에서 [4] 선택하면 최신 상태 업데이트

---

## ✅ 요약

**모니터링 시작** (Enhanced!):
1. `monitor_dashboard.bat` 더블클릭
2. Bot 상태 확인
3. 원하는 모니터링 도구 선택 (1-9)
   - [1] 성과 요약 ⭐ 신규!
   - [2] 거래 히스토리 ⭐ 신규!
   - [3] 신호 강도 ⭐ 신규!
   - [7] 에러 모니터 ⭐ 신규!

**첫 거래 대기**:
- 데이터 수집 완료까지 대기 (~5일)
- Entry 신호 발생 대기 (threshold 0.7)
- [3] 신호 강도로 실시간 확률 추적

**거래 발생 시**:
- [1] 성과 요약으로 전체 통계 확인
- [2] 거래 히스토리로 상세 정보 추적
- [5] 포지션 모니터로 P&L 실시간 추적
- [3] 신호 강도로 Exit 확률 모니터링

**1주일 후**:
- [1] 성과 요약으로 ML Exit 비율 검증
- 승률, 평균 수익률, 보유 시간 분석
- ML Exit 효율성 검증 (목표 87.6%)
- Production 배포 결정

---

**문제 발생 시**:
- [7] 에러 모니터로 즉시 문제 파악 ⭐ 신규!
- [8] 대시보드 Refresh
- Bot 상태 확인 (Status, Lock File)
- 로그 파일 확인 (manual)

**Enhanced 모니터링 준비 완료!** 🚀
**8가지 전문 도구로 완벽한 Bot 관리!** ⭐
