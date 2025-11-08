# Sweet-2 테스트넷 검증 가이드

**Date**: 2025-10-10
**Status**: ✅ **실시간 API 연동 완료**

---

## 🎉 검증 완료 사항

### 1. ✅ BingX API 실시간 데이터 수집 검증

**테스트 결과**:
```
✅ API Connection: Successful
✅ 5-minute Candlestick Data: 200 candles retrieved
✅ Real-time Price Updates: Working ($121,715.20)
✅ Data Quality: Valid OHLCV data
```

**테스트 실행**:
```bash
python scripts/production/test_bingx_api.py
```

---

### 2. ✅ Sweet-2 Bot 실시간 API 연동 완료

**업그레이드**:
- ✅ Live BingX API 데이터 수집 구현
- ✅ Fallback to simulation mode (파일 데이터)
- ✅ 실시간 가격 추적
- ✅ 5분마다 자동 업데이트

**검증 완료**:
```bash
# 실시간 API 데이터로 작동 확인
✅ Live data from BingX API: 200 candles
✅ Latest BTC Price: $121,715.20
✅ Sweet-2 thresholds applied
✅ Update cycle working
```

---

## 🚀 실행 방법

### Option 1: 실시간 API 모드 (권장)

**현재 상태**: 인터넷 연결만 있으면 자동으로 실시간 데이터 사용

```bash
cd C:\Users\J\OneDrive\CLAUDE_CODE_FIN\bingx_rl_trading_bot
python scripts/production/sweet2_paper_trading.py
```

**특징**:
- ✅ BingX Public API 사용 (credentials 불필요)
- ✅ 실시간 5분 캔들 데이터
- ✅ API 실패 시 자동으로 simulation mode로 fallback
- ✅ 현재 시장 상황에서 Sweet-2 검증

**로그 확인**:
```
✅ Live data from BingX API: 200 candles, Latest: $XXX,XXX.XX
```

---

### Option 2: Simulation 모드

**사용 시나리오**: 인터넷 없거나 API 제한 시

1. API를 일부러 실패시키거나
2. 인터넷 연결 끊기

**자동 fallback**:
```
⚠️ Failed to get live data from API: [error]
Falling back to simulation mode (file data)
📁 Simulation data from file: 200 candles
```

---

## 📊 실시간 검증 시나리오

### Scenario 1: 단기 검증 (1시간)

**목적**: Sweet-2 작동 확인

```bash
# Sweet-2 실행 (Ctrl+C로 중단)
python scripts/production/sweet2_paper_trading.py

# 로그 실시간 모니터링
tail -f logs/sweet2_paper_trading_20251010.log
```

**체크리스트**:
- [ ] 5분마다 새 데이터 수집 확인
- [ ] XGBoost 신호 체크 작동
- [ ] Technical Strategy 신호 작동
- [ ] Market Regime 분류 작동
- [ ] Buy & Hold baseline 추적

**예상 결과**:
- 12 updates (1시간 = 12 × 5분)
- 0-2 trades (Sweet-2 보수적)
- 실시간 가격 변동 추적

---

### Scenario 2: 일일 검증 (24시간)

**목적**: 거래 발생 확인

```bash
# 백그라운드 실행
nohup python scripts/production/sweet2_paper_trading.py &

# 로그 확인
tail -f logs/sweet2_paper_trading_20251010.log
```

**체크리스트**:
- [ ] 288 updates (24시간 = 288 × 5분)
- [ ] 0-3 trades 발생 (Sweet-2: 2.5 trades/week)
- [ ] Regime 변화 관찰 (Bull/Bear/Sideways)
- [ ] 승률 추적
- [ ] vs Buy & Hold 계산

**예상 결과**:
- 1일 = 주당 거래 2.5 / 7 = 0.36 trades
- 실제: 0-3 trades 가능
- Bull/Bear/Sideways 다양한 regime 경험

---

### Scenario 3: 주간 검증 (7일)

**목적**: 통계적 샘플 확보

**목표**:
- 7일 × 0.36 = 2-3 trades
- 승률 > 50% 확인
- vs Buy & Hold > 0% 확인

**실행**:
```bash
# systemd/cron으로 지속 실행
# 또는 screen/tmux 세션

# 주간 성과 체크
python -c "
import pandas as pd
from pathlib import Path

trades = pd.read_csv('results/sweet2_paper_trading_trades_*.csv')
print(f'Total Trades: {len(trades)}')
print(f'Win Rate: {(trades[\"pnl_usd_net\"] > 0).mean() * 100:.1f}%')
print(f'vs B&H: {trades[\"pnl_usd_net\"].sum():.2f}')
"
```

---

## 🔧 BingX Testnet 설정 (선택사항)

현재 Sweet-2는 **Public API**를 사용하므로 credentials가 **불필요**합니다. 하지만 실제 주문을 테스트하려면 Testnet 계정이 필요합니다.

### Testnet 계정 생성

1. **BingX Testnet 가입**
   - https://testnet.bingx.com/ (존재 시)
   - 또는 BingX 공식 문서 참고

2. **API Key 생성**
   - Testnet Account → API Management
   - Create API Key
   - API Key와 Secret Key 저장

3. **환경 변수 설정**
```bash
# Windows (PowerShell)
$env:BINGX_API_KEY="your_api_key_here"
$env:BINGX_API_SECRET="your_secret_key_here"
$env:BINGX_USE_TESTNET="true"

# Linux/Mac
export BINGX_API_KEY="your_api_key_here"
export BINGX_API_SECRET="your_secret_key_here"
export BINGX_USE_TESTNET="true"
```

4. **영구 설정** (`.env` 파일)
```bash
# Create .env file in project root
echo "BINGX_API_KEY=your_api_key_here" > .env
echo "BINGX_API_SECRET=your_secret_key_here" >> .env
echo "BINGX_USE_TESTNET=true" >> .env
```

**Note**: 현재는 데이터 수집만 하므로 API credentials 불필요. 실제 주문 실행 시에만 필요.

---

## 📈 모니터링 및 로그

### 로그 위치

```
logs/
└── sweet2_paper_trading_20251010.log  # 일일 로그
```

### 실시간 모니터링

```bash
# 전체 로그 보기
tail -f logs/sweet2_paper_trading_20251010.log

# 신호만 보기
tail -f logs/sweet2_paper_trading_20251010.log | grep "Signal Check"

# 거래만 보기
tail -f logs/sweet2_paper_trading_20251010.log | grep "ENTRY\|EXIT"

# 성과만 보기
tail -f logs/sweet2_paper_trading_20251010.log | grep "PERFORMANCE"
```

### 결과 파일

```
results/
├── sweet2_paper_trading_trades_*.csv      # 거래 기록
├── sweet2_market_regime_history_*.csv     # Regime 히스토리
└── sweet2_paper_trading_state.json        # 현재 상태
```

---

## 🎯 검증 체크리스트

### 기술적 검증

- [x] BingX API 연결 성공
- [x] 5분 캔들 데이터 수집
- [x] 실시간 가격 업데이트
- [x] Sweet-2 bot 초기화
- [x] Feature 계산 (XGBoost + Technical)
- [x] Hybrid Strategy 신호 생성
- [x] Market Regime 분류
- [x] Buy & Hold baseline 추적
- [ ] 실제 거래 발생 (시간 필요)
- [ ] 승률 계산 (거래 필요)
- [ ] vs Buy & Hold 계산 (거래 필요)

### 성과 검증 (1-2주 필요)

- [ ] 10+ trades 발생
- [ ] 승률 > 50%
- [ ] vs Buy & Hold > 0%
- [ ] Per-trade net > 0%
- [ ] Bull/Bear/Sideways 각 regime 경험

---

## 🔍 비판적 검증 질문

### Q1: "실시간 API 데이터가 백테스팅과 다른가?"

**Answer**:
- ✅ API 데이터: 실제 현재 시장 상황
- ✅ 백테스팅 데이터: 과거 historical 데이터
- ⚠️ Market conditions가 달라서 성과가 다를 수 있음

**Expected**:
- Sweet-2는 과거 11 windows에서 +0.75% vs B&H
- 실시간에서도 유사한 성과 예상
- 하지만 통계적 유의성 부족 (p=0.51)

---

### Q2: "왜 거래가 안 생기는가?"

**Answer**: Sweet-2는 **매우 보수적**

**Thresholds**:
```
xgb_strong = 0.7      # 70% 이상 확률 필요
xgb_moderate = 0.6    # 60% 이상 확률 필요
tech_strength = 0.75  # 75% 이상 기술적 강도 필요
```

**Expected frequency**:
- 주당 2.5 trades
- 일일 0.36 trades
- **1-2일 거래 없어도 정상**

**Log example**:
```
Signal Check:
  XGBoost Prob: 0.499  # < 0.6 threshold ❌
  Tech Signal: LONG (strength: 0.600)  # < 0.75 threshold ❌
  Should Enter: False
```

---

### Q3: "실시간 검증 vs 백테스팅 차이는?"

**백테스팅**:
- ✅ 빠른 검증 (몇 분)
- ✅ 통제된 환경
- ⚠️ Historical data bias
- ⚠️ No slippage, perfect execution

**실시간 검증**:
- ✅ 실제 시장 조건
- ✅ 실시간 가격 변동
- ⚠️ 느린 검증 (1-2주 필요)
- ⚠️ API limitations, network issues

**비판적 결론**:
> "백테스팅으로 가능성을 보았고,
> 실시간 검증으로 진실을 확인한다."

---

## 📊 예상 결과

### Week 1 (Best Case)

```
Total Trades: 3
Win Rate: 66.7% (2/3)
vs B&H: +0.5%
Per-trade Net: +0.17%

→ ✅ Continue to Week 2
```

### Week 1 (Realistic Case)

```
Total Trades: 1-2
Win Rate: 50-100% (insufficient sample)
vs B&H: -0.2% to +0.3%
Per-trade Net: -0.1% to +0.2%

→ ⏳ Continue, need more data
```

### Week 1 (Worst Case)

```
Total Trades: 0-1
Win Rate: N/A or 0%
vs B&H: -0.5%
Per-trade Net: N/A or negative

→ ⚠️ Review settings, check logs
```

---

## 🚨 문제 해결

### Issue 1: "No trades for days"

**Cause**: Sweet-2 보수적, 또는 market conditions

**Solution**:
1. Check logs: Signal 확인 (XGBoost prob, Tech strength)
2. Check regime: Bull에서 거래 적음 (예상됨)
3. Wait: 1주일 기다려도 < 2 trades면 threshold 검토

### Issue 2: "API connection failed"

**Cause**: Network issues, API down

**Solution**:
1. Check internet connection
2. Test: `python scripts/production/test_bingx_api.py`
3. Fallback: Simulation mode 자동 활성화됨

### Issue 3: "Negative returns"

**Cause**: Market conditions, Bull market

**Solution**:
1. Check regime distribution: Bull에서 -4.45% 예상
2. Wait: 2주 후 전체 판단
3. IF persistent: Implement 15m features or regime-specific

---

## 🎓 최종 권장사항

### 즉시 실행 (오늘!)

```bash
# Sweet-2 실시간 검증 시작
python scripts/production/sweet2_paper_trading.py

# 별도 터미널에서 로그 모니터링
tail -f logs/sweet2_paper_trading_*.log
```

### Week 1 목표

- [ ] 24/7 실행 유지 (또는 최대한 많은 시간)
- [ ] 거래 발생 확인 (0-3 trades)
- [ ] 신호 패턴 관찰
- [ ] Regime 분포 확인

### Week 2 판정

- [ ] 10-20 trades 확보
- [ ] 승률 > 50% 달성
- [ ] vs B&H > 0% 확인
- [ ] Go/No-go 결정

---

## ✅ 검증 완료 상태

**완료된 검증**:
- [x] BingX API 실시간 데이터 수집 ✅
- [x] Sweet-2 bot live API 연동 ✅
- [x] Update cycle 작동 확인 ✅
- [x] XGBoost + Technical 신호 생성 ✅
- [x] Market Regime 분류 ✅

**실행 대기 중**:
- [ ] 1-2주 실시간 검증 (사용자 선택)
- [ ] 통계적 샘플 확보 (20+ trades)
- [ ] 최종 go/no-go 결정

---

**"비판적 사고를 통해 실시간 API 연동까지 완료했습니다. Sweet-2는 실제 시장 데이터로 작동할 준비가 되었습니다. 이제 시간을 주고 진실을 밝힐 차례입니다."** 🎯

**Date**: 2025-10-10
**Status**: ✅ **실시간 검증 준비 완료**
**Next**: `python scripts/production/sweet2_paper_trading.py` 실행
