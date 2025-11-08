# 🚀 실행 가이드 - Paper Trading & Hybrid Strategy

**Date**: 2025-10-09
**Status**: ✅ 즉시 실행 가능
**비판적 사고**: "분석만 하고 실행 안 하면 무용지물"

---

## 📋 빠른 시작 (5분)

### Option A: Paper Trading (추천 - 제로 리스크) ⭐⭐⭐

```bash
# 1. 환경 변수 설정 (선택)
export BINGX_TESTNET_API_KEY="your_api_key"
export BINGX_TESTNET_API_SECRET="your_api_secret"

# 2. Paper Trading Bot 실행
cd bingx_rl_trading_bot
python scripts/paper_trading_bot.py
```

**결과**:
- 5분마다 시장 데이터 수집
- XGBoost 예측 및 거래
- 시장 상태 분류 (상승/횡보/하락)
- 실시간 성과 추적
- 로그 및 CSV 파일 자동 저장

---

### Option B: Hybrid Strategy (실전 준비) ⭐⭐⭐

```bash
# 1. 데모 실행 (시뮬레이션)
python scripts/hybrid_strategy_manager.py demo

# 2. 실제 사용 (코드 통합 필요)
# - 70% BTC 매수
# - 30% Paper Trading Bot 실행
# - Hybrid Manager로 통합 추적
```

---

## 🎯 상세 실행 가이드

### Step 1: 환경 준비 (10분)

#### 1.1 Python 패키지 확인

```bash
pip install -r requirements.txt

# 필수 패키지:
# - pandas
# - numpy
# - scikit-learn
# - xgboost
# - ta (technical analysis)
# - loguru
# - requests
```

#### 1.2 XGBoost 모델 확인

```bash
# 모델 파일 위치 확인
ls models/xgboost_model.pkl

# 없으면 학습 필요
python scripts/train_xgboost.py
```

#### 1.3 BingX Testnet 계정 (선택)

**API 사용 시** (실제 testnet):
1. https://testnet.bingx.com 접속
2. 계정 생성
3. API Key 발급
4. 환경 변수 설정

**API 없이 사용 시** (시뮬레이션):
- 로컬 데이터 파일 사용
- `data/BTCUSDT_5m_max.csv` 필요

---

### Step 2: Paper Trading 실행 (2-4주)

#### 2.1 Bot 시작

```bash
# 기본 실행
python scripts/paper_trading_bot.py

# 백그라운드 실행
nohup python scripts/paper_trading_bot.py > paper_trading.log 2>&1 &

# 로그 확인
tail -f logs/paper_trading_20251009.log
```

#### 2.2 모니터링

**실시간 로그**:
```
================================================================================
Paper Trading Bot Started
================================================================================
Initial Capital: $10,000.00
Entry Threshold: 0.20%
Stop Loss: 1.0%
Take Profit: 3.0%
================================================================================

Update: 2025-10-09 14:30:00
Market Regime: Bull
Current Price: $60,523.45
Capital: $10,000.00

Prediction: 1, Probability: 0.652, Expected Return: 0.304, Should Enter: True

🔔 ENTRY: LONG 0.1580 BTC @ $60,523.45
   Position Value: $9,500.00
   Market Regime: Bull
   Prediction Probability: 0.652
```

**성과 파일**:
- `results/paper_trading_trades_YYYYMMDD_HHMMSS.csv` - 거래 내역
- `results/market_regime_history_YYYYMMDD_HHMMSS.csv` - 시장 상태 이력
- `results/paper_trading_state.json` - 현재 상태

#### 2.3 성과 평가 (2-4주 후)

```python
import pandas as pd

# 거래 내역 로드
df = pd.read_csv('results/paper_trading_trades_20251109.csv')

# 전체 통계
total_trades = len(df)
win_rate = (len(df[df['pnl_usd'] > 0]) / total_trades) * 100
total_return = df['pnl_usd'].sum()

print(f"Total Trades: {total_trades}")
print(f"Win Rate: {win_rate:.1f}%")
print(f"Total P&L: ${total_return:,.2f}")

# 시장 상태별 성과
for regime in ['Bull', 'Bear', 'Sideways']:
    regime_df = df[df['regime'] == regime]
    if len(regime_df) > 0:
        regime_pnl = regime_df['pnl_usd'].sum()
        regime_wr = (len(regime_df[regime_df['pnl_usd'] > 0]) / len(regime_df)) * 100
        print(f"{regime}: {len(regime_df)} trades, {regime_wr:.1f}% WR, ${regime_pnl:+,.2f}")
```

**성공 기준**:
- ✅ Win Rate ≥ 50%
- ✅ 상승장: 70%+ 포착
- ✅ 횡보장: 양수 수익
- ✅ 하락장: 50%+ 방어 (if 있으면)

---

### Step 3: Hybrid Strategy 실행 (실전)

#### 3.1 자본 배분 ($1000 예시)

**70% Buy & Hold ($700)**:
```python
# 수동 실행
# 1. 거래소에서 $700 BTC 매수
# 2. 지갑 보관 또는 거래소 보관
# 3. 기록: 매수 가격, 수량

# 예시
btc_entry_price = 60000.0
btc_quantity = 700 / btc_entry_price  # 0.01166 BTC
```

**30% XGBoost Trading ($300)**:
```bash
# Paper Trading Bot 실행 (capital $300)
# config 수정 필요:
# - INITIAL_CAPITAL = 300.0
python scripts/paper_trading_bot.py
```

#### 3.2 통합 모니터링

```python
from scripts.hybrid_strategy_manager import HybridStrategyManager

# 초기화
manager = HybridStrategyManager(initial_capital=1000.0)
manager.initialize_buy_hold(current_btc_price=60000.0)

# 매일 업데이트 (수동)
current_btc_price = 61000.0  # API 또는 수동 입력
xgboost_capital = 310.0  # Paper Trading Bot 결과

portfolio = manager.get_portfolio_value(current_btc_price, xgboost_capital)
manager.print_portfolio_status(portfolio)

# 리밸런싱 체크
if manager.check_rebalancing_needed(portfolio):
    xgboost_capital = manager.rebalance(current_btc_price, xgboost_capital)

manager.record_performance(portfolio)
```

#### 3.3 자동화 (선택)

```python
# 완전 자동화 스크립트 예시
import time
from scripts.hybrid_strategy_manager import HybridStrategyManager
import ccxt  # 거래소 API

# 초기화
exchange = ccxt.binance()  # 또는 bingx
manager = HybridStrategyManager(initial_capital=1000.0)

# ... (매수 로직)

# 일일 루프
while True:
    # BTC 가격 조회
    ticker = exchange.fetch_ticker('BTC/USDT')
    current_btc_price = ticker['last']

    # XGBoost capital 조회 (paper_trading_bot state)
    with open('results/paper_trading_state.json') as f:
        state = json.load(f)
        xgboost_capital = state['capital']

    # 포트폴리오 업데이트
    portfolio = manager.get_portfolio_value(current_btc_price, xgboost_capital)
    manager.print_portfolio_status(portfolio)

    # 리밸런싱
    if manager.check_rebalancing_needed(portfolio):
        xgboost_capital = manager.rebalance(current_btc_price, xgboost_capital)
        # 실제 리밸런싱 거래 실행

    # 기록
    manager.record_performance(portfolio)
    manager.save_state()

    # 24시간 대기
    time.sleep(86400)
```

---

## 📊 성과 모니터링

### 일일 체크리스트

**Paper Trading**:
- [ ] Bot이 정상 실행 중인가? (`ps aux | grep paper_trading`)
- [ ] 오늘 거래가 발생했는가? (로그 확인)
- [ ] Win rate는? (현재 통계)
- [ ] 시장 상태는? (Bull/Bear/Sideways)

**Hybrid Strategy**:
- [ ] Buy & Hold BTC 수량 확인
- [ ] XGBoost capital 확인
- [ ] 전체 포트폴리오 가치
- [ ] 목표 비율 유지되는가? (70:30)

### 주간 리뷰 (매주 일요일)

```bash
# 1. Paper Trading 성과
python -c "
import pandas as pd
df = pd.read_csv('results/paper_trading_trades_latest.csv')
print('Week Stats:')
print(f'Trades: {len(df)}')
print(f'Win Rate: {(len(df[df.pnl_usd > 0]) / len(df)) * 100:.1f}%')
print(f'Total P&L: {df.pnl_usd.sum():.2f}')
"

# 2. Hybrid Strategy 리밸런싱 체크
python scripts/hybrid_strategy_manager.py check_rebalance

# 3. 시장 상태 분포
python -c "
import pandas as pd
df = pd.read_csv('results/market_regime_history_latest.csv')
print(df['regime'].value_counts())
"
```

### 월간 평가 (매월 1일)

**질문**:
1. Paper Trading 승률은? (목표: 50%+)
2. 각 시장 상태별 성과는?
3. Hybrid Strategy가 pure Buy & Hold보다 나은가?
4. Max drawdown은? (목표: <5%)

**결정**:
- ✅ 성공: 소액 실전 배포 ($100-300)
- ⚠️ 부분 성공: 계속 paper trading
- ❌ 실패: 파라미터 조정 또는 Buy & Hold 전환

---

## 🔧 설정 및 최적화

### Paper Trading Bot 설정

`scripts/paper_trading_bot.py`의 `Config` 클래스:

```python
class Config:
    # Entry threshold (낮출수록 더 많은 거래)
    ENTRY_THRESHOLD = 0.002  # 0.2%

    # Risk management
    STOP_LOSS = 0.01  # 1%
    TAKE_PROFIT = 0.03  # 3%

    # Volatility filter
    MIN_VOLATILITY = 0.0008

    # Position sizing
    POSITION_SIZE_PCT = 0.95  # 95% of capital

    # Max holding period
    MAX_POSITION_HOURS = 24
```

### Hybrid Strategy 설정

`scripts/hybrid_strategy_manager.py`의 `HybridConfig`:

```python
class HybridConfig:
    # Allocation
    BUY_HOLD_PCT = 0.70  # 70%
    XGBOOST_PCT = 0.30   # 30%

    # Rebalancing
    REBALANCE_THRESHOLD = 0.05  # 5% deviation
    REBALANCE_FREQUENCY_DAYS = 7  # Weekly

    # Risk
    STOP_LOSS_PORTFOLIO_PCT = 0.15  # 15% max loss
```

---

## 🚨 문제 해결

### Bot이 실행 안 됨

```bash
# 모델 파일 확인
ls -la models/xgboost_model.pkl

# 없으면 학습
python scripts/train_xgboost.py

# 데이터 파일 확인
ls -la data/BTCUSDT_5m_max.csv
```

### 거래가 발생 안 함

**원인**:
1. Entry threshold가 너무 높음 (0.003 → 0.002로 낮추기)
2. Volatility가 너무 낮음 (MIN_VOLATILITY 조정)
3. 예측 확률이 낮음 (모델 재학습 필요)

### API 에러

```python
# BingX Testnet API 확인
import requests
response = requests.get('https://open-api-vst.bingx.com/openApi/swap/v3/quote/klines',
                       params={'symbol': 'BTC-USDT', 'interval': '5m', 'limit': 10})
print(response.json())
```

---

## 📚 참조 문서

**분석 결과**:
- `START_HERE_FINAL.md` - 최종 요약
- `claudedocs/MARKET_REGIME_TRUTH.md` - 시장 상태 분석
- `claudedocs/CRITICAL_CONTRADICTIONS_FOUND.md` - 통계적 분석

**스크립트**:
- `scripts/paper_trading_bot.py` - Paper trading
- `scripts/hybrid_strategy_manager.py` - Hybrid strategy
- `scripts/market_regime_analysis.py` - 시장 상태 분석

**결과 파일**:
- `results/paper_trading_trades_*.csv` - 거래 내역
- `results/market_regime_history_*.csv` - 시장 상태
- `results/hybrid_strategy_performance_*.csv` - Hybrid 성과

---

## ✅ 실행 체크리스트

### 오늘 (즉시)

- [ ] **XGBoost 모델 확인** (`models/xgboost_model.pkl`)
- [ ] **Paper Trading Bot 실행** (`python scripts/paper_trading_bot.py`)
- [ ] **로그 확인** (`tail -f logs/paper_trading_*.log`)

### 이번 주

- [ ] **일일 모니터링** (거래, 승률, 시장 상태)
- [ ] **성과 추적** (CSV 파일 확인)
- [ ] **Bot 정상 작동** 확인

### 2-4주 후

- [ ] **Paper Trading 평가**
  - Win rate ≥ 50%?
  - 각 시장 상태별 성과는?
  - Sharpe ratio > 0.3?

- [ ] **결정**
  - ✅ 성공 → 소액 실전 또는 Hybrid
  - ❌ 실패 → 파라미터 조정 또는 Buy & Hold

---

## 🏆 성공 기준

### Paper Trading 성공

- ✅ **Win Rate**: 50%+
- ✅ **상승장**: 70%+ 포착
- ✅ **횡보장**: 양수 수익
- ✅ **하락장**: 50%+ 방어
- ✅ **Sharpe Ratio**: > 0.3
- ✅ **Max DD**: < 5%
- ✅ **안정성**: 2-4주 지속

### Hybrid Strategy 성공

- ✅ **Total Return**: ≥ Pure Buy & Hold × 0.95
- ✅ **Max DD**: < Pure Buy & Hold
- ✅ **Sharpe Ratio**: ≥ Buy & Hold
- ✅ **비율 유지**: 70:30 ± 5%

---

**비판적 사고**: "분석은 완료했다. 이제 실행하고 검증할 시간이다."

**다음 단계**: Paper Trading Bot을 지금 바로 실행하세요! 🚀

```bash
python scripts/paper_trading_bot.py
```
