# CLAUDE_CODE_FIN - Workspace Overview

**Last Updated**: 2025-12-25 KST (RSI Martingale 불일치 분석 + 연구-프로덕션 검증 가이드 추가)

---

## 🎯 Active Bots

### SuperTrend 5m Bot v1.0 ✅ NEW - High Frequency
**파일**: `scripts/production/supertrend_5m_bot.py`
**설정**: `config/supertrend_5m_config.yaml`
**상태**: ✅ **v1.0 배포 완료** - 고빈도 전략

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| **Entry (LONG)** | **SuperTrend Direction -1 → +1** | 하락추세 → 상승추세 전환 |
| **Entry (SHORT)** | **SuperTrend Direction +1 → -1** | 상승추세 → 하락추세 전환 |
| ATR Period | 10 | SuperTrend 계산용 |
| Multiplier | 2.2 | 밴드 폭 |
| **Take Profit** | **0.7%** | 고정 |
| **Stop Loss** | **1.0%** | 고정 |
| **Cooldown** | **1 candle** | **5분** |
| **Position Mode** | **One-Way** | |
| **Exchange Leverage** | **10x** | |
| Effective Leverage | 4x | 포지션 크기 계산용 |
| Timeframe | 5m | **고빈도** |

**검증 결과 (Walk-Forward 6 Windows, 90 days)**:

| 메트릭 | 값 | 비고 |
|--------|-----|------|
| **Profitable Windows** | **83%** (5/6) | 높은 일관성 |
| **Total PnL** | **+42.8%** | 90일 기준 |
| **Win Rate** | **69.5%** | LONG 66.2%, SHORT 73.0% |
| **Trades/Day** | **1.36** | 목표 달성 |
| **Max Drawdown** | **16.4%** | 안정적 |

**Deep Comparison (Strategy A vs B)**:
- **Strategy A (TP0.7/SL1.0)**: LONG +20.4%, SHORT +48.2% - **양방향 수익** ✅
- Strategy B (TP1.0/SL0.9): LONG -7.6% (손실!), SHORT +76.0% - 롱 손실

```bash
# Commands
START_SUPERTREND_5M.bat                              # Start (background)
STOP_SUPERTREND_5M.bat                               # Stop bot
MONITOR_SUPERTREND_5M.bat                            # Monitor
python scripts/production/supertrend_5m_bot.py       # Start (direct)
cat results/supertrend_5m_bot_state.json             # State
cat config/supertrend_5m_config.yaml                 # Config
```

---

### MACD Martingale Bot v1.0 ✅ NEW - Corrected Backtest Logic
**파일**: `scripts/production/macd_martingale_bot.py`
**설정**: `config/macd_martingale_config.yaml`
**상태**: ✅ **v1.0 배포 완료** - 수정된 백테스트 로직 적용

> **✅ BACKTEST CORRECTIONS APPLIED** (2025-12-24):
> 1. Entry at **NEXT bar OPEN** (not same bar close)
> 2. TP/SL detection using **HIGH/LOW** (not close)
> 3. Position capped at **10x balance** (exchange limit)
> 4. Conservative: **SL hit first** when both TP/SL possible

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| **Entry (LONG)** | **MACD Hist < 0 → ≥ 0 + ADX ≥ 12** | Histogram 제로선 상향 돌파 |
| **Entry (SHORT)** | **MACD Hist > 0 → ≤ 0 + ADX ≥ 12** | Histogram 제로선 하향 돌파 |
| MACD Fast | 12 | 단기 EMA |
| MACD Slow | 26 | 장기 EMA |
| MACD Signal | 9 | 시그널 라인 |
| **ADX Filter** | **≥ 12** | 추세 강도 필터 (낮은 임계값) |
| **Take Profit** | **2.0%** | 고정 |
| **Stop Loss** | **2.0%** | 고정 (1:1 R:R) |
| **Martingale** | **Enabled** | 손실 후 포지션 증가 |
| **Martingale Max** | **8x** | 최대 마틴게일 배수 |
| **Exchange Max** | **10x** | 포지션 하드 캡 |
| Leverage | 5 | 유효 레버리지 |
| Position Size | 25% | 기본 포지션 크기 |
| Timeframe | 15m | |

**Entry Logic (MACD Histogram Cross + ADX)**:
- **LONG**: MACD Hist(prev) < 0 AND MACD Hist(now) ≥ 0 AND ADX ≥ 12
- **SHORT**: MACD Hist(prev) > 0 AND MACD Hist(now) ≤ 0 AND ADX ≥ 12

**Position Sizing (Martingale with Exchange Cap)**:
```python
# Martingale: 2^consecutive_losses, capped at 8x
mult = min(2 ** consecutive_losses, 8)
raw_pos = balance * 25% * 5 * mult
# Exchange cap at 10x
position = min(raw_pos, balance * 10)
```

**Exit Logic (High/Low Detection)**:
- **TP Hit**: bar_high ≥ tp_price (LONG) or bar_low ≤ tp_price (SHORT)
- **SL Hit**: bar_low ≤ sl_price (LONG) or bar_high ≥ sl_price (SHORT)
- **Conservative**: When both TP/SL possible in same bar → SL assumed first

**검증 결과 (수정된 백테스트, Walk-Forward 6 Windows, 314일)**:

| 메트릭 | 원래 (결함) | 수정 후 | 비고 |
|--------|------------|---------|------|
| **Daily Return** | 0.93% | **0.65%** | 여전히 0.5%+ 목표 달성 |
| **Total Return** | +291.9% | **+203.1%** | 현실적 수치 |
| **Max Drawdown** | 47% | **59%** | 더 높은 리스크 |
| **Walk-Forward** | 5/6 | **5/6** | 일관성 유지 |

| 윈도우 | 기간 | PnL | Daily % | 상태 |
|--------|------|-----|---------|------|
| W1 | Days 1-52 | +94.4% | 1.82% | ✅ 0.5%+ |
| W2 | Days 53-104 | +6.5% | 0.13% | ⚠️ 저조 |
| W3 | Days 105-156 | +38.0% | 0.73% | ✅ 0.5%+ |
| W4 | Days 157-208 | +52.1% | 0.98% | ✅ 0.5%+ |
| W5 | Days 209-260 | +29.3% | 0.56% | ✅ 0.5%+ |
| W6 | Days 261-314 | -17.2% | -0.33% | ❌ 손실 |

**⚠️ 리스크 경고**:
- **Max Drawdown 59%**: 높은 드로다운 예상
- **6연속 손실 관측**: 마틴게일 최대 배수 도달 시나리오 발생
- **Window 6 손실**: 일부 기간 손실 가능
- **적합 대상**: 리스크 감내 가능한 트레이더만

```bash
# Commands
START_MACD_MARTINGALE.bat                            # Start (background)
STOP_MACD_MARTINGALE.bat                             # Stop bot
MONITOR_MACD_MARTINGALE.bat                          # Monitor
python scripts/production/macd_martingale_bot.py     # Start (direct)
cat results/macd_martingale_bot_state.json           # State
cat config/macd_martingale_config.yaml               # Config
```

---

### MACD+DCA Bot v1.1 ❌ DEPRECATED - 백테스트 방법론 오류
**파일**: `scripts/production/macd_dca_bot.py`
**설정**: `config/macd_dca_bot_config.yaml`
**상태**: ❌ **폐기됨** - 백테스트 방법론 결함 발견

> ⚠️ **CRITICAL WARNING**: 원래 백테스트(+918.9%)는 **고정 단위(1.0)** 기반으로 계산되어 **복리 효과를 반영하지 않았습니다**.
> 잔고 기반 포지션 사이징(33% × 4x)으로 재계산 시 **실제 수익률: -46.5%** (손실 전략!)
>
> 📊 **교훈**: 모든 백테스트는 반드시 잔고 기반 포지션 사이징과 복리 효과를 반영해야 합니다.
> 상세 내용은 아래 "⚠️ Backtest Methodology (CRITICAL)" 섹션 참조

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| **Entry (LONG)** | **MACD Hist < 0 → ≥ 0** | Histogram 제로선 상향 돌파 |
| **Entry (SHORT)** | **MACD Hist > 0 → ≤ 0** | Histogram 제로선 하향 돌파 |
| MACD Fast | 12 | 단기 EMA |
| MACD Slow | 26 | 장기 EMA |
| MACD Signal | 9 | 시그널 라인 |
| **ADX Filter** | **≥ 20** | 추세 강도 필터 |
| **Take Profit** | **2.0%** | 고정 |
| **Stop Loss** | **1.5%** | 고정 |
| **BE Trigger** | **1.0%** | SL → Entry로 이동 |
| **Trail %** | **0.5%** | BE 이후 추적 |
| **DCA Trigger** | **0.8%** | 손실 시 추가 진입 |
| **DCA Max** | **2회** | 최대 추가 진입 횟수 |
| **Reverse Close** | **50%** | **v1.1 핵심** - 역신호 시 50% 청산 |
| **Position Mode** | **One-Way** | |
| **Exchange Leverage** | **10x** | |
| Effective Leverage | 4x | 포지션 크기 계산용 |
| Base Position | 33% | DCA 여유분 확보 |
| Timeframe | 15m | |

**Entry Logic (MACD Histogram Crossover)**:
- **LONG**: MACD Hist(prev) < 0 AND MACD Hist(now) ≥ 0 AND ADX ≥ 20
- **SHORT**: MACD Hist(prev) > 0 AND MACD Hist(now) ≤ 0 AND ADX ≥ 20

**Exit Logic (BE+Trail + DCA + Reverse Signal)**:
1. **Initial**: TP 2.0%, SL 1.5%
2. **BE Trigger**: 1% 수익 도달 시 → SL을 Entry 가격으로 이동
3. **Trail**: BE 활성화 후 → 0.5% 뒤에서 추적
4. **DCA**: 0.8% 손실 시 → 동일 방향 추가 진입 (최대 2회)
5. **⚡ Reverse Signal (v1.1)**: 역방향 신호 시 → **50% 부분 청산**
   - LONG 보유 중 SHORT 신호 → 50% 청산 후 잔여 유지
   - SHORT 보유 중 LONG 신호 → 50% 청산 후 잔여 유지

**❌ 폐기 사유** (2025-12-23):

| 백테스트 방법 | Total PnL | Walk-Forward | Max DD | 결론 |
|--------------|-----------|--------------|--------|------|
| ~~고정 단위 (1.0)~~ | ~~+918.9%~~ | ~~6/6~~ | ~~1.5%~~ | ~~"좋아 보임"~~ |
| **잔고 기반 (33% × 4x)** | **-46.5%** | **2/6** | **60.5%** | **손실 전략!** |

- 원래 백테스트는 `size = 1.0` 고정으로 복리 효과 미반영
- 잔고 기반 재계산 시 **실제로 -46.5% 손실** 발생
- Walk-Forward 2/6 (33%)로 일관성 부족 확인
- **결론**: 전략 자체가 손실 구조, 봇 운영 중단

```bash
# ❌ DEPRECATED - 사용 금지
# 백테스트 방법론 결함으로 폐기된 봇입니다
# 상세 교훈은 "⚠️ Backtest Methodology (CRITICAL)" 섹션 참조
```

---

### MS_ChoCH Bot v1.2 ❌ DEPRECATED - Look-Ahead Bias 발견
**파일**: `scripts/production/ms_choch_bot.py`
**설정**: `config/ms_choch_bot_config.yaml`
**상태**: ❌ **폐기** - Look-Ahead Bias로 연구 결과 무효화

> **🔴 CRITICAL ISSUE (2025-12-24)**:
> 연구 스크립트에서 **Look-Ahead Bias** 발견
> - 연구 결과: **+609.1%** (허위)
> - 실제 백테스트: **-6.50%** (손실)
> - 원인: Swing Point 감지 시 미래 데이터 (`shift(-1)`, `center=True`) 참조
> - 상세: `claudedocs/MS_CHOCH_DISCREPANCY_ANALYSIS_20251224.md`

**폐기 사유**:
1. Swing High/Low 감지에 **미래 5봉** 데이터 사용
2. Walk-Forward 결과도 동일 데이터셋 내에서 수행 (진정한 OOS 없음)
3. 월별 백테스트 전 기간 손실 (-6.50%, 0% 수익월)

| 기간 | PnL | Win Rate | 비고 |
|------|-----|----------|------|
| Month 1 (11/24-12/24) | -1.50% | 36.4% | 손실 |
| Month 2 (10/25-11/24) | -1.00% | 36.8% | 손실 |
| Month 3 (09/25-10/25) | -4.00% | 33.3% | 손실 |
| **합계** | **-6.50%** | **35.5%** | **전 기간 손실** |

```bash
# Commands
START_MS_CHOCH.bat                              # Start (background)
STOP_MS_CHOCH.bat                               # Stop bot
MONITOR_MS_CHOCH.bat                            # Monitor
python scripts/production/ms_choch_bot.py       # Start (direct)
cat results/ms_choch_bot_state.json             # State
cat config/ms_choch_bot_config.yaml             # Config
```

---

### RSI Trend Filter Bot v2.0 ✅ ACTIVE - BE+Trail + PreBE Spike Protection
**파일**: `scripts/production/rsi_trend_filter_bot.py`
**설정**: `config/rsi_trend_filter_config.yaml`
**상태**: ✅ **v2.0 운영 중** - BE+Trail + PreBE 스파이크 보호 (LONG 수익 전환!)

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| **Entry (LONG)** | **RSI < 35 + BB% < 0.2 + EMA100** | Buy Low 전략 |
| **Entry (SHORT)** | **RSI > 65 + BB% > 0.8 + EMA100** | Sell High 전략 |
| RSI Period | 14 | 표준 RSI |
| EMA Period | 100 | 추세 필터 |
| **Bollinger Bands** | **BB(20), <0.2/>0.8** | v1.8 추가 |
| **Take Profit** | **4.0%** | v1.7 변경 |
| **Initial Stop Loss** | **1.5%** | v1.7 변경 |
| **BE Trigger** | **1.0%** | SL → Entry로 이동 |
| **Trail %** | **0.5%** | 일반 추적 |
| **Tight Trail %** | **0.15%** | **v1.9 신규** - 스파이크 감지 시 |
| **Spike Lookback** | **12 candles** | **v1.9 신규** - EMA slope 측정 |
| **Slope Threshold** | **0.05%** | **v1.9 신규** - 추세 판단 |
| **Cooldown** | **0** | **v1.10**: 백테스트 검증 결과 cooldown 불필요 |
| **Position Mode** | **One-Way** | |
| **Exchange Leverage** | **10x** | |
| Effective Leverage | 4x | 포지션 크기 계산용 |
| Timeframe | 5m | v1.8 변경 |

**Entry Logic (v1.8 Buy Low)**:
- **LONG**: Close > EMA(100) AND RSI < **35** AND BB% < **0.2**
- **SHORT**: Close < EMA(100) AND RSI > **65** AND BB% > **0.8**

**Exit Logic (v1.9 BE+Trail + Spike Protection)**:
1. **Initial**: TP 4.0%, SL 1.5%
2. **BE Trigger**: 1% 수익 도달 시 → SL을 Entry 가격으로 이동
3. **Normal Trail**: BE 활성화 후 → SL이 highest/lowest에서 **0.5%** 뒤를 추적
4. **⚡ Spike Protection (v1.9)**: 역추세 스파이크 감지 시 → Trail을 **0.15%**로 단축
   - LONG + 하락추세 + 가격이 EMA 위 = 스파이크 → Tight Trail
   - SHORT + 상승추세 + 가격이 EMA 아래 = 스파이크 → Tight Trail
5. **Exit Types**: TP (익절), SL (손절), TRAIL (추적), TIGHT (스파이크 추적)

**⚠️ v3.2 Time Exit 연구 중단** (2025-12-25):
> **🔴 CRITICAL**: PM Research에서 Time Exit 백테스트에 **Look-Ahead Bias** 발견
> - 원래 결과: +2772% ~ +3335% (허위)
> - 수정 결과: **-35% ~ -39%** (손실 전략!)
> - 원인: Entry 봉에서 동일 봉 High/Low로 Exit 체크 (bars_held: 0)
> - 조치: Time Exit v3.2 **비활성화**, Structure Exit v3.0 유지
> - 상세: `scripts/analysis/pm_research_correct.py`

**v2.0 변경사항** (2025-12-22):
- **🚀 PreBE Spike Protection**: BE 활성화 전 스파이크 감지 시 조기 청산
- **LONG 수익 전환**: -1.62% → **+2.02%** (손실 → 수익!)
- **SHORT 개선**: +3.34% → **+8.46%** (+5.12% 개선)
- **Win Rate 상승**: 60.0% → **69.6%** (+9.6%p)
- **Spike Exits**: 8건 스파이크 조기 청산으로 손실 방지
- **Walk-Forward PnL**: +1.0% → **+11.3%** (+10.4% 개선)

**v1.10 변경사항** (2025-12-22):
- Cooldown 제거 (1 → 0), Risk Limits 비활성화

**v1.9 변경사항** (2025-12-21):
- **⚡ 스파이크 보호**: 역추세 스파이크 감지 시 Trail 단축 (0.5% → 0.15%)
- **LONG 손실 감소**: -28.4% → -18.1% (**36% 개선**)
- **Walk-Forward PnL**: +47.7% → +61.0% (**+13.3% 개선**)
- **연구 기반**: 12 lookback, 0.05% slope, 0.15% tight trail 최적화

**v1.8 변경사항** (2025-12-21):
- 5m 타임프레임, BB 필터 추가, MTF/ADX/ATR 비활성화

**검증 결과 (v1.9 Spike Protection)**:

| 메트릭 | v1.8 (기존) | v1.9 (Spike) | 개선 |
|--------|------------|--------------|------|
| **Full Period PnL** | +47.7% | **+61.0%** | **+13.3%** |
| **LONG PnL** | -28.4% | **-18.1%** | **+10.3%** |
| **SHORT PnL** | +76.0% | +79.1% | +3.1% |
| **Walk-Forward PnL** | +49.0% | **+59.3%** | **+10.3%** |
| **LONG 손실 감소** | - | - | **36%** |

```bash
# Commands
START_RSI_TREND_FILTER.bat                              # Start (foreground)
START_RSI_TREND_FILTER_BACKGROUND.bat                   # Start (background, VS Code 독립)
STOP_RSI_TREND_FILTER.bat                               # Stop bot
MONITOR_RSI_TREND_FILTER.bat                            # Monitor
python scripts/production/rsi_trend_filter_bot.py       # Start (direct)
python scripts/monitoring/rsi_trend_filter_monitor.py   # Monitor (direct)
cat results/rsi_trend_filter_bot_state.json             # State
cat config/rsi_trend_filter_config.yaml                 # Config
```

**프로세스 독립성** (2025-12-20 추가):
- `START_RSI_TREND_FILTER_BACKGROUND.bat` 사용 시 VS Code/Claude Code 종료해도 봇 계속 실행
- 로그 파일: `logs/rsi_trend_filter_bot_YYYYMMDD.log`

### 파라미터 변경 예시
```
# 전략 파라미터
"TP를 2.5%로 변경해줘" → config/rsi_trend_filter_config.yaml 수정
"RSI 기준을 35/65로" → strategy.rsi_long_threshold: 35, rsi_short_threshold: 65
"EMA 기간을 200으로" → strategy.ema_period: 200
```

---

## 🔬 Strategy Research (2025-12-16)

### 연구 배경
- ADX Supertrend Trail Bot 백테스트 버그 발견 (+1276%는 허위 결과)
- 수정된 백테스트: **-234.1%** (손실 전략)
- 8개 대안 전략 비교 연구 진행
- RSI Trend Filter가 최적 전략으로 선정

### 대안 전략 비교 (8 strategies × 5 TP/SL combinations)

| Strategy | Best TP/SL | Return | Notes |
|----------|------------|--------|-------|
| **RSI Trend Filter** | **3.0/2.0** | **+120.8%** | **선정됨** |
| RSI Reversal | 2.5/1.5 | +78.3% | - |
| EMA Crossover | 3.0/2.0 | +65.2% | - |
| Bollinger Bounce | 2.0/1.5 | +45.7% | - |
| Donchian Breakout | 3.5/2.5 | +32.1% | - |
| Supertrend Flip | 3.0/2.0 | +28.4% | - |
| Long-Only Pullback | 2.5/2.0 | +21.3% | - |
| Volatility Breakout | 3.0/2.5 | +15.8% | - |

### RSI Parameter Optimization

| Variant | Windows | PnL | Sharpe | P-value |
|---------|---------|-----|--------|---------|
| RSI 35/65 EMA200 (original) | 4/7 | +54.4% | 0.89 | 0.38 |
| **RSI 40/60 EMA100** | **6/7** | **+120.8%** | **1.31** | **0.013** |
| RSI 45/55 EMA100 | 5/7 | +87.3% | 1.12 | 0.08 |
| RSI 30/70 EMA100 | 3/7 | +23.1% | 0.45 | 0.52 |

**문서**: `claudedocs/RSI_TREND_FILTER_RESEARCH_20251216.md`
**스크립트**:
- `scripts/analysis/alternative_strategies_research.py`
- `scripts/analysis/rsi_strategy_deep_research.py`
- `scripts/analysis/best_strategy_validation.py`

### 🔬 Comprehensive V3 Research (2025-12-22)

**Entry/Exit 최적화 연구** - 17개 전략 비교

| 전략 | Full PnL | WF PnL | WR | Sharpe | WF 일관성 |
|------|----------|--------|-----|--------|----------|
| **Exit_Partial_50%@2.0%** | **+20.16%** | **+19.21%** | 72.5% | 1.83 | 4/6 |
| **Exit_Partial_50%@1.5%** | **+20.02%** | **+18.51%** | 75.1% | 1.79 | 4/6 |
| **Entry_RSI_Divergence** | +14.80% | +17.97% | 66.2% | **2.18** | **5/6** ✅ |
| v2.0 Baseline | +17.59% | +16.62% | 67.7% | 1.56 | 4/6 |
| Entry_BodyRatio_0.5 | +15.30% | +14.26% | 69.6% | 1.59 | 4/6 |

**핵심 발견**:
1. **Partial Exit (50%@2.0%)**: 최고 PnL (+2.57% 개선), 50% 조기 수익 실현 효과적
2. **RSI Divergence Entry**: 최고 일관성 (5/6 WF), Sharpe 2.18로 리스크 대비 최고
3. **Entry 필터 실패**: PriceEff, Volume 필터는 과도한 거래 감소로 성능 악화
4. **ATR Dynamic Exit 실패**: WR 38.3%로 whipsaw 다수

**결론**: v2.0 유지 권장. 업그레이드 시 Partial Exit 또는 RSI Divergence 선택적 적용.

**결과 파일**:
- `results/comprehensive_v3_full_20251222_172833.csv`
- `results/comprehensive_v3_wf_20251222_172833.csv`

### 🏆 MACD+DCA v1.1 장기 전략 연구 (2025-12-23)

**연구 목적**: 314일 장기 검증에서 양방향 수익 + 낮은 Drawdown 달성

#### 핵심 결과 (MACD+DCA + 50% Reverse Close)

| 메트릭 | 값 | 비고 |
|--------|-----|------|
| **Total PnL** | **+918.9%** | 314일, 4x 레버리지 |
| **LONG PnL** | **+488.5%** | ✅ 양방향 수익 |
| **SHORT PnL** | **+430.4%** | ✅ 양방향 수익 |
| **Walk-Forward** | **6/6 (100%)** | 완벽한 일관성 |
| **Max Drawdown** | **1.5%** | 68.6% 감소! |
| **Win Rate** | 58.1% | 양호 |

#### 반대 신호 포지션 관리 연구

| 청산 비율 | Total PnL | Max DD | DD 감소 | WF |
|----------|-----------|--------|---------|-----|
| 0% (기존) | +628.6% | 4.7% | - | 6/6 |
| 25% | +801.1% | 1.8% | 61.7% | 6/6 |
| **50% ★** | **+918.9%** | **1.5%** | **68.6%** | 6/6 |
| 75% | +863.6% | 1.5% | 67.1% | 6/6 |
| 100% | +433.9% | 3.3% | 28.8% | 6/6 |

**핵심 발견**:
1. **50% 부분 청산이 최적**: PnL +46% 개선, DD 68.6% 감소
2. **DCA 필수**: DCA 없으면 LONG 손실 (-5.7%)
3. **100% 청산 피해야**: 과도한 청산은 -194% PnL 손실

#### 최종 권장 설정

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| MACD | 12/26/9 | 표준 설정 |
| ADX Threshold | 20 | 추세 강도 필터 |
| TP | 2.0% | 최적 익절선 |
| SL | 1.5% | 최적 손절선 |
| DCA Trigger | 0.8% | 물타기 시점 |
| Max DCA | 2 | 최대 2회 추가 |
| **Reverse Close** | **50%** | **핵심 리스크 관리** |

**설정 파일**: `config/macd_dca_bot_config.yaml`
**연구 문서**: `claudedocs/BALANCED_LONGSHORT_RESEARCH_20251223.md`
**분석 스크립트**:
- `scripts/analysis/macd_dca_deep_validation.py`
- `scripts/analysis/reverse_signal_position_management.py`

---

### 📚 전략 연구 통합 문서 (2025-12-21)

> **💡 Serena 메모리**: `strategy_research_compendium_20251221` - 핵심 내용 빠른 참조용
> - 세션 시작 시 `mcp__serena__read_memory("strategy_research_compendium_20251221.md")` 호출 가능

**전체 연구 결과는 아래 문서 참조:**

| 문서 | 내용 |
|------|------|
| **[STRATEGY_RESEARCH_COMPENDIUM.md](bingx_rl_trading_bot/claudedocs/STRATEGY_RESEARCH_COMPENDIUM.md)** | **통합 연구 문서** - Entry/Exit/Position Management 전체 정리 |
| [strategy_research_summary_20251221.csv](bingx_rl_trading_bot/results/strategy_research_summary_20251221.csv) | 모든 전략 백테스트 결과 CSV (48개 전략) |

**Compendium 포함 내용**:
- 21개 Entry 전략 비교 (RSI, Supertrend, EMA, VWAP 등)
- 8가지 Exit 전략 (Fixed SL, BE, Trail, 동적 SL)
- Position Sizing 패턴 (마틴게일 vs 역마틴게일)
- Walk-Forward 검증 결과 및 통계적 유의성
- 실패 전략 교훈 (Supertrend Trail 버그 등)

**개별 연구 문서**:
| 문서 | 내용 |
|------|------|
| [RSI_TREND_FILTER_BOT_ANALYSIS_20251216.md](bingx_rl_trading_bot/claudedocs/RSI_TREND_FILTER_BOT_ANALYSIS_20251216.md) | RSI 전략 심층 분석 |
| [ENTRY_SIGNAL_RESEARCH_COMPREHENSIVE_20251212.md](bingx_rl_trading_bot/claudedocs/ENTRY_SIGNAL_RESEARCH_COMPREHENSIVE_20251212.md) | 진입 신호 21개 비교 |
| [POSITION_SIZING_PATTERNS_20251212.md](bingx_rl_trading_bot/claudedocs/POSITION_SIZING_PATTERNS_20251212.md) | 순환매/마틴게일 연구 |
| [DYNAMIC_STOPLOSS_RESEARCH_20251213.md](bingx_rl_trading_bot/claudedocs/DYNAMIC_STOPLOSS_RESEARCH_20251213.md) | 동적 손절 연구 |

---

## 📦 Legacy Bots (Standby)

### ADX Supertrend Trail Bot v1.0 ❌ DEPRECATED
**파일**: `scripts/production/adx_supertrend_trail_bot.py`
**설정**: `config/adx_supertrend_trail_config.yaml`
**상태**: ❌ **폐기** - 백테스트 버그로 인한 허위 성과 발견

| 파라미터 | 값 |
|---------|-----|
| Entry | ADX > 20 + DI Crossover |
| TP | 2.0% |
| SL | Supertrend Trail (동적) |

**백테스트 버그**: Exit price를 Supertrend 값으로 사용 (불가능한 가격)
- 버그 결과: +1276.6% (허위)
- **수정 결과**: **-234.1%** (손실 전략)

### Supertrend + MTF Regime Bot v1.0 ⏸️ LEGACY
**파일**: `scripts/production/supertrend_regime_bot.py`
**설정**: `config/supertrend_regime_bot_config.yaml`
**상태**: ⏸️ **레거시**

| 파라미터 | 값 |
|---------|-----|
| Entry | Supertrend Direction Change |
| TP/SL | 3.5%/1.8% (고정) |
| Regime Filter | MTF 3단계 |

**성과**: Full Period +129.7%, 13 trades (거래 빈도 낮음)

### RSI Zone Entry Bot v2.2 ⏸️ LEGACY
**파일**: `scripts/production/rsi_zone_bot.py`
**설정**: `config/rsi_zone_bot_config.yaml`
**상태**: ⏸️ **레거시**

| 파라미터 | 값 |
|---------|-----|
| RSI Zone | 30/70 |
| TP/SL | 2.0%/1.5% |
| BE_SL | 1.2% |

**성과**: Full Period -6.2%, Test -13.5%

### Other Legacy Bots
- **EMA Crossover Bot v1.5**: `scripts/production/ema_crossover_bot.py`
- **VWAP Band Bot**: `scripts/production/vwap_band_bot.py`
- **Donchian Scalping Bot v20**: `scripts/production/donchian_scalping_bot.py`

---

## 📁 File Structure

```
CLAUDE_CODE_FIN/
├── CLAUDE.md (this file)
│
└── bingx_rl_trading_bot/
    ├── config/
    │   ├── ms_choch_bot_config.yaml          ← ✅ ACTIVE (v1.2) - ATR Position Sizing
    │   ├── macd_martingale_config.yaml       ← ❌ DEPRECATED
    │   ├── macd_dca_bot_config.yaml          ← ❌ DEPRECATED
    │   ├── supertrend_5m_config.yaml         ← ✅ ACTIVE (v1.0)
    │   ├── rsi_trend_filter_config.yaml      ← ✅ ACTIVE (v2.0)
    │   ├── adx_supertrend_trail_config.yaml  ← DEPRECATED
    │   ├── supertrend_regime_bot_config.yaml ← LEGACY
    │   └── rsi_zone_bot_config.yaml          ← LEGACY
    │
    ├── scripts/
    │   ├── production/
    │   │   ├── ms_choch_bot.py              ← ✅ ACTIVE (v1.2) - ATR Position Sizing
    │   │   ├── macd_martingale_bot.py       ← ❌ DEPRECATED
    │   │   ├── macd_dca_bot.py              ← ❌ DEPRECATED
    │   │   ├── supertrend_5m_bot.py         ← ✅ ACTIVE (v1.0)
    │   │   ├── rsi_trend_filter_bot.py      ← ✅ ACTIVE (v2.0)
    │   │   ├── adx_supertrend_trail_bot.py  ← DEPRECATED
    │   │   ├── supertrend_regime_bot.py     ← LEGACY
    │   │   ├── rsi_zone_bot.py              ← LEGACY
    │   │   ├── ema_crossover_bot.py         ← LEGACY
    │   │   ├── vwap_band_bot.py             ← LEGACY
    │   │   └── donchian_scalping_bot.py     ← LEGACY
    │   │
    │   ├── monitoring/
    │   │   ├── macd_dca_monitor.py              ← ❌ DEPRECATED
    │   │   ├── rsi_trend_filter_monitor.py      ← ✅ ACTIVE
    │   │   ├── adx_supertrend_trail_monitor.py  ← DEPRECATED
    │   │   ├── supertrend_regime_monitor.py     ← LEGACY
    │   │   └── rsi_zone_monitor.py              ← LEGACY
    │   │
    │   └── analysis/
    │       ├── comprehensive_v3_research.py         ← ✅ NEW (17 strategies)
    │       ├── alternative_strategies_research.py   ← 8 strategies comparison
    │       ├── rsi_strategy_deep_research.py        ← RSI parameter optimization
    │       ├── best_strategy_validation.py          ← Final validation
    │       ├── rsi_trend_filter_walkforward.py      ← Walk-forward testing
    │       ├── corrected_full_backtest.py           ← ADX bug fix verification
    │       └── ...
    │
    ├── results/
    │   ├── comprehensive_v3_*.csv               ← ✅ NEW (V3 Research)
    │   ├── ms_choch_bot_state.json              ← ✅ ACTIVE (v1.2) - ATR Position Sizing
    │   ├── macd_martingale_bot_state.json       ← ❌ DEPRECATED
    │   ├── macd_dca_bot_state.json              ← ❌ DEPRECATED
    │   ├── supertrend_5m_bot_state.json         ← ✅ ACTIVE
    │   ├── rsi_trend_filter_bot_state.json      ← ✅ ACTIVE
    │   ├── adx_supertrend_trail_bot_state.json  ← DEPRECATED
    │   ├── supertrend_regime_bot_state.json     ← LEGACY
    │   ├── rsi_zone_bot_state.json              ← LEGACY
    │   └── backups/
    │
    ├── claudedocs/
    │   ├── RSI_TREND_FILTER_RESEARCH_20251216.md
    │   ├── DYNAMIC_STOPLOSS_RESEARCH_20251213.md
    │   └── ...
    │
    ├── logs/
    │   ├── macd_martingale_bot_YYYYMMDD.log       ← ✅ NEW (v1.0)
    │   ├── macd_dca_bot_YYYYMMDD.log             ← ❌ DEPRECATED
    │   ├── supertrend_5m_bot_YYYYMMDD.log        ← ✅ ACTIVE
    │   └── rsi_trend_filter_bot_YYYYMMDD.log     ← ✅ ACTIVE
    │
    ├── START_MACD_MARTINGALE.bat            ← ✅ NEW (v1.0, background)
    ├── STOP_MACD_MARTINGALE.bat             ← ✅ NEW (v1.0)
    ├── MONITOR_MACD_MARTINGALE.bat          ← ✅ NEW (v1.0)
    ├── START_MACD_DCA.bat                   ← ❌ DEPRECATED
    ├── START_MACD_DCA_FOREGROUND.bat        ← ❌ DEPRECATED
    ├── STOP_MACD_DCA.bat                    ← ❌ DEPRECATED
    ├── MONITOR_MACD_DCA.bat                 ← ❌ DEPRECATED
    ├── START_SUPERTREND_5M.bat               ← ✅ ACTIVE (background)
    ├── STOP_SUPERTREND_5M.bat                ← ✅ ACTIVE
    ├── MONITOR_SUPERTREND_5M.bat             ← ✅ ACTIVE
    ├── START_RSI_TREND_FILTER.bat            ← ✅ ACTIVE (foreground)
    ├── START_RSI_TREND_FILTER_BACKGROUND.bat ← ✅ ACTIVE (VS Code 독립)
    ├── STOP_RSI_TREND_FILTER.bat             ← ✅ ACTIVE (봇 정지)
    ├── MONITOR_RSI_TREND_FILTER.bat          ← ✅ ACTIVE
    ├── START_ADX_SUPERTREND_TRAIL.bat        ← DEPRECATED
    └── MONITOR_ADX_SUPERTREND_TRAIL.bat      ← DEPRECATED
```

---

## ⚠️ Backtest Methodology (CRITICAL)

> **🔴 CRITICAL**: 이 섹션은 MACD+DCA 전략에서 발견된 백테스트 방법론 결함에서 얻은 교훈입니다.
> 모든 백테스트는 아래 규칙을 반드시 준수해야 합니다.

### 필수 요구사항

| 요구사항 | 설명 | 잘못된 예 | 올바른 예 |
|----------|------|----------|----------|
| **잔고 기반 포지션 사이징** | 포지션 크기는 현재 잔고를 기준으로 계산 | `size = 1.0` (고정) | `size = balance * 0.33 * leverage` |
| **복리 효과 반영** | 수익/손실이 다음 거래에 반영되어야 함 | PnL이 잔고에 미반영 | `balance += realized_pnl` |
| **수수료 포함** | Entry/Exit 양방향 수수료 반영 | 수수료 무시 | `fee = position * 0.05% * 2` |
| **미실현 손익 기반 Drawdown** | 포지션 보유 중 Drawdown 계산 | Exit 시점만 계산 | 매 캔들 unrealized PnL 포함 |

### 잘못된 백테스트 (고정 단위)

```python
# ❌ WRONG: 고정 단위 기반 - 복리 효과 없음
total_size = 1.0  # 항상 1.0 고정
pnl_pct = (exit_price - entry_price) / entry_price * 100
final_pnl = pnl_pct * total_size * leverage / 100  # 2% = $0.08
balance += final_pnl  # 잔고 변화가 다음 거래에 미반영
```

### 올바른 백테스트 (잔고 기반)

```python
# ✅ CORRECT: 잔고 기반 - 복리 효과 반영
POSITION_PCT = 0.33  # 잔고의 33%
LEVERAGE = 4
FEE_PCT = 0.05  # 0.05% per side

# 포지션 크기: 현재 잔고 기준
position_value = balance * POSITION_PCT * LEVERAGE  # 예: $100 * 0.33 * 4 = $132

# 수수료 차감
entry_fee = position_value * (FEE_PCT / 100)
balance -= entry_fee

# 청산 시 PnL 계산
pnl_pct = (exit_price - entry_price) / entry_price * 100
realized_pnl = position_value * (pnl_pct / 100)  # 2% = $2.64

# 수수료 차감 및 잔고 반영
exit_fee = position_value * (FEE_PCT / 100)
balance += realized_pnl - exit_fee  # 다음 거래에 복리 적용
```

### MACD+DCA 케이스 스터디

| 지표 | 잘못된 백테스트 (고정 단위) | 올바른 백테스트 (잔고 기반) | 차이 |
|------|--------------------------|--------------------------|------|
| **Total PnL** | +918.9% | **-46.5%** | ⚠️ **-965%p 차이!** |
| **Walk-Forward** | 6/6 (100%) | **2/6 (33%)** | 과적합 숨김 |
| **Max Drawdown** | 1.5% | **60.5%** | 리스크 과소평가 |
| **결론** | "훌륭한 전략" | **"손실 전략"** | 완전히 반대 결론 |

### 관련 스크립트

| 파일 | 용도 | 방법론 |
|------|------|--------|
| `scripts/analysis/macd_dca_backtest_with_fees.py` | 수수료 포함 | ❌ 고정 단위 |
| `scripts/analysis/macd_dca_backtest_correct_dd.py` | 미실현 PnL DD | ❌ 고정 단위 |
| `scripts/analysis/macd_dca_backtest_realistic.py` | **정확한 백테스트** | ✅ 잔고 기반 |

### 체크리스트 (새 전략 검증 시)

1. [ ] 포지션 크기가 `balance * position_pct * leverage`로 계산되는가?
2. [ ] 수익/손실이 다음 거래의 `balance`에 반영되는가?
3. [ ] Entry/Exit 수수료(0.05% × 2)가 포함되었는가?
4. [ ] Drawdown이 매 캔들마다 미실현 손익 포함하여 계산되는가?
5. [ ] Walk-Forward 검증에서 50% 이상 윈도우가 수익인가?

---

## 🔧 Known Issues & Fixes

| 날짜 | 이슈 | 해결 |
|------|------|------|
| 2025-12-25 | **🔴 RSI Martingale 연구-프로덕션 불일치** | 연구 +1.37% daily vs 프로덕션 -3.33% daily → **RSI 계산 방식 차이** (연구: SMA, 프로덕션: EWM), 봇 배포 금지 (상세: RSI_MARTINGALE_DISCREPANCY_ANALYSIS) |
| 2025-12-25 | **🔴 PM Research Look-Ahead Bias 발견** | Time Exit 백테스트 +3335% → **실제 -35%** (손실), Entry 봉에서 Exit 체크 (동일 봉 High/Low 사용 = 미래 데이터), v3.2 Time Exit **비활성화** |
| 2025-12-24 | **🔴 MS_ChoCH Look-Ahead Bias 발견** | 연구 +609.1% vs 실제 -6.50% 불일치 → **봇 폐기**, 전면 감사 완료 (상세: LOOK_AHEAD_BIAS_AUDIT) |
| 2025-12-24 | **MACD Martingale Bot v1.0 배포** | ✅ 수정된 백테스트 로직 적용: Entry@NextOpen, TP/SL@High/Low, 10x Cap, 0.65% daily |
| 2025-12-24 | **백테스트 3가지 결함 발견 및 수정** | Entry 타이밍, Exit 가격 감지, 포지션 캡 문제 수정 (상세: BACKTEST_VERIFICATION_REPORT) |
| 2025-12-23 | **MACD+DCA 백테스트 방법론 결함 발견** | ❌ 고정 단위 백테스트 +918.9% → 잔고 기반 **-46.5%** (손실 전략!), 봇 폐기 |
| 2025-12-22 | **Comprehensive V3 Research** | 17개 Entry/Exit 전략 비교, Partial Exit +20.16%, RSI Divergence 5/6 WF 일관성 |
| 2025-12-22 | **RSI Trend Filter v2.0 배포** | 🚀 PreBE Spike Protection: LONG -1.62%→+2.02% (수익 전환!), WR 69.6%, WF PnL +10.4% 개선 |
| 2025-12-22 | RSI Trend Filter v1.10 | Cooldown 제거 (1→0), Risk limits 비활성화, 백테스트 정합성 확보 |
| 2025-12-21 | **RSI Trend Filter v1.9 배포** | ⚡ 스파이크 보호: Trail 0.5% → 0.15%, LONG 손실 36% 감소, WF PnL +13.3% 개선 |
| 2025-12-21 | **RSI Trend Filter v1.8 배포** | 5m 타임프레임, BB 필터, v1.7 백테스트 일치 |
| 2025-12-20 | **SuperTrend 5m Bot v1.0 배포** | 5분 타임프레임 고빈도 전략, 1.36 T/day, WR 69.5%, +42.8% (90일) |
| 2025-12-20 | **RSI Trend Filter v1.7 배포** | BE+Trail 포지션 관리, RSI 35/65, TP4%/SL1.5%, BE@1%, Trail@0.5% |
| 2025-12-20 | **v1.7.8 락 메커니즘 개선** | Windows PID 검증, 원자적 락 생성, race condition 방지 |
| 2025-12-20 | **v1.6 프로세스 독립성** | Background 실행 스크립트 추가, VS Code 종료 시 봇 유지 |
| 2025-12-19 | **v1.6 TP 최적화** | TP 2.5% → 3.0%, R:R 1.5:1, +33%p PnL 개선 |
| 2025-12-16 | **RSI Trend Filter v1.0 배포** | 통계적 유의성 검증 완료 (p=0.013) |
| 2025-12-16 | **ADX Supertrend 버그 발견** | Exit price 버그로 +1276% 허위 → 실제 -234% |
| 2025-12-16 | **대안 전략 연구** | 8개 전략 비교, RSI Trend Filter 최적 |
| 2025-12-13 | ADX Supertrend Trail v1.0 배포 | 동적 SL 연구 (버그 있었음) |
| 2025-12-12 | Entry Signal Research | 21 methods, 1080 combinations 테스트 |

---

## 🛡️ Look-Ahead Bias 예방 가이드

### 금지 패턴 (백테스트/신호 생성에서)
```python
# ❌ 절대 금지
df['column'].shift(-1)          # 미래 1봉 참조
df['column'].shift(-n)          # 미래 n봉 참조 (n > 0)
df.rolling(n, center=True)      # 양방향 롤링 (미래 데이터 포함)

# ❌ 동일 봉 Exit 체크 (2025-12-25 발견)
# Entry가 bar[i].open에서 발생하면, 해당 봉의 high/low는 Entry 이후 데이터!
for i in range(len(df)):
    if position and i == entry_idx:   # ❌ 동일 봉에서 Exit 체크 = Look-Ahead
        if row['high'] >= tp_price:   # bar[i].high는 open 이후 발생!
            exit()
```

### 🔴 동일 봉 Exit 체크 문제 상세 (PM Research Case)
```
시나리오:
- Bar N: Signal 발생 → Entry at Bar N Open (100,000)
- 동일 Bar N: High = 102,000, Low = 99,000

잘못된 로직 (Look-Ahead):
- Bar N에서 Entry하면서 동시에 Bar N.high를 체크
- Bar N.high (102,000) ≥ TP (101,500) → 즉시 Exit
- 결과: bars_held = 0, Win Rate 77%, PnL +3335% (허위)

올바른 로직:
- Bar N에서 Entry → Exit 체크는 Bar N+1부터
- Bar N+1.high/low로 TP/SL 체크
- 결과: bars_held ≥ 1, Win Rate 28%, PnL -35% (실제)

핵심: Entry 봉의 High/Low는 Entry 이후에 결정되므로 사용 불가!
```

### 허용 패턴
```python
# ✅ 안전 - 과거만 참조
df['column'].shift(1)           # 과거 1봉
df.rolling(n).xxx()             # 기본값 = 과거만

# ✅ ML 라벨용 (신호 생성에 사용 금지)
labels['future_return'] = ...   # 라벨 전용, 명확히 분리

# ✅ 올바른 Exit 체크 (2025-12-25 추가)
for i in range(len(df)):
    if position:
        entry_idx = position['idx']
        if i <= entry_idx:        # ✅ Entry 봉은 건너뛰기
            continue              # Exit 체크는 entry_idx + 1부터
        # 이제 안전하게 row['high'], row['low'] 사용 가능
        if row['high'] >= tp_price:
            exit()
```

### 검증 체크리스트
1. [ ] `grep -rn "shift(-" scripts/analysis/` → 0건 확인
2. [ ] `grep -rn "center=True" scripts/analysis/` → 0건 확인
3. [ ] Entry 시점이 시그널 발생 **후** (다음 봉 Open) 확인
4. [ ] Walk-Forward가 **새로운 기간**에서 Out-of-Sample 테스트 확인
5. [ ] 프로덕션 봇 로직과 백테스트 로직 **동일** 확인
6. [ ] **🔴 동일 봉 Exit 체크 금지**: Entry 봉(i)에서 Exit 체크 시 `if i <= entry_idx: continue` 확인
7. [ ] **bars_held 검증**: 백테스트 결과에서 `bars_held: 0` 거래가 없어야 함

### 전면 감사 결과 (2025-12-25 업데이트)
- ✅ **RSI Trend Filter (Structure Exit)**: Look-Ahead 없음 → 연구 유효
- ✅ **SuperTrend 5m**: Look-Ahead 없음 → 연구 유효
- ❌ **MS_ChoCH**: Look-Ahead 발견 (shift(-1), center=True) → 연구 무효, 폐기
- ❌ **PM Research (Time Exit)**: Look-Ahead 발견 (동일 봉 Exit 체크) → **+3335% → -35%** (손실), v3.2 비활성화
- 상세: `claudedocs/LOOK_AHEAD_BIAS_AUDIT_20251224.md`
- 수정 스크립트: `scripts/analysis/pm_research_correct.py`

---

## 🔬 연구-프로덕션 일치 검증 가이드 (2025-12-25)

> **🔴 CRITICAL**: RSI Martingale 불일치 사례에서 발견된 교훈
> - 연구: +1.37% daily, 프로덕션 백테스트: **-3.33% daily** (완전히 다른 결과!)
> - 원인: RSI 계산 방식 차이 (연구: SMA, 프로덕션: EWM)
> - 상세: `claudedocs/RSI_MARTINGALE_DISCREPANCY_ANALYSIS_20251225.md`

### 인디케이터 계산 표준화

**RSI 계산 (CRITICAL)**:
```python
# ✅ 표준 RSI (Wilder's Smoothing = EWM) - 프로덕션 및 연구에서 동일하게 사용
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

# ❌ SMA RSI - 절대 사용 금지 (결과가 다름!)
def calculate_rsi_sma_WRONG(df, period=14):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()  # ❌ SMA
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()  # ❌ SMA
    ...
```

**EMA/MACD 계산**:
```python
# ✅ 표준 EMA
ema = df['close'].ewm(span=period, adjust=False).mean()

# ✅ 표준 MACD
ema_fast = df['close'].ewm(span=12, adjust=False).mean()
ema_slow = df['close'].ewm(span=26, adjust=False).mean()
macd_line = ema_fast - ema_slow
signal_line = macd_line.ewm(span=9, adjust=False).mean()
histogram = macd_line - signal_line
```

### 연구-프로덕션 검증 체크리스트

새 전략 연구 시 **반드시** 확인:

| 항목 | 검증 방법 | 확인란 |
|------|----------|--------|
| **인디케이터 계산** | 연구 스크립트 vs 프로덕션 봇 코드 diff | [ ] |
| **RSI 방식** | `rolling().mean()` 금지 → `ewm().mean()` 사용 | [ ] |
| **신호 로직** | 동일 데이터에서 동일 신호 발생 확인 | [ ] |
| **Entry 조건** | `>` vs `>=` 등 미묘한 차이 확인 | [ ] |
| **파라미터** | 연구 결과 파라미터 = 프로덕션 설정 확인 | [ ] |
| **레버리지/포지션** | 연구와 프로덕션 동일한 값 사용 | [ ] |
| **수수료/슬리피지** | 연구에 포함 여부 확인 (0.05% × 2) | [ ] |

### 프로덕션 백테스트 필수 프로세스

```
1. 연구 완료 (연구 스크립트로 유망한 결과 발견)
   ↓
2. 프로덕션 로직으로 백테스트 재실행 (필수!)
   - 프로덕션 봇과 동일한 인디케이터 계산 사용
   - 동일한 Entry/Exit 로직 사용
   ↓
3. 결과 비교 (10% 이상 차이 시 원인 분석 필수)
   - 연구: +100%, 프로덕션: +90% → ✅ OK (10% 차이)
   - 연구: +100%, 프로덕션: -50% → ❌ 원인 분석 필요!
   ↓
4. 원인 파악 및 수정
   - 인디케이터 계산 방식 차이?
   - 신호 조건 차이?
   - 파라미터 불일치?
   ↓
5. 최종 검증 후 배포
```

### 전면 감사 결과 (2025-12-25 업데이트)

| 전략 | 상태 | 이슈 | 조치 |
|------|------|------|------|
| ✅ RSI Trend Filter | 유효 | 없음 | 운영 중 |
| ✅ SuperTrend 5m | 유효 | 없음 | 운영 중 |
| ❌ RSI Martingale | 무효 | **RSI 계산 차이 (SMA vs EWM)** | 배포 금지 |
| ❌ MS_ChoCH | 무효 | Look-Ahead Bias | 폐기 |
| ❌ PM Research Time Exit | 무효 | Look-Ahead Bias | v3.2 비활성화 |
| ❌ MACD+DCA | 무효 | 백테스트 방법론 결함 | 폐기 |

---

## 🧠 AI Assistant Instructions

### RSI Trend Filter Bot 핵심 사항 (v2.0)
1. **Entry (LONG)**: Close > EMA(100) AND RSI(14) < **35** AND BB% < **0.2**
2. **Entry (SHORT)**: Close < EMA(100) AND RSI(14) > **65** AND BB% > **0.8**
3. **TP**: **4.0%** 고정
4. **SL**: **1.5%** 고정
5. **BE+Trail**: BE@1%, Trail@0.5%, Tight@0.15% (스파이크 시)
6. **⚡ PreBE Spike**: BE 활성화 전 스파이크 → 조기 청산 (v2.0)
7. **Cooldown**: 0 (없음)

### 신호 로직 설명 (v2.0)
```
Buy Low Strategy (RSI + BB + EMA Trend Filter):
- RSI < 35 = 과매도 상태 진입 (Buy Low)
- RSI > 65 = 과매수 상태 진입 (Sell High)
- BB% < 0.2 = 볼린저 밴드 하단 근처 (추가 확인)
- BB% > 0.8 = 볼린저 밴드 상단 근처 (추가 확인)
- EMA100 = 추세 방향 필터 (추세 역행 거래 방지)

LONG 조건:
- 가격 > EMA100 (상승 추세)
- RSI < 35 (과매도)
- BB% < 0.2 (하단 근처)

SHORT 조건:
- 가격 < EMA100 (하락 추세)
- RSI > 65 (과매수)
- BB% > 0.8 (상단 근처)

Exit 로직 (BE+Trail + Spike Protection):
- 1% 수익 도달 → SL을 Entry로 이동 (BE)
- BE 활성화 후 → 0.5% 뒤에서 Trail
- 역추세 스파이크 감지 → 0.15% Tight Trail
```

### 통계적 검증 결과
```
Walk-Forward Validation:
- 7개 윈도우 중 6개 수익 (86%)
- P-value: 0.013 (< 0.05 = 통계적 유의)
- Monte Carlo: 100% 수익 확률
- Sharpe: 1.31 (양호)

결론: 과적합이 아닌 실제 유효한 전략으로 검증됨
```

### Code Modification Rules
1. **Order Creation**: Hedge Mode에서 reduce_only 사용 불가
2. **Position Sizing**: EFFECTIVE_LEVERAGE (4x) 기준 계산
3. **State Management**: state.json 백업 후 변경
4. **CCXT 제한**: conditional orders는 Raw API 사용

### 관련 문서
| 문서 | 내용 |
|------|------|
| `config/rsi_trend_filter_config.yaml` | 봇 설정 |
| `scripts/analysis/best_strategy_validation.py` | 최종 검증 스크립트 |

---

## 🔌 MCP Integration (자동 활용 규칙)

**Priority**: 🔴 **CRITICAL** - 세션 초기화 및 코드 작업 시 필수

### Session Start Protocol
**매 세션 시작 시 자동 실행** (반드시 순서대로):
```
1. mcp__serena__activate_project("bingx_rl_trading_bot")
2. mcp__serena__check_onboarding_performed()
3. mcp__serena__list_memories() → 관련 메모리 확인
```
✅ **Right**: Serena 먼저 활성화 → Context7로 API 문서 → Sequential로 분석
❌ **Wrong**: MCP 없이 네이티브 도구만 사용, Serena 활성화 안 함

### Task → MCP 자동 매핑

| 작업 유형 | Primary MCP | Secondary MCP | 사용 시점 |
|----------|-------------|---------------|----------|
| **코드 분석** | Serena | Sequential | symbol find, rename, references |
| **함수 찾기** | Serena | - | 어디서 호출? 어디에 정의? |
| **CCXT/Pandas 문서** | Context7 | - | library API, framework patterns |
| **복잡한 디버깅** | Sequential | Serena | 왜 안됨? 원인 분석 |
| **BingX API 정보** | Tavily | Context7 | 최신 정보, 외부 검색 |
| **백테스트 분석** | Sequential | Serena | 결과 해석, 설계 |
| **다중 파일 수정** | Morphllm | Serena | 패턴 교체, 리팩토링 |
| **UI 컴포넌트** | Magic | - | React/Vue, /ui 명령 |
| **브라우저 테스트** | Playwright | - | E2E, 스크린샷, 자동화 |

### 트레이딩 봇 전용 MCP 규칙

**Serena 사용 시점**:
- `generate_signal`, `execute_trade` 등 함수 분석
- 인디케이터 계산 로직 이해
- state.json 관리 관련 코드 추적
- 세션 간 작업 연속성 유지 (메모리)

**Context7 사용 시점**:
- CCXT 라이브러리 API (fetch_ohlcv, create_order 등)
- Pandas DataFrame 조작
- TA-Lib 인디케이터 사용법

**Sequential 사용 시점**:
- "왜 진입 안됨?" → 체계적 원인 분석
- 백테스트 결과 해석
- Walk-forward 검증 분석
- 전략 설계 및 개선

**Tavily 사용 시점**:
- BingX API 최신 업데이트 확인
- 거래소 정책 변경 검색
- 시장 상황 리서치

### 키워드 기반 자동 활성화

| 키워드 (한국어) | 키워드 (영어) | 활성화 MCP |
|----------------|--------------|-----------|
| "찾아줘", "어디", "참조" | "find", "where", "references", "rename" | Serena |
| "문서", "사용법", "API" | "how to", "docs", library names | Context7 |
| "왜", "원인", "분석" | "why", "debug", "analyze" | Sequential |
| "최신", "검색", "업데이트" | "latest", "search", "news" | Tavily |
| "테스트", "E2E" | "browser", "screenshot" | Playwright |
| "UI", "/ui" | "component", "21st" | Magic |
| "백테스트", "검증" | "backtest", "validate" | Sequential + Serena |

### 복합 패턴

**디버깅 패턴**:
```
Serena (코드 위치) → Sequential (원인 분석) → Serena (수정)
```

**리서치 패턴**:
```
Tavily (최신 정보) → Context7 (공식 문서) → Sequential (종합)
```

**구현 패턴**:
```
Serena (기존 코드 이해) → Context7 (베스트 프랙티스) → Morphllm (적용)
```

### 메모리 관리

**Memory Naming Convention**:
```
{task_type}_{date}_{description}
예: debug_20251220_signal_fix
    research_20251220_strategy_compare
    config_20251220_bot_params
```

**저장된 메모리**:
- `rsi_trend_filter_v19_20251221` - **현재 봇 v1.10 설정 (최신)** ⭐
- `strategy_research_compendium_20251221` - **전략 연구 통합 문서** ⭐
- `rsi_trend_filter_v178_20251221` - v1.7.8 설정
- `workspace_setup_20251220` - MCP/워크스페이스 설정
- `rsi_trend_filter_v16_20251220` - v1.6 설정 (레거시)
- `session_20251202_bugfixes` - 버그 수정 세션 (레거시)
- `exit_threshold_optimization_20251024` - Exit 최적화 연구 (레거시)
- `investigation_results_20251028` - 조사 결과 (레거시)

**새 메모리 저장 시점**:
- 중요한 디버깅 발견
- 전략 연구 결과
- API 이슈 해결책
- 세션 간 연속성 필요한 작업

**세션 시작 시 메모리 확인**:
- 현재 작업 도메인과 관련된 메모리 먼저 읽기
- 오래된 레거시 메모리는 건너뛰기 가능

---

**Last Updated**: 2025-12-23 KST (MACD+DCA 폐기 + Backtest Methodology 추가)
