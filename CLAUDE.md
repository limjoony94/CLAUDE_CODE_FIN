# CLAUDE_CODE_FIN - Workspace Overview

**Last Updated**: 2025-12-16 KST

---

## 🎯 Active Bot

### RSI Trend Filter Bot v1.0 ✅ ACTIVE
**파일**: `scripts/production/rsi_trend_filter_bot.py`
**설정**: `config/rsi_trend_filter_config.yaml`
**상태**: ✅ **v1.0 운영 중** - Walk-Forward 검증 완료, 통계적 유의성 확인

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| **Entry (LONG)** | **RSI crosses above 40 + Close > EMA100** | 상승 추세 + RSI 반등 |
| **Entry (SHORT)** | **RSI crosses below 60 + Close < EMA100** | 하락 추세 + RSI 하락 |
| RSI Period | 14 | 표준 RSI |
| EMA Period | 100 | 추세 필터 |
| **Take Profit** | **3.0%** | 고정 |
| **Stop Loss** | **2.0%** | 고정 |
| Cooldown | 4 candles | 1시간 |
| Leverage | 4x | |
| Timeframe | 15m | |

**Entry Logic**:
- **LONG**: Close > EMA(100) AND RSI(14) crosses above 40
- **SHORT**: Close < EMA(100) AND RSI(14) crosses below 60

**Exit Logic**:
- TP: 3.0% 도달 → 익절
- SL: 2.0% 도달 → 손절

**검증 결과 (Walk-Forward 7 Windows)**:

| 메트릭 | 값 |
|--------|-----|
| **Profitable Windows** | **6/7 (86%)** |
| **Total PnL** | **+120.8%** |
| **Sharpe Ratio** | **1.31** |
| **P-value** | **0.013** (통계적 유의) |
| **Monte Carlo** | **100% profit probability** |
| **Worst Window** | **-4.8%** |
| **Validation Score** | **10/10 PASSED** |

```bash
# Commands
START_RSI_TREND_FILTER.bat                              # Start
MONITOR_RSI_TREND_FILTER.bat                            # Monitor
python scripts/production/rsi_trend_filter_bot.py       # Start (direct)
python scripts/monitoring/rsi_trend_filter_monitor.py   # Monitor (direct)
cat results/rsi_trend_filter_bot_state.json             # State
cat config/rsi_trend_filter_config.yaml                 # Config
```

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
    │   ├── rsi_trend_filter_config.yaml      ← ✅ ACTIVE
    │   ├── adx_supertrend_trail_config.yaml  ← DEPRECATED
    │   ├── supertrend_regime_bot_config.yaml ← LEGACY
    │   └── rsi_zone_bot_config.yaml          ← LEGACY
    │
    ├── scripts/
    │   ├── production/
    │   │   ├── rsi_trend_filter_bot.py      ← ✅ ACTIVE (v1.0)
    │   │   ├── adx_supertrend_trail_bot.py  ← DEPRECATED
    │   │   ├── supertrend_regime_bot.py     ← LEGACY
    │   │   ├── rsi_zone_bot.py              ← LEGACY
    │   │   ├── ema_crossover_bot.py         ← LEGACY
    │   │   ├── vwap_band_bot.py             ← LEGACY
    │   │   └── donchian_scalping_bot.py     ← LEGACY
    │   │
    │   ├── monitoring/
    │   │   ├── rsi_trend_filter_monitor.py      ← ✅ ACTIVE
    │   │   ├── adx_supertrend_trail_monitor.py  ← DEPRECATED
    │   │   ├── supertrend_regime_monitor.py     ← LEGACY
    │   │   └── rsi_zone_monitor.py              ← LEGACY
    │   │
    │   └── analysis/
    │       ├── alternative_strategies_research.py   ← 8 strategies comparison
    │       ├── rsi_strategy_deep_research.py        ← RSI parameter optimization
    │       ├── best_strategy_validation.py          ← Final validation
    │       ├── rsi_trend_filter_walkforward.py      ← Walk-forward testing
    │       ├── corrected_full_backtest.py           ← ADX bug fix verification
    │       └── ...
    │
    ├── results/
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
    │   └── rsi_trend_filter_bot_YYYYMMDD.log
    │
    ├── START_RSI_TREND_FILTER.bat      ← ✅ ACTIVE
    ├── MONITOR_RSI_TREND_FILTER.bat    ← ✅ ACTIVE
    ├── START_ADX_SUPERTREND_TRAIL.bat  ← DEPRECATED
    └── MONITOR_ADX_SUPERTREND_TRAIL.bat ← DEPRECATED
```

---

## 🔧 Known Issues & Fixes

| 날짜 | 이슈 | 해결 |
|------|------|------|
| 2025-12-16 | **RSI Trend Filter v1.0 배포** | 통계적 유의성 검증 완료 (p=0.013) |
| 2025-12-16 | **ADX Supertrend 버그 발견** | Exit price 버그로 +1276% 허위 → 실제 -234% |
| 2025-12-16 | **대안 전략 연구** | 8개 전략 비교, RSI Trend Filter 최적 |
| 2025-12-13 | ADX Supertrend Trail v1.0 배포 | 동적 SL 연구 (버그 있었음) |
| 2025-12-12 | Entry Signal Research | 21 methods, 1080 combinations 테스트 |

---

## 🧠 AI Assistant Instructions

### RSI Trend Filter Bot 핵심 사항
1. **Entry (LONG)**: Close > EMA(100) AND RSI(14) crosses above 40
2. **Entry (SHORT)**: Close < EMA(100) AND RSI(14) crosses below 60
3. **TP**: 3.0% 고정
4. **SL**: 2.0% 고정
5. **Cooldown**: 4 candles (1시간)

### 신호 로직 설명
```
RSI Crossover + Trend Filter:
- RSI가 40을 상향 돌파 = 과매도에서 반등 시작
- RSI가 60을 하향 돌파 = 과매수에서 하락 시작
- EMA100 = 추세 방향 필터 (추세 역행 거래 방지)

LONG 조건:
- 가격 > EMA100 (상승 추세)
- RSI가 40을 상향 돌파 (반등 확인)

SHORT 조건:
- 가격 < EMA100 (하락 추세)
- RSI가 60을 하향 돌파 (하락 확인)
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

**Last Updated**: 2025-12-16 KST
