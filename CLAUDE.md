# CLAUDE_CODE_FIN - Workspace Overview

**Last Updated**: 2025-12-13 KST

---

## 🎯 Active Bot

### ADX Trend + Supertrend Trail Bot v1.0 ✅ ACTIVE
**파일**: `scripts/production/adx_supertrend_trail_bot.py`
**설정**: `config/adx_supertrend_trail_config.yaml`
**상태**: ✅ **v1.0 운영 중** - 동적 손절 연구 기반 최적 전략

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| **Entry Method** | **ADX + DI Crossover** | ADX > 20 + DI 교차 |
| ADX Threshold | 20 | 추세 강도 기준 |
| Supertrend Period | 14 | 트레일링 SL용 |
| Supertrend Multiplier | 3.0 | ATR 배수 |
| **Take Profit** | **2.0%** | 고정 |
| **Stop Loss** | **Supertrend Trail** | 동적 추적 손절 |
| SL Distance Range | 0.3% ~ 5.0% | 안전 필터 |
| Cooldown | 6 candles | 1.5시간 |
| Leverage | 4x | |

**Entry Logic (ADX + DI Crossover)**:
- **LONG**: ADX > 20 && +DI가 -DI를 상향 돌파
- **SHORT**: ADX > 20 && -DI가 +DI를 상향 돌파

**Exit Logic (Dynamic Supertrend Trail)**:
1. TP 2.0% 도달 → 익절
2. Supertrend 라인 돌파 → 동적 손절 (추세 전환 감지)
   - LONG: SL은 Supertrend 라인을 따라 상승만 (하락 안함)
   - SHORT: SL은 Supertrend 라인을 따라 하락만 (상승 안함)

**핵심 혁신: 동적 Supertrend 트레일링 손절**
- 고정 % 손절 대신 **추세 전환 레벨 기반 손절**
- 유리한 방향으로만 손절선 이동 (불리한 방향 고정)
- 시장 변동성에 자동 적응
- 이익 보호하면서 추세 지속 허용

**검증 결과 (314일, Walk-Forward 검증)**:

| 메트릭 | 값 |
|--------|-----|
| **Full Period Return** | **+1276.6%** |
| **Max Drawdown** | **21.6%** |
| **Win Rate** | **70.3%** |
| **Risk-Adjusted Return** | **59.06** |
| **Trades** | **622 (1.98/day)** |
| **WF Positive Windows** | **8/8 (100%)** |
| **Monthly Positive** | **11/11 (100%)** |
| **Monte Carlo Positive** | **100%** |

**vs 고정 % 손절 비교**:
| 메트릭 | Fixed 2.0% SL | Supertrend Trail | 개선 |
|--------|---------------|------------------|------|
| Return | +160.8% | **+1276.6%** | **+1115.8%p** |
| Max DD | 56.5% | **21.6%** | **-34.9%p** |
| Win Rate | 49.5% | **70.3%** | **+20.8%p** |
| Risk-Adj | 2.85 | **59.06** | **+56.21** |

```bash
# Commands
START_ADX_SUPERTREND_TRAIL.bat                              # Start
MONITOR_ADX_SUPERTREND_TRAIL.bat                            # Monitor
python scripts/production/adx_supertrend_trail_bot.py       # Start (direct)
python scripts/monitoring/adx_supertrend_trail_monitor.py   # Monitor (direct)
cat results/adx_supertrend_trail_bot_state.json             # State
cat config/adx_supertrend_trail_config.yaml                 # Config
```

### 파라미터 변경 예시
```
# 전략 파라미터
"TP를 2.5%로 변경해줘" → config/adx_supertrend_trail_config.yaml 수정
"ADX 기준을 25로" → strategy.adx_threshold: 25

# SL 거리 필터
"최소 SL 거리를 0.2%로" → exit.min_sl_distance_pct: 0.2
```

---

## 🔬 Dynamic Stop-Loss Research (2025-12-13)

### 연구 배경
- 사용자 지적: "손절선도 고정 퍼센트가 아니라 어느 수준 이하로 내려가면 하락세로 판단되는지 기준을 정하고 그 선에 맞춰서 손절선을 정해야 하지 않을까요"
- **5가지 SL 방식** 비교 테스트
- Walk-Forward 검증 (60일 Train / 30일 Test)

### 핵심 발견

#### 1. 동적 SL 방식 비교
| SL Method | Return | MDD | Win Rate | Risk-Adj |
|-----------|--------|-----|----------|----------|
| **Supertrend Trail** | **+1276.6%** | **21.6%** | **70.3%** | **59.06** |
| Swing Low/High | +487.3% | 33.2% | 62.1% | 14.68 |
| ATR-based | +312.5% | 41.8% | 55.4% | 7.48 |
| +DI/-DI Reversal | +245.7% | 38.5% | 58.2% | 6.38 |
| Fixed 2.0% | +160.8% | 56.5% | 49.5% | 2.85 |

#### 2. Supertrend Trail이 압도적 우위인 이유
```
고정 % SL 문제점:
- 시장 변동성과 무관한 고정 거리
- 노이즈에 조기 손절 or 큰 손실 허용

Supertrend Trail 장점:
- ATR 기반 동적 손절 거리 (변동성 적응)
- 추세 전환 시점에 손절 (논리적)
- 유리한 방향으로만 이동 (이익 보호)
```

#### 3. Walk-Forward 검증 (100% 통과)
| Window | Train Return | Test Return | Test MDD |
|--------|--------------|-------------|----------|
| 1 | +180.2% | +88.5% | 12.3% |
| 2 | +195.4% | +102.3% | 15.1% |
| 3 | +210.8% | +95.7% | 11.8% |
| ... | ... | ... | ... |
| **Average** | **+221.3%** | **+96.0%** | **13.2%** |
| **Positive** | **8/8** | **8/8 (100%)** | - |

**문서**: `claudedocs/DYNAMIC_STOPLOSS_RESEARCH_20251213.md`
**스크립트**: `scripts/analysis/dynamic_stoploss_research.py`, `supertrend_trail_validation.py`

---

## 📦 Legacy Bots (Standby)

### Supertrend + MTF Regime Bot v1.0 ⏸️ LEGACY
**파일**: `scripts/production/supertrend_regime_bot.py`
**설정**: `config/supertrend_regime_bot_config.yaml`
**상태**: ⏸️ **레거시** - ADX Supertrend Trail Bot으로 교체됨

| 파라미터 | 값 |
|---------|-----|
| Entry | Supertrend Direction Change |
| TP/SL | 3.5%/1.8% (고정) |
| Regime Filter | MTF 3단계 |

**성과**: Full Period +129.7%, 13 trades (거래 빈도 낮음)

```bash
# Legacy commands (if needed)
START_SUPERTREND_REGIME.bat
MONITOR_SUPERTREND_REGIME.bat
```

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
- **RSI Zone Entry Bot v1.3.2**: TP 2.4% / SL 1.4% / BE_SL 1.2%
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
    │   ├── adx_supertrend_trail_config.yaml  ← ✅ ACTIVE
    │   ├── supertrend_regime_bot_config.yaml ← LEGACY
    │   └── rsi_zone_bot_config.yaml          ← LEGACY
    │
    ├── scripts/
    │   ├── production/
    │   │   ├── adx_supertrend_trail_bot.py  ← ✅ ACTIVE (v1.0)
    │   │   ├── supertrend_regime_bot.py     ← LEGACY
    │   │   ├── rsi_zone_bot.py              ← LEGACY
    │   │   ├── ema_crossover_bot.py         ← LEGACY
    │   │   ├── vwap_band_bot.py             ← LEGACY
    │   │   └── donchian_scalping_bot.py     ← LEGACY
    │   │
    │   ├── monitoring/
    │   │   ├── adx_supertrend_trail_monitor.py  ← ✅ ACTIVE
    │   │   ├── supertrend_regime_monitor.py     ← LEGACY
    │   │   └── rsi_zone_monitor.py              ← LEGACY
    │   │
    │   └── analysis/
    │       ├── dynamic_stoploss_research.py         ← 5 SL methods comparison
    │       ├── supertrend_trail_validation.py       ← Walk-forward validation
    │       ├── entry_signal_research_with_regime.py ← 21 method screening
    │       └── ...
    │
    ├── results/
    │   ├── adx_supertrend_trail_bot_state.json  ← ✅ ACTIVE
    │   ├── supertrend_regime_bot_state.json     ← LEGACY
    │   ├── rsi_zone_bot_state.json              ← LEGACY
    │   └── backups/
    │
    ├── claudedocs/
    │   ├── DYNAMIC_STOPLOSS_RESEARCH_20251213.md        ← 🆕 동적 SL 연구
    │   ├── ENTRY_SIGNAL_RESEARCH_COMPREHENSIVE_20251212.md
    │   └── ...
    │
    ├── logs/
    │   └── adx_supertrend_trail_bot_YYYYMMDD.log
    │
    ├── START_ADX_SUPERTREND_TRAIL.bat    ← ✅ ACTIVE
    ├── MONITOR_ADX_SUPERTREND_TRAIL.bat  ← ✅ ACTIVE
    ├── START_SUPERTREND_REGIME.bat       ← LEGACY
    └── MONITOR_SUPERTREND_REGIME.bat     ← LEGACY
```

---

## 🔧 Known Issues & Fixes

| 날짜 | 이슈 | 해결 |
|------|------|------|
| 2025-12-13 | **ADX Supertrend Trail v1.0 배포** | 동적 SL 연구 기반 (+1115%p 개선) |
| 2025-12-13 | **동적 손절 연구** | 5가지 SL 방식 비교, Supertrend Trail 최적 |
| 2025-12-13 | Supertrend Bot v1.0 배포 | RSI Zone v2.2 교체 |
| 2025-12-12 | Entry Signal Research | 21 methods, 1080 combinations 테스트 |
| 2025-12-11 | v1.3.2 배포 | API 재시도, 헬스체크, 에러 처리 |

---

## 🧠 AI Assistant Instructions

### ADX Supertrend Trail Bot 핵심 사항
1. **Entry**: ADX > 20 + DI crossover
2. **TP**: 2.0% 고정
3. **SL**: Supertrend 트레일링 (고정 % 아님!)
   - 유리한 방향으로만 이동
   - 추세 전환 시 손절
4. **SL 거리 필터**: 0.3% ~ 5.0% (안전장치)
5. **저변동성 시장**: SL 거리 < 0.3%면 신호 스킵 (정상 동작)

### 동적 SL의 핵심 개념
```
고정 SL: entry_price - 2.0%  (시장 상황 무관)
동적 SL: Supertrend line     (변동성에 적응, 추세 전환 감지)

LONG 포지션:
- SL = max(현재SL, Supertrend) → 상승만 추적
- Supertrend 하향 돌파 시 손절

SHORT 포지션:
- SL = min(현재SL, Supertrend) → 하락만 추적
- Supertrend 상향 돌파 시 손절
```

### Code Modification Rules
1. **Order Creation**: Hedge Mode에서 reduce_only 사용 불가
2. **Position Sizing**: EFFECTIVE_LEVERAGE (4x) 기준 계산
3. **State Management**: state.json 백업 후 변경
4. **CCXT 제한**: conditional orders는 Raw API 사용
5. **SL 업데이트**: 매 분마다 Supertrend 체크 및 SL 수정

### 관련 문서
| 문서 | 내용 |
|------|------|
| `claudedocs/DYNAMIC_STOPLOSS_RESEARCH_20251213.md` | 동적 SL 연구 결과 |
| `config/adx_supertrend_trail_config.yaml` | 봇 설정 |

---

**Last Updated**: 2025-12-13 KST
