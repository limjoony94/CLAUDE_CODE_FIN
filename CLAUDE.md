# CLAUDE_CODE_FIN - Trading Bot Workspace

**Last Updated**: 2026-01-26 KST | **Active Bot**: Pattern 5m v1.17 (Statistical Validation + TP/SL Auto-Adjust)

---

## Active Bot: Pattern 5m v1.17 (Statistical Validation + TP/SL Auto-Adjust)

| 항목 | 값 |
|------|-----|
| **패키지** | [pattern_5m/](bingx_rl_trading_bot/scripts/production/pattern_5m/) |
| **엔트리** | [pattern_5m_bot.py](bingx_rl_trading_bot/scripts/production/pattern_5m_bot.py) |
| **설정** | [pattern_5m_config.yaml](bingx_rl_trading_bot/config/pattern_5m_config.yaml) |
| **상태** | `results/pattern_5m_bot_state.json` |
| **메트릭** | `results/pattern_5m_metrics.json` |

### Strategy Overview

**12-Type Candle Classification System**:
| Code | Type | Description |
|------|------|-------------|
| D | DOJI | body < 10% of range |
| DF | DRAGONFLY | lower wick > 70% of range |
| GS | GRAVESTONE | upper wick > 70% of range |
| H | HAMMER | lower wick > 2x body |
| IH | INV_HAMMER | upper wick > 2x body |
| ST | SPINNING_TOP | small body, balanced wicks |
| MU | MARUBOZU_UP | bullish, wicks < 15% |
| MD | MARUBOZU_DOWN | bearish, wicks < 15% |
| BU | BIG_UP | normalized body > 1.5 |
| BD | BIG_DOWN | normalized body > 1.5 |
| U | MED_UP | medium bullish |
| DN | MED_DOWN | medium bearish |

### Validated Patterns (v1.17 Statistical Validation)

> **Research**: [production_pattern_validation.py](bingx_rl_trading_bot/scripts/analysis/production_pattern_validation.py)
>
> **Report**: [PRODUCTION_VALIDATION_REPORT_20260126.md](bingx_rl_trading_bot/claudedocs/PRODUCTION_VALIDATION_REPORT_20260126.md)

**v1.17 Key Changes** (Statistical Validation):
- **D-DN-BD 제거**: 6 trades only, p=1.0 (통계적으로 무의미)
- **18개 패턴 검증**: 14/18 통계적 유의 (p<0.05)
- **검증 기준**: WF ≥ 4/5, WR ≥ 60%, Edge > 0, Trades ≥ 10

**LONG Patterns** (8개):
| Pattern | Trades | WR | Edge | WF | p-value | TP/SL |
|---------|--------|------|------|-----|---------|-------|
| U-BU-U | 27 | 70.4% | +1.29 | 4/5 | 0.091 | 1.5/2.0% |
| ST-BD-DN | 11 | 90.9% | +4.54 | 4/5 | **0.004** | 2.0/3.0% |
| **DN-DN-DN** | **148** | **87.8%** | **+1.44** | **5/5** | **<0.001** | 1.0/3.0% |
| DN-U-U | 145 | 80.0% | +0.50 | 5/5 | 0.107 | 1.0/3.0% |
| DN-DN-U | 145 | 83.4% | +0.91 | 4/5 | **0.008** | 1.0/3.0% |
| DN-ST-U | 94 | 85.1% | +1.11 | 5/5 | **0.007** | 1.0/3.0% |
| U-ST-U | 85 | 84.7% | +1.06 | 5/5 | **0.013** | 1.0/3.0% |
| U-U-U | 89 | 71.9% | +0.61 | 4/5 | 0.175 | 1.5/3.0% |

**SHORT Patterns** (10개 - D-DN-BD removed v1.17):
| Pattern | Trades | WR | Edge | WF | p-value | TP/SL |
|---------|--------|------|------|-----|---------|-------|
| BD-BD-BD | 13 | 84.6% | +6.36 | 5/5 | **0.002** | 3.0/2.5% |
| DN-DN-BD | 38 | 89.5% | +2.98 | 4/5 | **<0.001** | 1.5/3.0% |
| MU-ST-DN | 33 | 93.9% | +2.26 | 5/5 | **<0.001** | 1.0/2.5% |
| IH-DN-DN | 17 | 88.2% | +1.49 | 4/5 | 0.072 | 1.0/3.0% |
| BD-ST-DN | 14 | 92.9% | +3.44 | 5/5 | **0.002** | 1.5/3.0% |
| BU-U-DN | 36 | 83.3% | +2.40 | 4/5 | **0.002** | 1.5/2.5% |
| ~~D-DN-BD~~ | ~~6~~ | ~~83.3%~~ | - | - | ~~1.0~~ | **REMOVED v1.17** |
| **U-DN-DN** | **172** | **90.1%** | **+1.71** | **4/5** | **<0.001** | 1.0/3.0% |
| DN-U-DN | 66 | 75.8% | +2.26 | 4/5 | **0.003** | 2.0/3.0% |
| DN-DN-ST | 53 | 83.0% | +2.11 | 5/5 | **0.002** | 1.5/3.0% |
| U-U-DN | 77 | 74.0% | +2.00 | 4/5 | **0.005** | 2.0/3.0% |

### Pattern-Specific TP/SL (v1.17)

> **v1.17**: 18개 패턴별 최적화 TP/SL (D-DN-BD removed)

| Pattern | TP% | SL% | Direction | WF | p-value |
|---------|-----|-----|-----------|-----|---------|
| U-BU-U | 1.5 | 2.0 | LONG | 4/5 | 0.091 |
| ST-BD-DN | 2.0 | 3.0 | LONG | 4/5 | 0.004 |
| DN-DN-DN | 1.0 | 3.0 | LONG | 5/5 | <0.001 |
| DN-U-U | 1.0 | 3.0 | LONG | 5/5 | 0.107 |
| DN-DN-U | 1.0 | 3.0 | LONG | 4/5 | 0.008 |
| DN-ST-U | 1.0 | 3.0 | LONG | 5/5 | 0.007 |
| U-ST-U | 1.0 | 3.0 | LONG | 5/5 | 0.013 |
| U-U-U | 1.5 | 3.0 | LONG | 4/5 | 0.175 |
| BD-BD-BD | 3.0 | 2.5 | SHORT | 5/5 | 0.002 |
| DN-DN-BD | 1.5 | 3.0 | SHORT | 4/5 | <0.001 |
| MU-ST-DN | 1.0 | 2.5 | SHORT | 5/5 | <0.001 |
| IH-DN-DN | 1.0 | 3.0 | SHORT | 4/5 | 0.072 |
| BD-ST-DN | 1.5 | 3.0 | SHORT | 5/5 | 0.002 |
| BU-U-DN | 1.5 | 2.5 | SHORT | 4/5 | 0.002 |
| U-DN-DN | 1.0 | 3.0 | SHORT | 4/5 | <0.001 |
| DN-U-DN | 2.0 | 3.0 | SHORT | 4/5 | 0.003 |
| DN-DN-ST | 1.5 | 3.0 | SHORT | 5/5 | 0.002 |
| U-U-DN | 2.0 | 3.0 | SHORT | 4/5 | 0.005 |

### Strategy Parameters

| 파라미터 | 값 | 비고 |
|---------|-----|------|
| Entry | 3-candle pattern match | 12-type classification |
| **TP / SL** | **패턴별 최적값** | **v1.17 (PATTERN_OPTIMAL_TPSL)** |
| Default TP/SL | 1.5% / 3.0% | 패턴 매칭 실패 시 |
| Double Exit | 50%@0.8x + 50%@1.0x | Scale-out strategy |
| Leverage | 3x (effective) | Position sizing 기준 |
| Cooldown | 0 candles | 연속 진입 허용 |
| Timeframe | 5m | |

### Early Exit Signal (v1.13)

> **Research**: [early_exit_deep_analysis.py](bingx_rl_trading_bot/scripts/analysis/early_exit_deep_analysis.py)

**개요**: 극단적 반전 시에만 조기 청산 (3연속 BD/BU)

| 파라미터 | 값 | 비고 |
|---------|-----|------|
| bearish_types | `['BD']` | LONG 청산 트리거 |
| bullish_types | `['BU']` | SHORT 청산 트리거 |
| confirm_candles | **3** | 극단적 반전에서만 |
| min_profit_pct | 0.3% | 최소 이익 조건 |

### Expected Performance (v1.17)

| 지표 | v1.16 | v1.17 | 변화 |
|------|-------|-------|------|
| Total Patterns | 19 | **18** | -1 (D-DN-BD removed) |
| LONG Patterns | 8 | **8** | 0 |
| SHORT Patterns | 11 | **10** | -1 |
| Statistically Significant | - | **14/18** | 77.8% |
| Avg WR | 87.8% | **83.9%** | (validated) |
| Avg Edge | +2.10 | **+2.21** | +5.2% |

### Features (Modular Architecture)

| 기능 | 설명 |
|------|------|
| **14 Modules** | bot, config, constants, exchange, indicators, models, orders, position, position_open, position_monitor, position_close, signals, state, utils |
| API Caching | TTL 5초 (ticker, balance, positions) |
| Circuit Breaker | 5회 실패 → 60초 차단 |
| Performance Metrics | Expected vs Actual 비교 |
| Crash Recovery | Orphan position 탐지/복구 |
| Health Check | API/CB/Metrics 종합 진단 |
| **Early Exit (v1.13)** | 3x BD/BU 연속 출현 + 0.3% 이익 시 조기청산 |
| **Pattern TP/SL (v1.17)** | 18개 패턴별 최적화된 TP/SL 자동 적용 |
| **TP/SL Auto-Adjust (v1.17)** | 봇 시작 시 기존 포지션의 TP/SL을 config에 맞게 자동 조정 |
| **Context Filters (v1.14)** | RSI/Vol/Trend/Position/Session 기반 필터링 |
| **Statistical Validation (v1.17)** | t-test, binomial test로 통계적 유의성 검증 |

### Commands

```bash
START_PATTERN_5M.bat    # 시작 (백그라운드)
STOP_PATTERN_5M.bat     # 종료
MONITOR_PATTERN_5M.bat  # 모니터링
```

### Context Filters (v1.14)

> **v1.14**: MU-ST-DN, BD-BD-BD 필터 추가

**상태**: ✅ 프로덕션 적용 (5개 필터 활성)

| Pattern | Direction | Filter Type | Context | Improvement |
|---------|-----------|-------------|---------|-------------|
| DN-DN-BD | SHORT | Required | RSI < 30 | Base filter |
| U-BU-U | LONG | Preferred | Downtrend | +14% WR |
| IH-DN-DN | SHORT | Excluded | High Vol | +17.5% WR |
| MU-ST-DN | SHORT | Preferred | position_zone=L | +36.2% WR |
| BD-BD-BD | SHORT | Preferred | session=ASIA | +29.8% WR |

**Filter Types**:
- `required`: 조건 불일치 시 신호 거부
- `preferred`: 조건 일치 시 confidence +10% 보너스
- `excluded`: 조건 일치 시 신호 거부

### Version History

| 버전 | 날짜 | 변경사항 |
|------|------|---------|
| **v1.17** | 01-26 | **Statistical Validation + TP/SL Auto-Adjustment** - D-DN-BD 제거 (6 trades, p=1.0), 18개 패턴 검증 (14/18 p<0.05), 봇 시작 시 TP/SL 자동 조정 기능 추가 ← **현재 운영** |
| v1.16 | 01-26 | Pattern Discovery Expansion - 1,728 패턴 전수검사, LONG 2→8개, SHORT 7→11개 |
| v1.15 | 01-26 | Regime-Validated TP/SL - 베어마켓 바이어스 수정, TP 타이트 + SL 와이드 |
| v1.14 | 01-26 | Context Research Optimization - MU-U-DN/DN-BD-BD 제거, 신규 패턴 추가, TP/SL 최적화 |
| v1.13 | 01-25 | Early Exit Optimization - confirm_candles 2→3 (baseline 대비 +21.7%) |
| v1.12 | 01-25 | Statistical Validity Optimization - U-DN-DN, MU-ST-ST 제거 |
| v1.11 | 01-25 | DN-BD-BD Pattern Added |
| v1.10 | 01-25 | WF-Validated TP/SL + RSI Filter |
| v1.9 | 01-25 | Low WR Pattern Optimization |
| v1.8 | 01-25 | Exclusion Filters |
| v1.7 | 01-24 | Context Filters |
| v1.6 | 01-24 | Pattern-Specific TP/SL |
| v1.5 | 01-24 | TP/SL Optimization |
| v1.4 | 01-24 | Exhaustive Search Optimization |
| v1.3 | 01-23 | Early Exit Signal |
| v1.2 | 01-22 | Confidence logging |
| **v1.0** | 01-22 | **Initial Release** |

---

## File Structure

```
bingx_rl_trading_bot/
├── config/
│   ├── pattern_5m_config.yaml  ← 전략 설정
│   └── api_keys.yaml           ← API 키
├── scripts/production/
│   ├── pattern_5m_bot.py       ← 엔트리포인트
│   ├── pattern_5m/             ← 모듈 패키지
│   │   ├── __init__.py
│   │   ├── bot.py              ← 메인 루프 + Early Exit + TP/SL 자동 조정 호출 ★
│   │   ├── config.py           ← 설정 관리
│   │   ├── constants.py        ← 상수 + 검증된 패턴 목록 ★ (v1.17 - 18개 패턴)
│   │   ├── exchange.py         ← API 인터페이스
│   │   ├── indicators.py       ← 기술 지표
│   │   ├── models.py           ← 데이터클래스
│   │   ├── orders.py           ← 주문 관리 + TP/SL 자동 조정 ★
│   │   ├── position.py         ← 포지션 관리 (facade)
│   │   ├── position_open.py    ← 포지션 진입 + Pattern TP/SL 적용
│   │   ├── position_monitor.py ← 포지션 모니터링
│   │   ├── position_close.py   ← 포지션 청산
│   │   ├── signals.py          ← 패턴 탐지 + Context Filter
│   │   ├── state.py            ← 상태 저장
│   │   └── utils/
│   │       ├── lock.py
│   │       └── logging_config.py
│   └── engulf_5m/              ← (Archived) Engulf bot
├── scripts/analysis/           ← 연구 스크립트
│   ├── production_pattern_validation.py           ← v1.17 통계적 검증 ★
│   ├── pattern_discovery_optimized.py             ← v1.16 패턴 발굴
│   ├── pattern_context_comprehensive_research.py  ← 컨텍스트 연구
│   ├── pattern_validity_analysis.py               ← 통계적 유효성 분석
│   └── early_exit_deep_analysis.py                ← Early Exit 연구
├── claudedocs/
│   ├── PRODUCTION_VALIDATION_REPORT_20260126.md   ← v1.17 검증 보고서 ★
│   ├── PATTERN_DISCOVERY_REPORT_20260126.md       ← v1.16 패턴 발굴 보고서
│   ├── EARLY_EXIT_SIGNAL_RESEARCH_20260123.md
│   └── STANDARD_RESEARCH_PROTOCOL.md
├── results/
│   ├── pattern_5m_bot_state.json
│   ├── pattern_5m_metrics.json
│   └── production_validation_20260126_*.csv       ← v1.17 검증 결과
├── logs/pattern_5m_bot_*.log
├── archive/deprecated_bots/
└── *.bat (START, STOP, MONITOR)
```

★ = v1.17에서 수정/추가된 파일 (TP/SL 자동 조정, 통계적 검증)

---

## Standard Research Protocol

> **Full Documentation**: [STANDARD_RESEARCH_PROTOCOL.md](bingx_rl_trading_bot/claudedocs/STANDARD_RESEARCH_PROTOCOL.md)

모든 전략 연구는 아래 프로토콜을 준수해야 합니다.

### Validation Framework (Two-Tier)

| Type | 목적 | 필수 조건 |
|------|------|----------|
| **Type 1** | 신호 품질 검증 | 신호 ≥100, 승률 ≥50%, 기대값 >0 |
| **Type 2** | 실제 거래 시뮬레이션 | PnL >0%, WF ≥50%, DD <50% |

### Validation Script

```python
from scripts.validation import StrategyValidator

validator = StrategyValidator(
    signal_func=my_signal_function,
    tp_pct=2.5, sl_pct=2.0,  # Or use pattern-specific values
    leverage=3, strategy_name="My Strategy"
)
report = validator.validate(df)
print(report)
```

### Backtest Rules

| 항목 | 표준 |
|------|------|
| **Entry 타이밍** | 신호 다음 봉 Open |
| **Exit 타이밍** | Intrabar High/Low |
| **Position Sizing** | Compound (복리) |
| **수수료** | 0.05% × 2 = 0.10% |
| **슬리피지** | 0.02% 버퍼 |

### Look-Ahead Bias Prevention

```python
# 금지: df['col'].shift(-1), df.rolling(n, center=True)
# 허용: df['col'].shift(1), df.rolling(n).xxx()
```

### Position Mode

```python
params={'positionSide': 'BOTH'}  # One-Way mode
```

---

## Archived Bots

> `archive/deprecated_bots/` 참조

### Engulf 5m v2.3 (Archived 2026-01-22)

- **Strategy**: Bullish/Bearish Engulfing pattern
- **WR**: 61.5%, **PnL**: +83.8% (90-day)
- **Reason for Archive**: Pattern 5m shows higher edge and better WF consistency

### Other Deprecated Bots

`archive/deprecated_bots/20260104/` - WR < 50%, Look-Ahead Bias, 백테스트 방법론 오류 등

---

## MCP Quick Reference

| 작업 | MCP | 예시 |
|-----|-----|------|
| 코드 분석 | Serena | `find_symbol`, `find_referencing_symbols` |
| CCXT 문서 | Context7 | `/ccxt/ccxt` |
| 복잡한 디버깅 | Sequential | `sequentialthinking` |
| 최신 정보 | Tavily | BingX API 변경사항 |
