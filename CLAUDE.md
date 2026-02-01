# CLAUDE_CODE_FIN - Trading Bot Workspace

**Last Updated**: 2026-02-01 KST | **Active Bot**: Pattern 5m v1.22.0 (DN-D-BD Removed)

---

## Active Bot: Pattern 5m v1.22.0 (12 Patterns, 7L+5S)

| 항목 | 값 |
|------|-----|
| **패키지** | [pattern_5m/](bingx_rl_trading_bot/scripts/production/pattern_5m/) |
| **엔트리** | [pattern_5m_bot.py](bingx_rl_trading_bot/scripts/production/pattern_5m_bot.py) |
| **설정** | [pattern_5m_config.yaml](bingx_rl_trading_bot/config/pattern_5m_config.yaml) |
| **상태** | `results/pattern_5m_bot_state.json` |
| **메트릭** | `results/pattern_5m_metrics.json` |

### Conservative Per-Pattern TP/SL (v1.21.0)

> **Research**: [per_pattern_tpsl_optimization.py](bingx_rl_trading_bot/scripts/analysis/per_pattern_tpsl_optimization.py), [portfolio_tpsl_comparison.py](bingx_rl_trading_bot/scripts/analysis/portfolio_tpsl_comparison.py)

**핵심**: 개별 패턴 MC<0.01만 per-pattern TP/SL 적용, 나머지는 uniform 1.0/1.0 유지 (Conservative 전략)

**v1.21.0 변경** (v1.20.1 → v1.21.0):
- 균일 1.0/1.0 → **Conservative per-pattern TP/SL** (8/13 optimized, 5/13 uniform)
- Portfolio 검증: WR 73.7→**78.5%**, PF 2.59→**3.19**, MDD 14.6→20.6%
- 13 패턴 유지 (7L+6S), 레짐 비활성화 유지

### 270-Day Validation (v1.21.0)

> **Data**: 2025-05-05 ~ 2026-01-30 (77,760 bars, H1: +22.6%, H2: -5.3%, H3: -24.2%)

| 지표 | v1.21.0 (Conservative) | v1.20.1 (Uniform) |
|------|------------------------|-------------------|
| **Patterns** | 13 (7L+6S) | 13 (7L+6S) |
| **TP / SL** | **Per-pattern (아래 표)** | 1.0% / 1.0% 균일 |
| **Trades** | **312** | 353 |
| **WR** | **78.5%** | 73.7% |
| **WF** | **5/5** | 5/5 |
| **Max DD** | 20.6% | 14.8% |
| **PF** | **3.19** | 2.59 |
| **3/3 Period Profit** | ✅ | ✅ |

**LONG Patterns (7)**:

| Pattern | TP/SL | Trades | WR | MC | WF | Note |
|---------|-------|--------|-----|--------|-----|------|
| U-MU-H | **1.5/1.5** | 57 | 68.4% | 0.0000 | 4/5 | optimized |
| MD-ST-MD | **2.0/2.0** | 48 | 70.8% | 0.0078 | 4/5 | optimized |
| GS-U-BD | 1.0/1.0 | 25 | 76.0% | 0.0372 | 4/5 | uniform (MC>0.01) |
| MD-MD-ST | **1.5/2.0** | 38 | 71.1% | 0.0002 | 5/5 | optimized |
| BU-IH-DN | **1.5/2.0** | 25 | 76.0% | 0.0022 | 4/5 | optimized |
| MD-H-MD | 1.0/1.0 | 18 | 83.3% | 0.0014 | 5/5 | uniform best |
| IH-MD-MD | **1.5/2.0** | 15 | 86.7% | 0.0020 | 4/5 | optimized |

**SHORT Patterns (6)**:

| Pattern | TP/SL | Trades | WR | MC | WF | Note |
|---------|-------|--------|-----|--------|-----|------|
| DN-D-BD | 1.0/1.0 | 46 | 67.4% | 0.2390 | 5/5 | uniform (MC fail) |
| BD-U-GS | **1.5/2.0** | 17 | 76.5% | 0.0042 | 4/5 | optimized |
| DN-GS-H | 1.0/1.0 | 15 | 80.0% | 0.0176 | 4/5 | uniform (MC>0.01) |
| U-DF-BU | **1.0/1.5** | 17 | 76.5% | 0.0010 | 4/5 | optimized |
| BD-GS-BD | 1.0/1.0 | 17 | 76.5% | 0.0120 | 4/5 | uniform (MC>0.01) |
| DN-IH-IH | **1.0/1.5** | 15 | 80.0% | 0.0000 | 5/5 | optimized |

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

### Strategy Parameters

| 파라미터 | 값 | 비고 |
|---------|-----|------|
| Entry | 3-candle pattern match | 12-type classification |
| **TP / SL** | **Per-pattern (1.0-2.0%)** | **v1.21.0: MC<0.01 optimized** |
| **Regime** | **비활성화** | **v1.19.0~: tight TP/SL은 레짐 독립적** |
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
| **TP/SL Auto-Adjust (v1.17)** | 봇 시작 시 기존 포지션의 TP/SL을 config에 맞게 자동 조정 |
| **Context Filters (v1.14)** | RSI/Vol/Trend/Position/Session 기반 필터링 |
| **Per-Pattern TP/SL (v1.21.0)** | MC<0.01 패턴 개별 최적화, 나머지 uniform 유지 |
| **Leverage Side Fix (v1.21.1)** | BingX setLeverage `side='BOTH'` 파라미터 추가 |

### Commands

```bash
START_PATTERN_5M.bat    # 시작 (백그라운드)
STOP_PATTERN_5M.bat     # 종료
MONITOR_PATTERN_5M.bat  # 모니터링
```

### Version History

| 버전 | 날짜 | 변경사항 |
|------|------|---------|
| **v1.22.0** | 02-01 | **DN-D-BD 제거** - 과적합 진단(4-test): MC=0.2390 Holm fail, WR 67.4%(최저). 제거 후 WR 80.3%, PF 3.36, WF 5/5 ← **현재 운영** |
| v1.21.1 | 02-01 | Leverage side fix + state cleanup - BingX setLeverage `side='BOTH'` 추가, stale regime 데이터 제거 |
| v1.21.0 | 02-01 | Conservative Per-Pattern TP/SL - MC<0.01 패턴 개별 최적화 (8/13), 나머지 uniform, WR 78.5%, PF 3.19, WF 5/5 |
| v1.20.1 | 02-01 | Improved early-bar classification (default avg_body_20=1.0) |
| v1.20.0 | 01-31 | Unified Classification Re-discovery - 연구/프로덕션 분류 불일치 수정, 21→13패턴(7L+6S), MDD 14.8%, PF 2.62 |
| v1.19.2 | 01-30 | Uniform 1.0/1.0 TP/SL |
| v1.19.1 | 01-30 | 21-Pattern Expansion |
| v1.19.0 | 01-30 | Tight TP/SL Regime-Independent |
| v1.18.2 | 01-30 | Regime Threshold 최적화 |
| v1.18.1 | 01-27 | Regime-Aware TP/SL Auto-Adjust Fix |
| v1.18 | 01-27 | Regime-Adaptive Strategy |
| v1.17 | 01-26 | Statistical Validation + TP/SL Auto-Adjustment |
| v1.16 | 01-26 | Pattern Discovery Expansion |
| v1.15 | 01-26 | Regime-Validated TP/SL |
| v1.14 | 01-26 | Context Research Optimization |
| v1.13 | 01-25 | Early Exit Optimization |
| v1.12 | 01-25 | Statistical Validity Optimization |
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
│   │   ├── bot.py              ← 메인 루프 + Early Exit + TP/SL 자동 조정 호출
│   │   ├── config.py           ← 설정 관리
│   │   ├── constants.py        ← 상수 + v1.21.0 per-pattern TP/SL ★
│   │   ├── exchange.py         ← API 인터페이스
│   │   ├── indicators.py       ← 기술 지표
│   │   ├── models.py           ← 데이터클래스
│   │   ├── orders.py           ← 주문 관리 + TP/SL 자동 조정
│   │   ├── position.py         ← 포지션 관리 (facade)
│   │   ├── position_open.py    ← 포지션 진입 + leverage side fix ★
│   │   ├── position_monitor.py ← 포지션 모니터링
│   │   ├── position_close.py   ← 포지션 청산
│   │   ├── signals.py          ← 패턴 탐지 + Context Filter + Regime 감지
│   │   ├── state.py            ← 상태 저장
│   │   └── utils/
│   │       ├── lock.py
│   │       └── logging_config.py
│   └── engulf_5m/              ← (Archived) Engulf bot
├── scripts/analysis/           ← 연구 스크립트
│   ├── overfitting_diagnosis.py                     ← v1.22.0 과적합 진단 (4-test) ★
│   ├── per_pattern_tpsl_optimization.py             ← v1.21.0 패턴별 TP/SL 최적화
│   ├── portfolio_tpsl_comparison.py                 ← v1.21.0 포트폴리오 비교
│   ├── unified_pattern_discovery.py                 ← v1.20.0 통합 재발굴
│   ├── deploy_comparison.py                         ← 배포 비교 검증
│   ├── tpsl_sensitivity_corrected.py                ← TP/SL 민감도 분석
│   ├── tight_tpsl_validation.py                     ← v1.19.0 tight TP/SL 검증
│   ├── comprehensive_regime_analysis.py             ← v1.18.2 270일 종합 검증
│   ├── download_extended_data.py                    ← 270일 데이터 다운로드
│   ├── per_pattern_backtest.py                      ← 패턴별 개별 백테스트
│   ├── production_pattern_validation.py             ← v1.17 통계적 검증
│   ├── pattern_discovery_optimized.py               ← v1.16 패턴 발굴
│   └── early_exit_deep_analysis.py                  ← Early Exit 연구
├── data/
│   ├── btc_5m_270days.csv      ← 270일 데이터 (Binance, 2025-05~2026-01)
│   ├── btc_5m_extended.csv     ← 105일 데이터 (BingX)
│   └── btc_5m_90days_*.csv
├── claudedocs/
│   ├── PRODUCTION_VALIDATION_REPORT_20260126.md
│   ├── PATTERN_DISCOVERY_REPORT_20260126.md
│   ├── EARLY_EXIT_SIGNAL_RESEARCH_20260123.md
│   └── STANDARD_RESEARCH_PROTOCOL.md
├── results/
│   ├── pattern_5m_bot_state.json
│   └── pattern_5m_metrics.json
├── logs/pattern_5m_bot_*.log
├── archive/deprecated_bots/
└── *.bat (START, STOP, MONITOR)
```

★ = v1.22.0에서 수정/추가된 파일

---

## Standard Research Protocol

> **Full Documentation**: [STANDARD_RESEARCH_PROTOCOL.md](bingx_rl_trading_bot/claudedocs/STANDARD_RESEARCH_PROTOCOL.md)

모든 전략 연구는 아래 프로토콜을 준수해야 합니다.

### Backtest Rules

| 항목 | 표준 |
|------|------|
| **Entry 타이밍** | 신호 다음 봉 Open |
| **Exit 타이밍** | Intrabar High/Low (distance-based TP/SL resolution) |
| **Position Sizing** | Compound (복리) |
| **수수료** | 0.05% × 2 = 0.10% |
| **슬리피지** | 0.02% 버퍼 |
| **MC Test** | Sign randomization (10k sims) |
| **WF** | 5-fold out-of-sample |

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
