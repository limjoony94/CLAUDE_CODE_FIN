# CLAUDE_CODE_FIN - Trading Bot Workspace

**Last Updated**: 2026-01-31 KST | **Active Bot**: Pattern 5m v1.20.0 (Unified Classification Re-discovery)

---

## Active Bot: Pattern 5m v1.20.0 (Unified Classification)

| 항목 | 값 |
|------|-----|
| **패키지** | [pattern_5m/](bingx_rl_trading_bot/scripts/production/pattern_5m/) |
| **엔트리** | [pattern_5m_bot.py](bingx_rl_trading_bot/scripts/production/pattern_5m_bot.py) |
| **설정** | [pattern_5m_config.yaml](bingx_rl_trading_bot/config/pattern_5m_config.yaml) |
| **상태** | `results/pattern_5m_bot_state.json` |
| **메트릭** | `results/pattern_5m_metrics.json` |

### Unified Classification Re-discovery (v1.20.0)

> **Research**: [unified_pattern_discovery.py](bingx_rl_trading_bot/scripts/analysis/unified_pattern_discovery.py)

**핵심**: v1.19.2까지 연구 스크립트와 프로덕션의 캔들 분류가 불일치 (연구: body/range, 프로덕션: avg_body_20 정규화). v1.20.0에서 프로덕션 분류체계로 1,728패턴 재발굴.

**v1.20.0 변경** (v1.19.2 → v1.20.0):
- 프로덕션 분류체계(avg_body_20)로 전체 재발굴
- 21패턴(10L+11S) → **13패턴(7L+6S)** (LONG Tier1: MC<0.01, SHORT Tier1.5: MC<0.03)
- MDD 32.9% → **14.8%**, PF 1.52 → **2.62**, WF 4/5 → **5/5**

### 270-Day Validation (v1.20.0)

> **Data**: 2025-05-05 ~ 2026-01-30 (77,760 bars, H1: +22.6%, H2: -5.3%, H3: -24.2%)

| 지표 | v1.20.0 (Unified Classification) |
|------|----------------------------------|
| **Patterns** | **13 (7L+6S)** |
| **TP / SL** | **1.0% / 1.0% (전 패턴 균일)** |
| **Trades** | **353** |
| **WR** | **73.7%** |
| **WF** | **5/5** |
| **MC p-value** | **0.0000** |
| **Max DD** | **14.8%** |
| **PF** | **2.62** |
| **3/3 Period Profit** | ✅ |

**LONG Patterns (7)** — Tier 1 (WF≥4, MC<0.01, excess>15), 균일 TP 1.0% / SL 1.0%:

| Pattern | Trades | WR | Excess | MC | WF | PP |
|---------|--------|-----|--------|--------|-----|-----|
| U-MU-H | 57 | 68.4% | +16.7% | 0.0033 | 4/5 | 3/3 |
| MD-ST-MD | 48 | 70.8% | +19.1% | 0.0023 | 4/5 | 3/3 |
| GS-U-BD | 25 | 76.0% | +24.3% | 0.0069 | 4/5 | 2/3 |
| MD-MD-ST | 38 | 71.1% | +19.3% | 0.0062 | 5/5 | 3/3 |
| BU-IH-DN | 25 | 76.0% | +24.3% | 0.0083 | 4/5 | 3/3 |
| MD-H-MD | 18 | 83.3% | +31.6% | 0.0045 | 5/5 | 3/3 |
| IH-MD-MD | 15 | 86.7% | +34.9% | 0.0033 | 4/5 | 2/3 |

**SHORT Patterns (6)** — Tier 1.5 (WF≥4, MC<0.03, excess>15), 균일 TP 1.0% / SL 1.0%:

| Pattern | Trades | WR | Excess | MC | WF | PP |
|---------|--------|-----|--------|--------|-----|-----|
| DN-D-BD | 46 | 67.4% | +19.1% | 0.0120 | 5/5 | 3/3 |
| BD-U-GS | 17 | 76.5% | +28.2% | 0.0238 | 4/5 | 3/3 |
| DN-GS-H | 15 | 80.0% | +31.7% | 0.0141 | 4/5 | 2/3 |
| U-DF-BU | 17 | 76.5% | +28.2% | 0.0208 | 4/5 | 2/3 |
| BD-GS-BD | 17 | 76.5% | +28.2% | 0.0265 | 4/5 | 3/3 |
| DN-IH-IH | 15 | 80.0% | +31.7% | 0.0162 | 5/5 | 3/3 |

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
| **TP / SL** | **1.0% / 1.0% (균일)** | **v1.20.0: 전 패턴 동일** |
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
| **Regime-Adaptive (v1.18)** | BULL/BEAR/SIDEWAYS 레짐별 패턴 + TP/SL 자동 선택 |
| **Unified Classification (v1.20.0)** | 프로덕션 분류체계 재발굴, 13패턴 (7L+6S), MDD 14.8%, PF 2.62 |

### Commands

```bash
START_PATTERN_5M.bat    # 시작 (백그라운드)
STOP_PATTERN_5M.bat     # 종료
MONITOR_PATTERN_5M.bat  # 모니터링
```

### Version History

| 버전 | 날짜 | 변경사항 |
|------|------|---------|
| **v1.20.0** | 01-31 | **Unified Classification Re-discovery** - 연구/프로덕션 분류 불일치 수정, avg_body_20 기반 1,728패턴 재발굴, 21→13패턴(7L+6S), MDD 14.8%, PF 2.62, WF 5/5, MC=0.0000 ← **현재 운영** |
| v1.19.2 | 01-30 | Uniform 1.0/1.0 TP/SL (분류 불일치 상태) |
| v1.19.1 | 01-30 | 21-Pattern Expansion - Tier 1 패턴 6개 추가, 15→21 패턴 |
| v1.19.0 | 01-30 | Tight TP/SL Regime-Independent - Wide TP/SL 비대칭 편향 발견, 0.3-1.0% tight TP/SL로 전환, 8L+7S, regime 비활성화, 270일 66개 검증 → 15개 선정 |
| v1.18.2 | 01-30 | Regime Threshold 최적화 - ±2.0% → ±1.5% (270일 검증) |
| v1.18.1 | 01-27 | Regime-Aware TP/SL Auto-Adjust Fix |
| v1.18 | 01-27 | Regime-Adaptive Strategy - BULL/BEAR/SIDEWAYS 레짐별 패턴 + TP/SL, WF 5/5 |
| v1.17 | 01-26 | Statistical Validation + TP/SL Auto-Adjustment - D-DN-BD 제거, 18개 패턴 검증 |
| v1.16 | 01-26 | Pattern Discovery Expansion - 1,728 패턴 전수검사 |
| v1.15 | 01-26 | Regime-Validated TP/SL |
| v1.14 | 01-26 | Context Research Optimization |
| v1.13 | 01-25 | Early Exit Optimization - confirm_candles 2→3 |
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
│   │   ├── constants.py        ← 상수 + v1.20.0 패턴/TP/SL ★
│   │   ├── exchange.py         ← API 인터페이스
│   │   ├── indicators.py       ← 기술 지표
│   │   ├── models.py           ← 데이터클래스
│   │   ├── orders.py           ← 주문 관리 + TP/SL 자동 조정
│   │   ├── position.py         ← 포지션 관리 (facade)
│   │   ├── position_open.py    ← 포지션 진입
│   │   ├── position_monitor.py ← 포지션 모니터링
│   │   ├── position_close.py   ← 포지션 청산
│   │   ├── signals.py          ← 패턴 탐지 + Context Filter + Regime 감지
│   │   ├── state.py            ← 상태 저장
│   │   └── utils/
│   │       ├── lock.py
│   │       └── logging_config.py
│   └── engulf_5m/              ← (Archived) Engulf bot
├── scripts/analysis/           ← 연구 스크립트
│   ├── unified_pattern_discovery.py                  ← v1.20.0 통합 재발굴 ★
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
│   ├── btc_5m_270days.csv      ← 270일 데이터 (Binance, 2025-05~2026-01) ★
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

★ = v1.20.0에서 수정/추가된 파일

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
