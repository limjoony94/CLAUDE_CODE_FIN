# CLAUDE_CODE_FIN - Trading Bot Workspace

**Last Updated**: 2026-01-19 KST | **Active Bot**: Engulf 5m v2.3 (Modular)

---

## Active Bot: Engulf 5m v2.3 (Quality)

| 항목 | 값 |
|------|-----|
| **패키지** | [engulf_5m/](bingx_rl_trading_bot/scripts/production/engulf_5m/) |
| **엔트리** | [engulf_5m_bot.py](bingx_rl_trading_bot/scripts/production/engulf_5m_bot.py) |
| **설정** | [engulf_5m_config.yaml](bingx_rl_trading_bot/config/engulf_5m_config.yaml) |
| **상태** | `results/engulf_5m_bot_state.json` |
| **메트릭** | `results/engulf_5m_metrics.json` |

### Strategy Parameters

| 파라미터 | 값 | 비고 |
|---------|-----|------|
| Entry (LONG) | Bullish Engulfing | `body>0, prev<0, close>prev_open, open<prev_close` |
| Entry (SHORT) | Bearish Engulfing | `body<0, prev>0, close<prev_open, open>prev_close` |
| Volume Filter | ≥1.2x (20-MA) | 저변동성 시장 적응 |
| Prev Body Filter | ≥30% | DOJI 제거 |
| Body % Filter | ≥0.24% | v1.9 추가 (작은 캔들 필터링) |
| TP / SL | 2.5% / 2.0% | 고정값 (Vol-Adaptive OFF) |
| Double Exit | 50%@0.8x + 50%@1.0x | Weighted TP 90% |
| Leverage | 3x (effective) | Position sizing 기준 |
| Timeframe | 5m | |

### v2.x Features (2026-01-14~)

| 기능 | 설명 |
|------|------|
| **Modular Architecture** | 16개 모듈로 분리 (v2.3: position 3분할) |
| API Caching | TTL 5초 (ticker, balance, positions) |
| Circuit Breaker | 5회 실패 → 60초 차단 |
| Performance Metrics | Expected vs Actual 비교 |
| Crash Recovery | Orphan position 탐지/복구 |
| Health Check | API/CB/Metrics 종합 진단 |

### Backtest Results (v1.9, 90일)

| 지표 | 값 |
|------|-----|
| Trades | 39 (L:31, S:36 signals) |
| Win Rate | **61.5%** |
| PnL (Scale-out) | **+83.8%** |
| Max DD | 11.3% |
| Walk-Forward | 4/8 (50%) |

### Commands

```bash
START_ENGULF_5M.bat    # 시작 (백그라운드)
STOP_ENGULF_5M.bat     # 종료
MONITOR_ENGULF_5M.bat  # 모니터링
```

### Version History

| 버전 | 날짜 | 변경사항 |
|------|------|---------|
| **v2.3** | 01-18 | **코드 품질 개선** - position.py 3분할, 타입힌트 100%, 구체적 예외처리 |
| v2.2 | 01-18 | 모듈화 리팩토링 - 13개 모듈로 분리, 매직 넘버 상수화 |
| v2.1 | 01-18 | Metrics/State 동기화 수정, 거래 종료 시 즉시 저장 |
| v2.0 | 01-14 | API 캐싱, Circuit Breaker, Performance Metrics |
| v1.9 | 01-14 | Body % Filter ≥0.24% 추가, 방법론 검증 완료 |
| v1.8 | 01-10 | Double Exit 50/50@0.8/1.0 |
| v1.7 | 01-09 | OS-level file locking, precision 수정 |

---

## File Structure

```
bingx_rl_trading_bot/
├── config/
│   ├── engulf_5m_config.yaml   ← 전략 설정
│   └── api_keys.yaml           ← API 키
├── scripts/production/
│   ├── engulf_5m_bot.py        ← 엔트리포인트
│   ├── engulf_5m/              ← v2.2 모듈 패키지
│   │   ├── __init__.py
│   │   ├── bot.py              ← 메인 루프
│   │   ├── config.py           ← 설정 관리
│   │   ├── constants.py        ← 상수/매직넘버
│   │   ├── exchange.py         ← API 인터페이스
│   │   ├── indicators.py       ← 기술 지표
│   │   ├── models.py           ← 데이터클래스
│   │   ├── orders.py           ← 주문 관리
│   │   ├── position.py         ← 포지션 관리 (facade)
│   │   ├── position_open.py    ← 포지션 진입
│   │   ├── position_monitor.py ← 포지션 모니터링
│   │   ├── position_close.py   ← 포지션 청산/복구
│   │   ├── signals.py          ← 시그널 탐지
│   │   ├── state.py            ← 상태 저장
│   │   └── utils/              ← 유틸리티
│   │       ├── lock.py
│   │       └── logging_config.py
│   └── *_backup.py             ← 백업 파일들
├── results/
│   ├── engulf_5m_bot_state.json
│   └── engulf_5m_metrics.json
├── logs/engulf_5m_bot_*.log
├── archive/deprecated_bots/
└── *.bat (START, STOP, MONITOR)
```

---

## Standard Research Protocol

> **📋 Full Documentation**: [STANDARD_RESEARCH_PROTOCOL.md](bingx_rl_trading_bot/claudedocs/STANDARD_RESEARCH_PROTOCOL.md)

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
    tp_pct=2.5, sl_pct=2.0,
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
# ❌ 금지: df['col'].shift(-1), df.rolling(n, center=True)
# ✅ 허용: df['col'].shift(1), df.rolling(n).xxx()
```

### Position Mode

```python
params={'positionSide': 'BOTH'}  # One-Way mode
```

---

## Archived Bots

> `archive/deprecated_bots/20260104/` 참조

폐기 사유: WR < 50%, Look-Ahead Bias, 백테스트 방법론 오류 등

---

## MCP Quick Reference

| 작업 | MCP | 예시 |
|-----|-----|------|
| 코드 분석 | Serena | `find_symbol`, `find_referencing_symbols` |
| CCXT 문서 | Context7 | `/ccxt/ccxt` |
| 복잡한 디버깅 | Sequential | `sequentialthinking` |
| 최신 정보 | Tavily | BingX API 변경사항 |
