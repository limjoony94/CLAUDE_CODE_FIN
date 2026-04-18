# bingx_rl_trading_bot — C1 Breakout v2.6

BingX BTC-USDT 15m 채널 돌파 자동 매매 봇.

## 핵심 디렉토리

| 경로 | 설명 |
|------|------|
| `scripts/production/c1_breakout/` | **운영 코드** (4개 모듈) |
| `scripts/production/c1_breakout_bot.py` | 엔트리포인트 |
| `config/c1_breakout_config.yaml` | 전략 파라미터 |
| `scripts/analysis/` | 연구/백테스트 스크립트 |
| `scripts/ops/` | 시작/중지/상태/헬스체크 |
| `results/` | 봇 상태 JSON |
| `logs/` | 운영 로그 (일일 회전) |
| `claudedocs/` | 설계 문서, 연구 보고서 |

## 현재 전략

- **C1 Breakout v2.6**: 15m Channel Breakout + Fractal SL + ATR Trailing TP
- **PnL**: +169.5% (additive 1x, 333일), **WR**: 36.6%, **R:R**: 3.36
- 상세: [CLAUDE.md](../CLAUDE.md)

## 테스트 스위트

- **113 pytest cases** (~5s) — `scripts/tests/`
- **Coverage**: 71% 전체, `signals.py`/`indicators.py`/`config.py` **100%**
- 각 BUG fix마다 identity regression guard
- 실행: `python -m pytest scripts/tests/ -v`

## 핵심 문서

- [claudedocs/c1_breakout_v2_design.md](claudedocs/c1_breakout_v2_design.md) — 전략 설계
- [claudedocs/BUG_HISTORY.md](claudedocs/BUG_HISTORY.md) — BUG#1~65 연대기
- [claudedocs/BACKTEST_LIVE_PARITY.md](claudedocs/BACKTEST_LIVE_PARITY.md) — 백테스트-라이브 정합성 (20/22)
- [claudedocs/STANDARD_RESEARCH_PROTOCOL.md](claudedocs/STANDARD_RESEARCH_PROTOCOL.md) — 연구 프로토콜

## 의존성

- Python 3.12+
- ccxt, pyyaml, requests, numpy, pytest, hypothesis
