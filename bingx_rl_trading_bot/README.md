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

## 의존성

- Python 3.12+
- ccxt, pyyaml, requests, numpy
