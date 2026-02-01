# bingx_rl_trading_bot — 메인 프로젝트 패키지

BingX 거래소 BTC-USDT 선물 자동 매매 봇. 5분봉 캔들 패턴 기반.

## 핵심 디렉토리

| 경로 | 설명 |
|------|------|
| `scripts/production/pattern_5m/` | **운영 코드** (14개 모듈) |
| `scripts/analysis/` | 연구/백테스트 스크립트 |
| `scripts/monitoring/` | 모니터링 스크립트 |
| `config/` | 설정 파일 (전략, API) |
| `data/` | 시장 데이터 CSV |
| `results/` | 봇 상태/메트릭 JSON |
| `logs/` | 운영 로그 |
| `claudedocs/` | 연구 리포트 |
| `archive/` | 레거시 아카이브 |

## 현재 전략

- **v1.22.0**: 12 패턴 (7L+5S), Per-pattern TP/SL
- **WR**: 80.3%, **PF**: 3.36, **WF**: 5/5
- 상세: [CLAUDE.md](../CLAUDE.md)

## 의존성

- Python 3.12 (WSL2 Ubuntu)
- ccxt, pandas, numpy, pyyaml
