# Tech Stack - CLAUDE_CODE_FIN

**Last Updated**: 2026-02-12 | **Bot Version**: v1.27.3

---

## Programming Language
- **Python 3.9+**: 전체 프로젝트 기본 언어

## Trading & Exchange
- **CCXT**: 암호화폐 거래소 통합 라이브러리 (BingX API)
- **PyYAML**: 설정 파일 관리 (`config/*.yaml`)

## Data Analysis
- **pandas**: 데이터 처리 및 백테스트
- **numpy**: 수치 연산, 통계 계산
- **scipy**: Monte Carlo 시뮬레이션, 통계 검정

## Technical Indicators
- **ta**: 기술적 지표 라이브러리 (RSI, ATR 등)

## Logging & Monitoring
- **loguru**: 구조화된 로깅

## Visualization (분석용)
- **matplotlib**: 데이터 시각화 (연구 스크립트)
- **plotly**: 인터랙티브 차트 (연구 스크립트)

## Development Tools
- **Git**: 버전 관리
- **Claude Code**: AI 지원 개발 도구
- **tmux**: 봇 프로세스 관리

## Project Structure
```
bingx_rl_trading_bot/
├── scripts/production/pattern_5m/  # 프로덕션 봇 (14 모듈)
├── scripts/scanner/               # Dynamic WF Pattern Scanner CLI
├── scripts/analysis/               # 연구 스크립트 (45+)
├── scripts/data/                   # 데이터 수집
├── scripts/utils/                  # 유틸리티
├── config/                         # 설정 (YAML)
├── data/                           # 시장 데이터 (CSV)
├── results/                        # 봇 상태/메트릭 (JSON)
├── logs/                           # 운영 로그
└── claudedocs/                     # 연구 문서
```

## Archived (미사용)
> 아래 도구들은 `archive/`에 보관된 레거시 코드에서만 참조됩니다.
- XGBoost, Stable-Baselines3, PyTorch, Gymnasium — ML/RL 시대 (2025년)
- aiohttp, websockets — 비동기 봇 시대
