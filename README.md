# CLAUDE_CODE_FIN — BTC Pattern Trading Bot

BingX 거래소 BTC-USDT 선물 자동 매매 봇. 5분봉 캔들 패턴 기반 전략.

## 현재 운영

- **전략**: Pattern 5m v1.22.0
- **패턴**: 12개 (7 Long + 5 Short), 3-캔들 조합
- **성과**: WR 80.3%, PF 3.36, WF 5/5 (270일 백테스트)

## 빠른 시작

### 요구사항
- Python 3.12+
- BingX API 키 (`config/api_keys.yaml`)

### 설치
```bash
cd bingx_rl_trading_bot
pip install ccxt pandas numpy pyyaml
```

### 실행
```bash
# 봇 시작
python3 scripts/production/pattern_5m_bot.py

# tmux 백그라운드 실행
tmux new-session -d -s pattern_5m "python3 scripts/production/pattern_5m_bot.py"
```

### 모니터링
```bash
# 상태 확인
cat results/pattern_5m_bot_state.json | jq .
cat results/pattern_5m_metrics.json | jq .

# 로그
tail -f logs/pattern_5m_bot_*.log
```

## 프로젝트 구조

```
bingx_rl_trading_bot/
├── config/                    # 설정 (API 키, 전략 파라미터)
├── scripts/production/        # 운영 코드
│   ├── pattern_5m_bot.py      # 엔트리포인트
│   └── pattern_5m/            # 14개 모듈 패키지
├── scripts/analysis/          # 연구/백테스트 스크립트
├── data/                      # 시장 데이터
├── results/                   # 상태/메트릭 JSON
└── logs/                      # 운영 로그
```

## 문서

- [CLAUDE.md](CLAUDE.md) — 전략 상세, 버전 히스토리, 연구 프로토콜
- [docs/analysis.md](docs/analysis.md) — 프로젝트 분석
- [docs/restructure-plan.md](docs/restructure-plan.md) — OpenClaw 구조 개선안
- [docs/agent-guides.md](docs/agent-guides.md) — 에이전트별 작업 가이드
- [docs/TECH_STACK.md](docs/TECH_STACK.md) — 기술 스택
- [docs/CODING_CONVENTIONS.md](docs/CODING_CONVENTIONS.md) — 코딩 컨벤션

## OpenClaw 에이전트 연동

| 에이전트 | 역할 | 채널 |
|---------|------|------|
| dev | 코드 수정, 연구, 백테스트 | #dev |
| automation | 봇 실행/중지/재시작 | #automation |
| monitor | 성과 모니터링, 알림 | #monitor |
