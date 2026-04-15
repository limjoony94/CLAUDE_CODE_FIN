# CLAUDE_CODE_FIN — C1 Breakout v2.6 BTC Trading Bot

BingX 거래소 BTC-USDT 15분봉 채널 돌파 전략 자동 매매 봇.

## 현재 운영

- **전략**: C1 Breakout v2.6 (15m Channel Breakout + Fractal SL + ATR Trailing TP)
- **성과**: PnL +169.5% (additive 1x, 333일), WR 36.6%, R:R 3.36
- **검증**: MC p=0.000 DISC, WF 5/5 PASS, 3-Way ALL PASS
- **포지션**: N=1, Exchange 10x / Trading 3x

## 빠른 시작

### 요구사항
- Python 3.12+
- BingX API 키 (`config/api_keys.yaml`)

### 설치
```bash
pip install -r requirements.txt
```

### 실행 (Windows)
```powershell
# 봇 시작
Start-Process -FilePath 'python' -ArgumentList 'scripts/production/c1_breakout_bot.py' -WindowStyle Hidden -WorkingDirectory 'bingx_rl_trading_bot'

# 상태 확인
Get-WmiObject Win32_Process -Filter "Name='python.exe' AND CommandLine LIKE '%c1_breakout%'" | Select-Object ProcessId
```

### 모니터링
```bash
# 상태
cat results/c1_breakout_state.json | python -m json.tool

# 로그
tail -f logs/c1_breakout.log
```

## 프로젝트 구조

```
bingx_rl_trading_bot/
├── scripts/production/
│   ├── c1_breakout_bot.py        # 엔트리포인트 (lock, 로깅)
│   └── c1_breakout/              # 봇 모듈
│       ├── bot.py                # 메인 루프, exchange, state
│       ├── signals.py            # 채널 돌파, 프랙탈 SL, 트레일 TP
│       ├── indicators.py         # ATR, 채널, 프랙탈 스윙
│       └── config.py             # 설정 로딩
├── config/
│   ├── c1_breakout_config.yaml   # 전략 파라미터 (유일한 설정 소스)
│   └── api_keys.yaml             # BingX API 키 (gitignored)
├── scripts/analysis/             # 연구/검증 스크립트
├── scripts/ops/                  # 시작/중지/상태/헬스체크
├── results/                      # 봇 상태, 검증 결과
├── logs/                         # c1_breakout.log (일일 회전)
├── claudedocs/                   # 설계 문서, 연구 보고서
└── archive/                      # 레거시 봇 (Pattern 5m 등)
```

## 문서

- [CLAUDE.md](CLAUDE.md) — 전략 상세, 검증 결과, 연구 프로토콜
- [AGENTS.md](AGENTS.md) — 에이전트 규칙
- [claudedocs/c1_breakout_v2_design.md](bingx_rl_trading_bot/claudedocs/c1_breakout_v2_design.md) — 설계 문서
