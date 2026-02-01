# 에이전트별 작업 가이드

---

## 🔧 dev 에이전트 가이드

### 작업 범위
- 전략 코드 수정 (`scripts/production/pattern_5m/`)
- 연구/백테스트 (`scripts/analysis/`)
- 설정 변경 (`config/pattern_5m_config.yaml`, `pattern_5m/constants.py`)
- CLAUDE.md 업데이트

### 핵심 파일
```
bingx_rl_trading_bot/
├── scripts/production/pattern_5m/   # 운영 코드 (14개 모듈)
├── scripts/analysis/                # 연구 스크립트
├── config/pattern_5m_config.yaml    # 전략 설정
└── data/btc_5m_270days.csv         # 백테스트 데이터
```

### 작업 프로토콜
1. **코드 수정 전** CLAUDE.md의 Standard Research Protocol 숙지
2. **백테스트 필수**: MC test (10k sims), WF 5-fold 검증
3. **Look-Ahead Bias 금지**: `shift(-1)`, `rolling(center=True)` 사용 금지
4. **버전 관리**: 변경 시 CLAUDE.md Version History 업데이트
5. **커밋 메시지**: `v1.XX.X: 간단한 설명`

### 자주 쓰는 명령
```bash
cd /home/sp/.openclaw/workspace/CLAUDE_CODE_FIN/bingx_rl_trading_bot
python3 scripts/analysis/overfitting_diagnosis.py    # 과적합 진단
python3 scripts/analysis/per_pattern_backtest.py     # 패턴별 백테스트
```

---

## 🤖 automation 에이전트 가이드

### 작업 범위
- 봇 프로세스 시작/중지/재시작
- 프로세스 상태 모니터링
- 크래시 복구

### 핵심 명령
```bash
# 봇 시작 (tmux)
cd /home/sp/.openclaw/workspace/CLAUDE_CODE_FIN/bingx_rl_trading_bot
tmux new-session -d -s pattern_5m "python3 scripts/production/pattern_5m_bot.py"

# 봇 상태 확인
tmux list-sessions | grep pattern_5m
ps aux | grep pattern_5m_bot

# 봇 중지
tmux send-keys -t pattern_5m C-c

# 로그 확인
tail -50 logs/pattern_5m_bot_*.log
```

### 주의사항
- 봇 중지 시 **열린 포지션 확인** 필수 (`results/pattern_5m_bot_state.json`)
- 재시작 시 봇이 자동으로 orphan position 복구 (Crash Recovery 기능)
- API 키 파일 (`config/api_keys.yaml`) 절대 수정/노출 금지

---

## 📊 monitor 에이전트 가이드

### 작업 범위
- 트레이딩 성과 모니터링
- 이상 징후 감지 및 알림
- 일일/주간 리포트 생성

### 메트릭 접근
```bash
# 봇 상태 확인
cat results/pattern_5m_bot_state.json | jq .

# 성과 메트릭
cat results/pattern_5m_metrics.json | jq .

# 최근 로그
tail -100 logs/pattern_5m_bot_*.log | grep -E "(TRADE|PROFIT|LOSS|ERROR)"
```

### 모니터링 항목

| 항목 | 확인 방법 | 알림 기준 |
|------|----------|----------|
| 봇 생존 | `ps aux \| grep pattern_5m` | 프로세스 없음 |
| 연속 손실 | metrics.json → consecutive_losses | ≥ 5회 |
| 일일 손실 | metrics.json → daily_pnl | ≤ -5% |
| MDD | metrics.json → max_drawdown | ≥ 25% |
| API 에러 | 로그 grep ERROR | ≥ 10회/시간 |
| WR 이탈 | metrics.json → win_rate | < 65% (20trades) |

### 리포트 포맷 (Discord #monitor)
```
📊 Pattern 5m 성과 리포트 (YYYY-MM-DD HH:MM)
━━━━━━━━━━━━━━━━━━━━
• 상태: ✅ 운영중 / ❌ 중단
• 오늘 거래: N건 (W승 L패)
• 오늘 PnL: +X.XX%
• 누적 WR: XX.X% (N trades)
• 현재 MDD: X.X%
• 열린 포지션: LONG/SHORT @ $XX,XXX
```
