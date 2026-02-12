# 에이전트별 작업 가이드

> **Version**: v1.27.3 | **Updated**: 2026-02-12

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
├── scripts/scanner/                 # Dynamic WF Pattern Scanner CLI
├── scripts/analysis/                # 연구 스크립트 (45+)
├── config/pattern_5m_config.yaml    # 전략 설정
├── data/btc_5m_270days_reclassified.csv  # 백테스트 데이터 (270일, Ground Truth)
└── results/                         # 봇 상태/메트릭 JSON + dynamic_patterns.json
```

### 작업 프로토콜
1. **코드 수정 전** CLAUDE.md의 Standard Research Protocol 숙지
2. **백테스트 필수**: MC sign randomization (10k sims), WF 5-fold 검증
3. **Look-Ahead Bias 금지**: `shift(-1)`, `rolling(center=True)` 사용 금지
4. **버전 관리**: 변경 시 CLAUDE.md Version History 업데이트
5. **커밋 메시지**: `feat(vX.XX.X): 간단한 설명`

### 현재 전략 파라미터 (v1.27.3)
- **패턴**: 51개 (32L+19S), Uniform TP 70% + Legacy reopt + Low-WR review
- **TP/SL**: Per-pattern 최적화 (v1.27.1 legacy reopt 포함) / Dynamic 모드: Universal TP 2.1/SL 3.0
- **레버리지**: 3x
- **리스크**: 일일 손실 **10%** (v1.28.0), 연속 3패 → 600초 pause
- **Pattern Source**: `static` (constants.py) 또는 `dynamic` (results/dynamic_patterns.json)

### 자주 쓰는 연구 스크립트
```bash
cd C:/Users/J/OneDrive/CLAUDE_CODE_FIN/bingx_rl_trading_bot
python scripts/analysis/uniform_tp_validation.py        # Uniform TP 검증
python scripts/analysis/tp_sl_optimization_v1264.py     # TP/SL 최적화
python scripts/analysis/tp_sl_deep_validation.py        # 심층 검증
python scripts/analysis/distance_edge_decomposition.py  # Edge 분해
python scripts/analysis/context_filter_research_v2.py   # Context filter 연구 (FAIL)
python scripts/analysis/portfolio_pruning_v4.py         # 포트폴리오 프루닝
python scripts/analysis/full_270d_revalidation.py       # 270일 전수 검증
python scripts/scanner/pattern_scanner.py               # Dynamic WF Pattern Selection
```

---

## 🤖 automation 에이전트 가이드

### 작업 범위
- 봇 프로세스 시작/중지/재시작
- 프로세스 상태 모니터링
- 크래시 복구

### 핵심 명령
```bash
cd C:/Users/J/OneDrive/CLAUDE_CODE_FIN/bingx_rl_trading_bot

# 봇 시작 (tmux)
tmux new-session -d -s pattern_5m "python3 scripts/production/pattern_5m_bot.py"

# 봇 상태 확인
tmux list-sessions | grep pattern_5m

# 봇 중지
tmux send-keys -t pattern_5m C-c

# 로그 확인
tail -50 logs/pattern_5m_bot_*.log
```

### 유틸리티 스크립트
```bash
python scripts/utils/check_status.py      # 봇 상태 확인
python scripts/utils/monitor_simple.py     # 간단 모니터링
python scripts/utils/get_current_balance.py # 잔고 조회
python scripts/utils/verify_state.py       # 상태 검증
python scripts/utils/stop_bot.py           # 봇 안전 중지
python scripts/utils/safe_reset.py         # 안전 리셋
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
| 봇 생존 | `tmux list-sessions` | 프로세스 없음 |
| 연속 손실 | state.json → consecutive_losses | ≥ 3회 (v1.27.0: pause 발동) |
| 일일 손실 | state.json → daily_pnl | ≤ -10% (v1.28.0: 자동 중단) |
| MDD | metrics.json → max_drawdown | ≥ 20% |
| API 에러 | 로그 grep ERROR | ≥ 10회/시간 |
| WR 이탈 | state.json → winning_trades/total_trades | < 60% (20trades) |
| 기대 WR | EXPECTED_WIN_RATE=68.0 (v1.27.3) | 실제 WR과 비교 |

### 리포트 포맷
```
📊 Pattern 5m v1.27.3 성과 리포트 (YYYY-MM-DD HH:MM)
━━━━━━━━━━━━━━━━━━━━
• 상태: ✅ 운영중 / ❌ 중단
• 오늘 거래: N건 (W승 L패)
• 오늘 PnL: +X.XX%
• 누적 WR: XX.X% (N trades) | 기대: 68.0%
• 연속손실: N회 / 일일손실: X.X%
• 열린 포지션: LONG/SHORT [패턴명] @ $XX,XXX
```
