# 에이전트별 작업 가이드

> **Version**: v1.56.2 | **Updated**: 2026-03-12

---

## 🔧 dev 에이전트 가이드

### 작업 범위
- 전략 코드 수정 (`scripts/production/pattern_5m/`)
- 연구/백테스트 (`scripts/analysis/`)
- 설정 변경 (`config/pattern_5m_config.yaml`, `pattern_5m/constants.py`)
- Scanner 실행/업데이트 (`scripts/scanner/pattern_scanner.py`)
- CLAUDE.md 업데이트

### 핵심 파일
```
bingx_rl_trading_bot/
├── scripts/production/pattern_5m/   # 운영 코드 (14개 모듈)
├── scripts/scanner/                 # Dynamic WF Pattern Scanner CLI (v2.4)
├── scripts/analysis/                # 연구 스크립트
├── config/pattern_5m_config.yaml    # 전략 설정
├── data/btc_5m_270days_reclassified.csv  # 백테스트 데이터 (303일, Ground Truth, 파일명 레거시)
└── results/                         # 봇 상태/메트릭 JSON + dynamic_patterns.json
```

### 작업 프로토콜
1. **코드 수정 전** CLAUDE.md의 Standard Research Protocol 숙지
2. **백테스트 필수**: MC sign randomization (10k sims), WF 3-fold 검증
3. **Look-Ahead Bias 금지**: `shift(-1)`, `rolling(center=True)` 사용 금지
4. **버전 관리**: 변경 시 CLAUDE.md Version History + docs/VERSION_HISTORY.md 업데이트
5. **커밋 메시지**: `feat(vX.XX.X): 간단한 설명`
6. **테스트 필수**: production 파일 변경 시 `pytest scripts/tests/ -x -q` 실행 (1061+ tests)

### 현재 전략 파라미터 (v1.56.2)
- **패턴**: **131개** (59L+72S), MAE/MFE + ATR scanner v2.4 + Neutral window ±1%
- **TP/SL**: Per-pattern ATR-scaled (TP 0.85-2.80%, SL 1.44-5.95%)
- **레버리지**: Fixed 3x (Adaptive 비활성화)
- **Max Positions**: 9 (virtual slots, 1/N=11.1% sizing, mixed-direction in Hedge)
- **Position Mode**: Hedge (LONG/SHORT 독립)
- **Direction Cap**: 7 (max same-direction)
- **Timeout**: 288 bars (24h)
- **리스크**: 일일 손실 **13%**, Aggregate risk cap (counter 8%/with 15%)
- **Quality Filter**: Edge>=18pp + WR>=60% + SL>=1.0% + MC<0.01 + min_trades>=25 + Holdout 7d
- **Cascade SL**: SL 피격 시 동일 방향 SL 거리 ×0.15 (85% 축소)
- **Momentum Guard**: BTC >1.5%/15min → 역방향 진입 1h 차단
- **DISABLED**: Regime Sizing, Adaptive Leverage, Equity Curve, Correlation-Aware, Loss Burst Brake

### Scanner CLI 사용법 (v2.4)
```bash
cd bingx_rl_trading_bot
# 기본 (neutral + ATR + N-pos, v1.38.1~ default)
python scripts/scanner/pattern_scanner.py --discovery-method mae_mfe --edge-threshold 18 --wf-folds 3 --holdout-days 7
# Legacy 1-pos 모드 (빠른 반복용)
python scripts/scanner/pattern_scanner.py --discovery-method mae_mfe --edge-threshold 18 --wf-folds 3 --holdout-days 7 --no-npos
# 주요 옵션: --no-neutral, --no-atr, --neutral-tol 2.0, --atr-clamp-lo 0.5 --atr-clamp-hi 1.5, --n-slots 5 --direction-cap 4
```

---

## 🤖 automation 에이전트 가이드

### 작업 범위
- 봇 프로세스 시작/중지/재시작
- 프로세스 상태 모니터링
- 크래시 복구

### 핵심 명령
```bash
cd bingx_rl_trading_bot

# 봇 시작 (tmux)
tmux new-session -d -s pattern_5m "python3 scripts/production/pattern_5m_bot.py"

# 봇 상태 확인
tmux list-sessions | grep pattern_5m

# 봇 중지 (열린 포지션 먼저 확인!)
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
- 봇 재시작 시 기존 포지션 TP/SL 자동 조정 (v1.17)
- `dynamic_patterns.json` 90일 초과 시 WARNING 출력 (Scan Staleness, v1.34.0)

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
| 연속 손실 | state.json → consecutive_losses | ≥ 3회 |
| 일일 손실 | state.json → daily_pnl | ≤ -13% (v1.28.5: 자동 중단) |
| MDD | metrics.json → max_drawdown | ≥ 25% |
| API 에러 | 로그 grep ERROR | ≥ 10회/시간 |
| WR 이탈 | state.json → winning_trades/total_trades | < 50% (20trades) |
| 기대 WR | EXPECTED_WIN_RATE=67.4 (v1.56.1 clean TP+SL, 129t) | 실제 WR과 비교 |
| LONG WR | 별도 추적 필요 | LONG WR 0% (03-05~08, 15t) — BTC 하락 레짐 편향, 소표본 |

### Clean Baseline (post-03-05)
- `results/pattern_5m_baseline_post0305.json` — pre-03-05 오염 데이터 제외한 정확한 기대치
- post-fix WR: 83.0% (53t)

### 리포트 포맷
```
📊 Pattern 5m v1.56.2 성과 리포트 (YYYY-MM-DD HH:MM)
━━━━━━━━━━━━━━━━━━━━
• 상태: ✅ 운영중 / ❌ 중단
• 오늘 거래: N건 (W승 L패)
• 오늘 PnL: +X.XX%
• 누적 WR: XX.X% (N trades) | 기대: 67.4%
• 연속손실: N회 / 일일손실: X.X%
• 열린 포지션: LONG/SHORT [패턴명] @ $XX,XXX (N/9 slots)
• MDD: X.X% (limit 25%)
```
