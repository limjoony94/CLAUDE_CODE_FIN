# OpenClaw 멀티 에이전트 환경 구조 개선안

**작성일**: 2026-02-01

---

## 1. 에이전트별 역할 정의

| 에이전트 | 채널 | 역할 |
|---------|------|------|
| **dev** | #dev | 코드 수정, 전략 연구, 백테스트, 모델 개선 |
| **automation** | #automation | 봇 실행/중지/재시작, 프로세스 관리 |
| **monitor** | #monitor | 트레이딩 성과 모니터링, 알림, 리포트 |

---

## 2. 추가할 스크립트/파일

### 2.1 automation 에이전트용 (`scripts/ops/`)

```bash
scripts/ops/
├── start_bot.sh        # 봇 시작 (tmux 세션)
├── stop_bot.sh         # 봇 중지 (graceful)
├── restart_bot.sh      # 재시작
├── status.sh           # 프로세스 상태 확인
└── health_check.sh     # 헬스체크 (API 연결, 메트릭 요약)
```

**start_bot.sh 예시**:
```bash
#!/bin/bash
cd /home/sp/.openclaw/workspace/CLAUDE_CODE_FIN/bingx_rl_trading_bot
tmux new-session -d -s pattern_5m "python3 scripts/production/pattern_5m_bot.py 2>&1 | tee logs/pattern_5m_bot_$(date +%Y%m%d).log"
echo "Bot started in tmux session 'pattern_5m'"
```

### 2.2 monitor 에이전트용 (`scripts/monitor/`)

```bash
scripts/monitor/
├── metrics_summary.py   # metrics.json → 요약 텍스트
├── daily_report.py      # 일일 성과 리포트 생성
├── alert_check.py       # 이상 징후 감지 (연속 손실, MDD 초과 등)
└── log_tail.sh          # 최근 로그 N줄
```

**메트릭 접근 경로**:
| 데이터 | 경로 | 형식 |
|--------|------|------|
| 봇 상태 | `results/pattern_5m_bot_state.json` | JSON |
| 성과 메트릭 | `results/pattern_5m_metrics.json` | JSON |
| 운영 로그 | `logs/pattern_5m_bot_*.log` | Text |
| 전략 설정 | `config/pattern_5m_config.yaml` | YAML |

### 2.3 dev 에이전트용 (기존 구조 활용)

dev 에이전트는 기존 구조를 그대로 사용:
- `scripts/analysis/` — 연구 스크립트 실행
- `scripts/production/pattern_5m/` — 코드 수정
- `config/` — 설정 변경
- `CLAUDE.md` — 변경사항 기록

---

## 3. 프로젝트 심볼릭 링크

각 에이전트 워크스페이스에서 접근 가능하도록:

```bash
# shared 디렉토리에 프로젝트 링크
ln -sf /home/sp/.openclaw/workspace/CLAUDE_CODE_FIN /home/sp/.openclaw/shared/trading-bot
```

---

## 4. Cron/Heartbeat 설정 제안

| 작업 | 에이전트 | 주기 | 방법 |
|------|---------|------|------|
| 봇 프로세스 확인 | automation | 30분 | heartbeat |
| 성과 리포트 | monitor | 4시간 | cron → #monitor |
| 일일 종합 리포트 | monitor | 매일 09:00 | cron → #monitor |
| MDD 알림 | monitor | 1시간 | heartbeat |

---

## 5. 파일 정리 제안

1. **models/ 정리**: 현재 사용하지 않는 pkl 파일 → `archive/legacy_models/`로 이동
2. **.bat 파일 제거**: WSL 환경이므로 `scripts/ops/` shell 스크립트로 대체
3. **src/ 정리**: 레거시 코드 명시적 아카이브 (`archive/legacy_src/`)
4. **experimental/ 정리**: 완료된 실험 아카이브

---

## 6. 구현 우선순위

| 순위 | 작업 | 난이도 | 효과 |
|------|------|--------|------|
| 1 | `scripts/ops/` 생성 (automation용) | 낮음 | 높음 |
| 2 | `scripts/monitor/metrics_summary.py` | 낮음 | 높음 |
| 3 | 심볼릭 링크 설정 | 낮음 | 중간 |
| 4 | Cron 설정 (monitor 리포트) | 낮음 | 높음 |
| 5 | models/ 정리 | 중간 | 중간 |
| 6 | 레거시 코드 아카이브 | 중간 | 낮음 |
