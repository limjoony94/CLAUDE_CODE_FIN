# CLAUDE_CODE_FIN - C1 Breakout v2.6 15m BTC 트레이딩 봇

> **Version**: v4.7.4 | **Bot**: C1 Breakout v2.6 (15m Channel Breakout + Fractal SL + Trail TP, N=1, Exch 10x / Trade 3x) | **Updated**: 2026-04-17
>
> **v4.7.4 (2026-04-17)**: 5차 Cycle — 복합 flow + property-based 테스트.
>   - `test_exchange_open.py` (7): MARKET retry (BUG#38), SL placement (BUG#28),
>     emergency close (BUG#26), partial fill (BUG#55)
>   - `test_sync_exchange.py` (8): orphan adoption 실제 flow (BUG#48 end-to-end),
>     ghost orderType 분류 (BUG#50), entry_time filter (BUG#36), exchange timestamp (BUG#45)
>   - `test_trail_update.py` (8): force_reset (BUG#35), LOOSEN-only (BUG#46),
>     failure streak (BUG#59)
>   - `test_properties.py` (6): hypothesis로 ~400 랜덤 케이스 자동 탐색.
>     ATR 인과성, signal 불변량, exit priority, timeout 보장
>   - **87 pytest cases PASS** (~17s, hypothesis 포함)
> **v4.7.3 (2026-04-17)**: 4차 Cycle — Test suite 영구화.
>   - `scripts/tests/` 신설: conftest.py + 4 test modules (test_indicators/signals/config/bot)
>   - **58 pytest cases PASS** (~1s). 각 케이스 = 특정 비판 각도(A엣지/B parity/C상호작용/D롤백) 코드화
>   - BUG#48/52/53/54/56/57/58/60 identity regression guard 확보
>   - run-tests.md 갱신: quick-start, deprecation-strict, bug→test 매핑
> **v4.7.2 (2026-04-17)**: 3차 4-Cycle 비판 평가. BUG#59~61 (3건) 추가.
>   - **BUG#59**: _update_exchange_trail 연속 실패 감지 (streak ≥3 → 보호 공백 경고)
>   - **BUG#60**: check_exit trail path current_close ≤ 0 / NaN 방어
>   - **BUG#61**: TimeSyncBingX offset ±60초 clamp (거래소 비정상 응답 차단)
> **v4.7.1 (2026-04-17)**: 2차 6-Cycle 비판 평가. BUG#54~58 (5건) 추가 수정.
>   - **BUG#54**: bars_since_last_exit wall-clock 보정 (last_exit_time 저장, 재시작 시 elapsed_bars 계산)
>   - **BUG#55**: MARKET 부분 체결 감지 경고 (shortfall > 1%)
>   - **BUG#56**: trade_history in-memory cap (1000→500)
>   - **BUG#57**: datetime.utcnow deprecation 대응 (_utc_now 헬퍼)
>   - **BUG#58**: state.json I/O 예외 방어 (OneDrive sync lock 대응)
> **v4.7.0 (2026-04-17)**: Opus 4.7 코드 리뷰. 6-Cycle 다각도 비판 평가. BUG#48~53 (6건) 수정.
>   - **BUG#48 [CRITICAL]**: Orphan 채택 시 거래소 실제 SL 복원 (3% fallback만 하던 것 → 실제 STOP 주문 조회)
>   - **BUG#49**: Fill price 기준 sl_pct 재검증 경고
>   - **BUG#50**: Ghost exit reason을 trade info.orderType로 1차 분류 (stale best_price 의존성 축소)
>   - **BUG#51**: 연속 candle fetch 실패 감지 → tighten 로직 공백 경고
>   - **BUG#52**: config load 시 trading_leverage > leverage 관계 검증
>   - **BUG#53**: channel_high <= channel_low sanity check
> **v4.6.0**: 49-Cycle 하네스 감사. 39건 수정. Pattern 5m 코드 완전 삭제. Lock 메커니즘 추가. 전략 변경 체크리스트 추가.
>
> **v2.6**: 20 Cycle 비판 평가. BUG#35~42 (8건) 수정. Trail 90%→0.9% CRITICAL fix. SL ID sync. 봇 재시작 완료.
> **v2.5**: 30 Cycle 비판 평가. SL-first 우선순위, Exchange trail=백테스트 수학 통일, lookback=10 통일
> **v2.3**: 21 Cycle 비판 평가로 16건 버그 수정
> **이전 봇**: MAVS-15 → 2026-04-12 WF 3/5 퇴화, C1에 의해 교체
> **모든 레거시 봇 폐기 완료** (archive/legacy_bots/)

---

## ⚡ 빠른 참조 (C1 Breakout v2 — ACTIVE)

| 항목      | 경로                                                                             |
| --------- | -------------------------------------------------------------------------------- |
| 엔트리    | `scripts/production/c1_breakout_bot.py`                                          |
| 모듈      | `scripts/production/c1_breakout/` (signals.py, bot.py, indicators.py, config.py) |
| 설정      | `config/c1_breakout_config.yaml`                                                 |
| 상태      | `results/c1_breakout_state.json`                                                 |
| 로그      | `logs/c1_breakout.log` (일일 자동 회전, 30일 보관)                               |
| 설계 문서 | `claudedocs/c1_breakout_v2_design.md`                                            |

### C1 Breakout v2 핵심

| 항목        | 값                                                      |
| ----------- | ------------------------------------------------------- |
| 전략        | 15m Channel Breakout + Fractal SL + ATR Trailing TP     |
| 진입        | close > 15봉 최고가 AND body > 40% of range → 돌파 방향 |
| SL          | 프랙탈 스윙 포인트 (최대 3.3×ATR 캡, 동적)              |
| TP          | 트레일링 best_price - 2.5×ATR (동적)                    |
| Emergency   | 3.0% hard SL                                            |
| Timeout     | 48h (192 bars)                                          |
| Leverage    | Exchange 10x / **Trading 3x**                           |
| 자산        | BTC/USDT (단일)                                         |
| 포지션      | **N=1**, One-Way 모드 (positionSide=BOTH)               |
| Exchange SL | STOP_MARKET @ fractal SL (crash protection)            |
| Exchange TP | TRAILING_STOP_MARKET @ ATR callback % (trailingPercent, 15분 갱신) |
| **검증**    | MC p=0.000 DISC, WF 5/5 PASS, 3-Way ALL PASS            |
| **리스크**  | Halt 없음 — SL/Trail/Emergency만 적용                    |

### 검증 결과 (v2.5, N=1, 333일 BTC 백테스트)

| 지표          | Additive 1x | Compound 1x | 기준                   |
| ------------- | ----------- | ----------- | ---------------------- |
| PnL           | **+169.5%** | +417.9%     | ✅ 양수                |
| MDD           | **5.4%**    | -           | ✅ 낮음                |
| WR            | **36.6%**   | -           | R:R 3.36으로 보상      |
| R:R           | **3.36**    | -           | ✅ >1.0                |
| Daily         | **+0.509%** | -           | ✅ >0.2%               |
| Trades/day    | **3.1**     | -           | ✅ ≥2.0               |
| MC Direction  | **p=0.000** | -           | ✅ DISC                |

### 심화 검증 (v2.5, 2026-04-13, additive 1x)

| 검증                     | 결과                                                |
| ------------------------ | --------------------------------------------------- |
| Look-ahead Progressive   | ✅ **10/10 PASS** (diff=0 전부)                     |
| Indicator causality      | ✅ ATR/Channel/Fractal 전부 causal                  |
| MC Direction (999 sims)  | ✅ p=0.0000 DISC                                    |
| WF 5-fold                | ✅ **5/5 PASS** (OOS total: +153.9%)                |
| 3-Way Split              | ✅ Train +61%, Valid +54%, **Test +55%** ALL PASS   |
| Param grid (60 combos)   | ✅ **60/60 양수** (파라미터 고원)                   |
| Rolling 60d              | ✅ **5/5 양수**                                     |
| Bootstrap 95% CI         | ✅ [+109%, +234%] (0 미포함)                        |
| Regime (High/Low vol)    | ✅ 둘 다 양수 (레짐 견고)                          |
| Purged CV 5-fold         | ✅ **5/5 PASS** (mean OOS +33.5%)                  |
| Top-20 제거              | ✅ +75.7% (양수 유지)                              |

### 전략 프로파일 (v2.5, 1028 trades, 333일)

| 항목            | 값                  |
| --------------- | ------------------- |
| LONG/SHORT 비율 | 50% / 50%           |
| Exit: TRAIL_TP  | 879회 (85.5%)       |
| Exit: SL        | 149회 (14.5%)       |
| Exit: EMERGENCY | 0회                 |
| Exit: TIMEOUT   | 0회                 |
| 최대 연속 손실  | 13회                |
| MDD (additive)  | 5.4%                |

> **이전 봇 MAVS-15**: WF 3/5 퇴화, 3-way Test 음수 → C1에 의해 교체
> **레거시 봇 전부 폐기**: `archive/legacy_bots/`

---

## 🤖 Auto-Trigger Rules (Claude 자율 판단 기준)

Claude는 사용자 의도를 감지하여 아래 규칙에 따라 **자동으로** 적절한 도구를 선택한다.

### Intent → Command 자동 매핑

| 사용자 의도 (키워드)                    | 자동 실행                                         | 비고                      |
| --------------------------------------- | ------------------------------------------------- | ------------------------- |
| "봇 상태", "봇 확인", "살아있어?"       | `/bot-status`                                     | 프로세스+메트릭+로그 종합 |
| "성과", "실적", "수익률", "얼마 벌었어" | `/check-live`                                     | 기대치 대비 성과 분석     |
| "일일 보고", "오늘 어때", "daily"       | `/daily-report`                                   | 일일 성과 리포트 생성     |
| "패턴 스캔", "재스캔", "새 패턴"        | `/scan-patterns`                                  | C1은 고정 파라미터 (DEPRECATED) |
| "연구", "가설", "백테스트", "분석해줘"  | `/research-template` + `trading-researcher` agent | 연구 프로토콜 강제        |
| "테스트", "tests"                       | `/run-tests`                                      | 모듈 import + 구문 검증   |
| "WF 검증", "walk-forward", "OOS"        | `/wf-validate`                                    | Expanding window WF       |
| "배포", "적용", "deploy"                | `/deploy-patterns`                                | 안전 배포 체크리스트      |
| "문제", "에러", "왜 안돼", "이상해"     | `/diagnose` + `root-cause-analyst` agent          | 종합 진단                 |
| "긴급", "중지", "emergency", "멈춰"     | `/emergency-stop`                                 | 긴급 정지 프로시저        |
| "리스크", "위험", "MDD", "drawdown"     | `trading-risk` agent                              | 리스크 평가               |

### Intent → Agent 자동 선택

| 작업 유형               | 에이전트               | 선택 이유                       |
| ----------------------- | ---------------------- | ------------------------------- |
| 연구 스크립트 작성/실행 | `trading-researcher`   | Standard Research Protocol 강제 |
| 봇 모니터링/성과 분석   | `trading-monitor`      | 기대치 대비 분석 로직 내장      |
| 리스크 평가/전략 안전성 | `trading-risk`         | MDD, WF, MC 전문                |
| 코드 품질/리팩토링      | `quality-engineer`     | 테스트+코드 품질                |
| 디버깅/장애 분석        | `root-cause-analyst`   | 체계적 원인 분석                |
| 성능 최적화             | `performance-engineer` | 프로파일링 기반                 |

### 자동 행동 규칙

1. **코드 변경 후** → 자동으로 `/run-tests` 제안 (production 파일 변경 시 필수)
2. **전략 파라미터 변경 제안 시** → 자동으로 `/wf-validate` 제안
3. **연구 스크립트 작성 시** → `trading-researcher` agent 사용 + 연구 프로토콜 검증
4. **git commit 후 production 파일 포함 시** → CLAUDE.md Version History 업데이트 제안
5. **비정상 결과 감지 시** (PnL > 5000%, WF 전부 FAIL) → 자동 경고 + 원인 분석 제안
6. **새 세션 시작 시** → Serena 메모리 확인 (`ccxt_bingx_pitfalls`, `research_protocol_standard`)

### Serena MCP 자동 활용

| 상황           | Serena 액션                                                         |
| -------------- | ------------------------------------------------------------------- |
| 세션 시작      | `activate_project` → `check_onboarding` → 관련 메모리 읽기          |
| 코드 탐색 요청 | `find_symbol` / `get_symbols_overview`                              |
| 함수 수정      | `find_referencing_symbols` → 영향 범위 확인 → `replace_symbol_body` |
| 중요 발견      | `write_memory` (다음 세션 활용)                                     |
| 연구 시작      | `read_memory("research_protocol_standard")`                         |
| 디버깅 시작    | `read_memory("common_pitfalls_and_lessons")`                        |

---

## 🎯 에이전트별 가이드

### dev — 코드/전략/연구

- **수정 대상**: `scripts/production/c1_breakout/`, `config/c1_breakout_config.yaml`
- **프로토콜**: 아래 Standard Research Protocol 반드시 준수
- **변경 후**: CLAUDE.md Version History 업데이트 + git commit

### automation — 봇 운영 (Windows)

- **시작**: `powershell -Command "Start-Process -FilePath 'python' -ArgumentList 'scripts/production/c1_breakout_bot.py' -WindowStyle Hidden -WorkingDirectory 'C:\Users\J\OneDrive\CLAUDE_CODE_FIN\bingx_rl_trading_bot'"`
- **상태**: `powershell -Command "Get-WmiObject Win32_Process -Filter \"Name='python.exe' AND CommandLine LIKE '%c1_breakout%'\" | Select-Object ProcessId"`
- **로그**: `tail -50 logs/c1_breakout.log`
- **중지**: `powershell -Command "Get-WmiObject Win32_Process -Filter \"Name='python.exe' AND CommandLine LIKE '%c1_breakout%'\" | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"`
- **복구**: 봇 재시작 시 state.json에서 포지션 복원 + orphan 자동 채택

### monitor — 성과 모니터링

- **상태**: `cat results/c1_breakout_state.json | python -m json.tool`
- **로그**: `tail -100 logs/c1_breakout.log | grep -E "(ENTRY|EXIT|PnL|ERROR|HALT|GHOST|HOURLY)"`
- **Halt**: 없음 — SL/Trail/Emergency만 적용
- **기대치**: WR ~36%, R:R ~3.4, daily ~+1.5% (additive 3x), ~3.1 trades/day

---

## 📁 프로젝트 구조

```
bingx_rl_trading_bot/
├── scripts/production/
│   ├── c1_breakout_bot.py        # C1 Breakout v2 엔트리포인트
│   └── c1_breakout/              # 봇 모듈
│       ├── bot.py                # C1BreakoutBot (메인 루프, N-pos, exchange, state)
│       ├── signals.py            # C1BreakoutSignal (채널 돌파, 프랙탈 SL, 트레일 TP)
│       ├── indicators.py         # ATR, 채널, 프랙탈 스윙 계산
│       └── config.py             # 설정 로딩 + 기본값
├── config/
│   ├── c1_breakout_config.yaml   # 전략+리스크 파라미터 (유일한 설정 소스)
│   └── api_keys.yaml             # BingX API 키
├── scripts/analysis/             # 연구/검증 스크립트
├── data/                         # BTC 5m 데이터 (15m 합성용, 백테스트 전용)
├── results/                      # 봇 상태, 검증 결과
├── logs/                         # c1_breakout.log (일일 회전, 30일 보관)
├── claudedocs/                   # 전략 문서, 연구 보고서
└── archive/legacy_bots/          # 폐기된 봇
```

## 🔬 Standard Research Protocol

| 항목         | 표준                                                   |
| ------------ | ------------------------------------------------------ |
| Entry        | 신호 bar[i] → 다음 봉 o[i+1] 진입                      |
| Exit         | Intrabar High/Low (distance-based same-bar resolution) |
| Fee          | 0.10% RT (taker 0.05% × 2)                             |
| MC Test      | Sign randomization (≥999 sims)                         |
| WF           | 5-fold expanding window                                |
| Look-ahead   | Progressive test 필수 (truncated vs full 비교)         |
| Overfit      | 3-way split (train/val/test) + sensitivity ±10%        |
| Additive PnL | Compound 왜곡 방지 — 단순합산 수익률 사용              |

## 🔗 문서 링크

- [C1 설계서](claudedocs/c1_breakout_v2_design.md)
- [연구 프로토콜](claudedocs/STANDARD_RESEARCH_PROTOCOL.md)

> **레거시 문서**: `archive/legacy_bots/docs/`, `docs/` 디렉토리 참조

## ✅ 전략 변경 시 체크리스트

전략 교체 또는 주요 파라미터 변경 시 아래 **모든 항목** 갱신 필수:

1. [ ] `CLAUDE.md` — 빠른 참조 테이블, 검증 결과, Auto-Trigger, 에이전트 가이드
2. [ ] `AGENTS.md` — 파일 경로, 금지 사항
3. [ ] `.claude/hooks/session-init.sh` — 봇 이름, 경로, 키 메모리
4. [ ] `.claude/hooks/guard-production.sh` — 감시 대상 경로
5. [ ] `.claude/hooks/post-commit-remind.sh` — 커밋 감지 경로
6. [ ] `.claude/commands/*.md` — 10개 커맨드 전부 (경로, 기대치, 프로세스명)
7. [ ] `Makefile` — 타겟, 경로, 프로세스명
8. [ ] `Serena project.yml` — initial_prompt
9. [ ] `MEMORY.md` — Active Strategy, Project State
10. [ ] `claudedocs/*_design.md` — 설계 문서 검증 수치, 파라미터
11. [ ] `requirements/runtime.txt` — 실제 의존성
12. [ ] 봇 코드 내 버전 문자열 (docstring + logger)
