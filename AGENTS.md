# AGENTS.md - C1 Breakout v2.6 프로젝트 에이전트 규칙

> 이 프로젝트에서 작업하는 모든 에이전트가 따라야 할 규칙
> **Updated**: 2026-04-18 (v4.7.9 기준)

---

## 필수 규칙

### 1. 문서 최신화 의무
- 코드 수정 시 **반드시** `CLAUDE.md` Version History 갱신 (BUG#XX 번호 + 1줄 설명)
- 설정 변경 시 빠른 참조 테이블 업데이트
- 신규 버그 수정 시 `claudedocs/BUG_HISTORY.md`에도 엔트리 추가
- 백테스트-라이브 정합성 관련 수정 시 `claudedocs/BACKTEST_LIVE_PARITY.md` 갱신

### 2. 커밋 메시지 컨벤션
```
docs: v4.X.Y — 간결한 변경 설명
fix: BUG#XX — 설명
test: C1 v2.6 <test scope> (+N cases)
research: 연구 내용 요약
refactor: <scope> — <reason>

예시:
docs: v4.7.9 — BUG#62~65 반영, BUG_HISTORY/PARITY 신설
fix: BUG#65 — capture actual MARKET close fill price
test: C1 v2.6 커버리지 62% → 71% (+26 cases)
```

### 3. 테스트 절차
전략/코드 변경 시 반드시:
1. **pytest 스위트 통과** (`scripts/tests/`, 113 cases, ~5s)
2. **MC test** (≥999 sims, p < 0.01) — 전략 파라미터 변경 시
3. **WF validation** (5-fold expanding window, 5/5 pass) — 진입/청산 로직 변경 시
4. **Progressive look-ahead test** — 지표 변경 시
5. 결과를 `claudedocs/`에 기록

### 4. 파일 구조 규칙
- 운영 코드: `scripts/production/c1_breakout/` (bot.py, signals.py, indicators.py, config.py)
- 엔트리: `scripts/production/c1_breakout_bot.py`
- 설정: `config/c1_breakout_config.yaml` (유일한 설정 소스)
- 연구 스크립트: `scripts/analysis/`에 추가 (C1 관련만)
- 테스트: `scripts/tests/` (pytest)
- 완료된 연구: `claudedocs/`에 기록
- 레거시/폐기: `archive/legacy_*/`로 이동 (`legacy_analysis`, `legacy_docs`, `legacy_results`, `legacy_bots`)

### 5. 금지 사항 (🔴 CRITICAL)

**봇 안전성**:
- `config/api_keys.yaml` 내용 절대 노출/수정 금지
- 봇 중지 시 열린 포지션/exchange 주문 미확인 금지
- MC/WF 미검증 전략 배포 금지

**백테스트 무결성**:
- Look-Ahead Bias: `shift(-1)`, `rolling(center=True)` 금지
- Compound PnL로 전략 수익 과장 금지 — **additive** 사용
- `.fillna(method='bfill')` 금지 (미래 값 누출)

**BingX CCXT 함정 (교훈)**:
- `priceRate` 파라미터 사용 금지 (BUG#35 — 90% callback 버그)
- `TRAILING_STOP_MARKET` 재배치 시 best_price 추적 리셋 사실 잊지 말 것 (BUG#46)
- `activatePrice`는 반드시 `trail_activation_pct`와 정합 (BUG#62)
- MARKET close 시 `order['average']` 또는 `fetch_my_trades`로 실제 체결가 캡처 (BUG#65)

### 6. Trail 메커니즘 원칙 (2026-04-18 확립)

| 상태 | 주문 타입 | 근거 |
|------|-----------|------|
| Pre-activation (best_pnl ≤ `trail_activation_pct`) | `TRAILING_STOP_MARKET` | baton-touch 미정의 (best_price 아직 추적 시작 전) |
| Post-activation, LOOSEN (ATR↑) | `STOP_MARKET` @ baton-touch trigger | BingX TRAILING cancel+replace는 best_price 리셋 → 2차방정식 해로 이전 레벨 복원 |
| Post-activation, TIGHTEN (best_price↑) | `STOP_MARKET` 갱신 | 백테스트 "re-check every bar" 정합 (BUG#63) |

**Baton-touch 수식** (signals.py와 100% 일치):
```
cur² - best·cur + trail_K·ATR·entry = 0
→ 상근(上根) = trigger_price
```

### 7. 백테스트-라이브 정합성 원칙
- `best_price`는 entry 시점 `fill_price`와 동기화 (BUG#64)
- 실제 MARKET 체결가를 `exit_price`로 기록 (BUG#65)
- Trail 재평가는 매 cycle 실행 (BUG#63, 백테스트의 매 bar 재평가 정합)
- 상세 체크리스트: `claudedocs/BACKTEST_LIVE_PARITY.md` (현재 20/22)

---

## 에이전트별 권한

| 에이전트 | 코드 수정 | 봇 운영 | 설정 변경 | 문서 수정 | 커밋 |
|---------|----------|---------|----------|----------|------|
| dev | ✅ | ❌ | ✅ | ✅ | ✅ |
| automation | ❌ | ✅ | ❌ | ❌ | ❌ |
| monitor | ❌ (읽기만) | ❌ (읽기만) | ❌ | ❌ | ❌ |
| trading-researcher | ❌ (스크립트만) | ❌ | ❌ | ✅ (claudedocs) | ✅ (docs/research) |
| trading-risk | ❌ | ❌ | ❌ | ✅ (claudedocs) | ❌ |

---

## 에이전트 공통 선행 작업

신규 세션 시작 시:
1. `MEMORY.md` 인덱스 확인 → 관련 메모리 읽기
2. Serena 활성화 + `list_memories`
3. 현재 봇 상태 확인 (`results/c1_breakout_state.json`)
4. 최근 로그 확인 (`logs/c1_breakout.log` 마지막 100줄)
5. `git status` + `git log --oneline -5`

---

## 참고 문서

- **전략·검증**: [CLAUDE.md](CLAUDE.md)
- **버그 연대기**: [claudedocs/BUG_HISTORY.md](bingx_rl_trading_bot/claudedocs/BUG_HISTORY.md)
- **정합성 체크**: [claudedocs/BACKTEST_LIVE_PARITY.md](bingx_rl_trading_bot/claudedocs/BACKTEST_LIVE_PARITY.md)
- **연구 프로토콜**: [claudedocs/STANDARD_RESEARCH_PROTOCOL.md](bingx_rl_trading_bot/claudedocs/STANDARD_RESEARCH_PROTOCOL.md)
- **설계 문서**: [claudedocs/c1_breakout_v2_design.md](bingx_rl_trading_bot/claudedocs/c1_breakout_v2_design.md)
