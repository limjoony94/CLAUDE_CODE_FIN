# C1 Breakout v2.6 — BUG History

> **Scope**: BUG#1~65 연대기. C1 Breakout v2.x 개발 과정 누적 수정 사항.
> **Updated**: 2026-04-21 (v4.8.0 — progressive_trail dynamic K helper 교훈 추가)
> **Source**: `scripts/production/c1_breakout/bot.py` docstring + 커밋 히스토리.

## 카테고리 요약

| 카테고리 | BUG# | 설명 |
|----------|------|------|
| **Exchange API / CCXT** | #35, #38, #44, #61(04-17) | BingX 특이 동작, TimeSync, Balance cache |
| **State / I/O** | #43, #45, #54, #56, #57, #58 | Persistence, deprecation, OneDrive lock |
| **Orphan / Ghost** | #36, #48, #50 | 재시작 복원, orderType 분류 |
| **Trail 메커니즘** | #35, #37, #46, #59, #60, #61(04-18), #62, #63, #65 | priceRate 버그, baton-touch, 정합성 |
| **Entry / Exit** | #49, #53, #55, #64 | Slippage, sanity check, fill sync |
| **Leverage / Sizing** | #52 | 관계 검증 |
| **Orchestration / Outage** | #51 | fetch_candles 연속 실패 |

---

## 상세 연대기

### BUG#35 (2026-04-14) 🚨 CRITICAL — TRAILING_STOP_MARKET priceRate 90% 버그
**위치**: `_update_exchange_trail`, `_exchange_open` trail 파라미터
**Root cause**: `priceRate` 파라미터 전달 시 CCXT 내부 ÷100 변환을 `extend(request,params)`가 덮어써서 BingX가 90% callback으로 해석 → trigger가 `best × 0.1 (~$7,212)` 위치.
**Fix**: `trailingPercent` 사용 (CCXT omit list 포함, 변환 생존). trigger → `best × 0.991 (~$71,778)` 정상화.
**영향**: 실운영 중 잘못 배치된 TRAILING 주문을 발견, 즉시 재시작 + replace 로직 추가 (`_force_trail_reset`).
**교훈**: BingX-CCXT 파라미터 충돌. `trailingPercent`만 쓸 것. `priceRate` 절대 금지 (AGENTS.md §5 명문화).

### BUG#36 (2026-04-14) — Ghost resolution timestamp filter
**위치**: `_resolve_ghost_exit`
**Root cause**: 최근 모든 sell을 ghost exit으로 매칭 → LONG close 후 SHORT open(sell)을 이전 LONG exit price로 오매칭.
**Fix**: `entry_time` 이후 close side 거래만 매칭.
**교훈**: 거래 매칭은 반드시 시간 경계 확인.

### BUG#37 (2026-04-14) — Trail tighten-only replacement
**위치**: `_update_exchange_trail`
**Root cause**: 모든 callback 변경에 re-place → ATR 상승 시에도 cancel+replace → BingX tracking 리셋.
**Fix**: `new_callback < old_callback`일 때만 replace. Startup forced reset은 예외.
**교훈**: (이후 BUG#46에서 완전 뒤집힘 — 실제로 LOOSEN-only가 맞음)

### BUG#38 (2026-04-14) — Insufficient margin retry
**위치**: `_exchange_open` MARKET 주문
**Root cause**: 단일 시도, 101253 에러 시 fallback 없음. 대형 손실 후 98% 사이징이 margin 부족 가능.
**Fix**: 95% → 90% 순차 retry.

### BUG#43 (2026-04-14) — `_force_trail_reset` when no position
**Root cause**: startup 플래그가 true인 채 포지션 없을 때도 유지 → 신규 진입 직후 첫 trail cycle이 방금 배치한 trail cancel+replace → 15분 보호 공백.
**Fix**: `_load_state` 후 `positions`가 비어있으면 플래그 clear.

### BUG#44 (2026-04-14) — Balance cache in retry loop
**Root cause**: `_calc_amount`가 retry마다 `_get_balance()` API 호출 → 3회 중복.
**Fix**: retry loop 이전 balance 1회 캐시.

### BUG#45 (2026-04-15) — Ghost exit_time uses detection time
**Root cause**: `trade_history`가 `datetime.utcnow()` (ghost 탐지 시각) 기록. 실제 exchange 체결 시각과 최대 15분 오차 → 일 경계에서 PnL 일별 집계 왜곡.
**Fix**: `_resolve_ghost_exit`가 거래소 timestamp 반환, ghost handler가 사용.

### BUG#46 (2026-04-15) 🚨 CRITICAL — Trail tighten resets tracking
**Root cause**: TRAILING_STOP_MARKET cancel+replace가 BingX 네이티브 best_price 추적을 파괴. 새 주문은 activatePrice 재도달 필요 → 도달 전 가격 하락 시 보호 공백.
**Fix**: 비대칭 정책
- **LOOSEN only** (ATR↑ > 0.1pp): 재배치. 넓은 trail은 best에서 멀어 premature trigger 낮음.
- **TIGHTEN never**: 봇의 `check_exit`이 매 15분 처리 (백테스트 정합).
**교훈**: 네이티브 TRAILING은 "지금 이 순간" 기준 재시작하는 것과 동치.

### BUG#48 (2026-04-17) 🚨 CRITICAL — Orphan SL discarded (Opus 4.7 review)
**위치**: `_sync_exchange` orphan 경로
**Root cause**: orphan 채택 시 항상 `emergency_sl_pct` (3%)를 `sl_price`로 기록. 원래 fractal SL이 0.5%였어도 재시작 후 3%로 간주 → 거래소 SL (0.5%)과 불일치 → `_update_exchange_trail`이 3%에 새 SL 배치 → 기존 0.5% SL이 루즈닝.
**Fix**: `_resolve_orphan_sl()` 신설 — 거래소의 live reduceOnly STOP 주문을 조회, 가장 타이트한 것 선택, `sl_price`와 `sl_order_id` 복원. 3% fallback은 정말 SL이 없을 때만.
**영향**: 재시작 시 실제 보호 레벨 보존. 이 버그가 있던 동안은 재시작이 리스크였음.

### BUG#49 (2026-04-17) — Fill price 기준 sl_pct 범위 체크
**Root cause**: Signal SL은 `bar_close` 기준 사이징. `fill_price`와 slippage 존재. `entry_price = fill_price` 업데이트 후 `sl_pct`가 `sl_min_pct` 미만 또는 `sl_max_pct` 초과 가능.
**Fix**: warn-only (fractal SL은 absolute market structure이므로 수정 안 함). 로그로 모니터링.

### BUG#50 (2026-04-17) — Ghost exit reason stale best_price
**Root cause**: 거리 휴리스틱이 `est_trail = last_callback × local best_price` 기반. Offline 기간 실제 극값이 봇 상태보다 멀어짐 → misclassification.
**Fix**: `trade.info.orderType` (STOP_MARKET vs TRAILING_STOP_MARKET) 우선 사용. Distance fallback 유지.

### BUG#51 (2026-04-17) — Silent candle fetch outage
**Root cause**: `fetch_candles` 실패 → `process_candles` 건너뜀 → `check_exit` 건너뜀 → ATR trail tighten 정지. BUG#46 이후 tighten은 봇 책임.
**Fix**: `_candle_fail_streak`. 포지션 있을 때 streak ≥3, 없을 때 ≥6에서 경고.

### BUG#52 (2026-04-17) — Leverage relationship 검증
**Root cause**: `trading_leverage > leverage`가 config load 시 허용 → 사이징이 거래소 cap 초과 → 즉시 청산 리스크.
**Fix**: `load_config`가 `ValueError` raise. `sl_min_pct < sl_max_pct`도 검증.

### BUG#53 (2026-04-17) — Channel sanity check
**Root cause**: `channel_high <= channel_low` (flat/inverted data)일 때도 breakout 조건이 비정상 발동 가능.
**Fix**: `check_entry`에서 명시적 reject.

### BUG#54 (2026-04-17) — `bars_since_last_exit` wall-clock
**Root cause**: cycle 당 증가만. 2h 다운타임 후 재시작 시 saved=0 → 1 (`min_bars_between=2` 여전히 1회 블록하지만 의미는 어긋남).
**Fix**: `last_exit_time` persist. `_load_state`에서 `elapsed_bars = (now - last_exit_time) / 15min` 과 saved 중 큰 값.

### BUG#55 (2026-04-17) — MARKET partial fill silent
**Root cause**: BingX MARKET 부분 체결은 드물지만 thin liquidity 순간에 가능. SL/Trail은 filled_qty 사이징(BUG#28)이라 보호는 정상이지만 operator 신호 없음.
**Fix**: `filled_qty < requested × 0.99` 시 shortfall % 경고.

### BUG#56 (2026-04-17) — `trade_history` in-memory 성장
**Root cause**: `_save_state`가 disk엔 `[-500:]` 기록하지만 in-memory는 계속 append.
**Fix**: exit 직후 1000 초과 시 500으로 trim.

### BUG#57 (2026-04-17) — `datetime.utcnow()` deprecated
**Root cause**: Python 3.12+에서 naive UTC 생성 deprecated, 3.14+에서 제거 가능.
**Fix**: `_utc_now()` / `_utc_now_naive_iso()` 헬퍼. 내부는 aware UTC, 직렬화는 `.replace(tzinfo=None).isoformat()` (기존 state.json 포맷 호환).

### BUG#58 (2026-04-17) — state.json I/O on OneDrive lock
**Root cause**: OneDrive sync 트리거된 `ERROR_SHARING_VIOLATION`이 `os.replace`에서 전파 → 메인 루프 크래시.
**Fix**: `json.dump`와 `os.replace` try/except. 다음 cycle 재시도. Orphan detection으로 재수화 가능.

### BUG#59 (2026-04-17) — `_update_exchange_trail` silent failures
**Root cause**: outer try/except가 cycle마다 1줄 경고만 → 지속적 구조 문제(API 권한 박탈, rate limit) 묻힘. 이 함수 안에서 SL 검증 + trail 재배치 둘 다 실행.
**Fix**: `_trail_update_fail_streak` 카운터. ≥3회 연속 실패 시 SL 검증 + tighten 백업 미작동 경고 격상.

### BUG#60 (2026-04-17) — check_exit trail path `current_close ≤ 0 / NaN`
**Root cause**: `trail_dist_pct = trail_K × atr / current_close × 100`. `current_close ≤ 0` → ZeroDivisionError. NaN → NaN 전파로 조용히 스킵 → 봇 측 보호 공백.
**Fix**: `not math.isnan(current_close) and current_close > 0` 가드.

### BUG#61 (2026-04-17) — TimeSync offset clamp
**Root cause**: BingX serverTime 응답을 sanity 없이 신뢰. 비정상 응답(0, year-2000) → `_time_offset` 오류 → 이후 모든 signed request timestamp 에러.
**Fix**: `_MAX_OFFSET_MS = 60_000`. 초과 시 reject, 이전 offset 유지.

---

## v4.7.8 (2026-04-18) 정합성 보강 — BUG#61~65 재번호

> ⚠️ **번호 충돌**: BUG#61은 04-17 TimeSync와 04-18 Trail baton-touch에서 동일 번호 사용. 컨텍스트로 구분.

### BUG#61 (2026-04-18) 🚨 CRITICAL — Trail LOOSEN baton-touch
**위치**: `_update_exchange_trail` LOOSEN 경로
**Root cause**: BUG#46 LOOSEN도 여전히 BingX tracking 리셋. 새 TRAILING의 `activatePrice = cur_price × 1.001` 의미:
1. 과거 best_price 망각 ($77K → 잊힘)
2. activatePrice 재도달까지 tracking 재개 못함
3. 즉시 price가 activatePrice 밑이면 무보호

**Fix**: LOOSEN 시 **baton-touch**:
1. `_calc_trail_trigger_price()`로 **정확한 백테스트 수식** trigger 계산
   수식: `cur² - best·cur + trail_K·ATR·entry = 0` → 상근(上根)
2. STOP_MARKET (고정 trigger, TRAILING 아님) 배치
3. 이전 trail 레벨 보존 (tracking이 계속되었을 경우의 가정적 위치)

`trail_order_id`를 pos state에 추적하여 식별/검증/취소 정확히.
Pre-activation (best_pnl ≤ activation_pct)은 여전히 TRAILING_STOP_MARKET (baton 미정의).

**BUG#61b**: `_calc_trail_trigger_price`는 `signals.py`의 `check_exit` trail 공식과 100% 동일.

**교훈**: 외부 거래소 추적 기능이 리셋되는 상황에서는, 봇 자체가 "가상의 트래킹" 상태를 계산해서 STOP_MARKET으로 동기화하는 것이 해법.

### BUG#62 (2026-04-18) — `activatePrice` ↔ `trail_activation_pct` 정합
**Root cause**: TRAILING_STOP_MARKET이 `activatePrice = entry × 1.001` (0.1%) — 백테스트의 `trail_activation_pct` (0.05%)보다 **2배 엄격**. Live trail이 백테스트보다 늦게 활성화.
**Fix**: `activatePrice = entry × (1 ± trail_activation_pct/100)` → 0.05% 정합.

### BUG#63 (2026-04-18) — Best-price-driven trail tighten
**위치**: `_update_exchange_trail` TIGHTEN 경로
**Root cause**: BUG#61 baton-touch STOP_MARKET은 static — best_price가 올라가도 갱신 안 됨. 백테스트는 매 bar 재평가 → trending 시장에서 live trail이 백테스트보다 뒤처짐.
**Fix**: 매 cycle 정확한 백테스트 trigger 재계산. 현재 STOP_MARKET trigger보다 타이트하면 cancel+replace. 임계 0.05% (과도한 churn 방지).

### BUG#64 (2026-04-18) — `best_price` ↔ `fill_price` 동기화
**위치**: `_do_open` fill_price 반영 후
**Root cause**: `_do_open`이 `best_price = signal_price`로 세팅, `_exchange_open`이 `entry_price = fill_price` 업데이트하지만 `best_price`는 signal로 남음. Slippage 존재 시 entry 시점 `best_pnl ≠ 0` (LONG + 양의 slip이면 음수).
**백테스트**: `best_price = entry_price` 정확히 → `best_pnl = 0` at entry.
**Fix**: fill_price 업데이트 후 `best_price = fill_price`도 동기화.

### BUG#65 (2026-04-18) — 실제 MARKET 체결가 캡처
**위치**: `_exchange_close`, `_do_close`
**Root cause**: `_do_close`가 `exit_price`를 `check_exit`의 이론적 trigger로 기록. TRAIL_TP 청산의 경우 sell-side slippage 미반영 → PnL 과장.
**Fix**: `_exchange_close`가 `(actual_fill, actual_ts)` 반환:
1. `order['average']` 또는 `order['price']` 우선
2. fallback: `fetch_my_trades(symbol, since=...)`

`_do_close`가 실제 fill 사용. `exit_slippage_pct` 기록 (모니터링). API 에러 시 theoretical로 safe degrade.

---

## 레거시 버그 (BUG#1~34, v2.5 이전)

v2.3~v2.5 사이클에서 16~30건 수정 (BUG#1~34). 상세는 이전 커밋 메시지 (`182a348` C1 Breakout v2.5: 30-Cycle critical review 등) 참조.

주요 카테고리:
- **BUG#18**: `entry_price` fill_price로 업데이트
- **BUG#26**: Emergency close 재시도 로직
- **BUG#28**: SL/Trail을 filled_qty로 사이징 (partial fill 대응)
- **BUG#42**: `candle_bars_fetch`가 잘못된 config 섹션 조회

---

## 교훈 (크로스 컷)

### BingX / CCXT 함정
- `priceRate` 금지 — `trailingPercent`만 (BUG#35)
- `TRAILING_STOP_MARKET` 재배치 = best_price 추적 리셋 (BUG#46)
- serverTime 맹신 금지 — clamp (BUG#61/04-17)

### 백테스트 ↔ 라이브 정합성
- `best_price` = `fill_price` at entry (BUG#64)
- `activatePrice` = backtest `trail_activation_pct` (BUG#62)
- 매 cycle 백테스트 수식 재평가 (BUG#63)
- 실제 체결가 기록 (BUG#65)
- 자세히: [BACKTEST_LIVE_PARITY.md](BACKTEST_LIVE_PARITY.md)

### 재시작 / 크래시 복구
- Orphan 채택은 거래소 실제 주문 조회 (BUG#48)
- `last_exit_time` wall-clock 복원 (BUG#54)
- state.json I/O 방어 — OneDrive lock 대응 (BUG#58)

### 방어적 가드
- Sanity check: channel (BUG#53), current_close (BUG#60), leverage 관계 (BUG#52), partial fill (BUG#55)
- 연속 실패 감지: candle fetch (BUG#51), trail update (BUG#59)

### Dynamic parameter helper (v4.8.0 progressive_trail)
- **Anti-pattern**: signals.py에 `k_post if best_pnl >= thr else trail_K` 직접 작성, bot.py에
  별도로 동일 로직 중복 → 수식 divergence 리스크 (BUG#61b 유형 재발)
- **Best practice**: 단일 helper 함수(`signal.get_effective_trail_k(best_pnl)`)로 양쪽 호출 통일
- **근거**: signals.py L105-109, bot.py L1092 모두 동일 helper 호출 → 수식 100% 일치가
  구조적으로 보장. "설계에 한 번, 구현에 여러 번"이 아니라 "설계와 구현 모두 한 함수"

---

## 관련 테스트

`scripts/tests/` 의 127개 pytest 케이스가 regression 방지 (v4.8.0 기준, progressive_trail 8 cases 포함).
BUG → Test 매핑은 [run-tests 커맨드 문서](../../.claude/commands/run-tests.md) 참조.
