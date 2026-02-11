# Design: pattern-5m-optimization

> Opus 4.6 전면 코드 리뷰 결과 — 구현 대상 이슈 목록

## 분석 요약

14개 모듈, ~200KB 코드를 전수 리뷰한 결과:
- 전체적으로 잘 구조화된 프로덕션 코드
- 기능 변경 없이 내부 품질만 개선하는 리팩토링 대상 도출

---

## 이슈 목록

### 1. [중복] signals.py ↔ indicators.py 캔들 분류 코드 중복

**심각도**: 🟡 Important
**모듈**: `signals.py:58-106`, `indicators.py:88-137`
**문제**: `add_candle_classification()` 과 `calculate_indicators()` 가 동일한 캔들 분류 + 패턴 생성 로직을 독립적으로 구현. 약 50줄 완전 중복.
**위험**: 한쪽만 수정하면 분류 불일치 발생 가능 (v1.24.0에서 실제 발생했던 버그)
**수정**: `signals.py`의 `add_candle_classification()`을 제거하고 `indicators.py`의 `calculate_indicators()`를 단일 소스로 사용. `check_entry_signal()`에서 이미 `df`에 `pattern_3` 컬럼이 있는지 확인하는 가드가 있음 (line 432-433).

### 2. [중복] position_close.py ↔ position_open.py scale-out 설정 중복

**심각도**: 🟢 Minor
**모듈**: `position_close.py:225-262`, `position_open.py:363-400`
**문제**: `_setup_scale_out_for_recovery()`와 `_setup_scale_out()`이 거의 동일한 로직. 차이점은 입력 인자 형태만 다름.
**수정**: `position_open.py`의 `_setup_scale_out()`을 공용으로 사용, `position_close.py`에서 import.

### 3. [중복] TP/SL 계산 로직 3곳 분산

**심각도**: 🟡 Important
**모듈**: `position_open.py:329-360`, `position_close.py:186-194`, `position_close.py:302-317`
**문제**: TP/SL 가격 계산이 `_calculate_tp_sl()`, `recover_position_to_state()`, `recalculate_position_orders()` 3곳에 독립 구현됨. `recover_position_to_state()`는 pattern-specific TP/SL을 사용하지 않아 복구 시 기본 TP/SL로 설정되는 문제.
**수정**: `_calculate_tp_sl()`을 공용 함수로 추출하여 3곳에서 호출.

### 4. [dead code] 비활성 Regime 관련 코드

**심각도**: 🟢 Minor
**모듈**: `constants.py:72-106`, `signals.py:109-162`, `signals.py:451-470`
**문제**: `REGIME_DETECTION_ENABLED = False`로 v1.19.0부터 비활성. 약 100줄의 dead code (`REGIME_PATTERNS`, `detect_market_regime()`, `check_entry_signal()` 내 regime 분기).
**수정**: 코드 제거 대신 주석으로 deprecated 섹션 명확히 표시 (향후 재활성화 가능성 보존).

### 5. [dead code] indicators.py의 미사용 함수/데이터

**심각도**: 🟢 Minor
**모듈**: `indicators.py:159-193`
**문제**: `get_pattern_description()`이 10+10=20개 패턴만 기술하는데 실제 봇은 6L+17S=23개 패턴 사용. 또한 이 함수를 호출하는 곳이 없음.
**수정**: 사용되지 않으므로 제거하거나 현재 23개 패턴으로 업데이트.

### 6. [성능] indicators.py row-by-row 캔들 분류 루프

**심각도**: 🟡 Important
**모듈**: `indicators.py:119-127`, `signals.py:86-94`
**문제**: `for i in range(len(df))` 루프로 매 봉마다 `classify_candle()` 호출. 150봉 × 2회(indicators + signals) = 300번 반복. Pandas vectorized operation으로 대체 가능하지만 classify_candle()의 분기 복잡도가 높아 현실적으로 어려움.
**수정**: 중복 제거(이슈 #1)로 300→150 호출로 50% 감소. 추가로 `pattern_3` 빌드도 vectorized shift 연산으로 대체 가능.

### 7. [성능] signals.py RSI/ATR 중복 계산

**심각도**: 🟡 Important
**모듈**: `signals.py:186-228`
**문제**: `calculate_context()`에서 RSI와 ATR을 자체 계산하지만, `indicators.py`의 `calculate_indicators()`에서도 ATR을 이미 계산. 동일 데이터에 대해 RSI/ATR을 2번 연산.
**수정**: `calculate_indicators()`에서 RSI와 ATR%를 미리 계산하여 df 컬럼으로 추가. `calculate_context()`는 이미 계산된 컬럼에서 직접 읽기.

### 8. [안정성] save_state() 호출 시 state_file 인자 누락

**심각도**: 🟡 Important
**모듈**: `bot.py:458`, `state.py:225`, 외 다수
**문제**: `save_state(state)` 호출 시 `state_file` 인자를 생략하면 기본값 `STATE_FILE`이 사용됨. 대부분의 코드에서 인자 없이 호출하므로 정상 동작하지만, `bot.py:_process_existing_position()`에서는 `save_state(state)` (line 458)로 호출하면서 import를 함수 내부에서 수행.
**수정**: 패턴 통일 — 모든 save_state 호출에서 일관된 방식 사용. 현재 동작에는 문제 없으나 인자 일관성 개선.

### 9. [안정성] position_close.py recover_position_to_state() pattern-specific TP/SL 미적용

**심각도**: 🟡 Important
**모듈**: `position_close.py:163-223`
**문제**: crash recovery 시 `recover_position_to_state()`가 기본 `strategy['tp_pct']`/`strategy['sl_pct']`만 사용. 실제 패턴 정보가 없어 per-pattern TP/SL (`PATTERN_OPTIMAL_TPSL`)이 적용되지 않음. 봇 재시작 시 `adjust_tpsl_to_config()`가 이를 보정하지만, recovery와 adjust 사이에 시간차가 발생.
**수정**: `recover_position_to_state()`에서 교환소 포지션의 entry price와 현재 open orders를 확인하여 pattern 추정 or needs_tpsl 플래그를 즉시 설정.

### 10. [안정성] exchange.py api_retry 데코레이터 내부 함수 생성

**심각도**: 🟢 Minor
**모듈**: `exchange.py:272-277`, `exchange.py:307-309`, `exchange.py:344-346`, `exchange.py:375-377`
**문제**: `fetch_ticker_cached()`, `fetch_balance_cached()`, `fetch_positions_cached()`, `fetch_ohlcv()` 각각에서 매 호출마다 내부 `@api_retry` 데코레이터로 새 함수 객체 생성.
**수정**: API retry 함수를 모듈 레벨로 한번만 생성하거나, 데코레이터 대신 try-retry 루프 직접 사용.

### 11. [구조] 순환 import 방지를 위한 지연 import 다수

**심각도**: 🟢 Minor
**모듈**: `bot.py:432`, `position_open.py:256`, `position_monitor.py:53`, `position_monitor.py:278`, `position_close.py:462`
**문제**: `from .xxx import yyy`를 함수 내부에서 수행하는 패턴이 5곳. 순환 의존성의 증상이나 현재 정상 동작.
**수정**: 현행 유지 (Python에서 허용된 패턴이며 순환 의존성 해소의 표준 방법). 문서화만 추가.

### 12. [구조] models.py APICache 반복 패턴

**심각도**: 🟢 Minor
**모듈**: `models.py:28-76`
**문제**: `get_ticker/set_ticker`, `get_balance/set_balance`, `get_positions/set_positions` 세 쌍이 완전히 동일한 패턴. DRY 위반이지만 명확성과 타입 안전성을 위한 의도적 설계.
**수정**: 현행 유지. 가독성과 타입 힌팅이 추상화보다 우선.

### 13. [안정성] signals.py _save_confidence_to_csv() 파일 경쟁

**심각도**: 🟢 Minor
**모듈**: `signals.py:561-611`
**문제**: CSV 파일에 append 모드로 쓰기할 때 파일 잠금 없이 접근. 단일 인스턴스 봇이므로 현재는 문제 없으나.
**수정**: 현행 유지 (단일 인스턴스 보장).

### 14. [구조] constants.py 중복/deprecated 상수

**심각도**: 🟢 Minor
**모듈**: `constants.py:329-341`, `constants.py:396-402`
**문제**: `CACHE_TTL_TICKER/BALANCE/POSITIONS` (개별)와 `CACHE_TTL_SECONDS` (통합)가 공존. `TP_SL_CHECK_INTERVAL`, `LOG_STATUS_INTERVAL`, `METRICS_SAVE_INTERVAL`은 iteration-based로 deprecated되고 time-based 버전이 추가됨.
**수정**: deprecated 상수에 명확한 주석 추가 및 사용처 확인 후 미사용 상수 제거.

---

## 구현 우선순위

### Phase 1: 중복 제거 (안전, 고영향)
1. 이슈 #1: signals.py ↔ indicators.py 캔들 분류 중복 제거
2. 이슈 #3: TP/SL 계산 로직 단일 함수로 통합
3. 이슈 #2: scale-out 설정 함수 통합

### Phase 2: 성능 개선
4. 이슈 #6: 캔들 분류 호출 50% 감소 (이슈 #1과 연동)
5. 이슈 #7: RSI/ATR 중복 계산 제거

### Phase 3: 안정성 개선
6. 이슈 #9: recovery 시 per-pattern TP/SL 미적용 수정
7. 이슈 #8: save_state() 호출 일관성

### Phase 4: 코드 정리
8. 이슈 #5: 미사용 함수 제거
9. 이슈 #14: deprecated 상수 정리
10. 이슈 #4: Regime dead code 표시

### 건너뛸 이슈 (현행 유지)
- 이슈 #10: api_retry 함수 생성 — 성능 영향 미미
- 이슈 #11: 지연 import — Python 표준 패턴
- 이슈 #12: APICache 반복 — 의도적 설계
- 이슈 #13: CSV 파일 경쟁 — 단일 인스턴스

---

## 제약 조건 (Plan에서 계승)

- 봇이 실시간 운영 중 — 기능 변경 금지
- TP/SL, 패턴, 전략 로직 변경 금지
- 외부 동작(API 호출 형식, 상태 파일 구조) 유지
- 모든 수정은 동작 보존 리팩토링만 허용

## 성공 기준

- 모든 Phase 1-4 이슈 해결
- 봇 재시작 후 정상 동작 확인
- 코드 라인 수 순감소 (중복 제거)
- Gap analysis match rate >= 90%
