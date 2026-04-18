# 백테스트 ↔ 라이브 정합성 체크리스트

> **목적**: 백테스트 결과가 라이브 환경에서 재현되는지 체계적으로 검증.
> **현재 상태 (v4.7.9, 2026-04-18)**: **20/22 정합**. 남은 2건은 구조적 한계로 수용.

## 검증 방식

1. **수학적 정합성**: 백테스트의 수식을 라이브 봇이 동일하게 구현
2. **타이밍 정합성**: 백테스트의 관측 시점과 라이브의 반응 시점 일치
3. **상태 정합성**: 내부 상태 변수의 의미·값이 일치
4. **데이터 정합성**: 가격·ATR·채널 계산의 look-ahead 없음

---

## 22-Item Parity Checklist

### ✅ 달성 (20/22)

| # | 항목 | 상태 | 근거 |
|---|------|------|------|
| 1 | **진입 신호 수식** | ✅ | `signals.py::check_entry` 공유 (백테스트 ↔ 라이브 동일 함수) |
| 2 | **채널 계산 (lookback=15)** | ✅ | `indicators.py::compute_channel`, `shift(1)`로 causal |
| 3 | **ATR 계산 (Wilder, period=14)** | ✅ | `indicators.py::compute_atr`, causal (no `center=True`) |
| 4 | **Fractal SL 계산 (lookback=10)** | ✅ | `indicators.py::compute_fractal_swings`, causal |
| 5 | **SL 최대 cap (3.3×ATR)** | ✅ | `signals.py::check_entry`의 `sl_max_pct` 클램프 |
| 6 | **Body ratio (>40% of range)** | ✅ | `check_entry`에서 동일 조건 |
| 7 | **다음 봉 시가 진입** | ✅ | 백테스트: `o[i+1]`. 라이브: 15m bar close 후 `market` order |
| 8 | **수수료 (0.10% RT)** | ✅ | 백테스트 시뮬. 라이브: BingX taker 0.05% × 2 (동일) |
| 9 | **Emergency SL (3.0%)** | ✅ | `signals.py::check_exit` 공유 |
| 10 | **Timeout (192 bars = 48h)** | ✅ | `check_exit` 공유, `bars_held` 증가는 cycle 단위 |
| 11 | **Trail 수식 — `cur² - best·cur + trail_K·ATR·entry = 0`** | ✅ | `signals.py::check_exit` TRAIL path. 라이브는 `bot.py::_calc_trail_trigger_price`가 **100% 동일 수식** (BUG#61b) |
| 12 | **Trail activation threshold (`trail_activation_pct` = 0.05%)** | ✅ | 백테스트: `check_exit` 내부. 라이브: `activatePrice = entry × (1 ± 0.0005)` (BUG#62) |
| 13 | **Trail 매 bar 재평가 (backtest "re-check every bar")** | ✅ | 백테스트: 매 bar close. 라이브: 매 cycle `_update_exchange_trail`에서 best_price-driven 재계산 (BUG#63) |
| 14 | **`best_price` 초기값 (= entry_price)** | ✅ | 백테스트: 진입 봉 close 또는 high. 라이브: `fill_price`와 동기화 (BUG#64) |
| 15 | **Exit priority (SL > Emergency > Timeout > Trail)** | ✅ | `check_exit` 공유 함수 — 백테스트/라이브 동일 우선순위 |
| 16 | **Leverage (exchange 10x / trading 3x)** | ✅ | 백테스트: `trading_leverage` × position size. 라이브: `setLeverage(10)` + qty × 3 |
| 17 | **One-Way 모드 (positionSide=BOTH)** | ✅ | 라이브 봇 명시 설정. 백테스트는 N=1이라 모드 무관 |
| 18 | **N=1 포지션 제약** | ✅ | `max_positions=1` config (백테스트/라이브 공유) |
| 19 | **최소 bar 간격 (min_bars_between=2)** | ✅ | 백테스트: 계산된 바 인덱스 diff. 라이브: `bars_since_last_exit` + wall-clock fallback (BUG#54) |
| 20 | **실제 체결가 기록 (PnL 정확도)** | ✅ | 라이브: `order['average']` 또는 `fetch_my_trades` fallback (BUG#65). 백테스트는 이론가 = 체결가 가정 |

### ⚠️ 구조적 한계로 미달성 (2/22)

| # | 항목 | 상태 | 이유 |
|---|------|------|------|
| 21 | **Pre-activation trail이 백테스트와 다른 type** | ⚠️ | 라이브: `best_pnl ≤ trail_activation_pct`일 때 TRAILING_STOP_MARKET 사용 (baton-touch 수식이 미정의 영역). 백테스트는 activation 전에는 trail 발동 자체 안 함. → **봇 다운타임 중 안전망**으로 수용 (fractal SL은 별도로 항상 존재) |
| 22 | **MARKET 주문 slippage** | ⚠️ | 라이브: 거래소 호가 구조에 따른 필연적 slippage. 백테스트: 이론가 = 체결가. BUG#65로 **측정·기록**만 가능 (`exit_slippage_pct`), 제거 불가 |

---

## 정합성 강화 히스토리

| 시점 | 정합성 | 주요 수정 |
|------|--------|-----------|
| v2.5 초기 (2026-04-13) | ~14/22 | 수학적 기본 일치, BUG#35 priceRate 이전 |
| BUG#35 적용 (2026-04-14) | 15/22 | Trail callback 90% → 0.9% 정상화 |
| BUG#46 정책 확립 (2026-04-15) | 17/22 | Tighten never / LOOSEN only, tracking 리셋 억제 |
| BUG#48 orphan 복원 (2026-04-17) | 18/22 | 재시작 시 실제 SL 보존 |
| BUG#61~65 통합 (2026-04-18) | **20/22** | Baton-touch + activation 정합 + best sync + 실제 체결가 |

---

## 수학적 증명: Baton-Touch Wrong-Side 불가능성

사용자의 우려: "급작스러운 가격 스파이크에서 다음 trail 계산 시 `baton_trigger`가 `cur_price`의 잘못된 쪽에 배치되면?"

### LONG 기준 증명

**정의**:
- `baton_trigger = cur² - best·cur + trail_K·ATR·entry = 0`의 상근(上根)
- `trail_dist = trail_K × ATR × entry / cur` (current 기준 상대 거리)
- `drawdown = (best - cur) / best × 100`

**Wrong-side (`baton_trigger > cur_price`)** ⟺ `drawdown < trail_dist` (수학적 동치)

즉, "trail이 현재가 위에 놓이는 상태" ⟺ "가격이 아직 trail 거리만큼 하락하지 않은 상태".

만약 `drawdown ≥ trail_dist`이면:
- 백테스트 `check_exit`이 TRAIL_TP로 청산.
- 라이브의 `process_candles`는 `check_exit → _do_close → _update_exchange_trail` 순서.
- `check_exit`이 먼저 청산을 트리거 → `_update_exchange_trail` 도달 못 함.

**결론**: `_update_exchange_trail`이 실행되는 모든 상황은 `drawdown < trail_dist`를 만족 → `baton_trigger < cur_price`가 구조적으로 보장됨.

### SHORT 대칭

SHORT에서도 부등호 방향 반대, 나머지 동일. 증명 대칭.

---

## 5-Layer 보호 구조 (정합성 + 안전망)

1. **BingX TRAILING_STOP_MARKET** — intrabar tick-level (pre-activation만, 봇 다운 대비)
2. **Bot `check_exit`** — 15m bar close 기준 (백테스트와 수식·타이밍 동일)
3. **BUG#61 sanity check** — baton-touch trigger 재평가 fallback
4. **Fractal SL `STOP_MARKET`** — 거래소 상주 절대 가격
5. **Emergency `-3%` hard stop** — `check_exit`이 최우선 검사

레이어 1만 백테스트와 엄밀히 다름(tick vs bar). 나머지 2~5는 백테스트와 정확히 일치.

---

## 향후 추가 가능한 정합성 체크

- [ ] **Intrabar tick resolution** (거래소 API가 5m 또는 1m 지원 시 accuracy 상승)
- [ ] **Queue-time slippage modeling** (백테스트에 실제 slippage 분포 주입)
- [ ] **Latency simulation** (bar close → order placement 100~500ms 지연 재현)

현 단계에서는 비용/효과상 우선순위 낮음 — 20/22로도 실운영 수익률이 기대치 이내.

---

## 관련 스크립트

- `scripts/analysis/live_vs_backtest_verification.py` — 1:1 trade matching
- `scripts/analysis/live_window_analysis.py` — 13-trade window 비교
- `scripts/analysis/shake_out_pattern_verification.py` — 특정 패턴 정합 검증
- `scripts/analysis/trail_alternatives_comparison.py` — trail 변종 영향 측정
- `scripts/analysis/intrabar_trail_impact.py` — bar vs tick 해상도 영향

## 관련 문서

- [BUG_HISTORY.md](BUG_HISTORY.md) — 전체 버그 연대기
- [c1_breakout_v2_design.md](c1_breakout_v2_design.md) — 전략 설계
- [STANDARD_RESEARCH_PROTOCOL.md](STANDARD_RESEARCH_PROTOCOL.md) — 연구 프로토콜
