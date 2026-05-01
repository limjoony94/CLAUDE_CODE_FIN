# Infrastructure Lessons Inherited (advisor non-blocking #5)

**Date**: 2026-05-01
**Source**: `bingx_rl_trading_bot/` 18d~13d 메모리 + BACKTEST_LIVE_PARITY.md + BUG_HISTORY.md
**Why preserve**: Zero-base는 strategy/conclusion 가정 차단이지, **CCXT/exchange API 미세 함정 재발견 의무가 아님**. P6 LIVE-deploy 단계에서 재필요.

---

## CCXT BingX API Pitfalls (P6 LIVE-deploy 시 의무 적용)

### 1. `priceRate` /100 자동 적용 (TRAILING_STOP_MARKET)

```python
# CCXT internal: trailingPercent = safe_string_2(params, 'trailingPercent', 'priceRate')
#                requestTrailingPercent = string_div(trailingPercent, '100')
```

**Rule**: `params['priceRate']`에 percent 값 그대로 전달 (예: `0.7` = 0.7%). `/100` 절대 금지.
- ✅ `priceRate: 0.7` → CCXT /100 → API 0.007 → BingX 0.7% 발동
- ❌ `priceRate: 0.007` → CCXT /100 → API 0.00007 → BingX 0.007% 즉시 트리거 (BUG#29)

### 2. `positionSide` for One-Way mode

BingX One-Way 모드의 모든 주문 (LONG/SHORT entry, close, SL, trail):
- ✅ `positionSide: 'BOTH'`
- ❌ `positionSide: 'LONG'` 또는 `'SHORT'` → One-Way에서 거부

Hedge mode 자동 전환 발견 시 (BUG#66): `set_position_mode(hedged=False)` + 검증 fail 시 즉시 abort.

### 3. `fetch_positions` side 파싱

CCXT 반환값 `position['side']` = 소문자 `'long'`/`'short'`. 비교 시 `.upper()` 또는 `.lower()` 통일 필수.

### 4. TimeSyncBingX

BingX 서버 시간 ~6-7s ahead of local clock. CCXT 기본 `milliseconds()` 오버라이드로 offset 적용 필수. ±60초 clamp (비정상 응답 차단, BUG#61 별도).

### 5. `stopPrice` (STOP_MARKET)

CCXT `price_to_precision`만 적용, 추가 변환 없음. percent 아닌 absolute price 그대로.

---

## BT-LIVE Parity Gotchas (Backtest 결과 재현 시 의무 검증)

### Trail Update Quadratic (BUG#61)

LOOSEN 경로에서 BingX TRAILING_STOP_MARKET cancel+replace 시 best_price reset됨 → STOP_MARKET 사용. 정확한 quadratic:
```
cur² - best * cur + trail_K * ATR * entry = 0
```
2차방정식 양수 해 = STOP_MARKET trigger price. 이전 trail level 보존.

### `activatePrice` 정합 (BUG#62)

```python
activatePrice = entry_price * (1 ± trail_activation_pct / 100)
```
기존 코드 `1.001` (0.1%)는 백테스트 0.05%의 2배 → activation 지연. trail_activation_pct 직접 사용 필수.

### Best Price Sync at Entry (BUG#64)

Entry 시 `best_price = fill_price` 동기화 (signal_price 아님). 백테스트는 entry bar에서 best_pnl=0.

### Actual Fill Price Capture (BUG#65)

`order.average` 우선 → fallback `fetch_my_trades`로 실제 MARKET 체결가 캡처. `_do_close`에서 사용, `exit_slippage_pct` 기록.

### state.json I/O Defense (BUG#58)

OneDrive sync lock 대응. Read/write 예외 try/except + retry.

### `bars_since_last_exit` Wall-clock (BUG#54)

봇 재시작 시 elapsed_bars 보정. `last_exit_time` 저장 → 재시작 시 wall-clock difference로 누락 bars 계산.

---

## Standard Research Protocol (P3-P5 백테스트 시 의무)

| Item | Standard |
|------|----------|
| Entry | 신호 bar[i] → 다음 봉 o[i+1] 진입 (bar-close → next-bar-open) |
| Exit | Intrabar High/Low (distance-based same-bar resolution) |
| Fee | 0.10% RT (taker 0.05% × 2) — **본 mandate v2는 0.16% realistic / 0.20% stress 사용** |
| MC Test | Sign randomization ≥999 sims |
| WF | 5-fold expanding window |
| Look-ahead | Progressive test 필수 (truncated vs full 비교) |
| Overfit | 3-way split (train/val/test) + sensitivity ±10% |
| Additive PnL | Compound 왜곡 방지, 단순합산 수익률 |

---

## Friction-Floor Empirical Evidence (skepticism prior)

R5/R13/R41/L2/R26 등 27 mechanisms × 5 substrates 검증:
- 0 deployable strategy except R5 single-coin BTC carry $49/yr
- avg_gross/trade [+0.010%, +0.050%] consistently < 0.07% taker friction
- "Substrate change does NOT lift edge above friction" — bar-level OHLCV, L2 microstructure, multi-coin XS 모두 동일 결론

**Mandate v2에 대한 함의**:
- H3-H5 force-flow reversal hypothesis는 prior에 대해 default skepticism 필수
- Phase A 결과가 6-criteria 통과해도 stress friction (0.20% RT)에서 재검증 의무
- "이번엔 다르다" 직관 회피 — anti-fishing charter가 protection

---

## R26 LIVE Postmortem (deploy gate trigger)

R26 14d LIVE -12.86% (BT 모델 +X% 가정 대비 catastrophic divergence):
- Mark-price ≠ fictional. Testnet ≠ mainnet.
- BT-realistic friction > BT-modeled friction → stress variant 의무 (precommit_amendment_001)
- 30분짜리 cohort distribution check가 catastrophe 회피 가능 (lessons_distribution_check_20260427.md)

**P6 LIVE-readiness 시 5-Gate Protocol 의무** (memory `strategy_deploy_5gate_protocol.md`):
- Gate 1: 모델 충실도 audit (LIVE 코드 ↔ BT 1:1 mapping)
- Gate 2: n≥20 multi-window mean+median 양수, sign test p<0.05
- Gate 3: 데이터 source 일치
- Gate 4: LIVE-realistic BT 6-sub-items (4a-4f)
- Gate 5: D-1/D-3/D-7 cron auto-halt

한 gate 미통과 → deploy BLOCK.

---

## How to Apply

1. P5에서 strategy candidate 식별 시: 본 문서의 §1-2 (CCXT/BT-LIVE parity)를 LIVE harness 설계 시 의무 체크
2. P6 LIVE-readiness 진입 시: 5-Gate Protocol 의무 적용
3. 모든 priority result 평가 시: friction-floor evidence (27 mechanisms 0 deployable)를 prior로 calibrated skepticism

본 문서는 **research conclusion이 아닌 infrastructure knowledge**로서 zero-base scope에 포함. 변경 사항 발견 시 update.
