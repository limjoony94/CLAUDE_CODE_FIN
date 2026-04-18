# Plan: Intrabar Parity 개선 (BT intrabar 모델 + LIVE tick 추적)

> **Feature**: intrabar_parity
> **Date**: 2026-04-19
> **Phase**: Plan
> **Scope**: Track A (BT intrabar slippage model) + Track B (LIVE tick-level best_price)
> **기반**: `claudedocs/BACKTEST_LIVE_PARITY.md` 구조적 한계 #21 (pre-activation TRAILING tick), #22 (MARKET slippage)

---

## 1. Background

### BT-LIVE 괴리의 잔여 원인 (2026-04-19 심층 검토 결과)

22-item 정합성 체크 중 **20/22 달성, 2건이 구조적 한계**:

| # | 항목 | 현재 상태 | 본 연구의 대응 |
|---|------|-----------|-----------------|
| 21 | Pre-activation TRAILING_STOP_MARKET이 tick 해상도 사용 | BT는 bar close 평가 | **Track A** — BT에 intrabar path 주입 |
| 22 | MARKET 주문 slippage | 이론가 = 체결가 가정 | **Track A** — slippage 분포 주입 / **Track B** — LIVE best_price tick 갱신 |

BT/LIVE 차이의 정량:
- BT-LIVE 갭 1x: **-7.72pp** (19 trades 1주일간)
- 이 중 ~51% (3.97pp)가 slippage·intrabar 비대칭에서 발생 (나머지는 pre-fix BUG 단건)
- 최근 5 trades 평균 |PnL diff| 1.3pp (fix 이후 절반 감소, 여전히 0이 아님)

### check_exit 내부의 intrabar 사용 현황

| Exit 경로 | 현재 BT | 현재 LIVE | 격차 |
|-----------|---------|-----------|------|
| **SL** | `lo ≤ sl` (intrabar touch) | 거래소 STOP_MARKET intrabar tick | ≈ parity ✓ |
| **Emergency** | `worst_pnl = (lo/entry-1)×100` intrabar | 없음 (봇 내부 최우선 체크) | ≈ parity (check_exit 우선) ✓ |
| **Timeout** | `bars_held ≥ 192` | 동일 | parity ✓ |
| **Trail** | `best` (intrabar 누적) + `cur_pnl` (bar **close**) | TRAILING 거래소 tick + bot per-cycle | **⚠ 격차** |

즉 **Trail이 핵심 격차**: `drawdown = best_pnl - cur_pnl`에서 `cur_pnl`이 bar close 기준이므로, 한 15m 봉 안에서 best peak 찍은 뒤 반등하여 close 시점엔 여전히 best 근처라면 BT는 trail 미발동. LIVE는 거래소 tick 기준으로 즉시 발동 가능.

### 5m 데이터 확보
- `data/btc_5m_270days_reclassified.csv`: 95,838 rows = 약 333일 분량
- 15m bar 1개 = 5m bar 3개 (intrabar 해상도 3x)
- 추가 1m 데이터는 현재 없음 — 5m로 시작

---

## 2. Goal

BT-LIVE PnL 갭을 **1x 기준 -7.72pp → -3pp 이내**로 축소.

두 방향에서 동시 개선:
- **Track A**: BT를 더 현실적으로 (intrabar simulation)
- **Track B**: LIVE를 더 BT-close (tick-level best_price)

단, Track A는 **파라미터 연구의 표준 메트릭**으로 영구 정착시키고, Track B는 **운영 안정성 비용 대비 개선 효과**가 명확할 때만 production 적용.

---

## 3. Hypotheses

| 가설 | 내용 | 검증 방법 |
|------|------|----------|
| **H1** | BT에 intrabar (5m path) + slippage 주입 시 baseline PnL이 clean +170% → +155~165% 감소 | C1 baseline intrabar BT vs clean BT 비교 |
| **H2** | Intrabar-BT가 LIVE 19-trades 결과를 더 잘 재현 (갭 4pp 이내) | intrabar-BT 결과 vs live_vs_backtest_verification.json 매칭 |
| **H3** | LIVE tick-level best_price 추적 시 trail exit 타이밍이 BT와 0.3% 이내로 근접 | WebSocket/REST polling 실험 데이터 |
| **H4** | sl_trail_tuning 결과의 robustness가 intrabar-adjusted에서도 유지 (baseline 우위) | top-3 combos intrabar 재평가 |
| **H5** | LIVE tick best_price는 운영 비용(rate limit, 코드 복잡도) 대비 PnL 개선 효과가 ≥ 1pp/월 | A/B 테스트 30일 이후 |

---

## 4. Goal 메트릭 & 성공 기준 (GO 조건 8개, `train_not_degraded` 포함)

적용 대상: **Track A intrabar-BT 적용 후 sl_trail_tuning 재평가 시점**.

1. **intrabar_realism**: intrabar-BT의 19-trade 기간 PnL이 LIVE 실제 -3.92%와 **±3pp 이내** 근접
2. **baseline_preservation**: intrabar-BT에서 baseline `(3.3, 2.5, 192)` PnL이 +150% 이상 (clean +170%의 88%+)
3. **wf_pass**: baseline이 intrabar WF 5/5 양수
4. **ratio_ok**: intrabar-BT의 baseline PnL/MDD ≥ clean 값의 85% (슬리피지 반영 후도 합리적)
5. **track_b_cost**: Track B 구현 시 REST/WebSocket rate 일일 호출 ≤ 10,000 (BingX 제한 내)
6. **track_b_benefit**: Track B 운영 시 trail exit PnL 개선 평균 ≥ +0.2pp/trade
7. **rollback_ready**: Track A/B 모두 config-driven on/off, 즉시 원복 가능
8. **train_not_degraded**: intrabar-BT 적용 후 train 구간 성과가 clean BT 대비 ≤ -5pp 이내

7/8 이상 충족 시 GO. 핵심 조건 1, 2, 3, 8 중 하나라도 실패 시 STOP.

---

## 5. Track A: BT Intrabar Model

### 5.1 접근법 — 5m sub-bar traversal

15m bar를 3개의 5m sub-bar로 분해:
```
15m bar at T:
  sub-bar 0: T + 0:00  (5m OHLC)
  sub-bar 1: T + 5:00  (5m OHLC)
  sub-bar 2: T + 10:00 (5m OHLC)
```

각 sub-bar에 대해:
- Entry 이후 각 sub-bar에서 SL/emergency 체크 (기존 BT와 동일)
- `best_price` 를 sub-bar `high` (LONG) 또는 `low` (SHORT) 로 갱신
- Sub-bar close 시점에 `check_exit` (trail 포함) 호출

**효과**: trail 평가 주기 15m → 5m. LIVE의 tick 해상도에는 미치지 못하지만 3배 정밀도 향상.

### 5.2 Slippage 모델

Round-trip 평균 slippage 주입 (측정 기반):
- Entry: MARKET → 방향당 0.15% adverse (측정 0.287% 의 절반, MARKET 평균치)
- Exit SL: STOP_MARKET → 0.30% adverse (측정 0.641% × SL 비율)
- Exit TRAIL: per-cycle re-place → 0.15% adverse
- Exit EMERGENCY: 0.50% adverse (worst slippage)

Config-driven:
```yaml
backtest:
  intrabar:
    enabled: true
    sub_bar_minutes: 5
  slippage:
    entry_pct: 0.15
    exit_sl_pct: 0.30
    exit_trail_pct: 0.15
    exit_emergency_pct: 0.50
```

### 5.3 위험 & 완화

| 위험 | 완화 |
|------|------|
| 5m 데이터 캡처 품질 (결측/중복) | `data/btc_5m_270days_reclassified.csv` 기존 검증된 데이터 사용 |
| Sub-bar path 가정 (o→h→l→c or o→l→h→c) | 두 경로 모두 시뮬 + 평균 / conservative (worse path) 선택 |
| 기존 파라미터 연구 결과 침해 | Clean BT 결과 별도 보존, intrabar BT는 보완 지표로만 |

---

## 6. Track B: LIVE Tick-level best_price

### 6.1 접근법

현재: 15분마다 `best_price = max(best, candles['high'][bar])` 갱신 (bar close 기준).

개선: 5m 주기 REST polling OR WebSocket ticker subscription → best_price 수시 갱신.

```python
# Option 1: 5m REST polling (safer, rate limit 내)
async def _poll_best_price(self, pos):
    while pos in self.positions:
        ticker = await self.exchange.fetch_ticker('BTC/USDT:USDT')
        cur_high = ticker['high']  # 24h high (부적합)
        # → ohlcv limit=1 5m: 현재 5m 봉 high/low
        ohlcv = await self.exchange.fetch_ohlcv('BTC/USDT:USDT', '5m', limit=1)
        new_high = ohlcv[-1][2]; new_low = ohlcv[-1][3]
        if pos['direction'] == 'LONG':
            pos['best_price'] = max(pos['best_price'], new_high)
        else:
            pos['best_price'] = min(pos['best_price'], new_low)
        # Trail trigger check + STOP_MARKET 재배치 if needed
        await asyncio.sleep(300)

# Option 2: WebSocket trades stream
async def _ws_best_price(self, pos):
    async for trade in self.exchange.watch_trades('BTC/USDT:USDT'):
        price = trade['price']
        if pos['direction'] == 'LONG':
            pos['best_price'] = max(pos['best_price'], price)
        # ...
```

### 6.2 Cost-benefit

| 지표 | 목표 | Option 1 (5m REST) | Option 2 (WebSocket) |
|------|------|--------------------|-----------------------|
| 정밀도 | tick-level | 5m bar 단위 | tick 단위 |
| Rate limit | ≤ 10K/day | 288회/day × 포지션 | 무관 (stream) |
| 구현 복잡도 | 낮음 | **저** | 고 |
| 네트워크 안정성 | 봇 다운 영향 | 제한적 | 끊김 시 재연결 필요 |
| BUG 리스크 | 낮음 유지 | **저** | 고 |

**Option 1 (5m REST) 우선 권장**. BT의 5m sub-bar 모델과도 일관.

### 6.3 적용 범위

- Trail activation 이후 (best_pnl > 0.05%) 만 활성화 — activation 전엔 거래소 TRAILING_STOP_MARKET이 보호
- Fractal SL은 변경 없음 (항상 거래소 STOP_MARKET)

---

## 7. Implementation Plan

### Phase 1: Track A BT Intrabar Model (1~2일)
1. `scripts/analysis/c1_intrabar_backtest.py` 신규 — 5m sub-bar traversal + slippage
2. Baseline/candidate combo 재평가
3. LIVE 19-trade 기간에 대해 intrabar-BT 실행하여 gap 측정
4. 결과 검증: sl_trail_tuning 결과 robustness 유지 확인

### Phase 2: Track B LIVE Tick Tracking (3~5일, Phase 1 이후)
1. `scripts/production/c1_breakout/bot.py` 에 `_poll_best_price_5m` 추가 (Option 1)
2. Config flag `tick_best_price.enabled=false` 기본값 (opt-in)
3. 백테스트 없이 운영 실측 A/B 준비
4. 30일 A/B 테스트 (enabled=true on/off 2주씩)

### Phase 3: GO/STOP 판정 (Phase 1 이후)
1. 7/8 GO 조건 평가
2. GO 시: Track A를 `research_protocol_overfit_guards.md`에 표준 편입
3. Track B 판정은 Phase 2 후 별도

---

## 8. Non-Goals

- 1m 데이터 수집 (비용 대비 효과 미확인, 추후 고려)
- Full tick data 백테스트 (데이터 확보/저장 비용 과다)
- Order book microstructure 모델 (범위 이탈)
- Smart order routing (MARKET vs LIMIT 최적화는 별도 PDCA)

---

## 9. Rollback

### Track A
- `scripts/analysis/c1_intrabar_backtest.py` 삭제만 하면 기존 BT에 영향 없음
- `research_protocol_overfit_guards.md`에 편입했다면 해당 섹션만 되돌림

### Track B
- Config `tick_best_price.enabled: false` 로 즉시 비활성화
- 코드 변경은 opt-in 분리 메서드로 최소화 — 메인 루프 영향 없음

---

## 10. Reference

- `claudedocs/BACKTEST_LIVE_PARITY.md` — 22-item 체크리스트 (#21, #22)
- `claudedocs/bt_live_gap_deep_review_20260419.md` — 심층 분석
- `results/live_vs_backtest_verification.json` — 19-trade 매칭 데이터
- `scripts/analysis/intrabar_trail_impact.py` — 기존 intrabar 실험 (재활용)
- `results/intrabar_trail_impact.json` — 이전 BT-only intrabar 결과
- `data/btc_5m_270days_reclassified.csv` — 5m 데이터
- `memory/bt_live_gap_20260419.md`, `memory/research_protocol_overfit_guards.md`
