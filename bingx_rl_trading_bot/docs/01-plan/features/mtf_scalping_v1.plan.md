# Plan: MTF Trend-Aligned Scalping (M1)

> **Strategy**: Multi-Timeframe (5m/15m execution + 1h/4h trend filter), trend-aligned scalping
> **Date**: 2026-04-27
> **Phase**: Plan (사용자 위임 "자체적으로 추가 분석 개선 이후 진행")
> **Origin**: C1 Breakout v2.6 SHELVED 후 다음 전략 (사용자 결정: Momentum + Scalping 통합)
> **Goal**: 방법론 학습 + 통계적으로 안정적인 수익 전략 develop
>
> **Advisor 검토 (2026-04-27)**: 직관 기반 preemptive spec 변경(M1-A → v2)은 C1 함정 재발 = **거부**.
> 사용자 "자체 분석 개선" 지시를 **분석 파이프라인 강화**로 재해석 (Phase 0 신설). Spec M1-A as-is 유지.

---

## 1. Background

### 이전 PDCA loop (C1 Breakout) 의 lessons (모두 적용)
- BT theoretical (slippage 0)와 LIVE 실측은 다른 distribution
- WF가 same distribution sample이면 OOS 검증 약함
- Friction floor (~-0.5%/trade)가 strategy edge 잠식 가능
- 사전 등록 평가 (pre-registration) > 사후 reframe
- **Bootstrap 3-day random window 안정성이 overfitting 방지의 핵심**

### 새 전략 paradigm 차이
- **C1**: Channel breakout (15m), counter-noise SL
- **M1**: MTF trend-aligned scalping (5m/15m execution, 1h/4h filter)
- 다른 mechanism (breakout → trend-following), 다른 timeframe (15m → 5m + 1h confirm)

---

## 2. Pre-Registered Success Criteria (사용자 명시)

| # | Criterion | Threshold | Type |
|---|-----------|-----------|------|
| 1 | **WR (Win Rate)** | **≥ 50%** | Hard |
| 2 | **평균 R:R** | **≥ 1.0** (유동적, fixed X) | Hard |
| 3 | **TP/SL 방식** | 상황 적응 (volatility/structure-aware) | Design |
| 4 | **진입/청산 로직** | 다를 수 있음 | Design |
| 5 | **일일 평균 수익 (1x)** | **≥ +0.2%** | Hard |
| 6 | **거래당 평균 수익** | **> 왕복 taker fee (0.10%)** | Hard |
| 7 | **일일 거래 횟수** | **≥ 2** (통계적 유의성) | Hard |
| 8 | 🎯 **Bootstrap 3-day random window** | 통계적 안정성 | **CRITICAL** |

### Bootstrap Stability (Criterion 8 — 가장 중요)

> "랜덤 3일 데이터 캔들을 인풋 하는 랜덤 구간 테스트를 유의미한 통계적 횟수로 반복 진행하였을 때 통계적으로 안정적인 수익성을 발생시키는 전략"

**구체화**:
- 1000회 random 3-day window sampling
- 각 window 누적 PnL 측정
- **Core 3 metrics** (이전 `research_protocol_3day_bootstrap` 메모리):
  - Mean: 양수
  - Pos rate: ≥ 50%
  - P5: > -1% (catastrophic loss 회피)
- **Relative**: cand BT vs no-trade baseline (시장 변동성 vs strategy edge 분리)

소수 outlier에 의존 안 하는 안정적 전략 검증.

---

## 3. Single Concrete Spec — DRAFT (사용자 confirm 필요)

advisor 권고: "Pick one specific spec, BT it. Don't grid-search 50 variants."

### Spec M1-A: Trend-Aligned Pullback Scalping (제안)

```yaml
trend_filter:                # 1h + 4h alignment
  htf_1h: EMA20 > EMA50 → LONG bias (반대도 동일)
  htf_4h: close > 4h EMA50 (LONG confirm)
  required: 1h AND 4h 둘 다 align (없으면 wait, no entry)

entry_signal:                # 5m + 15m execution
  prerequisite: trend_filter 통과 시만
  ltf_5m:
    - RSI(14) crossed above 40 (LONG, 직전 3봉 내 ≤ 40 후 회복)
    - close > EMA9
    - body / range > 0.4 (body confirmation)
  ltf_15m:                    # buffered (Phase 0.2 data 근거)
    - LONG : 15m EMA9 ≥ 15m EMA21 × 0.999  (0.1% buffer above-or-near)
    - SHORT: 15m EMA9 ≤ 15m EMA21 × 1.001  (symmetric)
  → entry on next 5m bar open

exit:                        # 유동적 (criterion 3, 4)
  TP_trail: best_price - 2.0 × 5m_ATR  # adaptive
  SL: max(직전 5m swing_low, entry - 1.5 × 5m_ATR)  # structure + ATR cap
  emergency_sl: -1.5% hard (slippage 회피)
  timeout: 24 bars (= 2h, 5m × 24)

position:
  N=1, leverage=1x (사용자 명시 1x 평가)

frequency:
  min_bars_between_trades: 2
```

### Why this spec

- **Trend filter (1h+4h)**: 5m noise 회피. C1의 85.7% wick은 noise heavy 환경. Trend confirmation 필수.
- **5m+15m entry**: 사용자 명시 timeframe. 15m은 secondary confirmation.
- **Pullback (RSI 40 회복)**: trend 안에서 mean-rev 진입 → R:R 좋음.
- **Body 40%**: C1과 동일, momentum confirmation.
- **Trail TP + Structural SL**: 유동적 (criterion 3 충족). ATR-based trail = volatility adaptive.
- **Timeout 2h**: scalping이라 짧게. friction floor 대비 성공률 ↑.
- **Emergency 1.5%**: 거래소 STOP_MARKET, slippage 회피.

### 3.1 Spec Evolution Log (data-driven changes only)

| Date | Change | Evidence | Rationale |
|------|--------|----------|-----------|
| 2026-04-27 | 15m role: strict EMA9>EMA21 → **buffered (0.1% symmetric)** | `results/m1_entry_15m_role_compare_20260427_211108.json` | D1 strict 1.73/d FAIL criterion 7. D3 buffered 4.52/d PASS, selectivity D4(9.27/d)대비 절반 보존. 사용자 spec literal "5m/15m 참고"에 부합 (alignment 강제 X, soft veto). |

> 향후 spec 변경은 모두 (a) Phase X.X 측정 결과 기반 + (b) advisor 검토 + (c) 본 evolution log 기록. 직관 기반 preemptive 변경 금지 (C1 함정 lesson).

### 대안 spec (참고만, BT 안 함 — variant search 회피)

| 대안 | 차이 |
|------|------|
| M1-B (Donchian) | Trend filter를 Donchian channel break으로 |
| M1-C (RSI extremes) | Entry를 RSI < 30 (LONG) 또는 > 70 (SHORT) — 더 selective |
| M1-D (VWAP) | VWAP 기반 mean-rev scalping |

→ M1-A 통과/실패 후 결정.

---

## 4. Design Constraints (5 must-have, C1 lessons)

1. ✅ **Friction-aware BT** — slippage 0.1%/trade (entry+exit 합계) inject
2. ✅ **Distribution check** — Trade #20-30에 BT rolling 14d distribution percentile 체크 의무
3. ✅ **Single-variable change** — production 변경 시 24h 간격
4. ✅ **Control group** — buy-and-hold, random entry baseline 동시 측정
5. ✅ **Out-of-regime 검증** — WF + bootstrap stability + 별도 OOS

---

## 5. Implementation Plan (Single concrete spec — variant search 회피)

### Phase 0: 분석 파이프라인 강화 (1일, NEW — advisor 권고)

**목적**: spec preemptive 변경 회피, 데이터 기반 결정 토대 구축.

#### 0.1 데이터 정합성 검증
- 5m / 15m / 1h / 4h timestamp alignment (UTC, missing bar 검증)
- 사용 데이터: `data/btc_5m_720days_binance.csv` (207,360 bars), `btc_15m_720days.csv`, `btc_1h_720days.csv`
- 4h는 1h resample (last 4h close align)
- Output: `results/m1_data_integrity_*.json`

#### 0.2 Entry-frequency sanity check (BT 아님, 1초 query)
- M1-A spec **그대로** 5m 캔들에서 entry condition pass 횟수 count
- 분해: trend_filter pass% / + RSI cross pass% / + body+EMA9 pass% / + 15m alignment pass%
- **GO 조건**: 일평균 ≥ 2 entries (criterion 7)
- **FAIL 시**: spec 너무 strict → 사용자 보고 후 결정 (filter 완화 vs 다른 paradigm)
- Output: `results/m1_entry_frequency_*.json`

#### 0.3 Friction model 사전 등록
- Entry MARKET slippage: 0.05% (C1 LIVE 측정 평균 ~0.03%, 보수적)
- Exit MARKET slippage: 0.05% (TP/SL/Emergency 모두 MARKET)
- Round-trip fee: 0.10% (taker 0.05% × 2)
- **Total friction floor**: -0.20% / trade
- 거래당 평균 수익이 0.20% 초과해야 양수 (criterion 6)
- Output: `claudedocs/m1_friction_model.md`

#### 0.4 Baseline 정의 사전 등록 (Bootstrap relative 기준)
- **Baseline A (no-trade)**: 0% / 3-day window
- **Baseline B (buy-and-hold)**: BTC 3-day return
- **Baseline C (random entry N=1)**: 5m 캔들에서 균등분포 entry, 같은 exit logic 적용
- Bootstrap에서 cand vs each baseline 비교
- Output: `claudedocs/m1_baseline_definition.md`

### Phase 1: BT Framework 구축 (1~2일)
- 5m + 15m + 1h + 4h MTF 데이터 fetch (BingX or Binance)
- check_entry / check_exit 코드 (C1 framework 재사용 가능 부분)
- Friction model (entry slip 0.05%, exit MARKET slip 0.05%)

### Phase 2: M1-A spec BT (1일)
- 272d full BT
- 8 success criteria 측정
- Reason 분포 (TP/SL/timeout)

### Phase 3: Bootstrap Stability Test (1일) — CRITICAL
- 1000 random 3-day windows
- Core 3 metrics + Relative
- Pass/Fail 명시

### Phase 4: WF + 3-way split (1일)
- 5-fold expanding WF
- Train/Val/Test split
- Sensitivity ±10% (parameter robustness)

### Phase 5: GO/NO-GO 결정
- 8/8 criteria 충족 → Phase 6 paper test
- 6~7/8 → variant 1개 검토 (M1-B 등)
- < 6 → Strategy shelve, 다른 paradigm

### Phase 6: Paper test (testnet 1~2주, no live capital)
- LIVE 환경 friction 측정
- Distribution check (사전 등록 lesson)
- LIVE 14d BT distribution 비교

### Phase 7: Live deployment (capital 0%부터 시작)
- 처음 1주 minimum size ($100)
- Trade #20에 distribution check
- 통과 시 size 점진 증가

---

## 6. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Spec parameter overfitting | Single spec only, variant search 금지 |
| BT-LIVE friction gap | Phase 1에서 friction-aware BT 의무 |
| Distribution mismatch | Phase 7 trade #20 distribution check |
| Anxiety pattern (즉시 deployment) | Paper test 1~2주 의무 |
| Drawdown 깊음 | Emergency 1.5% hard, max risk per trade $20 (1.5% of $1495) |

---

## 7. Hard Stops (자동 halt criteria)

LIVE 운영 중 다음 시 즉시 halt:
- Trade #20에 distribution check fail (LIVE < BT P5 14d window)
- 5 consecutive SL hits
- Daily PnL 1x ≤ -3% in any single day
- WR < 30% in last 20 trades

---

## 8. Open Questions (사용자 confirm 필요)

1. Spec M1-A 이대로 진행? 또는 변형?
2. Higher TF 1h만으로 충분? 4h도 필수?
3. Lower TF 5m primary, 15m confirmation 맞나?
4. Trail K=2.0, SL ATR×1.5 이 합리적? (BT 후 calibrate 가능)
5. Bootstrap 1000회 충분? (이전 protocol에서는 1000회였음)

---

## 9. Reference

- Lessons: `lessons_distribution_check_20260427.md`, `lessons_process_audit_20260425.md`, `research_protocol_3day_bootstrap.md`
- Postmortem: `c1_breakout_postmortem_20260427.md`
- Reusable code: `scripts/production/c1_breakout/indicators.py` (ATR, EMA 계산 등)
