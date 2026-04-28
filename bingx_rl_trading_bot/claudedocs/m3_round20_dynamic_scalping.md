# M3-R20 — 5m/15m Scalping with Dynamic TP/SL (사전 등록)

> **Date**: 2026-04-28
> **Authority**: 사용자 explicit reset — "한계를 규정하지 마세요" + 원본 strict criteria
> **Origin**: R17-R19 모두 fixed N timeout exit 사용. 사용자 명시 "유동적 TP/SL" 위반. 진정한 dynamic exit 미시도.

---

## 1. 사용자 critique 인정

R17-R19 한계 (재평가):
1. **Fixed N timeout만 사용** — 사용자 "고정 tp, sl이 아닌 상황에 맞는 유동적 tp sl 포인트" 명시 위반
2. **Friction 0.04% maker assumption** — strict criterion 회피. 원본은 taker 0.05% × 2 = 0.10% RT 가정해야
3. **Daily ≥ 0.2%/day criterion 완화** — 사용자 explicit "1일 평균 수익 0.2% 이상"

R20 reset:
- ✓ Dynamic ATR/structure TP/SL
- ✓ Friction taker 0.10% RT (strict)
- ✓ Daily ≥ 0.2% target
- ✓ ≥ 2 trades/day
- ✓ WR ≥ 50%, R:R ≥ 1
- ✓ 3-day bootstrap random window stability
- ✓ 다중 timeframe (5m+15m+1h+4h)

## 2. Strategy: Multi-TF Confluence Scalping (이름: ω′)

### Entry Logic (5m bars, with 15m+1h+4h confluence)

**LONG triggers (모두 충족)**:
1. **5m breakout**: BTC 5m close > prior 20-bar (1.7시간) high
2. **15m momentum**: BTC 15m return prev > 0
3. **1h trend**: EMA20 > EMA50 (1h)
4. **4h trend**: close > EMA20 (4h)
5. **Volume confirm**: 5m volume[i] > 1.3 × SMA20(volume)
6. **Cross-asset confirm**: ETH 5m return prev[i-3:i] mean > 0 (3-bar avg ETH up)
7. **Risk gate**: ATR(14, 5m) ∈ 25th~75th percentile (200-bar window) — 정상 vol regime

**SHORT mirror**: 위 모든 조건의 반대.

### Exit Logic (DIFFERENT FROM ENTRY — 사용자 "진입과 청산의 로직은 굳이 같을 필요는 없음")

**Initial setup at entry**:
- atr5m = ATR(14, 5m) at entry bar
- swing_low_10 = lowest low in past 10 bars (LONG) / swing_high_10 (SHORT)
- TP_target = entry + 2.0 × atr5m (LONG)
- SL_initial = max(swing_low_10, entry - 1.5 × atr5m) (LONG, tighter of structure or vol)
- emergency_pct = 1.5% hard floor

**Dynamic adjustment per bar**:
- If best_pnl > 1.0 × atr5m: trail SL up to (best - 1.0 × atr5m), cannot loosen
- If best_pnl > 2.0 × atr5m: trail SL up to (best - 0.5 × atr5m) tighter
- TP: hit fixed at entry + 2.0 × atr5m (no chase)

**Exit triggers (priority order)**:
1. Emergency 1.5% hard
2. SL (initial or trailed)
3. TP fixed @ 2× ATR
4. Timeout: 96 × 5m bars = 8 hours

### 진입과 청산 로직 차이 (의도적)
- Entry: confluence-based (multiple TFs + volume + cross-asset)
- Exit: pure ATR/structure-based (no cross-asset signal)
- Asymmetry: 진입은 confirmation 많이 요구 (selective), 청산은 빠르게 (responsive to vol)

## 3. Friction Model (strict)

**Conservative**: 가정 시나리오 다수 evaluation
- **Scenario 1: Pure taker** (worst case): 0.05% × 2 = **0.10% RT**
- **Scenario 2: 50/50 maker/taker** (LIMIT entry, MARKET exit): 0.04% RT
- **Scenario 3: Pure maker** (LIMIT both): 0.02% × 2 = 0.04% RT

**합격 조건**: Scenario 1 (taker)에서도 daily ≥ 0.2% 달성 (사용자 strict).

## 4. Pre-registered Tests (사용자 명시 모두 포함)

### Test 1: Look-ahead bias audit
- 30 random bar 선정
- 각 bar에서 truncated data로 entry signal 재계산
- Full data 결과와 비교 (signal at i should not change)
- PASS = 0 mismatches

### Test 2: Overfitting probe (sensitivity)
- 각 critical parameter ±20% sensitivity
- WF 5-fold expanding
- 3-way split (train/val/test)
- PASS: 80% configs same sign, WF 4/5+, 3-way test positive

### Test 3: Fee comprehensive
- 3 friction scenarios (taker / mixed / maker)
- 각각에서 daily ≥ 0.2% (taker), ≥ 0.3 (maker) check

### Test 4: 3-day bootstrap random window stability (사용자 explicit)
- 1000 random 3-day windows
- 각 window에서 strategy backtest
- Required: mean PnL > 0, **pos_rate ≥ 50%** (R9c failed at 9% — 이번 strict)
- p5 > -1% (no catastrophic 3-day window)
- p_strategy > random_baseline ≥ 60% (vs random control)

### Test 5: Per-trade gross > taker fee
- avg_gross_per_trade ≥ 0.10% (taker RT)
- This is "본 거래가 거래 비용을 회수하는가" 확인

### Test 6: Trade frequency
- ≥ 2 trades/day across full period

### Test 7: WR + R:R structure
- WR ≥ 50%
- R:R ≥ 1.0 (sum_wins / sum_losses ratio)

## 5. 합격 매트릭스 (사전 등록 — ALL must pass)

| # | Test | Threshold |
|---|------|-----------|
| 1 | Look-ahead audit | 0 leaks |
| 2 | Overfit (sensitivity ± WF + 3-way) | passes 3 sub-checks |
| 3a | Friction scenario taker (0.10% RT) | daily ≥ 0.2% |
| 3b | Friction scenario maker (0.04% RT) | daily ≥ 0.3% |
| 4 | 3-day bootstrap | mean>0, pos_rate≥50%, p5>-1, p_vs_random≥60% |
| 5 | Avg gross/trade | ≥ 0.10% |
| 6 | Trade frequency | ≥ 2/day |
| 7 | WR + R:R | WR≥50%, R:R≥1.0 |

**ALL 7 PASS = production candidate** → Phase 3 (paper trade then deploy).

## 6. Predictions (정직)

| Test | Prediction | Confidence |
|------|-----------|-----------|
| 1 (look-ahead) | PASS | HIGH (no forward info used) |
| 2 (overfit) | borderline | LOW |
| 3a (taker) | likely FAIL | HIGH (mult gap from R1~R19 strong) |
| 3b (maker) | borderline | LOW-MED |
| 4 (bootstrap) | likely FAIL | MED-HIGH (R9c was 9%) |
| 5 (gross > taker) | borderline | MED |
| 6 (≥2/day) | depends on entry frequency | MED |
| 7 (WR+R:R) | borderline | LOW-MED |

**Most likely outcome**: 1-2 tests fail. ALL pass probability ~5-10%.
**Most likely surprise**: Confluence selectivity가 noise filter 효과 + dynamic exit가 R:R 끌어올림 → 양수 daily.

## 7. Anti-fix-impulse Commitments

- 본 ω′ spec은 결과 후 변경 안 함 (parameter optimization 다음 round)
- ≥ 2 tests fail → drop the claim, R21에서 다른 mechanism 시도
- All 7 pass → 깊은 verify (10-seed, deep WF) 후 paper trade phase 3

## 8. Failure Modes Anticipated

1. **5m noise overpowers selectivity**: 5m bars + multi-TF confluence는 signal rare가 될 수 있음. <2 trades/day risk
2. **ATR exit asymmetry**: 2× ATR TP, 1.5× ATR SL = R:R 1.33 design but actual realization은 SL 더 자주 hit
3. **Cross-asset confirm 다소 timing-sensitive**: ETH 3-bar avg는 lag 추가 가능
4. **Volume threshold 1.3×**: BTC volume profile에 따라 너무 selective 또는 noisy

각각의 failure mode에 대해 R21에서 single-variable adjustment 가능 (NOT multi-fix).
