# M3-R12 — Paradigm Shift (사전 등록)

> **Date**: 2026-04-28
> **Authority**: 사용자 옵션 C 명시 + "수익성 있는 모델을 찾아낼 때까지 계속 진행"
> **Origin**: 17 directional mechanisms × 11 rounds + R10 multi-dim grid + R11 reversal hypothesis = directional alpha paradigm 전부 fail. 다른 return source 시도.

---

## 1. Paradigm shift 정의

**기존 paradigm** (R1~R11): Directional alpha — 가격 방향 예측해서 LONG/SHORT 베팅
**새 paradigm 후보**:
- A. **Market-neutral mean-reversion** (pair trade): spread 자체 mean-revert, 방향 무관
- B. **Yield/carry harvesting** (funding rate): funding rate 수령, 방향 hedge
- C. (추후) Cross-exchange basis arb — 데이터 부재 (single source 의심)
- D. (추후) Volatility selling — options 데이터 없음

R12에서 A, B 두 paradigm 사전 등록 후 검증.

## 2. Spec π* (Pair trade — true market-neutral)

**기존 β와 차이**:
- β: spread z-score extreme → 같은 방향으로 BTC만 directional 베팅
- π*: spread extreme → BTC LONG + ETH SHORT (또는 반대) — **순 delta = 0**

**Mechanism**:
- log_ratio = log(BTC_price / ETH_price)
- z = (log_ratio - rolling_mean) / rolling_std
- Entry: |z| ≥ 2.5 (extreme deviation)
- 방향: z > 0 (BTC 비싸짐) → SHORT BTC + LONG ETH (=1:1 dollar-neutral)
- 방향: z < 0 (BTC 싸짐) → LONG BTC + SHORT ETH
- Exit: |z| ≤ 0.5 (mean reverted) OR timeout 24h
- Hedge ratio: 1:1 dollar-neutral (entries에 같은 USDT notional)

**Friction**: 0.04% maker × 2 legs × 2 (open+close) = **0.16%/round-trip**.

**합격 조건** (사전 등록):
1. Net daily PnL > 0 @ 0.16% RT friction
2. Pair trades/day ≥ 0.5 (≥ 1/2-day minimum)
3. WR ≥ 50% (mean-rev은 trend-following과 달리 high WR 가능)
4. WF 5-fold 3/5 positive
5. 3-way test split positive
6. Bootstrap 200 windows pos_rate ≥ 50%

## 3. Spec ω* (Funding yield harvest — directional 무관)

**Mechanism**:
- BingX BTC perp funding 8h 마다 결제 (avg 0.01%)
- 가설: funding 절대값이 충분 크면, 받는 쪽에 sit + delta hedge로 carry 수확
- Simplified backtest (no spot data → synthetic hedge):
  - 매 8h funding 시점에 funding rate 부호 확인
  - funding > +threshold (longs pay shorts) → SHORT perp 진입, 8h 후 청산. funding 받음.
  - funding < -threshold → LONG perp 진입, 8h 후 청산.
  - 가정: 매 8h만 hold하므로 directional drift는 noise (mean-zero) — 단, 실제로는 변동
  - Net PnL = funding 받음 - directional drift - friction
- Threshold: 0.005, 0.01, 0.015, 0.02 (4 levels)

**Friction**: 0.04% × 2 (open+close) = 0.08% per cycle (1 leg, perp만).

**합격 조건**:
1. Net daily PnL > 0 @ 0.08% per-cycle friction at any threshold
2. Cycles/day ≥ 1 (즉 funding events 충분 활용)
3. WR ≥ 40% (funding 받음 + drift hedge 가정)
4. WF 5-fold 3/5+
5. Bootstrap pos_rate ≥ 50%

## 4. Predictions

### π* (pair trade)
| 조건 | Predicted | Confidence | Rationale |
|------|-----------|-----------|-----------|
| C1 daily > 0 | borderline | LOW | Spread이 진짜 mean-revert면 가능. 단 R6의 β가 spread-based directional fail이니까 spread 자체 mean-rev도 marginal일 수 있음 |
| C4 WF 3/5 | uncertain | LOW | Spread regime-dependent 가능 |

### ω* (funding harvest)
| 조건 | Predicted | Confidence | Rationale |
|------|-----------|-----------|-----------|
| C1 daily > 0 | likely FAIL | HIGH | 8h hold = ~0.5% std of price drift. Funding ~0.01% << drift std. Hedge 없이는 drift overpowers funding |
| 단 threshold 높이면 (0.02+) | maybe | LOW | Extreme funding은 capitulation event, drift bias 있을 수 있음 |

**Most likely outcome**: π* borderline OR fail, ω* fail. → 다음 paradigm으로 진행.

**Most likely surprise**: π* PASS — true market-neutral edge가 모든 directional fail와 무관하게 존재.

## 5. Stop conditions

- 0/2 PASS: 다음 paradigm 후보 (cross-exchange basis arb data 확보, 또는 calendar mean-rev with 1-7 day hold)
- 1/2 PASS C5: 사용자 결정
- 2/2 PASS: 둘 다 보고

**사용자 mandate "수익성 있는 모델 찾을 때까지" → R13 자동 진행 (다음 paradigm)**.
