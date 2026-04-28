# M3-R13 — Orthogonal Axes (사전 등록)

> **Date**: 2026-04-28
> **Authority**: 사용자 명시 — "캔들 데이터로 수익성 모델 찾을 때까지"
> **Origin**: 17 mechanisms × 12 rounds 모두 15m signal × hard threshold 패턴. 진짜 미테스트 axes로 이동.

---

## 1. 진짜 미테스트 axes 정리

| Axis | 12 rounds tested? |
|------|-------------------|
| 15m signal generation | ✓ (전부) |
| 1h signal generation | ❌ |
| 4h signal generation | ❌ |
| 1d signal generation | ❌ |
| Hard thresholds | ✓ |
| Continuous prediction (regression) | ❌ |
| Position size variation | ❌ (1× only) |
| Multi-position N>1 | ❌ |

R13에서 2개 axis 시도:
- **δ — 1h signal generation** (lower frequency, fewer noise events)
- **ε — Continuous regression-based signal** (no hard threshold)

## 2. Spec δ (1h timeframe ETH-lag mechanism)

**Concept**: α/ι의 ETH-lag concept을 1h timeframe으로 이동. 
- 15m bars → 1h bars (4× aggregation)
- Filter: 4h + 1d trend (bigger picture)
- Same ETH-lag idea but at lower frequency

**Hypothesis**: 15m noise saturates entry alpha. 1h signals are more meaningful. Lower frequency → friction-frequency profile better.

**Entry rule (LONG)**:
- ETH 1h return prev > 0.6% (2× of 15m 0.3% — scale-adjusted)
- AND BTC 1h return prev < 0.2% (2× of 15m 0.1%)
- AND BTC ATR(14, 1h) > 70th percentile of 200 bars
- AND 4h+1d trend LONG

**Entry rule (SHORT)**: mirror.

**Exit**: fixed N=4 bars (= 4 hours, similar concept) timeout-only, no trail/SL, emergency 1.5%.

**Friction**: 0.04% maker × 2 = 0.08% RT (single-leg perp, like prior tests).

## 3. Spec ε (Continuous regression-based signal)

**Concept**: 모든 prior mechanisms는 hard threshold (ETH > 0.3, RSI < 30 등). Linear regression on next-N return with 5+ features → continuous prediction → sign of prediction = direction, magnitude = confidence.

**Features** (15m bar at time t-1, predict t to t+N return):
1. eth_return[t-1]
2. btc_return[t-1]
3. atr14_normalized[t-1]
4. rsi14[t-1]
5. log_ratio[t-1] (BTC/ETH log ratio)
6. ratio_z[t-1]

**Target**: BTC return from open[t+1] to close[t+N], where N=4 (predict 1h forward).

**Training**:
- Train on first 60% of data (~432 days)
- Fit OLS regression (sklearn LinearRegression)
- Save coefficients

**Trading rule** (test on last 40%):
- Predict t+1 to t+4 return for each bar
- If prediction > +X (e.g., 0.20%) → LONG entry, exit N=4 bars later
- If prediction < -X → SHORT entry, exit N=4 bars later
- 1h+4h trend filter applied

**Friction**: 0.04% × 2 = 0.08% RT.

## 4. 합격 조건 (사전 등록)

For both δ and ε:
1. OOS daily_net > 0 @ 0.04% friction
2. n_OOS ≥ 50 trades
3. WR ≥ 40%
4. RR ≥ 1.0
5. WF 5-fold (within OOS): 3/5 positive
6. Bootstrap pos_rate ≥ 30%

For ε specifically:
7. Sign consistency: sign(prediction) and sign(actual) agreement > 50% (i.e., regression has some forecast skill, not just noise)

## 5. Predictions

| Mechanism | Predicted | Confidence |
|-----------|-----------|-----------|
| δ (1h signal) | borderline | LOW — lower frequency 자체가 alpha 보장 안 함. 단 untested timeframe 이므로 가능성 있음 |
| ε (regression) | likely FAIL | MED — linear regression on cryptos는 거의 모두 fail. Sign skill ~52% rare |

**Most likely outcome**: 둘 다 fail. → R14 다음 axes (4h, 1d, multi-position)
**Most likely surprise**: δ가 lower-frequency clean signal 발견 — α의 1h 버전이 production-grade 가능

## 6. Anti-fix-impulse commitment

본 R13 결과 무관하게:
- Same axes (1h timeframe, regression) 추가 sweep 안 함
- δ가 PASS 시 deep verify pre-reg → OOS strict test
- 둘 다 fail 시 R14 (4h/1d timeframe + multi-position)
- 사용자 "수익성 모델까지" mandate 따라 자동 진행
