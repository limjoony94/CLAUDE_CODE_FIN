# M3-R7 — ψ + τ Specs (사전 등록)

> **Date**: 2026-04-28
> **Frame**: 사용자 명시 명령 ("계속 후보 탐색"). 13/13 dies 후 새 axes.

## Spec ψ (pre-funding-settlement window × funding extreme)

**Hypothesis**: BingX funding은 0/8/16 UTC settle. 직전 1h (7, 15, 23 UTC)에 funding extreme positions이 강제 unwound 가능 → 가격 변동 큼. Time × funding compound이 standalone funding (γ, ξ, μ)보다 informative 가능.

**Entry rule (LONG)**:
- hour_utc ∈ {7, 15, 23} (settlement 직전 1h)
- AND funding_8sum ≤ -0.24 (shorts crowded → settlement squeeze 가능)
- AND BTC return prev > 0
- AND 1h+4h trend LONG

**Entry rule (SHORT)**: mirror.

**Distinction**: γ는 funding × cross-asset, μ는 funding momentum, ξ는 funding × ETH break. ψ는 funding × **time-of-day axis**. 새 third axis.

**Exit**: α 표준.

## Spec τ (3-bar reversal + cross-asset)

**Hypothesis**: 3개 연속 같은 방향 바 후 4번째 바 reversal (engulfing 또는 strong opposite close)는 momentum exhaustion + reversal signal. ETH same-direction이 mean-rev confirmation 추가.

**Entry rule (LONG)**:
- bars [i-3, i-2, i-1] 3개 모두 close < open (3 down bars)
- AND bar [i] close > open AND close > prev 3 bars의 max(close)
- AND ETH return prev > 0 (ETH already turning)
- 트렌드 무관 (counter-trend)

**Entry rule (SHORT)**: mirror.

**Distinction from σ**: σ는 ETH break + RSI. τ는 BTC bar pattern + ETH return. 다른 axes.

**Exit**: α 표준.

---

## Predictions

### ψ
| Critique | Predicted | Confidence |
|----------|-----------|-----------|
| C1 | borderline | LOW — small sample (3 hours/day × 166 days × ~conditional) |
| C3 | FAIL | HIGH |

### τ
| Critique | Predicted | Confidence |
|----------|-----------|-----------|
| C1 | borderline | LOW — multi-bar pattern × cross-asset untested |
| C3 | FAIL | HIGH (counter-trend) |

**Most likely outcome**: 15/15 die.

## Stop conditions

- 0/2 PASS: 사용자에게 진척 상황 보고 + 추가 axes 의향 확인.
- 1/2 PASS C5: deep verify.
