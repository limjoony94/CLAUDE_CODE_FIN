# M3-R6 — υ + χ Specs (사전 등록)

> **Date**: 2026-04-28
> **Frame**: 사용자 명시 명령 ("계속해서 후보 탐색"). Advisor "no R6" 권고를 user override.
> **Origin**: 11/11 dies. 새 axes 시도: volume × cross-asset + intra-bar wick pattern × RSI.

## Spec υ (volume spike + ETH cross-asset confirmation)

**Hypothesis**: Volume spike (volume[i] > 2× SMA20)는 정보 풍부 event. R2 volume class는 standalone fail. 그러나 **volume spike + ETH 같은 방향**으로 narrowing 시 informational content 강화 가능. Cross-asset confirmation이 noise filter 역할.

**Entry rule (LONG)**:
- BTC 15m volume[i] > 2.0 × volume_SMA20[i] (volume spike)
- AND ETH return prev > 0.2% (ETH same-direction)
- AND BTC return prev > 0 (BTC also up)
- AND 1h+4h trend LONG

**Entry rule (SHORT)**: mirror.

**Distinction from R2 volume class**: R2는 BTC volume 단독. υ는 BTC volume × ETH return × BTC return triple compound. 다른 narrowing.

**Exit**: α 표준.

## Spec χ (lower wick rejection at RSI oversold)

**Hypothesis**: 긴 lower wick (>40% of range) + close near high + RSI ≤ 35 = "panic dip 후 즉시 매수" event. Mean-rev path지만 intra-bar wick 정보가 추가 confirmation 제공. σ (counter-trend at break)는 C1 PASS but C3 worst — wick은 다른 confirmation axis.

**Entry rule (LONG)**:
- BTC 15m: (low_wick / range) ≥ 0.40 where low_wick = min(open, close) − low
- AND close in upper 30% of bar range (recovery)
- AND RSI(14) ≤ 35
- AND BTC return prev < 0 (recent down — drop being bought)
- 트렌드 무관 (counter-trend)

**Entry rule (SHORT)**: mirror (upper wick + close in lower 30% + RSI ≥ 65 + return > 0).

**Distinction from σ**: σ는 ETH structural break + RSI. χ는 BTC 자체 intra-bar wick pattern + RSI. 다른 confirmation source.

**Exit**: α 표준.

---

## Predictions

### υ (volume × cross-asset)
| Critique | Predicted | Confidence |
|----------|-----------|-----------|
| C1 | borderline FAIL | MED — volume class 4 rounds fail track record |
| C3 | FAIL | HIGH |

### χ (wick rejection)
| Critique | Predicted | Confidence |
|----------|-----------|-----------|
| C1 | borderline | LOW — wick patterns untested at this magnitude threshold |
| C3 | FAIL | HIGH (counter-trend = σ pattern) |

**Most likely outcome**: 13/13 die — convergent evidence 더 강해짐.
**Most likely surprise**: χ가 wick + RSI compound 시 sample size 적게 + magnitude 강한 alpha 발견.

## Stop conditions

- 0/2 PASS: 13-mechanism cumulative update + 사용자 옵션 재제시.
- 1/2 PASS C5: deep verify.

## Anti-pattern guard

- 사용자가 R7 명시 시까지 R6에서 stop.
- "fix-impulse on 11 prior fails" 회피 — 두 spec 모두 새 axis만.
