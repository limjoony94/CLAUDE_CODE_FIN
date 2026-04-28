# M3-R5 — ρ×ι + σ Specs (사전 등록, LAST ROUND per advisor)

> **Date**: 2026-04-28
> **Frame**: 2 specs × 5 critiques. **마지막 round**.
> **Origin**: 9 mechanisms 누적 dies. ι (Δp50 +0.226) = strongest entry alpha but daily_net @ 0.20% friction = **-0.0453%/day** vs criterion +0.2%/day = +0.2453%/day gap (sign change + magnitude).
> **Advisor 권고**: ρ standalone (filter)는 entry alpha 없음 — ρ를 ι에 적용해서 narrowing이 magnitude 충분한지 검증이 informative.

## Spec ρ×ι (session-filtered ι — narrowing test)

**Hypothesis**: ι의 ETH structural break alpha가 특정 session (US 13-21 UTC, Asia 0-8 UTC, EU 8-13 UTC) 에 집중되어 있다면, session 제한으로 magnitude 강화 가능. Trade-off: sample size 1/3 가까이 감소.

**Entry rule (LONG)**:
- ι 모든 조건 (α 조건 + ETH 24-bar high break + 1h+4h LONG)
- AND hour-of-day ∈ session window

**Sessions tested**:
- US: 13:00–20:59 UTC
- Asia: 00:00–07:59 UTC
- EU: 08:00–12:59 UTC
- 24h baseline (= ι, 비교용)

**Best session 자동 선정 후 strict 5-critique**.

**Distinction**: ρ를 standalone mechanism이 아닌 **filter**로 적용. fix-impulse 우려는 있으나 mechanism class testing 차원.

**Exit**: α 표준.

## Spec σ (mean-reversion at structural break — untested class)

**Hypothesis**: ETH가 24-bar high break 시점은 momentum signal로 작용 가능 (ι의 가설), 또는 **counter-trend exhaustion** signal로 작용 가능 (정반대 가설). BTC RSI > 70 같이 overextension confirm 시 mean-rev 진입.

**Entry rule (SHORT)**:
- ETH 15m close > prev 24-bar high (ETH break)
- AND BTC RSI(14) ≥ 70 (overextended)
- AND BTC 15m return prev > 0 (recent up — exhaustion candidate)
- 트렌드 무관 (counter-trend mechanism)

**Entry rule (LONG)**: mirror (ETH break down + BTC RSI ≤ 30 + return down).

**Distinction from γ**: γ는 funding × cross-asset, σ는 ETH break × BTC RSI. 다른 axes.

**Distinction from ι**: ι는 trend-follow (with trend), σ는 trend-fade (counter-trend).

**Exit**: α 표준.

---

## Predictions

### ρ×ι
| Critique | Predicted | Confidence | Rationale |
|----------|-----------|-----------|-----------|
| C1 | session 중 1개 PASS 가능 | LOW | Sample 1/3 → noise vs concentration trade-off. selection-bias 위험 (best session 자동 선정 = data dredging) |
| C3 | FAIL even if C1 pass | HIGH | ι @ 0.20% = -0.045%/day. Sample 1/3 → daily 이미 (-0.045 × 3 trades/day → -0.015 × 1 trade/day) ≈ same → magnitude 변화 없음 |

### σ
| Critique | Predicted | Confidence |
|----------|-----------|-----------|
| C1 | borderline | LOW — counter-trend at structural break는 untested |
| C3 | FAIL likely | MED |

**Most likely outcome**: 11-mechanism 누적 11/11 die. Strong convergent finding.

**Most likely surprise**: σ가 mean-rev path에서 alpha 발견 (지금까지 모든 mean-rev path FAIL했으나 structural confirmation 추가는 새 axis).

## Stop conditions (FINAL)

- 0/2 PASS: 11-mechanism cumulative memo + 사용자 decision options.
- 1/2 PASS C5: deep verify + advisor surprise call → memo + 결과.

**Per advisor: do NOT queue R6**.

## Anti-pattern guard

- ρ×ι의 best session pick은 명시적 selection bias로 카운트. C1 PASS → strict 10-seed 검증 필수.
- σ가 PASS 시 9 mechanisms와 다른 mechanism family이므로 mean-rev paradigm 재방문 가치.
