# M2 Round 2 — Pre-BT Map Screening (사전 등록, 별도 commit)

> **Date**: 2026-04-28 (변형 정의 + 예측 등록 단계)
> **Frame**: 16-cell map (12 NEW × Gate 5+6 + 4 Round 1 cited).
> **Authority**: 사용자 명시 위임 ("전부 병렬 심층 연구").
> **Constraint** (advisor): 단일 script 일괄 실행, MFE+MAE asymmetry column 신설, Phase 3 BT 절대 안 함.
> **Deliverable**: territory map. 각 cell은 ship candidate 아님.

## Constants

- Asset: BTC/USDT
- Trend filter (D1, D2): 1h EMA20>EMA50 AND 4h close>EMA50 (LONG) / mirror (SHORT)
- Trend filter (D3): NONE
- Friction: 0.20%/trade
- Random baseline: 5 seeds × n_target on same eligible universe per dimension
- Horizons: D1=4/8/16 (1h: 4=4h, 8=8h, 16=16h), D2/D3=4/8/16 bars (15m: 1h/2h/4h)
- Gate 6 thresholds: Δp50 ≥ 0.05pp AND Δ%>fr ≥ 5pp
- 신설 column: **MFE_P50 + MAE_P50** (sum). Pure noise = 0. >> 0 = favorable asymmetry.

## Dimension D1 — Timeframe shift (1h execution)

V1-V4 on 1h timeframe, same trend filter.

### D1.V1 — Mean-rev on 1h
LONG: 1h RSI(14) ≤ 25 AND current 1h close > open. SHORT: RSI ≥ 75 AND close < open.
Direction must match trend filter.

### D1.V2 — Squeeze breakout on 1h
LONG: 1h BB width at min of past 50 bars + close > prev BB upper. SHORT mirror.

### D1.V3 — Multi-bar momentum on 1h
LONG: 3 consecutive 1h bullish bars + 3-bar move ≥ 0.3%. SHORT mirror.

### D1.V4 — M1-A minus RSI on 1h
LONG: 1h body/range > 0.4 + close > 1h EMA9. SHORT mirror.

## Dimension D2 — NEW signal classes (BTC 15m)

### D2.V1 — Volatility regime shift
LONG: 15m ATR(14) > ATR(14) SMA50 (volatility expanding) AND current bar bullish (close > open).
SHORT: ATR > ATR SMA50 AND bearish bar.
Direction must match trend filter.

### D2.V2 — Range break with retest
LONG: 24-bar high broken by previous bar (prev close > 24-bar high before that bar) AND current bar pulls back into range (low < broken level) AND current close back above level.
SHORT mirror with 24-bar low.
Direction must match trend filter.

### D2.V3 — Higher-low / lower-high structural pivot
LONG: in past 10 15m bars, lows make higher-low sequence (last 10-bar low > 20-bar low) AND current bar bullish (close > open).
SHORT mirror.
Direction must match trend filter.
(Funding rate divergence는 데이터 align 부족으로 swap.)

### D2.V4 — Trend pullback to 1h EMA20
LONG: 15m close within 0.3% of 1h EMA20 (close ≥ 1h EMA20 × 0.997 AND ≤ × 1.003) AND 15m close > prev close (bounce).
SHORT mirror.
Direction must match trend filter.

## Dimension D3 — No trend filter (V1-V4 on BTC 15m, same triggers)

### D3.V1, D3.V2, D3.V3, D3.V4
Same as Round 1 V1-V4 BUT trend filter REMOVED. Direction determined by signal direction only.
Eligible universe: all valid bars (no h1+h4 alignment requirement).

## Predictions (commit before run; calibration source)

| # | Variant | Predicted vs random | Confidence | Rationale |
|---|---------|--------------------|-----------|-----------|
| D1.V1 | mean-rev 1h | ≈ random / marginal positive | LOW | RSI extremes on 1h이 더 의미 있을 수 있으나 sample size 매우 적을 것 |
| D1.V2 | squeeze 1h | negative | MED | V2 15m 이미 wrong; 1h도 비슷 예상 |
| D1.V3 | momentum 1h | negative | LOW | 1h consecutive bars trend 가능하지만 noise dominant 가능 |
| D1.V4 | M1-A minus RSI 1h | ≈ random | MED | M1 momentum-following family anti-edge — timeframe 무관 |
| D2.V1 | volatility regime | marginal positive | LOW-MED | Volatility expansion이 momentum 신호 가능 |
| D2.V2 | range break+retest | marginal positive | MED | Classic level pattern (재시험 = false breakout filter) |
| D2.V3 | HL/LH pivot | ≈ random | LOW | Pattern recognition 가능하나 noise heavy |
| D2.V4 | pullback to 1h EMA20 | ≈ random | LOW-MED | M1-A pullback에서 이미 fail; 다른 anchor라 다를 수도 |
| D3.V1 | mean-rev no filter | negative | MED | Filter 없이는 directional context 없음 |
| D3.V2 | squeeze no filter | negative | MED | Filter 없이 breakout 방향 미정 → false break 다수 |
| D3.V3 | momentum no filter | negative | HIGH | 가장 명백한 noise (no filter + noise autocorrelation) |
| D3.V4 | M1-A minus RSI no filter | negative | HIGH | Filter 없는 momentum-following = pure noise momentum |

**Honest calibration note**: 12 중 5개가 negative HIGH/MED, 5개가 ≈ random / LOW, 2개가 marginal positive. 전체적으로 비관적 — Round 1 0/4 + C1/M1 negative 종합 신호.

**Most likely surprise**: D2.V2 (range break+retest) PASS, D1.V4 PASS (timeframe shift이 anti-edge family 깨뜨릴 가능성). 둘 중 하나라도 surprise면 informative.

## Pass condition (Gate 5 + Gate 6)

- Gate 5: gross_sum > 0 in ≥ 2 of 3 horizons
- Gate 6: Δ MFE_P50 ≥ +0.05pp AND Δ %MFE > friction ≥ +5pp
- 신설 (참고만, gate 아님): MFE+MAE sum > 0.10 → 비대칭 favorable

## Stop conditions

- **0 PASS / 16**: map = deliverable. Round 3 (paradigm class shift) 사용자 결정.
- **1-3 PASS**: 모두 보고 → 사용자 picking. assistant 자체 picking 금지.
- **4+ PASS**: threshold too lax 신호. Δp50 ≥ 0.10pp로 strict re-run.

## Anti-pattern guard

- Variant 12개 cap. 13번째 추가 충동 = Round 3 candidates list로 deferral.
- Variant 결과 partial 또는 hidden 금지. 16-cell map은 모든 cell 보고.
- "Best variant" 자체 picking 또는 deeper Phase 3 BT 충동 = 함정. user의 explicit pick만 수용.
