# M2 Round 1 — Pre-BT Variant Screening (사전 등록)

> **Date**: 2026-04-28
> **Scope**: Gate 5 (entry isolation) + Gate 6 (random baseline) only. **Phase 3 BT 안 함.**
> **Authority**: 사용자 명시 위임 ("다양한 조합을 직접 테스트 연구")
> **Constraint** (advisor): 4 variants, BTC 15m fixed, single dimension (signal class) varies.

## Constants across all variants

- **Asset**: BTC/USDT
- **Timeframe (execution)**: 15m bars (M1 5m noise lesson 적용)
- **Trend filter**: 1h EMA20>EMA50 (LONG) AND 4h close > 4h EMA50 (LONG); SHORT mirror. 둘 다 align 필수.
- **Fixed-N-bar exit horizons**: 4 / 8 / 16 bars (= 1h / 2h / 4h)
- **N=1 sequencing**: 2-bar cooldown
- **Friction floor**: 0.20%/trade
- **Random baseline**: same 1h+4h trend-filtered universe, 5 seeds × ~target_n samples

## Variants (4)

### V1 — Mean-reversion at extremes
**Rule (LONG)**: 15m RSI(14) on previous bar ≤ 25 AND current 15m bar close > current open (bullish bar).
**Rule (SHORT)**: RSI ≥ 75 AND close < open.
Direction must match trend filter.
**Hypothesis**: RSI extremes oversold/overbought reversion 발생.

### V2 — Volatility squeeze breakout
**Rule (LONG)**: 15m BB width (BB(20, 2.0)) at lowest of past 50 bars (squeeze) AND current close > previous bar BB upper (breakout up).
**Rule (SHORT)**: squeeze AND current close < previous bar BB lower.
Direction must match trend filter.
**Hypothesis**: Volatility 압축 후 expansion = momentum 유지.

### V3 — Multi-bar momentum continuation
**Rule (LONG)**: 3 consecutive 15m bars all bullish (close > open) AND total 3-bar move ≥ 0.3% (high - low_3bar_ago / low_3bar_ago).
**Rule (SHORT)**: 3 consecutive bearish + 3-bar move ≤ -0.3%.
Direction must match trend filter.
**Hypothesis**: Trend persistence — 연속 동방향 봉은 trend 발현.

### V4 — M1-A minus RSI cross (ablation)
**Rule (LONG)**: 15m body/range > 0.4 AND close > 15m EMA9. (RSI cross 제거)
**Rule (SHORT)**: body/range > 0.4 AND close < 15m EMA9.
Direction must match trend filter.
**Hypothesis**: M1-A의 anti-selectivity가 RSI cross 단독 원인이었는지 isolation.

## Predictions (사전 등록 — post-hoc rationalization 회피)

| Variant | Predicted vs random (MFE P50) | Confidence | Rationale |
|---------|-------------------------------|-----------|-----------|
| V1 mean-rev | marginal positive | LOW | Extremes는 distribution tail → reversion 가능. BTC noise heavy 환경에서 직관 약함. |
| V2 squeeze breakout | **positive** | MEDIUM | Volatility expansion이 가장 알려진 momentum pattern. C1 breakout 실패 우려, but C1은 channel breakout (다른 mechanism). |
| V3 momentum continuation | ≈ random or **negative** | MEDIUM | Crypto mean-reversion bias 우세. 연속 동방향은 noise autocorrelation일 가능성. |
| V4 M1-A minus RSI | ≈ random | LOW | Body+EMA9도 momentum-following이라 anti-edge 가능. RSI cross 단독 원인이 아닐 수 있음. |

**가장 surprise expected**: V2 fail 또는 V3 pass. 둘 다 선험적 직관 위반.
**가장 informative outcome**: V4 vs M1-A diff — RSI cross의 isolated effect.

## Pass condition (Gate 6)

각 variant 다음 두 조건 모두 충족해야 PASS:
1. MFE P50 ≥ random MFE P50 + **0.05 percentage point**
2. % MFE > friction (0.20%) ≥ random + **5 percentage point**

(M1-A가 random에 0.10pp 뒤졌던 점 고려, +0.05pp 정도면 "clearly beats random"로 판정.)

## Stop conditions

- **Exactly 1 PASS**: 사용자에게 단일 후보 보고 → user confirms or rejects → plan 시작 (Phase 2.5 gates 1~6)
- **0 PASS**: 보고 + Round 2 제안 (timeframe 변경 or 다른 4 signal classes). 사용자 결정.
- **Multiple PASS**: 모두 보고 + 사용자 선택 (assistant 자체 picking 금지)

## Anti-pattern guard

- 5번째 variant 추가 충동 → "Round 2 candidates" list에 기록만, Round 1은 4개 fix
- 결과 보고 partial 금지: MFE P50 / MAE P50 / gross sum at 3 horizons / random comparison 표준 fields 모두 보고
- Variant fail 시 entry rule "수정" 후 재실행 금지 — 사전 정의 그대로 측정

## Output

- `results/m2_round1_screening_*.json` (raw data, 4 variants × all metrics)
- 결과 요약은 본 문서 "Results" 섹션에 추가 (post-run, pre-decision)
- 사용자 보고서에 prediction vs result 명시 (calibration 학습)
