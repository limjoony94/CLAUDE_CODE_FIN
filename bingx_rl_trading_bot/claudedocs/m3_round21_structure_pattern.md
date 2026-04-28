# M3-R21 — Structure-based Dynamic TP/SL + Pattern Reversal Entry (사전 등록)

> **Date**: 2026-04-28
> **Origin**: R20 ATR-based dynamic exit FAIL (gross -0.007% pre-fee, whipsaw).
> **Insight**: ATR-based SL ignores price structure. Real traders use swing points / S&R levels. R21 = pure structural.

## 1. R20 failure 분석

R20 ATR-based:
- TP = entry + 2×ATR (fixed distance, ignores price action)
- SL = max(swing_low_10, entry − 1.5×ATR) (mostly ATR-bound, structural backstop only)
- Result: WR 23.6% (way too low), avg_gross -0.007% pre-fee

**Root cause**: 1.5×ATR distance가 BTC 5m noise floor와 비슷 → 정상 noise에 SL 자주 hit. C1 폐기 사유 동일.

**R21 fix**:
- SL = strict swing point only (구조적 stop)
- TP = NEXT significant swing high/low (구조적 target, multi-bar)
- 진짜 "structure-based" — ATR 폐기

## 2. Strategy Spec: ψ′ (Pattern Reversal at S&R)

### Entry Logic — Pattern Reversal at Recent Extreme

**Definition**: 최근 swing high/low를 시도한 후 reversal candle 형성. Mean-reversion at structural level.

**LONG conditions**:
1. **Recent extreme**: low[i-1] or low[i-2] = recent 20-bar low (swing low touched)
2. **Reversal candle [i]**: bullish engulfing OR hammer
   - Bullish engulfing: cl[i] > op[i] AND op[i] < cl[i-1] AND cl[i] > op[i-1] (current bull body covers prev bear body)
   - Hammer: lower_wick[i] ≥ 2 × body[i] AND close near top (close > body_mid)
3. **1h trend**: NEUTRAL or LONG (don't fight strong downtrend) — close > 200-bar SMA(1h) gives leeway
4. **Volume**: bar [i] volume > 1.2 × SMA20

**SHORT mirror**: high touched, bearish engulfing or shooting star, etc.

### Exit Logic — Structure-based Dynamic

**Initial setup**:
- SL = recent_swing_low - 0.05% buffer (tight structural)
- TP1 = nearest resistance (closest 20-bar high prior to entry, above current)
- TP2 = next resistance (50-bar high prior)

**Dynamic management**:
- Half exit at TP1 (partial profit-taking)
- Trail SL to entry +0.05% after TP1 hit (breakeven+)
- Final exit at TP2 OR trailing SL hit
- Emergency: -1% (tighter than R20's 1.5%)
- Timeout: 24 bars (2 hours @ 5m)

This is genuinely closer to how discretionary traders manage trades.

### Friction Model
- Entry: LIMIT (maker, 0.02%) — pattern-based has time, not chasing
- Exit TP1/TP2: LIMIT (maker, 0.02%) — at known structural levels
- Exit SL: MARKET (taker, 0.05%)
- Per trade RT: ~0.02 + 0.02 = 0.04% if TP, 0.02 + 0.05 = 0.07% if SL
- Conservative: assume **0.07% RT** (mixed model, weighted toward SL friction since WR <100%)

## 3. Pre-registered Tests (사용자 spec 동일 — 7 tests)

| # | Test | Threshold |
|---|------|-----------|
| 1 | Look-ahead audit | 0 leaks |
| 2 | Overfit | sensitivity ±20%, WF 5-fold 4/5+, 3-way test positive |
| 3a | Friction taker (0.10% RT) | daily ≥ 0.2% |
| 3b | Friction mixed (0.07% RT) | daily ≥ 0.2% |
| 3c | Friction maker (0.04% RT) | daily ≥ 0.3% |
| 4 | Bootstrap 3-day | mean>0, pos_rate≥50%, p5>-1, p_vs_BH≥60% |
| 5 | Avg gross ≥ 0.10% | per-trade > taker fee |
| 6 | Frequency ≥ 2/day | sample 충분 |
| 7 | WR ≥ 50%, R:R ≥ 1 | core profitability |

## 4. Predictions

| Test | Prediction | Rationale |
|------|-----------|-----------|
| 1 (look-ahead) | PASS | Backward-looking |
| 2 (overfit) | borderline | LOW |
| 3a (taker) | likely FAIL | mult gap 강함 |
| 3b (mixed) | borderline | More realistic |
| 3c (maker) | borderline-PASS | Maker rebate path |
| 4 (bootstrap) | likely FAIL | Counter-trend nature, R9c pattern |
| 5 (gross) | borderline | Pattern reversal R:R 잠재력 vs noise |
| 6 (freq) | borderline | Pattern + structure 조건 selective |
| 7 (WR+RR) | borderline-PASS | Mean-rev high WR potential |

**Most likely outcome**: 3-4 tests fail. ALL pass probability ~5-10%.
**Most likely surprise**: ψ′ pattern + structural exit가 R20 ATR exit보다 + R:R 끌어올리고 + WR 50%+ 달성.

## 5. R20 vs R21 비교 가능 (controlled experiment)

같은 5m bar dataset에 두 spec:
- R20: confluence breakout entry + ATR exit
- R21: pattern reversal entry + structural exit

이 비교가 "ATR exit vs structural exit 어느 게 BTC에 적합한가" 정량화 시도.

## 6. Anti-fix-impulse

- 본 spec 결과 후 변경 안 함
- ≥ 2 tests fail → drop
- ALL pass → deep verify (10-seed strict, paper trade plan)
- 결과 무관 R22 자동 진행 안 함 (사용자 명시 redirect 후)
