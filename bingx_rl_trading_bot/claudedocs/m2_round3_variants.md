# M2 Round 3 — 9 Variants Pre-Registered (별도 commit, retrofitting 회피)

> **Date**: 2026-04-28
> **Authority**: 사용자 명시 위임 ("필요하다면 전부 확인")
> **Cap**: advisor 권고 6 → 사용자 explicit 9 (override). Hard ceiling 9.
> **Frame**: 9-cell map = 3 data families × 3 signal classes each.
> **Constraint**: Gate 5+6, NO Phase 3 BT, single screening script, MFE+MAE asymmetry column.

## Constants

- Asset (B, C): BTC/USDT primary; C uses BTC vs ETH spread
- Asset (A): BTC funding rate
- Timeframe: 15m execution
- Trend filter: 1h EMA20>EMA50 AND 4h close>EMA50 (LONG); SHORT mirror (where applicable)
- Friction: 0.20%/trade
- Random baseline: 5 seeds × n_target on same eligible universe per variant
- Horizons: 4 / 8 / 16 bars (= 1h / 2h / 4h)
- Gate 6 thresholds: Δp50 ≥ 0.05pp AND Δ%>fr ≥ 5pp
- 신설 column: MFE_P50 + MAE_P50 (asymmetry sum)

## Family A — Funding Rate Divergence

**Data**: BingX BTC funding rate (8h intervals). 720d fetch 진행 중. fetch 실패 시 A 후보 dropped.

### A.1 — Extreme funding fade
LONG: funding < −0.04% (shorts crowded) AND 15m RSI > 70 (mean-rev setup). Direction = LONG (fade short crowding).
SHORT: funding > +0.04% (longs crowded) AND 15m RSI < 30. Direction = SHORT.
Trend filter NOT used (funding is the directional bias).

### A.2 — Funding cross-zero
LONG: funding crosses from ≤ 0 to > 0 in latest 8h (positioning shift bullish). Trend filter aligned.
SHORT: funding crosses from ≥ 0 to < 0. Trend filter mirror.
Entry on 15m bar at the funding period boundary.

### A.3 — Sustained extreme
LONG: 8 consecutive funding periods (~64h) at funding ≥ +0.03% (overheated longs) → fade entry on RSI < 30 reversal.
SHORT: 8 consecutive ≤ -0.03% → fade on RSI > 70.

## Family B — Volume / Volume Delta

**Data**: candle volume column (즉시).

### B.1 — Volume spike at level break
LONG: 15m close > 24-bar high AND volume > 2 × volume_SMA(20). Trend filter aligned.
SHORT mirror with 24-bar low.

### B.2 — Volume divergence (negative)
LONG: 24-bar window — price makes higher-high (high[i] > rolling_max(high[:i], 24)) but volume_avg of last 5 bars < volume_avg of preceding 5 bars (weakening) → fade entry SHORT (going against the higher-high since volume is weak).
SHORT mirror.
**Direction = OPPOSITE of price move** (signal is divergence-fade).
Trend filter NOT used (signal anti-trend).

### B.3 — VWAP touch + bounce
LONG: Daily VWAP (rolling 96-bar = 24h on 15m) — close pulls back to within 0.2% of VWAP (close ≥ VWAP × 0.998 AND ≤ × 1.002) AND close > prev close (bounce). Trend filter aligned.
SHORT mirror.

## Family C — Cross-Asset BTC-ETH

**Data**: BTC 15m + ETH 15m (resampled from `eth_binance_5m.csv`, 365d aligned).

### C.1 — BTC-ETH spread mean-rev (z-score)
LONG: log(BTC/ETH) z-score (50-bar) < −2σ (BTC underpriced vs ETH) → BTC LONG.
SHORT: z-score > +2σ → BTC SHORT.
Trend filter aligned (entry only when trend supports the mean-rev direction).

### C.2 — Correlation breakdown
LONG: rolling 50-bar correlation(BTC_15m_returns, ETH_15m_returns) drops < 0.5 (보통 ~0.85 normal regime) AND BTC trend filter LONG → directional LONG.
SHORT mirror.

### C.3 — ETH-leads-BTC lag
LONG: ETH 15m return prev bar > +0.3% AND BTC 15m return prev bar < +0.1% (BTC lagging ETH up-move) AND BTC trend LONG → BTC LONG entry on next bar.
SHORT mirror with negative ETH return.

## Predictions (commit before run)

| # | Variant | Predicted vs random | Confidence | Rationale |
|---|---------|--------------------|-----------|-----------|
| A.1 | Extreme funding fade | marginal positive | MED | Position imbalance은 real signal. 단 8h 간격이라 15m timing precision 부족 |
| A.2 | Funding cross-zero | ≈ random | LOW | Pivot pattern이지만 noise heavy |
| A.3 | Sustained extreme | marginal positive | LOW-MED | Rare event (8 consecutive) → sample size 우려 |
| B.1 | Volume spike at break | marginal positive | MED | Classic "true breakout" pattern. Volume confirmation 의미 |
| B.2 | Volume divergence (fade) | ≈ random | LOW | Weak trend 신호이지만 BTC 15m noise 환경에서 미약 |
| B.3 | VWAP touch + bounce | ≈ random | LOW | Pullback variant. M1-A pullback에서 이미 fail |
| C.1 | BTC-ETH spread mean-rev | marginal positive | MED | Classic stat-arb. 단 crypto는 cointegrated 아님 (separation 가능) |
| C.2 | Correlation breakdown | ≈ random | LOW | Correlation regime 변화가 directional alpha 만드는지 의문 |
| C.3 | ETH-leads-BTC lag | ≈ random | LOW | Crypto lead-lag은 보통 미세 + not exploitable |

**Distribution**: 3 marginal positive MED, 6 ≈ random LOW-MED. 비관적이지만 honest.

**Most likely surprise**:
- B.1 (volume spike at break) FAIL — classic pattern이라 직관 강하지만 BTC 15m에서 random 못 넘을 가능성
- C.1 (spread mean-rev) PASS — cross-asset은 single-asset OHLCV가 잡지 못한 정보

**Most informative outcome**:
- A.1 if PASS → funding rate가 alpha source 시사 → BingX live deployment 후속 가능
- C.1 if PASS → cross-asset arbitrage 후속 가능 (다른 framework 영역)
- All FAIL → 4 rounds × 0 PASS = strong "no edge in this data class" 증거 → paradigm shift / pause / portfolio

## Pass condition (Gate 5 + Gate 6)

- Gate 5: gross_sum > 0 in ≥ 2 of 3 horizons
- Gate 6: Δ MFE_P50 ≥ +0.05pp AND Δ %>0.20% ≥ +5pp

## Stop conditions

- 0/9 PASS: convergent evidence memo 작성 → paradigm shift / portfolio / pause 사용자 결정
- 1-3 PASS: 모두 보고 → 사용자 picking (assistant 자체 picking 금지)
- 4+ PASS: threshold 의심 (random baseline 미달 가능성), strict re-run

## Anti-pattern guard

- 10번째 variant 추가 충동 = Round 4 deferral
- "B.1 PASS면 deeper Phase 3 BT" 충동 = 함정. Round 3 deliverable는 map only
- Funding fetch 실패 시 A drop, B+C 6 cells로 진행. A.1~A.3 NaN 처리 (NO_SIGNALS verdict)
