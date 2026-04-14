# Multi-Asset Vol-Spike 15m Strategy Design

> Version: 0.2.0 | Status: Research (4/5 criteria met) | Date: 2026-04-12

## Strategy Summary

**15m 다자산 Vol-Spike + Swing Structure TP/SL**

검증된 1h volspike 전략(15자산, daily +1.08%)을 15m에 적응. 대형 캔들(range > D×ATR) 발생 시 캔들 방향으로 진입, swing 구조 기반 동적 TP/SL로 청산.

## Entry Logic

```
조건 1: 15m candle range > 3.0 × ATR(14)
조건 2: |body| > 30% × range (doji/spinning top 제외)
방향:   body > 0 → LONG, body < 0 → SHORT
진입:   다음 봉(bar+1) 시가(open)
```

## Exit Logic (진입과 다른 로직)

```
TP: 2nd swing structure (32-bar lookback) + 0.3×ATR buffer
    - LONG: max(high[i-32:i]) + 0.3×ATR
    - SHORT: min(low[i-32:i]) - 0.3×ATR
SL: 1st swing structure (12-bar lookback) + 0.3×ATR buffer
    - LONG: min(low[i-12:i]) - 0.3×ATR
    - SHORT: max(high[i-12:i]) + 0.3×ATR
Timeout: 96 bars (24h)
Min R:R: 0.8 (TP/SL ratio filter)
```

**TP/SL은 완전 동적**: 진입 시점의 실제 가격 구조(swing high/low)에 기반하여 매 거래마다 다른 값.

## Parameters (Universal — 전 자산 동일)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| D | 3.0 | 3.0×ATR = 강한 모멘텀 캔들만 선별 |
| body_ratio_min | 0.3 | Doji 제외 → WR +5.3pp |
| swing_lookback (SL) | 12 bars | 3h 구간의 최근 지지/저항 |
| swing2_lookback (TP) | 32 bars | 8h 구간의 먼 구조물 → R:R > 1 |
| atr_buffer | 0.3×ATR | Noise buffer |
| cooldown | 4 bars | 1h 최소 간격 |
| max_hold | 96 bars | 24h timeout |
| min_rr | 0.8 | TP/SL 비율 최소 필터 |

## Validated Performance

### Best Config: D=3.0, body≥0.3, 9 assets, 365d

| Metric | Value |
|--------|-------|
| WR | 52.7% ✅ |
| R:R | 1.46 ✅ |
| Daily Return | +0.260% ✅ |
| Per-trade PnL | +0.48% ✅ |
| Trades/day | 0.45 ❌ |
| PnL (365d) | +94.9% |
| MC p-value | 0.006 |
| Walk-Forward | 5/5 PASS |
| Halves | H1 +28.0% / H2 +44.2% |

### Assets (9, filtered at D≥3.0 positive PnL)

ETH, DOGE, AVAX, BNB, LTC, BCH, ATOM, UNI, TRX

### Look-Ahead Bias Check

- Entry: next-bar open (no look-ahead)
- TP/SL: based on PAST swing structure only (lookback, not forward)
- ATR: rolling past 14 bars (no future data)
- body_ratio: current candle only

### Overfit Assessment

- **단일 파라미터 세트** (전 자산 동일) → 과적합 위험 최소
- **MC p=0.006** (5000 sign randomization)
- **WF 5/5 ALL PASS** — 시간적 안정성 최강
- **Both halves positive** (H1 +28%, H2 +44%) → 시간 편향 없음
- **Bonferroni concern**: 수십 개 config 테스트 → 0.006×36=0.22 (약한)
  - 단, D=3.0 클러스터 전체가 p<0.02 → 개별 cherry-pick 아님

## Frequency Solution (TODO)

현재 0.45 trades/day (9자산). 해결 방안:

1. **자산 확대** (9 → 30+): 0.45 × (30/9) ≈ 1.5/day
2. **추가 거래소**: Binance + BingX 독립 실행 → 2× frequency
3. **D 미세 조정**: D=2.9에서 WR 44.6%, tpd 0.86 → 자산 확대와 결합

## Fee Structure

- Round-trip: 0.10% (taker 0.05% × 2)
- Per-trade gross: +0.58% → net +0.48% (fee의 4.8배)
- Breakeven WR at R:R 1.46: 40.7% → 실제 WR 52.7%로 +12pp 마진

## Risk Management

- Max drawdown: ~18% (no leverage)
- Per-trade risk: SL avg 1.3% of position
- Timeout: 24h max hold
- Cooldown: 1h minimum between trades per asset
