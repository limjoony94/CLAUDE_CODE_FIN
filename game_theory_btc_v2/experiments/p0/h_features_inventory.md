# H1-H9 Feature Inventory + P0.2 Fetch Coverage Cross-Check

**Date**: 2026-05-01
**Mandate**: v2 § 1 hypotheses + § P0.5
**Purpose**: P0.2 fetch list가 H1-H9 정량 검증을 모두 cover하는지 walk-through (advisor 권고 #2)

---

## Feature Catalog by Hypothesis

### H1 — long/short imbalance proxy → 1-12h price direction
**Required features** (lag-aware, t-1 shift on snapshot data):
- `funding_rate_t-1` (8h native, forward-fill to 1h grid)
- `oi_delta_pct_t-1 = (oi_t - oi_{t-N}) / oi_{t-N}` (window N=1h, 4h, 24h)
- `price_change_t = log(close_t / close_{t-1})` (forward label)
- Optional: CVD (cumulative volume delta) — derived from trade tape

**Data source coverage**:
| Field | Source | Available? |
|-------|--------|------------|
| funding_rate | Binance funding history 720d | ✅ |
| oi history | Binance OI 1h 720d | ✅ |
| close price | Binance perp 1h | ✅ |
| CVD (trade-tape) | Binance trades aggregator | ⚠️ requires aggregation, large |

**Defer decision**: CVD은 P5에서 도입. H1 P2 단계는 funding × OI delta + price proxy로 검증.

---

### H2 — Kyle's λ low (non-toxic) = retail entry safest
**Required features**:
- `kyle_lambda = price_change / signed_volume` (regression slope, rolling window)
- Best estimated from 5m or 1m bar with `signed_volume` proxy

**Signed volume proxy options**:
- A: trade tape buy/sell aggressor flag (best, requires trade aggregator)
- B: tick rule on 1m bar (close > previous close → buy classification)
- C: Hasbrouck signature variance ratio

**Data source coverage**:
| Approach | Source | Available? |
|----------|--------|------------|
| A. trade tape | Binance/BingX websocket | ⚠️ defer — 별도 collector 필요 (P5/L2 영역) |
| B. tick rule on 1m | Binance perp 1m 90d | ✅ — H2 first-pass에 사용 |

**Plan**: P2에서 tick-rule based λ. P5에서 가능 시 trade tape 기반 정밀화.

---

### H3-revised — long-liquidation magnitude × low quantile × neg funding
**Required features** (mandate § 1.2.5):
- `liquidation_long_5m_usd`: 5m aggregate long liquidation magnitude (USD)
- `price_quantile_30d`: rolling 30d trailing close quantile rank
- `funding_rate_8h`: native 8h, forward-filled

**Data source coverage**:
| Field | Source | Available? |
|-------|--------|------------|
| liquidation aggregate | Coinglass (key 등록 후) 1h aggregate | ⚠️ pending key |
| liquidation aggregate (fallback) | Binance forced orders (sapi REST 30d) | ⚠️ 30d only |
| price quantile | Binance perp 1h (30d trailing) | ✅ |
| funding | Binance funding history | ✅ |

**Critical dependency**: liquidation data without Coinglass = fall back to:
- Binance forced orders sapi (30d window) — H3 검증 sample size 약화
- OR liquidation proxy: `funding_rate_jump_8h × oi_delta_neg × price_velocity_neg` — indirect, less precise

**Recommendation**: 사용자 Coinglass 무료 키 등록 (5분), 등록 안 하면 H3 검증 약화 수용 + closure 시 명시.

---

### H4-revised — cascade window 5-30분 mean reversion edge > 0.16% RT
**Required features**: H3와 동일 + bar resolution 1m (5-30분 window 정밀 측정)
- `liquidation_event_t` (binary or magnitude)
- `forward_return_5m, 15m, 30m` (lag-free label)
- 1m bar high/low for intrabar analysis

**Data source coverage**:
- liquidation: H3와 동일 dependency
- 1m bar: ✅ Binance perp 1m × 90d (P0.2에 추가됨)

---

### H5 — short-side mirror (BTC drift bias 약화)
**Required features**: H3-H4와 동일 + 반대방향
- `liquidation_short_5m_usd`
- 가격 high quantile (30d trailing >= 70%)
- positive funding extreme

**Data source coverage**: H3-H4와 동일

---

### H6 — price ↑ + funding spike + OI ↑ → distribution
**Required features**:
- `price_change_24h`: rolling
- `funding_rate_8h`
- `oi_delta_24h`
- Optional: `taker_buy_volume_ratio` (aggressive buying detection)

**Data source coverage**:
| Field | Source | Available? |
|-------|--------|------------|
| price_change | Binance perp 1h | ✅ |
| funding | Binance funding | ✅ |
| oi_delta | Binance OI history | ✅ |
| taker_buy_volume | Binance kline `takerBuyVolume` field | ⚠️ check CCXT support, alternative `quoteVolume` ratio |

---

### H7 — spot-perp basis extreme + L/S ratio extreme → fade
**Required features**:
- `basis_pct = (perp_close - spot_close) / spot_close * 100`
- `long_short_ratio` (positions, top accounts, taker)

**Data source coverage**:
| Field | Source | Available? |
|-------|--------|------------|
| spot price | Binance spot 1h × 720d | ✅ (P0.2에 추가됨) |
| perp price | Binance perp 1h | ✅ |
| L/S ratio | Binance fapi/data L/S endpoint | ⚠️ via REST `fapi/futures/data/topLongShortAccountRatio`, CCXT 미지원 → custom fetcher |

**Plan**: H7 first-pass는 basis 단독, L/S ratio는 P5에서 fetch.

---

### H8 — MAP-Elites archive ≥3 cells 6-criteria 통과
**Required features**: 모든 mechanism (P0.6 catalog) + regime grid (volatility × trend)
- mechanism returns (per-trade, daily aggregate)
- regime classifiers:
  - `volatility_regime = atr_30d_quantile (low/mid/high)`
  - `trend_regime = sma_50_slope (up/flat/down)`

**Data source coverage**: ✅ 모든 input은 1h OHLCV로 충분

---

### H9 — Risk-Aware Thompson Sampling > static archive
**Required features**: H8 결과 + bandit reward signal
- per-arm daily reward (각 mechanism의 daily return)
- contextual features (regime + volatility)

**Data source coverage**: ✅ H8과 동일

---

## P0.2 Fetch List Cross-Check (확정)

✅ = 이미 fetch list에 포함, ⚠️ = 추가 또는 결정 필요

| Data | Coverage | Status |
|------|---------|--------|
| BTC perp 1d × 1500d | H1-H9 baseline + 멀티-coin compare | ✅ |
| BTC perp 1h × 720d | H1, H6 primary + 30d quantile | ✅ |
| BTC perp 5m × 720d | H3-H5 medium-resolution | ✅ |
| BTC perp 1m × 90d | H4 cascade window | ✅ |
| BTC spot 1h × 720d | H7 basis | ✅ (advisor 권고로 추가됨) |
| BTC OI 1h × 720d | H1, H6 | ✅ |
| BTC funding 8h × 720d | H1, H3-H6 | ✅ |
| BTC liquidations 1h × 90-720d | H3-H5 ⚠️ | ⚠️ Coinglass key 결정 필요 |
| Binance L/S ratio 5m × 30d | H7 정밀 | ⚠️ defer to P5 |
| Multi-asset 1d × 800d (5 coins) | cross-asset reference | ✅ |

---

## Hypothesis Test Strategy by Coverage

| Hypothesis | Direct test? | Fallback if Coinglass blocked |
|-----------|-------------|------------------------------|
| H1 | ✅ | (covered) |
| H2 | ✅ tick-rule | trade tape (P5) for precision |
| H3 | ⚠️ Coinglass needed | proxy: funding spike + oi delta + price velocity |
| H4 | ⚠️ Coinglass needed | proxy + Binance sapi 30d sample |
| H5 | ⚠️ Coinglass needed | mirror H3 fallback |
| H6 | ✅ | (covered) |
| H7 | ✅ basis | L/S ratio P5 정밀 |
| H8 | ✅ (subject to H1-H7 results) | — |
| H9 | ✅ (subject to H8) | — |

---

## Decisions Required from User

1. **Coinglass 무료 API key 등록 여부** — 5분 작업, H3-H5 직접 검증 가능. 안 하면 proxy로 약화 수용.

P0.2는 사용자 Coinglass 결정과 무관하게 즉시 시작 가능 (Binance/BingX 부분).
