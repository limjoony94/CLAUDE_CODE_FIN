# P0.2 Feasibility Probe — Data Source Sanity Check

**Date**: 2026-05-01 13:15 UTC
**Probe script**: `scripts/data/probe_feasibility.py`
**Raw output**: `experiments/p0/feasibility_probe_raw.json`
**Purpose**: 1-2일 fetch 시작 전 single-call 가용성 확인 (advisor 권고 #1)

---

## Result Summary

| Source | OHLCV perp 1h/5m | OHLCV spot 1h | OI history | Funding history | Liquidation |
|--------|-----------------|---------------|------------|-----------------|-------------|
| **Binance** | ✅ | ✅ | ✅ | ✅ | (forced orders sapi 30d only via REST) |
| **BingX** | ✅ | ✅ | ❌ `not_supported_by_ccxt` | ✅ | (없음) |
| **Coinglass** | — | — | — | — | ❌ 4/4 endpoint `API key missing` |

## Sample Evidence

### Binance perp 1h OHLCV
```
[1777626000000, 77340.1, 77358.5, 77160.0, 77239.3, 3412.386]
[1777629600000, 77239.3, 77408.7, 77118.1, 77268.9, 2931.392]
[1777633200000, 77268.9, 77588.4, 77232.4, 77429.5, 4473.362]
```

### Binance OI history sample
```python
{'symbol': 'BTC/USDT:USDT',
 'baseVolume': 97471.389,
 'quoteVolume': 7538446972.40,
 'openInterestAmount': 97471.389,
 'openInterestValue': 7538446972.40,
 'timestamp': 1777626000000}
```

### Binance funding sample
```python
{'symbol': 'BTC/USDT:USDT', 'fundingRate': -4.22e-05, 'timestamp': 1777507200002}
```

### BingX OI fail
```
err: not_supported_by_ccxt
```

### Coinglass endpoints (all 4 fail)
```
v2 liquidation_history h1: 200 {"code":"30001","msg":"API key missing.","success":false}
v2 liquidation:            500 Internal Server Error
v3 aggregated-history:     200 {"code":"30001","msg":"API key missing.","success":false}
v3 liquidation history:    200 {"code":"30001","msg":"API key missing.","success":false}
```

---

## Decisions (P0.2 fetch list 변경)

### 결정 1: Backtest data source = Binance (primary)
- **이유**: OI history 가용 (BingX 미지원). LIVE-side BingX OHLCV는 sample-rate parity check용으로만 일부 fetch
- **함의**: BT-LIVE parity 시 Binance vs BingX OHLCV cross-check 의무 (90d window)
- **timestamp 정책**: candle close timestamp = `t_open + interval`. Binance native는 open ts → P0.3 friction model에서 +interval 보정

### 결정 2: BingX = LIVE/cross-check only
- 90d × 1h, 5m fetch (parity 검증용)
- Funding rate는 양측 모두 fetch (BingX live + Binance bt)

### 결정 3: Liquidation data — 사용자 결정 필요
3가지 옵션:

| 옵션 | 비용 | Coverage | 추천 |
|------|-----|---------|------|
| A. Coinglass 무료 키 등록 | 0 (email signup) | 90d-1y, rate-limited | ⭐ 추천 — H3-H4 직접 검증 |
| B. Binance forced orders (sapi REST) | 0 | **30d only** + auth required | 백업, 단기 sample |
| C. Liquidation proxy (funding spike + OI delta + price velocity) | 0 | full 720d | H3-H4 indirect — magnitude 측정 약함 |

**기본 plan**: 옵션 A (Coinglass 무료 키 사용자 등록) + 옵션 C (proxy fallback). 사용자가 키 등록 확인하면 즉시 fetch.

### 결정 4: Spot price 추가 (advisor 권고 #2)
- H6 (price↑ + funding spike + OI↑)는 perp price만으로 충분하지만, H7 (spot-perp basis extreme)는 spot 필요
- Binance BTC/USDT spot 1h × 720d 추가 fetch list

### 결정 5: 1m bar 90d 추가
- H4 cascade window edge (5-30분 mean reversion) 검증 시 5m bar 해상도 부족 가능
- Binance perp 1m × 90d 추가 (1.3M rows, ~50MB parquet — manageable)

---

## Updated P0.2 Fetch List

| Series | Source | Timeframe | Window | Est. size |
|--------|--------|-----------|--------|-----------|
| BTC perp OHLCV | Binance | 1d | 1500d | <1MB |
| BTC perp OHLCV | Binance | 1h | 720d | ~3MB |
| BTC perp OHLCV | Binance | 5m | 720d | ~30MB |
| BTC perp OHLCV | Binance | 1m | 90d | ~50MB |
| BTC spot OHLCV | Binance | 1h | 720d | ~3MB |
| BTC OI history | Binance | 1h | 720d (max API limit per call applies) | ~2MB |
| BTC funding rate | Binance | 8h native | 720d | <1MB |
| BTC funding rate | BingX | 8h native | 720d | <1MB |
| BTC perp/spot OHLCV | BingX | 1h, 5m | 90d (parity) | ~5MB |
| BTC liquidations | Coinglass (key 등록 후) | 1h aggregate | 90d-1y | ~1MB |
| Multi-asset OHLCV | Binance | 1d | 800d, 5 coins (ETH/SOL/BNB/XRP/DOGE) | ~1MB |

**Total**: ~95MB raw → ~30-50MB compressed parquet.

---

## Rate Limit / Blocker Risk

### Binance
- weight 5/req for 1h klines × 720d = ~30 requests (limit 500/req each), well within 6000 weight/min
- OI history 1h × 720d → ~30 req at limit 500. Same.
- 5m × 720d → ~150 req, ~750 weight. Single user fine.
- 1m × 90d → ~270 req, ~1350 weight.
- **risk: low**. Add `time.sleep(0.3)` between calls for safety.

### BingX
- Less documented. Use `enableRateLimit=True` in CCXT, time.sleep on errors.

### Coinglass (post key registration)
- Free tier typically 30 req/min. 720d × 1h aggregate 단일 endpoint 1 call로 가능.

---

## Open Action Items (사용자 결정 또는 자동 진행)

1. ⚠️ **Coinglass 무료 API key 등록** (사용자 작업, ~5분):
   - https://www.coinglass.com/api 또는 https://open-api.coinglass.com/
   - 키를 `config/api_keys.yaml`에 추가:
     ```yaml
     coinglass:
       api_key: "YOUR_KEY"
     ```
   - 등록 안 하면 옵션 C (proxy)로 진행 — H3-H4 검증 약화 수용
2. ✅ Binance fetch는 즉시 진행 가능 (no key required for public data)
3. ✅ BingX existing key 재사용

---

**Conclusion**: P0.2 진입 viable. Binance가 primary BT source. Coinglass key 등록 시 H3-H4 직접 검증 가능, 안 하면 proxy로 약화 수용. 7-day timeline은 Coinglass blocker 없이 유지 가능.
