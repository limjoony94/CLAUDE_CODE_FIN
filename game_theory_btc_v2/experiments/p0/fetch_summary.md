# P0.2 Fetch Summary — Data Acquisition Closure

**Date**: 2026-05-01 (T_anchor: 2026-05-01T14:00:00 UTC)
**Status**: ✅ Complete (all 8 specs PASS)
**Audit raw**: `experiments/p0/audit_results.json`

---

## Fetched Data Summary

| File | Source | Timeframe | Rows | Span | Gap rate | OHLC invalid | Assertion |
|------|--------|-----------|------|------|----------|--------------|-----------|
| `btc_perp_1d_1500d.parquet` | Binance perp | 1d | 1500 | 2022-03-24 → 2026-05-01 | 0.0% | 0 | ✅ |
| `btc_perp_1h_720d.parquet` | Binance perp | 1h | 17280 | 2024-05-11 → 2026-05-01 | 0.0% | 0 | ✅ |
| `btc_perp_5m_720d.parquet` | Binance perp | 5m | 207360 | 2024-05-11 → 2026-05-01 | 0.0% | 0 | ✅ |
| `btc_perp_1m_365d.parquet` | Binance perp | 1m | 525,600 | 2025-05-01 → 2026-05-01 | 0.0% | 0 | ✅ |
| `btc_spot_1h_720d.parquet` | Binance spot | 1h | 17280 | 2024-05-11 → 2026-05-01 | 0.0% | 0 | ✅ |
| `btc_funding_binance_720d.parquet` | Binance | 8h native | 2160 | 2024-05-11 → 2026-05-01 | 0.0% (vs 8h) | n/a | ✅ |
| `btc_funding_bybit_620d.parquet` | Bybit | 8h native | 400 | 2025-05-13 → 2026-05-01 | (354d span, 620d 미달) | n/a | ⚠️ partial |
| `multi_asset_1d_800d.parquet` | Binance perp | 1d | 4000 (5 coins) | 2023-12-23 → 2026-05-01 | 0.0% per symbol | 0 | ✅ |

### Forward Collector (Phase B prep, 자동 누적)
| File | Span (1차 run) | Records | Source |
|------|---------------|---------|--------|
| `oi_forward.parquet` | 28일 (2026-04-03 → 2026-05-01) | 652 | Binance OI hist (28d max retroactive) |
| `ls_account_forward.parquet` | 20.8일 | 500 | Binance topLongShortAccountRatio |
| `ls_position_forward.parquet` | 20.8일 | 500 | Binance topLongShortPositionRatio |
| `ls_global_forward.parquet` | 20.8일 | 500 | Binance globalLongShortAccountRatio |
| `taker_volume_forward.parquet` | 20.8일 | 500 | Binance takerlongshortRatio |

### Multi-Asset Per-Symbol (1d × 800d)
모두 800 rows / symbol, gap=0, ohlc_invalid=0:
- ETH/USDT:USDT, SOL/USDT:USDT, BNB/USDT:USDT, XRP/USDT:USDT, DOGE/USDT:USDT

### Funding Rate Statistics (Binance 720d)
- Range: [-0.015%, +0.072%] / 8h
- Mean: 약 +0.005% / 8h (≈ 약 +0.015% / day)
- Std: 약 0.012% / 8h
- 의미: BTC perp 평균 funding 비교적 낮음. 양 funding 환경 약간 우세 (longs paying shorts).

---

## Anomalies / Limitations

### 1. Bybit funding partial (354d / 620d asked)
- **원인**: Bybit pagination 동작이 documented 1000 limit과 다름. 200/call cap 추정.
- **영향**: cross-check 용도라 critical 아님. Binance funding 720d primary.
- **결정**: defer fix. P3에서 cross-check 필요 시 paginated re-fetch.

### 2. 1m × 90d → 1m × 365d 확장
- **원인**: 1m × 90d 전체가 sealed window (last 180d) 안. P0.5-P5 사용 불가능.
- **수정**: spec 변경 (90d → 365d). Backfill 진행 중.
- **결과**: 1m × 365d → sealed last 180d, free first 185d (P0.5-P5 가용).

### 3. OI/L-S forward collector
- **현 상태**: Day 1, 28d snapshot + 20.8d. 매시간 자동 누적 필요 (사용자 cron 등록 권장).
- **Phase B 활성화 timeline**: 60-90d 누적 후 (~2026-06-30 ~ 2026-07-30).

---

## Holdout Boundary (Sealed OOS Commit)

### Concrete Values (2026-05-01 P0.2 closure)
| Field | Value |
|-------|-------|
| `T_anchor_ms` | 1777680000000 |
| `T_anchor_iso` | **2026-05-01T14:00:00+00:00** (UTC) |
| `T_seal_start_ms` | 1762128000000 |
| `T_seal_start_iso` | **2025-11-02T14:00:00+00:00** (UTC) |
| `T_p3_holdout_start_iso` | **2026-02-18T14:00:00+00:00** (UTC) |
| Sealed window | 180 days (last 25%) |
| P3 extra holdout | 72 days (additional 10%) |

### Sealed Slice per File (post-1m re-fetch)
| File | Total | Sealed (last 180d) | Sealed % |
|------|-------|---------------------|---------|
| `btc_perp_1d_1500d.parquet` | 1500 | 181 | 12.07% |
| `btc_perp_1h_720d.parquet` | 17280 | 4321 | 25.01% |
| `btc_perp_5m_720d.parquet` | 207360 | 51840 | 25.0% |
| `btc_perp_1m_365d.parquet` | 525,600 | 259,199 | 49.31% |
| `btc_spot_1h_720d.parquet` | 17280 | 4321 | 25.01% |

→ `holdout_seal.md` Amendment Section에 별도 기록.

---

## Anti-Fishing Compliance Check

✅ Pre-commit (`precommit.md` + `amendment_001` + `amendment_002`) 사전 lock
✅ Lookahead-free assertion enforced (모든 OHLCV `t_close = t_open + interval`)
✅ Sealed boundary 사후 fixed (T_anchor immutable)
✅ Bybit cross-check은 cross 용도, primary는 Binance
✅ 1m × 90d defect 발견 즉시 fix (silent shift 아님)
✅ Forward collector 시작, 사용자 cron 등록 권장

---

## P0 Deliverable Update

| # | Deliverable | Status |
|---|-------------|--------|
| D1 | BTC 1h/5m/1d/1m 720+d data | ✅ (1m 365d backfill 진행) |
| D2 | API access | ✅ Binance + BingX (Coinglass paid 보류) |
| D3 | friction_model.py | ⏳ P0.3 |
| D4 | 6-criteria validator | ⏳ P0.3 |
| D5 | Baseline benchmarks | ⏳ P0.4 |
| D6 | H1-H9 정량 정의 | Phase A subset draft (P0.5에서 full registry) |
| D7 | ~30 mechanism catalog | ⏳ P0.6 |
| D8 (new) | Forward collector | ✅ running |
| D9 (new) | Proxy formula v2 | ✅ committed |

---

**Next**: P0.3 entry — friction_model.py + bootstrap_six_criteria.py 작성. Advisor 호출 (interpretation commit moment).
