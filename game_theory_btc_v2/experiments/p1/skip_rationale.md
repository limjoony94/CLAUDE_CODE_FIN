# P1 Skip Rationale (advisor 2026-05-01)

**Date**: 2026-05-01
**Mandate basis**: § P1 (BingX API + 공개 데이터 인벤토리, 1d)
**Decision**: P1을 P0.2에 흡수하여 별도 priority 진행 안 함. 이 문서로 explicit 등록 (anti-fishing § 0.7 honest closure 준수, silent omission 회피)

---

## Why Skip Justified

### 1. P0.2 covered the substantive content
P0.2 fetcher 작성 시 BingX endpoint inventory 수행:
- ✅ OHLCV (perp/spot, 1d/1h/5m/1m): CCXT 작동 확인
- ❌ OI history: `not_supported_by_ccxt` (REST 미지원)
- ✅ Funding rate history: 720d coverage (Bybit 354d cross-check)

### 2. Mandate § P1 의 implicit goal은 "Endpoint inventory"
- `data/bingx_api_inventory.md` 상응 → `experiments/p0/p02_feasibility_probe.md`에 통합
- `scripts/data/fetch_bingx_extended.py` 상응 → 별도 fetcher 안 만듦. 기존 `fetch_binance.py`가 BingX 90d parity check 포함 (defer to need-basis)
- BingX OFFICIAL doc fetch (WebFetch) → 미진행. Phase A에서 BingX endpoint 직접 사용 안 함 (Binance primary)

### 3. Industry-wide 30d OI/L-S limit이 BingX 별도 inventory 의의 약화
- BingX OI not supported via CCXT
- Direct REST 시도 시 다른 거래소와 동일 30d 제약 가정
- BingX는 LIVE 거래소로만 P6에서 재방문 (no Phase A dependency)

---

## What's NOT Done (compared to mandate § P1)

| 항목 | 상태 | 영향 |
|------|------|------|
| BingX official API doc fetch | ❌ skip | P6 LIVE-deploy 시 재수행 |
| `data/bingx_api_inventory.md` 별도 doc | ❌ skip | `p02_feasibility_probe.md` + `infrastructure_lessons_inherited.md` 흡수 |
| L2 orderbook snapshot test (BingX) | ❌ skip | Phase B / P5 microstructure 단계에서 |
| `fetch_bingx_extended.py` 별도 fetcher | ❌ skip | 필요 시 P3-P5에서 |

---

## When P1 Will Be Revisited

P6 LIVE-readiness 진입 시 의무 재방문 (mandate § P6 + `infrastructure_lessons_inherited.md`):
- BingX positionSide=BOTH 검증
- TimeSyncBingX offset 적용
- priceRate /100 함정 회피
- BT-LIVE parity 22-checklist (memory: backtest_live_parity_20260418)
- 5-Gate Protocol (memory: strategy_deploy_5gate_protocol)

이는 strategy candidate가 P5 통과 후 의미 있어짐. 현 Phase A 시점에서 작업 가치 낮음.

---

## P0 Deliverable Adjustment

기존 P0 7-deliverable + 2 derived (D8, D9) = 9.
P1 skip 결정으로 P0 deliverable 변경 없음. P1는 별도 priority인데 mandate scope adjustment에 따라 merge.

`experiments/p0/result.md`의 "9 Deliverable Status"는 변경 없음.

---

## Anti-Fishing Compliance

❌ Silent omission ("P1 was done") — 회피
✅ Explicit skip with rationale (this doc) — 준수
✅ Reschedule to P6 LIVE-deploy (rationale: dependency on strategy candidates)
✅ Mandate § 0.7 honest closure 정합

---

**Pre-commit signed**: Claude Code agent, 2026-05-01.
