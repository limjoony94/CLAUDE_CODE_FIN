# Coinglass Free-Tier Blocker — Discovery + Resolution Options

**Date**: 2026-05-01 13:34 UTC
**Trigger**: 사용자 무료 키 등록 후 authenticated probe 실행 결과
**Raw probe**: `experiments/p0/coinglass_authed_probe_raw.json`

---

## Discovery

| Endpoint | Header | Status | Response |
|---------|--------|--------|----------|
| v3 liquidation/aggregated-history | `CG-API-KEY` | 200 | `{"code":"401","msg":"Upgrade plan"}` |
| v3 liquidation/history | `CG-API-KEY` | 200 | `{"code":"401","msg":"Upgrade plan"}` |
| v3 liquidation/aggregated-coin-history | `CG-API-KEY` | 500 | Internal Server Error |
| v3 fundingRate/ohlc-history | `CG-API-KEY` | 200 | `{"code":"401","msg":"Upgrade plan"}` |
| v3 openInterest/ohlc-aggregated-history | `CG-API-KEY` | 200 | `{"code":"401","msg":"Upgrade plan"}` |
| v2 liquidation_history | `coinglassSecret` | 200 | `{"code":"401","msg":"Upgrade plan"}` |

**Key 자체는 valid** (server가 인증 인지하나 plan 부족). **Free tier에서는 liquidation/funding/OI history endpoint 모두 차단**.

추가 cost 정보: Coinglass paid plans (Standard/Pro/Premium) ~$29-$79/month (가격 변동 가능, 사용자 site 확인 필요).

---

## 3 Resolution Options

### Option A — Coinglass paid plan ($29-79/month)
- **Pros**: H3-H5 direct validation (long-liquidation magnitude 측정)
- **Cons**: 정기 비용. mandate scope ($1.5K capital)에 비례 부담 큼
- **Decision needed from user**: 유료 결제 의향

### Option B — Liquidation Proxy (Binance free)
H3-H5의 "long-liquidation cascade" 신호를 다음 proxy로 대체:
```
liquidation_proxy_t = funding_spike × oi_delta_negative × price_velocity_negative
```
구체적으로:
- `funding_spike_t = |funding_rate_t - rolling_mean_30d| > 2σ`
- `oi_drop_t = (oi_t - oi_{t-1h}) / oi_{t-1h} < -1%` (cascade 시 forced close → OI 하락)
- `price_velocity_t = |close_t - close_{t-5}| / atr_14` > 2

세 조건 동시 만족 시 cascade event candidate. 강도(magnitude)는 OI drop %로 정량화.

- **Pros**: 100% 무료, 720d full coverage, lookahead-free
- **Cons**: indirect — 실제 liquidation USD 수치 미측정. H3 hypothesis text 일부 약화 (cascade event는 detect되나 magnitude precision 낮음)
- **Mandate compatibility**: § 1.2.5 "검증 가능 정량 신호" 정신과 정합 (이미 funding/OI 사용 명시). PARTIAL 인정 가능.

### Option C — Binance forced orders REST (30d only)
- Binance Futures `/fapi/v1/allForceOrders` (UID-private) 또는 `/fapi/v1/forceOrders`
- API key + signature 필요 (BingX 키 있지만 Binance 키 추가 필요)
- 30일 window만 (REST historic limit)
- **Pros**: 진짜 liquidation USD 수치, free
- **Cons**: 30d window는 H3-H5 sample size 부족 (mandate 720d 가정 대비 4%). Binance 키 추가 발급 필요

### Option D — Hybrid (B + C)
- B (proxy 720d) primary 신호
- C (sapi 30d) calibration: proxy magnitude 정확도 검증용 30d ground truth
- **Pros**: 양 옵션 장점 결합. PARTIAL 인정 가능
- **Cons**: Binance 키 발급 필요

---

## Recommendation

**Option B (proxy 단독) 우선 추천** + 선택적 D 확장.

Rationale:
1. **Cost-effective**: $0 추가
2. **Mandate § 0.7 honest closure 정합**: H3-H5 검증 약화는 PARTIAL로 명시
3. **Friction-floor 27 mechanisms 0 deployable evidence 감안**: H3-H5가 PASS할 prior 자체가 LOW. 비용 투입 ROI 의심
4. **Hypothesis structure 보존**: § 1.2.5 H3-revised의 신호 구성 (liquidation magnitude × low quantile × neg funding)에서 "magnitude"만 proxy로 대체. low quantile + neg funding은 직접 측정 가능

만약 P2에서 proxy로 H3-H4가 BORDERLINE/PROMISING 결과 나오면, 그 시점에 Option A 비용 투입 ROI 재평가.

**Option A는 보류** (현 단계 비용 ROI 불명확).

---

## Effects on Existing Documents

### precommit.md (D6 deliverable)
변경 없음. H3-H5는 proxy 기반 정량 정의 추가만.

### h_features_inventory.md
H3-H5 row의 "Coinglass needed" → "Proxy: funding spike + OI delta + price velocity (all Binance free)"로 정정.

### holdout_seal.md
변경 없음.

### precommit_amendment_001.md (friction stress)
변경 없음.

---

## Action Items

1. ⚠️ **사용자 결정**: Option B 단독 진행 OK인지 확정. 또는 Option A/D 선호 시 알림.
2. ✅ Coinglass 키는 보존 (향후 paid plan 결정 시 재사용 가능). `keys.py`의 `get_coinglass()`는 작동.
3. ✅ P0.2 fetch는 즉시 시작 가능 (Binance 무료 endpoints만 사용). Coinglass blocker는 P0.2 timeline에 영향 없음.

---

## Decision Default

사용자 응답 없이 30분 경과 시 (또는 진행 명시 시): **Option B 채택, P0.2 진입**.
