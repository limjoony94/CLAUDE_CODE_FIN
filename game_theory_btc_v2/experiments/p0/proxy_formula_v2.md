# Liquidation Proxy Formula v2 — OI-free, sum-of-z-scores

**Date**: 2026-05-01
**Trigger**: Coinglass free-tier blocker + Binance OI 30d limit + advisor sparsity warning
**Replaces**: v1 multiplicative AND (`funding_spike × oi_drop × price_velocity`) — sparse + OI 의존
**Used in**: H3, H4, H5 hypothesis testing (Phase A)

---

## v1 Issues

1. **Sparse**: 3 thresholds × 5% frequency each → joint ~0.01% (advisor critical)
2. **OI-dependent**: `oi_drop` requires 720d OI which is unavailable free
3. **Magnitude unclear**: binary 조건 결합 — proxy strength quantification 약함

---

## v2 Design

### Components (모두 720d Binance free 가용)

| Component | Definition | Lookahead-aware? | 근거 (mandate § 1.2) |
|-----------|------------|-----------------|---------------------|
| `z_funding` | `(funding_t-1 - rolling_mean_30d) / rolling_std_30d` (8h native, ffill to 1h grid, **t-1 shift**) | ✅ t-1 explicit | "음 funding (shorts paying)"의 z-score |
| `z_velocity_neg` | LONG entry: `-(close_t-1 - close_t-N) / (atr_14_t-1)` for N=12 (1h) — negative price move strength | ✅ t-1 | "스트레스 받은 long의 forced flow" |
| `z_volume_surge` | `(volume_t-1 - rolling_mean_30d) / rolling_std_30d` (1h volume) | ✅ t-1 | Liquidation cascade는 volume spike와 동행 (Brunnermeier-Pedersen 2005) |

추가 (optional, Phase B 도입 시 활성):
- `z_oi_drop_28d` (only when forward collector accumulates ≥30d) → activate amendment 003 시점

### Aggregation: sum-of-z-scores (advisor 권고)

```
proxy_score_t = z_funding_t + z_velocity_neg_t + z_volume_surge_t
```

(symmetric for short-side: H5 uses negative funding excluded, positive velocity, same volume)

### Threshold candidates (parameter sweep at P2)

| Variant | Threshold | Expected freq |
|---------|-----------|---------------|
| Loose | proxy_score > 4.0 (each ~1.3σ) | ~10-15% |
| Median | proxy_score > 5.5 | ~3-5% |
| Tight | proxy_score > 7.0 | ~1-2% |

P2 sweep: optimize on training data, evaluate on OOS holdout. Threshold는 P2 phase에 단일 lock.

### Edge case handling

- `rolling_std == 0` (constant period): set z = 0, exclude from event detection
- `rolling_window` < 30d (warmup): mark as `warmup_zone`, don't include in P2 evaluation
- Funding rate gap (data missing): forward-fill last value, mark `funding_stale=True`

---

## Anti-Fishing Locks

1. **t-1 shift mandatory**: 모든 z-score 컴포넌트는 t-1 OHLCV/funding 사용. t 시점 정보 의존 금지. Validator unit test로 enforce
2. **Threshold lock**: P2 sweep 후 단일 threshold 채택. 후속 priority에서 변경 금지 (post-hoc tuning)
3. **OI 추가 시 amendment 003 필수**: forward collector 활성화 시 `proxy_score_v3 = v2 + z_oi_drop_28d` 별도 amendment로 등록
4. **Negative result 인정**: H3-H5 proxy v2로 6-criteria 미통과 → "envelope falsified at OI-free proxy", retry with OI 우선시 합리화 금지 (paid plan 결정은 별도 process)

---

## Limitation Disclosure (Phase A honest closure 시 명시)

Proxy v2는 **liquidation cascade 자체의 ground truth가 아니라, cascade와 correlated 공개 신호의 합성**:
- Funding spike: 실제 funding rate jump 측정 가능, but cascade 직접 신호 아님
- Volume surge: cascade 시 동행하지만 normal trend day도 발생
- Velocity neg: cascade는 빠른 가격 움직임 동행, but normal pullback도 동일

따라서 H3-H5의 "long-liquidation cascade" hypothesis 검증은 **proxy 가정 하에서**의 결과. 진짜 liquidation USD 측정은 paid (Coinglass) 또는 forward websocket collector 필요.

이 한계는 Phase A result 보고에 명시. P3+ Phase B 활성화 또는 paid 결정 시 재검토.

---

## Calibration (advisor 권고 #3 partial)

Forward `!forceOrder@arr` websocket collector를 P0.2 동안 시작 → 30d 누적 후:
- Proxy_score_v2 ≥ threshold 시점의 실제 liquidation USD 측정
- Spearman ρ between proxy_score and liquidation_usd 계산
- ρ > 0.6 → proxy 충분, Phase A H3-H5 결과 valid
- ρ < 0.6 → proxy 약함, Phase A 결과는 "proxy-based detection" 한정 의미. Coinglass 결정 trigger

(websocket collector 별도 script — 본 amendment 외 spec)

---

**Signed**: Claude Code agent, 2026-05-01.
P2 entry 전 변경 시 새 v3 doc.
