# P0 Closure — Zero-Base Inventory + Theory Grounding (COMPLETE)

**Date**: 2026-05-01 (Day 1 wrap; budget 7d, used 1d via background fetcher + parallel writes)
**Status**: ✅ All 7 mandate deliverables + 2 derived deliverables PASS
**Mandate version**: v2 (zero-base from R26/C1 lineage)
**Project root**: `C:\Users\J\OneDrive\CLAUDE_CODE_FIN\game_theory_btc_v2\`

---

## 9 Deliverable Status

| # | Deliverable | Status | Evidence |
|---|-------------|--------|----------|
| **D1** | BTC perp 1d/1h/5m/1m + spot 1h, 720+d | ✅ | `data/btc_perp_*.parquet` + `audit_results.json` (gap=0%, ohlc_invalid=0, t_close_assertion=True for all) |
| **D2** | Binance + BingX API access | ✅ | `keys.py` helper + 3 probe scripts. Binance primary, BingX live. Coinglass paid-tier 차단 confirmed |
| **D3** | friction_model.py (3 scenarios + funding) | ✅ | `scripts/validators/friction_model.py`. 3 scenarios + funding accrual separated. Maker stub via amendment_003 |
| **D4** | bootstrap_six_criteria.py | ✅ | `scripts/validators/bootstrap_six_criteria.py`. Vectorized CBB. p5 = bootstrap-of-means (post-advisor fix) |
| **D5** | Buy-and-hold + 1× long + random baselines | ✅ | `baselines_result.md`. Validator end-to-end demonstrated. B&H 5/6 PASS, random 1/6 PASS |
| **D6** | H1-H9 정량 정의 (registry) | ✅ | `hypothesis_registry.yaml` (7 Phase A + 3 Phase B). YAML schema locked |
| **D7** | ~30 mechanism catalog (Phase A subset) | ✅ | `mechanism_catalog.yaml` (14 Phase A + 4 Phase B = 18, NOT padded to 30 per advisor) |
| **D8** | Forward collector (Phase B prep) | ✅ | `collect_oi_ls_forward.py` + 1차 run. 5 parquets accumulating. 사용자 cron 등록 권장 |
| **D9** | Proxy formula v2 (OI-free) | ✅ | `proxy_formula_v2.md`. Sum-of-z-scores: z_funding_neg + z_velocity_neg + z_volume_surge |

**총 36/36 unit tests PASS** (`tests/test_p03_validators.py`).

---

## Major Discoveries (2026-05-01)

### Industry-wide constraint: OI history 30d max
Mandate v2 § 1.2.5 720d OI 가정이 free retail에서 infeasible:
- Binance OI: 28d max via REST
- Bybit OI: 8d via CCXT
- OKX OI: 30d max
- Coinglass paid: 720d+ ($29-79/month)

→ **Phase A/B split** (precommit_amendment_002): Phase A free 720d data로 H3-H5/H7-basis/H8-H9 즉시. Phase B (H1/H6/H7-full) forward collector 30-90d 누적 후.

### Coinglass free tier ≠ liquidation history
- 무료 키 자체는 valid, 그러나 모든 liquidation/funding/OI history endpoint = `{"code":"401","msg":"Upgrade plan"}`
- Resolution: **Option B (proxy v2)** — Binance free funding + price + volume sum-of-z-scores
- Coinglass paid 결정은 Phase A 결과 후 (현 단계 ROI 불명확)

### Buy-and-hold drift +0.110%/day exceeds P2 target 0.10%/day
- 강력한 anti-fishing 함의: P2 strategy가 단순 holding보다 못해도 mandate's 0.55 p_beats threshold 통과 가능
- **Lock**: P2-P6 `min_p_beats_baseline = 0.70` + `baseline_pnl mandatory` (advisor 2026-05-01)

### p5 interpretation correction
- 원안 (raw daily 5-percentile): BTC volatility로 사실상 통과 불가 (mandate intent 왜곡)
- **Lock**: bootstrap 5-percentile of MEANS = 95% one-sided lower CI of mean = "95% confident edge is positive"
- Anti-fishing aligned

---

## Holdout Boundary (Sealed OOS, IMMUTABLE)

| Field | Value |
|-------|-------|
| T_anchor | 2026-05-01T14:00:00 UTC |
| T_seal_start | 2025-11-02T14:00:00 UTC |
| T_p3_holdout_start | 2026-02-18T14:00:00 UTC |
| Sealed window | 180 days (last 25%) |

P0.5-P5 fitting/searching 단계에서 sealed window 사용 시 `assert_no_sealed_data()` AssertionError. P6에서 단 1회 final eval.

---

## Key Documents (P0 deliverable artifacts)

| Path | 내용 |
|------|------|
| `precommit.md` | 7-deliverable PASS criteria (immutable post-execution) |
| `precommit_amendment_001.md` | Friction stress 시나리오 의무 |
| `precommit_amendment_002.md` | Hypothesis Phase A/B split |
| `precommit_amendment_003.md` | Maker scenario flag (P3에서 활성화) |
| `holdout_seal.md` | Sealed boundary + concrete amendment |
| `proxy_formula_v2.md` | OI-free liquidation proxy 정의 |
| `six_criteria_thresholds.md` | v2 (post-advisor fix): p5 + min_p_beats=0.70 + baseline mandatory |
| `h_features_inventory.md` | H1-H9 feature input × P0.2 fetch coverage |
| `coinglass_blocker_resolution.md` | Coinglass blocker discovery + Option B 선택 |
| `infrastructure_lessons_inherited.md` | CCXT pitfalls + BT-LIVE parity (P6 deploy 시 의무) |
| `fetch_summary.md` | P0.2 closure summary |
| `audit_results.json` | Quality audit raw |
| `baselines_result.md` | P0.4 closure (validator end-to-end) |
| `hypothesis_registry.yaml` | P0.5: 7 Phase A + 3 Phase B hypotheses |
| `mechanism_catalog.yaml` | P0.6: 14 Phase A + 4 Phase B mechanisms |
| `result.md` (this) | P0 closure |

---

## Next Steps

### Immediate (사용자 작업, 5분, 1건)
**Forward collector cron 등록** (advisor 강력 권고). Windows Task Scheduler:
```powershell
schtasks /create /sc hourly /tn "GameTheoryBTC_ForwardCollector" /tr "powershell -Command \"cd 'C:\Users\J\OneDrive\CLAUDE_CODE_FIN\game_theory_btc_v2'; python scripts\data\collect_oi_ls_forward.py\"" /st 00:05
```
매일 지연 = 손실 데이터. 등록 안 해도 Phase A 진행은 가능.

### P2 Entry (Force-Flow Reversal Hypothesis Testing)
- H3 (M001 force_flow_long_proxy) primary
- H4 (M003 cascade window 5m) secondary
- H5 (M002 short mirror) optional
- Pre-commit before code: `experiments/p2/precommit.md` (per anti-fishing § 0.1)
- 다음 advisor 호출은 P2 closure (PASS/FAIL/PARTIAL 결과 시점)

### Phase B Activation (deferred ~2026-06-30)
- Forward collector 30d 누적 후 H1, H6, H7-full activation amendment_004
- OI/L-S 데이터로 proxy v2 calibration

---

## Honesty Disclosure

- Phase A 6-criteria 통과 prior probability: <30% per friction-floor 27 mechanisms 0 deployable evidence
- Phase A FAIL은 expected outcome; honest closure로 next steps
- Coinglass paid plan / Phase B 활성화는 Phase A 결과 후 ROI 평가
- 1%/day → whale-tier 2.5년 시나리오 D는 < 2% probability (mandate § 10)

P0 budget 7d 중 1d 사용. 6d 여유로 P2 진입 가능.

**P0 CLOSED** ✅
