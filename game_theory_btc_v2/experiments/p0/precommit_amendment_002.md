# P0 Pre-Commit Amendment #002 — Hypothesis Phase A/B Split

**Amendment date**: 2026-05-01
**Original precommit**: `experiments/p0/precommit.md`
**Trigger**: Industry-wide 30d OI/L-S ratio REST limit (Binance/Bybit/OKX 모두 동일). Mandate v2 § 1.2.5의 720d OI 가정이 무료 인프라에서 infeasible.
**Authority**: 자율 mandate per user authorization (2026-05-01). Anti-fishing § 0.7 honest closure 정합 — silent re-scope 금지, explicit pre-commit으로 등록.

---

## Discovery

| Source | OI history depth | L/S ratio | Funding |
|--------|------------------|-----------|---------|
| Binance | 28d max | 28d max | 720d ✅ |
| Bybit | ~8d only | — | (TBD probe in progress) |
| OKX | 30d max | — | (TBD) |
| Coinglass paid | 720d+ | — | (paid only) |

→ **Free retail에서 720d OI/L-S = 불가능**. Hypothesis 재구성 또는 paid plan 필요.

---

## Decision: Hypothesis Phase A/B Split

### Phase A — 720d primary research (immediate, P0.2-P5)
**Testable hypotheses with free 720d data**:

| H | Original definition (mandate § 1.2.5) | Phase A revised version |
|---|---------------------------------------|-------------------------|
| H3 | 큰 long-liquidation magnitude × low quantile × neg funding → long edge | Liquidation proxy v2 (no OI) × low quantile × neg funding (see `proxy_formula_v2.md`) |
| H4 | Cascade window 5-30분 mean reversion edge | 동일 + 1m bar primary, proxy v2 |
| H5 | Short-side mirror (price high quantile, pos funding extreme) | 동일 + proxy v2 mirror |
| H7 (partial) | Spot-perp basis extreme + L/S ratio extreme → fade | **Basis-only**; L/S ratio 컴포넌트 Phase B로 |
| H8 | MAP-Elites archive | Phase A mechanisms (~30 catalog) only |
| H9 | Risk-Aware Thompson Sampling | Phase A mechanisms only |

### Phase B — Forward-collected, deferred to P3+ or 60-90d 후
**Hypotheses requiring OI/L-S/taker-ratio**:

| H | Required data | Earliest test date |
|---|--------------|-------------------|
| H1 | OI delta 720d | Forward collection 60d 누적 후 = ~2026-06-30 |
| H6 | OI rise + funding spike | 동일 |
| H7 (full) | spot-perp basis + L/S ratio | 동일 |

**Forward collector**: `scripts/data/collect_oi_ls_forward.py` 즉시 실행. 매시간 last 28d snapshot fetch + dedupe append. 60-90d 누적 시 Phase B 가능.

### Hypotheses Unchanged
- H2 (Kyle's λ): tick-rule on 1m bar — Phase A
- H8/H9: Phase A subset of mechanisms

---

## Effects on Existing Documents

### precommit.md (original)
- D6 deliverable "H1-H9 정량 정의" → Phase A subset (H2/H3/H4/H5/H7-partial/H8/H9) 우선. Phase B는 placeholder def + forward collector spec.

### h_features_inventory.md
- H1/H6/H7-full row → "deferred Phase B" 표시
- Forward collector dependency 추가

### holdout_seal.md
- 변경 없음 (sealed boundary는 fetched data 기준)

### precommit_amendment_001.md (friction stress)
- 변경 없음. 모든 Phase A test에 stress 적용

---

## Anti-Fishing Locks

1. ❌ Phase A 결과를 Phase B 가설로 retrofit 금지 (예: H3 fail 시 "OI 없어서 그래" 변명 silent shift)
2. ❌ Forward collector 데이터 누적 부족하다는 이유로 P3-P5 일정 연장 금지 — Phase A subset만으로 P3-P5 통과 평가
3. ❌ Coinglass paid 결정은 Phase A 결과 후만 (현 단계 비용 투입 금지)
4. ✅ Phase A에서 Phase B testable hypothesis가 추가로 free로 가능해지면 그 시점에 amendment 003 작성

---

## Coinglass Paid Decision Trigger

Phase A 결과에 따라:
- Phase A H3-H5 6-criteria 통과 → Phase B에서 OI/L-S로 정밀화 가치 생김 → Coinglass paid 검토
- Phase A H3-H5 모두 FAIL → 720d primary envelope falsified, OI augmentation 무의미 → Coinglass paid 비용 ROI 낮음 → 보류

---

## P0 7-Deliverable Update

| # | Deliverable | Phase A 영향 |
|---|-------------|------------|
| D1 | BTC data 720d | OHLCV/funding/spot 완전 / OI는 28d snapshot only |
| D2 | API access | Binance + BingX (변경 없음) |
| D3 | friction_model.py | 변경 없음 |
| D4 | 6-criteria validator | 변경 없음 |
| D5 | Baseline | 변경 없음 |
| D6 | H1-H9 정량 정의 | **Phase A subset 우선, Phase B placeholder** |
| D7 | ~30 mechanism catalog | 변경 없음 |

추가:
- D8: Forward collector spec + running cron (Phase B 활성화 condition)
- D9: Proxy formula v2 (OI 제거, sum-of-z-scores 또는 multiplicative re-design)

---

**Pre-commit signed (amendment 002)**: Claude Code agent, 2026-05-01.
사용자 surface 후 변경 없으면 default 적용. 본 amendment 변경 시 새 amendment doc.
