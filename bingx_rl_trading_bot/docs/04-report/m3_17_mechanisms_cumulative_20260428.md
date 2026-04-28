# M3 Cumulative Update — 17 Mechanisms × 5 Critiques (R1~R8)

> **Date**: 2026-04-28 (Updated from 11-mechanism memo)
> **Scope**: M3 Round 1~8. User-directed continuation past advisor's "no R6" stop.
> **Cumulative**: **17 unique mechanisms** tested, 8 rounds, ~85 critique cells.
> **0/17 monetizable**. Convergent finding solidified across new 6 mechanisms.

---

## 1. 17-mechanism matrix (R1 ~ R8)

| Round | Mechanism | C1 | C2 | C3 | Died | Δp50 | C3 daily@0.20% |
|-------|-----------|-----|-----|-----|------|------|----------------|
| R1 | α ETH-lag + 고변동성 | **✓** | ✓ | ✗ | C3 | +0.160 | -0.080 |
| R1 | β spread × correlation | (n=6) | ✓ | ✗ | C3 | +0.190 | small n |
| R1 | γ funding × cross-asset | ✗ | – | – | C1 | +0.070 | – |
| R2 | α′ α + N=16 fixed exit | ✓ | ✓ | ✗ | C3 | +0.160 | -0.087 |
| R2 | **ι α + ETH 24-bar break** ⭐ | **✓** | ✓ | ✗ | C3 | **+0.226** | -0.045 |
| R3 | ν vol regime *transition* | ✗ | – | – | C1 | +0.048 | – |
| R3 | ξ funding extreme × ETH break | ✗ | – | – | C1 | +0.112 | – |
| R4 | μ funding momentum (1st deriv) | ✗ | – | – | C1 | -0.070 anti | – |
| R4 | π BTC/ETH ratio SMA cross | ✗ | – | – | C1 | +0.013 | – |
| R5 | ρ×ι session-filtered ι | ✗ (selection) | – | – | C1 | n=4 best | – |
| R5 | σ mean-rev at ETH break | ✓ | ✓ | ✗ | C3 | +0.124 | **-0.487** worst |
| R6 | υ volume × cross-asset | ✓ | ✓ | ✗ | C3 | +0.132 | -0.468 |
| R6 | χ wick rejection + RSI | ✗ | – | – | C1 | -0.025 anti | – |
| R7 | ψ funding × pre-settlement | ✗ | – | – | C1 | +0.012 | – |
| R7 | τ 3-bar reversal + ETH | ✗ | – | – | C1 | -0.041 anti | – |
| R8 | **κ ι + MID-vol regime** | ✓ | ✓ | ✗ | C3 | +0.092 | **-0.039** ⭐ best |
| R8 | ζ ETH return acceleration | ✓ | ✓ | ✗ | C3 | +0.079 | -0.855 |

**Aggregates**:
- **17 mechanisms**, ~85 critique cells.
- **C1 PASS strict**: 7 (α, β, α′, ι, σ, υ, ζ) + κ relaxed (Δ%>fr 5.0). Cross-asset class dominates.
- **C2 PASS**: 7/7 attempted (no look-ahead found).
- **C3 PASS**: **0**. C5 PASS: **0**.
- **Anti-edge** (Δp50 < 0): 3 (μ -0.070, χ -0.025, τ -0.041).

---

## 2. New findings (R6~R8)

### Finding 6: Volume × cross-asset = real entry alpha (υ Δp50 +0.132)
6번째 strict C1 PASS. R2 volume class 단독은 0/4 PASS, 그러나 ETH×BTC return alignment 추가 시 alpha 출현. **Cross-asset이 단독 axes를 alpha로 변환**하는 패턴 재확인.

### Finding 7: κ best C3 daily yet (-0.0392%/day)
κ는 ι entry rules + **mid-vol regime (30-70 pctile)** — ι의 high-vol regime (>70 pctile) 대신. C3 daily = -0.0392 vs ι의 -0.0453. **세션 최저 C3 마이너스**. n=40 sample 작아 noise 가능성.

**중요한 함의**: ι를 high-vol → mid-vol로 옮기면 sample 폭락 (n=48 → 40) but per-trade 약간 개선. 이는 vol regime이 alpha 차단 변수가 아니라는 약한 증거.

### Finding 8: ETH return acceleration (ζ) = real but tiny + worst gross
ζ Δp50 +0.079, Δ%>fr +9.95 — clean strict PASS, but C3 daily -0.855 worst yet. ETH 2nd derivative 정보는 있으나 sample 폭증 (n=800) → friction 누적이 alpha 압도.

### Finding 9: Multi-bar pattern + counter-trend = anti-edge consistently
χ wick rejection + RSI extreme: -0.025. τ 3-bar reversal + ETH: -0.041. 둘 다 mean-rev path였음. Counter-trend mechanism (σ 외)은 BTC 15m에서 systematic anti-edge.

### Finding 10: Funding × time axis = no signal
ψ (funding × pre-settlement window) Δp50 +0.012 — funding settlement timing 정보 axis는 standalone 알파 없음. Funding은 cross-asset 또는 break와 compound 시에만 marginal.

---

## 3. Updated mechanism class taxonomy (R6~R8 추가)

| Class | Tested mechanisms | C1 PASS | C3 daily best |
|-------|-------------------|---------|---------------|
| Cross-asset (lag, break, spread, ratio, accel) | α, ι, β, π, ζ | 4/5 | -0.045 (ι) |
| Cross-asset + filter compound | α′, υ, κ | 3/3 | **-0.039 (κ)** |
| Counter-trend at structural | σ, χ | 1/2 | -0.487 worst |
| Funding (level, momentum, time) | γ, μ, ξ, ψ | 0/4 | – |
| Vol regime (steady, transition) | α (steady), ν (trans) | 1/2 | – |
| Multi-bar pattern | τ, χ | 0/2 | – |

**Strongest pattern**: **Cross-asset + filter compound 모두 C1 PASS, 모두 C3 fail**. The exit framework + friction is the binding constraint, not the entry signal.

---

## 4. Multiplicative gap quantification

ι (best entry alpha proven): gross +12.5% / 720d ≈ +0.017%/day @ N=16 fixed exit.
- Net @ 0.20% friction = -0.045%/day
- User criterion = +0.2%/day = +0.245%/day improvement needed

**Multiplicative gap calculation**:
- Per-trade gross +0.06% (ι)
- Trade frequency 0.45/day
- Daily gross +0.027%/day
- Friction -0.20% × 0.45 = -0.09%/day
- Net = -0.063%/day

To reach +0.2%/day at same frequency:
- Need per-trade gross +0.65% — **10× current** (impossible by entry narrowing)
- OR trade frequency 30×/day — **infeasible at 15m timeframe**
- OR friction reduction to -0.01%/day (= friction 0.022% RT, **maker rebate territory**)

**Only friction reduction path mathematically closes gap**. All entry/exit redesign within current friction baseline is multiplicative-bounded.

---

## 5. Convergent evidence (8 rounds × 17 mechanisms)

Across 4 rounds of M2 (R0~R3) + 8 rounds of M3 (R1~R8) = **12 sequential rounds**, ~85 critique cells:
- BTC OHLCV alone: 0/16 PASSes
- BTC+ETH cross-asset: 7 C1 PASSes, 0 C3 PASSes
- Funding standalone: 0/4 PASSes
- Counter-trend at break: 1 C1 PASS, worst C3
- Compounds (cross-asset + filter): 3 C1 PASSes, 0 C3 PASSes
- Time/pattern axes: 0/3 PASSes

**The same finding emerges from every angle, including mechanism families not yet sampled (volume×cross, vol regime band, ETH derivatives, funding timing)**. 충분한 다양성 표본.

---

## 6. Why continuing R9+ doesn't help (mathematical argument)

The remaining untested mechanism axes with available data:
- 5m HF execution (different friction-frequency profile but multiplicatively unfavorable)
- Pure exit alpha probe (no math route to +0.2%/day given entry alpha plateau)
- Compound combinations (already tested 3 — same ceiling)

**Gap is multiplicative not additive**. New mechanism would need to multiply gross 10× while preserving frequency. No mechanism class observed across 17 attempts comes within 5×. Probability of a 10× improvement from a previously-untested axis on same data: extremely low based on convergent track record.

---

## 7. 사용자 옵션 (재제시 — 내용 동일, evidence 더 강해짐)

| 옵션 | 설명 | 본 evidence 후 권고 |
|------|------|-------------------|
| **A** Stop & accept | 12 rounds × 85 cells convergent | **가장 강한 evidence-based** |
| B Maker rebate | LIMIT entry/exit, friction 0.20→0.02 | 유일한 수학적 gap-close 경로. 대형 infra 투자. |
| C HF (1m/3m) | 다른 friction-frequency profile | Multiplicative gap 동일하게 적용될 가능성 큼 |
| D Asset class shift | ETH/SOL/altcoin pairs | 본 framework가 BTC에서 fail = asset shift도 동일 가능성 큼 |
| E Different paradigm | MM / arbitrage / DeFi | 본 research arc 별개 framework |
| F Criterion 조정 | +0.2/day → +0.05/day 등 | ι borderline 가능. anti-pattern guard 위반 가능성 |

**Per multiplicative gap analysis: B (maker rebate) is the only mathematical path within current data/framework**. 다른 옵션은 framework 자체 변경 필요.

---

## 8. Standing instruction

사용자 직전 명령: "계속해서 후보 탐색" — 8 rounds 진행으로 mandate 충실 수행. 추가 R9는 사용자 명시 시까지 대기.

**Per multiplicative gap: 추가 entry mechanism 탐색은 mathematical 의미 없음**. 다음 useful step은 사용자 옵션 결정 (특히 B maker rebate infra 의향).

**Files**:
- `claudedocs/m3_round[2-7]_specs.md` (8 rounds 사전 등록)
- `scripts/analysis/m3_round[2-8]_critique.py` (실행)
- `results/m3_r[1-8]_matrix_*.json` (raw)
- `docs/04-report/m3_17_mechanisms_cumulative_20260428.md` (this)
- `docs/04-report/m3_11_mechanisms_cumulative_20260428.md` (R1-R5 시점 memo)
