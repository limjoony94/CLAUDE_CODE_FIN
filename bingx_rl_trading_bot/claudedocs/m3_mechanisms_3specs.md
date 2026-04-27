# M3 — 3 Creative Mechanisms (사전 등록, 별도 commit)

> **Date**: 2026-04-28
> **Authority**: 사용자 명시 위임 ("전부 동의, 전부 확인 및 연구해서 비교 분석 진행")
> **Frame**: 3 mechanisms × 5 critiques = comparative 3×5 matrix (winner label 금지).
> **Origin**: 26-cell map family-level pattern (cross-instrument > single-asset OHLCV)

## Constants

- Asset: BTC/USDT (모든 specs)
- Timeframe (entry/exit): 15m
- Trend filter (specs that use): 1h EMA20>EMA50 AND 4h close>EMA50 (LONG); SHORT mirror
- Friction (default): 0.20%/trade (entry 0.05% slip + exit 0.05% slip + 0.10% taker fee RT)
- Position: N=1, leverage 1x
- min_bars_between_trades: 2
- Exit common framework (모든 specs):
  - SL: max(직전 15m swing_low, entry − 2.0 × 15m_ATR)  ← C1 D1 lesson 적용 (M1-A 1.5×ATR보다 wider)
  - Emergency_SL: −1.5% hard
  - TP_trail: best_price − 2.0 × 15m_ATR (trailing, ratchet)
  - Timeout: 16 bars (= 4h)

> Entry rules는 mechanism별 다름. Exit framework는 통일 (3 mechanisms 비교 시 entry alpha만 분리).

---

## Mechanism α — ETH-leads-BTC + 고변동성 regime conditional

**Origin**: C.3 (Δp50 +0.06, Δ%>fr +6.24 pre-strict PASS) idea 위에 regime filter 추가.

**Hypothesis**: ETH-BTC lag relationship은 universal 아님. 고변동성 regime에서만 발현. 평균적 분석은 dilute.

**Entry rule (LONG)**:
1. ETH 15m return prev bar > +0.3%
2. BTC 15m return prev bar < +0.1% (BTC lagging ETH up-move)
3. 1h EMA20 > 1h EMA50 AND 4h close > 4h EMA50 (LONG trend)
4. **Regime gate**: BTC 15m ATR(14) > 70th percentile of past 200 bars (high-vol regime active)

**Entry rule (SHORT)**: mirror (ETH < -0.3%, BTC > -0.1%, trend SHORT, regime gate).

**Critical parameters** (sensitivity probe targets):
- ETH return threshold: 0.3% (probe ±20% = 0.24/0.36)
- BTC lag threshold: 0.1% (probe ±20%)
- ATR percentile: 70th (probe 60th/80th)
- ATR lookback: 200 bars (probe 150/250)

---

## Mechanism β — Spread mean-rev × correlation breakdown compound

**Origin**: C.1 (Δp50 +0.06, Δ%>fr +8.90) AND C.2 (Δ%>fr +10.23, asym +0.094)을 두 독립 신호 아닌 **compound condition**으로.

**Hypothesis**: Spread 이탈 (statistical) + correlation 깨짐 (regime) = 둘 다 만족 시 directional edge. 단일 조건은 noise.

**Entry rule (LONG)**:
1. log(BTC_close / ETH_close) z-score (50-bar) < −2.0 (BTC underpriced vs ETH)
2. rolling 50-bar correlation(BTC_15m_return, ETH_15m_return) < 0.5 (correlation broken — 보통 ~0.85)
3. 1h+4h trend filter LONG
4. Both conditions (1) AND (2) must hold simultaneously on entry bar

**Entry rule (SHORT)**: z > +2.0 AND correlation < 0.5 AND trend SHORT.

**Critical parameters**:
- z threshold: 2.0σ (probe 1.6/2.4)
- correlation threshold: 0.5 (probe 0.4/0.6)
- correlation lookback: 50 bars (probe 40/60)

---

## Mechanism γ — Funding sustained extreme + cross-asset confirmation

**Origin**: A.3 (Δp50 +0.26 — 26 cells 중 single 가장 강한 metric, asym +0.18 favorable) + ETH 방향 확인 추가.

**Hypothesis**: Funding은 position imbalance 누적 신호 (rebalance 잠재력). Cross-asset 확인은 entry timing 정확도 향상. 단일 funding은 timing 부정확.

**Entry rule (SHORT — fade overheated longs)**:
1. funding_8period_sum ≥ +0.24% (= 8 periods × 0.03% average sustained)
2. 15m RSI(14) ≥ 70 (overbought 확인)
3. **Cross-asset confirm**: ETH 15m return prev bar < 0 (ETH already declining → BTC reversion likely)
4. Trend filter NOT used (counter-trend trade by design)

**Entry rule (LONG — fade overheated shorts)**:
1. funding_8sum ≤ −0.24%
2. RSI ≤ 30
3. ETH return prev bar > 0
4. Trend filter NOT used

**Critical parameters**:
- funding sustained threshold: 0.24% (probe 0.18/0.30)
- funding consecutive periods: 8 (probe 6/10)
- RSI threshold: 70/30 (probe 65/75)

---

## Predictions (사전 등록 — calibration source)

### Mechanism α (ETH-lag + 고변동성)

| Critique | Predicted | Confidence | Rationale |
|----------|-----------|-----------|-----------|
| C1 random baseline | marginal positive | LOW | C.3 pre-strict PASS + regime filter 가능성. Sample size 줄어들 우려 |
| C2 look-ahead | PASS | MED | Careful coding default. ETH causal merge 주의 |
| C3 friction stress | dies at MED (0.30%) | MED | Δp50 +0.06 너무 작아 friction 못 넘음 |
| C4 overfitting | sharp peak risk | MED | Regime threshold 70th percentile sharpness 의심 |
| C5 bootstrap | skip (won't reach) | – | C3에서 die 예상 |

### Mechanism β (spread × correlation compound)

| Critique | Predicted | Confidence | Rationale |
|----------|-----------|-----------|-----------|
| C1 random baseline | FAIL | MED | Compound condition으로 sample size 매우 작아질 가능성 |
| C2 look-ahead | PASS | MED | Spread + correlation 모두 backward-looking |
| C3 friction stress | skip | – | C1에서 die 예상 |
| C4 overfitting | – | – | Skip |
| C5 bootstrap | – | – | Skip |

### Mechanism γ (funding sustained + cross-asset)

| Critique | Predicted | Confidence | Rationale |
|----------|-----------|-----------|-----------|
| C1 random baseline | marginal positive | MED | A.3 strongest single metric. Cross-asset filter quality 향상 가능 |
| C2 look-ahead | PASS | MED | Funding backward, RSI backward, ETH prev bar |
| C3 friction stress | dies at HIGH (0.50%) | MED | Δp50 +0.26 + selectivity로 friction 0.50%까지 견딜 가능성 |
| C4 overfitting | plateau possible | LOW-MED | Funding threshold가 sharp가 아닌 plateau pattern일 수 있음 |
| C5 bootstrap | ≈ random or marginal | LOW | 26 cells 1 PASS 통계 무의미. window-level positive mean 의문 |

### Distribution
- 2 marginal positive (LOW-MED): α, γ
- 1 FAIL prediction: β
- 0 expected to survive C5 bootstrap

### Most likely surprise
- α survives C3 friction MED: regime conditional이 진짜 alpha면 small per-trade edge OK
- γ FAIL at C1 random: cross-asset confirmation이 signal 줄이지만 alpha는 못 만들 수 있음

### Most informative outcome
- All three die at C1: 26-cell pattern은 spurious. Paradigm shift 강한 evidence.
- γ survives all 5: A.3 single near-PASS가 진짜 alpha. funding-based strategy 후속 가능.

---

## Critique 정의 (모든 specs 동일 적용)

### C1 — Random baseline
- 5 random seeds × cand sample size random entries on same eligible universe
- Pass: Δp50 ≥ 0.05pp AND Δ%>fr ≥ 5pp (relaxed, R3 strict는 verify 단계에서)

### C2 — Look-ahead audit
- Manual code review: 모든 indicator는 past data만, MTF merge는 backward direction
- Automated leak test: indicator 계산을 truncated(t) vs full(t+future) 비교, t 시점 값 동일해야 함
- Pass: 모든 indicator/merge에서 leak 0

### C3 — Friction stress
- 4 scenarios: 0.20% (BASE) / 0.30% (MED) / 0.50% (HIGH) / 0.80% (STRESS)
- Pass: BASE 양수 net daily PnL AND MED 양수

### C4 — Overfitting probe
- 각 critical parameter ±20% sensitivity (3 cells per parameter: low/base/high)
- 3-fold expanding WF
- Parameter surface plot (sharp peak vs plateau)
- Pass: 모든 ±20% cell 동일 부호 (양수) AND WF 2/3 PASS

### C5 — Bootstrap 3-day stability (criterion 8)
- 1000 random 3-day windows, full BT each
- Pass: Core 3 (mean > 0, pos rate ≥ 50%, P5 > -1%) AND Relative (P(cand > random_baseline) ≥ 60%)
- Expensive — survivors of C1-C4 only

---

## Anti-pattern guard

- 4번째 mechanism 추가 충동 → 기록만 (별도 round)
- C1 PASS = "endorsement" 아님 — 각 critique 독립 threat
- "Mechanism α best" 같은 winner label 금지 — matrix만 보고
- Critique 5 (bootstrap) 미survivor에 사전 실행 금지 (compute 비용)
- Critique 결과 fail-fast은 mechanism별, 비교 전체 fail-fast 아님

## Output

`results/m3_3x5_matrix_*.json`:
```json
{
  "mechanisms": ["α", "β", "γ"],
  "critique_results": {
    "α": {"C1": {"pass": ..., "metrics": ...}, "C2": ..., ...},
    ...
  },
  "matrix": [
    ["α", "PASS", "PASS", "FAIL", "skip", "skip"],
    ["β", "FAIL", "skip", ..., ...],
    ["γ", "PASS", "PASS", "PASS", "PASS", "PASS"]
  ]
}
```

비교 분석 보고서: `docs/04-report/m3_comparative_analysis_*.md` (matrix + per-mechanism die location + 사용자 결정 옵션, winner label 없음).

---

## Multi-session expectation

- **Session 1 (이번)**: spec doc commit + BT runner (spec-agnostic) + C1+C2 implementations + 부분 C3 → 3 mechanisms 1차 critique 1-2 결과
- **Session 2**: C3+C4 완료 + C5 (survivors only) + 3×5 matrix + 보고서

총 코드량 추정: BT runner ~250 lines + 5 critiques ~150 lines each = ~1000 lines reusable artifact.
