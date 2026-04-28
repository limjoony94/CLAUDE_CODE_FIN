# Path B R2 Pre-Registration — Cross-Sectional Reversal (10 Crypto, Daily, Weekly)

**Date pre-registered**: 2026-04-29
**Status**: PRE-COMMIT
**Track**: Path B (advisor structural opposite to R1)

**Honest prior**: R1 (cross-sectional momentum, lookback=30d) was qualitatively different from prior arc — edge above friction (+0.13%/wk net), WF 3/5 PASS, but bootstrap 48.4% (1σ below 50%, borderline) and train/test sign disagreement → strict OOS FAIL. R2 tests the **structural opposite** factor.

---

## Why R2

Per advisor:
> "If both momentum (R1) and reversal (R2) fail at strict OOS but R1 has the regime-dependent positive edge: cross-sectional dimension is real but unstable single-factor.
> If R2 passes where R1 failed: you've characterized regime — momentum works in trending, reversal in ranging.
> If both pass at edge > friction: blending becomes viable.
> If both fail with broken economics (gross < friction): cross-sectional crypto factors at retail friction don't survive."

This is the cleanest binary disambiguation possible in this envelope.

---

## Locked Mechanism — `xs_reversal_weekly_top3`

**Theory** (De Bondt-Thaler 1985 + Lehmann 1990):
Short-term reversal: assets with worst recent returns outperform; best recent returns underperform. Lehmann (1990, QJE) documented 1-week reversal in equities. Crypto extension expected via overreaction-correction dynamics.

**Algorithm** (identical to R1 except direction):
1. Universe (locked, identical to R1): BTC, ETH, SOL, BNB, XRP, ADA, DOGE, AVAX, TRX, LINK
2. Each Monday at UTC 00:00:
   - Compute **trailing 7-day total return** for each coin
   - Rank coins
   - **Long bottom-3** (worst recent), **Short top-3** (best recent), equal weight
3. Hold 7 days; rebalance Monday
4. Friction: 0.07% per transaction (identical to R1)

**Locked Parameters (NO retuning)**:
```python
LOCKED = {
    'universe_size': 10,
    'lookback_days': 7,           # Lehmann 1-week reversal
    'long_bottom_n': 3,           # buy worst (reversal direction)
    'short_top_n': 3,             # short best
    'rebalance_frequency_days': 7,
    'friction_per_transaction': 0.07,
    'equal_weight': True,
}
```

**Theory source**:
- De Bondt & Thaler (1985): "Does the Stock Market Overreact?" — long-term reversal
- Lehmann (1990): "Fads, Martingales, and Market Efficiency" — 1-week reversal
- 7-day lookback: Lehmann's exact horizon

**Difference from R1** (advisor's "structural opposite"):
- R1: lookback=30d, long winners, short losers (momentum continuation)
- R2: lookback=7d, long losers, short winners (reversal)

---

## Exit / Friction Identical to R1

Same `run_xs_reversal` framework with reversed long/short selection. Friction 0.07%/transaction unchanged.

---

## Pre-Registered Tests

### Pre-run Dispersion Gate
- Median cross-sectional dispersion of trailing 7d returns ≥ 5%
- If FAIL → R2 inconclusive

### Test 1: WF 5-fold Expanding
- **Pass**: ≥3/5 folds avg_weekly_net > 0

### Test 2: Bootstrap 1000 × 30-day NET-return windows (from full BT)
- **Pass**: pos_rate ≥ 50%

### Test 3: Train/Test 60/40
- **Pass**: BOTH train AND test avg_weekly_net > 0

### Friction-Aware Reporting
- Log avg_weekly_gross at every fold

---

## Verdict Logic (per advisor's 4 patterns)

After R2 results:

| R1 outcome | R2 outcome | Action |
|------------|------------|--------|
| FAIL (regime edge) | FAIL (broken econ) | Cross-sectional dimension real but unstable single-factor → Path B closure |
| FAIL | PASS | Regime split: reversal works where momentum doesn't → R2 candidate for deployment |
| PASS | PASS | Both factors edge > friction → blending viable → synthesis next |
| FAIL | FAIL (broken econ) | Cross-sectional crypto factors at retail friction don't survive → Path B closure |

R1 was FAIL with regime-dependent edge. R2 outcomes:
- FAIL with edge > friction: pattern 1 (cross-sectional unstable)
- FAIL with broken econ (gross < friction like R36-R41): pattern 4 (Path B closure)
- PASS: pattern 2 or 3

**No advisor call after R2 unless result doesn't fit one of these patterns** (per advisor explicit instruction: "synthesize, don't escalate").

---

## Anti-Adjustment Provisions

1. No sweep, no retuning if FAIL
2. No friction relaxation
3. No criterion relaxation
4. No mechanism swap as "R2 v2" — different mechanism = R3
5. ALL params locked from theory before any data observation

---

## Sizing/Deployment Note (advisor)

Even if R1+R2 blend passes everything: $1,500 capital with R1's observed 58% MDD = ~$870 drawdown. Position sizing rule required (e.g., fixed-fractional cap at 20-25% MDD) before any deployment plan. To be addressed in synthesis if applicable.

---

## Hash anchor

Committed BEFORE backtest run.
