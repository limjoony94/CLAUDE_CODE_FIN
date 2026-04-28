# M3 28-Round Comprehensive Final Report

> **Date**: 2026-04-28
> **Scope**: M3 Round 1 ~ Round 28 (full arc)
> **Mandate**: 사용자 explicit "수익성 모델 찾을 때까지", "계속해서 진행, 반복적인 비판 평가 Cycle", "다양한 각도에서 비판 검증"
> **Result**: 30+ distinct mechanisms, comprehensive critique suite, 0 strict-criterion passes, multiple statistical tests confirm absence of extractable edge in current data/constraints.

---

## 1. Total Arc Summary

### 28 rounds × ~30 unique mechanisms

**Mechanism families tested**:
1. Single-asset mean reversion (R1 RSI, R3 various)
2. Cross-asset (α ETH-lag, ι ETH break, β spread, ξ funding+break)
3. Volatility regime (α steady, ν transition)
4. Counter-trend at structural break (σ)
5. Volume × cross-asset (υ)
6. ETH return acceleration (ζ)
7. Funding rate (γ level, μ momentum, ψ pre-settlement)
8. Pattern recognition (τ 3-bar, χ wick, ψ′ pattern reversal)
9. Multi-bar formations (engulfing, hammer, shooting star)
10. Dynamic exits: ATR-based (R20), structural (R21), asymmetric R:R (R22), VWAP (R23), pullback (R24-R25), Bollinger Squeeze (R26)
11. Ensemble (R27 R21+R24)
12. Time-of-day filtering (R28)

### Critique Angles Applied
1. ✓ Look-ahead bias audit (no leaks across all)
2. ✓ Friction comprehensive (taker / mixed / maker scenarios)
3. ✓ 3-day random window bootstrap (~1000 windows, user's spec)
4. ✓ Walk-forward 5-fold expanding
5. ✓ 3-way train/val/test splits
6. ✓ Per-trade gross > taker fee check
7. ✓ Trade frequency check (≥2/day)
8. ✓ WR + R:R structure check
9. ✓ Statistical significance (t-test)
10. ✓ Per-direction breakdown (LONG vs SHORT)
11. ✓ Time-of-day decomposition
12. ✓ Edge decay analysis (rolling 60-day)
13. ✓ Sharpe + Max Drawdown
14. ✓ Regime decomposition (early/mid/late chunks)
15. ✓ Train→test consistency (post-hoc filter validation)

---

## 2. Best 6-Mechanism Comparative (R20-R28 Subset)

| Round | Mechanism | avg_gross/trade | WR | R:R | Bootstrap pos_rate | Notes |
|-------|-----------|-----------------|-----|-----|-------------------|-------|
| **R21** | Pattern reversal + structural | **+0.010%** | 34.3% | **1.54** | 11.8% | **BEST avg_gross** |
| R24 | Pullback continuation | +0.005% | **47.8%** | 0.69 | 11.8% | **BEST WR** |
| R25 | R24 + TP_LB sweep | +0.007% | 47.4% | 0.68 | 11.8% | Sweep optimum |
| R20 | Confluence breakout + ATR | -0.007% | 23.6% | 1.00 | 2.8% | – |
| R23 | VWAP reversion | -0.013% | 23.8% | 1.73 | 10.8% | – |
| R26 | Bollinger Squeeze | -0.010% | 47.1% | 0.82 | 12.8% | t-stat -4.18 sig negative |
| R22 | Stop-hunt asymmetric | -0.016% | 28.9% | 1.22 | 10.4% | TP rarely hit |
| R27 | Ensemble (R21+R24) | -0.029% | 36.4% | 1.13 | 9.8% | t-stat -6.12 sig negative |

**Pattern**: avg_gross ceiling at +0.010%/trade across all mechanism classes.

## 3. Strict Criterion Math

User's strict criteria:
- WR ≥ 50%, R:R ≥ 1.0
- Daily ≥ +0.2% (1× leverage)
- ≥ 2 trades/day
- Per-trade gross ≥ taker fee (0.10% RT)
- Bootstrap stability

**Required avg_gross/trade**:
- 0.10% (taker fee) + 0.10% (target net daily / 2 trades) = **+0.20%/trade**

**Achieved avg_gross/trade ceiling (28 rounds)**: +0.010%/trade

**Multiplicative gap**: **20×**

## 4. Statistical Conclusion

Multiple independent statistical tests over 28 rounds confirm:

| Test | Finding |
|------|---------|
| t-test on per-trade returns (R26/R27) | Significantly NEGATIVE (p ≈ 1.0 one-sided) |
| WF 5-fold positive count | 0/5 across all rounds tested |
| Bootstrap 3-day pos_rate | 2-13% (necessary ≥50%) |
| Train→test consistency (TOD R28) | 0/2 train positive hours stay positive in test |
| Sharpe ratio | -6.61 (R27 ensemble) |
| Max drawdown | -57.31% (R27 ensemble) |

**Cumulative evidence**: Strategy edge is **statistically significantly NEGATIVE or zero** across all tested mechanisms. The "best" mechanism (R21) has avg_gross +0.01%/trade — barely above zero, and structurally cannot reach +0.20%/trade required.

## 5. Why Real Traders Make Money (User's Argument Addressed)

User's valid point: real traders / banks make money. So edge exists.

**True at professional level. Different access**:

| 자원 | Professional | Retail (this study) |
|------|--------------|---------------------|
| Order book depth | ✓ Full L2 | ✗ Not available |
| Cross-venue arbitrage | ✓ Multi-exchange | ✗ Single (BingX) |
| News/sentiment APIs | ✓ Bloomberg, etc. | ✗ Not available |
| Prime brokerage rates | ✓ ~0.001% RT | ✗ 0.05% taker |
| Information latency | ✓ ms-level | ✗ 5min polling minimum |
| Capital | ✓ $millions+ | ✗ Limited |
| Algos/HFT infra | ✓ Co-location | ✗ Cloud/home |
| Microstructure data | ✓ Trade tape, OI | ✗ Aggregated only |

**Retail trader with OHLCV + funding only**: structurally bound by data limits. Traditional candle-pattern + indicator scalping cannot extract +0.2%/day edge per 28 rounds × 30 mechanisms.

**Profitable retail trading typically requires**:
- Lower frequency (swing/position trading, weeks-months hold)
- Different paradigm (pair trading, basis arb if multi-venue)
- Information edge (news, fundamentals, on-chain analysis)
- Lower cost structure (rebates, multiple exchanges)

## 6. 사용자 Decision Matrix

| 옵션 | 설명 | 28-round 기반 권고 |
|------|------|-------------------|
| **A** Accept finding | 28-round 완전 evidence | **수학적 정합** |
| **B** Maker rebate full infra | LIMIT-only execution | Math doesn't close gap (best avg_gross +0.01% even at 0% friction) |
| **C** New data axis | Order book / multi-exchange / on-chain | 데이터 확보 필요, 새 PDCA arc |
| **D** Lower frequency paradigm | Swing/position trading | 다른 framework 진입 |
| **E** Adjust criterion | WR≥45%, daily≥+0.05%, etc | 28 rounds 모두 fail at strict — 어디까지 완화? |
| **F** Paper trade R21 best | "혹시" 기대치 (50% fail prior) | 시간/자원 고려 |

**가장 evidence-based**: **A or D**.
**B/C/E**: framework/scope 변경.

## 7. Files (this arc)

```
claudedocs/
├── m3_round[1-28]_*.md             (round-specific pre-regs)
├── m3_round14_potential_assessment.md (Phase 1 methodology)
├── m3_round16_phase2_4h.md         (Phase 2 4h)
├── m3_round17_comprehensive_potential.md
├── m3_round19_alpha_native_regime.md (R19 false positive)
├── m3_round20_dynamic_scalping.md  (dynamic exit 시작)
├── m3_round21_structure_pattern.md (best individual)
├── m3_round22_stop_hunt.md         (microstructure edge attempt)
├── m3_round23_vwap_scalping.md
├── m3_round25_pullback_tp_sweep.md (param sensitivity)
├── m3_round27_ensemble_deep.md (deep critique)
└── phase3_alpha_paper_trade_plan.md (R19 candidate paper trade — superseded)

scripts/analysis/m3_round[2-28]_*.py
results/m3_r[1-28]_*.json (raw data — 모든 round commit됨)

docs/04-report/
├── m3_3x5_matrix_comparative_20260428.md   (R1 시점)
├── m3_11_mechanisms_cumulative_20260428.md (R5 시점)
├── m3_17_mechanisms_cumulative_20260428.md (R8 시점)
├── m3_paradigm_shift_data_limit_20260428.md (R12)
├── m3_final_arc_20260428.md (R11 + R10 + R11 addenda)
└── m3_28round_comprehensive_final_20260428.md (this — final)
```

## 8. Standing Instruction

28 rounds + 30 mechanisms + 15 critique angles + multiple statistical tests = **comprehensive evidence**. 

본 dataset/constraint에서 strict criterion 미달성은 **mathematical impossibility region** 진입. 추가 round 자동 진행은 fix-impulse pattern (memory file `lessons_fix_impulse_pattern_20260427` 명시).

**사용자 명시 redirect 후 다음 단계**:
- A 선택 → 본 memo 마무리
- C/D 선택 → 새 framework PDCA arc start (별도 capital/time investment)
- F 선택 → R21 paper trade (50% fail prior 인지)

**자동 R29 안 함**.
