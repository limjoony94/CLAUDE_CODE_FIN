# 32-Mechanism Sweep Retry Master Plan

**Trigger**: User critique (2026-05-01) — single-config falsification 부당, parameter sweep으로 mechanism potential 측정 의무.

**Goal**: 누적 32 surface-tested mechanism 전부 sweep retry. Framework: `mechanism_sweep_standard.py`.

**Anti-fishing**: Per-mechanism pre-reg + 50/25/25 split + IS sweep → top-5 → val → fresh OOS.

---

## Phase 1 — OHLCV single-asset (framework reuse 즉시)

| # | Mechanism | Data | Surface result | Sweep value |
|---|-----------|------|----------------|-------------|
| 1 | **M3 R21 Pattern reversal** | BTC 5m | +0.010%/trade gross | Edge양수, sweep으로 frequency↑ 가능성 |
| 2 | **M3 R9b Donchian** | BTC 5m | best M3 family | 사용자 critique 가장 잘 적용 |
| 3 | **C1 Breakout v2** | BTC 15m | BT +169.5%, LIVE -12.86% | BT 자체 sweep → 더 좋은 config? |
| 4 | **R8 1h Donchian** | BTC 1h | taker +0.04% gross | R25 base, sweep으로 frequency 증가? |
| 5 | **M3 R37 Compression breakout** | BTC 5m | 6th OOS NEG | sweep으로 envelope confirm |
| 6 | **M3 R39 Daily ORB** | BTC 5m+1d | OOS FAIL | sweep으로 envelope confirm |
| 7 | **M3 R40 Volume absorption** | BTC 5m | n=2,760 +0.034% | edge 양수, sweep으로 ↑? |
| 8 | **M3 R41 MACD minimal** | BTC 5m | +0.034%/trade | arithmetic falsified, sweep retry |
| 9 | **M3 R36 A pullback** | BTC 5m | retracted (FP) | sweep으로 false positive 확정 |
| 10 | **R42b Ehlers cycle** | BTC 1h | ✅ DONE 2026-05-01 | 0/144 IS PASS, edge×freq trade-off |

**Phase 1 expected**: ~30-40% sweep으로 envelope 한계 confirm, ~5-10% borderline 발견 가능성.

---

## Phase 2 — Special substrate (framework 확장 필요)

| # | Mechanism | Data | Surface result | Sweep retry note |
|---|-----------|------|----------------|------------------|
| 11 | **Path B R1 XS momentum 30d** | 8-coin daily | +0.13%/wk net (first edge>friction!) | **가장 promising**, cross-asset framework 1회 확장 |
| 12 | **Path B R2 XS reversal 7d** | 8-coin daily | vacuous (4.64% disp <5%) | sweep으로 dispersion threshold 변경 |
| 13 | **N1 Funding skim** | 8-coin funding | 0/7 sweep FAIL | wider grid retry |
| 14 | **N2 Triangular arb** | BTC/ETH | data limited | feasibility re-check |
| 15 | **N7 Cointegration** | 28 pairs | 0/28 PASS Engle-Granger | rolling window sweep |
| 16 | **N8 Macro regime** | BTC + DXY/SPY/GLD | +41%/720d, W4 -29.53% borderline | threshold + lookback sweep |
| 17 | **C2 Funding z-score spread** | 8-coin funding | 144 sweep already done | 이미 sweep done — Phase 2 skip |
| 18 | **R26 Grid trading** | BTC 5m | BingX -46.46%/100d, Binance ruin | LIVE-realistic param sweep |
| 19 | **R5 Cash-and-carry** | BTC perp + spot | 3.28%/yr deployable | leverage sweep already done — skip |
| 20 | **R13 Multi-coin carry** | 8-coin | 6/9 PASS, 0/3 HARD | weighting sweep |

---

## Phase 3 — Lower priority / data-limited

| # | Mechanism | Note |
|---|-----------|------|
| 21 | **R24 ICT liquidity sweep** | -0.05%/trade negative — sweep으로 변하지 않을 prior |
| 22 | **R25 Maker entry** | adverse selection, structural — sweep으로 변하지 않음 |
| 23 | **DeFi-R1 L2 yield** | $26/yr, friction-bound — capital scale 문제 |
| 24 | **L2 microstructure** | 4 features 0/4 falsified — friction floor |
| 25-32 | **Remaining M3 rounds** | R1-R8, R10-R20, R22-R28, R30-R35 등 — surface-only |

---

## Execution order

1. **Now**: Phase 1 #1 M3 R21 sweep
2. Phase 1 진행 (각 mechanism ~30분-1시간)
3. Phase 1 끝나면 누적 결과 표 + 사용자 surface
4. Phase 2 (framework 확장 1회 후 batch)
5. Phase 3 (낮은 priority, 빠른 batch)

---

## Cumulative tracking

각 mechanism sweep 완료 시 아래 표 업데이트:

| # | Mechanism | IS PASS | VAL PASS | OOS PASS | Best daily | Best avg_gross | Note |
|---|-----------|---------|----------|----------|------------|----------------|------|
| 10 | R42b Ehlers cycle (1h) | 0/144 | 0/5 | - | +0.060% | +1.131% | edge×freq trade-off, F1 PASS but F6 FAIL |
| 1 | R21b Pattern reversal (5m) | 0/144 | 0/5 | - | -0.055% | +0.018% | edge < friction floor (avg_gross < 0.07%) |
| 2 | R9b Donchian fixed exit (5m) | TBD | - | - | TBD | TBD | Phase 1 next |
| 3 | C1 Breakout v2 (15m) | TBD | - | - | TBD | TBD | Phase 1 |
| ... | ... | ... | ... | ... | ... | ... | ... |

**Pattern observed (sweep 2/32 done)**:
- R42b: high edge per trade (+1.13%), low frequency (n=21/360d)
- R21b: low edge (+0.018%), normal frequency (n=427/360d)
- 둘 다 envelope 한계: edge × frequency = constant, friction floor binding

이 패턴이 추가 sweep에서도 반복되면 "32 surface-tested + sweep-tested 모두 envelope 한계" 강한 evidence.

**Final synthesis**: 32 sweep-tested 결과 + envelope evidence 정량화 + deployable count.
