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

| # | Mechanism | Configs | IS PASS | Best daily | Best avg_gross | Note |
|---|-----------|---------|---------|------------|----------------|------|
| 10 | R42b Ehlers cycle (1h) | 144 | 0/144 | +0.060% | +1.131% | edge↑↑ × freq↓↓ (n=21), F6 FAIL |
| 1 | R21b Pattern reversal (5m+1h) | 144 | 0/144 | -0.055% | +0.018% | edge < friction floor |
| 4 | R8b 1h Donchian | 1296 | 0/1296 | +0.043% | +0.232% | edge>fric, daily<target |
| 8 | R41b MACD cross (1h) | 648 | 0/648 | +0.051% | +0.232% | 동일 envelope |
| 5 | R37b compression breakout (1h) | 864 | 0/864 | +0.067% | +0.468% | edge>fric, daily<target |
| 6 | R39b Daily ORB (1h) | 216 | 0/216 | +0.006% | +0.146% | low edge, low daily |
| 11 | **R1b XS momentum (10coin daily)** | 108 | 0/108 | +0.105% | +1.315% | borderline (long-only 60d/7d rebal) |
| 16 | **N8b Macro regime BTC vs DXY/SPY/GLD** | 108 | 0/108 | **+0.312%** ✅ | **+4.077%** ✅ | **F6 FAIL n=28<50** — sample size only |
| 9 | R36b EMA pullback (1h) | 192 | 0/192 | +0.080% | +0.218% | 동일 envelope |

| 12 | **R2b XS reversal (10coin daily)** | 72 | 0/72 | **+0.299%** ✅ | +0.532% ✅ | **distribution stability FAIL** (p5=-3.08%, freq, p_beats) |
| 7 | R40b volume absorption (1h) | 432 | 0/432 | +0.055% | +1.041% | edge↑×freq↓↓ (n=22) |

**Cumulative: 11 mechanisms × 4,224 configs = 0/4,224 IS PASS overall**

**Distribution stability discovery (2026-05-01)**:
R2b가 mean +0.299%/day PASS, n=262 ✅, avg_gross +0.532% ✅ 등 통과하나
**bootstrap distribution test에서 3개 fail**: p5_daily=-3.08% (tail risk),
sufficient_trades_per_window (0.73/day), p_beats_baseline 0.519<0.55.

→ 사용자 criteria의 "3-day random window stability" 조건이 R26 LIVE -12.86% 같은
  catastrophe 사전 차단 핵심. Mean만 보지 말고 distribution stability 보라는
  user criterion이 정확히 이런 case 잡는 evidence.

**Pattern observed (sweep 9/32 done)**:
- All 9 mechanisms 0 IS PASS at strict criteria
- **N8b**: F2 ✅ daily target 통과 (first!) but F6 FAIL (n=28 < 50)
- **R1b**: borderline (+0.105% daily, half target)
- 다수: edge >> friction (avg_gross 0.15-1.13%) but daily +0.04-0.10%
- "edge × frequency = constant" hypothesis 강한 evidence
- N8b가 envelope 한계 가장 흥미로운 case — high-edge low-freq pathway

**다음 priorities**:
- Phase 1 #5 R40 volume absorption sweep
- Phase 2 #12 Path B R2 XS reversal sweep
- Phase 2 #15 N7 cointegration sweep (rolling window)
- Phase 1 #3 C1 Breakout v2 — already extensive sweep done elsewhere

**Pattern observed (sweep 2/32 done)**:
- R42b: high edge per trade (+1.13%), low frequency (n=21/360d)
- R21b: low edge (+0.018%), normal frequency (n=427/360d)
- 둘 다 envelope 한계: edge × frequency = constant, friction floor binding

이 패턴이 추가 sweep에서도 반복되면 "32 surface-tested + sweep-tested 모두 envelope 한계" 강한 evidence.

**Final synthesis**: 32 sweep-tested 결과 + envelope evidence 정량화 + deployable count.
