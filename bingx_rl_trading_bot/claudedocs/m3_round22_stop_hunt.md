# M3-R22 — Stop-Hunt Liquidity Reversal Scalping (사전 등록)

> **Date**: 2026-04-28
> **Authority**: 사용자 명시 — "한계를 규정하지 마세요" + creative new technique
> **Origin**: 전문 trader / ICT-SMC framework microstructure edge. 21 rounds prior 결과 무관 — 본 spec은 진정한 untested mechanism.

---

## 1. Mechanism Concept (전문 scalper edge)

**Stop-Hunt** (a.k.a. Liquidity Sweep, Stop Run): 가격이 swing high/low를 일시적으로 breach하면서 그 위/아래 stop loss orders를 hunt(체결시킴) → 즉시 reversal. 시장 maker / large traders가 자주 활용하는 microstructure pattern.

**왜 edge가 있는가** (literature):
1. Stop orders가 swing 직접 위/아래 cluster
2. Smart money는 이 liquidity를 "수확"하기 위해 일시적 spike 만듦
3. Spike 직후 가격 정상 range로 reversion → 재빨리 잡으면 high R:R trade
4. False breakouts의 정량 정의 + tradable signal

**Detection** (objective, no discretion):
- Bar [i]가 swing_low_20을 (2 bars 전 기준) breach: lo[i] < swing_low_at_t_minus_2
- 단 close back above: cl[i] > swing_low_at_t_minus_2
- Lower wick 길이 ≥ 2× body (rejection candle)
- Volume[i] ≥ 1.5× SMA20 (real flow, not drift)
- Cross-asset divergence: ETH 5m return [i] not also significantly negative (ETH > -0.3%)

LONG: 위 모든 조건 → 다음 bar open 진입.
SHORT mirror: high breach + rejection + cross-asset divergence.

## 2. Entry/Exit (사용자 spec — 진입≠청산 logic 다름)

### Entry (Stop-Hunt detection)
- 5m bar [i]: breach + rejection + volume + cross-asset divergence
- 1h trend filter: NEUTRAL or aligned (close > 100-bar SMA(1h)에서는 LONG, 그 반대 SHORT)
- 4h: 같은 방향 (broader trend confirm)

### Exit (Structure-based asymmetric R:R)
- **SL**: lo[i] - 0.05% (spike의 wick 직전, tight structural)
- **TP1**: entry + (entry - SL) × 1.0 (1R) — half exit (or trail trigger)
- **TP2**: entry + (entry - SL) × 2.5 (2.5R) — final
- **Trail after TP1**: SL → entry (breakeven)
- **Emergency**: -0.8% hard
- **Timeout**: 12 bars (1 hour)

R:R structure:
- Risk: ~0.3-0.5% (tight wick distance)
- TP2: 0.75-1.25% potential
- Asymmetric design encourages R:R ≥ 2.5 (단 actual realized 검증 필요)

## 3. Friction Model

- LIMIT entry (waiting for next-bar trigger): maker 0.02%
- LIMIT TP exits: maker 0.02% (price comes to known level)
- MARKET SL: taker 0.05%
- Per-trade RT: TP path 0.04%, SL path 0.07%

## 4. Pre-registered Test Suite (사용자 spec, 7 tests)

| # | Test | Threshold |
|---|------|-----------|
| 1 | Look-ahead audit | 0 leaks across 20 random signals |
| 2 | Overfitting probe | sensitivity ±20% same sign, WF 5-fold ≥3/5 positive, 3-way test positive |
| 3a | Friction taker (0.10% RT all) | daily ≥ 0.2% |
| 3b | Friction mixed (TP=0.04, SL=0.07) | daily ≥ 0.2% |
| 3c | Friction maker (0.04% RT all) | daily ≥ 0.3% |
| 4 | Bootstrap 1000 × 3-day windows | mean>0, pos_rate≥50%, p5>-1%, p_vs_BH≥60% |
| 5 | Avg gross/trade ≥ 0.10% | per-trade > taker fee |
| 6 | Frequency ≥ 2/day | sample sufficient |
| 7 | WR ≥ 50%, R:R ≥ 1.0 | core profitability |

**ALL 7 PASS** = production candidate → Phase 3 paper trade.

## 5. Predictions (정직)

| Test | Pred | Rationale |
|------|------|-----------|
| 1 | PASS | Backward-looking |
| 2 | borderline | LOW |
| 3a (taker) | borderline | mult gap continues, but stop-hunt has higher inherent R:R |
| 3b (mixed) | borderline-PASS | Maker TP 가능 |
| 3c (maker) | borderline-PASS | Best case |
| 4 (bootstrap) | borderline | Small but consistent edge가 있어야 통과 |
| 5 (gross > 0.10%) | borderline | Stop-hunt R:R 2.5 design이 평균 끌어올림 |
| 6 (freq ≥ 2) | borderline | Stop-hunt rare event, 1-2/day 예상 |
| 7 (WR ≥ 50%) | borderline-FAIL | Stop-hunt false signal 확률 (실제 breakout 시 SL hit), 단 R:R 2.5는 BE_WR=29%이므로 50% 도달 가능성 있음 |

**Most likely outcome**: 2-3 tests fail (특히 freq, taker friction). All pass: ~10-15% chance.
**Most likely surprise**: Asymmetric R:R + structural tight SL이 충분 → strict 통과.

## 6. R20/R21 vs R22 (controlled comparison)

같은 5m dataset 3-spec 비교:
- R20: confluence breakout + ATR exit
- R21: pattern reversal at extreme + structural exit
- R22: stop-hunt liquidity reversal + asymmetric R:R structural exit

이 3개 직접 비교가 "BTC 5m에서 어떤 mechanism class가 가장 가까이 가는가" 정량 measurement.

## 7. Anti-fix-impulse

- 본 R22 결과 후 parameter 변경 안 함
- ≥ 3 tests fail → drop, R23은 별도 mechanism
- ALL pass → 깊은 verify (10-seed, expanded WF) 후 paper trade
