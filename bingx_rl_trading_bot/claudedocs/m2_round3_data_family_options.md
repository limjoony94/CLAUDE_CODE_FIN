# M2 Round 3 — Data Family Options (사용자 선택 대기)

> **Date**: 2026-04-28
> **Reframe** (advisor): 사용자 메시지 "단순 파라미터 조합 한계"는 **paradigm 시사** = 새 data family.
> **Cap**: 1 family × 3 signal classes = 3 cells. 또는 2 families × 3 = 6 cells. **Hard ceiling 6**.
> **NOT this round**: cross-asset 10-asset sweep (sprawl), regime conditional on noise, ensemble of weak signals.

## Why this reframe matters

Round 1 (4) + Round 2 (12) + R0 (M1-A 1) = **20 cells uniformly negative on BTC candle OHLCV**.
이는 단순 더 많은 variants 추가로 해결 안 됨. **새 information source** 필요.

각 후보는 다른 data family이며 candle OHLCV가 잡지 못하는 정보를 포함.

---

## Candidate A — Funding rate divergence

**Hypothesis**: 누가 시장에 어느 방향으로 over-positioned인지 (longs crowded vs shorts crowded) → 반대 방향 mean-rev edge.

**Data availability**: ⚠️ `data/bingx_funding_rates.json` 데이터 2026-03~04 (~50일). 720d full range 미달 → **추가 fetch 필요** (BingX API funding rate history, 8h 간격 = ~2160 records over 720d).

**3 signal classes**:
- A.1 Extreme funding fade — funding > +0.05% (longs crowded) → SHORT bias entry on RSI overbought 15m
- A.2 Funding cross-zero — funding crosses from positive to negative → continuation in trend direction
- A.3 Sustained extreme — 8 consecutive funding periods (~64h) at extreme → reversal entry

**Effort**: API fetch (~10분) + signal coding + screening. 1세션 가능.

---

## Candidate B — Volume / Volume delta

**Hypothesis**: Price move가 high-volume일 때 정보 함량 ↑. Volume divergence는 weak move 신호.

**Data availability**: ✅ candle data already has volume column. 즉시 사용 가능. **추가 fetch 불필요**.

**3 signal classes**:
- B.1 Volume spike at level break — close > 24-bar high AND volume > 2× SMA(20) of volume
- B.2 Volume divergence — price makes higher-high but volume makes lower-high (weakening trend signal → fade)
- B.3 Volume-weighted price reaction — VWAP touch + bounce direction matches trend filter

**Effort**: 즉시. 데이터 fetch 없음.

---

## Candidate C — Cross-asset (BTC-ETH spread)

**Hypothesis**: BTC-ETH는 강하게 correlated. spread가 std 이탈 시 mean-rev. 또는 lead-lag relationship.

**Data availability**: ✅ `data/eth_binance_5m.csv` (105K bars, ~365 days from 2025-04-06). 720d 미달이지만 365일이면 충분. BTC와 timestamp align 검증 필요. **추가 fetch 불필요**.

**3 signal classes**:
- C.1 BTC-ETH spread mean-rev — log(BTC/ETH) ratio z-score > 2σ → fade (BTC가 ETH 대비 too high → BTC SHORT)
- C.2 Correlation breakdown — rolling 50-bar correlation < 0.5 (보통 ~0.85) → directional signal
- C.3 ETH-leads-BTC — ETH 변동률 vs BTC 변동률 lag relationship 측정 (ETH 15m 변동 → BTC 다음 15m 변동 predictor)

**Effort**: BTC-ETH timestamp align + signal coding + screening. 데이터 알라인이 첫 hurdle.

---

## Comparison

| Family | Data ready? | Effort | Information uniqueness | Risk |
|--------|-------------|--------|------------------------|------|
| A Funding rate | ❌ fetch 필요 | MED | Position imbalance (다른 dim) | API 의존 + 데이터 history limited |
| **B Volume** | ✅ 즉시 | LOW | OHLCV 내포 but underused | 가장 안전한 첫 시도 |
| **C Cross-asset** | ✅ 즉시 | MED | Cross-instrument correlation | timestamp align 첫 hurdle |

## Recommendation framework (assistant 권고 X — user picks)

advisor 권고대로 **assistant 자체 선택 금지**. 단 사용자 결정 도움 위해:

- **3 cells (1 family) 원칙**: 첫 시도는 가장 informative single family. B (volume)이 effort 낮고 OHLCV 내 정보 추출 잠재력.
- **6 cells (2 families) 원칙**: B + C 또는 B + A. 단 다른 data family 결합 시 시간 부담 ↑.

## Stop conditions (각 family 적용)

- 3 cells PASS = 0: 보고 + Round 4 (다른 family 또는 paradigm shift) 사용자 결정
- 1-2 PASS: 보고 + 사용자 picking (assistant 자체 picking 금지)
- 3 PASS: threshold 의심, strict re-run

## Convergent evidence memo (Round 3도 negative 시 deliverable)

advisor 권고: Round 3 결과 후 별도 memo 작성:
- **C1 + M1-A + R1 (4 variants) + R2 (12 cells) + R3 (3-6 cells) 종합 negative evidence**
- 사용자 옵션: paradigm shift class / portfolio approach / pause / 다른 asset
- "심층"은 negative result 진지하게 받아들이는 것 포함. 27번째 시도에서 발견 X.

## 사용자 결정 요청

다음 중 선택:
1. **Family B (volume) only** — 3 cells, 가장 빠름
2. **Family B + C** — 6 cells, OHLCV+cross-asset
3. **Family B + A** — 6 cells, OHLCV+funding (A는 fetch 시간 +10분)
4. **All three (A + B + C)** — 9 cells. ⚠️ advisor cap 6 초과. 강하게 권고하지 않음.
5. **Stop** — 4 rounds × negative 종합 + paradigm shift/pause 결정

선택 후 진행. 추가 dimension 없음 (single round 단위).
