# M1 Scalping — Baseline Definition (Pre-Registered)

> **Date**: 2026-04-27
> **Phase**: 0.4 (사전 등록)
> **Purpose**: Bootstrap 3-day stability test (criterion 8)에서 candidate vs baseline relative 비교. 시장 변동성 vs strategy edge 분리.
> **Origin**: `lessons_distribution_check_20260427.md`, `research_protocol_3day_bootstrap.md`

## 1. Baseline 종류 (3개)

### Baseline A — No-Trade
- 정의: window 동안 거래 안 함, return = 0%
- 의미: 전략이 무거래보다 나은가?
- 측정: PnL_A = 0% (모든 window)

### Baseline B — Buy-and-Hold (BTC)
- 정의: window 시작 시 BTC LONG 진입, window 종료 시 청산
- 의미: 전략이 BTC 단순 보유보다 나은가? (시장 trend exposure 제거)
- 측정: PnL_B(window) = (close_end / close_start − 1) × 100%
- 단: leverage 1x, slippage entry+exit 0.05% × 2 + fee 0.10% = 0.20% friction inject

### Baseline C — Random Entry N=1
- 정의: 동일 5m 캔들 universe에서 trend filter (1h+4h)만 통과하는 시점에 균등분포로 entry, 동일 exit logic 적용 (TP_trail, SL, timeout)
- 의미: 전략 edge가 entry timing(RSI cross + body + 15m buffer)에서 오는가, exit logic에서 오는가?
- 측정: 각 window에서 random seed로 entry timing 추출, BT 실행, PnL 1x 누적
- 동일 N=1 constraint, 동일 friction 적용

## 2. Bootstrap 비교 룰 (criterion 8 CRITICAL)

각 random 3-day window에서:

| Metric | M1-A | A (no-trade) | B (B&H) | C (random) |
|--------|------|--------------|---------|------------|
| PnL 1x | x_M | 0 | x_B | x_C |

**Pass 조건** (1000 windows over):
1. **Core 3** (`lessons_distribution_check`):
   - Mean(x_M) > 0
   - Pos rate(x_M) ≥ 50%
   - P5(x_M) > −1.0% (catastrophic loss 회피)
2. **Relative**:
   - P(x_M > x_A) ≥ 60% (60% windows에서 무거래보다 나음)
   - P(x_M > x_B) ≥ 55% (55% windows에서 B&H보다 나음)
   - P(x_M > x_C) ≥ 60% (60% windows에서 random entry보다 나음 — entry edge 증명)

3개 baseline 중 **모두 통과 시** PASS, 하나라도 fail → strategy shelve 검토.

## 3. Random seed 관리

- Bootstrap: seed 0~999 (1000 windows)
- Random entry C: 각 window 내 seed 0~999 randomly select entry timing
- 모든 seed 사전 고정 (재현 가능성)

## 4. Output schema

```json
{
  "n_windows": 1000,
  "candidate": {"mean": ..., "pos_rate": ..., "p5": ...},
  "baseline_A": {"mean": 0, "pos_rate": 0, "p5": 0},
  "baseline_B": {"mean": ..., "pos_rate": ..., "p5": ...},
  "baseline_C": {"mean": ..., "pos_rate": ..., "p5": ...},
  "relative": {
    "p_cand_gt_A": ...,
    "p_cand_gt_B": ...,
    "p_cand_gt_C": ...
  },
  "verdict": "PASS|FAIL"
}
```

## 5. Sample size note

3-day window @ 720d data → 720 / 3 = 240 unique non-overlapping windows. 1000 random sampling은 일부 overlap 허용 (Sampling with replacement on start dates).

이 문서는 **사전 등록**. Phase 3 Bootstrap 실행 전 변경 금지. 변경 시 evolution log + advisor 검토.
