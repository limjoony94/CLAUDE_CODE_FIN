# M3-R16 — Phase 2: 4h Optimal Parameter Search (사전 등록)

> **Date**: 2026-04-28
> **Authority**: 사용자 3-phase methodology Phase 2
> **Origin**: R15 distribution-level evidence — 4h timeframe phase-2 eligible (p_both_pos=80%, corr_tt=+0.22, potential 92.6).

## 1. R15 결과 요약 (Phase 1 완료)

4h α-style ETH-lag (no ATR filter), 15 valid configs:
- 12/15 (80%) cross-period positive (train AND test daily > 0)
- Top cluster: **eth_thresh=0.60, btc_lag=0.50/0.80, N_exit=2** (4-bar = 8h hold)
- Best test daily: +0.0648% (eth=0.60, btc=0.50, N=2)
- Median test daily: +0.014%

## 2. Phase 2 Plan

R15 train/test split (60/40)이 이미 사용되어 contamination risk. **WF 5-fold expanding window**로 fresh validation.

### Refined parameter grid (R15 best cluster 중심)
| Param | Grid values | n |
|-------|-------------|---|
| eth_thresh | 0.40, 0.50, 0.60, 0.70, 0.80, 1.00 | 6 |
| btc_lag_thresh | 0.30, 0.40, 0.50, 0.60, 0.80, 1.00 | 6 |
| N_exit | 1, 2, 3, 4 | 4 |
| **Total combos** | | **144** |

Friction: 0.08% RT (single perp leg, maker-tier).

### WF 5-fold expanding
- Total 4h bars: 4321 (720 days × 6 per day)
- Each fold: train uses 1/6 to 5/6 expanding, test on next 1/6
- Per config: count of folds with daily_net > 0

### Pre-registered selection criteria (결과 보기 전)
**Robust optimum** = config with:
1. **WF 5-fold positive ≥ 4/5 folds** (high consistency)
2. **WF mean daily_net > 0** across all 5 folds
3. **min n_test (per fold) ≥ 5** (each fold has trades)
4. **No fold daily < -0.1%** (no catastrophic fold)

### Tie-breaking
조건 모두 만족 configs 중:
- 가장 높은 `WF mean daily_net` 선정

### Bonferroni-style multi-comparison check
144 combos × 5 folds = 720 fold-tests. p=0.05 raw → ~36 false positives expected.
**Strict gate**: 4+/5 folds = (5 choose 4) × p^4 = 5 × 0.0625 = 0.3125. With p=0.5 (random walk to either sign), 4/5 prob = 0.1875 (~18.75%). At 144 trials, expected ~27 by chance.

→ **stricter**: require **5/5 folds positive** for primary candidate. Expected by chance: 144 × 0.5^5 = 4.5. So ≥10 such configs = strong signal vs 1.5 = noise.

## 3. Predictions

| 결과 | Predicted | Confidence |
|------|-----------|-----------|
| 5/5 fold robust configs ≥ 10 | borderline | LOW-MED. R15 cluster suggests yes, but small samples per fold. |
| Best config WF mean > +0.02%/day | borderline | LOW. R15 max +0.065 was full test, WF folds will be smaller |
| 4+/5 fold robust configs ≥ 30 | likely YES | MED |

## 4. Phase 3 (next) gate

Phase 2 PASS = ≥ 1 config with 5/5 folds positive AND mean > +0.02%/day.
- → 별도 Phase 3 OOS sanity test 가능 (WF 사용했으므로 OOS 잘 검증됨)
- → 실제 거래 적용 가능 ramp-up: paper trade 2 weeks → 0.5× position 1 month → full

Phase 2 FAIL (0 configs 5/5 robust):
- → R10 패턴 재현 (selection-from-grid noise)
- → 사용자 결정 (data acquisition / paradigm change)

## 5. Anti-fix-impulse

- 144 configs × WF 한 번 실행. 결과 후 grid 확장 안 함
- 5/5 robust 조건 결과 보기 전 정의됨
- 0 configs pass 5/5 → drop 4h family. R15 was sample noise.
- 사용자 mandate "수익성 모델 찾을 때까지" but 본 R16 fail 시 다른 timeframe 또는 paradigm 시도 (1d small sample, 또는 hybrid 5m/15m/1h)
