# Round 30 — Parameter Grid Search (Entry / SL / TP) with Train/Test Split

**Date pre-registered**: 2026-04-30
**Status**: PRE-COMMIT (locked before sweep code)
**Track**: User-specified 3-axis parameter sweep

---

## DISCLOSURE

User explicitly requested grid sweep across entry/SL/TP. To prevent post-hoc
winner selection (overfitting), procedure follows:

1. Split 720d into TRAIN (60% = first 432d) and TEST (40% = last 288d)
2. Run all 27 configs on TRAIN
3. Select best config by daily_pct on TRAIN
4. Apply ONLY THAT CONFIG to TEST
5. TEST daily_pct is the OOS estimate
6. Report ALL 27 train results + selected winner test result

This is standard ML procedure for hyperparameter selection with held-out data.
Multiple comparison is controlled by reserving test data.

---

## What's distinct from prior rounds

R29 = single locked config on full 720d. R30 = systematic 27-config sweep with
TRAIN/TEST split for honest multiple-comparison handling.

R30 baseline = R29 (15m fade, lookback 16 bars, fade direction). Variations on
entry/SL/TP only.

---

## Locked Grid

```python
GRID = {
    'entry_body_filter_pct': [30, 50, 70],          # 2-bar combined body % of range
    'sl_period_range_multiple': [0.5, 1.0, 1.5],
    'tp_period_range_multiple': [1.5, 2.5, 4.0],
}
# Total: 3 × 3 × 3 = 27 configs

LOCKED_FIXED = {
    'asset': 'BTC/USDT',
    'tf': '15m',
    'period_lookback_bars': 16,           # 4h
    'direction': 'fade',
    'max_hold_bars': 96,                  # 24h
    'taker_per_side_pct': 0.05,
    'maker_per_side_pct': 0.02,
    'capital_usd': 1500,
    'train_test_split': 0.60,             # 60% train, 40% test
}
```

Per config: same R29 logic but with that config's (body, SL, TP) values.
Friction: entry market taker, TP limit maker, SL market taker, timeout taker.

---

## Pre-Registered Procedure

### Step 1: TRAIN phase (first 432 days, 41,472 bars)
- Run all 27 configs
- For each: compute daily_pct, n_trades, WR, R:R
- Rank by train_daily_pct

### Step 2: WINNER selection on TRAIN
- Pick config with highest train_daily_pct
- Lock the winning (entry, SL, TP) values

### Step 3: TEST phase (last 288 days, 27,648 bars)
- Apply ONLY winning config on test data
- Compute test_daily_pct, test_bs_pos_rate, etc.
- This is the OOS estimate (no further selection)

### Step 4: Report
- Full 27-config train results table
- Winner config + test result
- Train vs test comparison (overfit check):
  - If test_daily ≥ 0.20% AND test_daily within 50% of train_daily: genuine
  - If test_daily ≪ train_daily: overfit confirmed
  - If test_daily ≤ 0%: clean overfit failure

---

## Pre-Registered Outcomes (Bonferroni-aware)

| Outcome | Probability (advisor + 33-round prior) |
|---------|----------------------------------------|
| Test daily ≥ 0.20% AND ≥ 0.5× train daily | **3-7%** (genuine signal) |
| Test daily 0.10-0.20% (sub-target but positive) | 10-15% |
| Test daily near 0 (overfit confirmed) | 50-60% |
| Test daily negative (catastrophic overfit) | 20-30% |

**Realistic prior**: train winner overfits in 60-70% of cases per literature
(Bailey et al. 2014, Harvey & Liu 2015). Multiple-comparison correction via
held-out test is standard but not perfect.

---

## Anti-Adjustment

GRID values LOCKED (no axis additions, no level changes). Train/test split 60/40
LOCKED. Winner selection by daily_pct LOCKED. **No post-hoc re-selection on
different metric** (e.g., re-pick by Sharpe if daily_pct winner fails).

If TEST winner fails: this is the empirical answer for the 27-config envelope.
Further specific axis variants (e.g., trying 80% body filter) require separate
pre-reg.

---

## Hash Anchor

Committed BEFORE grid sweep code.
