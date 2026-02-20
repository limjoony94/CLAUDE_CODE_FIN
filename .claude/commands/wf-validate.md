Run Walk-Forward validation on the current or proposed strategy.

## Standard WF Protocol
- Method: Expanding Window (IS=[0..T], OOS=[T..T+1], T increases)
- NEVER use cross-validation / leave-one-out (produces false positives)
- Folds: 3 (720d data, ~240d per fold)
- Metrics per fold: OOS Trades, OOS WR, OOS PnL, OOS MDD

## Pass Criteria
- All 3 folds must have positive OOS PnL
- Average OOS WR > breakeven WR (= SL / (TP + SL))
- No single fold OOS MDD > 50%

## Current Baseline (v1.28.42 MAE/MFE 59 patterns)
| Fold | OOS Trades | OOS WR | OOS PnL | OOS MDD |
|------|-----------|--------|---------|---------|
| 1    | 156       | 69.2%  | +80.7%  | 36.9%   |
| 2    | 131       | 72.5%  | +112.1% | 39.6%   |
| 3    | 154       | 79.9%  | +127.7% | 37.0%   |
Total OOS PnL: +320.5% | Avg OOS WR: 73.9%

## Usage
1. Read the proposed strategy changes
2. Run WF validation with the scanner or custom script
3. Compare against baseline
4. Report: PASS/FAIL per fold + comparison table

Ask the user: What strategy changes should be validated?
