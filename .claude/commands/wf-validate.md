Run Walk-Forward validation on the C1 Breakout v2 strategy.

## Standard WF Protocol
- Method: Expanding Window (IS=[0..T], OOS=[T..T+1], T increases)
- Formula: ie = int(n * (fi + 1) / (n_folds + 1))
- NEVER use cross-validation / leave-one-out (produces false positives)
- Folds: 5
- PnL: Additive (no compound)
- Fee: 0.10% RT

## Pass Criteria
- All 5 folds must have positive OOS PnL
- No single fold OOS MDD > 15% (additive 1x)

## Current Baseline (v2.5, 333 days, additive 1x)
| Fold | OOS PnL |
|------|---------|
| 1-5  | Total +153.9% (ALL PASS) |

Overall: PnL +169.5%, MDD 5.4%, WR 36.6%, R:R 3.36

## Usage
1. Read the proposed strategy changes
2. Run WF validation script
3. Compare against baseline
4. Report: PASS/FAIL per fold + comparison table

Ask the user: What strategy changes should be validated?
