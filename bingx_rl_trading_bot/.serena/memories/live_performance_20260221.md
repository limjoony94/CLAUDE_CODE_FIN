# Live Performance Analysis — 2026-02-21

## 90-Day OOS Test Progress (24% complete)
- Elapsed: 22d / 90d, Trade rate: 1.9/day, Projected: 168 trades
- **WR: 63.4%** vs expected 68% (YELLOW: -4.6pp)
- **PnL: +13.43%** (additive), Edge/trade: 0.04% vs expected 0.74%
- AvgWin: 3.51% (62% of expected 5.63%), AvgLoss: 5.97% (62% of expected 9.64%)
- **R:R preserved**: 0.588 actual vs 0.584 expected — ATR scaling compresses proportionally
- Consecutive losses: 0, Max single loss: 9.55%

## ATR Scaling Live Status
- **Enabled and working correctly** in production config
- Diagnostic (49.5h): +54% PnL vs baseline, 0 SL hits vs 2 baseline
- Mean vol_mult: 1.199 (range 0.642-1.7)
- No errors. No debug logging (expected — computed internally)
- Current position vol_mult=1.0 is transient (per-trade, not persisted)

## Multi-Position Study Conclusion
- **N=1 optimal. Multi-position NOT recommended.**
- N=1: PnL/MDD 27.59x (best), N=5: 23.24x (-16%), N=50: 10.50x (-62%)
- Capital dilution > diversification benefit. Edge erosion structural.
- WF 3/3 PASS for all N, but ratio collapse is definitive.

## IS=135d Re-scan Status
- Scanner lacks `--is-days` parameter — not yet implemented
- Edge Decay Study shows IS=135d → 2.86x better OOS PnL than 270d
- Recommendation: Implement scanner enhancement, but wait for 90-day test completion (April 30) before deploying
