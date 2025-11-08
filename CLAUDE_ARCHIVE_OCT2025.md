# CLAUDE.md Archive - October 2025

This file contains archived updates from October 2025 that were removed from CLAUDE.md to improve performance.

---

## October 31st Production Models

### Models Used on October 31st Successful Trades
**Date**: 2025-10-31
**Status**: ✅ **MODELS IDENTIFIED FROM LOG FILES**

**Models Used**: Enhanced 5-Fold CV (20251024_012445)
- LONG Entry: xgboost_long_entry_enhanced_20251024_012445.pkl (85 features)
- SHORT Entry: xgboost_short_entry_enhanced_20251024_012445.pkl (79 features)
- Exit Models: Opportunity Gating Improved (20251024_043527/044510)

**Successful Trades (Oct 31 - Nov 2)**:
- 3 trades reconciled from exchange
- Combined P&L: +$7.51
- Win Rate: 75% (3/4 wins)

---

## October Updates Archive

### Walk-Forward Decoupled Entry Models (Oct 27)
- Triple Integration: Filtered Simulation + Walk-Forward + Decoupled
- Performance: +38.04% per 5-day, 73.86% WR
- Not deployed (lower than Enhanced 5-Fold CV)

### Feature Reduction Analysis (Oct 28)
- Removed 22 zero-importance features
- LONG: 85 → 73 features, SHORT: 79 → 74 features
- Backtest: +34.02% but only 63 trades (insufficient sample)
- Rolled back due to statistical concerns

### Exit Threshold Optimizations (Oct 22-26)
- Multi-parameter grid search (64 combinations)
- Optimal: Exit 0.75, SL -3%, MaxHold 10h
- Currently in production

### Balance-Based Stop Loss (Oct 20-21)
- Changed from fixed price SL to balance-based
- Formula: price_sl_pct = balance_sl / (position_size × leverage)
- Result: Consistent -3% balance risk regardless of position size

### Trade Reconciliation System (Oct 19)
- Automatic detection of manual trades via exchange API
- P&L calculation from order history
- State file synchronization

---

## Historical Performance Metrics

### October Production Performance
- Session Start: 2025-10-17 17:14:57
- Total Trades: 15 (3 bot + 12 manual)
- Bot Win Rate: 66.7%
- Manual Trades Impact: -$26.41

### Backtest Validation (Oct 14-26)
- 108-window backtest (540 days)
- Entry 0.80/Exit 0.80: +25.21% per 5-day, 72.3% WR
- ML Exit Usage: 94.2%
- Max Drawdown: 1.34%

---

## Archived for reference - Last updated: 2025-11-03
