# Deprecated Bots - DO NOT USE

This folder contains batch files for bots that have been **deprecated** due to critical issues.

## RSI Martingale Bot (2025-12-25)

**Status**: ❌ **DEPLOYMENT PROHIBITED**

**Issue**: Research vs Production Discrepancy
- Research claimed: +1.37% daily return
- Production backtest: **-3.33% daily** (bankruptcy in 30 days)
- Root cause: RSI calculation method difference (Research: SMA, Production: EWM)

**Files Archived**:
- `START_RSI_MARTINGALE.bat`
- `STOP_RSI_MARTINGALE.bat`
- `MONITOR_RSI_MARTINGALE.bat`

**Documentation**: `claudedocs/RSI_MARTINGALE_DISCREPANCY_ANALYSIS_20251225.md`

---

**WARNING**: Do not move these files back or attempt to run these bots. They will result in significant losses.
