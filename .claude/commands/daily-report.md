Generate a daily performance report for the C1 Breakout v2 trading bot.

## Data Collection
1. Read `bingx_rl_trading_bot/results/c1_breakout_state.json` for current state
2. Read last 300 lines from `bingx_rl_trading_bot/logs/c1_breakout.log` for today's trades
3. Parse ENTRY/EXIT/PnL/ERROR/HOURLY lines from log

## Report Structure

### Daily Summary
- Date, trading hours active
- Trades today: count, wins, losses
- Daily PnL (%, additive)
- Daily win rate vs expected (~36.6%)
- R:R today vs expected (~3.36)

### Exit Type Distribution
- TRAIL_TP count (expected ~85%)
- SL count (expected ~15%)
- Emergency/Timeout (should be 0)

### Risk Assessment
- Current drawdown from peak
- Consecutive loss count (max expected: 13)
- Emergency SL triggers (should be 0)

### Health Indicators
- GREEN/YELLOW/RED status
- API errors or connectivity issues
- State file integrity
- SL/Trail exchange order sync status

### Recommendations
- Continue / monitor closely / investigate / pause
