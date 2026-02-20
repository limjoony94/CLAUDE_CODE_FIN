Generate a daily performance report for the Pattern 5m trading bot.

## Data Collection
1. Read `bingx_rl_trading_bot/results/pattern_5m_metrics.json` for overall stats
2. Read `bingx_rl_trading_bot/results/pattern_5m_bot_state.json` for current state
3. Read last 200 lines from latest log in `bingx_rl_trading_bot/logs/` for today's trades
4. Run: `cd bingx_rl_trading_bot && python scripts/monitor/daily_report.py` if available

## Report Structure

### Daily Summary
- Date, trading hours active
- Trades today: count, wins, losses
- Daily PnL (%, absolute)
- Daily win rate vs expected (68%)

### Pattern Performance
- Which patterns triggered today
- Per-pattern WR for today
- Any patterns with unexpected results (WR < 50% or > 95%)

### Risk Assessment
- Current drawdown from peak
- Daily loss vs limit (13%)
- Consecutive loss count
- ATR scaling factor (current volatility regime)

### Trend Analysis
- 7-day rolling WR
- 7-day rolling PnL
- Comparison vs first 30 days of OOS test
- On-track vs expected trajectory?

### Health Indicators
- GREEN/YELLOW/RED status per alert threshold
- API errors or circuit breaker activations
- State file integrity

### Recommendations
- Continue / monitor closely / investigate / pause
- Any patterns to watch
- Upcoming events that may affect performance
