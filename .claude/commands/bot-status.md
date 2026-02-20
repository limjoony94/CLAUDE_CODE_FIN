Check the current pattern_5m trading bot status:

1. Check if the bot process is running (look for pattern_5m_bot.py process)
2. Read the latest state from `bingx_rl_trading_bot/results/pattern_5m_bot_state.json`
3. Read the latest metrics from `bingx_rl_trading_bot/results/pattern_5m_metrics.json`
4. Read the last 50 lines from the most recent log file in `bingx_rl_trading_bot/logs/`
5. Summarize:
   - Bot running status (process alive or not)
   - Current position (if any): pattern, direction, entry price, unrealized PnL
   - Today's stats: trades, wins, losses, daily PnL
   - Overall stats: total trades, win rate, total PnL, max drawdown
   - Recent trade history (last 5 trades)
   - Any warnings or errors in recent logs
6. Compare actual performance against expected (EXPECTED_WIN_RATE=68%, EXPECTED_EDGE=0.27%)
7. Flag any anomalies: consecutive losses >= 3, daily loss > 10%, WR < 60%
