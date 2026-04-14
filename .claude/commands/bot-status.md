Check the current C1 Breakout v2 trading bot status:

1. Check if the bot process is running:
   ```powershell
   Get-WmiObject Win32_Process -Filter "Name='python.exe' AND CommandLine LIKE '%c1_breakout%'" | Select-Object ProcessId
   ```
2. Read state from `bingx_rl_trading_bot/results/c1_breakout_state.json`
3. Read last 50 lines from `bingx_rl_trading_bot/logs/c1_breakout.log`
4. Summarize:
   - Bot running status (process alive or not)
   - Current position (if any): direction, entry price, unrealized PnL
   - SL/Trail order status (exchange orders placed?)
   - Today's stats: trades, wins, losses, daily PnL
   - Overall stats: total trades, win rate, total PnL
   - Recent trade history (last 5 trades from log)
   - Any warnings or errors in recent logs
5. Compare actual performance against expected:
   - Expected WR: ~36.6%, R:R: ~3.36, daily: ~+1.5% (additive 3x), ~3.1 trades/day
6. Flag anomalies: consecutive losses >= 15, daily loss > 5%, no trades in 12h
