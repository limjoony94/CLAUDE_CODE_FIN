Emergency C1 Breakout bot shutdown procedure. Use when critical anomaly detected.

## Pre-Shutdown Checks
1. Read current position from `bingx_rl_trading_bot/results/c1_breakout_state.json`
2. If open position exists: WARN — stopping bot will NOT close the position or cancel exchange orders
3. Check recent log: `tail -20 bingx_rl_trading_bot/logs/c1_breakout.log`

## Shutdown Steps
```powershell
# 1. Find and kill bot process
Get-WmiObject Win32_Process -Filter "Name='python.exe' AND CommandLine LIKE '%c1_breakout%'" | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }

# 2. Verify stopped
Get-WmiObject Win32_Process -Filter "Name='python.exe' AND CommandLine LIKE '%c1_breakout%'" | Select-Object ProcessId
```

## Post-Shutdown
1. If open position remains: user must manually close on BingX or use CCXT script
2. Check for orphan exchange orders (SL/Trail) that may still be active
3. Save state snapshot for incident log
4. Document: what triggered emergency, state at time of stop

## When to Auto-Suggest Emergency Stop
- MDD exceeds 15% (additive 1x)
- State corruption detected
- Exchange API errors repeating > 10 in 1 hour
- Bot process consuming >90% CPU or memory
