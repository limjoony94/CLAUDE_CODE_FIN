Safely deploy changes to the live C1 Breakout v2 trading bot.

## CRITICAL: This modifies the live trading system. Always ask for user confirmation.

## Pre-Deployment Checklist
1. [ ] Code changes tested and reviewed
2. [ ] WF validation passed (5/5 folds positive OOS PnL)
3. [ ] MC test passed (>=999 sims, p < 0.01)
4. [ ] Test suite passes
5. [ ] Current bot has no open position (check state)
6. [ ] Config changes match backtest parameters exactly

## Deployment Steps

### 1. Backup Current Config
```bash
cd bingx_rl_trading_bot
cp config/c1_breakout_config.yaml "config/c1_breakout_config_backup_$(date +%Y%m%d_%H%M%S).yaml"
```

### 2. Stop Bot
```powershell
Get-WmiObject Win32_Process -Filter "Name='python.exe' AND CommandLine LIKE '%c1_breakout%'" | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

### 3. Apply Changes (code or config)

### 4. Restart Bot
```powershell
Start-Process -FilePath 'python' -ArgumentList 'scripts/production/c1_breakout_bot.py' -WindowStyle Hidden -WorkingDirectory 'C:\Users\J\OneDrive\CLAUDE_CODE_FIN\bingx_rl_trading_bot'
```

### 5. Post-Deploy Verification
- Monitor first 3 trades after restart
- Verify log shows correct config loaded
- Check exchange SL/Trail orders placed correctly on first entry

## Rollback
```bash
cp config/c1_breakout_config_backup_TIMESTAMP.yaml config/c1_breakout_config.yaml
# Restart bot
```

## When NOT to Deploy
- Open position exists
- Test suite has failures
- WF validation not completed
