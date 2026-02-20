Emergency bot shutdown procedure. Use when critical anomaly detected.

## Pre-Shutdown Checks
1. Read current position from `bingx_rl_trading_bot/results/pattern_5m_bot_state.json`
2. If open position exists: WARN the user — closing the bot will NOT close the position
3. Check recent log for errors: `tail -20` latest log in `bingx_rl_trading_bot/logs/`

## Shutdown Steps
```bash
# 1. Graceful stop (preferred)
cd bingx_rl_trading_bot && python scripts/utils/stop_bot.py

# 2. If graceful fails, force kill
cd bingx_rl_trading_bot && python scripts/utils/force_kill_bot.py

# 3. Verify stopped
ps aux | grep pattern_5m_bot | grep -v grep
```

## Post-Shutdown
1. Save current metrics snapshot for incident log
2. If open position remains: ask user whether to close it
   ```bash
   cd bingx_rl_trading_bot && python scripts/utils/close_position.py
   ```
3. Verify state file integrity: `python scripts/utils/verify_state.py`
4. Document incident: what triggered the emergency, metrics at time of stop

## When to Auto-Suggest Emergency Stop
- Daily loss exceeds 13% (hard limit breached)
- 5+ consecutive losses in a single day
- State corruption detected (state < metrics discrepancy)
- Exchange API circuit breaker tripped 3+ times in 1 hour
- Bot process consuming >90% CPU or memory
