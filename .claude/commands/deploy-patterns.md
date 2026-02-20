Safely deploy new patterns to the live trading bot.

## CRITICAL: This modifies the live trading system. Always ask for user confirmation.

## Pre-Deployment Checklist
1. [ ] New patterns file exists and is valid JSON
2. [ ] WF validation passed (3/3 folds positive OOS PnL)
3. [ ] MC test passed (3-seed, p < 0.01)
4. [ ] All quality filters applied (E>=21.8pp, WR>=60%, SL>=1.0%, min_trades>=25)
5. [ ] Test suite passes (1139+ tests)
6. [ ] Current bot has no open position (check state)

## Deployment Steps

### 1. Backup Current Patterns
```bash
cd bingx_rl_trading_bot
cp results/dynamic_patterns.json "results/dynamic_patterns_backup_$(date +%Y%m%d_%H%M%S).json"
```

### 2. Validate New Patterns
```bash
# Check JSON validity
python -c "import json; d=json.load(open('results/dynamic_patterns_NEW.json')); print(f'{len(d[\"patterns\"])} patterns loaded')"
```

### 3. Stop Bot (if running)
```bash
python scripts/utils/stop_bot.py
# Wait for confirmation of stop
ps aux | grep pattern_5m_bot | grep -v grep
```

### 4. Deploy
```bash
cp results/dynamic_patterns_NEW.json results/dynamic_patterns.json
```

### 5. Run Tests
```bash
python -m pytest scripts/tests/ -v --tb=short 2>&1 | tail -30
```

### 6. Restart Bot
```bash
# User must restart manually or via tmux
```

### 7. Post-Deploy Verification
- Monitor first 5 trades after deployment
- Verify correct patterns loaded (check bot startup log)
- Verify TP/SL values match expected

## Rollback
```bash
cp results/dynamic_patterns_backup_TIMESTAMP.json results/dynamic_patterns.json
# Restart bot
```

## When NOT to Deploy
- Market is in extreme volatility (ATR ratio > 2.0)
- Open position exists
- Test suite has failures
- WF validation not completed
