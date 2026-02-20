Run comprehensive system diagnostic and anomaly detection.

## Diagnostic Steps

### 1. Process Health
```bash
# Check bot process
ps aux | grep pattern_5m_bot | grep -v grep

# Check system resources
cd bingx_rl_trading_bot && python scripts/utils/system_diagnostic.py
```

### 2. State Integrity
```bash
cd bingx_rl_trading_bot && python scripts/utils/verify_state.py
```
- Compare state vs metrics for consistency (trades count, PnL)
- Check for corruption markers (state < metrics = corruption)
- Verify .bak and .new files don't exist (indicates previous write failure)

### 3. Exchange Connectivity
- Check last successful API call timestamp in logs
- Count API errors in last hour
- Circuit breaker status (tripped count, current state)

### 4. Log Analysis
- Parse last 100 log lines for ERROR/WARNING/CRITICAL
- Check for recurring patterns (same error repeating)
- Identify any unhandled exceptions

### 5. Performance Anomaly Detection
Read metrics and flag:
- WR deviation > 10pp from expected (68%)
- PnL trajectory diverging from expected
- Trade frequency anomaly (< 1 or > 8 trades/day)
- Unexpected pattern distribution (one pattern dominating)

### 6. File System Check
- State file age (should be < 10 minutes if bot running)
- Log file rotation (not too large)
- Lock file status (should not exist if bot not running, should exist if running)
- OneDrive sync issues (check for .tmp or locked files)

## Output
Produce a diagnostic report with:
- Overall health: HEALTHY / DEGRADED / CRITICAL
- Issues found (sorted by severity)
- Recommended actions per issue
