Run comprehensive C1 Breakout v2 system diagnostic and anomaly detection.

## Diagnostic Steps

### 1. Process Health
```powershell
Get-WmiObject Win32_Process -Filter "Name='python.exe' AND CommandLine LIKE '%c1_breakout%'" | Select-Object ProcessId, WorkingSetSize, CreationDate
```

### 2. State Integrity
- Read `bingx_rl_trading_bot/results/c1_breakout_state.json`
- Verify JSON validity
- Check for .tmp or .bak files (indicates previous write failure)
- Verify positions array consistency

### 3. Exchange Connectivity
- Check last successful API call timestamp in log
- Count ERROR lines in last hour
- Verify SL/Trail exchange order IDs match state

### 4. Log Analysis
- Parse last 100 lines from `bingx_rl_trading_bot/logs/c1_breakout.log`
- Filter: ERROR, WARNING, CRITICAL, GHOST, ORPHAN
- Check for recurring patterns (same error repeating)

### 5. Performance Anomaly Detection
Flag:
- WR deviation > 15pp from expected (36.6%)
- Trade frequency anomaly (< 1 or > 8 trades/day)
- Consecutive losses > 15
- Emergency SL triggers (should be 0)

### 6. File System Check
- State file age (should be < 15 minutes if bot running)
- Log file size and rotation
- OneDrive sync issues (check for locked files)

## Output
Produce diagnostic report: HEALTHY / DEGRADED / CRITICAL with issues and actions.
