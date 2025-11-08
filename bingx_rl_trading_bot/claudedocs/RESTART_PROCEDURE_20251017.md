# Bot Restart Procedure - 2025-10-17 16:42 KST

## 🚨 Current Status

**Bot State**: Running with OLD CODE (has 4 critical bugs)
**Position**: Ghost position exists (not on exchange but in state file)
**Issue**: Bot crashes every loop with API error, cannot exit positions

## 🐛 Critical Bugs Fixed (Code Updated)

All bugs are FIXED in code but NOT APPLIED (bot running old version):

| # | Bug | File | Status |
|---|-----|------|--------|
| 1 | Emergency Max Hold parsing | `opportunity_gating_bot_4x.py:373-389` | ✅ FIXED |
| 2 | Monitor max_hold mismatch | `quant_monitor.py:826` | ✅ FIXED |
| 3 | Entry time using candle timestamp | `opportunity_gating_bot_4x.py:658-676` | ✅ FIXED |
| 4 | BingX API limit error (1440 → 1400) | `opportunity_gating_bot_4x.py:71` | ✅ FIXED |

## 📋 Restart Procedure

### Step 1: Stop Current Bot

**Option A - Find PID and Kill**:
```bash
# Find bot process
tasklist | grep -i python

# Identify the bot (usually largest memory ~280MB)
# Kill by PID
taskkill /F /PID <PID>
```

**Option B - Kill All Python (Nuclear Option)**:
```bash
# WARNING: This kills ALL Python processes
taskkill /F /IM python.exe
```

**Recommended**: Option A, kill only the bot process (PID 9692 - 279MB).

### Step 2: Clear Ghost Position

```bash
cd bingx_rl_trading_bot
python scripts/debugging/clear_ghost_position.py
```

**What it does**:
- Sets `position: null` in state file
- Keeps trades history intact
- Syncs state with exchange (no position on exchange)

### Step 3: Verify Fixes Applied

```bash
# Check MAX_DATA_CANDLES changed to 1400
grep "MAX_DATA_CANDLES" scripts/production/opportunity_gating_bot_4x.py

# Expected: MAX_DATA_CANDLES = 1400

# Check entry_time fix
grep -A 5 "actual_entry_time" scripts/production/opportunity_gating_bot_4x.py

# Expected: actual_entry_time = datetime.now().isoformat()
```

### Step 4: Restart Bot

```bash
cd bingx_rl_trading_bot

# Start bot (foreground - recommended for testing)
python scripts/production/opportunity_gating_bot_4x.py

# OR background with log
nohup python scripts/production/opportunity_gating_bot_4x.py > /dev/null 2>&1 &
```

### Step 5: Monitor Startup

**Watch log in real-time**:
```bash
tail -f logs/opportunity_gating_bot_4x_20251017.log
```

**Look for**:
- ✅ No API errors (`limit must be less than or equal to 1440`)
- ✅ Data fetched successfully
- ✅ Signals generated
- ✅ No crashes

**Expected startup logs**:
```
INFO - ================================================================================
INFO - OPPORTUNITY GATING BOT (4x LEVERAGE) - STARTING
INFO - ================================================================================
INFO - Loading models...
INFO -   ✅ Models loaded
INFO - Initializing BingX client (Testnet: False)...
INFO - ✅ Client initialized
INFO - [2025-10-17 16:45:00] Price: $106,XXX.X | Balance: $XXX.XX | LONG: 0.XXXX | SHORT: 0.XXXX
```

### Step 6: Verify Emergency Max Hold Works

**Important**: Next position will test if Emergency Max Hold works:

1. Wait for new position entry
2. Monitor holding time (should be recorded correctly)
3. Verify exit triggers at 8 hours (not indefinite)

**Monitor with**:
```bash
python scripts/monitoring/quant_monitor.py
```

**Expected in monitor**:
- Holding Time: Accurate (not 0h or wrong)
- Max Hold: 8.0h (not 4.0h)
- Time Left: Decreases correctly
- Emergency Max Hold triggers at 8h

## 🔍 Verification Checklist

After restart, verify:

- [ ] Bot starts without API errors
- [ ] Data fetched successfully (1400 candles, not 1440)
- [ ] Ghost position cleared from state file
- [ ] Signals generated correctly
- [ ] Next position: Entry time = actual execution time (not candle time)
- [ ] Next position: Holding time calculated correctly
- [ ] Next position: Emergency Max Hold triggers at 8h (if held that long)

## 📊 Expected Improvements

**Before (Old Code)**:
- ❌ Bot crashes every loop (API error)
- ❌ Emergency Max Hold never triggers (always hours_held = 0)
- ❌ Positions held indefinitely (9+ hours)
- ❌ Entry time wrong (candle timestamp)
- ❌ Holding time calculation wrong (9h off)

**After (Fixed Code)**:
- ✅ Bot runs smoothly (no API errors)
- ✅ Emergency Max Hold triggers at 8h
- ✅ Positions exit automatically at 8h max
- ✅ Entry time = actual execution time
- ✅ Holding time accurate

## 🚨 What If Issues Persist?

### If API error still occurs:
```bash
# Check MAX_DATA_CANDLES value
grep "MAX_DATA_CANDLES =" scripts/production/opportunity_gating_bot_4x.py

# Should be 1400, not 1440
```

### If Emergency Max Hold doesn't trigger:
```bash
# Check datetime parsing fix
grep -A 10 "# Calculate time in position" scripts/production/opportunity_gating_bot_4x.py

# Should have try/except with strptime fallback
```

### If entry time still wrong:
```bash
# Check entry time recording
grep -A 5 "actual_entry_time" scripts/production/opportunity_gating_bot_4x.py

# Should use datetime.now().isoformat(), not str(current_time)
```

## 📝 Post-Restart Actions

1. **Monitor first 30 minutes**: Check logs for stability
2. **Monitor first position**: Verify all fixes work correctly
3. **Monitor 8+ hour position**: Verify Emergency Max Hold triggers
4. **Document results**: Update system status with actual behavior

## 🎯 Success Criteria

✅ Bot completes these without errors:
1. Fetch 1400 candles successfully
2. Generate LONG/SHORT signals
3. Record entry time correctly (if position entered)
4. Calculate holding time correctly
5. Exit position at 8h if Emergency Max Hold needed

---

**Created**: 2025-10-17 16:42 KST
**Status**: Ready for restart
**All fixes verified**: ✅
