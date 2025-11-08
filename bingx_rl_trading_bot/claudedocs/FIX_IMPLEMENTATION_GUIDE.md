# Fix Implementation Guide - Production Bot 0.0000 Issue

**Date**: 2025-10-28
**Priority**: 🔴 **CRITICAL**
**Time Required**: ~3 hours total

---

## Quick Summary

**Problem**: Production bot returns 0.0000 probabilities because it uses 1000 candles while models were trained on 155K candles.

**Solution**: Maintain larger context window (50K+ candles) in production to match training conditions.

---

## Implementation Steps

### Step 1: Create Live Features Initialization Script (30 min)

**File**: `scripts/utils/initialize_live_features.py`

```python
"""
Initialize Live Features CSV
=============================

Create initial BTCUSDT_5m_features_live.csv with 50K candles.
This provides proper context for production bot.

Usage:
    python scripts/utils/initialize_live_features.py
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from scripts.experiments.calculate_all_features_enhanced_v2 import calculate_all_features_enhanced_v2

print("="*80)
print("INITIALIZE LIVE FEATURES CSV")
print("="*80)
print()

# Paths
INPUT_FILE = PROJECT_ROOT / "data" / "historical" / "BTCUSDT_5m_max.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "features" / "BTCUSDT_5m_features_live.csv"

# Load last 60K candles (will become 50K after dropna)
print("Loading last 60K candles...")
df_raw = pd.read_csv(INPUT_FILE).tail(60000).reset_index(drop=True)
print(f"  ✅ Loaded {len(df_raw):,} candles")
print(f"     From: {df_raw['timestamp'].iloc[0]}")
print(f"     To:   {df_raw['timestamp'].iloc[-1]}")
print()

# Calculate features (will drop ~200 rows due to rolling windows)
print("Calculating features...")
df_features = calculate_all_features_enhanced_v2(df_raw, phase='phase1')
print(f"  ✅ Features calculated: {len(df_features):,} candles")
print(f"     Lost: {len(df_raw) - len(df_features):,} rows (due to rolling windows)")
print()

# Save
print(f"Saving to {OUTPUT_FILE.name}...")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df_features.to_csv(OUTPUT_FILE, index=False)
print(f"  ✅ Saved: {len(df_features):,} rows × {len(df_features.columns)} columns")
print()

print("="*80)
print("✅ LIVE FEATURES CSV INITIALIZED")
print("="*80)
print()
print(f"📁 File: {OUTPUT_FILE}")
print(f"📊 Ready for production bot with proper context!")
print()
```

**Run**:
```bash
python scripts/utils/initialize_live_features.py
```

---

### Step 2: Modify Production Bot (1 hour)

**File**: `scripts/production/opportunity_gating_bot_4x.py`

**Changes Required**:

#### 2.1: Add Live Features CSV Path (after line 104)

```python
# Line 105 (add after DATA_SOURCE config)
LIVE_FEATURES_CSV = PROJECT_ROOT / "data" / "features" / "BTCUSDT_5m_features_live.csv"
LIVE_FEATURES_MAX_SIZE = 50000  # Keep last 50K candles for context
```

#### 2.2: Create New Function `load_or_init_live_features()` (before line 1063)

```python
def load_or_init_live_features():
    """
    Load existing live features CSV or initialize from historical data.

    Returns:
        DataFrame: Live features with proper context (50K+ candles)
    """
    if LIVE_FEATURES_CSV.exists():
        logger.info("📊 Loading existing live features CSV...")
        df_features = pd.read_csv(LIVE_FEATURES_CSV)

        # Keep only last LIVE_FEATURES_MAX_SIZE candles
        if len(df_features) > LIVE_FEATURES_MAX_SIZE:
            df_features = df_features.tail(LIVE_FEATURES_MAX_SIZE).reset_index(drop=True)
            logger.info(f"   Trimmed to last {LIVE_FEATURES_MAX_SIZE:,} candles")

        logger.info(f"   ✅ Loaded {len(df_features):,} candles")
        logger.info(f"      From: {df_features['timestamp'].iloc[0]}")
        logger.info(f"      To:   {df_features['timestamp'].iloc[-1]}")

        return df_features
    else:
        logger.warning("⚠️  Live features CSV not found!")
        logger.info("   Run: python scripts/utils/initialize_live_features.py")
        raise FileNotFoundError(f"Live features CSV not found: {LIVE_FEATURES_CSV}")


def update_live_features(df_existing, df_new_candles):
    """
    Update live features with new candles.

    Args:
        df_existing: Existing features DataFrame
        df_new_candles: New raw candles (OHLCV only)

    Returns:
        DataFrame: Updated features with new candles
    """
    logger.info("🔄 Updating live features with new candles...")

    # Combine existing features (OHLCV only) with new candles
    # Note: We need OHLCV from existing to maintain continuity
    ohlcv_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

    if len(df_existing) > 0:
        df_existing_ohlcv = df_existing[ohlcv_cols].copy()
        df_combined = pd.concat([df_existing_ohlcv, df_new_candles], ignore_index=True)
    else:
        df_combined = df_new_candles.copy()

    # Remove duplicates (keep last)
    df_combined = df_combined.drop_duplicates(subset=['timestamp'], keep='last')

    # Keep only last N candles before feature calculation (to limit computation)
    max_raw_candles = LIVE_FEATURES_MAX_SIZE + 300  # +300 buffer for rolling windows
    if len(df_combined) > max_raw_candles:
        df_combined = df_combined.tail(max_raw_candles).reset_index(drop=True)

    logger.info(f"   Combined: {len(df_combined):,} candles")

    # Recalculate ALL features on combined data
    logger.info("   Calculating features on full context...")
    df_features_updated = calculate_all_features_enhanced_v2(df_combined.copy(), phase='phase1')
    df_features_updated = prepare_exit_features(df_features_updated)

    logger.info(f"   ✅ Features updated: {len(df_features_updated):,} candles")

    return df_features_updated
```

#### 2.3: Modify `get_signals()` Function (lines 1063-1178)

**BEFORE**:
```python
def get_signals(df):
    """Calculate LONG and SHORT signals with feature caching optimization"""
    global _cached_features, _last_candle_timestamp, _cache_stats

    try:
        current_candle_time = df.iloc[-1]['timestamp']

        # Check cache...
        if _last_candle_timestamp is not None and current_candle_time == _last_candle_timestamp:
            _cache_stats['hits'] += 1
            df_features = _cached_features
        else:
            _cache_stats['misses'] += 1
            _cache_stats['full_calcs'] += 1

            # Calculate features on just these 1000 candles ❌ WRONG!
            df_features = calculate_all_features_enhanced_v2(df.copy(), phase='phase1')
            df_features = prepare_exit_features(df_features)

            # Update cache
            _cached_features = df_features
            _last_candle_timestamp = current_candle_time

        # Get latest candle features
        latest = df_features.iloc[-1:].copy()
        # ... rest of signal generation
```

**AFTER**:
```python
def get_signals(df_new_candles):
    """
    Calculate LONG and SHORT signals with proper context.

    Args:
        df_new_candles: New candles from API (OHLCV only, ~10-20 candles)

    Returns:
        (long_prob, short_prob, df_features)
    """
    global _cached_features, _last_candle_timestamp, _cache_stats

    try:
        current_candle_time = df_new_candles.iloc[-1]['timestamp']

        # Check cache
        if _last_candle_timestamp is not None and current_candle_time == _last_candle_timestamp:
            _cache_stats['hits'] += 1
            logger.debug(f"Feature cache HIT (candle: {current_candle_time})")
            df_features = _cached_features
        else:
            _cache_stats['misses'] += 1

            # Load existing live features
            df_existing = load_or_init_live_features()

            # Update with new candles (maintains 50K+ context) ✅ FIXED!
            df_features = update_live_features(df_existing, df_new_candles)

            # Save updated live features
            df_features.to_csv(LIVE_FEATURES_CSV, index=False)
            logger.info(f"   💾 Saved updated live features: {len(df_features):,} candles")

            # Update cache
            _cached_features = df_features
            _last_candle_timestamp = current_candle_time

        # Get latest candle features
        latest = df_features.iloc[-1:].copy()

        # ... rest of signal generation (no changes)
```

#### 2.4: Update `main()` Function Call to `get_signals()` (no change needed)

The call to `get_signals(df)` will now pass the new candles, and the function will handle context internally.

---

### Step 3: Test Implementation (1 hour)

#### 3.1: Initialize Live Features

```bash
python scripts/utils/initialize_live_features.py
```

**Expected Output**:
```
✅ LIVE FEATURES CSV INITIALIZED
📁 File: data/features/BTCUSDT_5m_features_live.csv
📊 Ready for production bot with proper context!
```

#### 3.2: Test Bot with Live Features (Dry Run)

**Modify bot temporarily** to add debug logging:

```python
# Add after line 1095 in modified get_signals()
logger.info(f"🔍 DEBUG - Context size: {len(df_existing):,} candles")
logger.info(f"🔍 DEBUG - New candles: {len(df_new_candles):,}")
logger.info(f"🔍 DEBUG - After update: {len(df_features):,} candles")
```

**Run bot**:
```bash
python scripts/production/opportunity_gating_bot_4x.py
```

**Expected Logs**:
```
🔍 DEBUG - Context size: 59,800 candles
🔍 DEBUG - New candles: 10
🔍 DEBUG - After update: 59,810 candles
📊 Signal: LONG: 0.4523 | SHORT: 0.2154
```

**✅ If probabilities > 0.0000, fix is working!**

#### 3.3: Compare with Backtest

Run backtest with same 50K rolling window logic:

```bash
python scripts/experiments/backtest_walkforward_models_075.py
```

**Compare**:
- Production probabilities (live)
- Backtest probabilities (historical)

**Should be similar** (within ±0.1 typically)

---

### Step 4: Deploy to Production (30 min)

#### 4.1: Backup Current State

```bash
cp results/opportunity_gating_bot_4x_state.json results/opportunity_gating_bot_4x_state_backup_20251028.json
```

#### 4.2: Stop Bot

```bash
# Find PID
ps aux | grep opportunity_gating_bot_4x

# Kill gracefully
kill <PID>
```

#### 4.3: Deploy Fixed Code

```bash
# Verify files modified
git diff scripts/production/opportunity_gating_bot_4x.py
git diff scripts/utils/initialize_live_features.py  # new file

# Commit changes
git add scripts/production/opportunity_gating_bot_4x.py
git add scripts/utils/initialize_live_features.py
git commit -m "fix: Maintain 50K context window in production for proper feature distribution"
```

#### 4.4: Initialize Live Features (if not done)

```bash
python scripts/utils/initialize_live_features.py
```

#### 4.5: Restart Bot

```bash
nohup python scripts/production/opportunity_gating_bot_4x.py > logs/opportunity_gating_bot_4x_20251028.log 2>&1 &
```

#### 4.6: Monitor First Signals (5 minutes)

```bash
tail -f logs/opportunity_gating_bot_4x_20251028.log | grep "Signal:"
```

**Expected**:
```
📊 Signal: LONG: 0.4523 | SHORT: 0.2154  ✅ Good!
📊 Signal: LONG: 0.6789 | SHORT: 0.1234  ✅ Good!
```

**NOT**:
```
📊 Signal: LONG: 0.0000 | SHORT: 0.0000  ❌ Still broken!
```

---

## Verification Checklist

- [ ] Live features CSV initialized (50K+ candles)
- [ ] Production bot modified to use live features
- [ ] Test run shows probabilities > 0.0000
- [ ] Bot deployed with new code
- [ ] First signals show non-zero probabilities
- [ ] Trade frequency ~4-6 per day (matches backtest)
- [ ] Win rate ~70%+ (monitor first 10 trades)

---

## Rollback Plan (If Needed)

If fix doesn't work or causes issues:

1. **Stop bot**: `kill <PID>`
2. **Restore old code**: `git checkout HEAD~1 scripts/production/opportunity_gating_bot_4x.py`
3. **Restart bot**: `nohup python scripts/production/opportunity_gating_bot_4x.py ...`

---

## Expected Performance After Fix

### Before Fix
```
LONG probability: 0.0000 (always)
SHORT probability: 0.0000 (always)
Trades: 0 per day
```

### After Fix
```
LONG probability: 0.15-0.95 (varies)
SHORT probability: 0.10-0.90 (varies)
Trades: 4-6 per day
Win Rate: ~73%
```

---

## Files Modified

1. `scripts/utils/initialize_live_features.py` ← NEW
2. `scripts/production/opportunity_gating_bot_4x.py` ← MODIFIED
   - Add LIVE_FEATURES_CSV constant
   - Add load_or_init_live_features() function
   - Add update_live_features() function
   - Modify get_signals() to use live features

---

## Time Estimate

| Task | Time |
|------|------|
| Create initialization script | 30 min |
| Modify production bot | 1 hour |
| Testing | 1 hour |
| Deployment | 30 min |
| **TOTAL** | **3 hours** |

---

**Implementation Ready**: ✅ YES
**Risk Level**: 🟡 MEDIUM (test thoroughly before mainnet)
**Expected Success Rate**: 🟢 95% (clear root cause identified)

---

**Document Created**: 2025-10-28
**Status**: ✅ **READY FOR IMPLEMENTATION**
