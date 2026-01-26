# Phase Transition Guide: Multiple Positions Deployment

**Document Version**: 1.0
**Created**: 2025-11-15 19:30 KST
**Author**: Claude Code
**Purpose**: Complete guide for Phase 1 → Phase 2 → Phase 3 transitions

---

## Table of Contents

1. [Phase Overview](#phase-overview)
2. [Phase 1: Conservative Start (MAX_POSITIONS = 2)](#phase-1-conservative-start)
3. [Phase 2: Moderate Expansion (MAX_POSITIONS = 3)](#phase-2-moderate-expansion)
4. [Phase 3: Full Production (MAX_POSITIONS = 5)](#phase-3-full-production)
5. [Success Criteria](#success-criteria)
6. [Transition Procedures](#transition-procedures)
7. [Monitoring Commands](#monitoring-commands)
8. [Emergency Procedures](#emergency-procedures)
9. [Red Flags](#red-flags)
10. [Timeline Summary](#timeline-summary)

---

## Phase Overview

### Strategy: Staged Conservative Deployment

```yaml
Phase 1 (24 hours):
  MAX_POSITIONS: 2
  Purpose: System validation, trailing SL testing
  Start: 2025-11-15 17:59 KST

Phase 2 (24 hours):
  MAX_POSITIONS: 3
  Purpose: Moderate load testing, performance validation
  Start: 2025-11-16 18:00 KST (if Phase 1 success)

Phase 3 (Ongoing):
  MAX_POSITIONS: 5
  Purpose: Full production capacity
  Start: 2025-11-17 18:00 KST (if Phase 2 success)
```

### Key Configuration Changes

| Parameter | Single Position | Phase 1 | Phase 2 | Phase 3 |
|-----------|----------------|---------|---------|---------|
| MAX_POSITIONS | 1 | 2 | 3 | 5 |
| Position Size | 218% | 40% | 40% | 40% |
| Margin Cap | N/A | 95% | 95% | 95% |
| Portfolio SL | N/A | -10% | -10% | -10% |
| Trailing SL | No | Yes (5 rules) | Yes | Yes |
| Timeframe | 5-min | 15-min | 15-min | 15-min |

---

## Phase 1: Conservative Start

### Configuration

```python
# scripts/production/opportunity_gating_bot_4x.py (Line 89)
MAX_POSITIONS = 2  # PHASE 1: Start with 2 positions
```

### Expected Performance

```yaml
Trade Frequency: 5-7 trades/day (vs 3.4 previous)
Win Rate: 60-70%
Daily Return: +2-4%
Concurrent Positions: 0-2 (avg ~1.2)
Exit Mechanisms: 60-70% ML Exit, 20-30% Trailing SL, <10% Stop Loss
Direction Split: 60-70% LONG, 30-40% SHORT (more balanced)
```

### Success Criteria (24-Hour Period)

```yaml
Required (ALL must pass):
  - Duration: >= 24 hours monitoring
  - Win Rate: >= 60%
  - Portfolio SL Triggers: 0
  - Positive P&L: > $0
  - System Stability: No crashes

Recommended (2+ should pass):
  - Trailing SL Usage: >= 15% of exits
  - Trade Frequency: 4-8/day
  - Direction Balance: LONG 50-75%, SHORT 25-50%
  - Individual SL Rate: < 20%
```

### Monitoring Checklist

**Every 4 Hours**:
- [ ] Check bot process alive (`ps aux | grep opportunity_gating_bot_4x`)
- [ ] Review current positions (`python scripts/utils/check_current_position.py`)
- [ ] Check recent trades in logs (`tail -100 logs/opportunity_gating_bot_4x_phase1_*.log`)
- [ ] Verify balance trend (should be stable or increasing)

**Every 12 Hours**:
- [ ] Run performance analysis (`python scripts/utils/analyze_phase_performance.py 1`)
- [ ] Check win rate trend
- [ ] Review exit mechanism distribution
- [ ] Verify no Portfolio SL triggers

**At 24 Hours**:
- [ ] Full performance evaluation
- [ ] Success criteria validation
- [ ] Go/No-Go decision for Phase 2

---

## Phase 2: Moderate Expansion

### Configuration

```python
# scripts/production/opportunity_gating_bot_4x.py (Line 89)
MAX_POSITIONS = 3  # PHASE 2: Moderate expansion
```

### Expected Performance

```yaml
Trade Frequency: 7-10 trades/day
Win Rate: 60-70%
Daily Return: +3-5%
Concurrent Positions: 0-3 (avg ~1.8)
Exit Mechanisms: 60-70% ML Exit, 20-30% Trailing SL, <10% Stop Loss
Direction Split: 58-65% LONG, 35-42% SHORT (approaching target)
```

### Success Criteria (24-Hour Period)

```yaml
Required (ALL must pass):
  - Duration: >= 24 hours monitoring
  - Win Rate: >= 60%
  - Portfolio SL Triggers: 0
  - Positive P&L: > $0
  - System Stability: No crashes
  - 3 Concurrent Positions Tested: At least once

Recommended (3+ should pass):
  - Trailing SL Usage: >= 20% of exits
  - Trade Frequency: 6-12/day
  - Direction Balance: LONG 55-70%, SHORT 30-45%
  - Individual SL Rate: < 15%
  - Daily Return: >= +2%
```

### Deployment Procedure

**Automated** (Recommended):
```bash
cd bingx_rl_trading_bot
bash scripts/utils/deploy_phase2.sh
```

**Manual** (If needed):
1. Stop Phase 1 bot: `pkill -f opportunity_gating_bot_4x`
2. Backup Phase 1 config: `cp scripts/production/opportunity_gating_bot_4x.py scripts/production/opportunity_gating_bot_4x_phase1.py.bak`
3. Update MAX_POSITIONS: `sed -i 's/MAX_POSITIONS = 2/MAX_POSITIONS = 3/' scripts/production/opportunity_gating_bot_4x.py`
4. Start Phase 2 bot: `nohup python scripts/production/opportunity_gating_bot_4x.py > logs/opportunity_gating_bot_4x_phase2_$(date +%Y%m%d_%H%M%S).log 2>&1 &`
5. Verify: `python scripts/utils/check_current_position.py`

---

## Phase 3: Full Production

### Configuration

```python
# scripts/production/opportunity_gating_bot_4x.py (Line 89)
MAX_POSITIONS = 5  # PHASE 3: Full production capacity
```

### Expected Performance

```yaml
Trade Frequency: 9-12 trades/day (approaching backtest 9.46/day)
Win Rate: 60-70%
Daily Return: +4-6%
Weekly Return: +25-35%
Monthly Return: +100-150%
Concurrent Positions: 0-5 (avg ~2.5)
Exit Mechanisms: 60-70% ML Exit, 20-30% Trailing SL, <10% Stop Loss
Direction Split: 58% LONG, 42% SHORT (backtest target achieved)
```

### Success Criteria (Ongoing Monitoring)

```yaml
Weekly Performance:
  - Win Rate: >= 58%
  - Weekly Return: >= +20%
  - Portfolio SL Triggers: <= 1/week
  - Trailing SL Usage: >= 20%
  - Direction Balance: LONG 50-65%, SHORT 35-50%

Monthly Performance:
  - Win Rate: >= 60%
  - Monthly Return: >= +80%
  - Portfolio SL Triggers: <= 2/month
  - Individual SL Rate: <= 20%
  - System Uptime: >= 98%
```

### Deployment Procedure

**Automated** (Recommended):
```bash
cd bingx_rl_trading_bot
bash scripts/utils/deploy_phase3.sh
```

**Manual** (If needed):
1. Stop Phase 2 bot: `pkill -f opportunity_gating_bot_4x`
2. Backup Phase 2 config: `cp scripts/production/opportunity_gating_bot_4x.py scripts/production/opportunity_gating_bot_4x_phase2.py.bak`
3. Update MAX_POSITIONS: `sed -i 's/MAX_POSITIONS = 3/MAX_POSITIONS = 5/' scripts/production/opportunity_gating_bot_4x.py`
4. Start Phase 3 bot: `nohup python scripts/production/opportunity_gating_bot_4x.py > logs/opportunity_gating_bot_4x_phase3_$(date +%Y%m%d_%H%M%S).log 2>&1 &`
5. Verify: `python scripts/utils/check_current_position.py`

---

## Success Criteria

### Required Criteria (MUST Pass ALL)

These criteria apply to ALL phases:

1. **Duration**: Complete full monitoring period (24h for Phase 1/2)
2. **Win Rate**: >= 60% (validates model quality)
3. **Portfolio Stop Loss**: 0 triggers (validates risk management)
4. **Positive P&L**: Total return > $0 (validates profitability)
5. **System Stability**: No crashes or freezes (validates code quality)

### Recommended Criteria (SHOULD Pass 2-3)

These improve confidence but are not blocking:

1. **Trailing SL Usage**: >= 15-20% of exits (validates trailing SL working)
2. **Trade Frequency**: Within expected range for phase
3. **Direction Balance**: Approaching 58/42 LONG/SHORT target
4. **Individual SL Rate**: < 20% (validates entry quality)
5. **Daily Return**: Positive and consistent

### Evaluation Process

```python
# Automated evaluation
python scripts/utils/analyze_phase_performance.py <phase_number>

# Manual evaluation
python scripts/utils/check_current_position.py
tail -200 logs/opportunity_gating_bot_4x_phase*.log | grep "TRADE CLOSED"
```

---

## Transition Procedures

### Phase 1 → Phase 2 Transition

**Timing**: 2025-11-16 18:00 KST (24 hours after Phase 1 start)

**Checklist**:
- [ ] **Evaluate Phase 1 Performance** (18:00-18:15 KST)
  - Run: `python scripts/utils/analyze_phase_performance.py 1`
  - Verify all Required criteria pass
  - Document results in `claudedocs/PHASE1_FINAL_RESULTS_20251116.md`

- [ ] **Go/No-Go Decision** (18:15-18:20 KST)
  - **GO**: All Required criteria pass → Proceed to Phase 2 deployment
  - **NO-GO**: Any Required criteria fails → Extend Phase 1 by 24h or rollback

- [ ] **Phase 2 Deployment** (18:20-18:25 KST - if GO)
  - Run: `bash scripts/utils/deploy_phase2.sh`
  - Verify bot started: `ps aux | grep opportunity_gating_bot_4x`
  - Check initial status: `python scripts/utils/check_current_position.py`
  - Monitor first 30 minutes for stability

- [ ] **Documentation** (18:25-18:30 KST)
  - Update CLAUDE.md with Phase 2 start time
  - Create Phase 2 monitoring schedule
  - Set alarm for 6-hour check (2025-11-17 00:00 KST)

### Phase 2 → Phase 3 Transition

**Timing**: 2025-11-17 18:00 KST (24 hours after Phase 2 start)

**Checklist**:
- [ ] **Evaluate Phase 2 Performance** (18:00-18:15 KST)
  - Run: `python scripts/utils/analyze_phase_performance.py 2`
  - Verify all Required criteria pass
  - Confirm 3 concurrent positions tested at least once
  - Document results in `claudedocs/PHASE2_FINAL_RESULTS_20251117.md`

- [ ] **Go/No-Go Decision** (18:15-18:20 KST)
  - **GO**: All Required + 3 concurrent test → Proceed to Phase 3
  - **NO-GO**: Any Required criteria fails → Extend Phase 2 or rollback to Phase 1

- [ ] **Phase 3 Deployment** (18:20-18:25 KST - if GO)
  - Run: `bash scripts/utils/deploy_phase3.sh`
  - Verify bot started: `ps aux | grep opportunity_gating_bot_4x`
  - Check initial status: `python scripts/utils/check_current_position.py`
  - Monitor first 1 hour for stability

- [ ] **Production Monitoring Setup** (18:25-18:30 KST)
  - Update CLAUDE.md with Phase 3 production status
  - Set up daily performance review schedule
  - Enable alert system for Portfolio SL triggers
  - Document baseline performance expectations

---

## Monitoring Commands

### Quick Status Check

```bash
# Current position and balance
python scripts/utils/check_current_position.py

# Bot process status
ps aux | grep opportunity_gating_bot_4x

# Latest signals
tail -20 logs/opportunity_gating_bot_4x_phase*.log | grep "Current signals"

# Recent trades
tail -100 logs/opportunity_gating_bot_4x_phase*.log | grep "TRADE CLOSED"
```

### Performance Analysis

```bash
# Phase-specific performance
python scripts/utils/analyze_phase_performance.py 1  # Phase 1
python scripts/utils/analyze_phase_performance.py 2  # Phase 2
python scripts/utils/analyze_phase_performance.py 3  # Phase 3

# Current session performance
tail -500 logs/opportunity_gating_bot_4x_phase*.log | grep -E "TRADE CLOSED|Portfolio"
```

### System Health

```bash
# Check for errors
tail -200 logs/opportunity_gating_bot_4x_phase*.log | grep -i error

# Check for warnings
tail -200 logs/opportunity_gating_bot_4x_phase*.log | grep -i warning

# Verify trailing SL adjustments
tail -200 logs/opportunity_gating_bot_4x_phase*.log | grep "Trailing SL"

# Check position sync
tail -200 logs/opportunity_gating_bot_4x_phase*.log | grep "Position sync"
```

### Exchange Verification

```bash
# Verify positions match exchange
python scripts/utils/check_current_position.py

# Check state file
cat results/opportunity_gating_bot_4x_state.json | grep -A 20 "positions"

# Monitor balance changes
tail -50 logs/opportunity_gating_bot_4x_phase*.log | grep "Current balance"
```

---

## Emergency Procedures

### Portfolio Stop Loss Triggered

**Severity**: 🚨 CRITICAL
**Action**: IMMEDIATE

```yaml
What Happened:
  - Total balance dropped >= 10% from session start
  - ALL positions automatically closed
  - Bot enters cooldown mode

Immediate Actions:
  1. Verify all positions closed on exchange
  2. Check total loss amount
  3. Review what caused the cascade
  4. Evaluate if models need emergency retraining

Commands:
  python scripts/utils/check_current_position.py
  tail -500 logs/opportunity_gating_bot_4x_phase*.log | grep -E "Portfolio|TRADE CLOSED"

Recovery:
  - DO NOT restart bot immediately
  - Analyze root cause (model failure? market regime change?)
  - Consider extending current phase or rolling back
  - Restart only after issue identified and resolved
```

### Individual Stop Loss Cascade (3+ in 1 hour)

**Severity**: ⚠️ HIGH
**Action**: Within 30 minutes

```yaml
What Happened:
  - 3 or more positions hit individual SL within 1 hour
  - Possible model degradation or regime mismatch

Immediate Actions:
  1. Check current signals (LONG/SHORT probabilities)
  2. Review recent price action
  3. Verify features calculating correctly
  4. Consider temporary threshold increase

Commands:
  tail -100 logs/opportunity_gating_bot_4x_phase*.log | grep "Stop Loss"
  python scripts/utils/check_feature_values.py

Recovery:
  - Increase Entry threshold temporarily (0.60 → 0.70)
  - Monitor for 2 hours
  - If continues, consider pausing bot and retraining
```

### Bot Process Crash

**Severity**: ⚠️ MEDIUM
**Action**: Within 15 minutes

```yaml
What Happened:
  - Bot process terminated unexpectedly
  - Positions may still be open on exchange

Immediate Actions:
  1. Check if positions still open on exchange
  2. Review crash logs for error cause
  3. Verify state file not corrupted
  4. Restart bot after fixing issue

Commands:
  ps aux | grep opportunity_gating_bot_4x
  tail -200 logs/opportunity_gating_bot_4x_phase*.log | tail -50
  python scripts/utils/check_current_position.py

Recovery:
  - Fix identified bug
  - Verify state file integrity
  - Restart bot: nohup python scripts/production/opportunity_gating_bot_4x.py > logs/opportunity_gating_bot_4x_recovery_$(date +%Y%m%d_%H%M%S).log 2>&1 &
  - Monitor closely for 1 hour
```

### Rollback to Previous Phase

**Severity**: 🔄 CONTROLLED
**Action**: Planned procedure

```yaml
Phase 3 → Phase 2 Rollback:
  1. Stop Phase 3 bot: pkill -f opportunity_gating_bot_4x
  2. Restore Phase 2 config: cp scripts/production/opportunity_gating_bot_4x_phase2.py.bak scripts/production/opportunity_gating_bot_4x.py
  3. Restart: nohup python scripts/production/opportunity_gating_bot_4x.py > logs/opportunity_gating_bot_4x_rollback_phase2_$(date +%Y%m%d_%H%M%S).log 2>&1 &
  4. Verify: python scripts/utils/check_current_position.py
  5. Document reason in claudedocs/

Phase 2 → Phase 1 Rollback:
  1. Stop Phase 2 bot: pkill -f opportunity_gating_bot_4x
  2. Restore Phase 1 config: cp scripts/production/opportunity_gating_bot_4x_phase1.py.bak scripts/production/opportunity_gating_bot_4x.py
  3. Restart: nohup python scripts/production/opportunity_gating_bot_4x.py > logs/opportunity_gating_bot_4x_rollback_phase1_$(date +%Y%m%d_%H%M%S).log 2>&1 &
  4. Verify: python scripts/utils/check_current_position.py
  5. Document reason in claudedocs/

Phase 1 → Single Position Rollback:
  1. Stop Phase 1 bot: pkill -f opportunity_gating_bot_4x
  2. Use previous single-position bot or restore from backup
  3. May need state migration back to single position schema
  4. Document detailed reasoning (why multiple positions failed)
```

---

## Red Flags

### 🚨 CRITICAL - Stop Bot Immediately

```yaml
1. Portfolio Stop Loss Triggered
   - Total loss >= 10% in session
   - Action: STOP bot, analyze root cause

2. Repeated Crashes (3+ in 1 hour)
   - Code bug or system instability
   - Action: STOP bot, fix bug, restart

3. Win Rate < 30% (over 20+ trades)
   - Model completely degraded
   - Action: STOP bot, retrain models immediately

4. Exchange API Errors (5+ in 10 minutes)
   - Connection issues or API rate limiting
   - Action: STOP bot, check API status
```

### ⚠️ WARNING - Increase Monitoring

```yaml
1. Win Rate 40-50% (over 10+ trades)
   - Model degrading, possible regime change
   - Action: Increase Entry threshold, monitor closely

2. Individual SL Rate > 30% (over 10+ trades)
   - Entry quality poor, model overconfident
   - Action: Increase Entry threshold 0.60 → 0.70

3. Trade Frequency < 2/day (for 8+ hours)
   - Signals too conservative or market quiet
   - Action: Check thresholds, verify features

4. Direction Imbalance > 80% (over 20+ trades)
   - Model bias returning (LONG or SHORT)
   - Action: Consider threshold adjustments

5. Trailing SL Usage < 5% (over 20+ trades)
   - Trailing SL not activating as expected
   - Action: Verify trailing SL logic, check profit levels
```

### 📊 MONITORING - Track Trends

```yaml
1. Win Rate 50-60%
   - Below target but acceptable
   - Action: Monitor trend, document if improving/declining

2. Trade Frequency 4-6/day
   - Lower than expected but functional
   - Action: Review threshold settings

3. Trailing SL Usage 10-15%
   - Lower than expected but working
   - Action: Monitor profit levels reached

4. Daily Return +1-2%
   - Lower than target but positive
   - Action: Ensure consistency over time
```

---

## Timeline Summary

### Full Deployment Schedule

```yaml
Phase 1 (24 hours):
  Start: 2025-11-15 17:59 KST ✅ DEPLOYED
  End: 2025-11-16 17:59 KST
  Evaluation: 2025-11-16 18:00-18:15 KST
  Decision: 2025-11-16 18:15-18:20 KST

Phase 2 (24 hours):
  Start: 2025-11-16 18:20 KST (if Phase 1 success)
  End: 2025-11-17 18:20 KST
  Evaluation: 2025-11-17 18:20-18:35 KST
  Decision: 2025-11-17 18:35-18:40 KST

Phase 3 (Ongoing):
  Start: 2025-11-17 18:40 KST (if Phase 2 success)
  Production: Continuous operation
  Reviews: Daily performance checks
```

### Monitoring Schedule

```yaml
Phase 1 Monitoring:
  Every 4 hours: Process check, position review, log check
  Every 12 hours: Performance analysis, exit mechanism review
  At 24 hours: Full evaluation, Go/No-Go decision

Phase 2 Monitoring:
  Every 4 hours: Same as Phase 1
  Every 12 hours: Same as Phase 1 + concurrent position check
  At 24 hours: Full evaluation, Go/No-Go decision

Phase 3 Monitoring:
  Daily (18:00 KST): Performance review, win rate check
  Weekly (Sunday 18:00 KST): Comprehensive analysis
  Monthly (1st of month): Full system audit
```

### Key Decision Points

```yaml
2025-11-16 18:15 KST: Phase 1 → Phase 2 Decision
  - Required: All success criteria pass
  - If PASS: Deploy Phase 2 at 18:20 KST
  - If FAIL: Extend Phase 1 by 24h OR rollback

2025-11-17 18:35 KST: Phase 2 → Phase 3 Decision
  - Required: All success criteria + 3 concurrent test
  - If PASS: Deploy Phase 3 at 18:40 KST (PRODUCTION)
  - If FAIL: Extend Phase 2 by 24h OR rollback to Phase 1

Ongoing: Weekly Reviews (Every Sunday 18:00 KST)
  - Evaluate weekly performance
  - Check for model degradation
  - Decide if retraining needed
```

---

## Appendix

### Configuration Reference

```python
# Multiple Positions Strategy Configuration
POSITION_SIZE_RATIO = 0.40  # 40% of available margin per signal
MAX_POSITIONS = 2  # Phase 1: 2 | Phase 2: 3 | Phase 3: 5
MARGIN_USAGE_CAP = 0.95  # Use up to 95% of total margin
PORTFOLIO_STOP_LOSS = 0.10  # -10% total balance stop

# Trailing Stop Loss Rules (15-min candle checks)
# Rule 1: Profit > 5% → SL to breakeven (0.1% above entry)
# Rule 2: Profit > 10% → SL locks 50% profit
# Rule 3: Profit > 20% → SL locks 70% profit
# Rule 4: Old position (>50 candles) + profit > 2% → Tighten to 30%
# Rule 5: High volatility + losing → Keep original SL

# Entry/Exit Thresholds
LONG_ENTRY_THRESHOLD = 0.60
SHORT_ENTRY_THRESHOLD = 0.60
EXIT_THRESHOLD = 0.75

# Risk Management
STOP_LOSS_PCT = 0.03  # -3% balance per position
MAX_HOLD_TIME = 120  # candles (10 hours @ 5-min)
LEVERAGE = 4
```

### File Locations

```yaml
Bot Script:
  scripts/production/opportunity_gating_bot_4x.py

Deployment Scripts:
  scripts/utils/deploy_phase2.sh
  scripts/utils/deploy_phase3.sh

Analysis Scripts:
  scripts/utils/analyze_phase_performance.py
  scripts/utils/check_current_position.py

State Files:
  results/opportunity_gating_bot_4x_state.json
  results/opportunity_gating_bot_4x_state.json.backup_pre_migration

Logs:
  logs/opportunity_gating_bot_4x_phase1_*.log
  logs/opportunity_gating_bot_4x_phase2_*.log
  logs/opportunity_gating_bot_4x_phase3_*.log

Documentation:
  claudedocs/PHASE1_DEPLOYMENT_RESULTS_20251115.md
  claudedocs/ROOT_CAUSE_ANALYSIS_AND_RESOLUTION_20251115.md
  claudedocs/PHASE_TRANSITION_GUIDE_20251115.md (this file)
```

---

**Document Status**: ✅ COMPLETE
**Next Review**: 2025-11-16 18:00 KST (Phase 1 → Phase 2 decision)
**Maintained By**: Claude Code + User
