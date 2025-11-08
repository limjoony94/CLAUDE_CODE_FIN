#!/usr/bin/env python3
"""
Trade #2 EXIT_THRESHOLD Validation Monitor
Tracks exit probability and validates early exit hypothesis

Purpose: Monitor Trade #2 to validate if EXIT_THRESHOLD=0.70 causes early exits
"""

import json
import time
from datetime import datetime
from pathlib import Path
import re

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
STATE_FILE = PROJECT_ROOT / "results" / "phase4_testnet_trading_state.json"
LOG_FILE = PROJECT_ROOT / "logs" / "phase4_dynamic_testnet_trading_20251015.log"
MONITOR_LOG = PROJECT_ROOT / "logs" / "trade2_exit_monitor.log"

def log_monitor(message):
    """Log monitoring messages"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"{timestamp} | {message}"
    print(log_msg)
    with open(MONITOR_LOG, 'a') as f:
        f.write(log_msg + "\n")

def get_trade2_status():
    """Get current Trade #2 status from state file"""
    try:
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)

        trades = state.get('trades', [])
        if len(trades) < 2:
            return None

        trade2 = trades[1]  # Second trade (index 1)
        return trade2
    except Exception as e:
        log_monitor(f"❌ Error reading state: {e}")
        return None

def parse_latest_exit_signal():
    """Parse latest Exit Model signal from logs"""
    try:
        with open(LOG_FILE, 'r') as f:
            lines = f.readlines()

        # Find most recent Exit Model signal
        exit_prob = None
        for line in reversed(lines):
            if "Exit Model Signal (LONG):" in line:
                # Extract probability: "Exit Model Signal (LONG): 0.216 (threshold: 0.70)"
                match = re.search(r"Exit Model Signal \(LONG\): ([\d.]+)", line)
                if match:
                    exit_prob = float(match.group(1))
                    break

        return exit_prob
    except Exception as e:
        log_monitor(f"⚠️ Error parsing logs: {e}")
        return None

def parse_latest_dynamic_threshold():
    """Parse latest dynamic threshold from logs"""
    try:
        with open(LOG_FILE, 'r') as f:
            lines = f.readlines()

        # Find most recent dynamic threshold info
        signal_rate = None
        threshold_long = None
        adjustment = None

        for line in reversed(lines):
            if "Recent Signal Rate:" in line:
                match = re.search(r"Recent Signal Rate: ([\d.]+)%", line)
                if match:
                    signal_rate = float(match.group(1))
            if "LONG Threshold:" in line:
                match = re.search(r"LONG Threshold: ([\d.]+)", line)
                if match:
                    threshold_long = float(match.group(1))
            if "Threshold Adjustment:" in line:
                match = re.search(r"Threshold Adjustment: ([+-]?[\d.]+)", line)
                if match:
                    adjustment = float(match.group(1))

            if signal_rate and threshold_long and adjustment:
                break

        return {
            'signal_rate': signal_rate,
            'threshold_long': threshold_long,
            'adjustment': adjustment
        }
    except Exception as e:
        log_monitor(f"⚠️ Error parsing dynamic threshold: {e}")
        return {}

def calculate_holding_time(entry_time_str):
    """Calculate holding time in minutes"""
    try:
        entry_time = datetime.fromisoformat(entry_time_str)
        current_time = datetime.now()
        delta = current_time - entry_time
        return delta.total_seconds() / 60
    except:
        return None

def monitor_trade2():
    """Main monitoring function"""
    log_monitor("=" * 80)
    log_monitor("🔍 Trade #2 EXIT_THRESHOLD Validation Monitor Started")
    log_monitor("=" * 80)
    log_monitor("")
    log_monitor("Monitoring Objective: Validate if EXIT_THRESHOLD=0.70 causes early exits")
    log_monitor("Trade #1 Pattern: 10-min exit, Exit prob 0.716, -$62.16 loss")
    log_monitor("Hypothesis: EXIT_THRESHOLD=0.70 is TOO HIGH → premature exits")
    log_monitor("")
    log_monitor("Update Interval: Every 5 minutes (bot cycle)")
    log_monitor("=" * 80)
    log_monitor("")

    check_count = 0
    trade_closed = False
    last_exit_prob = None

    while not trade_closed:
        check_count += 1
        log_monitor(f"📊 Status Check #{check_count} - {datetime.now().strftime('%H:%M:%S')}")
        log_monitor("-" * 80)

        # Get Trade #2 status
        trade2 = get_trade2_status()

        if not trade2:
            log_monitor("⚠️ Trade #2 not found in state file")
            time.sleep(300)  # 5 minutes
            continue

        # Check if trade closed
        status = trade2.get('status', 'UNKNOWN')
        if status == 'CLOSED':
            trade_closed = True
            log_monitor("🎯 TRADE #2 CLOSED - Beginning final analysis...")
            log_monitor("")

            # Final analysis
            entry_time = trade2.get('entry_time')
            exit_time = trade2.get('exit_time')
            entry_price = trade2.get('entry_price')
            exit_price = trade2.get('exit_price')
            exit_reason = trade2.get('exit_reason', 'Unknown')
            pnl_net = trade2.get('pnl_usd_net', 0)
            pnl_gross = trade2.get('pnl_usd_gross', 0)
            transaction_cost = trade2.get('transaction_cost', 0)

            holding_minutes = calculate_holding_time(entry_time) if entry_time else None

            log_monitor("=" * 80)
            log_monitor("📋 TRADE #2 FINAL RESULTS")
            log_monitor("=" * 80)
            log_monitor(f"Entry Time: {entry_time}")
            log_monitor(f"Exit Time: {exit_time}")
            log_monitor(f"Holding Duration: {holding_minutes:.1f} minutes" if holding_minutes else "Unknown")
            log_monitor(f"Entry Price: ${entry_price:,.2f}")
            log_monitor(f"Exit Price: ${exit_price:,.2f}" if exit_price else "Unknown")
            log_monitor(f"Exit Reason: {exit_reason}")
            log_monitor("")
            log_monitor(f"Gross P&L: ${pnl_gross:,.2f}")
            log_monitor(f"Transaction Cost: ${transaction_cost:,.2f}")
            log_monitor(f"Net P&L: ${pnl_net:,.2f}")
            log_monitor("")

            # Pattern Analysis
            log_monitor("🔍 PATTERN ANALYSIS:")
            log_monitor("-" * 80)

            # Compare with Trade #1
            if holding_minutes:
                if holding_minutes < 30:
                    log_monitor("🚨 EARLY EXIT PATTERN CONFIRMED")
                    log_monitor(f"   Trade #1: 10 minutes")
                    log_monitor(f"   Trade #2: {holding_minutes:.1f} minutes")
                    log_monitor(f"   Pattern: Both trades exited early (< 30 min)")
                    log_monitor("")
                    log_monitor("✅ STRONG EVIDENCE: EXIT_THRESHOLD=0.70 is TOO HIGH")
                    log_monitor("   Recommendation: Change EXIT_THRESHOLD to 0.2-0.3")
                elif 30 <= holding_minutes < 60:
                    log_monitor("⚠️ MODERATE EXIT DURATION")
                    log_monitor(f"   Trade #1: 10 minutes (very early)")
                    log_monitor(f"   Trade #2: {holding_minutes:.1f} minutes (moderate)")
                    log_monitor("   Pattern: Inconsistent - need more data")
                    log_monitor("")
                    log_monitor("🟡 INCONCLUSIVE: Monitor next 3-5 trades")
                else:
                    log_monitor("✅ NORMAL EXIT DURATION")
                    log_monitor(f"   Trade #1: 10 minutes (anomaly?)")
                    log_monitor(f"   Trade #2: {holding_minutes:.1f} minutes (normal)")
                    log_monitor("   Pattern: Trade #1 may have been outlier")
                    log_monitor("")
                    log_monitor("🟢 WEAK EVIDENCE: EXIT_THRESHOLD=0.70 may be acceptable")
                    log_monitor("   Note: Backtest optimal was still 0.2, not 0.70")

            # Transaction cost analysis
            log_monitor("")
            log_monitor("💰 TRANSACTION COST ANALYSIS:")
            if pnl_gross > 0 and transaction_cost > pnl_gross:
                log_monitor(f"🚨 TRANSACTION COST > GROSS PROFIT")
                log_monitor(f"   Gross Profit: ${pnl_gross:,.2f}")
                log_monitor(f"   Transaction Cost: ${transaction_cost:,.2f}")
                log_monitor(f"   Cost Ratio: {transaction_cost/pnl_gross:.1f}x gross profit")
                log_monitor("")
                log_monitor("   ⚠️ Early exits eating into profits")

            # Exit reason analysis
            log_monitor("")
            log_monitor("🎯 EXIT MECHANISM:")
            if "ML Exit" in exit_reason:
                log_monitor(f"   Exit triggered by: ML Exit Model")
                log_monitor(f"   Last Exit Prob: {last_exit_prob}")
                if last_exit_prob and last_exit_prob >= 0.70:
                    log_monitor(f"   ✅ EXIT_THRESHOLD (0.70) triggered as expected")
            elif "Stop Loss" in exit_reason:
                log_monitor(f"   Exit triggered by: Stop Loss")
                log_monitor(f"   ⚠️ Cannot validate EXIT_THRESHOLD from this trade")
            elif "Take Profit" in exit_reason:
                log_monitor(f"   Exit triggered by: Take Profit")
                log_monitor(f"   ✅ Profitable exit, but cannot validate EXIT_THRESHOLD")

            log_monitor("")
            log_monitor("=" * 80)
            log_monitor("📊 NEXT ACTIONS:")
            log_monitor("=" * 80)
            log_monitor("1. Review this analysis")
            log_monitor("2. Deploy EXPECTED_SIGNAL_RATE fix (0.101 → 0.0612)")
            log_monitor("3. Decide on EXIT_THRESHOLD adjustment based on evidence")
            log_monitor("4. Wait for V4 Bayesian optimization results")
            log_monitor("=" * 80)

            break

        # Trade still OPEN - continue monitoring
        entry_time = trade2.get('entry_time')
        entry_price = trade2.get('entry_price')
        entry_prob = trade2.get('probability')

        holding_minutes = calculate_holding_time(entry_time) if entry_time else None

        # Get current exit probability
        exit_prob = parse_latest_exit_signal()

        # Get dynamic threshold info
        threshold_info = parse_latest_dynamic_threshold()

        log_monitor(f"Status: {status}")
        log_monitor(f"Entry: ${entry_price:,.2f} @ {entry_time}")
        log_monitor(f"Entry Probability: {entry_prob}")
        if holding_minutes:
            log_monitor(f"Holding Time: {holding_minutes:.1f} minutes")

        if exit_prob is not None:
            log_monitor(f"")
            log_monitor(f"Exit Model Probability: {exit_prob:.3f}")
            log_monitor(f"EXIT_THRESHOLD: 0.70")

            if exit_prob >= 0.70:
                log_monitor(f"🚨 EXIT THRESHOLD REACHED! Exit likely on next cycle")
            elif exit_prob >= 0.65:
                log_monitor(f"⚠️ Exit probability approaching threshold (0.65-0.69)")
            elif exit_prob >= 0.50:
                log_monitor(f"📈 Exit probability rising (0.50-0.64)")
            else:
                log_monitor(f"✅ Exit probability low (< 0.50)")

            last_exit_prob = exit_prob

        if threshold_info:
            log_monitor("")
            log_monitor(f"Dynamic Threshold System:")
            if threshold_info.get('signal_rate'):
                log_monitor(f"  Recent Signal Rate: {threshold_info['signal_rate']:.2f}%")
            if threshold_info.get('threshold_long'):
                log_monitor(f"  Current LONG Threshold: {threshold_info['threshold_long']:.3f}")
            if threshold_info.get('adjustment'):
                log_monitor(f"  Adjustment: {threshold_info['adjustment']:+.3f}")

        log_monitor("")
        log_monitor("⏳ Next check in 5 minutes...")
        log_monitor("=" * 80)
        log_monitor("")

        # Wait 5 minutes (bot update cycle)
        time.sleep(300)

if __name__ == "__main__":
    try:
        monitor_trade2()
    except KeyboardInterrupt:
        log_monitor("")
        log_monitor("⚠️ Monitoring stopped by user (Ctrl+C)")
    except Exception as e:
        log_monitor(f"❌ Monitoring error: {e}")
        import traceback
        log_monitor(traceback.format_exc())
