#!/usr/bin/env python
"""
Unified Monitor - All-in-One Bot Monitoring Dashboard
Shows: Performance, Signals, Positions, and Latest Activity in ONE window
"""

import os
import time
from datetime import datetime
from pathlib import Path

def clear_screen():
    """Clear console screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def get_latest_log_file():
    """Get the most recent log file"""
    log_dir = Path("logs")
    log_files = sorted(log_dir.glob("phase4_dynamic_testnet_trading_*.log"),
                      key=lambda x: x.stat().st_mtime, reverse=True)
    return log_files[0] if log_files else None

def read_log_lines(log_file, pattern, last_n=1):
    """Read lines matching pattern from log file"""
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if pattern in line]
            return lines[-last_n:] if lines else []
    except:
        return []

def display_monitor():
    """Display unified monitoring dashboard"""
    clear_screen()

    log_file = get_latest_log_file()
    if not log_file:
        print("❌ Error: Log file not found. Bot may not be running.")
        return False

    print("=" * 100)
    print(" " * 35 + "ML EXIT BOT - UNIFIED MONITOR")
    print("=" * 100)
    print(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log: {log_file.name}")
    print()

    # Section 1: Account & Performance
    print("\n[1] ACCOUNT & PERFORMANCE")
    print("-" * 100)

    initial_balance = read_log_lines(log_file, "Initial Balance", 1)
    if initial_balance:
        print(f"   {initial_balance[0].split('|')[-1].strip()}")

    current_balance = read_log_lines(log_file, "Account Balance", 1)
    if current_balance:
        print(f"   {current_balance[0].split('|')[-1].strip()}")

    session_pnl = read_log_lines(log_file, "Session P&L", 1)
    if session_pnl:
        print(f"   {session_pnl[0].split('|')[-1].strip()}")

    closed_count = len(read_log_lines(log_file, "POSITION CLOSED", 999))
    print(f"   Total Trades: {closed_count if closed_count > 0 else 'No trades completed yet'}")

    # Section 2: Signal Monitoring
    print("\n[2] SIGNAL MONITORING (Latest)")
    print("-" * 100)

    long_prob = read_log_lines(log_file, "LONG Model Prob", 1)
    if long_prob:
        prob_value = long_prob[0].split(":")[-1].strip()
        print(f"   LONG Model Probability: {prob_value}")

    short_prob = read_log_lines(log_file, "SHORT Model Prob", 1)
    if short_prob:
        prob_value = short_prob[0].split(":")[-1].strip()
        print(f"   SHORT Model Probability: {prob_value}")

    threshold = read_log_lines(log_file, "Threshold:", 1)
    if threshold:
        thresh_value = threshold[0].split("Threshold:")[-1].strip()
        print(f"   Entry Threshold: {thresh_value}")

    should_enter = read_log_lines(log_file, "Should Enter", 1)
    if should_enter:
        enter_value = should_enter[0].split("Should Enter:")[-1].strip()
        print(f"   ➜ {enter_value}")

    # Section 3: Position Status
    print("\n[3] POSITION STATUS")
    print("-" * 100)

    position = read_log_lines(log_file, "Position:", 1)
    if position and "BTC @" in position[0]:
        print(f"   {position[0].split('|')[-1].strip()}")
    else:
        print("   No open position")

    pnl = read_log_lines(log_file, "P&L:", 1)
    if pnl and "%" in pnl[0]:
        print(f"   {pnl[0].split('|')[-1].strip()}")

    # Section 4: Recent Activity
    print("\n[4] RECENT ACTIVITY (Last 5 log entries)")
    print("-" * 100)

    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
            recent = all_lines[-5:]
            for line in recent:
                if "|" in line:
                    parts = line.split("|")
                    if len(parts) >= 3:
                        time_part = parts[0].strip()
                        level = parts[1].strip()
                        msg = parts[-1].strip()[:80]  # Truncate long messages
                        print(f"   [{time_part[:19]}] {msg}")
    except:
        print("   Unable to read recent activity")

    # Section 5: Status
    print("\n[5] STATUS")
    print("-" * 100)

    next_update = read_log_lines(log_file, "Next update", 1)
    if next_update:
        print(f"   {next_update[0].split('|')[-1].strip()}")

    market_regime = read_log_lines(log_file, "Market Regime", 1)
    if market_regime:
        print(f"   {market_regime[0].split('|')[-1].strip()}")

    current_price = read_log_lines(log_file, "Current Price", 1)
    if current_price:
        print(f"   {current_price[0].split('|')[-1].strip()}")

    print("\n" + "=" * 100)
    print("Press Ctrl+C to exit | Auto-refresh every 30 seconds")
    print("=" * 100)

    return True

def main():
    """Main monitoring loop"""
    print("Starting ML Exit Bot Unified Monitor...")
    print("This window will auto-refresh every 30 seconds.")
    print()

    try:
        while True:
            if not display_monitor():
                print("\nWaiting 5 seconds before retry...")
                time.sleep(5)
            else:
                time.sleep(30)  # Refresh every 30 seconds
    except KeyboardInterrupt:
        print("\n\nMonitor stopped by user. Goodbye!")

if __name__ == "__main__":
    # Change to bot directory
    bot_dir = Path(__file__).parent
    os.chdir(bot_dir)

    main()
