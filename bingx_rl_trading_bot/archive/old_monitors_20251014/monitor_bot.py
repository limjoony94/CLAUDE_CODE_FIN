#!/usr/bin/env python3
"""
Real-time Bot Monitoring Dashboard
Shows current bot status, position, and performance
"""
import os
import json
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def load_state():
    """Load current bot state"""
    state_file = RESULTS_DIR / "phase4_testnet_trading_state.json"
    if state_file.exists():
        with open(state_file, 'r') as f:
            return json.load(f)
    return None

def get_latest_log_lines(n=30):
    """Get latest log lines"""
    today = datetime.now().strftime('%Y%m%d')
    log_file = LOGS_DIR / f"phase4_dynamic_testnet_trading_{today}.log"

    if log_file.exists():
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            return lines[-n:]
    return []

def format_time_ago(iso_string):
    """Format time difference"""
    try:
        dt = datetime.fromisoformat(iso_string)
        delta = datetime.now() - dt

        hours = delta.total_seconds() / 3600
        if hours < 1:
            minutes = delta.total_seconds() / 60
            return f"{minutes:.0f}m ago"
        elif hours < 24:
            return f"{hours:.1f}h ago"
        else:
            days = hours / 24
            return f"{days:.1f}d ago"
    except:
        return "Unknown"

def display_dashboard():
    """Display monitoring dashboard"""
    clear_screen()

    print("=" * 80)
    print("🤖 Phase 4 Dynamic Testnet Trading Bot - Live Monitor")
    print("=" * 80)
    print()

    # Load state
    state = load_state()

    if not state:
        print("❌ No state file found. Is bot running?")
        return

    # Basic info
    print("📊 BOT STATUS")
    print("-" * 80)
    print(f"Session Start:    {state.get('session_start', 'Unknown')[:19]}")
    print(f"Last Update:      {state.get('timestamp', 'Unknown')[:19]} ({format_time_ago(state.get('timestamp', ''))})")
    print()

    # Balance
    initial_balance = state.get('initial_balance', 0)
    current_balance = state.get('current_balance', 0)
    pnl = current_balance - initial_balance
    pnl_pct = (pnl / initial_balance * 100) if initial_balance > 0 else 0

    print("💰 BALANCE")
    print("-" * 80)
    print(f"Initial:          ${initial_balance:,.2f} USDT")
    print(f"Current:          ${current_balance:,.2f} USDT")
    print(f"Net P&L:          ${pnl:+,.2f} ({pnl_pct:+.2f}%)")
    print()

    # Trades
    trades = state.get('trades', [])
    trades_count = state.get('trades_count', 0)
    closed_trades = state.get('closed_trades', 0)
    open_trades = trades_count - closed_trades

    print("📈 TRADING ACTIVITY")
    print("-" * 80)
    print(f"Total Trades:     {trades_count}")
    print(f"Open:             {open_trades}")
    print(f"Closed:           {closed_trades}")
    print()

    # Buy & Hold comparison
    bh_quantity = state.get('bh_btc_quantity', 0)
    bh_entry_price = state.get('bh_entry_price', 0)

    if bh_quantity > 0 and bh_entry_price > 0:
        print("🏆 vs BUY & HOLD")
        print("-" * 80)
        print(f"B&H Entry:        ${bh_entry_price:,.2f}")
        print(f"B&H Quantity:     {bh_quantity:.6f} BTC")
        print(f"B&H Value:        ${bh_quantity * bh_entry_price:,.2f}")
        # Note: Current price not in state, would need to fetch from API
        print()

    # Recent logs
    print("📝 RECENT ACTIVITY (Last 10 lines)")
    print("-" * 80)

    log_lines = get_latest_log_lines(10)
    for line in log_lines:
        # Clean up log line
        if '|' in line:
            parts = line.split('|')
            if len(parts) >= 3:
                # Extract timestamp and message
                timestamp = parts[0].strip()
                message = '|'.join(parts[2:]).strip()
                # Truncate if too long
                if len(message) > 70:
                    message = message[:67] + "..."
                print(f"{timestamp} | {message}")

    print()
    print("=" * 80)
    print("Press Ctrl+C to exit | Refreshing every 30 seconds...")
    print("=" * 80)

def main():
    """Main monitoring loop"""
    print("Starting bot monitor...")
    time.sleep(1)

    try:
        while True:
            display_dashboard()
            time.sleep(30)  # Refresh every 30 seconds
    except KeyboardInterrupt:
        print("\n\n👋 Monitor stopped by user")

if __name__ == "__main__":
    main()
