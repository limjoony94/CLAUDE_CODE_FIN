#!/usr/bin/env python3
"""
RSI Trend Filter Bot Monitor
============================

Real-time monitoring dashboard for the RSI Trend Filter Bot.
Displays:
- Current position status
- P&L performance
- RSI and EMA indicators
- Trade statistics
- System health

Usage:
    python scripts/monitoring/rsi_trend_filter_monitor.py
    or
    MONITOR_RSI_TREND_FILTER.bat
"""

import os
import sys
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from colorama import init, Fore, Back, Style

# Initialize colorama for Windows
init()

# Constants
STATE_FILE = project_root / "results" / "rsi_trend_filter_bot_state.json"
CONFIG_FILE = project_root / "config" / "rsi_trend_filter_config.yaml"
LOG_DIR = project_root / "logs"
BOT_NAME = "rsi_trend_filter_bot"

REFRESH_INTERVAL = 5  # seconds


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def load_state() -> dict:
    """Load bot state from JSON file."""
    if not STATE_FILE.exists():
        return None

    try:
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        return None


def format_pnl(pnl: float) -> str:
    """Format P&L with color."""
    if pnl > 0:
        return f"{Fore.GREEN}+{pnl:.2f}%{Style.RESET_ALL}"
    elif pnl < 0:
        return f"{Fore.RED}{pnl:.2f}%{Style.RESET_ALL}"
    else:
        return f"{Fore.YELLOW}{pnl:.2f}%{Style.RESET_ALL}"


def format_direction(direction: str) -> str:
    """Format direction with color."""
    if direction == "LONG":
        return f"{Fore.GREEN}LONG{Style.RESET_ALL}"
    elif direction == "SHORT":
        return f"{Fore.RED}SHORT{Style.RESET_ALL}"
    else:
        return f"{Fore.YELLOW}NONE{Style.RESET_ALL}"


def format_rsi(rsi: float) -> str:
    """Format RSI with color based on value."""
    if rsi is None:
        return f"{Fore.YELLOW}N/A{Style.RESET_ALL}"
    if rsi >= 70:
        return f"{Fore.RED}{rsi:.1f}{Style.RESET_ALL}"  # Overbought
    elif rsi <= 30:
        return f"{Fore.GREEN}{rsi:.1f}{Style.RESET_ALL}"  # Oversold
    else:
        return f"{Fore.WHITE}{rsi:.1f}{Style.RESET_ALL}"  # Neutral


def get_bot_uptime(state: dict) -> str:
    """Calculate bot uptime from state."""
    if not state.get('created_at'):
        return "Unknown"

    try:
        created = datetime.fromisoformat(state['created_at'])
        uptime = datetime.now() - created
        days = uptime.days
        hours = uptime.seconds // 3600
        minutes = (uptime.seconds % 3600) // 60
        return f"{days}d {hours}h {minutes}m"
    except:
        return "Unknown"


def get_last_update(state: dict) -> str:
    """Get time since last update."""
    if not state.get('updated_at'):
        return "Unknown"

    try:
        updated = datetime.fromisoformat(state['updated_at'])
        diff = datetime.now() - updated
        seconds = diff.total_seconds()

        if seconds < 60:
            return f"{int(seconds)}s ago"
        elif seconds < 3600:
            return f"{int(seconds // 60)}m ago"
        else:
            return f"{int(seconds // 3600)}h ago"
    except:
        return "Unknown"


def display_header():
    """Display monitor header."""
    print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}   RSI Trend Filter Bot Monitor v1.0{Style.RESET_ALL}")
    print(f"{Fore.CYAN}   Strategy: RSI(14) 40/60 + EMA100{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")


def display_status(state: dict):
    """Display bot status section."""
    print(f"\n{Fore.YELLOW}== BOT STATUS =={Style.RESET_ALL}")

    updated = get_last_update(state)
    uptime = get_bot_uptime(state)

    # Check if bot is running (updated within last 2 minutes)
    try:
        updated_time = datetime.fromisoformat(state.get('updated_at', ''))
        is_running = (datetime.now() - updated_time).total_seconds() < 120
        status = f"{Fore.GREEN}RUNNING{Style.RESET_ALL}" if is_running else f"{Fore.RED}STOPPED{Style.RESET_ALL}"
    except:
        status = f"{Fore.YELLOW}UNKNOWN{Style.RESET_ALL}"

    print(f"  Status: {status}")
    print(f"  Last Update: {updated}")
    print(f"  Uptime: {uptime}")


def display_position(state: dict):
    """Display current position section."""
    print(f"\n{Fore.YELLOW}== POSITION =={Style.RESET_ALL}")

    position = state.get('position')

    if position:
        direction = format_direction(position.get('direction'))
        entry_price = position.get('entry_price', 0)
        tp_price = position.get('tp_price', 0)
        sl_price = position.get('sl_price', 0)
        quantity = position.get('quantity', 0)
        entry_time = position.get('entry_time', 'Unknown')
        reason = position.get('reason', 'Unknown')

        # Calculate current P&L (estimated)
        print(f"  Direction: {direction}")
        print(f"  Entry: ${entry_price:.1f}")
        print(f"  Quantity: {quantity:.4f}")
        print(f"  TP: ${tp_price:.1f} ({Fore.GREEN}+3.0%{Style.RESET_ALL})")
        print(f"  SL: ${sl_price:.1f} ({Fore.RED}-2.0%{Style.RESET_ALL})")
        print(f"  Entry Time: {entry_time[:19] if len(entry_time) > 19 else entry_time}")
        print(f"  Reason: {reason[:50]}...")
    else:
        print(f"  {Fore.WHITE}No open position{Style.RESET_ALL}")


def display_indicators(state: dict):
    """Display current indicators section."""
    print(f"\n{Fore.YELLOW}== INDICATORS =={Style.RESET_ALL}")

    rsi = state.get('last_rsi')
    rsi_display = format_rsi(rsi)

    print(f"  RSI(14): {rsi_display}")
    print(f"  Long Threshold: 40 (RSI crosses above)")
    print(f"  Short Threshold: 60 (RSI crosses below)")
    print(f"  Trend Filter: EMA(100)")


def display_performance(state: dict):
    """Display performance statistics section."""
    print(f"\n{Fore.YELLOW}== PERFORMANCE =={Style.RESET_ALL}")

    total_trades = state.get('total_trades', 0)
    total_pnl = state.get('total_pnl', 0)
    winning_trades = state.get('winning_trades', 0)
    daily_trades = state.get('daily_trades', 0)
    daily_pnl = state.get('daily_pnl', 0)

    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

    print(f"  Total Trades: {total_trades}")
    print(f"  Win Rate: {win_rate:.1f}%")
    print(f"  Total P&L: {format_pnl(total_pnl)}")
    print(f"  Daily Trades: {daily_trades}")
    print(f"  Daily P&L: {format_pnl(daily_pnl)}")


def display_last_trade(state: dict):
    """Display last trade information."""
    print(f"\n{Fore.YELLOW}== LAST TRADE =={Style.RESET_ALL}")

    last_trade = state.get('last_trade')

    if last_trade:
        direction = format_direction(last_trade.get('direction'))
        entry = last_trade.get('entry_price', 0)
        exit_price = last_trade.get('exit_price', 0)
        pnl = last_trade.get('pnl_pct', 0)
        exit_reason = last_trade.get('exit_reason', 'Unknown')
        closed_at = last_trade.get('closed_at', 'Unknown')

        print(f"  Direction: {direction}")
        print(f"  Entry: ${entry:.1f}")
        print(f"  Exit: ${exit_price:.1f}")
        print(f"  P&L: {format_pnl(pnl)}")
        print(f"  Exit Reason: {exit_reason}")
        print(f"  Closed: {closed_at[:19] if len(str(closed_at)) > 19 else closed_at}")
    else:
        print(f"  {Fore.WHITE}No trades yet{Style.RESET_ALL}")


def display_footer():
    """Display footer with refresh info."""
    print(f"\n{Fore.CYAN}{'─'*60}{Style.RESET_ALL}")
    print(f"  Refreshing every {REFRESH_INTERVAL}s | Press Ctrl+C to exit")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    """Main monitor loop."""
    try:
        while True:
            clear_screen()
            display_header()

            state = load_state()

            if state:
                display_status(state)
                display_position(state)
                display_indicators(state)
                display_performance(state)
                display_last_trade(state)
            else:
                print(f"\n{Fore.RED}  [!] Could not load bot state{Style.RESET_ALL}")
                print(f"  State file: {STATE_FILE}")
                print(f"\n  Make sure the bot is running:")
                print(f"    python scripts/production/rsi_trend_filter_bot.py")

            display_footer()

            time.sleep(REFRESH_INTERVAL)

    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Monitor stopped.{Style.RESET_ALL}")


if __name__ == '__main__':
    main()
