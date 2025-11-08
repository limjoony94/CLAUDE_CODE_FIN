"""
Real-time Bot Monitoring Window

Displays live bot status, positions, and performance metrics.
Updates every 10 seconds.
"""

import json
import time
import os
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
STATE_FILE = PROJECT_ROOT / "results" / "phase4_testnet_trading_state.json"
LOG_FILE = PROJECT_ROOT / "logs" / f"phase4_dynamic_testnet_trading_{datetime.now().strftime('%Y%m%d')}.log"

def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def load_state():
    """Load bot state from JSON"""
    if not STATE_FILE.exists():
        return None

    with open(STATE_FILE, 'r') as f:
        return json.load(f)

def format_duration(seconds):
    """Format duration in human-readable format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours}h {minutes}m"

def display_monitoring(state):
    """Display formatted monitoring dashboard"""
    clear_screen()

    # Header
    print("=" * 80)
    print("📊 BINGX DUAL MODEL BOT - LIVE MONITORING")
    print("=" * 80)
    print(f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"State File: {state.get('timestamp', 'N/A')}")
    print("")

    # Bot Status
    initial_balance = state.get('initial_balance', 0)
    current_balance = state.get('current_balance', 0)
    pnl = current_balance - initial_balance
    pnl_pct = (pnl / initial_balance * 100) if initial_balance > 0 else 0

    print("🤖 BOT STATUS")
    print("-" * 80)
    session_start = datetime.fromisoformat(state.get('session_start', datetime.now().isoformat()))
    runtime = (datetime.now() - session_start).total_seconds()
    print(f"Runtime: {format_duration(runtime)} (since {session_start.strftime('%Y-%m-%d %H:%M')})")
    print(f"Initial Balance: ${initial_balance:,.2f} USDT")
    print(f"Current Balance: ${current_balance:,.2f} USDT")

    pnl_emoji = "📈" if pnl >= 0 else "📉"
    print(f"Session P&L: {pnl_emoji} ${pnl:+,.2f} ({pnl_pct:+.2f}%)")
    print("")

    # Positions
    trades = state.get('trades', [])
    open_trades = [t for t in trades if t.get('status') == 'OPEN']
    closed_trades = [t for t in trades if t.get('status') == 'CLOSED']

    print("📈 OPEN POSITION")
    print("-" * 80)

    if len(open_trades) > 0:
        for trade in open_trades:
            side = trade.get('side', 'LONG')
            entry_price = trade.get('entry_price', 0)
            quantity = trade.get('quantity', 0)
            entry_time = datetime.fromisoformat(trade.get('entry_time'))
            hours_held = (datetime.now() - entry_time).total_seconds() / 3600
            probability = trade.get('probability', 0)

            # Estimate current price (would need API call for real price)
            # For now, show entry info
            position_value = entry_price * quantity

            side_emoji = "🟢" if side == "LONG" else "🔴"
            print(f"{side_emoji} {side} Position")
            print(f"   Entry: ${entry_price:,.2f}")
            print(f"   Quantity: {quantity:.4f} BTC")
            print(f"   Value: ${position_value:,.2f} USDT")
            print(f"   Holding: {hours_held:.1f} hours (Max: 4.0h)")
            print(f"   Signal Confidence: {probability:.1%}")
            print(f"   Position Size: {trade.get('position_size_pct', 0)*100:.1f}% of capital")
            print(f"   Regime: {trade.get('regime', 'Unknown')}")

            # Exit conditions
            print(f"   Exit Triggers:")
            print(f"      • Stop Loss: -1.0% from entry")
            print(f"      • Take Profit: +3.0% from entry")
            print(f"      • Max Holding: {4.0 - hours_held:.1f} hours remaining")
    else:
        print("No open positions")
        print("   Waiting for entry signal...")
        print(f"   Signal threshold: LONG >= 0.7 OR SHORT >= 0.7")

    print("")

    # Trade Statistics
    print("📊 TRADE STATISTICS")
    print("-" * 80)

    total_trades = len(trades)
    print(f"Total Trades: {total_trades}")

    if len(closed_trades) > 0:
        winning = len([t for t in closed_trades if t.get('pnl_usd_net', 0) > 0])
        win_rate = (winning / len(closed_trades)) * 100
        total_pnl = sum(t.get('pnl_usd_net', 0) for t in closed_trades)

        # LONG vs SHORT breakdown
        long_trades = [t for t in trades if t.get('side') == 'LONG']
        short_trades = [t for t in trades if t.get('side') == 'SHORT']

        print(f"Closed Trades: {len(closed_trades)}")
        print(f"   Winning: {winning} ({win_rate:.1f}%)")
        print(f"   Total P&L: ${total_pnl:+,.2f}")
        print(f"")
        print(f"Trade Direction:")
        print(f"   LONG: {len(long_trades)} trades ({len(long_trades)/total_trades*100:.1f}%)")
        print(f"   SHORT: {len(short_trades)} trades ({len(short_trades)/total_trades*100:.1f}%)")
    else:
        print("No completed trades yet")

    print("")

    # Expected Performance
    print("🎯 EXPECTED PERFORMANCE (From Backtest)")
    print("-" * 80)
    print("Dual Model (LONG + SHORT):")
    print("   Return: +14.98% per 5 days")
    print("   Win Rate: 66.2%")
    print("   LONG Ratio: 87.6% (주력)")
    print("   SHORT Ratio: 12.4% (하락장 보완)")
    print("   Improvement: +2.31%p vs LONG-only")

    print("")
    print("=" * 80)
    print("Press Ctrl+C to exit monitoring")
    print("Refreshing in 10 seconds...")

def main():
    """Main monitoring loop"""
    print("Starting bot monitoring...")
    print("Loading state...")

    try:
        while True:
            state = load_state()

            if state is None:
                clear_screen()
                print("=" * 80)
                print("❌ Bot state file not found!")
                print("=" * 80)
                print(f"Looking for: {STATE_FILE}")
                print("")
                print("Is the bot running?")
                print("Retrying in 10 seconds...")
                time.sleep(10)
                continue

            display_monitoring(state)
            time.sleep(10)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")

if __name__ == "__main__":
    main()
