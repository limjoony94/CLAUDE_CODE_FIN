#!/usr/bin/env python3
"""
Simple Real-time Monitor for Opportunity Gating Bot 4x
Displays key metrics including initial balance, current balance, and trading stats
"""

import json
import time
import os
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATE_FILE = PROJECT_ROOT / "results" / "opportunity_gating_bot_4x_state.json"

def load_state():
    """Load current state from file"""
    if not STATE_FILE.exists():
        return None

    with open(STATE_FILE, 'r') as f:
        return json.load(f)

def format_time_ago(iso_time_str):
    """Convert ISO timestamp to time ago"""
    try:
        dt = datetime.fromisoformat(iso_time_str.replace('Z', '+00:00'))
        now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
        delta = now - dt

        days = delta.days
        hours = delta.seconds // 3600
        minutes = (delta.seconds % 3600) // 60

        if days > 0:
            return f"{days}d {hours}h ago"
        elif hours > 0:
            return f"{hours}h {minutes}m ago"
        else:
            return f"{minutes}m ago"
    except:
        return "unknown"

def display_status(state):
    """Display formatted status"""
    os.system('cls' if os.name == 'nt' else 'clear')

    print("=" * 80)
    print("🤖 OPPORTUNITY GATING BOT 4X - REAL-TIME MONITOR")
    print("=" * 80)

    # Session Info
    session_start = state.get('session_start', 'N/A')
    if session_start != 'N/A':
        session_ago = format_time_ago(session_start)
        print(f"📅 Session Start:     {session_start[:19]} ({session_ago})")
    else:
        print(f"📅 Session Start:     N/A")

    # Balance Info
    initial = state.get('initial_balance', 0)
    current = state.get('current_balance', 0)
    realized = state.get('realized_balance', 0)
    unrealized_pnl = state.get('unrealized_pnl', 0)
    net_balance = state.get('net_balance', 0)

    total_return = ((current - initial) / initial * 100) if initial > 0 else 0

    print(f"\n💰 Balance:")
    print(f"   Initial:          ${initial:,.2f}")
    print(f"   Current:          ${current:,.2f}")
    print(f"   Realized:         ${realized:,.2f}")
    print(f"   Net (with unreal): ${net_balance:,.2f}")
    print(f"   Total Return:     {total_return:+.2f}% (${current - initial:+,.2f})")

    # Position Info
    position = state.get('position', {})
    if position is None:
        position = {}
    pos_status = position.get('status', 'NONE')

    print(f"\n📊 Position:")
    if pos_status == 'OPEN':
        side = position.get('side', 'UNKNOWN')
        entry_price = position.get('entry_price', 0)
        quantity = position.get('quantity', 0)
        position_size_pct = position.get('position_size_pct', 0)
        leveraged_value = position.get('leveraged_value', 0)
        entry_time = position.get('entry_time', 'N/A')

        if entry_time != 'N/A':
            entry_ago = format_time_ago(entry_time)
            print(f"   Status:           {side} OPEN ({entry_ago})")
        else:
            print(f"   Status:           {side} OPEN")

        print(f"   Entry Price:      ${entry_price:,.2f}")
        print(f"   Quantity:         {quantity:.6f} BTC")
        print(f"   Position Size:    {position_size_pct*100:.2f}%")
        print(f"   Leveraged Value:  ${leveraged_value:,.2f}")
        print(f"   Unrealized P&L:   ${unrealized_pnl:+,.2f}")
    else:
        print(f"   Status:           No open position")

    # Trading Stats
    stats = state.get('stats', {})
    total_trades = stats.get('total_trades', 0)
    long_trades = stats.get('long_trades', 0)
    short_trades = stats.get('short_trades', 0)
    wins = stats.get('wins', 0)
    losses = stats.get('losses', 0)
    total_pnl = stats.get('total_pnl_usd', 0)

    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    avg_pnl = total_pnl / total_trades if total_trades > 0 else 0

    print(f"\n📈 Trading Stats:")
    print(f"   Total Trades:     {total_trades} ({long_trades}L / {short_trades}S)")
    print(f"   Win / Loss:       {wins}W / {losses}L")
    print(f"   Win Rate:         {win_rate:.1f}%")
    print(f"   Total P&L:        ${total_pnl:+,.2f}")
    print(f"   Avg P&L:          ${avg_pnl:+,.2f} per trade")

    # Latest Signals
    latest_signals = state.get('latest_signals', {})
    entry_signals = latest_signals.get('entry', {})
    exit_signals = latest_signals.get('exit', {})

    print(f"\n🎯 Latest Signals:")
    if entry_signals:
        long_prob = entry_signals.get('long_prob', 0)
        short_prob = entry_signals.get('short_prob', 0)
        long_thresh = entry_signals.get('long_threshold', 0.7)
        short_thresh = entry_signals.get('short_threshold', 0.7)

        long_signal = "✅" if long_prob >= long_thresh else "❌"
        short_signal = "✅" if short_prob >= short_thresh else "❌"

        print(f"   LONG:             {long_prob:.4f} (thresh: {long_thresh}) {long_signal}")
        print(f"   SHORT:            {short_prob:.4f} (thresh: {short_thresh}) {short_signal}")

    if exit_signals and pos_status == 'OPEN':
        exit_prob = exit_signals.get('exit_prob', 0)
        exit_thresh = exit_signals.get('exit_threshold_current', 0.8)
        exit_signal = "✅ EXIT" if exit_prob >= exit_thresh else "⏳ Hold"

        print(f"   EXIT:             {exit_prob:.4f} (thresh: {exit_thresh}) {exit_signal}")

    # Configuration
    config = state.get('configuration', {})
    print(f"\n⚙️  Configuration:")
    print(f"   Leverage:         {config.get('leverage', 4)}x")
    print(f"   Entry LONG:       {config.get('long_threshold', 0.7):.2f}")
    print(f"   Entry SHORT:      {config.get('short_threshold', 0.7):.2f}")
    print(f"   ML Exit:          {config.get('ml_exit_threshold_base_long', 0.8):.2f}")
    print(f"   Stop Loss:        {config.get('emergency_stop_loss', 0.03)*100:.0f}%")
    print(f"   Max Hold:         {config.get('emergency_max_hold_hours', 10):.0f}h")

    # Last Update
    timestamp = state.get('timestamp', 'N/A')
    if timestamp != 'N/A':
        update_ago = format_time_ago(timestamp)
        print(f"\n⏰ Last Update:       {timestamp[:19]} ({update_ago})")

    print("=" * 80)
    print("🔄 Refreshing every 10 seconds... (Ctrl+C to stop)")

def main():
    """Main monitoring loop"""
    try:
        while True:
            state = load_state()
            if state:
                display_status(state)
            else:
                print("⚠️  State file not found. Waiting...")

            time.sleep(10)  # Refresh every 10 seconds
    except KeyboardInterrupt:
        print("\n\n✅ Monitor stopped by user")

if __name__ == "__main__":
    main()
