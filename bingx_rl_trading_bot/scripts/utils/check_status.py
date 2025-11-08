#!/usr/bin/env python3
"""
Quick Status Check for Opportunity Gating Bot 4x
Single execution to check current status
"""

import json
import os
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATE_FILE = PROJECT_ROOT / "results" / "opportunity_gating_bot_4x_state.json"

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

def main():
    """Display current status once"""
    if not STATE_FILE.exists():
        print("⚠️  State file not found")
        return

    with open(STATE_FILE, 'r') as f:
        state = json.load(f)

    print("=" * 80)
    print("🤖 OPPORTUNITY GATING BOT 4X - STATUS CHECK")
    print("=" * 80)

    # Session Info
    session_start = state.get('session_start', 'N/A')
    if session_start != 'N/A':
        session_ago = format_time_ago(session_start)
        print(f"📅 Session Start:     {session_start[:19]} ({session_ago})")

    # Balance Info
    initial = state.get('initial_balance', 0)
    current = state.get('current_balance', 0)
    total_return = ((current - initial) / initial * 100) if initial > 0 else 0

    print(f"\n💰 Balance:")
    print(f"   Initial:          ${initial:,.2f}")
    print(f"   Current:          ${current:,.2f}")
    print(f"   Total Return:     {total_return:+.2f}% (${current - initial:+,.2f})")

    # Position Info
    position = state.get('position')
    if position is None:
        print(f"\n📊 Position:          No open position")
    elif position.get('status') == 'OPEN':
        side = position.get('side', 'UNKNOWN')
        entry_price = position.get('entry_price', 0)
        unrealized_pnl = state.get('unrealized_pnl', 0)
        print(f"\n📊 Position:          {side} OPEN")
        print(f"   Entry Price:      ${entry_price:,.2f}")
        print(f"   Unrealized P&L:   ${unrealized_pnl:+,.2f}")

    # Trading Stats
    stats = state.get('stats', {})
    total_trades = stats.get('total_trades', 0)
    wins = stats.get('wins', 0)
    losses = stats.get('losses', 0)
    total_pnl = stats.get('total_pnl_usd', 0)
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

    print(f"\n📈 Trading Stats:")
    print(f"   Total Trades:     {total_trades}")
    print(f"   Win / Loss:       {wins}W / {losses}L")
    print(f"   Win Rate:         {win_rate:.1f}%")
    print(f"   Total P&L:        ${total_pnl:+,.2f}")

    # Latest Signals
    latest_signals = state.get('latest_signals', {})
    entry_signals = latest_signals.get('entry', {})
    if entry_signals:
        long_prob = entry_signals.get('long_prob', 0)
        short_prob = entry_signals.get('short_prob', 0)
        print(f"\n🎯 Latest Signals:")
        print(f"   LONG:             {long_prob:.4f}")
        print(f"   SHORT:            {short_prob:.4f}")

    print("=" * 80)

if __name__ == "__main__":
    main()
