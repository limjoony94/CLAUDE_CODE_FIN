#!/usr/bin/env python3
"""
Safe Trading History Reset - 올바른 순서로 리셋 수행
Safe reset that stops bot, resets state, then allows restart
"""

import json
import os
import sys
import time
import signal
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATE_FILE = PROJECT_ROOT / "results" / "opportunity_gating_bot_4x_state.json"
LOCK_FILE = PROJECT_ROOT / "opportunity_gating_bot_4x.lock"

def find_bot_processes():
    """Find running bot processes"""
    import psutil
    bot_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline', [])
            if cmdline and any('opportunity_gating_bot_4x.py' in str(arg) for arg in cmdline):
                bot_processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return bot_processes

def stop_bot():
    """Stop running bot process"""
    print("\n" + "="*60)
    print("🛑 STEP 1: Stopping Bot")
    print("="*60)

    # Check lock file
    if LOCK_FILE.exists():
        try:
            with open(LOCK_FILE, 'r') as f:
                pid = int(f.read().strip())
            print(f"   Lock file found: PID {pid}")

            try:
                import psutil
                proc = psutil.Process(pid)
                print(f"   Terminating process {pid}...")
                proc.terminate()
                proc.wait(timeout=5)
                print(f"   ✅ Bot stopped (PID {pid})")
            except psutil.NoSuchProcess:
                print(f"   ⚠️  Process {pid} not found (already stopped)")
            except Exception as e:
                print(f"   ⚠️  Error stopping process: {e}")

            # Remove lock file
            LOCK_FILE.unlink()
            print(f"   ✅ Lock file removed")

        except Exception as e:
            print(f"   ⚠️  Error reading lock file: {e}")
    else:
        print("   No lock file found")

    # Check for any remaining bot processes
    try:
        import psutil
        bot_procs = find_bot_processes()
        if bot_procs:
            print(f"\n   ⚠️  Found {len(bot_procs)} bot process(es) still running:")
            for proc in bot_procs:
                print(f"      PID {proc.pid}: {' '.join(proc.cmdline())}")
                print(f"      Terminating...")
                proc.terminate()

            # Wait for processes to stop
            gone, alive = psutil.wait_procs(bot_procs, timeout=5)
            if alive:
                print(f"   ⚠️  Force killing {len(alive)} process(es)...")
                for proc in alive:
                    proc.kill()

            print(f"   ✅ All bot processes stopped")
    except ImportError:
        print("   ⚠️  psutil not available, skipping process check")

    # Wait a moment for file handles to release
    print("   Waiting 2 seconds for file handles to release...")
    time.sleep(2)
    print("   ✅ Bot stopped successfully\n")

def reset_state():
    """Reset trading history"""
    print("="*60)
    print("🔄 STEP 2: Resetting State")
    print("="*60)

    # Load current state
    with open(STATE_FILE, 'r') as f:
        state = json.load(f)

    current_balance = state.get('current_balance', 0)
    position = state.get('position')
    if position is None:
        position = {}

    print(f"   Current Balance: ${current_balance:,.2f}")
    print(f"   Position: {position.get('status', 'NONE')}")

    # Backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = STATE_FILE.parent / f"opportunity_gating_bot_4x_state_backup_{timestamp}.json"
    with open(backup_file, 'w') as f:
        json.dump(state, f, indent=2)
    print(f"   ✅ Backup: {backup_file.name}")

    # Reset trading history
    state['initial_balance'] = current_balance
    state['trades'] = []
    state['closed_trades'] = 0
    state['ledger'] = []
    state['stats'] = {
        'total_trades': 0,
        'long_trades': 0,
        'short_trades': 0,
        'wins': 0,
        'losses': 0,
        'total_pnl_usd': 0,
        'total_pnl_pct': 0
    }

    # Add reset event to reconciliation log
    reset_event = {
        'timestamp': datetime.now().isoformat(),
        'event': 'trading_history_reset',
        'reason': 'Safe manual reset - Bot stopped, history cleared',
        'balance': current_balance,
        'previous_balance': current_balance,
        'notes': f'Safe reset on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}. Bot was stopped, history cleared, position maintained.'
    }

    if 'reconciliation_log' not in state:
        state['reconciliation_log'] = []
    state['reconciliation_log'].append(reset_event)

    # Save state
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

    print(f"\n   ✅ Reset Complete:")
    print(f"      Initial Balance: ${current_balance:,.2f}")
    print(f"      Trades: 0")
    print(f"      Stats: Reset to 0")
    print(f"      Position: {position.get('status', 'NONE')} (maintained)")
    print(f"      Reset event added to reconciliation_log\n")

def verify_reset():
    """Verify reset was successful"""
    print("="*60)
    print("✅ STEP 3: Verifying Reset")
    print("="*60)

    with open(STATE_FILE, 'r') as f:
        state = json.load(f)

    initial = state.get('initial_balance', 0)
    current = state.get('current_balance', 0)
    trades = len(state.get('trades', []))
    closed = state.get('closed_trades', 0)

    # Check for reset event
    reconciliation_log = state.get('reconciliation_log', [])
    reset_found = False
    for entry in reversed(reconciliation_log):
        if entry.get('event') == 'trading_history_reset':
            reset_time = entry.get('timestamp', '')
            reset_found = True
            break

    print(f"   Initial Balance: ${initial:,.2f}")
    print(f"   Current Balance: ${current:,.2f}")
    print(f"   Trades: {trades}")
    print(f"   Closed: {closed}")
    print(f"   Reset Event: {'✅ Found' if reset_found else '❌ Missing'}")

    if trades == 0 and closed == 0 and reset_found:
        print(f"\n   ✅ Reset Verified Successfully!")
        print(f"\n📌 Next Steps:")
        print(f"   1. Bot is stopped")
        print(f"   2. State is reset and clean")
        print(f"   3. You can now restart the bot:")
        print(f"      cd bingx_rl_trading_bot")
        print(f"      python scripts/production/opportunity_gating_bot_4x.py")
        return True
    else:
        print(f"\n   ❌ Reset Verification Failed!")
        print(f"      Trades should be 0, got {trades}")
        print(f"      Closed should be 0, got {closed}")
        return False

def main():
    """Main reset flow"""
    print("\n" + "="*60)
    print("🛡️ SAFE TRADING HISTORY RESET")
    print("="*60)
    print("This will:")
    print("  1. Stop the bot")
    print("  2. Reset trading history (keep positions)")
    print("  3. Verify reset was successful")
    print("="*60)

    response = input("\nProceed with safe reset? (yes/no): ").strip().lower()
    if response != 'yes':
        print("❌ Reset cancelled")
        return

    try:
        # Step 1: Stop bot
        stop_bot()

        # Step 2: Reset state
        reset_state()

        # Step 3: Verify
        success = verify_reset()

        if success:
            print("\n" + "="*60)
            print("✅ SAFE RESET COMPLETE")
            print("="*60)
            return 0
        else:
            print("\n" + "="*60)
            print("❌ RESET FAILED - CHECK STATE FILE")
            print("="*60)
            return 1

    except Exception as e:
        print(f"\n❌ Error during reset: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
