#!/usr/bin/env python3
"""
Bot Restart Script
Safely stops all running trading bot instances and starts a new one
"""
import os
import sys
import time
import psutil
import subprocess

def kill_existing_bots():
    """Kill all existing trading bot processes"""
    killed_count = 0

    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and 'phase4_dynamic_testnet_trading.py' in ' '.join(cmdline):
                print(f"Killing process {proc.info['pid']}: {proc.info['name']}")
                proc.kill()
                killed_count += 1
                time.sleep(0.5)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    if killed_count > 0:
        print(f"✅ Killed {killed_count} existing bot process(es)")
        time.sleep(2)  # Wait for processes to fully terminate
    else:
        print("ℹ️ No existing bot processes found")

    return killed_count

def start_bot():
    """Start the trading bot"""
    bot_script = os.path.join(os.path.dirname(__file__), 'production', 'phase4_dynamic_testnet_trading.py')

    if not os.path.exists(bot_script):
        print(f"❌ Bot script not found: {bot_script}")
        return False

    print(f"🚀 Starting bot: {bot_script}")

    # Start bot in background
    if sys.platform == 'win32':
        # Windows: use CREATE_NEW_CONSOLE to run in background
        subprocess.Popen(
            [sys.executable, bot_script],
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
    else:
        # Unix: use nohup
        subprocess.Popen(
            ['nohup', sys.executable, bot_script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    print("✅ Bot started successfully")
    return True

def main():
    print("=" * 80)
    print("Phase 4 Dynamic Testnet Trading Bot - Restart Script")
    print("=" * 80)

    # Step 1: Kill existing bots
    print("\n[Step 1/2] Stopping existing bot processes...")
    kill_existing_bots()

    # Step 2: Start new bot
    print("\n[Step 2/2] Starting new bot...")
    success = start_bot()

    if success:
        print("\n" + "=" * 80)
        print("✅ Bot restart completed successfully!")
        print("=" * 80)
        print("\nTo view logs:")
        print("  tail -f logs/phase4_dynamic_testnet_trading_20251014.log")
        print("\nTo check if bot is running:")
        print("  ps aux | grep phase4_dynamic_testnet_trading.py")
    else:
        print("\n❌ Bot restart failed")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
