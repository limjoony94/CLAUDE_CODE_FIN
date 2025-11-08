"""
Stop Opportunity Gating Bot
"""
import psutil
import sys
import time

def find_bot_process():
    """Find the bot process"""
    bot_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and any('opportunity_gating_bot_4x.py' in arg for arg in cmdline):
                # Skip nohup wrapper, get actual python process
                if 'nohup' not in proc.info['name'].lower():
                    bot_processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return bot_processes

def main():
    print("🔍 Searching for bot processes...")

    processes = find_bot_process()

    if not processes:
        print("⚠️  Bot not running")
        return

    print(f"✅ Found {len(processes)} bot process(es)")

    for proc in processes:
        print(f"\n🛑 Stopping bot (PID: {proc.pid})...")
        print(f"   Name: {proc.name()}")
        print(f"   Command: {' '.join(proc.cmdline())}")

        try:
            # Graceful termination
            proc.terminate()
            proc.wait(timeout=5)
            print(f"✅ Bot stopped gracefully (PID: {proc.pid})")
        except psutil.TimeoutExpired:
            # Force kill if graceful fails
            print(f"⚠️  Graceful stop timed out, forcing (PID: {proc.pid})...")
            proc.kill()
            proc.wait(timeout=2)
            print(f"✅ Bot force stopped (PID: {proc.pid})")
        except Exception as e:
            print(f"❌ Error stopping bot (PID: {proc.pid}): {e}")
            sys.exit(1)

    # Verify stopped
    time.sleep(1)
    remaining = find_bot_process()
    if remaining:
        print(f"❌ {len(remaining)} bot process(es) still running")
        sys.exit(1)
    else:
        print("\n✅ Verified: All bot processes stopped")

if __name__ == "__main__":
    main()
