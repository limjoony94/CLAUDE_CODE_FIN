#!/usr/bin/env python3
"""
Force kill all opportunity_gating_bot_4x.py processes
"""
import psutil
import sys

def kill_bot_processes():
    """Find and kill all bot processes"""
    killed = []

    print("Searching for bot processes...")
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and isinstance(cmdline, list):
                cmdline_str = ' '.join(cmdline)
                if 'opportunity_gating_bot_4x.py' in cmdline_str:
                    pid = proc.info['pid']
                    print(f"Found bot process: PID {pid}")
                    proc.kill()
                    proc.wait(timeout=5)
                    killed.append(pid)
                    print(f"  ✅ Killed PID {pid}")
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired) as e:
            pass

    if killed:
        print(f"\n✅ Successfully killed {len(killed)} bot processes: {killed}")
        return True
    else:
        print("\n❌ No bot processes found")
        return False

if __name__ == "__main__":
    success = kill_bot_processes()
    sys.exit(0 if success else 1)
