#!/usr/bin/env python3
"""Quick cascade health check — run after v1.67.1 to verify fix."""
import os, glob, re, json
from datetime import datetime
from collections import defaultdict

LOG_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'logs')
STATE_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'pattern_5m_bot_state.json')


def check():
    logs = sorted(glob.glob(os.path.join(LOG_DIR, 'pattern_5m_bot_*.log')))
    if not logs:
        print("No log files found.")
        return

    cascade_triggers = 0
    cascade_skips = 0
    cascade_success = 0
    pre_emptive = 0
    sl_exits = 0
    tp_exits = 0

    for logf in logs[-3:]:  # last 3 log files
        with open(logf, encoding='utf-8', errors='ignore') as f:
            for line in f:
                if '🔗 CASCADE SL:' in line:
                    cascade_triggers += 1
                elif 'CASCADE slot' in line and 'SKIP' in line:
                    cascade_skips += 1
                elif 'CASCADE slot' in line and 'SL $' in line and '→' in line:
                    cascade_success += 1
                elif '🔗 PRE-EMPTIVE CASCADE:' in line:
                    pre_emptive += 1
                elif 'TRADE CLOSED' in line and 'SL' in line:
                    sl_exits += 1
                elif 'TRADE CLOSED' in line and 'TP' in line:
                    tp_exits += 1

    total_targeted = cascade_skips + cascade_success
    skip_rate = cascade_skips / max(total_targeted, 1) * 100

    print("=" * 50)
    print("CASCADE HEALTH CHECK")
    print("=" * 50)
    print(f"  Cascade triggers:     {cascade_triggers}")
    print(f"  Positions targeted:   {total_targeted}")
    print(f"  Successfully updated: {cascade_success}")
    print(f"  SKIPped:              {cascade_skips} ({skip_rate:.0f}%)")
    print(f"  Pre-emptive cascade:  {pre_emptive}")
    print(f"  SL exits:             {sl_exits}")
    print(f"  TP exits:             {tp_exits}")

    if skip_rate > 50:
        print(f"\n  🔴 CASCADE SKIP rate {skip_rate:.0f}% > 50% — CHECK FIX")
    elif skip_rate > 20:
        print(f"\n  🟡 CASCADE SKIP rate {skip_rate:.0f}% — MONITOR")
    else:
        print(f"\n  🟢 CASCADE SKIP rate {skip_rate:.0f}% — HEALTHY")

    # Check current positions for _sl_price_original
    try:
        with open(STATE_FILE) as f:
            state = json.load(f)
        positions = state.get('positions', {})
        has_orig = sum(1 for p in positions.values() if p.get('_sl_price_original'))
        print(f"\n  Active positions: {len(positions)}")
        print(f"  With _sl_price_original: {has_orig}/{len(positions)}")
    except Exception:
        pass


if __name__ == '__main__':
    check()
