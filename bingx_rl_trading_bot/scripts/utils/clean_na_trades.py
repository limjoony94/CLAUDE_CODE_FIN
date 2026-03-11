#!/usr/bin/env python3
"""
One-time cleanup: Remove N/A pattern trades and duplicates from metrics.

Backs up original file before modification.
Recalculates state counters to match clean trade_history.
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from collections import defaultdict

METRICS_FILE = Path("results/pattern_5m_metrics.json")
STATE_FILE = Path("results/pattern_5m_bot_state.json")

def main():
    # Backup
    backup_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = METRICS_FILE.parent / f"pattern_5m_metrics_backup_{backup_ts}.json"
    shutil.copy2(METRICS_FILE, backup_path)
    print(f"Backup saved to {backup_path}")

    with open(METRICS_FILE) as f:
        metrics = json.load(f)

    history = metrics['trade_history']
    print(f"Original trade_history: {len(history)} entries")

    # Remove N/A trades and duplicates
    clean = []
    seen_keys = set()
    na_removed = 0
    dup_removed = 0

    for t in history:
        # Remove N/A pattern trades
        if t.get('pattern', '') == 'N/A':
            na_removed += 1
            continue

        # Remove duplicates (same entry+exit+direction)
        key = (round(t['entry_price'], 1), round(t['exit_price'], 1), t['direction'])
        if key in seen_keys:
            dup_removed += 1
            continue
        seen_keys.add(key)
        clean.append(t)

    print(f"Removed: {na_removed} N/A + {dup_removed} duplicates = {na_removed + dup_removed}")
    print(f"Clean trade_history: {len(clean)} entries")

    # Recalculate metrics from clean history
    total_trades = len(clean)
    winning_trades = sum(1 for t in clean if t.get('pnl_portfolio', 0) > 0)
    total_pnl = sum(t.get('pnl_portfolio', 0) for t in clean)

    # Update metrics
    metrics['trade_history'] = clean
    metrics['total_trades'] = total_trades
    metrics['winning_trades'] = winning_trades
    metrics['total_pnl'] = round(total_pnl, 4)
    if total_trades > 0:
        metrics['actual_win_rate'] = round(winning_trades / total_trades * 100, 2)
    else:
        metrics['actual_win_rate'] = 0.0

    # Save
    with open(METRICS_FILE, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"Saved cleaned metrics to {METRICS_FILE}")

    # Also fix state counters
    if STATE_FILE.exists():
        state_backup = STATE_FILE.parent / f"pattern_5m_bot_state_backup_{backup_ts}.json"
        shutil.copy2(STATE_FILE, state_backup)
        print(f"State backup saved to {state_backup}")

        with open(STATE_FILE) as f:
            state = json.load(f)

        state['total_trades'] = total_trades
        state['winning_trades'] = winning_trades
        state['total_pnl'] = round(total_pnl, 4)

        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        print(f"Updated state counters in {STATE_FILE}")

    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"Trades: {len(history)} → {len(clean)} (-{len(history)-len(clean)})")
    print(f"WR: {metrics['actual_win_rate']:.1f}%")
    print(f"PnL: {total_pnl:+.3f}%")

if __name__ == '__main__':
    main()
