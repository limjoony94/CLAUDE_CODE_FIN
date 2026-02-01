#!/usr/bin/env python3
"""alert_check.py — 이상 징후 감지 (연속 손실, MDD 초과 등)"""

import json
import sys
from pathlib import Path
from datetime import datetime
import platform

# Cross-platform path detection
BOT_DIR = Path(__file__).parent.parent.parent  # scripts/monitor/ -> bingx_rl_trading_bot/
METRICS_FILE = BOT_DIR / "results" / "pattern_5m_metrics.json"
STATE_FILE = BOT_DIR / "results" / "pattern_5m_bot_state.json"
CONFIG_FILE = BOT_DIR / "config" / "pattern_5m_config.yaml"

# Alert thresholds (three-tier: warning → danger)
THRESHOLDS = {
    "consecutive_losses_warning": 5,
    "consecutive_losses_danger": 8,
    "mdd_warning_pct": 15.0,
    "mdd_danger_pct": 25.0,
    "win_rate_warning_pct": 65.0,      # WR < 65% = warning (last 50 trades)
    "win_rate_danger_pct": 50.0,       # WR < 50% = danger (last 50 trades)
    "stale_state_minutes": 60,         # 1 hour = warning
    "daily_loss_danger_pct": -5.0,     # Daily loss ≤ -5% = danger
    "recent_trades_window": 50,        # Window for WR calculation
}


def load_json(path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def check_alerts():
    alerts = []
    max_severity = 0  # 0=normal, 1=warning, 2=danger
    now = datetime.now()

    # 1. Check bot process is alive (state file freshness)
    state = load_json(STATE_FILE)
    if STATE_FILE.exists():
        import os
        mtime = datetime.fromtimestamp(os.path.getmtime(STATE_FILE))
        age_min = (now - mtime).total_seconds() / 60
        if age_min > THRESHOLDS["stale_state_minutes"]:
            alerts.append(f"🟡 WARNING: Bot state file stale ({age_min:.0f}min old, threshold={THRESHOLDS['stale_state_minutes']}min)")
            max_severity = max(max_severity, 1)
    else:
        alerts.append("🔴 DANGER: No bot state file found — bot may not be running")
        max_severity = max(max_severity, 2)

    # 2. Check metrics
    metrics = load_json(METRICS_FILE)
    if metrics is None:
        alerts.append("🟡 WARNING: No metrics file found")
        max_severity = max(max_severity, 1)
    else:
        data = metrics if isinstance(metrics, dict) else (metrics[-1] if isinstance(metrics, list) and metrics else {})

        # Consecutive losses (three-tier)
        consec = data.get("consecutive_losses", data.get("max_consecutive_losses", 0))
        if isinstance(consec, (int, float)):
            if consec >= THRESHOLDS["consecutive_losses_danger"]:
                alerts.append(f"🔴 DANGER: {consec} consecutive losses (danger threshold={THRESHOLDS['consecutive_losses_danger']})")
                max_severity = max(max_severity, 2)
            elif consec >= THRESHOLDS["consecutive_losses_warning"]:
                alerts.append(f"🟡 WARNING: {consec} consecutive losses (warning threshold={THRESHOLDS['consecutive_losses_warning']})")
                max_severity = max(max_severity, 1)

        # MDD (three-tier)
        mdd = data.get("max_drawdown_pct", data.get("max_drawdown", 0))
        if isinstance(mdd, (int, float)):
            mdd_abs = abs(mdd)
            if mdd_abs >= THRESHOLDS["mdd_danger_pct"]:
                alerts.append(f"🔴 DANGER: MDD={mdd:.2f}% exceeds danger threshold={THRESHOLDS['mdd_danger_pct']}%")
                max_severity = max(max_severity, 2)
            elif mdd_abs >= THRESHOLDS["mdd_warning_pct"]:
                alerts.append(f"🟡 WARNING: MDD={mdd:.2f}% exceeds warning threshold={THRESHOLDS['mdd_warning_pct']}%")
                max_severity = max(max_severity, 1)

        # Win rate (last 50 trades, three-tier)
        # Get recent trade history from state file
        if state and "trade_history" in state:
            recent_trades = state["trade_history"][-THRESHOLDS["recent_trades_window"]:]
            if len(recent_trades) >= 10:
                wins = sum(1 for t in recent_trades if t.get("pnl", 0) > 0)
                wr_recent = wins / len(recent_trades) * 100
                if wr_recent < THRESHOLDS["win_rate_danger_pct"]:
                    alerts.append(f"🔴 DANGER: Win rate={wr_recent:.1f}% below danger threshold={THRESHOLDS['win_rate_danger_pct']}% (last {len(recent_trades)} trades)")
                    max_severity = max(max_severity, 2)
                elif wr_recent < THRESHOLDS["win_rate_warning_pct"]:
                    alerts.append(f"🟡 WARNING: Win rate={wr_recent:.1f}% below warning threshold={THRESHOLDS['win_rate_warning_pct']}% (last {len(recent_trades)} trades)")
                    max_severity = max(max_severity, 1)

        # Daily loss check
        daily_pnl_pct = data.get("daily_pnl_pct", 0)
        if isinstance(daily_pnl_pct, (int, float)) and daily_pnl_pct <= THRESHOLDS["daily_loss_danger_pct"]:
            alerts.append(f"🔴 DANGER: Daily loss={daily_pnl_pct:.2f}% exceeds danger threshold={THRESHOLDS['daily_loss_danger_pct']}%")
            max_severity = max(max_severity, 2)

    # 3. Check process status (platform-specific)
    if platform.system() != "Windows":
        # Unix-like: Check tmux session
        import subprocess
        try:
            result = subprocess.run(["tmux", "has-session", "-t", "pattern_5m"],
                                  capture_output=True, timeout=5)
            if result.returncode != 0:
                alerts.append("🔴 DANGER: tmux session 'pattern_5m' not running")
                max_severity = max(max_severity, 2)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            alerts.append("🟡 WARNING: tmux not available for process check")
            max_severity = max(max_severity, 1)
    # Windows: Rely on state file freshness check (already done above)

    # Output
    print(f"🔍 Alert Check — {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    if not alerts:
        print("✅ All clear — no alerts")
        sys.exit(0)
    else:
        for alert in alerts:
            print(alert)
        print(f"\nSeverity: {'DANGER' if max_severity == 2 else 'WARNING' if max_severity == 1 else 'NORMAL'}")
        # Exit code: 0=normal, 1=warning, 2=danger
        sys.exit(max_severity)


if __name__ == "__main__":
    check_alerts()
