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

# Alert thresholds
THRESHOLDS = {
    "max_consecutive_losses": 5,
    "max_drawdown_pct": 10.0,      # from config risk.max_daily_loss_pct
    "min_win_rate": 30.0,           # below this is alarming
    "stale_state_minutes": 30,      # no update = possibly crashed
}


def load_json(path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def check_alerts():
    alerts = []
    now = datetime.now()

    # 1. Check bot process is alive (state file freshness)
    state = load_json(STATE_FILE)
    if STATE_FILE.exists():
        import os
        mtime = datetime.fromtimestamp(os.path.getmtime(STATE_FILE))
        age_min = (now - mtime).total_seconds() / 60
        if age_min > THRESHOLDS["stale_state_minutes"]:
            alerts.append(f"🔴 CRITICAL: Bot state file stale ({age_min:.0f}min old, threshold={THRESHOLDS['stale_state_minutes']}min)")
    else:
        alerts.append("🔴 CRITICAL: No bot state file found — bot may not be running")

    # 2. Check metrics
    metrics = load_json(METRICS_FILE)
    if metrics is None:
        alerts.append("🟡 WARNING: No metrics file found")
    else:
        data = metrics if isinstance(metrics, dict) else (metrics[-1] if isinstance(metrics, list) and metrics else {})

        # Consecutive losses
        consec = data.get("consecutive_losses", data.get("max_consecutive_losses", 0))
        if isinstance(consec, (int, float)) and consec >= THRESHOLDS["max_consecutive_losses"]:
            alerts.append(f"🔴 CRITICAL: {consec} consecutive losses (threshold={THRESHOLDS['max_consecutive_losses']})")

        # MDD
        mdd = data.get("max_drawdown_pct", data.get("max_drawdown", 0))
        if isinstance(mdd, (int, float)) and abs(mdd) >= THRESHOLDS["max_drawdown_pct"]:
            alerts.append(f"🔴 CRITICAL: MDD={mdd:.2f}% exceeds threshold={THRESHOLDS['max_drawdown_pct']}%")

        # Win rate
        wr = data.get("win_rate", None)
        if isinstance(wr, (int, float)):
            wr_val = wr * 100 if wr < 1 else wr
            total = data.get("total_trades", 0)
            if total >= 10 and wr_val < THRESHOLDS["min_win_rate"]:
                alerts.append(f"🟡 WARNING: Win rate={wr_val:.1f}% below threshold={THRESHOLDS['min_win_rate']}% ({total} trades)")

    # 3. Check process status (platform-specific)
    if platform.system() != "Windows":
        # Unix-like: Check tmux session
        import subprocess
        try:
            result = subprocess.run(["tmux", "has-session", "-t", "pattern_5m"],
                                  capture_output=True, timeout=5)
            if result.returncode != 0:
                alerts.append("🔴 CRITICAL: tmux session 'pattern_5m' not running")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            alerts.append("🟡 WARNING: tmux not available for process check")
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
        # Exit 1 if any critical alerts
        if any("CRITICAL" in a for a in alerts):
            sys.exit(1)
        sys.exit(0)


if __name__ == "__main__":
    check_alerts()
