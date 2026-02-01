#!/usr/bin/env python3
"""metrics_summary.py — pattern_5m_metrics.json 파싱해서 요약 텍스트 출력"""

import json
import sys
from pathlib import Path
from datetime import datetime

BOT_DIR = Path("/home/sp/.openclaw/workspace/CLAUDE_CODE_FIN/bingx_rl_trading_bot")
METRICS_FILE = BOT_DIR / "results" / "pattern_5m_metrics.json"
STATE_FILE = BOT_DIR / "results" / "pattern_5m_bot_state.json"


def load_json(path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def summarize_metrics():
    print(f"📊 Metrics Summary — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    # Bot state
    state = load_json(STATE_FILE)
    if state and isinstance(state, dict):
        print(f"\n🤖 Bot State:")
        for key in ["status", "uptime", "start_time", "last_signal_time"]:
            if key in state:
                print(f"  {key}: {state[key]}")

    # Metrics
    metrics = load_json(METRICS_FILE)
    if metrics is None:
        print(f"\n⚠️  Metrics file not found: {METRICS_FILE}")
        return

    if isinstance(metrics, dict):
        data = metrics
    elif isinstance(metrics, list) and len(metrics) > 0:
        data = metrics[-1] if isinstance(metrics[-1], dict) else {"raw": metrics}
    else:
        print("⚠️  Unexpected metrics format")
        return

    # Key metrics
    print(f"\n📈 Performance:")
    metric_keys = [
        ("total_trades", "Total Trades"),
        ("win_rate", "Win Rate"),
        ("total_pnl", "Total PnL"),
        ("total_pnl_pct", "Total PnL %"),
        ("realized_pnl", "Realized PnL"),
        ("unrealized_pnl", "Unrealized PnL"),
        ("max_drawdown", "Max Drawdown"),
        ("max_drawdown_pct", "MDD %"),
        ("sharpe_ratio", "Sharpe Ratio"),
        ("profit_factor", "Profit Factor"),
        ("avg_win", "Avg Win"),
        ("avg_loss", "Avg Loss"),
        ("consecutive_losses", "Consecutive Losses"),
        ("consecutive_wins", "Consecutive Wins"),
        ("long_trades", "Long Trades"),
        ("short_trades", "Short Trades"),
        ("long_win_rate", "Long Win Rate"),
        ("short_win_rate", "Short Win Rate"),
    ]

    for key, label in metric_keys:
        if key in data:
            val = data[key]
            if isinstance(val, float):
                val = f"{val:.4f}" if abs(val) < 1 else f"{val:.2f}"
            print(f"  {label}: {val}")

    # Any remaining keys
    shown = {k for k, _ in metric_keys}
    remaining = {k: v for k, v in data.items() if k not in shown}
    if remaining:
        print(f"\n📋 Other:")
        for k, v in list(remaining.items())[:15]:
            if isinstance(v, (dict, list)):
                continue
            print(f"  {k}: {v}")


if __name__ == "__main__":
    summarize_metrics()
