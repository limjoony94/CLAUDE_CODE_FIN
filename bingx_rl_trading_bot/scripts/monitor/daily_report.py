#!/usr/bin/env python3
"""daily_report.py — 일일 성과 리포트"""

import json
import sys
from pathlib import Path
from datetime import datetime, timedelta

BOT_DIR = Path("/home/sp/.openclaw/workspace/CLAUDE_CODE_FIN/bingx_rl_trading_bot")
METRICS_FILE = BOT_DIR / "results" / "pattern_5m_metrics.json"
STATE_FILE = BOT_DIR / "results" / "pattern_5m_bot_state.json"
CONFIG_FILE = BOT_DIR / "config" / "pattern_5m_config.yaml"


def load_json(path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_config():
    if not CONFIG_FILE.exists():
        return {}
    try:
        import yaml
        with open(CONFIG_FILE) as f:
            return yaml.safe_load(f)
    except ImportError:
        # Fallback: basic parsing
        config = {}
        with open(CONFIG_FILE) as f:
            for line in f:
                line = line.strip()
                if ":" in line and not line.startswith("#") and not line.startswith("-"):
                    k, v = line.split(":", 1)
                    config[k.strip()] = v.strip()
        return config


def generate_report():
    now = datetime.now()
    print(f"📊 Daily Trading Report — {now.strftime('%Y-%m-%d')}")
    print("=" * 60)

    # Config info
    config = load_config()
    if config:
        leverage = config.get("leverage", config.get("exchange_leverage", "?"))
        print(f"\n⚙️  Config: leverage={leverage}")

    # Bot state
    state = load_json(STATE_FILE)
    if state and isinstance(state, dict):
        status = state.get("status", "unknown")
        emoji = "✅" if status in ("running", "active") else "❌"
        print(f"\n{emoji} Bot Status: {status}")
        if "start_time" in state:
            print(f"  Started: {state['start_time']}")

    # Metrics
    metrics = load_json(METRICS_FILE)
    if metrics is None:
        print(f"\n⚠️  No metrics data available")
        print(f"   Expected: {METRICS_FILE}")
        return

    data = metrics if isinstance(metrics, dict) else (metrics[-1] if isinstance(metrics, list) and metrics else {})

    # Performance summary
    print(f"\n💰 Performance Summary:")
    fields = {
        "total_trades": "Total Trades",
        "win_rate": "Win Rate",
        "total_pnl": "Total PnL (USDT)",
        "total_pnl_pct": "Total PnL %",
        "max_drawdown_pct": "Max Drawdown %",
        "sharpe_ratio": "Sharpe Ratio",
        "profit_factor": "Profit Factor",
    }
    for key, label in fields.items():
        if key in data:
            val = data[key]
            if isinstance(val, float):
                if "pct" in key or "rate" in key:
                    val = f"{val:.2f}%"
                else:
                    val = f"{val:.4f}" if abs(val) < 1 else f"{val:.2f}"
            print(f"  {label}: {val}")

    # Direction breakdown
    print(f"\n📊 By Direction:")
    for direction in ["long", "short"]:
        trades = data.get(f"{direction}_trades", "?")
        wr = data.get(f"{direction}_win_rate", "?")
        pnl = data.get(f"{direction}_pnl", "?")
        if trades != "?":
            print(f"  {direction.upper()}: {trades} trades, WR={wr}, PnL={pnl}")

    # Risk indicators
    print(f"\n⚠️  Risk Indicators:")
    consecutive_losses = data.get("consecutive_losses", data.get("max_consecutive_losses", 0))
    mdd = data.get("max_drawdown_pct", data.get("max_drawdown", 0))
    print(f"  Consecutive Losses: {consecutive_losses}")
    print(f"  Max Drawdown: {mdd}")

    # Timestamp
    print(f"\n📅 Report generated: {now.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    generate_report()
