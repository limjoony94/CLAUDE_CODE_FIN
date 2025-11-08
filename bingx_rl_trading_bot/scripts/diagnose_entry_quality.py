"""
Entry Quality Diagnostic Tool

Purpose: Analyze entry conditions to understand why win rate is 0%

Compares:
- Current entry conditions vs backtest "winning trade" characteristics
- Feature values at entry vs training distributions
- Model predictions at entry vs expected ranges
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

logger.add(LOGS_DIR / "entry_quality_diagnosis.log")


def load_trades():
    """Load trade history from state file"""
    state_path = RESULTS_DIR / "phase4_testnet_trading_state.json"

    if not state_path.exists():
        raise FileNotFoundError(f"State file not found: {state_path}")

    with open(state_path, 'r') as f:
        state = json.load(f)

    return state['trades']


def analyze_entry_conditions(trades):
    """Analyze entry conditions for all trades"""

    print("=" * 100)
    print("ENTRY QUALITY DIAGNOSIS")
    print("=" * 100)
    print()

    if len(trades) == 0:
        print("No trades to analyze")
        return

    print(f"Total Trades: {len(trades)}")
    print()

    # Separate winners and losers
    winners = [t for t in trades if t.get('pnl_usd_net', 0) > 0]
    losers = [t for t in trades if t.get('pnl_usd_net', 0) <= 0]

    print(f"Winners: {len(winners)} ({len(winners)/len(trades)*100:.1f}%)")
    print(f"Losers: {len(losers)} ({len(losers)/len(trades)*100:.1f}%)")
    print()

    # Analyze entry conditions
    print("=" * 100)
    print("ENTRY CONDITIONS ANALYSIS")
    print("=" * 100)
    print()

    # Extract entry conditions
    probabilities = []
    regimes = []
    position_sizes = []
    entry_prices = []
    exit_prices = []
    pnls = []

    for trade in trades:
        probabilities.append(trade.get('probability', 0.0))
        regimes.append(trade.get('regime', 'Unknown'))
        position_sizes.append(trade.get('position_size_pct', 0.0))

        entry_price = trade.get('entry_price', 0)
        exit_price = trade.get('exit_price', 0)
        pnl = trade.get('pnl_pct', 0) * 100

        entry_prices.append(entry_price)
        exit_prices.append(exit_price)
        pnls.append(pnl)

    # Summary statistics
    print("Entry Signal Probabilities:")
    if all(p == 0.0 for p in probabilities):
        print("  ❌ NOT RECORDED (all 0.000)")
        print("  Cannot analyze entry quality without probability data!")
    else:
        print(f"  Mean: {np.mean(probabilities):.3f}")
        print(f"  Range: [{min(probabilities):.3f}, {max(probabilities):.3f}]")

    print()

    print("Market Regimes:")
    regime_counts = {}
    for regime in regimes:
        regime_counts[regime] = regime_counts.get(regime, 0) + 1

    for regime, count in regime_counts.items():
        pct = count / len(regimes) * 100
        print(f"  {regime}: {count} ({pct:.1f}%)")

    if all(r == 'Unknown' for r in regimes):
        print("  ❌ NOT RECORDED (all Unknown)")

    print()

    print("Position Sizes:")
    if all(p == 0.0 for p in position_sizes):
        print("  ❌ NOT RECORDED")
    else:
        print(f"  Mean: {np.mean(position_sizes)*100:.1f}%")
        print(f"  Range: [{min(position_sizes)*100:.1f}%, {max(position_sizes)*100:.1f}%]")

    print()

    # Trade outcomes
    print("=" * 100)
    print("TRADE OUTCOMES")
    print("=" * 100)
    print()

    print("P&L Distribution:")
    print(f"  Mean: {np.mean(pnls):.2f}%")
    print(f"  Median: {np.median(pnls):.2f}%")
    print(f"  Worst: {min(pnls):.2f}%")
    print(f"  Best: {max(pnls):.2f}%")
    print()

    # Price movement analysis
    price_changes = [(exit - entry) / entry * 100 for entry, exit in zip(entry_prices, exit_prices)]

    print("Price Movements:")
    print(f"  Mean: {np.mean(price_changes):+.2f}%")
    print(f"  Median: {np.median(price_changes):+.2f}%")
    print()

    # Entry timing analysis
    print("=" * 100)
    print("ENTRY TIMING ANALYSIS")
    print("=" * 100)
    print()

    # Check if entries are all at similar times (potential bias)
    entry_times = [datetime.fromisoformat(t['entry_time']) for t in trades]
    entry_hours = [t.hour for t in entry_times]

    print("Entry Hour Distribution:")
    hour_counts = {}
    for hour in entry_hours:
        hour_counts[hour] = hour_counts.get(hour, 0) + 1

    for hour in sorted(hour_counts.keys()):
        count = hour_counts[hour]
        print(f"  {hour:02d}:00 - {hour:02d}:59: {count} trades")

    print()

    # Recommendations
    print("=" * 100)
    print("DIAGNOSTIC RECOMMENDATIONS")
    print("=" * 100)
    print()

    issues = []

    if all(p == 0.0 for p in probabilities):
        issues.append("❌ Entry probabilities not recorded - Cannot diagnose entry quality")
        issues.append("   Action: Verify state file is saving probability correctly")

    if all(r == 'Unknown' for r in regimes):
        issues.append("❌ Market regimes not recorded - Cannot analyze regime bias")
        issues.append("   Action: Verify regime calculation and storage")

    if len(winners) == 0:
        issues.append("❌ 0% win rate - Entry model not working")
        issues.append("   Possible causes:")
        issues.append("   1. Model overfitted to training period")
        issues.append("   2. Feature distribution shift (training vs production)")
        issues.append("   3. Threshold too aggressive (capturing weak signals)")
        issues.append("   4. Market regime different from training")
        issues.append("   Action: Run feature distribution analysis")

    if np.mean(pnls) < -0.5:
        issues.append(f"❌ Average loss {np.mean(pnls):.2f}% - Systemic entry problem")
        issues.append("   Action: Compare entry features to backtest winning trades")

    if len(issues) == 0:
        print("✅ No critical issues detected (but sample size small)")
    else:
        for issue in issues:
            print(issue)

    print()

    # Next steps
    print("=" * 100)
    print("NEXT STEPS")
    print("=" * 100)
    print()

    print("1. Wait for more trades (need 10+ for statistical significance)")
    print("2. Run feature distribution analysis:")
    print("   python scripts/analyze_feature_distributions.py")
    print("3. Collect 24h prediction distribution:")
    print("   python scripts/collect_prediction_distribution.py")
    print("4. Compare to backtest metrics:")
    print("   - Expected win rate: 82.9%")
    print("   - Expected avg position: 71.6%")
    print("   - Expected trades/week: 42.5")
    print()


def main():
    """Run entry quality diagnosis"""
    try:
        trades = load_trades()
        analyze_entry_conditions(trades)
    except Exception as e:
        logger.error(f"Diagnosis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
