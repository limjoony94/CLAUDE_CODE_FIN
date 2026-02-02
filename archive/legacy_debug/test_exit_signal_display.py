#!/usr/bin/env python3
"""
Test exit signal display in monitor
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.monitoring.quant_monitor import format_signal_probability


def test_exit_signal_display():
    """Test exit signal display with actual state data"""

    print("\n" + "="*80)
    print("🔍 EXIT SIGNAL DISPLAY TEST")
    print("="*80)

    # Load actual state
    state_file = PROJECT_ROOT / "results" / "opportunity_gating_bot_4x_state.json"

    if not state_file.exists():
        print("❌ State file not found!")
        return

    with open(state_file, 'r') as f:
        state = json.load(f)

    # Check position
    position = state.get('position', {})
    if position.get('status') != 'OPEN':
        print("⚠️  No open position")
        return

    print(f"\n📊 Current Position:")
    print(f"  Side: {position.get('side')}")
    print(f"  Entry Price: ${position.get('entry_price', 0):,.2f}")
    print(f"  Entry Time: {position.get('entry_time', 'N/A')}")

    # Check exit signals
    exit_signals = state.get('latest_signals', {}).get('exit', {})

    if not exit_signals:
        print("\n❌ No exit signals in state!")
        return

    print(f"\n📈 Exit Signal Data:")
    for key, value in exit_signals.items():
        print(f"  {key}: {value}")

    # Test display formatting
    exit_prob = exit_signals.get('exit_prob', 0.0)
    exit_thresh = exit_signals.get('exit_threshold_current',
                                   exit_signals.get('exit_threshold', 0.70))
    exit_pct = (exit_prob / exit_thresh * 100) if exit_thresh > 0 else 0

    print(f"\n🎨 Display Format Test:")
    exit_prob_str = format_signal_probability(exit_prob, exit_thresh)
    exit_pct_color = "\033[1;92m" if exit_pct >= 100 else "\033[92m" if exit_pct >= 85 else "\033[93m" if exit_pct >= 70 else "\033[0m" if exit_pct >= 50 else "\033[91m"

    side = position.get('side', 'LONG')
    print(f"  Exit Signal ({side:<5s}): {exit_prob_str} ({exit_pct_color}{exit_pct:>4.0f}%\033[0m)")
    print(f"  Threshold: ML Exit ({exit_thresh:.2f})")

    # Status interpretation
    print(f"\n💡 Signal Interpretation:")
    if exit_pct >= 100:
        status = "🟢 READY TO EXIT - Signal above threshold!"
    elif exit_pct >= 85:
        status = "🟢 VERY CLOSE - Almost ready to exit"
    elif exit_pct >= 70:
        status = "🟡 APPROACHING - Getting close to exit threshold"
    elif exit_pct >= 50:
        status = "⚪ MODERATE - Halfway to exit threshold"
    else:
        status = "🔴 FAR - Still far from exit threshold"

    print(f"  {status}")

    # Volatility info
    volatility = exit_signals.get('volatility', 0.0)
    print(f"\n📊 Market Conditions:")
    print(f"  Volatility: {volatility*100:.4f}%")

    if volatility > 0.02:
        vol_status = "HIGH - Using lower exit threshold (0.65)"
    elif volatility < 0.01:
        vol_status = "LOW - Using higher exit threshold (0.75)"
    else:
        vol_status = "NORMAL - Using base exit threshold"

    print(f"  Status: {vol_status}")

    print("\n" + "="*80)
    print("✅ Test complete! Exit signal should now display in monitor.")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_exit_signal_display()
