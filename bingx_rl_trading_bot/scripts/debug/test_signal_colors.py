#!/usr/bin/env python3
"""
Test script for signal probability color formatting
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.monitoring.quant_monitor import format_signal_probability


def test_signal_colors():
    """Test signal probability color formatting"""

    print("\n" + "="*80)
    print("🎨 SIGNAL PROBABILITY COLOR TEST")
    print("="*80)

    # Test LONG signals (threshold: 0.65)
    print("\n📈 LONG Signals (Threshold: 0.65)")
    print("-" * 80)

    test_cases_long = [
        (0.85, 0.65, "Above threshold - Ready to enter!"),
        (0.70, 0.65, "Above threshold - Ready to enter!"),
        (0.65, 0.65, "Exactly at threshold - Ready to enter!"),
        (0.60, 0.65, "92% of threshold - Very close"),
        (0.55, 0.65, "85% of threshold - Close"),
        (0.50, 0.65, "77% of threshold - Approaching"),
        (0.40, 0.65, "62% of threshold - Moderate"),
        (0.30, 0.65, "46% of threshold - Far"),
        (0.10, 0.65, "15% of threshold - Very far"),
    ]

    for prob, thresh, description in test_cases_long:
        formatted = format_signal_probability(prob, thresh)
        pct = (prob / thresh * 100) if thresh > 0 else 0
        print(f"  {formatted} ({pct:>4.0f}%)  │  {description}")

    # Test SHORT signals (threshold: 0.70)
    print("\n📉 SHORT Signals (Threshold: 0.70)")
    print("-" * 80)

    test_cases_short = [
        (0.90, 0.70, "Above threshold - Ready to enter!"),
        (0.75, 0.70, "Above threshold - Ready to enter!"),
        (0.70, 0.70, "Exactly at threshold - Ready to enter!"),
        (0.65, 0.70, "93% of threshold - Very close"),
        (0.60, 0.70, "86% of threshold - Close"),
        (0.55, 0.70, "79% of threshold - Approaching"),
        (0.45, 0.70, "64% of threshold - Moderate"),
        (0.30, 0.70, "43% of threshold - Far"),
        (0.10, 0.70, "14% of threshold - Very far"),
    ]

    for prob, thresh, description in test_cases_short:
        formatted = format_signal_probability(prob, thresh)
        pct = (prob / thresh * 100) if thresh > 0 else 0
        print(f"  {formatted} ({pct:>4.0f}%)  │  {description}")

    # Test EXIT signals (threshold: 0.70 for LONG, 0.72 for SHORT)
    print("\n🚪 EXIT Signals (LONG Threshold: 0.70, SHORT Threshold: 0.72)")
    print("-" * 80)

    test_cases_exit = [
        (0.85, 0.70, "LONG", "Above threshold - Ready to exit!"),
        (0.68, 0.70, "LONG", "97% of threshold - Very close"),
        (0.60, 0.70, "LONG", "86% of threshold - Close"),
        (0.50, 0.70, "LONG", "71% of threshold - Approaching"),
        (0.30, 0.70, "LONG", "43% of threshold - Far"),
        (0.90, 0.72, "SHORT", "Above threshold - Ready to exit!"),
        (0.70, 0.72, "SHORT", "97% of threshold - Very close"),
        (0.65, 0.72, "SHORT", "90% of threshold - Close"),
        (0.55, 0.72, "SHORT", "76% of threshold - Approaching"),
        (0.35, 0.72, "SHORT", "49% of threshold - Far"),
    ]

    for prob, thresh, side, description in test_cases_exit:
        formatted = format_signal_probability(prob, thresh)
        pct = (prob / thresh * 100) if thresh > 0 else 0
        print(f"  {side:>5s}  {formatted} ({pct:>4.0f}%)  │  {description}")

    # Color legend
    print("\n🎨 COLOR LEGEND")
    print("-" * 80)
    print(f"  \033[1;92m●\033[0m Bright Green (Bold)  : ≥100% of threshold (Ready!)")
    print(f"  \033[92m●\033[0m Green               : 85-99% of threshold (Very close)")
    print(f"  \033[93m●\033[0m Yellow              : 70-84% of threshold (Approaching)")
    print(f"  \033[0m●\033[0m White (Default)     : 50-69% of threshold (Moderate)")
    print(f"  \033[91m●\033[0m Red                 : <50% of threshold (Far)")

    print("\n" + "="*80)
    print("✅ Color test complete! All signals should display with appropriate colors.")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_signal_colors()
