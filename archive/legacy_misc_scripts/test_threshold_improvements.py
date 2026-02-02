"""
Test Threshold Improvements: V1 (Linear) vs V2 (Non-linear)

Verify that new threshold calculation handles extreme conditions better.
"""

import numpy as np


def calculate_threshold_v1(signal_rate, expected_rate=0.0612, base_threshold=0.70):
    """V1: Linear adjustment (OLD)"""
    THRESHOLD_ADJUSTMENT_FACTOR = 0.15
    MIN_THRESHOLD = 0.55
    MAX_THRESHOLD = 0.85

    adjustment_ratio = signal_rate / expected_rate if expected_rate > 0 else 1.0
    threshold_delta = (1.0 - adjustment_ratio) * THRESHOLD_ADJUSTMENT_FACTOR

    adjusted = base_threshold - threshold_delta
    adjusted = np.clip(adjusted, MIN_THRESHOLD, MAX_THRESHOLD)

    return {
        'adjusted_threshold': adjusted,
        'adjustment_ratio': adjustment_ratio,
        'threshold_delta': threshold_delta,
        'clipped': adjusted == MIN_THRESHOLD or adjusted == MAX_THRESHOLD
    }


def calculate_threshold_v2(signal_rate, expected_rate=0.0612, base_threshold=0.70):
    """V2: Non-linear adjustment (NEW)"""
    THRESHOLD_ADJUSTMENT_FACTOR = 0.25  # Increased
    MIN_THRESHOLD = 0.50  # Lowered
    MAX_THRESHOLD = 0.92  # Raised

    adjustment_ratio = signal_rate / expected_rate if expected_rate > 0 else 1.0

    # Non-linear threshold adjustment
    if adjustment_ratio > 2.0:  # Extreme high
        threshold_delta = -THRESHOLD_ADJUSTMENT_FACTOR * ((adjustment_ratio - 1.0) ** 0.75)
    elif adjustment_ratio < 0.5:  # Extreme low
        threshold_delta = THRESHOLD_ADJUSTMENT_FACTOR * ((1.0 - adjustment_ratio) ** 0.75)
    else:  # Normal range
        threshold_delta = (1.0 - adjustment_ratio) * THRESHOLD_ADJUSTMENT_FACTOR

    adjusted = base_threshold - threshold_delta
    adjusted = np.clip(adjusted, MIN_THRESHOLD, MAX_THRESHOLD)

    return {
        'adjusted_threshold': adjusted,
        'adjustment_ratio': adjustment_ratio,
        'threshold_delta': threshold_delta,
        'clipped': adjusted == MIN_THRESHOLD or adjusted == MAX_THRESHOLD
    }


def test_scenarios():
    """Test various signal rate scenarios"""

    scenarios = [
        ("Very Quiet", 0.02, 0.327),  # 2% (33% of expected)
        ("Quiet", 0.04, 0.65),  # 4% (65% of expected)
        ("Normal", 0.06, 0.98),  # 6% (98% of expected)
        ("Active", 0.10, 1.63),  # 10% (163% of expected)
        ("Very Active", 0.15, 2.45),  # 15% (245% of expected)
        ("CURRENT STATE", 0.194, 3.17),  # 19.4% (317% of expected) ← Current issue
        ("Extreme", 0.25, 4.08),  # 25% (408% of expected)
    ]

    print("=" * 100)
    print("THRESHOLD CALCULATION COMPARISON: V1 (Linear) vs V2 (Non-linear)")
    print("=" * 100)
    print()

    print(f"{'Scenario':<15} {'Signal':<8} {'Ratio':<8} {'V1 Threshold':<15} {'V1 Clip?':<10} {'V2 Threshold':<15} {'V2 Clip?':<10} {'Improvement'}")
    print("-" * 100)

    for name, signal_rate, expected_ratio in scenarios:
        v1 = calculate_threshold_v1(signal_rate)
        v2 = calculate_threshold_v2(signal_rate)

        v1_clipped = "YES ❌" if v1['clipped'] else "NO ✅"
        v2_clipped = "YES ❌" if v2['clipped'] else "NO ✅"

        improvement = v2['adjusted_threshold'] - v1['adjusted_threshold']
        improvement_str = f"{improvement:+.3f}"

        if name == "CURRENT STATE":
            print(f">>> {name:<12} {signal_rate*100:>6.1f}% {expected_ratio:>6.2f}x {v1['adjusted_threshold']:>13.3f} {v1_clipped:<10} {v2['adjusted_threshold']:>13.3f} {v2_clipped:<10} {improvement_str} <<<")
        else:
            print(f"    {name:<12} {signal_rate*100:>6.1f}% {expected_ratio:>6.2f}x {v1['adjusted_threshold']:>13.3f} {v1_clipped:<10} {v2['adjusted_threshold']:>13.3f} {v2_clipped:<10} {improvement_str}")

    print()
    print("=" * 100)
    print("KEY FINDINGS")
    print("=" * 100)
    print()

    # Analyze current state
    current_v1 = calculate_threshold_v1(0.194)
    current_v2 = calculate_threshold_v2(0.194)

    print("CURRENT STATE (19.4% signal rate):")
    print()
    print(f"V1 (OLD):")
    print(f"  Ratio: {current_v1['adjustment_ratio']:.2f}x expected")
    print(f"  Delta: {current_v1['threshold_delta']:.3f}")
    print(f"  Adjusted Threshold: {current_v1['adjusted_threshold']:.3f}")
    print(f"  Clipped: {current_v1['clipped']} ❌")
    print(f"  Problem: At maximum (0.85), still generates 19.4% signals!")
    print()

    print(f"V2 (NEW):")
    print(f"  Ratio: {current_v2['adjustment_ratio']:.2f}x expected")
    print(f"  Delta: {current_v2['threshold_delta']:.3f} (non-linear adjustment)")
    print(f"  Adjusted Threshold: {current_v2['adjusted_threshold']:.3f}")
    print(f"  Clipped: {current_v2['clipped']} {'✅' if not current_v2['clipped'] else '❌'}")
    print(f"  Improvement: +{current_v2['adjusted_threshold'] - current_v1['adjusted_threshold']:.3f} (higher threshold)")
    print()

    print("IMPROVEMENTS:")
    print("1. ✅ Wider dynamic range: 0.50-0.92 (was 0.55-0.85)")
    print("2. ✅ Stronger adjustment: 0.25 factor (was 0.15)")
    print("3. ✅ Non-linear scaling: Exponential for extreme conditions")
    print("4. ✅ Emergency monitoring: Alerts if at max for >1 hour")
    print()

    print("EXPECTED IMPACT:")
    print(f"  Current signal rate: 19.4% @ threshold 0.85")
    print(f"  New threshold: 0.92 (+0.07)")
    print(f"  Expected signal rate: ~10-12% (estimated 40-50% reduction)")
    print(f"  Note: May still be higher than 6.12% target if model distribution shifted")
    print()

    print("=" * 100)


if __name__ == "__main__":
    test_scenarios()
