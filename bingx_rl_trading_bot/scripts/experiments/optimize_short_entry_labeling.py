"""
Optimize SHORT Entry Labeling Parameters with Real BTC Data

Test improved 2of3 scoring labeling with real market data to find optimal parameters.
Target: 10-20% positive rate for good ML training.
"""

import pandas as pd
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.labeling.improved_short_entry_labeling import diagnose_short_entry_labeling


def main():
    print("="*80)
    print("SHORT Entry Labeling Parameter Optimization with Real BTC Data")
    print("="*80)

    # Load real BTC data
    data_path = project_root / "data" / "historical" / "BTCUSDT_5m_max.csv"
    print(f"\nLoading data from: {data_path}")

    df = pd.read_csv(data_path)
    print(f"Data loaded: {len(df):,} candles")
    print(f"Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
    print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

    # Run diagnostic with real data
    print("\n" + "="*80)
    print("Running diagnostic to find optimal parameters...")
    print("="*80)

    results = diagnose_short_entry_labeling(df)

    # Additional parameter tests for fine-tuning
    print("\n" + "="*80)
    print("ADDITIONAL FINE-TUNING TESTS")
    print("="*80)

    from src.labeling.improved_short_entry_labeling import create_improved_short_entry_labels
    import numpy as np

    # Test more aggressive profit thresholds
    fine_tune_configs = [
        (0.010, 3, 24, "Very Strict (1.0%, 3-24 candles)"),
        (0.015, 3, 24, "Ultra Strict (1.5%, 3-24 candles)"),
        (0.007, 6, 18, "Medium Strict (0.7%, 6-18 candles)"),
    ]

    fine_tune_results = []

    for profit_th, lead_min, lead_max, desc in fine_tune_configs:
        print(f"\n{'='*80}")
        print(f"Testing: {desc}")
        print(f"{'='*80}")

        labels = create_improved_short_entry_labels(
            df,
            lookahead=max(lead_max + 12, 24),
            profit_threshold=profit_th,
            lead_min=lead_min,
            lead_max=lead_max,
            relative_delay=12
        )

        positive_rate = np.sum(labels) / len(labels) * 100

        fine_tune_results.append({
            'config': desc,
            'positive_rate': positive_rate,
            'positive_count': np.sum(labels),
            'profit_threshold': profit_th,
            'lead_min': lead_min,
            'lead_max': lead_max
        })

    # Combined summary
    print("\n" + "="*80)
    print("COMPLETE RESULTS SUMMARY")
    print("="*80)
    print(f"\n{'Configuration':<50} {'Positive Rate':<15} {'Count':<10}")
    print("-"*80)

    # Show all results
    all_results = results + fine_tune_results

    for r in all_results:
        print(f"{r['config']:<50} {r['positive_rate']:>6.2f}% {r['positive_count']:>10,}")

    # Find configs in ideal range (10-20%)
    ideal_configs = [r for r in all_results if 10.0 <= r['positive_rate'] <= 20.0]

    print("\n" + "="*80)
    print("OPTIMAL CONFIGURATION RECOMMENDATION")
    print("="*80)

    if ideal_configs:
        # Sort by positive rate (closer to 15% is better)
        ideal_configs.sort(key=lambda r: abs(r['positive_rate'] - 15.0))
        best = ideal_configs[0]

        print(f"\n✅ RECOMMENDED CONFIGURATION:")
        print(f"   {best['config']}")
        print(f"   Positive rate: {best['positive_rate']:.2f}%")
        print(f"   Profit threshold: {best['profit_threshold']*100:.1f}%")
        print(f"   Lead-time: {best['lead_min']}-{best['lead_max']} candles")
        print(f"\n   These parameters will be used for SHORT Entry retraining.")

        # Save recommended config
        config_file = project_root / "models" / "short_entry_optimal_labeling_config.txt"
        with open(config_file, 'w') as f:
            f.write(f"# Optimal SHORT Entry Labeling Configuration\n")
            f.write(f"# Generated: {pd.Timestamp.now()}\n")
            f.write(f"# Based on real BTC data analysis\n\n")
            f.write(f"profit_threshold={best['profit_threshold']}\n")
            f.write(f"lead_min={best['lead_min']}\n")
            f.write(f"lead_max={best['lead_max']}\n")
            f.write(f"lookahead={max(best['lead_max'] + 12, 24)}\n")
            f.write(f"relative_delay=12\n")
            f.write(f"\n# Expected positive rate: {best['positive_rate']:.2f}%\n")

        print(f"\n✅ Configuration saved to: {config_file}")
    else:
        # Find closest to 15%
        best = min(all_results, key=lambda r: abs(r['positive_rate'] - 15.0))
        print(f"\n⚠️  No config in ideal range (10-20%)")
        print(f"   Closest: {best['config']}")
        print(f"   Positive rate: {best['positive_rate']:.2f}%")
        print(f"\n   Recommend: Further parameter tuning needed")

    print("\n" + "="*80)
    print("✅ Parameter optimization complete!")
    print("="*80)


if __name__ == "__main__":
    main()
