"""
Optimize Hybrid Strategy Thresholds

Test multiple threshold configurations to find optimal balance:
- xgb_threshold_strong: How confident XGBoost must be for "strong" entry
- xgb_threshold_moderate: Threshold for "moderate" entry
- tech_strength_threshold: Minimum technical signal strength

Goal: Beat Buy & Hold while maintaining reasonable trade frequency
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
    from scripts.production.technical_strategy import TechnicalStrategy
    from scripts.production.backtest_hybrid_v4 import HybridStrategy, rolling_window_backtest

    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    RESULTS_DIR = PROJECT_ROOT / "results"

    print("=" * 80)
    print("Hybrid Strategy Threshold Optimization")
    print("=" * 80)

    # Load XGBoost model
    model_file = MODELS_DIR / "xgboost_v3_lookahead3_thresh1_phase2.pkl"
    with open(model_file, 'rb') as f:
        xgboost_model = pickle.load(f)

    # Load feature columns
    feature_file = MODELS_DIR / "xgboost_v3_lookahead3_thresh1_phase2_features.txt"
    with open(feature_file, 'r') as f:
        feature_columns = [line.strip() for line in f.readlines()]

    # Load data
    data_file = DATA_DIR / "historical" / "BTCUSDT_5m_max.csv"
    df = pd.read_csv(data_file)

    # Calculate features
    df = calculate_features(df)

    # Calculate technical indicators
    technical_strategy = TechnicalStrategy()
    df = technical_strategy.calculate_indicators(df)
    df = df.dropna()

    print(f"✅ Data loaded: {len(df)} candles\n")

    # Test configurations
    configs = [
        # More aggressive (more trades)
        {'name': 'Aggressive', 'xgb_strong': 0.35, 'xgb_moderate': 0.25, 'tech_strength': 0.5},
        {'name': 'Semi-Aggressive', 'xgb_strong': 0.4, 'xgb_moderate': 0.3, 'tech_strength': 0.55},

        # Baseline (current)
        {'name': 'Baseline', 'xgb_strong': 0.5, 'xgb_moderate': 0.4, 'tech_strength': 0.6},

        # More conservative (fewer trades, higher quality)
        {'name': 'Semi-Conservative', 'xgb_strong': 0.55, 'xgb_moderate': 0.45, 'tech_strength': 0.65},
        {'name': 'Conservative', 'xgb_strong': 0.6, 'xgb_moderate': 0.5, 'tech_strength': 0.7},
    ]

    all_results = []

    for config in configs:
        print(f"\n{'=' * 80}")
        print(f"Testing: {config['name']}")
        print(f"  xgb_strong={config['xgb_strong']}, xgb_moderate={config['xgb_moderate']}, tech_strength={config['tech_strength']}")
        print(f"{'=' * 80}")

        # Initialize Hybrid Strategy with this config
        hybrid_strategy = HybridStrategy(
            xgboost_model=xgboost_model,
            feature_columns=feature_columns,
            technical_strategy=technical_strategy,
            xgb_threshold_strong=config['xgb_strong'],
            xgb_threshold_moderate=config['xgb_moderate'],
            tech_strength_threshold=config['tech_strength']
        )

        # Run backtest
        results = rolling_window_backtest(df, hybrid_strategy)

        # Store results
        results['config_name'] = config['name']
        all_results.append(results)

        # Print summary
        avg_return = results['hybrid_return'].mean()
        avg_bh = results['bh_return'].mean()
        avg_diff = results['difference'].mean()
        avg_trades = results['num_trades'].mean()
        avg_winrate = results['win_rate'].mean()
        avg_sharpe = results['sharpe'].mean()

        print(f"\n📊 Results:")
        print(f"  Hybrid: {avg_return:.2f}%")
        print(f"  Buy & Hold: {avg_bh:.2f}%")
        print(f"  Difference: {avg_diff:.2f}%")
        print(f"  Trades: {avg_trades:.1f}")
        print(f"  Win Rate: {avg_winrate:.1f}%")
        print(f"  Sharpe: {avg_sharpe:.3f}")

        if avg_diff > 0:
            print(f"  🎉 BEATS BUY & HOLD by {avg_diff:.2f}%!")
        elif avg_diff > -1.0:
            print(f"  ✅ Close to Buy & Hold ({avg_diff:.2f}%)")
        else:
            print(f"  ❌ Under Buy & Hold ({avg_diff:.2f}%)")

    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)

    # Summary comparison
    print(f"\n{'=' * 80}")
    print("SUMMARY COMPARISON")
    print(f"{'=' * 80}\n")

    summary_data = []
    for config_name in [c['name'] for c in configs]:
        config_results = combined_df[combined_df['config_name'] == config_name]
        summary_data.append({
            'Config': config_name,
            'Hybrid': f"{config_results['hybrid_return'].mean():.2f}%",
            'vs B&H': f"{config_results['difference'].mean():.2f}%",
            'Trades': f"{config_results['num_trades'].mean():.1f}",
            'Win Rate': f"{config_results['win_rate'].mean():.1f}%",
            'Sharpe': f"{config_results['sharpe'].mean():.3f}"
        })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    # Find best configuration
    print(f"\n{'=' * 80}")
    print("BEST CONFIGURATION")
    print(f"{'=' * 80}\n")

    best_idx = combined_df.groupby('config_name')['difference'].mean().idxmax()
    best_config = combined_df[combined_df['config_name'] == best_idx]

    print(f"🏆 Best: {best_idx}")
    print(f"  vs B&H: {best_config['difference'].mean():.2f}%")
    print(f"  Trades: {best_config['num_trades'].mean():.1f}")
    print(f"  Win Rate: {best_config['win_rate'].mean():.1f}%")
    print(f"  Sharpe: {best_config['sharpe'].mean():.3f}")

    # Get best config params
    best_config_dict = [c for c in configs if c['name'] == best_idx][0]
    print(f"\n📋 Parameters:")
    print(f"  xgb_threshold_strong: {best_config_dict['xgb_strong']}")
    print(f"  xgb_threshold_moderate: {best_config_dict['xgb_moderate']}")
    print(f"  tech_strength_threshold: {best_config_dict['tech_strength']}")

    # Save best results
    best_results_file = RESULTS_DIR / f"backtest_hybrid_v4_best.csv"
    best_config.to_csv(best_results_file, index=False)
    print(f"\n✅ Best results saved: {best_results_file}")

    # Save all results
    all_results_file = RESULTS_DIR / "backtest_hybrid_v4_all_configs.csv"
    combined_df.to_csv(all_results_file, index=False)
    print(f"✅ All results saved: {all_results_file}")

    print(f"\n{'=' * 80}")
    print("Threshold Optimization Complete!")
    print(f"{'=' * 80}")
