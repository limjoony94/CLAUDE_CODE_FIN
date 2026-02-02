"""
Test Ultra-Conservative Thresholds

Hypothesis: Even more conservative settings might achieve positive returns
Target: < 10 trades per window, win rate > 50%, beat Buy & Hold

Rationale:
- Conservative config got us to -0.66% (very close!)
- Trend shows: more conservative = better performance
- Transaction costs: Fewer trades = less cost drag
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
    print("Ultra-Conservative Threshold Test")
    print("Target: Beat Buy & Hold with < 10 trades, > 50% win rate")
    print("=" * 80)

    # Load model and data
    model_file = MODELS_DIR / "xgboost_v3_lookahead3_thresh1_phase2.pkl"
    with open(model_file, 'rb') as f:
        xgboost_model = pickle.load(f)

    feature_file = MODELS_DIR / "xgboost_v3_lookahead3_thresh1_phase2_features.txt"
    with open(feature_file, 'r') as f:
        feature_columns = [line.strip() for line in f.readlines()]

    data_file = DATA_DIR / "historical" / "BTCUSDT_5m_max.csv"
    df = pd.read_csv(data_file)
    df = calculate_features(df)

    technical_strategy = TechnicalStrategy()
    df = technical_strategy.calculate_indicators(df)
    df = df.dropna()

    print(f"✅ Data loaded: {len(df)} candles\n")

    # Ultra-conservative configurations
    configs = [
        # Previous best
        {'name': 'Conservative (baseline)', 'xgb_strong': 0.6, 'xgb_moderate': 0.5, 'tech_strength': 0.7},

        # Ultra-conservative variations
        {'name': 'Ultra-1', 'xgb_strong': 0.65, 'xgb_moderate': 0.55, 'tech_strength': 0.7},
        {'name': 'Ultra-2', 'xgb_strong': 0.7, 'xgb_moderate': 0.6, 'tech_strength': 0.7},
        {'name': 'Ultra-3', 'xgb_strong': 0.65, 'xgb_moderate': 0.55, 'tech_strength': 0.75},
        {'name': 'Ultra-4', 'xgb_strong': 0.7, 'xgb_moderate': 0.6, 'tech_strength': 0.75},
        {'name': 'Ultra-5 (Extreme)', 'xgb_strong': 0.75, 'xgb_moderate': 0.65, 'tech_strength': 0.8},
    ]

    all_results = []

    for config in configs:
        print(f"\n{'=' * 80}")
        print(f"Testing: {config['name']}")
        print(f"  xgb_strong={config['xgb_strong']}, xgb_moderate={config['xgb_moderate']}, tech_strength={config['tech_strength']}")
        print(f"{'=' * 80}")

        hybrid_strategy = HybridStrategy(
            xgboost_model=xgboost_model,
            feature_columns=feature_columns,
            technical_strategy=technical_strategy,
            xgb_threshold_strong=config['xgb_strong'],
            xgb_threshold_moderate=config['xgb_moderate'],
            tech_strength_threshold=config['tech_strength']
        )

        results = rolling_window_backtest(df, hybrid_strategy)
        results['config_name'] = config['name']
        all_results.append(results)

        avg_diff = results['difference'].mean()
        avg_trades = results['num_trades'].mean()
        avg_winrate = results['win_rate'].mean()
        avg_sharpe = results['sharpe'].mean()

        print(f"\n📊 Results:")
        print(f"  vs B&H: {avg_diff:.2f}%")
        print(f"  Trades: {avg_trades:.1f}")
        print(f"  Win Rate: {avg_winrate:.1f}%")
        print(f"  Sharpe: {avg_sharpe:.3f}")

        if avg_diff > 0:
            print(f"  🎉 BEATS BUY & HOLD by {avg_diff:.2f}%!")
        elif avg_diff > -0.5:
            print(f"  ✅ Very close ({avg_diff:.2f}%)")
        elif avg_diff > -1.0:
            print(f"  ⚠️ Close ({avg_diff:.2f}%)")
        else:
            print(f"  ❌ Under B&H ({avg_diff:.2f}%)")

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}\n")

    summary_data = []
    for config_name in [c['name'] for c in configs]:
        config_results = pd.concat([r for r in all_results if (r['config_name'] == config_name).any()])
        summary_data.append({
            'Config': config_name,
            'vs B&H': f"{config_results['difference'].mean():.2f}%",
            'Trades': f"{config_results['num_trades'].mean():.1f}",
            'Win Rate': f"{config_results['win_rate'].mean():.1f}%",
            'Sharpe': f"{config_results['sharpe'].mean():.3f}"
        })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    # Find best
    combined_df = pd.concat(all_results, ignore_index=True)
    best_idx = combined_df.groupby('config_name')['difference'].mean().idxmax()
    best_config = combined_df[combined_df['config_name'] == best_idx]

    print(f"\n{'=' * 80}")
    print("BEST CONFIGURATION")
    print(f"{'=' * 80}\n")

    print(f"🏆 Best: {best_idx}")
    print(f"  vs B&H: {best_config['difference'].mean():.2f}%")
    print(f"  Trades: {best_config['num_trades'].mean():.1f}")
    print(f"  Win Rate: {best_config['win_rate'].mean():.1f}%")
    print(f"  Sharpe: {best_config['sharpe'].mean():.3f}")

    # Get best params
    best_config_dict = [c for c in configs if c['name'] == best_idx][0]
    print(f"\n📋 Optimal Parameters:")
    print(f"  xgb_threshold_strong: {best_config_dict['xgb_strong']}")
    print(f"  xgb_threshold_moderate: {best_config_dict['xgb_moderate']}")
    print(f"  tech_strength_threshold: {best_config_dict['tech_strength']}")

    # Check if we beat Buy & Hold
    if best_config['difference'].mean() > 0:
        print(f"\n🎉🎉🎉 SUCCESS! 🎉🎉🎉")
        print(f"Hybrid Strategy BEATS Buy & Hold by {best_config['difference'].mean():.2f}%!")
    elif best_config['difference'].mean() > -0.3:
        print(f"\n✅ Nearly there! Only {abs(best_config['difference'].mean()):.2f}% away from Buy & Hold")
        print(f"Further optimization or market regime filtering may help")
    else:
        print(f"\n⚠️ Still {abs(best_config['difference'].mean()):.2f}% behind Buy & Hold")
        print(f"May need different approach or market regime specific strategies")

    # Save
    best_results_file = RESULTS_DIR / "backtest_hybrid_v4_ultraconservative.csv"
    best_config.to_csv(best_results_file, index=False)
    print(f"\n✅ Results saved: {best_results_file}")

    print(f"\n{'=' * 80}")
    print("Ultra-Conservative Test Complete!")
    print(f"{'=' * 80}")
