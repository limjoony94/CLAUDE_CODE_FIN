"""
Optimize for Profitability - Sweet Spot Strategy

목표: VIP 없이도 수익 내는 설정 찾기
- Transaction costs: 0.12% per trade
- 목표 거래: 4-6 trades (sweet spot)
- 목표 승률: 52%+ (quality first)
- 목표: vs B&H > +0.3%

전략: Ultra-5와 Conservative 사이의 sweet spot
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
    print("Sweet Spot Optimization: VIP 없이 수익 내기")
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

    print(f"✅ Data loaded: {len(df)} candles (5분 캔들 기반)\n")

    # Sweet Spot configurations
    # Ultra-5와 Conservative 사이의 스펙트럼
    configs = [
        # Ultra-5 (baseline) - 2.1 trades, +1.26%
        {'name': 'Ultra-5 (Baseline)', 'xgb_strong': 0.75, 'xgb_moderate': 0.65, 'tech_strength': 0.8},

        # Sweet Spot candidates (4-6 trades 목표)
        {'name': 'Sweet-1', 'xgb_strong': 0.72, 'xgb_moderate': 0.62, 'tech_strength': 0.78},
        {'name': 'Sweet-2', 'xgb_strong': 0.70, 'xgb_moderate': 0.60, 'tech_strength': 0.75},
        {'name': 'Sweet-3', 'xgb_strong': 0.68, 'xgb_moderate': 0.58, 'tech_strength': 0.73},
        {'name': 'Sweet-4', 'xgb_strong': 0.66, 'xgb_moderate': 0.56, 'tech_strength': 0.72},
        {'name': 'Sweet-5', 'xgb_strong': 0.64, 'xgb_moderate': 0.54, 'tech_strength': 0.71},

        # Conservative (비교) - 10.6 trades, -0.66%
        {'name': 'Conservative (Reference)', 'xgb_strong': 0.6, 'xgb_moderate': 0.5, 'tech_strength': 0.7},
    ]

    all_results = []
    TRANSACTION_COST = 0.12  # 0.06% maker + 0.06% taker

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

        # Metrics
        avg_diff = results['difference'].mean()
        avg_trades = results['num_trades'].mean()
        avg_winrate = results['win_rate'].mean()
        avg_sharpe = results['sharpe'].mean()

        # Calculate profitability
        total_cost = avg_trades * TRANSACTION_COST
        strategy_return = avg_diff + total_cost

        if avg_trades > 0:
            per_trade_profit = strategy_return / avg_trades
            per_trade_net = per_trade_profit - TRANSACTION_COST
        else:
            per_trade_profit = 0
            per_trade_net = 0

        print(f"\n📊 Results:")
        print(f"  vs B&H: {avg_diff:.2f}%")
        print(f"  Trades: {avg_trades:.1f}")
        print(f"  Win Rate: {avg_winrate:.1f}%")
        print(f"  Sharpe: {avg_sharpe:.3f}")
        print(f"\n💰 Profitability:")
        print(f"  전략 수익 (비용 전): {strategy_return:.2f}%")
        print(f"  거래 비용: {total_cost:.2f}%")
        print(f"  거래당 순이익: {per_trade_net:.3f}%")

        # Evaluation
        if avg_diff > 0:
            print(f"  🎉 PROFITABLE! vs B&H by {avg_diff:.2f}%")
        elif per_trade_net > 0:
            print(f"  ✅ 거래당 순이익 양수 (+{per_trade_net:.3f}%)")
        elif avg_diff > -0.3:
            print(f"  ⚠️ Close ({avg_diff:.2f}%)")
        else:
            print(f"  ❌ Under B&H ({avg_diff:.2f}%)")

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY: Sweet Spot Analysis")
    print(f"{'=' * 80}\n")

    summary_data = []
    for config_name in [c['name'] for c in configs]:
        config_results = pd.concat([r for r in all_results if (r['config_name'] == config_name).any()])

        avg_trades = config_results['num_trades'].mean()
        avg_diff = config_results['difference'].mean()
        total_cost = avg_trades * TRANSACTION_COST
        strategy_return = avg_diff + total_cost

        if avg_trades > 0:
            per_trade_net = (strategy_return / avg_trades) - TRANSACTION_COST
        else:
            per_trade_net = 0

        summary_data.append({
            'Config': config_name,
            'vs B&H': f"{avg_diff:.2f}%",
            'Trades': f"{avg_trades:.1f}",
            'Win Rate': f"{config_results['win_rate'].mean():.1f}%",
            '거래당순익': f"{per_trade_net:.3f}%",
            'Sharpe': f"{config_results['sharpe'].mean():.3f}",
            'Profitable': '✅' if avg_diff > 0 else ('⚠️' if per_trade_net > 0 else '❌')
        })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    # Find best profitable configuration
    combined_df = pd.concat(all_results, ignore_index=True)

    # Calculate profitability metrics for each config
    config_metrics = []
    for config_name in combined_df['config_name'].unique():
        config_data = combined_df[combined_df['config_name'] == config_name]
        avg_trades = config_data['num_trades'].mean()
        avg_diff = config_data['difference'].mean()

        total_cost = avg_trades * TRANSACTION_COST
        strategy_return = avg_diff + total_cost
        per_trade_net = (strategy_return / avg_trades - TRANSACTION_COST) if avg_trades > 0 else 0

        config_metrics.append({
            'config_name': config_name,
            'vs_bh': avg_diff,
            'trades': avg_trades,
            'per_trade_net': per_trade_net,
            'win_rate': config_data['win_rate'].mean()
        })

    metrics_df = pd.DataFrame(config_metrics)

    # Best by vs B&H
    best_vs_bh = metrics_df.loc[metrics_df['vs_bh'].idxmax()]

    # Best by per-trade net (profitability)
    best_profitable = metrics_df.loc[metrics_df['per_trade_net'].idxmax()]

    print(f"\n{'=' * 80}")
    print("BEST CONFIGURATIONS")
    print(f"{'=' * 80}\n")

    print(f"🏆 Best vs B&H: {best_vs_bh['config_name']}")
    print(f"  vs B&H: {best_vs_bh['vs_bh']:.2f}%")
    print(f"  Trades: {best_vs_bh['trades']:.1f}")
    print(f"  Win Rate: {best_vs_bh['win_rate']:.1f}%")
    print(f"  거래당 순이익: {best_vs_bh['per_trade_net']:.3f}%")

    print(f"\n💰 Best Profitability (거래당 순이익): {best_profitable['config_name']}")
    print(f"  vs B&H: {best_profitable['vs_bh']:.2f}%")
    print(f"  Trades: {best_profitable['trades']:.1f}")
    print(f"  Win Rate: {best_profitable['win_rate']:.1f}%")
    print(f"  거래당 순이익: {best_profitable['per_trade_net']:.3f}%")

    # Recommend sweet spot
    # Criteria: 4-6 trades, per_trade_net > 0, vs_bh highest
    sweet_spot = metrics_df[
        (metrics_df['trades'] >= 4) &
        (metrics_df['trades'] <= 6) &
        (metrics_df['per_trade_net'] > 0)
    ]

    if len(sweet_spot) > 0:
        best_sweet = sweet_spot.loc[sweet_spot['vs_bh'].idxmax()]

        print(f"\n🎯 SWEET SPOT (4-6 trades, 수익 가능): {best_sweet['config_name']}")
        print(f"  vs B&H: {best_sweet['vs_bh']:.2f}%")
        print(f"  Trades: {best_sweet['trades']:.1f}")
        print(f"  Win Rate: {best_sweet['win_rate']:.1f}%")
        print(f"  거래당 순이익: {best_sweet['per_trade_net']:.3f}%")

        # Get config params
        best_sweet_config = [c for c in configs if c['name'] == best_sweet['config_name']][0]
        print(f"\n📋 Optimal Parameters:")
        print(f"  xgb_threshold_strong: {best_sweet_config['xgb_strong']}")
        print(f"  xgb_threshold_moderate: {best_sweet_config['xgb_moderate']}")
        print(f"  tech_strength_threshold: {best_sweet_config['tech_strength']}")

        print(f"\n✅ VIP 없이도 수익 가능!")
    else:
        print(f"\n⚠️ Sweet spot (4-6 trades) 범위에 수익 가능한 설정 없음")
        print(f"   하지만 {best_profitable['config_name']}가 거래당 순이익 최고")

    # Save results
    best_results_file = RESULTS_DIR / "backtest_sweet_spot_profitable.csv"
    best_config_data = combined_df[combined_df['config_name'] == best_profitable['config_name']]
    best_config_data.to_csv(best_results_file, index=False)
    print(f"\n✅ Best results saved: {best_results_file}")

    # Save all results
    all_results_file = RESULTS_DIR / "backtest_sweet_spot_all.csv"
    combined_df.to_csv(all_results_file, index=False)
    print(f"✅ All results saved: {all_results_file}")

    print(f"\n{'=' * 80}")
    print("Sweet Spot Optimization Complete!")
    print(f"{'=' * 80}")
