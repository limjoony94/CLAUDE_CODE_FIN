"""
Regime-Specific Hybrid Strategy Backtest

핵심 아이디어:
- 이전 window의 regime을 보고 현재 threshold 결정
- Bull regime: Aggressive threshold (기회 포착)
- Bear regime: Conservative threshold (리스크 관리)
- Sideways: Moderate threshold (균형)

가설: Bull 시장에서 -4% 실패는 너무 보수적인 threshold 때문
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
    from scripts.production.backtest_hybrid_v4 import backtest_hybrid_strategy, classify_market_regime, HybridStrategy

    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    RESULTS_DIR = PROJECT_ROOT / "results"

    # Trading parameters
    WINDOW_SIZE = 1440  # 5 days

    print("=" * 80)
    print("Regime-Specific Hybrid Strategy Backtest")
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

    # Regime-specific threshold configurations
    # Based on critical analysis: Bull 시장에서 너무 보수적이었음
    regime_thresholds = {
        'Bull': {
            'name': 'Bull (Aggressive)',
            'xgb_strong': 0.35,  # Very aggressive!
            'xgb_moderate': 0.25,
            'tech_strength': 0.5
        },
        'Bear': {
            'name': 'Bear (Conservative)',
            'xgb_strong': 0.65,  # Conservative
            'xgb_moderate': 0.55,
            'tech_strength': 0.7
        },
        'Sideways': {
            'name': 'Sideways (Moderate)',
            'xgb_strong': 0.55,  # Moderate
            'xgb_moderate': 0.45,
            'tech_strength': 0.65
        }
    }

    def rolling_window_regime_specific_backtest(df, xgboost_model, feature_columns, technical_strategy):
        """
        Rolling window backtest with regime-specific thresholds

        Key: Use PREVIOUS window's regime to set thresholds for CURRENT window
        (Trend continuation assumption)
        """
        windows = []
        start_idx = 0
        previous_regime = 'Sideways'  # Default initial regime

        while start_idx + WINDOW_SIZE <= len(df):
            end_idx = start_idx + WINDOW_SIZE
            window_df = df.iloc[start_idx:end_idx].reset_index(drop=True)

            # Determine actual regime (for comparison)
            actual_regime = classify_market_regime(window_df)

            # Use PREVIOUS regime to set thresholds (realistic)
            # (We can't know future regime at window start)
            thresholds = regime_thresholds[previous_regime]

            # Initialize strategy with regime-specific thresholds
            hybrid_strategy = HybridStrategy(
                xgboost_model=xgboost_model,
                feature_columns=feature_columns,
                technical_strategy=technical_strategy,
                xgb_threshold_strong=thresholds['xgb_strong'],
                xgb_threshold_moderate=thresholds['xgb_moderate'],
                tech_strength_threshold=thresholds['tech_strength']
            )

            # Backtest with regime-specific thresholds
            trades, metrics = backtest_hybrid_strategy(window_df, hybrid_strategy)

            # Buy & Hold
            bh_start = window_df['close'].iloc[0]
            bh_end = window_df['close'].iloc[-1]
            bh_return = ((bh_end - bh_start) / bh_start) * 100
            bh_cost = 2 * 0.0006 * 100
            bh_return -= bh_cost

            # Confidence distribution
            if len(trades) > 0:
                strong_trades = [t for t in trades if t['confidence'] == 'strong']
                moderate_trades = [t for t in trades if t['confidence'] == 'moderate']
                num_strong = len(strong_trades)
                num_moderate = len(moderate_trades)
            else:
                num_strong = 0
                num_moderate = 0

            windows.append({
                'start_idx': start_idx,
                'end_idx': end_idx,
                'actual_regime': actual_regime,
                'used_regime': previous_regime,  # Which regime threshold was used
                'hybrid_return': metrics['total_return_pct'],
                'bh_return': bh_return,
                'difference': metrics['total_return_pct'] - bh_return,
                'num_trades': metrics['num_trades'],
                'num_strong': num_strong,
                'num_moderate': num_moderate,
                'win_rate': metrics['win_rate'],
                'sharpe': metrics['sharpe_ratio'],
                'max_dd': metrics['max_drawdown']
            })

            # Update previous regime for next window
            previous_regime = actual_regime

            start_idx += WINDOW_SIZE

        return pd.DataFrame(windows)


    print(f"{'=' * 80}")
    print("Running Regime-Specific Backtest...")
    print(f"{'=' * 80}\n")

    print("Threshold Configuration:")
    for regime, config in regime_thresholds.items():
        print(f"\n{regime}:")
        print(f"  xgb_strong: {config['xgb_strong']}")
        print(f"  xgb_moderate: {config['xgb_moderate']}")
        print(f"  tech_strength: {config['tech_strength']}")

    print(f"\n{'=' * 80}\n")

    results = rolling_window_regime_specific_backtest(
        df, xgboost_model, feature_columns, technical_strategy
    )

    # Summary
    print(f"📊 Overall Results ({len(results)} windows):\n")
    print(f"  Hybrid Return: {results['hybrid_return'].mean():.2f}% ± {results['hybrid_return'].std():.2f}%")
    print(f"  Buy & Hold Return: {results['bh_return'].mean():.2f}% ± {results['bh_return'].std():.2f}%")
    print(f"  Difference: {results['difference'].mean():.2f}% ± {results['difference'].std():.2f}%")
    print(f"  Avg Trades: {results['num_trades'].mean():.1f}")
    print(f"  Avg Win Rate: {results['win_rate'].mean():.1f}%")
    print(f"  Avg Sharpe: {results['sharpe'].mean():.3f}")

    # By actual regime (how well we performed in each regime)
    print(f"\n📈 Performance by Actual Market Regime:\n")
    for regime in ['Bull', 'Bear', 'Sideways']:
        regime_df = results[results['actual_regime'] == regime]
        if len(regime_df) > 0:
            print(f"{regime} ({len(regime_df)} windows):")
            print(f"  Used thresholds: {regime_df['used_regime'].value_counts().to_dict()}")
            print(f"  Hybrid: {regime_df['hybrid_return'].mean():.2f}%")
            print(f"  Buy & Hold: {regime_df['bh_return'].mean():.2f}%")
            print(f"  Difference: {regime_df['difference'].mean():.2f}%")
            print(f"  Trades: {regime_df['num_trades'].mean():.1f}\n")

    # Statistical significance
    from scipy import stats
    t_stat, p_value = stats.ttest_rel(results['hybrid_return'], results['bh_return'])

    print(f"🔬 Statistical Test:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Significant (p < 0.05): {'✅ Yes' if p_value < 0.05 else '❌ No'}\n")

    # Detailed window-by-window analysis
    print(f"{'=' * 80}")
    print("Window-by-Window Results:")
    print(f"{'=' * 80}\n")
    print(results[['actual_regime', 'used_regime', 'difference', 'num_trades', 'win_rate']].to_string(index=False))

    # Save results
    output_file = RESULTS_DIR / "backtest_regime_specific_v5.csv"
    results.to_csv(output_file, index=False)
    print(f"\n✅ Results saved: {output_file}")

    # Critical analysis
    print(f"\n{'=' * 80}")
    print("CRITICAL ANALYSIS")
    print(f"{'=' * 80}\n")

    print("비교:")
    print(f"\n  Phase 3 Conservative (고정 threshold):")
    print(f"    - vs B&H: -0.66%")
    print(f"    - Bull: -4.42%")
    print(f"    - Bear: +5.46%")
    print(f"    - Sideways: +1.28%")

    print(f"\n  Phase 4 Regime-Specific:")
    print(f"    - vs B&H: {results['difference'].mean():.2f}%")
    bull_results = results[results['actual_regime'] == 'Bull']
    bear_results = results[results['actual_regime'] == 'Bear']
    sideways_results = results[results['actual_regime'] == 'Sideways']

    if len(bull_results) > 0:
        print(f"    - Bull: {bull_results['difference'].mean():.2f}%")
    if len(bear_results) > 0:
        print(f"    - Bear: {bear_results['difference'].mean():.2f}%")
    if len(sideways_results) > 0:
        print(f"    - Sideways: {sideways_results['difference'].mean():.2f}%")

    # Check if Bull improved
    if len(bull_results) > 0:
        bull_improvement = bull_results['difference'].mean() - (-4.42)
        print(f"\n  Bull 시장 개선: {bull_improvement:+.2f}%p")

        if bull_results['difference'].mean() > -2.0:
            print(f"  ✅ Bull 시장 성과 크게 개선!")
        elif bull_improvement > 0:
            print(f"  ⚠️ Bull 시장 일부 개선, 추가 최적화 필요")
        else:
            print(f"  ❌ Bull 시장 개선 실패")

    if results['difference'].mean() > 0:
        print(f"\n  🎉 SUCCESS! Beat Buy & Hold by {results['difference'].mean():.2f}%!")
    elif results['difference'].mean() > -0.5:
        print(f"\n  ✅ Very close to Buy & Hold ({results['difference'].mean():.2f}%)")
    else:
        print(f"\n  ❌ Still behind Buy & Hold ({results['difference'].mean():.2f}%)")

    print(f"\n{'=' * 80}")
    print("Regime-Specific Backtest Complete")
    print(f"{'=' * 80}")
