"""
Backtest All Labeling Methods - Comprehensive Comparison

Models to test:
1. Baseline (Phase 4 Base) - Simple threshold labeling
2. Realistic Labels - P&L-based labeling
3. Regression - Direct P&L prediction
4. With Regime - Unsupervised + realistic labels

Goal: Find best labeling/learning approach for production
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys
from loguru import logger

# Project setup
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Trading parameters (same as baseline)
WINDOW_SIZE = 576  # 2 days
INITIAL_CAPITAL = 10000.0
POSITION_SIZE_PCT = 0.95
STOP_LOSS = 0.01
TAKE_PROFIT = 0.03
MAX_HOLDING_HOURS = 4
TRANSACTION_COST = 0.0002
ENTRY_THRESHOLD = 0.7


def classify_market_regime(df_window):
    """Classify market regime"""
    start_price = df_window['close'].iloc[0]
    end_price = df_window['close'].iloc[-1]
    return_pct = ((end_price / start_price) - 1) * 100
    if return_pct > 3.0:
        return "Bull"
    elif return_pct < -2.0:
        return "Bear"
    else:
        return "Sideways"


def backtest_classifier(df, model, feature_columns, entry_threshold):
    """Backtest classification model"""
    capital = INITIAL_CAPITAL
    position = None
    trades = []

    for i in range(len(df)):
        current_price = df['close'].iloc[i]

        # Manage position
        if position is not None:
            entry_idx = position['entry_idx']
            entry_price = position['entry_price']
            hours_held = (i - entry_idx) / 12

            pnl_pct = (current_price - entry_price) / entry_price

            exit_reason = None
            if pnl_pct <= -STOP_LOSS:
                exit_reason = "Stop Loss"
            elif pnl_pct >= TAKE_PROFIT:
                exit_reason = "Take Profit"
            elif hours_held >= MAX_HOLDING_HOURS:
                exit_reason = "Max Holding"

            if exit_reason:
                quantity = position['quantity']
                pnl_usd = pnl_pct * (entry_price * quantity)

                entry_cost = entry_price * quantity * TRANSACTION_COST
                exit_cost = current_price * quantity * TRANSACTION_COST
                pnl_usd -= (entry_cost + exit_cost)

                capital += pnl_usd

                trades.append({
                    'entry_idx': entry_idx,
                    'exit_idx': i,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl_pct': pnl_pct,
                    'pnl_usd': pnl_usd,
                    'exit_reason': exit_reason,
                    'holding_hours': hours_held
                })

                position = None

        # Look for entry
        if position is None and i < len(df) - 1:
            features = df[feature_columns].iloc[i:i+1].values

            if np.isnan(features).any():
                continue

            probability = model.predict_proba(features)[0][1]

            if probability > entry_threshold:
                position_value = capital * POSITION_SIZE_PCT
                quantity = position_value / current_price

                position = {
                    'entry_idx': i,
                    'entry_price': current_price,
                    'quantity': quantity
                }

    # Calculate metrics
    if len(trades) == 0:
        return {
            'total_return_pct': 0.0,
            'num_trades': 0,
            'win_rate': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }

    total_return_pct = ((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    winning_trades = [t for t in trades if t['pnl_usd'] > 0]
    win_rate = (len(winning_trades) / len(trades)) * 100

    # Sharpe
    if len(trades) > 1:
        returns = [t['pnl_pct'] for t in trades]
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0.0
    else:
        sharpe = 0.0

    # Drawdown
    cumulative_returns = []
    running_capital = INITIAL_CAPITAL
    for trade in trades:
        running_capital += trade['pnl_usd']
        cumulative_returns.append(running_capital)

    if len(cumulative_returns) > 0:
        peak = cumulative_returns[0]
        max_dd = 0
        for value in cumulative_returns:
            if value > peak:
                peak = value
            dd = ((peak - value) / peak) * 100
            if dd > max_dd:
                max_dd = dd
    else:
        max_dd = 0.0

    return {
        'total_return_pct': total_return_pct,
        'num_trades': len(trades),
        'win_rate': win_rate,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd
    }


def backtest_regressor(df, model, feature_columns, entry_threshold_pnl=0.01):
    """Backtest regression model (predicts P&L directly)"""
    capital = INITIAL_CAPITAL
    position = None
    trades = []

    for i in range(len(df)):
        current_price = df['close'].iloc[i]

        # Manage position
        if position is not None:
            entry_idx = position['entry_idx']
            entry_price = position['entry_price']
            hours_held = (i - entry_idx) / 12

            pnl_pct = (current_price - entry_price) / entry_price

            exit_reason = None
            if pnl_pct <= -STOP_LOSS:
                exit_reason = "Stop Loss"
            elif pnl_pct >= TAKE_PROFIT:
                exit_reason = "Take Profit"
            elif hours_held >= MAX_HOLDING_HOURS:
                exit_reason = "Max Holding"

            if exit_reason:
                quantity = position['quantity']
                pnl_usd = pnl_pct * (entry_price * quantity)

                entry_cost = entry_price * quantity * TRANSACTION_COST
                exit_cost = current_price * quantity * TRANSACTION_COST
                pnl_usd -= (entry_cost + exit_cost)

                capital += pnl_usd

                trades.append({
                    'entry_idx': entry_idx,
                    'exit_idx': i,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl_pct': pnl_pct,
                    'pnl_usd': pnl_usd,
                    'exit_reason': exit_reason,
                    'holding_hours': hours_held
                })

                position = None

        # Look for entry (regression predicts P&L)
        if position is None and i < len(df) - 1:
            features = df[feature_columns].iloc[i:i+1].values

            if np.isnan(features).any():
                continue

            predicted_pnl = model.predict(features)[0]

            # Enter if predicted P&L > threshold
            if predicted_pnl > entry_threshold_pnl:
                position_value = capital * POSITION_SIZE_PCT
                quantity = position_value / current_price

                position = {
                    'entry_idx': i,
                    'entry_price': current_price,
                    'quantity': quantity
                }

    # Calculate metrics (same as classifier)
    if len(trades) == 0:
        return {
            'total_return_pct': 0.0,
            'num_trades': 0,
            'win_rate': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }

    total_return_pct = ((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    winning_trades = [t for t in trades if t['pnl_usd'] > 0]
    win_rate = (len(winning_trades) / len(trades)) * 100

    # Sharpe
    if len(trades) > 1:
        returns = [t['pnl_pct'] for t in trades]
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0.0
    else:
        sharpe = 0.0

    # Drawdown
    cumulative_returns = []
    running_capital = INITIAL_CAPITAL
    for trade in trades:
        running_capital += trade['pnl_usd']
        cumulative_returns.append(running_capital)

    if len(cumulative_returns) > 0:
        peak = cumulative_returns[0]
        max_dd = 0
        for value in cumulative_returns:
            if value > peak:
                peak = value
            dd = ((peak - value) / peak) * 100
            if dd > max_dd:
                max_dd = dd
    else:
        max_dd = 0.0

    return {
        'total_return_pct': total_return_pct,
        'num_trades': len(trades),
        'win_rate': win_rate,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd
    }


def rolling_window_backtest(df, model, feature_columns, model_type='classifier'):
    """Rolling window backtest"""
    windows = []
    start_idx = 0

    while start_idx + WINDOW_SIZE <= len(df):
        end_idx = start_idx + WINDOW_SIZE
        window_df = df.iloc[start_idx:end_idx].reset_index(drop=True)

        regime = classify_market_regime(window_df)

        # Choose backtest function based on model type
        if model_type == 'regressor':
            metrics = backtest_regressor(window_df, model, feature_columns)
        else:
            metrics = backtest_classifier(window_df, model, feature_columns, ENTRY_THRESHOLD)

        # Buy & Hold
        bh_start = window_df['close'].iloc[0]
        bh_end = window_df['close'].iloc[-1]
        bh_return = ((bh_end - bh_start) / bh_start) * 100
        bh_cost = 2 * TRANSACTION_COST * 100
        bh_return -= bh_cost

        windows.append({
            'start_idx': start_idx,
            'end_idx': end_idx,
            'regime': regime,
            'return': metrics['total_return_pct'],
            'bh_return': bh_return,
            'difference': metrics['total_return_pct'] - bh_return,
            'num_trades': metrics['num_trades'],
            'win_rate': metrics['win_rate'],
            'sharpe': metrics['sharpe_ratio'],
            'max_dd': metrics['max_drawdown']
        })

        start_idx += WINDOW_SIZE

    return pd.DataFrame(windows)


def main():
    logger.info("=" * 80)
    logger.info("Backtest All Labeling Methods - Comprehensive Comparison")
    logger.info("=" * 80)

    # Load data
    logger.info("\nLoading and preparing data...")
    data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
    df = pd.read_csv(data_file)
    logger.success(f"✅ Data loaded: {len(df)} candles")

    # Calculate features
    df = calculate_features(df)
    adv_features = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)
    df = adv_features.calculate_all_features(df)
    df = df.ffill().dropna()
    logger.success(f"✅ Features calculated: {len(df)} rows after NaN handling")

    # Define models to test
    models_to_test = [
        {
            'name': 'Baseline (Phase 4 Base)',
            'model_file': 'xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl',
            'features_file': 'xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt',
            'model_type': 'classifier',
            'expected_f1': 0.089
        },
        {
            'name': 'Realistic Labels',
            'model_file': 'xgboost_v4_realistic_labels.pkl',
            'features_file': 'xgboost_v4_realistic_labels_features.txt',
            'model_type': 'classifier',
            'expected_f1': 0.513
        },
        {
            'name': 'Regression',
            'model_file': 'xgboost_v4_regression.pkl',
            'features_file': 'xgboost_v4_regression_features.txt',
            'model_type': 'regressor',
            'expected_f1': None
        },
        {
            'name': 'With Regime',
            'model_file': 'xgboost_v4_with_regime.pkl',
            'features_file': 'xgboost_v4_with_regime_features.txt',
            'model_type': 'classifier',
            'expected_f1': 0.512
        }
    ]

    # Run backtests
    all_results = {}

    for model_info in models_to_test:
        model_name = model_info['name']
        model_path = MODELS_DIR / model_info['model_file']
        features_path = MODELS_DIR / model_info['features_file']

        if not model_path.exists():
            logger.warning(f"⚠️ {model_name}: Model not found, skipping...")
            continue

        logger.info(f"\n{'=' * 80}")
        logger.info(f"Testing: {model_name}")
        logger.info(f"{'=' * 80}")

        # Load model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        with open(features_path, 'r') as f:
            feature_columns = [line.strip() for line in f.readlines()]

        logger.info(f"  Model type: {model_info['model_type']}")
        logger.info(f"  Features: {len(feature_columns)}")
        if model_info['expected_f1']:
            logger.info(f"  Expected F1: {model_info['expected_f1']:.3f}")

        # Special handling for "With Regime" model
        df_test = df.copy()
        if model_name == 'With Regime':
            logger.info("  Loading K-Means and Scaler for regime calculation...")
            kmeans_path = MODELS_DIR / 'xgboost_v4_with_regime_kmeans.pkl'
            scaler_path = MODELS_DIR / 'xgboost_v4_with_regime_scaler.pkl'

            with open(kmeans_path, 'rb') as f:
                kmeans = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)

            # Calculate regime features
            logger.info("  Calculating market regime features...")
            regime_features = []
            for i in range(len(df_test)):
                if i < 20:
                    regime_features.append(0)  # Default for insufficient history
                else:
                    window = df_test.iloc[i-20:i]
                    returns = window['close'].pct_change().dropna()

                    features = np.array([[
                        returns.mean(),
                        returns.std(),
                        returns.std(),  # volatility
                        window['volume'].mean(),
                        abs(returns.mean())  # trend_strength
                    ]])

                    features_scaled = scaler.transform(features)
                    regime = kmeans.predict(features_scaled)[0]
                    regime_features.append(regime)

            df_test['market_regime'] = regime_features
            logger.success(f"  ✅ Market regime calculated: {len(set(regime_features))} unique regimes")

        # Run backtest
        results = rolling_window_backtest(df_test, model, feature_columns, model_info['model_type'])

        all_results[model_name] = results

        # Print summary
        logger.info(f"\n📊 Results ({len(results)} windows):")
        logger.info(f"  Returns: {results['return'].mean():.2f}% per 2 days")
        logger.info(f"  vs B&H: {results['difference'].mean():+.2f}%")
        logger.info(f"  Win Rate: {results['win_rate'].mean():.1f}%")
        logger.info(f"  Trades/window: {results['num_trades'].mean():.1f}")
        logger.info(f"  Sharpe: {results['sharpe'].mean():.2f}")
        logger.info(f"  Max DD: {results['max_dd'].mean():.2f}%")

    # Comparison table
    logger.info(f"\n{'=' * 80}")
    logger.info("FINAL COMPARISON")
    logger.info(f"{'=' * 80}")

    comparison = []
    for model_name, results in all_results.items():
        comparison.append({
            'Model': model_name,
            'Returns (2d)': results['return'].mean(),
            'vs B&H': results['difference'].mean(),
            'Win Rate': results['win_rate'].mean(),
            'Trades': results['num_trades'].mean(),
            'Sharpe': results['sharpe'].mean()
        })

    comp_df = pd.DataFrame(comparison)
    comp_df = comp_df.sort_values('Returns (2d)', ascending=False)

    logger.info("\n" + comp_df.to_string(index=False))

    # Find winner
    best_model = comp_df.iloc[0]
    logger.info(f"\n🏆 WINNER: {best_model['Model']}")
    logger.info(f"   Returns: {best_model['Returns (2d)']:.2f}% per 2 days")
    logger.info(f"   vs B&H: {best_model['vs B&H']:+.2f}%")
    logger.info(f"   Win Rate: {best_model['Win Rate']:.1f}%")

    # Save results
    output_file = RESULTS_DIR / "labeling_methods_comparison.csv"
    comp_df.to_csv(output_file, index=False)
    logger.success(f"\n✅ Comparison saved: {output_file}")

    logger.info(f"\n{'=' * 80}")
    logger.success("✅ All Backtests Complete!")
    logger.info(f"{'=' * 80}")


if __name__ == "__main__":
    main()
