"""
Backtest Phase 4 Models with Mainnet Data + MinMaxScaler

CRITICAL FIX:
- Load and apply MinMaxScaler before predictions
- Models trained with normalized data need normalized input
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys

# Project setup
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Trading parameters
WINDOW_SIZE = 1440  # 5 days
INITIAL_CAPITAL = 10000.0
POSITION_SIZE_PCT = 0.95
STOP_LOSS = 0.01  # 1%
TAKE_PROFIT = 0.03  # 3%
MAX_HOLDING_HOURS = 4
TRANSACTION_COST = 0.0002  # 0.02% maker fee


def classify_market_regime(df_window):
    """Classify market regime based on price movement"""
    start_price = df_window['close'].iloc[0]
    end_price = df_window['close'].iloc[-1]
    return_pct = ((end_price / start_price) - 1) * 100
    if return_pct > 3.0:
        return "Bull"
    elif return_pct < -2.0:
        return "Bear"
    else:
        return "Sideways"


def backtest_longshort_strategy(df, model, scaler, feature_columns,
                                 long_threshold=0.7, short_threshold=0.3):
    """
    Backtest LONG + SHORT strategy with MinMaxScaler applied

    Args:
        df: DataFrame with features
        model: Trained XGBoost model
        scaler: MinMaxScaler fitted on training data
        feature_columns: List of feature names
        long_threshold: Probability threshold for LONG entry
        short_threshold: Probability threshold for SHORT entry (prob <= this)
    """
    capital = INITIAL_CAPITAL
    position = None
    trades = []

    for i in range(len(df)):
        current_price = df['close'].iloc[i]

        # Manage existing position
        if position is not None:
            side = position['side']
            entry_idx = position['entry_idx']
            entry_price = position['entry_price']
            hours_held = (i - entry_idx) / 12

            # Calculate P&L based on position side
            if side == 'LONG':
                pnl_pct = (current_price - entry_price) / entry_price
            else:  # SHORT
                pnl_pct = (entry_price - current_price) / entry_price

            # Check exit conditions
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

                # Transaction costs
                entry_cost = entry_price * quantity * TRANSACTION_COST
                exit_cost = current_price * quantity * TRANSACTION_COST
                pnl_usd -= (entry_cost + exit_cost)

                capital += pnl_usd

                trades.append({
                    'side': side,
                    'entry_idx': entry_idx,
                    'exit_idx': i,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl_pct': pnl_pct,
                    'pnl_usd': pnl_usd,
                    'exit_reason': exit_reason,
                    'holding_hours': hours_held,
                    'probability': position['probability'],
                    'regime': position['regime']
                })

                position = None

        # Look for entry (LONG or SHORT)
        if position is None and i < len(df) - 1:
            features = df[feature_columns].iloc[i:i+1].values

            if np.isnan(features).any():
                continue

            # ✅ CRITICAL: Apply scaler before prediction
            features_scaled = scaler.transform(features)
            probability = model.predict_proba(features_scaled)[0][1]

            # Determine entry direction
            side = None
            signal_strength = None

            if probability >= long_threshold:
                # LONG signal
                side = 'LONG'
                signal_strength = probability
            elif probability <= short_threshold:
                # SHORT signal
                side = 'SHORT'
                signal_strength = 1 - probability

            if side is not None:
                position_value = capital * POSITION_SIZE_PCT
                quantity = position_value / current_price
                current_regime = classify_market_regime(df.iloc[max(0, i-20):i+1])

                position = {
                    'side': side,
                    'entry_idx': i,
                    'entry_price': current_price,
                    'quantity': quantity,
                    'probability': probability,
                    'signal_strength': signal_strength,
                    'regime': current_regime
                }

    # Calculate metrics
    if len(trades) == 0:
        return trades, {
            'total_return_pct': 0.0,
            'num_trades': 0,
            'num_long': 0,
            'num_short': 0,
            'win_rate': 0.0,
            'win_rate_long': 0.0,
            'win_rate_short': 0.0,
            'avg_pnl_pct': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }

    total_return_pct = ((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100

    # Overall metrics
    winning_trades = [t for t in trades if t['pnl_usd'] > 0]
    win_rate = (len(winning_trades) / len(trades)) * 100
    avg_pnl_pct = np.mean([t['pnl_pct'] for t in trades]) * 100

    # LONG vs SHORT breakdown
    long_trades = [t for t in trades if t['side'] == 'LONG']
    short_trades = [t for t in trades if t['side'] == 'SHORT']

    win_rate_long = (len([t for t in long_trades if t['pnl_usd'] > 0]) / len(long_trades) * 100) if len(long_trades) > 0 else 0.0
    win_rate_short = (len([t for t in short_trades if t['pnl_usd'] > 0]) / len(short_trades) * 100) if len(short_trades) > 0 else 0.0

    # Drawdown
    cumulative_returns = []
    running_capital = INITIAL_CAPITAL
    for trade in trades:
        running_capital += trade['pnl_usd']
        cumulative_returns.append(running_capital)

    peak = cumulative_returns[0]
    max_dd = 0
    for value in cumulative_returns:
        if value > peak:
            peak = value
        dd = ((peak - value) / peak) * 100
        if dd > max_dd:
            max_dd = dd

    # Sharpe
    if len(trades) > 1:
        returns = [t['pnl_pct'] for t in trades]
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0.0
    else:
        sharpe = 0.0

    metrics = {
        'total_return_pct': total_return_pct,
        'num_trades': len(trades),
        'num_long': len(long_trades),
        'num_short': len(short_trades),
        'win_rate': win_rate,
        'win_rate_long': win_rate_long,
        'win_rate_short': win_rate_short,
        'avg_pnl_pct': avg_pnl_pct,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd
    }

    return trades, metrics


def rolling_window_backtest(df, model, scaler, feature_columns,
                            long_threshold=0.7, short_threshold=0.3):
    """Rolling window backtest"""
    windows = []
    start_idx = 0

    while start_idx + WINDOW_SIZE <= len(df):
        end_idx = start_idx + WINDOW_SIZE
        window_df = df.iloc[start_idx:end_idx].reset_index(drop=True)

        regime = classify_market_regime(window_df)

        trades, metrics = backtest_longshort_strategy(
            window_df, model, scaler, feature_columns, long_threshold, short_threshold
        )

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
            'xgb_return': metrics['total_return_pct'],
            'bh_return': bh_return,
            'difference': metrics['total_return_pct'] - bh_return,
            'num_trades': metrics['num_trades'],
            'num_long': metrics['num_long'],
            'num_short': metrics['num_short'],
            'win_rate': metrics['win_rate'],
            'win_rate_long': metrics['win_rate_long'],
            'win_rate_short': metrics['win_rate_short'],
            'sharpe': metrics['sharpe_ratio'],
            'max_dd': metrics['max_drawdown']
        })

        start_idx += WINDOW_SIZE

    return pd.DataFrame(windows)


def main():
    print("=" * 80)
    print("Mainnet Backtest with MinMaxScaler Applied")
    print("=" * 80)

    # Load model
    model_file = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl"
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    print(f"✅ Model loaded: {model_file.name}")

    # Load scaler
    scaler_file = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_scaler.pkl"
    with open(scaler_file, 'rb') as f:
        scaler = pickle.load(f)
    print(f"✅ Scaler loaded: {scaler_file.name} (range: {scaler.feature_range})")

    # Load feature columns
    feature_file = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt"
    with open(feature_file, 'r') as f:
        feature_columns = [line.strip() for line in f.readlines()]
    print(f"✅ Features loaded: {len(feature_columns)} features")

    # Load data
    data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
    df = pd.read_csv(data_file)
    print(f"✅ Data loaded: {len(df)} candles")

    # Calculate features
    print("\nCalculating features...")
    df = calculate_features(df)
    adv_features = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)
    df = adv_features.calculate_all_features(df)
    df = df.ffill().dropna()
    print(f"✅ Features calculated: {len(df)} rows")

    # Analyze probability distribution with scaler
    print("\n" + "=" * 80)
    print("Probability Distribution Analysis (with scaler)")
    print("=" * 80)

    features_scaled = scaler.transform(df[feature_columns].values)
    probabilities = model.predict_proba(features_scaled)[:, 1]

    print(f"Mean: {np.mean(probabilities):.4f}")
    print(f"Std: {np.std(probabilities):.4f}")
    print(f"Min: {np.min(probabilities):.4f}")
    print(f"Max: {np.max(probabilities):.4f}")
    print("\nThreshold Distribution:")
    for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
        count = np.sum(probabilities >= threshold)
        pct = (count / len(probabilities)) * 100
        print(f"  Prob >= {threshold:.1f}: {count:,} ({pct:.2f}%)")

    # Run backtest
    print("\n" + "=" * 80)
    print("Running Backtest: LONG + SHORT Strategy")
    print("=" * 80)
    print(f"LONG Threshold: 0.7")
    print(f"SHORT Threshold: 0.3")

    results = rolling_window_backtest(
        df=df,
        model=model,
        scaler=scaler,
        feature_columns=feature_columns,
        long_threshold=0.7,
        short_threshold=0.3
    )

    # Summary
    print(f"\n📊 Results ({len(results)} windows):")
    print(f"  XGBoost Return: {results['xgb_return'].mean():.2f}% ± {results['xgb_return'].std():.2f}%")
    print(f"  Buy & Hold Return: {results['bh_return'].mean():.2f}% ± {results['bh_return'].std():.2f}%")
    print(f"  Difference: {results['difference'].mean():.2f}% ± {results['difference'].std():.2f}%")
    print(f"\n  📊 Trade Breakdown:")
    print(f"    Total Trades: {results['num_trades'].mean():.1f}")
    print(f"    LONG Trades: {results['num_long'].mean():.1f} ({results['num_long'].sum() / results['num_trades'].sum() * 100:.1f}%)")
    print(f"    SHORT Trades: {results['num_short'].mean():.1f} ({results['num_short'].sum() / results['num_trades'].sum() * 100:.1f}%)")
    print(f"\n  🎯 Win Rates:")
    print(f"    Overall: {results['win_rate'].mean():.1f}%")
    print(f"    LONG: {results['win_rate_long'].mean():.1f}%")
    print(f"    SHORT: {results['win_rate_short'].mean():.1f}%")
    print(f"\n  📈 Risk Metrics:")
    print(f"    Sharpe: {results['sharpe'].mean():.3f}")
    print(f"    Max DD: {results['max_dd'].mean():.2f}%")

    # By regime
    print(f"\n{'=' * 80}")
    print(f"Performance by Market Regime:")
    print(f"{'=' * 80}")
    for regime in ['Bull', 'Bear', 'Sideways']:
        regime_df = results[results['regime'] == regime]
        if len(regime_df) > 0:
            print(f"\n  {regime} ({len(regime_df)} windows):")
            print(f"    XGBoost: {regime_df['xgb_return'].mean():.2f}%")
            print(f"    Buy & Hold: {regime_df['bh_return'].mean():.2f}%")
            print(f"    Trades: {regime_df['num_trades'].mean():.1f} (LONG: {regime_df['num_long'].mean():.1f}, SHORT: {regime_df['num_short'].mean():.1f})")
            print(f"    Win Rate: {regime_df['win_rate'].mean():.1f}%")

    # Save results
    output_file = RESULTS_DIR / "backtest_mainnet_with_scaler.csv"
    results.to_csv(output_file, index=False)
    print(f"\n✅ Results saved: {output_file}")

    print(f"\n{'=' * 80}")
    print("Backtest Complete!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
