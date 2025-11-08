"""
Threshold Sensitivity Analysis

Purpose: Analyze trade-off between signal quantity and quality at different thresholds

Methodology:
1. Test thresholds: 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
2. For each threshold, measure:
   - Number of trades (signal quantity)
   - Win rate (signal quality)
   - Returns
   - Risk metrics
3. Identify optimal threshold balancing quantity and quality

This addresses: "Is 0.7 the optimal threshold or can we improve?"
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys
import matplotlib.pyplot as plt

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

# Test these thresholds
TEST_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


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


def backtest_dual_model_strategy(df, long_model, short_model, long_scaler, short_scaler,
                                  feature_columns, threshold=0.7):
    """Backtest with dual-model strategy at given threshold"""
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

            # Get predictions from BOTH models
            features_scaled_long = long_scaler.transform(features)
            long_prob = long_model.predict_proba(features_scaled_long)[0][1]

            features_scaled_short = short_scaler.transform(features)
            short_prob = short_model.predict_proba(features_scaled_short)[0][1]

            # Determine entry direction
            side = None
            probability = None

            if long_prob >= threshold and short_prob < threshold:
                side = 'LONG'
                probability = long_prob
            elif short_prob >= threshold and long_prob < threshold:
                side = 'SHORT'
                probability = short_prob
            elif long_prob >= threshold and short_prob >= threshold:
                if long_prob > short_prob:
                    side = 'LONG'
                    probability = long_prob
                else:
                    side = 'SHORT'
                    probability = short_prob

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


def rolling_window_backtest(df, long_model, short_model, long_scaler, short_scaler,
                            feature_columns, threshold=0.7):
    """Rolling window backtest with dual models"""
    windows = []
    start_idx = 0

    while start_idx + WINDOW_SIZE <= len(df):
        end_idx = start_idx + WINDOW_SIZE
        window_df = df.iloc[start_idx:end_idx].reset_index(drop=True)

        regime = classify_market_regime(window_df)

        trades, metrics = backtest_dual_model_strategy(
            window_df, long_model, short_model, long_scaler, short_scaler,
            feature_columns, threshold
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
    print("THRESHOLD SENSITIVITY ANALYSIS")
    print("=" * 80)
    print("\n🎯 Purpose: Find optimal threshold balancing signal quantity vs quality")

    # Load models
    long_model_file = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl"
    long_scaler_file = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_scaler.pkl"
    short_model_file = MODELS_DIR / "xgboost_short_model_lookahead3_thresh0.3.pkl"
    short_scaler_file = MODELS_DIR / "xgboost_short_model_lookahead3_thresh0.3_scaler.pkl"

    with open(long_model_file, 'rb') as f:
        long_model = pickle.load(f)
    with open(long_scaler_file, 'rb') as f:
        long_scaler = pickle.load(f)
    with open(short_model_file, 'rb') as f:
        short_model = pickle.load(f)
    with open(short_scaler_file, 'rb') as f:
        short_scaler = pickle.load(f)
    print(f"✅ Models loaded")

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

    # Test each threshold
    print("\n" + "=" * 80)
    print(f"Testing Thresholds: {TEST_THRESHOLDS}")
    print("=" * 80)

    all_results = []

    for threshold in TEST_THRESHOLDS:
        print(f"\n🔍 Testing threshold {threshold}...")

        results = rolling_window_backtest(
            df=df,
            long_model=long_model,
            short_model=short_model,
            long_scaler=long_scaler,
            short_scaler=short_scaler,
            feature_columns=feature_columns,
            threshold=threshold
        )

        summary = {
            'threshold': threshold,
            'avg_return': results['xgb_return'].mean(),
            'std_return': results['xgb_return'].std(),
            'avg_trades': results['num_trades'].mean(),
            'avg_win_rate': results['win_rate'].mean(),
            'avg_sharpe': results['sharpe'].mean(),
            'avg_max_dd': results['max_dd'].mean(),
            'pct_long': results['num_long'].sum() / results['num_trades'].sum() * 100 if results['num_trades'].sum() > 0 else 0,
            'pct_short': results['num_short'].sum() / results['num_trades'].sum() * 100 if results['num_trades'].sum() > 0 else 0
        }

        all_results.append(summary)

        print(f"   Return: {summary['avg_return']:+.2f}% | Trades: {summary['avg_trades']:.1f} | WR: {summary['avg_win_rate']:.1f}% | Sharpe: {summary['avg_sharpe']:.2f}")

    # Create summary DataFrame
    summary_df = pd.DataFrame(all_results)

    # Display results
    print("\n" + "=" * 80)
    print("📊 THRESHOLD SENSITIVITY RESULTS")
    print("=" * 80)

    print("\n" + summary_df.to_string(index=False))

    # Find optimal threshold
    print("\n" + "=" * 80)
    print("🎯 OPTIMAL THRESHOLD ANALYSIS")
    print("=" * 80)

    # Best by return
    best_return = summary_df.loc[summary_df['avg_return'].idxmax()]
    print(f"\n  Best Return: Threshold {best_return['threshold']}")
    print(f"    Return: {best_return['avg_return']:+.2f}%")
    print(f"    Trades: {best_return['avg_trades']:.1f}")
    print(f"    Win Rate: {best_return['avg_win_rate']:.1f}%")

    # Best by Sharpe
    best_sharpe = summary_df.loc[summary_df['avg_sharpe'].idxmax()]
    print(f"\n  Best Sharpe: Threshold {best_sharpe['threshold']}")
    print(f"    Return: {best_sharpe['avg_return']:+.2f}%")
    print(f"    Sharpe: {best_sharpe['avg_sharpe']:.2f}")
    print(f"    Trades: {best_sharpe['avg_trades']:.1f}")

    # Best by Win Rate
    best_wr = summary_df.loc[summary_df['avg_win_rate'].idxmax()]
    print(f"\n  Best Win Rate: Threshold {best_wr['threshold']}")
    print(f"    Win Rate: {best_wr['avg_win_rate']:.1f}%")
    print(f"    Return: {best_wr['avg_return']:+.2f}%")
    print(f"    Trades: {best_wr['avg_trades']:.1f}")

    # Trade-off analysis
    print("\n" + "=" * 80)
    print("📈 TRADE-OFF ANALYSIS")
    print("=" * 80)

    print("\n  As threshold increases:")
    print(f"    Trades:   {summary_df['avg_trades'].iloc[0]:.1f} → {summary_df['avg_trades'].iloc[-1]:.1f}")
    print(f"    Win Rate: {summary_df['avg_win_rate'].iloc[0]:.1f}% → {summary_df['avg_win_rate'].iloc[-1]:.1f}%")
    print(f"    Return:   {summary_df['avg_return'].iloc[0]:+.2f}% → {summary_df['avg_return'].iloc[-1]:+.2f}%")

    # Recommendation
    print("\n" + "=" * 80)
    print("💡 RECOMMENDATION")
    print("=" * 80)

    # Score each threshold: return * 0.4 + win_rate * 0.3 + sharpe * 0.3
    summary_df['score'] = (
        summary_df['avg_return'] * 0.4 +
        summary_df['avg_win_rate'] * 0.3 +
        summary_df['avg_sharpe'] * 0.3
    )

    best_overall = summary_df.loc[summary_df['score'].idxmax()]

    print(f"\n  Recommended Threshold: {best_overall['threshold']}")
    print(f"    Return: {best_overall['avg_return']:+.2f}%")
    print(f"    Win Rate: {best_overall['avg_win_rate']:.1f}%")
    print(f"    Sharpe: {best_overall['avg_sharpe']:.2f}")
    print(f"    Trades: {best_overall['avg_trades']:.1f}")

    if best_overall['threshold'] == 0.7:
        print("\n  ✅ Current threshold (0.7) is optimal!")
    elif best_overall['threshold'] < 0.7:
        print(f"\n  📊 Consider lowering threshold to {best_overall['threshold']} for:")
        print(f"     - More trades ({best_overall['avg_trades']:.1f} vs {summary_df[summary_df['threshold']==0.7]['avg_trades'].values[0]:.1f})")
        print(f"     - Better returns ({best_overall['avg_return']:+.2f}% vs {summary_df[summary_df['threshold']==0.7]['avg_return'].values[0]:+.2f}%)")
    else:
        print(f"\n  🎯 Consider raising threshold to {best_overall['threshold']} for:")
        print(f"     - Higher win rate ({best_overall['avg_win_rate']:.1f}% vs {summary_df[summary_df['threshold']==0.7]['avg_win_rate'].values[0]:.1f}%)")
        print(f"     - Better risk/reward")

    # Save results
    output_file = RESULTS_DIR / "threshold_sensitivity_analysis.csv"
    summary_df.to_csv(output_file, index=False)
    print(f"\n✅ Results saved: {output_file}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
