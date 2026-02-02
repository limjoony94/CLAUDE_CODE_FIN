"""
Backtest Exit Models Comparison

Compare:
1. Entry Dual + Exit Rules (current system)
2. Entry Dual + Exit ML (with LONG/SHORT exit models)

Goal: Determine if ML-learned exits beat fixed rules
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

# Trading parameters
WINDOW_SIZE = 576  # 2 days
INITIAL_CAPITAL = 10000.0
POSITION_SIZE_PCT = 0.95
STOP_LOSS = 0.01
TAKE_PROFIT = 0.03
MAX_HOLDING_HOURS = 4
TRANSACTION_COST = 0.0002
ENTRY_THRESHOLD = 0.7
EXIT_THRESHOLD = 0.75  # For ML exit models


def calculate_position_features(current_idx, entry_idx, entry_price, df, trade_direction):
    """
    Calculate position-specific features for exit model

    Returns:
        dict with 8 position features
    """
    offset = current_idx - entry_idx

    # Feature 1: Time held (hours)
    time_held = offset / 12  # 5min candles, 12 = 1 hour

    # Feature 2: Current P&L percentage
    current_price = df['close'].iloc[current_idx]
    if trade_direction == "LONG":
        current_pnl_pct = (current_price - entry_price) / entry_price
    else:  # SHORT
        current_pnl_pct = (entry_price - current_price) / entry_price

    # Feature 3-4: Peak and Trough P&L so far
    pnl_peak = -999999
    pnl_trough = 999999
    for i in range(entry_idx + 1, current_idx + 1):
        price = df['close'].iloc[i]
        if trade_direction == "LONG":
            pnl = (price - entry_price) / entry_price
        else:
            pnl = (entry_price - price) / entry_price

        if pnl > pnl_peak:
            pnl_peak = pnl
        if pnl < pnl_trough:
            pnl_trough = pnl

    # Feature 5: P&L from peak (drawdown)
    pnl_from_peak = current_pnl_pct - pnl_peak

    # Feature 6: Volatility since entry
    entry_to_current = df['close'].iloc[entry_idx:current_idx+1]
    returns = entry_to_current.pct_change().dropna()
    volatility_since_entry = returns.std() if len(returns) > 1 else 0

    # Feature 7: Volume change
    entry_volume = df['volume'].iloc[entry_idx]
    current_volume = df['volume'].iloc[current_idx]
    volume_change = (current_volume - entry_volume) / entry_volume if entry_volume > 0 else 0

    # Feature 8: Momentum shift
    if current_idx >= 3:
        recent_prices = df['close'].iloc[current_idx-2:current_idx+1]
        recent_returns = recent_prices.pct_change().dropna()
        momentum_shift = recent_returns.mean() if len(recent_returns) > 0 else 0
    else:
        momentum_shift = 0

    return {
        'time_held': time_held,
        'current_pnl_pct': current_pnl_pct,
        'pnl_peak': pnl_peak,
        'pnl_trough': pnl_trough,
        'pnl_from_peak': pnl_from_peak,
        'volatility_since_entry': volatility_since_entry,
        'volume_change': volume_change,
        'momentum_shift': momentum_shift
    }


def backtest_with_rules(df, entry_model, feature_columns):
    """Backtest with rule-based exits (SL/TP/Max Hold)"""
    capital = INITIAL_CAPITAL
    position = None
    trades = []

    for i in range(len(df)):
        current_price = df['close'].iloc[i]

        # Manage position
        if position is not None:
            entry_idx = position['entry_idx']
            entry_price = position['entry_price']
            direction = position['direction']
            hours_held = (i - entry_idx) / 12

            # Calculate P&L based on direction
            if direction == "LONG":
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

                entry_cost = entry_price * quantity * TRANSACTION_COST
                exit_cost = current_price * quantity * TRANSACTION_COST
                pnl_usd -= (entry_cost + exit_cost)

                capital += pnl_usd

                trades.append({
                    'entry_idx': entry_idx,
                    'exit_idx': i,
                    'direction': direction,
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

            probability = entry_model.predict_proba(features)[0][1]

            if probability > ENTRY_THRESHOLD:
                # Determine direction (using same model for simplicity)
                # In real system, would use LONG/SHORT entry models
                direction = "LONG"  # Simplified

                position_value = capital * POSITION_SIZE_PCT
                quantity = position_value / current_price

                position = {
                    'entry_idx': i,
                    'entry_price': current_price,
                    'direction': direction,
                    'quantity': quantity
                }

    # Calculate metrics
    if len(trades) == 0:
        return {
            'total_return_pct': 0.0,
            'num_trades': 0,
            'win_rate': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'avg_holding_hours': 0.0,
            'exit_reasons': {}
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

    # Average holding time
    avg_holding_hours = np.mean([t['holding_hours'] for t in trades])

    # Exit reasons distribution
    exit_reasons = {}
    for trade in trades:
        reason = trade['exit_reason']
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

    return {
        'total_return_pct': total_return_pct,
        'num_trades': len(trades),
        'win_rate': win_rate,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'avg_holding_hours': avg_holding_hours,
        'exit_reasons': exit_reasons
    }


def backtest_with_ml_exits(df, entry_model, long_exit_model, short_exit_model,
                           entry_features, exit_features):
    """Backtest with ML-learned exits"""
    capital = INITIAL_CAPITAL
    position = None
    trades = []

    for i in range(len(df)):
        current_price = df['close'].iloc[i]

        # Manage position
        if position is not None:
            entry_idx = position['entry_idx']
            entry_price = position['entry_price']
            direction = position['direction']
            hours_held = (i - entry_idx) / 12

            # Calculate P&L based on direction
            if direction == "LONG":
                pnl_pct = (current_price - entry_price) / entry_price
            else:  # SHORT
                pnl_pct = (entry_price - current_price) / entry_price

            # Hard stops (safety)
            exit_reason = None
            if pnl_pct <= -0.015:  # -1.5% hard stop
                exit_reason = "Hard Stop Loss"
            elif pnl_pct >= 0.035:  # +3.5% hard take profit
                exit_reason = "Hard Take Profit"
            elif hours_held >= MAX_HOLDING_HOURS:
                exit_reason = "Max Holding"

            # ML Exit Decision (if no hard stops)
            if exit_reason is None:
                # Get base features (only those in exit model)
                # Exit model was trained with 36 base features (volume_ma_ratio removed from entry's 37)
                exit_base_features = [f for f in exit_features if f not in [
                    'time_held', 'current_pnl_pct', 'pnl_peak', 'pnl_trough',
                    'pnl_from_peak', 'volatility_since_entry', 'volume_change', 'momentum_shift'
                ]]

                base_features = df[exit_base_features].iloc[i].values

                # Calculate position features
                pos_features = calculate_position_features(
                    current_idx=i,
                    entry_idx=entry_idx,
                    entry_price=entry_price,
                    df=df,
                    trade_direction=direction
                )

                # Combine features (match training order)
                combined_features = np.concatenate([
                    base_features,
                    [pos_features[f] for f in [
                        'time_held', 'current_pnl_pct', 'pnl_peak', 'pnl_trough',
                        'pnl_from_peak', 'volatility_since_entry', 'volume_change', 'momentum_shift'
                    ]]
                ]).reshape(1, -1)

                # Predict exit
                if direction == "LONG":
                    exit_prob = long_exit_model.predict_proba(combined_features)[0][1]
                else:  # SHORT
                    exit_prob = short_exit_model.predict_proba(combined_features)[0][1]

                if exit_prob >= EXIT_THRESHOLD:
                    exit_reason = "ML Exit"

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
                    'direction': direction,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl_pct': pnl_pct,
                    'pnl_usd': pnl_usd,
                    'exit_reason': exit_reason,
                    'holding_hours': hours_held
                })

                position = None

        # Look for entry (same as rule-based)
        if position is None and i < len(df) - 1:
            features = df[entry_features].iloc[i:i+1].values

            if np.isnan(features).any():
                continue

            probability = entry_model.predict_proba(features)[0][1]

            if probability > ENTRY_THRESHOLD:
                direction = "LONG"  # Simplified

                position_value = capital * POSITION_SIZE_PCT
                quantity = position_value / current_price

                position = {
                    'entry_idx': i,
                    'entry_price': current_price,
                    'direction': direction,
                    'quantity': quantity
                }

    # Calculate metrics (same as rule-based)
    if len(trades) == 0:
        return {
            'total_return_pct': 0.0,
            'num_trades': 0,
            'win_rate': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'avg_holding_hours': 0.0,
            'exit_reasons': {}
        }

    total_return_pct = ((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    winning_trades = [t for t in trades if t['pnl_usd'] > 0]
    win_rate = (len(winning_trades) / len(trades)) * 100

    if len(trades) > 1:
        returns = [t['pnl_pct'] for t in trades]
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0.0
    else:
        sharpe = 0.0

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

    avg_holding_hours = np.mean([t['holding_hours'] for t in trades])

    exit_reasons = {}
    for trade in trades:
        reason = trade['exit_reason']
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

    return {
        'total_return_pct': total_return_pct,
        'num_trades': len(trades),
        'win_rate': win_rate,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'avg_holding_hours': avg_holding_hours,
        'exit_reasons': exit_reasons
    }


def rolling_window_backtest(df, entry_model, long_exit_model, short_exit_model,
                            entry_features, exit_features, use_ml_exit=False):
    """Rolling window backtest"""
    windows = []
    start_idx = 0

    while start_idx + WINDOW_SIZE <= len(df):
        end_idx = start_idx + WINDOW_SIZE
        window_df = df.iloc[start_idx:end_idx].reset_index(drop=True)

        # Choose backtest method
        if use_ml_exit:
            metrics = backtest_with_ml_exits(
                window_df, entry_model, long_exit_model, short_exit_model,
                entry_features, exit_features
            )
        else:
            metrics = backtest_with_rules(window_df, entry_model, entry_features)

        windows.append({
            'start_idx': start_idx,
            'end_idx': end_idx,
            'return': metrics['total_return_pct'],
            'num_trades': metrics['num_trades'],
            'win_rate': metrics['win_rate'],
            'sharpe': metrics['sharpe_ratio'],
            'max_dd': metrics['max_drawdown'],
            'avg_holding_hours': metrics['avg_holding_hours'],
            'exit_reasons': metrics['exit_reasons']
        })

        start_idx += WINDOW_SIZE

    return pd.DataFrame(windows)


def main():
    logger.info("=" * 80)
    logger.info("Backtest Exit Models Comparison")
    logger.info("=" * 80)

    # Load data
    logger.info("\nLoading data...")
    data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
    df = pd.read_csv(data_file)
    logger.success(f"✅ Data loaded: {len(df)} candles")

    # Calculate features
    logger.info("Calculating features...")
    df = calculate_features(df)
    adv_features = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)
    df = adv_features.calculate_all_features(df)
    df = df.ffill().dropna()
    logger.success(f"✅ Features calculated: {len(df)} rows")

    # Load entry model
    entry_model_path = MODELS_DIR / "xgboost_v4_realistic_labels.pkl"
    entry_features_path = MODELS_DIR / "xgboost_v4_realistic_labels_features.txt"

    with open(entry_model_path, 'rb') as f:
        entry_model = pickle.load(f)

    with open(entry_features_path, 'r') as f:
        entry_features = [line.strip() for line in f.readlines()]

    logger.info(f"Entry model: {entry_model_path.name}")
    logger.info(f"Entry features: {len(entry_features)}")

    # Load exit models
    long_exit_path = MODELS_DIR / "xgboost_v4_long_exit.pkl"
    short_exit_path = MODELS_DIR / "xgboost_v4_short_exit.pkl"
    exit_features_path = MODELS_DIR / "xgboost_v4_long_exit_features.txt"

    with open(long_exit_path, 'rb') as f:
        long_exit_model = pickle.load(f)

    with open(short_exit_path, 'rb') as f:
        short_exit_model = pickle.load(f)

    with open(exit_features_path, 'r') as f:
        exit_features = [line.strip() for line in f.readlines()]

    logger.info(f"LONG exit model: {long_exit_path.name}")
    logger.info(f"SHORT exit model: {short_exit_path.name}")
    logger.info(f"Exit features: {len(exit_features)}")

    # Test 1: Rule-based exits
    logger.info(f"\n{'=' * 80}")
    logger.info("Test 1: Entry Dual + Exit Rules (Current System)")
    logger.info(f"{'=' * 80}")

    results_rules = rolling_window_backtest(
        df=df,
        entry_model=entry_model,
        long_exit_model=None,
        short_exit_model=None,
        entry_features=entry_features,
        exit_features=None,
        use_ml_exit=False
    )

    logger.info(f"\n📊 Results ({len(results_rules)} windows):")
    logger.info(f"  Returns: {results_rules['return'].mean():.2f}% per 2 days")
    logger.info(f"  Win Rate: {results_rules['win_rate'].mean():.1f}%")
    logger.info(f"  Trades/window: {results_rules['num_trades'].mean():.1f}")
    logger.info(f"  Sharpe: {results_rules['sharpe'].mean():.2f}")
    logger.info(f"  Max DD: {results_rules['max_dd'].mean():.2f}%")
    logger.info(f"  Avg Holding: {results_rules['avg_holding_hours'].mean():.2f}h")

    # Exit reasons
    all_exit_reasons = {}
    for reasons_dict in results_rules['exit_reasons']:
        for reason, count in reasons_dict.items():
            all_exit_reasons[reason] = all_exit_reasons.get(reason, 0) + count

    total_exits = sum(all_exit_reasons.values())
    logger.info(f"\n  Exit Reasons:")
    for reason, count in sorted(all_exit_reasons.items(), key=lambda x: x[1], reverse=True):
        pct = (count / total_exits) * 100 if total_exits > 0 else 0
        logger.info(f"    {reason}: {count} ({pct:.1f}%)")

    # Test 2: ML-based exits
    logger.info(f"\n{'=' * 80}")
    logger.info("Test 2: Entry Dual + Exit ML (New System)")
    logger.info(f"{'=' * 80}")

    results_ml = rolling_window_backtest(
        df=df,
        entry_model=entry_model,
        long_exit_model=long_exit_model,
        short_exit_model=short_exit_model,
        entry_features=entry_features,
        exit_features=exit_features,
        use_ml_exit=True
    )

    logger.info(f"\n📊 Results ({len(results_ml)} windows):")
    logger.info(f"  Returns: {results_ml['return'].mean():.2f}% per 2 days")
    logger.info(f"  Win Rate: {results_ml['win_rate'].mean():.1f}%")
    logger.info(f"  Trades/window: {results_ml['num_trades'].mean():.1f}")
    logger.info(f"  Sharpe: {results_ml['sharpe'].mean():.2f}")
    logger.info(f"  Max DD: {results_ml['max_dd'].mean():.2f}%")
    logger.info(f"  Avg Holding: {results_ml['avg_holding_hours'].mean():.2f}h")

    # Exit reasons
    all_exit_reasons_ml = {}
    for reasons_dict in results_ml['exit_reasons']:
        for reason, count in reasons_dict.items():
            all_exit_reasons_ml[reason] = all_exit_reasons_ml.get(reason, 0) + count

    total_exits_ml = sum(all_exit_reasons_ml.values())
    logger.info(f"\n  Exit Reasons:")
    for reason, count in sorted(all_exit_reasons_ml.items(), key=lambda x: x[1], reverse=True):
        pct = (count / total_exits_ml) * 100 if total_exits_ml > 0 else 0
        logger.info(f"    {reason}: {count} ({pct:.1f}%)")

    # Comparison
    logger.info(f"\n{'=' * 80}")
    logger.info("COMPARISON: Rule-based vs ML-based Exits")
    logger.info(f"{'=' * 80}")

    returns_improvement = ((results_ml['return'].mean() - results_rules['return'].mean()) / abs(results_rules['return'].mean())) * 100
    winrate_diff = results_ml['win_rate'].mean() - results_rules['win_rate'].mean()
    sharpe_improvement = ((results_ml['sharpe'].mean() - results_rules['sharpe'].mean()) / results_rules['sharpe'].mean()) * 100
    holding_time_reduction = ((results_rules['avg_holding_hours'].mean() - results_ml['avg_holding_hours'].mean()) / results_rules['avg_holding_hours'].mean()) * 100

    logger.info(f"\nReturns:")
    logger.info(f"  Rule-based: {results_rules['return'].mean():.2f}%")
    logger.info(f"  ML-based: {results_ml['return'].mean():.2f}%")
    logger.info(f"  Improvement: {returns_improvement:+.1f}%")

    logger.info(f"\nWin Rate:")
    logger.info(f"  Rule-based: {results_rules['win_rate'].mean():.1f}%")
    logger.info(f"  ML-based: {results_ml['win_rate'].mean():.1f}%")
    logger.info(f"  Difference: {winrate_diff:+.1f}%")

    logger.info(f"\nSharpe Ratio:")
    logger.info(f"  Rule-based: {results_rules['sharpe'].mean():.2f}")
    logger.info(f"  ML-based: {results_ml['sharpe'].mean():.2f}")
    logger.info(f"  Improvement: {sharpe_improvement:+.1f}%")

    logger.info(f"\nAverage Holding Time:")
    logger.info(f"  Rule-based: {results_rules['avg_holding_hours'].mean():.2f}h")
    logger.info(f"  ML-based: {results_ml['avg_holding_hours'].mean():.2f}h")
    logger.info(f"  Reduction: {holding_time_reduction:.1f}%")

    # Decision
    logger.info(f"\n{'=' * 80}")
    logger.info("DECISION")
    logger.info(f"{'=' * 80}")

    if returns_improvement > 10 and winrate_diff > 5:
        logger.success("\n✅ DEPLOY ML EXIT MODELS!")
        logger.success("   ML exits significantly outperform rule-based exits")
        logger.success(f"   Returns: +{returns_improvement:.1f}%")
        logger.success(f"   Win Rate: +{winrate_diff:.1f}%")
    elif returns_improvement > 5:
        logger.warning("\n⚠️ MARGINAL IMPROVEMENT")
        logger.warning("   ML exits show improvement but not overwhelming")
        logger.warning(f"   Consider: Tune threshold ({EXIT_THRESHOLD}) or keep rules")
    else:
        logger.error("\n❌ KEEP RULE-BASED EXITS")
        logger.error("   ML exits do not justify added complexity")
        logger.error(f"   Returns improvement: {returns_improvement:+.1f}% (target: >10%)")

    # Save results
    comparison_df = pd.DataFrame({
        'System': ['Rule-based', 'ML-based'],
        'Returns': [results_rules['return'].mean(), results_ml['return'].mean()],
        'Win Rate': [results_rules['win_rate'].mean(), results_ml['win_rate'].mean()],
        'Sharpe': [results_rules['sharpe'].mean(), results_ml['sharpe'].mean()],
        'Avg Hold (h)': [results_rules['avg_holding_hours'].mean(), results_ml['avg_holding_hours'].mean()]
    })

    output_file = RESULTS_DIR / "exit_models_comparison.csv"
    comparison_df.to_csv(output_file, index=False)
    logger.success(f"\n✅ Comparison saved: {output_file}")

    logger.info(f"\n{'=' * 80}")
    logger.success("✅ Backtest Complete!")
    logger.info(f"{'=' * 80}")


if __name__ == "__main__":
    main()
