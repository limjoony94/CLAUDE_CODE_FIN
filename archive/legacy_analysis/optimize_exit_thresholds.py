#!/usr/bin/env python3
"""
Optimize Exit Thresholds - Grid Search
=======================================
Optimizes ML Exit thresholds for LONG and SHORT positions using grid search.

Tests multiple threshold combinations on recent data (7-30 days) to find
the optimal balance between exit timing and performance.

Usage:
    python scripts/analysis/optimize_exit_thresholds.py
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yaml
import joblib
from itertools import product
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from api.bingx_client import BingXClient

# ============================================================================
# Configuration
# ============================================================================

BACKTEST_DAYS = 30  # 30 days of data for optimization
CANDLE_INTERVAL = '5m'
INITIAL_BALANCE = 10000

# Fixed Parameters (not optimizing these)
LONG_THRESHOLD = 0.65
SHORT_THRESHOLD = 0.70
GATE_THRESHOLD = 0.001
LONG_AVG_RETURN = 0.0041
SHORT_AVG_RETURN = 0.0047
EMERGENCY_STOP_LOSS = 0.03  # -3% of balance
EMERGENCY_MAX_HOLD_CANDLES = 120  # 10 hours
LEVERAGE = 4

# Optimization Grid
EXIT_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]

# Model paths
MODELS_DIR = PROJECT_ROOT / 'models'
LONG_ENTRY_MODEL = MODELS_DIR / 'xgboost_long_trade_outcome_full_20251018_233146.pkl'
SHORT_ENTRY_MODEL = MODELS_DIR / 'xgboost_short_trade_outcome_full_20251018_233146.pkl'
LONG_EXIT_MODEL = MODELS_DIR / 'xgboost_long_exit_oppgating_improved_20251017_151624.pkl'
SHORT_EXIT_MODEL = MODELS_DIR / 'xgboost_short_exit_oppgating_improved_20251017_152440.pkl'

LONG_SCALER = MODELS_DIR / 'xgboost_long_trade_outcome_full_20251018_233146_scaler.pkl'
SHORT_SCALER = MODELS_DIR / 'xgboost_short_trade_outcome_full_20251018_233146_scaler.pkl'
LONG_EXIT_SCALER = MODELS_DIR / 'xgboost_long_exit_oppgating_improved_20251017_151624_scaler.pkl'
SHORT_EXIT_SCALER = MODELS_DIR / 'xgboost_short_exit_oppgating_improved_20251017_152440_scaler.pkl'


# ============================================================================
# Feature Calculation
# ============================================================================

def calculate_all_features(df):
    """Calculate all features needed for models"""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / 'scripts' / 'experiments'))
    from calculate_all_features import calculate_all_features as calc_features
    return calc_features(df)


def load_models():
    """Load all ML models and scalers"""
    models = {
        'long_entry': joblib.load(LONG_ENTRY_MODEL),
        'short_entry': joblib.load(SHORT_ENTRY_MODEL),
        'long_exit': joblib.load(LONG_EXIT_MODEL),
        'short_exit': joblib.load(SHORT_EXIT_MODEL),
        'long_scaler': joblib.load(LONG_SCALER),
        'short_scaler': joblib.load(SHORT_SCALER),
        'long_exit_scaler': joblib.load(LONG_EXIT_SCALER),
        'short_exit_scaler': joblib.load(SHORT_EXIT_SCALER),
    }

    # Load feature lists
    long_features_path = MODELS_DIR / 'xgboost_long_trade_outcome_full_20251018_233146_features.txt'
    with open(long_features_path, 'r') as f:
        models['long_features'] = [line.strip() for line in f.readlines() if line.strip()]

    short_features_path = MODELS_DIR / 'xgboost_short_trade_outcome_full_20251018_233146_features.txt'
    with open(short_features_path, 'r') as f:
        models['short_features'] = [line.strip() for line in f.readlines() if line.strip()]

    long_exit_features_path = MODELS_DIR / 'xgboost_long_exit_oppgating_improved_20251017_151624_features.txt'
    with open(long_exit_features_path, 'r') as f:
        models['long_exit_features'] = [line.strip() for line in f.readlines() if line.strip()]

    short_exit_features_path = MODELS_DIR / 'xgboost_short_exit_oppgating_improved_20251017_152440_features.txt'
    with open(short_exit_features_path, 'r') as f:
        models['short_exit_features'] = [line.strip() for line in f.readlines() if line.strip()]

    return models


def get_entry_signals(features, models, idx):
    """Get LONG and SHORT entry signals"""
    long_features = features[models['long_features']].iloc[idx:idx+1].values
    long_features_scaled = models['long_scaler'].transform(long_features)
    long_prob = models['long_entry'].predict_proba(long_features_scaled)[0][1]

    short_features = features[models['short_features']].iloc[idx:idx+1].values
    short_features_scaled = models['short_scaler'].transform(short_features)
    short_prob = models['short_entry'].predict_proba(short_features_scaled)[0][1]

    return long_prob, short_prob


def get_exit_signal(features, models, idx, direction):
    """Get exit signal for current position"""
    try:
        if direction == 'LONG':
            missing = [f for f in models['long_exit_features'] if f not in features.columns]
            if missing:
                return 0.0

            exit_features = features[models['long_exit_features']].iloc[idx:idx+1].values
            exit_features_scaled = models['long_exit_scaler'].transform(exit_features)
            exit_prob = models['long_exit'].predict_proba(exit_features_scaled)[0][1]
        else:
            missing = [f for f in models['short_exit_features'] if f not in features.columns]
            if missing:
                return 0.0

            exit_features = features[models['short_exit_features']].iloc[idx:idx+1].values
            exit_features_scaled = models['short_exit_scaler'].transform(exit_features)
            exit_prob = models['short_exit'].predict_proba(exit_features_scaled)[0][1]

        return exit_prob
    except:
        return 0.0


def check_opportunity_gate(long_prob, short_prob):
    """Check if SHORT should be allowed through opportunity gate"""
    if short_prob < SHORT_THRESHOLD:
        return False, 0.0

    long_ev = long_prob * LONG_AVG_RETURN
    short_ev = short_prob * SHORT_AVG_RETURN
    opportunity_cost = short_ev - long_ev

    return opportunity_cost > GATE_THRESHOLD, opportunity_cost


# ============================================================================
# Backtest Engine
# ============================================================================

def run_backtest(df, features, models, ml_exit_long, ml_exit_short):
    """Run backtest with specific exit thresholds"""
    balance = INITIAL_BALANCE
    position = None
    trades = []

    balance_history = [balance]
    max_balance = balance

    for idx in range(len(df)):
        candle = df.iloc[idx]
        price = candle['close']

        # Get entry signals
        long_prob, short_prob = get_entry_signals(features, models, idx)

        # Check exit conditions
        if position:
            hold_time = idx - position['entry_idx']
            current_pnl_pct = (price - position['entry_price']) / position['entry_price'] * LEVERAGE
            if position['direction'] == 'SHORT':
                current_pnl_pct = -current_pnl_pct

            current_pnl_usd = current_pnl_pct * position['size_usd']

            # Get exit signal
            exit_prob = get_exit_signal(features, models, idx, position['direction'])

            exit_reason = None

            # 1. ML Exit
            exit_threshold = ml_exit_long if position['direction'] == 'LONG' else ml_exit_short
            if exit_prob >= exit_threshold:
                exit_reason = 'ML_EXIT'

            # 2. Stop Loss
            elif current_pnl_usd / balance <= -EMERGENCY_STOP_LOSS:
                exit_reason = 'STOP_LOSS'

            # 3. Max Hold
            elif hold_time >= EMERGENCY_MAX_HOLD_CANDLES:
                exit_reason = 'MAX_HOLD'

            if exit_reason:
                position['exit_price'] = price
                position['hold_candles'] = hold_time
                position['pnl_pct'] = current_pnl_pct
                position['pnl_usd'] = current_pnl_usd
                position['exit_reason'] = exit_reason
                position['exit_prob'] = exit_prob

                balance += current_pnl_usd
                trades.append(position)
                position = None

        # Check entry signals
        if not position:
            direction = None

            if long_prob >= LONG_THRESHOLD:
                direction = 'LONG'
            elif short_prob >= SHORT_THRESHOLD:
                gate_passed, _ = check_opportunity_gate(long_prob, short_prob)
                if gate_passed:
                    direction = 'SHORT'

            if direction:
                position_pct = 0.50
                size_usd = balance * position_pct

                position = {
                    'direction': direction,
                    'entry_price': price,
                    'entry_idx': idx,
                    'size_usd': size_usd,
                }

        # Track balance history
        if position:
            unrealized_pnl_pct = (price - position['entry_price']) / position['entry_price'] * LEVERAGE
            if position['direction'] == 'SHORT':
                unrealized_pnl_pct = -unrealized_pnl_pct
            unrealized_pnl_usd = unrealized_pnl_pct * position['size_usd']
            current_balance = balance + unrealized_pnl_usd
        else:
            current_balance = balance

        balance_history.append(current_balance)
        max_balance = max(max_balance, current_balance)

    # Close any open position
    if position:
        price = df.iloc[-1]['close']
        hold_time = len(df) - 1 - position['entry_idx']
        current_pnl_pct = (price - position['entry_price']) / position['entry_price'] * LEVERAGE
        if position['direction'] == 'SHORT':
            current_pnl_pct = -current_pnl_pct
        current_pnl_usd = current_pnl_pct * position['size_usd']

        position['exit_price'] = price
        position['hold_candles'] = hold_time
        position['pnl_pct'] = current_pnl_pct
        position['pnl_usd'] = current_pnl_usd
        position['exit_reason'] = 'END_OF_PERIOD'
        position['exit_prob'] = get_exit_signal(features, models, len(df)-1, position['direction'])

        balance += current_pnl_usd
        trades.append(position)

    return trades, balance_history


def calculate_metrics(trades, balance_history, initial_balance):
    """Calculate performance metrics"""
    if not trades:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'total_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'profit_factor': 0,
            'ml_exit_rate': 0,
        }

    wins = [t for t in trades if t['pnl_usd'] > 0]
    losses = [t for t in trades if t['pnl_usd'] <= 0]

    final_balance = balance_history[-1]
    total_return = (final_balance - initial_balance) / initial_balance

    # Calculate returns for Sharpe ratio
    returns = np.diff(balance_history) / balance_history[:-1]
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(288 * 30) if np.std(returns) > 0 else 0

    # Calculate max drawdown
    balance_array = np.array(balance_history)
    running_max = np.maximum.accumulate(balance_array)
    drawdown = (balance_array - running_max) / running_max
    max_drawdown = np.min(drawdown)

    # Profit factor
    gross_profit = sum([t['pnl_usd'] for t in wins]) if wins else 0
    gross_loss = abs(sum([t['pnl_usd'] for t in losses])) if losses else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    # ML Exit rate
    ml_exits = [t for t in trades if t['exit_reason'] == 'ML_EXIT']
    ml_exit_rate = len(ml_exits) / len(trades)

    return {
        'total_trades': len(trades),
        'win_rate': len(wins) / len(trades),
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'profit_factor': profit_factor,
        'ml_exit_rate': ml_exit_rate,
        'avg_hold': np.mean([t['hold_candles'] for t in trades]),
    }


# ============================================================================
# Optimization
# ============================================================================

def optimize_thresholds(df, features, models):
    """Grid search for optimal exit thresholds"""
    print("\n" + "="*100)
    print("🔍 OPTIMIZING EXIT THRESHOLDS - GRID SEARCH")
    print("="*100)
    print(f"Testing {len(EXIT_THRESHOLDS)}² = {len(EXIT_THRESHOLDS)**2} combinations")
    print(f"LONG Exit: {EXIT_THRESHOLDS}")
    print(f"SHORT Exit: {EXIT_THRESHOLDS}")
    print("="*100 + "\n")

    results = []
    combinations = list(product(EXIT_THRESHOLDS, repeat=2))

    for ml_exit_long, ml_exit_short in tqdm(combinations, desc="Testing combinations"):
        trades, balance_history = run_backtest(df, features, models, ml_exit_long, ml_exit_short)
        metrics = calculate_metrics(trades, balance_history, INITIAL_BALANCE)

        results.append({
            'ml_exit_long': ml_exit_long,
            'ml_exit_short': ml_exit_short,
            **metrics
        })

    return pd.DataFrame(results)


def print_results(results_df):
    """Print optimization results"""
    print("\n" + "="*100)
    print("📊 OPTIMIZATION RESULTS")
    print("="*100)

    # Sort by total return
    results_df = results_df.sort_values('total_return', ascending=False)

    print("\n🏆 TOP 10 CONFIGURATIONS (by Total Return):")
    print("-"*100)
    print(f"{'Rank':<5} {'LONG':<6} {'SHORT':<7} {'Return':<10} {'WinRate':<9} {'Sharpe':<8} {'MaxDD':<10} {'ML Exit':<9} {'Trades':<7}")
    print("-"*100)

    for i, row in results_df.head(10).iterrows():
        print(f"{i+1:<5} "
              f"{row['ml_exit_long']:<6.2f} "
              f"{row['ml_exit_short']:<7.2f} "
              f"{row['total_return']:>8.2%}  "
              f"{row['win_rate']:>7.1%}  "
              f"{row['sharpe_ratio']:>6.3f}  "
              f"{row['max_drawdown']:>8.2%}  "
              f"{row['ml_exit_rate']:>7.1%}  "
              f"{row['total_trades']:>5.0f}")

    print("\n" + "="*100)

    # Best by different metrics
    print("\n📈 BEST BY DIFFERENT METRICS:")
    print("-"*100)

    best_return = results_df.iloc[0]
    print(f"Best Return: LONG={best_return['ml_exit_long']:.2f}, SHORT={best_return['ml_exit_short']:.2f} "
          f"→ {best_return['total_return']:.2%}")

    best_sharpe = results_df.nlargest(1, 'sharpe_ratio').iloc[0]
    print(f"Best Sharpe: LONG={best_sharpe['ml_exit_long']:.2f}, SHORT={best_sharpe['ml_exit_short']:.2f} "
          f"→ {best_sharpe['sharpe_ratio']:.3f}")

    best_winrate = results_df.nlargest(1, 'win_rate').iloc[0]
    print(f"Best WinRate: LONG={best_winrate['ml_exit_long']:.2f}, SHORT={best_winrate['ml_exit_short']:.2f} "
          f"→ {best_winrate['win_rate']:.1%}")

    best_ml_exit = results_df.nlargest(1, 'ml_exit_rate').iloc[0]
    print(f"Most ML Exits: LONG={best_ml_exit['ml_exit_long']:.2f}, SHORT={best_ml_exit['ml_exit_short']:.2f} "
          f"→ {best_ml_exit['ml_exit_rate']:.1%}")

    print("\n" + "="*100)

    # Current production settings
    current = results_df[(results_df['ml_exit_long'] == 0.75) & (results_df['ml_exit_short'] == 0.75)]
    if not current.empty:
        current = current.iloc[0]
        rank = results_df[results_df['total_return'] >= current['total_return']].shape[0]
        print(f"\n⚙️  CURRENT PRODUCTION (0.75/0.75): Rank #{rank}/{len(results_df)}")
        print(f"   Return: {current['total_return']:+.2%} | WinRate: {current['win_rate']:.1%} | "
              f"Sharpe: {current['sharpe_ratio']:.3f} | ML Exit: {current['ml_exit_rate']:.1%}")
        print("="*100)


# ============================================================================
# Main
# ============================================================================

def main():
    print("\n🚀 EXIT THRESHOLD OPTIMIZATION")
    print("="*100)

    # Load API keys
    config_path = PROJECT_ROOT / 'config' / 'api_keys.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize client
    api_key = config['bingx']['mainnet']['api_key']
    secret_key = config['bingx']['mainnet']['secret_key']
    client = BingXClient(api_key, secret_key, testnet=False)

    # Load models
    print("🧠 Loading ML models...")
    models = load_models()

    # Fetch data
    print(f"📥 Fetching {BACKTEST_DAYS} days of data...")
    candles_needed = BACKTEST_DAYS * 288 + 200  # 288 candles per day + buffer
    ohlcv = client.get_klines('BTC-USDT', CANDLE_INTERVAL, limit=min(candles_needed, 1440))

    df = pd.DataFrame(ohlcv)
    df = df.rename(columns={'time': 'timestamp'})
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    print(f"  Fetched: {len(df)} candles ({len(df)/288:.1f} days)")
    print(f"  Period: {df.iloc[0]['timestamp']} → {df.iloc[-1]['timestamp']}")

    # Calculate features
    print("🔧 Calculating features...")
    features = calculate_all_features(df)

    # Run optimization
    results_df = optimize_thresholds(df, features, models)

    # Print results
    print_results(results_df)

    # Save results
    output_path = PROJECT_ROOT / 'results' / 'exit_threshold_optimization.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\n💾 Full results saved to: {output_path}")


if __name__ == '__main__':
    main()
