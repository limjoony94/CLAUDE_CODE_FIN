#!/usr/bin/env python3
"""
Test optimal parameter combination identified from grid search.
Tests: Baseline vs Individual optimizations vs Combined optimization
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.dynamic_position_sizing import DynamicPositionSizer

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Production settings
LEVERAGE = 4
INITIAL_CAPITAL = 10000
LONG_ENTRY_THRESHOLD = 0.65
SHORT_ENTRY_THRESHOLD = 0.70
GATE_THRESHOLD = 0.001
EMERGENCY_STOP_LOSS = 0.03
EMERGENCY_MAX_HOLD_TIME = 120
ML_EXIT_THRESHOLD_LONG = 0.75
ML_EXIT_THRESHOLD_SHORT = 0.75

def calculate_minimal_features(df, params):
    """Calculate only features affected by parameter changes"""
    df = df.copy()

    # RSI with variable period
    rsi_period = params['rsi_period']
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD with variable periods
    macd_fast = params['macd_fast']
    macd_slow = params['macd_slow']
    macd_signal = params['macd_signal']
    ema_fast = df['close'].ewm(span=macd_fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=macd_slow, adjust=False).mean()
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd'].ewm(span=macd_signal, adjust=False).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']

    # Bollinger Bands with variable period
    bb_period = params['bb_period']
    df['bb_mid'] = df['close'].rolling(window=bb_period).mean()
    bb_std = df['close'].rolling(window=bb_period).std()
    df['bb_high'] = df['bb_mid'] + (bb_std * 2)
    df['bb_low'] = df['bb_mid'] - (bb_std * 2)
    df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']

    # ATR with variable period
    atr_period = params['atr_period']
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(window=atr_period).mean()
    df['atr_pct'] = (df['atr'] / df['close']) * 100

    # EMA with variable period
    ema_period = params['ema_period']
    df[f'ema_{ema_period}'] = df['close'].ewm(span=ema_period, adjust=False).mean()
    # Also update derived features
    for period in [8, 12, 20]:
        if period == ema_period:
            df[f'price_vs_ema{period}'] = (df['close'] - df[f'ema_{ema_period}']) / df['close']

    return df

def run_backtest(df, models, params, config_name):
    """Run backtest with given parameters"""

    # Update only affected features
    df_test = calculate_minimal_features(df, params)

    position_sizer = DynamicPositionSizer()

    trades = []
    balance = INITIAL_CAPITAL
    position = None

    for i in range(len(df_test)):
        current_candle = df_test.iloc[i]

        # Exit logic
        if position is not None:
            hold_time = i - position['entry_idx']
            current_price = current_candle['close']

            # Calculate P&L
            if position['direction'] == 'LONG':
                pnl_pct = ((current_price - position['entry_price']) / position['entry_price']) * LEVERAGE
            else:
                pnl_pct = ((position['entry_price'] - current_price) / position['entry_price']) * LEVERAGE

            exit_signal = False
            exit_reason = None

            # Emergency exits
            if pnl_pct <= -EMERGENCY_STOP_LOSS:
                exit_signal = True
                exit_reason = 'stop_loss'
            elif hold_time >= EMERGENCY_MAX_HOLD_TIME:
                exit_signal = True
                exit_reason = 'max_hold'
            else:
                # ML Exit
                try:
                    if position['direction'] == 'LONG':
                        features = df_test.iloc[i][models['long_exit'].feature_names_in_].values.reshape(1, -1)
                        exit_prob = models['long_exit'].predict_proba(features)[0][1]
                        if exit_prob >= ML_EXIT_THRESHOLD_LONG:
                            exit_signal = True
                            exit_reason = 'ml_exit'
                    else:
                        features = df_test.iloc[i][models['short_exit'].feature_names_in_].values.reshape(1, -1)
                        exit_prob = models['short_exit'].predict_proba(features)[0][1]
                        if exit_prob >= ML_EXIT_THRESHOLD_SHORT:
                            exit_signal = True
                            exit_reason = 'ml_exit'
                except:
                    pass

            if exit_signal:
                # Close position
                pnl = balance * position['size_pct'] * pnl_pct
                balance += pnl

                trades.append({
                    'direction': position['direction'],
                    'entry_price': position['entry_price'],
                    'exit_price': current_price,
                    'size_pct': position['size_pct'],
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'hold_time': hold_time,
                    'exit_reason': exit_reason,
                    'balance': balance
                })

                position = None

        # Entry logic (only if no position)
        if position is None and i < len(df_test) - EMERGENCY_MAX_HOLD_TIME:
            try:
                # Get LONG probability
                long_features = df_test.iloc[i][models['long_entry'].feature_names_in_].values.reshape(1, -1)
                long_prob = models['long_entry'].predict_proba(long_features)[0][1]

                # Get SHORT probability
                short_features = df_test.iloc[i][models['short_entry'].feature_names_in_].values.reshape(1, -1)
                short_prob = models['short_entry'].predict_proba(short_features)[0][1]

                # LONG Entry
                if long_prob >= LONG_ENTRY_THRESHOLD:
                    sizing = position_sizer.get_position_size_simple(
                        capital=balance,
                        signal_strength=long_prob,
                        leverage=LEVERAGE
                    )

                    position = {
                        'direction': 'LONG',
                        'entry_price': current_candle['close'],
                        'entry_idx': i,
                        'size_pct': sizing['position_size_pct']
                    }

                # SHORT Entry (with gating)
                elif short_prob >= SHORT_ENTRY_THRESHOLD:
                    long_ev = long_prob * 0.0041
                    short_ev = short_prob * 0.0047
                    opportunity_cost = short_ev - long_ev

                    if opportunity_cost > GATE_THRESHOLD:
                        sizing = position_sizer.get_position_size_simple(
                            capital=balance,
                            signal_strength=short_prob,
                            leverage=LEVERAGE
                        )

                        position = {
                            'direction': 'SHORT',
                            'entry_price': current_candle['close'],
                            'entry_idx': i,
                            'size_pct': sizing['position_size_pct']
                        }
            except:
                pass

    # Calculate metrics
    if len(trades) == 0:
        return {
            'config': config_name,
            'total_return': 0,
            'win_rate': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'num_trades': 0,
            'final_balance': INITIAL_CAPITAL
        }

    df_trades = pd.DataFrame(trades)

    total_return = ((balance - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    win_rate = (df_trades['pnl'] > 0).sum() / len(df_trades) * 100

    # Calculate drawdown
    balances = df_trades['balance'].values
    running_max = np.maximum.accumulate(balances)
    drawdowns = (balances - running_max) / running_max * 100
    max_drawdown = drawdowns.min()

    # Sharpe ratio
    returns = df_trades['pnl_pct'].values
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(len(returns)) if returns.std() > 0 else 0

    return {
        'config': config_name,
        'total_return': total_return,
        'win_rate': win_rate,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'num_trades': len(trades),
        'final_balance': balance,
        'trades': df_trades
    }

def main():
    print("=" * 80)
    print("OPTIMAL PARAMETER COMBINATION TEST")
    print("=" * 80)
    print()

    # Test configurations
    BASELINE = {
        'rsi_period': 14,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'bb_period': 20,
        'atr_period': 14,
        'ema_period': 12
    }

    # Individual optimizations
    ATR_21 = {**BASELINE, 'atr_period': 21}
    BB_25 = {**BASELINE, 'bb_period': 25}
    EMA_8 = {**BASELINE, 'ema_period': 8}

    # Combined optimization
    COMBINED = {
        'rsi_period': 14,      # Keep baseline (best)
        'macd_fast': 12,       # Keep baseline (best)
        'macd_slow': 26,
        'macd_signal': 9,
        'bb_period': 25,       # Change to 25
        'atr_period': 21,      # Change to 21
        'ema_period': 8        # Change to 8
    }

    TEST_CONFIGS = [
        ('Baseline', BASELINE),
        ('ATR_21', ATR_21),
        ('BB_25', BB_25),
        ('EMA_8', EMA_8),
        ('COMBINED (ATR21+BB25+EMA8)', COMBINED),
    ]

    print("Test Configurations:")
    for name, config in TEST_CONFIGS:
        print(f"  {name}:")
        for k, v in config.items():
            if k in ['rsi_period', 'bb_period', 'atr_period', 'ema_period']:
                baseline_val = BASELINE[k]
                if v != baseline_val:
                    print(f"    {k}: {v} (baseline: {baseline_val}) ✓")
        print()

    # Load models
    print("=" * 80)
    print("Loading Production Models")
    print("=" * 80)

    models = {}
    model_files = {
        'long_entry': 'models/xgboost_long_trade_outcome_full_20251018_233146.pkl',
        'short_entry': 'models/xgboost_short_trade_outcome_full_20251018_233146.pkl',
        'long_exit': 'models/xgboost_long_exit_oppgating_improved_20251017_151624.pkl',
        'short_exit': 'models/xgboost_short_exit_oppgating_improved_20251017_152440.pkl',
    }

    for name, path in model_files.items():
        with open(PROJECT_ROOT / path, 'rb') as f:
            models[name] = pickle.load(f)

    print("✅ Models loaded")
    print()

    # Load base data
    print("=" * 80)
    print("Loading Base Data")
    print("=" * 80)

    data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
    df_full = pd.read_csv(data_file)

    print(f"✅ Loaded {len(df_full):,} candles")
    print()

    # Run tests
    print("=" * 80)
    print("Running Tests")
    print("=" * 80)
    print()

    results = []
    for i, (name, config) in enumerate(TEST_CONFIGS, 1):
        print(f"[{i}/{len(TEST_CONFIGS)}] Testing {name}...", end=" ", flush=True)

        result = run_backtest(df_full, models, config, name)
        results.append(result)

        print(f"✅ Return: {result['total_return']:+.2f}% | WR: {result['win_rate']:.1f}%")

    print()

    # Display results
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()

    df_results = pd.DataFrame([{
        'Config': r['config'],
        'Return': r['total_return'],
        'Win Rate': r['win_rate'],
        'Sharpe': r['sharpe_ratio'],
        'Max DD': r['max_drawdown'],
        'Trades': r['num_trades']
    } for r in results])

    # Calculate composite scores
    def calc_score(row):
        return_norm = (row['Return'] - df_results['Return'].min()) / (df_results['Return'].max() - df_results['Return'].min()) if df_results['Return'].max() > df_results['Return'].min() else 0
        wr_norm = (row['Win Rate'] - df_results['Win Rate'].min()) / (df_results['Win Rate'].max() - df_results['Win Rate'].min()) if df_results['Win Rate'].max() > df_results['Win Rate'].min() else 0
        sharpe_norm = (row['Sharpe'] - df_results['Sharpe'].min()) / (df_results['Sharpe'].max() - df_results['Sharpe'].min()) if df_results['Sharpe'].max() > df_results['Sharpe'].min() else 0
        dd_norm = (row['Max DD'] - df_results['Max DD'].min()) / (df_results['Max DD'].max() - df_results['Max DD'].min()) if df_results['Max DD'].max() > df_results['Max DD'].min() else 0

        return 0.3 * return_norm + 0.2 * wr_norm + 0.4 * sharpe_norm + 0.1 * dd_norm

    df_results['Score'] = df_results.apply(calc_score, axis=1)
    df_results = df_results.sort_values('Score', ascending=False)

    print(df_results.to_string(index=False))
    print()

    # Comparison
    print("=" * 80)
    print("COMPARISON: Combined vs Baseline")
    print("=" * 80)
    print()

    baseline = next(r for r in results if r['config'] == 'Baseline')
    combined = next(r for r in results if r['config'].startswith('COMBINED'))

    print(f"BASELINE:")
    print(f"  Return:    {baseline['total_return']:+.2f}%")
    print(f"  Win Rate:  {baseline['win_rate']:.1f}%")
    print(f"  Sharpe:    {baseline['sharpe_ratio']:.2f}")
    print(f"  Max DD:    {baseline['max_drawdown']:.2f}%")
    print()

    print(f"COMBINED (ATR21+BB25+EMA8):")
    print(f"  Return:    {combined['total_return']:+.2f}%  ({combined['total_return'] - baseline['total_return']:+.2f}pp)")
    print(f"  Win Rate:  {combined['win_rate']:.1f}%  ({combined['win_rate'] - baseline['win_rate']:+.1f}pp)")
    print(f"  Sharpe:    {combined['sharpe_ratio']:.2f}  ({combined['sharpe_ratio'] - baseline['sharpe_ratio']:+.2f})")
    print(f"  Max DD:    {combined['max_drawdown']:.2f}%  ({combined['max_drawdown'] - baseline['max_drawdown']:+.2f}pp)")
    print()

    improvement = ((combined['total_return'] - baseline['total_return']) / abs(baseline['total_return'])) * 100 if baseline['total_return'] != 0 else 0
    print(f"Improvement: {improvement:+.1f}%")
    print()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = PROJECT_ROOT / "results" / f"optimal_combination_test_{timestamp}.csv"
    df_results.to_csv(output_file, index=False)
    print(f"✅ Results saved to: {output_file.name}")
    print()

    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
