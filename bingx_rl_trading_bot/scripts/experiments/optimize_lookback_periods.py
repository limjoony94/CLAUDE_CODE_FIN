"""
Lookback Period Grid Search Optimization
=========================================

Systematically test different lookback period combinations to find optimal
technical indicator parameters for the trading strategy.

Parameters to Optimize:
- RSI: [10, 14, 20] periods
- MACD: [8/17/9, 12/26/9, 16/35/9] (fast/slow/signal)
- Bollinger Bands: [15, 20, 25] periods
- ATR: [7, 14, 21] periods
- EMA: [8, 12, 20] periods

Total Combinations: 3^5 = 243 combinations

Evaluation Metrics:
- Return (30%)
- Win Rate (20%)
- Sharpe Ratio (40%)
- Max Drawdown (10%)

Created: 2025-10-23
"""

import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
import sys
import time
from datetime import datetime
from itertools import product
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import feature calculation
from scripts.experiments.calculate_all_features import (
    calculate_features,
    AdvancedTechnicalFeatures
)
from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features
from scripts.production.dynamic_position_sizing import DynamicPositionSizer

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

print("="*80)
print("LOOKBACK PERIOD GRID SEARCH OPTIMIZATION")
print("="*80)
print(f"\nSystematically testing indicator parameter combinations")
print(f"Total combinations: 243 (3^5)")
print(f"Using PRODUCTION models with varying feature calculations\n")

# ============================================================================
# Grid Search Parameters
# ============================================================================

PARAM_GRID = {
    'rsi_period': [10, 14, 20],
    'macd_params': [(8, 17, 9), (12, 26, 9), (16, 35, 9)],  # (fast, slow, signal)
    'bb_period': [15, 20, 25],
    'atr_period': [7, 14, 21],
    'ema_period': [8, 12, 20]
}

print(f"Parameter Grid:")
print(f"  RSI Periods:  {PARAM_GRID['rsi_period']}")
print(f"  MACD Params:  {PARAM_GRID['macd_params']}")
print(f"  BB Periods:   {PARAM_GRID['bb_period']}")
print(f"  ATR Periods:  {PARAM_GRID['atr_period']}")
print(f"  EMA Periods:  {PARAM_GRID['ema_period']}")

# Generate all combinations
all_combinations = list(product(
    PARAM_GRID['rsi_period'],
    PARAM_GRID['macd_params'],
    PARAM_GRID['bb_period'],
    PARAM_GRID['atr_period'],
    PARAM_GRID['ema_period']
))

print(f"\nTotal combinations to test: {len(all_combinations)}")

# ============================================================================
# Load Production Models (UNCHANGED - using original 107 features)
# ============================================================================
print(f"\n{'='*80}")
print("Loading Production Models")
print(f"{'='*80}")

timestamp = "20251018_233146"

# LONG Entry
long_model_path = MODELS_DIR / f"xgboost_long_trade_outcome_full_{timestamp}.pkl"
with open(long_model_path, 'rb') as f:
    long_model = pickle.load(f)
long_scaler = joblib.load(MODELS_DIR / f"xgboost_long_trade_outcome_full_{timestamp}_scaler.pkl")
with open(MODELS_DIR / f"xgboost_long_trade_outcome_full_{timestamp}_features.txt", 'r') as f:
    long_feature_columns = [line.strip() for line in f.readlines()]

# SHORT Entry
short_model_path = MODELS_DIR / f"xgboost_short_trade_outcome_full_{timestamp}.pkl"
with open(short_model_path, 'rb') as f:
    short_model = pickle.load(f)
short_scaler = joblib.load(MODELS_DIR / f"xgboost_short_trade_outcome_full_{timestamp}_scaler.pkl")
with open(MODELS_DIR / f"xgboost_short_trade_outcome_full_{timestamp}_features.txt", 'r') as f:
    short_feature_columns = [line.strip() for line in f.readlines()]

# Exit Models
long_exit_model_path = MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251017_151624.pkl"
with open(long_exit_model_path, 'rb') as f:
    long_exit_model = pickle.load(f)
long_exit_scaler = joblib.load(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251017_151624_scaler.pkl")
with open(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251017_151624_features.txt", 'r') as f:
    long_exit_feature_columns = [line.strip() for line in f.readlines()]

short_exit_model_path = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251017_152440.pkl"
with open(short_exit_model_path, 'rb') as f:
    short_exit_model = pickle.load(f)
short_exit_scaler = joblib.load(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251017_152440_scaler.pkl")
with open(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251017_152440_features.txt", 'r') as f:
    short_exit_feature_columns = [line.strip() for line in f.readlines()]

print(f"✅ Models loaded")
print(f"   LONG Entry: {len(long_feature_columns)} features")
print(f"   SHORT Entry: {len(short_feature_columns)} features")
print(f"   Exit: {len(long_exit_feature_columns)} features")

# ============================================================================
# Load Base Data (Will recalculate features for each combination)
# ============================================================================
print(f"\n{'='*80}")
print("Loading Base Data")
print(f"{'='*80}")

data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df_base = pd.read_csv(data_file)
print(f"✅ Loaded {len(df_base):,} candles")

# ============================================================================
# Feature Calculation Function (with variable lookback periods)
# ============================================================================

def calculate_features_with_params(df, rsi_period, macd_params, bb_period, atr_period, ema_period):
    """
    Calculate all features with custom lookback periods

    This replaces hardcoded periods in calculate_all_features
    """
    df = df.copy()

    # Basic price features
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

    # RSI (variable period)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD (variable periods)
    fast_period, slow_period, signal_period = macd_params
    ema_fast = df['close'].ewm(span=fast_period, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow_period, adjust=False).mean()
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd'].ewm(span=signal_period, adjust=False).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']

    # Bollinger Bands (variable period)
    df['bb_mid'] = df['close'].rolling(window=bb_period).mean()
    bb_std = df['close'].rolling(window=bb_period).std()
    df['bb_high'] = df['bb_mid'] + (2 * bb_std)
    df['bb_low'] = df['bb_mid'] - (2 * bb_std)
    df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']

    # ATR (variable period)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['atr'] = true_range.rolling(window=atr_period).mean()
    df['atr_pct'] = df['atr'] / df['close']

    # EMA (variable period)
    df['ema_12'] = df['close'].ewm(span=ema_period, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=ema_period*2, adjust=False).mean()

    # Volume features
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']

    # Volatility (20-period standard)
    df['volatility'] = df['returns'].rolling(window=20).std()

    # Add remaining features from original calculator
    # (These use standard periods and don't need optimization)
    from scripts.experiments.calculate_all_features import calculate_features
    df_full = calculate_features(df)

    # Override with our optimized features
    for col in ['rsi', 'macd', 'macd_signal', 'macd_diff', 'bb_mid', 'bb_high', 'bb_low',
                'bb_width', 'atr', 'atr_pct', 'ema_12', 'ema_26']:
        if col in df.columns:
            df_full[col] = df[col]

    return df_full

# ============================================================================
# Backtest Function
# ============================================================================

def run_backtest(df, combination_id, params):
    """
    Run backtest with given parameter combination

    Returns: dict with performance metrics
    """
    rsi_period, macd_params, bb_period, atr_period, ema_period = params

    # Trading parameters (optimized settings)
    LEVERAGE = 4
    LONG_THRESHOLD = 0.65
    SHORT_THRESHOLD = 0.70
    GATE_THRESHOLD = 0.001
    ML_EXIT_THRESHOLD_LONG = 0.75
    ML_EXIT_THRESHOLD_SHORT = 0.75
    EMERGENCY_MAX_HOLD_TIME = 120
    EMERGENCY_STOP_LOSS = -0.03
    LONG_AVG_RETURN = 0.0041
    SHORT_AVG_RETURN = 0.0047
    INITIAL_CAPITAL = 10000.0
    TAKER_FEE = 0.0005

    # Initialize
    position_sizer = DynamicPositionSizer(min_position_pct=0.20, max_position_pct=0.95, base_position_pct=0.50)
    balance = INITIAL_CAPITAL
    position = None
    trades = []

    # Backtest loop
    for idx in range(len(df)):
        current_price = df['close'].iloc[idx]

        # Position Management
        if position is not None:
            direction = position['direction']
            entry_price = position['entry_price']
            size_pct = position['size_pct']
            entry_idx = position['entry_idx']
            candles_held = idx - entry_idx

            # Calculate P&L
            if direction == 'LONG':
                price_change_pct = (current_price - entry_price) / entry_price
            else:
                price_change_pct = (entry_price - current_price) / entry_price

            leveraged_pnl_pct = price_change_pct * LEVERAGE
            balance_pnl_pct = leveraged_pnl_pct * size_pct

            # Exit conditions
            should_exit = False
            exit_reason = None

            # Stop Loss
            if balance_pnl_pct <= EMERGENCY_STOP_LOSS:
                should_exit = True
                exit_reason = 'STOP_LOSS'
            # Max Hold
            elif candles_held >= EMERGENCY_MAX_HOLD_TIME:
                should_exit = True
                exit_reason = 'MAX_HOLD'
            # ML Exit
            else:
                exit_features = df[long_exit_feature_columns if direction == 'LONG' else short_exit_feature_columns].iloc[idx].values.reshape(1, -1)
                exit_scaler_obj = long_exit_scaler if direction == 'LONG' else short_exit_scaler
                exit_model_obj = long_exit_model if direction == 'LONG' else short_exit_model
                threshold = ML_EXIT_THRESHOLD_LONG if direction == 'LONG' else ML_EXIT_THRESHOLD_SHORT

                exit_features_scaled = exit_scaler_obj.transform(exit_features)
                exit_prob = exit_model_obj.predict_proba(exit_features_scaled)[0][1]

                if exit_prob >= threshold:
                    should_exit = True
                    exit_reason = 'ML'

            # Execute Exit
            if should_exit:
                position_value = balance * size_pct
                leveraged_position = position_value * LEVERAGE
                entry_fee = leveraged_position * TAKER_FEE
                exit_fee = leveraged_position * TAKER_FEE
                total_fees = entry_fee + exit_fee
                gross_pnl = balance * balance_pnl_pct
                net_pnl = gross_pnl - total_fees
                balance += net_pnl

                trades.append({
                    'direction': direction,
                    'net_pnl': net_pnl,
                    'balance_pnl_pct': balance_pnl_pct,
                    'exit_reason': exit_reason,
                    'balance_after': balance
                })

                position = None

        # Entry Signals
        if position is None:
            long_features = df[long_feature_columns].iloc[idx].values.reshape(1, -1)
            long_features_scaled = long_scaler.transform(long_features)
            long_prob = long_model.predict_proba(long_features_scaled)[0][1]

            short_features = df[short_feature_columns].iloc[idx].values.reshape(1, -1)
            short_features_scaled = short_scaler.transform(short_features)
            short_prob = short_model.predict_proba(short_features_scaled)[0][1]

            # LONG Entry
            if long_prob >= LONG_THRESHOLD:
                sizing_result = position_sizer.get_position_size_simple(
                    capital=balance, signal_strength=long_prob, leverage=LEVERAGE
                )
                position = {
                    'direction': 'LONG',
                    'entry_price': current_price,
                    'size_pct': sizing_result['position_size_pct'],
                    'entry_idx': idx
                }

            # SHORT Entry (gated)
            elif short_prob >= SHORT_THRESHOLD:
                long_ev = long_prob * LONG_AVG_RETURN
                short_ev = short_prob * SHORT_AVG_RETURN
                if (short_ev - long_ev) > GATE_THRESHOLD:
                    sizing_result = position_sizer.get_position_size_simple(
                        capital=balance, signal_strength=short_prob, leverage=LEVERAGE
                    )
                    position = {
                        'direction': 'SHORT',
                        'entry_price': current_price,
                        'size_pct': sizing_result['position_size_pct'],
                        'entry_idx': idx
                    }

    # Calculate metrics
    if len(trades) == 0:
        return None

    trades_df = pd.DataFrame(trades)
    total_return = (balance - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    win_rate = len(trades_df[trades_df['net_pnl'] > 0]) / len(trades_df) * 100

    # Max Drawdown
    cumulative = trades_df['balance_after'].values
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max * 100
    max_drawdown = drawdown.min()

    # Sharpe Ratio
    returns = trades_df['balance_pnl_pct'].values
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 24 * 12) if returns.std() > 0 else 0

    return {
        'combination_id': combination_id,
        'rsi_period': rsi_period,
        'macd_fast': macd_params[0],
        'macd_slow': macd_params[1],
        'macd_signal': macd_params[2],
        'bb_period': bb_period,
        'atr_period': atr_period,
        'ema_period': ema_period,
        'total_return': total_return,
        'win_rate': win_rate,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe,
        'num_trades': len(trades_df),
        'final_balance': balance
    }

# ============================================================================
# Grid Search Execution
# ============================================================================

print(f"\n{'='*80}")
print("Starting Grid Search")
print(f"{'='*80}\n")

results = []
start_time = time.time()

for i, params in enumerate(all_combinations, 1):
    rsi_period, macd_params, bb_period, atr_period, ema_period = params

    print(f"\r[{i}/{len(all_combinations)}] Testing: RSI={rsi_period}, MACD={macd_params}, BB={bb_period}, ATR={atr_period}, EMA={ema_period}", end='')

    try:
        # Calculate features with these parameters
        df_test = calculate_features_with_params(
            df_base.copy(),
            rsi_period=rsi_period,
            macd_params=macd_params,
            bb_period=bb_period,
            atr_period=atr_period,
            ema_period=ema_period
        )

        # Add advanced features and exit features
        adv_features = AdvancedTechnicalFeatures(lookback_sr=200, lookback_trend=50)
        df_test = adv_features.calculate_all_features(df_test)
        df_test = prepare_exit_features(df_test)

        # Remove NaN
        df_test = df_test.dropna()

        # Run backtest
        result = run_backtest(df_test, i, params)

        if result is not None:
            results.append(result)

    except Exception as e:
        print(f"\n⚠️ Error in combination {i}: {e}")
        continue

elapsed = time.time() - start_time
print(f"\n\n✅ Grid search complete in {elapsed/60:.1f} minutes")
print(f"   Successful combinations: {len(results)}/{len(all_combinations)}")

# ============================================================================
# Results Analysis
# ============================================================================

print(f"\n{'='*80}")
print("GRID SEARCH RESULTS")
print(f"{'='*80}\n")

if len(results) > 0:
    results_df = pd.DataFrame(results)

    # Calculate composite score (weighted)
    # Return: 30%, Win Rate: 20%, Sharpe: 40%, Max DD: 10% (negative is good)
    results_df['return_score'] = (results_df['total_return'] - results_df['total_return'].min()) / (results_df['total_return'].max() - results_df['total_return'].min())
    results_df['wr_score'] = (results_df['win_rate'] - results_df['win_rate'].min()) / (results_df['win_rate'].max() - results_df['win_rate'].min())
    results_df['sharpe_score'] = (results_df['sharpe_ratio'] - results_df['sharpe_ratio'].min()) / (results_df['sharpe_ratio'].max() - results_df['sharpe_ratio'].min())
    results_df['dd_score'] = (results_df['max_drawdown'].max() - results_df['max_drawdown']) / (results_df['max_drawdown'].max() - results_df['max_drawdown'].min())

    results_df['composite_score'] = (
        results_df['return_score'] * 0.30 +
        results_df['wr_score'] * 0.20 +
        results_df['sharpe_score'] * 0.40 +
        results_df['dd_score'] * 0.10
    )

    # Sort by composite score
    results_df = results_df.sort_values('composite_score', ascending=False)

    # Top 10 combinations
    print("Top 10 Combinations:")
    print("="*120)
    print(f"{'Rank':<6} {'RSI':<5} {'MACD':<12} {'BB':<5} {'ATR':<5} {'EMA':<5} {'Return':<8} {'WR%':<7} {'Sharpe':<8} {'MDD%':<7} {'Score':<7}")
    print("="*120)

    for idx, row in results_df.head(10).iterrows():
        print(f"{row['combination_id']:<6} {row['rsi_period']:<5} {row['macd_fast']}/{row['macd_slow']}/{row['macd_signal']:<6} {row['bb_period']:<5} {row['atr_period']:<5} {row['ema_period']:<5} {row['total_return']:>7.2f}% {row['win_rate']:>6.1f}% {row['sharpe_ratio']:>7.3f} {row['max_drawdown']:>6.2f}% {row['composite_score']:>6.4f}")

    # Save full results
    results_path = RESULTS_DIR / f"lookback_optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n✅ Full results saved to: {results_path.name}")

    # Best combination
    best = results_df.iloc[0]
    print(f"\n{'='*80}")
    print("OPTIMAL COMBINATION (Rank 1)")
    print(f"{'='*80}")
    print(f"""
Parameters:
  RSI Period:      {best['rsi_period']}
  MACD:            {best['macd_fast']}/{best['macd_slow']}/{best['macd_signal']} (fast/slow/signal)
  BB Period:       {best['bb_period']}
  ATR Period:      {best['atr_period']}
  EMA Period:      {best['ema_period']}

Performance:
  Total Return:    {best['total_return']:+.2f}%
  Win Rate:        {best['win_rate']:.1f}%
  Sharpe Ratio:    {best['sharpe_ratio']:.3f}
  Max Drawdown:    {best['max_drawdown']:.2f}%
  Num Trades:      {int(best['num_trades'])}
  Final Balance:   ${best['final_balance']:,.2f}

Composite Score: {best['composite_score']:.4f}
""")

    # Comparison with baseline (RSI=14, MACD=12/26/9, BB=20, ATR=14, EMA=12)
    baseline = results_df[
        (results_df['rsi_period'] == 14) &
        (results_df['macd_fast'] == 12) &
        (results_df['macd_slow'] == 26) &
        (results_df['bb_period'] == 20) &
        (results_df['atr_period'] == 14) &
        (results_df['ema_period'] == 12)
    ]

    if len(baseline) > 0:
        baseline = baseline.iloc[0]
        print(f"\n{'='*80}")
        print("COMPARISON: Optimal vs Baseline (Standard Periods)")
        print(f"{'='*80}")
        print(f"""
BASELINE (RSI=14, MACD=12/26/9, BB=20, ATR=14, EMA=12):
  Return:    {baseline['total_return']:+.2f}%
  Win Rate:  {baseline['win_rate']:.1f}%
  Sharpe:    {baseline['sharpe_ratio']:.3f}
  Max DD:    {baseline['max_drawdown']:.2f}%

OPTIMAL:
  Return:    {best['total_return']:+.2f}%  ({(best['total_return']/baseline['total_return']-1)*100:+.1f}%)
  Win Rate:  {best['win_rate']:.1f}%  ({(best['win_rate']/baseline['win_rate']-1)*100:+.1f}%)
  Sharpe:    {best['sharpe_ratio']:.3f}  ({(best['sharpe_ratio']/baseline['sharpe_ratio']-1)*100:+.1f}%)
  Max DD:    {best['max_drawdown']:.2f}%  ({(best['max_drawdown']/baseline['max_drawdown']-1)*100:+.1f}%)

Improvement: {(best['composite_score']/baseline['composite_score']-1)*100:+.1f}%
""")
else:
    print("⚠️ No successful backtest results")

print(f"\n{'='*80}")
print("GRID SEARCH COMPLETE")
print(f"{'='*80}\n")
