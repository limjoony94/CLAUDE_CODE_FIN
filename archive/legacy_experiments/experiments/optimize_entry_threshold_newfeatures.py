"""
Entry Threshold Optimization for New Feature Models
====================================================

Tests thresholds [0.75, 0.80, 0.85, 0.90] to find optimal 3-8 trades/day frequency.

Context:
- Models CAN reach 0.75 with excellent quality (67.9% WR, +20% benchmark outperformance)
- But frequency too high (23.79/day vs target 3-8/day)
- Testing HIGHER thresholds for optimal selectivity

Created: 2025-10-29
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import joblib
from datetime import datetime

def load_models(timestamp="20251029_191359"):
    """Load Entry models with new features"""
    models_dir = PROJECT_ROOT / "models"

    # LONG Entry
    long_entry_model = joblib.load(models_dir / f"xgboost_long_entry_newfeatures_{timestamp}.pkl")
    long_entry_scaler = joblib.load(models_dir / f"xgboost_long_entry_newfeatures_{timestamp}_scaler.pkl")
    with open(models_dir / f"xgboost_long_entry_newfeatures_{timestamp}_features.txt", 'r') as f:
        long_entry_features = [line.strip() for line in f]

    # SHORT Entry
    short_entry_model = joblib.load(models_dir / f"xgboost_short_entry_newfeatures_{timestamp}.pkl")
    short_entry_scaler = joblib.load(models_dir / f"xgboost_short_entry_newfeatures_{timestamp}_scaler.pkl")
    with open(models_dir / f"xgboost_short_entry_newfeatures_{timestamp}_features.txt", 'r') as f:
        short_entry_features = [line.strip() for line in f]

    # Exit Models (UPDATED 2025-10-29: Use same models as original backtest for fair comparison)
    long_exit_model = joblib.load(models_dir / "xgboost_long_exit_threshold_075_20251027_190512.pkl")
    long_exit_scaler = joblib.load(models_dir / "xgboost_long_exit_threshold_075_20251027_190512_scaler.pkl")
    with open(models_dir / "xgboost_long_exit_threshold_075_20251027_190512_features.txt", 'r') as f:
        long_exit_features = [line.strip() for line in f]

    short_exit_model = joblib.load(models_dir / "xgboost_short_exit_threshold_075_20251027_190512.pkl")
    short_exit_scaler = joblib.load(models_dir / "xgboost_short_exit_threshold_075_20251027_190512_scaler.pkl")
    with open(models_dir / "xgboost_short_exit_threshold_075_20251027_190512_features.txt", 'r') as f:
        short_exit_features = [line.strip() for line in f]

    return {
        'long_entry': (long_entry_model, long_entry_scaler, long_entry_features),
        'short_entry': (short_entry_model, short_entry_scaler, short_entry_features),
        'long_exit': (long_exit_model, long_exit_scaler, long_exit_features),
        'short_exit': (short_exit_model, short_exit_scaler, short_exit_features)
    }

def prepare_exit_features(df):
    """Prepare additional features needed for Exit models"""
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    df['volume_surge'] = (df['volume'] > df['volume'].rolling(20).mean() * 1.5).astype(float)

    if 'ma_50' not in df.columns:
        df['ma_50'] = df['close'].rolling(50).mean()
    if 'ma_200' not in df.columns:
        df['ma_200'] = df['close'].rolling(200).mean()

    df['price_vs_ma20'] = (df['close'] - df['ma_50']) / df['ma_50']
    df['price_vs_ma50'] = (df['close'] - df['ma_200']) / df['ma_200']

    df['returns'] = df['close'].pct_change()
    df['volatility_20'] = df['returns'].rolling(20).std()
    df['volatility_regime'] = (df['volatility_20'] > df['volatility_20'].rolling(100).median()).astype(float)

    if 'rsi' in df.columns:
        df['rsi_slope'] = df['rsi'].diff(3) / 3
        df['rsi_overbought'] = (df['rsi'] > 70).astype(float)
        df['rsi_oversold'] = (df['rsi'] < 30).astype(float)
    else:
        df['rsi_slope'] = 0
        df['rsi_overbought'] = 0
        df['rsi_oversold'] = 0

    df['rsi_divergence'] = 0

    if 'macd_hist' in df.columns:
        df['macd_histogram_slope'] = df['macd_hist'].diff(3) / 3
    else:
        df['macd_histogram_slope'] = 0

    if 'macd' in df.columns and 'macd_signal' in df.columns:
        df['macd_crossover'] = ((df['macd'] > df['macd_signal']) &
                                (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(float)
        df['macd_crossunder'] = ((df['macd'] < df['macd_signal']) &
                                 (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(float)
    else:
        df['macd_crossover'] = 0
        df['macd_crossunder'] = 0

    df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(float)
    df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(float)
    df['price_acceleration'] = df['close'].diff().diff()
    df['near_resistance'] = 0
    df['near_support'] = 0

    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    else:
        df['bb_position'] = 0.5

    df = df.ffill().bfill()
    return df

def backtest_threshold(df, models, entry_threshold, ml_exit_threshold=0.75):
    """Backtest with specific entry threshold"""

    # DEBUG: Print header for this threshold test
    print(f"\n{'='*60}")
    print(f"TESTING THRESHOLD: {entry_threshold:.2f}")
    print(f"{'='*60}")

    # Unpack models
    long_entry_model, long_entry_scaler, long_entry_features = models['long_entry']
    short_entry_model, short_entry_scaler, short_entry_features = models['short_entry']
    long_exit_model, long_exit_scaler, long_exit_features = models['long_exit']
    short_exit_model, short_exit_scaler, short_exit_features = models['short_exit']

    # Config
    LEVERAGE = 4
    EMERGENCY_STOP_LOSS = -0.03
    EMERGENCY_MAX_HOLD = 120
    INITIAL_CAPITAL = 10000

    # State
    balance = INITIAL_CAPITAL
    position = None
    trades = []

    # DEBUG: Add comprehensive logging for first few iterations
    debug_iteration_count = 0
    debug_max_iterations = 10  # Increased to see more iterations

    # Backtest loop
    for i in range(100, len(df) - EMERGENCY_MAX_HOLD):
        row = df.iloc[i]
        price = row['close']

        # Exit Logic
        if position is not None:
            side = position['side']
            entry_price = position['entry_price']
            entry_index = position['entry_index']
            hold_time = i - entry_index

            # Calculate P&L
            if side == 'LONG':
                price_change_pct = (price - entry_price) / entry_price
            else:
                price_change_pct = (entry_price - price) / entry_price

            leveraged_pnl_pct = price_change_pct * LEVERAGE

            # Check exits
            exit_reason = None

            # Emergency Stop Loss
            if leveraged_pnl_pct <= EMERGENCY_STOP_LOSS:
                exit_reason = "STOP_LOSS"

            # Max Hold Time
            elif hold_time >= EMERGENCY_MAX_HOLD:
                exit_reason = "MAX_HOLD"

            # ML Exit
            else:
                if side == 'LONG':
                    exit_model, exit_scaler, exit_features = models['long_exit']
                else:
                    exit_model, exit_scaler, exit_features = models['short_exit']

                X_exit = row[exit_features].values.reshape(1, -1)
                X_exit_scaled = exit_scaler.transform(X_exit)
                exit_prob = exit_model.predict_proba(X_exit_scaled)[0][1]

                if exit_prob >= ml_exit_threshold:
                    exit_reason = "ML_EXIT"

            # Execute exit
            if exit_reason is not None:
                # BUG FIX: Use position_size_pct from position dict
                position_size_pct = position['position_size_pct']
                pnl_dollars = balance * position_size_pct * leveraged_pnl_pct
                balance += pnl_dollars

                trades.append({
                    'entry_time': df.iloc[entry_index]['timestamp'],
                    'exit_time': row['timestamp'],
                    'side': side,
                    'entry_price': entry_price,
                    'exit_price': price,
                    'entry_prob': position['entry_prob'],
                    'position_size_pct': position_size_pct,
                    'hold_time': hold_time,
                    'pnl_pct': leveraged_pnl_pct,
                    'pnl_dollars': pnl_dollars,
                    'exit_reason': exit_reason,
                    'balance': balance
                })

                position = None

        # Entry Logic
        if position is None:
            # LONG signal
            X_long = row[long_entry_features].values.reshape(1, -1)
            X_long_scaled = long_entry_scaler.transform(X_long)
            long_prob = long_entry_model.predict_proba(X_long_scaled)[0][1]

            # SHORT signal
            X_short = row[short_entry_features].values.reshape(1, -1)
            X_short_scaled = short_entry_scaler.transform(X_short)
            short_prob = short_entry_model.predict_proba(X_short_scaled)[0][1]

            # DEBUG: Log first few iterations to trace threshold logic
            if debug_iteration_count < debug_max_iterations:
                print(f"[DEBUG] Iter {i}: LONG_prob={long_prob:.4f}, SHORT_prob={short_prob:.4f}, threshold={entry_threshold:.2f}")
                print(f"[DEBUG]   LONG >= threshold? {long_prob >= entry_threshold}")
                print(f"[DEBUG]   SHORT >= threshold? {short_prob >= entry_threshold}")
                debug_iteration_count += 1

            # Entry decision with dynamic position sizing
            if long_prob >= entry_threshold:
                entry_prob = long_prob

                # Dynamic Position Sizing (matching original backtest)
                if entry_prob < 0.65:
                    position_size_pct = 0.20
                elif entry_prob >= 0.85:
                    position_size_pct = 0.95
                else:
                    position_size_pct = 0.20 + (0.95 - 0.20) * ((entry_prob - 0.65) / (0.85 - 0.65))

                # DEBUG: Log first 3 positions
                if len(trades) < 3:
                    print(f"[DEBUG] LONG Entry #{len(trades)+1}: prob={entry_prob:.4f}, position_size={position_size_pct:.2%}")

                position = {
                    'side': 'LONG',
                    'entry_price': price,
                    'entry_index': i,
                    'entry_prob': entry_prob,
                    'position_size_pct': position_size_pct
                }
            elif short_prob >= entry_threshold:
                entry_prob = short_prob

                # Dynamic Position Sizing (matching original backtest)
                if entry_prob < 0.65:
                    position_size_pct = 0.20
                elif entry_prob >= 0.85:
                    position_size_pct = 0.95
                else:
                    position_size_pct = 0.20 + (0.95 - 0.20) * ((entry_prob - 0.65) / (0.85 - 0.65))

                position = {
                    'side': 'SHORT',
                    'entry_price': price,
                    'entry_index': i,
                    'entry_prob': entry_prob,
                    'position_size_pct': position_size_pct
                }

    # Calculate metrics
    if len(trades) == 0:
        return None

    trades_df = pd.DataFrame(trades)

    # Performance metrics
    total_return = ((balance - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    wins = trades_df[trades_df['pnl_pct'] > 0]
    losses = trades_df[trades_df['pnl_pct'] <= 0]
    win_rate = len(wins) / len(trades_df) * 100

    # Calculate trade frequency (trades per day)
    test_days = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[100]).total_seconds() / 86400
    trades_per_day = len(trades_df) / test_days

    # Direction split
    long_trades = len(trades_df[trades_df['side'] == 'LONG'])
    short_trades = len(trades_df[trades_df['side'] == 'SHORT'])

    # Exit distribution
    ml_exits = len(trades_df[trades_df['exit_reason'] == 'ML_EXIT'])
    sl_exits = len(trades_df[trades_df['exit_reason'] == 'STOP_LOSS'])
    mh_exits = len(trades_df[trades_df['exit_reason'] == 'MAX_HOLD'])

    # Benchmark
    market_change = ((df['close'].iloc[-1] - df['close'].iloc[100]) / df['close'].iloc[100]) * 100
    leveraged_market = market_change * LEVERAGE
    outperformance = total_return - leveraged_market

    return {
        'entry_threshold': entry_threshold,
        'total_trades': len(trades_df),
        'trades_per_day': trades_per_day,
        'total_return': total_return,
        'win_rate': win_rate,
        'long_pct': (long_trades / len(trades_df)) * 100,
        'short_pct': (short_trades / len(trades_df)) * 100,
        'ml_exit_pct': (ml_exits / len(trades_df)) * 100,
        'sl_exit_pct': (sl_exits / len(trades_df)) * 100,
        'mh_exit_pct': (mh_exits / len(trades_df)) * 100,
        'market_change': market_change,
        'leveraged_market': leveraged_market,
        'outperformance': outperformance,
        'avg_trade': trades_df['pnl_pct'].mean() * 100,
        'best_trade': trades_df['pnl_pct'].max() * 100,
        'worst_trade': trades_df['pnl_pct'].min() * 100,
        'avg_hold': trades_df['hold_time'].mean()
    }

def main():
    print("="*80)
    print("ENTRY THRESHOLD OPTIMIZATION - NEW FEATURE MODELS")
    print("="*80)
    print()

    # Load data
    print("Loading data with new features...")
    data_file = PROJECT_ROOT / "data" / "features" / "BTCUSDT_5m_pattern_features.csv"
    df = pd.read_csv(data_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"✅ Loaded {len(df):,} candles")
    print()

    # Use last 14 days as holdout
    holdout_start = df['timestamp'].iloc[-1] - pd.Timedelta(days=14)
    df_holdout = df[df['timestamp'] >= holdout_start].copy()
    print(f"Holdout period: {df_holdout['timestamp'].iloc[0]} to {df_holdout['timestamp'].iloc[-1]}")
    print(f"Holdout candles: {len(df_holdout):,} ({14} days)")
    print()

    # Prepare exit features (including 'returns')
    print("Preparing exit features...")
    df_holdout = prepare_exit_features(df_holdout)
    print("✅ Exit features prepared")
    print()

    # Load models
    print("Loading models...")
    models = load_models()
    print("✅ All models loaded")
    print()

    # Test thresholds
    thresholds = [0.75, 0.80, 0.85, 0.90]
    results = []

    print("Testing thresholds...")
    print("-" * 80)

    for threshold in thresholds:
        print(f"\nThreshold: {threshold:.2f}")
        result = backtest_threshold(df_holdout, models, threshold)

        if result is None:
            print("  ⚠️ No trades generated at this threshold")
            continue

        results.append(result)

        # Print summary
        print(f"  Total Trades: {result['total_trades']}")
        print(f"  Trades/Day: {result['trades_per_day']:.2f} {'✅' if 3 <= result['trades_per_day'] <= 8 else '❌'}")
        print(f"  Total Return: {result['total_return']:.2f}%")
        print(f"  Win Rate: {result['win_rate']:.1f}%")
        print(f"  LONG/SHORT: {result['long_pct']:.1f}% / {result['short_pct']:.1f}%")
        print(f"  Outperformance: {result['outperformance']:+.2f}%")

    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print()

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Find optimal threshold (3-8 trades/day)
    in_target = results_df[
        (results_df['trades_per_day'] >= 3) &
        (results_df['trades_per_day'] <= 8)
    ]

    if len(in_target) > 0:
        # Among those in target, pick highest return
        optimal = in_target.loc[in_target['total_return'].idxmax()]
        print("🎯 OPTIMAL THRESHOLD FOUND:")
        print(f"  Threshold: {optimal['entry_threshold']:.2f}")
        print(f"  Trades/Day: {optimal['trades_per_day']:.2f} ✅ (target: 3-8)")
        print(f"  Total Return: {optimal['total_return']:.2f}%")
        print(f"  Win Rate: {optimal['win_rate']:.1f}%")
        print(f"  Outperformance: {optimal['outperformance']:+.2f}%")
        print()
    else:
        print("⚠️ No threshold achieved 3-8 trades/day target")
        print()

        # Show closest
        results_df['distance_from_target'] = results_df['trades_per_day'].apply(
            lambda x: abs(x - 5.5)  # 5.5 is middle of 3-8 range
        )
        closest = results_df.loc[results_df['distance_from_target'].idxmin()]
        print("📊 CLOSEST TO TARGET:")
        print(f"  Threshold: {closest['entry_threshold']:.2f}")
        print(f"  Trades/Day: {closest['trades_per_day']:.2f}")
        print(f"  Total Return: {closest['total_return']:.2f}%")
        print(f"  Win Rate: {closest['win_rate']:.1f}%")
        print()

    # Full comparison table
    print("Full Comparison:")
    print()
    print("Threshold | Trades | Trades/Day | Return | WR    | LONG/SHORT    | Outperf | ML Exit")
    print("-" * 90)
    for _, r in results_df.iterrows():
        target_marker = "✅" if 3 <= r['trades_per_day'] <= 8 else "  "
        print(f"  {r['entry_threshold']:.2f}    | {int(r['total_trades']):4d}   | "
              f"{r['trades_per_day']:5.2f} {target_marker}  | "
              f"{r['total_return']:+6.2f}% | {r['win_rate']:5.1f}% | "
              f"{r['long_pct']:5.1f}%/{r['short_pct']:5.1f}% | "
              f"{r['outperformance']:+6.2f}% | {r['ml_exit_pct']:5.1f}%")
    print()

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = PROJECT_ROOT / "results" / f"threshold_optimization_newfeatures_{timestamp}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"✅ Results saved: {output_file.name}")
    print()

if __name__ == '__main__':
    main()
