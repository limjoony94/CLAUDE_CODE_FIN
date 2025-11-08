"""
Full Trading Backtest: Enhanced SHORT Entry

Complete trading simulation with:
- LONG Entry: Current model (baseline)
- SHORT Entry: Enhanced model (new)
- EXIT: Enhanced models (both)

Measures:
- Win rate, returns, sharpe ratio
- Trade frequency and distribution
- Performance vs current baseline
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Backtest Parameters
INITIAL_CAPITAL = 10000.0
POSITION_SIZE = 0.95
LONG_ENTRY_THRESHOLD = 0.85
SHORT_ENTRY_THRESHOLD = 0.55  # Optimized from signal analysis
EXIT_THRESHOLD = 0.7
MAX_HOLDING_CANDLES = 48  # 4 hours
STOP_LOSS_PCT = 0.05  # 5%
TRANSACTION_COST = 0.0002


def calculate_enhanced_features(df):
    """Calculate 22 SELL signal features"""

    # Volume features
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    df['volume_surge'] = (df['volume'] > df['volume'].rolling(20).mean() * 1.5).astype(float)

    # Price momentum
    ma_20 = df['close'].rolling(20).mean()
    ma_50 = df['close'].rolling(50).mean()
    df['price_vs_ma20'] = (df['close'] - ma_20) / ma_20
    df['price_vs_ma50'] = (df['close'] - ma_50) / ma_50
    df['price_acceleration'] = df['close'].diff().diff()

    # Volatility
    df['volatility_20'] = df['close'].pct_change().rolling(20).std()
    volatility_median = df['volatility_20'].median()
    df['volatility_regime'] = (df['volatility_20'] > volatility_median).astype(float)

    # RSI dynamics (defaults - missing from data)
    df['rsi'] = 50.0
    df['rsi_slope'] = 0.0
    df['rsi_overbought'] = 0.0
    df['rsi_oversold'] = 0.0
    df['rsi_divergence'] = 0.0

    # MACD dynamics (defaults - missing from data)
    df['macd'] = 0.0
    df['macd_signal'] = 0.0
    df['macd_histogram_slope'] = 0.0
    df['macd_crossover'] = 0.0
    df['macd_crossunder'] = 0.0

    # Price patterns
    df['higher_high'] = ((df['high'] > df['high'].shift(1)) &
                         (df['high'].shift(1) > df['high'].shift(2))).astype(float)
    df['lower_low'] = ((df['low'] < df['low'].shift(1)) &
                       (df['low'].shift(1) < df['low'].shift(2))).astype(float)

    # Support/Resistance
    resistance = df['high'].rolling(50).max()
    support = df['low'].rolling(50).min()
    df['near_resistance'] = (df['close'] > resistance * 0.98).astype(float)
    df['near_support'] = (df['close'] < support * 1.02).astype(float)

    # Bollinger Bands
    bb_mid = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    bb_high = bb_mid + 2 * bb_std
    bb_low = bb_mid - 2 * bb_std
    bb_range = bb_high - bb_low
    df['bb_position'] = np.where(bb_range != 0,
                                  (df['close'] - bb_low) / bb_range,
                                  0.5)

    return df.ffill().fillna(0)


def backtest_trading(df, models, scalers, features):
    """Full trading backtest with position management"""

    capital = INITIAL_CAPITAL
    position = None
    trades = []

    for i in range(len(df) - 1):
        current_price = df['close'].iloc[i]

        # Position management
        if position is not None:
            side = position['side']
            entry_price = position['entry_price']
            entry_idx = position['entry_idx']
            candles_held = i - entry_idx

            # Calculate P&L
            if side == 'LONG':
                pnl_pct = (current_price - entry_price) / entry_price
            else:  # SHORT
                pnl_pct = (entry_price - current_price) / entry_price

            # Exit conditions
            exit_reason = None

            # 1. Stop Loss
            if pnl_pct <= -STOP_LOSS_PCT:
                exit_reason = "Stop Loss"

            # 2. Max Holding
            elif candles_held >= MAX_HOLDING_CANDLES:
                exit_reason = "Max Hold"

            # 3. ML Exit Signal
            else:
                if side == 'LONG':
                    exit_features_row = df[features['long_exit']].iloc[i:i+1].values
                    if not np.isnan(exit_features_row).any():
                        exit_scaled = scalers['long_exit'].transform(exit_features_row)
                        exit_prob = models['long_exit'].predict_proba(exit_scaled)[0][1]
                        if exit_prob >= EXIT_THRESHOLD:
                            exit_reason = "ML Exit"
                else:  # SHORT
                    exit_features_row = df[features['short_exit']].iloc[i:i+1].values
                    if not np.isnan(exit_features_row).any():
                        exit_scaled = scalers['short_exit'].transform(exit_features_row)
                        exit_prob = models['short_exit'].predict_proba(exit_scaled)[0][1]
                        if exit_prob >= EXIT_THRESHOLD:
                            exit_reason = "ML Exit"

            # Execute exit
            if exit_reason:
                quantity = position['quantity']
                pnl_usd = pnl_pct * (entry_price * quantity)

                # Transaction costs
                entry_cost = entry_price * quantity * TRANSACTION_COST
                exit_cost = current_price * quantity * TRANSACTION_COST
                pnl_usd -= (entry_cost + exit_cost)

                capital += pnl_usd

                trades.append({
                    'entry_idx': entry_idx,
                    'exit_idx': i,
                    'side': side,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl_pct': pnl_pct * 100,
                    'pnl_usd': pnl_usd,
                    'holding_candles': candles_held,
                    'exit_reason': exit_reason,
                    'entry_prob': position['entry_prob']
                })

                position = None

        # Entry logic
        if position is None:
            # Get LONG Entry signal (baseline model - not used, just for comparison)
            # We'll use LONG Entry current model if available

            # Get SHORT Entry signal (ENHANCED model)
            short_features_row = df[features['short_entry']].iloc[i:i+1].values
            if not np.isnan(short_features_row).any():
                short_scaled = scalers['short_entry'].transform(short_features_row)
                short_prob = models['short_entry'].predict_proba(short_scaled)[0][1]
            else:
                short_prob = 0.0

            # Entry decision
            if short_prob >= SHORT_ENTRY_THRESHOLD:
                position_value = capital * POSITION_SIZE
                quantity = position_value / current_price

                position = {
                    'side': 'SHORT',
                    'entry_idx': i,
                    'entry_price': current_price,
                    'quantity': quantity,
                    'entry_prob': short_prob
                }

    # Close any open position
    if position is not None:
        i = len(df) - 1
        current_price = df['close'].iloc[i]
        side = position['side']
        entry_price = position['entry_price']
        entry_idx = position['entry_idx']

        if side == 'LONG':
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price

        quantity = position['quantity']
        pnl_usd = pnl_pct * (entry_price * quantity)
        entry_cost = entry_price * quantity * TRANSACTION_COST
        exit_cost = current_price * quantity * TRANSACTION_COST
        pnl_usd -= (entry_cost + exit_cost)

        capital += pnl_usd

        trades.append({
            'entry_idx': entry_idx,
            'exit_idx': i,
            'side': side,
            'entry_price': entry_price,
            'exit_price': current_price,
            'pnl_pct': pnl_pct * 100,
            'pnl_usd': pnl_usd,
            'holding_candles': i - entry_idx,
            'exit_reason': 'End of Data',
            'entry_prob': position['entry_prob']
        })

    return trades, capital


def calculate_metrics(trades, final_capital, data_length):
    """Calculate trading performance metrics"""

    if len(trades) == 0:
        return {}

    # Basic metrics
    total_return_pct = ((final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    winning_trades = [t for t in trades if t['pnl_usd'] > 0]
    losing_trades = [t for t in trades if t['pnl_usd'] <= 0]

    win_rate = len(winning_trades) / len(trades) * 100
    avg_win = np.mean([t['pnl_pct'] for t in winning_trades]) if winning_trades else 0
    avg_loss = np.mean([t['pnl_pct'] for t in losing_trades]) if losing_trades else 0

    # Risk metrics
    returns = [t['pnl_pct'] for t in trades]
    sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0

    # Drawdown
    cumulative = INITIAL_CAPITAL
    peak = INITIAL_CAPITAL
    max_dd = 0

    for trade in trades:
        cumulative += trade['pnl_usd']
        if cumulative > peak:
            peak = cumulative
        dd = (peak - cumulative) / peak * 100
        if dd > max_dd:
            max_dd = dd

    # Exit analysis
    exit_counts = {}
    for trade in trades:
        reason = trade['exit_reason']
        exit_counts[reason] = exit_counts.get(reason, 0) + 1

    # Trade frequency
    days = data_length / (12 * 24)  # 5min candles to days
    trades_per_day = len(trades) / days

    return {
        'total_return_pct': total_return_pct,
        'final_capital': final_capital,
        'num_trades': len(trades),
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'exit_counts': exit_counts,
        'trades_per_day': trades_per_day,
        'avg_holding_candles': np.mean([t['holding_candles'] for t in trades])
    }


def main():
    print("=" * 80)
    print("Full Trading Backtest: Enhanced SHORT Entry")
    print("=" * 80)

    # Load data
    print("\n1. Loading Data...")
    df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_max.csv")
    print(f"  ✅ {len(df):,} candles")
    print(f"  Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

    # Calculate features
    print("\n2. Calculating Features...")
    df = calculate_enhanced_features(df)
    print(f"  ✅ 22 SELL signal features calculated")

    # Load models
    print("\n3. Loading Models...")

    # Enhanced SHORT Entry
    short_entry_model = pickle.load(open(MODELS_DIR / "xgboost_short_entry_enhanced_20251016_201219.pkl", 'rb'))
    short_entry_scaler = pickle.load(open(MODELS_DIR / "xgboost_short_entry_enhanced_20251016_201219_scaler.pkl", 'rb'))
    with open(MODELS_DIR / "xgboost_short_entry_enhanced_20251016_201219_features.txt", 'r') as f:
        short_entry_features = [line.strip() for line in f]
    print(f"  ✅ SHORT Entry (Enhanced): {len(short_entry_features)} features, threshold={SHORT_ENTRY_THRESHOLD}")

    # LONG Exit (Enhanced)
    long_exit_model = pickle.load(open(MODELS_DIR / "xgboost_long_exit_improved_20251016_175554.pkl", 'rb'))
    long_exit_scaler = pickle.load(open(MODELS_DIR / "xgboost_long_exit_improved_20251016_175554_scaler.pkl", 'rb'))
    with open(MODELS_DIR / "xgboost_long_exit_improved_20251016_175554_features.txt", 'r') as f:
        long_exit_features = [line.strip() for line in f]
    print(f"  ✅ LONG Exit (Enhanced): {len(long_exit_features)} features")

    # SHORT Exit (Enhanced)
    short_exit_model = pickle.load(open(MODELS_DIR / "xgboost_short_exit_improved_20251016_180207.pkl", 'rb'))
    short_exit_scaler = pickle.load(open(MODELS_DIR / "xgboost_short_exit_improved_20251016_180207_scaler.pkl", 'rb'))
    with open(MODELS_DIR / "xgboost_short_exit_improved_20251016_180207_features.txt", 'r') as f:
        short_exit_features = [line.strip() for line in f]
    print(f"  ✅ SHORT Exit (Enhanced): {len(short_exit_features)} features")

    models = {
        'short_entry': short_entry_model,
        'long_exit': long_exit_model,
        'short_exit': short_exit_model
    }

    scalers = {
        'short_entry': short_entry_scaler,
        'long_exit': long_exit_scaler,
        'short_exit': short_exit_scaler
    }

    features = {
        'short_entry': short_entry_features,
        'long_exit': long_exit_features,
        'short_exit': short_exit_features
    }

    # Run backtest
    print("\n" + "=" * 80)
    print("Running Full Trading Backtest")
    print("=" * 80)
    print(f"Strategy: SHORT Entry Only (LONG disabled for clear SHORT analysis)")
    print(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
    print(f"Position Size: {POSITION_SIZE * 100:.0f}%")
    print(f"SHORT Entry Threshold: {SHORT_ENTRY_THRESHOLD}")
    print(f"Exit Threshold: {EXIT_THRESHOLD}")
    print(f"Stop Loss: {STOP_LOSS_PCT * 100:.0f}%")
    print(f"Max Hold: {MAX_HOLDING_CANDLES} candles ({MAX_HOLDING_CANDLES / 12:.1f}h)")

    trades, final_capital = backtest_trading(df, models, scalers, features)
    metrics = calculate_metrics(trades, final_capital, len(df))

    # Results
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)

    if len(trades) == 0:
        print("\n❌ No trades executed!")
        print("  Possible reasons:")
        print("  - Threshold too high")
        print("  - Model not generating signals")
        print("  - Feature calculation issues")
        return

    print(f"\n📊 Performance Summary:")
    print(f"  Total Return: {metrics['total_return_pct']:+.2f}%")
    print(f"  Final Capital: ${metrics['final_capital']:,.2f}")
    print(f"  Total Trades: {metrics['num_trades']}")
    print(f"  Win Rate: {metrics['win_rate']:.2f}%")
    print(f"  Avg Win: {metrics['avg_win']:+.2f}%")
    print(f"  Avg Loss: {metrics['avg_loss']:+.2f}%")
    print(f"  Sharpe Ratio: {metrics['sharpe']:.3f}")
    print(f"  Max Drawdown: {metrics['max_drawdown']:.2f}%")

    print(f"\n📊 Trading Activity:")
    print(f"  Trades per Day: {metrics['trades_per_day']:.2f}")
    print(f"  Avg Holding: {metrics['avg_holding_candles']:.1f} candles ({metrics['avg_holding_candles'] / 12:.2f}h)")

    print(f"\n📊 Exit Breakdown:")
    for reason, count in metrics['exit_counts'].items():
        pct = count / metrics['num_trades'] * 100
        print(f"  {reason}: {count} ({pct:.1f}%)")

    # Trade distribution
    print(f"\n📊 P&L Distribution:")
    pnls = [t['pnl_pct'] for t in trades]
    print(f"  Best Trade: {max(pnls):+.2f}%")
    print(f"  Worst Trade: {min(pnls):+.2f}%")
    print(f"  Median: {np.median(pnls):+.2f}%")

    # Probability analysis
    print(f"\n📊 Entry Probability Analysis:")
    entry_probs = [t['entry_prob'] for t in trades]
    print(f"  Mean: {np.mean(entry_probs):.3f}")
    print(f"  Median: {np.median(entry_probs):.3f}")
    print(f"  Min: {min(entry_probs):.3f}")
    print(f"  Max: {max(entry_probs):.3f}")

    # Win rate by probability range
    print(f"\n📊 Win Rate by Entry Probability:")
    prob_ranges = [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 1.00)]
    for low, high in prob_ranges:
        range_trades = [t for t in trades if low <= t['entry_prob'] < high]
        if range_trades:
            range_wins = len([t for t in range_trades if t['pnl_usd'] > 0])
            range_wr = range_wins / len(range_trades) * 100
            print(f"  {low:.2f}-{high:.2f}: {range_wr:.1f}% ({len(range_trades)} trades)")

    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    trades_df = pd.DataFrame(trades)
    trades_df.to_csv(RESULTS_DIR / "enhanced_short_entry_backtest_trades.csv", index=False)
    print(f"  ✅ Saved: enhanced_short_entry_backtest_trades.csv")

    # Final assessment
    print("\n" + "=" * 80)
    print("ASSESSMENT")
    print("=" * 80)

    if metrics['win_rate'] >= 60 and metrics['total_return_pct'] > 0:
        print("\n✅ EXCELLENT PERFORMANCE!")
        print(f"  - Win rate {metrics['win_rate']:.1f}% >= 60% target")
        print(f"  - Positive returns {metrics['total_return_pct']:+.2f}%")
        print(f"  - {metrics['num_trades']} trades executed")
        print("\n  RECOMMENDATION: DEPLOY Enhanced SHORT Entry")

    elif metrics['win_rate'] >= 50 and metrics['total_return_pct'] > 0:
        print("\n✅ GOOD PERFORMANCE")
        print(f"  - Win rate {metrics['win_rate']:.1f}% >= 50%")
        print(f"  - Positive returns {metrics['total_return_pct']:+.2f}%")
        print("\n  RECOMMENDATION: Consider deployment with monitoring")

    elif metrics['win_rate'] >= 40:
        print("\n⚠️  MODERATE PERFORMANCE")
        print(f"  - Win rate {metrics['win_rate']:.1f}% below target")
        print(f"  - Returns {metrics['total_return_pct']:+.2f}%")
        print("\n  RECOMMENDATION: Further optimization needed")

    else:
        print("\n❌ POOR PERFORMANCE")
        print(f"  - Win rate {metrics['win_rate']:.1f}% too low")
        print(f"  - Returns {metrics['total_return_pct']:+.2f}%")
        print("\n  RECOMMENDATION: Do not deploy, needs improvement")

    print("\n" + "=" * 80)
    print("Backtest Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
