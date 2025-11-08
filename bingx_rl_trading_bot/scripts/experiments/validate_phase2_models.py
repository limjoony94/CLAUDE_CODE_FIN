"""
Validate Phase 2 Enhanced Models on 28-Day Holdout

Compares:
- Phase 2 Enhanced models (20251102_200202) - Walk-Forward Decoupled + Enhanced Features
- Current models (20251027_194313 Entry + 20251027_190512 Exit)

Training: Jul 14 - Sep 28 (21,940 candles)
Test Period: Sep 28 - Oct 26, 2025 (28 days, 8,064 candles)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

# Paths
DATA_DIR = BASE_DIR / "data" / "features"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Phase 2 Enhanced Models (NEW - Walk-Forward Decoupled + Enhanced Features)
PHASE2_LONG_ENTRY = MODELS_DIR / "xgboost_long_entry_phase2_enhanced_20251102_200202.pkl"
PHASE2_SHORT_ENTRY = MODELS_DIR / "xgboost_short_entry_phase2_enhanced_20251102_200202.pkl"
PHASE2_LONG_EXIT = MODELS_DIR / "xgboost_long_exit_phase2_enhanced_20251102_200202.pkl"
PHASE2_SHORT_EXIT = MODELS_DIR / "xgboost_short_exit_phase2_enhanced_20251102_200202.pkl"

# Current Production Models
CURRENT_LONG_ENTRY = MODELS_DIR / "xgboost_long_entry_walkforward_decoupled_20251027_194313.pkl"
CURRENT_SHORT_ENTRY = MODELS_DIR / "xgboost_short_entry_walkforward_decoupled_20251027_194313.pkl"
CURRENT_LONG_EXIT = MODELS_DIR / "xgboost_long_exit_threshold_075_20251027_190512.pkl"
CURRENT_SHORT_EXIT = MODELS_DIR / "xgboost_short_exit_threshold_075_20251027_190512.pkl"

# Configuration
LONG_THRESHOLD = 0.75
SHORT_THRESHOLD = 0.75
ML_EXIT_THRESHOLD_LONG = 0.75
ML_EXIT_THRESHOLD_SHORT = 0.75
EMERGENCY_STOP_LOSS = 0.03  # -3% balance
EMERGENCY_MAX_HOLD_CANDLES = 120  # 10 hours
LEVERAGE = 4
TAKER_FEE = 0.0005  # 0.05%
INITIAL_CAPITAL = 10000.0


def load_validation_data():
    """Load 28-day validation holdout (Sep 28 - Oct 26)"""
    df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_features_complete.csv")  # 195 features (Enhanced)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Last 28 days = 8,064 candles
    validation_candles = 28 * 24 * 60 // 5
    df_val = df.iloc[-validation_candles:].copy()

    print(f"\n{'='*80}")
    print(f"VALIDATION DATA (28-Day Holdout)")
    print(f"{'='*80}")
    print(f"Period: {df_val['timestamp'].min()} to {df_val['timestamp'].max()}")
    print(f"Candles: {len(df_val):,}")
    print(f"Days: {(df_val['timestamp'].max() - df_val['timestamp'].min()).days}")

    return df_val


def load_models(model_set='phase2'):
    """Load model set"""
    if model_set == 'phase2':
        long_entry_path = PHASE2_LONG_ENTRY
        short_entry_path = PHASE2_SHORT_ENTRY
        long_exit_path = PHASE2_LONG_EXIT
        short_exit_path = PHASE2_SHORT_EXIT
        label = "Phase 2 Enhanced (20251102_200202)"
    else:
        long_entry_path = CURRENT_LONG_ENTRY
        short_entry_path = CURRENT_SHORT_ENTRY
        long_exit_path = CURRENT_LONG_EXIT
        short_exit_path = CURRENT_SHORT_EXIT
        label = "Current Production (20251027)"

    # Load models
    long_entry_model = joblib.load(long_entry_path)
    short_entry_model = joblib.load(short_entry_path)
    long_exit_model = joblib.load(long_exit_path)
    short_exit_model = joblib.load(short_exit_path)

    # Load scalers (both use '_scaler.pkl' suffix)
    long_entry_scaler = joblib.load(str(long_entry_path).replace('.pkl', '_scaler.pkl'))
    short_entry_scaler = joblib.load(str(short_entry_path).replace('.pkl', '_scaler.pkl'))
    long_exit_scaler = joblib.load(str(long_exit_path).replace('.pkl', '_scaler.pkl'))
    short_exit_scaler = joblib.load(str(short_exit_path).replace('.pkl', '_scaler.pkl'))

    # Load features (both use '_features.txt' suffix)
    with open(str(long_entry_path).replace('.pkl', '_features.txt'), 'r') as f:
        long_entry_features = [line.strip() for line in f.readlines()]
    with open(str(short_entry_path).replace('.pkl', '_features.txt'), 'r') as f:
        short_entry_features = [line.strip() for line in f.readlines()]
    with open(str(long_exit_path).replace('.pkl', '_features.txt'), 'r') as f:
        long_exit_features = [line.strip() for line in f.readlines()]
    with open(str(short_exit_path).replace('.pkl', '_features.txt'), 'r') as f:
        short_exit_features = [line.strip() for line in f.readlines()]

    return {
        'label': label,
        'long_entry': (long_entry_model, long_entry_scaler, long_entry_features),
        'short_entry': (short_entry_model, short_entry_scaler, short_entry_features),
        'long_exit': (long_exit_model, long_exit_scaler, long_exit_features),
        'short_exit': (short_exit_model, short_exit_scaler, short_exit_features)
    }


def prepare_exit_features(df):
    """Prepare Exit model enhanced features"""
    # Add enhanced features if not present
    if 'volume_surge' not in df.columns:
        df['volume_surge'] = (df['volume'] > df['volume'].rolling(20).mean() * 1.5).astype(float)

    if 'price_acceleration' not in df.columns:
        df['price_acceleration'] = df['close'].pct_change(5)

    if 'price_vs_ma20' not in df.columns:
        if 'sma_20' in df.columns:
            df['price_vs_ma20'] = (df['close'] - df['sma_20']) / df['sma_20']
        else:
            ma_20 = df['close'].rolling(20).mean()
            df['price_vs_ma20'] = (df['close'] - ma_20) / ma_20

    if 'price_vs_ma50' not in df.columns:
        ma_50 = df['close'].rolling(50).mean()
        df['price_vs_ma50'] = (df['close'] - ma_50) / ma_50

    if 'rsi_slope' not in df.columns:
        df['rsi_slope'] = df['rsi'].diff(3) / 3

    if 'rsi_overbought' not in df.columns:
        df['rsi_overbought'] = (df['rsi'] > 70).astype(float)

    if 'rsi_oversold' not in df.columns:
        df['rsi_oversold'] = (df['rsi'] < 30).astype(float)

    if 'rsi_divergence' not in df.columns:
        df['rsi_divergence'] = 0

    if 'macd_histogram_slope' not in df.columns:
        if 'macd_diff' in df.columns:
            df['macd_histogram_slope'] = df['macd_diff'].diff(3) / 3
        else:
            df['macd_histogram_slope'] = 0

    if 'macd_crossover' not in df.columns:
        df['macd_crossover'] = ((df['macd'] > df['macd_signal']) &
                                (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(float)

    if 'macd_crossunder' not in df.columns:
        df['macd_crossunder'] = ((df['macd'] < df['macd_signal']) &
                                 (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(float)

    if 'bb_position' not in df.columns:
        if 'bb_high' in df.columns and 'bb_low' in df.columns:
            df['bb_position'] = (df['close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'])
        else:
            df['bb_position'] = 0.5

    if 'higher_high' not in df.columns:
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(float)

    if 'near_support' not in df.columns:
        df['near_support'] = 0

    # Clean NaN
    df = df.ffill().bfill()

    return df


def calculate_position_size(signal_prob, min_size=0.20, max_size=0.95):
    """Simple position sizing based on signal strength"""
    # Linear scaling from min to max based on probability
    # 0.75 threshold → 20% position
    # 1.00 probability → 95% position
    if signal_prob < 0.75:
        return min_size

    scale = (signal_prob - 0.75) / (1.0 - 0.75)
    position_pct = min_size + (max_size - min_size) * scale
    return min(max(position_pct, min_size), max_size)


def backtest(df, models):
    """Run backtest with given models"""
    # Prepare exit features
    df = prepare_exit_features(df)

    # State
    balance = INITIAL_CAPITAL
    position = None
    trades = []

    for i in range(len(df)):
        row = df.iloc[i]

        # === ENTRY LOGIC ===
        if position is None:
            # Get LONG signal
            long_model, long_scaler, long_features = models['long_entry']
            X_long = row[long_features].values.reshape(1, -1)
            X_long_scaled = long_scaler.transform(X_long)
            long_prob = long_model.predict_proba(X_long_scaled)[0][1]

            # Get SHORT signal
            short_model, short_scaler, short_features = models['short_entry']
            X_short = row[short_features].values.reshape(1, -1)
            X_short_scaled = short_scaler.transform(X_short)
            short_prob = short_model.predict_proba(X_short_scaled)[0][1]

            # Entry decision
            if long_prob >= LONG_THRESHOLD:
                # Calculate position size
                position_pct = calculate_position_size(long_prob)

                position_value = balance * position_pct
                quantity = (position_value * LEVERAGE) / row['close']

                position = {
                    'side': 'LONG',
                    'entry_price': row['close'],
                    'entry_time': row['timestamp'],
                    'entry_idx': i,
                    'quantity': quantity,
                    'position_value': position_value,
                    'leveraged_value': position_value * LEVERAGE,
                    'entry_fee': position_value * LEVERAGE * TAKER_FEE,
                    'entry_prob': long_prob
                }

            elif short_prob >= SHORT_THRESHOLD:
                position_pct = calculate_position_size(short_prob)

                position_value = balance * position_pct
                quantity = (position_value * LEVERAGE) / row['close']

                position = {
                    'side': 'SHORT',
                    'entry_price': row['close'],
                    'entry_time': row['timestamp'],
                    'entry_idx': i,
                    'quantity': quantity,
                    'position_value': position_value,
                    'leveraged_value': position_value * LEVERAGE,
                    'entry_fee': position_value * LEVERAGE * TAKER_FEE,
                    'entry_prob': short_prob
                }

        # === EXIT LOGIC ===
        else:
            hold_candles = i - position['entry_idx']

            # Calculate current P&L
            if position['side'] == 'LONG':
                price_change = (row['close'] - position['entry_price']) / position['entry_price']
            else:
                price_change = (position['entry_price'] - row['close']) / position['entry_price']

            leveraged_pnl_pct = price_change * LEVERAGE
            pnl_usd = position['position_value'] * leveraged_pnl_pct

            # Get ML Exit signal
            if position['side'] == 'LONG':
                exit_model, exit_scaler, exit_features = models['long_exit']
                threshold = ML_EXIT_THRESHOLD_LONG
            else:
                exit_model, exit_scaler, exit_features = models['short_exit']
                threshold = ML_EXIT_THRESHOLD_SHORT

            X_exit = row[exit_features].values.reshape(1, -1)
            X_exit_scaled = exit_scaler.transform(X_exit)
            exit_prob = exit_model.predict_proba(X_exit_scaled)[0][1]

            # Exit conditions
            ml_exit = exit_prob >= threshold
            stop_loss = leveraged_pnl_pct <= -EMERGENCY_STOP_LOSS
            max_hold = hold_candles >= EMERGENCY_MAX_HOLD_CANDLES

            if ml_exit or stop_loss or max_hold:
                # Calculate final P&L
                exit_fee = position['leveraged_value'] * TAKER_FEE
                total_fee = position['entry_fee'] + exit_fee
                net_pnl = pnl_usd - total_fee

                balance += net_pnl

                # Record trade
                if ml_exit:
                    exit_reason = f"ML Exit ({exit_prob:.3f})"
                elif stop_loss:
                    exit_reason = "Stop Loss"
                else:
                    exit_reason = "Max Hold"

                trades.append({
                    'side': position['side'],
                    'entry_time': position['entry_time'],
                    'exit_time': row['timestamp'],
                    'entry_price': position['entry_price'],
                    'exit_price': row['close'],
                    'hold_candles': hold_candles,
                    'entry_prob': position['entry_prob'],
                    'exit_prob': exit_prob,
                    'pnl_usd': pnl_usd,
                    'total_fee': total_fee,
                    'net_pnl': net_pnl,
                    'pnl_pct': leveraged_pnl_pct,
                    'exit_reason': exit_reason
                })

                position = None

    return balance, trades


def analyze_results(balance, trades, label):
    """Analyze backtest results"""
    df_trades = pd.DataFrame(trades)

    total_return = ((balance - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100

    if len(trades) == 0:
        print(f"\n{'='*80}")
        print(f"{label}: NO TRADES")
        print(f"{'='*80}")
        return None

    wins = df_trades[df_trades['net_pnl'] > 0]
    losses = df_trades[df_trades['net_pnl'] <= 0]
    win_rate = len(wins) / len(trades) * 100 if len(trades) > 0 else 0

    avg_win = wins['net_pnl'].mean() if len(wins) > 0 else 0
    avg_loss = losses['net_pnl'].mean() if len(losses) > 0 else 0

    ml_exits = df_trades[df_trades['exit_reason'].str.contains('ML Exit')]
    ml_exit_rate = len(ml_exits) / len(trades) * 100 if len(trades) > 0 else 0

    # Calculate Sharpe (daily returns)
    daily_returns = []
    current_balance = INITIAL_CAPITAL
    for trade in trades:
        current_balance += trade['net_pnl']
        daily_return = trade['net_pnl'] / INITIAL_CAPITAL
        daily_returns.append(daily_return)

    if len(daily_returns) > 1:
        sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-10) * np.sqrt(252)
    else:
        sharpe = 0

    print(f"\n{'='*80}")
    print(f"{label}")
    print(f"{'='*80}")
    print(f"\nPerformance:")
    print(f"  Final Balance: ${balance:,.2f}")
    print(f"  Total Return: {total_return:+.2f}%")
    print(f"  Sharpe Ratio: {sharpe:.3f}")

    print(f"\nTrades:")
    print(f"  Total: {len(trades)}")
    print(f"  Wins: {len(wins)} ({win_rate:.1f}%)")
    print(f"  Losses: {len(losses)}")

    print(f"\nP&L:")
    print(f"  Avg Win: ${avg_win:.2f}")
    print(f"  Avg Loss: ${avg_loss:.2f}")
    print(f"  Total Fees: ${df_trades['total_fee'].sum():.2f}")

    print(f"\nExit Distribution:")
    print(f"  ML Exit: {len(ml_exits)} ({ml_exit_rate:.1f}%)")
    print(f"  Stop Loss: {len(df_trades[df_trades['exit_reason'] == 'Stop Loss'])}")
    print(f"  Max Hold: {len(df_trades[df_trades['exit_reason'] == 'Max Hold'])}")

    print(f"\nHold Time:")
    print(f"  Avg: {df_trades['hold_candles'].mean():.1f} candles ({df_trades['hold_candles'].mean()/12:.1f} hours)")

    return {
        'label': label,
        'final_balance': balance,
        'total_return': total_return,
        'sharpe': sharpe,
        'total_trades': len(trades),
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'ml_exit_rate': ml_exit_rate
    }


def main():
    print(f"\n{'='*80}")
    print("PHASE 2 MODEL VALIDATION (28-Day Holdout)")
    print(f"{'='*80}")

    # Load data
    df_val = load_validation_data()

    # Test Phase 2 models
    print(f"\n{'='*80}")
    print("Testing Phase 2 Models (NEW)")
    print(f"{'='*80}")
    models_phase2 = load_models('phase2')
    balance_phase2, trades_phase2 = backtest(df_val, models_phase2)
    results_phase2 = analyze_results(balance_phase2, trades_phase2, "Phase 2 Models")

    # Test Current models
    print(f"\n{'='*80}")
    print("Testing Current Production Models")
    print(f"{'='*80}")
    models_current = load_models('current')
    balance_current, trades_current = backtest(df_val, models_current)
    results_current = analyze_results(balance_current, trades_current, "Current Production")

    # Comparison
    if results_phase2 and results_current:
        print(f"\n{'='*80}")
        print("COMPARISON")
        print(f"{'='*80}")

        metrics = ['total_return', 'sharpe', 'win_rate', 'ml_exit_rate', 'total_trades']

        for metric in metrics:
            phase2_val = results_phase2[metric]
            current_val = results_current[metric]
            diff = phase2_val - current_val
            pct_change = (diff / current_val * 100) if current_val != 0 else 0

            winner = "Phase 2" if phase2_val > current_val else "Current"

            print(f"\n{metric}:")
            print(f"  Phase 2: {phase2_val:.2f}")
            print(f"  Current: {current_val:.2f}")
            print(f"  Diff: {diff:+.2f} ({pct_change:+.1f}%)")
            print(f"  Winner: {winner}")

        # Recommendation
        print(f"\n{'='*80}")
        print("RECOMMENDATION")
        print(f"{'='*80}")

        if results_phase2['total_return'] > results_current['total_return']:
            print("\n✅ DEPLOY Phase 2 Enhanced Models")
            print(f"   Return improvement: {results_phase2['total_return'] - results_current['total_return']:+.2f}%")
            print(f"   Timestamp: 20251102_200202 (Walk-Forward Decoupled + Enhanced Features)")
        else:
            print("\n⚠️ KEEP Current Models")
            print(f"   Phase 2 underperforms by: {results_current['total_return'] - results_phase2['total_return']:.2f}%")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"phase2_validation_{timestamp}.csv"

    if trades_phase2:
        df_phase2 = pd.DataFrame(trades_phase2)
        df_phase2['model_set'] = 'phase2'
        df_phase2.to_csv(results_file, index=False)
        print(f"\n✅ Results saved: {results_file}")


if __name__ == "__main__":
    main()
