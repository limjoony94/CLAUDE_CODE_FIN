"""
Exit Threshold Sweep Analysis - Find Optimal Trade Frequency

Purpose:
- Test Exit thresholds: 0.60, 0.65, 0.70, 0.75, 0.80, 0.85
- Entry fixed: LONG 0.85, SHORT 0.80
- Target: 2-10 trades/day (54-270 trades in 27 days)
- Current: 0.37 trades/day (10 trades in 27 days)

Models:
- Entry: 90-day Trade Outcome (171 features)
- Exit: Optimal Triple Barrier (171 features, 15% exit rate)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import pickle
import joblib
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.production.production_features_v1 import calculate_all_features_enhanced_v2

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "features"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
FEATURES_FILE = DATA_DIR / "BTCUSDT_5m_features_90days_complete_20251106_164542.csv"
START_DATE = "2025-09-29 00:00:00"
END_DATE = "2025-10-26 00:00:00"
STARTING_BALANCE = 300.0
LEVERAGE = 4

# Entry Models (90-day Trade Outcome)
LONG_ENTRY_MODEL = MODELS_DIR / "xgboost_long_entry_90days_tradeoutcome_20251106_193900.pkl"
LONG_ENTRY_SCALER = MODELS_DIR / "xgboost_long_entry_90days_tradeoutcome_20251106_193900_scaler.pkl"
LONG_ENTRY_FEATURES = MODELS_DIR / "xgboost_long_entry_90days_tradeoutcome_20251106_193900_features.txt"

SHORT_ENTRY_MODEL = MODELS_DIR / "xgboost_short_entry_90days_tradeoutcome_20251106_193900.pkl"
SHORT_ENTRY_SCALER = MODELS_DIR / "xgboost_short_entry_90days_tradeoutcome_20251106_193900_scaler.pkl"
SHORT_ENTRY_FEATURES = MODELS_DIR / "xgboost_short_entry_90days_tradeoutcome_20251106_193900_features.txt"

# Exit Models (Optimal Triple Barrier)
LONG_EXIT_MODEL = MODELS_DIR / "xgboost_long_exit_optimal_20251106_223613.pkl"
LONG_EXIT_SCALER = MODELS_DIR / "xgboost_long_exit_optimal_20251106_223613_scaler.pkl"
LONG_EXIT_FEATURES = MODELS_DIR / "xgboost_long_exit_optimal_20251106_223613_features.txt"

SHORT_EXIT_MODEL = MODELS_DIR / "xgboost_short_exit_optimal_20251106_223613.pkl"
SHORT_EXIT_SCALER = MODELS_DIR / "xgboost_short_exit_optimal_20251106_223613_scaler.pkl"
SHORT_EXIT_FEATURES = MODELS_DIR / "xgboost_short_exit_optimal_20251106_223613_features.txt"

# Entry Thresholds (Fixed)
LONG_ENTRY_THRESHOLD = 0.85
SHORT_ENTRY_THRESHOLD = 0.80

# Exit Thresholds to Test
EXIT_THRESHOLDS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]

# Risk Management
STOP_LOSS_PCT = 0.03  # 3% of balance
MAX_HOLD_CANDLES = 120  # 10 hours


def load_models():
    """Load all models and scalers"""
    print("\n" + "="*80)
    print("LOADING MODELS")
    print("="*80)

    # Load Entry models
    print("\n📦 Loading Entry models...")
    with open(LONG_ENTRY_MODEL, 'rb') as f:
        long_entry_model = pickle.load(f)
    long_entry_scaler = joblib.load(LONG_ENTRY_SCALER)
    with open(LONG_ENTRY_FEATURES, 'r') as f:
        long_entry_features = [line.strip() for line in f.readlines()]
    print(f"   ✅ LONG Entry: {len(long_entry_features)} features")

    with open(SHORT_ENTRY_MODEL, 'rb') as f:
        short_entry_model = pickle.load(f)
    short_entry_scaler = joblib.load(SHORT_ENTRY_SCALER)
    with open(SHORT_ENTRY_FEATURES, 'r') as f:
        short_entry_features = [line.strip() for line in f.readlines()]
    print(f"   ✅ SHORT Entry: {len(short_entry_features)} features")

    # Load Exit models
    print("\n📦 Loading Exit models (Optimal Triple Barrier)...")
    with open(LONG_EXIT_MODEL, 'rb') as f:
        long_exit_model = pickle.load(f)
    long_exit_scaler = joblib.load(LONG_EXIT_SCALER)
    with open(LONG_EXIT_FEATURES, 'r') as f:
        long_exit_features = [line.strip() for line in f.readlines()]
    print(f"   ✅ LONG Exit: {len(long_exit_features)} features")

    with open(SHORT_EXIT_MODEL, 'rb') as f:
        short_exit_model = pickle.load(f)
    short_exit_scaler = joblib.load(SHORT_EXIT_SCALER)
    with open(SHORT_EXIT_FEATURES, 'r') as f:
        short_exit_features = [line.strip() for line in f.readlines()]
    print(f"   ✅ SHORT Exit: {len(short_exit_features)} features")

    return {
        'long_entry': {'model': long_entry_model, 'scaler': long_entry_scaler, 'features': long_entry_features},
        'short_entry': {'model': short_entry_model, 'scaler': short_entry_scaler, 'features': short_entry_features},
        'long_exit': {'model': long_exit_model, 'scaler': long_exit_scaler, 'features': long_exit_features},
        'short_exit': {'model': short_exit_model, 'scaler': short_exit_scaler, 'features': short_exit_features}
    }


def calculate_probabilities(df, models):
    """Calculate all probabilities"""
    print("\n" + "="*80)
    print("CALCULATING PROBABILITIES")
    print("="*80)

    # Long Entry
    X_long_entry = df[models['long_entry']['features']]
    X_long_entry_scaled = models['long_entry']['scaler'].transform(X_long_entry)
    long_entry_prob = models['long_entry']['model'].predict_proba(X_long_entry_scaled)[:, 1]
    df['long_entry_prob'] = long_entry_prob
    print(f"\n✅ LONG Entry: max {long_entry_prob.max():.4f}")

    # Short Entry
    X_short_entry = df[models['short_entry']['features']]
    X_short_entry_scaled = models['short_entry']['scaler'].transform(X_short_entry)
    short_entry_prob = models['short_entry']['model'].predict_proba(X_short_entry_scaled)[:, 1]
    df['short_entry_prob'] = short_entry_prob
    print(f"✅ SHORT Entry: max {short_entry_prob.max():.4f}")

    # Long Exit
    X_long_exit = df[models['long_exit']['features']]
    X_long_exit_scaled = models['long_exit']['scaler'].transform(X_long_exit)
    long_exit_prob = models['long_exit']['model'].predict_proba(X_long_exit_scaled)[:, 1]
    df['long_exit_prob'] = long_exit_prob
    print(f"✅ LONG Exit: max {long_exit_prob.max():.4f}")

    # Short Exit
    X_short_exit = df[models['short_exit']['features']]
    X_short_exit_scaled = models['short_exit']['scaler'].transform(X_short_exit)
    short_exit_prob = models['short_exit']['model'].predict_proba(X_short_exit_scaled)[:, 1]
    df['short_exit_prob'] = short_exit_prob
    print(f"✅ SHORT Exit: max {short_exit_prob.max():.4f}")

    return df


def run_backtest(df, exit_threshold):
    """Run backtest with specific exit threshold"""
    balance = STARTING_BALANCE
    position = None
    trades = []

    for i in range(len(df)):
        current_candle = df.iloc[i]

        # Position management
        if position is not None:
            position['hold_candles'] += 1
            current_price = current_candle['close']

            # Calculate unrealized P&L
            if position['side'] == 'LONG':
                price_change_pct = (current_price - position['entry_price']) / position['entry_price']
            else:  # SHORT
                price_change_pct = (position['entry_price'] - current_price) / position['entry_price']

            leveraged_pnl_pct = price_change_pct * LEVERAGE
            unrealized_pnl = position['position_value'] * leveraged_pnl_pct

            # Check exit conditions
            exit_signal = False
            exit_reason = None

            # 1. Stop Loss
            if leveraged_pnl_pct <= -STOP_LOSS_PCT:
                exit_signal = True
                exit_reason = "Stop Loss"

            # 2. Max Hold
            elif position['hold_candles'] >= MAX_HOLD_CANDLES:
                exit_signal = True
                exit_reason = "Max Hold"

            # 3. ML Exit
            else:
                exit_prob = current_candle[f"{position['side'].lower()}_exit_prob"]
                if exit_prob >= exit_threshold:
                    exit_signal = True
                    exit_reason = f"ML Exit ({exit_prob:.3f})"

            # Execute exit
            if exit_signal:
                balance += unrealized_pnl

                trade = {
                    'side': position['side'],
                    'entry_time': position['entry_time'],
                    'entry_price': position['entry_price'],
                    'exit_time': current_candle['timestamp'],
                    'exit_price': current_price,
                    'pnl_usd': unrealized_pnl,
                    'pnl_pct': leveraged_pnl_pct,
                    'hold_candles': position['hold_candles'],
                    'exit_reason': exit_reason
                }
                trades.append(trade)
                position = None

        # Entry logic (no position)
        if position is None:
            long_prob = current_candle['long_entry_prob']
            short_prob = current_candle['short_entry_prob']

            # Opportunity gating
            long_signal = long_prob >= LONG_ENTRY_THRESHOLD
            short_signal = short_prob >= SHORT_ENTRY_THRESHOLD

            if long_signal or short_signal:
                # Select side
                if long_signal and not short_signal:
                    side = 'LONG'
                elif short_signal and not long_signal:
                    side = 'SHORT'
                else:  # Both signals
                    gate_threshold = 0.001
                    if short_prob > long_prob + gate_threshold:
                        side = 'SHORT'
                    else:
                        side = 'LONG'

                # Open position
                entry_price = current_candle['close']
                position_value = balance * 0.95  # 95% of balance

                position = {
                    'side': side,
                    'entry_time': current_candle['timestamp'],
                    'entry_price': entry_price,
                    'position_value': position_value,
                    'hold_candles': 0
                }

    # Close any remaining position
    if position is not None:
        current_candle = df.iloc[-1]
        current_price = current_candle['close']

        if position['side'] == 'LONG':
            price_change_pct = (current_price - position['entry_price']) / position['entry_price']
        else:
            price_change_pct = (position['entry_price'] - current_price) / position['entry_price']

        leveraged_pnl_pct = price_change_pct * LEVERAGE
        unrealized_pnl = position['position_value'] * leveraged_pnl_pct
        balance += unrealized_pnl

        trade = {
            'side': position['side'],
            'entry_time': position['entry_time'],
            'entry_price': position['entry_price'],
            'exit_time': current_candle['timestamp'],
            'exit_price': current_price,
            'pnl_usd': unrealized_pnl,
            'pnl_pct': leveraged_pnl_pct,
            'hold_candles': position['hold_candles'],
            'exit_reason': 'Backtest End'
        }
        trades.append(trade)

    # Calculate statistics
    df_trades = pd.DataFrame(trades)

    if len(trades) == 0:
        return {
            'exit_threshold': exit_threshold,
            'total_trades': 0,
            'trades_per_day': 0,
            'final_balance': balance,
            'total_return': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'avg_hold': 0,
            'long_trades': 0,
            'short_trades': 0,
            'ml_exit_pct': 0,
            'stop_loss_pct': 0,
            'max_hold_pct': 0
        }

    wins = len(df_trades[df_trades['pnl_usd'] > 0])
    losses = len(df_trades[df_trades['pnl_usd'] <= 0])
    win_rate = wins / len(trades) if len(trades) > 0 else 0

    gross_profit = df_trades[df_trades['pnl_usd'] > 0]['pnl_usd'].sum() if wins > 0 else 0
    gross_loss = abs(df_trades[df_trades['pnl_usd'] <= 0]['pnl_usd'].sum()) if losses > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    days = (pd.to_datetime(END_DATE) - pd.to_datetime(START_DATE)).days
    trades_per_day = len(trades) / days

    # Exit mechanism breakdown
    ml_exits = len(df_trades[df_trades['exit_reason'].str.contains('ML Exit')])
    stop_losses = len(df_trades[df_trades['exit_reason'] == 'Stop Loss'])
    max_holds = len(df_trades[df_trades['exit_reason'] == 'Max Hold'])

    return {
        'exit_threshold': exit_threshold,
        'total_trades': len(trades),
        'trades_per_day': trades_per_day,
        'final_balance': balance,
        'total_return': (balance - STARTING_BALANCE) / STARTING_BALANCE * 100,
        'win_rate': win_rate * 100,
        'profit_factor': profit_factor,
        'avg_hold': df_trades['hold_candles'].mean(),
        'long_trades': len(df_trades[df_trades['side'] == 'LONG']),
        'short_trades': len(df_trades[df_trades['side'] == 'SHORT']),
        'ml_exit_pct': ml_exits / len(trades) * 100,
        'stop_loss_pct': stop_losses / len(trades) * 100,
        'max_hold_pct': max_holds / len(trades) * 100
    }


def main():
    print("="*80)
    print("EXIT THRESHOLD SWEEP ANALYSIS")
    print("="*80)

    print(f"\n📊 Configuration:")
    print(f"   Period: {START_DATE} to {END_DATE}")
    print(f"   Entry Thresholds: LONG {LONG_ENTRY_THRESHOLD}, SHORT {SHORT_ENTRY_THRESHOLD}")
    print(f"   Exit Thresholds: {EXIT_THRESHOLDS}")
    print(f"   Target: 2-10 trades/day")

    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)

    df = pd.read_csv(FEATURES_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Filter date range
    df = df[(df['timestamp'] >= START_DATE) & (df['timestamp'] <= END_DATE)].copy()
    print(f"\n✅ Loaded {len(df):,} candles")
    print(f"   Period: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"   Days: {(df['timestamp'].max() - df['timestamp'].min()).days}")

    # Load models
    models = load_models()

    # Calculate probabilities
    df = calculate_probabilities(df, models)

    # Run backtests
    print("\n" + "="*80)
    print("RUNNING BACKTESTS")
    print("="*80)

    results = []
    for threshold in EXIT_THRESHOLDS:
        print(f"\n🔄 Testing Exit threshold: {threshold:.2f}")
        result = run_backtest(df.copy(), threshold)
        results.append(result)

        print(f"   Trades: {result['total_trades']} ({result['trades_per_day']:.2f}/day)")
        print(f"   Return: {result['total_return']:+.2f}%")
        print(f"   Win Rate: {result['win_rate']:.1f}%")

    # Analyze results
    print("\n" + "="*80)
    print("RESULTS COMPARISON")
    print("="*80)

    df_results = pd.DataFrame(results)

    print("\n📊 Full Comparison:")
    print(df_results.to_string(index=False))

    # Find best for trade frequency
    print("\n" + "="*80)
    print("TRADE FREQUENCY ANALYSIS")
    print("="*80)

    target_min = 2.0
    target_max = 10.0

    in_range = df_results[(df_results['trades_per_day'] >= target_min) &
                          (df_results['trades_per_day'] <= target_max)]

    if len(in_range) > 0:
        print(f"\n✅ {len(in_range)} configurations meet target (2-10 trades/day):")
        print(in_range.to_string(index=False))

        # Best by return
        best_return = in_range.loc[in_range['total_return'].idxmax()]
        print(f"\n🏆 BEST (by return):")
        print(f"   Exit Threshold: {best_return['exit_threshold']:.2f}")
        print(f"   Trades/Day: {best_return['trades_per_day']:.2f}")
        print(f"   Total Return: {best_return['total_return']:+.2f}%")
        print(f"   Win Rate: {best_return['win_rate']:.1f}%")
        print(f"   Profit Factor: {best_return['profit_factor']:.2f}×")
    else:
        print(f"\n⚠️ NO configurations meet target (2-10 trades/day)")
        print(f"\n   Closest:")
        closest = df_results.iloc[(df_results['trades_per_day'] - target_min).abs().argsort()[:3]]
        print(closest.to_string(index=False))

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = RESULTS_DIR / f"exit_threshold_sweep_{timestamp}.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\n💾 Saved: {output_file.name}")


if __name__ == "__main__":
    main()
