"""
Backtest 30% Exit Rate Models with Threshold Sweep

Purpose: Test if 30% exit rate achieves 2-10 trades/day target

Comparison:
- Current (15% exit rate): 0.37 trades/day
- Target: 2-10 trades/day
- Expected (30% exit rate): ~1.1 trades/day

Test Thresholds: 0.60, 0.65, 0.70, 0.75
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import pickle
import joblib

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
FEATURES_DIR = BASE_DIR / "data" / "features"
RESULTS_DIR = BASE_DIR / "results"

# Timestamp
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Models (30% exit rate)
LONG_EXIT_30PCT = MODELS_DIR / "xgboost_long_exit_30pct_20251107_171927.pkl"
SHORT_EXIT_30PCT = MODELS_DIR / "xgboost_short_exit_30pct_20251107_171927.pkl"
LONG_EXIT_SCALER_30PCT = MODELS_DIR / "xgboost_long_exit_30pct_20251107_171927_scaler.pkl"
SHORT_EXIT_SCALER_30PCT = MODELS_DIR / "xgboost_short_exit_30pct_20251107_171927_scaler.pkl"
EXIT_FEATURES_30PCT = MODELS_DIR / "xgboost_long_exit_30pct_20251107_171927_features.txt"

# Entry models (90-day Trade Outcome)
LONG_ENTRY = MODELS_DIR / "xgboost_long_entry_90days_tradeoutcome_20251106_193900.pkl"
SHORT_ENTRY = MODELS_DIR / "xgboost_short_entry_90days_tradeoutcome_20251106_193900.pkl"
LONG_ENTRY_SCALER = MODELS_DIR / "xgboost_long_entry_90days_tradeoutcome_20251106_193900_scaler.pkl"
SHORT_ENTRY_SCALER = MODELS_DIR / "xgboost_short_entry_90days_tradeoutcome_20251106_193900_scaler.pkl"
ENTRY_FEATURES = MODELS_DIR / "xgboost_long_entry_90days_tradeoutcome_20251106_193900_features.txt"

# Features
FEATURES_FILE = FEATURES_DIR / "BTCUSDT_5m_features_90days_complete_20251106_164542.csv"

# Configuration
LONG_ENTRY_THRESHOLD = 0.85
SHORT_ENTRY_THRESHOLD = 0.80
EXIT_THRESHOLDS = [0.60, 0.65, 0.70, 0.75]
LEVERAGE = 4
BALANCE_STOP_LOSS = 0.03  # -3%
MAX_HOLD_CANDLES = 120  # 10 hours

# Validation period
VAL_START = "2025-10-09 00:00:00"
VAL_END = "2025-11-06 07:35:00"


def load_models():
    """Load Entry and Exit models"""
    print(f"\n{'='*80}")
    print("LOADING MODELS")
    print(f"{'='*80}")

    # Load Entry models
    with open(LONG_ENTRY, 'rb') as f:
        long_entry_model = pickle.load(f)
    with open(SHORT_ENTRY, 'rb') as f:
        short_entry_model = pickle.load(f)
    long_entry_scaler = joblib.load(LONG_ENTRY_SCALER)
    short_entry_scaler = joblib.load(SHORT_ENTRY_SCALER)

    with open(ENTRY_FEATURES, 'r') as f:
        entry_features = [line.strip() for line in f.readlines() if line.strip()]

    print(f"✅ Entry models loaded (171 features)")

    # Load Exit models (30% rate)
    with open(LONG_EXIT_30PCT, 'rb') as f:
        long_exit_model = pickle.load(f)
    with open(SHORT_EXIT_30PCT, 'rb') as f:
        short_exit_model = pickle.load(f)
    long_exit_scaler = joblib.load(LONG_EXIT_SCALER_30PCT)
    short_exit_scaler = joblib.load(SHORT_EXIT_SCALER_30PCT)

    with open(EXIT_FEATURES_30PCT, 'r') as f:
        exit_features = [line.strip() for line in f.readlines() if line.strip()]

    print(f"✅ Exit models loaded (30% rate, 171 features)")

    return {
        'long_entry': long_entry_model,
        'short_entry': short_entry_model,
        'long_entry_scaler': long_entry_scaler,
        'short_entry_scaler': short_entry_scaler,
        'long_exit': long_exit_model,
        'short_exit': short_exit_model,
        'long_exit_scaler': long_exit_scaler,
        'short_exit_scaler': short_exit_scaler,
        'entry_features': entry_features,
        'exit_features': exit_features
    }


def calculate_probabilities(df, models):
    """Calculate Entry and Exit probabilities for all candles"""
    print(f"\n{'='*80}")
    print("CALCULATING PROBABILITIES")
    print(f"{'='*80}")

    # Entry probabilities
    X_entry = df[models['entry_features']].values
    X_entry_long_scaled = models['long_entry_scaler'].transform(X_entry)
    X_entry_short_scaled = models['short_entry_scaler'].transform(X_entry)

    long_entry_proba = models['long_entry'].predict_proba(X_entry_long_scaled)[:, 1]
    short_entry_proba = models['short_entry'].predict_proba(X_entry_short_scaled)[:, 1]

    print(f"✅ Entry probabilities calculated")
    print(f"   LONG: max={long_entry_proba.max():.4f}, mean={long_entry_proba.mean():.4f}")
    print(f"   SHORT: max={short_entry_proba.max():.4f}, mean={short_entry_proba.mean():.4f}")

    # Exit probabilities
    X_exit = df[models['exit_features']].values
    X_exit_long_scaled = models['long_exit_scaler'].transform(X_exit)
    X_exit_short_scaled = models['short_exit_scaler'].transform(X_exit)

    long_exit_proba = models['long_exit'].predict_proba(X_exit_long_scaled)[:, 1]
    short_exit_proba = models['short_exit'].predict_proba(X_exit_short_scaled)[:, 1]

    print(f"✅ Exit probabilities calculated (30% rate)")
    print(f"   LONG: max={long_exit_proba.max():.4f}, mean={long_exit_proba.mean():.4f}")
    print(f"   SHORT: max={short_exit_proba.max():.4f}, mean={short_exit_proba.mean():.4f}")

    return long_entry_proba, short_entry_proba, long_exit_proba, short_exit_proba


def simulate_trading(df, long_entry_proba, short_entry_proba, long_exit_proba, short_exit_proba, exit_threshold):
    """Simulate trading with given Exit threshold"""
    balance = 300.0  # Starting balance
    position = None
    trades = []

    for i in range(len(df) - MAX_HOLD_CANDLES):
        current_price = df.iloc[i]['close']
        current_time = df.iloc[i]['timestamp']

        # Exit logic (if in position)
        if position is not None:
            hold_time = i - position['entry_idx']

            # Get exit probability
            if position['side'] == 'LONG':
                exit_prob = long_exit_proba[i]
            else:
                exit_prob = short_exit_proba[i]

            # Exit conditions
            exit_reason = None

            # 1. ML Exit
            if exit_prob >= exit_threshold:
                exit_reason = f"ML Exit ({exit_prob:.3f})"

            # 2. Balance-based Stop Loss
            if exit_reason is None:
                if position['side'] == 'LONG':
                    loss_pct = (current_price - position['entry_price']) / position['entry_price']
                    if loss_pct * LEVERAGE <= -BALANCE_STOP_LOSS:
                        exit_reason = "Stop Loss"
                else:  # SHORT
                    loss_pct = (position['entry_price'] - current_price) / position['entry_price']
                    if loss_pct * LEVERAGE <= -BALANCE_STOP_LOSS:
                        exit_reason = "Stop Loss"

            # 3. Max Hold Time
            if exit_reason is None and hold_time >= MAX_HOLD_CANDLES:
                exit_reason = "Max Hold"

            if exit_reason:
                # Calculate P&L
                if position['side'] == 'LONG':
                    pnl_pct = (current_price - position['entry_price']) / position['entry_price']
                else:
                    pnl_pct = (position['entry_price'] - current_price) / position['entry_price']

                leveraged_pnl_pct = pnl_pct * LEVERAGE
                pnl_usd = balance * leveraged_pnl_pct
                balance += pnl_usd

                trades.append({
                    'entry_time': position['entry_time'],
                    'entry_price': position['entry_price'],
                    'entry_prob': position['entry_prob'],
                    'exit_time': current_time,
                    'exit_price': current_price,
                    'exit_prob': exit_prob,
                    'side': position['side'],
                    'pnl_pct': pnl_pct,
                    'leveraged_pnl_pct': leveraged_pnl_pct,
                    'pnl_usd': pnl_usd,
                    'balance': balance,
                    'hold_candles': hold_time,
                    'exit_reason': exit_reason
                })

                position = None

        # Entry logic (if no position)
        if position is None:
            long_prob = long_entry_proba[i]
            short_prob = short_entry_proba[i]

            # LONG Entry
            if long_prob >= LONG_ENTRY_THRESHOLD and long_prob > short_prob + 0.001:
                position = {
                    'side': 'LONG',
                    'entry_idx': i,
                    'entry_time': current_time,
                    'entry_price': current_price,
                    'entry_prob': long_prob
                }

            # SHORT Entry
            elif short_prob >= SHORT_ENTRY_THRESHOLD and short_prob > long_prob + 0.001:
                position = {
                    'side': 'SHORT',
                    'entry_idx': i,
                    'entry_time': current_time,
                    'entry_price': current_price,
                    'entry_prob': short_prob
                }

    return trades, balance


def analyze_results(trades, balance, exit_threshold, validation_days):
    """Analyze trading results"""
    if len(trades) == 0:
        return {
            'exit_threshold': exit_threshold,
            'trades': 0,
            'trades_per_day': 0,
            'return_pct': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'avg_hold_candles': 0
        }

    df_trades = pd.DataFrame(trades)

    total_return = ((balance - 300.0) / 300.0) * 100
    wins = df_trades[df_trades['pnl_usd'] > 0]
    losses = df_trades[df_trades['pnl_usd'] <= 0]
    win_rate = len(wins) / len(trades) * 100

    total_wins = wins['pnl_usd'].sum() if len(wins) > 0 else 0
    total_losses = abs(losses['pnl_usd'].sum()) if len(losses) > 0 else 0
    profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

    trades_per_day = len(trades) / validation_days

    return {
        'exit_threshold': exit_threshold,
        'trades': len(trades),
        'trades_per_day': trades_per_day,
        'return_pct': total_return,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_hold_candles': df_trades['hold_candles'].mean(),
        'long_trades': len(df_trades[df_trades['side'] == 'LONG']),
        'short_trades': len(df_trades[df_trades['side'] == 'SHORT']),
        'ml_exit_pct': (df_trades['exit_reason'].str.contains('ML Exit').sum() / len(trades)) * 100,
        'stop_loss_pct': (df_trades['exit_reason'].str.contains('Stop Loss').sum() / len(trades)) * 100,
        'max_hold_pct': (df_trades['exit_reason'].str.contains('Max Hold').sum() / len(trades)) * 100
    }


def main():
    print("="*80)
    print("BACKTEST: 30% EXIT RATE MODELS - THRESHOLD SWEEP")
    print("="*80)

    print(f"\n📊 Configuration:")
    print(f"   Entry: LONG {LONG_ENTRY_THRESHOLD}, SHORT {SHORT_ENTRY_THRESHOLD}")
    print(f"   Exit Thresholds: {EXIT_THRESHOLDS}")
    print(f"   Target: 2-10 trades/day")
    print(f"   Current (15% exit rate): 0.37 trades/day")
    print(f"   Expected (30% exit rate): ~1.1 trades/day")

    # Load data
    print(f"\n{'='*80}")
    print("LOADING DATA")
    print(f"{'='*80}")

    df = pd.read_csv(FEATURES_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Filter validation period
    df = df[(df['timestamp'] >= VAL_START) & (df['timestamp'] <= VAL_END)].copy()
    df = df.reset_index(drop=True)

    validation_days = (df['timestamp'].max() - df['timestamp'].min()).days

    print(f"\n✅ Validation period loaded:")
    print(f"   Period: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"   Days: {validation_days}")
    print(f"   Candles: {len(df):,}")

    # Load models
    models = load_models()

    # Calculate probabilities
    long_entry_proba, short_entry_proba, long_exit_proba, short_exit_proba = calculate_probabilities(df, models)

    # Test all Exit thresholds
    print(f"\n{'='*80}")
    print("TESTING EXIT THRESHOLDS")
    print(f"{'='*80}")

    all_results = []

    for exit_threshold in EXIT_THRESHOLDS:
        print(f"\n{'='*80}")
        print(f"EXIT THRESHOLD: {exit_threshold:.2f}")
        print(f"{'='*80}")

        trades, final_balance = simulate_trading(
            df, long_entry_proba, short_entry_proba,
            long_exit_proba, short_exit_proba, exit_threshold
        )

        results = analyze_results(trades, final_balance, exit_threshold, validation_days)
        all_results.append(results)

        print(f"\n📊 Results:")
        print(f"   Trades: {results['trades']} ({results['trades_per_day']:.2f}/day)")
        print(f"   Return: {results['return_pct']:+.2f}%")
        print(f"   Win Rate: {results['win_rate']:.2f}%")
        print(f"   Profit Factor: {results['profit_factor']:.2f}×")
        print(f"   Avg Hold: {results['avg_hold_candles']:.1f} candles")
        print(f"   LONG/SHORT: {results['long_trades']}/{results['short_trades']}")
        print(f"   Exit Breakdown:")
        print(f"     ML Exit: {results['ml_exit_pct']:.1f}%")
        print(f"     Stop Loss: {results['stop_loss_pct']:.1f}%")
        print(f"     Max Hold: {results['max_hold_pct']:.1f}%")

        # Target check
        if 2.0 <= results['trades_per_day'] <= 10.0:
            print(f"   ✅ MEETS TARGET (2-10 trades/day)")
        else:
            print(f"   ❌ MISSES TARGET (current: {results['trades_per_day']:.2f}, need: 2-10)")

    # Summary
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")

    df_results = pd.DataFrame(all_results)

    print(f"\n📊 All Results:")
    print(df_results[['exit_threshold', 'trades', 'trades_per_day', 'return_pct', 'win_rate', 'profit_factor']].to_string(index=False))

    # Best by trade frequency
    best_freq = df_results.loc[df_results['trades_per_day'].idxmax()]
    print(f"\n🏆 Highest Trade Frequency:")
    print(f"   Threshold: {best_freq['exit_threshold']:.2f}")
    print(f"   Trades/Day: {best_freq['trades_per_day']:.2f}")
    print(f"   Return: {best_freq['return_pct']:+.2f}%")

    # Best by return
    best_return = df_results.loc[df_results['return_pct'].idxmax()]
    print(f"\n💰 Highest Return:")
    print(f"   Threshold: {best_return['exit_threshold']:.2f}")
    print(f"   Return: {best_return['return_pct']:+.2f}%")
    print(f"   Trades/Day: {best_return['trades_per_day']:.2f}")

    # Target achievement
    meets_target = df_results[(df_results['trades_per_day'] >= 2.0) & (df_results['trades_per_day'] <= 10.0)]

    print(f"\n🎯 Target Achievement (2-10 trades/day):")
    if len(meets_target) > 0:
        print(f"   ✅ {len(meets_target)} threshold(s) meet target:")
        for _, row in meets_target.iterrows():
            print(f"      Exit {row['exit_threshold']:.2f}: {row['trades_per_day']:.2f}/day, {row['return_pct']:+.2f}%")
    else:
        print(f"   ❌ NO thresholds meet target")
        print(f"   Best: {best_freq['trades_per_day']:.2f}/day at {best_freq['exit_threshold']:.2f} threshold")
        print(f"   Gap: {(2.0 / best_freq['trades_per_day']):.1f}× below target")

    # Save results
    output_file = RESULTS_DIR / f"backtest_30pct_exit_sweep_{TIMESTAMP}.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\n💾 Results saved: {output_file.name}")

    # Recommendation
    print(f"\n{'='*80}")
    print("RECOMMENDATION")
    print(f"{'='*80}")

    if len(meets_target) > 0:
        best_target = meets_target.loc[meets_target['return_pct'].idxmax()]
        print(f"\n✅ DEPLOY Exit {best_target['exit_threshold']:.2f} (30% rate models)")
        print(f"   Expected: {best_target['trades_per_day']:.2f} trades/day, {best_target['return_pct']:+.2f}% return")
    else:
        print(f"\n⚠️ 30% Exit Rate INSUFFICIENT")
        print(f"   Current: {best_freq['trades_per_day']:.2f}/day (need {2.0/best_freq['trades_per_day']:.1f}× more)")
        print(f"\n🔄 Next Steps:")
        print(f"   1. Consider Entry threshold adjustment (0.85/0.80 → lower)")
        print(f"   2. OR retrain Entry models with relaxed labeling")
        print(f"   3. Trade-off: Frequency ↑ but Quality ↓")


if __name__ == "__main__":
    main()
