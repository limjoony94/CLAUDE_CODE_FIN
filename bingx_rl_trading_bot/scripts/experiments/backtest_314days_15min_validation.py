"""
Backtest 314-Day 15-Min Models on 28-Day Validation Period
===========================================================

Purpose: Validate 314-day 15-min models on 100% out-of-sample period

Models:
- 314-day 15-min (trained Dec 29, 2024 - Oct 9, 2025)
- Timestamp: 20251106_155610

Backtest Period: Oct 9 - Nov 6, 2025 (28 days, 100% unseen)

Configuration:
- Entry: 0.85/0.80 (LONG/SHORT)
- Exit: 0.75
- Stop Loss: -3% balance
- Max Hold: 120 candles (30 hours @ 15-min)
- Leverage: 4x
- Position Size: 95%

Date: 2025-11-06
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from datetime import datetime

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "features"
LABELS_DIR = BASE_DIR / "data" / "labels"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Model timestamp
MODEL_TIMESTAMP = "20251106_155610"

# Production configuration
LONG_THRESHOLD = 0.85
SHORT_THRESHOLD = 0.80
GATE_THRESHOLD = 0.001
EXIT_THRESHOLD = 0.75
LEVERAGE = 4
STOP_LOSS_PCT = 0.03
MAX_HOLD_CANDLES = 120  # 30 hours @ 15-min (vs 10 hours @ 5-min)
POSITION_SIZE_PCT = 0.95
INITIAL_BALANCE = 10000.0

# Validation period (28 days, 100% out-of-sample)
VALIDATION_START = "2025-10-09 00:45:00"
VALIDATION_END = "2025-11-06 05:30:00"


def load_models():
    """Load 314-day 15-min models"""
    print(f"\n{'=' * 80}")
    print(f"LOADING 314-DAY 15-MIN MODELS")
    print('=' * 80)

    # Load LONG Entry
    long_entry = joblib.load(MODELS_DIR / f"xgboost_long_entry_314days_15min_{MODEL_TIMESTAMP}.pkl")
    long_entry_scaler = joblib.load(MODELS_DIR / f"xgboost_long_entry_314days_15min_{MODEL_TIMESTAMP}_scaler.pkl")
    with open(MODELS_DIR / f"xgboost_long_entry_314days_15min_{MODEL_TIMESTAMP}_features.txt", 'r') as f:
        long_entry_features = [line.strip() for line in f.readlines()]

    # Load SHORT Entry
    short_entry = joblib.load(MODELS_DIR / f"xgboost_short_entry_314days_15min_{MODEL_TIMESTAMP}.pkl")
    short_entry_scaler = joblib.load(MODELS_DIR / f"xgboost_short_entry_314days_15min_{MODEL_TIMESTAMP}_scaler.pkl")
    with open(MODELS_DIR / f"xgboost_short_entry_314days_15min_{MODEL_TIMESTAMP}_features.txt", 'r') as f:
        short_entry_features = [line.strip() for line in f.readlines()]

    # Load LONG Exit
    long_exit = joblib.load(MODELS_DIR / f"xgboost_long_exit_314days_15min_{MODEL_TIMESTAMP}.pkl")
    long_exit_scaler = joblib.load(MODELS_DIR / f"xgboost_long_exit_314days_15min_{MODEL_TIMESTAMP}_scaler.pkl")
    with open(MODELS_DIR / f"xgboost_long_exit_314days_15min_{MODEL_TIMESTAMP}_features.txt", 'r') as f:
        long_exit_features = [line.strip() for line in f.readlines()]

    # Load SHORT Exit
    short_exit = joblib.load(MODELS_DIR / f"xgboost_short_exit_314days_15min_{MODEL_TIMESTAMP}.pkl")
    short_exit_scaler = joblib.load(MODELS_DIR / f"xgboost_short_exit_314days_15min_{MODEL_TIMESTAMP}_scaler.pkl")
    with open(MODELS_DIR / f"xgboost_short_exit_314days_15min_{MODEL_TIMESTAMP}_features.txt", 'r') as f:
        short_exit_features = [line.strip() for line in f.readlines()]

    print(f"✅ Models Loaded:")
    print(f"   LONG Entry: {len(long_entry_features)} features")
    print(f"   SHORT Entry: {len(short_entry_features)} features")
    print(f"   LONG Exit: {len(long_exit_features)} features")
    print(f"   SHORT Exit: {len(short_exit_features)} features")

    return {
        'long_entry': (long_entry, long_entry_scaler, long_entry_features),
        'short_entry': (short_entry, short_entry_scaler, short_entry_features),
        'long_exit': (long_exit, long_exit_scaler, long_exit_features),
        'short_exit': (short_exit, short_exit_scaler, short_exit_features)
    }


def load_validation_data():
    """Load validation period data"""
    print(f"\n{'=' * 80}")
    print("LOADING VALIDATION DATA")
    print('=' * 80)

    # Load features
    df = pd.read_csv(DATA_DIR / "BTCUSDT_15m_features_314days_20251106_152653.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Filter to validation period
    val_start = pd.to_datetime(VALIDATION_START)
    val_end = pd.to_datetime(VALIDATION_END)

    df_val = df[(df['timestamp'] >= val_start) & (df['timestamp'] <= val_end)].copy()
    df_val = df_val.reset_index(drop=True)

    print(f"\n✅ Validation Period (100% Out-of-Sample):")
    print(f"   Period: {df_val['timestamp'].min()} to {df_val['timestamp'].max()}")
    print(f"   Rows: {len(df_val):,}")
    print(f"   Days: {(df_val['timestamp'].max() - df_val['timestamp'].min()).days}")
    print(f"   Price Range: ${df_val['close'].min():,.2f} - ${df_val['close'].max():,.2f}")

    return df_val


def backtest(df, models):
    """Run backtest"""
    print(f"\n{'=' * 80}")
    print(f"RUNNING BACKTEST")
    print('=' * 80)

    balance = INITIAL_BALANCE
    position = None
    trades = []

    # Get models
    long_entry_model, long_entry_scaler, long_entry_features = models['long_entry']
    short_entry_model, short_entry_scaler, short_entry_features = models['short_entry']
    long_exit_model, long_exit_scaler, long_exit_features = models['long_exit']
    short_exit_model, short_exit_scaler, short_exit_features = models['short_exit']

    # Predict Entry probabilities
    X_long_entry = df[long_entry_features].fillna(0).replace([np.inf, -np.inf], 0).values
    X_short_entry = df[short_entry_features].fillna(0).replace([np.inf, -np.inf], 0).values

    long_entry_probs = long_entry_model.predict_proba(long_entry_scaler.transform(X_long_entry))[:, 1]
    short_entry_probs = short_entry_model.predict_proba(short_entry_scaler.transform(X_short_entry))[:, 1]

    # Predict Exit probabilities
    X_long_exit = df[long_exit_features].fillna(0).replace([np.inf, -np.inf], 0).values
    X_short_exit = df[short_exit_features].fillna(0).replace([np.inf, -np.inf], 0).values

    long_exit_probs = long_exit_model.predict_proba(long_exit_scaler.transform(X_long_exit))[:, 1]
    short_exit_probs = short_exit_model.predict_proba(short_exit_scaler.transform(X_short_exit))[:, 1]

    df['long_entry_prob'] = long_entry_probs
    df['short_entry_prob'] = short_entry_probs
    df['long_exit_prob'] = long_exit_probs
    df['short_exit_prob'] = short_exit_probs

    print(f"\nEntry Signal Coverage:")
    print(f"  LONG >= {LONG_THRESHOLD}: {(long_entry_probs >= LONG_THRESHOLD).sum()} ({(long_entry_probs >= LONG_THRESHOLD).mean() * 100:.2f}%)")
    print(f"  SHORT >= {SHORT_THRESHOLD}: {(short_entry_probs >= SHORT_THRESHOLD).sum()} ({(short_entry_probs >= SHORT_THRESHOLD).mean() * 100:.2f}%)")

    # Backtest loop
    for i in range(len(df)):
        row = df.iloc[i]
        current_price = row['close']
        current_time = row['timestamp']

        # Check existing position
        if position:
            hold_candles = i - position['entry_idx']

            # Calculate P&L
            if position['side'] == 'LONG':
                exit_prob = row['long_exit_prob']
                pnl_pct = (current_price - position['entry_price']) / position['entry_price']
            else:
                exit_prob = row['short_exit_prob']
                pnl_pct = (position['entry_price'] - current_price) / position['entry_price']

            leveraged_pnl_pct = pnl_pct * LEVERAGE
            balance_pnl_pct = leveraged_pnl_pct * POSITION_SIZE_PCT

            # ML Exit
            if exit_prob >= EXIT_THRESHOLD:
                pnl = balance * balance_pnl_pct
                balance += pnl
                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': current_time,
                    'side': position['side'],
                    'entry_price': position['entry_price'],
                    'exit_price': current_price,
                    'entry_prob': position['entry_prob'],
                    'exit_prob': exit_prob,
                    'hold_candles': hold_candles,
                    'pnl': pnl,
                    'balance': balance,
                    'exit_reason': 'ML Exit'
                })
                position = None
                continue

            # Stop Loss
            if balance_pnl_pct <= -STOP_LOSS_PCT:
                pnl = balance * balance_pnl_pct
                balance += pnl
                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': current_time,
                    'side': position['side'],
                    'entry_price': position['entry_price'],
                    'exit_price': current_price,
                    'entry_prob': position['entry_prob'],
                    'exit_prob': exit_prob,
                    'hold_candles': hold_candles,
                    'pnl': pnl,
                    'balance': balance,
                    'exit_reason': 'Stop Loss'
                })
                position = None
                continue

            # Max Hold
            if hold_candles >= MAX_HOLD_CANDLES:
                pnl = balance * balance_pnl_pct
                balance += pnl
                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': current_time,
                    'side': position['side'],
                    'entry_price': position['entry_price'],
                    'exit_price': current_price,
                    'entry_prob': position['entry_prob'],
                    'exit_prob': exit_prob,
                    'hold_candles': hold_candles,
                    'pnl': pnl,
                    'balance': balance,
                    'exit_reason': 'Max Hold'
                })
                position = None
                continue

        # No position - check entry
        if not position:
            long_prob = row['long_entry_prob']
            short_prob = row['short_entry_prob']

            should_long = long_prob >= LONG_THRESHOLD
            should_short = short_prob >= SHORT_THRESHOLD and short_prob > long_prob + GATE_THRESHOLD

            if should_long or should_short:
                position = {
                    'side': 'SHORT' if should_short else 'LONG',
                    'entry_time': current_time,
                    'entry_idx': i,
                    'entry_price': current_price,
                    'entry_prob': short_prob if should_short else long_prob
                }

    # Close remaining position
    if position:
        row = df.iloc[-1]
        current_price = row['close']
        current_time = row['timestamp']
        hold_candles = len(df) - 1 - position['entry_idx']

        if position['side'] == 'LONG':
            exit_prob = row['long_exit_prob']
            pnl_pct = (current_price - position['entry_price']) / position['entry_price']
        else:
            exit_prob = row['short_exit_prob']
            pnl_pct = (position['entry_price'] - current_price) / position['entry_price']

        leveraged_pnl_pct = pnl_pct * LEVERAGE
        balance_pnl_pct = leveraged_pnl_pct * POSITION_SIZE_PCT
        pnl = balance * balance_pnl_pct
        balance += pnl

        trades.append({
            'entry_time': position['entry_time'],
            'exit_time': current_time,
            'side': position['side'],
            'entry_price': position['entry_price'],
            'exit_price': current_price,
            'entry_prob': position['entry_prob'],
            'exit_prob': exit_prob,
            'hold_candles': hold_candles,
            'pnl': pnl,
            'balance': balance,
            'exit_reason': 'Force Close'
        })

    return trades, balance


def analyze_results(trades, final_balance):
    """Analyze backtest results"""
    print(f"\n{'=' * 80}")
    print("BACKTEST RESULTS")
    print('=' * 80)

    df_trades = pd.DataFrame(trades) if len(trades) > 0 else pd.DataFrame()

    # Overall metrics
    total_return = (final_balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100

    print(f"\n📊 Overall Performance:")
    print(f"   Initial Balance: ${INITIAL_BALANCE:,.2f}")
    print(f"   Final Balance: ${final_balance:,.2f}")
    print(f"   Total Return: {total_return:+.2f}%")
    print(f"   Total Trades: {len(trades)}")

    if len(trades) == 0:
        print("\n⚠️  No trades executed!")
        return

    # Win rate
    wins = len(df_trades[df_trades['pnl'] > 0])
    losses = len(df_trades[df_trades['pnl'] <= 0])
    win_rate = wins / len(df_trades) * 100

    print(f"\n📈 Trade Statistics:")
    print(f"   Wins: {wins} ({win_rate:.2f}%)")
    print(f"   Losses: {losses} ({100-win_rate:.2f}%)")
    print(f"   Avg P&L: ${df_trades['pnl'].mean():,.2f}")
    print(f"   Best Trade: ${df_trades['pnl'].max():,.2f}")
    print(f"   Worst Trade: ${df_trades['pnl'].min():,.2f}")

    # Side distribution
    long_trades = len(df_trades[df_trades['side'] == 'LONG'])
    short_trades = len(df_trades[df_trades['side'] == 'SHORT'])

    print(f"\n📊 Side Distribution:")
    print(f"   LONG: {long_trades} ({long_trades/len(df_trades)*100:.1f}%)")
    print(f"   SHORT: {short_trades} ({short_trades/len(df_trades)*100:.1f}%)")

    # Exit reasons
    print(f"\n🚪 Exit Reasons:")
    for reason, count in df_trades['exit_reason'].value_counts().items():
        print(f"   {reason}: {count} ({count/len(df_trades)*100:.1f}%)")

    # Hold time
    print(f"\n⏱️  Hold Time:")
    print(f"   Avg: {df_trades['hold_candles'].mean():.1f} candles ({df_trades['hold_candles'].mean()*15/60:.1f} hours)")
    print(f"   Max: {df_trades['hold_candles'].max()} candles ({df_trades['hold_candles'].max()*15/60:.1f} hours)")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = RESULTS_DIR / f"backtest_314days_15min_{timestamp}.csv"
    df_trades.to_csv(output_file, index=False)

    print(f"\n💾 Results saved: {output_file.name}")


if __name__ == "__main__":
    print(f"\n{'=' * 80}")
    print("314-DAY 15-MIN MODEL BACKTEST")
    print('=' * 80)
    print(f"\nValidation Period: Oct 9 - Nov 6, 2025 (28 days)")
    print(f"Configuration: Entry {LONG_THRESHOLD}/{SHORT_THRESHOLD}, Exit {EXIT_THRESHOLD}")
    print(f"Max Hold: {MAX_HOLD_CANDLES} candles ({MAX_HOLD_CANDLES*15/60:.1f} hours @ 15-min)")

    models = load_models()
    df_val = load_validation_data()

    trades, final_balance = backtest(df_val, models)

    analyze_results(trades, final_balance)

    print(f"\n{'=' * 80}")
    print("BACKTEST COMPLETE")
    print('=' * 80)
