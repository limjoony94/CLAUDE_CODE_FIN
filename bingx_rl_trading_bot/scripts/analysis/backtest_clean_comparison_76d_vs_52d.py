#!/usr/bin/env python3
"""
Clean Backtest Comparison: 76-Day vs 52-Day Models
===================================================

Purpose: Fair performance comparison with 0% data leakage
Both models backtested on: Sep 29 - Oct 26, 2025 (27 days)

76-Day Models: Trained Jul 14 - Sep 28
52-Day Models: Trained Aug 7 - Sep 28

Backtest: Sep 29 - Oct 26 (100% out-of-sample for both)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from datetime import datetime

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "features"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Model timestamp
MODEL_TIMESTAMP = "20251106_140955"

# Production configuration
LONG_THRESHOLD = 0.85
SHORT_THRESHOLD = 0.80
GATE_THRESHOLD = 0.001
EXIT_THRESHOLD = 0.75
LEVERAGE = 4
STOP_LOSS_PCT = 0.03
MAX_HOLD_CANDLES = 120
POSITION_SIZE_PCT = 0.95
INITIAL_BALANCE = 10000.0

# Clean validation period (NO overlap with training)
VALIDATION_START = "2025-09-29 00:00:00"
VALIDATION_END = "2025-10-26 23:59:59"


def load_models(model_set='76day'):
    """Load model set (76day or 52day)"""
    print(f"\n{'=' * 80}")
    print(f"LOADING {model_set.upper()} MODELS")
    print('=' * 80)

    long_entry = joblib.load(MODELS_DIR / f"xgboost_long_entry_{model_set}_{MODEL_TIMESTAMP}.pkl")
    long_entry_scaler = joblib.load(MODELS_DIR / f"xgboost_long_entry_{model_set}_{MODEL_TIMESTAMP}_scaler.pkl")
    with open(MODELS_DIR / f"xgboost_long_entry_{model_set}_{MODEL_TIMESTAMP}_features.txt", 'r') as f:
        long_entry_features = [line.strip() for line in f.readlines()]

    short_entry = joblib.load(MODELS_DIR / f"xgboost_short_entry_{model_set}_{MODEL_TIMESTAMP}.pkl")
    short_entry_scaler = joblib.load(MODELS_DIR / f"xgboost_short_entry_{model_set}_{MODEL_TIMESTAMP}_scaler.pkl")
    with open(MODELS_DIR / f"xgboost_short_entry_{model_set}_{MODEL_TIMESTAMP}_features.txt", 'r') as f:
        short_entry_features = [line.strip() for line in f.readlines()]

    long_exit = joblib.load(MODELS_DIR / f"xgboost_long_exit_{model_set}_{MODEL_TIMESTAMP}.pkl")
    long_exit_scaler = joblib.load(MODELS_DIR / f"xgboost_long_exit_{model_set}_{MODEL_TIMESTAMP}_scaler.pkl")
    with open(MODELS_DIR / f"xgboost_long_exit_{model_set}_{MODEL_TIMESTAMP}_features.txt", 'r') as f:
        long_exit_features = [line.strip() for line in f.readlines()]

    short_exit = joblib.load(MODELS_DIR / f"xgboost_short_exit_{model_set}_{MODEL_TIMESTAMP}.pkl")
    short_exit_scaler = joblib.load(MODELS_DIR / f"xgboost_short_exit_{model_set}_{MODEL_TIMESTAMP}_scaler.pkl")
    with open(MODELS_DIR / f"xgboost_short_exit_{model_set}_{MODEL_TIMESTAMP}_features.txt", 'r') as f:
        short_exit_features = [line.strip() for line in f.readlines()]

    print(f"✅ {model_set.upper()} Models Loaded:")
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
    """Load clean validation period"""
    print(f"\n{'=' * 80}")
    print("LOADING CLEAN VALIDATION DATA")
    print('=' * 80)

    df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_features.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    val_start = pd.to_datetime(VALIDATION_START)
    val_end = pd.to_datetime(VALIDATION_END)

    df_val = df[(df['timestamp'] >= val_start) & (df['timestamp'] <= val_end)].copy()
    df_val = df_val.reset_index(drop=True)

    print(f"\n✅ Validation Period (100% Out-of-Sample):")
    print(f"   Period: {df_val['timestamp'].min()} to {df_val['timestamp'].max()}")
    print(f"   Rows: {len(df_val):,}")
    print(f"   Days: {(df_val['timestamp'].max() - df_val['timestamp'].min()).days}")

    return df_val


def backtest_models(df, models, model_name):
    """Run backtest with specified models"""
    print(f"\n{'=' * 80}")
    print(f"BACKTESTING {model_name.upper()}")
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


def analyze_results(trades_76d, balance_76d, trades_52d, balance_52d):
    """Compare results"""
    print(f"\n{'=' * 80}")
    print("BACKTEST COMPARISON RESULTS")
    print('=' * 80)

    df_76d = pd.DataFrame(trades_76d) if len(trades_76d) > 0 else pd.DataFrame()
    df_52d = pd.DataFrame(trades_52d) if len(trades_52d) > 0 else pd.DataFrame()

    # Metrics
    return_76d = (balance_76d - INITIAL_BALANCE) / INITIAL_BALANCE * 100
    return_52d = (balance_52d - INITIAL_BALANCE) / INITIAL_BALANCE * 100

    wr_76d = len(df_76d[df_76d['pnl'] > 0]) / len(df_76d) * 100 if len(df_76d) > 0 else 0
    wr_52d = len(df_52d[df_52d['pnl'] > 0]) / len(df_52d) * 100 if len(df_52d) > 0 else 0

    print(f"\n{'Metric':<25} | {'76-Day Models':<20} | {'52-Day Models':<20} | {'Winner':<10}")
    print("-" * 80)
    print(f"{'Total Return (%)':<25} | {return_76d:>19.2f}% | {return_52d:>19.2f}% | {'76d' if return_76d > return_52d else '52d':<10}")
    print(f"{'Total Trades':<25} | {len(trades_76d):>20} | {len(trades_52d):>20} | {'-':<10}")
    print(f"{'Win Rate (%)':<25} | {wr_76d:>19.2f}% | {wr_52d:>19.2f}% | {'76d' if wr_76d > wr_52d else '52d':<10}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if len(df_76d) > 0:
        df_76d.to_csv(RESULTS_DIR / f"backtest_76day_clean_{timestamp}.csv", index=False)
    if len(df_52d) > 0:
        df_52d.to_csv(RESULTS_DIR / f"backtest_52day_clean_{timestamp}.csv", index=False)

    print(f"\n✅ RECOMMENDATION: {'Keep 76-Day Models' if return_76d > return_52d else 'Keep 52-Day Models'}")
    print(f"   76-Day: {return_76d:+.2f}%")
    print(f"   52-Day: {return_52d:+.2f}%")


if __name__ == "__main__":
    print(f"\n{'=' * 80}")
    print("CLEAN BACKTEST COMPARISON: 76-DAY vs 52-DAY")
    print('=' * 80)
    print(f"\nValidation Period: Sep 29 - Oct 26, 2025")
    print(f"Configuration: Entry {LONG_THRESHOLD}/{SHORT_THRESHOLD}, Exit {EXIT_THRESHOLD}")

    df_val = load_validation_data()

    models_76d = load_models('76day')
    models_52d = load_models('52day')

    trades_76d, balance_76d = backtest_models(df_val, models_76d, '76-Day Models')
    trades_52d, balance_52d = backtest_models(df_val, models_52d, '52-Day Models')

    analyze_results(trades_76d, balance_76d, trades_52d, balance_52d)

    print(f"\n{'=' * 80}")
    print("COMPARISON COMPLETE")
    print('=' * 80)
