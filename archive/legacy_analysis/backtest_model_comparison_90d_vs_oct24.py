#!/usr/bin/env python3
"""
Model Comparison Backtest: 90-Day Models vs Oct 24 Models
==========================================================

Purpose: Compare performance of new 90-day models vs Oct 24 models on clean validation period
Models:
  - New 90d: xgboost_*_90days_20251106_103807.pkl
  - Oct 24: xgboost_*_enhanced_20251024_012445.pkl (Entry), xgboost_*_oppgating_improved_20251024_043527/044510.pkl (Exit)

Validation Period: Sep 28 - Oct 26, 2025 (28 days, ~8,064 candles)
Data Status: 100% Out-of-Sample for both model sets (0% training overlap)

Configuration: Production settings (Entry 0.85/0.80, Exit 0.75/0.75, 4x leverage, -3% SL, 120 candle max hold)
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

# Configuration (Production settings)
LONG_THRESHOLD = 0.85  # Production LONG threshold (Nov 5 update)
SHORT_THRESHOLD = 0.80  # Production SHORT threshold (Oct 24)
GATE_THRESHOLD = 0.001
EXIT_THRESHOLD = 0.75  # ML Exit threshold
LEVERAGE = 4
STOP_LOSS_PCT = 0.03  # -3% balance
MAX_HOLD_CANDLES = 120  # 10 hours
POSITION_SIZE_PCT = 0.95  # 95% of balance
INITIAL_BALANCE = 10000.0

# Validation period (CLEAN - no training overlap for both model sets)
# Oct 24 training ends at Sep 28 18:35, so start validation from Sep 29 00:00
VALIDATION_START = "2025-09-29 00:00:00"
VALIDATION_END = "2025-10-26 23:59:59"


def load_models(model_set='90d'):
    """Load model set (90d or oct24)"""
    print(f"\n{'=' * 80}")
    print(f"LOADING {model_set.upper()} MODELS")
    print('=' * 80)

    if model_set == '90d':
        # New 90-day models
        timestamp = '20251106_103807'

        long_entry_model = joblib.load(MODELS_DIR / f"xgboost_long_entry_90days_{timestamp}.pkl")
        long_entry_scaler = joblib.load(MODELS_DIR / f"xgboost_long_entry_90days_{timestamp}_scaler.pkl")
        with open(MODELS_DIR / f"xgboost_long_entry_90days_{timestamp}_features.txt", 'r') as f:
            long_entry_features = [line.strip() for line in f.readlines()]

        short_entry_model = joblib.load(MODELS_DIR / f"xgboost_short_entry_90days_{timestamp}.pkl")
        short_entry_scaler = joblib.load(MODELS_DIR / f"xgboost_short_entry_90days_{timestamp}_scaler.pkl")
        with open(MODELS_DIR / f"xgboost_short_entry_90days_{timestamp}_features.txt", 'r') as f:
            short_entry_features = [line.strip() for line in f.readlines()]

        long_exit_model = joblib.load(MODELS_DIR / f"xgboost_long_exit_90days_{timestamp}.pkl")
        long_exit_scaler = joblib.load(MODELS_DIR / f"xgboost_long_exit_90days_{timestamp}_scaler.pkl")
        with open(MODELS_DIR / f"xgboost_long_exit_90days_{timestamp}_features.txt", 'r') as f:
            long_exit_features = [line.strip() for line in f.readlines()]

        short_exit_model = joblib.load(MODELS_DIR / f"xgboost_short_exit_90days_{timestamp}.pkl")
        short_exit_scaler = joblib.load(MODELS_DIR / f"xgboost_short_exit_90days_{timestamp}_scaler.pkl")
        with open(MODELS_DIR / f"xgboost_short_exit_90days_{timestamp}_features.txt", 'r') as f:
            short_exit_features = [line.strip() for line in f.readlines()]

        print(f"✅ 90-Day Models ({timestamp}):")
        print(f"   LONG Entry: {len(long_entry_features)} features")
        print(f"   SHORT Entry: {len(short_entry_features)} features")
        print(f"   LONG Exit: {len(long_exit_features)} features")
        print(f"   SHORT Exit: {len(short_exit_features)} features")

    else:  # oct24
        # Oct 24 models
        entry_timestamp = '20251024_012445'
        long_exit_timestamp = '20251024_043527'
        short_exit_timestamp = '20251024_044510'

        long_entry_model = joblib.load(MODELS_DIR / f"xgboost_long_entry_enhanced_{entry_timestamp}.pkl")
        long_entry_scaler = joblib.load(MODELS_DIR / f"xgboost_long_entry_enhanced_{entry_timestamp}_scaler.pkl")
        with open(MODELS_DIR / f"xgboost_long_entry_enhanced_{entry_timestamp}_features.txt", 'r') as f:
            long_entry_features = [line.strip() for line in f.readlines()]

        short_entry_model = joblib.load(MODELS_DIR / f"xgboost_short_entry_enhanced_{entry_timestamp}.pkl")
        short_entry_scaler = joblib.load(MODELS_DIR / f"xgboost_short_entry_enhanced_{entry_timestamp}_scaler.pkl")
        with open(MODELS_DIR / f"xgboost_short_entry_enhanced_{entry_timestamp}_features.txt", 'r') as f:
            short_entry_features = [line.strip() for line in f.readlines()]

        long_exit_model = joblib.load(MODELS_DIR / f"xgboost_long_exit_oppgating_improved_{long_exit_timestamp}.pkl")
        long_exit_scaler = joblib.load(MODELS_DIR / f"xgboost_long_exit_oppgating_improved_{long_exit_timestamp}_scaler.pkl")
        with open(MODELS_DIR / f"xgboost_long_exit_oppgating_improved_{long_exit_timestamp}_features.txt", 'r') as f:
            long_exit_features = [line.strip() for line in f.readlines()]

        short_exit_model = joblib.load(MODELS_DIR / f"xgboost_short_exit_oppgating_improved_{short_exit_timestamp}.pkl")
        short_exit_scaler = joblib.load(MODELS_DIR / f"xgboost_short_exit_oppgating_improved_{short_exit_timestamp}_scaler.pkl")
        with open(MODELS_DIR / f"xgboost_short_exit_oppgating_improved_{short_exit_timestamp}_features.txt", 'r') as f:
            short_exit_features = [line.strip() for line in f.readlines()]

        print(f"✅ Oct 24 Models (Enhanced 5-Fold CV):")
        print(f"   LONG Entry: {len(long_entry_features)} features")
        print(f"   SHORT Entry: {len(short_entry_features)} features")
        print(f"   LONG Exit: {len(long_exit_features)} features")
        print(f"   SHORT Exit: {len(short_exit_features)} features")

    return {
        'long_entry': (long_entry_model, long_entry_scaler, long_entry_features),
        'short_entry': (short_entry_model, short_entry_scaler, short_entry_features),
        'long_exit': (long_exit_model, long_exit_scaler, long_exit_features),
        'short_exit': (short_exit_model, short_exit_scaler, short_exit_features)
    }


def load_clean_validation_data():
    """Load CLEAN validation period only (Sep 28 - Oct 26)"""
    print("\n" + "=" * 80)
    print("LOADING CLEAN VALIDATION DATA")
    print("=" * 80)

    # Load features
    df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_features.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    print(f"Full Dataset: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Total Rows: {len(df):,}")

    # Extract CLEAN validation period only
    val_start = pd.to_datetime(VALIDATION_START)
    val_end = pd.to_datetime(VALIDATION_END)

    df_val = df[(df['timestamp'] >= val_start) & (df['timestamp'] <= val_end)].copy()
    df_val = df_val.reset_index(drop=True)

    print(f"\n✅ CLEAN Validation Period (NO training overlap):")
    print(f"   Period: {df_val['timestamp'].min()} to {df_val['timestamp'].max()}")
    print(f"   Rows: {len(df_val):,}")
    print(f"   Days: {(df_val['timestamp'].max() - df_val['timestamp'].min()).days}")

    return df_val


def backtest_with_models(df, models, model_name):
    """Run backtest using specified model set"""
    print(f"\n{'=' * 80}")
    print(f"BACKTESTING WITH {model_name.upper()} MODELS")
    print('=' * 80)
    print(f"Configuration:")
    print(f"  Entry Threshold: {LONG_THRESHOLD}/{SHORT_THRESHOLD} (LONG/SHORT)")
    print(f"  Exit Threshold: {EXIT_THRESHOLD}")
    print(f"  Gate Threshold: {GATE_THRESHOLD}")
    print(f"  Leverage: {LEVERAGE}×")
    print(f"  Stop Loss: -{STOP_LOSS_PCT * 100}% balance")
    print(f"  Max Hold: {MAX_HOLD_CANDLES} candles ({MAX_HOLD_CANDLES / 12:.1f} hours)")
    print(f"  Position Size: {POSITION_SIZE_PCT * 100}% balance")
    print(f"  Initial Balance: ${INITIAL_BALANCE:,.2f}")

    balance = INITIAL_BALANCE
    position = None
    trades = []

    # Predict probabilities for Entry models
    long_entry_model, long_entry_scaler, long_entry_features = models['long_entry']
    short_entry_model, short_entry_scaler, short_entry_features = models['short_entry']
    long_exit_model, long_exit_scaler, long_exit_features = models['long_exit']
    short_exit_model, short_exit_scaler, short_exit_features = models['short_exit']

    # Entry probabilities
    X_long_entry = df[long_entry_features].fillna(0).replace([np.inf, -np.inf], 0).values
    X_short_entry = df[short_entry_features].fillna(0).replace([np.inf, -np.inf], 0).values

    X_long_entry_scaled = long_entry_scaler.transform(X_long_entry)
    X_short_entry_scaled = short_entry_scaler.transform(X_short_entry)

    long_entry_probs = long_entry_model.predict_proba(X_long_entry_scaled)[:, 1]
    short_entry_probs = short_entry_model.predict_proba(X_short_entry_scaled)[:, 1]

    # Exit probabilities (handle missing features by filling with zeros)
    available_features = df.columns.tolist()

    # Check for missing features and fill with zeros
    long_exit_missing = [f for f in long_exit_features if f not in available_features]
    short_exit_missing = [f for f in short_exit_features if f not in available_features]

    if len(long_exit_missing) > 0:
        print(f"\n⚠️  LONG Exit: {len(long_exit_missing)}/{len(long_exit_features)} features missing (filling with zeros)")
        print(f"   Missing: {sorted(long_exit_missing)[:5]}...")
        for feat in long_exit_missing:
            df[feat] = 0.0

    if len(short_exit_missing) > 0:
        print(f"⚠️  SHORT Exit: {len(short_exit_missing)}/{len(short_exit_features)} features missing (filling with zeros)")
        print(f"   Missing: {sorted(short_exit_missing)[:5]}...")
        for feat in short_exit_missing:
            if feat not in df.columns:  # Avoid duplicate if already added for LONG
                df[feat] = 0.0

    X_long_exit = df[long_exit_features].fillna(0).replace([np.inf, -np.inf], 0).values
    X_short_exit = df[short_exit_features].fillna(0).replace([np.inf, -np.inf], 0).values

    X_long_exit_scaled = long_exit_scaler.transform(X_long_exit)
    X_short_exit_scaled = short_exit_scaler.transform(X_short_exit)

    long_exit_probs = long_exit_model.predict_proba(X_long_exit_scaled)[:, 1]
    short_exit_probs = short_exit_model.predict_proba(X_short_exit_scaled)[:, 1]

    # Add to dataframe for analysis
    df['long_entry_prob'] = long_entry_probs
    df['short_entry_prob'] = short_entry_probs
    df['long_exit_prob'] = long_exit_probs
    df['short_exit_prob'] = short_exit_probs

    print(f"\nEntry Signal Coverage:")
    print(f"  LONG >= {LONG_THRESHOLD}: {(long_entry_probs >= LONG_THRESHOLD).sum()} candles ({(long_entry_probs >= LONG_THRESHOLD).mean() * 100:.2f}%)")
    print(f"  SHORT >= {SHORT_THRESHOLD}: {(short_entry_probs >= SHORT_THRESHOLD).sum()} candles ({(short_entry_probs >= SHORT_THRESHOLD).mean() * 100:.2f}%)")

    # Backtest loop
    for i in range(len(df)):
        row = df.iloc[i]
        current_price = row['close']
        current_time = row['timestamp']

        # Check existing position
        if position:
            # Calculate hold time
            hold_candles = i - position['entry_idx']

            # Check ML Exit
            if position['side'] == 'LONG':
                exit_prob = row['long_exit_prob']
                pnl_pct = (current_price - position['entry_price']) / position['entry_price']
            else:  # SHORT
                exit_prob = row['short_exit_prob']
                pnl_pct = (position['entry_price'] - current_price) / position['entry_price']

            leveraged_pnl_pct = pnl_pct * LEVERAGE
            balance_pnl_pct = leveraged_pnl_pct * position['position_size_pct']

            # ML Exit
            if exit_prob >= EXIT_THRESHOLD:
                # Close position
                exit_price = current_price
                pnl = balance * balance_pnl_pct
                balance += pnl

                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': current_time,
                    'side': position['side'],
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
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
                # Close position
                exit_price = current_price
                pnl = balance * balance_pnl_pct
                balance += pnl

                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': current_time,
                    'side': position['side'],
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
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
                # Close position
                exit_price = current_price
                pnl = balance * balance_pnl_pct
                balance += pnl

                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': current_time,
                    'side': position['side'],
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'entry_prob': position['entry_prob'],
                    'exit_prob': exit_prob,
                    'hold_candles': hold_candles,
                    'pnl': pnl,
                    'balance': balance,
                    'exit_reason': 'Max Hold'
                })

                position = None
                continue

        # No position - check entry signals
        if not position:
            long_entry_prob = row['long_entry_prob']
            short_entry_prob = row['short_entry_prob']

            # Opportunity Gating
            should_long = long_entry_prob >= LONG_THRESHOLD
            should_short = short_entry_prob >= SHORT_THRESHOLD and short_entry_prob > long_entry_prob + GATE_THRESHOLD

            if should_long or should_short:
                side = 'SHORT' if should_short else 'LONG'
                entry_prob = short_entry_prob if should_short else long_entry_prob

                # Enter position
                position = {
                    'side': side,
                    'entry_time': current_time,
                    'entry_idx': i,
                    'entry_price': current_price,
                    'entry_prob': entry_prob,
                    'position_size_pct': POSITION_SIZE_PCT
                }

    # Close any remaining position at end
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
        balance_pnl_pct = leveraged_pnl_pct * position['position_size_pct']
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


def analyze_results(trades_90d, balance_90d, trades_oct24, balance_oct24):
    """Analyze and compare backtest results"""
    print("\n" + "=" * 80)
    print("BACKTEST COMPARISON RESULTS")
    print("=" * 80)

    if len(trades_90d) == 0 and len(trades_oct24) == 0:
        print("❌ No trades executed in either model set")
        return

    df_90d = pd.DataFrame(trades_90d) if len(trades_90d) > 0 else pd.DataFrame()
    df_oct24 = pd.DataFrame(trades_oct24) if len(trades_oct24) > 0 else pd.DataFrame()

    # Overall metrics comparison
    print(f"\n{'=' * 80}")
    print("PERFORMANCE COMPARISON")
    print('=' * 80)

    print(f"\n{'Metric':<25} | {'90-Day Models':<20} | {'Oct 24 Models':<20} | {'Winner':<10}")
    print("-" * 80)

    # Return
    return_90d = (balance_90d - INITIAL_BALANCE) / INITIAL_BALANCE * 100 if len(trades_90d) > 0 else 0
    return_oct24 = (balance_oct24 - INITIAL_BALANCE) / INITIAL_BALANCE * 100 if len(trades_oct24) > 0 else 0
    winner_return = "90d" if return_90d > return_oct24 else "Oct24" if return_oct24 > return_90d else "Tie"
    print(f"{'Total Return (%)':<25} | {return_90d:>19.2f}% | {return_oct24:>19.2f}% | {winner_return:<10}")

    # Trades
    trades_90d_count = len(trades_90d)
    trades_oct24_count = len(trades_oct24)
    print(f"{'Total Trades':<25} | {trades_90d_count:>20} | {trades_oct24_count:>20} | {'-':<10}")

    # Win Rate
    if len(df_90d) > 0:
        wins_90d = df_90d[df_90d['pnl'] > 0]
        wr_90d = len(wins_90d) / len(df_90d) * 100
    else:
        wr_90d = 0

    if len(df_oct24) > 0:
        wins_oct24 = df_oct24[df_oct24['pnl'] > 0]
        wr_oct24 = len(wins_oct24) / len(df_oct24) * 100
    else:
        wr_oct24 = 0

    winner_wr = "90d" if wr_90d > wr_oct24 else "Oct24" if wr_oct24 > wr_90d else "Tie"
    print(f"{'Win Rate (%)':<25} | {wr_90d:>19.2f}% | {wr_oct24:>19.2f}% | {winner_wr:<10}")

    # Profit Factor
    if len(df_90d) > 0:
        wins_90d = df_90d[df_90d['pnl'] > 0]
        losses_90d = df_90d[df_90d['pnl'] <= 0]
        if len(wins_90d) > 0 and len(losses_90d) > 0:
            pf_90d = wins_90d['pnl'].sum() / abs(losses_90d['pnl'].sum())
        else:
            pf_90d = 0
    else:
        pf_90d = 0

    if len(df_oct24) > 0:
        wins_oct24 = df_oct24[df_oct24['pnl'] > 0]
        losses_oct24 = df_oct24[df_oct24['pnl'] <= 0]
        if len(wins_oct24) > 0 and len(losses_oct24) > 0:
            pf_oct24 = wins_oct24['pnl'].sum() / abs(losses_oct24['pnl'].sum())
        else:
            pf_oct24 = 0
    else:
        pf_oct24 = 0

    winner_pf = "90d" if pf_90d > pf_oct24 else "Oct24" if pf_oct24 > pf_90d else "Tie"
    print(f"{'Profit Factor':<25} | {pf_90d:>19.2f}× | {pf_oct24:>19.2f}× | {winner_pf:<10}")

    # Average Trade P&L
    avg_pnl_90d = df_90d['pnl'].mean() if len(df_90d) > 0 else 0
    avg_pnl_oct24 = df_oct24['pnl'].mean() if len(df_oct24) > 0 else 0
    winner_avg = "90d" if avg_pnl_90d > avg_pnl_oct24 else "Oct24" if avg_pnl_oct24 > avg_pnl_90d else "Tie"
    print(f"{'Avg Trade P&L ($)':<25} | ${avg_pnl_90d:>18.2f} | ${avg_pnl_oct24:>18.2f} | {winner_avg:<10}")

    # Trade Breakdown
    print(f"\n{'=' * 80}")
    print("TRADE BREAKDOWN")
    print('=' * 80)

    if len(df_90d) > 0:
        long_90d = df_90d[df_90d['side'] == 'LONG']
        short_90d = df_90d[df_90d['side'] == 'SHORT']
        print(f"\n90-Day Models:")
        print(f"  LONG: {len(long_90d)} ({len(long_90d) / len(df_90d) * 100:.1f}%)")
        print(f"  SHORT: {len(short_90d)} ({len(short_90d) / len(df_90d) * 100:.1f}%)")
    else:
        print(f"\n90-Day Models: No trades")

    if len(df_oct24) > 0:
        long_oct24 = df_oct24[df_oct24['side'] == 'LONG']
        short_oct24 = df_oct24[df_oct24['side'] == 'SHORT']
        print(f"\nOct 24 Models:")
        print(f"  LONG: {len(long_oct24)} ({len(long_oct24) / len(df_oct24) * 100:.1f}%)")
        print(f"  SHORT: {len(short_oct24)} ({len(short_oct24) / len(df_oct24) * 100:.1f}%)")
    else:
        print(f"\nOct 24 Models: No trades")

    # Exit Reasons
    print(f"\n{'=' * 80}")
    print("EXIT MECHANISM DISTRIBUTION")
    print('=' * 80)

    if len(df_90d) > 0:
        print(f"\n90-Day Models:")
        for reason, count in df_90d['exit_reason'].value_counts().items():
            pct = count / len(df_90d) * 100
            print(f"  {reason}: {count} ({pct:.1f}%)")
    else:
        print(f"\n90-Day Models: No trades")

    if len(df_oct24) > 0:
        print(f"\nOct 24 Models:")
        for reason, count in df_oct24['exit_reason'].value_counts().items():
            pct = count / len(df_oct24) * 100
            print(f"  {reason}: {count} ({pct:.1f}%)")
    else:
        print(f"\nOct 24 Models: No trades")

    # Recommendation
    print(f"\n{'=' * 80}")
    print("DEPLOYMENT RECOMMENDATION")
    print('=' * 80)

    if return_90d > return_oct24:
        improvement = return_90d - return_oct24
        print(f"\n✅ RECOMMENDATION: Deploy 90-Day Models")
        print(f"   Reason: +{improvement:.2f}% better return on clean validation")
        print(f"   90-Day Return: {return_90d:+.2f}%")
        print(f"   Oct 24 Return: {return_oct24:+.2f}%")
    elif return_oct24 > return_90d:
        improvement = return_oct24 - return_90d
        print(f"\n✅ RECOMMENDATION: Keep Oct 24 Models")
        print(f"   Reason: +{improvement:.2f}% better return on clean validation")
        print(f"   Oct 24 Return: {return_oct24:+.2f}%")
        print(f"   90-Day Return: {return_90d:+.2f}%")
        print(f"   Note: Oct 24 models have 24 more days of training data (76d vs 52d)")
    else:
        print(f"\n⚖️ RECOMMENDATION: Either model acceptable")
        print(f"   Reason: Identical performance on clean validation")
        print(f"   Both Return: {return_90d:+.2f}%")
        print(f"   Preference: Oct 24 models (more training data)")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if len(df_90d) > 0:
        output_90d = RESULTS_DIR / f"backtest_90d_models_{timestamp}.csv"
        df_90d.to_csv(output_90d, index=False)
        print(f"\n💾 90-Day Results: {output_90d.name}")

    if len(df_oct24) > 0:
        output_oct24 = RESULTS_DIR / f"backtest_oct24_models_{timestamp}.csv"
        df_oct24.to_csv(output_oct24, index=False)
        print(f"💾 Oct 24 Results: {output_oct24.name}")

    return {
        '90d': {'trades': trades_90d_count, 'return': return_90d, 'win_rate': wr_90d, 'profit_factor': pf_90d},
        'oct24': {'trades': trades_oct24_count, 'return': return_oct24, 'win_rate': wr_oct24, 'profit_factor': pf_oct24}
    }


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("MODEL COMPARISON BACKTEST: 90-DAY vs OCT 24")
    print("=" * 80)
    print(f"\nObjective: Compare performance on clean validation period")
    print(f"Period: Sep 28 - Oct 26, 2025 (28 days)")
    print(f"Data Status: 100% Out-of-Sample for both model sets")
    print(f"\nModels:")
    print(f"  - New 90d: 20251106_103807 (52 days training)")
    print(f"  - Oct 24: 20251024_012445 (76 days training)")
    print(f"\nConfiguration:")
    print(f"  Entry: {LONG_THRESHOLD}/{SHORT_THRESHOLD} (LONG/SHORT)")
    print(f"  Exit: {EXIT_THRESHOLD}")
    print(f"  Leverage: {LEVERAGE}×, SL: -{STOP_LOSS_PCT * 100}%, Max Hold: {MAX_HOLD_CANDLES} candles")

    # Load validation data
    df_val = load_clean_validation_data()

    # Load both model sets
    models_90d = load_models('90d')
    models_oct24 = load_models('oct24')

    # Run backtests
    trades_90d, balance_90d = backtest_with_models(df_val, models_90d, '90-Day')
    trades_oct24, balance_oct24 = backtest_with_models(df_val, models_oct24, 'Oct 24')

    # Analyze and compare
    results = analyze_results(trades_90d, balance_90d, trades_oct24, balance_oct24)

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)
