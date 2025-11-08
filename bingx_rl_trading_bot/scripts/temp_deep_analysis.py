"""
Deep Analysis: LONG Probability Trend and Performance Correlation
==================================================================

Investigate:
1. Time-series trend of LONG probabilities
2. Performance in high vs low LONG probability periods
3. Root cause of probability shift
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import pickle
import joblib
from experiments.calculate_all_features import calculate_all_features
from experiments.retrain_exit_models_opportunity_gating import prepare_exit_features

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"

print("="*80)
print("DEEP ANALYSIS: LONG PROBABILITY TREND")
print("="*80)

# Load data
print("\n1️⃣ Loading data...")
df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_max.csv")
print(f"   Loaded {len(df):,} candles")
print(f"   Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

# Calculate features
print("\n2️⃣ Calculating features...")
df = calculate_all_features(df)
df = prepare_exit_features(df)
print(f"   Features calculated")

# Load models
timestamp = "20251018_233146"
print(f"\n3️⃣ Loading models: {timestamp}")

# LONG Entry
with open(MODELS_DIR / f"xgboost_long_trade_outcome_full_{timestamp}.pkl", 'rb') as f:
    long_model = pickle.load(f)
long_scaler = joblib.load(MODELS_DIR / f"xgboost_long_trade_outcome_full_{timestamp}_scaler.pkl")
with open(MODELS_DIR / f"xgboost_long_trade_outcome_full_{timestamp}_features.txt", 'r') as f:
    long_features = [line.strip() for line in f.readlines() if line.strip()]

# LONG Exit
with open(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251017_151624.pkl", 'rb') as f:
    long_exit_model = pickle.load(f)
long_exit_scaler = joblib.load(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251017_151624_scaler.pkl")
with open(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251017_151624_features.txt", 'r') as f:
    long_exit_features = [line.strip() for line in f.readlines() if line.strip()]

print(f"   Models loaded")

# Calculate probabilities
print("\n4️⃣ Calculating probabilities for all candles...")
long_probs = []
timestamps = []

for i in range(len(df)):
    try:
        long_feat = df[long_features].iloc[i:i+1].values
        long_scaled = long_scaler.transform(long_feat)
        long_prob = long_model.predict_proba(long_scaled)[0][1]
        long_probs.append(long_prob)
        timestamps.append(df['timestamp'].iloc[i])

        if (i + 1) % 5000 == 0:
            print(f"   Processed {i+1:,} candles...")
    except:
        long_probs.append(np.nan)
        timestamps.append(df['timestamp'].iloc[i])

prob_df = pd.DataFrame({
    'timestamp': timestamps,
    'long_prob': long_probs
})
prob_df['timestamp'] = pd.to_datetime(prob_df['timestamp'])

print(f"   ✅ Probabilities calculated")

# ============================================================================
# ANALYSIS 1: Time-series trend
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS 1: TIME-SERIES TREND OF LONG PROBABILITIES")
print("="*80)

# Split into 10 equal periods
n_periods = 10
period_size = len(prob_df) // n_periods

print(f"\nDividing {len(prob_df):,} candles into {n_periods} periods of ~{period_size:,} candles each:")
print(f"\n{'Period':<10} {'Date Range':<40} {'LONG Avg':<12} {'LONG >= 0.65':<15}")
print("-"*80)

for i in range(n_periods):
    start_idx = i * period_size
    end_idx = (i + 1) * period_size if i < n_periods - 1 else len(prob_df)

    period_data = prob_df.iloc[start_idx:end_idx]
    avg_prob = period_data['long_prob'].mean()
    high_prob_pct = (period_data['long_prob'] >= 0.65).sum() / len(period_data) * 100

    date_start = period_data['timestamp'].iloc[0].strftime('%Y-%m-%d')
    date_end = period_data['timestamp'].iloc[-1].strftime('%Y-%m-%d')

    print(f"Period {i+1:<3} {date_start} to {date_end:<15} {avg_prob:>6.2%}      {high_prob_pct:>6.1f}%")

# ============================================================================
# ANALYSIS 2: Performance in high vs low probability periods
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS 2: PERFORMANCE BY PROBABILITY LEVEL")
print("="*80)

# Simulate trades for different probability ranges
LONG_THRESHOLD = 0.65
LEVERAGE = 4.0
INITIAL_BALANCE = 10000
FEE_RATE = 0.0005

def simulate_window(window_df, start_balance):
    """Simulate trading for a single window"""
    balance = start_balance
    trades = []
    position = None

    for i in range(len(window_df)):
        current_candle = window_df.iloc[i]

        # Exit check
        if position is not None:
            hold_time = i - position['entry_idx']

            # Get exit probability
            exit_feat = window_df[long_exit_features].iloc[i:i+1].values
            exit_scaled = long_exit_scaler.transform(exit_feat)
            exit_prob = long_exit_model.predict_proba(exit_scaled)[0][1]

            # Exit conditions
            entry_price = position['entry_price']
            current_price = current_candle['close']
            pnl_pct = (current_price - entry_price) / entry_price
            leveraged_pnl_pct = pnl_pct * LEVERAGE

            should_exit = False
            exit_reason = None

            # ML Exit
            if exit_prob >= 0.70:
                should_exit = True
                exit_reason = 'ML_EXIT'
            # Take Profit
            elif leveraged_pnl_pct >= 0.03:
                should_exit = True
                exit_reason = 'TP'
            # Stop Loss
            elif leveraged_pnl_pct <= -0.04:
                should_exit = True
                exit_reason = 'SL'
            # Max Hold
            elif hold_time >= 96:  # 8 hours
                should_exit = True
                exit_reason = 'MAX_HOLD'

            if should_exit:
                position_value = balance * position['position_size_pct']
                pnl_usd = position_value * leveraged_pnl_pct

                # Fees
                entry_fee = position_value * FEE_RATE
                exit_fee = position_value * FEE_RATE
                total_fees = entry_fee + exit_fee

                pnl_after_fees = pnl_usd - total_fees
                balance += pnl_after_fees

                trades.append({
                    'entry_prob': position['entry_prob'],
                    'pnl_after_fees': pnl_after_fees,
                    'exit_reason': exit_reason
                })

                position = None

        # Entry check
        if position is None:
            long_prob = current_candle['long_prob']

            if long_prob >= LONG_THRESHOLD:
                # Enter LONG
                position = {
                    'entry_idx': i,
                    'entry_price': current_candle['close'],
                    'entry_prob': long_prob,
                    'position_size_pct': 0.95  # Fixed for analysis
                }

    return balance, trades

# Merge probabilities with data
df_with_prob = df.copy()
df_with_prob['long_prob'] = long_probs

# Analyze performance in different time periods
print("\n📊 Backtest Performance by Time Period:")
print(f"\n{'Period':<10} {'LONG Avg':<12} {'Return':<12} {'Win Rate':<12} {'Trades':<10}")
print("-"*60)

for i in range(n_periods):
    start_idx = i * period_size
    end_idx = (i + 1) * period_size if i < n_periods - 1 else len(df_with_prob)

    period_data = df_with_prob.iloc[start_idx:end_idx].copy()
    avg_prob = period_data['long_prob'].mean()

    # Simulate
    final_balance, trades = simulate_window(period_data, INITIAL_BALANCE)

    if len(trades) > 0:
        returns = (final_balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100
        win_rate = sum(1 for t in trades if t['pnl_after_fees'] > 0) / len(trades) * 100
        print(f"Period {i+1:<3} {avg_prob:>6.2%}      {returns:>6.1f}%      {win_rate:>6.1f}%      {len(trades):<10}")
    else:
        print(f"Period {i+1:<3} {avg_prob:>6.2%}      No trades  -            -")

# ============================================================================
# ANALYSIS 3: Feature importance in recent vs old data
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS 3: MARKET CHARACTERISTICS COMPARISON")
print("="*80)

# Compare first 3000 vs last 3000 candles
old_data = df.iloc[:3000]
recent_data = df.iloc[-3000:]

print(f"\nComparing OLD (first 3000) vs RECENT (last 3000) candles:")
print(f"\n{'Feature':<25} {'OLD Avg':<15} {'RECENT Avg':<15} {'Change':<15}")
print("-"*70)

# Key features to compare
key_features = [
    'close', 'volume', 'rsi_14', 'macd', 'bb_width',
    'ema_12', 'ema_26', 'atr_14', 'obv'
]

for feat in key_features:
    if feat in df.columns:
        old_avg = old_data[feat].mean()
        recent_avg = recent_data[feat].mean()
        change_pct = (recent_avg - old_avg) / old_avg * 100 if old_avg != 0 else 0

        print(f"{feat:<25} {old_avg:>14.2f} {recent_avg:>14.2f} {change_pct:>+13.1f}%")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

# Overall statistics
prob_df_clean = prob_df.dropna()
overall_avg = prob_df_clean['long_prob'].mean()
recent_1000_avg = prob_df_clean.iloc[-1000:]['long_prob'].mean()
recent_500_avg = prob_df_clean.iloc[-500:]['long_prob'].mean()
recent_100_avg = prob_df_clean.iloc[-100:]['long_prob'].mean()

print(f"\nLONG Probability Trend:")
print(f"  Overall (all {len(prob_df_clean):,} candles): {overall_avg:.2%}")
print(f"  Recent 1000 candles: {recent_1000_avg:.2%} ({(recent_1000_avg/overall_avg - 1)*100:+.1f}%)")
print(f"  Recent 500 candles:  {recent_500_avg:.2%} ({(recent_500_avg/overall_avg - 1)*100:+.1f}%)")
print(f"  Recent 100 candles:  {recent_100_avg:.2%} ({(recent_100_avg/overall_avg - 1)*100:+.1f}%)")

print("\n" + "="*80)
