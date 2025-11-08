"""
Fine-Tuned Exit Threshold Optimization (0.15-0.20 Range)
=========================================================
Tests intermediate thresholds to find better Win Rate / ML Exit balance

User Request: "각각 확인을 진행해 봅시다" (Let's check each one)
Purpose: Find optimal balance between Win Rate (target 70-75%) and ML Exit (target 75-85%)

Previous Results:
  Exit 0.15: WR 49.9%, ML Exit 87.1% (low WR, excellent ML)
  Exit 0.20: WR 73.2%, ML Exit 64.8% (excellent WR, low ML)

Hypothesis: Sweet spot exists between 0.15-0.20 achieving both targets
"""

import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
from datetime import datetime, timedelta

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

# Fine-tuned thresholds to test
EXIT_THRESHOLDS = [0.16, 0.17, 0.18, 0.19]

# Configuration (match actual production)
LONG_ENTRY_THRESHOLD = 0.80
SHORT_ENTRY_THRESHOLD = 0.80
EMERGENCY_STOP_LOSS = 0.03
EMERGENCY_MAX_HOLD_TIME = 120
LEVERAGE = 4
OPPORTUNITY_GATE_THRESHOLD = 0.001

print("=" * 80)
print("Fine-Tuned Exit Threshold Optimization")
print("=" * 80)
print(f"Testing {len(EXIT_THRESHOLDS)} thresholds: {EXIT_THRESHOLDS}")
print(f"Range: 0.15-0.20 (finding Win Rate / ML Exit balance)")
print()

# Load models (ACTUAL production models)
print("Loading models...")
with open(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445.pkl", 'rb') as f:
    long_entry_model = pickle.load(f)
long_entry_scaler = joblib.load(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_scaler.pkl")
with open(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_features.txt", 'r') as f:
    long_entry_features = [line.strip() for line in f.readlines()]

with open(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445.pkl", 'rb') as f:
    short_entry_model = pickle.load(f)
short_entry_scaler = joblib.load(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_scaler.pkl")
with open(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_features.txt", 'r') as f:
    short_entry_features = [line.strip() for line in f.readlines()]

with open(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527.pkl", 'rb') as f:
    long_exit_model = pickle.load(f)
with open(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527_features.txt", 'r') as f:
    long_exit_features = [line.strip() for line in f.readlines() if line.strip()]

with open(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510.pkl", 'rb') as f:
    short_exit_model = pickle.load(f)
with open(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510_features.txt", 'r') as f:
    short_exit_features = [line.strip() for line in f.readlines() if line.strip()]

print(f"✓ LONG Entry: 85 features, threshold {LONG_ENTRY_THRESHOLD}")
print(f"✓ SHORT Entry: 79 features, threshold {SHORT_ENTRY_THRESHOLD}")
print(f"✓ LONG Exit: {len(long_exit_features)} features")
print(f"✓ SHORT Exit: {len(short_exit_features)} features")
print()

# Load data
print("Loading data...")
df_master = pd.read_csv(DATA_DIR / "features" / "BTCUSDT_5m_features.csv")

# Enhanced features for Exit models (same as production)
df_master['volume_surge'] = (df_master['volume'] / df_master['volume'].rolling(20).mean() - 1).fillna(0)
df_master['price_acceleration'] = df_master['close'].diff(2).fillna(0)

if 'sma_20' in df_master.columns:
    df_master['price_vs_ma20'] = ((df_master['close'] - df_master['sma_20']) / df_master['sma_20']).fillna(0)
else:
    df_master['price_vs_ma20'] = 0

if 'sma_50' not in df_master.columns:
    df_master['sma_50'] = df_master['close'].rolling(50).mean()
df_master['price_vs_ma50'] = ((df_master['close'] - df_master['sma_50']) / df_master['sma_50']).fillna(0)

df_master['volatility_20'] = df_master['close'].pct_change().rolling(20).std().fillna(0)

if 'rsi' in df_master.columns:
    df_master['rsi_slope'] = df_master['rsi'].diff(5).fillna(0)
    df_master['rsi_overbought'] = (df_master['rsi'] > 70).astype(int)
    df_master['rsi_oversold'] = (df_master['rsi'] < 30).astype(int)
    df_master['rsi_divergence'] = (df_master['rsi'].diff() * df_master['close'].pct_change() < 0).astype(int)

if 'macd' in df_master.columns and 'macd_signal' in df_master.columns:
    df_master['macd_histogram_slope'] = (df_master['macd'] - df_master['macd_signal']).diff(3).fillna(0)
    df_master['macd_crossover'] = ((df_master['macd'] > df_master['macd_signal']) & (df_master['macd'].shift(1) <= df_master['macd_signal'].shift(1))).astype(int)
    df_master['macd_crossunder'] = ((df_master['macd'] < df_master['macd_signal']) & (df_master['macd'].shift(1) >= df_master['macd_signal'].shift(1))).astype(int)

if 'bb_high' in df_master.columns and 'bb_low' in df_master.columns:
    bb_range = df_master['bb_high'] - df_master['bb_low']
    bb_range = bb_range.replace(0, 1e-10)
    df_master['bb_position'] = ((df_master['close'] - df_master['bb_low']) / bb_range).fillna(0.5)

df_master['higher_high'] = (df_master['high'] > df_master['high'].shift(1)).astype(int)

support_level = df_master['low'].rolling(20).min()
df_master['near_support'] = (df_master['close'] <= support_level * 1.01).astype(int)

df_master['timestamp'] = pd.to_datetime(df_master['timestamp'])
df_master = df_master.sort_values('timestamp').reset_index(drop=True)
print(f"✓ Loaded {len(df_master):,} candles (with enhanced exit features)")
print()

# Define 5-day windows (1440 candles per window)
window_size = 1440
num_windows = len(df_master) // window_size
print(f"Testing {num_windows} windows (5-day periods)")
print()

def backtest_threshold(exit_threshold_long, exit_threshold_short):
    """Run backtest with specific Exit thresholds"""

    all_trades = []

    for window_idx in range(num_windows):
        start_idx = window_idx * window_size
        end_idx = start_idx + window_size
        df_window = df_master.iloc[start_idx:end_idx].copy()

        balance = 10000
        position = None

        for i in range(len(df_window)):
            current_candle = df_window.iloc[i]

            # Check exit if position open
            if position is not None:
                hold_time = i - position['entry_idx']
                current_price = current_candle['close']

                # Calculate P&L
                if position['side'] == 'LONG':
                    pnl_pct = (current_price - position['entry_price']) / position['entry_price']
                else:
                    pnl_pct = (position['entry_price'] - current_price) / position['entry_price']

                leveraged_pnl_pct = pnl_pct * LEVERAGE

                # Emergency Stop Loss (balance-based)
                if leveraged_pnl_pct <= -EMERGENCY_STOP_LOSS:
                    pnl_amount = balance * leveraged_pnl_pct
                    balance += pnl_amount

                    all_trades.append({
                        'side': position['side'],
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'pnl_pct': leveraged_pnl_pct * 100,
                        'pnl_amount': pnl_amount,
                        'hold_time': hold_time,
                        'exit_reason': 'stop_loss'
                    })
                    position = None
                    continue

                # Emergency Max Hold
                if hold_time >= EMERGENCY_MAX_HOLD_TIME:
                    pnl_amount = balance * leveraged_pnl_pct
                    balance += pnl_amount

                    all_trades.append({
                        'side': position['side'],
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'pnl_pct': leveraged_pnl_pct * 100,
                        'pnl_amount': pnl_amount,
                        'hold_time': hold_time,
                        'exit_reason': 'max_hold'
                    })
                    position = None
                    continue

                # ML Exit
                exit_features = current_candle[long_exit_features if position['side'] == 'LONG' else short_exit_features].values.reshape(1, -1)
                exit_model = long_exit_model if position['side'] == 'LONG' else short_exit_model
                exit_threshold = exit_threshold_long if position['side'] == 'LONG' else exit_threshold_short

                exit_prob = exit_model.predict_proba(exit_features)[0, 1]

                if exit_prob >= exit_threshold:
                    pnl_amount = balance * leveraged_pnl_pct
                    balance += pnl_amount

                    all_trades.append({
                        'side': position['side'],
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'pnl_pct': leveraged_pnl_pct * 100,
                        'pnl_amount': pnl_amount,
                        'hold_time': hold_time,
                        'exit_reason': 'ml_exit'
                    })
                    position = None
                    continue

            # Entry signals (only if no position)
            if position is None:
                # LONG Entry
                long_feat = current_candle[long_entry_features].values.reshape(1, -1)
                long_feat_scaled = long_entry_scaler.transform(long_feat)
                long_prob = long_entry_model.predict_proba(long_feat_scaled)[0, 1]

                # SHORT Entry
                short_feat = current_candle[short_entry_features].values.reshape(1, -1)
                short_feat_scaled = short_entry_scaler.transform(short_feat)
                short_prob = short_entry_model.predict_proba(short_feat_scaled)[0, 1]

                # Opportunity Gating
                long_ev = long_prob * 0.0041
                short_ev = short_prob * 0.0047

                if long_prob >= LONG_ENTRY_THRESHOLD:
                    position = {
                        'side': 'LONG',
                        'entry_price': current_candle['close'],
                        'entry_idx': i
                    }
                elif short_prob >= SHORT_ENTRY_THRESHOLD and (short_ev - long_ev) > OPPORTUNITY_GATE_THRESHOLD:
                    position = {
                        'side': 'SHORT',
                        'entry_price': current_candle['close'],
                        'entry_idx': i
                    }

    # Calculate metrics
    if not all_trades:
        return None

    df_trades = pd.DataFrame(all_trades)

    wins = df_trades[df_trades['pnl_amount'] > 0]
    losses = df_trades[df_trades['pnl_amount'] <= 0]

    win_rate = len(wins) / len(df_trades) * 100
    avg_return = df_trades['pnl_pct'].mean()

    ml_exits = len(df_trades[df_trades['exit_reason'] == 'ml_exit'])
    ml_exit_pct = ml_exits / len(df_trades) * 100

    avg_hold = df_trades['hold_time'].mean()

    long_trades = len(df_trades[df_trades['side'] == 'LONG'])
    short_trades = len(df_trades[df_trades['side'] == 'SHORT'])

    return {
        'total_trades': len(df_trades),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': win_rate,
        'avg_return': avg_return,
        'ml_exit_pct': ml_exit_pct,
        'avg_hold': avg_hold,
        'long_trades': long_trades,
        'short_trades': short_trades
    }

# Test all fine-tuned thresholds
results = []

for threshold in EXIT_THRESHOLDS:
    print(f"Testing Exit threshold {threshold:.2f}...", end=" ", flush=True)

    metrics = backtest_threshold(threshold, threshold)

    if metrics is None:
        print("❌ No trades")
        continue

    # Composite scoring (same as before)
    wr_score = np.clip(metrics['win_rate'] / 72.5, 0, 1.2)
    ml_score = np.clip(metrics['ml_exit_pct'] / 80, 0, 1.2)
    hold_score = 1 - np.abs(metrics['avg_hold'] - 25) / 100

    composite_score = wr_score * 0.3 + ml_score * 0.4 + hold_score * 0.3

    result = {
        'exit_threshold': threshold,
        **metrics,
        'wr_score': wr_score,
        'ml_score': ml_score,
        'hold_score': hold_score,
        'composite_score': composite_score
    }

    results.append(result)

    print(f"✓ WR {metrics['win_rate']:.1f}% | ML Exit {metrics['ml_exit_pct']:.1f}% | Hold {metrics['avg_hold']:.1f} | Score {composite_score:.3f}")

print()
print("=" * 80)
print("Fine-Tuned Results Summary")
print("=" * 80)

df_results = pd.DataFrame(results)
df_results = df_results.sort_values('composite_score', ascending=False).reset_index(drop=True)

print()
print("Rank  Exit   WR      Avg Ret   ML Exit   Hold    Score")
print("─" * 60)
for idx, row in df_results.iterrows():
    rank = idx + 1
    marker = "⭐" if rank == 1 else "  "
    print(f"{rank:2d}    {row['exit_threshold']:.2f}   {row['win_rate']:5.1f}%  {row['avg_return']:+7.2f}%  {row['ml_exit_pct']:5.1f}%   {row['avg_hold']:5.1f}   {row['composite_score']:.3f} {marker}")

# Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = RESULTS_DIR / f"fine_tuned_exit_threshold_{timestamp}.csv"
df_results.to_csv(output_file, index=False)

print()
print(f"✓ Results saved: {output_file.name}")
print()

# Show best configuration
best = df_results.iloc[0]
print("=" * 80)
print("✅ OPTIMAL CONFIGURATION (Fine-Tuned)")
print("=" * 80)
print(f"Exit Threshold: {best['exit_threshold']:.2f}")
print()
print(f"Win Rate: {best['win_rate']:.1f}% (target: 70-75%)")
print(f"Avg Return: {best['avg_return']:+.2f}% per window (target: 35-40%)")
print(f"ML Exit: {best['ml_exit_pct']:.1f}% (target: 75-85%)")
print(f"Avg Hold: {best['avg_hold']:.1f} candles (target: 20-30)")
print()
print(f"Composite Score: {best['composite_score']:.3f}")
print()
print(f"Trades: {best['total_trades']} ({best['wins']}W / {best['losses']}L)")
print(f"LONG/SHORT: {best['long_trades']}/{best['short_trades']}")
print("=" * 80)
