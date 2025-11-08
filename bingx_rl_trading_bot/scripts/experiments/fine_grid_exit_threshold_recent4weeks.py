"""
Fine Grid Exit Threshold Search - Recent 4 Weeks
=================================================
User Request: "추가 실험" (Additional experiments)
Purpose: Find optimal balance between ML Exit usage and profitability
Strategy: Test intermediate thresholds [0.25, 0.30, ..., 0.70] on recent 4-week data
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

# Fine grid: Exit thresholds from 0.25 to 0.70 (every 0.05)
EXIT_THRESHOLDS = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

# Configuration (match actual production)
LONG_ENTRY_THRESHOLD = 0.80
SHORT_ENTRY_THRESHOLD = 0.80
EMERGENCY_STOP_LOSS = 0.03
EMERGENCY_MAX_HOLD_TIME = 120
LEVERAGE = 4
OPPORTUNITY_GATE_THRESHOLD = 0.001

print("=" * 80)
print("FINE GRID EXIT THRESHOLD SEARCH - RECENT 4 WEEKS")
print("=" * 80)
print(f"Testing {len(EXIT_THRESHOLDS)} intermediate thresholds")
print(f"Range: 0.25 to 0.70 (every 0.05)")
print(f"Goal: Find optimal balance between ML Exit and profitability")
print()

# Load models
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

print(f"✓ Models loaded")
print()

# Load recent 4 weeks data
print("Loading dataset...")
df_master = pd.read_csv(DATA_DIR / "features" / "BTCUSDT_5m_features.csv")
df_master['timestamp'] = pd.to_datetime(df_master['timestamp'])

cutoff_date = df_master['timestamp'].max() - timedelta(days=28)
df_recent = df_master[df_master['timestamp'] >= cutoff_date].copy().reset_index(drop=True)

print(f"✓ Recent 4 weeks: {len(df_recent):,} candles")
print()

# Add enhanced exit features
print("Adding enhanced exit features...")
df_recent['volume_surge'] = (df_recent['volume'] / df_recent['volume'].rolling(20).mean() - 1).fillna(0)
df_recent['price_acceleration'] = df_recent['close'].diff(2).fillna(0)

if 'sma_20' in df_recent.columns:
    df_recent['price_vs_ma20'] = ((df_recent['close'] - df_recent['sma_20']) / df_recent['sma_20']).fillna(0)
else:
    df_recent['price_vs_ma20'] = 0

if 'sma_50' not in df_recent.columns:
    df_recent['sma_50'] = df_recent['close'].rolling(50).mean()
df_recent['price_vs_ma50'] = ((df_recent['close'] - df_recent['sma_50']) / df_recent['sma_50']).fillna(0)

df_recent['volatility_20'] = df_recent['close'].pct_change().rolling(20).std().fillna(0)

if 'rsi' in df_recent.columns:
    df_recent['rsi_slope'] = df_recent['rsi'].diff(5).fillna(0)
    df_recent['rsi_overbought'] = (df_recent['rsi'] > 70).astype(int)
    df_recent['rsi_oversold'] = (df_recent['rsi'] < 30).astype(int)
    df_recent['rsi_divergence'] = (df_recent['rsi'].diff() * df_recent['close'].pct_change() < 0).astype(int)

if 'macd' in df_recent.columns and 'macd_signal' in df_recent.columns:
    df_recent['macd_histogram_slope'] = (df_recent['macd'] - df_recent['macd_signal']).diff(3).fillna(0)
    df_recent['macd_crossover'] = ((df_recent['macd'] > df_recent['macd_signal']) & (df_recent['macd'].shift(1) <= df_recent['macd_signal'].shift(1))).astype(int)
    df_recent['macd_crossunder'] = ((df_recent['macd'] < df_recent['macd_signal']) & (df_recent['macd'].shift(1) >= df_recent['macd_signal'].shift(1))).astype(int)

if 'bb_high' in df_recent.columns and 'bb_low' in df_recent.columns:
    bb_range = df_recent['bb_high'] - df_recent['bb_low']
    bb_range = bb_range.replace(0, 1e-10)
    df_recent['bb_position'] = ((df_recent['close'] - df_recent['bb_low']) / bb_range).fillna(0.5)

df_recent['higher_high'] = (df_recent['high'] > df_recent['high'].shift(1)).astype(int)
support_level = df_recent['low'].rolling(20).min()
df_recent['near_support'] = (df_recent['close'] <= support_level * 1.01).astype(int)

print("✓ Enhanced features added")
print()

window_size = 1440
num_windows = len(df_recent) // window_size
print(f"Testing {num_windows} windows (5-day periods)")
print()

def backtest_threshold(exit_threshold_long, exit_threshold_short):
    """Run backtest with specific Exit thresholds"""

    all_trades = []

    for window_idx in range(num_windows):
        start_idx = window_idx * window_size
        end_idx = start_idx + window_size
        df_window = df_recent.iloc[start_idx:end_idx].copy()

        balance = 10000
        position = None

        for i in range(len(df_window)):
            current_candle = df_window.iloc[i]

            if position is not None:
                hold_time = i - position['entry_idx']
                current_price = current_candle['close']

                if position['side'] == 'LONG':
                    pnl_pct = (current_price - position['entry_price']) / position['entry_price']
                else:
                    pnl_pct = (position['entry_price'] - current_price) / position['entry_price']

                leveraged_pnl_pct = pnl_pct * LEVERAGE

                if leveraged_pnl_pct <= -EMERGENCY_STOP_LOSS:
                    pnl_amount = balance * leveraged_pnl_pct
                    balance += pnl_amount
                    all_trades.append({
                        'side': position['side'],
                        'pnl_pct': leveraged_pnl_pct * 100,
                        'pnl_amount': pnl_amount,
                        'hold_time': hold_time,
                        'exit_reason': 'stop_loss'
                    })
                    position = None
                    continue

                if hold_time >= EMERGENCY_MAX_HOLD_TIME:
                    pnl_amount = balance * leveraged_pnl_pct
                    balance += pnl_amount
                    all_trades.append({
                        'side': position['side'],
                        'pnl_pct': leveraged_pnl_pct * 100,
                        'pnl_amount': pnl_amount,
                        'hold_time': hold_time,
                        'exit_reason': 'max_hold'
                    })
                    position = None
                    continue

                exit_features = current_candle[long_exit_features if position['side'] == 'LONG' else short_exit_features].values.reshape(1, -1)
                exit_model = long_exit_model if position['side'] == 'LONG' else short_exit_model
                exit_threshold = exit_threshold_long if position['side'] == 'LONG' else exit_threshold_short

                exit_prob = exit_model.predict_proba(exit_features)[0, 1]

                if exit_prob >= exit_threshold:
                    pnl_amount = balance * leveraged_pnl_pct
                    balance += pnl_amount
                    all_trades.append({
                        'side': position['side'],
                        'pnl_pct': leveraged_pnl_pct * 100,
                        'pnl_amount': pnl_amount,
                        'hold_time': hold_time,
                        'exit_reason': 'ml_exit'
                    })
                    position = None
                    continue

            if position is None:
                long_feat = current_candle[long_entry_features].values.reshape(1, -1)
                long_feat_scaled = long_entry_scaler.transform(long_feat)
                long_prob = long_entry_model.predict_proba(long_feat_scaled)[0, 1]

                short_feat = current_candle[short_entry_features].values.reshape(1, -1)
                short_feat_scaled = short_entry_scaler.transform(short_feat)
                short_prob = short_entry_model.predict_proba(short_feat_scaled)[0, 1]

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

    if not all_trades:
        return None

    df_trades = pd.DataFrame(all_trades)

    wins = df_trades[df_trades['pnl_amount'] > 0]
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
        'losses': len(df_trades) - len(wins),
        'win_rate': win_rate,
        'avg_return': avg_return,
        'ml_exit_pct': ml_exit_pct,
        'avg_hold': avg_hold,
        'long_trades': long_trades,
        'short_trades': short_trades
    }

# Test all thresholds
results = []

for threshold in EXIT_THRESHOLDS:
    print(f"Testing Exit {threshold:.2f}...", end=" ", flush=True)

    metrics = backtest_threshold(threshold, threshold)

    if metrics is None:
        print("❌ No trades")
        continue

    result = {
        'exit_threshold': threshold,
        **metrics
    }

    results.append(result)

    print(f"✓ WR {metrics['win_rate']:.1f}% | Return {metrics['avg_return']:+.2f}% | ML Exit {metrics['ml_exit_pct']:.1f}% | Hold {metrics['avg_hold']:.1f}")

print()
print("=" * 80)
print("FINE GRID RESULTS - RECENT 4 WEEKS")
print("=" * 80)

df_results = pd.DataFrame(results)
df_results = df_results.sort_values('exit_threshold').reset_index(drop=True)

# Calculate composite score (balanced)
df_results['ml_score'] = np.clip(df_results['ml_exit_pct'] / 60, 0, 1.2)  # Target 60%
df_results['return_score'] = np.clip(df_results['avg_return'] / 3.0, 0, 1.2)  # Target 3%
df_results['wr_score'] = np.clip(df_results['win_rate'] / 75, 0, 1.2)  # Target 75%
df_results['composite_score'] = (
    df_results['ml_score'] * 0.40 +      # ML Exit 40% weight
    df_results['return_score'] * 0.35 +  # Return 35% weight
    df_results['wr_score'] * 0.25        # Win Rate 25% weight
)

df_results = df_results.sort_values('composite_score', ascending=False).reset_index(drop=True)

print()
print("Rank  Exit   Trades  WR      Return   ML Exit   Hold    Score")
print("─" * 75)
for idx, row in df_results.iterrows():
    rank = idx + 1
    marker = "⭐" if rank == 1 else ("🥈" if rank == 2 else ("🥉" if rank == 3 else "  "))
    print(f"{rank:2d}    {row['exit_threshold']:.2f}   {int(row['total_trades']):4d}   {row['win_rate']:5.1f}%  {row['avg_return']:+7.2f}%  {row['ml_exit_pct']:5.1f}%   {row['avg_hold']:5.1f}   {row['composite_score']:.3f} {marker}")

# Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = RESULTS_DIR / f"fine_grid_exit_recent4weeks_{timestamp}.csv"
df_results.to_csv(output_file, index=False)

print()
print(f"✓ Results saved: {output_file.name}")
print()

# Best threshold analysis
best = df_results.iloc[0]
print("=" * 80)
print("⭐ OPTIMAL THRESHOLD (Balanced Score)")
print("=" * 80)
print(f"Exit Threshold: {best['exit_threshold']:.2f}")
print()
print(f"Win Rate: {best['win_rate']:.1f}%")
print(f"Return: {best['avg_return']:+.2f}% per 5-day window")
print(f"ML Exit: {best['ml_exit_pct']:.1f}% ← {'✅ WORKING' if best['ml_exit_pct'] > 30 else '❌ NOT WORKING'}")
print(f"Avg Hold: {best['avg_hold']:.1f} candles ({best['avg_hold']/12:.1f} hours)")
print()
print(f"Composite Score: {best['composite_score']:.3f}")
print(f"Trades: {int(best['total_trades'])} ({int(best['wins'])}W / {int(best['losses'])}L)")
print(f"LONG/SHORT: {int(best['long_trades'])}/{int(best['short_trades'])}")
print()

# Compare with Exit 0.75
print("=" * 80)
print("COMPARISON WITH CURRENT PRODUCTION (Exit 0.75)")
print("=" * 80)
print()
print("Metric                Optimal          Exit 0.75       Difference")
print("─" * 75)
print(f"Exit Threshold        {best['exit_threshold']:.2f}               0.75            -")
print(f"Win Rate              {best['win_rate']:5.1f}%           79.2%           {best['win_rate']-79.2:+.1f}pp")
print(f"Return/5-day          {best['avg_return']:+.2f}%          +3.83%          {best['avg_return']-3.83:+.2f}%")
print(f"ML Exit               {best['ml_exit_pct']:5.1f}%            0.0%           {best['ml_exit_pct']-0.0:+.1f}pp ⭐")
print(f"Avg Hold              {best['avg_hold']:5.1f}           115.5           {best['avg_hold']-115.5:+.1f}")
print()

if best['ml_exit_pct'] > 30:
    print("✅ ML Exit system is WORKING with this threshold")
    print(f"✅ ML Exit improvement: {best['ml_exit_pct']:.1f}% (from 0%)")
    if best['avg_return'] >= 3.0:
        print(f"✅ Return comparable: {best['avg_return']:+.2f}% (target: +3.00%)")
        print()
        print("🎯 RECOMMENDATION: Switch to this threshold")
        print("   - ML Exit system restored")
        print("   - Profitability maintained")
        print("   - Market adaptability improved")
    else:
        print(f"⚠️ Return trade-off: {best['avg_return']:+.2f}% vs +3.83% (-{3.83-best['avg_return']:.2f}%)")
        print()
        print("🤔 CONSIDERATION NEEDED:")
        print(f"   - ML Exit restored (+{best['ml_exit_pct']:.1f}%)")
        print(f"   - But return decreased ({best['avg_return']-3.83:+.2f}%)")
        print("   - User decision: ML system vs higher return?")
else:
    print("❌ ML Exit still not working well with this threshold")
    print("⚠️ May need more aggressive threshold (e.g., 0.15-0.25)")

print("=" * 80)
