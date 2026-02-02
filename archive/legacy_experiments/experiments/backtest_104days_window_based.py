"""
EXIT Threshold Comparison - 104 Days Full Backtest
===================================================
Using CURRENT production settings:
- Entry: Enhanced 5-Fold CV (20251024_012445)
- Exit: oppgating_improved (20251024_044510)
- Data: 104 days (Jul 14 - Oct 26, 2025)
- Full logic: Entry signals + Opportunity gating + Exit thresholds
- Compare: EXIT 0.15, 0.20, 0.75
"""

import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
from datetime import datetime

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "features"
RESULTS_DIR = BASE_DIR / "results"

def prepare_exit_features(df):
    """Add EXIT-specific features required by oppgating_improved models"""
    print("\nPreparing EXIT features...")

    # Volume features
    df['volume_surge'] = (df['volume'] > df['volume'].rolling(20).mean() * 1.5).astype(float)

    # Price momentum features
    if 'sma_20' in df.columns:
        df['price_vs_ma20'] = (df['close'] - df['sma_20']) / df['sma_20']
    else:
        ma_20 = df['close'].rolling(20).mean()
        df['price_vs_ma20'] = (df['close'] - ma_20) / ma_20

    if 'sma_50' in df.columns:
        df['price_vs_ma50'] = (df['close'] - df['sma_50']) / df['sma_50']
    else:
        ma_50 = df['close'].rolling(50).mean()
        df['price_vs_ma50'] = (df['close'] - ma_50) / ma_50

    # Volatility features
    df['returns'] = df['close'].pct_change()
    df['volatility_20'] = df['returns'].rolling(20).std()

    # RSI dynamics
    df['rsi_slope'] = df['rsi'].diff(3) / 3
    df['rsi_overbought'] = (df['rsi'] > 70).astype(float)
    df['rsi_oversold'] = (df['rsi'] < 30).astype(float)
    df['rsi_divergence'] = 0

    # MACD dynamics
    if 'macd_diff' in df.columns:
        df['macd_histogram_slope'] = df['macd_diff'].diff(3) / 3
    else:
        df['macd_histogram_slope'] = 0
    df['macd_crossover'] = ((df['macd'] > df['macd_signal']) &
                            (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(float)
    df['macd_crossunder'] = ((df['macd'] < df['macd_signal']) &
                             (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(float)

    # Price patterns
    df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(float)
    df['price_acceleration'] = df['close'].diff().diff()

    # Support/Resistance proximity
    df['near_support'] = 0

    # Bollinger Band position
    if 'bb_high' in df.columns and 'bb_low' in df.columns:
        df['bb_position'] = (df['close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'])
    else:
        df['bb_position'] = 0.5

    # Clean NaN
    df = df.ffill().bfill()

    print(f"✓ EXIT features prepared")
    return df

print("=" * 80)
print("EXIT THRESHOLD COMPARISON - 104 DAYS FULL BACKTEST")
print("=" * 80)
print()

# Load data
print("Loading market data...")
df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_features.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
print(f"✓ Data loaded: {len(df):,} candles")
print(f"  Period: {df['timestamp'].min()} to {df['timestamp'].max()}")

# Prepare EXIT features (ADD missing 15 features)
df = prepare_exit_features(df)
print()

# Load CURRENT production models
print("Loading CURRENT production models...")

# Entry models
with open(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445.pkl", 'rb') as f:
    long_entry_model = pickle.load(f)
with open(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_scaler.pkl", 'rb') as f:
    long_entry_scaler = joblib.load(f)
with open(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_features.txt", 'r') as f:
    long_entry_features = [line.strip() for line in f.readlines() if line.strip()]

with open(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445.pkl", 'rb') as f:
    short_entry_model = pickle.load(f)
with open(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_scaler.pkl", 'rb') as f:
    short_entry_scaler = joblib.load(f)
with open(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_features.txt", 'r') as f:
    short_entry_features = [line.strip() for line in f.readlines() if line.strip()]

# Exit models
with open(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527.pkl", 'rb') as f:
    long_exit_model = pickle.load(f)
with open(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527_scaler.pkl", 'rb') as f:
    long_exit_scaler = joblib.load(f)
with open(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527_features.txt", 'r') as f:
    long_exit_features = [line.strip() for line in f.readlines() if line.strip()]

with open(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510.pkl", 'rb') as f:
    short_exit_model = pickle.load(f)
with open(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510_scaler.pkl", 'rb') as f:
    short_exit_scaler = joblib.load(f)
with open(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510_features.txt", 'r') as f:
    short_exit_features = [line.strip() for line in f.readlines() if line.strip()]

print(f"✓ LONG Entry: {len(long_entry_features)} features")
print(f"✓ SHORT Entry: {len(short_entry_features)} features")
print(f"✓ LONG Exit: {len(long_exit_features)} features")
print(f"✓ SHORT Exit: {len(short_exit_features)} features")
print()

# Backtest parameters (production settings)
LEVERAGE = 4
INITIAL_CAPITAL = 10000
EMERGENCY_MAX_HOLD = 120  # 10 hours
LONG_THRESHOLD = 0.65
SHORT_THRESHOLD = 0.70
GATE_THRESHOLD = 0.001  # Opportunity gating
THRESHOLDS_TO_TEST = [0.15, 0.20, 0.75]

def simulate_trade(df, entry_idx, side, exit_threshold):
    """Simulate a single trade with given exit threshold"""
    entry_price = df.iloc[entry_idx]['close']

    # Select models
    if side == 'LONG':
        exit_model = long_exit_model
        exit_scaler = long_exit_scaler
        exit_feats = long_exit_features
    else:  # SHORT
        exit_model = short_exit_model
        exit_scaler = short_exit_scaler
        exit_feats = short_exit_features

    # Simulate holding position
    for hold_time in range(1, min(len(df) - entry_idx, EMERGENCY_MAX_HOLD + 1)):
        current_idx = entry_idx + hold_time
        if current_idx >= len(df):
            break

        current_candle = df.iloc[current_idx]
        current_price = current_candle['close']

        # Calculate P&L
        if side == 'LONG':
            pnl_pct = (current_price - entry_price) / entry_price
        else:  # SHORT
            pnl_pct = (entry_price - current_price) / entry_price

        leveraged_pnl = pnl_pct * LEVERAGE

        # Check ML Exit
        try:
            exit_feat = current_candle[exit_feats].values.reshape(1, -1)
            exit_feat_scaled = exit_scaler.transform(exit_feat)
            exit_prob = exit_model.predict_proba(exit_feat_scaled)[0, 1]

            if exit_prob >= exit_threshold:
                return {
                    'exit_reason': 'ml_exit',
                    'hold_time': hold_time,
                    'pnl': leveraged_pnl,
                    'exit_prob': exit_prob
                }
        except:
            pass

        # Emergency Max Hold
        if hold_time >= EMERGENCY_MAX_HOLD:
            return {
                'exit_reason': 'max_hold',
                'hold_time': hold_time,
                'pnl': leveraged_pnl,
                'exit_prob': None
            }

    # End of data
    return {
        'exit_reason': 'end_of_data',
        'hold_time': len(df) - entry_idx - 1,
        'pnl': leveraged_pnl if 'leveraged_pnl' in locals() else 0,
        'exit_prob': None
    }

def backtest_exit_threshold(df, exit_threshold):
    """Backtest with given exit threshold using production Entry logic"""
    results = []
    capital = INITIAL_CAPITAL

    next_available_idx = 200  # Track when we can enter next trade

    for i in range(200, len(df) - EMERGENCY_MAX_HOLD - 10):  # Skip warmup period
        # Skip if still holding previous position
        if i < next_available_idx:
            continue

        current_candle = df.iloc[i]

        # Get Entry signals
        try:
            # LONG Entry
            long_feat = current_candle[long_entry_features].values.reshape(1, -1)
            long_feat_scaled = long_entry_scaler.transform(long_feat)
            long_prob = long_entry_model.predict_proba(long_feat_scaled)[0, 1]

            # SHORT Entry
            short_feat = current_candle[short_entry_features].values.reshape(1, -1)
            short_feat_scaled = short_entry_scaler.transform(short_feat)
            short_prob = short_entry_model.predict_proba(short_feat_scaled)[0, 1]

            # Entry decision with opportunity gating
            entered = False

            if long_prob >= LONG_THRESHOLD:
                # LONG entry (no gating)
                trade_result = simulate_trade(df, i, 'LONG', exit_threshold)
                pnl_amount = capital * trade_result['pnl']
                capital += pnl_amount

                results.append({
                    'entry_idx': i,
                    'side': 'LONG',
                    'entry_prob': long_prob,
                    'pnl': trade_result['pnl'],
                    'pnl_amount': pnl_amount,
                    'hold_time': trade_result['hold_time'],
                    'exit_reason': trade_result['exit_reason'],
                    'exit_prob': trade_result['exit_prob'],
                    'capital': capital
                })
                entered = True

            elif short_prob >= SHORT_THRESHOLD:
                # SHORT entry (with opportunity gating)
                long_ev = long_prob * 0.0041
                short_ev = short_prob * 0.0047
                opportunity_cost = short_ev - long_ev

                if opportunity_cost > GATE_THRESHOLD:
                    trade_result = simulate_trade(df, i, 'SHORT', exit_threshold)
                    pnl_amount = capital * trade_result['pnl']
                    capital += pnl_amount

                    results.append({
                        'entry_idx': i,
                        'side': 'SHORT',
                        'entry_prob': short_prob,
                        'pnl': trade_result['pnl'],
                        'pnl_amount': pnl_amount,
                        'hold_time': trade_result['hold_time'],
                        'exit_reason': trade_result['exit_reason'],
                        'exit_prob': trade_result['exit_prob'],
                        'capital': capital
                    })
                    entered = True

            if entered:
                # Update next available entry index (after this trade completes)
                next_available_idx = i + trade_result['hold_time'] + 1

        except Exception as e:
            # Missing features, skip this candle
            continue

    df_results = pd.DataFrame(results)

    if len(df_results) == 0:
        return {
            'threshold': exit_threshold,
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0,
            'total_return': 0,
            'avg_pnl': 0,
            'final_capital': INITIAL_CAPITAL,
            'ml_exits': 0,
            'max_hold_exits': 0,
            'ml_exit_rate': 0,
            'avg_hold_time': 0
        }

    # Calculate metrics
    total_trades = len(df_results)
    wins = len(df_results[df_results['pnl'] > 0])
    losses = len(df_results[df_results['pnl'] < 0])
    win_rate = wins / total_trades if total_trades > 0 else 0

    total_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
    avg_pnl = df_results['pnl'].mean()

    ml_exits = len(df_results[df_results['exit_reason'] == 'ml_exit'])
    max_hold_exits = len(df_results[df_results['exit_reason'] == 'max_hold'])
    ml_exit_rate = ml_exits / total_trades if total_trades > 0 else 0

    long_trades = len(df_results[df_results['side'] == 'LONG'])
    short_trades = len(df_results[df_results['side'] == 'SHORT'])

    return {
        'threshold': exit_threshold,
        'total_trades': total_trades,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'total_return': total_return,
        'avg_pnl': avg_pnl,
        'final_capital': capital,
        'ml_exits': ml_exits,
        'max_hold_exits': max_hold_exits,
        'ml_exit_rate': ml_exit_rate,
        'avg_hold_time': df_results['hold_time'].mean(),
        'long_trades': long_trades,
        'short_trades': short_trades
    }

# Run backtests for all thresholds
print("Running backtests for EXIT thresholds...")
print("(This may take several minutes...)")
print()

all_results = []

for threshold in THRESHOLDS_TO_TEST:
    print(f"Testing EXIT threshold: {threshold:.2f}...")
    result = backtest_exit_threshold(df, threshold)
    all_results.append(result)

    print(f"  Trades: {result['total_trades']} (LONG: {result['long_trades']}, SHORT: {result['short_trades']})")
    print(f"  Win Rate: {result['win_rate']:.1%}")
    print(f"  Total Return: {result['total_return']:+.2%}")
    print(f"  Final Capital: ${result['final_capital']:,.2f}")
    print(f"  ML Exit Rate: {result['ml_exit_rate']:.1%}")
    print(f"  Avg Hold Time: {result['avg_hold_time']:.1f} candles")
    print()

# Create comparison DataFrame
df_comparison = pd.DataFrame(all_results)

# Save results
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file = RESULTS_DIR / f"exit_comparison_104days_{timestamp}.csv"
df_comparison.to_csv(output_file, index=False)

print("=" * 80)
print("COMPARISON SUMMARY - 104 DAYS")
print("=" * 80)
print()

print(df_comparison[['threshold', 'total_trades', 'win_rate', 'total_return', 'ml_exit_rate', 'avg_hold_time']].to_string(index=False))
print()

# Find winner
if len(df_comparison) > 0 and df_comparison['total_trades'].sum() > 0:
    best_idx = df_comparison['total_return'].idxmax()
    best_threshold = df_comparison.loc[best_idx, 'threshold']

    print(f"✅ BEST THRESHOLD: {best_threshold:.2f}")
    print(f"   Return: {df_comparison.loc[best_idx, 'total_return']:+.2%}")
    print(f"   Win Rate: {df_comparison.loc[best_idx, 'win_rate']:.1%}")
    print(f"   ML Exit Rate: {df_comparison.loc[best_idx, 'ml_exit_rate']:.1%}")
    print(f"   Trades: {df_comparison.loc[best_idx, 'total_trades']}")
else:
    print("⚠️ No trades generated")

print()
print(f"Results saved: {output_file.name}")
print()
print("=" * 80)
