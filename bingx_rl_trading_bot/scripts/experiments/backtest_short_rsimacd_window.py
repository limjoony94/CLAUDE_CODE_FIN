"""
Enhanced SHORT Entry (with RSI/MACD) + LONG Entry Window Backtest

Compares:
- Old: 48.3% win rate (without RSI/MACD)
- New: ??? (with real RSI/MACD calculation)
"""

import pandas as pd
import numpy as np
import pickle
import talib
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Trading parameters
WINDOW_SIZE = 1440  # 5 days
INITIAL_CAPITAL = 10000.0
POSITION_SIZE_PCT = 0.95
LEVERAGE = 4
STOP_LOSS = 0.01
TAKE_PROFIT = 0.02
MAX_HOLDING_HOURS = 4
TRANSACTION_COST = 0.0002

LONG_ENTRY_THRESHOLD = 0.70
SHORT_ENTRY_THRESHOLD = 0.55  # Same as before

def calculate_rsi_macd_features(df):
    """Calculate RSI/MACD features (same as training)"""

    # RSI (14-period)
    df['rsi'] = talib.RSI(df['close'], timeperiod=14)
    df['rsi_slope'] = df['rsi'].diff(3)
    df['rsi_overbought'] = (df['rsi'] > 70).astype(float)
    df['rsi_oversold'] = (df['rsi'] < 30).astype(float)

    # RSI divergence
    price_change = df['close'].diff(5)
    rsi_change = df['rsi'].diff(5)
    df['rsi_divergence'] = (
        ((price_change > 0) & (rsi_change < 0)) |
        ((price_change < 0) & (rsi_change > 0))
    ).astype(float)

    # MACD (12, 26, 9)
    macd, macd_signal, macd_hist = talib.MACD(
        df['close'],
        fastperiod=12,
        slowperiod=26,
        signalperiod=9
    )

    df['macd'] = macd
    df['macd_signal'] = macd_signal
    df['macd_histogram_slope'] = pd.Series(macd_hist).diff(3)

    # MACD crossovers
    df['macd_crossover'] = (
        (macd > macd_signal) &
        (macd.shift(1) <= macd_signal.shift(1))
    ).astype(float)

    df['macd_crossunder'] = (
        (macd < macd_signal) &
        (macd.shift(1) >= macd_signal.shift(1))
    ).astype(float)

    return df

def calculate_enhanced_short_features(df):
    """Calculate other 22 SELL features"""

    # Volume features
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    df['volume_surge'] = (df['volume'] > df['volume'].rolling(20).mean() * 1.5).astype(float)

    # Price momentum
    ma_20 = df['close'].rolling(20).mean()
    ma_50 = df['close'].rolling(50).mean()
    df['price_vs_ma20'] = (df['close'] - ma_20) / ma_20
    df['price_vs_ma50'] = (df['close'] - ma_50) / ma_50
    df['price_acceleration'] = df['close'].diff().diff()

    # Volatility
    df['volatility_20'] = df['close'].pct_change().rolling(20).std()
    volatility_median = df['volatility_20'].median()
    df['volatility_regime'] = (df['volatility_20'] > volatility_median).astype(float)

    # Price patterns
    df['higher_high'] = (
        (df['high'] > df['high'].shift(1)) &
        (df['high'].shift(1) > df['high'].shift(2))
    ).astype(float)

    df['lower_low'] = (
        (df['low'] < df['low'].shift(1)) &
        (df['low'].shift(1) < df['low'].shift(2))
    ).astype(float)

    # Support/Resistance
    resistance = df['high'].rolling(50).max()
    support = df['low'].rolling(50).min()
    df['near_resistance'] = (df['close'] > resistance * 0.98).astype(float)
    df['near_support'] = (df['close'] < support * 1.02).astype(float)

    # Bollinger Bands
    bb_mid = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    bb_high = bb_mid + 2 * bb_std
    bb_low = bb_mid - 2 * bb_std
    bb_range = bb_high - bb_low
    df['bb_position'] = np.where(
        bb_range != 0,
        (df['close'] - bb_low) / bb_range,
        0.5
    )

    return df.ffill().fillna(0)

def classify_market_regime(df_window):
    """Classify market regime"""
    start_price = df_window['close'].iloc[0]
    end_price = df_window['close'].iloc[-1]
    return_pct = ((end_price / start_price) - 1) * 100
    if return_pct > 3.0:
        return "Bull"
    elif return_pct < -2.0:
        return "Bear"
    else:
        return "Sideways"

def backtest_longshort_enhanced(window_df, long_model, long_scaler, short_model, short_scaler, long_features, short_features):
    """Backtest LONG Entry + Enhanced SHORT Entry (with RSI/MACD)"""
    capital = INITIAL_CAPITAL
    position = None
    trades = []

    for i in range(len(window_df)):
        current_price = window_df['close'].iloc[i]

        # Manage existing position
        if position is not None:
            side = position['side']
            entry_idx = position['entry_idx']
            entry_price = position['entry_price']
            hours_held = (i - entry_idx) / 12

            # Calculate P&L
            if side == 'LONG':
                pnl_pct = (current_price - entry_price) / entry_price
            else:  # SHORT
                pnl_pct = (entry_price - current_price) / entry_price

            # Check exit conditions
            exit_reason = None
            if pnl_pct <= -STOP_LOSS:
                exit_reason = "Stop Loss"
            elif pnl_pct >= TAKE_PROFIT:
                exit_reason = "Take Profit"
            elif hours_held >= MAX_HOLDING_HOURS:
                exit_reason = "Max Holding"

            if exit_reason:
                quantity = position['quantity']
                pnl_usd = pnl_pct * (entry_price * quantity)

                entry_cost = entry_price * quantity * TRANSACTION_COST
                exit_cost = current_price * quantity * TRANSACTION_COST
                total_cost = entry_cost + exit_cost
                pnl_usd -= total_cost

                capital += pnl_usd

                trades.append({
                    'side': side,
                    'entry_idx': entry_idx,
                    'exit_idx': i,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'quantity': quantity,
                    'pnl_pct': pnl_pct,
                    'pnl_usd': pnl_usd,
                    'exit_reason': exit_reason,
                    'holding_hours': hours_held,
                    'probability': position['probability']
                })

                position = None

        # Look for entry
        if position is None and i < len(window_df) - 1:
            # Try LONG Entry (with scaler)
            long_feat = window_df[long_features].iloc[i:i+1].values
            if not np.isnan(long_feat).any():
                long_feat_scaled = long_scaler.transform(long_feat)
                long_prob = long_model.predict_proba(long_feat_scaled)[0][1]
            else:
                long_prob = 0.0

            # Try SHORT Entry (with scaler + RSI/MACD!)
            short_feat = window_df[short_features].iloc[i:i+1].values
            if not np.isnan(short_feat).any():
                short_feat_scaled = short_scaler.transform(short_feat)
                short_prob = short_model.predict_proba(short_feat_scaled)[0][1]
            else:
                short_prob = 0.0

            # Determine entry direction
            side = None
            signal_prob = 0.0

            if long_prob >= LONG_ENTRY_THRESHOLD:
                side = 'LONG'
                signal_prob = long_prob
            elif short_prob >= SHORT_ENTRY_THRESHOLD:
                side = 'SHORT'
                signal_prob = short_prob

            if side is not None:
                position_value = capital * POSITION_SIZE_PCT
                quantity = position_value / current_price

                position = {
                    'side': side,
                    'entry_idx': i,
                    'entry_price': current_price,
                    'quantity': quantity,
                    'probability': signal_prob
                }

    # Calculate metrics
    if len(trades) == 0:
        return trades, {
            'total_return_pct': 0.0,
            'num_trades': 0,
            'num_long': 0,
            'num_short': 0,
            'win_rate': 0.0,
            'win_rate_long': 0.0,
            'win_rate_short': 0.0
        }

    total_return_pct = ((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100

    # Overall metrics
    winning_trades = [t for t in trades if t['pnl_usd'] > 0]
    win_rate = (len(winning_trades) / len(trades)) * 100

    # LONG vs SHORT breakdown
    long_trades = [t for t in trades if t['side'] == 'LONG']
    short_trades = [t for t in trades if t['side'] == 'SHORT']

    if len(long_trades) > 0:
        long_wins = [t for t in long_trades if t['pnl_usd'] > 0]
        win_rate_long = (len(long_wins) / len(long_trades)) * 100
    else:
        win_rate_long = 0.0

    if len(short_trades) > 0:
        short_wins = [t for t in short_trades if t['pnl_usd'] > 0]
        win_rate_short = (len(short_wins) / len(short_trades)) * 100
    else:
        win_rate_short = 0.0

    metrics = {
        'total_return_pct': total_return_pct,
        'num_trades': len(trades),
        'num_long': len(long_trades),
        'num_short': len(short_trades),
        'win_rate': win_rate,
        'win_rate_long': win_rate_long,
        'win_rate_short': win_rate_short
    }

    return trades, metrics

def rolling_window_backtest(df, long_model, long_scaler, short_model, short_scaler, long_features, short_features):
    """Rolling window backtest"""
    windows = []
    start_idx = 0

    while start_idx + WINDOW_SIZE <= len(df):
        end_idx = start_idx + WINDOW_SIZE
        window_df = df.iloc[start_idx:end_idx].reset_index(drop=True)

        regime = classify_market_regime(window_df)

        trades, metrics = backtest_longshort_enhanced(
            window_df, long_model, long_scaler, short_model, short_scaler, long_features, short_features
        )

        # Buy & Hold
        bh_start = window_df['close'].iloc[0]
        bh_end = window_df['close'].iloc[-1]
        bh_return = ((bh_end - bh_start) / bh_start) * 100
        bh_cost = 2 * TRANSACTION_COST * 100
        bh_return -= bh_cost

        windows.append({
            'start_idx': start_idx,
            'end_idx': end_idx,
            'regime': regime,
            'return': metrics['total_return_pct'],
            'bh_return': bh_return,
            'difference': metrics['total_return_pct'] - bh_return,
            'num_trades': metrics['num_trades'],
            'num_long': metrics['num_long'],
            'num_short': metrics['num_short'],
            'win_rate': metrics['win_rate'],
            'long_win_rate': metrics['win_rate_long'],
            'short_win_rate': metrics['win_rate_short']
        })

        start_idx += WINDOW_SIZE

    return pd.DataFrame(windows)

print("="*80)
print("Enhanced SHORT Entry (RSI/MACD) + LONG Entry - Window Backtest")
print("="*80)
print("\n🎯 Testing: NEW model with real RSI/MACD calculation")
print("  - OLD: 48.3% SHORT win rate (defaults)")
print("  - NEW: ??? SHORT win rate (real RSI/MACD)")
print()

# Load LONG Entry model
print("Loading LONG Entry model...")
long_model_path = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl"
with open(long_model_path, 'rb') as f:
    long_model = pickle.load(f)

long_scaler_path = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_scaler.pkl"
with open(long_scaler_path, 'rb') as f:
    long_scaler = pickle.load(f)

long_features_path = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt"
with open(long_features_path, 'r') as f:
    long_feature_columns = [line.strip() for line in f.readlines()]
print(f"  ✅ LONG model loaded: {len(long_feature_columns)} features + scaler")

# Load NEW Enhanced SHORT Entry model (with RSI/MACD)
print("\nLoading NEW Enhanced SHORT Entry model (RSI/MACD)...")
short_model_path = MODELS_DIR / "xgboost_short_entry_enhanced_rsimacd_20251016_223048.pkl"
with open(short_model_path, 'rb') as f:
    short_model = pickle.load(f)

short_scaler_path = MODELS_DIR / "xgboost_short_entry_enhanced_rsimacd_20251016_223048_scaler.pkl"
with open(short_scaler_path, 'rb') as f:
    short_scaler = pickle.load(f)

short_features_path = MODELS_DIR / "xgboost_short_entry_enhanced_rsimacd_20251016_223048_features.txt"
with open(short_features_path, 'r') as f:
    short_feature_columns = [line.strip() for line in f.readlines()]
print(f"  ✅ NEW SHORT model loaded: {len(short_feature_columns)} features + scaler")

# Load data
print("\nLoading data...")
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df = pd.read_csv(data_file)
print(f"  ✅ Data loaded: {len(df)} candles")

# Calculate LONG Entry features
print("\nCalculating LONG Entry features...")
df = calculate_features(df)
adv_features = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)
df = adv_features.calculate_all_features(df)

# Calculate RSI/MACD (CRITICAL for NEW model!)
print("Calculating RSI/MACD features...")
df = calculate_rsi_macd_features(df)

# Calculate SHORT Entry features
print("Calculating SHORT Entry features (22 SELL signals)...")
df = calculate_enhanced_short_features(df)

# Handle NaN
df = df.ffill()
df = df.dropna()
print(f"  ✅ Features calculated: {len(df)} rows after dropna")

# Run window backtest
print(f"\n{'='*80}")
print(f"Running Window Backtest (NEW MODEL)")
print(f"{'='*80}")

results = rolling_window_backtest(df, long_model, long_scaler, short_model, short_scaler, long_feature_columns, short_feature_columns)

# Summary
print(f"\nNEW Model Results ({len(results)} windows):")
print(f"  Avg Return: {results['return'].mean():.2f}% per window")
print(f"  Total Return: {results['return'].sum():.2f}%")
print(f"  Win Rate: {(results['return'] > 0).sum() / len(results) * 100:.1f}%")
print(f"  Buy & Hold: {results['bh_return'].mean():.2f}% per window")
print(f"  vs B&H: {results['difference'].mean():.2f}%")
print(f"\n  📊 Trade Breakdown:")
print(f"    Avg Trades: {results['num_trades'].mean():.1f} per window")
print(f"    Avg LONG: {results['num_long'].mean():.1f} per window")
print(f"    Avg SHORT: {results['num_short'].mean():.1f} per window")
print(f"\n  🎯 Win Rates:")
print(f"    Overall: {results['win_rate'].mean():.1f}%")
print(f"    LONG: {results['long_win_rate'].mean():.1f}%")
print(f"    SHORT: {results['short_win_rate'].mean():.1f}%")

# By regime
print(f"\n{'='*80}")
print(f"By Market Regime:")
print(f"{'='*80}")
for regime in ['Bull', 'Bear', 'Sideways']:
    regime_df = results[results['regime'] == regime]
    if len(regime_df) > 0:
        print(f"\n  {regime} ({len(regime_df)} windows):")
        print(f"    Avg Return: {regime_df['return'].mean():.2f}%")
        print(f"    LONG: {regime_df['num_long'].mean():.1f} trades, {regime_df['long_win_rate'].mean():.1f}% WR")
        print(f"    SHORT: {regime_df['num_short'].mean():.1f} trades, {regime_df['short_win_rate'].mean():.1f}% WR")

# Save
output_file = RESULTS_DIR / "backtest_short_rsimacd_window.csv"
results.to_csv(output_file, index=False)
print(f"\n✅ Results saved: {output_file}")

# Compare with OLD model
print(f"\n{'='*80}")
print(f"COMPARISON: NEW (RSI/MACD) vs OLD (defaults)")
print(f"{'='*80}")

old_file = RESULTS_DIR / "backtest_enhanced_short_window.csv"
if old_file.exists():
    old_results = pd.read_csv(old_file)

    print(f"\nOLD Model (without RSI/MACD):")
    print(f"  Avg Return: {old_results['return'].mean():.2f}% per window")
    print(f"  Total Return: {old_results['return'].sum():.2f}%")
    print(f"  Win Rate: {(old_results['return'] > 0).sum() / len(old_results) * 100:.1f}%")
    print(f"  SHORT Win Rate: {old_results['short_win_rate'].mean():.1f}%")
    print(f"  Avg SHORT trades: {old_results['num_short'].mean():.1f} per window")

    print(f"\nNEW Model (with real RSI/MACD):")
    print(f"  Avg Return: {results['return'].mean():.2f}% per window")
    print(f"  Total Return: {results['return'].sum():.2f}%")
    print(f"  Win Rate: {(results['return'] > 0).sum() / len(results) * 100:.1f}%")
    print(f"  SHORT Win Rate: {results['short_win_rate'].mean():.1f}%")
    print(f"  Avg SHORT trades: {results['num_short'].mean():.1f} per window")

    print(f"\nImprovement:")
    return_diff = results['return'].mean() - old_results['return'].mean()
    short_wr_diff = results['short_win_rate'].mean() - old_results['short_win_rate'].mean()

    print(f"  Return: {return_diff:+.2f}% per window")
    print(f"  SHORT Win Rate: {short_wr_diff:+.1f}%")

    # Decision
    print(f"\n{'='*80}")
    print(f"DECISION")
    print(f"{'='*80}")

    if short_wr_diff > 5:
        print(f"\n  ✅ SIGNIFICANT IMPROVEMENT!")
        print(f"     SHORT Win Rate: {old_results['short_win_rate'].mean():.1f}% → {results['short_win_rate'].mean():.1f}% ({short_wr_diff:+.1f}%)")
        print(f"\n  🎯 RECOMMENDATION: DEPLOY NEW MODEL")
    elif short_wr_diff > 0:
        print(f"\n  ⚠️  MINOR IMPROVEMENT")
        print(f"     SHORT Win Rate: {old_results['short_win_rate'].mean():.1f}% → {results['short_win_rate'].mean():.1f}% ({short_wr_diff:+.1f}%)")
        print(f"\n  🎯 RECOMMENDATION: Test 1 week, then decide")
    else:
        print(f"\n  ❌ NO IMPROVEMENT")
        print(f"     SHORT Win Rate: {old_results['short_win_rate'].mean():.1f}% → {results['short_win_rate'].mean():.1f}% ({short_wr_diff:+.1f}%)")
        print(f"\n  🎯 RECOMMENDATION: Need further improvement (stricter labeling?)")
else:
    print("\n  ⚠️  OLD results not found. Cannot compare.")

print(f"\n{'='*80}")
print("Backtest Complete!")
print(f"{'='*80}\n")
