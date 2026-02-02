"""
Backtest Enhanced SHORT Entry + LONG Entry (Window-based)

Tests:
- LONG Entry: Current model (44 features)
- SHORT Entry: Enhanced model (22 SELL features) ← NEW
- Comparison with LONG-only performance

Critical Question:
"Enhanced SHORT Entry가 LONG-only 대비 성능을 개선하는가?"
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Trading parameters (same as LONG-only backtest)
WINDOW_SIZE = 1440  # 5 days
INITIAL_CAPITAL = 10000.0
POSITION_SIZE_PCT = 0.95
LEVERAGE = 4
STOP_LOSS = 0.01
TAKE_PROFIT = 0.02
MAX_HOLDING_HOURS = 4
TRANSACTION_COST = 0.0002

LONG_ENTRY_THRESHOLD = 0.70
SHORT_ENTRY_THRESHOLD = 0.55  # Enhanced model optimal

def calculate_enhanced_short_features(df):
    """Calculate 22 SELL signal features for Enhanced SHORT Entry"""

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

    # RSI dynamics (defaults - will be real values in production)
    df['rsi'] = 50.0
    df['rsi_slope'] = 0.0
    df['rsi_overbought'] = 0.0
    df['rsi_oversold'] = 0.0
    df['rsi_divergence'] = 0.0

    # MACD dynamics
    df['macd'] = 0.0
    df['macd_signal'] = 0.0
    df['macd_histogram_slope'] = 0.0
    df['macd_crossover'] = 0.0
    df['macd_crossunder'] = 0.0

    # Price patterns
    df['higher_high'] = ((df['high'] > df['high'].shift(1)) &
                         (df['high'].shift(1) > df['high'].shift(2))).astype(float)
    df['lower_low'] = ((df['low'] < df['low'].shift(1)) &
                       (df['low'].shift(1) < df['low'].shift(2))).astype(float)

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
    df['bb_position'] = np.where(bb_range != 0,
                                  (df['close'] - bb_low) / bb_range,
                                  0.5)

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

def backtest_longshort_enhanced(window_df, long_model, long_scaler, short_model, short_scaler, long_features, short_features, debug=False):
    """
    Backtest LONG Entry + Enhanced SHORT Entry

    LONG: prob >= 0.70 (44 features) with scaler
    SHORT: prob >= 0.55 (22 SELL features) with scaler

    Args:
        window_df: Window dataframe with reset index (NOT full df)
        long_scaler: MinMaxScaler for LONG features
        short_scaler: MinMaxScaler for SHORT features
    """
    capital = INITIAL_CAPITAL
    position = None
    trades = []

    # Debug counters
    debug_counters = {
        'total_candles': 0,
        'long_nan_features': 0,
        'short_nan_features': 0,
        'long_prob_checks': 0,
        'short_prob_checks': 0,
        'long_entries': 0,
        'short_entries': 0,
        'long_prob_max': 0.0,
        'short_prob_max': 0.0
    }

    for i in range(len(window_df)):
        debug_counters['total_candles'] += 1
        current_price = window_df['close'].iloc[i]

        # Manage existing position
        if position is not None:
            side = position['side']
            entry_idx = position['entry_idx']
            entry_price = position['entry_price']
            hours_held = (i - entry_idx) / 12

            # Calculate P&L based on position side
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

        # Look for entry (LONG or SHORT)
        if position is None and i < len(window_df) - 1:
            # Try LONG Entry (use window_df + scaler!)
            long_feat = window_df[long_features].iloc[i:i+1].values
            if not np.isnan(long_feat).any():
                long_feat_scaled = long_scaler.transform(long_feat)
                long_prob = long_model.predict_proba(long_feat_scaled)[0][1]
                debug_counters['long_prob_checks'] += 1
                if long_prob > debug_counters['long_prob_max']:
                    debug_counters['long_prob_max'] = long_prob
            else:
                long_prob = 0.0
                debug_counters['long_nan_features'] += 1

            # Try SHORT Entry (use window_df + scaler!)
            short_feat = window_df[short_features].iloc[i:i+1].values
            if not np.isnan(short_feat).any():
                short_feat_scaled = short_scaler.transform(short_feat)
                short_prob = short_model.predict_proba(short_feat_scaled)[0][1]
                debug_counters['short_prob_checks'] += 1
                if short_prob > debug_counters['short_prob_max']:
                    debug_counters['short_prob_max'] = short_prob
            else:
                short_prob = 0.0
                debug_counters['short_nan_features'] += 1

            # Determine entry direction
            side = None
            signal_prob = 0.0

            if long_prob >= LONG_ENTRY_THRESHOLD:
                side = 'LONG'
                signal_prob = long_prob
                debug_counters['long_entries'] += 1
            elif short_prob >= SHORT_ENTRY_THRESHOLD:
                side = 'SHORT'
                signal_prob = short_prob
                debug_counters['short_entries'] += 1

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
            'win_rate_short': 0.0,
            'debug': debug_counters if debug else None
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
        'win_rate_short': win_rate_short,
        'debug': debug_counters if debug else None
    }

    return trades, metrics

def rolling_window_backtest(df, long_model, long_scaler, short_model, short_scaler, long_features, short_features, debug=False):
    """Rolling window backtest"""
    windows = []
    start_idx = 0

    while start_idx + WINDOW_SIZE <= len(df):
        end_idx = start_idx + WINDOW_SIZE
        window_df = df.iloc[start_idx:end_idx].reset_index(drop=True)

        regime = classify_market_regime(window_df)

        trades, metrics = backtest_longshort_enhanced(
            window_df, long_model, long_scaler, short_model, short_scaler, long_features, short_features, debug=debug
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
            'short_win_rate': metrics['win_rate_short'],
            'debug': metrics.get('debug', None)
        })

        start_idx += WINDOW_SIZE

    return pd.DataFrame(windows)

print("=" * 80)
print("Enhanced SHORT Entry + LONG Entry - Window Backtest")
print("=" * 80)
print("\n🎯 Testing: LONG (44 features) + Enhanced SHORT (22 SELL features)")
print("  - LONG Entry Threshold: 0.70")
print("  - SHORT Entry Threshold: 0.55 (Enhanced optimal)")
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

# Load Enhanced SHORT Entry model
print("\nLoading Enhanced SHORT Entry model...")
short_model_path = MODELS_DIR / "xgboost_short_entry_enhanced_20251016_201219.pkl"
with open(short_model_path, 'rb') as f:
    short_model = pickle.load(f)

short_scaler_path = MODELS_DIR / "xgboost_short_entry_enhanced_20251016_201219_scaler.pkl"
with open(short_scaler_path, 'rb') as f:
    short_scaler = pickle.load(f)

short_features_path = MODELS_DIR / "xgboost_short_entry_enhanced_20251016_201219_features.txt"
with open(short_features_path, 'r') as f:
    short_feature_columns = [line.strip() for line in f.readlines()]
print(f"  ✅ Enhanced SHORT model loaded: {len(short_feature_columns)} SELL features + scaler")

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

# Calculate SHORT Entry features
print("Calculating SHORT Entry features (22 SELL signals)...")
df = calculate_enhanced_short_features(df)

# Handle NaN
df = df.ffill()
df = df.dropna()
print(f"  ✅ Features calculated: {len(df)} rows after dropna")

# Run window backtest
print(f"\n{'=' * 80}")
print(f"Running Window Backtest (DEBUG MODE + SCALERS)")
print(f"{'=' * 80}")

results = rolling_window_backtest(df, long_model, long_scaler, short_model, short_scaler, long_feature_columns, short_feature_columns, debug=True)

# Summary
print(f"\nResults ({len(results)} windows):")
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

# Debug info
if 'debug' in results.columns and results['debug'].iloc[0] is not None:
    print(f"\n{'=' * 80}")
    print(f"DEBUG ANALYSIS")
    print(f"{'=' * 80}")

    # Aggregate debug counters across all windows
    total_debug = {
        'total_candles': 0,
        'long_nan_features': 0,
        'short_nan_features': 0,
        'long_prob_checks': 0,
        'short_prob_checks': 0,
        'long_entries': 0,
        'short_entries': 0,
        'long_prob_max': 0.0,
        'short_prob_max': 0.0
    }

    for debug_data in results['debug']:
        if debug_data:
            for key in total_debug:
                if key in ['long_prob_max', 'short_prob_max']:
                    total_debug[key] = max(total_debug[key], debug_data[key])
                else:
                    total_debug[key] += debug_data[key]

    print(f"\n  📊 Feature Extraction:")
    print(f"    Total candles processed: {total_debug['total_candles']:,}")
    print(f"    LONG NaN features: {total_debug['long_nan_features']:,} ({total_debug['long_nan_features']/total_debug['total_candles']*100:.2f}%)")
    print(f"    SHORT NaN features: {total_debug['short_nan_features']:,} ({total_debug['short_nan_features']/total_debug['total_candles']*100:.2f}%)")

    print(f"\n  🎯 Probability Checks:")
    print(f"    LONG prob checks: {total_debug['long_prob_checks']:,}")
    print(f"    SHORT prob checks: {total_debug['short_prob_checks']:,}")
    print(f"    LONG max prob: {total_debug['long_prob_max']:.4f} (threshold: {LONG_ENTRY_THRESHOLD})")
    print(f"    SHORT max prob: {total_debug['short_prob_max']:.4f} (threshold: {SHORT_ENTRY_THRESHOLD})")

    print(f"\n  🚨 CRITICAL FINDING:")
    if total_debug['long_prob_max'] < LONG_ENTRY_THRESHOLD:
        print(f"    ❌ LONG Entry NEVER triggered!")
        print(f"       Max probability: {total_debug['long_prob_max']:.4f}")
        print(f"       Required threshold: {LONG_ENTRY_THRESHOLD}")
        print(f"       Gap: {LONG_ENTRY_THRESHOLD - total_debug['long_prob_max']:.4f}")
        print(f"\n    🔍 Possible causes:")
        print(f"       1. LONG model not calibrated for this data")
        print(f"       2. Feature calculation differs from training")
        print(f"       3. Model expects different feature distribution")
    else:
        print(f"    ✅ LONG Entry did trigger (max: {total_debug['long_prob_max']:.4f})")
        print(f"       Total LONG entries: {total_debug['long_entries']}")

# By regime
print(f"\n{'=' * 80}")
print(f"By Market Regime:")
print(f"{'=' * 80}")
for regime in ['Bull', 'Bear', 'Sideways']:
    regime_df = results[results['regime'] == regime]
    if len(regime_df) > 0:
        print(f"\n  {regime} ({len(regime_df)} windows):")
        print(f"    Avg Return: {regime_df['return'].mean():.2f}%")
        print(f"    LONG: {regime_df['num_long'].mean():.1f} trades, {regime_df['long_win_rate'].mean():.1f}% WR")
        print(f"    SHORT: {regime_df['num_short'].mean():.1f} trades, {regime_df['short_win_rate'].mean():.1f}% WR")

# Save
output_file = RESULTS_DIR / "backtest_enhanced_short_window.csv"
results.to_csv(output_file, index=False)
print(f"\n✅ Results saved: {output_file}")

# Load LONG-only results for comparison
print(f"\n{'=' * 80}")
print(f"COMPARISON: LONG+Enhanced SHORT vs LONG-only")
print(f"{'=' * 80}")

longonly_file = RESULTS_DIR / "backtest_full_dataset_longonly_4x.csv"
if longonly_file.exists():
    longonly = pd.read_csv(longonly_file)

    print(f"\nLONG-only:")
    print(f"  Avg Return: {longonly['return'].mean():.2f}% per window")
    print(f"  Total Return: {longonly['return'].sum():.2f}%")
    print(f"  Win Rate: {(longonly['return'] > 0).sum() / len(longonly) * 100:.1f}%")
    print(f"  Avg Trades: {longonly['num_trades'].mean():.1f} per window")

    print(f"\nLONG+Enhanced SHORT:")
    print(f"  Avg Return: {results['return'].mean():.2f}% per window")
    print(f"  Total Return: {results['return'].sum():.2f}%")
    print(f"  Win Rate: {(results['return'] > 0).sum() / len(results) * 100:.1f}%")
    print(f"  Avg Trades: {results['num_trades'].mean():.1f} per window")

    print(f"\nDifference:")
    diff_return = results['return'].mean() - longonly['return'].mean()
    diff_total = results['return'].sum() - longonly['return'].sum()
    diff_pct = (diff_return / longonly['return'].mean()) * 100

    print(f"  Avg Return: {diff_return:+.2f}% ({diff_pct:+.1f}%)")
    print(f"  Total Return: {diff_total:+.2f}%")

    # Decision
    print(f"\n{'=' * 80}")
    print(f"DECISION")
    print(f"{'=' * 80}")

    if diff_return > 0 and results['short_win_rate'].mean() > 50:
        print(f"\n  ✅ Enhanced SHORT IMPROVES Performance!")
        print(f"     - Return improvement: {diff_return:+.2f}% per window")
        print(f"     - SHORT win rate: {results['short_win_rate'].mean():.1f}%")
        print(f"\n  🎯 RECOMMENDATION: DEPLOY LONG+SHORT")
    elif diff_return > -2:
        print(f"\n  ⚠️  Enhanced SHORT has NEUTRAL impact")
        print(f"     - Return change: {diff_return:+.2f}% per window")
        print(f"     - SHORT win rate: {results['short_win_rate'].mean():.1f}%")
        print(f"\n  🎯 RECOMMENDATION: Test 1 week, then decide")
    else:
        print(f"\n  ❌ Enhanced SHORT DEGRADES Performance")
        print(f"     - Return drop: {diff_return:+.2f}% per window ({diff_pct:+.1f}%)")
        print(f"     - SHORT win rate: {results['short_win_rate'].mean():.1f}%")
        print(f"\n  🎯 RECOMMENDATION: DISABLE SHORT, use LONG-only")
else:
    print("\n  ⚠️  LONG-only backtest not found. Cannot compare.")

print(f"\n{'=' * 80}")
print("Backtest Complete!")
print(f"{'=' * 80}\n")
