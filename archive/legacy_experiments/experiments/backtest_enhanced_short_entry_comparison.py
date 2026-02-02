"""
Enhanced SHORT Entry Model Comparison Backtest

Compares:
1. Current SHORT Entry (67 multi-timeframe features, Peak/Trough labeling)
2. Enhanced SHORT Entry (22 SELL features, 2of3 scoring labeling)

Expected improvement:
- Signal rate: 1% → 13%
- Win rate: 20% → 60-70%
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Parameters
WINDOW_SIZE = 1440  # 5 days
INITIAL_CAPITAL = 10000.0
POSITION_SIZE_PCT = 0.95
LONG_ENTRY_THRESHOLD = 0.7
SHORT_ENTRY_THRESHOLD = 0.7  # Test enhanced model
EXIT_THRESHOLD = 0.7
TRANSACTION_COST = 0.0002
MAX_HOLDING_HOURS = 4


def calculate_enhanced_features(df):
    """Calculate 22 SELL signal features for enhanced model"""

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

    # RSI dynamics (using existing rsi if available)
    if 'rsi' not in df.columns:
        df['rsi'] = 50.0  # Default

    df['rsi_slope'] = df['rsi'].diff(3)
    df['rsi_overbought'] = (df['rsi'] > 70).astype(float)
    df['rsi_oversold'] = (df['rsi'] < 30).astype(float)
    df['rsi_divergence'] = 0.0

    # MACD dynamics
    if 'macd' not in df.columns:
        df['macd'] = 0.0
        df['macd_signal'] = 0.0

    macd_histogram = df['macd'] - df['macd_signal']
    df['macd_histogram_slope'] = macd_histogram.diff(3)
    df['macd_crossover'] = ((df['macd'] > df['macd_signal']) &
                            (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(float)
    df['macd_crossunder'] = ((df['macd'] < df['macd_signal']) &
                             (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(float)

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
    start_price = df_window['close'].iloc[0]
    end_price = df_window['close'].iloc[-1]
    return_pct = ((end_price / start_price) - 1) * 100
    if return_pct > 3.0:
        return "Bull"
    elif return_pct < -2.0:
        return "Bear"
    else:
        return "Sideways"


def backtest_with_short_model(df, models, scalers, features, model_name="current"):
    """Backtest with specified SHORT Entry model"""
    capital = INITIAL_CAPITAL
    position = None
    trades = []

    for i in range(len(df)):
        current_price = df['close'].iloc[i]

        # Manage existing position
        if position is not None:
            side = position['side']
            entry_idx = position['entry_idx']
            entry_price = position['entry_price']
            hours_held = (i - entry_idx) / 12

            # Calculate P&L
            if side == 'LONG':
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - current_price) / entry_price

            # Check ML EXIT signal
            exit_prob = None
            if side == 'LONG':
                exit_row = df[features['long_exit']].iloc[i:i+1].values
                if not np.isnan(exit_row).any():
                    exit_scaled = scalers['long_exit'].transform(exit_row)
                    exit_prob = models['long_exit'].predict_proba(exit_scaled)[0][1]
            else:  # SHORT
                exit_row = df[features['short_exit']].iloc[i:i+1].values
                if not np.isnan(exit_row).any():
                    exit_scaled = scalers['short_exit'].transform(exit_row)
                    exit_prob = models['short_exit'].predict_proba(exit_scaled)[0][1]

            # Exit conditions
            exit_reason = None
            if exit_prob is not None:
                if exit_prob >= EXIT_THRESHOLD:
                    exit_reason = "ML Exit"
                elif hours_held >= MAX_HOLDING_HOURS:
                    exit_reason = "Max Holding"

            if exit_reason:
                quantity = position['quantity']
                pnl_usd = pnl_pct * (entry_price * quantity)

                entry_cost = entry_price * quantity * TRANSACTION_COST
                exit_cost = current_price * quantity * TRANSACTION_COST
                pnl_usd -= (entry_cost + exit_cost)

                capital += pnl_usd

                trades.append({
                    'side': side,
                    'entry_idx': entry_idx,
                    'exit_idx': i,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl_pct': pnl_pct,
                    'pnl_usd': pnl_usd,
                    'exit_reason': exit_reason,
                    'holding_hours': hours_held,
                    'entry_probability': position['probability'],
                    'exit_probability': exit_prob,
                    'regime': position['regime']
                })

                position = None

        # Look for ENTRY
        if position is None and i < len(df) - 1:
            # LONG Entry
            long_row = df[features['long_entry']].iloc[i:i+1].values
            if np.isnan(long_row).any():
                long_prob = 0
            else:
                long_scaled = scalers['long_entry'].transform(long_row)
                long_prob = models['long_entry'].predict_proba(long_scaled)[0][1]

            # SHORT Entry (using specified model)
            short_row = df[features['short_entry']].iloc[i:i+1].values
            if np.isnan(short_row).any():
                short_prob = 0
            else:
                short_scaled = scalers['short_entry'].transform(short_row)
                short_prob = models['short_entry'].predict_proba(short_scaled)[0][1]

            # Entry logic
            side = None
            probability = None

            if long_prob >= LONG_ENTRY_THRESHOLD and short_prob < SHORT_ENTRY_THRESHOLD:
                side = 'LONG'
                probability = long_prob
            elif short_prob >= SHORT_ENTRY_THRESHOLD and long_prob < LONG_ENTRY_THRESHOLD:
                side = 'SHORT'
                probability = short_prob
            elif long_prob >= LONG_ENTRY_THRESHOLD and short_prob >= SHORT_ENTRY_THRESHOLD:
                if long_prob > short_prob:
                    side = 'LONG'
                    probability = long_prob
                else:
                    side = 'SHORT'
                    probability = short_prob

            if side is not None:
                position_value = capital * POSITION_SIZE_PCT
                quantity = position_value / current_price
                current_regime = classify_market_regime(df.iloc[max(0, i-20):i+1])

                position = {
                    'side': side,
                    'entry_idx': i,
                    'entry_price': current_price,
                    'quantity': quantity,
                    'probability': probability,
                    'regime': current_regime
                }

    # Metrics
    if len(trades) == 0:
        return trades, {}

    total_return_pct = ((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    winning_trades = [t for t in trades if t['pnl_usd'] > 0]
    win_rate = (len(winning_trades) / len(trades)) * 100

    long_trades = [t for t in trades if t['side'] == 'LONG']
    short_trades = [t for t in trades if t['side'] == 'SHORT']

    win_rate_long = (len([t for t in long_trades if t['pnl_usd'] > 0]) / len(long_trades) * 100) if len(long_trades) > 0 else 0
    win_rate_short = (len([t for t in short_trades if t['pnl_usd'] > 0]) / len(short_trades) * 100) if len(short_trades) > 0 else 0

    # SHORT Entry signal rate
    short_entry_rate = (len(short_trades) / len(trades) * 100) if len(trades) > 0 else 0

    ml_exits = [t for t in trades if t['exit_reason'] == 'ML Exit']
    ml_exit_rate = (len(ml_exits) / len(trades)) * 100

    # Sharpe
    if len(trades) > 1:
        returns = [t['pnl_pct'] for t in trades]
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
    else:
        sharpe = 0

    return trades, {
        'total_return_pct': total_return_pct,
        'num_trades': len(trades),
        'num_long': len(long_trades),
        'num_short': len(short_trades),
        'short_entry_rate': short_entry_rate,
        'win_rate': win_rate,
        'win_rate_long': win_rate_long,
        'win_rate_short': win_rate_short,
        'sharpe_ratio': sharpe,
        'ml_exit_rate': ml_exit_rate
    }


def rolling_window_backtest(df, models, scalers, features, model_name):
    windows = []
    start_idx = 0

    while start_idx + WINDOW_SIZE <= len(df):
        end_idx = start_idx + WINDOW_SIZE
        window_df = df.iloc[start_idx:end_idx].reset_index(drop=True)

        regime = classify_market_regime(window_df)
        trades, metrics = backtest_with_short_model(window_df, models, scalers, features, model_name)

        if len(metrics) > 0:
            windows.append({
                'start_idx': start_idx,
                'end_idx': end_idx,
                'regime': regime,
                'model': model_name,
                'return_pct': metrics['total_return_pct'],
                'num_trades': metrics['num_trades'],
                'num_long': metrics['num_long'],
                'num_short': metrics['num_short'],
                'short_rate': metrics['short_entry_rate'],
                'win_rate': metrics['win_rate'],
                'win_rate_long': metrics['win_rate_long'],
                'win_rate_short': metrics['win_rate_short'],
                'sharpe': metrics['sharpe_ratio'],
                'ml_exit_rate': metrics['ml_exit_rate']
            })

        start_idx += WINDOW_SIZE

    return pd.DataFrame(windows)


def main():
    print("=" * 80)
    print("Enhanced SHORT Entry Model Comparison Backtest")
    print("=" * 80)

    # Load data
    print("\n1. Loading Data...")
    df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_max.csv")
    print(f"  ✅ {len(df):,} candles")

    # Calculate enhanced features
    print("\n2. Calculating Enhanced Features...")
    df = calculate_enhanced_features(df)
    print(f"  ✅ 22 SELL signal features calculated")

    # Load shared models (LONG Entry, LONG Exit, SHORT Exit)
    print("\n3. Loading Shared Models...")

    long_entry_model = pickle.load(open(MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl", 'rb'))
    long_entry_scaler = pickle.load(open(MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_scaler.pkl", 'rb'))
    with open(MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt", 'r') as f:
        long_entry_features = [line.strip() for line in f]
    print(f"  ✅ LONG Entry: {len(long_entry_features)} features")

    long_exit_model = pickle.load(open(MODELS_DIR / "xgboost_long_exit_improved_20251016_175554.pkl", 'rb'))
    long_exit_scaler = pickle.load(open(MODELS_DIR / "xgboost_long_exit_improved_20251016_175554_scaler.pkl", 'rb'))
    with open(MODELS_DIR / "xgboost_long_exit_improved_20251016_175554_features.txt", 'r') as f:
        long_exit_features = [line.strip() for line in f]
    print(f"  ✅ LONG Exit: {len(long_exit_features)} features (enhanced)")

    short_exit_model = pickle.load(open(MODELS_DIR / "xgboost_short_exit_improved_20251016_180207.pkl", 'rb'))
    short_exit_scaler = pickle.load(open(MODELS_DIR / "xgboost_short_exit_improved_20251016_180207_scaler.pkl", 'rb'))
    with open(MODELS_DIR / "xgboost_short_exit_improved_20251016_180207_features.txt", 'r') as f:
        short_exit_features = [line.strip() for line in f]
    print(f"  ✅ SHORT Exit: {len(short_exit_features)} features (enhanced)")

    # Test 1: Current SHORT Entry
    print("\n" + "=" * 80)
    print("TEST 1: Current SHORT Entry Model")
    print("=" * 80)

    # Use production bot's current SHORT Entry model
    current_short_model = pickle.load(open(MODELS_DIR / "xgboost_short_model_lookahead3_thresh0.3.pkl", 'rb'))
    current_short_scaler = pickle.load(open(MODELS_DIR / "xgboost_short_model_lookahead3_thresh0.3_scaler.pkl", 'rb'))
    # SHORT Entry uses same features as LONG Entry (44 features)
    current_short_features = long_entry_features
    print(f"  Features: {len(current_short_features)} (multi-timeframe)")
    print(f"  Expected: 1% signal rate, 20% win rate")

    current_models = {
        'long_entry': long_entry_model,
        'short_entry': current_short_model,
        'long_exit': long_exit_model,
        'short_exit': short_exit_model
    }

    current_scalers = {
        'long_entry': long_entry_scaler,
        'short_entry': current_short_scaler,
        'long_exit': long_exit_scaler,
        'short_exit': short_exit_scaler
    }

    current_features = {
        'long_entry': long_entry_features,
        'short_entry': current_short_features,
        'long_exit': long_exit_features,
        'short_exit': short_exit_features
    }

    current_results = rolling_window_backtest(df, current_models, current_scalers, current_features, "current")

    # Test 2: Enhanced SHORT Entry
    print("\n" + "=" * 80)
    print("TEST 2: Enhanced SHORT Entry Model")
    print("=" * 80)

    enhanced_short_model = pickle.load(open(MODELS_DIR / "xgboost_short_entry_enhanced_20251016_201219.pkl", 'rb'))
    enhanced_short_scaler = pickle.load(open(MODELS_DIR / "xgboost_short_entry_enhanced_20251016_201219_scaler.pkl", 'rb'))
    with open(MODELS_DIR / "xgboost_short_entry_enhanced_20251016_201219_features.txt", 'r') as f:
        enhanced_short_features = [line.strip() for line in f]
    print(f"  Features: {len(enhanced_short_features)} (SELL signals)")
    print(f"  Expected: 13% signal rate, 60-70% win rate")

    enhanced_models = {
        'long_entry': long_entry_model,
        'short_entry': enhanced_short_model,
        'long_exit': long_exit_model,
        'short_exit': short_exit_model
    }

    enhanced_scalers = {
        'long_entry': long_entry_scaler,
        'short_entry': enhanced_short_scaler,
        'long_exit': long_exit_scaler,
        'short_exit': short_exit_scaler
    }

    enhanced_features = {
        'long_entry': long_entry_features,
        'short_entry': enhanced_short_features,
        'long_exit': long_exit_features,
        'short_exit': short_exit_features
    }

    enhanced_results = rolling_window_backtest(df, enhanced_models, enhanced_scalers, enhanced_features, "enhanced")

    # Comparison
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)

    print(f"\n📊 {'Metric':<30} {'Current':<15} {'Enhanced':<15} {'Change'}")
    print("-" * 80)

    metrics = [
        ('Return per window', 'return_pct', '%'),
        ('Total trades', 'num_trades', ''),
        ('SHORT trades', 'num_short', ''),
        ('SHORT signal rate', 'short_rate', '%'),
        ('Win rate (overall)', 'win_rate', '%'),
        ('Win rate (SHORT)', 'win_rate_short', '%'),
        ('Sharpe ratio', 'sharpe', ''),
        ('ML Exit rate', 'ml_exit_rate', '%')
    ]

    for label, col, unit in metrics:
        current_val = current_results[col].mean()
        enhanced_val = enhanced_results[col].mean()
        change = enhanced_val - current_val
        change_pct = (change / current_val * 100) if current_val != 0 else 0

        if unit == '%':
            print(f"{label:<30} {current_val:>6.2f}% {enhanced_val:>6.2f}% {change:>+6.2f}% ({change_pct:+.1f}%)")
        else:
            print(f"{label:<30} {current_val:>6.2f}  {enhanced_val:>6.2f}  {change:>+6.2f} ({change_pct:+.1f}%)")

    # Statistical significance
    from scipy import stats
    if len(current_results) > 1 and len(enhanced_results) > 1:
        t_stat, p_value = stats.ttest_ind(enhanced_results['return_pct'], current_results['return_pct'])
        print(f"\n📊 Statistical Significance:")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Significant: {'✅ Yes' if p_value < 0.05 else '❌ No'} (alpha=0.05)")

    # Key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    short_rate_current = current_results['short_rate'].mean()
    short_rate_enhanced = enhanced_results['short_rate'].mean()
    win_rate_short_current = current_results['win_rate_short'].mean()
    win_rate_short_enhanced = enhanced_results['win_rate_short'].mean()
    return_current = current_results['return_pct'].mean()
    return_enhanced = enhanced_results['return_pct'].mean()

    print(f"\n1. SHORT Entry Signal Rate:")
    print(f"   Current: {short_rate_current:.1f}%")
    print(f"   Enhanced: {short_rate_enhanced:.1f}%")
    print(f"   Change: {short_rate_enhanced - short_rate_current:+.1f}% ({(short_rate_enhanced/short_rate_current-1)*100:+.1f}%)")

    print(f"\n2. SHORT Entry Win Rate:")
    print(f"   Current: {win_rate_short_current:.1f}%")
    print(f"   Enhanced: {win_rate_short_enhanced:.1f}%")
    print(f"   Change: {win_rate_short_enhanced - win_rate_short_current:+.1f}% ({(win_rate_short_enhanced/win_rate_short_current-1)*100 if win_rate_short_current > 0 else 0:+.1f}%)")

    print(f"\n3. Overall Return:")
    print(f"   Current: {return_current:+.2f}%")
    print(f"   Enhanced: {return_enhanced:+.2f}%")
    print(f"   Change: {return_enhanced - return_current:+.2f}%")

    # Recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    if (short_rate_enhanced >= short_rate_current * 5 and
        win_rate_short_enhanced >= win_rate_short_current * 1.5 and
        return_enhanced >= return_current):
        print("\n✅ DEPLOY ENHANCED MODEL")
        print("  - Significantly higher signal rate")
        print("  - Much better win rate")
        print("  - Equal or better returns")
        print("  - BUY/SELL paradigm alignment")
    elif win_rate_short_enhanced >= win_rate_short_current * 1.2:
        print("\n✅ CONSIDER DEPLOYMENT")
        print("  - Improved win rate")
        print("  - May need signal rate tuning")
    else:
        print("\n⚠️  NEEDS IMPROVEMENT")
        print("  - Performance not significantly better")
        print("  - Consider further optimization")

    # Save results
    combined_results = pd.concat([current_results, enhanced_results], ignore_index=True)
    combined_results.to_csv(RESULTS_DIR / "backtest_short_entry_comparison.csv", index=False)
    print(f"\n✅ Results saved: backtest_short_entry_comparison.csv")

    print("\n" + "=" * 80)
    print("Backtest Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
