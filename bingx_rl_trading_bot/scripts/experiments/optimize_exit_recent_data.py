"""
EXIT Threshold Optimization - Recent Data (Oct 1-26)
====================================================
Goal: Find EXIT threshold that beats production performance (+7.05% per 5-day)

Test Period: Oct 1-26, 2025 (closest to production period Oct 30+)
Test Thresholds: 0.60, 0.65, 0.70, 0.75
Current Production: EXIT 0.75, +7.05% per 5-day (estimated)

Approach: Window-based backtest with realistic fees on RECENT data
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

# ============================================================================
# Backtest Parameters (PRODUCTION settings)
# ============================================================================
LEVERAGE = 4
INITIAL_CAPITAL = 10000
EMERGENCY_MAX_HOLD = 120  # 10 hours
LONG_THRESHOLD = 0.80  # PRODUCTION
SHORT_THRESHOLD = 0.80  # PRODUCTION
GATE_THRESHOLD = 0.001
TAKER_FEE = 0.0005  # 0.05%
SLIPPAGE = 0.0001  # 0.01%
WINDOW_DAYS = 5

# Test thresholds
THRESHOLDS_TO_TEST = [0.60, 0.65, 0.70, 0.75]

# Production baseline
PRODUCTION_RETURN_5DAY = 0.0705  # +7.05% per 5-day window

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

def load_models():
    """Load Entry and Exit models"""
    print("\n=== Loading Models ===")

    # Entry models (Enhanced 5-Fold CV)
    long_entry_path = MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445.pkl"
    short_entry_path = MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445.pkl"

    # Exit models (oppgating improved)
    long_exit_path = MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527.pkl"
    short_exit_path = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510.pkl"

    models = {
        'long_entry': joblib.load(long_entry_path),
        'short_entry': joblib.load(short_entry_path),
        'long_exit': joblib.load(long_exit_path),
        'short_exit': joblib.load(short_exit_path)
    }

    # Load scalers (corrected naming: model_name_scaler.pkl)
    models['long_entry_scaler'] = joblib.load(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_scaler.pkl")
    models['short_entry_scaler'] = joblib.load(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_scaler.pkl")
    models['long_exit_scaler'] = joblib.load(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527_scaler.pkl")
    models['short_exit_scaler'] = joblib.load(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510_scaler.pkl")

    # Load feature columns (from .txt files)
    with open(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_features.txt", 'r') as f:
        models['long_entry_features'] = [line.strip() for line in f.readlines() if line.strip()]

    with open(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_features.txt", 'r') as f:
        models['short_entry_features'] = [line.strip() for line in f.readlines() if line.strip()]

    with open(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527_features.txt", 'r') as f:
        models['long_exit_features'] = [line.strip() for line in f.readlines() if line.strip()]

    with open(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510_features.txt", 'r') as f:
        models['short_exit_features'] = [line.strip() for line in f.readlines() if line.strip()]

    print(f"✅ LONG Entry: {len(models['long_entry_features'])} features")
    print(f"✅ SHORT Entry: {len(models['short_entry_features'])} features")
    print(f"✅ LONG Exit: {len(models['long_exit_features'])} features")
    print(f"✅ SHORT Exit: {len(models['short_exit_features'])} features")

    return models

def simulate_trade(df, entry_idx, side, exit_threshold):
    """Simulate a single trade with EXIT threshold"""
    entry_price = df.loc[entry_idx, 'close']

    # Opportunity gating for SHORT
    if side == 'SHORT':
        long_prob = df.loc[entry_idx, 'long_entry_prob']
        short_prob = df.loc[entry_idx, 'short_entry_prob']

        long_ev = long_prob * 0.0041
        short_ev = short_prob * 0.0047
        opportunity_cost = short_ev - long_ev

        if opportunity_cost <= GATE_THRESHOLD:
            return None  # Skip this SHORT trade

    # Simulate holding
    for hold_time in range(1, min(EMERGENCY_MAX_HOLD + 1, len(df) - entry_idx)):
        current_idx = entry_idx + hold_time
        current_price = df.loc[current_idx, 'close']

        # Calculate P&L
        if side == 'LONG':
            price_change = (current_price - entry_price) / entry_price
        else:  # SHORT
            price_change = (entry_price - current_price) / entry_price

        leveraged_pnl = price_change * LEVERAGE

        # Apply fees and slippage
        total_cost = 2 * (TAKER_FEE + SLIPPAGE)  # Entry + Exit
        net_pnl = leveraged_pnl - total_cost

        # Get exit probability
        exit_prob_col = 'long_exit_prob' if side == 'LONG' else 'short_exit_prob'
        exit_prob = df.loc[current_idx, exit_prob_col]

        # ML Exit (threshold varies)
        if exit_prob >= exit_threshold:
            return {
                'exit_reason': 'ml_exit',
                'hold_time': hold_time,
                'pnl': net_pnl,
                'exit_prob': exit_prob
            }

        # Max Hold
        if hold_time >= EMERGENCY_MAX_HOLD:
            return {
                'exit_reason': 'max_hold',
                'hold_time': hold_time,
                'pnl': net_pnl,
                'exit_prob': exit_prob
            }

    # End of data
    return {
        'exit_reason': 'end_of_data',
        'hold_time': len(df) - entry_idx - 1,
        'pnl': net_pnl,
        'exit_prob': 0
    }

def backtest_exit_threshold(df, exit_threshold):
    """WINDOW-BASED Backtest with given EXIT threshold"""
    all_results = []
    window_returns = []

    # Calculate windows (5-day = 1440 candles)
    candles_per_window = WINDOW_DAYS * 24 * 60 // 5
    num_windows = (len(df) - 200) // candles_per_window

    print(f"\n=== Testing EXIT Threshold: {exit_threshold} ===")
    print(f"Windows: {num_windows}, Candles/window: {candles_per_window}")

    for window_idx in range(num_windows):
        # Reset capital
        capital = INITIAL_CAPITAL
        window_start = 200 + window_idx * candles_per_window
        window_end = window_start + candles_per_window
        window_trades = []

        # Trade within window
        idx = window_start
        while idx < window_end - EMERGENCY_MAX_HOLD:
            # Entry signals
            long_prob = df.loc[idx, 'long_entry_prob']
            short_prob = df.loc[idx, 'short_entry_prob']

            side = None
            if long_prob >= LONG_THRESHOLD:
                side = 'LONG'
            elif short_prob >= SHORT_THRESHOLD:
                side = 'SHORT'

            if side:
                # Simulate trade
                result = simulate_trade(df, idx, side, exit_threshold)

                if result:
                    # Calculate position size (dynamic 20-95%)
                    entry_prob = long_prob if side == 'LONG' else short_prob
                    position_size = 0.20 + (entry_prob - 0.65) * (0.75 / 0.35)
                    position_size = np.clip(position_size, 0.20, 0.95)

                    # Update capital
                    trade_pnl = capital * position_size * result['pnl']
                    capital += trade_pnl

                    # Record trade
                    window_trades.append({
                        'side': side,
                        'entry_idx': idx,
                        'position_size': position_size,
                        'pnl': result['pnl'],
                        'pnl_usd': trade_pnl,
                        'hold_time': result['hold_time'],
                        'exit_reason': result['exit_reason'],
                        'exit_prob': result['exit_prob']
                    })

                    # Move past this trade
                    idx += result['hold_time']
                else:
                    idx += 1
            else:
                idx += 1

        # Window complete
        window_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
        window_returns.append(window_return)
        all_results.extend(window_trades)

    # Aggregate results
    if len(all_results) == 0:
        return None

    df_results = pd.DataFrame(all_results)

    # Calculate metrics
    total_trades = len(df_results)
    wins = len(df_results[df_results['pnl'] > 0])
    win_rate = wins / total_trades if total_trades > 0 else 0

    # Window-based return
    avg_window_return = np.mean(window_returns)
    std_window_return = np.std(window_returns)
    sharpe = (avg_window_return / std_window_return * np.sqrt(73)) if std_window_return > 0 else 0

    # ML Exit rate
    ml_exits = len(df_results[df_results['exit_reason'] == 'ml_exit'])
    ml_exit_rate = ml_exits / total_trades if total_trades > 0 else 0

    # Trades per window
    trades_per_window = total_trades / num_windows if num_windows > 0 else 0

    return {
        'threshold': exit_threshold,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_window_return': avg_window_return,
        'std_window_return': std_window_return,
        'sharpe_ratio': sharpe,
        'ml_exit_rate': ml_exit_rate,
        'trades_per_window': trades_per_window,
        'num_windows': num_windows
    }

def main():
    print("=" * 70)
    print("EXIT Threshold Optimization - Recent Data (Oct 1-26)")
    print("=" * 70)

    # Load data
    print("\n=== Loading Data ===")
    data_path = DATA_DIR / "BTCUSDT_5m_features.csv"
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Filter to Oct 1-26, 2025 (recent data, close to production)
    df = df[(df['timestamp'] >= '2025-10-01') & (df['timestamp'] <= '2025-10-26')].copy()
    df = df.reset_index(drop=True)

    print(f"Period: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Total candles: {len(df):,}")
    print(f"Days: {(df['timestamp'].max() - df['timestamp'].min()).days}")

    # Prepare EXIT features
    df = prepare_exit_features(df)

    # Load models
    models = load_models()

    # Generate Entry probabilities
    print("\n=== Generating Entry Probabilities ===")

    # LONG Entry
    long_feat = df[models['long_entry_features']].values
    long_feat_scaled = models['long_entry_scaler'].transform(long_feat)
    df['long_entry_prob'] = models['long_entry'].predict_proba(long_feat_scaled)[:, 1]

    # SHORT Entry
    short_feat = df[models['short_entry_features']].values
    short_feat_scaled = models['short_entry_scaler'].transform(short_feat)
    df['short_entry_prob'] = models['short_entry'].predict_proba(short_feat_scaled)[:, 1]

    print(f"✅ LONG Entry probs: mean={df['long_entry_prob'].mean():.4f}")
    print(f"✅ SHORT Entry probs: mean={df['short_entry_prob'].mean():.4f}")

    # Generate Exit probabilities
    print("\n=== Generating Exit Probabilities ===")

    # LONG Exit
    long_exit_feat = df[models['long_exit_features']].values
    long_exit_feat_scaled = models['long_exit_scaler'].transform(long_exit_feat)
    df['long_exit_prob'] = models['long_exit'].predict_proba(long_exit_feat_scaled)[:, 1]

    # SHORT Exit
    short_exit_feat = df[models['short_exit_features']].values
    short_exit_feat_scaled = models['short_exit_scaler'].transform(short_exit_feat)
    df['short_exit_prob'] = models['short_exit'].predict_proba(short_exit_feat_scaled)[:, 1]

    print(f"✅ LONG Exit probs: mean={df['long_exit_prob'].mean():.4f}")
    print(f"✅ SHORT Exit probs: mean={df['short_exit_prob'].mean():.4f}")

    # Test each threshold
    results = []
    for threshold in THRESHOLDS_TO_TEST:
        result = backtest_exit_threshold(df, threshold)
        if result:
            results.append(result)

    # Display results
    print("\n" + "=" * 70)
    print("RESULTS - EXIT Threshold Optimization (Oct 1-26)")
    print("=" * 70)

    if len(results) == 0:
        print("❌ No results generated!")
        return

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('avg_window_return', ascending=False)

    print(f"\nCurrent Production Baseline: +{PRODUCTION_RETURN_5DAY*100:.2f}% per 5-day window")
    print("\nThreshold Comparison:")
    print("-" * 70)

    for idx, row in df_results.iterrows():
        vs_prod = (row['avg_window_return'] / PRODUCTION_RETURN_5DAY - 1) * 100
        status = "✅ BEATS" if row['avg_window_return'] > PRODUCTION_RETURN_5DAY else "❌ BELOW"

        print(f"\nEXIT {row['threshold']:.2f}:")
        print(f"  Return/5-day: {row['avg_window_return']*100:+.2f}% (±{row['std_window_return']*100:.2f}%) {status} production")
        print(f"  vs Production: {vs_prod:+.1f}%")
        print(f"  Sharpe Ratio: {row['sharpe_ratio']:.2f}")
        print(f"  Win Rate: {row['win_rate']*100:.1f}%")
        print(f"  ML Exit Rate: {row['ml_exit_rate']*100:.1f}%")
        print(f"  Trades/window: {row['trades_per_window']:.1f}")
        print(f"  Total Trades: {row['total_trades']} across {row['num_windows']} windows")

    # Best threshold
    best = df_results.iloc[0]
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    if best['avg_window_return'] > PRODUCTION_RETURN_5DAY:
        improvement = (best['avg_window_return'] / PRODUCTION_RETURN_5DAY - 1) * 100
        print(f"\n🎉 FOUND BETTER THRESHOLD: EXIT {best['threshold']}")
        print(f"   Current Production: +{PRODUCTION_RETURN_5DAY*100:.2f}% per 5-day")
        print(f"   Optimized (EXIT {best['threshold']}): +{best['avg_window_return']*100:.2f}% per 5-day")
        print(f"   Improvement: +{improvement:.1f}%")
        print(f"\n✅ READY TO DEPLOY - Update production bot EXIT threshold to {best['threshold']}")
    else:
        print(f"\n⚠️ NO IMPROVEMENT FOUND")
        print(f"   Best threshold (EXIT {best['threshold']}): +{best['avg_window_return']*100:.2f}%")
        print(f"   Current Production: +{PRODUCTION_RETURN_5DAY*100:.2f}%")
        print(f"\n❌ KEEP CURRENT SETTINGS - No deployment needed")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"exit_optimization_recent_{timestamp}.csv"
    df_results.to_csv(output_path, index=False)
    print(f"\n📊 Results saved: {output_path}")

if __name__ == "__main__":
    main()
