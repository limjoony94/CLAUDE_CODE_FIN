"""
Entry Threshold Optimization for Opportunity Gating Strategy
=============================================================

Test different LONG/SHORT entry thresholds to find optimal configuration
that maximizes win rate and risk-adjusted returns while minimizing false positives.

Current Issue:
- Too many trades in some windows (50-101 trades vs optimal 30-40)
- Low win rate in high-trade periods (32-42% vs target 60%+)
- Entry thresholds too aggressive (LONG 0.65, SHORT 0.70)

Optimization Strategy:
- Test LONG thresholds: 0.60, 0.65, 0.70, 0.75, 0.80
- Test SHORT thresholds: 0.65, 0.70, 0.75, 0.80, 0.85
- Evaluate on: Win Rate, Sharpe Ratio, Avg Trades, Return
- Find balance between opportunity and quality
"""

import sys
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.calculate_all_features import calculate_all_features
from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features
from scripts.production.dynamic_position_sizing import DynamicPositionSizer

# Load models
MODELS_DIR = PROJECT_ROOT / "models"

print("Loading models...")
# LONG Entry
with open(MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl", 'rb') as f:
    long_model = pickle.load(f)
with open(MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_scaler.pkl", 'rb') as f:
    long_scaler = pickle.load(f)
with open(MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt", 'r') as f:
    long_feature_columns = [line.strip() for line in f.readlines()]

# SHORT Entry
with open(MODELS_DIR / "xgboost_short_redesigned_20251016_233322.pkl", 'rb') as f:
    short_model = pickle.load(f)
with open(MODELS_DIR / "xgboost_short_redesigned_20251016_233322_scaler.pkl", 'rb') as f:
    short_scaler = pickle.load(f)
with open(MODELS_DIR / "xgboost_short_redesigned_20251016_233322_features.txt", 'r') as f:
    short_feature_columns = [line.strip() for line in f.readlines() if line.strip()]

# LONG Exit
with open(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251017_151624.pkl", 'rb') as f:
    long_exit_model = pickle.load(f)
with open(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251017_151624_scaler.pkl", 'rb') as f:
    long_exit_scaler = pickle.load(f)
with open(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251017_151624_features.txt", 'r') as f:
    long_exit_feature_columns = [line.strip() for line in f.readlines() if line.strip()]

# SHORT Exit
with open(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251017_152440.pkl", 'rb') as f:
    short_exit_model = pickle.load(f)
with open(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251017_152440_scaler.pkl", 'rb') as f:
    short_exit_scaler = pickle.load(f)
with open(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251017_152440_features.txt", 'r') as f:
    short_exit_feature_columns = [line.strip() for line in f.readlines() if line.strip()]

print("✅ Models loaded\n")

# Strategy parameters
GATE_THRESHOLD = 0.001
LEVERAGE = 4
LONG_AVG_RETURN = 0.0041
SHORT_AVG_RETURN = 0.0047

# Exit parameters (COMBINED strategy - optimized)
ML_EXIT_THRESHOLD_BASE_LONG = 0.70
ML_EXIT_THRESHOLD_BASE_SHORT = 0.72
EMERGENCY_STOP_LOSS = -0.03  # Balance-Based: -3% total balance (optimized 2025-10-22, Return: +50.31%, WR: 57.1%, MDD: -15%)
EMERGENCY_MAX_HOLD = 96
FIXED_TAKE_PROFIT = 0.03
TRAILING_TP_ACTIVATION = 0.02
TRAILING_TP_DRAWDOWN = 0.10
VOLATILITY_HIGH = 0.02
VOLATILITY_LOW = 0.01
ML_THRESHOLD_HIGH_VOL = 0.65
ML_THRESHOLD_LOW_VOL = 0.75

sizer = DynamicPositionSizer(base_position_pct=0.50, max_position_pct=0.95, min_position_pct=0.20)

def calculate_market_volatility(df_features, lookback=20):
    """Calculate market volatility"""
    try:
        if len(df_features) < lookback:
            lookback = len(df_features)
        if lookback < 2:
            return 0.015
        recent_prices = df_features['close'].iloc[-lookback:]
        returns = recent_prices.pct_change().dropna()
        if len(returns) < 2:
            return 0.015
        return float(returns.std())
    except:
        return 0.015

def check_entry_signal(i, df_features, position, balance, long_threshold, short_threshold):
    """Check for entry signal with configurable thresholds"""
    if position is not None:
        return False, None, None, None, None

    # Get features
    latest = df_features.iloc[i:i+1].copy()

    # LONG signal
    long_feat = latest[long_feature_columns].values
    long_feat_scaled = long_scaler.transform(long_feat)
    long_prob = long_model.predict_proba(long_feat_scaled)[0][1]

    # SHORT signal
    short_feat_df = latest[short_feature_columns]
    if short_feat_df.shape[1] != 38:
        short_feat = short_feat_df.iloc[:, :38].values
    else:
        short_feat = short_feat_df.values
    short_feat_scaled = short_scaler.transform(short_feat)
    short_prob = short_model.predict_proba(short_feat_scaled)[0][1]

    # LONG entry
    if long_prob >= long_threshold:
        sizing_result = sizer.get_position_size_simple(balance, long_prob, LEVERAGE)
        return True, "LONG", long_prob, short_prob, sizing_result

    # SHORT entry (gated)
    if short_prob >= short_threshold:
        long_ev = long_prob * LONG_AVG_RETURN
        short_ev = short_prob * SHORT_AVG_RETURN
        opportunity_cost = short_ev - long_ev

        if opportunity_cost > GATE_THRESHOLD:
            sizing_result = sizer.get_position_size_simple(balance, short_prob, LEVERAGE)
            return True, "SHORT", long_prob, short_prob, sizing_result

    return False, None, long_prob, short_prob, None

def check_exit_signal(position, current_price, hours_held, df_features, i):
    """Check for exit signal using COMBINED strategy"""
    # Calculate P&L
    entry_notional = position['quantity'] * position['entry_price']
    current_notional = position['quantity'] * current_price

    if position['side'] == "LONG":
        pnl_usd = current_notional - entry_notional
        price_change_pct = (current_price - position['entry_price']) / position['entry_price']
    else:
        pnl_usd = entry_notional - current_notional
        price_change_pct = (position['entry_price'] - current_price) / position['entry_price']

    leveraged_pnl_pct = price_change_pct * LEVERAGE

    # Track peak
    if 'peak_pnl_pct' not in position:
        position['peak_pnl_pct'] = leveraged_pnl_pct
    else:
        position['peak_pnl_pct'] = max(position['peak_pnl_pct'], leveraged_pnl_pct)

    # 1. Fixed Take Profit
    if leveraged_pnl_pct >= FIXED_TAKE_PROFIT:
        return True, f"Fixed TP ({leveraged_pnl_pct*100:.2f}%)", leveraged_pnl_pct

    # 2. Trailing Take Profit
    if position['peak_pnl_pct'] >= TRAILING_TP_ACTIVATION:
        drawdown_from_peak = (position['peak_pnl_pct'] - leveraged_pnl_pct) / position['peak_pnl_pct']
        if drawdown_from_peak >= TRAILING_TP_DRAWDOWN:
            return True, f"Trailing TP ({drawdown_from_peak*100:.1f}% dd)", leveraged_pnl_pct

    # 3. Dynamic ML Exit
    latest = df_features.iloc[i:i+1].copy()

    if position['side'] == "LONG":
        exit_features = latest[long_exit_feature_columns].values
        exit_features_scaled = long_exit_scaler.transform(exit_features)
        exit_prob = long_exit_model.predict_proba(exit_features_scaled)[0][1]
        base_threshold = ML_EXIT_THRESHOLD_BASE_LONG
    else:
        exit_features = latest[short_exit_feature_columns].values
        exit_features_scaled = short_exit_scaler.transform(exit_features)
        exit_prob = short_exit_model.predict_proba(exit_features_scaled)[0][1]
        base_threshold = ML_EXIT_THRESHOLD_BASE_SHORT

    # Adjust for volatility
    volatility = calculate_market_volatility(df_features.iloc[:i+1])
    if volatility > VOLATILITY_HIGH:
        ml_threshold = ML_THRESHOLD_HIGH_VOL
    elif volatility < VOLATILITY_LOW:
        ml_threshold = ML_THRESHOLD_LOW_VOL
    else:
        ml_threshold = base_threshold

    if exit_prob >= ml_threshold:
        return True, f"ML Exit (prob={exit_prob:.3f})", leveraged_pnl_pct

    # 4. Emergency Stop Loss
    if leveraged_pnl_pct <= EMERGENCY_STOP_LOSS:
        return True, f"Emergency SL ({leveraged_pnl_pct*100:.2f}%)", leveraged_pnl_pct

    # 5. Emergency Max Hold
    if hours_held >= EMERGENCY_MAX_HOLD:
        return True, f"Max Hold ({hours_held:.1f}h)", leveraged_pnl_pct

    return False, None, None

def run_backtest_window(df, window_start_idx, window_size, long_threshold, short_threshold):
    """Run backtest for one window with specific thresholds"""
    window_end_idx = min(window_start_idx + window_size, len(df))
    window_df = df.iloc[window_start_idx:window_end_idx].copy()

    balance = 10000.0
    position = None
    trades = []

    for i in range(len(window_df)):
        current_price = window_df.iloc[i]['close']

        # Check exit
        if position is not None:
            hours_held = (i - position['entry_idx']) * (5 / 60)  # 5min candles
            should_exit, exit_reason, exit_pnl = check_exit_signal(
                position, current_price, hours_held, window_df, i
            )

            if should_exit:
                position['exit_price'] = current_price
                position['exit_pnl_pct'] = exit_pnl
                position['exit_reason'] = exit_reason
                trades.append(position)
                balance += position['quantity'] * position['entry_price'] * exit_pnl
                position = None

        # Check entry
        if position is None:
            should_enter, side, long_prob, short_prob, sizing_result = check_entry_signal(
                i, window_df, position, balance, long_threshold, short_threshold
            )

            if should_enter and sizing_result:
                quantity = sizing_result['leveraged_value'] / current_price
                position = {
                    'side': side,
                    'entry_idx': i,
                    'entry_price': current_price,
                    'quantity': quantity,
                    'position_value': sizing_result['position_value'],
                    'peak_pnl_pct': 0.0
                }

    # Close any open position at window end
    if position is not None:
        current_price = window_df.iloc[-1]['close']
        hours_held = (len(window_df) - 1 - position['entry_idx']) * (5 / 60)
        _, _, exit_pnl = check_exit_signal(position, current_price, hours_held, window_df, len(window_df)-1)
        position['exit_price'] = current_price
        position['exit_pnl_pct'] = exit_pnl if exit_pnl is not None else 0.0
        position['exit_reason'] = "Window End"
        trades.append(position)
        balance += position['quantity'] * position['entry_price'] * position['exit_pnl_pct']

    # Calculate metrics
    if len(trades) == 0:
        return {
            'total_trades': 0,
            'long_trades': 0,
            'short_trades': 0,
            'win_rate': 0,
            'avg_return': 0,
            'sharpe': 0,
            'total_return': 0
        }

    wins = sum(1 for t in trades if t.get('exit_pnl_pct', 0) > 0)
    win_rate = wins / len(trades)

    returns = [t.get('exit_pnl_pct', 0) for t in trades]
    avg_return = np.mean(returns)
    std_return = np.std(returns) if len(returns) > 1 else 0.01
    sharpe = avg_return / std_return if std_return > 0 else 0

    total_return = (balance - 10000) / 10000

    long_trades = sum(1 for t in trades if t['side'] == 'LONG')
    short_trades = len(trades) - long_trades

    return {
        'total_trades': len(trades),
        'long_trades': long_trades,
        'short_trades': short_trades,
        'win_rate': win_rate,
        'avg_return': avg_return,
        'sharpe': sharpe,
        'total_return': total_return
    }

def optimize_entry_thresholds():
    """Test multiple entry threshold combinations"""
    print("Loading data...")
    DATA_FILE = PROJECT_ROOT / "data" / "btc_usdt_5m_with_cache.csv"
    df = pd.read_csv(DATA_FILE)
    df['timestamp'] = pd.to_datetime(df['time'], unit='ms')

    print("Calculating features...")
    df_features = calculate_all_features(df.copy())
    df_features = prepare_exit_features(df_features)
    print(f"Features calculated: {len(df_features)} candles\n")

    # Test ranges
    long_thresholds = [0.60, 0.65, 0.70, 0.75, 0.80]
    short_thresholds = [0.65, 0.70, 0.75, 0.80, 0.85]

    # Test on 100 windows (same as baseline)
    window_size = int(24 * 60 / 5 * 5)  # 5 days of 5min candles
    num_windows = 100

    results = []

    print(f"Testing {len(long_thresholds)} LONG × {len(short_thresholds)} SHORT thresholds = {len(long_thresholds)*len(short_thresholds)} combinations")
    print(f"Across {num_windows} independent windows...\n")

    for long_th in long_thresholds:
        for short_th in short_thresholds:
            print(f"Testing LONG={long_th:.2f}, SHORT={short_th:.2f}... ", end='', flush=True)

            window_results = []
            for window_idx in range(num_windows):
                start_idx = window_idx * window_size
                if start_idx + window_size > len(df_features):
                    break

                result = run_backtest_window(df_features, start_idx, window_size, long_th, short_th)
                window_results.append(result)

            # Aggregate metrics
            avg_total_trades = np.mean([r['total_trades'] for r in window_results])
            avg_win_rate = np.mean([r['win_rate'] for r in window_results])
            avg_return = np.mean([r['total_return'] for r in window_results])
            avg_sharpe = np.mean([r['sharpe'] for r in window_results])

            results.append({
                'long_threshold': long_th,
                'short_threshold': short_th,
                'avg_total_trades': avg_total_trades,
                'avg_win_rate': avg_win_rate,
                'avg_return_pct': avg_return * 100,
                'avg_sharpe': avg_sharpe
            })

            print(f"Trades: {avg_total_trades:.1f}, WR: {avg_win_rate*100:.1f}%, Return: {avg_return*100:.2f}%, Sharpe: {avg_sharpe:.3f}")

    # Save results
    results_df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = PROJECT_ROOT / "results" / f"entry_threshold_optimization_{timestamp}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n✅ Results saved to: {output_file}")

    # Find best configurations
    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS")
    print("="*80 + "\n")

    # Best by Sharpe
    best_sharpe = results_df.nlargest(5, 'avg_sharpe')
    print("🏆 Best by Sharpe Ratio:")
    for idx, row in best_sharpe.iterrows():
        print(f"   LONG={row['long_threshold']:.2f}, SHORT={row['short_threshold']:.2f}: "
              f"Sharpe {row['avg_sharpe']:.3f}, WR {row['avg_win_rate']*100:.1f}%, "
              f"Trades {row['avg_total_trades']:.1f}, Return {row['avg_return_pct']:.2f}%")

    # Best by Win Rate
    best_wr = results_df.nlargest(5, 'avg_win_rate')
    print("\n✅ Best by Win Rate:")
    for idx, row in best_wr.iterrows():
        print(f"   LONG={row['long_threshold']:.2f}, SHORT={row['short_threshold']:.2f}: "
              f"WR {row['avg_win_rate']*100:.1f}%, Sharpe {row['avg_sharpe']:.3f}, "
              f"Trades {row['avg_total_trades']:.1f}, Return {row['avg_return_pct']:.2f}%")

    # Best by Return
    best_return = results_df.nlargest(5, 'avg_return_pct')
    print("\n💰 Best by Return:")
    for idx, row in best_return.iterrows():
        print(f"   LONG={row['long_threshold']:.2f}, SHORT={row['short_threshold']:.2f}: "
              f"Return {row['avg_return_pct']:.2f}%, Sharpe {row['avg_sharpe']:.3f}, "
              f"WR {row['avg_win_rate']*100:.1f}%, Trades {row['avg_total_trades']:.1f}")

    # Current baseline (LONG 0.65, SHORT 0.70)
    baseline = results_df[(results_df['long_threshold'] == 0.65) & (results_df['short_threshold'] == 0.70)]
    if not baseline.empty:
        print("\n📊 Current Baseline (LONG 0.65, SHORT 0.70):")
        row = baseline.iloc[0]
        print(f"   Trades: {row['avg_total_trades']:.1f}, WR: {row['avg_win_rate']*100:.1f}%, "
              f"Return: {row['avg_return_pct']:.2f}%, Sharpe: {row['avg_sharpe']:.3f}")

    print("\n" + "="*80)
    return results_df

if __name__ == "__main__":
    results = optimize_entry_thresholds()