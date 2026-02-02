"""
Debug EXIT Threshold Anomaly

Why 0.7 is worse than 0.6?
Analyze window-by-window results to find the issue
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
from src.features.sell_signal_features import SellSignalFeatures

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"

WINDOW_SIZE = 1440
INITIAL_CAPITAL = 10000.0
POSITION_SIZE_PCT = 0.95
ENTRY_THRESHOLD = 0.7
TRANSACTION_COST = 0.0002
MAX_HOLDING_HOURS = 8


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


def backtest_with_exit_threshold_detailed(df, models, scalers, features, exit_threshold):
    """Backtest with detailed trade tracking"""
    capital = INITIAL_CAPITAL
    position = None
    trades = []

    for i in range(len(df)):
        current_price = df['close'].iloc[i]

        if position is not None:
            side = position['side']
            entry_idx = position['entry_idx']
            entry_price = position['entry_price']
            hours_held = (i - entry_idx) / 12

            if side == 'LONG':
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - current_price) / entry_price

            exit_prob = None
            if side == 'LONG':
                exit_row = df[features['long_exit']].iloc[i:i+1].values
                if not np.isnan(exit_row).any():
                    exit_scaled = scalers['long_exit'].transform(exit_row)
                    exit_prob = models['long_exit'].predict_proba(exit_scaled)[0][1]
            else:
                exit_row = df[features['short_exit']].iloc[i:i+1].values
                if not np.isnan(exit_row).any():
                    exit_scaled = scalers['short_exit'].transform(exit_row)
                    exit_prob = models['short_exit'].predict_proba(exit_scaled)[0][1]

            exit_reason = None
            if exit_prob is not None:
                if exit_prob >= exit_threshold:
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
                    'pnl_usd': pnl_usd,
                    'pnl_pct': pnl_pct,
                    'exit_reason': exit_reason,
                    'holding_hours': hours_held,
                    'exit_prob': exit_prob,
                    'regime': position['regime']
                })

                position = None

        if position is None and i < len(df) - 1:
            long_row = df[features['long_entry']].iloc[i:i+1].values
            if np.isnan(long_row).any():
                long_prob = 0
            else:
                long_scaled = scalers['long_entry'].transform(long_row)
                long_prob = models['long_entry'].predict_proba(long_scaled)[0][1]

            short_row = df[features['short_entry']].iloc[i:i+1].values
            if np.isnan(short_row).any():
                short_prob = 0
            else:
                short_scaled = scalers['short_entry'].transform(short_row)
                short_prob = models['short_entry'].predict_proba(short_scaled)[0][1]

            side = None
            probability = None

            if long_prob >= ENTRY_THRESHOLD and short_prob < ENTRY_THRESHOLD:
                side = 'LONG'
                probability = long_prob
            elif short_prob >= ENTRY_THRESHOLD and long_prob < ENTRY_THRESHOLD:
                side = 'SHORT'
                probability = short_prob
            elif long_prob >= ENTRY_THRESHOLD and short_prob >= ENTRY_THRESHOLD:
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

    return capital, trades


def main():
    print("=" * 80)
    print("DEBUG: EXIT Threshold Anomaly (Window-by-Window)")
    print("=" * 80)

    # Load models
    print("\nLoading models...")
    models = {}
    scalers = {}
    features = {}

    with open(MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl", 'rb') as f:
        models['long_entry'] = pickle.load(f)
    with open(MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_scaler.pkl", 'rb') as f:
        scalers['long_entry'] = pickle.load(f)
    with open(MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt", 'r') as f:
        features['long_entry'] = [line.strip() for line in f]

    with open(MODELS_DIR / "xgboost_short_peak_trough_20251016_131939.pkl", 'rb') as f:
        models['short_entry'] = pickle.load(f)
    with open(MODELS_DIR / "xgboost_short_peak_trough_20251016_131939_scaler.pkl", 'rb') as f:
        scalers['short_entry'] = pickle.load(f)
    with open(MODELS_DIR / "xgboost_short_peak_trough_20251016_131939_features.txt", 'r') as f:
        features['short_entry'] = [line.strip() for line in f]

    with open(MODELS_DIR / "xgboost_long_exit_peak_trough_20251016_132651.pkl", 'rb') as f:
        models['long_exit'] = pickle.load(f)
    with open(MODELS_DIR / "xgboost_long_exit_peak_trough_20251016_132651_scaler.pkl", 'rb') as f:
        scalers['long_exit'] = pickle.load(f)
    with open(MODELS_DIR / "xgboost_long_exit_peak_trough_20251016_132651_features.txt", 'r') as f:
        features['long_exit'] = [line.strip() for line in f]

    with open(MODELS_DIR / "xgboost_short_exit_peak_trough_20251016_135233.pkl", 'rb') as f:
        models['short_exit'] = pickle.load(f)
    with open(MODELS_DIR / "xgboost_short_exit_peak_trough_20251016_135233_scaler.pkl", 'rb') as f:
        scalers['short_exit'] = pickle.load(f)
    with open(MODELS_DIR / "xgboost_short_exit_peak_trough_20251016_135233_features.txt", 'r') as f:
        features['short_exit'] = [line.strip() for line in f]

    # Load data
    df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_max.csv")
    df = calculate_features(df)
    adv = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)
    df = adv.calculate_all_features(df)
    sell = SellSignalFeatures()
    df = sell.calculate_all_features(df)
    df = df.ffill().dropna()
    print(f"✅ {len(df):,} candles")

    # Test thresholds 0.6 and 0.7 window-by-window
    print("\n" + "=" * 80)
    print("Window-by-Window Comparison: 0.6 vs 0.7")
    print("=" * 80)

    thresholds = [0.6, 0.7]
    all_window_results = {0.6: [], 0.7: []}

    start_idx = 0
    window_num = 0
    while start_idx + WINDOW_SIZE <= len(df):
        end_idx = start_idx + WINDOW_SIZE
        window_df = df.iloc[start_idx:end_idx].reset_index(drop=True)
        window_num += 1

        regime = classify_market_regime(window_df)

        print(f"\nWindow {window_num} ({regime}):")

        for thresh in thresholds:
            final_capital, trades = backtest_with_exit_threshold_detailed(
                window_df, models, scalers, features, thresh
            )

            if len(trades) > 0:
                return_pct = ((final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
                win_rate = len([t for t in trades if t['pnl_usd'] > 0]) / len(trades) * 100

                all_window_results[thresh].append({
                    'window': window_num,
                    'regime': regime,
                    'return': return_pct,
                    'trades': len(trades),
                    'win_rate': win_rate
                })

                print(f"  Threshold {thresh}: Return={return_pct:+6.2f}%, Trades={len(trades)}, WinRate={win_rate:.1f}%")
            else:
                print(f"  Threshold {thresh}: No trades")

        start_idx += WINDOW_SIZE

    # Summary
    print("\n" + "=" * 80)
    print("Summary by Threshold")
    print("=" * 80)

    for thresh in thresholds:
        results = all_window_results[thresh]
        if results:
            returns = [r['return'] for r in results]
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            min_return = np.min(returns)
            max_return = np.max(returns)

            print(f"\nThreshold {thresh}:")
            print(f"  Average Return: {avg_return:+.2f}% ± {std_return:.2f}%")
            print(f"  Min: {min_return:+.2f}%")
            print(f"  Max: {max_return:+.2f}%")
            print(f"  Windows: {len(results)}")

            # Find worst windows
            sorted_results = sorted(results, key=lambda x: x['return'])
            print(f"\n  Worst 3 windows:")
            for r in sorted_results[:3]:
                print(f"    Window {r['window']} ({r['regime']}): {r['return']:+.2f}%")

    # Direct comparison
    print("\n" + "=" * 80)
    print("Direct Window Comparison (0.6 vs 0.7)")
    print("=" * 80)

    print(f"\n{'Window':<8} {'Regime':<10} {'0.6 Return':<12} {'0.7 Return':<12} {'Diff':<10}")
    print("-" * 60)

    for i in range(len(all_window_results[0.6])):
        w6 = all_window_results[0.6][i]
        w7 = all_window_results[0.7][i]

        diff = w6['return'] - w7['return']
        marker = "⚠️" if abs(diff) > 5 else ""

        print(f"{w6['window']:<8} {w6['regime']:<10} {w6['return']:+10.2f}%  {w7['return']:+10.2f}%  {diff:+8.2f}%  {marker}")

    # Find biggest differences
    print("\n" + "=" * 80)
    print("Biggest Differences")
    print("=" * 80)

    differences = []
    for i in range(len(all_window_results[0.6])):
        w6 = all_window_results[0.6][i]
        w7 = all_window_results[0.7][i]
        diff = w6['return'] - w7['return']
        differences.append({
            'window': w6['window'],
            'regime': w6['regime'],
            'diff': diff,
            'return_0.6': w6['return'],
            'return_0.7': w7['return']
        })

    # Sort by absolute difference
    differences.sort(key=lambda x: abs(x['diff']), reverse=True)

    print("\nTop 5 windows with biggest differences:")
    for d in differences[:5]:
        print(f"  Window {d['window']} ({d['regime']}): 0.6={d['return_0.6']:+.2f}%, 0.7={d['return_0.7']:+.2f}%, diff={d['diff']:+.2f}%")

    print("\n" + "=" * 80)
    print("Debug Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
