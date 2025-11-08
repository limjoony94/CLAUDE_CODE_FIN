"""
Test Inverted EXIT Logic

Hypothesis: EXIT models learned the OPPOSITE
Solution: Use LOW probability as exit signal (invert logic)

Original: Exit when prob >= 0.7 (BAD!)
Inverted: Exit when prob <= 0.3 (opposite)
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


def backtest_with_inverted_exit(df, models, scalers, features, exit_threshold_low):
    """
    Backtest with INVERTED exit logic
    Exit when probability is LOW (opposite of normal)
    """
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
                # INVERTED LOGIC: Exit when probability is LOW
                if exit_prob <= exit_threshold_low:
                    exit_reason = "Inverted ML Exit"
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

    if len(trades) == 0:
        return {}

    total_return_pct = ((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    winning_trades = [t for t in trades if t['pnl_usd'] > 0]
    win_rate = (len(winning_trades) / len(trades)) * 100

    long_trades = [t for t in trades if t['side'] == 'LONG']
    short_trades = [t for t in trades if t['side'] == 'SHORT']

    ml_exits = [t for t in trades if t['exit_reason'] == 'Inverted ML Exit']
    ml_exit_rate = (len(ml_exits) / len(trades)) * 100

    if len(trades) > 1:
        returns = [t['pnl_pct'] for t in trades]
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
    else:
        sharpe = 0

    return {
        'total_return_pct': total_return_pct,
        'num_trades': len(trades),
        'num_long': len(long_trades),
        'num_short': len(short_trades),
        'win_rate': win_rate,
        'sharpe_ratio': sharpe,
        'ml_exit_rate': ml_exit_rate
    }


def main():
    print("=" * 80)
    print("TEST: Inverted EXIT Logic")
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

    print("✅ Models loaded")

    # Load data
    print("\nLoading data...")
    df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_max.csv")
    df = calculate_features(df)
    adv = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)
    df = adv.calculate_all_features(df)
    sell = SellSignalFeatures()
    df = sell.calculate_all_features(df)
    df = df.ffill().dropna()
    print(f"✅ {len(df):,} candles")

    # Test INVERTED thresholds
    print("\n" + "=" * 80)
    print("Testing INVERTED EXIT Thresholds (prob <= X)")
    print("=" * 80)

    # For inverted logic, threshold 0.5 means "exit when prob <= 0.5"
    # which includes the good signals in 0.0-0.5 range
    inverted_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    results = []

    for thresh in inverted_thresholds:
        print(f"\nInverted Threshold: <= {thresh}")

        all_results = []
        start_idx = 0
        while start_idx + WINDOW_SIZE <= len(df):
            end_idx = start_idx + WINDOW_SIZE
            window_df = df.iloc[start_idx:end_idx].reset_index(drop=True)

            metrics = backtest_with_inverted_exit(window_df, models, scalers, features, thresh)
            if metrics:
                all_results.append(metrics)

            start_idx += WINDOW_SIZE

        if all_results:
            avg_return = np.mean([r['total_return_pct'] for r in all_results])
            avg_win_rate = np.mean([r['win_rate'] for r in all_results])
            avg_trades = np.mean([r['num_trades'] for r in all_results])
            avg_sharpe = np.mean([r['sharpe_ratio'] for r in all_results])
            avg_ml_exit = np.mean([r['ml_exit_rate'] for r in all_results])

            results.append({
                'threshold': thresh,
                'return': avg_return,
                'win_rate': avg_win_rate,
                'trades_per_window': avg_trades,
                'sharpe': avg_sharpe,
                'ml_exit_rate': avg_ml_exit
            })

            print(f"  Return: {avg_return:+.2f}%")
            print(f"  Win Rate: {avg_win_rate:.1f}%")
            print(f"  Trades/window: {avg_trades:.1f}")
            print(f"  Sharpe: {avg_sharpe:.3f}")
            print(f"  ML Exit Rate: {avg_ml_exit:.1f}%")

    # Summary
    print("\n" + "=" * 80)
    print("INVERTED EXIT Threshold Comparison")
    print("=" * 80)

    results_df = pd.DataFrame(results)
    print("\n" + results_df.to_string(index=False))

    # Compare to original (non-inverted) results
    print("\n" + "=" * 80)
    print("Comparison: Inverted vs Original")
    print("=" * 80)

    print("\nOriginal EXIT Logic (prob >= threshold):")
    print("  Threshold 0.5: +4.05% return, 40.0% win rate")
    print("  Threshold 0.7: -9.54% return, 33.5% win rate (WORST)")
    print("  Threshold 0.9: +2.88% return, 56.1% win rate")

    if len(results) > 0:
        best_inverted = results_df.iloc[results_df['return'].idxmax()]
        print(f"\nBest Inverted EXIT Logic (prob <= {best_inverted['threshold']}):")
        print(f"  Return: {best_inverted['return']:+.2f}%")
        print(f"  Win Rate: {best_inverted['win_rate']:.1f}%")
        print(f"  Trades/window: {best_inverted['trades_per_window']:.1f}")
        print(f"  Sharpe: {best_inverted['sharpe']:.3f}")

        # Calculate improvement
        original_best = 4.05  # Original threshold 0.5
        improvement = best_inverted['return'] - original_best
        print(f"\n📊 Improvement over original: {improvement:+.2f}%")

    print("\n" + "=" * 80)
    print("Conclusion")
    print("=" * 80)

    print("\nHypothesis: EXIT models learned OPPOSITE of intended")
    print("  Original labels: Peak/Trough = exit signal")
    print("  Model learned: High prob = BAD exit, Low prob = GOOD exit")
    print()
    print("If inverted logic performs better:")
    print("  ✅ Confirms model is inverted")
    print("  → Quick fix: Use inverted logic (prob <= X)")
    print("  → Proper fix: Retrain with corrected labels")
    print()
    print("If inverted logic performs same/worse:")
    print("  ❌ Model has fundamental calibration issue")
    print("  → Must retrain with better labeling methodology")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
