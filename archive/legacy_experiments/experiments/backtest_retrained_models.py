"""
Backtest ENHANCED EXIT Models with Feature Engineering

Purpose: Test enhanced models (22 features) with proper (non-inverted) logic
Compare to:
- Baseline: Inverted logic (+11.60% return, 75.6% win rate)
- Basic: 3-feature models (+12.64% return, 64.2% win rate)

Models:
- LONG EXIT: xgboost_long_exit_improved_20251016_175554.pkl (22 features)
- SHORT EXIT: xgboost_short_exit_improved_20251016_180207.pkl (22 features)

Features: Enhanced market context (volume, momentum, volatility, RSI/MACD dynamics)
Logic: NORMAL (Exit when prob >= threshold)
- NOT inverted (models trained with correct 2of3 labels)

Author: Claude Code
Date: 2025-10-16
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
from scripts.experiments.retrain_exit_models_improved import prepare_exit_features

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"

WINDOW_SIZE = 1440
INITIAL_CAPITAL = 10000.0
POSITION_SIZE_PCT = 0.95
ENTRY_THRESHOLD = 0.7
TRANSACTION_COST = 0.0002
MAX_HOLDING_HOURS = 8


def classify_market_regime(df_window):
    """Classify market regime for analysis"""
    start_price = df_window['close'].iloc[0]
    end_price = df_window['close'].iloc[-1]
    return_pct = ((end_price / start_price) - 1) * 100
    if return_pct > 3.0:
        return "Bull"
    elif return_pct < -2.0:
        return "Bear"
    else:
        return "Sideways"


def backtest_with_normal_exit(df, models, scalers, features, exit_threshold):
    """
    Backtest with NORMAL exit logic (NOT inverted)
    Exit when probability is HIGH (as intended)

    This tests retrained models with proper 2of3 labeling
    """
    capital = INITIAL_CAPITAL
    position = None
    trades = []

    for i in range(len(df)):
        current_price = df['close'].iloc[i]

        # === EXIT LOGIC ===
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

            # Get EXIT probability
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

            # EXIT decision (NORMAL LOGIC: high prob = good exit)
            exit_reason = None
            if exit_prob is not None:
                # NORMAL LOGIC: Exit when probability is HIGH
                if exit_prob >= exit_threshold:
                    exit_reason = "Normal ML Exit"
                elif hours_held >= MAX_HOLDING_HOURS:
                    exit_reason = "Max Holding"

            # Execute exit
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

        # === ENTRY LOGIC ===
        if position is None and i < len(df) - 1:
            # Get LONG entry probability
            long_row = df[features['long_entry']].iloc[i:i+1].values
            if np.isnan(long_row).any():
                long_prob = 0
            else:
                long_scaled = scalers['long_entry'].transform(long_row)
                long_prob = models['long_entry'].predict_proba(long_scaled)[0][1]

            # Get SHORT entry probability
            short_row = df[features['short_entry']].iloc[i:i+1].values
            if np.isnan(short_row).any():
                short_prob = 0
            else:
                short_scaled = scalers['short_entry'].transform(short_row)
                short_prob = models['short_entry'].predict_proba(short_scaled)[0][1]

            # Entry decision
            side = None
            probability = None

            if long_prob >= ENTRY_THRESHOLD and short_prob < ENTRY_THRESHOLD:
                side = 'LONG'
                probability = long_prob
            elif short_prob >= ENTRY_THRESHOLD and long_prob < ENTRY_THRESHOLD:
                side = 'SHORT'
                probability = short_prob
            elif long_prob >= ENTRY_THRESHOLD and short_prob >= ENTRY_THRESHOLD:
                # Both signals: choose stronger
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

    # Calculate metrics
    total_return_pct = ((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    winning_trades = [t for t in trades if t['pnl_usd'] > 0]
    win_rate = (len(winning_trades) / len(trades)) * 100

    long_trades = [t for t in trades if t['side'] == 'LONG']
    short_trades = [t for t in trades if t['side'] == 'SHORT']

    ml_exits = [t for t in trades if t['exit_reason'] == 'Normal ML Exit']
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
    print("BACKTEST: Retrained EXIT Models (Normal Logic)")
    print("=" * 80)
    print()
    print("Models: 2of3 Scoring System (LONG + SHORT)")
    print("Logic: NORMAL (Exit when prob >= threshold)")
    print("Baseline: Inverted logic (+11.60% return, 75.6% win rate)")
    print()

    # Load models
    print("Loading models...")
    models = {}
    scalers = {}
    features = {}

    # ENTRY models (unchanged)
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

    # RETRAINED EXIT models - ENHANCED (22 features)
    with open(MODELS_DIR / "xgboost_long_exit_improved_20251016_175554.pkl", 'rb') as f:
        models['long_exit'] = pickle.load(f)
    with open(MODELS_DIR / "xgboost_long_exit_improved_20251016_175554_scaler.pkl", 'rb') as f:
        scalers['long_exit'] = pickle.load(f)
    with open(MODELS_DIR / "xgboost_long_exit_improved_20251016_175554_features.txt", 'r') as f:
        features['long_exit'] = [line.strip() for line in f]

    with open(MODELS_DIR / "xgboost_short_exit_improved_20251016_180207.pkl", 'rb') as f:
        models['short_exit'] = pickle.load(f)
    with open(MODELS_DIR / "xgboost_short_exit_improved_20251016_180207_scaler.pkl", 'rb') as f:
        scalers['short_exit'] = pickle.load(f)
    with open(MODELS_DIR / "xgboost_short_exit_improved_20251016_180207_features.txt", 'r') as f:
        features['short_exit'] = [line.strip() for line in f]

    print("✅ Models loaded")
    print()
    print("EXIT Models:")
    print(f"  LONG: {len(features['long_exit'])} features - {features['long_exit']}")
    print(f"  SHORT: {len(features['short_exit'])} features - {features['short_exit']}")

    # Load data
    print("\nLoading data...")
    df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_max.csv")
    df = calculate_features(df)
    adv = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)
    df = adv.calculate_all_features(df)
    sell = SellSignalFeatures()
    df = sell.calculate_all_features(df)

    # Calculate enhanced EXIT features (2025-10-16)
    df = prepare_exit_features(df)

    df = df.ffill().dropna()
    print(f"✅ {len(df):,} candles loaded")

    # Test NORMAL thresholds
    print("\n" + "=" * 80)
    print("Testing NORMAL EXIT Thresholds (prob >= X)")
    print("=" * 80)

    normal_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    results = []

    for thresh in normal_thresholds:
        print(f"\nNormal Threshold: >= {thresh}")

        all_results = []
        start_idx = 0
        num_windows = 0

        while start_idx + WINDOW_SIZE <= len(df):
            end_idx = start_idx + WINDOW_SIZE
            window_df = df.iloc[start_idx:end_idx].reset_index(drop=True)

            metrics = backtest_with_normal_exit(window_df, models, scalers, features, thresh)
            if metrics:
                all_results.append(metrics)

            num_windows += 1
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

            print(f"  Windows tested: {num_windows}")
            print(f"  Return: {avg_return:+.2f}%")
            print(f"  Win Rate: {avg_win_rate:.1f}%")
            print(f"  Trades/window: {avg_trades:.1f}")
            print(f"  Sharpe: {avg_sharpe:.3f}")
            print(f"  ML Exit Rate: {avg_ml_exit:.1f}%")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Retrained Models Performance")
    print("=" * 80)

    if len(results) > 0:
        results_df = pd.DataFrame(results)
        print("\n" + results_df.to_string(index=False))

        best_retrained = results_df.iloc[results_df['return'].idxmax()]
        print(f"\n🏆 Best Retrained Model Configuration:")
        print(f"  Threshold: >= {best_retrained['threshold']}")
        print(f"  Return: {best_retrained['return']:+.2f}%")
        print(f"  Win Rate: {best_retrained['win_rate']:.1f}%")
        print(f"  Trades/window: {best_retrained['trades_per_window']:.1f}")
        print(f"  Sharpe: {best_retrained['sharpe']:.3f}")
        print(f"  ML Exit Rate: {best_retrained['ml_exit_rate']:.1f}%")

    # Comparison to baseline
    print("\n" + "=" * 80)
    print("COMPARISON: Retrained vs Inverted Baseline")
    print("=" * 80)

    print("\nInverted Logic Baseline (Current Production):")
    print("  Threshold: <= 0.5")
    print("  Return: +11.60% per window")
    print("  Win Rate: 75.6%")
    print("  Trades/window: 92.2 (~19/day)")
    print("  Sharpe: 9.82")
    print("  ML Exit Rate: 100.0%")

    if len(results) > 0:
        best_retrained = results_df.iloc[results_df['return'].idxmax()]

        print(f"\nRetrained Models (Best: >= {best_retrained['threshold']}):")
        print(f"  Return: {best_retrained['return']:+.2f}% per window")
        print(f"  Win Rate: {best_retrained['win_rate']:.1f}%")
        print(f"  Trades/window: {best_retrained['trades_per_window']:.1f}")
        print(f"  Sharpe: {best_retrained['sharpe']:.3f}")
        print(f"  ML Exit Rate: {best_retrained['ml_exit_rate']:.1f}%")

        # Calculate differences
        baseline_return = 11.60
        baseline_win_rate = 75.6
        baseline_sharpe = 9.82

        return_diff = best_retrained['return'] - baseline_return
        win_rate_diff = best_retrained['win_rate'] - baseline_win_rate
        sharpe_diff = best_retrained['sharpe'] - baseline_sharpe

        print(f"\n📊 Differences:")
        print(f"  Return: {return_diff:+.2f}% {'✅ Better' if return_diff > 0 else '❌ Worse'}")
        print(f"  Win Rate: {win_rate_diff:+.1f}% {'✅ Better' if win_rate_diff > 0 else '❌ Worse'}")
        print(f"  Sharpe: {sharpe_diff:+.3f} {'✅ Better' if sharpe_diff > 0 else '❌ Worse'}")

    # Decision recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    if len(results) > 0:
        best_retrained = results_df.iloc[results_df['return'].idxmax()]
        baseline_return = 11.60

        if best_retrained['return'] >= baseline_return:
            print("\n✅ DEPLOY RETRAINED MODELS")
            print(f"  Retrained models outperform baseline: {best_retrained['return']:+.2f}% >= {baseline_return:+.2f}%")
            print(f"  Recommended threshold: >= {best_retrained['threshold']}")
            print("\n  Next steps:")
            print("  1. Update production bot to use retrained models")
            print("  2. Monitor performance for 24-48 hours")
            print("  3. Consider feature engineering for further improvement")
        elif best_retrained['return'] >= baseline_return * 0.9:  # Within 10%
            print("\n⚠️ MARGINAL PERFORMANCE")
            print(f"  Retrained: {best_retrained['return']:+.2f}% vs Baseline: {baseline_return:+.2f}%")
            print(f"  Difference: {best_retrained['return'] - baseline_return:+.2f}% (within 10%)")
            print("\n  Options:")
            print("  A. Deploy retrained (better logic, similar performance)")
            print("  B. Feature engineering first (add position-specific features)")
            print("  C. Keep inverted logic (working well, lower risk)")
        else:
            print("\n❌ FEATURE ENGINEERING NEEDED")
            print(f"  Retrained underperforms: {best_retrained['return']:+.2f}% < {baseline_return:+.2f}%")
            print(f"  Gap: {baseline_return - best_retrained['return']:.2f}%")
            print("\n  Root cause: Limited features (only rsi, macd, macd_signal)")
            print("  Required action: Add position-specific features")
            print("\n  Recommended features:")
            print("  - current_pnl_pct (profit/loss state)")
            print("  - pnl_from_peak/trough (give-back detection)")
            print("  - holding_time (time-based exits)")
            print("  - entry_signal_strength (entry context)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
