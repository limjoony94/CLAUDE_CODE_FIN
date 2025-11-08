"""
Threshold-Frequency Optimization (Approach #18)

비판적 발견:
- Approach #17은 수학적으로 수익성 있지만
- 거래 빈도가 너무 적음 (월 2.7 trades)
- 실용성 부족!

새로운 접근:
1. Threshold를 낮춰서 거래 빈도 증가
2. 승률과 거래 빈도의 균형 찾기
3. Expected Value가 여전히 양수인지 확인
4. 실용적인 거래 빈도 달성 (월 10+ trades)

테스트 파라미터:
- Thresholds: 0.3, 0.4, 0.5, 0.6, 0.7 (원래), 0.8
- Optimal R:R: SL 1.5%, TP 6.0% (Approach #17에서 발견)
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

WINDOW_SIZE = 1440  # 5 days
INITIAL_CAPITAL = 10000.0
POSITION_SIZE_PCT = 0.95
MAX_HOLDING_HOURS = 4
TRANSACTION_COST = 0.0002

# Optimal R:R from Approach #17
STOP_LOSS = 0.015  # 1.5%
TAKE_PROFIT = 0.06  # 6.0%


def backtest_with_threshold(df, model, feature_columns, threshold):
    """Backtest with specific threshold"""
    capital = INITIAL_CAPITAL
    position = None
    trades = []

    for i in range(len(df)):
        current_price = df['close'].iloc[i]

        # Manage position
        if position is not None:
            entry_idx = position['entry_idx']
            entry_price = position['entry_price']
            hours_held = (i - entry_idx) / 12

            # SHORT P&L
            pnl_pct = (entry_price - current_price) / entry_price

            # Exit conditions
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
                pnl_usd -= (entry_cost + exit_cost)

                capital += pnl_usd

                trades.append({
                    'entry_idx': entry_idx,
                    'exit_idx': i,
                    'pnl_pct': pnl_pct,
                    'pnl_usd': pnl_usd,
                    'exit_reason': exit_reason,
                    'short_prob': position['short_prob']
                })

                position = None

        # Entry
        if position is None and i < len(df) - 1:
            features = df[feature_columns].iloc[i:i+1].values

            if np.isnan(features).any():
                continue

            probs = model.predict_proba(features)[0]
            short_prob = probs[2]

            if short_prob >= threshold:
                position_value = capital * POSITION_SIZE_PCT
                quantity = position_value / current_price

                position = {
                    'entry_idx': i,
                    'entry_price': current_price,
                    'quantity': quantity,
                    'short_prob': short_prob
                }

    # Metrics
    if len(trades) == 0:
        return None

    winning_trades = [t for t in trades if t['pnl_usd'] > 0]
    win_rate = len(winning_trades) / len(trades)

    avg_win = np.mean([t['pnl_pct'] for t in winning_trades]) if winning_trades else 0
    avg_loss = np.mean([t['pnl_pct'] for t in trades if t['pnl_usd'] <= 0]) if len(trades) > len(winning_trades) else 0

    expected_value = win_rate * avg_win + (1 - win_rate) * avg_loss
    total_return_pct = ((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100

    return {
        'num_trades': len(trades),
        'win_rate': win_rate * 100,
        'avg_win_pct': avg_win * 100,
        'avg_loss_pct': avg_loss * 100,
        'expected_value_pct': expected_value * 100,
        'total_return_pct': total_return_pct
    }


def rolling_window_test(df, model, feature_columns, threshold):
    """Test with rolling windows"""
    windows = []
    start_idx = 0

    while start_idx + WINDOW_SIZE <= len(df):
        end_idx = start_idx + WINDOW_SIZE
        window_df = df.iloc[start_idx:end_idx].reset_index(drop=True)

        metrics = backtest_with_threshold(window_df, model, feature_columns, threshold)

        if metrics:
            windows.append(metrics)

        start_idx += WINDOW_SIZE

    if not windows:
        return None

    # Aggregate
    return {
        'num_trades_avg': np.mean([w['num_trades'] for w in windows]),
        'win_rate_avg': np.mean([w['win_rate'] for w in windows]),
        'expected_value_avg': np.mean([w['expected_value_pct'] for w in windows]),
        'total_return_avg': np.mean([w['total_return_pct'] for w in windows]),
        'num_windows': len(windows)
    }


def main():
    """Main optimization"""
    print("="*80)
    print("Threshold-Frequency Optimization (Approach #18)")
    print("="*80)
    print("\n비판적 발견:")
    print("  Approach #17: 수익성 있지만 거래 너무 적음 (월 2.7 trades)")
    print("  문제: 실용성 부족")
    print("")
    print("해결책:")
    print("  Threshold 낮춰서 거래 빈도 증가")
    print("  Expected Value 유지하면서 실용적 빈도 달성")
    print("="*80)

    # Load model
    model_file = MODELS_DIR / "xgboost_v4_phase4_3class_lookahead3_thresh3.pkl"
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    print(f"\n✅ Model loaded")

    # Load features
    feature_file = MODELS_DIR / "xgboost_v4_phase4_3class_lookahead3_thresh3_features.txt"
    with open(feature_file, 'r') as f:
        feature_columns = [line.strip() for line in f.readlines()]
    print(f"✅ Features loaded: {len(feature_columns)}")

    # Load data
    data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
    df = pd.read_csv(data_file)

    print("\n계산 중...")
    df = calculate_features(df)
    adv_features = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)
    df = adv_features.calculate_all_features(df)
    df = df.ffill().dropna()
    print(f"✅ Data prepared: {len(df)} candles")

    # Test thresholds
    print(f"\n" + "="*80)
    print("Testing Different Thresholds")
    print("="*80)
    print(f"\nOptimal R:R (from Approach #17): SL {STOP_LOSS*100:.1f}%, TP {TAKE_PROFIT*100:.1f}%")

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    results = []

    print(f"\n{'Threshold':<12} {'Trades/5d':<12} {'Trades/mo':<12} {'Win%':<10} {'EV%':<10} {'Return%':<12} {'Status':<15}")
    print("-" * 90)

    for threshold in thresholds:
        metrics = rolling_window_test(df, model, feature_columns, threshold)

        if metrics:
            trades_per_month = metrics['num_trades_avg'] * 6  # 30 days / 5 days
            monthly_return = metrics['expected_value_avg'] * metrics['num_trades_avg'] * 6

            status = "✅ Profitable" if metrics['expected_value_avg'] > 0 else "❌ Loss"

            if trades_per_month >= 10 and metrics['expected_value_avg'] > 0:
                status = "✅✅ Optimal"
            elif trades_per_month < 5:
                status = "⚠️ Too Few"

            print(f"{threshold:<12.2f} {metrics['num_trades_avg']:<12.1f} {trades_per_month:<12.1f} "
                  f"{metrics['win_rate_avg']:<10.1f} {metrics['expected_value_avg']:<10.3f} "
                  f"{monthly_return:<12.2f} {status:<15}")

            results.append({
                'threshold': threshold,
                'trades_per_5d': metrics['num_trades_avg'],
                'trades_per_month': trades_per_month,
                'win_rate': metrics['win_rate_avg'],
                'expected_value': metrics['expected_value_avg'],
                'monthly_return': monthly_return
            })

    # Find optimal
    valid_results = [r for r in results if r['expected_value'] > 0 and r['trades_per_month'] >= 10]

    print(f"\n" + "="*80)
    print("🎯 OPTIMAL CONFIGURATION")
    print("="*80)

    if valid_results:
        # Sort by monthly return
        best = max(valid_results, key=lambda x: x['monthly_return'])

        print(f"\nBest Balance (Frequency + Profitability):")
        print(f"  Threshold: {best['threshold']:.2f}")
        print(f"  Trades per month: {best['trades_per_month']:.1f}")
        print(f"  Win Rate: {best['win_rate']:.1f}%")
        print(f"  Expected Value: {best['expected_value']:.3f}% per trade")
        print(f"  Monthly Return: {best['monthly_return']:+.2f}%")

        print(f"\n✅ SUCCESS! Practical SHORT strategy achieved!")
        print(f"\n비판적 사고 검증:")
        print(f"  - Approach #17: 수익성 O, 실용성 X (월 2.7 trades)")
        print(f"  - Approach #18: 수익성 O, 실용성 O (월 {best['trades_per_month']:.1f} trades)")

    else:
        print(f"\n⚠️ No configuration meets criteria:")
        print(f"  - Expected Value > 0")
        print(f"  - Trades per month >= 10")

        # Show best by EV
        if results:
            best_ev = max(results, key=lambda x: x['expected_value'])
            print(f"\nBest Expected Value:")
            print(f"  Threshold: {best_ev['threshold']:.2f}")
            print(f"  Trades/month: {best_ev['trades_per_month']:.1f}")
            print(f"  EV: {best_ev['expected_value']:.3f}%")

    # Save
    output_file = RESULTS_DIR / "threshold_frequency_optimization.csv"
    pd.DataFrame(results).to_csv(output_file, index=False)
    print(f"\n✅ Results saved: {output_file}")

    print(f"\nProgress:")
    print(f"  #1-16: Win rate optimization → Failed")
    print(f"  #17: Profitability optimization → Success but impractical")
    print(f"  #18: Frequency-Profit balance → {'SUCCESS!' if valid_results else 'Testing...'}")

    return results


if __name__ == "__main__":
    results = main()
