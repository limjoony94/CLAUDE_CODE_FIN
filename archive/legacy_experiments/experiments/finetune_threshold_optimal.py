"""
Threshold Fine-Tuning - Approach #20

비판적 발견:
- Threshold 0.6을 검증했지만
- 주변 값 (0.55, 0.65)는 테스트 안함
- 작은 차이가 큰 성능 변화 가능

Critical Question:
"0.6이 정말 최적인가? 0.55나 0.65가 더 나을 수도?"

이 스크립트:
- 0.55, 0.60, 0.65 비교 테스트
- 진짜 최적 threshold 찾기
- Monthly return 기준으로 순위 매기기
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

# Configuration
STOP_LOSS = 0.015  # 1.5%
TAKE_PROFIT = 0.06  # 6.0%
WINDOW_SIZE = 1440  # 5 days
INITIAL_CAPITAL = 10000.0
POSITION_SIZE_PCT = 0.95
MAX_HOLDING_HOURS = 4
TRANSACTION_COST = 0.0002


def backtest_threshold(df, model, feature_columns, threshold):
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

                # Transaction costs
                entry_cost = entry_price * quantity * TRANSACTION_COST
                exit_cost = current_price * quantity * TRANSACTION_COST
                pnl_usd -= (entry_cost + exit_cost)

                capital += pnl_usd

                trades.append({
                    'entry_idx': entry_idx,
                    'exit_idx': i,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl_pct': pnl_pct * 100,
                    'pnl_usd': pnl_usd,
                    'exit_reason': exit_reason,
                    'short_prob': position['short_prob'],
                    'hours_held': hours_held
                })

                position = None

        # Look for entry
        if position is None and i < len(df) - 1:
            features = df[feature_columns].iloc[i:i+1].values

            if np.isnan(features).any():
                continue

            probs = model.predict_proba(features)[0]
            short_prob = probs[2]  # Class 2 = SHORT

            if short_prob >= threshold:
                position_value = capital * POSITION_SIZE_PCT
                quantity = position_value / current_price

                position = {
                    'entry_idx': i,
                    'entry_price': current_price,
                    'quantity': quantity,
                    'short_prob': short_prob
                }

    return trades, capital


def rolling_window_test(df, model, feature_columns, threshold):
    """Test with rolling windows"""
    windows = []
    start_idx = 0

    while start_idx + WINDOW_SIZE <= len(df):
        end_idx = start_idx + WINDOW_SIZE
        window_df = df.iloc[start_idx:end_idx].reset_index(drop=True)

        trades, final_capital = backtest_threshold(window_df, model, feature_columns, threshold)

        if len(trades) > 0:
            winning_trades = [t for t in trades if t['pnl_usd'] > 0]
            win_rate = len(winning_trades) / len(trades) * 100

            avg_win = np.mean([t['pnl_pct'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['pnl_pct'] for t in trades if t['pnl_usd'] <= 0]) if len(trades) > len(winning_trades) else 0

            expected_value = (win_rate/100) * avg_win + (1 - win_rate/100) * avg_loss
            total_return = ((final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100

            windows.append({
                'num_trades': len(trades),
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'expected_value': expected_value,
                'total_return': total_return,
                'final_capital': final_capital
            })

        start_idx += WINDOW_SIZE

    return windows


def main():
    """Fine-tune threshold around 0.6"""
    print("="*80)
    print("THRESHOLD FINE-TUNING - Approach #20")
    print("="*80)
    print("\n❓ Critical Question:")
    print("   Is 0.6 really optimal? What about 0.55 or 0.65?")
    print("\n🔍 Testing Strategy:")
    print("   - Test: 0.55, 0.60, 0.65")
    print("   - Metric: Monthly return (EV × frequency)")
    print("   - Find true optimal threshold")
    print("="*80)

    # Load model
    model_file = MODELS_DIR / "xgboost_v4_phase4_3class_lookahead3_thresh3.pkl"
    print(f"\n✅ Loading model: {model_file.name}")
    with open(model_file, 'rb') as f:
        model = pickle.load(f)

    # Load features
    feature_file = MODELS_DIR / "xgboost_v4_phase4_3class_lookahead3_thresh3_features.txt"
    with open(feature_file, 'r') as f:
        feature_columns = [line.strip() for line in f.readlines()]
    print(f"✅ Features loaded: {len(feature_columns)}")

    # Load data
    data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
    print(f"✅ Loading data: {data_file.name}")
    df = pd.read_csv(data_file)

    # Calculate features
    print("\n⚙️  Calculating features...")
    df = calculate_features(df)
    adv_features = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)
    df = adv_features.calculate_all_features(df)
    df = df.ffill().dropna()
    print(f"✅ Data prepared: {len(df):,} candles")

    # Test thresholds
    thresholds = [0.55, 0.60, 0.65]
    results = []

    print(f"\n" + "="*80)
    print("TESTING THRESHOLDS")
    print("="*80)

    for threshold in thresholds:
        print(f"\n🔬 Testing threshold {threshold}...")

        windows = rolling_window_test(df, model, feature_columns, threshold)

        if not windows:
            print(f"   ❌ No trades found")
            continue

        # Aggregate
        avg_trades = np.mean([w['num_trades'] for w in windows])
        avg_win_rate = np.mean([w['win_rate'] for w in windows])
        avg_ev = np.mean([w['expected_value'] for w in windows])
        avg_return = np.mean([w['total_return'] for w in windows])

        trades_per_month = avg_trades * 6  # 30 days / 5 days
        monthly_return = avg_ev * avg_trades * 6

        print(f"   Trades/5d: {avg_trades:.1f} → {trades_per_month:.1f}/month")
        print(f"   Win Rate: {avg_win_rate:.1f}%")
        print(f"   EV/trade: {avg_ev:+.3f}%")
        print(f"   Monthly Return: {monthly_return:+.2f}%")

        results.append({
            'threshold': threshold,
            'trades_per_5d': avg_trades,
            'trades_per_month': trades_per_month,
            'win_rate': avg_win_rate,
            'expected_value': avg_ev,
            'monthly_return': monthly_return,
            'total_trades': sum(w['num_trades'] for w in windows),
            'num_windows': len(windows),
            'profitable_windows': sum(1 for w in windows if w['expected_value'] > 0)
        })

    # Analysis
    print("\n" + "="*80)
    print("📊 COMPARISON ANALYSIS")
    print("="*80)

    if not results:
        print("\n❌ No results to compare!")
        return

    # Sort by monthly return
    results_sorted = sorted(results, key=lambda x: x['monthly_return'], reverse=True)

    print(f"\n{'Rank':<6} {'Threshold':<12} {'Trades/mo':<12} {'Win%':<10} {'EV%':<10} {'Monthly%':<12} {'Status':<15}")
    print("-" * 90)

    for rank, r in enumerate(results_sorted, 1):
        status = "🥇 OPTIMAL" if rank == 1 else "🥈 Runner-up" if rank == 2 else "🥉 Third"

        print(f"{rank:<6} {r['threshold']:<12.2f} {r['trades_per_month']:<12.1f} "
              f"{r['win_rate']:<10.1f} {r['expected_value']:<10.3f} "
              f"{r['monthly_return']:<12.2f} {status:<15}")

    # Detailed comparison
    best = results_sorted[0]

    print("\n" + "="*80)
    print("🎯 OPTIMAL THRESHOLD IDENTIFIED")
    print("="*80)

    print(f"\n🥇 Winner: Threshold {best['threshold']}")
    print(f"\nPerformance:")
    print(f"   Trades per month: {best['trades_per_month']:.1f}")
    print(f"   Win Rate: {best['win_rate']:.1f}%")
    print(f"   Expected Value: {best['expected_value']:+.3f}% per trade")
    print(f"   Monthly Return: {best['monthly_return']:+.2f}%")

    print(f"\nValidation:")
    print(f"   Total trades tested: {best['total_trades']}")
    print(f"   Windows tested: {best['num_windows']}")
    print(f"   Profitable windows: {best['profitable_windows']}/{best['num_windows']} ({best['profitable_windows']/best['num_windows']*100:.0f}%)")

    # Compare to others
    if len(results_sorted) > 1:
        print(f"\n📈 Improvement Over Alternatives:")

        for i, r in enumerate(results_sorted[1:], 2):
            improvement = ((best['monthly_return'] - r['monthly_return']) / abs(r['monthly_return'])) * 100 if r['monthly_return'] != 0 else 0

            print(f"\n   vs Threshold {r['threshold']}:")
            print(f"      Monthly return: {best['monthly_return']:+.2f}% vs {r['monthly_return']:+.2f}%")
            print(f"      Improvement: {improvement:+.1f}%")
            print(f"      Trade frequency: {best['trades_per_month']:.1f} vs {r['trades_per_month']:.1f}")
            print(f"      Win rate: {best['win_rate']:.1f}% vs {r['win_rate']:.1f}%")

    # Critical thinking conclusion
    print("\n" + "="*80)
    print("🧠 CRITICAL THINKING CONCLUSION")
    print("="*80)

    if best['threshold'] == 0.60:
        print("\n✅ ORIGINAL CHOICE VALIDATED!")
        print("   Threshold 0.6 is indeed optimal")
        print("   No need to change configuration")
    else:
        print(f"\n🔄 OPTIMIZATION FOUND!")
        print(f"   Threshold {best['threshold']} is better than 0.6")
        print(f"   Recommendation: Update to {best['threshold']}")
        print(f"   Expected improvement: {((best['monthly_return'] - [r for r in results if r['threshold']==0.6][0]['monthly_return']) / abs([r for r in results if r['threshold']==0.6][0]['monthly_return'])) * 100:+.1f}%")

    print("\n비판적 사고 적용:")
    print("  1. 0.6이 최적이라고 가정했지만")
    print("  2. 주변 값들을 실제로 테스트")
    print(f"  3. 결과: Threshold {best['threshold']}가 최적 ({best['monthly_return']:+.2f}% monthly)")
    print("  4. 증거 기반 의사결정 완료 ✅")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
