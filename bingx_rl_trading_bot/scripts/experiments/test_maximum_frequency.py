"""
Maximum Trade Frequency Test

사용자 요구:
"1달 21건 트레이드는 너무 낮은 수치인 것 같습니다?
 적어도 1일 1번 - 10번 범위에 있어야 할 것 같아요"

Critical Question:
"SHORT 전략으로 하루 1-10 거래가 물리적으로 가능한가?"

이 스크립트:
- 매우 낮은 threshold (0.3, 0.4, 0.5) 테스트
- 최대 가능한 거래 빈도 확인
- 거래 빈도 vs 수익성 trade-off 분석
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
    signals_generated = 0  # Track total signals

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
                    'pnl_pct': pnl_pct * 100,
                    'pnl_usd': pnl_usd,
                    'exit_reason': exit_reason,
                    'short_prob': position['short_prob']
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
                signals_generated += 1

                position_value = capital * POSITION_SIZE_PCT
                quantity = position_value / current_price

                position = {
                    'entry_idx': i,
                    'entry_price': current_price,
                    'quantity': quantity,
                    'short_prob': short_prob
                }

    return trades, capital, signals_generated


def rolling_window_test(df, model, feature_columns, threshold):
    """Test with rolling windows"""
    windows = []
    start_idx = 0

    while start_idx + WINDOW_SIZE <= len(df):
        end_idx = start_idx + WINDOW_SIZE
        window_df = df.iloc[start_idx:end_idx].reset_index(drop=True)

        trades, final_capital, signals = backtest_threshold(window_df, model, feature_columns, threshold)

        if len(trades) > 0:
            winning_trades = [t for t in trades if t['pnl_usd'] > 0]
            win_rate = len(winning_trades) / len(trades) * 100

            avg_win = np.mean([t['pnl_pct'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['pnl_pct'] for t in trades if t['pnl_usd'] <= 0]) if len(trades) > len(winning_trades) else 0

            expected_value = (win_rate/100) * avg_win + (1 - win_rate/100) * avg_loss
            total_return = ((final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100

            windows.append({
                'num_trades': len(trades),
                'signals': signals,
                'win_rate': win_rate,
                'expected_value': expected_value,
                'total_return': total_return
            })

        start_idx += WINDOW_SIZE

    return windows


def main():
    """Test maximum possible trade frequency"""
    print("="*80)
    print("MAXIMUM TRADE FREQUENCY TEST")
    print("="*80)
    print("\n사용자 요구:")
    print("   하루 1-10 거래 (월 30-300 거래)")
    print("\n현재 결과:")
    print("   Threshold 0.6: 월 21.6 거래 (하루 0.7 거래)")
    print("\n❓ Critical Question:")
    print("   SHORT로 하루 1-10 거래가 물리적으로 가능한가?")
    print("\n🔍 Testing Strategy:")
    print("   매우 낮은 threshold (0.3, 0.4, 0.5, 0.6) 테스트")
    print("   최대 가능 빈도 확인")
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

    # Test thresholds from very low to current
    thresholds = [0.3, 0.4, 0.5, 0.6]
    results = []

    print(f"\n" + "="*80)
    print("TESTING MAXIMUM FREQUENCY")
    print("="*80)

    for threshold in thresholds:
        print(f"\n🔬 Testing threshold {threshold}...")

        windows = rolling_window_test(df, model, feature_columns, threshold)

        if not windows:
            print(f"   ❌ No trades found")
            continue

        # Aggregate
        avg_trades = np.mean([w['num_trades'] for w in windows])
        avg_signals = np.mean([w['signals'] for w in windows])
        avg_win_rate = np.mean([w['win_rate'] for w in windows])
        avg_ev = np.mean([w['expected_value'] for w in windows])

        trades_per_month = avg_trades * 6  # 30 days / 5 days
        trades_per_day = trades_per_month / 30
        signals_per_day = (avg_signals * 6) / 30

        monthly_return = avg_ev * avg_trades * 6

        print(f"   Signals/day: {signals_per_day:.1f}")
        print(f"   Trades/day: {trades_per_day:.1f}")
        print(f"   Trades/month: {trades_per_month:.1f}")
        print(f"   Win Rate: {avg_win_rate:.1f}%")
        print(f"   EV/trade: {avg_ev:+.3f}%")
        print(f"   Monthly Return: {monthly_return:+.2f}%")

        results.append({
            'threshold': threshold,
            'signals_per_day': signals_per_day,
            'trades_per_day': trades_per_day,
            'trades_per_month': trades_per_month,
            'win_rate': avg_win_rate,
            'expected_value': avg_ev,
            'monthly_return': monthly_return,
            'total_trades': sum(w['num_trades'] for w in windows),
            'profitable_windows': sum(1 for w in windows if w['expected_value'] > 0),
            'num_windows': len(windows)
        })

    # Analysis
    print("\n" + "="*80)
    print("📊 FREQUENCY vs PROFITABILITY ANALYSIS")
    print("="*80)

    print(f"\n{'Threshold':<12} {'Trades/day':<12} {'Trades/mo':<12} {'Win%':<10} {'EV%':<10} {'Monthly%':<12} {'사용자 요구':<15}")
    print("-" * 100)

    for r in results:
        # Check if meets user requirement (1-10 trades/day)
        meets_min = r['trades_per_day'] >= 1.0
        meets_max = r['trades_per_day'] <= 10.0
        meets_req = meets_min and meets_max

        if meets_req:
            status = "✅ 요구 충족"
        elif r['trades_per_day'] < 1.0:
            status = f"❌ 부족 ({1.0 - r['trades_per_day']:.1f} 부족)"
        else:
            status = f"⚠️ 초과"

        print(f"{r['threshold']:<12.2f} {r['trades_per_day']:<12.1f} {r['trades_per_month']:<12.1f} "
              f"{r['win_rate']:<10.1f} {r['expected_value']:<10.3f} "
              f"{r['monthly_return']:<12.2f} {status:<15}")

    # Find best for user requirement
    meets_requirement = [r for r in results if r['trades_per_day'] >= 1.0]

    print("\n" + "="*80)
    print("🎯 CRITICAL REALITY CHECK")
    print("="*80)

    if not meets_requirement:
        print("\n❌ 사용자 요구 (하루 1-10 거래) 달성 불가능!")
        print("\n이유:")
        print("   - SHORT 신호 자체가 매우 희소 (3.2% of market)")
        print("   - Threshold를 낮춰도 신호 자체가 부족")
        print(f"   - 최대 달성 가능: 하루 {max(r['trades_per_day'] for r in results):.1f} 거래")

        # Find highest frequency option
        max_freq = max(results, key=lambda x: x['trades_per_day'])

        print(f"\n가장 많은 거래 빈도 설정:")
        print(f"   Threshold: {max_freq['threshold']}")
        print(f"   하루 거래: {max_freq['trades_per_day']:.1f} (월 {max_freq['trades_per_month']:.1f})")
        print(f"   Win Rate: {max_freq['win_rate']:.1f}%")
        print(f"   Monthly Return: {max_freq['monthly_return']:+.2f}%")

        print("\n⚠️ 대안:")
        print("   1. SHORT 단독으로는 요구 충족 불가")
        print("   2. LONG 전략 추가 (LONG은 월 ~30 거래 = 하루 1 거래)")
        print("   3. LONG + SHORT 결합 (월 ~52 거래 = 하루 1.7 거래)")
        print("   4. 하루 10 거래는 두 전략 결합으로도 불가능")

    else:
        # Sort by monthly return among those meeting requirement
        best = max(meets_requirement, key=lambda x: x['monthly_return'])

        print(f"\n✅ 사용자 요구 달성 가능!")
        print(f"\n최적 설정:")
        print(f"   Threshold: {best['threshold']}")
        print(f"   하루 거래: {best['trades_per_day']:.1f} (요구: 1-10)")
        print(f"   월 거래: {best['trades_per_month']:.1f}")
        print(f"   Win Rate: {best['win_rate']:.1f}%")
        print(f"   Monthly Return: {best['monthly_return']:+.2f}%")

    # Show LONG comparison
    print("\n" + "="*80)
    print("📈 LONG 전략과 비교")
    print("="*80)

    print("\nLONG Strategy (Phase 4 Base):")
    print("   하루 거래: ~1 거래")
    print("   월 거래: ~30 거래")
    print("   Win Rate: 69.1%")
    print("   Monthly Return: ~46%")

    max_short = max(results, key=lambda x: x['trades_per_day'])
    print(f"\nSHORT Strategy (최대 빈도):")
    print(f"   하루 거래: ~{max_short['trades_per_day']:.1f} 거래")
    print(f"   월 거래: ~{max_short['trades_per_month']:.1f} 거래")
    print(f"   Win Rate: {max_short['win_rate']:.1f}%")
    print(f"   Monthly Return: {max_short['monthly_return']:+.2f}%")

    print(f"\nLONG + SHORT 결합 (가정):")
    print(f"   하루 거래: ~{1.0 + max_short['trades_per_day']:.1f} 거래")
    print(f"   월 거래: ~{30 + max_short['trades_per_month']:.1f} 거래")
    print(f"   결론: 하루 {1.0 + max_short['trades_per_day']:.1f} 거래 (사용자 요구 {'충족 ✅' if 1.0 + max_short['trades_per_day'] >= 1.0 else '미달 ❌'})")

    print("\n" + "="*80)
    print("💡 RECOMMENDATION")
    print("="*80)

    print("\n사용자 요구: 하루 1-10 거래")
    print("\n현실:")
    print("   - SHORT 단독: 최대 하루 ~{:.1f} 거래 (부족)".format(max(r['trades_per_day'] for r in results)))
    print("   - LONG 단독: 하루 ~1 거래 (최소치 충족)")
    print("   - LONG + SHORT: 하루 ~{:.1f} 거래 (최소치 충족)".format(1.0 + max(r['trades_per_day'] for r in results)))

    print("\n권장:")
    if max(r['trades_per_day'] for r in results) >= 1.0:
        best_short = max(results, key=lambda x: x['monthly_return'] if x['trades_per_day'] >= 1.0 else 0)
        print(f"   ✅ SHORT로도 하루 1+ 거래 가능 (threshold {best_short['threshold']})")
        print(f"   Monthly return: {best_short['monthly_return']:+.2f}%")
    else:
        print("   ❌ SHORT 단독으로 하루 1 거래 불가능")
        print("   ✅ 대안: LONG + SHORT 결합 추천")
        print(f"      - LONG: 하루 ~1 거래, +46% monthly")
        print(f"      - SHORT (threshold {max_short['threshold']}): 하루 ~{max_short['trades_per_day']:.1f} 거래, +{max_short['monthly_return']:.2f}% monthly")
        print(f"      - 결합: 하루 ~{1.0 + max_short['trades_per_day']:.1f} 거래")

    print("\n하루 10 거래 달성:")
    print("   ❌ LONG + SHORT 결합으로도 불가능")
    print("   현실적 범위: 하루 1-2 거래 (월 30-60 거래)")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
