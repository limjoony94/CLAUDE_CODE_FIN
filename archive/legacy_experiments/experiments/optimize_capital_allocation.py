"""
Capital Allocation Optimization - Approach #22

비판적 발견:
- 70/30 비율은 "합리적으로 보이는" 추정치
- 실제로 백테스트로 검증하지 않음!
- 다른 비율이 더 나을 수도 있음

Critical Question:
"70/30이 정말 최적인가? 다른 비율은 어떤가?"

이 스크립트:
- 50/50, 60/40, 70/30, 80/20, 90/10 테스트
- 각 비율의 combined return 계산
- 진짜 최적 allocation 찾기
- Sharpe ratio와 risk-adjusted return도 고려
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

WINDOW_SIZE = 1440  # 5 days
INITIAL_CAPITAL = 10000.0

# LONG Configuration
LONG_THRESHOLD = 0.7
LONG_STOP_LOSS = 0.01
LONG_TAKE_PROFIT = 0.03
LONG_MAX_HOLDING_HOURS = 4
LONG_POSITION_SIZE_PCT = 0.95

# SHORT Configuration
SHORT_THRESHOLD = 0.4  # Optimal from Approach #21
SHORT_STOP_LOSS = 0.015
SHORT_TAKE_PROFIT = 0.06
SHORT_MAX_HOLDING_HOURS = 4
SHORT_POSITION_SIZE_PCT = 0.95

TRANSACTION_COST = 0.0002


def backtest_strategy(df, model, feature_columns, strategy_type, threshold, sl, tp, allocation):
    """Backtest LONG or SHORT strategy with specific allocation"""
    capital = INITIAL_CAPITAL * allocation
    position = None
    trades = []

    for i in range(len(df)):
        current_price = df['close'].iloc[i]

        # Manage position
        if position is not None:
            entry_idx = position['entry_idx']
            entry_price = position['entry_price']
            hours_held = (i - entry_idx) / 12

            # Calculate P&L based on strategy type
            if strategy_type == "LONG":
                pnl_pct = (current_price - entry_price) / entry_price
            else:  # SHORT
                pnl_pct = (entry_price - current_price) / entry_price

            # Exit conditions
            exit_reason = None
            if pnl_pct <= -sl:
                exit_reason = "Stop Loss"
            elif pnl_pct >= tp:
                exit_reason = "Take Profit"
            elif hours_held >= (LONG_MAX_HOLDING_HOURS if strategy_type == "LONG" else SHORT_MAX_HOLDING_HOURS):
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
                    'exit_reason': exit_reason
                })

                position = None

        # Look for entry
        if position is None and i < len(df) - 1:
            features = df[feature_columns].iloc[i:i+1].values

            if np.isnan(features).any():
                continue

            if strategy_type == "LONG":
                prob = model.predict_proba(features)[0][1]
                signal = prob >= threshold
            else:  # SHORT
                probs = model.predict_proba(features)[0]
                prob = probs[2]
                signal = prob >= threshold

            if signal:
                position_value = capital * (LONG_POSITION_SIZE_PCT if strategy_type == "LONG" else SHORT_POSITION_SIZE_PCT)
                quantity = position_value / current_price

                position = {
                    'entry_idx': i,
                    'entry_price': current_price,
                    'quantity': quantity
                }

    return trades, capital


def rolling_window_test(df, long_model, long_features, short_model, short_features, long_alloc, short_alloc):
    """Test combined strategy with rolling windows"""
    windows = []
    start_idx = 0

    while start_idx + WINDOW_SIZE <= len(df):
        end_idx = start_idx + WINDOW_SIZE
        window_df = df.iloc[start_idx:end_idx].reset_index(drop=True)

        # Backtest LONG
        long_trades, long_capital = backtest_strategy(
            window_df, long_model, long_features, "LONG",
            LONG_THRESHOLD, LONG_STOP_LOSS, LONG_TAKE_PROFIT, long_alloc
        )

        # Backtest SHORT
        short_trades, short_capital = backtest_strategy(
            window_df, short_model, short_features, "SHORT",
            SHORT_THRESHOLD, SHORT_STOP_LOSS, SHORT_TAKE_PROFIT, short_alloc
        )

        # Combined performance
        total_capital = long_capital + short_capital
        total_return = ((total_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100

        long_return = ((long_capital - INITIAL_CAPITAL * long_alloc) / (INITIAL_CAPITAL * long_alloc)) * 100 if long_trades else 0
        short_return = ((short_capital - INITIAL_CAPITAL * short_alloc) / (INITIAL_CAPITAL * short_alloc)) * 100 if short_trades else 0

        windows.append({
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
            'total_trades': len(long_trades) + len(short_trades),
            'long_return': long_return,
            'short_return': short_return,
            'total_return': total_return
        })

        start_idx += WINDOW_SIZE

    return windows


def main():
    """Optimize capital allocation between LONG and SHORT"""
    print("="*80)
    print("CAPITAL ALLOCATION OPTIMIZATION - Approach #22")
    print("="*80)
    print("\n❓ Critical Question:")
    print("   Is 70/30 allocation really optimal?")
    print("   Or did we just assume it without testing?")
    print("\n🔍 Testing Strategy:")
    print("   Test allocations: 50/50, 60/40, 70/30, 80/20, 90/10")
    print("   Find allocation that maximizes combined return")
    print("   Consider risk-adjusted performance")
    print("="*80)

    # Load LONG model
    long_model_file = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl"
    print(f"\n✅ Loading LONG model: {long_model_file.name}")
    with open(long_model_file, 'rb') as f:
        long_model = pickle.load(f)

    long_feature_file = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt"
    with open(long_feature_file, 'r') as f:
        long_features = [line.strip() for line in f.readlines()]
    print(f"   LONG features: {len(long_features)}")

    # Load SHORT model
    short_model_file = MODELS_DIR / "xgboost_v4_phase4_3class_lookahead3_thresh3.pkl"
    print(f"✅ Loading SHORT model: {short_model_file.name}")
    with open(short_model_file, 'rb') as f:
        short_model = pickle.load(f)

    short_feature_file = MODELS_DIR / "xgboost_v4_phase4_3class_lookahead3_thresh3_features.txt"
    with open(short_feature_file, 'r') as f:
        short_features = [line.strip() for line in f.readlines()]
    print(f"   SHORT features: {len(short_features)}")

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

    # Test different allocations
    allocations = [
        (0.5, 0.5),   # 50/50
        (0.6, 0.4),   # 60/40
        (0.7, 0.3),   # 70/30 (current assumption)
        (0.8, 0.2),   # 80/20
        (0.9, 0.1),   # 90/10
    ]

    results = []

    print(f"\n" + "="*80)
    print("TESTING DIFFERENT ALLOCATIONS")
    print("="*80)

    for long_alloc, short_alloc in allocations:
        print(f"\n🔬 Testing {long_alloc*100:.0f}% LONG / {short_alloc*100:.0f}% SHORT...")

        windows = rolling_window_test(df, long_model, long_features, short_model, short_features, long_alloc, short_alloc)

        if not windows:
            print(f"   ❌ No data")
            continue

        # Aggregate
        avg_total_return = np.mean([w['total_return'] for w in windows])
        std_total_return = np.std([w['total_return'] for w in windows])
        avg_long_return = np.mean([w['long_return'] for w in windows])
        avg_short_return = np.mean([w['short_return'] for w in windows])
        avg_total_trades = np.mean([w['total_trades'] for w in windows])

        # Monthly extrapolation
        monthly_return = avg_total_return * 6  # 30 days / 5 days
        monthly_trades = avg_total_trades * 6

        # Risk-adjusted metrics
        sharpe_ratio = (avg_total_return / std_total_return) if std_total_return > 0 else 0

        print(f"   5-day return: {avg_total_return:+.2f}%")
        print(f"   Monthly return: {monthly_return:+.2f}%")
        print(f"   Return volatility: {std_total_return:.2f}%")
        print(f"   Sharpe ratio: {sharpe_ratio:.2f}")
        print(f"   Trades/month: {monthly_trades:.1f}")

        results.append({
            'long_alloc': long_alloc,
            'short_alloc': short_alloc,
            '5d_return': avg_total_return,
            'monthly_return': monthly_return,
            'volatility': std_total_return,
            'sharpe_ratio': sharpe_ratio,
            'long_contribution': avg_long_return * long_alloc,
            'short_contribution': avg_short_return * short_alloc,
            'monthly_trades': monthly_trades,
            'num_windows': len(windows)
        })

    # Analysis
    print("\n" + "="*80)
    print("📊 ALLOCATION COMPARISON")
    print("="*80)

    print(f"\n{'LONG%':<8} {'SHORT%':<8} {'5d%':<10} {'Monthly%':<12} {'Sharpe':<10} {'Trades/mo':<12} {'Rank':<8}")
    print("-" * 80)

    # Sort by monthly return
    results_sorted = sorted(results, key=lambda x: x['monthly_return'], reverse=True)

    for rank, r in enumerate(results_sorted, 1):
        marker = "⭐ BEST" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else ""

        print(f"{r['long_alloc']*100:<8.0f} {r['short_alloc']*100:<8.0f} "
              f"{r['5d_return']:<10.2f} {r['monthly_return']:<12.2f} "
              f"{r['sharpe_ratio']:<10.2f} {r['monthly_trades']:<12.1f} {marker:<8}")

    # Best allocation
    best = results_sorted[0]

    print("\n" + "="*80)
    print("🎯 OPTIMAL ALLOCATION FOUND")
    print("="*80)

    print(f"\n🥇 Best Allocation: {best['long_alloc']*100:.0f}% LONG / {best['short_alloc']*100:.0f}% SHORT")
    print(f"\nPerformance:")
    print(f"   Monthly Return: {best['monthly_return']:+.2f}%")
    print(f"   Sharpe Ratio: {best['sharpe_ratio']:.2f}")
    print(f"   Trades per month: {best['monthly_trades']:.1f}")

    print(f"\nContributions:")
    print(f"   LONG: {best['long_contribution']:+.2f}% (from {best['long_alloc']*100:.0f}% allocation)")
    print(f"   SHORT: {best['short_contribution']:+.2f}% (from {best['short_alloc']*100:.0f}% allocation)")

    # Compare to 70/30
    original_70_30 = [r for r in results if r['long_alloc'] == 0.7][0]

    if best['long_alloc'] != 0.7:
        improvement = ((best['monthly_return'] - original_70_30['monthly_return']) / abs(original_70_30['monthly_return'])) * 100

        print(f"\n📈 vs 70/30 Allocation:")
        print(f"   70/30 return: {original_70_30['monthly_return']:+.2f}%")
        print(f"   Optimal return: {best['monthly_return']:+.2f}%")
        print(f"   Improvement: {improvement:+.1f}%")

        print(f"\n🔄 RECOMMENDATION: Update from 70/30 to {best['long_alloc']*100:.0f}/{best['short_alloc']*100:.0f}")
    else:
        print(f"\n✅ ORIGINAL CHOICE VALIDATED!")
        print(f"   70/30 is indeed the optimal allocation")
        print(f"   No need to change configuration")

    # Risk-adjusted analysis
    print("\n" + "="*80)
    print("⚖️ RISK-ADJUSTED ANALYSIS")
    print("="*80)

    # Sort by Sharpe ratio
    results_by_sharpe = sorted(results, key=lambda x: x['sharpe_ratio'], reverse=True)
    best_sharpe = results_by_sharpe[0]

    print(f"\nBest Sharpe Ratio: {best_sharpe['long_alloc']*100:.0f}/{best_sharpe['short_alloc']*100:.0f}")
    print(f"   Sharpe: {best_sharpe['sharpe_ratio']:.2f}")
    print(f"   Monthly Return: {best_sharpe['monthly_return']:+.2f}%")

    if best_sharpe['long_alloc'] == best['long_alloc']:
        print(f"\n✅ Same as best return → Clear optimal choice!")
    else:
        print(f"\n⚠️ Trade-off:")
        print(f"   Best return: {best['long_alloc']*100:.0f}/{best['short_alloc']*100:.0f} → {best['monthly_return']:+.2f}%")
        print(f"   Best Sharpe: {best_sharpe['long_alloc']*100:.0f}/{best_sharpe['short_alloc']*100:.0f} → {best_sharpe['monthly_return']:+.2f}%")
        print(f"\n   Recommendation: Choose best return (maximize profit)")

    print("\n" + "="*80)
    print("🧠 CRITICAL THINKING CONCLUSION")
    print("="*80)

    print("\n비판적 질문: \"70/30이 최적인가?\"")
    print(f"\n실제 테스트 결과:")
    print(f"   최적 allocation: {best['long_alloc']*100:.0f}% LONG / {best['short_alloc']*100:.0f}% SHORT")
    print(f"   70/30 대비: {'+' if best['long_alloc'] != 0.7 else ''}{'개선 필요' if best['long_alloc'] != 0.7 else '이미 최적!'}")

    if best['long_alloc'] == 0.7:
        print(f"\n✅ 원래 선택 (70/30)이 데이터로 검증됨!")
        print(f"   추정이 우연히 맞았지만, 이제는 증거가 있음")
    else:
        print(f"\n🔄 더 나은 allocation 발견!")
        print(f"   {best['long_alloc']*100:.0f}/{best['short_alloc']*100:.0f}로 업데이트 권장")
        print(f"   기대 개선: {((best['monthly_return'] - original_70_30['monthly_return']) / abs(original_70_30['monthly_return'])) * 100:+.1f}%")

    print("\n비판적 사고 적용:")
    print("  1. 70/30 가정 의심 → \"정말 최적인가?\"")
    print("  2. 다양한 allocation 실제 테스트")
    print(f"  3. 결과: {best['long_alloc']*100:.0f}/{best['short_alloc']*100:.0f}가 최적 ({best['monthly_return']:+.2f}% monthly)")
    print("  4. 증거 기반 의사결정 완료 ✅")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
