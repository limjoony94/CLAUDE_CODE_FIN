"""
Threshold 0.6 Validation (Critical Thinking Check)

비판적 발견:
- SHORT_DEPLOYMENT_GUIDE.md에서 threshold 0.6 추천
- 하지만 이것은 "예상"이지 실제 테스트가 아님!
- optimize_threshold_frequency.py가 타임아웃으로 검증 실패

Critical Question:
"Threshold 0.6이 정말 8-12 trades/month와 30-35% win rate를 달성하는가?"

이 스크립트:
- Threshold 0.6만 집중 테스트 (빠른 실행)
- 실제 데이터로 가정 검증
- 배포 전 마지막 검증
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

# Test Configuration
THRESHOLD = 0.6  # The recommended threshold to validate
STOP_LOSS = 0.015  # 1.5%
TAKE_PROFIT = 0.06  # 6.0%
WINDOW_SIZE = 1440  # 5 days
INITIAL_CAPITAL = 10000.0
POSITION_SIZE_PCT = 0.95
MAX_HOLDING_HOURS = 4
TRANSACTION_COST = 0.0002


def backtest_threshold_06(df, model, feature_columns):
    """Backtest specifically with threshold 0.6"""
    capital = INITIAL_CAPITAL
    position = None
    trades = []

    for i in range(len(df)):
        current_price = df['close'].iloc[i]

        # Manage existing position
        if position is not None:
            entry_idx = position['entry_idx']
            entry_price = position['entry_price']
            hours_held = (i - entry_idx) / 12

            # SHORT P&L (profit when price drops)
            pnl_pct = (entry_price - current_price) / entry_price

            # Check exit conditions
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

            if short_prob >= THRESHOLD:
                position_value = capital * POSITION_SIZE_PCT
                quantity = position_value / current_price

                position = {
                    'entry_idx': i,
                    'entry_price': current_price,
                    'quantity': quantity,
                    'short_prob': short_prob
                }

    return trades, capital


def rolling_window_validation(df, model, feature_columns):
    """Test with rolling windows"""
    windows = []
    start_idx = 0

    while start_idx + WINDOW_SIZE <= len(df):
        end_idx = start_idx + WINDOW_SIZE
        window_df = df.iloc[start_idx:end_idx].reset_index(drop=True)

        trades, final_capital = backtest_threshold_06(window_df, model, feature_columns)

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
    """Validate threshold 0.6 assumptions"""
    print("="*80)
    print("THRESHOLD 0.6 VALIDATION - Critical Thinking Check")
    print("="*80)
    print("\n❓ Critical Question:")
    print("   Does threshold 0.6 really achieve 8-12 trades/month and 30-35% win rate?")
    print("\n📋 Deployment Guide Claims:")
    print("   - Win Rate: 30-35%")
    print("   - Trades/Month: 8-12")
    print("   - Expected Value: +0.75% per trade")
    print("\n🔍 Let's verify with ACTUAL DATA...")
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

    # Run validation
    print(f"\n🔬 Testing threshold 0.6 with rolling windows...")
    print(f"   Window size: {WINDOW_SIZE} candles (5 days)")
    print(f"   R:R: SL {STOP_LOSS*100:.1f}%, TP {TAKE_PROFIT*100:.1f}%")

    windows = rolling_window_validation(df, model, feature_columns)

    if not windows:
        print("\n❌ No trades found!")
        return

    # Aggregate results
    avg_trades = np.mean([w['num_trades'] for w in windows])
    avg_win_rate = np.mean([w['win_rate'] for w in windows])
    avg_ev = np.mean([w['expected_value'] for w in windows])
    avg_return = np.mean([w['total_return'] for w in windows])

    trades_per_month = avg_trades * 6  # 30 days / 5 days
    monthly_return = avg_ev * avg_trades * 6

    # Display results
    print("\n" + "="*80)
    print("📊 VALIDATION RESULTS")
    print("="*80)

    print(f"\n🎯 Actual Performance (Threshold 0.6):")
    print(f"   Trades per 5 days: {avg_trades:.1f}")
    print(f"   Trades per month: {trades_per_month:.1f}")
    print(f"   Win Rate: {avg_win_rate:.1f}%")
    print(f"   Expected Value: {avg_ev:+.3f}% per trade")
    print(f"   Monthly Return (estimated): {monthly_return:+.2f}%")
    print(f"   5-day Return (actual): {avg_return:+.2f}%")

    print(f"\n📋 Deployment Guide Assumptions:")
    print(f"   Trades per month: 8-12")
    print(f"   Win Rate: 30-35%")
    print(f"   Expected Value: +0.75%")

    # Validation check
    print("\n" + "="*80)
    print("✅ VALIDATION CHECK")
    print("="*80)

    trades_match = 8 <= trades_per_month <= 12
    winrate_match = 30 <= avg_win_rate <= 35
    ev_positive = avg_ev > 0

    print(f"\nTrade Frequency: {'✅ MATCH' if trades_match else '❌ MISMATCH'}")
    print(f"   Expected: 8-12/month")
    print(f"   Actual: {trades_per_month:.1f}/month")

    print(f"\nWin Rate: {'✅ MATCH' if winrate_match else '❌ MISMATCH'}")
    print(f"   Expected: 30-35%")
    print(f"   Actual: {avg_win_rate:.1f}%")

    print(f"\nExpected Value: {'✅ POSITIVE' if ev_positive else '❌ NEGATIVE'}")
    print(f"   Expected: +0.75%")
    print(f"   Actual: {avg_ev:+.3f}%")

    # Final conclusion
    print("\n" + "="*80)
    print("🎯 CRITICAL THINKING CONCLUSION")
    print("="*80)

    if trades_match and winrate_match and ev_positive:
        print("\n✅✅ ASSUMPTIONS VALIDATED!")
        print("   Threshold 0.6 performs as expected")
        print("   Deployment guide assumptions are CORRECT")
        print("   Safe to deploy with current configuration")
    elif ev_positive:
        print("\n⚠️ PARTIAL VALIDATION")
        print("   Expected Value is positive (profitable)")
        print("   But frequency or win rate differs from estimates")
        print("   Deployment possible but update expectations")
    else:
        print("\n❌ ASSUMPTIONS FAILED")
        print("   Threshold 0.6 does not meet expectations")
        print("   Need to reconsider deployment strategy")

    # Detailed statistics
    print(f"\n📈 Detailed Statistics ({len(windows)} windows):")
    print(f"   Total trades: {sum(w['num_trades'] for w in windows)}")
    print(f"   Win rate range: {min(w['win_rate'] for w in windows):.1f}% - {max(w['win_rate'] for w in windows):.1f}%")
    print(f"   EV range: {min(w['expected_value'] for w in windows):+.3f}% - {max(w['expected_value'] for w in windows):+.3f}%")
    print(f"   Positive EV windows: {sum(1 for w in windows if w['expected_value'] > 0)}/{len(windows)}")

    print("\n" + "="*80)
    print("비판적 사고 결과: 가정이 실제 데이터와 일치하는지 확인 완료")
    print("="*80)


if __name__ == "__main__":
    main()
