"""
NaN 처리 방법 비교 백테스트

목표: 다양한 NaN 처리 방법이 모델 성능에 미치는 영향 측정
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures
from scripts.production.dynamic_position_sizing import DynamicPositionSizer

# Load model
MODELS_DIR = PROJECT_ROOT / "models"
model_path = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl"
feature_path = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt"

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(feature_path, 'r') as f:
    feature_columns = [line.strip() for line in f.readlines()]

print(f"✅ Model loaded: {len(feature_columns)} features")

# Load data
data_file = PROJECT_ROOT / "data" / "historical" / "BTCUSDT_5m_max.csv"
df = pd.read_csv(data_file)

print("="*80)
print("NaN 처리 방법 비교 백테스트")
print("="*80)
print(f"Data: {len(df)} rows")

# Calculate features
print("\nCalculating features...")
df = calculate_features(df)
adv_features = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)
df = adv_features.calculate_all_features(df)

print(f"✅ Features calculated: {len(df)} rows")

# Backtest function
def backtest_with_nan_handling(df, model, feature_columns, threshold, leverage, nan_handling_name, nan_handling_func):
    """
    Run backtest with specific NaN handling method

    Args:
        nan_handling_func: Function that takes df and returns df with NaN handled
    """
    print(f"\n{'='*80}")
    print(f"백테스트: {nan_handling_name}")
    print(f"{'='*80}")

    # Apply NaN handling
    df_processed = nan_handling_func(df.copy())
    rows_before = len(df)
    rows_after = len(df_processed)
    print(f"Data: {rows_before} → {rows_after} rows ({rows_before - rows_after} lost)")

    if len(df_processed) < 1440:
        print(f"⚠️ 데이터 부족: {len(df_processed)} < 1440")
        return None

    # Initialize position sizer
    position_sizer = DynamicPositionSizer(
        base_position_pct=0.50,
        max_position_pct=0.95,
        min_position_pct=0.20,
        signal_weight=0.4,
        volatility_weight=0.3,
        regime_weight=0.2,
        streak_weight=0.1
    )

    # Rolling window backtest
    WINDOW_SIZE = 1440  # 5 days
    STEP_SIZE = 288     # 1 day

    all_trades = []
    all_returns = []

    for start_idx in range(0, len(df_processed) - WINDOW_SIZE, STEP_SIZE):
        end_idx = start_idx + WINDOW_SIZE
        window_df = df_processed.iloc[start_idx:end_idx].copy()

        # Initial capital
        capital = 100000
        position = None
        trades = []

        for i in range(50, len(window_df)):
            current_price = window_df['close'].iloc[i]

            # Get features
            features = window_df[feature_columns].iloc[i:i+1].values

            # Check for NaN in features
            if np.isnan(features).any():
                continue  # Skip if NaN found after processing

            # Predict
            probabilities = model.predict_proba(features)[0]
            prob_long = probabilities[1]
            prob_short = probabilities[0]

            # Determine signal
            signal_direction = None
            signal_probability = None

            if prob_long >= threshold:
                signal_direction = "LONG"
                signal_probability = prob_long
            elif prob_short >= threshold:
                signal_direction = "SHORT"
                signal_probability = prob_short

            # Manage position
            if position is not None:
                # Check exit conditions
                entry_price = position['entry_price']
                side = position['side']

                if side == "SHORT":
                    pnl_pct = (entry_price - current_price) / entry_price
                else:
                    pnl_pct = (current_price - entry_price) / entry_price

                hours_held = (i - position['entry_idx']) / 12  # 5-min candles

                exit_reason = None
                if pnl_pct <= -0.01:  # Stop Loss 1%
                    exit_reason = "SL"
                elif pnl_pct >= 0.03:  # Take Profit 3%
                    exit_reason = "TP"
                elif hours_held >= 4:  # Max Holding 4 hours
                    exit_reason = "MH"

                if exit_reason:
                    # Close position
                    position['exit_idx'] = i
                    position['exit_price'] = current_price
                    position['pnl_pct'] = pnl_pct
                    position['exit_reason'] = exit_reason

                    # Calculate P&L
                    position_value = position['position_value']
                    gross_pnl = pnl_pct * position_value

                    # Transaction costs
                    entry_cost = position_value * 0.0006
                    exit_cost = position_value * 0.0006
                    total_cost = entry_cost + exit_cost
                    net_pnl = gross_pnl - total_cost

                    position['pnl_usd'] = net_pnl
                    capital += net_pnl

                    trades.append(position)
                    position = None

            # Check entry (if no position)
            if position is None and signal_direction is not None:
                # Calculate volatility
                current_volatility = window_df['atr_pct'].iloc[i] if 'atr_pct' in window_df.columns else 0.01
                avg_volatility = window_df['atr_pct'].iloc[max(0, i-50):i].mean() if 'atr_pct' in window_df.columns else 0.01

                # Classify regime
                lookback = 20
                recent_data = window_df.iloc[max(0, i-lookback):i]
                if len(recent_data) >= lookback:
                    start_price = recent_data['close'].iloc[0]
                    end_price = recent_data['close'].iloc[-1]
                    price_change_pct = ((end_price / start_price) - 1) * 100

                    if price_change_pct > 3.0:
                        regime = "Bull"
                    elif price_change_pct < -2.0:
                        regime = "Bear"
                    else:
                        regime = "Sideways"
                else:
                    regime = "Unknown"

                # Calculate position size
                sizing_result = position_sizer.calculate_position_size(
                    capital=capital,
                    signal_strength=signal_probability,
                    current_volatility=current_volatility,
                    avg_volatility=avg_volatility,
                    market_regime=regime,
                    recent_trades=trades[-10:] if len(trades) > 0 else [],
                    leverage=leverage
                )

                position = {
                    'entry_idx': i,
                    'entry_price': current_price,
                    'side': signal_direction,
                    'position_value': sizing_result['position_value'],
                    'position_size_pct': sizing_result['position_size_pct'],
                    'signal_probability': signal_probability,
                    'regime': regime
                }

        # Calculate window return
        if len(trades) > 0:
            window_return = ((capital - 100000) / 100000) * 100
            all_returns.append(window_return)
            all_trades.extend(trades)

    if len(all_trades) == 0:
        print("⚠️ No trades executed!")
        return None

    # Calculate metrics
    df_trades = pd.DataFrame(all_trades)

    total_trades = len(df_trades)
    winning_trades = len(df_trades[df_trades['pnl_usd'] > 0])
    win_rate = (winning_trades / total_trades) * 100

    avg_return = np.mean(all_returns)
    std_return = np.std(all_returns)
    sharpe = (avg_return / std_return) if std_return > 0 else 0

    avg_position = df_trades['position_size_pct'].mean() * 100

    print(f"\n결과:")
    print(f"  총 윈도우: {len(all_returns)}개")
    print(f"  총 거래: {total_trades}")
    print(f"  승률: {win_rate:.1f}%")
    print(f"  평균 포지션: {avg_position:.1f}%")
    print(f"  평균 수익률: {avg_return:+.2f}% per 5일")
    print(f"  수익률 표준편차: {std_return:.2f}%")
    print(f"  Sharpe Ratio: {sharpe:.2f}")

    return {
        'nan_handling': nan_handling_name,
        'data_rows': rows_after,
        'data_lost': rows_before - rows_after,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_position': avg_position,
        'avg_return': avg_return,
        'std_return': std_return,
        'sharpe': sharpe
    }

# Test different NaN handling methods
print("\n" + "="*80)
print("NaN 처리 방법별 백테스트")
print("="*80)

threshold = 0.7
leverage = 4

nan_strategies = {
    "ffill+dropna (현재)": lambda df: df.ffill().dropna(),
    "fillna(0)": lambda df: df.fillna(0),
    "ffill+bfill+dropna": lambda df: df.ffill().bfill().dropna(),
}

results = []

for strategy_name, strategy_func in nan_strategies.items():
    result = backtest_with_nan_handling(
        df=df,
        model=model,
        feature_columns=feature_columns,
        threshold=threshold,
        leverage=leverage,
        nan_handling_name=strategy_name,
        nan_handling_func=strategy_func
    )

    if result:
        results.append(result)

# Compare results
print("\n" + "="*80)
print("종합 비교")
print("="*80)

if len(results) > 0:
    df_results = pd.DataFrame(results)

    print(f"\n{'NaN 처리':20s} | {'데이터':8s} | {'손실':6s} | {'거래':6s} | {'승률':6s} | {'포지션':8s} | {'수익률':8s} | {'Sharpe':7s}")
    print(f"{'-'*20} | {'-'*8} | {'-'*6} | {'-'*6} | {'-'*6} | {'-'*8} | {'-'*8} | {'-'*7}")

    for _, row in df_results.iterrows():
        print(f"{row['nan_handling']:20s} | {row['data_rows']:8d} | {row['data_lost']:6d} | "
              f"{row['total_trades']:6d} | {row['win_rate']:5.1f}% | {row['avg_position']:7.1f}% | "
              f"{row['avg_return']:+7.2f}% | {row['sharpe']:7.2f}")

    # Find best method
    best_by_return = df_results.loc[df_results['avg_return'].idxmax()]
    best_by_sharpe = df_results.loc[df_results['sharpe'].idxmax()]

    print(f"\n✨ 최고 수익률: {best_by_return['nan_handling']} ({best_by_return['avg_return']:+.2f}%)")
    print(f"✨ 최고 Sharpe: {best_by_sharpe['nan_handling']} ({best_by_sharpe['sharpe']:.2f})")

    # Recommendation
    print(f"\n" + "="*80)
    print("권장 사항")
    print("="*80)

    if best_by_return['nan_handling'] == "ffill+dropna (현재)":
        print("""
✅ 현재 방식(ffill+dropna)이 최적입니다!

이유:
1. 가장 높은 수익률 달성
2. 백테스트와 일치 (신뢰성 보장)
3. 잘못된 정보를 모델에 제공하지 않음
4. 손실되는 50개 행(0.29%)은 무시 가능한 수준

→ 변경 불필요. 현재 방식 유지하세요!
        """)
    else:
        print(f"""
⚠️ {best_by_return['nan_handling']}가 더 높은 수익률을 달성했습니다.

그러나 주의 사항:
1. 백테스트와 프로덕션 불일치 가능성
2. 잘못된 정보(0, mean 등)를 모델에 제공할 수 있음
3. 실제 성능은 다를 수 있음

→ 신중한 검토 후 결정하세요!
        """)
else:
    print("⚠️ No valid backtest results!")

print("\n✅ 분석 완료!")
