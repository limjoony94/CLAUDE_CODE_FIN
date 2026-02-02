"""
전체 데이터셋으로 LONG-only 백테스트

목표: 11 windows vs 55 windows 차이 확인
      "좋은 구간"만 테스트했는지, 전체 성능은 어떤지 검증
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
from scripts.production.dynamic_position_sizing import DynamicPositionSizer

MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "historical"
RESULTS_DIR = PROJECT_ROOT / "results"

# Load model + scaler
model_path = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl"
scaler_path = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_scaler.pkl"
feature_path = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt"

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

with open(feature_path, 'r') as f:
    feature_columns = [line.strip() for line in f.readlines()]

print("="*80)
print("전체 데이터셋 LONG-only 백테스트 (SCALER 적용)")
print("="*80)
print(f"✅ Model loaded: {len(feature_columns)} features")
print(f"✅ Scaler loaded: MinMaxScaler(-1, 1)")

# Load data
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df = pd.read_csv(data_file)
print(f"✅ Data loaded: {len(df)} rows")

# Calculate features
print("\nCalculating features...")
df = calculate_features(df)
adv_features = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)
df = adv_features.calculate_all_features(df)
df = df.ffill().dropna()
print(f"✅ Features calculated: {len(df)} rows")

# Backtest parameters
WINDOW_SIZE = 1440  # 5 days
STEP_SIZE = 288     # 1 day (non-overlapping)
INITIAL_CAPITAL = 10000.0
STOP_LOSS = 0.01
TAKE_PROFIT = 0.03
MAX_HOLDING_HOURS = 4
TRANSACTION_COST = 0.0002
THRESHOLD = 0.7
LEVERAGE = 4

def backtest_window(window_df, leverage):
    """Backtest single window with LONG-only + leverage"""
    capital = INITIAL_CAPITAL
    position = None
    trades = []

    # Debug counters
    max_prob = 0.0
    num_checks = 0

    # Position sizer
    sizer = DynamicPositionSizer(
        base_position_pct=0.50,
        max_position_pct=0.95,
        min_position_pct=0.20,
        signal_weight=0.4,
        volatility_weight=0.3,
        regime_weight=0.2,
        streak_weight=0.1
    )

    for i in range(len(window_df)):
        current_price = window_df['close'].iloc[i]

        # Manage position
        if position is not None:
            entry_idx = position['entry_idx']
            entry_price = position['entry_price']
            hours_held = (i - entry_idx) / 12

            # P&L with leverage (LONG only)
            price_change_pct = (current_price - entry_price) / entry_price
            leveraged_pnl_pct = price_change_pct * leverage
            leveraged_pnl_usd = leveraged_pnl_pct * position['base_value']

            # Liquidation check
            liquidation_threshold = -0.95 / leverage
            if leveraged_pnl_pct <= liquidation_threshold:
                # LIQUIDATION
                leveraged_pnl_usd = -position['base_value']
                net_pnl_usd = leveraged_pnl_usd

                trades.append({
                    'entry_idx': entry_idx,
                    'exit_idx': i,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'base_value': position['base_value'],
                    'leveraged_value': position['leveraged_value'],
                    'position_size_pct': position['position_size_pct'],
                    'leveraged_pnl_pct': leveraged_pnl_pct,
                    'pnl_usd_net': net_pnl_usd,
                    'exit_reason': 'LIQUIDATION',
                    'probability': position['probability']
                })

                capital += net_pnl_usd
                position = None
                continue

            # Normal exits
            exit_reason = None
            if leveraged_pnl_pct <= -STOP_LOSS:
                exit_reason = "SL"
            elif leveraged_pnl_pct >= TAKE_PROFIT:
                exit_reason = "TP"
            elif hours_held >= MAX_HOLDING_HOURS:
                exit_reason = "MH"

            if exit_reason:
                # Transaction costs
                entry_cost = position['leveraged_value'] * TRANSACTION_COST
                exit_cost = (current_price / entry_price) * position['leveraged_value'] * TRANSACTION_COST
                total_cost = entry_cost + exit_cost

                net_pnl_usd = leveraged_pnl_usd - total_cost

                trades.append({
                    'entry_idx': entry_idx,
                    'exit_idx': i,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'base_value': position['base_value'],
                    'leveraged_value': position['leveraged_value'],
                    'position_size_pct': position['position_size_pct'],
                    'leveraged_pnl_pct': leveraged_pnl_pct,
                    'pnl_usd_net': net_pnl_usd,
                    'exit_reason': exit_reason,
                    'probability': position['probability']
                })

                capital += net_pnl_usd
                position = None

        # Entry logic (LONG only)
        if position is None and i < len(window_df) - 1:
            if capital <= 0:
                break

            # Get features
            features = window_df[feature_columns].iloc[i:i+1].values

            if np.isnan(features).any():
                continue

            # Apply scaler (CRITICAL!)
            features_scaled = scaler.transform(features)

            # Predict (LONG only - Class 1)
            probabilities = model.predict_proba(features_scaled)[0]
            prob_long = probabilities[1]

            # Debug tracking
            num_checks += 1
            if prob_long > max_prob:
                max_prob = prob_long

            if prob_long < THRESHOLD:
                continue

            # Calculate regime
            lookback = 20
            recent_data = window_df.iloc[max(0, i-lookback):i+1]
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

            # Volatility
            current_volatility = window_df['atr_pct'].iloc[i] if 'atr_pct' in window_df.columns else 0.01
            avg_volatility = window_df['atr_pct'].iloc[max(0, i-50):i].mean() if 'atr_pct' in window_df.columns else 0.01

            # Calculate position size
            sizing_result = sizer.calculate_position_size(
                capital=capital,
                signal_strength=prob_long,
                current_volatility=current_volatility,
                avg_volatility=avg_volatility,
                market_regime=regime,
                recent_trades=trades[-10:] if len(trades) > 0 else [],
                leverage=leverage
            )

            position = {
                'entry_idx': i,
                'entry_price': current_price,
                'base_value': sizing_result['position_value'],
                'leveraged_value': sizing_result['leveraged_value'],
                'position_size_pct': sizing_result['position_size_pct'],
                'probability': prob_long,
                'regime': regime
            }

    return trades, capital, max_prob, num_checks

# Rolling window backtest
print(f"\n{'='*80}")
print(f"Rolling Window Backtest")
print(f"{'='*80}")
print(f"Window Size: {WINDOW_SIZE} candles (5 days)")
print(f"Step Size: {STEP_SIZE} candles (1 day)")
print(f"Leverage: {LEVERAGE}x")
print(f"Strategy: LONG-only")

all_windows = []
start_idx = 0

# Aggregate debug info
total_max_prob = 0.0
total_checks = 0

while start_idx + WINDOW_SIZE <= len(df):
    end_idx = start_idx + WINDOW_SIZE
    window_df = df.iloc[start_idx:end_idx].copy().reset_index(drop=True)

    # Backtest
    trades, final_capital, max_prob, num_checks = backtest_window(window_df, LEVERAGE)

    # Track debug
    total_max_prob = max(total_max_prob, max_prob)
    total_checks += num_checks

    # Calculate metrics
    window_return = ((final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100

    # Buy & Hold
    bh_start = window_df['close'].iloc[0]
    bh_end = window_df['close'].iloc[-1]
    bh_return = ((bh_end - bh_start) / bh_start) * 100
    bh_cost = 2 * TRANSACTION_COST * 100
    bh_return -= bh_cost

    # Trade metrics
    if len(trades) > 0:
        winning_trades = [t for t in trades if t['pnl_usd_net'] > 0]
        win_rate = (len(winning_trades) / len(trades)) * 100
        avg_position = np.mean([t['position_size_pct'] for t in trades]) * 100
        liquidations = len([t for t in trades if t['exit_reason'] == 'LIQUIDATION'])
    else:
        win_rate = 0
        avg_position = 0
        liquidations = 0

    # Regime
    window_start_price = window_df['close'].iloc[0]
    window_end_price = window_df['close'].iloc[-1]
    window_change = ((window_end_price / window_start_price) - 1) * 100

    if window_change > 3.0:
        regime = "Bull"
    elif window_change < -2.0:
        regime = "Bear"
    else:
        regime = "Sideways"

    all_windows.append({
        'start_idx': start_idx,
        'end_idx': end_idx,
        'regime': regime,
        'return': window_return,
        'bh_return': bh_return,
        'difference': window_return - bh_return,
        'num_trades': len(trades),
        'win_rate': win_rate,
        'final_capital': final_capital,
        'avg_position_size': avg_position,
        'liquidations': liquidations
    })

    start_idx += STEP_SIZE

df_results = pd.DataFrame(all_windows)

# Results
print(f"\n{'='*80}")
print(f"결과: {len(df_results)} windows")
print(f"{'='*80}")

# Debug info
print(f"\n🔍 DEBUG INFO:")
print(f"  Total probability checks: {total_checks:,}")
print(f"  Max probability seen: {total_max_prob:.4f}")
print(f"  Threshold: {THRESHOLD}")

print(f"\n전체 성능:")
print(f"  평균 수익률: {df_results['return'].mean():+.2f}% per 5일")
print(f"  vs B&H: {df_results['difference'].mean():+.2f}%")
print(f"  평균 승률: {df_results['win_rate'].mean():.1f}%")
print(f"  평균 거래: {df_results['num_trades'].mean():.1f}개/window")
print(f"  평균 포지션: {df_results['avg_position_size'].mean():.1f}%")
print(f"  총 청산: {df_results['liquidations'].sum()}건")

# Regime breakdown
print(f"\n시장 환경별:")
for regime in ['Bull', 'Bear', 'Sideways']:
    regime_df = df_results[df_results['regime'] == regime]
    if len(regime_df) > 0:
        print(f"  {regime:10s}: {len(regime_df):2d} windows, "
              f"{regime_df['return'].mean():+6.2f}%, "
              f"Win Rate: {regime_df['win_rate'].mean():5.1f}%")

# Time analysis
print(f"\n기간별 분석:")
total_windows = len(df_results)
for i, label in enumerate(['전반 25%', '중반 25%', '후반 25%', '마지막 25%']):
    start_w = int(i * total_windows / 4)
    end_w = int((i+1) * total_windows / 4)
    period_df = df_results.iloc[start_w:end_w]
    print(f"  {label:15s}: {period_df['return'].mean():+6.2f}%, "
          f"Win Rate: {period_df['win_rate'].mean():5.1f}%")

# Save
output_file = RESULTS_DIR / f"backtest_full_dataset_longonly_4x.csv"
df_results.to_csv(output_file, index=False)
print(f"\n✅ Saved: {output_file.name}")

# Critical analysis
print(f"\n{'='*80}")
print(f"🎯 비판적 분석")
print(f"{'='*80}")

avg_return = df_results['return'].mean()

if avg_return > 10:
    print(f"\n✅ 전체 데이터에서도 우수한 성능!")
    print(f"   평균 수익률: {avg_return:+.2f}% per 5일")
elif avg_return > 5:
    print(f"\n⚠️ 전체 데이터에서 중간 성능")
    print(f"   평균 수익률: {avg_return:+.2f}% per 5일")
    print(f"   일부 구간에서 저조한 성능 가능")
elif avg_return > 0:
    print(f"\n⚠️ 전체 데이터에서 낮은 성능")
    print(f"   평균 수익률: {avg_return:+.2f}% per 5일")
    print(f"   나쁜 구간에서 큰 손실 발생")
else:
    print(f"\n🚨 전체 데이터에서 마이너스 성능!")
    print(f"   평균 수익률: {avg_return:+.2f}% per 5일")
    print(f"   전략이 특정 시장에서만 작동")

# Compare with 11 windows result
print(f"\n비교:")
print(f"  11 windows (이전): +12.06%")
print(f"  {len(df_results)} windows (전체): {avg_return:+.2f}%")

if abs(avg_return - 12.06) > 5:
    print(f"\n⚠️ 큰 차이 발견!")
    print(f"   이전 11 windows는 '좋은 구간'만 선택한 것일 수 있음")
    print(f"   전체 데이터가 더 현실적인 성능")
else:
    print(f"\n✅ 일관된 성능")
    print(f"   이전 결과와 유사함")

print(f"\n{'='*80}")
print("분석 완료!")
print(f"{'='*80}")
