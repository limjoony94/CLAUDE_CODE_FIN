"""
SHORT Model Standalone Backtest

목표: SHORT 모델 단독 성능 검증
- SHORT 예측 전용 모델
- Threshold 0.7로 필터링
- 4x leverage
- Dynamic Position Sizing

검증 목표:
- SHORT 모델 승률 > 60% 달성?
- 하락장에서 실제 수익?
- 신호 빈도가 적정한가?
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

# Load SHORT model
model_path = MODELS_DIR / "xgboost_short_model_lookahead3_thresh0.3.pkl"
feature_path = MODELS_DIR / "xgboost_short_model_lookahead3_thresh0.3_features.txt"

with open(model_path, 'rb') as f:
    short_model = pickle.load(f)

with open(feature_path, 'r') as f:
    feature_columns = [line.strip() for line in f.readlines()]

print("=" * 80)
print("SHORT Model Standalone Backtest")
print("=" * 80)
print(f"✅ SHORT Model loaded: {len(feature_columns)} features")

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
STEP_SIZE = 288     # 1 day
INITIAL_CAPITAL = 10000.0
STOP_LOSS = 0.01
TAKE_PROFIT = 0.03
MAX_HOLDING_HOURS = 4
TRANSACTION_COST = 0.0002
THRESHOLD = 0.7
LEVERAGE = 4


def backtest_window(window_df, leverage):
    """Backtest single window with SHORT-only + leverage"""
    capital = INITIAL_CAPITAL
    position = None
    trades = []

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

            # P&L for SHORT
            price_change_pct = (entry_price - current_price) / entry_price
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

        # Entry logic (SHORT only)
        if position is None and i < len(window_df) - 1:
            if capital <= 0:
                break

            # Get features
            features = window_df[feature_columns].iloc[i:i+1].values

            if np.isnan(features).any():
                continue

            # Predict SHORT
            prob_short = short_model.predict_proba(features)[0][1]

            if prob_short < THRESHOLD:
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
                signal_strength=prob_short,
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
                'probability': prob_short,
                'regime': regime
            }

    return trades, capital


# Rolling window backtest
print(f"\n{'='*80}")
print(f"Rolling Window Backtest")
print(f"{'='*80}")
print(f"Window Size: {WINDOW_SIZE} candles (5 days)")
print(f"Step Size: {STEP_SIZE} candles (1 day)")
print(f"Leverage: {LEVERAGE}x")
print(f"Strategy: SHORT-only")

all_windows = []
start_idx = 0

while start_idx + WINDOW_SIZE <= len(df):
    end_idx = start_idx + WINDOW_SIZE
    window_df = df.iloc[start_idx:end_idx].copy().reset_index(drop=True)

    # Backtest
    trades, final_capital = backtest_window(window_df, LEVERAGE)

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

# Save
output_file = RESULTS_DIR / f"backtest_short_only_4x.csv"
df_results.to_csv(output_file, index=False)
print(f"\n✅ Saved: {output_file.name}")

# Critical analysis
print(f"\n{'='*80}")
print(f"🎯 SHORT 모델 검증")
print(f"{'='*80}")

short_return = df_results['return'].mean()
short_win_rate = df_results['win_rate'].mean()
total_trades = df_results['num_trades'].sum()

print(f"\nSHORT 모델 성능:")
print(f"  평균 수익률: {short_return:+.2f}% per 5일")
print(f"  평균 승률: {short_win_rate:.1f}%")
print(f"  총 거래: {total_trades}개 ({total_trades/len(df_results):.1f}개/window)")

print(f"\n검증 목표:")
if short_win_rate >= 60:
    print(f"  ✅ 승률 목표 달성: {short_win_rate:.1f}% >= 60%")
else:
    print(f"  ❌ 승률 목표 미달: {short_win_rate:.1f}% < 60%")

if short_return > 0:
    print(f"  ✅ 수익 달성: {short_return:+.2f}%")
else:
    print(f"  ❌ 손실 발생: {short_return:+.2f}%")

# Bear market performance
bear_df = df_results[df_results['regime'] == 'Bear']
if len(bear_df) > 0:
    bear_return = bear_df['return'].mean()
    if bear_return > 0:
        print(f"  ✅ 하락장 수익: {bear_return:+.2f}%")
    else:
        print(f"  ❌ 하락장 손실: {bear_return:+.2f}%")

print(f"\n{'='*80}")
print("분석 완료!")
print(f"{'='*80}")
