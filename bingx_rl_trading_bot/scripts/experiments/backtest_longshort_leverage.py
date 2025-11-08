"""
LONG/SHORT 백테스트 (레버리지 4x)

목표: 프로덕션 봇과 동일한 설정으로 백테스트
      - LONG + SHORT 모두 거래
      - 레버리지 4x
      - Dynamic Position Sizing
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

# Load model
model_path = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl"
feature_path = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt"

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(feature_path, 'r') as f:
    feature_columns = [line.strip() for line in f.readlines()]

print("="*80)
print("LONG/SHORT 백테스트 (프로덕션 설정)")
print("="*80)
print(f"✅ Model loaded: {len(feature_columns)} features")

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
    """Backtest single window with LONG/SHORT + leverage"""
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
            side = position['side']
            hours_held = (i - entry_idx) / 12

            # P&L with leverage (LONG or SHORT)
            if side == "SHORT":
                price_change_pct = (entry_price - current_price) / entry_price
            else:  # LONG
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
                    'side': side,
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
                    'side': side,
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

        # Entry logic (LONG + SHORT)
        if position is None and i < len(window_df) - 1:
            if capital <= 0:
                break

            # Get features
            features = window_df[feature_columns].iloc[i:i+1].values

            if np.isnan(features).any():
                continue

            # Predict (LONG and SHORT)
            probabilities = model.predict_proba(features)[0]
            prob_long = probabilities[1]   # Class 1 = LONG
            prob_short = probabilities[0]  # Class 0 = SHORT

            # Determine signal direction
            signal_direction = None
            signal_probability = None

            if prob_long >= THRESHOLD:
                signal_direction = "LONG"
                signal_probability = prob_long
            elif prob_short >= THRESHOLD:
                signal_direction = "SHORT"
                signal_probability = prob_short

            if signal_direction is None:
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
                'base_value': sizing_result['position_value'],
                'leveraged_value': sizing_result['leveraged_value'],
                'position_size_pct': sizing_result['position_size_pct'],
                'probability': signal_probability,
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
print(f"Strategy: LONG + SHORT (프로덕션 설정)")

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

        # LONG/SHORT breakdown
        long_trades = [t for t in trades if t['side'] == 'LONG']
        short_trades = [t for t in trades if t['side'] == 'SHORT']

        long_wins = len([t for t in long_trades if t['pnl_usd_net'] > 0])
        short_wins = len([t for t in short_trades if t['pnl_usd_net'] > 0])

        long_win_rate = (long_wins / len(long_trades)) * 100 if len(long_trades) > 0 else 0
        short_win_rate = (short_wins / len(short_trades)) * 100 if len(short_trades) > 0 else 0
    else:
        win_rate = 0
        avg_position = 0
        liquidations = 0
        long_trades = []
        short_trades = []
        long_win_rate = 0
        short_win_rate = 0

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
        'num_long': len(long_trades),
        'num_short': len(short_trades),
        'win_rate': win_rate,
        'long_win_rate': long_win_rate,
        'short_win_rate': short_win_rate,
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

# LONG vs SHORT performance
total_long = df_results['num_long'].sum()
total_short = df_results['num_short'].sum()
avg_long_wr = df_results['long_win_rate'].mean()
avg_short_wr = df_results['short_win_rate'].mean()

print(f"\nLONG vs SHORT:")
print(f"  LONG 거래:  {total_long:4d}개 ({total_long/(total_long+total_short)*100:5.1f}%), 승률: {avg_long_wr:5.1f}%")
print(f"  SHORT 거래: {total_short:4d}개 ({total_short/(total_long+total_short)*100:5.1f}%), 승률: {avg_short_wr:5.1f}%")

# Regime breakdown
print(f"\n시장 환경별:")
for regime in ['Bull', 'Bear', 'Sideways']:
    regime_df = df_results[df_results['regime'] == regime]
    if len(regime_df) > 0:
        print(f"  {regime:10s}: {len(regime_df):2d} windows, "
              f"{regime_df['return'].mean():+6.2f}%, "
              f"Win Rate: {regime_df['win_rate'].mean():5.1f}%")

# Save
output_file = RESULTS_DIR / f"backtest_longshort_4x.csv"
df_results.to_csv(output_file, index=False)
print(f"\n✅ Saved: {output_file.name}")

# Critical analysis
print(f"\n{'='*80}")
print(f"🎯 비판적 분석")
print(f"{'='*80}")

longshort_return = df_results['return'].mean()
longonly_return = 12.67  # From previous backtest

print(f"\n비교:")
print(f"  LONG-only @ 4x:    {longonly_return:+.2f}%")
print(f"  LONG/SHORT @ 4x:   {longshort_return:+.2f}%")
print(f"  차이:              {longshort_return - longonly_return:+.2f}%")

if longshort_return > longonly_return:
    print(f"\n✅ LONG/SHORT이 더 우수!")
    print(f"   SHORT 거래가 수익에 기여")
elif longshort_return > longonly_return - 2:
    print(f"\n⚠️ LONG/SHORT이 약간 낮음")
    print(f"   SHORT 성능 점검 필요")
else:
    print(f"\n🚨 LONG/SHORT이 크게 낮음!")
    print(f"   SHORT 거래가 수익을 악화시킴")
    print(f"   → SHORT 비활성화 고려 필요")

# SHORT performance analysis
if avg_short_wr < 50:
    print(f"\n🚨 SHORT 승률 매우 낮음: {avg_short_wr:.1f}%")
    print(f"   모델이 SHORT 예측에 취약")
elif avg_short_wr < avg_long_wr - 10:
    print(f"\n⚠️ SHORT 승률이 LONG보다 낮음:")
    print(f"   LONG: {avg_long_wr:.1f}% vs SHORT: {avg_short_wr:.1f}%")
else:
    print(f"\n✅ LONG/SHORT 균형:")
    print(f"   LONG: {avg_long_wr:.1f}% vs SHORT: {avg_short_wr:.1f}%")

print(f"\n{'='*80}")
print("분석 완료!")
print(f"{'='*80}")
