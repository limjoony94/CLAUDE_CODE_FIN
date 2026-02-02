"""
10x Scaled vs 4x Comparison: 30-Day Backtest
=============================================

목표: 동일한 자본 노출로 10배 vs 4배 비교

Configuration:
- 10x @ 8-38% (base 20%) = 0.8x-3.8x 노출
- 4x @ 20-95% (base 50%) = 0.8x-3.8x 노출

Expected Outcome:
- 동일한 수익/손실 패턴
- 10배의 마진 여유 더 많음 (평균 80% vs 50%)
- 10배의 청산 위험 더 높음 (-10% vs -25%)
"""

import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
import sys
import time
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.calculate_all_features import calculate_all_features
from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features
from scripts.production.dynamic_position_sizing import DynamicPositionSizer
from scripts.production.dynamic_position_sizing_10x_scaled import DynamicPositionSizer10xScaled
from scripts.production.dynamic_position_sizing_10x_wider import DynamicPositionSizer10xWider

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

print("="*80)
print("LEVERAGE & POSITION SIZER COMPARISON - 3 VARIANTS")
print("="*80)
print("\n4x @ 20-95% vs 10x @ 8-38% vs 10x @ 5-50%\n")

# =============================================================================
# CONFIGURATION (4배 최적화된 설정 사용)
# =============================================================================

# Strategy Parameters
LONG_THRESHOLD = 0.65
SHORT_THRESHOLD = 0.70
GATE_THRESHOLD = 0.001

# Exit Parameters (최적화된 설정)
ML_EXIT_THRESHOLD_LONG = 0.75
ML_EXIT_THRESHOLD_SHORT = 0.75
EMERGENCY_MAX_HOLD_TIME = 120  # 10 hours
EMERGENCY_STOP_LOSS = -0.03  # -3% balance

# Expected Values
LONG_AVG_RETURN = 0.0041
SHORT_AVG_RETURN = 0.0047

# Backtest Parameters
INITIAL_CAPITAL = 10000.0
TAKER_FEE = 0.0005

# =============================================================================
# LOAD MODELS
# =============================================================================

print("Loading models...")
long_model_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146.pkl"
with open(long_model_path, 'rb') as f:
    long_model = pickle.load(f)

long_scaler_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146_scaler.pkl"
long_scaler = joblib.load(long_scaler_path)

long_features_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146_features.txt"
with open(long_features_path, 'r') as f:
    long_feature_columns = [line.strip() for line in f.readlines()]

short_model_path = MODELS_DIR / "xgboost_short_trade_outcome_full_20251018_233146.pkl"
with open(short_model_path, 'rb') as f:
    short_model = pickle.load(f)

short_scaler_path = MODELS_DIR / "xgboost_short_trade_outcome_full_20251018_233146_scaler.pkl"
short_scaler = joblib.load(short_scaler_path)

short_features_path = MODELS_DIR / "xgboost_short_trade_outcome_full_20251018_233146_features.txt"
with open(short_features_path, 'r') as f:
    short_feature_columns = [line.strip() for line in f.readlines()]

long_exit_model_path = MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251017_151624.pkl"
with open(long_exit_model_path, 'rb') as f:
    long_exit_model = pickle.load(f)

long_exit_scaler_path = MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251017_151624_scaler.pkl"
long_exit_scaler = joblib.load(long_exit_scaler_path)

long_exit_features_path = MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251017_151624_features.txt"
with open(long_exit_features_path, 'r') as f:
    long_exit_feature_columns = [line.strip() for line in f.readlines()]

short_exit_model_path = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251017_152440.pkl"
with open(short_exit_model_path, 'rb') as f:
    short_exit_model = pickle.load(f)

short_exit_scaler_path = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251017_152440_scaler.pkl"
short_exit_scaler = joblib.load(short_exit_scaler_path)

short_exit_features_path = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251017_152440_features.txt"
with open(short_exit_features_path, 'r') as f:
    short_exit_feature_columns = [line.strip() for line in f.readlines()]

print("  ✅ All models loaded\n")

# =============================================================================
# LOAD DATA
# =============================================================================

print("Loading historical data...")
btc_5m = pd.read_csv(DATA_DIR / "BTCUSDT_5m.csv")
btc_5m['timestamp'] = pd.to_datetime(btc_5m['timestamp'])
btc_5m = btc_5m.sort_values('timestamp').reset_index(drop=True)

print(f"  Total candles: {len(btc_5m):,}")
print(f"  Date range: {btc_5m['timestamp'].min()} to {btc_5m['timestamp'].max()}\n")

# Last 30 days
end_date = btc_5m['timestamp'].max()
start_date = end_date - pd.Timedelta(days=30)
test_df = btc_5m[btc_5m['timestamp'] >= start_date].copy().reset_index(drop=True)

print(f"Test period: {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")
print(f"Test candles: {len(test_df):,} ({len(test_df)/288:.1f} days)\n")

# =============================================================================
# CALCULATE FEATURES
# =============================================================================

print("Calculating features...")
test_df = calculate_all_features(test_df)
test_df = prepare_exit_features(test_df)  # Exit features
test_df = test_df.dropna().reset_index(drop=True)
print(f"  After dropna: {len(test_df):,} candles\n")

# ATR for volatility
test_df['atr'] = test_df['close'].rolling(14).apply(
    lambda x: np.mean(np.abs(np.diff(x))) if len(x) > 1 else 0,
    raw=True
)

# =============================================================================
# RUN BACKTESTS
# =============================================================================

def run_backtest(leverage, sizer, sizer_name):
    """Run backtest with given leverage and sizer"""

    print(f"\n{'='*80}")
    print(f"BACKTEST: {sizer_name}")
    print(f"{'='*80}\n")

    capital = INITIAL_CAPITAL
    position = None
    trades = []
    equity_curve = []
    recent_trades = []

    # Stats
    total_trades = 0
    winning_trades = 0
    losing_trades = 0
    long_trades = 0
    short_trades = 0
    ml_exits = 0
    sl_exits = 0
    maxhold_exits = 0

    peak_capital = capital
    max_drawdown = 0.0

    for i in range(len(test_df)):
        row = test_df.iloc[i]
        current_timestamp = row['timestamp']
        current_price = row['close']

        # Track equity
        if position:
            if position['side'] == 'LONG':
                pnl_pct = (current_price - position['entry_price']) / position['entry_price']
            else:  # SHORT
                pnl_pct = (position['entry_price'] - current_price) / position['entry_price']

            leveraged_pnl_pct = pnl_pct * leverage
            unrealized_pnl = position['margin'] * leveraged_pnl_pct
            current_equity = capital + unrealized_pnl
        else:
            current_equity = capital

        equity_curve.append(current_equity)

        # Update peak and drawdown
        if current_equity > peak_capital:
            peak_capital = current_equity
        drawdown = (current_equity - peak_capital) / peak_capital
        if drawdown < max_drawdown:
            max_drawdown = drawdown

        # EXIT CHECK
        if position:
            exit_reason = None
            exit_price = current_price

            # 1. ML Exit (use pre-calculated features)
            if position['side'] == 'LONG':
                X_exit = pd.DataFrame([test_df.iloc[i]])[long_exit_feature_columns]
                X_exit_scaled = long_exit_scaler.transform(X_exit)
                exit_prob = long_exit_model.predict_proba(X_exit_scaled)[0][1]

                if exit_prob >= ML_EXIT_THRESHOLD_LONG:
                    exit_reason = 'ML_EXIT'
                    ml_exits += 1
            else:  # SHORT
                X_exit = pd.DataFrame([test_df.iloc[i]])[short_exit_feature_columns]
                X_exit_scaled = short_exit_scaler.transform(X_exit)
                exit_prob = short_exit_model.predict_proba(X_exit_scaled)[0][1]

                if exit_prob >= ML_EXIT_THRESHOLD_SHORT:
                    exit_reason = 'ML_EXIT'
                    ml_exits += 1

            # 2. Stop Loss (balance-based)
            if position['side'] == 'LONG':
                pnl_pct = (current_price - position['entry_price']) / position['entry_price']
            else:
                pnl_pct = (position['entry_price'] - current_price) / position['entry_price']

            leveraged_pnl_pct = pnl_pct * leverage
            balance_pnl_pct = leveraged_pnl_pct * (position['margin'] / capital)

            if balance_pnl_pct <= EMERGENCY_STOP_LOSS:
                exit_reason = 'STOP_LOSS'
                sl_exits += 1

            # 3. Max Hold Time
            hold_time = i - position['entry_idx']
            if hold_time >= EMERGENCY_MAX_HOLD_TIME:
                exit_reason = 'MAX_HOLD'
                maxhold_exits += 1

            # EXIT
            if exit_reason:
                # Calculate P&L
                if position['side'] == 'LONG':
                    pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
                else:
                    pnl_pct = (position['entry_price'] - exit_price) / position['entry_price']

                leveraged_pnl_pct = pnl_pct * leverage
                gross_pnl = position['margin'] * leveraged_pnl_pct

                # Fees
                entry_fee = position['position_size'] * TAKER_FEE
                exit_fee = position['position_size'] * TAKER_FEE
                total_fees = entry_fee + exit_fee

                net_pnl = gross_pnl - total_fees

                # Update capital
                capital += net_pnl

                # Record trade
                trade_record = {
                    'entry_time': position['entry_time'],
                    'exit_time': current_timestamp,
                    'side': position['side'],
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'margin': position['margin'],
                    'position_size': position['position_size'],
                    'hold_candles': hold_time,
                    'pnl': net_pnl,
                    'pnl_pct': (net_pnl / position['margin']) * 100,
                    'exit_reason': exit_reason
                }
                trades.append(trade_record)
                recent_trades.append({'pnl_usd_net': net_pnl})

                # Stats
                total_trades += 1
                if net_pnl > 0:
                    winning_trades += 1
                else:
                    losing_trades += 1

                if position['side'] == 'LONG':
                    long_trades += 1
                else:
                    short_trades += 1

                position = None

        # ENTRY CHECK
        if not position and i >= 100:
            # LONG entry features
            X_long = pd.DataFrame([test_df.iloc[i]])[long_feature_columns]
            X_long_scaled = long_scaler.transform(X_long)
            long_prob = long_model.predict_proba(X_long_scaled)[0][1]

            # SHORT entry features
            X_short = pd.DataFrame([test_df.iloc[i]])[short_feature_columns]
            X_short_scaled = short_scaler.transform(X_short)
            short_prob = short_model.predict_proba(X_short_scaled)[0][1]

            # Position sizing
            avg_volatility = test_df['atr'].iloc[max(0, i-100):i].mean()
            current_volatility = test_df['atr'].iloc[i]

            # Determine market regime (simplified)
            recent_returns = test_df['close'].iloc[max(0, i-20):i].pct_change().dropna()
            if len(recent_returns) > 0:
                avg_return = recent_returns.mean()
                if avg_return > 0.001:
                    regime = "Bull"
                elif avg_return < -0.001:
                    regime = "Bear"
                else:
                    regime = "Sideways"
            else:
                regime = "Sideways"

            # Opportunity Gating
            if long_prob >= LONG_THRESHOLD:
                side = 'LONG'
                signal_strength = long_prob
            elif short_prob >= SHORT_THRESHOLD:
                long_ev = long_prob * LONG_AVG_RETURN
                short_ev = short_prob * SHORT_AVG_RETURN
                opportunity_cost = short_ev - long_ev

                if opportunity_cost > GATE_THRESHOLD:
                    side = 'SHORT'
                    signal_strength = short_prob
                else:
                    side = None
                    signal_strength = 0
            else:
                side = None
                signal_strength = 0

            # ENTER
            if side:
                sizing_result = sizer.calculate_position_size(
                    capital=capital,
                    signal_strength=signal_strength,
                    current_volatility=current_volatility,
                    avg_volatility=avg_volatility,
                    market_regime=regime,
                    recent_trades=recent_trades[-10:],
                    leverage=leverage
                )

                # Handle both sizer return formats
                if 'position_dollars' in sizing_result:
                    # 10x_scaled format
                    margin = sizing_result['position_dollars']
                    position_size = sizing_result['leverage_exposure']
                else:
                    # Original format
                    margin = sizing_result['position_value']
                    position_size = sizing_result['leveraged_value']

                # Entry fees
                entry_fee = position_size * TAKER_FEE

                position = {
                    'side': side,
                    'entry_time': current_timestamp,
                    'entry_price': current_price,
                    'entry_idx': i,
                    'margin': margin,
                    'position_size': position_size
                }

    # Close final position if any
    if position:
        exit_price = test_df.iloc[-1]['close']
        if position['side'] == 'LONG':
            pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
        else:
            pnl_pct = (position['entry_price'] - exit_price) / position['entry_price']

        leveraged_pnl_pct = pnl_pct * leverage
        gross_pnl = position['margin'] * leveraged_pnl_pct

        entry_fee = position['position_size'] * TAKER_FEE
        exit_fee = position['position_size'] * TAKER_FEE
        net_pnl = gross_pnl - entry_fee - exit_fee

        capital += net_pnl

        trade_record = {
            'entry_time': position['entry_time'],
            'exit_time': test_df.iloc[-1]['timestamp'],
            'side': position['side'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'margin': position['margin'],
            'position_size': position['position_size'],
            'hold_candles': len(test_df) - 1 - position['entry_idx'],
            'pnl': net_pnl,
            'pnl_pct': (net_pnl / position['margin']) * 100,
            'exit_reason': 'END_OF_TEST'
        }
        trades.append(trade_record)

        total_trades += 1
        if net_pnl > 0:
            winning_trades += 1
        else:
            losing_trades += 1

        if position['side'] == 'LONG':
            long_trades += 1
        else:
            short_trades += 1

    # Calculate metrics
    final_capital = capital
    total_return = ((final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

    # Average position size
    if trades:
        avg_margin_pct = np.mean([t['margin'] / INITIAL_CAPITAL for t in trades]) * 100
    else:
        avg_margin_pct = 0

    # Sharpe
    returns = pd.Series(equity_curve).pct_change().dropna()
    if len(returns) > 0 and returns.std() > 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(288 * 30)  # Annualized
    else:
        sharpe = 0

    # Exit breakdown
    ml_exit_pct = (ml_exits / total_trades * 100) if total_trades > 0 else 0
    sl_exit_pct = (sl_exits / total_trades * 100) if total_trades > 0 else 0
    maxhold_exit_pct = (maxhold_exits / total_trades * 100) if total_trades > 0 else 0

    # Print results
    print(f"Final Capital: ${final_capital:,.2f}")
    print(f"Total Return: {total_return:+.2f}%")
    print(f"Max Drawdown: {max_drawdown*100:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.3f}")
    print(f"\nTrades: {total_trades}")
    print(f"Win Rate: {win_rate:.1f}% ({winning_trades}W / {losing_trades}L)")
    print(f"LONG: {long_trades} | SHORT: {short_trades}")
    print(f"Avg Margin: {avg_margin_pct:.1f}%")
    print(f"\nExit Breakdown:")
    print(f"  ML Exit: {ml_exits} ({ml_exit_pct:.1f}%)")
    print(f"  Stop Loss: {sl_exits} ({sl_exit_pct:.1f}%)")
    print(f"  Max Hold: {maxhold_exits} ({maxhold_exit_pct:.1f}%)")

    return {
        'sizer_name': sizer_name,
        'leverage': leverage,
        'final_capital': final_capital,
        'total_return': total_return,
        'max_drawdown': max_drawdown * 100,
        'sharpe_ratio': sharpe,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'long_trades': long_trades,
        'short_trades': short_trades,
        'avg_margin_pct': avg_margin_pct,
        'ml_exits': ml_exits,
        'sl_exits': sl_exits,
        'maxhold_exits': maxhold_exits,
        'trades': trades
    }

# Run all three backtests
sizer_4x = DynamicPositionSizer(
    base_position_pct=0.50,
    max_position_pct=0.95,
    min_position_pct=0.20
)

sizer_10x_scaled = DynamicPositionSizer10xScaled(
    base_position_pct=0.20,
    max_position_pct=0.38,
    min_position_pct=0.08
)

sizer_10x_wider = DynamicPositionSizer10xWider(
    base_position_pct=0.25,
    max_position_pct=0.50,
    min_position_pct=0.05
)

result_4x = run_backtest(4, sizer_4x, "4x @ 20-95% (Original)")
result_10x_scaled = run_backtest(10, sizer_10x_scaled, "10x @ 8-38% (Scaled)")
result_10x_wider = run_backtest(10, sizer_10x_wider, "10x @ 5-50% (Wider)")

# =============================================================================
# COMPARISON
# =============================================================================

print(f"\n{'='*80}")
print("COMPARISON SUMMARY")
print(f"{'='*80}\n")

print(f"{'Metric':<25} {'4x Original':<15} {'10x Scaled':<15} {'10x Wider':<15} {'Winner':<10}")
print("-" * 90)

metrics = [
    ('Return', 'total_return', '%', 'higher'),
    ('Max Drawdown', 'max_drawdown', '%', 'lower'),
    ('Sharpe Ratio', 'sharpe_ratio', '', 'higher'),
    ('Win Rate', 'win_rate', '%', 'higher'),
    ('Total Trades', 'total_trades', '', 'similar'),
    ('Avg Margin', 'avg_margin_pct', '%', 'info'),
    ('ML Exit Rate', None, '%', 'info'),
]

for metric_name, metric_key, unit, comparison in metrics:
    if metric_key:
        val_4x = result_4x[metric_key]
        val_10x_scaled = result_10x_scaled[metric_key]
        val_10x_wider = result_10x_wider[metric_key]
    elif metric_name == 'ML Exit Rate':
        val_4x = result_4x['ml_exits'] / result_4x['total_trades'] * 100 if result_4x['total_trades'] > 0 else 0
        val_10x_scaled = result_10x_scaled['ml_exits'] / result_10x_scaled['total_trades'] * 100 if result_10x_scaled['total_trades'] > 0 else 0
        val_10x_wider = result_10x_wider['ml_exits'] / result_10x_wider['total_trades'] * 100 if result_10x_wider['total_trades'] > 0 else 0

    val_4x_str = f"{val_4x:+.2f}{unit}" if unit == '%' else f"{val_4x:.3f}"
    val_10x_scaled_str = f"{val_10x_scaled:+.2f}{unit}" if unit == '%' else f"{val_10x_scaled:.3f}"
    val_10x_wider_str = f"{val_10x_wider:+.2f}{unit}" if unit == '%' else f"{val_10x_wider:.3f}"

    if comparison == 'higher':
        vals = {'4x': val_4x, '10x_s': val_10x_scaled, '10x_w': val_10x_wider}
        winner = max(vals, key=vals.get).replace('_s', ' Scaled').replace('_w', ' Wider')
    elif comparison == 'lower':
        vals = {'4x': val_4x, '10x_s': val_10x_scaled, '10x_w': val_10x_wider}
        winner = min(vals, key=vals.get).replace('_s', ' Scaled').replace('_w', ' Wider')
    else:
        winner = "-"

    print(f"{metric_name:<25} {val_4x_str:<15} {val_10x_scaled_str:<15} {val_10x_wider_str:<15} {winner:<10}")

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

# Capital efficiency
avg_reserve_4x = 100 - result_4x['avg_margin_pct']
avg_reserve_10x_scaled = 100 - result_10x_scaled['avg_margin_pct']
avg_reserve_10x_wider = 100 - result_10x_wider['avg_margin_pct']

print(f"\n1. Capital Efficiency:")
print(f"   4x @ 20-95%: {result_4x['avg_margin_pct']:.1f}% used, {avg_reserve_4x:.1f}% reserve")
print(f"   10x @ 8-38%: {result_10x_scaled['avg_margin_pct']:.1f}% used, {avg_reserve_10x_scaled:.1f}% reserve")
print(f"   10x @ 5-50%: {result_10x_wider['avg_margin_pct']:.1f}% used, {avg_reserve_10x_wider:.1f}% reserve")
print(f"   → 10x variants have {avg_reserve_10x_scaled - avg_reserve_4x:.1f}%-{avg_reserve_10x_wider - avg_reserve_4x:.1f}% more reserve")

print(f"\n2. Exposure Range:")
print(f"   4x @ 20-95%: 0.8x - 3.8x exposure")
print(f"   10x @ 8-38%: 0.8x - 3.8x exposure (same as 4x)")
print(f"   10x @ 5-50%: 0.5x - 5.0x exposure (wider range)")

print(f"\n3. Liquidation Risk:")
print(f"   4x: ~-25% price move (safest)")
print(f"   10x: ~-10% price move (higher risk)")
print(f"   → 4x 2.5x safer from liquidation")

print(f"\n4. Performance:")
results_sorted = sorted(
    [
        ('4x @ 20-95%', result_4x['total_return'], result_4x['sharpe_ratio']),
        ('10x @ 8-38%', result_10x_scaled['total_return'], result_10x_scaled['sharpe_ratio']),
        ('10x @ 5-50%', result_10x_wider['total_return'], result_10x_wider['sharpe_ratio'])
    ],
    key=lambda x: x[1],
    reverse=True
)
print(f"   Return Ranking:")
for i, (name, ret, sharpe) in enumerate(results_sorted, 1):
    print(f"   {i}. {name}: {ret:+.2f}% (Sharpe {sharpe:.3f})")

print(f"\n5. Recommendation:")
best_return = max(result_4x['total_return'], result_10x_scaled['total_return'], result_10x_wider['total_return'])
best_sharpe = max(result_4x['sharpe_ratio'], result_10x_scaled['sharpe_ratio'], result_10x_wider['sharpe_ratio'])

if result_4x['total_return'] == best_return and result_4x['sharpe_ratio'] == best_sharpe:
    print("   🥇 4x @ 20-95%: Best returns AND Sharpe → Keep current setup")
elif result_10x_scaled['total_return'] == best_return and result_10x_scaled['sharpe_ratio'] == best_sharpe:
    print("   🥇 10x @ 8-38%: Best performance + more capital flexibility")
elif result_10x_wider['total_return'] == best_return and result_10x_wider['sharpe_ratio'] == best_sharpe:
    print("   🥇 10x @ 5-50%: Highest returns + widest range")
else:
    print("   Mixed results → Choose based on priorities:")
    print("   - Max safety: 4x @ 20-95% (lowest liquidation risk)")
    print("   - Same exposure: 10x @ 8-38% (capital flexibility)")
    print("   - Max range: 10x @ 5-50% (aggressive upside)")

print("\n" + "="*80)

# Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = RESULTS_DIR / f"leverage_position_comparison_3variants_{timestamp}.txt"

with open(results_file, 'w') as f:
    f.write("LEVERAGE & POSITION SIZER COMPARISON - 3 VARIANTS\n")
    f.write("="*80 + "\n\n")
    f.write(f"4x @ 20-95%: {result_4x['total_return']:+.2f}% | Sharpe {result_4x['sharpe_ratio']:.3f}\n")
    f.write(f"10x @ 8-38%: {result_10x_scaled['total_return']:+.2f}% | Sharpe {result_10x_scaled['sharpe_ratio']:.3f}\n")
    f.write(f"10x @ 5-50%: {result_10x_wider['total_return']:+.2f}% | Sharpe {result_10x_wider['sharpe_ratio']:.3f}\n")
    f.write(f"\nWinner: {results_sorted[0][0]}\n")

print(f"\nResults saved: {results_file}")
print("="*80)
