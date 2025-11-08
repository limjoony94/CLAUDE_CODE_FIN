"""
Comprehensive Position Sizing Optimization v2.0 (Fast Version)

최적화 전략:
1. 더 작은 테스트셋 사용 (최근 2주)
2. Weight 조합 축소 (가장 유망한 조합만 테스트)
3. 2단계 최적화:
   - Phase 1: Weight 최적화 (27 조합) - 축소된 그리드
   - Phase 2: BASE/MAX/MIN 최적화 (6 조합)

예상 소요 시간: 3-5분
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys
from itertools import product
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures

# Configuration
DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Fixed parameters
LONG_ENTRY_THRESHOLD = 0.70
SHORT_ENTRY_THRESHOLD = 0.65
EXIT_THRESHOLD = 0.70
STOP_LOSS = 0.01
TAKE_PROFIT = 0.02
MAX_HOLDING_HOURS = 4

print("=" * 100)
print("COMPREHENSIVE POSITION SIZING OPTIMIZATION V2.0 (FAST)")
print("🚀 Optimized for speed: Reduced test period + focused grid search")
print("=" * 100)
print()

start_time = datetime.now()

# Load data
df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_max.csv")
print(f"✅ Loaded {len(df)} candles")

# Calculate features
df = calculate_features(df)
adv_features = AdvancedTechnicalFeatures()
df = adv_features.calculate_all_features(df)
df = df.ffill().dropna()
print(f"✅ Features calculated: {len(df)} rows")

# Load models
with open(MODELS_DIR / 'xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl', 'rb') as f:
    long_model = pickle.load(f)
with open(MODELS_DIR / 'xgboost_v4_phase4_advanced_lookahead3_thresh0_scaler.pkl', 'rb') as f:
    long_scaler = pickle.load(f)
with open(MODELS_DIR / 'xgboost_short_model_lookahead3_thresh0.3.pkl', 'rb') as f:
    short_model = pickle.load(f)
with open(MODELS_DIR / 'xgboost_short_model_lookahead3_thresh0.3_scaler.pkl', 'rb') as f:
    short_scaler = pickle.load(f)
with open(MODELS_DIR / 'xgboost_v4_long_exit.pkl', 'rb') as f:
    long_exit_model = pickle.load(f)
with open(MODELS_DIR / 'xgboost_v4_long_exit_scaler.pkl', 'rb') as f:
    long_exit_scaler = pickle.load(f)
with open(MODELS_DIR / 'xgboost_v4_short_exit.pkl', 'rb') as f:
    short_exit_model = pickle.load(f)
with open(MODELS_DIR / 'xgboost_v4_short_exit_scaler.pkl', 'rb') as f:
    short_exit_scaler = pickle.load(f)
with open(MODELS_DIR / 'xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt', 'r') as f:
    feature_columns = [line.strip() for line in f]

print(f"✅ Models loaded")
print()

# Get predictions
X = df[feature_columns].values
X_long_scaled = long_scaler.transform(X)
X_short_scaled = short_scaler.transform(X)

df['prob_long_entry'] = long_model.predict_proba(X_long_scaled)[:, 1]
df['prob_short_entry'] = short_model.predict_proba(X_short_scaled)[:, 1]

# Calculate volatility and regime
df['atr_pct'] = df['atr'] / df['close'] * 100 if 'atr' in df.columns else df['close'].pct_change().rolling(14).std() * 100
df['avg_volatility'] = df['atr_pct'].rolling(window=50).mean()
df['sma_20'] = df['close'].rolling(20).mean()
df['sma_50'] = df['close'].rolling(50).mean()

def detect_regime(row):
    if pd.isna(row['sma_20']) or pd.isna(row['sma_50']):
        return "Sideways"
    if row['close'] > row['sma_20'] and row['sma_20'] > row['sma_50']:
        return "Bull"
    elif row['close'] < row['sma_20'] and row['sma_20'] < row['sma_50']:
        return "Bear"
    else:
        return "Sideways"

df['market_regime'] = df.apply(detect_regime, axis=1)

# Use smaller test set for speed (last 2 weeks instead of 3)
two_weeks_candles = 12 * 24 * 14  # 2016 candles
df_test = df.iloc[-two_weeks_candles:].copy().reset_index(drop=True)

weeks = len(df_test) / (12 * 24 * 7)
print(f"✅ Test set: {len(df_test)} candles ({weeks:.1f} weeks) - FAST MODE")
print()


class DynamicPositionSizer:
    """Integrated DynamicPositionSizer"""

    def __init__(self, base_position_pct=0.50, max_position_pct=0.95, min_position_pct=0.20,
                 signal_weight=0.4, volatility_weight=0.3, regime_weight=0.2, streak_weight=0.1):
        self.base_position_pct = base_position_pct
        self.max_position_pct = max_position_pct
        self.min_position_pct = min_position_pct

        total_weight = signal_weight + volatility_weight + regime_weight + streak_weight
        self.signal_weight = signal_weight / total_weight
        self.volatility_weight = volatility_weight / total_weight
        self.regime_weight = regime_weight / total_weight
        self.streak_weight = streak_weight / total_weight

    def calculate_position_size(self, signal_strength, current_volatility, avg_volatility, market_regime, recent_trades):
        # Signal factor
        normalized = (signal_strength - 0.5) / 0.5
        signal_factor = np.clip(normalized ** 1.5, 0.0, 1.0)

        # Volatility factor
        if avg_volatility == 0 or pd.isna(avg_volatility):
            volatility_factor = 0.5
        else:
            volatility_ratio = current_volatility / avg_volatility
            if volatility_ratio <= 0.5:
                volatility_factor = 1.0
            elif volatility_ratio <= 1.0:
                volatility_factor = 1.0 - (volatility_ratio - 0.5) / 0.5 * 0.5
            elif volatility_ratio <= 2.0:
                volatility_factor = 0.5 - (volatility_ratio - 1.0) / 1.0 * 0.5
            else:
                volatility_factor = 0.0

        # Regime factor
        regime_factors = {"Bull": 1.0, "Sideways": 0.6, "Bear": 0.3}
        regime_factor = regime_factors.get(market_regime, 0.5)

        # Streak factor
        if not recent_trades:
            streak_factor = 1.0
        else:
            recent = recent_trades[-5:]
            consecutive_wins = consecutive_losses = 0
            for trade in reversed(recent):
                if trade.get('realized_pnl', 0) > 0:
                    if consecutive_losses > 0:
                        break
                    consecutive_wins += 1
                else:
                    if consecutive_wins > 0:
                        break
                    consecutive_losses += 1
            if consecutive_wins >= 3:
                streak_factor = 0.8
            elif consecutive_losses >= 3:
                streak_factor = 0.3
            elif consecutive_losses == 2:
                streak_factor = 0.6
            elif consecutive_losses == 1:
                streak_factor = 0.9
            else:
                streak_factor = 1.0

        combined_factor = (self.signal_weight * signal_factor + self.volatility_weight * volatility_factor +
                          self.regime_weight * regime_factor + self.streak_weight * streak_factor)

        position_size_pct = self.base_position_pct * (0.5 + combined_factor)
        return np.clip(position_size_pct, self.min_position_pct, self.max_position_pct)


def run_backtest(df_test, sizer, feature_columns, long_exit_model, long_exit_scaler, short_exit_model, short_exit_scaler):
    """Optimized backtest"""
    df_test['signal_long'] = (df_test['prob_long_entry'] >= LONG_ENTRY_THRESHOLD).astype(int)
    df_test['signal_short'] = (df_test['prob_short_entry'] >= SHORT_ENTRY_THRESHOLD).astype(int)

    position = None
    trades = []
    equity = 1.0
    equity_curve = [1.0]
    max_hold_candles = MAX_HOLDING_HOURS * 12

    for i in range(len(df_test)):
        current_price = df_test.loc[i, 'close']

        if position is not None:
            hold_time = i - entry_idx
            hours_held = hold_time / 12
            pnl_pct = (current_price - entry_price) / entry_price if direction == 'LONG' else (entry_price - current_price) / entry_price
            realized_pnl = pnl_pct * position_size_pct
            should_exit = False
            exit_reason = None

            if pnl_pct <= -STOP_LOSS:
                should_exit, exit_reason = True, "SL"
            elif pnl_pct >= TAKE_PROFIT:
                should_exit, exit_reason = True, "TP"
            elif hold_time >= max_hold_candles:
                should_exit, exit_reason = True, "MaxHold"
            else:
                base_features = df_test[feature_columns].iloc[i].values[:36]
                position_features = np.array([hours_held / 1.0, pnl_pct, max(pnl_pct, 0.0), min(pnl_pct, 0.0),
                                             pnl_pct - max(pnl_pct, 0.0), df_test['atr_pct'].iloc[i], 0.0, 0.0])
                exit_features = np.concatenate([base_features, position_features]).reshape(1, -1)
                exit_model = long_exit_model if direction == 'LONG' else short_exit_model
                exit_scaler = long_exit_scaler if direction == 'LONG' else short_exit_scaler
                exit_prob = exit_model.predict_proba(exit_scaler.transform(exit_features))[0][1]
                if exit_prob >= EXIT_THRESHOLD:
                    should_exit, exit_reason = True, "ML"

            if should_exit:
                trades.append({'direction': direction, 'pnl_pct': pnl_pct, 'realized_pnl': realized_pnl,
                              'position_size': position_size_pct, 'hold_time_hours': hours_held, 'exit_reason': exit_reason})
                equity *= (1 + realized_pnl * 0.999)
                equity_curve.append(equity)
                position = None

        if position is None:
            current_volatility = df_test.loc[i, 'atr_pct']
            avg_volatility = df_test.loc[i, 'avg_volatility']
            market_regime = df_test.loc[i, 'market_regime']

            if df_test.loc[i, 'signal_long'] == 1:
                position_size_pct = sizer.calculate_position_size(df_test.loc[i, 'prob_long_entry'],
                                                                  current_volatility, avg_volatility, market_regime, trades)
                position, entry_price, entry_idx, direction = 'LONG', current_price, i, 'LONG'

            elif df_test.loc[i, 'signal_short'] == 1:
                position_size_pct = sizer.calculate_position_size(df_test.loc[i, 'prob_short_entry'],
                                                                   current_volatility, avg_volatility, market_regime, trades)
                position, entry_price, entry_idx, direction = 'SHORT', current_price, i, 'SHORT'

    if len(trades) > 0:
        trades_df = pd.DataFrame(trades)
        total_return = (equity - 1.0) * 100
        win_rate = (trades_df['realized_pnl'] > 0).sum() / len(trades_df) * 100
        trades_per_week = len(trades_df) / weeks
        avg_holding = trades_df['hold_time_hours'].mean()
        avg_position_size = trades_df['position_size'].mean() * 100

        returns = pd.Series(equity_curve).pct_change().dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(12 * 24 * 7) if returns.std() > 0 else 0

        equity_series = pd.Series(equity_curve)
        drawdown = (equity_series - equity_series.expanding().max()) / equity_series.expanding().max() * 100
        max_dd = drawdown.min()

        return {'total_return': total_return, 'sharpe': sharpe, 'win_rate': win_rate, 'avg_holding_hours': avg_holding,
                'avg_position_size': avg_position_size, 'trades_per_week': trades_per_week, 'max_dd': max_dd, 'total_trades': len(trades_df)}
    return None


# ============================================================================
# Phase 1: Weight Optimization (Reduced Grid - 27 combinations)
# ============================================================================

print("=" * 100)
print("PHASE 1: WEIGHT OPTIMIZATION (Focused Grid)")
print("=" * 100)
print()

# Reduced grid: Focus on most promising ranges
signal_weights = [0.35, 0.40, 0.45]  # 3 values (current 0.4 ± 0.05)
volatility_weights = [0.25, 0.30, 0.35]  # 3 values (current 0.3 ± 0.05)
regime_weights = [0.15, 0.20, 0.25]  # 3 values (current 0.2 ± 0.05)
# streak_weight will be computed as: 1.0 - (sig + vol + reg)

PHASE1_BASE = 0.60
PHASE1_MAX = 1.00
PHASE1_MIN = 0.20

total_combinations = len(signal_weights) * len(volatility_weights) * len(regime_weights)
print(f"Testing {total_combinations} weight combinations (focused grid)")
print(f"  SIGNAL_WEIGHT: {signal_weights}")
print(f"  VOLATILITY_WEIGHT: {volatility_weights}")
print(f"  REGIME_WEIGHT: {regime_weights}")
print(f"  STREAK_WEIGHT: Auto-computed (1.0 - sum)")
print(f"  Fixed: BASE={PHASE1_BASE}, MAX={PHASE1_MAX}, MIN={PHASE1_MIN}")
print()

phase1_results = []
combo_count = 0

for sig_w, vol_w, reg_w in product(signal_weights, volatility_weights, regime_weights):
    str_w = 1.0 - (sig_w + vol_w + reg_w)  # Auto-compute streak weight
    if str_w < 0:  # Skip if negative
        continue

    combo_count += 1
    print(f"Phase 1: {combo_count}/{total_combinations} | SIG={sig_w:.2f} VOL={vol_w:.2f} REG={reg_w:.2f} STR={str_w:.2f}", end='\r')

    sizer = DynamicPositionSizer(base_position_pct=PHASE1_BASE, max_position_pct=PHASE1_MAX, min_position_pct=PHASE1_MIN,
                                 signal_weight=sig_w, volatility_weight=vol_w, regime_weight=reg_w, streak_weight=str_w)

    result = run_backtest(df_test, sizer, feature_columns, long_exit_model, long_exit_scaler, short_exit_model, short_exit_scaler)

    if result:
        result.update({'signal_weight': sig_w, 'volatility_weight': vol_w, 'regime_weight': reg_w, 'streak_weight': str_w,
                      'base_position': PHASE1_BASE, 'max_position': PHASE1_MAX, 'min_position': PHASE1_MIN})
        phase1_results.append(result)

print()

phase1_df = pd.DataFrame(phase1_results).sort_values('total_return', ascending=False)

print()
print("=" * 120)
print("PHASE 1 RESULTS: TOP 10 WEIGHT COMBINATIONS")
print("=" * 120)
print(f"{'Rank':<5} {'SigW':<5} {'VolW':<5} {'RegW':<5} {'StrW':<5} {'Return%':>8} {'Sharpe':>7} {'WinR%':>7} {'AvgPos%':>8} {'Tr/W':>6} {'DD%':>7}")
print("-" * 120)

for idx, row in phase1_df.head(10).iterrows():
    rank = phase1_df.index.get_loc(idx) + 1
    marker = " ← BEST" if rank == 1 else ""
    print(f"{rank:<5} {row['signal_weight']:<5.2f} {row['volatility_weight']:<5.2f} {row['regime_weight']:<5.2f} {row['streak_weight']:<5.2f} "
          f"{row['total_return']:>8.2f} {row['sharpe']:>7.2f} {row['win_rate']:>7.1f} {row['avg_position_size']:>8.1f} "
          f"{row['trades_per_week']:>6.1f} {row['max_dd']:>7.2f}{marker}")

phase1_file = RESULTS_DIR / "position_sizing_weights_optimization_results.csv"
phase1_df.to_csv(phase1_file, index=False)
print()
print(f"✅ Phase 1 results saved: {phase1_file}")

best_weights = phase1_df.iloc[0]


# ============================================================================
# Phase 2: BASE/MAX/MIN Optimization
# ============================================================================

print()
print("=" * 100)
print("PHASE 2: BASE/MAX/MIN OPTIMIZATION (Using Best Weights)")
print("=" * 100)
print()

base_positions = [0.55, 0.60, 0.65]
max_positions = [0.95, 1.00]
min_positions = [0.20]

total_combinations_p2 = len(base_positions) * len(max_positions) * len(min_positions)
print(f"Testing {total_combinations_p2} BASE/MAX/MIN combinations")
print(f"  BASE: {base_positions} | MAX: {max_positions} | MIN: {min_positions}")
print(f"  Best weights: SIG={best_weights['signal_weight']:.2f} VOL={best_weights['volatility_weight']:.2f} "
      f"REG={best_weights['regime_weight']:.2f} STR={best_weights['streak_weight']:.2f}")
print()

phase2_results = []
combo_count = 0

for base_pos, max_pos, min_pos in product(base_positions, max_positions, min_positions):
    combo_count += 1
    print(f"Phase 2: {combo_count}/{total_combinations_p2} | BASE={base_pos:.2f} MAX={max_pos:.2f}", end='\r')

    sizer = DynamicPositionSizer(base_position_pct=base_pos, max_position_pct=max_pos, min_position_pct=min_pos,
                                 signal_weight=best_weights['signal_weight'], volatility_weight=best_weights['volatility_weight'],
                                 regime_weight=best_weights['regime_weight'], streak_weight=best_weights['streak_weight'])

    result = run_backtest(df_test, sizer, feature_columns, long_exit_model, long_exit_scaler, short_exit_model, short_exit_scaler)

    if result:
        result.update({'base_position': base_pos, 'max_position': max_pos, 'min_position': min_pos,
                      'signal_weight': best_weights['signal_weight'], 'volatility_weight': best_weights['volatility_weight'],
                      'regime_weight': best_weights['regime_weight'], 'streak_weight': best_weights['streak_weight']})
        phase2_results.append(result)

print()

phase2_df = pd.DataFrame(phase2_results).sort_values('total_return', ascending=False)

print()
print("=" * 110)
print("PHASE 2 RESULTS: BASE/MAX/MIN COMBINATIONS")
print("=" * 110)
print(f"{'Rank':<5} {'Base%':<6} {'Max%':<5} {'Min%':<5} {'Return%':>8} {'Sharpe':>7} {'WinR%':>7} {'AvgPos%':>8} {'Tr/W':>6} {'DD%':>7}")
print("-" * 110)

for idx, row in phase2_df.iterrows():
    rank = phase2_df.index.get_loc(idx) + 1
    marker = " ← BEST" if rank == 1 else ""
    print(f"{rank:<5} {row['base_position']*100:<6.0f} {row['max_position']*100:<5.0f} {row['min_position']*100:<5.0f} "
          f"{row['total_return']:>8.2f} {row['sharpe']:>7.2f} {row['win_rate']:>7.1f} {row['avg_position_size']:>8.1f} "
          f"{row['trades_per_week']:>6.1f} {row['max_dd']:>7.2f}{marker}")

phase2_file = RESULTS_DIR / "position_sizing_comprehensive_final_results.csv"
phase2_df.to_csv(phase2_file, index=False)
print()
print(f"✅ Phase 2 results saved: {phase2_file}")

best_config = phase2_df.iloc[0]

print()
print("=" * 80)
print("🏆 FINAL BEST CONFIGURATION")
print("=" * 80)
print()
print("Weights:")
print(f"  SIGNAL={best_config['signal_weight']:.2f} | VOL={best_config['volatility_weight']:.2f} | "
      f"REG={best_config['regime_weight']:.2f} | STREAK={best_config['streak_weight']:.2f}")
print()
print("Position Sizing:")
print(f"  BASE={best_config['base_position']:.2f} ({best_config['base_position']*100:.0f}%)")
print(f"  MAX={best_config['max_position']:.2f} ({best_config['max_position']*100:.0f}%)")
print(f"  MIN={best_config['min_position']:.2f} ({best_config['min_position']*100:.0f}%)")
print()
print(f"Performance ({weeks:.1f} weeks):")
print(f"  Return: {best_config['total_return']:.2f}% | Weekly: {best_config['total_return']/weeks:.2f}%")
print(f"  Sharpe: {best_config['sharpe']:.2f} | Win Rate: {best_config['win_rate']:.1f}%")
print(f"  Avg Position: {best_config['avg_position_size']:.1f}% | Trades/Week: {best_config['trades_per_week']:.1f}")
print(f"  Max DD: {best_config['max_dd']:.2f}%")
print()

# Comparison
previous_best = 35.67  # 3 weeks
current_projected = best_config['total_return'] / weeks * 3  # Project to 3 weeks
improvement = ((current_projected - previous_best) / previous_best) * 100

print("=" * 80)
print("📊 COMPARISON (Projected to 3 weeks)")
print("=" * 80)
print(f"Previous (Simplified Logic): {previous_best:.2f}%")
print(f"New (4-Factor Logic): {current_projected:.2f}% (projected)")
print(f"Improvement: {improvement:+.2f}%")
print()
if improvement > 0:
    print(f"✅ SUCCESS! 4-factor optimization is +{improvement:.2f}% better")
else:
    print(f"⚠️ Current config is {improvement:.2f}% vs simplified (validate with longer test)")
print("=" * 80)

elapsed = datetime.now() - start_time
print()
print(f"⏱️ Total time: {elapsed.total_seconds():.1f} seconds")
print()
