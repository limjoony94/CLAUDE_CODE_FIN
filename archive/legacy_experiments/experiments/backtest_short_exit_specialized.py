"""
Backtest SHORT-Specialized Exit Model vs Current System
========================================================

Compare:
- Current: Threshold 0.72 + generic exit model
- New: Threshold 0.72 + SHORT-specialized exit model

Metrics:
- Opportunity cost (target: <-0.5% vs -2.27% original)
- Late exits % (target: <30% vs 61.9% original)
- SHORT win rate (target: >79% vs 79.3% with 0.72)
- Total return per window

Author: Claude Code
Date: 2025-10-18
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys

# Add parent directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.experiments.calculate_all_features import calculate_all_features
from scripts.experiments.reversal_detection_features import add_all_reversal_features

# Configuration
DATA_FILE = project_root / "data/historical/BTCUSDT_5m_max.csv"
MODELS_DIR = project_root / "models"

# Entry models
LONG_ENTRY_MODEL = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl"
LONG_ENTRY_FEATURES_FILE = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt"
SHORT_ENTRY_MODEL = MODELS_DIR / "xgboost_short_redesigned_20251016_233322.pkl"
SHORT_ENTRY_FEATURES_FILE = MODELS_DIR / "xgboost_short_redesigned_20251016_233322_features.txt"

# Exit models (current SHORT exit model used by bot)
CURRENT_EXIT_MODEL = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251017_152440.pkl"
CURRENT_EXIT_SCALER = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251017_152440_scaler.pkl"

# Find the latest specialized model (exclude scaler files)
specialized_models = [
    f for f in MODELS_DIR.glob("xgboost_short_exit_specialized_*.pkl")
    if not f.stem.endswith("_scaler")
]
if not specialized_models:
    raise FileNotFoundError("No specialized SHORT exit model found!")

NEW_EXIT_MODEL = sorted(specialized_models)[-1]
NEW_EXIT_SCALER = NEW_EXIT_MODEL.parent / f"{NEW_EXIT_MODEL.stem}_scaler.pkl"
NEW_EXIT_FEATURES = NEW_EXIT_MODEL.parent / f"{NEW_EXIT_MODEL.stem}_features.txt"

print("="*80)
print("Backtest SHORT-Specialized Exit Model vs Current System")
print("="*80)
print(f"\nEntry Models:")
print(f"  LONG: {LONG_ENTRY_MODEL.name}")
print(f"  SHORT: {SHORT_ENTRY_MODEL.name}")
print(f"\nExit Models:")
print(f"  Current: {CURRENT_EXIT_MODEL.name}")
print(f"  New (Specialized): {NEW_EXIT_MODEL.name}")

# Thresholds
LONG_THRESHOLD = 0.65
SHORT_THRESHOLD = 0.70
GATE_THRESHOLD = 0.001
EXIT_THRESHOLD = 0.72  # Same threshold for fair comparison

# Leverage & position sizing
LEVERAGE = 4
MIN_POSITION_SIZE = 0.20
MAX_POSITION_SIZE = 0.95

# Exit rules
MAX_HOLD_CANDLES = 240  # 20 hours
TAKE_PROFIT_LEVERAGED = 0.03  # 3% on 4x position
STOP_LOSS_LEVERAGED = -0.015  # -1.5% on 4x position

# Performance tracking
LONG_AVG_RETURN = 0.0041
SHORT_AVG_RETURN = 0.0047

print("\n" + "="*80)
print("Loading data and models...")
print("="*80)

# Load data
df = pd.read_csv(DATA_FILE)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)
print(f"✅ {len(df):,} candles loaded")

# Calculate features
print("\nCalculating features...")
df = calculate_all_features(df)
print("✅ Standard features calculated")

# Calculate enhanced market context features (if not already present)
if 'volume_ratio' not in df.columns:
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
if 'price_vs_ma20' not in df.columns:
    df['price_vs_ma20'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).mean()
if 'price_vs_ma50' not in df.columns:
    df['price_vs_ma50'] = (df['close'] - df['close'].rolling(50).mean()) / df['close'].rolling(50).mean()
if 'volatility_20' not in df.columns:
    df['volatility_20'] = df['close'].pct_change().rolling(20).std()
if 'rsi_slope' not in df.columns:
    df['rsi_slope'] = df['rsi'].diff(5)
if 'rsi_overbought' not in df.columns:
    df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
if 'rsi_oversold' not in df.columns:
    df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
if 'volume_surge' not in df.columns:
    df['volume_surge'] = (df['volume'] > df['volume'].rolling(20).mean() * 2).astype(int)
if 'price_acceleration' not in df.columns:
    df['price_acceleration'] = df['close'].pct_change().diff()
if 'trend_strength' not in df.columns:
    df['trend_strength'] = abs(df['close'] - df['close'].rolling(20).mean()) / df['atr']
if 'volatility_regime' not in df.columns:
    df['volatility_regime'] = (df['volatility_20'] > df['volatility_20'].rolling(100).mean()).astype(int)
print("✅ Enhanced context features calculated")

# Calculate reversal detection features
df = add_all_reversal_features(df)
print("✅ Reversal detection features calculated")

# Clean NaN
df = df.dropna().reset_index(drop=True)
print(f"✅ Cleaned data: {len(df):,} candles")

# Load entry models
with open(LONG_ENTRY_MODEL, 'rb') as f:
    long_entry_model = pickle.load(f)
with open(LONG_ENTRY_FEATURES_FILE, 'r') as f:
    long_entry_features = [line.strip() for line in f]

with open(SHORT_ENTRY_MODEL, 'rb') as f:
    short_entry_model = pickle.load(f)
with open(SHORT_ENTRY_FEATURES_FILE, 'r') as f:
    short_entry_features = [line.strip() for line in f]

# Load exit models
with open(CURRENT_EXIT_MODEL, 'rb') as f:
    current_exit_model = pickle.load(f)
with open(CURRENT_EXIT_SCALER, 'rb') as f:
    current_exit_scaler = pickle.load(f)

with open(NEW_EXIT_MODEL, 'rb') as f:
    new_exit_model = pickle.load(f)
with open(NEW_EXIT_SCALER, 'rb') as f:
    new_exit_scaler = pickle.load(f)

# Load features from files
CURRENT_EXIT_FEATURES_FILE = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251017_152440_features.txt"
with open(CURRENT_EXIT_FEATURES_FILE, 'r') as f:
    current_exit_features = [line.strip() for line in f]

with open(NEW_EXIT_FEATURES, 'r') as f:
    new_exit_features = [line.strip() for line in f]

print("✅ All models loaded")

print(f"\nFeature comparison:")
print(f"  Current exit model: {len(current_exit_features)} features")
print(f"  New exit model: {len(new_exit_features)} features")


def calculate_position_size(prob, side):
    """Calculate dynamic position size (20-95%)"""
    if side == 'LONG':
        kelly = (prob * (1 + LONG_AVG_RETURN) - (1 - prob)) / LONG_AVG_RETURN
    else:  # SHORT
        kelly = (prob * (1 + SHORT_AVG_RETURN) - (1 - prob)) / SHORT_AVG_RETURN

    kelly = max(0, min(kelly, 1))
    position_size = MIN_POSITION_SIZE + kelly * (MAX_POSITION_SIZE - MIN_POSITION_SIZE)
    return np.clip(position_size, MIN_POSITION_SIZE, MAX_POSITION_SIZE)


def backtest_system(df, exit_model, exit_scaler, exit_features, system_name):
    """Run backtest with specified exit model"""

    print(f"\n{'='*80}")
    print(f"Backtesting: {system_name}")
    print(f"{'='*80}")

    balance = 10000.0
    position = None
    trades = []

    for i in range(100, len(df)):
        current_price = df['close'].iloc[i]
        current_time = df['timestamp'].iloc[i]

        # Entry logic (same for both systems)
        if position is None:
            # Get entry probabilities
            long_prob = long_entry_model.predict_proba(df[long_entry_features].iloc[i:i+1].values)[0, 1]
            short_prob = short_entry_model.predict_proba(df[short_entry_features].iloc[i:i+1].values)[0, 1]

            # LONG entry
            if long_prob >= LONG_THRESHOLD:
                position_size_pct = calculate_position_size(long_prob, 'LONG')
                position_value = balance * position_size_pct
                leveraged_value = position_value * LEVERAGE
                quantity = leveraged_value / current_price

                position = {
                    'side': 'LONG',
                    'entry_price': current_price,
                    'entry_time': current_time,
                    'entry_idx': i,
                    'quantity': quantity,
                    'position_value': position_value,
                    'leveraged_value': leveraged_value,
                    'probability': long_prob,
                    'position_size_pct': position_size_pct
                }

            # SHORT entry (with opportunity gating)
            elif short_prob >= SHORT_THRESHOLD:
                long_ev = long_prob * LONG_AVG_RETURN
                short_ev = short_prob * SHORT_AVG_RETURN
                opportunity_cost = short_ev - long_ev

                if opportunity_cost > GATE_THRESHOLD:
                    position_size_pct = calculate_position_size(short_prob, 'SHORT')
                    position_value = balance * position_size_pct
                    leveraged_value = position_value * LEVERAGE
                    quantity = leveraged_value / current_price

                    position = {
                        'side': 'SHORT',
                        'entry_price': current_price,
                        'entry_time': current_time,
                        'entry_idx': i,
                        'quantity': quantity,
                        'position_value': position_value,
                        'leveraged_value': leveraged_value,
                        'probability': short_prob,
                        'position_size_pct': position_size_pct,
                        'opportunity_cost': opportunity_cost
                    }

        # Exit logic (different for each system)
        else:
            hold_time = i - position['entry_idx']

            # Calculate P&L
            if position['side'] == 'LONG':
                price_change = (current_price - position['entry_price']) / position['entry_price']
            else:  # SHORT
                price_change = (position['entry_price'] - current_price) / position['entry_price']

            leveraged_pnl = price_change * LEVERAGE

            # Emergency exits
            exit_reason = None
            if leveraged_pnl <= STOP_LOSS_LEVERAGED:
                exit_reason = f"Emergency Stop Loss ({leveraged_pnl:.2%})"
            elif leveraged_pnl >= TAKE_PROFIT_LEVERAGED:
                exit_reason = f"Take Profit ({leveraged_pnl:.2%})"
            elif hold_time >= MAX_HOLD_CANDLES:
                exit_reason = f"Max Hold ({hold_time} candles)"
            else:
                # ML exit signal
                X = df[exit_features].iloc[i:i+1].values
                X_scaled = exit_scaler.transform(X)
                exit_prob = exit_model.predict_proba(X_scaled)[0, 1]

                if exit_prob >= EXIT_THRESHOLD:
                    exit_reason = f"ML Exit (prob={exit_prob:.4f})"

            if exit_reason:
                # Close position
                pnl_usd = position['position_value'] * leveraged_pnl
                balance += pnl_usd

                # Calculate exit quality metrics
                max_favorable_price = None
                max_favorable_pnl = None
                exit_timing = None

                # Look ahead to find best exit in next 240 candles
                future_window = df.iloc[i:min(i+240, len(df))]
                if position['side'] == 'LONG':
                    best_price = future_window['high'].max()
                    max_favorable_pnl = (best_price - position['entry_price']) / position['entry_price'] * LEVERAGE
                else:  # SHORT
                    best_price = future_window['low'].min()
                    max_favorable_pnl = (position['entry_price'] - best_price) / position['entry_price'] * LEVERAGE

                opportunity_cost_pct = max_favorable_pnl - leveraged_pnl

                # Classify exit timing
                if max_favorable_pnl < 0:
                    exit_timing = 'GOOD (avoided worse)'
                elif leveraged_pnl > 0 and opportunity_cost_pct < 0.005:
                    exit_timing = 'GOOD (near peak)'
                elif leveraged_pnl > 0 and opportunity_cost_pct >= 0.005:
                    exit_timing = 'LATE (gave back profit)'
                elif leveraged_pnl < 0:
                    exit_timing = 'EARLY (missed profit)'
                else:
                    exit_timing = 'NEUTRAL'

                trade = {
                    'side': position['side'],
                    'entry_price': position['entry_price'],
                    'entry_time': position['entry_time'],
                    'exit_price': current_price,
                    'exit_time': current_time,
                    'hold_candles': hold_time,
                    'pnl_pct': leveraged_pnl,
                    'pnl_usd': pnl_usd,
                    'exit_reason': exit_reason,
                    'position_size_pct': position['position_size_pct'],
                    'probability': position['probability'],
                    'max_favorable_pnl': max_favorable_pnl,
                    'opportunity_cost_pct': opportunity_cost_pct,
                    'exit_timing': exit_timing
                }

                trades.append(trade)
                position = None

    # Analysis
    df_trades = pd.DataFrame(trades)

    print(f"\n{'='*80}")
    print(f"Results: {system_name}")
    print(f"{'='*80}")

    if len(df_trades) == 0:
        print("❌ No trades executed")
        return None

    print(f"\nOverall Performance:")
    print(f"  Total Trades: {len(df_trades)}")
    print(f"  Final Balance: ${balance:,.2f}")
    print(f"  Total Return: {(balance - 10000) / 10000:.2%}")
    print(f"  Win Rate: {(df_trades['pnl_usd'] > 0).mean():.1%}")

    # LONG vs SHORT
    long_trades = df_trades[df_trades['side'] == 'LONG']
    short_trades = df_trades[df_trades['side'] == 'SHORT']

    print(f"\nLONG Trades: {len(long_trades)} ({len(long_trades)/len(df_trades):.1%})")
    if len(long_trades) > 0:
        print(f"  Win Rate: {(long_trades['pnl_usd'] > 0).mean():.1%}")
        print(f"  Avg Return: {long_trades['pnl_pct'].mean():.2%}")
        print(f"  Total P&L: ${long_trades['pnl_usd'].sum():.2f}")

    print(f"\nSHORT Trades: {len(short_trades)} ({len(short_trades)/len(df_trades):.1%})")
    if len(short_trades) > 0:
        print(f"  Win Rate: {(short_trades['pnl_usd'] > 0).mean():.1%}")
        print(f"  Avg Return: {short_trades['pnl_pct'].mean():.2%}")
        print(f"  Total P&L: ${short_trades['pnl_usd'].sum():.2f}")

    # Exit timing analysis (SHORT only)
    if len(short_trades) > 0:
        print(f"\nSHORT Exit Timing Quality:")
        timing_counts = short_trades['exit_timing'].value_counts()
        for timing, count in timing_counts.items():
            pct = count / len(short_trades) * 100
            print(f"  {timing}: {count} ({pct:.1f}%)")

        # Opportunity cost
        avg_opp_cost = short_trades['opportunity_cost_pct'].mean()
        print(f"\n  Avg Opportunity Cost: {avg_opp_cost:.2%}")

        # Late exits specifically
        late_exits = short_trades[short_trades['exit_timing'] == 'LATE (gave back profit)']
        if len(late_exits) > 0:
            late_pct = len(late_exits) / len(short_trades) * 100
            avg_late_cost = late_exits['opportunity_cost_pct'].mean()
            print(f"  Late Exits: {len(late_exits)} ({late_pct:.1f}%)")
            print(f"  Late Exit Opportunity Cost: {avg_late_cost:.2%}")

    # Return metrics
    return {
        'system_name': system_name,
        'total_trades': len(df_trades),
        'final_balance': balance,
        'total_return': (balance - 10000) / 10000,
        'win_rate': (df_trades['pnl_usd'] > 0).mean(),
        'long_trades': len(long_trades),
        'long_win_rate': (long_trades['pnl_usd'] > 0).mean() if len(long_trades) > 0 else 0,
        'short_trades': len(short_trades),
        'short_win_rate': (short_trades['pnl_usd'] > 0).mean() if len(short_trades) > 0 else 0,
        'short_avg_opp_cost': short_trades['opportunity_cost_pct'].mean() if len(short_trades) > 0 else 0,
        'short_late_exits_pct': (short_trades['exit_timing'] == 'LATE (gave back profit)').mean() * 100 if len(short_trades) > 0 else 0,
        'trades_df': df_trades
    }


# Run backtests
print("\n" + "="*80)
print("Running Backtests")
print("="*80)

current_results = backtest_system(
    df,
    current_exit_model,
    current_exit_scaler,
    current_exit_features,
    "Current System (Generic Exit + Threshold 0.72)"
)

new_results = backtest_system(
    df,
    new_exit_model,
    new_exit_scaler,
    new_exit_features,
    "New System (SHORT-Specialized Exit + Threshold 0.72)"
)

# Comparison
if current_results and new_results:
    print("\n" + "="*80)
    print("COMPARISON: New vs Current")
    print("="*80)

    print(f"\nTotal Return:")
    print(f"  Current: {current_results['total_return']:.2%}")
    print(f"  New: {new_results['total_return']:.2%}")
    print(f"  Improvement: {(new_results['total_return'] - current_results['total_return']):.2%} ({(new_results['total_return'] / current_results['total_return'] - 1) * 100:+.1f}%)")

    print(f"\nSHORT Performance:")
    print(f"  Win Rate:")
    print(f"    Current: {current_results['short_win_rate']:.1%}")
    print(f"    New: {new_results['short_win_rate']:.1%}")
    print(f"    Change: {(new_results['short_win_rate'] - current_results['short_win_rate']) * 100:+.1f} pp")

    print(f"\n  Opportunity Cost:")
    print(f"    Current: {current_results['short_avg_opp_cost']:.2%}")
    print(f"    New: {new_results['short_avg_opp_cost']:.2%}")
    print(f"    Improvement: {(current_results['short_avg_opp_cost'] - new_results['short_avg_opp_cost']):.2%}")

    print(f"\n  Late Exits:")
    print(f"    Current: {current_results['short_late_exits_pct']:.1f}%")
    print(f"    New: {new_results['short_late_exits_pct']:.1f}%")
    print(f"    Reduction: {(current_results['short_late_exits_pct'] - new_results['short_late_exits_pct']):.1f} pp")

    print(f"\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)

    # Decision logic
    improvements = 0

    if new_results['total_return'] > current_results['total_return']:
        improvements += 1
        print("✅ Total return improved")
    else:
        print("❌ Total return declined")

    if new_results['short_win_rate'] > current_results['short_win_rate']:
        improvements += 1
        print("✅ SHORT win rate improved")
    else:
        print("❌ SHORT win rate declined")

    if new_results['short_avg_opp_cost'] > current_results['short_avg_opp_cost']:  # Less negative = better
        improvements += 1
        print("✅ SHORT opportunity cost improved")
    else:
        print("❌ SHORT opportunity cost worsened")

    if new_results['short_late_exits_pct'] < current_results['short_late_exits_pct']:
        improvements += 1
        print("✅ SHORT late exits reduced")
    else:
        print("❌ SHORT late exits increased")

    print(f"\nScore: {improvements}/4 metrics improved")

    if improvements >= 3:
        print("\n🎉 DEPLOY NEW MODEL - Strong improvement across metrics")
    elif improvements == 2:
        print("\n⚠️ BORDERLINE - Consider deployment if total return improvement is significant")
    else:
        print("\n❌ KEEP CURRENT - New model did not show sufficient improvement")

    print("\n" + "="*80)
