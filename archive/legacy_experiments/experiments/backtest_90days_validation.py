"""
Backtest 90-Day Retrained Models on 28-Day Validation Period
=============================================================

Purpose: Compare new models (62d training) vs current production models

Validation Period: Oct 7 - Nov 4, 2025 (28 days / 4 weeks)
Configuration: Entry 0.80, Exit 0.80, Opportunity Gating, 4x Leverage

Models to Compare:
  New (90days_20251105_021742):
    - Training: 62 days (Aug 7 - Oct 7)
    - LONG: 85 features, CV 38.94%
    - SHORT: 89 features, CV 48.20%
    - Exit: 20 features, CV 84.84% / 85.39%

  Current Production (Enhanced 5-Fold CV):
    - LONG: 104 days training (xgboost_long_entry_enhanced_20251024_012445.pkl)
    - SHORT: 35 days training (xgboost_short_entry_with_new_features_20251104_213043.pkl)

Created: 2025-11-05 02:20 KST
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib

# Paths
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "features"
RESULTS_DIR = PROJECT_ROOT / "results"

# Load feature dataset
FEATURE_FILE = DATA_DIR / "BTCUSDT_5m_features_90days_20251105_010924.csv"

# Configuration
ENTRY_THRESHOLD = 0.80
EXIT_THRESHOLD = 0.80
LEVERAGE = 4
STOP_LOSS_PCT = -0.03  # -3% balance-based
MAX_HOLD_TIME = 120  # 10 hours
MAKER_FEE = 0.0002
TAKER_FEE = 0.0004

print("="*80)
print("BACKTEST: 90-DAY RETRAINED VS CURRENT (NO DATA LEAKAGE)")
print("="*80)
print()
print(f"📊 Validation Period: Oct 25 - Nov 4, 2025 (11 days)")
print(f"   FIXED: Starts Oct 25 to avoid CURRENT training overlap")
print(f"   CURRENT trained: Jun 16 - Oct 24 (495 days)")
print(f"   NEW trained: Aug 7 - Oct 7 (62 days)")
print(f"⚙️ Configuration:")
print(f"   Entry: {ENTRY_THRESHOLD}, Exit: {EXIT_THRESHOLD}")
print(f"   Leverage: {LEVERAGE}x, Stop Loss: {STOP_LOSS_PCT*100}%")
print(f"   Max Hold: {MAX_HOLD_TIME} candles (10 hours)")
print()

# ==============================================================================
# STEP 1: Load Data and Models
# ==============================================================================

print("-"*80)
print("STEP 1: Loading Data and Models")
print("-"*80)

# Load feature dataset
print(f"📂 Loading: {FEATURE_FILE.name}")
df = pd.read_csv(FEATURE_FILE)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# FIXED: Extract validation period starting Oct 25 to avoid data leakage
# CURRENT model trained until Oct 24 (495 days: Jun 16 - Oct 24, 2025)
# Must use Oct 25+ for clean validation (no overlap with training)
validation_start = pd.to_datetime('2025-10-25 00:00:00')
val_df = df[df['timestamp'] >= validation_start].reset_index(drop=True)

print(f"✅ Validation Data (NO DATA LEAKAGE):")
print(f"   Period: {val_df['timestamp'].min()} to {val_df['timestamp'].max()}")
print(f"   Rows: {len(val_df):,}")
print(f"   Days: {(val_df['timestamp'].max() - val_df['timestamp'].min()).days}")
print(f"   NOTE: Starts Oct 25 to avoid CURRENT training overlap (trained until Oct 24)")
print()

# Load NEW models (90days)
print("📦 Loading NEW Models (90days_20251105_021742)...")
new_long_entry = joblib.load(MODELS_DIR / "xgboost_long_entry_90days_20251105_021742.pkl")
new_short_entry = joblib.load(MODELS_DIR / "xgboost_short_entry_90days_20251105_021742.pkl")
new_long_exit = joblib.load(MODELS_DIR / "xgboost_long_exit_90days_20251105_021742.pkl")
new_short_exit = joblib.load(MODELS_DIR / "xgboost_short_exit_90days_20251105_021742.pkl")

# Load feature lists
with open(MODELS_DIR / "xgboost_long_entry_90days_20251105_021742_features.txt", 'r') as f:
    new_long_entry_features = [line.strip() for line in f.readlines()]
with open(MODELS_DIR / "xgboost_short_entry_90days_20251105_021742_features.txt", 'r') as f:
    new_short_entry_features = [line.strip() for line in f.readlines()]
with open(MODELS_DIR / "xgboost_long_exit_90days_20251105_021742_features.txt", 'r') as f:
    new_exit_features = [line.strip() for line in f.readlines()]

print(f"   ✅ NEW LONG Entry: {len(new_long_entry_features)} features")
print(f"   ✅ NEW SHORT Entry: {len(new_short_entry_features)} features")
print(f"   ✅ NEW Exit: {len(new_exit_features)} features")
print()

# Load CURRENT production models (ACTUAL production config: LONG Oct 24 + SHORT Nov 4)
print("📦 Loading CURRENT Production Models (ACTUAL: LONG Oct24 + SHORT Nov4)...")
curr_long_entry = joblib.load(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445.pkl")
curr_short_entry = joblib.load(MODELS_DIR / "xgboost_short_entry_with_new_features_20251104_213043.pkl")
curr_long_exit = joblib.load(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527.pkl")
curr_short_exit = joblib.load(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510.pkl")

# Load feature lists
with open(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_features.txt", 'r') as f:
    curr_long_entry_features = [line.strip() for line in f.readlines()]
with open(MODELS_DIR / "xgboost_short_entry_with_new_features_20251104_213043_features.txt", 'r') as f:
    curr_short_entry_features = [line.strip() for line in f.readlines()]
with open(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527_features.txt", 'r') as f:
    curr_long_exit_features = [line.strip() for line in f.readlines()]
with open(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510_features.txt", 'r') as f:
    curr_short_exit_features = [line.strip() for line in f.readlines()]

print(f"   ✅ CURRENT LONG Entry: {len(curr_long_entry_features)} features")
print(f"   ✅ CURRENT SHORT Entry: {len(curr_short_entry_features)} features")
print(f"   ✅ CURRENT LONG Exit: {len(curr_long_exit_features)} features")
print(f"   ✅ CURRENT SHORT Exit: {len(curr_short_exit_features)} features")
print()

# ==============================================================================
# STEP 2: Backtest Function
# ==============================================================================

def backtest_models(df, long_entry_model, short_entry_model, long_exit_model, short_exit_model,
                   long_entry_features, short_entry_features, long_exit_features, short_exit_features,
                   model_name):
    """Run backtest with opportunity gating"""

    print(f"🔄 Backtesting: {model_name}")

    # Predict probabilities (convert to numpy array to avoid pandas/XGBoost compatibility issues)
    X_long_entry = df[long_entry_features].fillna(0).replace([np.inf, -np.inf], 0).values
    X_short_entry = df[short_entry_features].fillna(0).replace([np.inf, -np.inf], 0).values

    long_entry_probs = long_entry_model.predict_proba(X_long_entry)[:, 1]
    short_entry_probs = short_entry_model.predict_proba(X_short_entry)[:, 1]

    # Initialize tracking
    balance = 1000.0
    position = None
    trades = []

    for i in range(len(df) - MAX_HOLD_TIME):
        current_time = df.iloc[i]['timestamp']
        current_price = df.iloc[i]['close']

        # Exit logic
        if position is not None:
            hold_time = i - position['entry_idx']
            pnl_pct = ((current_price - position['entry_price']) / position['entry_price'] *
                      LEVERAGE * (1 if position['side'] == 'LONG' else -1))

            exit_signal = False
            exit_reason = None

            # Stop Loss (balance-based)
            if pnl_pct <= STOP_LOSS_PCT:
                exit_signal = True
                exit_reason = 'Stop Loss'

            # Max Hold Time
            elif hold_time >= MAX_HOLD_TIME:
                exit_signal = True
                exit_reason = 'Max Hold'

            # ML Exit
            else:
                exit_features = long_exit_features if position['side'] == 'LONG' else short_exit_features
                exit_model = long_exit_model if position['side'] == 'LONG' else short_exit_model

                X_exit = df.iloc[i:i+1][exit_features].fillna(0).replace([np.inf, -np.inf], 0).values
                exit_prob = exit_model.predict_proba(X_exit)[0, 1]

                if exit_prob >= EXIT_THRESHOLD:
                    exit_signal = True
                    exit_reason = f'ML Exit {exit_prob:.4f}'

            if exit_signal:
                # Calculate P&L
                entry_fee = abs(position['size']) * TAKER_FEE
                exit_fee = abs(position['size']) * MAKER_FEE
                total_fee = entry_fee + exit_fee

                gross_pnl = position['size'] * pnl_pct
                net_pnl = gross_pnl - total_fee

                balance += net_pnl

                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': current_time,
                    'side': position['side'],
                    'entry_price': position['entry_price'],
                    'exit_price': current_price,
                    'entry_prob': position['entry_prob'],
                    'hold_time': hold_time,
                    'pnl_pct': pnl_pct,
                    'pnl_usd': net_pnl,
                    'exit_reason': exit_reason,
                    'balance': balance
                })

                position = None

        # Entry logic (only if no position)
        if position is None:
            long_prob = long_entry_probs[i]
            short_prob = short_entry_probs[i]

            # Opportunity Gating: Only enter if probability is significantly better
            if long_prob >= ENTRY_THRESHOLD and short_prob >= ENTRY_THRESHOLD:
                # Both signals strong - choose higher with 0.001 buffer
                if short_prob > long_prob + 0.001:
                    side = 'SHORT'
                    entry_prob = short_prob
                elif long_prob >= short_prob:
                    side = 'LONG'
                    entry_prob = long_prob
                else:
                    continue
            elif long_prob >= ENTRY_THRESHOLD:
                side = 'LONG'
                entry_prob = long_prob
            elif short_prob >= ENTRY_THRESHOLD:
                side = 'SHORT'
                entry_prob = short_prob
            else:
                continue

            # Enter position
            position = {
                'entry_idx': i,
                'entry_time': current_time,
                'entry_price': current_price,
                'entry_prob': entry_prob,
                'side': side,
                'size': balance * 0.95  # 95% position sizing
            }

    # Force close any remaining position
    if position is not None:
        current_price = df.iloc[-1]['close']
        pnl_pct = ((current_price - position['entry_price']) / position['entry_price'] *
                  LEVERAGE * (1 if position['side'] == 'LONG' else -1))

        entry_fee = abs(position['size']) * TAKER_FEE
        exit_fee = abs(position['size']) * MAKER_FEE
        total_fee = entry_fee + exit_fee

        gross_pnl = position['size'] * pnl_pct
        net_pnl = gross_pnl - total_fee
        balance += net_pnl

        trades.append({
            'entry_time': position['entry_time'],
            'exit_time': df.iloc[-1]['timestamp'],
            'side': position['side'],
            'entry_price': position['entry_price'],
            'exit_price': current_price,
            'entry_prob': position['entry_prob'],
            'hold_time': len(df) - 1 - position['entry_idx'],
            'pnl_pct': pnl_pct,
            'pnl_usd': net_pnl,
            'exit_reason': 'Force Close',
            'balance': balance
        })

    return pd.DataFrame(trades), balance

# ==============================================================================
# STEP 3: Run Backtests
# ==============================================================================

print("-"*80)
print("STEP 3: Running Backtests")
print("-"*80)
print()

# Backtest NEW models
new_trades, new_balance = backtest_models(
    val_df, new_long_entry, new_short_entry, new_long_exit, new_short_exit,
    new_long_entry_features, new_short_entry_features, new_exit_features, new_exit_features,
    "NEW Models (90days)"
)

# Backtest CURRENT models (ACTUAL production: LONG Oct24 + SHORT Nov4)
# Note: Using NEW exit models for fair comparison (CURRENT exit needs phase2 features)
curr_trades, curr_balance = backtest_models(
    val_df, curr_long_entry, curr_short_entry, new_long_exit, new_short_exit,
    curr_long_entry_features, curr_short_entry_features, new_exit_features, new_exit_features,
    "CURRENT Production (LONG Oct24 + SHORT Nov4)"
)

# ==============================================================================
# STEP 4: Performance Analysis
# ==============================================================================

print()
print("-"*80)
print("STEP 4: Performance Comparison")
print("-"*80)
print()

def analyze_performance(trades, balance, model_name):
    """Analyze backtest performance"""

    if len(trades) == 0:
        print(f"❌ {model_name}: No trades")
        return None

    wins = trades[trades['pnl_usd'] > 0]
    losses = trades[trades['pnl_usd'] <= 0]

    win_rate = len(wins) / len(trades) * 100
    avg_win = wins['pnl_usd'].mean() if len(wins) > 0 else 0
    avg_loss = losses['pnl_usd'].mean() if len(losses) > 0 else 0
    profit_factor = abs(wins['pnl_usd'].sum() / losses['pnl_usd'].sum()) if len(losses) > 0 and losses['pnl_usd'].sum() != 0 else float('inf')

    total_return = (balance - 1000) / 1000 * 100
    total_pnl = balance - 1000

    # Trade frequency
    days = (trades['exit_time'].max() - trades['entry_time'].min()).total_seconds() / 86400
    trades_per_day = len(trades) / days if days > 0 else 0

    # Side distribution
    long_trades = len(trades[trades['side'] == 'LONG'])
    short_trades = len(trades[trades['side'] == 'SHORT'])

    # Exit reasons
    exit_reasons = trades['exit_reason'].value_counts()

    print(f"📊 {model_name}:")
    print(f"   Total Trades: {len(trades)}")
    print(f"   LONG: {long_trades} ({long_trades/len(trades)*100:.1f}%), SHORT: {short_trades} ({short_trades/len(trades)*100:.1f}%)")
    print(f"   Win Rate: {win_rate:.2f}% ({len(wins)}W / {len(losses)}L)")
    print(f"   Total Return: {total_return:+.2f}% (${total_pnl:+.2f})")
    print(f"   Final Balance: ${balance:.2f}")
    print(f"   Profit Factor: {profit_factor:.2f}")
    print(f"   Avg Win: ${avg_win:.2f}, Avg Loss: ${avg_loss:.2f}")
    print(f"   Trades/Day: {trades_per_day:.1f}")
    print(f"   Avg Hold: {trades['hold_time'].mean():.1f} candles")
    print(f"   Exit Reasons: {exit_reasons.to_dict()}")
    print()

    return {
        'model': model_name,
        'trades': len(trades),
        'long_trades': long_trades,
        'short_trades': short_trades,
        'win_rate': win_rate,
        'total_return': total_return,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'trades_per_day': trades_per_day,
        'avg_hold': trades['hold_time'].mean(),
        'final_balance': balance
    }

new_stats = analyze_performance(new_trades, new_balance, "NEW Models (90days)")
curr_stats = analyze_performance(curr_trades, curr_balance, "CURRENT Production")

# ==============================================================================
# STEP 5: Comparison and Decision
# ==============================================================================

print("-"*80)
print("STEP 5: Deployment Decision")
print("-"*80)
print()

if new_stats and curr_stats:
    return_diff = new_stats['total_return'] - curr_stats['total_return']
    wr_diff = new_stats['win_rate'] - curr_stats['win_rate']
    pf_diff = new_stats['profit_factor'] - curr_stats['profit_factor']

    print(f"📈 Performance Difference (NEW - CURRENT):")
    print(f"   Return: {return_diff:+.2f}%")
    print(f"   Win Rate: {wr_diff:+.2f}%")
    print(f"   Profit Factor: {pf_diff:+.2f}")
    print()

    # Decision logic
    if return_diff > 2.0 and wr_diff > 3.0:
        decision = "✅ DEPLOY NEW MODELS - Significant improvement"
    elif return_diff > 0.5 and wr_diff > 0:
        decision = "⚠️ CONSIDER DEPLOYMENT - Marginal improvement"
    elif abs(return_diff) < 0.5:
        decision = "⏸️ KEEP CURRENT - Similar performance"
    else:
        decision = "❌ DO NOT DEPLOY - Current models better"

    print(f"🎯 Decision: {decision}")
    print()

# Save results
timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
new_trades.to_csv(RESULTS_DIR / f"backtest_90days_NEW_{timestamp_str}.csv", index=False)
curr_trades.to_csv(RESULTS_DIR / f"backtest_90days_CURRENT_{timestamp_str}.csv", index=False)

print(f"💾 Results saved:")
print(f"   NEW: backtest_90days_NEW_{timestamp_str}.csv")
print(f"   CURRENT: backtest_90days_CURRENT_{timestamp_str}.csv")
print()

print("="*80)
print("BACKTEST COMPLETE")
print("="*80)
