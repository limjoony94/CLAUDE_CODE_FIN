"""
Retrain Entry Models - FIXED Walk-Forward (Performance-Based Selection)
=========================================================================

CRITICAL FIX: Model selection based on ACTUAL PERFORMANCE (not prediction rate)

Previous Bug:
  - Selected model with highest prediction rate (% of positive predictions)
  - Resulted in models that predict many entries but with terrible win rate
  - Example: 40.47% WR, -9.83% return (LOSING MONEY!)

Fixed Approach:
  - Run mini-backtest on each fold's validation set
  - Calculate win rate and return for each fold's model
  - Select model with best performance (composite score: 70% WR + 30% return)

Expected Result:
  - Models that predict FEWER but BETTER entries
  - Higher win rate (target: > 60%)
  - Positive returns

Created: 2025-10-27 (FIXED VERSION)
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import pickle
import joblib
from datetime import datetime
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

# Configuration
ENTRY_THRESHOLD = 0.75
ML_EXIT_THRESHOLD = 0.75
LEVERAGE = 4
EMERGENCY_STOP_LOSS = -0.03
EMERGENCY_MAX_HOLD = 120
INITIAL_CAPITAL = 10000

DATA_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models"

print("="*80)
print("FIXED WALK-FORWARD ENTRY MODEL TRAINING")
print("="*80)
print()
print("🔧 CRITICAL FIX: Performance-Based Model Selection")
print("   Previous: Selected by prediction rate (quantity)")
print("   Fixed: Selected by win rate + return (quality)")
print()
print(f"Configuration:")
print(f"  Entry Threshold: {ENTRY_THRESHOLD}")
print(f"  ML Exit Threshold: {ML_EXIT_THRESHOLD}")
print(f"  Emergency Stop Loss: {EMERGENCY_STOP_LOSS}")
print(f"  Emergency Max Hold: {EMERGENCY_MAX_HOLD} candles")
print()

# Load Data
print("-"*80)
print("STEP 1: Loading Data")
print("-"*80)

df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_features.csv")
print(f"✅ Loaded {len(df):,} candles with {len(df.columns)} features")
print(f"   Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
print()

# Load Entry Feature Lists
print("-"*80)
print("STEP 2: Loading Entry Feature Lists")
print("-"*80)

with open(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_features.txt", 'r') as f:
    long_entry_features = [line.strip() for line in f]

with open(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_features.txt", 'r') as f:
    short_entry_features = [line.strip() for line in f]

print(f"  ✅ LONG Entry: {len(long_entry_features)} features")
print(f"  ✅ SHORT Entry: {len(short_entry_features)} features")
print()

# ==============================================================================
# Candidate Filtering (Option A - Unchanged)
# ==============================================================================

def filter_entry_candidates(df, side):
    """Pre-filter candidates using simple heuristics"""
    print(f"\n🔍 Filtering {side} Entry Candidates...")

    mask = pd.Series(True, index=df.index)

    # Volume filter
    if 'volume' in df.columns:
        volume_filter = df['volume'] > df['volume'].rolling(20).mean()
        mask = mask & volume_filter

    # RSI filter
    if 'rsi' in df.columns:
        rsi_filter = (df['rsi'] > 30) & (df['rsi'] < 70)
        mask = mask & rsi_filter

    if side == 'LONG':
        if 'ma_50' in df.columns:
            trend_filter = df['close'] > df['ma_50']
            mask = mask & trend_filter
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            macd_filter = df['macd'] > df['macd_signal']
            mask = mask & macd_filter
    else:
        if 'ma_50' in df.columns:
            trend_filter = df['close'] < df['ma_50']
            mask = mask & trend_filter
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            macd_filter = df['macd'] < df['macd_signal']
            mask = mask & macd_filter

    valid_range = (df.index >= 100) & (df.index < len(df) - EMERGENCY_MAX_HOLD)
    mask = mask & valid_range

    candidates = df[mask].index.tolist()

    total_possible = len(df) - 100 - EMERGENCY_MAX_HOLD
    reduction_pct = (1 - len(candidates) / total_possible) * 100

    print(f"  After filtering: {len(candidates):,} ({reduction_pct:.1f}% reduction)")

    return candidates

# ==============================================================================
# Rule-Based Exit Labels (Option C - Unchanged)
# ==============================================================================

def label_exits_rule_based(df, side):
    """Create rule-based exit labels"""
    print(f"\n🔨 Creating Rule-Based Exit Labels for {side}...")

    exit_labels = np.zeros(len(df))

    for i in range(100, len(df) - EMERGENCY_MAX_HOLD):
        entry_price = df['close'].iloc[i]

        for j in range(1, EMERGENCY_MAX_HOLD + 1):
            if i + j >= len(df):
                break

            current_price = df['close'].iloc[i + j]

            if side == 'LONG':
                pnl = (current_price - entry_price) / entry_price
            else:
                pnl = (entry_price - current_price) / entry_price

            leveraged_pnl = pnl * LEVERAGE

            if leveraged_pnl > 0.02 and j < 60:
                exit_labels[i + j] = 1
                break
            elif leveraged_pnl <= EMERGENCY_STOP_LOSS:
                break

    positive = (exit_labels == 1).sum()
    print(f"  Good exits: {positive:,} ({positive/len(df)*100:.2f}%)")

    return exit_labels

def train_exit_model_fold(df, side):
    """Train Exit model using rule-based labels"""

    exit_labels = label_exits_rule_based(df, side)

    EXIT_FEATURES = [
        'close', 'volume', 'volume_surge', 'price_acceleration',
        'ma_20', 'ma_50', 'price_vs_ma20', 'price_vs_ma50',
        'volatility_20',
        'rsi', 'rsi_slope', 'rsi_overbought', 'rsi_oversold', 'rsi_divergence',
        'macd', 'macd_signal', 'macd_histogram', 'macd_histogram_slope',
        'macd_crossover', 'macd_crossunder',
        'bb_upper', 'bb_lower', 'bb_position',
        'higher_high', 'near_support'
    ]

    available_features = [f for f in EXIT_FEATURES if f in df.columns]

    X = df[available_features].values
    y = exit_labels

    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[mask]
    y = y[mask]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1),
        random_state=42,
        eval_metric='logloss'
    )

    model.fit(X_scaled, y, verbose=False)

    return model, scaler, available_features

# ==============================================================================
# Trade Simulation (Unchanged)
# ==============================================================================

def simulate_single_trade(df, entry_idx, exit_model, exit_scaler, exit_features, side):
    """Simulate a single trade"""
    entry_price = df['close'].iloc[entry_idx]

    max_hold_end = min(entry_idx + EMERGENCY_MAX_HOLD + 1, len(df))
    prices = df['close'].iloc[entry_idx+1:max_hold_end].values

    if side == 'LONG':
        pnl_series = (prices - entry_price) / entry_price
    else:
        pnl_series = (entry_price - prices) / entry_price

    for i, pnl_pct in enumerate(pnl_series):
        current_idx = entry_idx + 1 + i
        hold_time = i + 1
        leveraged_pnl = pnl_pct * LEVERAGE

        # Stop Loss
        if leveraged_pnl <= EMERGENCY_STOP_LOSS:
            return {
                'entry_idx': entry_idx,
                'pnl_pct': pnl_pct,
                'leveraged_pnl': leveraged_pnl,
                'hold_time': hold_time,
                'exit_reason': 'stop_loss'
            }

        # ML Exit
        try:
            exit_feat = df[exit_features].iloc[current_idx:current_idx+1].values
            if not np.isnan(exit_feat).any():
                exit_scaled = exit_scaler.transform(exit_feat)
                exit_prob = exit_model.predict_proba(exit_scaled)[0][1]

                if exit_prob >= ML_EXIT_THRESHOLD:
                    return {
                        'entry_idx': entry_idx,
                        'pnl_pct': pnl_pct,
                        'leveraged_pnl': leveraged_pnl,
                        'hold_time': hold_time,
                        'exit_reason': 'ml_exit'
                    }
        except:
            pass

        # Max Hold
        if hold_time >= EMERGENCY_MAX_HOLD:
            return {
                'entry_idx': entry_idx,
                'pnl_pct': pnl_pct,
                'leveraged_pnl': leveraged_pnl,
                'hold_time': hold_time,
                'exit_reason': 'max_hold'
            }

    final_pnl = pnl_series[-1] if len(pnl_series) > 0 else 0
    return {
        'entry_idx': entry_idx,
        'pnl_pct': final_pnl,
        'leveraged_pnl': final_pnl * LEVERAGE,
        'hold_time': len(pnl_series),
        'exit_reason': 'data_end'
    }

def label_entries_with_simulation(df, entry_candidates, exit_model, exit_scaler, exit_features, side):
    """Label entry points using trade simulation"""
    print(f"\n🎯 Labeling {side} Entries...")

    entry_labels = np.zeros(len(df))

    for entry_idx in entry_candidates:
        result = simulate_single_trade(df, entry_idx, exit_model, exit_scaler, exit_features, side)

        # 2-of-3 criteria
        score = 0
        if result['leveraged_pnl'] > 0.02:
            score += 1
        if result['hold_time'] <= 60:
            score += 1
        if result['exit_reason'] == 'ml_exit':
            score += 1

        if score >= 2:
            entry_labels[entry_idx] = 1

    positive = (entry_labels == 1).sum()
    print(f"  Good entries: {positive:,} ({positive/len(df)*100:.2f}%)")

    return entry_labels

# ==============================================================================
# 🔧 NEW: Performance-Based Validation
# ==============================================================================

def validate_fold_with_backtest(df_val, entry_model, entry_scaler, entry_features,
                                 exit_model, exit_scaler, exit_features, side):
    """
    🔧 CRITICAL FIX: Validate model by running realistic mini-backtest

    Returns actual performance metrics (win rate, return) instead of prediction rate
    """
    print(f"\n  📊 Running validation backtest on {side} model...")

    # Simulate trades with ONE position at a time (realistic)
    capital = INITIAL_CAPITAL
    trades = []
    position = None

    for i in range(100, len(df_val) - EMERGENCY_MAX_HOLD):
        # Entry logic
        if position is None:
            # Get entry probability
            try:
                X_entry = df_val[entry_features].iloc[i:i+1].values
                if np.isnan(X_entry).any():
                    continue

                X_entry_scaled = entry_scaler.transform(X_entry)
                entry_prob = entry_model.predict_proba(X_entry_scaled)[0][1]

                if entry_prob >= ENTRY_THRESHOLD:
                    # Start position
                    position = {
                        'entry_idx': i,
                        'entry_price': df_val['close'].iloc[i],
                        'side': side
                    }
            except:
                continue

        # Exit logic
        if position is not None:
            hold_time = i - position['entry_idx']

            if hold_time > 0:
                current_price = df_val['close'].iloc[i]

                # Calculate P&L
                if side == 'LONG':
                    pnl_pct = (current_price - position['entry_price']) / position['entry_price']
                else:
                    pnl_pct = (position['entry_price'] - current_price) / position['entry_price']

                leveraged_pnl = pnl_pct * LEVERAGE

                should_exit = False
                exit_reason = None

                # Stop Loss
                if leveraged_pnl <= EMERGENCY_STOP_LOSS:
                    should_exit = True
                    exit_reason = 'stop_loss'

                # ML Exit
                if not should_exit:
                    try:
                        X_exit = df_val[exit_features].iloc[i:i+1].values
                        if not np.isnan(X_exit).any():
                            X_exit_scaled = exit_scaler.transform(X_exit)
                            exit_prob = exit_model.predict_proba(X_exit_scaled)[0][1]

                            if exit_prob >= ML_EXIT_THRESHOLD:
                                should_exit = True
                                exit_reason = 'ml_exit'
                    except:
                        pass

                # Max Hold
                if not should_exit and hold_time >= EMERGENCY_MAX_HOLD:
                    should_exit = True
                    exit_reason = 'max_hold'

                # Execute exit
                if should_exit:
                    pnl_usd = capital * leveraged_pnl * 0.2  # Assume 20% position size
                    capital += pnl_usd

                    trades.append({
                        'pnl': leveraged_pnl,
                        'pnl_usd': pnl_usd,
                        'win': leveraged_pnl > 0,
                        'exit_reason': exit_reason
                    })

                    position = None

    if len(trades) == 0:
        return {'win_rate': 0, 'return_pct': 0, 'trades': 0, 'score': 0}

    # Calculate metrics
    win_rate = sum(1 for t in trades if t['win']) / len(trades)
    return_pct = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL

    # Composite score: 70% win rate + 30% return
    # Normalize return: +10% return = 0.5 score, +20% = 1.0 score
    normalized_return = min(return_pct / 0.20, 1.0)
    composite_score = (win_rate * 0.7) + (normalized_return * 0.3)

    print(f"    Trades: {len(trades)}, WR: {win_rate*100:.1f}%, Return: {return_pct*100:.1f}%, Score: {composite_score:.3f}")

    return {
        'win_rate': win_rate,
        'return_pct': return_pct,
        'trades': len(trades),
        'score': composite_score
    }

# ==============================================================================
# 🔧 FIXED: Walk-Forward Training with Performance-Based Selection
# ==============================================================================

def train_entry_model_walkforward(df, entry_features, side):
    """
    🔧 FIXED: Select model based on ACTUAL PERFORMANCE (not prediction rate)
    """
    print(f"\n{'='*80}")
    print(f"Walk-Forward Training: {side} Entry Model (FIXED)")
    print(f"{'='*80}")

    tscv = TimeSeriesSplit(n_splits=5)

    fold_models = []
    fold_performances = []

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(df)):
        print(f"\n{'─'*80}")
        print(f"FOLD {fold_idx+1}/5")
        print(f"{'─'*80}")
        print(f"  Train: {len(train_idx):,} samples")
        print(f"  Val: {len(val_idx):,} samples")

        df_train = df.iloc[train_idx].copy()
        df_val = df.iloc[val_idx].copy()

        # Step 1: Train Exit model
        print(f"\n  Step 1: Training Exit model...")
        exit_model, exit_scaler, exit_features = train_exit_model_fold(df_train, side)

        # Step 2: Filter candidates
        train_candidates = filter_entry_candidates(df_train, side)

        # Step 3: Label entries
        entry_labels = label_entries_with_simulation(
            df_train, train_candidates, exit_model, exit_scaler, exit_features, side
        )

        # Step 4: Train Entry model
        print(f"\n  Step 4: Training {side} Entry model...")

        available_features = [f for f in entry_features if f in df_train.columns]
        X_train = df_train[available_features].values
        y_train = entry_labels

        mask = ~np.isnan(X_train).any(axis=1) & ~np.isnan(y_train)
        X_train = X_train[mask]
        y_train = y_train[mask]

        if (y_train == 1).sum() < 10:
            print(f"  ⚠️  Too few positive samples, skipping...")
            continue

        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            random_state=42,
            eval_metric='logloss'
        )

        model.fit(X_train_scaled, y_train, verbose=False)

        # 🔧 Step 5: VALIDATE WITH ACTUAL PERFORMANCE (not prediction rate!)
        performance = validate_fold_with_backtest(
            df_val, model, scaler, available_features,
            exit_model, exit_scaler, exit_features, side
        )

        if performance['trades'] > 0:
            fold_models.append((model, scaler, available_features))
            fold_performances.append(performance)

        print(f"  ✅ Fold {fold_idx+1} complete")

    # 🔧 Select model with BEST PERFORMANCE (not highest prediction rate!)
    if len(fold_models) == 0:
        raise ValueError(f"No valid models trained for {side}")

    best_idx = np.argmax([p['score'] for p in fold_performances])
    best_model, best_scaler, best_features = fold_models[best_idx]
    best_perf = fold_performances[best_idx]

    print(f"\n{'='*80}")
    print(f"✅ Best Model: Fold {best_idx+1}")
    print(f"   Win Rate: {best_perf['win_rate']*100:.1f}%")
    print(f"   Return: {best_perf['return_pct']*100:.1f}%")
    print(f"   Trades: {best_perf['trades']}")
    print(f"   Composite Score: {best_perf['score']:.3f}")
    print(f"{'='*80}")

    return best_model, best_scaler, best_features

# ==============================================================================
# Main Training Pipeline
# ==============================================================================

print("-"*80)
print("STEP 3: Training LONG Entry Model (FIXED)")
print("-"*80)

long_entry_model, long_entry_scaler, long_entry_features_used = train_entry_model_walkforward(
    df, long_entry_features, 'LONG'
)

print("\n" + "-"*80)
print("STEP 4: Training SHORT Entry Model (FIXED)")
print("-"*80)

short_entry_model, short_entry_scaler, short_entry_features_used = train_entry_model_walkforward(
    df, short_entry_features, 'SHORT'
)

# Save Models
print("\n" + "="*80)
print("STEP 5: Saving Models")
print("="*80)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# LONG Entry
long_path = MODELS_DIR / f"xgboost_long_entry_walkforward_fixed_{timestamp}"
with open(f"{long_path}.pkl", 'wb') as f:
    pickle.dump(long_entry_model, f)
joblib.dump(long_entry_scaler, f"{long_path}_scaler.pkl")
with open(f"{long_path}_features.txt", 'w') as f:
    f.write('\n'.join(long_entry_features_used))

print(f"✅ LONG Entry: {long_path.name}.pkl")
print(f"   Features: {len(long_entry_features_used)}")

# SHORT Entry
short_path = MODELS_DIR / f"xgboost_short_entry_walkforward_fixed_{timestamp}"
with open(f"{short_path}.pkl", 'wb') as f:
    pickle.dump(short_entry_model, f)
joblib.dump(short_entry_scaler, f"{short_path}_scaler.pkl")
with open(f"{short_path}_features.txt", 'w') as f:
    f.write('\n'.join(short_entry_features_used))

print(f"✅ SHORT Entry: {short_path.name}.pkl")
print(f"   Features: {len(short_entry_features_used)}")

print("\n" + "="*80)
print("🔧 FIXED TRAINING COMPLETE")
print("="*80)
print()
print("Critical Fix Applied:")
print("  ❌ Old: Selected by prediction rate (many but bad entries)")
print("  ✅ New: Selected by performance (fewer but good entries)")
print()
print("Next Steps:")
print("  1. Run 108-window realistic backtest")
print("  2. Compare with 0.80 baseline (72.3% WR, +25.21% return)")
print("  3. Deploy if comparable or better")
print()
