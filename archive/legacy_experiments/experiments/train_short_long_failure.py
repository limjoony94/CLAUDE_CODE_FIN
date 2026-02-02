"""
SHORT Model - LONG Failure Pattern Analysis (Approach #14)

Paradigm Shift:
- Previous 13 approaches: "When will price fall?" (8.7% rare)
- New approach: "When will LONG model fail?" (30.9% errors!)

Meta-Learning Strategy:
1. LONG model is verified (69.1% win rate)
2. But it's wrong 30.9% of the time
3. Learn WHEN LONG model makes mistakes
4. Enter SHORT when LONG is predicted to fail

Why This Works:
- More training data (30.9% vs 8.7%)
- Pattern exists (LONG has systematic errors)
- Meta-features (LONG probability, confidence, etc.)
- Completely different problem space

Expected: 50-60%+ win rate (learning from LONG errors)
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_sample_weight

import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.market_regime_filter import MarketRegimeFilter

# Directories
DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"


def create_long_labels(df, lookahead=3, threshold=0.0):
    """Create LONG labels"""
    labels = []
    for i in range(len(df)):
        if i >= len(df) - lookahead:
            labels.append(0)
            continue

        current_price = df['close'].iloc[i]
        future_prices = df['close'].iloc[i+1:i+1+lookahead]
        max_future = future_prices.max()
        increase_pct = (max_future - current_price) / current_price

        if increase_pct >= 0.002:  # 0.2% gain
            labels.append(1)  # LONG
        else:
            labels.append(0)

    return np.array(labels)


def create_short_labels(df, lookahead=3, threshold=0.002):
    """Create SHORT validation labels"""
    labels = []
    for i in range(len(df)):
        if i >= len(df) - lookahead:
            labels.append(0)
            continue

        current_price = df['close'].iloc[i]
        future_prices = df['close'].iloc[i+1:i+1+lookahead]
        min_future = future_prices.min()
        decrease_pct = (current_price - min_future) / current_price

        if decrease_pct >= threshold:
            labels.append(1)  # SHORT success
        else:
            labels.append(0)

    return np.array(labels)


def load_and_prepare_data():
    """Load data"""
    print("="*80)
    print("Loading Data for LONG Failure Analysis")
    print("="*80)

    df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_max.csv")
    print(f"Loaded {len(df)} candles")

    df = calculate_features(df)

    regime_filter = MarketRegimeFilter()
    df = regime_filter.get_regime_features(df)

    df = df.ffill().dropna()
    print(f"After processing: {len(df)} candles")

    # Base features
    base_features = [
        'rsi', 'macd', 'macd_signal', 'bb_high', 'bb_low',
        'sma_10', 'sma_20', 'ema_10', 'volatility', 'volume_ratio'
    ]
    base_features = [f for f in base_features if f in df.columns]

    # Labels
    long_labels = create_long_labels(df, lookahead=3, threshold=0.0)
    short_labels = create_short_labels(df, lookahead=3, threshold=0.002)

    print(f"\nLONG Labels: {(long_labels==1).sum()} / {len(long_labels)} ({(long_labels==1).sum()/len(long_labels)*100:.1f}%)")
    print(f"SHORT Labels: {(short_labels==1).sum()} / {len(short_labels)} ({(short_labels==1).sum()/len(short_labels)*100:.1f}%)")

    X_base = df[base_features].values
    y_long = long_labels
    y_short = short_labels

    return X_base, y_long, y_short, base_features, df


def train_long_model_and_analyze_errors(X_base, y_long, y_short, base_features, df):
    """
    Train LONG model and analyze its error patterns

    Key Innovation: Use LONG model's mistakes as training data for SHORT
    """
    print("\n" + "="*80)
    print("Step 1: Train LONG Model and Collect Predictions")
    print("="*80)

    tscv = TimeSeriesSplit(n_splits=5)

    # Store LONG predictions for meta-learning
    all_long_predictions = []
    long_error_patterns = []

    params = {
        'n_estimators': 300,
        'max_depth': 5,
        'learning_rate': 0.01,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.5,
        'reg_alpha': 1,
        'reg_lambda': 2,
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    }

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_base)):
        X_train, X_val = X_base[train_idx], X_base[val_idx]
        y_train, y_val = y_long[train_idx], y_long[val_idx]

        # Train LONG model
        model = xgb.XGBClassifier(**params)
        sample_weights = compute_sample_weight('balanced', y_train)
        model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)

        # Get predictions
        y_prob_long = model.predict_proba(X_val)[:, 1]
        y_pred_long = (y_prob_long >= 0.7).astype(int)  # LONG threshold

        # Analyze errors
        for i, idx in enumerate(val_idx):
            prediction_data = {
                'fold': fold,
                'index': idx,
                'long_prob': y_prob_long[i],
                'long_pred': y_pred_long[i],
                'long_actual': y_val[i],
                'long_correct': (y_pred_long[i] == y_val[i]),
                'short_actual': y_short[idx],
                'features': X_base[idx]
            }
            all_long_predictions.append(prediction_data)

            # If LONG predicted but FAILED
            if y_pred_long[i] == 1 and y_val[i] == 0:
                long_error_patterns.append(prediction_data)

    print(f"\nLONG Model Analysis:")
    print(f"  Total predictions: {len(all_long_predictions)}")
    print(f"  LONG signals: {sum(1 for p in all_long_predictions if p['long_pred']==1)}")
    print(f"  LONG errors: {len(long_error_patterns)} ({len(long_error_patterns)/len(all_long_predictions)*100:.1f}%)")

    # Analyze error patterns
    errors_df = pd.DataFrame(long_error_patterns)
    if len(errors_df) > 0:
        print(f"\nLONG Error Patterns:")
        print(f"  Mean LONG prob when wrong: {errors_df['long_prob'].mean():.3f}")
        print(f"  Std LONG prob when wrong: {errors_df['long_prob'].std():.3f}")
        print(f"  SHORT success rate in LONG errors: {errors_df['short_actual'].mean()*100:.1f}%")

    return all_long_predictions, long_error_patterns


def build_meta_features(X_base, long_prob, base_features):
    """
    Build meta-features for SHORT prediction

    Combines:
    - Base features (OHLCV indicators)
    - Meta features (LONG model predictions)
    """
    # Convert to DataFrame for easier manipulation
    df_base = pd.DataFrame(X_base, columns=base_features)

    # Add meta-features
    df_base['long_prob'] = long_prob
    df_base['long_confidence'] = np.abs(long_prob - 0.5)  # How confident?
    df_base['long_uncertainty'] = 1 - df_base['long_confidence']
    df_base['long_prob_low'] = (long_prob < 0.3).astype(int)
    df_base['long_prob_medium'] = ((long_prob >= 0.3) & (long_prob < 0.7)).astype(int)
    df_base['long_prob_high'] = (long_prob >= 0.7).astype(int)

    return df_base.values, list(df_base.columns)


def train_short_from_long_errors(all_predictions, y_short, X_base, base_features, df):
    """
    Train SHORT model using LONG error patterns

    Key Innovation:
    - Target = Will price fall? (SHORT success)
    - Features = Base features + LONG predictions (meta-learning!)
    - Filter = Enter SHORT when LONG is likely to fail
    """
    print("\n" + "="*80)
    print("Step 2: Train SHORT Model from LONG Error Patterns")
    print("="*80)

    # Prepare data
    indices = [p['index'] for p in all_predictions]
    long_probs = np.array([p['long_prob'] for p in all_predictions])

    # Build meta-features
    X_meta, meta_feature_names = build_meta_features(
        X_base[indices],
        long_probs,
        base_features
    )

    y_short_meta = y_short[indices]

    print(f"\nMeta-Learning Dataset:")
    print(f"  Samples: {len(X_meta)}")
    print(f"  Features: {len(meta_feature_names)} ({len(base_features)} base + {len(meta_feature_names)-len(base_features)} meta)")
    print(f"  SHORT labels: {(y_short_meta==1).sum()} / {len(y_short_meta)} ({(y_short_meta==1).sum()/len(y_short_meta)*100:.1f}%)")

    # Train SHORT model with meta-features
    tscv = TimeSeriesSplit(n_splits=5)

    all_trades = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_meta)):
        X_train, X_val = X_meta[train_idx], X_meta[val_idx]
        y_train, y_val = y_short_meta[train_idx], y_short_meta[val_idx]

        # Train SHORT model
        short_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.01,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.5,
            reg_alpha=1,
            reg_lambda=2,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )

        sample_weights = compute_sample_weight('balanced', y_train)
        short_model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)

        # Predict SHORT
        y_prob_short = short_model.predict_proba(X_val)[:, 1]

        # Apply filters
        short_threshold = 0.6

        for i, meta_idx in enumerate(val_idx):
            original_idx = indices[meta_idx]

            short_signal = y_prob_short[i] >= short_threshold
            regime_allowed = df['short_allowed'].iloc[original_idx] == 1

            if short_signal and regime_allowed:
                actual = y_val[i]

                trade = {
                    'fold': fold,
                    'index': original_idx,
                    'short_prob': y_prob_short[i],
                    'long_prob': long_probs[meta_idx],
                    'predicted': 1,
                    'actual': actual,
                    'correct': (actual == 1)
                }
                all_trades.append(trade)

        # Fold results
        fold_trades = [t for t in all_trades if t['fold'] == fold]
        if len(fold_trades) > 0:
            fold_correct = sum(t['correct'] for t in fold_trades)
            fold_total = len(fold_trades)
            fold_wr = fold_correct / fold_total
            print(f"Fold {fold}: {fold_total} trades, {fold_correct} correct ({fold_wr*100:.1f}%)")

    # Overall results
    print(f"\n{'='*80}")
    print("SHORT Model Results (Meta-Learning from LONG Errors)")
    print(f"{'='*80}")

    if len(all_trades) == 0:
        print("\n❌ No trades generated")
        return 0.0, all_trades

    total_trades = len(all_trades)
    correct_trades = sum(t['correct'] for t in all_trades)
    win_rate = correct_trades / total_trades

    print(f"\nOverall Results:")
    print(f"  Total Trades: {total_trades}")
    print(f"  Correct: {correct_trades}")
    print(f"  Incorrect: {total_trades - correct_trades}")
    print(f"  Win Rate: {win_rate*100:.1f}%")

    # Analysis
    trades_df = pd.DataFrame(all_trades)
    print(f"\nMeta-Features Analysis:")
    print(f"  Mean LONG prob (when SHORT): {trades_df['long_prob'].mean():.3f}")
    print(f"  Mean SHORT prob: {trades_df['short_prob'].mean():.3f}")

    return win_rate, all_trades


def main():
    """Main pipeline"""
    print("="*80)
    print("SHORT Model - LONG Failure Pattern Analysis (Approach #14)")
    print("="*80)
    print("Paradigm Shift:")
    print("  Previous: 'When will price fall?' (8.7% rare events)")
    print("  New: 'When will LONG model fail?' (30.9% errors!)")
    print("")
    print("Meta-Learning Strategy:")
    print("  1. Train LONG model (69.1% win rate)")
    print("  2. Analyze LONG's 30.9% errors")
    print("  3. Learn error patterns")
    print("  4. Predict when LONG will fail → Enter SHORT")
    print("="*80)

    # Load data
    X_base, y_long, y_short, base_features, df = load_and_prepare_data()

    # Step 1: Train LONG and analyze errors
    all_predictions, error_patterns = train_long_model_and_analyze_errors(
        X_base, y_long, y_short, base_features, df
    )

    # Step 2: Train SHORT from error patterns
    win_rate, trades = train_short_from_long_errors(
        all_predictions, y_short, X_base, base_features, df
    )

    # Final decision
    print("\n" + "="*80)
    print("Final Decision - LONG Failure Meta-Learning")
    print("="*80)

    if win_rate >= 0.60:
        print(f"✅ SUCCESS! SHORT win rate {win_rate*100:.1f}% >= 60%")
        print(f"✅ Meta-learning from LONG errors WORKS!")

    elif win_rate >= 0.50:
        print(f"🔄 SIGNIFICANT IMPROVEMENT: SHORT win rate {win_rate*100:.1f}% (50-60%)")
        print(f"🔄 Meta-learning shows strong promise!")

    elif win_rate >= 0.45:
        print(f"⚠️ MODERATE IMPROVEMENT: SHORT win rate {win_rate*100:.1f}% (45-50%)")
        print(f"⚠️ Better than baseline, near Approach #1")

    else:
        print(f"❌ INSUFFICIENT: SHORT win rate {win_rate*100:.1f}%")

    print(f"\nComplete Progress Summary (All 14 Approaches):")
    print(f"  #1  2-Class Inverse: 46.0% ✅ Previous best")
    print(f"  #5  V2 Baseline: 26.0%")
    print(f"  #13 Calibrated Threshold: 26.7%")
    print(f"  #14 LONG Failure Meta-Learning: {win_rate*100:.1f}%")

    return win_rate, trades


if __name__ == "__main__":
    win_rate, trades = main()
