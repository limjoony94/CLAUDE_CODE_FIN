"""
Retrain All Models with Reduced Feature Set
============================================

Retrain LONG/SHORT Entry + Exit models using reduced feature set after
redundancy removal.

Performance Comparison:
- Before: 107 features (44 LONG, 38 SHORT, 25 Exit)
- After: 90 features (37 LONG, 30 SHORT, 23 Exit)
- Reduction: -15.9% features

Expected Benefits:
- Reduced overfitting risk
- Faster training/inference
- Better generalization
- Cleaner model interpretation

Created: 2025-10-23
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

# Import reduced feature calculator
import importlib.util
spec = importlib.util.spec_from_file_location(
    "calculate_reduced_features",
    PROJECT_ROOT / "scripts" / "experiments" / "calculate_reduced_features.py"
)
calc_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calc_module)

calculate_reduced_features = calc_module.calculate_reduced_features
LONG_ENTRY_REDUCED_FEATURES = calc_module.LONG_ENTRY_REDUCED_FEATURES
SHORT_ENTRY_REDUCED_FEATURES = calc_module.SHORT_ENTRY_REDUCED_FEATURES
EXIT_REDUCED_FEATURES = calc_module.EXIT_REDUCED_FEATURES

# Simple labeling function defined below (no external dependencies)


def label_simple_trade_outcomes(df, direction='LONG', max_hold=120, tp_pct=0.03, sl_pct=0.015, leverage=4):
    """
    Simple trade outcome labeling without ML exit simulation

    Label = 1 if trade hits TP before SL or max hold
    Label = 0 otherwise
    """
    labels = []

    for i in range(len(df) - max_hold):
        entry_price = df['close'].iloc[i]
        future_prices = df['close'].iloc[i+1:i+max_hold+1]

        if direction == 'LONG':
            # LONG: profit when price rises
            tp_price = entry_price * (1 + tp_pct / leverage)
            sl_price = entry_price * (1 - sl_pct / leverage)

            # Check if TP hit
            tp_hit = (future_prices >= tp_price).any()
            # Check if SL hit
            sl_hit = (future_prices <= sl_price).any()

            if tp_hit and not sl_hit:
                # TP hit first
                tp_idx = (future_prices >= tp_price).idxmax()
                sl_idx = (future_prices <= sl_price).idxmax() if sl_hit else len(future_prices)
                label = 1 if tp_idx < sl_idx else 0
            elif sl_hit:
                label = 0  # SL hit
            else:
                # Neither hit, check final P&L
                final_price = future_prices.iloc[-1]
                pnl_pct = (final_price - entry_price) / entry_price * leverage
                label = 1 if pnl_pct >= tp_pct/2 else 0  # 50% of TP is acceptable

        else:  # SHORT
            # SHORT: profit when price falls
            tp_price = entry_price * (1 - tp_pct / leverage)
            sl_price = entry_price * (1 + sl_pct / leverage)

            # Check if TP hit
            tp_hit = (future_prices <= tp_price).any()
            # Check if SL hit
            sl_hit = (future_prices >= sl_price).any()

            if tp_hit and not sl_hit:
                # TP hit first
                tp_idx = (future_prices <= tp_price).idxmax()
                sl_idx = (future_prices >= sl_price).idxmax() if sl_hit else len(future_prices)
                label = 1 if tp_idx < sl_idx else 0
            elif sl_hit:
                label = 0  # SL hit
            else:
                # Neither hit, check final P&L
                final_price = future_prices.iloc[-1]
                pnl_pct = (entry_price - final_price) / entry_price * leverage
                label = 1 if pnl_pct >= tp_pct/2 else 0  # 50% of TP is acceptable

        labels.append(label)

    # Pad with NaN for last max_hold candles
    labels.extend([np.nan] * max_hold)

    return np.array(labels)


def prepare_data(df, feature_list, target_col='target'):
    """Prepare data for training"""
    # Remove NaN
    df_clean = df.dropna(subset=feature_list + [target_col])

    X = df_clean[feature_list].copy()
    y = df_clean[target_col].copy()

    print(f"\n{'='*80}")
    print(f"Data Preparation")
    print(f"{'='*80}")
    print(f"Total samples: {len(df_clean):,}")
    print(f"Features: {len(feature_list)}")
    print(f"Target distribution:")
    print(y.value_counts().sort_index())

    return X, y


def train_model(X, y, model_name):
    """Train XGBoost model with optimized hyperparameters"""

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n{'='*80}")
    print(f"Training: {model_name}")
    print(f"{'='*80}")
    print(f"Train samples: {len(X_train):,}")
    print(f"Test samples: {len(X_test):,}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # XGBoost parameters (from production models)
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'tree_method': 'hist',
        'n_jobs': -1
    }

    # Train (early_stopping_rounds is now a constructor param in XGBoost 2.0+)
    model = xgb.XGBClassifier(**params, n_estimators=200, early_stopping_rounds=20)

    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False
    )

    # Evaluate
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)

    print(f"\n{'='*80}")
    print(f"Results: {model_name}")
    print(f"{'='*80}")
    print(f"Train Accuracy: {train_score:.4f}")
    print(f"Test Accuracy: {test_score:.4f}")
    print(f"Best iteration: {model.best_iteration}")

    # Feature importance
    importances = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\nTop 10 Features:")
    print(importances.head(10).to_string(index=False))

    return model, scaler, test_score


def main():
    print(f"\n{'='*80}")
    print("REDUCED FEATURE MODEL RETRAINING")
    print(f"{'='*80}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nFeature Reduction:")
    print(f"  LONG Entry: 44 → 37 features (-7)")
    print(f"  SHORT Entry: 38 → 30 features (-8)")
    print(f"  Exit: 25 → 23 features (-2)")
    print(f"  Total: 107 → 90 features (-15.9%)")

    # Load data
    data_file = PROJECT_ROOT / "data" / "historical" / "BTCUSDT_5m_max.csv"
    print(f"\nLoading data from: {data_file.name}")
    df = pd.read_csv(data_file)
    print(f"Original data: {len(df):,} rows")

    # Calculate reduced features
    print(f"\n{'='*80}")
    print("Calculating Reduced Features")
    print(f"{'='*80}")
    df = calculate_reduced_features(df)
    print(f"After features: {len(df):,} rows, {len(df.columns)} columns")

    # Label trade outcomes (simple TP/SL-based)
    print(f"\n{'='*80}")
    print("Labeling Trade Outcomes (Simple TP/SL)")
    print(f"{'='*80}")

    # LONG Entry
    print("\nLabeling LONG Entry outcomes...")
    df['long_entry_target'] = label_simple_trade_outcomes(
        df, direction='LONG', max_hold=120, tp_pct=0.03, sl_pct=0.015, leverage=4
    )

    # SHORT Entry
    print("\nLabeling SHORT Entry outcomes...")
    df['short_entry_target'] = label_simple_trade_outcomes(
        df, direction='SHORT', max_hold=120, tp_pct=0.03, sl_pct=0.015, leverage=4
    )

    # Exit models use same labels
    df['long_exit_target'] = df['long_entry_target'].copy()
    df['short_exit_target'] = df['short_entry_target'].copy()

    # Print label statistics
    print(f"\nLabel Statistics:")
    print(f"  LONG Entry: {df['long_entry_target'].sum():.0f}/{df['long_entry_target'].notna().sum()} positive ({df['long_entry_target'].mean()*100:.2f}%)")
    print(f"  SHORT Entry: {df['short_entry_target'].sum():.0f}/{df['short_entry_target'].notna().sum()} positive ({df['short_entry_target'].mean()*100:.2f}%)")

    # Train all models
    results = {}
    models_dir = PROJECT_ROOT / "models"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. LONG Entry Model
    print(f"\n{'='*80}")
    print("1/4 LONG ENTRY MODEL")
    print(f"{'='*80}")
    X_long, y_long = prepare_data(df, LONG_ENTRY_REDUCED_FEATURES, 'long_entry_target')
    model_long, scaler_long, score_long = train_model(X_long, y_long, "LONG Entry")
    results['long_entry'] = score_long

    # Save
    model_path = models_dir / f"xgboost_long_entry_reduced_{timestamp}.pkl"
    scaler_path = models_dir / f"scaler_long_entry_reduced_{timestamp}.pkl"
    features_path = models_dir / f"xgboost_long_entry_reduced_{timestamp}_features.txt"

    joblib.dump(model_long, model_path)
    joblib.dump(scaler_long, scaler_path)
    with open(features_path, 'w') as f:
        f.write('\n'.join(LONG_ENTRY_REDUCED_FEATURES))

    print(f"\n✅ Saved:")
    print(f"   Model: {model_path.name}")
    print(f"   Scaler: {scaler_path.name}")
    print(f"   Features: {features_path.name}")

    # 2. SHORT Entry Model
    print(f"\n{'='*80}")
    print("2/4 SHORT ENTRY MODEL")
    print(f"{'='*80}")
    X_short, y_short = prepare_data(df, SHORT_ENTRY_REDUCED_FEATURES, 'short_entry_target')
    model_short, scaler_short, score_short = train_model(X_short, y_short, "SHORT Entry")
    results['short_entry'] = score_short

    # Save
    model_path = models_dir / f"xgboost_short_entry_reduced_{timestamp}.pkl"
    scaler_path = models_dir / f"scaler_short_entry_reduced_{timestamp}.pkl"
    features_path = models_dir / f"xgboost_short_entry_reduced_{timestamp}_features.txt"

    joblib.dump(model_short, model_path)
    joblib.dump(scaler_short, scaler_path)
    with open(features_path, 'w') as f:
        f.write('\n'.join(SHORT_ENTRY_REDUCED_FEATURES))

    print(f"\n✅ Saved:")
    print(f"   Model: {model_path.name}")
    print(f"   Scaler: {scaler_path.name}")
    print(f"   Features: {features_path.name}")

    # 3. LONG Exit Model
    print(f"\n{'='*80}")
    print("3/4 LONG EXIT MODEL")
    print(f"{'='*80}")
    X_long_exit, y_long_exit = prepare_data(df, EXIT_REDUCED_FEATURES, 'long_exit_target')
    model_long_exit, scaler_long_exit, score_long_exit = train_model(X_long_exit, y_long_exit, "LONG Exit")
    results['long_exit'] = score_long_exit

    # Save
    model_path = models_dir / f"xgboost_long_exit_reduced_{timestamp}.pkl"
    scaler_path = models_dir / f"scaler_long_exit_reduced_{timestamp}.pkl"
    features_path = models_dir / f"xgboost_long_exit_reduced_{timestamp}_features.txt"

    joblib.dump(model_long_exit, model_path)
    joblib.dump(scaler_long_exit, scaler_path)
    with open(features_path, 'w') as f:
        f.write('\n'.join(EXIT_REDUCED_FEATURES))

    print(f"\n✅ Saved:")
    print(f"   Model: {model_path.name}")
    print(f"   Scaler: {scaler_path.name}")
    print(f"   Features: {features_path.name}")

    # 4. SHORT Exit Model
    print(f"\n{'='*80}")
    print("4/4 SHORT EXIT MODEL")
    print(f"{'='*80}")
    X_short_exit, y_short_exit = prepare_data(df, EXIT_REDUCED_FEATURES, 'short_exit_target')
    model_short_exit, scaler_short_exit, score_short_exit = train_model(X_short_exit, y_short_exit, "SHORT Exit")
    results['short_exit'] = score_short_exit

    # Save
    model_path = models_dir / f"xgboost_short_exit_reduced_{timestamp}.pkl"
    scaler_path = models_dir / f"scaler_short_exit_reduced_{timestamp}.pkl"
    features_path = models_dir / f"xgboost_short_exit_reduced_{timestamp}_features.txt"

    joblib.dump(model_short_exit, model_path)
    joblib.dump(scaler_short_exit, scaler_path)
    with open(features_path, 'w') as f:
        f.write('\n'.join(EXIT_REDUCED_FEATURES))

    print(f"\n✅ Saved:")
    print(f"   Model: {model_path.name}")
    print(f"   Scaler: {scaler_path.name}")
    print(f"   Features: {features_path.name}")

    # Summary
    print(f"\n{'='*80}")
    print("TRAINING SUMMARY")
    print(f"{'='*80}")
    print(f"\nAll 4 models trained with REDUCED features")
    print(f"\nTest Accuracy:")
    print(f"  LONG Entry:  {results['long_entry']:.4f}")
    print(f"  SHORT Entry: {results['short_entry']:.4f}")
    print(f"  LONG Exit:   {results['long_exit']:.4f}")
    print(f"  SHORT Exit:  {results['short_exit']:.4f}")
    print(f"  Average:     {np.mean(list(results.values())):.4f}")

    print(f"\n{'='*80}")
    print("NEXT STEPS")
    print(f"{'='*80}")
    print("1. Run backtest with reduced feature models")
    print("2. Compare performance vs original models (107 features)")
    print("3. If performance maintained or improved:")
    print("   - Deploy to production")
    print("   - Expected benefits:")
    print("     * Reduced overfitting")
    print("     * Faster inference (15.9% fewer features)")
    print("     * Better generalization")
    print("4. If performance degraded:")
    print("   - Analyze which removed features were critical")
    print("   - Consider keeping borderline features (0.8-0.85 correlation)")

    print(f"\n{'='*80}")
    print("✅ RETRAINING COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()