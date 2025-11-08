"""
Train Exit Model: Optimal Exit Timing Prediction

목표: 포지션 청산 시점을 학습하는 별도 모델 훈련
- Input: Technical features (37) + Position features (4)
- Target: "지금 청산하는 것이 최적인가?" (0 or 1)

Labeling Logic:
1. Future Max Profit Reached: current_pnl >= future_max * 0.95 → Exit
2. Downside Risk: future_min < current - 0.5% → Exit
3. Otherwise: Hold

Expected: Exit Model이 Fixed보다 더 나은 타이밍 포착
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"

# Parameters
ENTRY_THRESHOLD = 0.7
LOOKAHEAD_STEPS = 12  # 1 hour (12 * 5min)
PROFIT_THRESHOLD_RATIO = 0.95  # 현재가 최대 수익의 95% 이상
DOWNSIDE_THRESHOLD = 0.005  # 0.5% 하락 위험

def generate_exit_training_data(df, entry_model, feature_columns, entry_threshold):
    """
    Generate training data for exit model

    For each position:
    - Track all candles while in position
    - Label: optimal exit timing (0=hold, 1=exit)
    """
    print("=" * 80)
    print("GENERATING EXIT TRAINING DATA")
    print("=" * 80)

    training_samples = []
    position = None
    positions_processed = 0

    for i in range(len(df) - LOOKAHEAD_STEPS):
        current_price = df['close'].iloc[i]

        # Entry logic (same as production)
        if position is None:
            features = df[feature_columns].iloc[i:i+1].values

            if np.isnan(features).any():
                continue

            prob_long = entry_model.predict_proba(features)[0][1]

            if prob_long >= entry_threshold:
                # Enter position
                position = {
                    'entry_idx': i,
                    'entry_price': current_price,
                    'entry_signal': prob_long,
                    'entry_features': features.copy()
                }
                positions_processed += 1

                if positions_processed % 100 == 0:
                    print(f"  Processed {positions_processed} positions...")

        # Position management and labeling
        else:
            entry_idx = position['entry_idx']
            entry_price = position['entry_price']
            entry_signal = position['entry_signal']

            # Current P&L
            current_pnl = (current_price - entry_price) / entry_price
            holding_time = (i - entry_idx) / 12  # hours

            # Look ahead: future max/min P&L
            future_prices = df['close'].iloc[i:i+LOOKAHEAD_STEPS+1].values
            future_pnls = (future_prices - entry_price) / entry_price
            future_max_pnl = future_pnls.max()
            future_min_pnl = future_pnls.min()

            # Get current features (technical + position)
            tech_features = df[feature_columns].iloc[i:i+1].values[0]

            if np.isnan(tech_features).any():
                # Skip this candle
                continue

            # Position features
            position_features = np.array([
                current_pnl,  # unrealized P&L
                holding_time,  # hours held
                entry_signal,  # entry signal strength
                (current_price - entry_price) / entry_price,  # price distance from entry
            ])

            # Combined features
            combined_features = np.concatenate([tech_features, position_features])

            # Labeling logic
            label = 0  # Default: hold
            exit_reason = None

            # Exit condition 1: Near peak (현재가 최대 수익의 95% 이상)
            if current_pnl >= future_max_pnl * PROFIT_THRESHOLD_RATIO and current_pnl > 0:
                label = 1
                exit_reason = "Peak Reached"

            # Exit condition 2: Downside risk (향후 0.5% 이상 하락)
            elif future_min_pnl < current_pnl - DOWNSIDE_THRESHOLD:
                label = 1
                exit_reason = "Downside Risk"

            # Exit condition 3: Already in loss and getting worse
            elif current_pnl < -0.005 and future_min_pnl < current_pnl:
                label = 1
                exit_reason = "Cut Loss"

            # Create training sample
            training_samples.append({
                'features': combined_features,
                'label': label,
                'exit_reason': exit_reason,
                'current_pnl': current_pnl,
                'future_max_pnl': future_max_pnl,
                'future_min_pnl': future_min_pnl,
                'holding_time': holding_time,
                'entry_signal': entry_signal,
            })

            # Exit position if:
            # 1. Labeled as exit
            # 2. Hit stop loss (-2%)
            # 3. Max holding time (8 hours)
            if label == 1 or current_pnl < -0.02 or holding_time >= 8:
                position = None

    print(f"\n✅ Generated {len(training_samples)} training samples from {positions_processed} positions")

    return training_samples

def prepare_training_data(training_samples):
    """Prepare X, y for training"""
    print("\n" + "=" * 80)
    print("PREPARING TRAINING DATA")
    print("=" * 80)

    X = np.array([sample['features'] for sample in training_samples])
    y = np.array([sample['label'] for sample in training_samples])

    print(f"Total samples: {len(X)}")
    print(f"Features per sample: {X.shape[1]}")
    print(f"  - Technical features: 37")
    print(f"  - Position features: 4")

    # Class distribution
    exit_count = np.sum(y == 1)
    hold_count = np.sum(y == 0)
    exit_ratio = exit_count / len(y) * 100

    print(f"\nClass Distribution:")
    print(f"  Hold (0): {hold_count} ({100-exit_ratio:.1f}%)")
    print(f"  Exit (1): {exit_count} ({exit_ratio:.1f}%)")

    # Exit reason breakdown
    exit_samples = [s for s in training_samples if s['label'] == 1]
    if len(exit_samples) > 0:
        reasons = pd.Series([s['exit_reason'] for s in exit_samples if s['exit_reason']])
        print(f"\nExit Reasons:")
        for reason, count in reasons.value_counts().items():
            print(f"  {reason}: {count} ({count/len(exit_samples)*100:.1f}%)")

    return X, y, training_samples

def train_exit_model(X, y):
    """Train XGBoost exit model"""
    print("\n" + "=" * 80)
    print("TRAINING EXIT MODEL")
    print("=" * 80)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # XGBoost model
    model = XGBClassifier(
        objective='binary:logistic',
        max_depth=6,
        learning_rate=0.05,
        n_estimators=300,
        min_child_weight=3,
        gamma=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )

    print("\nTraining model...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50
    )

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE")
    print("=" * 80)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Hold', 'Exit']))

    print(f"\nROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"              Predicted Hold  Predicted Exit")
    print(f"Actual Hold   {cm[0][0]:>14}  {cm[0][1]:>14}")
    print(f"Actual Exit   {cm[1][0]:>14}  {cm[1][1]:>14}")

    # Cross-validation
    print("\n5-Fold Cross-Validation:")
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
    print(f"  Mean ROC AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    return model, X_test, y_test, y_pred_proba

def analyze_feature_importance(model, feature_columns):
    """Analyze feature importance"""
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE")
    print("=" * 80)

    # Get feature importances
    importances = model.feature_importances_

    # Feature names (37 technical + 4 position)
    feature_names = feature_columns + [
        'current_pnl',
        'holding_time',
        'entry_signal',
        'price_distance'
    ]

    # Create DataFrame
    df_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    print("\nTop 20 Most Important Features:")
    for idx, row in df_importance.head(20).iterrows():
        print(f"  {row['feature']:<30} {row['importance']:.4f}")

    # Position features importance
    position_features = df_importance[df_importance['feature'].isin([
        'current_pnl', 'holding_time', 'entry_signal', 'price_distance'
    ])]

    print("\nPosition Features Importance:")
    for idx, row in position_features.iterrows():
        print(f"  {row['feature']:<30} {row['importance']:.4f}")

    return df_importance

def save_model(model, feature_columns):
    """Save trained exit model"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save model
    model_file = MODELS_DIR / f"xgboost_exit_model_{timestamp}.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n✅ Model saved: {model_file.name}")

    # Save feature list (technical + position features)
    feature_file = MODELS_DIR / f"xgboost_exit_model_{timestamp}_features.txt"
    all_features = feature_columns + [
        'current_pnl',
        'holding_time',
        'entry_signal',
        'price_distance'
    ]
    with open(feature_file, 'w') as f:
        f.write('\n'.join(all_features))
    print(f"✅ Features saved: {feature_file.name}")

    return model_file


# Main execution
print("=" * 80)
print("EXIT MODEL TRAINING PIPELINE")
print("=" * 80)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Load entry model (for generating positions)
print("\n1. Loading entry model...")
entry_model_file = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl"
feature_file = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt"

with open(entry_model_file, 'rb') as f:
    entry_model = pickle.load(f)

with open(feature_file, 'r') as f:
    feature_columns = [line.strip() for line in f.readlines()]

print(f"✅ Entry model loaded: {len(feature_columns)} features")

# Load data
print("\n2. Loading market data...")
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df = pd.read_csv(data_file)
print(f"✅ Data loaded: {len(df)} candles")

# Calculate features
print("\n3. Calculating features...")
df = calculate_features(df)
adv_features = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)
df = adv_features.calculate_all_features(df)
df = df.ffill().dropna().reset_index(drop=True)
print(f"✅ Features ready: {len(df)} rows")

# Generate exit training data
print("\n4. Generating exit training data...")
training_samples = generate_exit_training_data(
    df, entry_model, feature_columns, ENTRY_THRESHOLD
)

# Prepare data
X, y, samples = prepare_training_data(training_samples)

# Train model
model, X_test, y_test, y_pred_proba = train_exit_model(X, y)

# Feature importance
df_importance = analyze_feature_importance(model, feature_columns)

# Save model
model_file = save_model(model, feature_columns)

print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)
print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nNext Steps:")
print(f"1. Backtest exit model performance")
print(f"2. Compare vs Fixed Exit (TP 1.5%, SL 1%, MaxHold 2h)")
print(f"3. Compare vs Dynamic Exit (Signal<0.3)")
print(f"4. Deploy best strategy to production")
print("=" * 80)
