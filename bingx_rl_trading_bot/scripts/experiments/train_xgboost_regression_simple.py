"""
XGBoost Regression: 수익률 직접 예측

개선 포인트:
- Classification → Regression
- Binary label → Continuous target (P&L%)
- 더 풍부한 정보 활용
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import baseline features
import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures
from scripts.experiments.train_xgboost_realistic_labels import simulate_trade, get_feature_columns

# Directories
DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_and_prepare_data():
    """Load data and calculate all features"""
    print("="*80)
    print("XGBoost Regression: Direct P&L Prediction")
    print("="*80)
    print("\nLoading data...")

    data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
    df = pd.read_csv(data_file)

    print(f"Raw data: {len(df)} candles")

    # Calculate baseline features
    print("\nCalculating baseline features...")
    df = calculate_features(df)

    # Calculate advanced features
    print("Calculating advanced technical features...")
    adv_features = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)
    df = adv_features.calculate_all_features(df)

    # Handle NaN
    rows_before = len(df)
    df = df.ffill()
    df = df.dropna()
    rows_after = len(df)

    print(f"After NaN handling: {rows_before} → {rows_after} rows")

    return df, adv_features


def create_regression_targets(df, max_hold=48, stop_loss=0.01, take_profit=0.03):
    """
    회귀 목표값 생성: 최종 P&L 직접 예측

    Returns:
        targets: 최종 P&L % (-0.01 ~ 0.03)
        target_stats: 통계 정보
    """
    print(f"\nCreating regression targets:")
    print(f"  Stop Loss: {stop_loss*100}%")
    print(f"  Take Profit: {take_profit*100}%")
    print(f"  Max Hold: {max_hold} candles")

    targets = []
    target_details = []

    for i in range(len(df)):
        if i >= len(df) - max_hold:
            targets.append(0.0)
            target_details.append({'reason': 'INSUFFICIENT_DATA'})
            continue

        entry_price = df['close'].iloc[i]
        future_prices = df['close'].iloc[i+1:i+1+max_hold]

        # 거래 시뮬레이션
        final_pnl, exit_reason, exit_candle = simulate_trade(
            entry_price, future_prices, stop_loss, take_profit, max_hold
        )

        targets.append(final_pnl)
        target_details.append({'pnl': final_pnl, 'reason': exit_reason})

    targets_array = np.array(targets)
    details_df = pd.DataFrame(target_details)

    # 통계
    target_stats = {
        'mean': float(targets_array.mean()),
        'std': float(targets_array.std()),
        'min': float(targets_array.min()),
        'max': float(targets_array.max()),
        'positive_pct': float((targets_array > 0).mean() * 100),
        'negative_pct': float((targets_array < 0).mean() * 100),
        'zero_pct': float((targets_array == 0).mean() * 100)
    }

    print(f"\nTarget Statistics:")
    print(f"  Mean P&L: {target_stats['mean']*100:.3f}%")
    print(f"  Std Dev: {target_stats['std']*100:.3f}%")
    print(f"  Range: [{target_stats['min']*100:.2f}%, {target_stats['max']*100:.2f}%]")
    print(f"  Positive: {target_stats['positive_pct']:.1f}%")
    print(f"  Negative: {target_stats['negative_pct']:.1f}%")
    print(f"  Zero: {target_stats['zero_pct']:.1f}%")

    return targets_array, target_stats


def train_xgboost_regression(df, feature_columns, targets, target_stats):
    """Train XGBoost Regressor"""

    print("\n" + "="*80)
    print("Training XGBoost Regression")
    print("="*80)

    X = df[feature_columns].values
    y = targets

    print(f"\nDataset:")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {len(feature_columns)}")
    print(f"  Target mean: {y.mean()*100:.3f}%")
    print(f"  Target std: {y.std()*100:.3f}%")

    # Time series split
    tscv = TimeSeriesSplit(n_splits=5)

    best_model = None
    best_score = float('inf')

    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Train Regressor
        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='rmse'
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # Evaluate
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        fold_scores.append({
            'fold': fold,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        })

        print(f"\nFold {fold}:")
        print(f"  RMSE: {rmse*100:.3f}%")
        print(f"  MAE: {mae*100:.3f}%")
        print(f"  R²: {r2:.4f}")

        if rmse < best_score:
            best_score = rmse
            best_model = model

    # Average scores
    avg_scores = pd.DataFrame(fold_scores).mean()

    print("\n" + "="*80)
    print("Cross-Validation Results (Average)")
    print("="*80)
    print(f"RMSE: {avg_scores['rmse']*100:.3f}%")
    print(f"MAE: {avg_scores['mae']*100:.3f}%")
    print(f"R²: {avg_scores['r2']:.4f}")

    # Train final model
    print("\n" + "="*80)
    print("Training Final Model on All Data")
    print("="*80)

    final_model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='rmse'
    )

    final_model.fit(X, y, verbose=False)

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 20 Most Important Features:")
    print(feature_importance.head(20).to_string(index=False))

    return final_model, feature_importance, avg_scores


def save_model(model, feature_columns, target_stats, scores):
    """Save model and metadata"""

    model_name = "xgboost_v4_regression"
    model_path = MODELS_DIR / f"{model_name}.pkl"
    feature_path = MODELS_DIR / f"{model_name}_features.txt"

    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # Save features
    with open(feature_path, 'w') as f:
        for feature in feature_columns:
            f.write(f"{feature}\n")

    # Metadata
    metadata = {
        "model_name": model_name,
        "model_type": "regression",
        "n_features": len(feature_columns),
        "target_method": "simulated_trade_pnl",
        "stop_loss": 0.01,
        "take_profit": 0.03,
        "max_hold_candles": 48,
        "timestamp": datetime.now().isoformat(),
        "target_stats": target_stats,
        "scores": {
            "rmse": float(scores['rmse']),
            "mae": float(scores['mae']),
            "r2": float(scores['r2'])
        }
    }

    import json
    metadata_path = MODELS_DIR / f"{model_name}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "="*80)
    print("Model Saved")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Features: {feature_path}")
    print(f"Metadata: {metadata_path}")

    return model_path


def main():
    """Main training pipeline"""

    # Load data
    df, adv_features = load_and_prepare_data()

    # Get features
    feature_columns = get_feature_columns(df, adv_features)

    # Create regression targets
    targets, target_stats = create_regression_targets(
        df,
        max_hold=48,
        stop_loss=0.01,
        take_profit=0.03
    )

    # Train model
    model, feature_importance, scores = train_xgboost_regression(
        df, feature_columns, targets, target_stats
    )

    # Save model
    model_path = save_model(model, feature_columns, target_stats, scores)

    print("\n" + "="*80)
    print("✅ XGBoost Regression Training Complete!")
    print("="*80)
    print(f"\nNext Steps:")
    print(f"1. Run backtest to validate performance")
    print(f"2. Compare to baseline (7.68%) and Realistic Labels")
    print(f"3. Regression provides continuous predictions!")


if __name__ == "__main__":
    main()
