"""
XGBoost Training with Realistic Labels (P&L-based)

개선된 레이블링:
- 기존: "다음 15분 내 0.3% 상승" → Label 1
- 개선: 실제 거래 시뮬레이션 (SL/TP/Max Hold) → 최종 P&L 기반 Label

목표: 레이블 신뢰도 향상으로 성능 개선 (7.68% → 9-11%)
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Import baseline features
import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures

# Directories
DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_and_prepare_data():
    """Load data and calculate all features"""
    print("="*80)
    print("XGBoost Realistic Labels: P&L-Based Labeling")
    print("="*80)
    print("\nLoading data...")

    data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
    df = pd.read_csv(data_file)

    print(f"Raw data: {len(df)} candles")

    # Calculate baseline features (Phase 2)
    print("\nCalculating baseline features (33 features)...")
    df = calculate_features(df)

    # Calculate advanced features (27 features)
    print("Calculating advanced technical features (27 features)...")
    adv_features = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)
    df = adv_features.calculate_all_features(df)

    # Handle NaN
    rows_before = len(df)
    df = df.ffill()
    df = df.dropna()
    rows_after = len(df)

    print(f"After NaN handling: {rows_before} → {rows_after} rows")

    return df, adv_features


def simulate_trade(entry_price, future_prices, stop_loss=0.01, take_profit=0.03, max_hold=48):
    """
    실제 거래를 시뮬레이션하여 최종 P&L 반환

    Args:
        entry_price: 진입 가격
        future_prices: 향후 가격 시리즈 (최대 max_hold 길이)
        stop_loss: Stop Loss % (0.01 = 1%)
        take_profit: Take Profit % (0.03 = 3%)
        max_hold: 최대 보유 시간 (candles)

    Returns:
        final_pnl: 최종 손익률 (-0.01 ~ 0.03)
        exit_reason: 종료 사유 ('TP', 'SL', 'MAX_HOLD')
        exit_candle: 종료 시점 (candle index)
    """
    sl_price = entry_price * (1 - stop_loss)
    tp_price = entry_price * (1 + take_profit)

    for i, price in enumerate(future_prices):
        # Take Profit 도달
        if price >= tp_price:
            return take_profit, 'TP', i

        # Stop Loss 도달
        if price <= sl_price:
            return -stop_loss, 'SL', i

        # Max Hold 도달
        if i == len(future_prices) - 1 or i >= max_hold - 1:
            final_pnl = (price - entry_price) / entry_price
            return final_pnl, 'MAX_HOLD', i

    # Shouldn't reach here, but safety
    final_pnl = (future_prices.iloc[-1] - entry_price) / entry_price
    return final_pnl, 'MAX_HOLD', len(future_prices) - 1


def create_realistic_labels(df, max_hold=48, stop_loss=0.01, take_profit=0.03,
                           positive_threshold=0.0):
    """
    실제 거래 시뮬레이션 기반 레이블 생성

    Label = 1: 최종 P&L > positive_threshold
    Label = 0: 최종 P&L <= positive_threshold

    Args:
        df: 데이터프레임
        max_hold: 최대 보유 시간 (candles, default=48 = 4 hours)
        stop_loss: Stop Loss % (default=0.01 = 1%)
        take_profit: Take Profit % (default=0.03 = 3%)
        positive_threshold: Label=1의 최소 P&L (default=0.0)

    Returns:
        labels: numpy array of labels
        label_stats: 레이블 통계 정보
    """
    print(f"\nCreating realistic labels:")
    print(f"  Stop Loss: {stop_loss*100}%")
    print(f"  Take Profit: {take_profit*100}%")
    print(f"  Max Hold: {max_hold} candles ({max_hold*5} minutes)")
    print(f"  Positive Threshold: {positive_threshold*100}%")

    labels = []
    label_details = []

    for i in range(len(df)):
        # 미래 데이터 부족 시 Label=0
        if i >= len(df) - max_hold:
            labels.append(0)
            label_details.append({
                'pnl': 0.0,
                'reason': 'INSUFFICIENT_DATA',
                'exit_candle': 0
            })
            continue

        entry_price = df['close'].iloc[i]
        future_prices = df['close'].iloc[i+1:i+1+max_hold]

        # 거래 시뮬레이션
        final_pnl, exit_reason, exit_candle = simulate_trade(
            entry_price, future_prices, stop_loss, take_profit, max_hold
        )

        # 레이블 결정
        label = 1 if final_pnl > positive_threshold else 0
        labels.append(label)

        label_details.append({
            'pnl': final_pnl,
            'reason': exit_reason,
            'exit_candle': exit_candle
        })

    # 통계 계산
    labels_array = np.array(labels)
    details_df = pd.DataFrame(label_details)

    label_stats = {
        'total': len(labels),
        'positive': np.sum(labels_array),
        'positive_pct': np.mean(labels_array) * 100,
        'negative': len(labels_array) - np.sum(labels_array),
        'negative_pct': (1 - np.mean(labels_array)) * 100,
        'avg_pnl_positive': details_df[details_df['pnl'] > positive_threshold]['pnl'].mean() * 100,
        'avg_pnl_negative': details_df[details_df['pnl'] <= positive_threshold]['pnl'].mean() * 100,
        'exit_reasons': details_df['reason'].value_counts().to_dict()
    }

    print(f"\nLabel Statistics:")
    print(f"  Total samples: {label_stats['total']}")
    print(f"  Positive (Label=1): {label_stats['positive']} ({label_stats['positive_pct']:.1f}%)")
    print(f"  Negative (Label=0): {label_stats['negative']} ({label_stats['negative_pct']:.1f}%)")
    print(f"  Avg P&L (Label=1): {label_stats['avg_pnl_positive']:.2f}%")
    print(f"  Avg P&L (Label=0): {label_stats['avg_pnl_negative']:.2f}%")
    print(f"\nExit Reasons:")
    for reason, count in label_stats['exit_reasons'].items():
        pct = (count / label_stats['total']) * 100
        print(f"  {reason}: {count} ({pct:.1f}%)")

    return labels_array, label_stats


def get_feature_columns(df, adv_features):
    """Get all feature column names"""

    # Baseline features (33 from Phase 2)
    baseline_features = [
        'returns', 'log_returns', 'close_change_1', 'close_change_3',
        'volume_change', 'volume_ma_ratio', 'rsi', 'rsi_ma', 'rsi_change',
        'macd', 'macd_signal', 'macd_diff', 'bb_high', 'bb_mid', 'bb_low',
        'bb_width', 'bb_position', 'atr', 'atr_pct', 'adx', 'ema_9', 'ema_21',
        'ema_diff', 'price_vs_ema9', 'price_vs_ema21', 'stoch_k', 'stoch_d',
        'stoch_diff', 'obv', 'obv_ema', 'obv_divergence', 'vwap', 'price_vs_vwap'
    ]

    # Advanced features (27)
    advanced_features = adv_features.get_feature_names()

    # Combine
    all_features = baseline_features + advanced_features

    # Filter to only existing columns
    available_features = [f for f in all_features if f in df.columns]

    print(f"\nFeature selection:")
    print(f"  Baseline features: {len([f for f in baseline_features if f in df.columns])}")
    print(f"  Advanced features: {len([f for f in advanced_features if f in df.columns])}")
    print(f"  Total features: {len(available_features)}")

    return available_features


def train_xgboost_realistic(df, feature_columns, labels, label_stats):
    """Train XGBoost with realistic labels"""

    print("\n" + "="*80)
    print("Training XGBoost with Realistic Labels")
    print("="*80)

    # Prepare features and labels
    X = df[feature_columns].values
    y = labels

    print(f"\nDataset:")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {len(feature_columns)}")
    print(f"  Positive samples: {np.sum(y)} ({np.mean(y)*100:.1f}%)")
    print(f"  Negative samples: {len(y) - np.sum(y)} ({(1-np.mean(y))*100:.1f}%)")

    # Time series split
    tscv = TimeSeriesSplit(n_splits=5)

    best_model = None
    best_score = 0

    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Calculate class weight (ratio of negative to positive)
        scale_pos_weight = (len(y_train) - np.sum(y_train)) / np.sum(y_train)

        # Train model with class weighting
        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss'
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # Evaluate
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)

        fold_scores.append({
            'fold': fold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })

        print(f"\nFold {fold}:")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1 Score: {f1:.3f}")

        if f1 > best_score:
            best_score = f1
            best_model = model

    # Average scores
    avg_scores = pd.DataFrame(fold_scores).mean()

    print("\n" + "="*80)
    print("Cross-Validation Results (Average)")
    print("="*80)
    print(f"Accuracy: {avg_scores['accuracy']:.3f}")
    print(f"Precision: {avg_scores['precision']:.3f}")
    print(f"Recall: {avg_scores['recall']:.3f}")
    print(f"F1 Score: {avg_scores['f1']:.3f}")

    print("\nComparison to Baseline:")
    baseline_f1 = 0.089
    improvement = ((avg_scores['f1'] - baseline_f1) / baseline_f1) * 100
    print(f"  Baseline F1: {baseline_f1:.3f}")
    print(f"  Realistic F1: {avg_scores['f1']:.3f}")
    print(f"  Improvement: {improvement:+.1f}%")

    # Train final model on all data
    print("\n" + "="*80)
    print("Training Final Model on All Data")
    print("="*80)

    scale_pos_weight = (len(y) - np.sum(y)) / np.sum(y)
    print(f"Class weight (scale_pos_weight): {scale_pos_weight:.2f}")

    final_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss'
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


def save_model(model, feature_columns, label_stats, scores):
    """Save model and metadata"""

    model_name = "xgboost_v4_realistic_labels"
    model_path = MODELS_DIR / f"{model_name}.pkl"
    feature_path = MODELS_DIR / f"{model_name}_features.txt"

    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # Save features
    with open(feature_path, 'w') as f:
        for feature in feature_columns:
            f.write(f"{feature}\n")

    # Convert numpy types to Python natives for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_native(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        else:
            return obj

    # Save metadata
    metadata = {
        "model_name": model_name,
        "n_features": len(feature_columns),
        "labeling_method": "realistic_pnl_simulation",
        "stop_loss": 0.01,
        "take_profit": 0.03,
        "max_hold_candles": 48,
        "timestamp": datetime.now().isoformat(),
        "label_stats": convert_to_native(label_stats),
        "scores": {
            "accuracy": float(scores['accuracy']),
            "precision": float(scores['precision']),
            "recall": float(scores['recall']),
            "f1": float(scores['f1'])
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

    # Load and prepare data
    df, adv_features = load_and_prepare_data()

    # Get feature columns
    feature_columns = get_feature_columns(df, adv_features)

    # Create realistic labels
    labels, label_stats = create_realistic_labels(
        df,
        max_hold=48,  # 4 hours = 48 candles
        stop_loss=0.01,  # 1%
        take_profit=0.03,  # 3%
        positive_threshold=0.0  # P&L > 0%
    )

    # Train model
    model, feature_importance, scores = train_xgboost_realistic(
        df, feature_columns, labels, label_stats
    )

    # Save model
    model_path = save_model(model, feature_columns, label_stats, scores)

    print("\n" + "="*80)
    print("✅ XGBoost Realistic Labels Training Complete!")
    print("="*80)
    print(f"\nNext Steps:")
    print(f"1. Run backtest to validate performance")
    print(f"2. Compare to baseline (7.68% returns)")
    print(f"3. If improved (>8%), deploy to production")
    print(f"\n레이블 신뢰도 개선으로 실제 거래 성공률 향상 기대!")


if __name__ == "__main__":
    main()
