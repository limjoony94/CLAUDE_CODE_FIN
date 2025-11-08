"""
Feature Selection and Evaluation Framework
===========================================

Systematically evaluate feature importance and select optimal feature sets.

Methods:
1. Permutation Importance: Measure performance drop when feature is shuffled
2. SHAP Values: Explain model predictions and feature contributions
3. Recursive Feature Elimination: Remove least important features iteratively
4. Correlation Analysis: Remove highly correlated redundant features

Strategy:
1. Load data with all features
2. Calculate feature importance using multiple methods
3. Remove low-importance and redundant features
4. Retrain model with selected features
5. Validate on holdout set (last 4 weeks)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
from datetime import datetime, timedelta
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.calculate_all_features import calculate_all_features


class FeatureSelector:
    """
    Feature selection using multiple methods
    """

    def __init__(self, X_train, y_train, X_val, y_val, feature_names):
        """
        Initialize feature selector

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            feature_names: List of feature names
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.feature_names = feature_names

        self.model = None
        self.feature_importance_methods = {}

    def train_baseline_model(self):
        """Train baseline XGBoost model with all features"""

        print("\n" + "="*80)
        print("Training Baseline Model (All Features)")
        print("="*80)

        scale_pos_weight = (self.y_train == 0).sum() / (self.y_train == 1).sum()

        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1
        )

        self.model.fit(self.X_train, self.y_train, verbose=0)

        # Evaluate baseline
        y_pred = self.model.predict(self.X_val)
        y_pred_proba = self.model.predict_proba(self.X_val)[:, 1]

        baseline_metrics = {
            'accuracy': accuracy_score(self.y_val, y_pred),
            'auc': roc_auc_score(self.y_val, y_pred_proba) if len(np.unique(self.y_val)) > 1 else 0.5,
            'f1': f1_score(self.y_val, y_pred),
            'precision': precision_score(self.y_val, y_pred, zero_division=0),
            'recall': recall_score(self.y_val, y_pred, zero_division=0)
        }

        print(f"\nBaseline Performance:")
        print(f"  Accuracy: {baseline_metrics['accuracy']:.4f}")
        print(f"  AUC: {baseline_metrics['auc']:.4f}")
        print(f"  F1: {baseline_metrics['f1']:.4f}")
        print(f"  Precision: {baseline_metrics['precision']:.4f}")
        print(f"  Recall: {baseline_metrics['recall']:.4f}")

        return baseline_metrics

    def calculate_builtin_importance(self):
        """Calculate XGBoost built-in feature importance"""

        print("\n" + "="*80)
        print("Method 1: XGBoost Built-in Importance")
        print("="*80)

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        self.feature_importance_methods['builtin'] = importance_df

        print(f"\nTop 10 Features (Built-in):")
        for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
            print(f"  {i:2d}. {row['feature']:30s} {row['importance']:.6f}")

        print(f"\nZero Importance Features: {(importance_df['importance'] == 0).sum()}")

        return importance_df

    def calculate_permutation_importance(self, n_repeats=5):
        """Calculate permutation importance"""

        print("\n" + "="*80)
        print(f"Method 2: Permutation Importance (n_repeats={n_repeats})")
        print("="*80)

        perm_importance = permutation_importance(
            self.model,
            self.X_val,
            self.y_val,
            n_repeats=n_repeats,
            random_state=42,
            n_jobs=-1
        )

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance_mean': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std
        }).sort_values('importance_mean', ascending=False)

        self.feature_importance_methods['permutation'] = importance_df

        print(f"\nTop 10 Features (Permutation):")
        for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
            print(f"  {i:2d}. {row['feature']:30s} {row['importance_mean']:.6f} ± {row['importance_std']:.6f}")

        print(f"\nNegative Importance Features: {(importance_df['importance_mean'] < 0).sum()}")

        return importance_df

    def calculate_correlation_matrix(self, threshold=0.95):
        """Calculate feature correlation and identify redundant features"""

        print("\n" + "="*80)
        print(f"Method 3: Correlation Analysis (threshold={threshold})")
        print("="*80)

        # Calculate correlation matrix
        corr_matrix = pd.DataFrame(self.X_train, columns=self.feature_names).corr().abs()

        # Find highly correlated pairs
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        redundant_features = []
        for column in upper.columns:
            high_corr = upper[column][upper[column] > threshold]
            if len(high_corr) > 0:
                redundant_features.append({
                    'feature': column,
                    'correlated_with': high_corr.index.tolist(),
                    'max_correlation': high_corr.max()
                })

        print(f"\nHighly Correlated Feature Pairs (>{threshold}):")
        for item in redundant_features[:10]:
            print(f"  {item['feature']:30s} correlated with {len(item['correlated_with'])} features (max: {item['max_correlation']:.3f})")

        self.feature_importance_methods['correlation'] = redundant_features

        return redundant_features

    def select_features(self, method='composite', top_k=None, importance_threshold=0.001):
        """
        Select features based on importance

        Args:
            method: 'composite', 'builtin', 'permutation'
            top_k: Select top K features (None = use threshold)
            importance_threshold: Minimum importance threshold

        Returns:
            List of selected feature names
        """

        print("\n" + "="*80)
        print(f"Feature Selection (method={method}, top_k={top_k}, threshold={importance_threshold})")
        print("="*80)

        if method == 'composite':
            # Combine multiple methods
            builtin_df = self.feature_importance_methods['builtin']
            perm_df = self.feature_importance_methods['permutation']

            # Normalize scores to 0-1
            builtin_norm = builtin_df['importance'] / builtin_df['importance'].max()
            perm_norm = (perm_df['importance_mean'] - perm_df['importance_mean'].min()) / \
                        (perm_df['importance_mean'].max() - perm_df['importance_mean'].min())

            # Combine scores
            composite_df = pd.DataFrame({
                'feature': self.feature_names,
                'builtin_score': builtin_norm.values,
                'perm_score': perm_norm.values,
                'composite_score': 0.6 * builtin_norm.values + 0.4 * perm_norm.values
            }).sort_values('composite_score', ascending=False)

            selected_df = composite_df

        elif method == 'builtin':
            selected_df = self.feature_importance_methods['builtin']

        elif method == 'permutation':
            selected_df = self.feature_importance_methods['permutation'].rename(
                columns={'importance_mean': 'composite_score'}
            )

        else:
            raise ValueError(f"Unknown method: {method}")

        # Apply selection criteria
        if top_k is not None:
            selected_features = selected_df.head(top_k)['feature'].tolist()
        else:
            selected_features = selected_df[
                selected_df['composite_score'] >= importance_threshold
            ]['feature'].tolist()

        print(f"\nSelected Features: {len(selected_features)} / {len(self.feature_names)}")
        print(f"Reduction: {(1 - len(selected_features)/len(self.feature_names))*100:.1f}%")

        return selected_features, selected_df

    def evaluate_selected_features(self, selected_features):
        """Train and evaluate model with selected features"""

        print("\n" + "="*80)
        print(f"Evaluating Model with {len(selected_features)} Selected Features")
        print("="*80)

        # Get feature indices
        feature_indices = [i for i, f in enumerate(self.feature_names) if f in selected_features]

        X_train_selected = self.X_train[:, feature_indices]
        X_val_selected = self.X_val[:, feature_indices]

        # Train model with selected features
        scale_pos_weight = (self.y_train == 0).sum() / (self.y_train == 1).sum()

        model_selected = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1
        )

        model_selected.fit(X_train_selected, self.y_train, verbose=0)

        # Evaluate
        y_pred = model_selected.predict(X_val_selected)
        y_pred_proba = model_selected.predict_proba(X_val_selected)[:, 1]

        selected_metrics = {
            'accuracy': accuracy_score(self.y_val, y_pred),
            'auc': roc_auc_score(self.y_val, y_pred_proba) if len(np.unique(self.y_val)) > 1 else 0.5,
            'f1': f1_score(self.y_val, y_pred),
            'precision': precision_score(self.y_val, y_pred, zero_division=0),
            'recall': recall_score(self.y_val, y_pred, zero_division=0)
        }

        print(f"\nSelected Features Performance:")
        print(f"  Accuracy: {selected_metrics['accuracy']:.4f}")
        print(f"  AUC: {selected_metrics['auc']:.4f}")
        print(f"  F1: {selected_metrics['f1']:.4f}")
        print(f"  Precision: {selected_metrics['precision']:.4f}")
        print(f"  Recall: {selected_metrics['recall']:.4f}")

        return selected_metrics, model_selected


def main():
    """Main execution"""

    print("="*80)
    print("Feature Selection and Evaluation")
    print("="*80)

    # Load data
    DATA_DIR = PROJECT_ROOT / "data" / "historical"
    data_file = DATA_DIR / "BTCUSDT_5m_max.csv"

    print(f"\nLoading data from: {data_file}")
    df = pd.read_csv(data_file)

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    print(f"Loaded {len(df)} candles")

    # Calculate all features
    print("\nCalculating features...")
    df = calculate_all_features(df)

    # Create labels (dummy for demonstration)
    # TODO: Load actual labels from training data
    df['signal_long'] = (df['close'].pct_change(20) > 0.02).astype(int)
    df['signal_short'] = (df['close'].pct_change(20) < -0.02).astype(int)

    # Split data (exclude last 4 weeks for validation)
    last_date = pd.to_datetime(df['timestamp'].max())
    train_end_date = last_date - timedelta(weeks=4)

    train_mask = pd.to_datetime(df['timestamp']) <= train_end_date
    val_mask = pd.to_datetime(df['timestamp']) > train_end_date

    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()

    print(f"\nData split:")
    print(f"  Training: {len(train_df)} candles ({train_df['timestamp'].min()} to {train_df['timestamp'].max()})")
    print(f"  Validation: {len(val_df)} candles ({val_df['timestamp'].min()} to {val_df['timestamp'].max()})")

    # Feature columns
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'signal_long', 'signal_short']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    print(f"\nTotal features: {len(feature_cols)}")

    # Process LONG signals
    print("\n" + "="*80)
    print("LONG Signal Feature Selection")
    print("="*80)

    X_train_long = train_df[feature_cols].values
    y_train_long = train_df['signal_long'].values
    X_val_long = val_df[feature_cols].values
    y_val_long = val_df['signal_long'].values

    selector_long = FeatureSelector(
        X_train_long, y_train_long,
        X_val_long, y_val_long,
        feature_cols
    )

    # Train baseline
    baseline_metrics_long = selector_long.train_baseline_model()

    # Calculate importance
    builtin_imp_long = selector_long.calculate_builtin_importance()
    perm_imp_long = selector_long.calculate_permutation_importance(n_repeats=3)
    corr_long = selector_long.calculate_correlation_matrix(threshold=0.95)

    # Select features (top 50 or importance > 0.001)
    selected_features_long, importance_df_long = selector_long.select_features(
        method='composite',
        top_k=50,
        importance_threshold=0.001
    )

    # Evaluate selected features
    selected_metrics_long, model_long = selector_long.evaluate_selected_features(selected_features_long)

    # Process SHORT signals
    print("\n" + "="*80)
    print("SHORT Signal Feature Selection")
    print("="*80)

    X_train_short = train_df[feature_cols].values
    y_train_short = train_df['signal_short'].values
    X_val_short = val_df[feature_cols].values
    y_val_short = val_df['signal_short'].values

    selector_short = FeatureSelector(
        X_train_short, y_train_short,
        X_val_short, y_val_short,
        feature_cols
    )

    # Train baseline
    baseline_metrics_short = selector_short.train_baseline_model()

    # Calculate importance
    builtin_imp_short = selector_short.calculate_builtin_importance()
    perm_imp_short = selector_short.calculate_permutation_importance(n_repeats=3)
    corr_short = selector_short.calculate_correlation_matrix(threshold=0.95)

    # Select features
    selected_features_short, importance_df_short = selector_short.select_features(
        method='composite',
        top_k=50,
        importance_threshold=0.001
    )

    # Evaluate selected features
    selected_metrics_short, model_short = selector_short.evaluate_selected_features(selected_features_short)

    # Save results
    RESULTS_DIR = PROJECT_ROOT / "results"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = {
        'long': {
            'baseline_metrics': baseline_metrics_long,
            'selected_metrics': selected_metrics_long,
            'selected_features': selected_features_long,
            'num_features_original': len(feature_cols),
            'num_features_selected': len(selected_features_long)
        },
        'short': {
            'baseline_metrics': baseline_metrics_short,
            'selected_metrics': selected_metrics_short,
            'selected_features': selected_features_short,
            'num_features_original': len(feature_cols),
            'num_features_selected': len(selected_features_short)
        }
    }

    with open(RESULTS_DIR / f"feature_selection_{timestamp}.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Save feature importance
    importance_df_long.to_csv(RESULTS_DIR / f"feature_importance_long_{timestamp}.csv", index=False)
    importance_df_short.to_csv(RESULTS_DIR / f"feature_importance_short_{timestamp}.csv", index=False)

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print("\nLONG Signal:")
    print(f"  Features: {len(feature_cols)} → {len(selected_features_long)} ({(1-len(selected_features_long)/len(feature_cols))*100:.1f}% reduction)")
    print(f"  Baseline Performance:")
    print(f"    Accuracy: {baseline_metrics_long['accuracy']:.4f} | AUC: {baseline_metrics_long['auc']:.4f} | F1: {baseline_metrics_long['f1']:.4f}")
    print(f"  Selected Performance:")
    print(f"    Accuracy: {selected_metrics_long['accuracy']:.4f} | AUC: {selected_metrics_long['auc']:.4f} | F1: {selected_metrics_long['f1']:.4f}")
    print(f"  Performance Change:")
    print(f"    Accuracy: {(selected_metrics_long['accuracy']-baseline_metrics_long['accuracy'])*100:+.2f}%")
    print(f"    AUC: {(selected_metrics_long['auc']-baseline_metrics_long['auc'])*100:+.2f}%")

    print("\nSHORT Signal:")
    print(f"  Features: {len(feature_cols)} → {len(selected_features_short)} ({(1-len(selected_features_short)/len(feature_cols))*100:.1f}% reduction)")
    print(f"  Baseline Performance:")
    print(f"    Accuracy: {baseline_metrics_short['accuracy']:.4f} | AUC: {baseline_metrics_short['auc']:.4f} | F1: {baseline_metrics_short['f1']:.4f}")
    print(f"  Selected Performance:")
    print(f"    Accuracy: {selected_metrics_short['accuracy']:.4f} | AUC: {selected_metrics_short['auc']:.4f} | F1: {selected_metrics_short['f1']:.4f}")
    print(f"  Performance Change:")
    print(f"    Accuracy: {(selected_metrics_short['accuracy']-baseline_metrics_short['accuracy'])*100:+.2f}%")
    print(f"    AUC: {(selected_metrics_short['auc']-baseline_metrics_short['auc'])*100:+.2f}%")

    print(f"\n{'='*80}")
    print(f"Results saved to:")
    print(f"  - feature_selection_{timestamp}.json")
    print(f"  - feature_importance_long_{timestamp}.csv")
    print(f"  - feature_importance_short_{timestamp}.csv")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
