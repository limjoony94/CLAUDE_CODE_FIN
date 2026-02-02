"""
Complete Optimization and Retraining Pipeline
==============================================

End-to-end pipeline for:
1. Indicator period optimization
2. Feature selection
3. Model retraining with optimal configuration
4. Backtest validation on last 4 weeks

Usage:
    python optimize_and_retrain_pipeline.py --signal-type LONG
    python optimize_and_retrain_pipeline.py --signal-type SHORT
    python optimize_and_retrain_pipeline.py --signal-type BOTH
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
import argparse
from datetime import datetime, timedelta
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
import joblib
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.calculate_all_features import calculate_all_features
from optimize_indicator_periods import grid_search_periods, calculate_features_with_periods
from feature_selection_evaluation import FeatureSelector


class OptimizationPipeline:
    """
    Complete optimization pipeline
    """

    def __init__(self, signal_type='LONG', holdout_weeks=4):
        """
        Initialize pipeline

        Args:
            signal_type: 'LONG', 'SHORT', or 'BOTH'
            holdout_weeks: Number of weeks to hold out for backtest
        """
        self.signal_type = signal_type
        self.holdout_weeks = holdout_weeks

        self.df = None
        self.optimal_periods = None
        self.selected_features = None
        self.model = None

        self.train_df = None
        self.val_df = None
        self.test_df = None

    def load_data(self):
        """Load and prepare data"""

        print("\n" + "="*80)
        print("Step 1: Loading Data")
        print("="*80)

        DATA_DIR = PROJECT_ROOT / "data" / "historical"
        data_file = DATA_DIR / "BTCUSDT_5m_max.csv"

        print(f"\nLoading data from: {data_file}")
        self.df = pd.read_csv(data_file)

        if 'timestamp' in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
            self.df.set_index('timestamp', inplace=True)

        print(f"Loaded {len(self.df)} candles")
        print(f"Period: {self.df.index[0]} to {self.df.index[-1]}")

        # Load actual trade outcome labels
        LABELS_DIR = PROJECT_ROOT / "data" / "labels"
        label_files = sorted(LABELS_DIR.glob("trade_outcome_labels_*.csv"))
        if len(label_files) == 0:
            raise FileNotFoundError("No trade outcome label files found in data/labels/!")

        latest_label_file = label_files[-1]
        print(f"\n✅ Loading trade outcome labels from: {latest_label_file.name}")

        labels_df = pd.read_csv(latest_label_file)
        labels_df['timestamp'] = pd.to_datetime(labels_df['timestamp'])
        labels_df.set_index('timestamp', inplace=True)

        # Merge labels with price data
        self.df = self.df.merge(
            labels_df[['signal_long', 'signal_short']],
            left_index=True, right_index=True, how='left'
        )

        # Fill any missing labels with 0 (safety)
        self.df['signal_long'] = self.df['signal_long'].fillna(0).astype(int)
        self.df['signal_short'] = self.df['signal_short'].fillna(0).astype(int)

        print(f"\nLabel distribution:")
        print(f"  LONG: {self.df['signal_long'].sum()} ({self.df['signal_long'].mean()*100:.2f}%)")
        print(f"  SHORT: {self.df['signal_short'].sum()} ({self.df['signal_short'].mean()*100:.2f}%)")

        # Split data
        last_date = self.df.index[-1]
        test_start = last_date - timedelta(weeks=self.holdout_weeks)
        val_start = test_start - timedelta(weeks=2)  # 2 weeks for validation

        train_mask = self.df.index < val_start
        val_mask = (self.df.index >= val_start) & (self.df.index < test_start)
        test_mask = self.df.index >= test_start

        self.train_df = self.df[train_mask].copy()
        self.val_df = self.df[val_mask].copy()
        self.test_df = self.df[test_mask].copy()

        print(f"\nData split:")
        print(f"  Training: {len(self.train_df)} candles ({self.train_df.index[0]} to {self.train_df.index[-1]})")
        print(f"  Validation: {len(self.val_df)} candles ({self.val_df.index[0]} to {self.val_df.index[-1]})")
        print(f"  Test (Backtest): {len(self.test_df)} candles ({self.test_df.index[0]} to {self.test_df.index[-1]})")

    def optimize_periods(self, max_combinations=30):
        """Optimize indicator periods"""

        print("\n" + "="*80)
        print(f"Step 2: Optimizing Indicator Periods ({self.signal_type})")
        print("="*80)

        # Combine train + val for period optimization
        optim_df = pd.concat([self.train_df, self.val_df])

        results = grid_search_periods(
            optim_df,
            signal_type=self.signal_type,
            max_combinations=max_combinations
        )

        if len(results) == 0:
            print("\n❌ No valid period combinations found!")
            return None

        # Select best configuration
        self.optimal_periods = results[0]['periods']

        print(f"\n✅ Optimal periods selected:")
        for param, value in self.optimal_periods.items():
            print(f"  {param}: {value}")

        print(f"\nPerformance with optimal periods:")
        print(f"  Composite Score: {results[0]['composite_score']:.4f}")
        print(f"  Accuracy: {results[0]['accuracy']:.4f}")
        print(f"  AUC: {results[0]['auc']:.4f}")
        print(f"  F1: {results[0]['f1']:.4f}")

        return results[0]

    def calculate_optimized_features(self):
        """Calculate features with optimized periods"""

        print("\n" + "="*80)
        print("Step 3: Calculating Features with Optimal Periods")
        print("="*80)

        if self.optimal_periods is None:
            print("\n⚠️  Using default periods (no optimization performed)")
            self.train_df = calculate_all_features(self.train_df)
            self.val_df = calculate_all_features(self.val_df)
            self.test_df = calculate_all_features(self.test_df)
        else:
            print(f"\nApplying optimized periods...")
            self.train_df = calculate_features_with_periods(self.train_df, self.optimal_periods)
            self.val_df = calculate_features_with_periods(self.val_df, self.optimal_periods)
            self.test_df = calculate_features_with_periods(self.test_df, self.optimal_periods)

        print(f"\nFeatures calculated:")
        print(f"  Training: {len(self.train_df.columns)} features")
        print(f"  Validation: {len(self.val_df.columns)} features")
        print(f"  Test: {len(self.test_df.columns)} features")

    def select_features(self, method='composite', top_k=50):
        """Select optimal feature subset"""

        print("\n" + "="*80)
        print(f"Step 4: Feature Selection ({self.signal_type})")
        print("="*80)

        # Feature columns
        exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'signal_long', 'signal_short']
        feature_cols = [col for col in self.train_df.columns if col not in exclude_cols]

        print(f"\nTotal features: {len(feature_cols)}")

        # Prepare data
        X_train = self.train_df[feature_cols].values
        y_train = self.train_df[f'signal_{self.signal_type.lower()}'].values
        X_val = self.val_df[feature_cols].values
        y_val = self.val_df[f'signal_{self.signal_type.lower()}'].values

        # Feature selection
        selector = FeatureSelector(X_train, y_train, X_val, y_val, feature_cols)

        # Train baseline
        baseline_metrics = selector.train_baseline_model()

        # Calculate importance
        selector.calculate_builtin_importance()
        selector.calculate_permutation_importance(n_repeats=3)
        selector.calculate_correlation_matrix(threshold=0.95)

        # Select features
        self.selected_features, importance_df = selector.select_features(
            method=method,
            top_k=top_k,
            importance_threshold=0.001
        )

        # Evaluate selected features
        selected_metrics, _ = selector.evaluate_selected_features(self.selected_features)

        print(f"\n✅ Selected {len(self.selected_features)} features")
        print(f"   Reduction: {(1 - len(self.selected_features)/len(feature_cols))*100:.1f}%")

        print(f"\nPerformance comparison:")
        print(f"  Baseline  : Acc={baseline_metrics['accuracy']:.4f} | AUC={baseline_metrics['auc']:.4f} | F1={baseline_metrics['f1']:.4f}")
        print(f"  Selected  : Acc={selected_metrics['accuracy']:.4f} | AUC={selected_metrics['auc']:.4f} | F1={selected_metrics['f1']:.4f}")
        print(f"  Change    : Acc={((selected_metrics['accuracy']-baseline_metrics['accuracy'])*100):+.2f}% | "
              f"AUC={((selected_metrics['auc']-baseline_metrics['auc'])*100):+.2f}% | "
              f"F1={((selected_metrics['f1']-baseline_metrics['f1'])*100):+.2f}%")

        return importance_df

    def train_final_model(self):
        """Train final model with optimal configuration"""

        print("\n" + "="*80)
        print(f"Step 5: Training Final Model ({self.signal_type})")
        print("="*80)

        # Get feature indices
        exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'signal_long', 'signal_short']
        all_feature_cols = [col for col in self.train_df.columns if col not in exclude_cols]
        feature_indices = [i for i, f in enumerate(all_feature_cols) if f in self.selected_features]

        # Prepare data
        X_train = self.train_df[self.selected_features].values
        y_train = self.train_df[f'signal_{self.signal_type.lower()}'].values

        print(f"\nTraining data:")
        print(f"  Samples: {len(X_train)}")
        print(f"  Features: {X_train.shape[1]}")
        print(f"  Positive ratio: {y_train.mean()*100:.2f}%")

        # Train model
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

        self.model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1
        )

        print(f"\nTraining model...")
        self.model.fit(X_train, y_train, verbose=0)

        print(f"✅ Model training complete")

        # Evaluate on validation set
        X_val = self.val_df[self.selected_features].values
        y_val = self.val_df[f'signal_{self.signal_type.lower()}'].values

        y_pred = self.model.predict(X_val)
        y_pred_proba = self.model.predict_proba(X_val)[:, 1]

        val_metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'auc': roc_auc_score(y_val, y_pred_proba) if len(np.unique(y_val)) > 1 else 0.5,
            'f1': f1_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, zero_division=0),
            'recall': recall_score(y_val, y_pred, zero_division=0)
        }

        print(f"\nValidation Performance:")
        print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  AUC: {val_metrics['auc']:.4f}")
        print(f"  F1: {val_metrics['f1']:.4f}")
        print(f"  Precision: {val_metrics['precision']:.4f}")
        print(f"  Recall: {val_metrics['recall']:.4f}")

        return val_metrics

    def backtest_validation(self):
        """Validate on holdout test set (last 4 weeks)"""

        print("\n" + "="*80)
        print(f"Step 6: Backtest Validation ({self.holdout_weeks} weeks holdout)")
        print("="*80)

        # Prepare test data
        X_test = self.test_df[self.selected_features].values
        y_test = self.test_df[f'signal_{self.signal_type.lower()}'].values

        print(f"\nTest data:")
        print(f"  Samples: {len(X_test)}")
        print(f"  Period: {self.test_df.index[0]} to {self.test_df.index[-1]}")
        print(f"  Positive ratio: {y_test.mean()*100:.2f}%")

        # Predict
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        test_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.5,
            'f1': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'prediction_rate': y_pred.mean()
        }

        print(f"\n✅ Backtest Performance (Unseen Data):")
        print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  AUC: {test_metrics['auc']:.4f}")
        print(f"  F1: {test_metrics['f1']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall: {test_metrics['recall']:.4f}")
        print(f"  Prediction Rate: {test_metrics['prediction_rate']*100:.2f}%")

        # Additional analysis
        print(f"\nSignal distribution on test set:")
        for threshold in [0.6, 0.65, 0.7, 0.75, 0.8]:
            signals = (y_pred_proba >= threshold).sum()
            print(f"  Threshold {threshold:.2f}: {signals} signals ({signals/len(y_pred_proba)*100:.2f}%)")

        return test_metrics

    def save_results(self, importance_df, val_metrics, test_metrics):
        """Save all results and model"""

        print("\n" + "="*80)
        print("Step 7: Saving Results")
        print("="*80)

        RESULTS_DIR = PROJECT_ROOT / "results"
        MODELS_DIR = PROJECT_ROOT / "models"
        RESULTS_DIR.mkdir(exist_ok=True)
        MODELS_DIR.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        signal_lower = self.signal_type.lower()

        # Save model
        model_filename = f"xgboost_{signal_lower}_optimized_{timestamp}.pkl"
        joblib.dump(self.model, MODELS_DIR / model_filename)
        print(f"\n✅ Model saved: {model_filename}")

        # Save feature list
        feature_filename = f"features_{signal_lower}_optimized_{timestamp}.txt"
        with open(MODELS_DIR / feature_filename, 'w') as f:
            f.write('\n'.join(self.selected_features))
        print(f"✅ Features saved: {feature_filename}")

        # Save periods
        if self.optimal_periods is not None:
            periods_filename = f"periods_{signal_lower}_optimized_{timestamp}.json"
            with open(RESULTS_DIR / periods_filename, 'w') as f:
                json.dump(self.optimal_periods, f, indent=2)
            print(f"✅ Periods saved: {periods_filename}")

        # Save comprehensive results
        results = {
            'signal_type': self.signal_type,
            'timestamp': timestamp,
            'data_info': {
                'total_candles': len(self.df),
                'train_candles': len(self.train_df),
                'val_candles': len(self.val_df),
                'test_candles': len(self.test_df),
                'test_period_weeks': self.holdout_weeks
            },
            'optimization': {
                'optimal_periods': self.optimal_periods,
                'selected_features': self.selected_features,
                'num_features_original': len([col for col in self.train_df.columns
                                             if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                                           'signal_long', 'signal_short']]),
                'num_features_selected': len(self.selected_features)
            },
            'performance': {
                'validation': val_metrics,
                'test_backtest': test_metrics
            },
            'model_file': model_filename,
            'feature_file': feature_filename
        }

        results_filename = f"optimization_results_{signal_lower}_{timestamp}.json"
        with open(RESULTS_DIR / results_filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✅ Results saved: {results_filename}")

        # Save feature importance
        importance_filename = f"feature_importance_{signal_lower}_{timestamp}.csv"
        importance_df.to_csv(RESULTS_DIR / importance_filename, index=False)
        print(f"✅ Feature importance saved: {importance_filename}")

        print(f"\n{'='*80}")
        print("All results saved successfully!")
        print(f"{'='*80}")

    def run_pipeline(self, optimize_periods=True, max_period_combinations=30, top_k_features=50):
        """Run complete optimization pipeline"""

        print("="*80)
        print(f"OPTIMIZATION PIPELINE - {self.signal_type} Signals")
        print("="*80)
        print(f"\nConfiguration:")
        print(f"  Optimize Periods: {optimize_periods}")
        print(f"  Period Combinations: {max_period_combinations}")
        print(f"  Top K Features: {top_k_features}")
        print(f"  Holdout Period: {self.holdout_weeks} weeks")

        # Run pipeline steps
        self.load_data()

        if optimize_periods:
            self.optimize_periods(max_combinations=max_period_combinations)

        self.calculate_optimized_features()
        importance_df = self.select_features(method='composite', top_k=top_k_features)
        val_metrics = self.train_final_model()
        test_metrics = self.backtest_validation()
        self.save_results(importance_df, val_metrics, test_metrics)

        print("\n" + "="*80)
        print("PIPELINE COMPLETE! ✅")
        print("="*80)

        return {
            'optimal_periods': self.optimal_periods,
            'selected_features': self.selected_features,
            'validation_metrics': val_metrics,
            'test_metrics': test_metrics
        }


def main():
    """Main execution with argparse"""

    parser = argparse.ArgumentParser(description='Optimize and retrain trading models')
    parser.add_argument('--signal-type', type=str, default='BOTH',
                       choices=['LONG', 'SHORT', 'BOTH'],
                       help='Signal type to optimize')
    parser.add_argument('--holdout-weeks', type=int, default=4,
                       help='Number of weeks to hold out for backtest')
    parser.add_argument('--optimize-periods', action='store_true', default=False,
                       help='Enable period optimization (slower)')
    parser.add_argument('--period-combinations', type=int, default=30,
                       help='Number of period combinations to test')
    parser.add_argument('--top-k', type=int, default=50,
                       help='Number of top features to select')

    args = parser.parse_args()

    if args.signal_type == 'BOTH':
        signal_types = ['LONG', 'SHORT']
    else:
        signal_types = [args.signal_type]

    # Run pipeline for each signal type
    all_results = {}

    for signal_type in signal_types:
        print(f"\n\n{'='*80}")
        print(f"Processing {signal_type} signals...")
        print(f"{'='*80}\n")

        pipeline = OptimizationPipeline(
            signal_type=signal_type,
            holdout_weeks=args.holdout_weeks
        )

        results = pipeline.run_pipeline(
            optimize_periods=args.optimize_periods,
            max_period_combinations=args.period_combinations,
            top_k_features=args.top_k
        )

        all_results[signal_type] = results

    # Print final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    for signal_type, results in all_results.items():
        print(f"\n{signal_type} Signals:")
        print(f"  Selected Features: {len(results['selected_features'])}")
        print(f"  Validation Performance:")
        print(f"    AUC: {results['validation_metrics']['auc']:.4f}")
        print(f"    F1: {results['validation_metrics']['f1']:.4f}")
        print(f"  Test Performance (Backtest):")
        print(f"    AUC: {results['test_metrics']['auc']:.4f}")
        print(f"    F1: {results['test_metrics']['f1']:.4f}")


if __name__ == "__main__":
    main()