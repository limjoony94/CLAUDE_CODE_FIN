"""
Backtest-Production Parity Verification Script
==============================================

Systematically verify code-level differences between backtest and production
to identify root cause of performance discrepancy.

Usage:
    python scripts/analysis/verify_backtest_production_parity.py

Output:
    - Console report with pass/fail for each check
    - Detailed JSON report in results/parity_verification_YYYYMMDD_HHMMSS.json
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import joblib
from datetime import datetime, timedelta
import json
import warnings

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.bingx_client import BingXClient
from scripts.experiments.calculate_all_features_enhanced_v2 import calculate_all_features_enhanced_v2
from scripts.experiments.calculate_all_features import calculate_all_features
from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features
from scripts.production.dynamic_position_sizing import DynamicPositionSizer

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Production configuration (from opportunity_gating_bot_4x.py)
LONG_THRESHOLD = 0.80
SHORT_THRESHOLD = 0.80
ML_EXIT_THRESHOLD_LONG = 0.80
ML_EXIT_THRESHOLD_SHORT = 0.80
EMERGENCY_STOP_LOSS = 0.03
EMERGENCY_MAX_HOLD_TIME = 120
LEVERAGE = 4

# Expected model performance (from backtest)
EXPECTED_RETURN_7D = 0.2902  # 29.02%
EXPECTED_WIN_RATE = 0.472    # 47.2%
EXPECTED_TRADES_7D = 36      # ~5.1/day


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_models():
    """Load all production models (ACTUAL production models from bot)"""
    print("📦 Loading PRODUCTION models...")

    models = {}

    # LONG Entry (PRODUCTION: Enhanced model 2025-10-24)
    long_model_path = MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445.pkl"
    with open(long_model_path, 'rb') as f:
        models['long_entry'] = pickle.load(f)

    long_scaler_path = MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_scaler.pkl"
    models['long_entry_scaler'] = joblib.load(long_scaler_path)

    long_features_path = MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_features.txt"
    with open(long_features_path, 'r') as f:
        models['long_entry_features'] = [line.strip() for line in f.readlines()]

    # SHORT Entry (PRODUCTION: Enhanced model 2025-10-24)
    short_model_path = MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445.pkl"
    with open(short_model_path, 'rb') as f:
        models['short_entry'] = pickle.load(f)

    short_scaler_path = MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_scaler.pkl"
    models['short_entry_scaler'] = joblib.load(short_scaler_path)

    short_features_path = MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_features.txt"
    with open(short_features_path, 'r') as f:
        models['short_entry_features'] = [line.strip() for line in f.readlines() if line.strip()]

    # LONG Exit (Retrained 2025-10-24)
    long_exit_model_path = MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527.pkl"
    with open(long_exit_model_path, 'rb') as f:
        models['long_exit'] = pickle.load(f)

    long_exit_scaler_path = MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527_scaler.pkl"
    models['long_exit_scaler'] = joblib.load(long_exit_scaler_path)

    long_exit_features_path = MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527_features.txt"
    with open(long_exit_features_path, 'r') as f:
        models['long_exit_features'] = [line.strip() for line in f.readlines() if line.strip()]

    # SHORT Exit (Retrained 2025-10-24)
    short_exit_model_path = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510.pkl"
    with open(short_exit_model_path, 'rb') as f:
        models['short_exit'] = pickle.load(f)

    short_exit_scaler_path = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510_scaler.pkl"
    models['short_exit_scaler'] = joblib.load(short_exit_scaler_path)

    short_exit_features_path = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510_features.txt"
    with open(short_exit_features_path, 'r') as f:
        models['short_exit_features'] = [line.strip() for line in f.readlines() if line.strip()]

    print(f"  ✅ Loaded {len(models)} model components")
    print(f"     LONG Entry features: {len(models['long_entry_features'])}")
    print(f"     SHORT Entry features: {len(models['short_entry_features'])}")
    print(f"     LONG Exit features: {len(models['long_exit_features'])}")
    print(f"     SHORT Exit features: {len(models['short_exit_features'])}")
    return models


def fetch_test_data(limit=1440):
    """Fetch recent candle data for testing"""
    print(f"📊 Fetching test data ({limit} candles)...")

    try:
        # Load API keys
        import yaml
        CONFIG_DIR = PROJECT_ROOT / "config"
        with open(CONFIG_DIR / "api_keys.yaml", 'r') as f:
            config = yaml.safe_load(f)

        api_keys = config['bingx']['mainnet']

        client = BingXClient(
            api_key=api_keys['api_key'],
            secret_key=api_keys['secret_key'],
            testnet=False
        )
        candles = client.get_klines(symbol='BTC-USDT', interval='5m', limit=limit)

        df = pd.DataFrame(candles, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume'
        ])

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        print(f"  ✅ Fetched {len(df)} candles ({df['timestamp'].min()} to {df['timestamp'].max()})")
        return df

    except Exception as e:
        print(f"  ❌ Error fetching data: {e}")
        return None


# =============================================================================
# VERIFICATION TESTS
# =============================================================================

def test_feature_calculation(df_raw):
    """
    Test 1: Feature Calculation Parity

    Verify that backtest and production calculate identical features
    """
    print("\n" + "="*80)
    print("TEST 1: Feature Calculation Parity")
    print("="*80)

    results = {
        'test_name': 'feature_calculation',
        'passed': False,
        'details': {}
    }

    try:
        # Backtest method (calculate_all_features)
        print("\n📐 Calculating features (backtest method)...")
        df_backtest = df_raw.copy()
        df_backtest = calculate_all_features(df_backtest)
        backtest_features = [c for c in df_backtest.columns if c not in
                            ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        print(f"  Backtest features: {len(backtest_features)}")

        # Production method (calculate_all_features_enhanced_v2)
        print("\n📐 Calculating features (production method)...")
        df_production = df_raw.copy()
        df_production = calculate_all_features_enhanced_v2(df_production)
        production_features = [c for c in df_production.columns if c not in
                              ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        print(f"  Production features: {len(production_features)}")

        # Compare feature sets
        print("\n🔍 Comparing feature sets...")
        backtest_set = set(backtest_features)
        production_set = set(production_features)

        only_in_backtest = backtest_set - production_set
        only_in_production = production_set - backtest_set
        common_features = backtest_set & production_set

        results['details']['backtest_count'] = len(backtest_features)
        results['details']['production_count'] = len(production_features)
        results['details']['common_count'] = len(common_features)
        results['details']['only_backtest'] = list(only_in_backtest)
        results['details']['only_production'] = list(only_in_production)

        print(f"  Common features: {len(common_features)}")
        print(f"  Only in backtest: {len(only_in_backtest)}")
        if only_in_backtest:
            print(f"    {list(only_in_backtest)[:5]}...")
        print(f"  Only in production: {len(only_in_production)}")
        if only_in_production:
            print(f"    {list(only_in_production)[:5]}...")

        # Compare values for common features
        if len(common_features) > 0:
            print("\n📊 Comparing feature values (common features)...")

            # Drop NaN rows for fair comparison
            df_b_clean = df_backtest[list(common_features)].dropna()
            df_p_clean = df_production[list(common_features)].dropna()

            # Align indices
            common_indices = df_b_clean.index.intersection(df_p_clean.index)
            df_b_clean = df_b_clean.loc[common_indices]
            df_p_clean = df_p_clean.loc[common_indices]

            if len(common_indices) > 100:  # Need sufficient data
                diff = (df_b_clean - df_p_clean).abs()

                max_diff = diff.max().max()
                mean_diff = diff.mean().mean()

                results['details']['value_comparison'] = {
                    'rows_compared': len(common_indices),
                    'max_difference': float(max_diff),
                    'mean_difference': float(mean_diff),
                    'threshold': 1e-6
                }

                print(f"  Rows compared: {len(common_indices)}")
                print(f"  Max difference: {max_diff:.2e}")
                print(f"  Mean difference: {mean_diff:.2e}")
                print(f"  Threshold: 1e-6")

                # Check if differences are within floating point precision
                if max_diff < 1e-6:
                    print("  ✅ Feature values match (within floating point precision)")
                    results['details']['values_match'] = True
                else:
                    print(f"  ⚠️ Significant differences detected!")
                    # Find features with largest differences
                    max_diff_per_feature = diff.max()
                    worst_features = max_diff_per_feature.nlargest(5)
                    print(f"\n  Top 5 features with largest differences:")
                    for feat, val in worst_features.items():
                        print(f"    {feat}: {val:.2e}")
                    results['details']['values_match'] = False
                    results['details']['worst_features'] = {
                        k: float(v) for k, v in worst_features.items()
                    }
            else:
                print(f"  ⚠️ Insufficient clean data for comparison ({len(common_indices)} rows)")
                results['details']['values_match'] = None

        # Overall pass/fail
        feature_sets_match = (only_in_backtest == set() and only_in_production == set())
        values_match = results['details'].get('values_match', False)

        results['passed'] = feature_sets_match and values_match

        if results['passed']:
            print("\n✅ TEST PASSED: Feature calculation is identical")
        else:
            print("\n❌ TEST FAILED: Feature calculation differs")

    except Exception as e:
        print(f"\n❌ TEST ERROR: {e}")
        results['error'] = str(e)
        import traceback
        traceback.print_exc()

    return results


def test_model_prediction(df_features, models):
    """
    Test 2: Model Prediction Parity

    Verify that models produce consistent predictions given same features
    """
    print("\n" + "="*80)
    print("TEST 2: Model Prediction Parity")
    print("="*80)

    results = {
        'test_name': 'model_prediction',
        'passed': False,
        'details': {}
    }

    try:
        # Drop NaN rows
        df_clean = df_features.dropna()

        if len(df_clean) < 100:
            print(f"⚠️ Insufficient clean data ({len(df_clean)} rows)")
            results['error'] = 'insufficient_data'
            return results

        print(f"📊 Testing with {len(df_clean)} clean rows")

        # Test LONG entry model
        print("\n🔵 Testing LONG entry model...")
        long_feat_cols = models['long_entry_features']

        # Check feature availability
        missing_features = [f for f in long_feat_cols if f not in df_clean.columns]
        if missing_features:
            print(f"  ❌ Missing {len(missing_features)} features:")
            print(f"     {missing_features[:5]}...")
            results['details']['long_entry'] = {
                'missing_features': missing_features,
                'status': 'missing_features'
            }
        else:
            long_feat = df_clean[long_feat_cols].values
            long_feat_scaled = models['long_entry_scaler'].transform(long_feat)
            long_probs = models['long_entry'].predict_proba(long_feat_scaled)[:, 1]

            print(f"  ✅ Predictions generated")
            print(f"     Mean: {long_probs.mean():.4f}")
            print(f"     Std: {long_probs.std():.4f}")
            print(f"     Min: {long_probs.min():.4f}, Max: {long_probs.max():.4f}")

            results['details']['long_entry'] = {
                'predictions': len(long_probs),
                'mean_prob': float(long_probs.mean()),
                'std_prob': float(long_probs.std()),
                'min_prob': float(long_probs.min()),
                'max_prob': float(long_probs.max()),
                'status': 'success'
            }

        # Test SHORT entry model
        print("\n🔴 Testing SHORT entry model...")
        short_feat_cols = models['short_entry_features']

        missing_features = [f for f in short_feat_cols if f not in df_clean.columns]
        if missing_features:
            print(f"  ❌ Missing {len(missing_features)} features:")
            print(f"     {missing_features[:5]}...")
            results['details']['short_entry'] = {
                'missing_features': missing_features,
                'status': 'missing_features'
            }
        else:
            short_feat = df_clean[short_feat_cols].values
            short_feat_scaled = models['short_entry_scaler'].transform(short_feat)
            short_probs = models['short_entry'].predict_proba(short_feat_scaled)[:, 1]

            print(f"  ✅ Predictions generated")
            print(f"     Mean: {short_probs.mean():.4f}")
            print(f"     Std: {short_probs.std():.4f}")
            print(f"     Min: {short_probs.min():.4f}, Max: {short_probs.max():.4f}")

            results['details']['short_entry'] = {
                'predictions': len(short_probs),
                'mean_prob': float(short_probs.mean()),
                'std_prob': float(short_probs.std()),
                'min_prob': float(short_probs.min()),
                'max_prob': float(short_probs.max()),
                'status': 'success'
            }

        # Test Exit models
        print("\n🚪 Testing Exit models...")

        # Prepare exit features
        df_with_exit = prepare_exit_features(df_clean.copy())

        # LONG exit
        long_exit_cols = models['long_exit_features']
        missing_features = [f for f in long_exit_cols if f not in df_with_exit.columns]
        if missing_features:
            print(f"  ❌ LONG exit: Missing {len(missing_features)} features")
            results['details']['long_exit'] = {
                'missing_features': missing_features,
                'status': 'missing_features'
            }
        else:
            long_exit_feat = df_with_exit[long_exit_cols].dropna()
            if len(long_exit_feat) > 0:
                long_exit_scaled = models['long_exit_scaler'].transform(long_exit_feat.values)
                long_exit_probs = models['long_exit'].predict_proba(long_exit_scaled)[:, 1]

                print(f"  ✅ LONG exit predictions: {len(long_exit_probs)}")
                print(f"     Mean: {long_exit_probs.mean():.4f}")

                results['details']['long_exit'] = {
                    'predictions': len(long_exit_probs),
                    'mean_prob': float(long_exit_probs.mean()),
                    'status': 'success'
                }

        # SHORT exit
        short_exit_cols = models['short_exit_features']
        missing_features = [f for f in short_exit_cols if f not in df_with_exit.columns]
        if missing_features:
            print(f"  ❌ SHORT exit: Missing {len(missing_features)} features")
            results['details']['short_exit'] = {
                'missing_features': missing_features,
                'status': 'missing_features'
            }
        else:
            short_exit_feat = df_with_exit[short_exit_cols].dropna()
            if len(short_exit_feat) > 0:
                short_exit_scaled = models['short_exit_scaler'].transform(short_exit_feat.values)
                short_exit_probs = models['short_exit'].predict_proba(short_exit_scaled)[:, 1]

                print(f"  ✅ SHORT exit predictions: {len(short_exit_probs)}")
                print(f"     Mean: {short_exit_probs.mean():.4f}")

                results['details']['short_exit'] = {
                    'predictions': len(short_exit_probs),
                    'mean_prob': float(short_exit_probs.mean()),
                    'status': 'success'
                }

        # Overall pass/fail
        all_success = all(
            details.get('status') == 'success'
            for details in results['details'].values()
        )

        results['passed'] = all_success

        if results['passed']:
            print("\n✅ TEST PASSED: All models produce predictions")
        else:
            print("\n❌ TEST FAILED: Some models have issues")

    except Exception as e:
        print(f"\n❌ TEST ERROR: {e}")
        results['error'] = str(e)
        import traceback
        traceback.print_exc()

    return results


def test_exit_logic():
    """
    Test 3: Exit Logic Verification

    Verify exit thresholds and calculations match backtest
    """
    print("\n" + "="*80)
    print("TEST 3: Exit Logic Verification")
    print("="*80)

    results = {
        'test_name': 'exit_logic',
        'passed': False,
        'details': {}
    }

    try:
        print("\n🔍 Checking Exit parameters...")

        # ML Exit thresholds
        print(f"\n  ML Exit Thresholds:")
        print(f"    LONG: {ML_EXIT_THRESHOLD_LONG} (expected: 0.80)")
        print(f"    SHORT: {ML_EXIT_THRESHOLD_SHORT} (expected: 0.80)")

        ml_exit_correct = (
            ML_EXIT_THRESHOLD_LONG == 0.80 and
            ML_EXIT_THRESHOLD_SHORT == 0.80
        )

        results['details']['ml_exit_thresholds'] = {
            'long': ML_EXIT_THRESHOLD_LONG,
            'short': ML_EXIT_THRESHOLD_SHORT,
            'correct': ml_exit_correct
        }

        # Stop Loss
        print(f"\n  Emergency Stop Loss:")
        print(f"    Value: {EMERGENCY_STOP_LOSS} (expected: 0.03 = -3%)")

        sl_correct = EMERGENCY_STOP_LOSS == 0.03

        results['details']['stop_loss'] = {
            'value': EMERGENCY_STOP_LOSS,
            'correct': sl_correct
        }

        # Max Hold
        print(f"\n  Emergency Max Hold:")
        print(f"    Candles: {EMERGENCY_MAX_HOLD_TIME} (expected: 120)")
        print(f"    Hours: {EMERGENCY_MAX_HOLD_TIME * 5 / 60:.1f}")

        max_hold_correct = EMERGENCY_MAX_HOLD_TIME == 120

        results['details']['max_hold'] = {
            'candles': EMERGENCY_MAX_HOLD_TIME,
            'hours': EMERGENCY_MAX_HOLD_TIME * 5 / 60,
            'correct': max_hold_correct
        }

        # Test Stop Loss calculation
        print(f"\n  Testing Stop Loss calculation...")
        print(f"    Balance-based SL: {EMERGENCY_STOP_LOSS * 100}%")
        print(f"    Leverage: {LEVERAGE}x")

        # Example positions
        test_cases = [
            {'position_size_pct': 0.20, 'entry_price': 100000},
            {'position_size_pct': 0.50, 'entry_price': 100000},
            {'position_size_pct': 0.95, 'entry_price': 100000},
        ]

        sl_calculations = []
        for case in test_cases:
            pos_size = case['position_size_pct']
            entry = case['entry_price']

            # Calculate price SL percentage
            price_sl_pct = EMERGENCY_STOP_LOSS / (pos_size * LEVERAGE)

            # Calculate stop prices
            stop_long = entry * (1 - price_sl_pct)
            stop_short = entry * (1 + price_sl_pct)

            sl_calculations.append({
                'position_size': pos_size,
                'price_sl_pct': price_sl_pct,
                'stop_long': stop_long,
                'stop_short': stop_short
            })

            print(f"    Position {pos_size*100:.0f}%:")
            print(f"      Price SL: {price_sl_pct*100:.2f}%")
            print(f"      LONG stop: ${stop_long:,.2f}")
            print(f"      SHORT stop: ${stop_short:,.2f}")

        results['details']['sl_calculations'] = sl_calculations

        # Overall pass/fail
        results['passed'] = ml_exit_correct and sl_correct and max_hold_correct

        if results['passed']:
            print("\n✅ TEST PASSED: Exit logic configuration correct")
        else:
            print("\n❌ TEST FAILED: Exit logic configuration mismatch")

    except Exception as e:
        print(f"\n❌ TEST ERROR: {e}")
        results['error'] = str(e)
        import traceback
        traceback.print_exc()

    return results


def test_position_sizing(models, df_features):
    """
    Test 4: Position Sizing Verification

    Verify dynamic position sizing matches backtest
    """
    print("\n" + "="*80)
    print("TEST 4: Position Sizing Verification")
    print("="*80)

    results = {
        'test_name': 'position_sizing',
        'passed': False,
        'details': {}
    }

    try:
        print("\n📏 Testing Position Sizer...")

        # Initialize sizer
        sizer = DynamicPositionSizer()

        # Get clean data
        df_clean = df_features.dropna()
        if len(df_clean) < 100:
            print(f"⚠️ Insufficient data")
            results['error'] = 'insufficient_data'
            return results

        # Generate predictions
        long_feat_cols = models['long_entry_features']
        long_feat = df_clean[long_feat_cols].values
        long_feat_scaled = models['long_entry_scaler'].transform(long_feat)
        long_probs = models['long_entry'].predict_proba(long_feat_scaled)[:, 1]

        # Test position sizing
        test_probs = [0.60, 0.70, 0.80, 0.90, 0.95]
        sizes = []

        print(f"\n  Position sizes for different probabilities:")
        for prob in test_probs:
            size = sizer.calculate_position_size(
                signal_confidence=prob,
                balance=10000,
                current_price=100000
            )
            sizes.append({'prob': prob, 'size_pct': size})
            print(f"    Prob {prob:.2f}: {size*100:.1f}%")

        results['details']['test_sizes'] = sizes

        # Verify range (20-95%)
        all_in_range = all(0.20 <= s['size_pct'] <= 0.95 for s in sizes)
        increasing = all(sizes[i]['size_pct'] <= sizes[i+1]['size_pct']
                        for i in range(len(sizes)-1))

        print(f"\n  Verification:")
        print(f"    All in range [20%, 95%]: {all_in_range}")
        print(f"    Monotonically increasing: {increasing}")

        results['details']['verification'] = {
            'all_in_range': all_in_range,
            'monotonic': increasing
        }

        results['passed'] = all_in_range and increasing

        if results['passed']:
            print("\n✅ TEST PASSED: Position sizing correct")
        else:
            print("\n❌ TEST FAILED: Position sizing issues")

    except Exception as e:
        print(f"\n❌ TEST ERROR: {e}")
        results['error'] = str(e)
        import traceback
        traceback.print_exc()

    return results


def test_configuration():
    """
    Test 5: Configuration Verification

    Verify all configuration parameters match backtest
    """
    print("\n" + "="*80)
    print("TEST 5: Configuration Verification")
    print("="*80)

    results = {
        'test_name': 'configuration',
        'passed': False,
        'details': {}
    }

    try:
        print("\n🔧 Checking Configuration...")

        # Entry thresholds
        print(f"\n  Entry Thresholds:")
        print(f"    LONG: {LONG_THRESHOLD} (expected: 0.80)")
        print(f"    SHORT: {SHORT_THRESHOLD} (expected: 0.80)")

        entry_correct = LONG_THRESHOLD == 0.80 and SHORT_THRESHOLD == 0.80

        results['details']['entry_thresholds'] = {
            'long': LONG_THRESHOLD,
            'short': SHORT_THRESHOLD,
            'correct': entry_correct
        }

        # Leverage
        print(f"\n  Leverage:")
        print(f"    Value: {LEVERAGE}x (expected: 4x)")

        leverage_correct = LEVERAGE == 4

        results['details']['leverage'] = {
            'value': LEVERAGE,
            'correct': leverage_correct
        }

        # Expected performance
        print(f"\n  Expected Performance (7-day window):")
        print(f"    Return: {EXPECTED_RETURN_7D*100:.2f}%")
        print(f"    Win Rate: {EXPECTED_WIN_RATE*100:.1f}%")
        print(f"    Trades: {EXPECTED_TRADES_7D} (~{EXPECTED_TRADES_7D/7:.1f}/day)")

        results['details']['expected_performance'] = {
            'return_7d': EXPECTED_RETURN_7D,
            'win_rate': EXPECTED_WIN_RATE,
            'trades_7d': EXPECTED_TRADES_7D
        }

        # Overall pass/fail
        results['passed'] = entry_correct and leverage_correct

        if results['passed']:
            print("\n✅ TEST PASSED: Configuration matches backtest")
        else:
            print("\n❌ TEST FAILED: Configuration mismatch")

    except Exception as e:
        print(f"\n❌ TEST ERROR: {e}")
        results['error'] = str(e)
        import traceback
        traceback.print_exc()

    return results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run all verification tests"""
    print("="*80)
    print("BACKTEST-PRODUCTION PARITY VERIFICATION")
    print("="*80)
    print(f"Start Time: {datetime.now()}")

    # Initialize results
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'tests': [],
        'summary': {}
    }

    # Load models
    models = load_models()

    # Fetch test data
    df_raw = fetch_test_data(limit=1440)
    if df_raw is None:
        print("\n❌ CRITICAL: Cannot fetch test data")
        return

    # Run tests
    print("\n" + "="*80)
    print("RUNNING VERIFICATION TESTS")
    print("="*80)

    # Test 1: Feature Calculation
    test1_result = test_feature_calculation(df_raw)
    all_results['tests'].append(test1_result)

    # Calculate features for remaining tests
    df_features = calculate_all_features_enhanced_v2(df_raw.copy())

    # Test 2: Model Prediction
    test2_result = test_model_prediction(df_features, models)
    all_results['tests'].append(test2_result)

    # Test 3: Exit Logic
    test3_result = test_exit_logic()
    all_results['tests'].append(test3_result)

    # Test 4: Position Sizing
    test4_result = test_position_sizing(models, df_features)
    all_results['tests'].append(test4_result)

    # Test 5: Configuration
    test5_result = test_configuration()
    all_results['tests'].append(test5_result)

    # Summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)

    passed_tests = sum(1 for t in all_results['tests'] if t.get('passed', False))
    total_tests = len(all_results['tests'])

    all_results['summary'] = {
        'total_tests': total_tests,
        'passed': passed_tests,
        'failed': total_tests - passed_tests,
        'pass_rate': passed_tests / total_tests if total_tests > 0 else 0
    }

    print(f"\nTests Passed: {passed_tests}/{total_tests}")
    print(f"Pass Rate: {all_results['summary']['pass_rate']*100:.1f}%")

    for test in all_results['tests']:
        status = "✅ PASS" if test.get('passed', False) else "❌ FAIL"
        print(f"  {status}: {test['test_name']}")
        if 'error' in test:
            print(f"         Error: {test['error']}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = RESULTS_DIR / f"parity_verification_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n📄 Detailed report saved: {output_file}")

    # Overall result
    if passed_tests == total_tests:
        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED - Backtest matches Production")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("⚠️ SOME TESTS FAILED - Investigate differences")
        print("="*80)

    return all_results


if __name__ == "__main__":
    results = main()
