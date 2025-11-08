"""
Feature Calculation Validation Script
======================================

프로덕션 모델에 입력되는 모든 지표가 제대로 계산되는지 검증

검증 항목:
1. Entry 피처 계산 (calculate_all_features)
2. Exit 피처 계산 (prepare_exit_features)
3. 모든 함수가 실제로 구현되어 있는지
4. 빈 껍데기 함수가 없는지
5. NaN 값 처리가 올바른지

2025-10-23: 프로덕션 모델 입력 검증
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.calculate_all_features import calculate_all_features, SHORT_FEATURE_COLUMNS
from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features
from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures


def validate_function_implementation(func, name):
    """
    함수가 실제로 구현되어 있는지 확인 (빈 껍데기 체크)

    Args:
        func: 검증할 함수
        name: 함수 이름

    Returns:
        bool: 제대로 구현되어 있으면 True
    """
    import inspect

    # 함수 소스 코드 가져오기
    try:
        source = inspect.getsource(func)

        # 빈 껍데기 패턴 감지
        empty_patterns = [
            'pass',
            'return None',
            'raise NotImplementedError',
            'TODO',
            '...'
        ]

        # 실제 구현이 있는지 확인 (단순히 패턴만 있는지)
        has_implementation = False
        for line in source.split('\n'):
            stripped = line.strip()
            # 주석이나 docstring 제외
            if stripped and not stripped.startswith('#') and not stripped.startswith('"""') and not stripped.startswith("'''"):
                # 함수 정의나 빈 껍데기 패턴이 아닌 코드가 있는지
                if not any(pattern in stripped for pattern in empty_patterns) and not stripped.startswith('def '):
                    has_implementation = True
                    break

        return has_implementation

    except Exception as e:
        print(f"⚠️ {name}: 소스 코드 검증 실패 - {e}")
        return False


def validate_entry_features():
    """Entry 피처 계산 검증"""
    print("\n" + "="*80)
    print("1. ENTRY FEATURES VALIDATION")
    print("="*80)

    # 샘플 데이터 생성
    print("\n📋 Creating sample data...")
    np.random.seed(42)
    n_candles = 300

    df = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=n_candles, freq='5min'),
        'open': 100 + np.random.randn(n_candles).cumsum() * 0.5,
        'high': 100 + np.random.randn(n_candles).cumsum() * 0.5 + np.abs(np.random.randn(n_candles)) * 0.5,
        'low': 100 + np.random.randn(n_candles).cumsum() * 0.5 - np.abs(np.random.randn(n_candles)) * 0.5,
        'close': 100 + np.random.randn(n_candles).cumsum() * 0.5,
        'volume': np.abs(np.random.randn(n_candles)) * 1000 + 5000
    })

    print(f"✅ Sample data: {len(df)} rows, {len(df.columns)} columns")

    # 1.1 LONG 기본 피처 검증
    print("\n📊 1.1 Validating LONG basic features...")

    # 함수 구현 확인
    if not validate_function_implementation(calculate_features, "calculate_features"):
        print("❌ calculate_features: 빈 껍데기 함수!")
        return False

    df_basic = calculate_features(df.copy())

    expected_basic_features = [
        'close_change_1', 'close_change_2', 'close_change_3', 'close_change_4', 'close_change_5',
        'sma_10', 'sma_20', 'ema_3', 'ema_5', 'ema_10',
        'macd', 'macd_signal', 'macd_diff',
        'rsi', 'rsi_5', 'rsi_7',
        'bb_high', 'bb_low', 'bb_mid',
        'volatility', 'volatility_5', 'volatility_10',
        'volume_sma', 'volume_ratio', 'volume_spike', 'volume_trend',
        'price_mom_3', 'price_mom_5', 'price_vs_ema3', 'price_vs_ema5',
        'body_size', 'upper_shadow', 'lower_shadow'
    ]

    missing_basic = [f for f in expected_basic_features if f not in df_basic.columns]

    if missing_basic:
        print(f"❌ Missing {len(missing_basic)} basic features:")
        for f in missing_basic:
            print(f"   - {f}")
        return False
    else:
        print(f"✅ All {len(expected_basic_features)} basic features present")

    # 1.2 LONG 고급 피처 검증
    print("\n📊 1.2 Validating LONG advanced features...")

    adv_features = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)

    # 각 계산 함수 구현 확인
    functions_to_check = [
        ('detect_support_resistance', adv_features.detect_support_resistance),
        ('calculate_trend_lines', adv_features.calculate_trend_lines),
        ('detect_divergences', adv_features.detect_divergences),
        ('detect_chart_patterns', adv_features.detect_chart_patterns),
        ('calculate_volume_profile', adv_features.calculate_volume_profile),
        ('calculate_price_action_features', adv_features.calculate_price_action_features),
        ('calculate_short_specific_features', adv_features.calculate_short_specific_features)
    ]

    for func_name, func in functions_to_check:
        if not validate_function_implementation(func, func_name):
            print(f"❌ {func_name}: 빈 껍데기 함수!")
            return False

    df_advanced = adv_features.calculate_all_features(df_basic.copy())

    expected_advanced_features = adv_features.get_feature_names()
    missing_advanced = [f for f in expected_advanced_features if f not in df_advanced.columns]

    if missing_advanced:
        print(f"❌ Missing {len(missing_advanced)} advanced features:")
        for f in missing_advanced:
            print(f"   - {f}")
        return False
    else:
        print(f"✅ All {len(expected_advanced_features)} advanced features present")

    # 1.3 SHORT 피처 검증
    print("\n📊 1.3 Validating SHORT features...")

    df_all = calculate_all_features(df.copy())

    missing_short = [f for f in SHORT_FEATURE_COLUMNS if f not in df_all.columns]

    if missing_short:
        print(f"❌ Missing {len(missing_short)} SHORT features:")
        for f in missing_short:
            print(f"   - {f}")
        return False
    else:
        print(f"✅ All {len(SHORT_FEATURE_COLUMNS)} SHORT features present")

    # 1.4 NaN 값 확인
    print("\n📊 1.4 Checking for NaN values...")

    nan_cols = df_all.columns[df_all.isna().any()].tolist()

    if nan_cols:
        print(f"⚠️ Found {len(nan_cols)} columns with NaN values:")
        for col in nan_cols[:10]:  # 처음 10개만 표시
            nan_count = df_all[col].isna().sum()
            nan_pct = (nan_count / len(df_all)) * 100
            print(f"   - {col}: {nan_count} ({nan_pct:.1f}%)")
        if len(nan_cols) > 10:
            print(f"   ... and {len(nan_cols) - 10} more")

        # NaN 비율이 높으면 실패
        max_nan_pct = (df_all.isna().sum().max() / len(df_all)) * 100
        if max_nan_pct > 50:
            print(f"❌ Too many NaN values (max: {max_nan_pct:.1f}%)")
            return False
        else:
            print(f"⚠️ Some NaN values present, but acceptable (max: {max_nan_pct:.1f}%)")
    else:
        print("✅ No NaN values found")

    print(f"\n✅ ENTRY FEATURES VALIDATION PASSED")
    print(f"   Total features: {len(df_all.columns)}")
    print(f"   LONG basic: {len(expected_basic_features)}")
    print(f"   LONG advanced: {len(expected_advanced_features)}")
    print(f"   SHORT: {len(SHORT_FEATURE_COLUMNS)}")

    return True


def validate_exit_features():
    """Exit 피처 계산 검증"""
    print("\n" + "="*80)
    print("2. EXIT FEATURES VALIDATION")
    print("="*80)

    # 샘플 데이터 생성 (Entry 피처 포함)
    print("\n📋 Creating sample data with entry features...")
    np.random.seed(42)
    n_candles = 300

    df = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=n_candles, freq='5min'),
        'open': 100 + np.random.randn(n_candles).cumsum() * 0.5,
        'high': 100 + np.random.randn(n_candles).cumsum() * 0.5 + np.abs(np.random.randn(n_candles)) * 0.5,
        'low': 100 + np.random.randn(n_candles).cumsum() * 0.5 - np.abs(np.random.randn(n_candles)) * 0.5,
        'close': 100 + np.random.randn(n_candles).cumsum() * 0.5,
        'volume': np.abs(np.random.randn(n_candles)) * 1000 + 5000
    })

    # Entry 피처 먼저 계산
    df = calculate_all_features(df)

    print(f"✅ Sample data with entry features: {len(df)} rows, {len(df.columns)} columns")

    # 함수 구현 확인
    print("\n📊 Validating prepare_exit_features implementation...")

    if not validate_function_implementation(prepare_exit_features, "prepare_exit_features"):
        print("❌ prepare_exit_features: 빈 껍데기 함수!")
        return False

    # Exit 피처 계산
    df_exit = prepare_exit_features(df.copy())

    # Exit 피처 확인
    expected_exit_features = [
        'volume_ratio', 'volume_surge',
        'price_vs_ma20', 'price_vs_ma50',
        'returns', 'volatility_20', 'volatility_regime',
        'rsi_slope', 'rsi_overbought', 'rsi_oversold', 'rsi_divergence',
        'macd_histogram_slope', 'macd_crossover', 'macd_crossunder',
        'higher_high', 'lower_low', 'price_acceleration',
        'near_resistance', 'near_support',
        'bb_position'
    ]

    missing_exit = [f for f in expected_exit_features if f not in df_exit.columns]

    if missing_exit:
        print(f"❌ Missing {len(missing_exit)} exit features:")
        for f in missing_exit:
            print(f"   - {f}")
        return False
    else:
        print(f"✅ All {len(expected_exit_features)} exit features present")

    # NaN 값 확인
    print("\n📊 Checking for NaN values in exit features...")

    nan_cols = [col for col in expected_exit_features if col in df_exit.columns and df_exit[col].isna().any()]

    if nan_cols:
        print(f"⚠️ Found {len(nan_cols)} exit features with NaN values:")
        for col in nan_cols[:10]:
            nan_count = df_exit[col].isna().sum()
            nan_pct = (nan_count / len(df_exit)) * 100
            print(f"   - {col}: {nan_count} ({nan_pct:.1f}%)")

        # NaN 비율이 높으면 실패
        max_nan_pct = max([df_exit[col].isna().sum() / len(df_exit) * 100 for col in nan_cols])
        if max_nan_pct > 50:
            print(f"❌ Too many NaN values (max: {max_nan_pct:.1f}%)")
            return False
        else:
            print(f"⚠️ Some NaN values present, but acceptable (max: {max_nan_pct:.1f}%)")
    else:
        print("✅ No NaN values found in exit features")

    print(f"\n✅ EXIT FEATURES VALIDATION PASSED")
    print(f"   Total exit features: {len(expected_exit_features)}")

    return True


def main():
    """메인 검증 실행"""
    print("="*80)
    print("PRODUCTION MODEL FEATURE CALCULATION VALIDATION")
    print("="*80)
    print("\n목적: 프로덕션 모델에 입력되는 모든 지표가 제대로 계산되는지 검증")
    print("검증 항목:")
    print("  1. Entry 피처 계산 (LONG + SHORT)")
    print("  2. Exit 피처 계산")
    print("  3. 함수 구현 상태 (빈 껍데기 체크)")
    print("  4. NaN 값 처리")

    # Entry 피처 검증
    entry_valid = validate_entry_features()

    # Exit 피처 검증
    exit_valid = validate_exit_features()

    # 최종 결과
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    results = {
        "Entry Features": "✅ PASSED" if entry_valid else "❌ FAILED",
        "Exit Features": "✅ PASSED" if exit_valid else "❌ FAILED"
    }

    for check, result in results.items():
        print(f"  {check}: {result}")

    if entry_valid and exit_valid:
        print("\n" + "="*80)
        print("🎉 ALL VALIDATION PASSED - 모든 지표가 제대로 계산됩니다!")
        print("="*80)
        return True
    else:
        print("\n" + "="*80)
        print("❌ VALIDATION FAILED - 일부 지표에 문제가 있습니다!")
        print("="*80)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
