"""
Test Paper Trading Bot - Single Cycle Test

목적: Paper Trading Bot의 단일 사이클 테스트
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.paper_trading_bot import (
    PaperTradingBot,
    Config,
    calculate_features,
    MarketRegimeClassifier,
    XGBoostTradingStrategy,
    MODELS_DIR
)
from loguru import logger
import pandas as pd

def test_model_loading():
    """Test 1: XGBoost 모델 로딩 테스트"""
    logger.info("=" * 80)
    logger.info("Test 1: XGBoost Model Loading (SMOTE version)")
    logger.info("=" * 80)

    model_path = MODELS_DIR / "xgboost_model_smote.pkl"
    strategy = XGBoostTradingStrategy(model_path)

    if strategy.model is None:
        logger.error("❌ Model loading failed")
        return False

    logger.success("✅ Model loaded successfully")
    return True

def test_data_loading():
    """Test 2: 데이터 로딩 테스트"""
    logger.info("\n" + "=" * 80)
    logger.info("Test 2: Market Data Loading")
    logger.info("=" * 80)

    bot = PaperTradingBot()
    df = bot._get_market_data()

    if df is None or len(df) == 0:
        logger.error("❌ Failed to load market data")
        return False

    logger.success(f"✅ Market data loaded: {len(df)} candles")
    logger.info(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    return True

def test_feature_calculation():
    """Test 3: Feature 계산 테스트"""
    logger.info("\n" + "=" * 80)
    logger.info("Test 3: Feature Calculation")
    logger.info("=" * 80)

    bot = PaperTradingBot()
    df = bot._get_market_data()

    if df is None:
        logger.error("❌ No data to calculate features")
        return False

    df = calculate_features(df)

    feature_columns = [
        'close_change_1', 'sma_10', 'sma_20', 'ema_10',
        'macd', 'macd_signal', 'macd_diff', 'rsi',
        'bb_high', 'bb_low', 'bb_mid', 'volatility',
        'volume_sma', 'volume_ratio'
    ]

    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        logger.error(f"❌ Missing features: {missing_features}")
        return False

    # Check for NaN
    nan_count = df[feature_columns].isna().sum().sum()
    logger.info(f"   Total features: {len(feature_columns)}")
    logger.info(f"   NaN values: {nan_count}")

    logger.success("✅ Features calculated successfully")
    return True

def test_market_regime_classification():
    """Test 4: Market regime 분류 테스트"""
    logger.info("\n" + "=" * 80)
    logger.info("Test 4: Market Regime Classification")
    logger.info("=" * 80)

    bot = PaperTradingBot()
    df = bot._get_market_data()

    if df is None:
        logger.error("❌ No data for regime classification")
        return False

    classifier = MarketRegimeClassifier()
    regime = classifier.classify(df)

    logger.success(f"✅ Current market regime: {regime}")

    # Show price trend
    recent_data = df.tail(20)
    price_change_pct = ((recent_data['close'].iloc[-1] / recent_data['close'].iloc[0]) - 1) * 100
    logger.info(f"   Recent price change (20 periods): {price_change_pct:+.2f}%")

    return True

def test_prediction():
    """Test 5: XGBoost 예측 테스트"""
    logger.info("\n" + "=" * 80)
    logger.info("Test 5: XGBoost Prediction")
    logger.info("=" * 80)

    bot = PaperTradingBot()
    df = bot._get_market_data()

    if df is None:
        logger.error("❌ No data for prediction")
        return False

    df = calculate_features(df)

    prediction, probability, should_enter = bot.strategy.predict(df)

    logger.info(f"   Prediction: {prediction} (0=not enter, 1=enter)")
    logger.info(f"   Probability (class 1): {probability:.3f}")
    logger.info(f"   Expected return: {(probability - 0.5) * 2:.3f}")
    logger.info(f"   Should enter: {should_enter}")
    logger.info(f"   Entry threshold: {Config.ENTRY_THRESHOLD}")

    current_volatility = df['volatility'].iloc[-1]
    logger.info(f"   Current volatility: {current_volatility:.6f}")
    logger.info(f"   Min volatility: {Config.MIN_VOLATILITY}")

    if should_enter:
        logger.success("✅ Entry signal detected!")
    else:
        logger.warning("⚠️ No entry signal")

    return True

def test_single_cycle():
    """Test 6: 전체 사이클 통합 테스트"""
    logger.info("\n" + "=" * 80)
    logger.info("Test 6: Full Cycle Integration Test")
    logger.info("=" * 80)

    try:
        bot = PaperTradingBot()
        bot._update_cycle()

        logger.success("✅ Single cycle completed successfully")
        logger.info(f"   Current capital: ${bot.capital:,.2f}")
        logger.info(f"   Position: {bot.position}")
        logger.info(f"   Total trades: {len(bot.trades)}")

        return True

    except Exception as e:
        logger.error(f"❌ Cycle failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    logger.info("=" * 80)
    logger.info("Paper Trading Bot - Test Suite")
    logger.info("=" * 80)
    logger.info("비판적 질문: '코드를 작성했지만 검증했는가?'")
    logger.info("=" * 80)

    tests = [
        ("Model Loading", test_model_loading),
        ("Data Loading", test_data_loading),
        ("Feature Calculation", test_feature_calculation),
        ("Market Regime Classification", test_market_regime_classification),
        ("XGBoost Prediction", test_prediction),
        ("Full Cycle Integration", test_single_cycle),
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            logger.error(f"Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)

    total_tests = len(results)
    passed_tests = sum(1 for r in results.values() if r)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {test_name}")

    logger.info("=" * 80)
    logger.info(f"Total: {passed_tests}/{total_tests} tests passed")
    logger.info("=" * 80)

    if passed_tests == total_tests:
        logger.success("\n🎉 All tests passed! Paper Trading Bot is ready!")
        logger.info("\n📝 Next steps:")
        logger.info("   1. Run: python scripts/paper_trading_bot.py")
        logger.info("   2. Monitor: tail -f logs/paper_trading_*.log")
        logger.info("   3. Wait: 2-4 weeks for comprehensive testing")
    else:
        logger.error("\n⚠️ Some tests failed. Please fix issues before running bot.")

    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
