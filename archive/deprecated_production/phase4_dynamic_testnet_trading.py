"""
Phase 4 Dual Entry + Dual Exit Model TESTNET Trading Bot

목표: 4-Model 전략 실시간 검증 (Entry Dual + Exit Dual)
- Entry Models:
  * LONG Model: XGBoost Phase 4 (상승 예측 전용)
  * SHORT Model: XGBoost Short (하락 예측 전용)
- Exit Models:
  * LONG Exit Model: ML-learned optimal exit timing for LONG positions
  * SHORT Exit Model: ML-learned optimal exit timing for SHORT positions
- Dynamic Position Sizing (20-95% adaptive)
- 실제 BingX Testnet API 주문 실행
- 5분 캔들 기반

⚠️ 중요: 이것은 TESTNET TRADING입니다!
- 실제 주문 제출 (가상 자본)
- 실제 계좌 잔고 사용
- 실제 포지션 관리
- Paper Trading이 아님!

Expected Performance (from ML Exit Model backtesting):
- Returns: +2.85% per 2 days (+39.2% vs rule-based)
- Win Rate: 94.7% (+5.0% vs rule-based 89.7%)
- Avg Holding: 2.36 hours (-41% vs rule-based 4.0h)
- Exit Efficiency: 87.6% ML Exit, 12.4% Max Hold
- Entry Distribution: 87.6% LONG, 12.4% SHORT
"""

import os
import time
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
from loguru import logger
import yaml

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"
CONFIG_DIR = PROJECT_ROOT / "config"

# Import project modules
import sys
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures
from scripts.production.dynamic_position_sizing import DynamicPositionSizer
from src.api.bingx_client import BingXClient
from src.api.exceptions import (
    BingXAPIError,
    BingXOrderError,
    BingXInsufficientBalanceError
)
from src.utils.institutional_logger import InstitutionalLogger

# Create directories
for dir_path in [RESULTS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Logging setup
log_file = LOGS_DIR / f"phase4_dynamic_testnet_trading_{datetime.now().strftime('%Y%m%d')}.log"
logger.add(log_file, rotation="1 day", retention="30 days")

# ============================================================================
# Bot Singleton - Prevent Multiple Instances
# ============================================================================

class BotSingleton:
    """
    Singleton pattern to ensure only one bot instance runs at a time

    Prevents duplicate bot execution that causes:
    - Conflicting orders
    - Statistical confusion
    - Increased API rate limit violations
    """

    def __init__(self):
        self.lock_file = None
        self.lock_path = RESULTS_DIR / "bot_instance.lock"

    def __enter__(self):
        """Acquire exclusive lock on bot execution"""
        try:
            # Create lock file
            self.lock_file = open(self.lock_path, 'w')

            # Try to acquire exclusive lock (non-blocking)
            # This will fail if another instance already holds the lock
            if sys.platform == 'win32':
                # Windows: Use msvcrt for file locking
                import msvcrt
                try:
                    msvcrt.locking(self.lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                except IOError:
                    logger.error("=" * 80)
                    logger.error("❌ BOT ALREADY RUNNING!")
                    logger.error("=" * 80)
                    logger.error("Another instance of the bot is currently active.")
                    logger.error("Only ONE bot instance can run at a time to prevent:")
                    logger.error("  - Duplicate order execution")
                    logger.error("  - API rate limit violations")
                    logger.error("  - Statistical tracking conflicts")
                    logger.error("")
                    logger.error("Check running processes:")
                    logger.error("  ps aux | grep phase4_dynamic_testnet_trading")
                    logger.error("=" * 80)
                    sys.exit(1)
            else:
                # Unix/Linux: Use fcntl for file locking
                import fcntl
                try:
                    fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except IOError:
                    logger.error("=" * 80)
                    logger.error("❌ BOT ALREADY RUNNING!")
                    logger.error("=" * 80)
                    logger.error("Another instance of the bot is currently active.")
                    logger.error("Only ONE bot instance can run at a time.")
                    logger.error("")
                    logger.error("Check running processes:")
                    logger.error("  ps aux | grep phase4_dynamic_testnet_trading")
                    logger.error("=" * 80)
                    sys.exit(1)

            # Write PID to lock file
            self.lock_file.write(f"{os.getpid()}\n")
            self.lock_file.write(f"{datetime.now().isoformat()}\n")
            self.lock_file.flush()

            logger.success("✅ Bot instance lock acquired")

        except Exception as e:
            logger.error(f"Failed to acquire bot lock: {e}")
            sys.exit(1)

        return self

    def __exit__(self, *args):
        """Release lock on bot exit"""
        if self.lock_file:
            try:
                # Release lock
                if sys.platform == 'win32':
                    import msvcrt
                    msvcrt.locking(self.lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN)

                self.lock_file.close()

                # Remove lock file
                if self.lock_path.exists():
                    self.lock_path.unlink()

                logger.info("✅ Bot instance lock released")

            except Exception as e:
                logger.error(f"Error releasing bot lock: {e}")

# ============================================================================
# Phase 4 Dynamic Testnet Configuration
# ============================================================================

class Phase4TestnetConfig:
    """Phase 4 Dual Model Testnet Configuration"""

    # XGBoost Thresholds (2025-10-15: DYNAMIC THRESHOLD SYSTEM - Root Cause Solution)
    # HISTORY:
    # - Previous: LONG 0.80, SHORT 0.50 (asymmetric labels - Train-Test Mismatch)
    # - Fixed: LONG 0.80, SHORT 0.80 (symmetric labels - 12.29% return)
    # - Entry Optimized: LONG 0.70, SHORT 0.65 (19.88% return, +62% improvement!)
    # - V2 Optimization (2 weeks): All parameters (35.67% return, +79% vs entry-only!)
    # - V3 Optimization (3 months): Same parameters validated (97.82% train return, robust!)
    # - 2025-10-15 17:40: DYNAMIC THRESHOLD SYSTEM (adaptive to market regime changes)

    # Base Thresholds (optimized from backtest, used as baseline for dynamic adjustment)
    BASE_LONG_ENTRY_THRESHOLD = 0.70   # V3 optimal (tested on 11.70% signal rate test set)
    BASE_SHORT_ENTRY_THRESHOLD = 0.70  # OPTIMIZED (2025-10-16 Threshold Testing)
                                        # Window backtest: 52.3% win rate, 3.18% return per window
                                        # Threshold testing (0.55→0.70): Quality > Quantity
                                        # 22 SELL features (RSI/MACD), 9.1 trades/window, balanced LONG/SHORT
    EXIT_THRESHOLD = 0.7  # ENHANCED MODELS OPTIMAL (2025-10-16 Feature Engineering Complete)
                          # DEPLOYED: Enhanced EXIT models with 22 features + 2of3 scoring
                          # - High probability (>=0.7) = GOOD exits (+14.44% return, 67.4% win)
                          # - Validated across 21 windows with proper NORMAL logic (not inverted)
                          # - Feature engineering: 3 → 22 features (+volume, momentum, volatility)
                          # - Improvement vs inverted baseline: +2.84% return (+24.5% relative)
                          # - Models: xgboost_long_exit_improved_20251016_175554.pkl + short_180207.pkl
                          # Previous: 0.603 (Bayesian optimization, but wrong logic direction)

    # Dynamic Threshold Configuration (2025-10-16 V3: ACTUAL ENTRY RATE - Root Cause Fix)
    ENABLE_DYNAMIC_THRESHOLD = True  # Enable adaptive threshold system

    # Target Metrics (from backtest validation)
    TARGET_TRADES_PER_WEEK = 22.0    # Realistic target from recent observations
                                      # Previous: 42.5 (overestimated from test set)
    TARGET_ENTRY_RATE = TARGET_TRADES_PER_WEEK / (7 * 24 * 12)  # ~0.011 (1.1% of candles)
                                      # 7 days * 24 hours * 12 candles/hour = 2016 candles/week

    # Lookback Configuration
    DYNAMIC_LOOKBACK_HOURS = 6       # Monitor recent 6-hour actual entry rate
    LOOKBACK_CANDLES = DYNAMIC_LOOKBACK_HOURS * 12  # 72 candles (5-min intervals)
    MIN_ENTRIES_FOR_FEEDBACK = 5     # Minimum entries needed for actual rate calculation

    # Adjustment Parameters
    THRESHOLD_ADJUSTMENT_FACTOR = 0.20  # Conservative adjustment (was 0.25)
    MIN_THRESHOLD = 0.50             # Allow aggressive entries in quiet markets
    MAX_THRESHOLD = 0.75             # CRITICAL FIX: Lowered from 0.92 (model's practical limit)
                                      # Reason: Model rarely outputs prob > 0.80
                                      # 0.92 was unreachable and caused trading shutdown

    # Expected Metrics (2025-10-15: V3 FULL-DATASET OPTIMIZATION - Temporal Bias Eliminated)
    # Backtest Results (3-month dataset with walk-forward validation):
    # - Training: 97.82% return (9.0 weeks), Sharpe 31.00
    # - Validation: 7.60% return (1.9 weeks), Sharpe 25.06
    # - Test (out-of-sample): 28.66% return (1.9 weeks), Sharpe 16.60
    # - Win Rate: 82.9% (validated on 3-month data, not just 2 weeks)
    # - Trades/Week: 42.5 (test set, similar to V2 but more robust)
    # - Avg Position: 71.6% (validated across multiple market regimes)
    # - Max Drawdown: -8.43% (V3 test set)
    # - Dataset: 3x more data than V2, eliminates Oct 10 outlier bias
    EXPECTED_RETURN_PER_WEEK = 14.86  # From V3 test set (28.66% / 1.9 weeks)
    EXPECTED_WIN_RATE = 82.9  # V3 test set result (out-of-sample validated)
    EXPECTED_TRADES_PER_WEEK = 42.5  # V3 test set (validated on diverse conditions)
    EXPECTED_SHARPE_RATIO = 16.60  # V3 test set (robust, out-of-sample)
    EXPECTED_MAX_DRAWDOWN = -8.43  # V3 test set
    EXPECTED_AVG_POSITION = 71.6  # average position size %
    EXPECTED_AVG_HOLDING = 1.53  # hours (unchanged)
    EXPECTED_LONG_RATIO = 91.7  # % of LONG trades (expected)
    EXPECTED_SHORT_RATIO = 8.3  # % of SHORT trades (expected)

    # Targets
    TARGET_WIN_RATE = 60.0
    TARGET_VS_BH = 10.0
    TARGET_AVG_POSITION = (40.0, 70.0)  # reasonable range

    # API Configuration (BingX Testnet) - Load from YAML config
    @staticmethod
    def _load_api_keys():
        """Load API keys from config/api_keys.yaml"""
        api_keys_file = CONFIG_DIR / "api_keys.yaml"
        if api_keys_file.exists():
            with open(api_keys_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                return config.get('bingx', {}).get('testnet', {})
        return {}

    _api_config = _load_api_keys()
    API_KEY = _api_config.get('api_key', os.getenv("BINGX_API_KEY", ""))
    API_SECRET = _api_config.get('secret_key', os.getenv("BINGX_API_SECRET", ""))
    USE_TESTNET = True  # ALWAYS testnet for this bot!

    # Trading Parameters (2025-10-15: EXIT OPTIMIZED)
    SYMBOL = "BTC-USDT"
    TIMEFRAME = "5m"
    STOP_LOSS = 0.01  # 1% (optimal from 81 combinations backtest)
    TAKE_PROFIT = 0.02  # 2% (optimized: 3% → 2%, early profit taking strategy!)
    MAX_HOLDING_HOURS = 4  # 4 hours (optimal from backtest)

    # Dynamic Position Sizing Parameters (2025-10-15: V3 FULL-DATASET OPTIMIZATION)
    # Optimized on 3-month dataset with walk-forward validation (27 weight + 6 position combos)
    # Key findings:
    # - Same optimal parameters as V2, but validated on 6x more data (robust!)
    # - Eliminates temporal bias from Oct 10 outlier (V2 used 11.46% signal rate, V3 uses 5.46%)
    # - Streak factor 2.5× more important than expected (0.10 → 0.25)
    # - Out-of-sample test confirms: 28.66% return, 82.9% win rate, Sharpe 16.60
    BASE_POSITION_PCT = 0.65  # 65% base (V3 validated: best across all market regimes)
    MAX_POSITION_PCT = 0.95   # 95% maximum (V3 validated: optimal risk management)
    MIN_POSITION_PCT = 0.20   # 20% minimum (V3 validated: unchanged, optimal)
    SIGNAL_WEIGHT = 0.35      # 35% (V3 validated: avoid over-reliance on signal)
    VOLATILITY_WEIGHT = 0.25  # 25% (V3 validated: balanced volatility impact)
    REGIME_WEIGHT = 0.15      # 15% (V3 validated: moderate regime impact)
    STREAK_WEIGHT = 0.25      # 25% (V3 validated: manages consecutive losses effectively)

    # Risk Management
    MAX_DAILY_LOSS_PCT = 0.05
    TRANSACTION_COST = 0.0006

    # Data Collection
    LOOKBACK_CANDLES = 1440  # CRITICAL: Match backtest window size (72 hours = 5 days)
    # Previously 500 (42h), but backtests used 1440 (72h) windows
    # Model learned patterns from 72h context, so must use 72h in production
    UPDATE_INTERVAL = 300  # 5 minutes

    # Market Regime Classification
    BULL_THRESHOLD = 3.0
    BEAR_THRESHOLD = -2.0

    # Order Configuration
    ORDER_TYPE = "MARKET"  # MARKET or LIMIT
    LEVERAGE = 4  # Optimized from backtest: Dynamic @ 4x = 12.06% per 5 days (vs 7.68% baseline)

# ============================================================================
# Phase 4 Dynamic Testnet Trading Bot
# ============================================================================

class Phase4DynamicTestnetTradingBot:
    """
    Phase 4 Dynamic Position Sizing TESTNET Trading Bot

    실제 BingX Testnet API를 사용하여 주문 실행 및 포지션 관리
    """

    def __init__(self):
        # Initialize BingX Client
        if not Phase4TestnetConfig.API_KEY or not Phase4TestnetConfig.API_SECRET:
            raise ValueError("BingX API credentials not found! Set BINGX_API_KEY and BINGX_API_SECRET environment variables.")

        self.client = BingXClient(
            api_key=Phase4TestnetConfig.API_KEY,
            secret_key=Phase4TestnetConfig.API_SECRET,
            testnet=Phase4TestnetConfig.USE_TESTNET,
            timeout=30
        )
        logger.success("✅ BingX Testnet Client initialized")

        # Initialize Institutional Logger
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.inst_logger = InstitutionalLogger(
            log_dir=LOGS_DIR,
            strategy_name="Phase4_Dynamic_4Model",
            session_id=session_id,
            initial_capital=0.0,  # Will be set after balance query
            enable_json=True,
            enable_text=True,
            enable_audit=True
        )
        logger.success("✅ Institutional Logger initialized")
        # Note: log_system_event not available in InstitutionalLogger
        logger.info("Bot initialization: Phase 4 Dynamic (4-Model System)")

        # Test connection
        if not self.client.ping():
            raise ConnectionError("Failed to connect to BingX Testnet API")
        logger.success("✅ BingX Testnet API connection verified")

        # Set leverage for the trading pair (One-Way mode requires side="BOTH")
        try:
            self.client.set_leverage(
                symbol=Phase4TestnetConfig.SYMBOL,
                side="BOTH",  # One-Way mode requires "BOTH"
                leverage=Phase4TestnetConfig.LEVERAGE
            )
            logger.success(f"✅ Leverage set to {Phase4TestnetConfig.LEVERAGE}x for {Phase4TestnetConfig.SYMBOL}")
        except Exception as e:
            logger.warning(f"⚠️ Could not set leverage (may already be set): {e}")
            logger.info(f"   Continuing with default/existing leverage setting...")

        # Load DUAL MODELS: LONG + SHORT (with MinMaxScaler normalization)
        # LONG Model + Scaler
        long_model_path = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl"
        long_scaler_path = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_scaler.pkl"
        feature_path = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt"

        if not long_model_path.exists():
            raise FileNotFoundError(f"XGBoost LONG model not found: {long_model_path}")
        if not long_scaler_path.exists():
            raise FileNotFoundError(f"XGBoost LONG scaler not found: {long_scaler_path}")

        with open(long_model_path, 'rb') as f:
            self.long_model = pickle.load(f)

        with open(long_scaler_path, 'rb') as f:
            self.long_scaler = pickle.load(f)

        with open(feature_path, 'r') as f:
            self.feature_columns = [line.strip() for line in f.readlines()]

        logger.success(f"✅ XGBoost LONG model loaded: {len(self.feature_columns)} features")
        logger.success(f"✅ XGBoost LONG scaler loaded: MinMaxScaler(-1, 1)")

        # SHORT Entry Model + Scaler (RSI/MACD Enhanced - 2025-10-16)
        # RSI/MACD model: 22 SELL features, real TA-Lib calculation, threshold 0.70 optimal
        # Window backtest: 52.3% win rate, 3.18% return, 9.1 trades/window (balanced)
        short_model_path = MODELS_DIR / "xgboost_short_entry_enhanced_rsimacd_20251016_223048.pkl"
        short_scaler_path = MODELS_DIR / "xgboost_short_entry_enhanced_rsimacd_20251016_223048_scaler.pkl"
        short_features_path = MODELS_DIR / "xgboost_short_entry_enhanced_rsimacd_20251016_223048_features.txt"

        if not short_model_path.exists():
            raise FileNotFoundError(f"XGBoost SHORT Entry model not found: {short_model_path}")
        if not short_scaler_path.exists():
            raise FileNotFoundError(f"XGBoost SHORT Entry scaler not found: {short_scaler_path}")
        if not short_features_path.exists():
            raise FileNotFoundError(f"XGBoost SHORT Entry features not found: {short_features_path}")

        with open(short_model_path, 'rb') as f:
            self.short_model = pickle.load(f)

        with open(short_scaler_path, 'rb') as f:
            self.short_scaler = pickle.load(f)

        with open(short_features_path, 'r') as f:
            self.short_feature_columns = [line.strip() for line in f.readlines()]

        logger.success(f"✅ XGBoost SHORT Entry RSI/MACD model loaded: {len(self.short_feature_columns)} SELL features")
        logger.success(f"✅ XGBoost SHORT Entry scaler loaded: MinMaxScaler(-1, 1)")
        logger.info(f"📊 SHORT Entry Strategy: 22 SELL features (RSI/MACD), 52.3% win rate, threshold=0.70")
        logger.info(f"📊 Dual Model Strategy: LONG + SHORT (balanced, independent predictions, normalized features)")

        # EXIT Models (LONG + SHORT - ENHANCED with 22 features, 2025-10-16)
        # Enhanced models use proper 2of3 labeling + market context features
        long_exit_model_path = MODELS_DIR / "xgboost_long_exit_improved_20251016_175554.pkl"
        long_exit_scaler_path = MODELS_DIR / "xgboost_long_exit_improved_20251016_175554_scaler.pkl"
        long_exit_features_path = MODELS_DIR / "xgboost_long_exit_improved_20251016_175554_features.txt"
        short_exit_model_path = MODELS_DIR / "xgboost_short_exit_improved_20251016_180207.pkl"
        short_exit_scaler_path = MODELS_DIR / "xgboost_short_exit_improved_20251016_180207_scaler.pkl"
        short_exit_features_path = MODELS_DIR / "xgboost_short_exit_improved_20251016_180207_features.txt"

        if not long_exit_model_path.exists():
            raise FileNotFoundError(f"XGBoost LONG EXIT model not found: {long_exit_model_path}")
        if not long_exit_scaler_path.exists():
            raise FileNotFoundError(f"XGBoost LONG EXIT scaler not found: {long_exit_scaler_path}")
        if not short_exit_model_path.exists():
            raise FileNotFoundError(f"XGBoost SHORT EXIT model not found: {short_exit_model_path}")
        if not short_exit_scaler_path.exists():
            raise FileNotFoundError(f"XGBoost SHORT EXIT scaler not found: {short_exit_scaler_path}")

        with open(long_exit_model_path, 'rb') as f:
            self.long_exit_model = pickle.load(f)

        with open(long_exit_scaler_path, 'rb') as f:
            self.long_exit_scaler = pickle.load(f)

        with open(long_exit_features_path, 'r') as f:
            self.long_exit_features = [line.strip() for line in f.readlines()]

        with open(short_exit_model_path, 'rb') as f:
            self.short_exit_model = pickle.load(f)

        with open(short_exit_scaler_path, 'rb') as f:
            self.short_exit_scaler = pickle.load(f)

        with open(short_exit_features_path, 'r') as f:
            self.short_exit_features = [line.strip() for line in f.readlines()]

        logger.success(f"✅ XGBoost LONG EXIT model loaded ({len(self.long_exit_features)} ENHANCED features: 22 market context)")
        logger.success(f"✅ XGBoost LONG EXIT scaler loaded: MinMaxScaler(-1, 1)")
        logger.success(f"✅ XGBoost SHORT EXIT model loaded ({len(self.short_exit_features)} ENHANCED features: 22 market context)")
        logger.success(f"✅ XGBoost SHORT EXIT scaler loaded: MinMaxScaler(-1, 1)")
        logger.info(f"📊 Exit Strategy: ENHANCED ML with NORMAL logic (threshold={Phase4TestnetConfig.EXIT_THRESHOLD}, HIGH prob = good exit)")
        logger.info(f"   ✅ NORMAL LOGIC: Exit when prob >= {Phase4TestnetConfig.EXIT_THRESHOLD} (2of3 scoring + feature engineering)")
        logger.info(f"   📈 Expected: +14.44% return, 67.4% win rate (enhanced with 22 features)")

        # Initialize Advanced Technical Features
        self.adv_features = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)
        logger.success("✅ Advanced Technical Features initialized")

        # Initialize Dynamic Position Sizer
        self.position_sizer = DynamicPositionSizer(
            base_position_pct=Phase4TestnetConfig.BASE_POSITION_PCT,
            max_position_pct=Phase4TestnetConfig.MAX_POSITION_PCT,
            min_position_pct=Phase4TestnetConfig.MIN_POSITION_PCT,
            signal_weight=Phase4TestnetConfig.SIGNAL_WEIGHT,
            volatility_weight=Phase4TestnetConfig.VOLATILITY_WEIGHT,
            regime_weight=Phase4TestnetConfig.REGIME_WEIGHT,
            streak_weight=Phase4TestnetConfig.STREAK_WEIGHT
        )
        logger.success("✅ Dynamic Position Sizer initialized")
        logger.info(f"   Position Range: {Phase4TestnetConfig.MIN_POSITION_PCT*100:.0f}% - {Phase4TestnetConfig.MAX_POSITION_PCT*100:.0f}%")
        logger.info(f"   Base Position: {Phase4TestnetConfig.BASE_POSITION_PCT*100:.0f}%")

        # Balance caching (reduce API calls and rate limiting)
        self._cached_balance = None
        self._balance_cache_time = None
        self._balance_cache_ttl = 240.0  # 4 minutes TTL (covers full 5-min cycle)

        # Rate limit tracking
        self._rate_limit_hit_count = 0
        self._last_rate_limit_time = None

        # Get initial balance with retry logic for initialization
        logger.info("📊 Getting initial account balance...")
        max_init_attempts = 3
        for init_attempt in range(max_init_attempts):
            try:
                self.initial_balance = self._get_account_balance(use_cache=False)
                logger.success(f"✅ Testnet Account Balance: ${self.initial_balance:,.2f} USDT")

                # Update institutional logger with initial capital
                self.inst_logger.initial_capital = self.initial_balance
                # Note: log_system_event not available in InstitutionalLogger
                logger.info(f"Initial balance set: ${self.initial_balance:,.2f} USDT")
                break
            except Exception as e:
                if init_attempt < max_init_attempts - 1:
                    wait_time = 120 * (init_attempt + 1)  # 2 min, 4 min
                    logger.warning(f"⚠️ Failed to get initial balance (attempt {init_attempt+1}/{max_init_attempts})")
                    logger.warning(f"   Error: {e}")
                    logger.info(f"   Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"❌ Failed to get initial balance after {max_init_attempts} attempts")
                    logger.error(f"   This usually indicates API rate limiting or connectivity issues")
                    logger.error(f"   Please wait 5-10 minutes and try again")
                    # Note: log_error signature: (error_type, error_message, stack_trace)
                    import traceback
                    self.inst_logger.log_error(
                        error_type="INITIALIZATION_ERROR",
                        error_message=f"Failed to retrieve initial balance: {str(e)}",
                        stack_trace=traceback.format_exc()
                    )
                    raise

        # Trading state
        self.trades = []
        self.session_start = datetime.now()
        self.market_regime_history = []

        # Try to load previous session state (for session continuity)
        self._load_previous_state()

        # Signal tracking (for opportunity cost analysis)
        self.signal_log = []

        # Real-time signal tracking (for state file and monitoring)
        # Entry signals (updated every cycle)
        self.latest_long_entry_prob = 0.0
        self.latest_short_entry_prob = 0.0
        self.latest_long_entry_threshold = Phase4TestnetConfig.BASE_LONG_ENTRY_THRESHOLD
        self.latest_short_entry_threshold = Phase4TestnetConfig.BASE_SHORT_ENTRY_THRESHOLD
        self.latest_entry_timestamp = None
        # Dynamic threshold context (V3: ACTUAL ENTRY RATE)
        self.latest_threshold_entry_rate = None  # Actual entry rate (entries/candles)
        self.latest_threshold_entries_count = None  # Number of entries in lookback
        self.latest_threshold_adjustment = 0.0  # Threshold delta
        self.latest_threshold_adjustment_ratio = None  # Ratio vs target

        # Exit signals (updated only when position exists)
        self.latest_long_exit_prob = None  # None = no LONG position
        self.latest_short_exit_prob = None  # None = no SHORT position
        self.latest_exit_threshold = Phase4TestnetConfig.EXIT_THRESHOLD
        self.latest_exit_timestamp = None
        self.latest_exit_position_side = None  # "LONG" or "SHORT"

        # Buy & Hold tracking
        self.bh_btc_quantity = 0.0
        self.bh_entry_price = 0.0
        self.bh_initialized = False

        logger.info("=" * 80)
        logger.info("Phase 4 Dual Entry + Dual Exit Model Testnet Trading Bot Initialized")
        logger.info("=" * 80)
        logger.info(f"Network: BingX TESTNET ✅ (Real Order Execution!)")
        logger.info(f"Entry Strategy: Dual Model (LONG + SHORT independent predictions)")
        logger.info(f"Exit Strategy: Dual ML Exit Model INVERTED @ {Phase4TestnetConfig.EXIT_THRESHOLD:.2f}")
        logger.info(f"  ⚠️ INVERTED LOGIC: Exit when prob <= {Phase4TestnetConfig.EXIT_THRESHOLD:.2f} (models learned opposite)")
        logger.info(f"  📊 Validation: +7.55% improvement over normal logic (21 windows tested)")
        logger.info(f"Initial Balance: ${self.initial_balance:,.2f} USDT")
        logger.info(f"")
        logger.info(f"Expected Performance (2025-10-15: V3 FULL-DATASET OPTIMIZATION - Temporal Bias Eliminated):")
        logger.info(f"  - Returns: +{Phase4TestnetConfig.EXPECTED_RETURN_PER_WEEK:.2f}% per week (V3 out-of-sample validated)")
        logger.info(f"  - Win Rate: {Phase4TestnetConfig.EXPECTED_WIN_RATE:.1f}% (validated on 3 months data)")
        logger.info(f"  - Avg Holding: {Phase4TestnetConfig.EXPECTED_AVG_HOLDING:.2f} hours")
        logger.info(f"  - Trades/Week: {Phase4TestnetConfig.EXPECTED_TRADES_PER_WEEK:.1f} (robust across market regimes)")
        logger.info(f"  - Avg Position: {Phase4TestnetConfig.EXPECTED_AVG_POSITION:.1f}%")
        logger.info(f"  - Max Drawdown: {Phase4TestnetConfig.EXPECTED_MAX_DRAWDOWN:.2f}%")
        logger.info(f"  - Sharpe Ratio: {Phase4TestnetConfig.EXPECTED_SHARPE_RATIO:.2f} (out-of-sample)")
        logger.info(f"  - Dataset: 3-month validation (eliminates Oct 10 outlier bias)")
        logger.info(f"")
        logger.info(f"Entry Thresholds (Dynamic System 2025-10-15):")
        logger.info(f"  - LONG Base: {Phase4TestnetConfig.BASE_LONG_ENTRY_THRESHOLD:.2f} (adaptive range: {Phase4TestnetConfig.MIN_THRESHOLD:.2f}-{Phase4TestnetConfig.MAX_THRESHOLD:.2f})")
        logger.info(f"  - SHORT Base: {Phase4TestnetConfig.BASE_SHORT_ENTRY_THRESHOLD:.2f} (adaptive range: {Phase4TestnetConfig.MIN_THRESHOLD:.2f}-{Phase4TestnetConfig.MAX_THRESHOLD:.2f})")
        logger.info(f"  - Dynamic Adjustment: {'ENABLED' if Phase4TestnetConfig.ENABLE_DYNAMIC_THRESHOLD else 'DISABLED'} (lookback: {Phase4TestnetConfig.DYNAMIC_LOOKBACK_HOURS}h)")
        logger.info(f"")
        logger.info(f"Exit Parameters (Optimized 2025-10-15):")
        logger.info(f"  - ML Exit Threshold: {Phase4TestnetConfig.EXIT_THRESHOLD:.2f}")
        logger.info(f"  - Stop Loss: {Phase4TestnetConfig.STOP_LOSS*100:.1f}%")
        logger.info(f"  - Take Profit: {Phase4TestnetConfig.TAKE_PROFIT*100:.1f}%")
        logger.info(f"  - Max Holding: {Phase4TestnetConfig.MAX_HOLDING_HOURS}h")
        logger.info(f"")
        logger.info(f"Position Sizing (V3 Optimized 2025-10-15 - Full-Dataset Validated):")
        logger.info(f"  - Base: {Phase4TestnetConfig.BASE_POSITION_PCT*100:.0f}% | Max: {Phase4TestnetConfig.MAX_POSITION_PCT*100:.0f}% | Min: {Phase4TestnetConfig.MIN_POSITION_PCT*100:.0f}%")
        logger.info(f"  - Weights: Signal={Phase4TestnetConfig.SIGNAL_WEIGHT:.2f} | Vol={Phase4TestnetConfig.VOLATILITY_WEIGHT:.2f} | Regime={Phase4TestnetConfig.REGIME_WEIGHT:.2f} | Streak={Phase4TestnetConfig.STREAK_WEIGHT:.2f}")
        logger.info(f"  - V3 Validation: Same as V2 but tested on 3-month dataset (6x more data)")
        logger.info(f"  - Key Insight: Streak factor 2.5× more important, manages consecutive losses!")
        logger.info("=" * 80)

    def _load_previous_state(self):
        """Load previous session state for session continuity"""
        state_file = RESULTS_DIR / "phase4_testnet_trading_state.json"

        if not state_file.exists():
            logger.info("🆕 Starting fresh session (no previous state)")
            return

        try:
            with open(state_file, 'r') as f:
                prev_state = json.load(f)

            # Check if continuing same session (within 30 minutes)
            prev_session_start_str = prev_state.get('session_start')
            if not prev_session_start_str:
                logger.info("🆕 Starting fresh session (no session_start in previous state)")
                return

            prev_session_start = datetime.fromisoformat(prev_session_start_str)
            time_since_prev = (datetime.now() - prev_session_start).total_seconds()

            # Get previous session's initial balance
            prev_initial_balance = prev_state.get('initial_balance')
            current_balance_at_restart = self.initial_balance  # Current balance (already set in __init__)

            # If within 4 hours (matches Max Holding time), consider it the same session
            if time_since_prev < 14400:  # 4 hours = 14400 seconds
                time_str = f"{time_since_prev/60:.1f} minutes" if time_since_prev < 3600 else f"{time_since_prev/3600:.1f} hours"

                # ✅ FIX: Log restart details for debugging
                logger.info("=" * 80)
                logger.success(f"🔄 BOT RESTART DETECTED")
                logger.info(f"   Previous session started: {time_str} ago")
                logger.info(f"   Previous initial balance: ${prev_initial_balance:,.2f} USDT")
                logger.info(f"   Current balance: ${current_balance_at_restart:,.2f} USDT")

                # Calculate P&L since previous session start
                if prev_initial_balance:
                    session_pnl = current_balance_at_restart - prev_initial_balance
                    session_pnl_pct = (session_pnl / prev_initial_balance) * 100
                    logger.info(f"   Session P&L: ${session_pnl:+,.2f} ({session_pnl_pct:+.2f}%)")

                logger.info("=" * 80)

                # ✅ FIX: Restore original initial_balance (not current balance!)
                if prev_initial_balance is not None:
                    self.initial_balance = prev_initial_balance
                    logger.success(f"✅ Restored original initial balance: ${self.initial_balance:,.2f} USDT")
                else:
                    logger.warning("⚠️ No initial_balance in state file, using current balance")
                    logger.warning(f"   This will cause ROI calculation inaccuracy!")

                # Restore trades list with datetime deserialization
                if 'trades' in prev_state and isinstance(prev_state['trades'], list):
                    self.trades = []
                    for trade_data in prev_state['trades']:
                        # Deserialize datetime fields
                        if 'entry_time' in trade_data and isinstance(trade_data['entry_time'], str):
                            trade_data['entry_time'] = datetime.fromisoformat(trade_data['entry_time'])
                        if 'exit_time' in trade_data and isinstance(trade_data['exit_time'], str):
                            trade_data['exit_time'] = datetime.fromisoformat(trade_data['exit_time'])

                        # ✅ FIX: Convert probability string to float (fixes numpy type error)
                        if 'probability' in trade_data and isinstance(trade_data['probability'], str):
                            trade_data['probability'] = float(trade_data['probability'])

                        # ✅ FIX: Convert sizing_factors strings to float
                        if 'sizing_factors' in trade_data and isinstance(trade_data['sizing_factors'], dict):
                            for key, value in trade_data['sizing_factors'].items():
                                if isinstance(value, str):
                                    trade_data['sizing_factors'][key] = float(value)

                        self.trades.append(trade_data)

                    open_trades = [t for t in self.trades if t.get('status') == 'OPEN']
                    closed_trades = [t for t in self.trades if t.get('status') == 'CLOSED']
                    logger.info(f"   Restored {len(self.trades)} trades ({len(open_trades)} open, {len(closed_trades)} closed)")

                    # Log open trade details
                    for trade in open_trades:
                        entry_time = trade.get('entry_time')
                        hours_held = (datetime.now() - entry_time).total_seconds() / 3600 if entry_time else 0
                        logger.info(f"   Open Position: {trade.get('quantity', 0):.4f} BTC @ ${trade.get('entry_price', 0):,.2f} ({hours_held:.1f}h held)")
                else:
                    logger.info("   No trades to restore")

                # Update session_start to maintain original session time
                self.session_start = prev_session_start

            else:
                logger.info(f"🆕 Starting fresh session (previous session {time_since_prev/3600:.1f} hours ago)")

        except Exception as e:
            logger.error(f"Failed to load previous state: {e}")
            logger.info("🆕 Starting fresh session")

    def _is_rate_limit_error(self, error) -> bool:
        """Check if error is a rate limit error"""
        error_str = str(error).lower()
        return ('100410' in error_str or
                'rate limit' in error_str or
                'frequency limit' in error_str or
                'please try again later' in error_str)

    def _handle_rate_limit_error(self, error):
        """
        Handle rate limit error with appropriate wait time

        BingX rate limit errors include unblock timestamp in the response.
        We extract this and wait appropriately.
        """
        self._rate_limit_hit_count += 1
        self._last_rate_limit_time = datetime.now()

        # Try to extract unblock timestamp from error message
        # Format: "unblocked after 1760369709268"
        error_str = str(error)
        wait_time = 300  # Default 5 minutes

        try:
            if 'unblocked after' in error_str:
                import re
                match = re.search(r'unblocked after (\d+)', error_str)
                if match:
                    unblock_timestamp_ms = int(match.group(1))
                    unblock_time = datetime.fromtimestamp(unblock_timestamp_ms / 1000)
                    wait_time = max(0, (unblock_time - datetime.now()).total_seconds())
                    wait_time = min(wait_time, 600)  # Cap at 10 minutes
        except Exception:
            pass  # Use default wait time

        logger.error("=" * 80)
        logger.error("🚨 BingX API RATE LIMIT EXCEEDED!")
        logger.error("=" * 80)
        logger.error(f"Rate limit hit #{self._rate_limit_hit_count} this session")
        logger.error(f"Error: {error}")
        logger.error(f"")
        logger.error(f"Required Action:")
        logger.error(f"  ⏳ Waiting {wait_time:.0f} seconds ({wait_time/60:.1f} minutes) for rate limit to reset...")
        logger.error(f"")
        logger.error(f"Root Cause:")
        logger.error(f"  - Too many API calls in short time period")
        logger.error(f"  - BingX enforces strict rate limits on testnet")
        logger.error(f"")
        logger.error(f"Prevention:")
        logger.error(f"  ✅ Balance cache TTL increased to 4 minutes")
        logger.error(f"  ✅ Retry logic reduced from 3 to 2 attempts")
        logger.error(f"  ✅ Longer wait times between retries")
        logger.error("=" * 80)

        # Log rate limit error to institutional logger
        self.inst_logger.log_compliance_event(
            event_type="API_RATE_LIMIT",
            description=f"BingX API rate limit exceeded (hit #{self._rate_limit_hit_count})",
            severity="HIGH",
            action_taken=f"Waiting {wait_time:.0f}s for rate limit reset"
        )
        # Note: log_error signature: (error_type, error_message, stack_trace)
        self.inst_logger.log_error(
            error_type="API_RATE_LIMIT",
            error_message=f"BingX API rate limit exceeded (hit #{self._rate_limit_hit_count}): {str(error)}"
        )

        # Wait for rate limit to reset
        time.sleep(wait_time)

        logger.success(f"✅ Rate limit wait completed, resuming operations...")

    def _get_account_balance(self, retry_count: int = 2, retry_delay: float = 60.0, use_cache: bool = True) -> float:
        """
        Get USDT balance from account with retry logic and caching

        Args:
            retry_count: Number of retry attempts (default: 2, reduced from 3)
            retry_delay: Initial delay between retries in seconds (default: 60s, increased from 2s)
            use_cache: Whether to use cached balance if available (default: True)

        Returns:
            USDT balance

        Raises:
            Exception: If all retry attempts fail and no cache available
        """
        # Check cache first (if enabled and valid)
        if use_cache and self._cached_balance is not None and self._balance_cache_time is not None:
            cache_age = (datetime.now() - self._balance_cache_time).total_seconds()
            if cache_age < self._balance_cache_ttl:
                logger.debug(f"Using cached balance (age: {cache_age:.1f}s / {self._balance_cache_ttl:.0f}s TTL)")
                return self._cached_balance

        last_error = None

        for attempt in range(retry_count):
            try:
                balance_data = self.client.get_balance()

                # BingX balance response format
                if isinstance(balance_data, dict):
                    balance = balance_data.get('balance', {})
                    usdt_balance = float(balance.get('balance', 0.0))

                    # Validate balance (must be > 0 for trading)
                    if usdt_balance > 0:
                        # Update cache on successful query
                        self._cached_balance = usdt_balance
                        self._balance_cache_time = datetime.now()
                        return usdt_balance
                    else:
                        logger.warning(f"Balance returned 0 (attempt {attempt+1}/{retry_count})")

                elif isinstance(balance_data, list):
                    for item in balance_data:
                        if item.get('asset') == 'USDT':
                            usdt_balance = float(item.get('balance', 0.0))
                            if usdt_balance > 0:
                                # Update cache on successful query
                                self._cached_balance = usdt_balance
                                self._balance_cache_time = datetime.now()
                                return usdt_balance
                            else:
                                logger.warning(f"Balance returned 0 (attempt {attempt+1}/{retry_count})")

                else:
                    logger.warning(f"Could not parse balance response (attempt {attempt+1}/{retry_count})")

            except Exception as e:
                last_error = e

                # Check if it's a rate limit error
                if self._is_rate_limit_error(e):
                    logger.warning(f"Rate limit error on attempt {attempt+1}/{retry_count}")
                    self._handle_rate_limit_error(e)
                    # After handling rate limit, continue to next attempt
                    continue
                else:
                    logger.warning(f"Balance query failed (attempt {attempt+1}/{retry_count}): {e}")

            # Wait before retry (exponential backoff for non-rate-limit errors)
            if attempt < retry_count - 1:
                wait_time = retry_delay * (2 ** attempt)
                logger.info(f"Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)

        # All retries failed - try to return cached balance if available
        if self._cached_balance is not None:
            cache_age = (datetime.now() - self._balance_cache_time).total_seconds()
            logger.warning(f"⚠️ Using cached balance after {retry_count} failed attempts (age: {cache_age:.1f}s)")
            logger.warning(f"   Cached value may be stale, but continuing to prevent bot crash")
            return self._cached_balance

        # No cache available - raise exception
        error_msg = f"Failed to get valid balance after {retry_count} attempts"
        if last_error:
            error_msg += f": {last_error}"
        logger.error(error_msg)
        raise Exception(error_msg)

    def _get_current_position(self) -> dict:
        """Get current open position"""
        try:
            positions = self.client.get_positions(Phase4TestnetConfig.SYMBOL)

            for pos in positions:
                position_amt = float(pos.get('positionAmt', 0))
                if position_amt != 0:  # Has open position
                    return {
                        'symbol': pos.get('symbol'),
                        'position_side': pos.get('positionSide'),
                        'position_amt': abs(position_amt),
                        'entry_price': float(pos.get('entryPrice', 0)),
                        'unrealized_pnl': float(pos.get('unrealizedProfit', 0)),
                        'leverage': int(float(pos.get('leverage', 1)))
                    }

            return None  # No position

        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return None

    def _get_actual_exit_from_history(self, trade: dict) -> Optional[dict]:
        """
        Get actual exit price and time from exchange order history

        CRITICAL: Matches by MULTIPLE criteria to prevent wrong trade matching:
        1. Side (SELL for LONG, BUY for SHORT)
        2. Quantity (±0.01 BTC tolerance)
        3. Time sequence (exit AFTER entry)
        4. Entry price proximity (optional validation)

        Args:
            trade: Trade dict with entry_time, entry_price, quantity, side

        Returns:
            Dict with exit_price, exit_time, order_id if found, None otherwise
        """
        try:
            # Get closed orders from last 24 hours
            since = int((datetime.now() - timedelta(hours=24)).timestamp() * 1000)

            closed_orders = self.client.exchange.fetch_closed_orders(
                symbol='BTC/USDT:USDT',
                since=since,
                limit=100
            )

            # Parse trade info
            trade_side = trade.get('side', 'LONG')
            expected_side = 'sell' if trade_side == 'LONG' else 'buy'
            quantity = trade.get('quantity', 0)
            entry_price = trade.get('entry_price', 0)

            # Parse entry time
            entry_time_str = trade.get('entry_time', '')
            try:
                entry_time = datetime.fromisoformat(entry_time_str)
            except:
                logger.warning(f"   ⚠️ Could not parse entry_time: {entry_time_str}")
                entry_time = None

            logger.info(f"   🔍 Searching for exit: {trade_side} {quantity} BTC @ ${entry_price:,.2f}")
            if entry_time:
                logger.info(f"      Entry time: {entry_time.strftime('%Y-%m-%d %H:%M:%S')}")

            # Find matching exit order with STRICT validation
            candidates = []
            for order in closed_orders:
                # Check 1: Side match
                if order['side'] != expected_side:
                    continue

                # Check 2: Quantity match (±0.01 tolerance)
                qty_diff = abs(order['amount'] - quantity)
                if qty_diff >= 0.01:
                    continue

                # Check 3: Time sequence (exit AFTER entry)
                exit_time = datetime.fromtimestamp(order['timestamp'] / 1000)
                if entry_time and exit_time <= entry_time:
                    logger.debug(f"      ⏭️ Skip: Exit {exit_time} before entry {entry_time}")
                    continue

                # Found valid candidate
                exit_price = order.get('price', order.get('average'))
                order_id = order.get('id')

                candidates.append({
                    'exit_time': exit_time,
                    'exit_price': float(exit_price),
                    'order_id': str(order_id),
                    'qty_diff': qty_diff
                })

            if not candidates:
                logger.warning(f"   ⚠️ No matching exit order found in exchange history")
                return None

            # If multiple candidates, pick the earliest one (closest to entry)
            best_match = min(candidates, key=lambda x: x['exit_time'])

            logger.info(f"   ✅ Found actual exit in exchange history:")
            logger.info(f"      Time: {best_match['exit_time'].strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"      Price: ${best_match['exit_price']:,.2f}")
            logger.info(f"      Order ID: {best_match['order_id']}")
            logger.info(f"      Qty diff: {best_match['qty_diff']:.4f} BTC")

            return {
                'exit_price': best_match['exit_price'],
                'exit_time': best_match['exit_time'].isoformat(),
                'order_id': best_match['order_id']
            }

        except Exception as e:
            logger.error(f"   ❌ Failed to fetch exchange history: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run(self):
        """Main trading loop"""
        logger.info("🚀 Starting Phase 4 Dynamic Testnet Trading...")
        logger.info(f"⚠️ WARNING: This bot will execute REAL orders on Testnet!")
        logger.info(f"Update: Synchronized to 5-minute candle completion")

        try:
            while True:
                self._update_cycle()

                # Sleep until next 5-minute candle completes
                now = datetime.now()
                current_minute = now.minute
                current_second = now.second

                # Next candle completion time (00, 05, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55)
                next_candle_minute = ((current_minute // 5) + 1) * 5
                if next_candle_minute >= 60:
                    next_candle_minute = 0

                # Calculate seconds to wait
                minutes_to_wait = (next_candle_minute - current_minute) % 60
                seconds_to_wait = minutes_to_wait * 60 - current_second + 5  # +5 sec buffer after candle completes

                if seconds_to_wait <= 0:
                    seconds_to_wait += 300  # Wait for next candle if we just missed one

                logger.info(f"⏳ Next update in {seconds_to_wait}s (at :{next_candle_minute:02d}:05)")
                time.sleep(seconds_to_wait)

        except KeyboardInterrupt:
            logger.info("\n" + "=" * 80)
            logger.info("Bot stopped by user")
            self._print_final_stats()

    def _update_cycle(self):
        """Single update cycle (every 5 minutes)"""
        try:
            # Get market data
            df = self._get_market_data()
            if df is None or len(df) < Phase4TestnetConfig.LOOKBACK_CANDLES:
                logger.warning("Insufficient market data")
                return

            # Calculate features
            df = calculate_features(df)
            df = self.adv_features.calculate_all_features(df)

            # Calculate RSI/MACD features (2025-10-16: Required for Enhanced SHORT Entry)
            df = self._calculate_rsi_macd_features(df)

            # Calculate ENHANCED EXIT features (2025-10-16: 22 market context features)
            df = self._calculate_enhanced_exit_features(df)

            # Handle NaN values (from Support/Resistance lookback warmup)
            rows_before = len(df)

            # Identify NaN columns before handling (for informative logging)
            nan_counts = df.isna().sum()
            nan_columns = nan_counts[nan_counts > 0]

            df = df.ffill()
            df = df.dropna()
            rows_after = len(df)
            rows_lost = rows_before - rows_after

            # Expected loss from S/R lookback (50 candles)
            expected_loss = 50  # lookback_sr parameter

            if rows_lost <= expected_loss + 10:  # Normal range (+10 tolerance)
                logger.info(f"✅ Data ready: {rows_after} rows (warmup removed {rows_lost} rows)")
                logger.debug(f"   Expected warmup loss: ~{expected_loss} rows (S/R lookback)")
                if len(nan_columns) > 0:
                    top_nan_cols = nan_columns.nlargest(3)
                    logger.debug(f"   NaN sources: {', '.join(top_nan_cols.index[:3])} (normal)")
            else:
                logger.warning(f"⚠️ Unexpected data loss: {rows_lost} rows (expected ~{expected_loss})")
                logger.warning(f"   This may indicate a data quality issue")
                if len(nan_columns) > 5:
                    logger.warning(f"   {len(nan_columns)} columns have NaN (check feature calculation)")

            if len(df) < 50:
                logger.error(f"❌ Insufficient data after NaN handling ({len(df)} < 50)")
                return

            # Current state
            # ✅ FIX: Use previous COMPLETE candle (BingX API returns forming candle as last row!)
            # Root Cause: API returns incomplete candle at iloc[-1], causing 75-91% lower probabilities
            # Solution: Use iloc[-2] for complete candle that matches backtest methodology
            current_idx = len(df) - 2  # Previous complete candle (not forming candle)
            current_price = df['close'].iloc[current_idx]

            # Get account balance
            current_balance = self._get_account_balance()

            # Initialize Buy & Hold if first run
            if not self.bh_initialized:
                self._initialize_buy_hold(current_price)

            # Classify market regime
            current_regime = self._classify_market_regime(df)
            self.market_regime_history.append({
                "timestamp": datetime.now(),
                "regime": current_regime,
                "price": current_price
            })

            # Log market data to institutional logger
            self.inst_logger.log_market_data(
                price=current_price,
                regime=current_regime,
                volume=float(df['volume'].iloc[-1]) if 'volume' in df.columns else None,
                volatility=float(df['atr_pct'].iloc[-1]) if 'atr_pct' in df.columns else None
            )

            logger.info(f"\n{'=' * 80}")
            logger.info(f"Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"Market Regime: {current_regime}")
            logger.info(f"Current Price: ${current_price:,.2f}")
            logger.info(f"Account Balance: ${current_balance:,.2f} USDT")

            # Check current position status from exchange
            current_position = self._get_current_position()

            # Manage existing position
            if current_position is not None:
                # Find matching trade in state for this API position
                # Match by quantity (within 0.01 tolerance) since entry_price may differ slightly
                matching_trade = None
                for t in self.trades:
                    if t.get('status') == 'OPEN' and abs(t.get('quantity', 0) - current_position['position_amt']) < 0.01:
                        matching_trade = t
                        break

                # Check for orphaned position (API has position but state doesn't)
                if matching_trade is None:
                    logger.warning("⚠️ ORPHANED POSITION DETECTED!")
                    logger.warning(f"   Position: {current_position['position_side']} {current_position['position_amt']:.4f} BTC @ ${current_position['entry_price']:,.2f}")
                    logger.warning(f"   Unrealized P&L: ${current_position['unrealized_pnl']:+,.2f}")
                    logger.warning(f"   Trades in state: {len(self.trades)} (0 OPEN)")
                    logger.warning(f"   Possible causes: Bot crash, manual trade, or state file corruption")
                    logger.warning(f"   Creating trade record with Max Holding time trigger...")

                    # Create trade record with entry_time set to 4 hours ago
                    # This will trigger Max Holding exit immediately
                    orphaned_entry_time = datetime.now() - timedelta(hours=Phase4TestnetConfig.MAX_HOLDING_HOURS)

                    orphaned_trade = {
                        'entry_time': orphaned_entry_time,
                        'order_id': 'ORPHANED',
                        'side': current_position['position_side'],  # ✅ Include position side
                        'entry_price': current_position['entry_price'],
                        'quantity': current_position['position_amt'],
                        'position_size_pct': 0.50,  # Assume 50%
                        'position_value': current_position['entry_price'] * current_position['position_amt'],
                        'regime': 'Unknown',
                        'probability': 0.0,
                        'sizing_factors': {},
                        'status': 'OPEN'
                    }
                    self.trades.append(orphaned_trade)
                    logger.info(f"   ✅ Trade record created - will exit via Max Holding")

                # REVERSE ORPHAN CHECK: Close any other OPEN trades that don't match API position
                # This handles case where state has multiple OPEN but API has only one
                other_open_trades = [t for t in self.trades if t.get('status') == 'OPEN' and t != matching_trade]
                if other_open_trades:
                    logger.warning(f"⚠️ REVERSE ORPHAN DETECTED (partial)!")
                    logger.warning(f"   Found {len(other_open_trades)} OPEN trade(s) in state but not on exchange")
                    logger.warning(f"   Auto-closing orphaned trades...")

                    for trade in other_open_trades:
                        # Handle both string and datetime objects
                        entry_time_raw = trade['entry_time']
                        entry_time = datetime.fromisoformat(entry_time_raw) if isinstance(entry_time_raw, str) else entry_time_raw
                        entry_price = trade['entry_price']
                        quantity = trade['quantity']

                        # Try to get actual exit from exchange history
                        logger.info(f"   🔍 Searching exchange history for actual exit...")
                        actual_exit = self._get_actual_exit_from_history(trade)

                        if actual_exit:
                            # Use actual exit from exchange
                            exit_price = actual_exit['exit_price']
                            exit_time_str = actual_exit['exit_time']
                            close_order_id = actual_exit['order_id']
                        else:
                            # Fallback: Use current price
                            logger.warning(f"   ⚠️ Using current market price as fallback")
                            exit_price = current_price
                            exit_time_str = datetime.now().isoformat()
                            close_order_id = None

                        side = trade.get('side', 'LONG')
                        if side == "LONG":
                            pnl_pct = (exit_price - entry_price) / entry_price
                        else:
                            pnl_pct = (entry_price - exit_price) / entry_price

                        position_value = quantity * entry_price
                        pnl_usd_gross = quantity * (exit_price - entry_price) if side == "LONG" else quantity * (entry_price - exit_price)
                        entry_cost = entry_price * quantity * Phase4TestnetConfig.TRANSACTION_COST
                        exit_cost = exit_price * quantity * Phase4TestnetConfig.TRANSACTION_COST
                        transaction_cost = entry_cost + exit_cost
                        pnl_usd_net = pnl_usd_gross - transaction_cost

                        trade['status'] = 'CLOSED'
                        trade['exit_time'] = exit_time_str
                        trade['exit_price'] = exit_price
                        trade['exit_reason'] = "Exchange closed (verified from history)" if actual_exit else "Auto-closed (position not found - reverse orphan)"
                        trade['close_order_id'] = close_order_id
                        trade['pnl_pct'] = pnl_pct
                        trade['pnl_usd_gross'] = pnl_usd_gross
                        trade['transaction_cost'] = transaction_cost
                        trade['pnl_usd_net'] = pnl_usd_net

                        logger.warning(f"   ✅ Auto-closed: {side} {quantity:.4f} BTC @ ${entry_price:,.2f} → ${exit_price:,.2f}")
                        logger.warning(f"      P&L: {pnl_pct*100:+.2f}% (${pnl_usd_net:+,.2f})")

                    logger.success(f"   ✅ Partial reverse orphan: {len(other_open_trades)} trade(s) auto-closed")

                self._manage_position(current_price, df, current_idx, current_position)
            else:
                # REVERSE ORPHAN DETECTION: state.json has OPEN but API doesn't
                # Automatically close any orphaned OPEN trades in state
                open_trades = [t for t in self.trades if t.get('status') == 'OPEN']
                if open_trades:
                    logger.warning("⚠️ REVERSE ORPHAN DETECTED!")
                    logger.warning(f"   Found {len(open_trades)} OPEN trade(s) in state.json but NO position on exchange")
                    logger.warning(f"   Possible causes: Position auto-closed by exchange, liquidation, or manual closure")
                    logger.warning(f"   Auto-closing orphaned trades...")

                    for trade in open_trades:
                        # Handle both string and datetime objects
                        entry_time_raw = trade['entry_time']
                        entry_time = datetime.fromisoformat(entry_time_raw) if isinstance(entry_time_raw, str) else entry_time_raw
                        entry_price = trade['entry_price']
                        quantity = trade['quantity']

                        # Try to get actual exit from exchange history
                        logger.info(f"   🔍 Searching exchange history for actual exit...")
                        actual_exit = self._get_actual_exit_from_history(trade)

                        if actual_exit:
                            # Use actual exit from exchange
                            exit_price = actual_exit['exit_price']
                            exit_time_str = actual_exit['exit_time']
                            close_order_id = actual_exit['order_id']
                        else:
                            # Fallback: Use current price
                            logger.warning(f"   ⚠️ Using current market price as fallback")
                            exit_price = current_price
                            exit_time_str = datetime.now().isoformat()
                            close_order_id = None

                        # Calculate P&L
                        side = trade.get('side', 'LONG')
                        if side == "LONG":
                            pnl_pct = (exit_price - entry_price) / entry_price
                        else:  # SHORT
                            pnl_pct = (entry_price - exit_price) / entry_price

                        position_value = quantity * entry_price
                        pnl_usd_gross = quantity * (exit_price - entry_price) if side == "LONG" else quantity * (entry_price - exit_price)
                        entry_cost = entry_price * quantity * Phase4TestnetConfig.TRANSACTION_COST
                        exit_cost = exit_price * quantity * Phase4TestnetConfig.TRANSACTION_COST
                        transaction_cost = entry_cost + exit_cost
                        pnl_usd_net = pnl_usd_gross - transaction_cost

                        # Update trade
                        trade['status'] = 'CLOSED'
                        trade['exit_time'] = exit_time_str
                        trade['exit_price'] = exit_price
                        trade['exit_reason'] = "Exchange closed (verified from history)" if actual_exit else "Auto-closed (position not found on exchange - reverse orphan)"
                        trade['close_order_id'] = close_order_id
                        trade['pnl_pct'] = pnl_pct
                        trade['pnl_usd_gross'] = pnl_usd_gross
                        trade['transaction_cost'] = transaction_cost
                        trade['pnl_usd_net'] = pnl_usd_net

                        logger.warning(f"   ✅ Auto-closed: {side} {quantity:.4f} BTC @ ${entry_price:,.2f} → ${exit_price:,.2f}")
                        logger.warning(f"      P&L: {pnl_pct*100:+.2f}% (${pnl_usd_net:+,.2f})")
                        logger.warning(f"      Entry: {entry_time.strftime('%Y-%m-%d %H:%M:%S')}")
                        logger.warning(f"      Duration: {(datetime.now() - entry_time).total_seconds() / 3600:.1f}h")

                    logger.success(f"   ✅ Reverse orphan detection: {len(open_trades)} trade(s) auto-closed")

                # Look for new entry (if no position)
                self._check_entry(df, current_idx, current_price, current_regime, current_balance)

            # Print stats (with current API price for accurate B&H comparison)
            self._print_stats(current_price=current_price)

            # Save state
            self._save_state()

        except Exception as e:
            logger.error(f"Error in update cycle: {e}")
            import traceback
            traceback.print_exc()
            # Note: log_error signature: (error_type, error_message, stack_trace)
            self.inst_logger.log_error(
                error_type="UPDATE_CYCLE_ERROR",
                error_message=f"Error in update cycle: {str(e)}",
                stack_trace=traceback.format_exc()
            )

    def _get_market_data(self) -> pd.DataFrame:
        """Get market data from BingX Testnet API"""
        try:
            # BingX API supports up to 1440 candles in single request
            klines = self.client.get_klines(
                symbol=Phase4TestnetConfig.SYMBOL,
                interval=Phase4TestnetConfig.TIMEFRAME,
                limit=Phase4TestnetConfig.LOOKBACK_CANDLES
            )

            if not klines:
                logger.error("No klines data received")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(klines)
            df = df.rename(columns={'time': 'timestamp'})
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df[['open', 'high', 'low', 'close', 'volume']] = \
                df[['open', 'high', 'low', 'close', 'volume']].astype(float)
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

            # Sort by timestamp (ensure chronological order)
            df = df.sort_values('timestamp').reset_index(drop=True)

            latest_price = df['close'].iloc[-1]
            latest_time = df['timestamp'].iloc[-1]
            logger.info(f"✅ Live data from BingX Testnet API: {len(df)} candles")
            logger.info(f"   Latest: ${latest_price:,.2f} @ {latest_time.strftime('%Y-%m-%d %H:%M')}")

            return df

        except Exception as e:
            logger.error(f"Failed to get market data: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _calculate_rsi_macd_features(self, df):
        """
        Calculate RSI/MACD features using TA-Lib (2025-10-16)

        Required for Enhanced SHORT Entry model (RSI/MACD Enhanced)
        - RSI (14-period) with slope, overbought/oversold zones, divergence
        - MACD (12,26,9) with histogram slope and crossovers
        """
        import talib

        # RSI (14-period)
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        df['rsi_slope'] = df['rsi'].diff(3)
        df['rsi_overbought'] = (df['rsi'] > 70).astype(float)
        df['rsi_oversold'] = (df['rsi'] < 30).astype(float)

        # RSI divergence (price vs RSI direction mismatch)
        price_change = df['close'].diff(5)
        rsi_change = df['rsi'].diff(5)
        df['rsi_divergence'] = (
            ((price_change > 0) & (rsi_change < 0)) |  # Bearish divergence
            ((price_change < 0) & (rsi_change > 0))    # Bullish divergence
        ).astype(float)

        # MACD (12, 26, 9)
        macd, macd_signal, macd_hist = talib.MACD(
            df['close'], fastperiod=12, slowperiod=26, signalperiod=9
        )
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_hist'] = macd_hist
        df['macd_histogram_slope'] = pd.Series(macd_hist).diff(3)

        # MACD crossovers
        df['macd_crossover'] = (
            (macd > macd_signal) &
            (macd.shift(1) <= macd_signal.shift(1))
        ).astype(float)
        df['macd_crossunder'] = (
            (macd < macd_signal) &
            (macd.shift(1) >= macd_signal.shift(1))
        ).astype(float)

        return df

    def _calculate_enhanced_exit_features(self, df):
        """
        Calculate ENHANCED EXIT features for improved model performance

        2025-10-16 Enhancement: 22 market context features
        - Volume analysis (volume_ratio, volume_surge)
        - Price momentum (price_vs_ma20, price_vs_ma50, price_acceleration)
        - Volatility metrics (volatility_20, volatility_regime)
        - RSI dynamics (rsi_slope, rsi_overbought, rsi_oversold, rsi_divergence)
        - MACD dynamics (macd_histogram_slope, macd_crossover, macd_crossunder)
        - Price patterns (higher_high, lower_low)
        - Support/Resistance proximity (near_resistance, near_support)
        - Bollinger Band position (bb_position)

        These features help differentiate good vs bad exit timing
        without requiring position-specific context.
        """

        # Volume features
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['volume_surge'] = (df['volume'] > df['volume'].rolling(20).mean() * 1.5).astype(float)

        # Price momentum features (use existing SMA columns from advanced features)
        if 'sma_50' in df.columns:
            df['price_vs_ma20'] = (df['close'] - df['sma_50']) / df['sma_50']
        else:
            ma_20 = df['close'].rolling(20).mean()
            df['price_vs_ma20'] = (df['close'] - ma_20) / ma_20

        if 'sma_200' in df.columns:
            df['price_vs_ma50'] = (df['close'] - df['sma_200']) / df['sma_200']
        else:
            ma_50 = df['close'].rolling(50).mean()
            df['price_vs_ma50'] = (df['close'] - ma_50) / ma_50

        # Volatility features
        df['returns'] = df['close'].pct_change()
        df['volatility_20'] = df['returns'].rolling(20).std()
        df['volatility_regime'] = (df['volatility_20'] > df['volatility_20'].rolling(100).median()).astype(float)

        # RSI dynamics
        if 'rsi' in df.columns:
            df['rsi_slope'] = df['rsi'].diff(3) / 3  # Rate of change over 3 candles (15 min)
            df['rsi_overbought'] = (df['rsi'] > 70).astype(float)
            df['rsi_oversold'] = (df['rsi'] < 30).astype(float)
        else:
            df['rsi_slope'] = 0
            df['rsi_overbought'] = 0
            df['rsi_oversold'] = 0
        df['rsi_divergence'] = 0  # Placeholder (complex calculation, not critical)

        # MACD dynamics
        if 'macd_hist' in df.columns:
            df['macd_histogram_slope'] = df['macd_hist'].diff(3) / 3
        else:
            df['macd_histogram_slope'] = 0

        if 'macd' in df.columns and 'macd_signal' in df.columns:
            df['macd_crossover'] = ((df['macd'] > df['macd_signal']) &
                                   (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(float)
            df['macd_crossunder'] = ((df['macd'] < df['macd_signal']) &
                                    (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(float)
        else:
            df['macd_crossover'] = 0
            df['macd_crossunder'] = 0

        # Price patterns
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(float)
        df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(float)
        df['price_acceleration'] = df['close'].diff().diff()  # Second derivative

        # Support/Resistance proximity (use existing columns if available)
        if 'support_level' in df.columns and 'resistance_level' in df.columns:
            df['near_resistance'] = (df['close'] > df['resistance_level'] * 0.98).astype(float)
            df['near_support'] = (df['close'] < df['support_level'] * 1.02).astype(float)
        else:
            df['near_resistance'] = 0
            df['near_support'] = 0

        # Bollinger Band position (use existing columns if available)
        if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            # Handle division by zero
            df['bb_position'] = df['bb_position'].fillna(0.5)
        else:
            df['bb_position'] = 0.5

        # Clean NaN values (forward fill then backward fill)
        df = df.ffill().bfill()

        return df

    def _initialize_buy_hold(self, current_price):
        """Initialize Buy & Hold baseline"""
        self.bh_entry_price = current_price
        self.bh_btc_quantity = self.initial_balance / current_price
        self.bh_initialized = True

        logger.success(f"📊 Buy & Hold Baseline Initialized (for comparison):")
        logger.info(f"   Virtual Buy: {self.bh_btc_quantity:.6f} BTC @ ${current_price:,.2f}")

    def _classify_market_regime(self, df):
        """Classify current market regime"""
        lookback = 20
        if len(df) < lookback:
            return "Unknown"

        recent_data = df.tail(lookback)
        start_price = recent_data['close'].iloc[0]
        end_price = recent_data['close'].iloc[-1]
        price_change_pct = ((end_price / start_price) - 1) * 100

        if price_change_pct > Phase4TestnetConfig.BULL_THRESHOLD:
            return "Bull"
        elif price_change_pct < Phase4TestnetConfig.BEAR_THRESHOLD:
            return "Bear"
        else:
            return "Sideways"

    def _calculate_dynamic_thresholds(self, df, current_idx):
        """
        Calculate adaptive thresholds based on ACTUAL ENTRY RATE (2025-10-16 V3 ROOT CAUSE FIX)

        FUNDAMENTAL FIX: Previous versions measured "signal rate at base threshold" but
        traded at "adjusted threshold", creating logical contradiction with no feedback loop.

        NEW APPROACH: Measure what actually matters - how many trades are we making?
        - If too many actual entries → raise threshold
        - If too few actual entries → lower threshold
        - True feedback loop: threshold affects entries, entries affect threshold

        Logic:
        - Count actual entries in last 6 hours
        - Calculate actual_entry_rate = entries / candles
        - Compare with target_entry_rate (~1.1% = 22 trades/week)
        - Adjust threshold to converge toward target

        Args:
            df: DataFrame with features
            current_idx: Current candle index

        Returns:
            dict: {'long': adjusted_threshold, 'short': adjusted_threshold,
                   'entry_rate': actual_entry_rate, 'adjustment': threshold_delta,
                   'entries_count': recent_entries_count}
        """
        if not Phase4TestnetConfig.ENABLE_DYNAMIC_THRESHOLD:
            return {
                'long': Phase4TestnetConfig.BASE_LONG_ENTRY_THRESHOLD,
                'short': Phase4TestnetConfig.BASE_SHORT_ENTRY_THRESHOLD,
                'entry_rate': None,
                'adjustment': 0.0,
                'reason': 'disabled'
            }

        # Calculate actual entry rate from recent trades
        lookback_time = timedelta(hours=Phase4TestnetConfig.DYNAMIC_LOOKBACK_HOURS)
        cutoff_time = datetime.now() - lookback_time

        # Count entries in lookback period
        recent_entries = [
            t for t in self.trades
            if (t.get('status') in ['OPEN', 'CLOSED'] and
                'entry_time' in t and
                datetime.fromisoformat(t['entry_time']) > cutoff_time)
        ]
        entries_count = len(recent_entries)

        # Calculate actual entry rate
        lookback_candles = Phase4TestnetConfig.LOOKBACK_CANDLES
        actual_entry_rate = entries_count / lookback_candles

        # Target entry rate (from config)
        target_entry_rate = Phase4TestnetConfig.TARGET_ENTRY_RATE

        # Determine if we have enough data for feedback-based adjustment
        min_entries = Phase4TestnetConfig.MIN_ENTRIES_FOR_FEEDBACK

        if entries_count < min_entries:
            # COLD START MODE (< 5 entries): Use volatility-based conservative approach
            # CRITICAL: Do NOT use base signal rate (causes V2 problem)
            # Instead: Use BASE threshold + small volatility-based adjustment
            logger.debug(f"📊 COLD START ({entries_count} < {min_entries}): Using volatility-based threshold")

            # Check if we have enough data for volatility calculation
            if len(df) < lookback_candles or current_idx < lookback_candles:
                # Very early, use BASE thresholds as-is
                logger.debug(f"   Insufficient candles ({current_idx} < {lookback_candles}), using BASE thresholds")
                return {
                    'long': Phase4TestnetConfig.BASE_LONG_ENTRY_THRESHOLD,
                    'short': Phase4TestnetConfig.BASE_SHORT_ENTRY_THRESHOLD,
                    'entry_rate': actual_entry_rate,
                    'entries_count': entries_count,
                    'adjustment': 0.0,
                    'adjustment_ratio': 0.0,
                    'reason': 'cold_start_insufficient_data'
                }

            # Calculate recent volatility for market condition assessment
            start_idx = max(0, current_idx - lookback_candles)
            recent_df = df.iloc[start_idx:current_idx]

            # Use ATR or simple volatility as market condition indicator
            if 'atr' in recent_df.columns:
                recent_volatility = recent_df['atr'].mean()
                baseline_volatility = df['atr'].mean()  # Overall average
            elif 'volatility' in recent_df.columns:
                recent_volatility = recent_df['volatility'].mean()
                baseline_volatility = df['volatility'].mean()
            else:
                # Fallback: calculate simple price volatility
                returns = recent_df['close'].pct_change().dropna()
                recent_volatility = returns.std()
                baseline_volatility = df['close'].pct_change().std()

            # Volatility-based adjustment (conservative)
            volatility_ratio = recent_volatility / baseline_volatility if baseline_volatility > 0 else 1.0

            if volatility_ratio > 1.3:
                # High volatility → raise threshold slightly (more conservative)
                initial_adjustment = 0.03
                reason_detail = f"High volatility ({volatility_ratio:.2f}x baseline)"
            elif volatility_ratio < 0.7:
                # Low volatility → lower threshold slightly (more opportunities)
                initial_adjustment = -0.03
                reason_detail = f"Low volatility ({volatility_ratio:.2f}x baseline)"
            else:
                # Normal volatility → use BASE threshold as-is
                initial_adjustment = 0.0
                reason_detail = f"Normal volatility ({volatility_ratio:.2f}x baseline)"

            # Apply adjustment
            adjusted_long = Phase4TestnetConfig.BASE_LONG_ENTRY_THRESHOLD + initial_adjustment
            adjusted_short = Phase4TestnetConfig.BASE_SHORT_ENTRY_THRESHOLD + initial_adjustment

            # Clip to allowed range
            adjusted_long = np.clip(adjusted_long, Phase4TestnetConfig.MIN_THRESHOLD, Phase4TestnetConfig.MAX_THRESHOLD)
            adjusted_short = np.clip(adjusted_short, Phase4TestnetConfig.MIN_THRESHOLD, Phase4TestnetConfig.MAX_THRESHOLD)

            logger.info(f"📊 COLD START MODE:")
            logger.info(f"   Entries: {entries_count}/{min_entries} (need {min_entries - entries_count} more for feedback)")
            logger.info(f"   Volatility: {recent_volatility:.4f} (baseline: {baseline_volatility:.4f}, ratio: {volatility_ratio:.2f}x)")
            logger.info(f"   Adjustment: {initial_adjustment:+.2f} ({reason_detail})")
            logger.info(f"   Thresholds: LONG {adjusted_long:.2f}, SHORT {adjusted_short:.2f}")

            return {
                'long': adjusted_long,
                'short': adjusted_short,
                'entry_rate': actual_entry_rate,
                'entries_count': entries_count,
                'adjustment': initial_adjustment,
                'adjustment_ratio': volatility_ratio,
                'reason': 'cold_start_volatility_based'
            }
        else:
            # FEEDBACK MODE (>= 5 entries): Use ACTUAL entry rate for adjustment
            # This is the true feedback loop
            adjustment_ratio = actual_entry_rate / target_entry_rate if target_entry_rate > 0 else 1.0

        # Calculate threshold adjustment based on ratio
        # Non-linear adjustment for extreme deviations
        if adjustment_ratio > 2.0:  # Too many entries (>200% of target)
            # Exponential increase - raise threshold significantly
            threshold_delta = -Phase4TestnetConfig.THRESHOLD_ADJUSTMENT_FACTOR * ((adjustment_ratio - 1.0) ** 0.75)
        elif adjustment_ratio < 0.5:  # Too few entries (<50% of target)
            # Exponential decrease - lower threshold significantly
            threshold_delta = Phase4TestnetConfig.THRESHOLD_ADJUSTMENT_FACTOR * ((1.0 - adjustment_ratio) ** 0.75)
        else:  # Normal range (50-200% of target)
            # Linear adjustment for gradual convergence
            threshold_delta = (1.0 - adjustment_ratio) * Phase4TestnetConfig.THRESHOLD_ADJUSTMENT_FACTOR

        # Apply adjustment to base thresholds
        adjusted_long = Phase4TestnetConfig.BASE_LONG_ENTRY_THRESHOLD - threshold_delta
        adjusted_short = Phase4TestnetConfig.BASE_SHORT_ENTRY_THRESHOLD - threshold_delta

        # Clip to allowed range
        adjusted_long = np.clip(adjusted_long, Phase4TestnetConfig.MIN_THRESHOLD, Phase4TestnetConfig.MAX_THRESHOLD)
        adjusted_short = np.clip(adjusted_short, Phase4TestnetConfig.MIN_THRESHOLD, Phase4TestnetConfig.MAX_THRESHOLD)

        # Emergency check: If at max threshold AND entry rate still >2.5x target
        # This could indicate model misconfiguration or extreme market
        if (adjusted_long >= Phase4TestnetConfig.MAX_THRESHOLD and
            actual_entry_rate > target_entry_rate * 2.5 and
            entries_count >= min_entries):
            logger.warning("=" * 80)
            logger.warning("⚠️  EMERGENCY: Threshold at maximum but entry rate still excessive")
            logger.warning(f"   Actual Entry Rate: {actual_entry_rate*100:.2f}% ({entries_count} entries in 6h)")
            logger.warning(f"   Target Entry Rate: {target_entry_rate*100:.2f}%")
            logger.warning(f"   Ratio: {adjustment_ratio:.2f}x target")
            logger.warning(f"   Threshold: {adjusted_long:.3f} (at maximum {Phase4TestnetConfig.MAX_THRESHOLD})")
            logger.warning("   Possible causes:")
            logger.warning("   1. Model misconfiguration or wrong feature scaling")
            logger.warning("   2. Extreme volatility exceeding training data range")
            logger.warning("   3. Target rate too low for current market conditions")
            logger.warning("=" * 80)

            # Check how long we've been in this state
            if not hasattr(self, '_emergency_threshold_start'):
                self._emergency_threshold_start = datetime.now()
                self._emergency_logged = False

            emergency_duration = (datetime.now() - self._emergency_threshold_start).total_seconds() / 3600

            if emergency_duration > 1.0 and not self._emergency_logged:  # 1 hour
                logger.error("🚨 CRITICAL: Threshold at maximum for >1 hour with excessive entry rate")
                logger.error(f"   Action: Review model config, consider raising MAX_THRESHOLD or TARGET_TRADES_PER_WEEK")
                self.inst_logger.log_compliance_event(
                    event_type="THRESHOLD_EMERGENCY",
                    description=f"Threshold at max for {emergency_duration:.1f}h, entry rate {actual_entry_rate*100:.2f}%",
                    severity="HIGH",
                    action_taken="Trading continues but requires investigation"
                )
                self._emergency_logged = True
        else:
            # Reset emergency tracking if conditions normalize
            if hasattr(self, '_emergency_threshold_start'):
                delattr(self, '_emergency_threshold_start')
                self._emergency_logged = False

        return {
            'long': adjusted_long,
            'short': adjusted_short,
            'entry_rate': actual_entry_rate,  # Actual entry rate (most important)
            'entries_count': entries_count,   # Number of entries in lookback
            'adjustment': threshold_delta,
            'adjustment_ratio': adjustment_ratio,
            'reason': 'actual_entry_rate' if entries_count >= min_entries else 'cold_start_volatility_based'
        }

    def _check_entry(self, df, idx, current_price, regime, current_balance):
        """Check for entry signal and execute order (DUAL MODEL: LONG + SHORT + DYNAMIC THRESHOLDS)"""
        # Calculate dynamic thresholds based on recent market regime
        dynamic_thresholds = self._calculate_dynamic_thresholds(df, idx)
        threshold_long = dynamic_thresholds['long']
        threshold_short = dynamic_thresholds['short']

        # Log dynamic threshold adjustment (V3.1: ACTUAL ENTRY RATE + Volatility-based Cold Start)
        if dynamic_thresholds['reason'] in ['actual_entry_rate', 'cold_start_volatility_based', 'cold_start_insufficient_data']:
            entry_rate = dynamic_thresholds['entry_rate']
            entries_count = dynamic_thresholds['entries_count']
            adjustment = dynamic_thresholds['adjustment']
            adjustment_ratio = dynamic_thresholds['adjustment_ratio']
            target_rate = Phase4TestnetConfig.TARGET_ENTRY_RATE

            logger.info(f"🎯 Dynamic Threshold System (V3.1):")
            logger.info(f"  Actual Entry Rate: {entry_rate*100:.2f}% ({entries_count} entries in 6h)")
            logger.info(f"  Target Entry Rate: {target_rate*100:.2f}% (~{Phase4TestnetConfig.TARGET_TRADES_PER_WEEK:.1f} trades/week)")
            logger.info(f"  Adjustment Ratio: {adjustment_ratio:.2f}x")
            logger.info(f"  Threshold Adjustment: {adjustment:+.3f}")
            logger.info(f"  LONG Threshold: {threshold_long:.3f} (base: {Phase4TestnetConfig.BASE_LONG_ENTRY_THRESHOLD:.2f})")
            logger.info(f"  SHORT Threshold: {threshold_short:.3f} (base: {Phase4TestnetConfig.BASE_SHORT_ENTRY_THRESHOLD:.2f})")

            if dynamic_thresholds['reason'] == 'cold_start_volatility_based':
                logger.info(f"  Mode: COLD START (volatility-based adjustment, need {Phase4TestnetConfig.MIN_ENTRIES_FOR_FEEDBACK} entries)")
            elif dynamic_thresholds['reason'] == 'cold_start_insufficient_data':
                logger.info(f"  Mode: COLD START (insufficient data, using BASE thresholds)")

        # Get features (LONG: 44 features, SHORT: 22 SELL features)
        features_long = df[self.feature_columns].iloc[idx:idx+1].values
        features_short = df[self.short_feature_columns].iloc[idx:idx+1].values

        if np.isnan(features_long).any() or np.isnan(features_short).any():
            logger.warning("NaN in features, skipping entry check")
            return

        # Apply MinMaxScaler normalization before prediction
        features_long_scaled = self.long_scaler.transform(features_long)
        features_short_scaled = self.short_scaler.transform(features_short)

        # Predict with DUAL MODELS (with normalized features)
        prob_long = self.long_model.predict_proba(features_long_scaled)[0][1]   # LONG model (44 features)
        prob_short = self.short_model.predict_proba(features_short_scaled)[0][1]  # SHORT model (22 SELL features)

        # Save latest ENTRY signals to instance variables (for state file and monitoring)
        self.latest_long_entry_prob = float(prob_long)
        self.latest_short_entry_prob = float(prob_short)
        self.latest_long_entry_threshold = float(threshold_long)
        self.latest_short_entry_threshold = float(threshold_short)
        self.latest_entry_timestamp = datetime.now()

        # Save dynamic threshold context (for monitor display) - V3: ACTUAL ENTRY RATE
        self.latest_threshold_entry_rate = dynamic_thresholds.get('entry_rate')
        self.latest_threshold_entries_count = dynamic_thresholds.get('entries_count')
        self.latest_threshold_adjustment = dynamic_thresholds.get('adjustment')
        self.latest_threshold_adjustment_ratio = dynamic_thresholds.get('adjustment_ratio')

        # Log model predictions to institutional logger (2025-10-15: Dynamic thresholds)
        self.inst_logger.log_model_prediction(
            model_name="LONG_Entry_Model",
            prediction="BUY" if prob_long >= threshold_long else "HOLD",
            confidence=prob_long
        )
        self.inst_logger.log_model_prediction(
            model_name="SHORT_Entry_Model",
            prediction="SELL" if prob_short >= threshold_short else "HOLD",
            confidence=prob_short
        )

        logger.info(f"Signal Check (Dual Model - Dynamic Thresholds 2025-10-15):")
        logger.info(f"  LONG Model Prob: {prob_long:.3f} (dynamic threshold: {threshold_long:.2f})")
        logger.info(f"  SHORT Model Prob: {prob_short:.3f} (dynamic threshold: {threshold_short:.2f})")

        # Determine signal direction (independent models with DYNAMIC thresholds)
        signal_direction = None
        signal_probability = None

        if prob_long >= threshold_long:
            signal_direction = "LONG"
            signal_probability = prob_long
        elif prob_short >= threshold_short:
            # Opportunity Gating: Only enter SHORT if clearly better than LONG
            long_ev = prob_long * 0.0041  # LONG avg return (from backtest)
            short_ev = prob_short * 0.0047  # SHORT avg return (from backtest)
            opportunity_cost = short_ev - long_ev

            if opportunity_cost > 0.001:  # Gate threshold (validated)
                signal_direction = "SHORT"
                signal_probability = prob_short
                logger.info(f"  ✅ SHORT passed Opportunity Gate (opp_cost={opportunity_cost:.6f} > 0.001)")
            else:
                logger.info(f"  ❌ SHORT blocked by Opportunity Gate (opp_cost={opportunity_cost:.6f} ≤ 0.001)")
                signal_direction = None
                signal_probability = None

        # Log signal data (for all signals, even if below threshold)
        signal_data = {
            'timestamp': datetime.now(),
            'price': current_price,
            'position_status': 'NONE',
            'position_pnl_pct': None,
            'position_probability': None,
            'current_signal_prob': prob_long,  # Backward compatibility (LONG prob)
            'current_signal_prob_long': prob_long,
            'current_signal_prob_short': prob_short,
            'signal_direction': signal_direction,  # "LONG", "SHORT", or None
            'signal_strength_delta': None,
            'missed_opportunity': False,
            'hours_held': 0,
            'dynamic_threshold_long': threshold_long,  # Track dynamic threshold
            'dynamic_threshold_short': threshold_short,
            'entry_rate_6h': dynamic_thresholds.get('entry_rate', 0.0),  # Actual entry rate (V3.1)
            'entries_count_6h': dynamic_thresholds.get('entries_count', 0)  # Number of entries
        }
        self.signal_log.append(signal_data)

        # Log signal regardless of threshold (for analysis)
        # Note: log_signal signature: (signal_type, direction, probability, price, features)
        self.inst_logger.log_signal(
            signal_type="ENTRY",
            direction=signal_direction if signal_direction else "NONE",
            probability=signal_probability if signal_probability else max(prob_long, prob_short),
            price=current_price,
            features={
                "long_prob": prob_long,
                "short_prob": prob_short,
                "regime": regime,
                "long_threshold": threshold_long,  # Dynamic threshold
                "short_threshold": threshold_short,  # Dynamic threshold
                "entry_rate_6h": dynamic_thresholds.get('entry_rate'),  # V3: Actual entry rate
                "entries_count_6h": dynamic_thresholds.get('entries_count'),
                "threshold_reason": dynamic_thresholds.get('reason')
            }
        )

        # Check threshold (different for LONG vs SHORT, DYNAMIC)
        if signal_direction is None:
            logger.info(f"  Should Enter: False (LONG {prob_long:.3f} < {threshold_long:.2f}, SHORT {prob_short:.3f} < {threshold_short:.2f})")
            return

        logger.info(f"  Should Enter: True ({signal_direction} signal = {signal_probability:.3f})")

        # Calculate volatility
        current_volatility = df['atr_pct'].iloc[idx] if 'atr_pct' in df.columns else 0.01
        avg_volatility = df['atr_pct'].iloc[max(0, idx-50):idx].mean() if 'atr_pct' in df.columns else 0.01

        # Calculate dynamic position size
        sizing_result = self.position_sizer.calculate_position_size(
            capital=current_balance,
            signal_strength=signal_probability,
            current_volatility=current_volatility,
            avg_volatility=avg_volatility,
            market_regime=regime,
            recent_trades=self.trades[-10:] if len(self.trades) > 0 else [],
            leverage=Phase4TestnetConfig.LEVERAGE
        )

        # Use leveraged_value for quantity calculation (4x leverage)
        position_value = sizing_result['position_value']  # Collateral amount
        leveraged_value = sizing_result['leveraged_value']  # Actual position size (4x)
        quantity = leveraged_value / current_price  # ✅ Fixed: Use leveraged value

        # Determine order side based on signal direction
        order_side = "BUY" if signal_direction == "LONG" else "SELL"

        # Execute REAL order on Testnet!
        try:
            logger.warning(f"⚡ EXECUTING REAL ORDER on Testnet!")
            logger.info(f"   Order Type: MARKET {order_side}")
            logger.info(f"   Direction: {signal_direction}")
            logger.info(f"   Quantity: {quantity:.4f} BTC")
            logger.info(f"   Est. Value: ${position_value:,.2f}")

            order_result = self.client.create_order(
                symbol=Phase4TestnetConfig.SYMBOL,
                side=order_side,  # BUY for LONG, SELL for SHORT
                position_side="BOTH",  # Fixed: One-way mode requires "BOTH"
                order_type="MARKET",
                quantity=quantity
            )

            logger.success(f"✅ ORDER EXECUTED!")
            logger.info(f"   Order ID: {order_result.get('orderId')}")
            logger.info(f"   Status: {order_result.get('status')}")
            logger.info(f"   Direction: {signal_direction}")
            logger.info(f"   Position Size: {sizing_result['position_size_pct']*100:.1f}% (collateral)")
            logger.info(f"   Collateral: ${position_value:,.2f}")
            logger.info(f"   Leveraged Position: ${leveraged_value:,.2f} ({Phase4TestnetConfig.LEVERAGE}x)")
            logger.info(f"   Quantity: {quantity:.4f} BTC")
            logger.info(f"   Market Regime: {regime}")
            logger.info(f"   XGBoost Prob: {signal_probability:.3f}")

            # Get actual fill price from exchange (may differ from current_price due to slippage)
            actual_fill_price = order_result.get('average') or order_result.get('price') or current_price

            # Log slippage if significant
            slippage = actual_fill_price - current_price
            if abs(slippage) > 1.0:  # More than $1 slippage
                logger.warning(f"   Slippage: ${slippage:+.2f} (Market: ${current_price:,.2f} → Fill: ${actual_fill_price:,.2f})")

            # Record entry with ACTUAL fill price
            trade_record = {
                'entry_time': datetime.now(),
                'order_id': order_result.get('orderId'),
                'side': signal_direction,  # Store LONG or SHORT
                'entry_price': actual_fill_price,  # ✅ ACTUAL fill price (not approximate!)
                'quantity': quantity,
                'position_size_pct': sizing_result['position_size_pct'],
                'position_value': position_value,
                'regime': regime,
                'probability': signal_probability,
                'sizing_factors': sizing_result['factors'],
                'status': 'OPEN'
            }
            self.trades.append(trade_record)

            # Log trade entry to institutional logger (audit trail)
            self.inst_logger.log_trade_entry(
                order_id=str(order_result.get('orderId')),
                side=signal_direction,
                quantity=quantity,
                price=actual_fill_price,
                position_size_pct=sizing_result['position_size_pct'],
                signal_probability=signal_probability,
                regime=regime,
                leverage=float(Phase4TestnetConfig.LEVERAGE)
            )

            logger.info(f"   Entry Price: ${actual_fill_price:,.2f} (filled)")

        except BingXInsufficientBalanceError as e:
            logger.error("❌ Insufficient balance for order!")
            # Note: log_error signature: (error_type, error_message, stack_trace)
            import traceback
            self.inst_logger.log_error(
                error_type="INSUFFICIENT_BALANCE",
                error_message=f"Insufficient balance for {order_side} order: {quantity:.4f} BTC @ ${current_price:,.2f}",
                stack_trace=traceback.format_exc()
            )
        except BingXOrderError as e:
            logger.error(f"❌ Order failed: {e.message}")
            import traceback
            self.inst_logger.log_error(
                error_type="ORDER_EXECUTION_ERROR",
                error_message=f"Order execution failed ({order_side} {quantity:.4f} BTC): {str(e)}",
                stack_trace=traceback.format_exc()
            )
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
            import traceback
            self.inst_logger.log_error(
                error_type="UNEXPECTED_ORDER_ERROR",
                error_message=f"Unexpected error during {order_side} order execution: {str(e)}",
                stack_trace=traceback.format_exc()
            )

    def _manage_position(self, current_price, df, current_idx, position):
        """Manage existing position and check exit conditions"""
        entry_price = position['entry_price']
        quantity = position['position_amt']
        unrealized_pnl = position['unrealized_pnl']

        # Find position side from trades
        position_side = None
        entry_time = None
        position_probability = None
        for trade in reversed(self.trades):
            if trade.get('status') == 'OPEN':
                position_side = trade.get('side', 'LONG')  # Default to LONG for backward compatibility
                entry_time = trade.get('entry_time')
                position_probability = trade.get('probability')
                break

        # Calculate P&L (different formula for LONG vs SHORT)
        if position_side == "SHORT":
            # SHORT: profit when price goes down
            pnl_pct = (entry_price - current_price) / entry_price
        else:
            # LONG: profit when price goes up
            pnl_pct = (current_price - entry_price) / entry_price

        pnl_usd = pnl_pct * (entry_price * quantity)

        # Calculate holding time
        if entry_time:
            hours_held = (datetime.now() - entry_time).total_seconds() / 3600
        else:
            hours_held = 0

        # Check current signal strength (to track missed opportunities)
        features = df[self.feature_columns].iloc[current_idx:current_idx+1].values
        current_signal_prob = None
        signal_strength_delta = None
        missed_opportunity = False

        if not np.isnan(features).any():
            # Use LONG model for signal tracking (backward compatibility)
            current_signal_prob = self.long_model.predict_proba(features)[0][1]
            if position_probability:
                signal_strength_delta = current_signal_prob - position_probability
                # Flag as missed opportunity if current signal is much stronger
                if current_signal_prob >= 0.75 and signal_strength_delta >= 0.10:
                    missed_opportunity = True

        # Log signal data for analysis
        signal_data = {
            'timestamp': datetime.now(),
            'price': current_price,
            'position_status': 'OPEN',
            'position_pnl_pct': pnl_pct,
            'position_probability': position_probability,
            'current_signal_prob': current_signal_prob,  # Backward compatibility
            'current_signal_prob_long': None,  # Not calculated in position management
            'current_signal_prob_short': None,
            'signal_direction': position_side,  # Current position direction
            'signal_strength_delta': signal_strength_delta,
            'missed_opportunity': missed_opportunity,
            'hours_held': hours_held
        }
        self.signal_log.append(signal_data)

        # Calculate current exposure and risk metrics
        current_balance = self._get_account_balance(use_cache=True)
        current_exposure = (entry_price * quantity) / current_balance
        position_value = entry_price * quantity * Phase4TestnetConfig.LEVERAGE

        # Log risk metrics to institutional logger
        self.inst_logger.log_risk_metrics(
            current_exposure=current_exposure,
            var_95=None,  # Can be calculated from historical returns if needed
            var_99=None,
            leverage=float(Phase4TestnetConfig.LEVERAGE),
            max_position_pct=Phase4TestnetConfig.MAX_POSITION_PCT
        )

        logger.info(f"Position: {position_side} {quantity:.4f} BTC @ ${entry_price:,.2f}")
        logger.info(f"P&L: {pnl_pct * 100:+.2f}% (${pnl_usd:+,.2f})")
        logger.info(f"Unrealized PnL (Exchange): ${unrealized_pnl:+,.2f}")
        logger.info(f"Holding: {hours_held:.1f} hours")

        if current_signal_prob is not None:
            # Format entry probability safely
            entry_prob_str = f"{position_probability:.3f}" if position_probability is not None else "N/A"
            logger.info(f"Current Signal: {current_signal_prob:.3f} (Entry: {entry_prob_str})")
            if missed_opportunity:
                logger.warning(f"⚠️ MISSED OPPORTUNITY: Current signal {current_signal_prob:.3f} is +{signal_strength_delta:.3f} stronger!")

        # ============================================================================
        # EXIT DECISION: Dual ML Exit Models (LONG/SHORT specialized)
        # ============================================================================
        exit_reason = None

        # ============================================================================
        # STANDARD EXIT CONDITIONS (from backtest validation)
        # Priority Order: SL > TP > Max Hold > ML Exit
        # These are the validated conditions from 81-combination optimization
        # ============================================================================

        # 1. Stop Loss (FIRST PRIORITY - Risk Control)
        if pnl_pct <= -Phase4TestnetConfig.STOP_LOSS:
            exit_reason = f"Stop Loss ({pnl_pct*100:.2f}%)"
            logger.warning(f"⚠️ STOP LOSS TRIGGERED: {exit_reason}")

        # 2. Take Profit (SECOND PRIORITY - Profit Capture)
        elif pnl_pct >= Phase4TestnetConfig.TAKE_PROFIT:
            exit_reason = f"Take Profit ({pnl_pct*100:.2f}%)"
            logger.info(f"✅ TAKE PROFIT TRIGGERED: {exit_reason}")

        # 3. Max Holding (THIRD PRIORITY - Capital Efficiency)
        elif hours_held >= Phase4TestnetConfig.MAX_HOLDING_HOURS:
            exit_reason = f"Max Hold ({hours_held:.1f}h)"
            logger.info(f"⏰ MAX HOLD TRIGGERED: {exit_reason}")

        # 4. ML Exit Model (FOURTH PRIORITY - Optimal Timing)
        # ENHANCED MODELS (2025-10-16): 22 features + 2of3 scoring + NORMAL logic
        # Only calculate if standard exits don't trigger (computational efficiency)
        elif not np.isnan(features).any():
            # Select appropriate exit model, scaler, and features based on position direction
            if position_side == "LONG":
                exit_model = self.long_exit_model
                exit_scaler = self.long_exit_scaler
                exit_features_list = self.long_exit_features
            else:  # SHORT
                exit_model = self.short_exit_model
                exit_scaler = self.short_exit_scaler
                exit_features_list = self.short_exit_features

            # Extract 22 ENHANCED features directly from current dataframe
            # These are market context features calculated by prepare_features()
            # No position-specific features (removed: time_held, pnl_peak, etc.)
            try:
                exit_features_values = df[exit_features_list].iloc[current_idx].values.reshape(1, -1)
            except KeyError as e:
                logger.error(f"❌ Missing enhanced features: {e}")
                logger.error(f"   Available: {df.columns.tolist()}")
                logger.error(f"   Required: {exit_features_list}")
                exit_features_values = None

            if exit_features_values is not None:
                # Apply MinMaxScaler normalization before prediction
                exit_features_scaled = exit_scaler.transform(exit_features_values)

                # Get exit signal from ENHANCED Exit Model (NORMAL logic)
                exit_prob = exit_model.predict_proba(exit_features_scaled)[0][1]

                logger.info(f"Exit Model Signal ENHANCED ({position_side}): {exit_prob:.3f} (exit if >= {Phase4TestnetConfig.EXIT_THRESHOLD:.2f})")
                logger.debug(f"  Position P&L: {pnl_pct*100:.2f}%, Held: {hours_held:.1f}h")
                if exit_prob >= Phase4TestnetConfig.EXIT_THRESHOLD:
                    logger.info(f"  ✅ EXIT SIGNAL TRIGGERED: {exit_prob:.3f} >= {Phase4TestnetConfig.EXIT_THRESHOLD:.2f}")

                # Save latest EXIT signal to instance variables (for state file and monitoring)
                if position_side == "LONG":
                    self.latest_long_exit_prob = float(exit_prob)
                    self.latest_short_exit_prob = None  # No SHORT position
                else:  # SHORT
                    self.latest_short_exit_prob = float(exit_prob)
                    self.latest_long_exit_prob = None  # No LONG position
                self.latest_exit_timestamp = datetime.now()
                self.latest_exit_position_side = position_side

                # ✅ NORMAL LOGIC: Exit when probability is HIGH (proper learning)
                # Enhanced models with 2of3 scoring (profit + lead-peak + relative performance)
                # Trained on 22 market context features (volume, momentum, volatility, patterns)
                # Validated: prob >= 0.7 achieves +14.44% return, 67.4% win rate
                # Improvement: +2.84% vs inverted baseline (+24.5% relative gain)
                if exit_prob >= Phase4TestnetConfig.EXIT_THRESHOLD:
                    exit_reason = f"ML Exit ENHANCED ({position_side}, prob={exit_prob:.3f}>=0.7)"

        else:
            logger.warning("⚠️ NaN in features, skipping Exit Model (position held)")

        # Safety exits (override ML Exit Model if critical conditions)
        # These act as safety nets for extreme situations where ML exit fails
        # Much more conservative than backtest hard stops (-1.5% SL, +3.5% TP)
        # to allow ML model room to operate optimally
        if pnl_pct <= -0.05:  # -5% emergency stop (very rare with leverage 4x)
            exit_reason = f"Emergency Stop Loss ({pnl_pct*100:.2f}%)"
            logger.error(f"🚨 EMERGENCY EXIT: {exit_reason}")
            # Log compliance event for emergency stop
            self.inst_logger.log_compliance_event(
                event_type="EMERGENCY_STOP_LOSS",
                description=f"Emergency stop loss triggered at {pnl_pct*100:.2f}% loss",
                severity="HIGH",
                action_taken="Position closed immediately"
            )
        elif hours_held >= 8:  # 8 hours maximum (3.4x longer than expected 2.36h)
            exit_reason = f"Emergency Max Holding ({hours_held:.1f}h)"
            logger.warning(f"⚠️ EMERGENCY EXIT: {exit_reason}")
            # Log compliance event for max holding violation
            self.inst_logger.log_compliance_event(
                event_type="MAX_HOLDING_EXCEEDED",
                description=f"Position held for {hours_held:.1f}h exceeds 8h limit",
                severity="MEDIUM",
                action_taken="Position closed via emergency exit"
            )

        if exit_reason:
            self._exit_position(current_price, exit_reason, position)

    def _exit_position(self, exit_price, reason, position):
        """Exit position by executing REAL close order"""
        try:
            position_side = position.get('position_side', 'LONG')  # Get actual position side

            logger.warning(f"⚡ CLOSING POSITION on Testnet!")
            logger.info(f"   Reason: {reason}")
            logger.info(f"   Side: {position_side}")
            logger.info(f"   Quantity: {position['position_amt']:.4f} BTC")

            close_result = self.client.close_position(
                symbol=Phase4TestnetConfig.SYMBOL,
                position_side=position_side,  # Use actual position side (LONG or SHORT)
                quantity=position['position_amt']
            )

            # Validate that close actually succeeded
            # CCXT returns 'id' at top level, not 'orderId'
            order_id = close_result.get('id') or close_result.get('orderId')
            if not close_result or not order_id:
                logger.error(f"❌ POSITION CLOSE FAILED!")
                logger.error(f"   API returned: {close_result}")
                logger.error(f"   Trade status remains OPEN")
                return  # Don't mark as closed if API failed

            # Close succeeded - verify position no longer exists
            logger.success(f"✅ POSITION CLOSED!")
            logger.info(f"   Close Order ID: {order_id}")
            logger.info(f"   Exit Price: ${exit_price:,.2f}")

            # Update trade record
            for trade in reversed(self.trades):
                if trade.get('status') == 'OPEN':
                    trade['status'] = 'CLOSED'
                    trade['exit_time'] = datetime.now()
                    trade['exit_price'] = exit_price
                    trade['exit_reason'] = reason
                    trade['close_order_id'] = order_id  # Already extracted above

                    # Calculate final P&L (different for LONG vs SHORT)
                    entry_price = trade['entry_price']
                    quantity = trade['quantity']
                    trade_side = trade.get('side', 'LONG')

                    if trade_side == "SHORT":
                        # SHORT: profit when price goes down
                        pnl_pct = (entry_price - exit_price) / entry_price
                    else:
                        # LONG: profit when price goes up
                        pnl_pct = (exit_price - entry_price) / entry_price

                    pnl_usd = pnl_pct * (entry_price * quantity)

                    # Transaction costs
                    entry_cost = entry_price * quantity * Phase4TestnetConfig.TRANSACTION_COST
                    exit_cost = exit_price * quantity * Phase4TestnetConfig.TRANSACTION_COST
                    total_cost = entry_cost + exit_cost
                    net_pnl = pnl_usd - total_cost

                    trade['pnl_pct'] = pnl_pct
                    trade['pnl_usd_gross'] = pnl_usd
                    trade['transaction_cost'] = total_cost
                    trade['pnl_usd_net'] = net_pnl

                    # Calculate holding time in hours
                    holding_time_hours = (trade['exit_time'] - trade['entry_time']).total_seconds() / 3600

                    # Log trade exit to institutional logger (audit trail with P&L)
                    self.inst_logger.log_trade_exit(
                        order_id=str(order_id),
                        side=trade_side,
                        quantity=quantity,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        pnl_usd=net_pnl,
                        pnl_pct=pnl_pct,
                        holding_time_hours=holding_time_hours,
                        exit_reason=reason,
                        transaction_costs=total_cost
                    )

                    logger.info(f"   Gross P&L: {pnl_pct * 100:+.2f}% (${pnl_usd:+,.2f})")
                    logger.info(f"   Transaction Cost: ${total_cost:.2f}")
                    logger.info(f"   Net P&L: ${net_pnl:+,.2f}")

                    # Reset EXIT signals after position closed
                    self.latest_long_exit_prob = None
                    self.latest_short_exit_prob = None
                    self.latest_exit_timestamp = None
                    self.latest_exit_position_side = None
                    logger.debug(f"Exit signals reset (position closed)")

                    break

        except Exception as e:
            logger.error(f"❌ Failed to close position: {e}")
            logger.error(f"   Trade status remains OPEN (exception occurred)")
            # Note: log_error signature: (error_type, error_message, stack_trace)
            import traceback
            self.inst_logger.log_error(
                error_type="POSITION_CLOSE_ERROR",
                error_message=f"Failed to close {position.get('position_side')} position ({position.get('position_amt')} BTC): {str(e)}",
                stack_trace=traceback.format_exc()
            )

    def _print_stats(self, current_price=None):
        """Print current performance statistics"""
        closed_trades = [t for t in self.trades if t.get('status') == 'CLOSED']

        if len(closed_trades) == 0:
            logger.info(f"\n📊 No completed trades yet")
            return

        df_trades = pd.DataFrame(closed_trades)

        # Basic stats
        total_trades = len(df_trades)
        winning_trades = len(df_trades[df_trades['pnl_usd_net'] > 0])
        win_rate = (winning_trades / total_trades) * 100

        # Returns
        total_net_pnl = df_trades['pnl_usd_net'].sum()
        current_balance = self._get_account_balance(use_cache=True)
        total_return_pct = ((current_balance - self.initial_balance) / self.initial_balance) * 100

        # Average position size
        avg_position_size = df_trades['position_size_pct'].mean() * 100

        # Buy & Hold comparison
        if self.bh_initialized and current_price is not None:
            bh_value = self.bh_btc_quantity * current_price
            bh_return_pct = ((bh_value - self.initial_balance) / self.initial_balance) * 100
            vs_bh = total_return_pct - bh_return_pct
        else:
            bh_return_pct = 0.0
            vs_bh = 0.0

        # Time-based metrics
        days_running = (datetime.now() - self.session_start).total_seconds() / 86400
        trades_per_week = (total_trades / days_running) * 7 if days_running > 0 else 0

        # Calculate Sharpe and Sortino ratios (if enough trades)
        sharpe_ratio = None
        sortino_ratio = None
        if len(df_trades) >= 10:
            returns = df_trades['pnl_pct'].values
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else None
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0 and downside_returns.std() > 0:
                sortino_ratio = (returns.mean() / downside_returns.std()) * np.sqrt(252)

        # Calculate max drawdown
        equity_curve = (df_trades['pnl_usd_net'].cumsum() + self.initial_balance).values
        max_drawdown = None
        if len(equity_curve) > 0:
            peak = np.maximum.accumulate(equity_curve)
            drawdown = (equity_curve - peak) / peak
            max_drawdown = drawdown.min() if len(drawdown) > 0 else None

        # Calculate profit factor
        winning_pnl = df_trades[df_trades['pnl_usd_net'] > 0]['pnl_usd_net'].sum()
        losing_pnl = abs(df_trades[df_trades['pnl_usd_net'] < 0]['pnl_usd_net'].sum())
        profit_factor = winning_pnl / losing_pnl if losing_pnl > 0 else None

        # Log performance metrics to institutional logger
        self.inst_logger.log_performance_metrics(
            current_capital=current_balance,
            total_trades=total_trades,
            win_rate=win_rate,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            profit_factor=profit_factor
        )

        logger.info(f"\n{'=' * 80}")
        logger.info("📊 PHASE 4 TESTNET TRADING PERFORMANCE")
        logger.info(f"{'=' * 80}")
        logger.info(f"Session Duration: {days_running:.1f} days")
        logger.info(f"")
        # LONG vs SHORT Distribution (2025-10-15: Track threshold optimization effectiveness)
        long_trades = df_trades[df_trades['side'] == 'LONG']
        short_trades = df_trades[df_trades['side'] == 'SHORT']
        long_count = len(long_trades)
        short_count = len(short_trades)
        long_pct = (long_count / total_trades * 100) if total_trades > 0 else 0
        short_pct = (short_count / total_trades * 100) if total_trades > 0 else 0

        # Win rates by direction
        long_win_rate = (len(long_trades[long_trades['pnl_usd_net'] > 0]) / long_count * 100) if long_count > 0 else 0
        short_win_rate = (len(short_trades[short_trades['pnl_usd_net'] > 0]) / short_count * 100) if short_count > 0 else 0

        # Trades per week by direction
        long_trades_per_week = (long_count / days_running) * 7 if days_running > 0 else 0
        short_trades_per_week = (short_count / days_running) * 7 if days_running > 0 else 0

        logger.info(f"Trading Performance:")
        logger.info(f"  Total Trades: {total_trades}")
        logger.info(f"  Winning: {winning_trades} ({win_rate:.1f}%) {'✅' if win_rate >= Phase4TestnetConfig.TARGET_WIN_RATE else '⚠️'}")
        logger.info(f"  Trades/Week: {trades_per_week:.1f}")
        logger.info(f"  Avg Position: {avg_position_size:.1f}%")
        logger.info(f"")
        logger.info(f"Trade Distribution (Symmetric Model 2025-10-15):")
        logger.info(f"  LONG: {long_count} trades ({long_pct:.1f}%) - {long_trades_per_week:.1f}/week - Win: {long_win_rate:.1f}%")
        logger.info(f"  SHORT: {short_count} trades ({short_pct:.1f}%) - {short_trades_per_week:.1f}/week - Win: {short_win_rate:.1f}%")
        logger.info(f"  Expected: 79% LONG (17.7/week, 28% prec) / 21% SHORT (4.6/week, 79% prec!)")
        logger.info(f"")
        logger.info(f"Returns:")
        logger.info(f"  Total Net P&L: ${total_net_pnl:+,.2f}")
        logger.info(f"  Total Return: {total_return_pct:+.2f}%")
        logger.info(f"  Current Balance: ${current_balance:,.2f} USDT")
        logger.info(f"  Initial Balance: ${self.initial_balance:,.2f} USDT")
        logger.info(f"")
        logger.info(f"vs Buy & Hold:")
        logger.info(f"  B&H Return: {bh_return_pct:+.2f}%")
        logger.info(f"  Strategy Return: {total_return_pct:+.2f}%")
        logger.info(f"  Difference: {vs_bh:+.2f}% {'✅' if vs_bh > 0 else '⚠️'}")
        logger.info(f"{'=' * 80}")

    def _print_final_stats(self):
        """Print final statistics on exit"""
        self._print_stats()

        # Generate and log session summary
        session_duration = (datetime.now() - self.session_start).total_seconds() / 3600
        closed_trades = [t for t in self.trades if t.get('status') == 'CLOSED']
        open_trades = [t for t in self.trades if t.get('status') == 'OPEN']

        if len(closed_trades) > 0:
            df_trades = pd.DataFrame(closed_trades)
            total_pnl = df_trades['pnl_usd_net'].sum()
            win_rate = (len(df_trades[df_trades['pnl_usd_net'] > 0]) / len(df_trades)) * 100
        else:
            total_pnl = 0.0
            win_rate = 0.0

        # Generate session summary
        summary = self.inst_logger.generate_session_summary(
            session_duration_hours=session_duration,
            final_capital=self._get_account_balance(use_cache=True),
            total_trades=len(closed_trades),
            open_positions=len(open_trades)
        )

        logger.info("\n" + "=" * 80)
        logger.info("📊 SESSION SUMMARY (Institutional Format)")
        logger.info("=" * 80)
        logger.info(f"Duration: {session_duration:.1f} hours")
        logger.info(f"Total Trades: {len(closed_trades)} closed, {len(open_trades)} open")
        logger.info(f"Win Rate: {win_rate:.1f}%")
        logger.info(f"Total P&L: ${total_pnl:+,.2f}")
        logger.info(f"Log files:")
        logger.info(f"  - JSON: {summary.get('json_log_path', 'N/A')}")
        logger.info(f"  - Text: {summary.get('text_log_path', 'N/A')}")
        logger.info(f"  - Audit: {summary.get('audit_log_path', 'N/A')}")
        logger.info("=" * 80)

        # Save trades to CSV
        if len(closed_trades) > 0:
            df_trades = pd.DataFrame(closed_trades)
            output_file = RESULTS_DIR / f"phase4_testnet_trading_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df_trades.to_csv(output_file, index=False)
            logger.success(f"\n✅ Trades saved to: {output_file}")

        # Save signal log for opportunity cost analysis
        if len(self.signal_log) > 0:
            df_signals = pd.DataFrame(self.signal_log)
            signal_file = RESULTS_DIR / f"phase4_testnet_signal_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df_signals.to_csv(signal_file, index=False)
            logger.success(f"✅ Signal log saved to: {signal_file}")

            # Print opportunity cost analysis
            missed_opps = df_signals[df_signals['missed_opportunity'] == True]
            if len(missed_opps) > 0:
                logger.warning(f"\n⚠️ MISSED OPPORTUNITIES DETECTED:")
                logger.warning(f"   Count: {len(missed_opps)}")
                logger.warning(f"   Avg Signal Delta: +{missed_opps['signal_strength_delta'].mean():.3f}")
                logger.warning(f"   Max Signal Delta: +{missed_opps['signal_strength_delta'].max():.3f}")
                logger.warning(f"   See {signal_file} for details")

    def _save_state(self):
        """Save bot state with trades persistence"""
        # Serialize trades list (convert datetime to ISO format)
        serializable_trades = []
        for trade in self.trades:
            trade_copy = trade.copy()
            # Convert datetime objects to ISO format strings
            if 'entry_time' in trade_copy and isinstance(trade_copy['entry_time'], datetime):
                trade_copy['entry_time'] = trade_copy['entry_time'].isoformat()
            if 'exit_time' in trade_copy and isinstance(trade_copy['exit_time'], datetime):
                trade_copy['exit_time'] = trade_copy['exit_time'].isoformat()
            serializable_trades.append(trade_copy)

        state = {
            "initial_balance": self.initial_balance,
            "current_balance": self._get_account_balance(use_cache=True),
            "trades": serializable_trades,  # Now persists full trades list
            "trades_count": len(self.trades),
            "closed_trades": len([t for t in self.trades if t.get('status') == 'CLOSED']),
            "bh_btc_quantity": self.bh_btc_quantity,
            "bh_entry_price": self.bh_entry_price,
            "session_start": self.session_start.isoformat(),
            "timestamp": datetime.now().isoformat(),
            "latest_signals": {
                "entry": {
                    "long_prob": self.latest_long_entry_prob,
                    "short_prob": self.latest_short_entry_prob,
                    "long_threshold": self.latest_long_entry_threshold,
                    "short_threshold": self.latest_short_entry_threshold,
                    "timestamp": self.latest_entry_timestamp.isoformat() if self.latest_entry_timestamp else None,
                    "threshold_context": {
                        # V3: ACTUAL ENTRY RATE context
                        "entry_rate": self.latest_threshold_entry_rate,  # Actual entry rate
                        "entries_count": self.latest_threshold_entries_count,  # Entries in lookback
                        "adjustment": self.latest_threshold_adjustment,  # Threshold delta
                        "adjustment_ratio": self.latest_threshold_adjustment_ratio,  # Ratio vs target
                        "base_long": Phase4TestnetConfig.BASE_LONG_ENTRY_THRESHOLD,
                        "base_short": Phase4TestnetConfig.BASE_SHORT_ENTRY_THRESHOLD,
                        "target_rate": Phase4TestnetConfig.TARGET_ENTRY_RATE,
                        "target_trades_per_week": Phase4TestnetConfig.TARGET_TRADES_PER_WEEK
                    }
                },
                "exit": {
                    "long_prob": self.latest_long_exit_prob,
                    "short_prob": self.latest_short_exit_prob,
                    "threshold": self.latest_exit_threshold,
                    "position_side": self.latest_exit_position_side,
                    "timestamp": self.latest_exit_timestamp.isoformat() if self.latest_exit_timestamp else None
                }
            }
        }

        def _json_serializer(obj):
            """Custom JSON serializer - only convert datetime, preserve floats"""
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            raise TypeError(f"Type {type(obj)} not serializable")

        state_file = RESULTS_DIR / "phase4_testnet_trading_state.json"
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2, default=_json_serializer)

# ============================================================================
# Main
# ============================================================================

def main():
    """Main entry point"""
    logger.info("=" * 80)
    logger.info("Phase 4 Dynamic Position Sizing TESTNET Trading Bot")
    logger.info("⚠️ WARNING: This bot executes REAL orders on BingX Testnet!")
    logger.info("=" * 80)

    # Check API credentials
    if not Phase4TestnetConfig.API_KEY or not Phase4TestnetConfig.API_SECRET:
        logger.error("BingX API credentials not found!")
        logger.error("Set BINGX_API_KEY and BINGX_API_SECRET environment variables")
        return

    # Check model exists
    model_path = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl"
    if not model_path.exists():
        logger.error(f"XGBoost Phase 4 model not found: {model_path}")
        return

    # Use Singleton pattern to prevent duplicate instances
    with BotSingleton():
        # Initialize and run bot
        try:
            bot = Phase4DynamicTestnetTradingBot()
            bot.run()
        except KeyboardInterrupt:
            logger.info("\n" + "=" * 80)
            logger.info("Bot stopped by user (Ctrl+C)")
            logger.info("=" * 80)
        except Exception as e:
            logger.error(f"Bot failed with exception: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
