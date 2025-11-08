"""
Hybrid Strategy Portfolio Manager

전략: 70% Buy & Hold + 30% XGBoost Trading

비판적 사고:
- "100% 한 쪽에 베팅하는 것은 위험"
- "Hybrid는 리스크 분산의 핵심"
- "상승장에서는 B&H, 하락/횡보에서는 XGB"

목적:
1. 리스크 분산 (70:30 비율)
2. 상승장: Buy & Hold 부분으로 안정적 수익
3. 하락/횡보: XGBoost 부분으로 방어
4. 자동 리밸런싱 및 성과 추적
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories
for dir_path in [RESULTS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Logging
log_file = LOGS_DIR / f"hybrid_strategy_{datetime.now().strftime('%Y%m%d')}.log"
logger.add(log_file, rotation="1 day", retention="30 days")

# ============================================================================
# Configuration
# ============================================================================

class HybridConfig:
    """Hybrid strategy configuration"""

    # Portfolio Allocation
    BUY_HOLD_PCT = 0.70  # 70% Buy & Hold
    XGBOOST_PCT = 0.30   # 30% XGBoost Trading

    # Rebalancing
    REBALANCE_THRESHOLD = 0.05  # Rebalance if deviation > 5%
    REBALANCE_FREQUENCY_DAYS = 7  # Weekly rebalancing

    # Capital
    INITIAL_CAPITAL = 1000.0  # $1000 starting capital
    MIN_CAPITAL = 100.0  # Minimum capital to continue

    # Risk Management
    MAX_PORTFOLIO_DRAWDOWN_PCT = 0.20  # 20% max portfolio drawdown
    STOP_LOSS_PORTFOLIO_PCT = 0.15  # Stop all trading if -15%

    # Performance Tracking
    UPDATE_INTERVAL = 3600  # 1 hour
    REPORT_FREQUENCY_HOURS = 24  # Daily reports

# ============================================================================
# Hybrid Strategy Manager
# ============================================================================

class HybridStrategyManager:
    """Manage 70% Buy & Hold + 30% XGBoost portfolio"""

    def __init__(self, initial_capital: float = HybridConfig.INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.capital = initial_capital

        # Portfolio components
        self.buy_hold_capital = initial_capital * HybridConfig.BUY_HOLD_PCT
        self.xgboost_capital = initial_capital * HybridConfig.XGBOOST_PCT

        # Buy & Hold position
        self.btc_quantity = 0.0
        self.btc_entry_price = 0.0

        # XGBoost trading (simulated - would integrate with paper_trading_bot)
        self.xgboost_trades = []
        self.xgboost_active = True

        # Performance tracking
        self.performance_history = []
        self.last_rebalance = datetime.now()
        self.session_start = datetime.now()

        # State
        self.is_initialized = False

        logger.info("=" * 80)
        logger.info("Hybrid Strategy Manager Initialized")
        logger.info("=" * 80)
        logger.info(f"Initial Capital: ${self.capital:,.2f}")
        logger.info(f"Buy & Hold: ${self.buy_hold_capital:,.2f} ({HybridConfig.BUY_HOLD_PCT * 100:.0f}%)")
        logger.info(f"XGBoost Trading: ${self.xgboost_capital:,.2f} ({HybridConfig.XGBOOST_PCT * 100:.0f}%)")
        logger.info("=" * 80)

    def initialize_buy_hold(self, current_btc_price: float):
        """Initialize Buy & Hold position"""
        self.btc_entry_price = current_btc_price
        self.btc_quantity = self.buy_hold_capital / current_btc_price
        self.is_initialized = True

        logger.success(f"🔔 Buy & Hold Initialized:")
        logger.info(f"   Bought {self.btc_quantity:.6f} BTC @ ${current_btc_price:,.2f}")
        logger.info(f"   Position Value: ${self.buy_hold_capital:,.2f}")

    def get_portfolio_value(self, current_btc_price: float, xgboost_capital: float = None) -> dict:
        """
        Calculate current portfolio value

        Args:
            current_btc_price: Current BTC price
            xgboost_capital: Current XGBoost trading capital (if None, use initial)

        Returns:
            dict with portfolio breakdown
        """
        if xgboost_capital is None:
            xgboost_capital = self.xgboost_capital

        # Buy & Hold value
        buy_hold_value = self.btc_quantity * current_btc_price if self.is_initialized else self.buy_hold_capital

        # Total portfolio
        total_value = buy_hold_value + xgboost_capital

        # Allocation percentages
        buy_hold_pct = (buy_hold_value / total_value) * 100 if total_value > 0 else 0
        xgboost_pct = (xgboost_capital / total_value) * 100 if total_value > 0 else 0

        # Returns
        buy_hold_return_pct = ((buy_hold_value - self.buy_hold_capital) / self.buy_hold_capital) * 100 if self.is_initialized else 0
        xgboost_return_pct = ((xgboost_capital - self.xgboost_capital) / self.xgboost_capital) * 100
        total_return_pct = ((total_value - self.initial_capital) / self.initial_capital) * 100

        return {
            'timestamp': datetime.now(),
            'btc_price': current_btc_price,
            'buy_hold_value': buy_hold_value,
            'xgboost_value': xgboost_capital,
            'total_value': total_value,
            'buy_hold_pct': buy_hold_pct,
            'xgboost_pct': xgboost_pct,
            'buy_hold_return_pct': buy_hold_return_pct,
            'xgboost_return_pct': xgboost_return_pct,
            'total_return_pct': total_return_pct
        }

    def check_rebalancing_needed(self, portfolio: dict) -> bool:
        """Check if portfolio rebalancing is needed"""
        # Time-based rebalancing
        days_since_rebalance = (datetime.now() - self.last_rebalance).days
        if days_since_rebalance >= HybridConfig.REBALANCE_FREQUENCY_DAYS:
            logger.info(f"Time-based rebalancing triggered ({days_since_rebalance} days)")
            return True

        # Threshold-based rebalancing
        target_buy_hold_pct = HybridConfig.BUY_HOLD_PCT * 100
        current_buy_hold_pct = portfolio['buy_hold_pct']
        deviation = abs(current_buy_hold_pct - target_buy_hold_pct)

        if deviation > HybridConfig.REBALANCE_THRESHOLD * 100:
            logger.info(f"Threshold-based rebalancing triggered (deviation: {deviation:.2f}%)")
            return True

        return False

    def rebalance(self, current_btc_price: float, xgboost_capital: float):
        """Rebalance portfolio to target allocation"""
        portfolio = self.get_portfolio_value(current_btc_price, xgboost_capital)
        total_value = portfolio['total_value']

        # Calculate target values
        target_buy_hold_value = total_value * HybridConfig.BUY_HOLD_PCT
        target_xgboost_value = total_value * HybridConfig.XGBOOST_PCT

        # Current values
        current_buy_hold_value = portfolio['buy_hold_value']
        current_xgboost_value = xgboost_capital

        # Calculate adjustments
        buy_hold_adjustment = target_buy_hold_value - current_buy_hold_value
        xgboost_adjustment = target_xgboost_value - current_xgboost_value

        logger.warning("🔄 REBALANCING PORTFOLIO")
        logger.info(f"Current Allocation: B&H {portfolio['buy_hold_pct']:.1f}% / XGB {portfolio['xgboost_pct']:.1f}%")
        logger.info(f"Target Allocation: B&H {HybridConfig.BUY_HOLD_PCT * 100:.0f}% / XGB {HybridConfig.XGBOOST_PCT * 100:.0f}%")

        if buy_hold_adjustment > 0:
            # Buy more BTC for Buy & Hold
            btc_to_buy = buy_hold_adjustment / current_btc_price
            self.btc_quantity += btc_to_buy
            logger.info(f"Buying {btc_to_buy:.6f} BTC for Buy & Hold (${buy_hold_adjustment:,.2f})")

        elif buy_hold_adjustment < 0:
            # Sell some BTC from Buy & Hold
            btc_to_sell = abs(buy_hold_adjustment) / current_btc_price
            self.btc_quantity -= btc_to_sell
            logger.info(f"Selling {btc_to_sell:.6f} BTC from Buy & Hold (${abs(buy_hold_adjustment):,.2f})")

        # Update XGBoost capital
        new_xgboost_capital = target_xgboost_value
        logger.info(f"XGBoost capital adjusted: ${current_xgboost_value:,.2f} → ${new_xgboost_capital:,.2f}")

        self.last_rebalance = datetime.now()

        # Return new XGBoost capital
        return new_xgboost_capital

    def check_stop_loss(self, portfolio: dict) -> bool:
        """Check if portfolio stop loss triggered"""
        total_return_pct = portfolio['total_return_pct']

        if total_return_pct <= -HybridConfig.STOP_LOSS_PORTFOLIO_PCT * 100:
            logger.error(f"⚠️ PORTFOLIO STOP LOSS TRIGGERED: {total_return_pct:.2f}%")
            logger.error("All trading halted")
            return True

        return False

    def print_portfolio_status(self, portfolio: dict):
        """Print current portfolio status"""
        logger.info("\n" + "=" * 80)
        logger.info("📊 HYBRID PORTFOLIO STATUS")
        logger.info("=" * 80)
        logger.info(f"Timestamp: {portfolio['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"BTC Price: ${portfolio['btc_price']:,.2f}")
        logger.info("")
        logger.info("Portfolio Composition:")
        logger.info(f"  Buy & Hold: ${portfolio['buy_hold_value']:,.2f} ({portfolio['buy_hold_pct']:.1f}%)")
        logger.info(f"  XGBoost Trading: ${portfolio['xgboost_value']:,.2f} ({portfolio['xgboost_pct']:.1f}%)")
        logger.info(f"  Total Value: ${portfolio['total_value']:,.2f}")
        logger.info("")
        logger.info("Returns:")
        logger.info(f"  Buy & Hold: {portfolio['buy_hold_return_pct']:+.2f}%")
        logger.info(f"  XGBoost Trading: {portfolio['xgboost_return_pct']:+.2f}%")
        logger.info(f"  Total Portfolio: {portfolio['total_return_pct']:+.2f}%")
        logger.info("")
        logger.info(f"vs Pure Buy & Hold: {self._calculate_vs_buy_hold(portfolio):.2f}%")
        logger.info("=" * 80)

    def _calculate_vs_buy_hold(self, portfolio: dict) -> float:
        """Calculate performance vs pure Buy & Hold"""
        # Pure Buy & Hold would have bought all capital initially
        if not self.is_initialized:
            return 0.0

        pure_buy_hold_btc = self.initial_capital / self.btc_entry_price
        pure_buy_hold_value = pure_buy_hold_btc * portfolio['btc_price']
        pure_buy_hold_return = ((pure_buy_hold_value - self.initial_capital) / self.initial_capital) * 100

        return portfolio['total_return_pct'] - pure_buy_hold_return

    def record_performance(self, portfolio: dict):
        """Record performance snapshot"""
        self.performance_history.append(portfolio.copy())

        # Save to file
        if len(self.performance_history) % 10 == 0:  # Save every 10 updates
            self._save_performance_history()

    def _save_performance_history(self):
        """Save performance history to CSV"""
        if len(self.performance_history) == 0:
            return

        df = pd.DataFrame(self.performance_history)
        output_file = RESULTS_DIR / f"hybrid_strategy_performance_{datetime.now().strftime('%Y%m%d')}.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Performance history saved: {output_file}")

    def save_state(self):
        """Save current state"""
        state = {
            'initial_capital': self.initial_capital,
            'current_capital': self.capital,
            'buy_hold_capital': self.buy_hold_capital,
            'xgboost_capital': self.xgboost_capital,
            'btc_quantity': self.btc_quantity,
            'btc_entry_price': self.btc_entry_price,
            'is_initialized': self.is_initialized,
            'last_rebalance': self.last_rebalance.isoformat(),
            'session_start': self.session_start.isoformat()
        }

        state_file = RESULTS_DIR / "hybrid_strategy_state.json"
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)

# ============================================================================
# Simulation / Demo
# ============================================================================

def run_demo():
    """Run demo simulation"""
    logger.info("Hybrid Strategy Demo Simulation")
    logger.info("비판적 사고: 'Hybrid는 리스크 분산의 핵심'")

    # Initialize manager
    manager = HybridStrategyManager(initial_capital=1000.0)

    # Simulate market data
    initial_btc_price = 60000.0
    manager.initialize_buy_hold(initial_btc_price)

    # Simulate price movements and XGBoost trading
    simulated_days = 30
    xgboost_capital = manager.xgboost_capital

    for day in range(1, simulated_days + 1):
        # Simulate BTC price (random walk)
        price_change_pct = np.random.normal(0.01, 0.03)  # 1% drift, 3% volatility
        current_btc_price = initial_btc_price * (1 + price_change_pct * day / 30)

        # Simulate XGBoost trading performance
        # Assume XGBoost slightly underperforms in bull market, outperforms in sideways/bear
        if price_change_pct > 0.02:  # Strong bull
            xgboost_daily_return = price_change_pct * 0.8  # 80% capture
        elif price_change_pct < -0.01:  # Bear
            xgboost_daily_return = price_change_pct * 0.5  # 50% downside protection
        else:  # Sideways
            xgboost_daily_return = 0.001  # Small positive

        xgboost_capital *= (1 + xgboost_daily_return / 30)

        # Get portfolio status
        portfolio = manager.get_portfolio_value(current_btc_price, xgboost_capital)

        # Print status every 7 days
        if day % 7 == 0:
            manager.print_portfolio_status(portfolio)

            # Check rebalancing
            if manager.check_rebalancing_needed(portfolio):
                xgboost_capital = manager.rebalance(current_btc_price, xgboost_capital)
                portfolio = manager.get_portfolio_value(current_btc_price, xgboost_capital)

        # Record performance
        manager.record_performance(portfolio)

        # Check stop loss
        if manager.check_stop_loss(portfolio):
            break

    # Final stats
    logger.info("\n" + "=" * 80)
    logger.info("DEMO SIMULATION COMPLETE")
    logger.info("=" * 80)
    final_portfolio = manager.get_portfolio_value(current_btc_price, xgboost_capital)
    manager.print_portfolio_status(final_portfolio)

    # Save results
    manager._save_performance_history()
    manager.save_state()

# ============================================================================
# Integration Instructions
# ============================================================================

def print_integration_guide():
    """Print guide for integrating with actual trading"""
    logger.info("""
    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║                     HYBRID STRATEGY INTEGRATION GUIDE                     ║
    ╚═══════════════════════════════════════════════════════════════════════════╝

    실제 사용 방법:

    1. Buy & Hold 부분 (70%):
       - 즉시 BTC 매수 ($700 예시)
       - 지갑에 보관 또는 거래소 보관
       - 주간 rebalancing 시에만 조정

    2. XGBoost Trading 부분 (30%):
       - paper_trading_bot.py 실행 ($300 예시)
       - 또는 실제 거래소 API 연동
       - 일일 성과 추적

    3. 통합 관리:
       - 이 스크립트로 전체 포트폴리오 추적
       - 주간 rebalancing 체크
       - 일일 성과 보고서

    4. 실행 예시:

       ```python
       # 초기화
       manager = HybridStrategyManager(initial_capital=1000.0)
       current_btc_price = get_current_btc_price()  # API call
       manager.initialize_buy_hold(current_btc_price)

       # 매일 업데이트
       while True:
           current_btc_price = get_current_btc_price()
           xgboost_capital = get_xgboost_capital()  # From paper_trading_bot

           portfolio = manager.get_portfolio_value(current_btc_price, xgboost_capital)
           manager.print_portfolio_status(portfolio)

           if manager.check_rebalancing_needed(portfolio):
               xgboost_capital = manager.rebalance(current_btc_price, xgboost_capital)

           manager.record_performance(portfolio)
           time.sleep(86400)  # Daily
       ```

    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║                              READY TO USE                                 ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """)

# ============================================================================
# Main
# ============================================================================

def main():
    """Main entry point"""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        run_demo()
    else:
        print_integration_guide()

if __name__ == "__main__":
    main()
