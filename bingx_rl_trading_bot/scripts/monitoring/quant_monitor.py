#!/usr/bin/env python3
"""
Professional Quantitative Trading Monitor
===========================================
Real-time monitoring with institutional-grade metrics

Features:
- Risk-adjusted performance metrics (Sharpe, Sortino, Calmar)
- Real-time risk analytics (VaR, CVaR, drawdown)
- Signal quality tracking & model diagnostics
- Execution quality monitoring (slippage, fill rate)
- Market regime analysis
- Anomaly detection
- Alert system for critical events
- ASCII visualization
"""

import json
import re
import time
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, List, Optional, Tuple
import numpy as np
import math
import yaml

# Configuration Auto-Sync Module (State file as Single Source of Truth)
from config_sync import (
    load_config_with_sync,
    ConfigurationSyncError,
    ConfigurationValidator,
    print_config_comparison
)


# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

LOGS_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "results"
CONFIG_DIR = PROJECT_ROOT / "config"
STATE_FILE = RESULTS_DIR / "opportunity_gating_bot_4x_state.json"

# Import BingX client
from src.api.bingx_client import BingXClient

# Monitoring parameters
# ✅ OPTIMIZED 2025-10-25: Safe API rate limit management
# API calls per refresh: 3 (balance + position + ticker)
# Current rate: 6 calls/min @ 30s refresh (0.5% of typical 1200/min limit)
# Recommended: 30-60s for safety, 10s minimum for active monitoring
REFRESH_INTERVAL = 30  # seconds (adjustable via command line)
MAX_HISTORY = 1000  # Keep last N data points
RISK_FREE_RATE = 0.04  # 4% annual (for Sharpe calculation)
SYMBOL = "BTC-USDT"  # Trading symbol for API queries

# Opportunity Gating Expected Values (ROLLBACK 2025-10-30 - ENHANCED MODELS) ✅
# Source: Enhanced Entry + oppgating Exit - 108-window backtest (540 days)
# Entry Models: xgboost_*_entry_enhanced_20251024_012445.pkl (85/79 features)
# Exit Models: xgboost_*_exit_oppgating_improved_20251024_04XXXX.pkl (27 features)
# Configuration: Entry 0.80/0.80, Exit 0.75/0.75, SL -3%, Max Hold 120 candles
# Backtest (Exit 0.80): 540 days, 2,506 trades, +25.21% return/window, 72.3% WR
# Note: Exit 0.75 performance TBD - monitoring actual vs expected (may differ from 0.80)
# Rationale: Rollback from Walk-Forward (-9.69%) to proven Enhanced models (+25.21%)
EXPECTED_RETURN_5D = 0.2521  # 25.21% per 5-day window (Exit 0.80 baseline - TBD for 0.75)
EXPECTED_WIN_RATE = 0.723    # 72.3% (Exit 0.80 baseline - TBD for 0.75)
EXPECTED_TRADES_PER_DAY = 4.6    # 4.64 trades/day (2,506 trades / 540 days)
EXPECTED_LONG_PCT = 0.618    # 61.8% LONG trades (1,549 / 2,506)
EXPECTED_SHORT_PCT = 0.382   # 38.2% SHORT trades (957 / 2,506)
EXPECTED_SHARPE = 6.610      # Annualized Sharpe Ratio (Exit 0.80 baseline)
EXPECTED_GATE_BLOCK_RATE = 0.382  # 38.2% SHORT entry rate (from backtest)

# Alert thresholds (adjusted for Enhanced models - UPDATED 2025-10-30)
ALERT_MAX_DRAWDOWN = 0.15  # 15% (conservative threshold)
ALERT_MIN_SHARPE = 3.5      # ~53% of expected 6.610 (conservative threshold)
ALERT_POSITION_RISK = 0.90  # 90% of balance
ALERT_MIN_WIN_RATE = 0.60   # 60% minimum (expected 72.3%, allow 17% degradation)
ALERT_SHORT_RATIO_MIN = 0.25  # SHORT < 25% (expected 38.2% - LONG-heavy bias)
ALERT_SHORT_RATIO_MAX = 0.55  # SHORT > 55% (expected 38.2% - monitor for shift)
ALERT_GATE_BLOCK_MIN = 0.50   # Gate blocking < 50% (expected ~61.8% LONG)
ALERT_GATE_BLOCK_MAX = 0.75   # Gate blocking > 75% (expected ~61.8% LONG)
ALERT_TRADES_PER_DAY_MIN = 3.0  # < 3.0 trades/day (expected 4.6/day)


# ============================================================================
# Data Structures
# ============================================================================

class TradingMetrics:
    """Container for trading metrics"""
    def __init__(self, config: Optional[Dict] = None):
        self.returns = deque(maxlen=MAX_HISTORY)
        self.trades = []
        self.prices = deque(maxlen=MAX_HISTORY)
        self.timestamps = deque(maxlen=MAX_HISTORY)

        # Store configuration (Single Source of Truth)
        # Configuration is loaded from state file (Single Source of Truth)
        # No hardcoded defaults - config must be provided from load_state()
        if config is None:
            raise ValueError(
                "Configuration must be provided (loaded from state file).\n"
                "Call load_state() first to get config from production bot's state file.\n"
                "Hardcoded defaults have been removed for auto-sync architecture."
            )
        self.config = config

        # Performance
        self.total_return = 0.0
        self.realized_return = 0.0      # From closed trades only
        self.unrealized_return = 0.0    # From open positions
        self.balance_change = 0.0       # Balance change (includes fees)
        self.balance_change_pct = 0.0   # Balance change percentage
        self.sharpe_ratio = 0.0
        self.sortino_ratio = 0.0
        self.calmar_ratio = 0.0
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0

        # Trading stats
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.win_rate = 0.0
        self.profit_factor = 0.0
        self.avg_win = 0.0
        self.avg_loss = 0.0
        self.largest_win = 0.0
        self.largest_loss = 0.0

        # Risk metrics
        self.var_95 = 0.0  # 95% Value at Risk
        self.cvar_95 = 0.0  # 95% Conditional VaR
        self.current_exposure = 0.0
        self.avg_holding_time = 0.0

        # Signal quality
        self.signal_rate = 0.0
        self.signal_accuracy = 0.0
        self.false_positive_rate = 0.0
        self.model_confidence_avg = 0.0

        # Execution quality
        self.avg_slippage = 0.0
        self.fill_rate = 1.0

        # Market regime
        self.volatility = 0.0
        self.trend_strength = 0.0
        self.regime = "Unknown"

        # Opportunity Gating specific metrics
        self.long_trades = 0
        self.short_trades = 0
        self.long_pct = 0.0
        self.short_pct = 0.0
        self.short_signals_total = 0
        self.short_signals_blocked = 0
        self.gate_block_rate = 0.0
        self.leverage = self.config.get('leverage', 4)  # From config
        self.avg_position_size = 0.0
        self.trades_per_day = 0.0
        self.session_start_time = None
        self.days_running = 0.0

        # Fee tracking (from exchange API)
        self.total_fees = 0.0
        self.entry_fees = 0.0
        self.exit_fees = 0.0

        # Deposits/Withdrawals detection
        self.deposits_withdrawals_detected = 0.0


# ============================================================================
# API Client Initialization
# ============================================================================

def init_api_client() -> Optional[BingXClient]:
    """Initialize BingX API client (matches production bot configuration)"""
    try:
        # Load API keys
        api_keys_file = CONFIG_DIR / "api_keys.yaml"
        if not api_keys_file.exists():
            print(f"⚠️  API keys file not found: {api_keys_file}")
            return None

        with open(api_keys_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # Match production bot: USE_TESTNET = False (mainnet)
        use_testnet = False

        # Read from mainnet config (since production bot uses mainnet)
        api_config = config.get('bingx', {}).get('mainnet', {})
        api_key = api_config.get('api_key')
        api_secret = api_config.get('secret_key')

        if not api_key or not api_secret or api_key == 'your_mainnet_api_key_here':
            print("⚠️  API credentials not configured in config/api_keys.yaml")
            print("   Update bingx.mainnet.api_key and bingx.mainnet.secret_key")
            return None

        # Initialize client
        client = BingXClient(api_key, api_secret, testnet=use_testnet)
        return client

    except Exception as e:
        print(f"⚠️  Failed to initialize API client: {e}")
        return None


def fetch_realtime_data(client: BingXClient, symbol: str = "BTC-USDT") -> Optional[Dict]:
    """Fetch real-time position and balance from exchange

    Returns:
        Dict with keys: equity, balance, position, current_price, timestamp
    """
    if not client:
        return None

    try:
        # Get balance (includes both wallet balance and equity from BingX API)
        balance_data = client.get_balance()
        balance_info = balance_data.get('balance', {})

        # ✅ FIXED 2025-10-25: Use BingX API's equity directly (already includes unrealized P&L)
        # - equity: Total account value (realized + unrealized)
        # - balance: Wallet balance (realized only)
        equity = float(balance_info.get('equity', 0))
        wallet_balance = float(balance_info.get('balance', 0))

        # Get positions
        positions = client.get_positions(symbol)
        position = None
        if positions and len(positions) > 0:
            pos = positions[0]
            # Convert to our format
            position = {
                'side': 'LONG' if float(pos.get('positionAmt', 0)) > 0 else 'SHORT',
                'quantity': abs(float(pos.get('positionAmt', 0))),
                'entry_price': float(pos.get('avgPrice', 0)),
                'position_value': abs(float(pos.get('positionAmt', 0))) * float(pos.get('avgPrice', 0)),
                'unrealized_pnl': float(pos.get('unrealizedProfit', 0)),
                'leverage': int(float(pos.get('leverage', 1)))  # Fix: convert to float first, then int
            }

        # Get current price
        ticker = client.get_ticker(symbol)
        current_price = float(ticker.get('lastPrice', 0))

        return {
            'equity': equity,  # BingX API's equity (realized + unrealized)
            'balance': wallet_balance,  # Wallet balance (realized only)
            'position': position,
            'current_price': current_price,
            'timestamp': datetime.now().isoformat(),
            'source': 'exchange_api'
        }

    except Exception as e:
        print(f"⚠️  Failed to fetch realtime data: {e}")
        return None


def fetch_order_fees(client: BingXClient, state: Dict, symbol: str = "BTC-USDT") -> Dict:
    """Fetch actual fees from exchange for all orders in current session

    Returns:
        Dict with keys: total_fees, entry_fees, exit_fees, order_details
    """
    if not client or not state:
        return {'total_fees': 0, 'entry_fees': 0, 'exit_fees': 0, 'order_details': []}

    try:
        # Collect all order IDs from trades
        order_ids = []
        trades = state.get('trades', [])

        for trade in trades:
            # Entry order
            entry_id = trade.get('order_id')
            if entry_id:
                order_ids.append(('entry', entry_id, trade.get('side', 'UNKNOWN')))

            # Exit order (if closed)
            if trade.get('status') == 'CLOSED':
                close_id = trade.get('close_order_id')
                if close_id and close_id != 'N/A':
                    order_ids.append(('exit', close_id, trade.get('side', 'UNKNOWN')))

        # Fetch fees for each order
        total_fees = 0.0
        entry_fees = 0.0
        exit_fees = 0.0
        order_details = []

        for order_type, order_id, side in order_ids:
            try:
                order = client.exchange.fetch_order(order_id, symbol)
                fee_info = order.get('fee', {})
                fee_cost = fee_info.get('cost', 0)
                fee_cost = float(fee_cost) if fee_cost else 0.0

                total_fees += fee_cost
                if order_type == 'entry':
                    entry_fees += fee_cost
                else:
                    exit_fees += fee_cost

                order_details.append({
                    'order_id': order_id,
                    'type': order_type,
                    'side': side,
                    'fee': fee_cost,
                    'currency': fee_info.get('currency', 'USDT')
                })
            except Exception as e:
                # Order not found or error - skip
                pass

        return {
            'total_fees': total_fees,
            'entry_fees': entry_fees,
            'exit_fees': exit_fees,
            'order_details': order_details
        }

    except Exception as e:
        return {'total_fees': 0, 'entry_fees': 0, 'exit_fees': 0, 'order_details': []}


def extract_fees_from_state(state: Dict) -> Dict:
    """Extract fee data from state file

    Extracts fees from both 'total_fee' (V2 reconciliation) and separate
    'entry_fee'/'exit_fee' fields (production bot format).

    Returns:
        Dict with keys: total_fees, entry_fees, exit_fees, order_details
    """
    if not state:
        return {'total_fees': 0, 'entry_fees': 0, 'exit_fees': 0, 'order_details': []}

    try:
        trades = state.get('trades', [])
        total_fees = 0.0
        entry_fees = 0.0
        exit_fees = 0.0
        order_details = []

        for trade in trades:
            # Try to get fees from different fields
            # Priority: total_fee (V2) > entry_fee + exit_fee (production bot)
            total_fee = trade.get('total_fee', 0)
            entry_fee = trade.get('entry_fee', 0)
            exit_fee = trade.get('exit_fee', 0)

            # Convert to float
            total_fee = float(total_fee) if total_fee else 0.0
            entry_fee = float(entry_fee) if entry_fee else 0.0
            exit_fee = float(exit_fee) if exit_fee else 0.0

            # Calculate trade fees
            if total_fee > 0:
                # V2 reconciliation format (total_fee field)
                trade_total = total_fee
                trade_entry = total_fee / 2
                trade_exit = total_fee / 2
                source = 'v2_reconciliation'
            else:
                # Production bot format (entry_fee/exit_fee fields)
                trade_entry = entry_fee
                trade_exit = exit_fee
                trade_total = entry_fee + exit_fee
                source = 'production_bot'

            # Add to totals
            if trade_total > 0:
                total_fees += trade_total
                entry_fees += trade_entry
                exit_fees += trade_exit

                order_details.append({
                    'order_id': trade.get('order_id', 'N/A'),
                    'type': trade.get('status', 'UNKNOWN'),
                    'side': trade.get('side', 'UNKNOWN'),
                    'fee': trade_total,
                    'entry_fee': trade_entry,
                    'exit_fee': trade_exit,
                    'currency': 'USDT',
                    'source': source
                })

        return {
            'total_fees': total_fees,
            'entry_fees': entry_fees,
            'exit_fees': exit_fees,
            'order_details': order_details
        }

    except Exception as e:
        return {'total_fees': 0, 'entry_fees': 0, 'exit_fees': 0, 'order_details': []}


# ============================================================================
# Data Collection
# ============================================================================

def find_latest_log() -> Optional[Path]:
    """Find the most recent log file"""
    try:
        log_files = sorted(
            LOGS_DIR.glob("opportunity_gating_bot_4x_*.log"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        return log_files[0] if log_files else None
    except Exception as e:
        print(f"Error finding log: {e}")
        return None


def load_trading_state() -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Load current trading state and configuration from JSON with AUTO-SYNC.

    Architecture: State file as Single Source of Truth (SSOT)
    - Production bot writes configuration to state file
    - Monitoring program reads configuration from state file
    - No hardcoded defaults (except emergency fallback)
    - Configuration changes propagate automatically

    Returns:
        (state, config) tuple or (None, None) if failed
    """
    try:
        # Load configuration using auto-sync module (State file = SSOT)
        config, source = load_config_with_sync(STATE_FILE)

        # Display configuration source
        if source == "EMERGENCY_FALLBACK":
            print("="*80)
            print("⚠️ WARNING: Using EMERGENCY FALLBACK configuration")
            print("   Production bot state file not found or corrupted")
            print("   Configuration may be outdated!")
            print("="*80)

        # Load full state
        if STATE_FILE.exists():
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
                return state, config
        else:
            # State file missing - return None for state, but config from fallback
            print("⚠️ State file not found, but configuration loaded from emergency fallback")
            return None, config

    except ConfigurationSyncError as e:
        print(f"❌ Configuration sync failed: {e}")
        print("   Cannot proceed without valid configuration")
        return None, None
    except Exception as e:
        print(f"❌ Error loading state: {e}")
        return None, None


def parse_log_metrics(log_file: Path, metrics: TradingMetrics) -> None:
    """Extract metrics from log file - only latest unique values"""
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        # Get last 500 lines for gate effectiveness tracking
        recent_lines = lines[-500:] if len(lines) > 500 else lines

        # Track seen prices to avoid duplicates
        seen_prices = set()
        regime_found = False
        short_signals = 0
        gate_blocks = 0

        for line in reversed(recent_lines):  # Read backwards for most recent
            # Price (only add unique values)
            # Actual log format: "Price: $107,950.1 | Balance: ..."
            if "Price:" in line and len(seen_prices) < 30:
                match = re.search(r'Price:\s*\$([0-9,]+\.\d+)', line)
                if match:
                    price = float(match.group(1).replace(',', ''))
                    if price not in seen_prices:
                        metrics.prices.append(price)
                        seen_prices.add(price)

            # Regime parsing removed - bot doesn't log market regime
            # Keeping regime as "Unknown" (default in TradingMetrics)

        # Count gate blocks in ALL lines (not just recent_lines)
        for line in lines:
            # SHORT signal above threshold
            if "SHORT: 0.7" in line or "SHORT: 0.8" in line or "SHORT: 0.9" in line:
                short_signals += 1

            # Gate blocked
            if "blocked by gate" in line.lower() or "gate check failed" in line.lower():
                gate_blocks += 1

        # Update gate metrics
        metrics.short_signals_total = short_signals
        metrics.short_signals_blocked = gate_blocks
        if short_signals > 0:
            metrics.gate_block_rate = gate_blocks / short_signals

    except Exception as e:
        print(f"Error parsing log: {e}")


def calculate_metrics(state: Dict, metrics: TradingMetrics, fee_data: Optional[Dict] = None, realtime_data: Optional[Dict] = None) -> None:
    """Calculate all trading metrics from state and optionally real-time API data

    Args:
        state: Bot state from JSON file
        metrics: TradingMetrics object to update
        fee_data: Optional fee data from exchange API (total_fees, entry_fees, exit_fees)
        realtime_data: Optional real-time data from exchange API (balance, unrealized_pnl)
                      If provided, overrides state file values for current balance/unrealized
    """
    if not state:
        return

    try:
        # Basic stats
        initial_balance = state.get('initial_balance', 100000)  # Initial equity

        # ✅ FIXED 2025-10-25 18:40: Get baseline values for proper return calculation
        initial_wallet_balance = state.get('initial_wallet_balance', initial_balance)  # Wallet at reset
        initial_unrealized_pnl = state.get('initial_unrealized_pnl', 0)  # Unrealized at reset

        # ✅ FIXED 2025-10-25: Use BingX API's equity directly (no duplicate calculation)
        # Priority: Realtime API > State File
        if realtime_data and realtime_data.get('source') == 'exchange_api':
            # BingX API provides both equity and wallet balance
            net_balance = realtime_data.get('equity', 0)  # Equity (realized + unrealized)
            current_balance = realtime_data.get('balance', 0)  # Wallet balance (realized only)
        else:
            # Fallback to state file (calculate equity from balance + unrealized)
            current_balance = state.get('current_balance', initial_balance)

        trades = state.get('trades', [])
        # Include exchange-reconciled trades (accurate P&L from exchange)
        # Exclude only pure manual trades (manual_trade=True AND NOT exchange_reconciled)
        closed_trades = [t for t in trades
                         if t.get('status') == 'CLOSED'
                         and (not t.get('manual_trade', False) or t.get('exchange_reconciled', False))]

        # Total trades: use actual count from trades array (don't trust state['closed_trades'])
        metrics.total_trades = len(closed_trades)

        # Unrealized P&L from positions
        # ✅ FIXED 2025-10-25: Use API data when available
        if realtime_data and realtime_data.get('source') == 'exchange_api':
            # Get unrealized from API (position data)
            position = realtime_data.get('position')
            unrealized_pnl = position.get('unrealized_pnl', 0) if position else 0
        else:
            unrealized_pnl = state.get('unrealized_pnl', 0)
            # Calculate equity from state file
            net_balance = current_balance + unrealized_pnl

        # Get fees from exchange API or default to 0
        total_fees = fee_data.get('total_fees', 0) if fee_data else 0
        closed_trade_fees = fee_data.get('entry_fees', 0) + fee_data.get('exit_fees', 0) if fee_data else 0

        # Calculate returns
        if initial_balance > 0:
            # ✅ FIXED 2025-10-25: Use equity directly from API (no duplicate calculation)
            # net_balance is already calculated above based on data source:
            # - From API: net_balance = BingX equity (already includes unrealized)
            # - From State: net_balance = current_balance + unrealized_pnl

            # Unrealized return (from open positions)
            # ✅ FIXED 2025-10-25 18:40: Compare current unrealized to initial unrealized
            metrics.unrealized_return = (unrealized_pnl - initial_unrealized_pnl) / initial_balance

            # Realized return: From CLOSED TRADES ONLY (actual P&L from completed trades)
            # This is the sum of P&L from all closed trades
            # Does NOT include fees, funding fees, or deposits/withdrawals
            # ✅ CORRECTED 2025-10-25 18:45: Reverted to original logic (only closed trades)
            total_realized_pnl = sum(trade.get('pnl_usd', 0) for trade in closed_trades)
            metrics.realized_return = total_realized_pnl / initial_balance if initial_balance > 0 else 0

            # Total return = Total account return (net_balance - initial_balance) / initial_balance
            # This accounts for ALL balance changes: trades, fees, funding, deposits, etc.
            # ✅ FIXED 2025-10-25: Was incorrectly using realized_return + unrealized_return
            #    which only counted trading P&L and ignored other balance changes
            metrics.total_return = (net_balance - initial_balance) / initial_balance

            # ✅ FIXED 2025-10-31: Auto-detect deposits/withdrawals
            # Balance change should only reflect trading P&L + fees (not deposits/withdrawals)

            # Calculate total trading P&L (net, after fees)
            total_trading_pnl_net = sum(trade.get('pnl_usd_net', 0) for trade in closed_trades)

            # Expected balance = initial + trading P&L (net)
            expected_balance = initial_wallet_balance + total_trading_pnl_net

            # Detect deposits/withdrawals (balance change not explained by trading)
            detected_deposits_withdrawals = current_balance - expected_balance

            # Balance change (Trading P&L + Fees only, excludes deposits/withdrawals)
            # This represents actual trading performance
            metrics.balance_change = current_balance - initial_wallet_balance - detected_deposits_withdrawals
            metrics.balance_change_pct = metrics.balance_change / initial_balance if initial_balance > 0 else 0

            # Store detection info for display
            metrics.deposits_withdrawals_detected = detected_deposits_withdrawals

            # Verification: total_return calculation
            # total_return = (net_balance - initial_balance) / initial_balance
            # This includes: trading P&L, fees, funding fees, deposits, withdrawals, etc.
            # It represents the true total account performance

        # Store fee information in metrics for display
        metrics.total_fees = total_fees
        metrics.entry_fees = fee_data.get('entry_fees', 0) if fee_data else 0
        metrics.exit_fees = fee_data.get('exit_fees', 0) if fee_data else 0

        # Trade statistics
        if closed_trades:
            winning = [t for t in closed_trades if t.get('pnl_usd_net', 0) > 0]
            losing = [t for t in closed_trades if t.get('pnl_usd_net', 0) <= 0]

            metrics.winning_trades = len(winning)
            metrics.losing_trades = len(losing)
            metrics.win_rate = metrics.winning_trades / len(closed_trades)

            if winning:
                metrics.avg_win = np.mean([t['pnl_usd_net'] for t in winning])
                metrics.largest_win = max([t['pnl_usd_net'] for t in winning])

            if losing:
                metrics.avg_loss = abs(np.mean([t['pnl_usd_net'] for t in losing]))
                metrics.largest_loss = abs(min([t['pnl_usd_net'] for t in losing]))

            # Profit factor
            total_profit = sum([t['pnl_usd_net'] for t in winning]) if winning else 0
            total_loss = abs(sum([t['pnl_usd_net'] for t in losing])) if losing else 1
            metrics.profit_factor = total_profit / total_loss if total_loss > 0 else 0

            # Returns for risk-adjusted metrics
            returns = np.array([t.get('pnl_pct', 0) for t in closed_trades])

            # Sharpe Ratio (annualized)
            if len(returns) > 1:
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                if std_return > 0:
                    # Assume 37.3 trades/week (V4 target), 52 weeks/year
                    trades_per_year = 37.3 * 52
                    sharpe_annual = (avg_return * trades_per_year - RISK_FREE_RATE) / (std_return * np.sqrt(trades_per_year))
                    metrics.sharpe_ratio = sharpe_annual

            # Sortino Ratio (downside deviation)
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 1:
                downside_std = np.std(downside_returns)
                if downside_std > 0:
                    trades_per_year = 37.3 * 52
                    sortino_annual = (np.mean(returns) * trades_per_year - RISK_FREE_RATE) / (downside_std * np.sqrt(trades_per_year))
                    metrics.sortino_ratio = sortino_annual

            # Maximum Drawdown
            cumulative = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / (1 + running_max)
            metrics.max_drawdown = abs(np.min(drawdown)) if len(drawdown) > 0 else 0
            metrics.current_drawdown = abs(drawdown[-1]) if len(drawdown) > 0 else 0

            # Calmar Ratio
            if metrics.max_drawdown > 0:
                annual_return = metrics.total_return * (37.3 * 52 / len(closed_trades))
                metrics.calmar_ratio = annual_return / metrics.max_drawdown

            # VaR and CVaR (95% confidence)
            if len(returns) > 10:
                metrics.var_95 = abs(np.percentile(returns, 5))
                var_threshold = np.percentile(returns, 5)
                tail_losses = returns[returns <= var_threshold]
                metrics.cvar_95 = abs(np.mean(tail_losses)) if len(tail_losses) > 0 else metrics.var_95

            # Average holding time
            holding_times = []
            for t in closed_trades:
                if 'entry_time' in t and 'exit_time' in t:
                    try:
                        entry = datetime.fromisoformat(t['entry_time'])
                        exit_time = datetime.fromisoformat(t['exit_time'])
                        duration = (exit_time - entry).total_seconds() / 3600  # hours
                        holding_times.append(duration)
                    except:
                        pass

            if holding_times:
                metrics.avg_holding_time = np.mean(holding_times)

        # Current position exposure
        # PRIORITY 1: Check 'position' field first (synced from exchange)
        current_position = state.get('position')
        if current_position and current_position.get('status') == 'OPEN':
            position_value = current_position.get('position_value', 0)
            if position_value > 0 and current_balance > 0:
                metrics.current_exposure = position_value / current_balance
        else:
            # PRIORITY 2: Fallback to trades array
            # Include exchange-reconciled trades
            open_trades = [t for t in trades
                           if t.get('status') == 'OPEN'
                           and (not t.get('manual_trade', False) or t.get('exchange_reconciled', False))]
            if open_trades and current_balance > 0:
                total_exposure = sum([t.get('position_value', 0) for t in open_trades])
                metrics.current_exposure = total_exposure / current_balance

        # Signal rate (recent 6 hours)
        # This would need to be calculated from log timestamps
        # For now, use a placeholder
        metrics.signal_rate = 0.0612  # Expected rate from analysis

        # Volatility (if we have price history)
        if len(metrics.prices) > 20:
            price_returns = np.diff(list(metrics.prices)) / list(metrics.prices)[:-1]
            metrics.volatility = np.std(price_returns) * np.sqrt(252 * 24)  # Annualized

        # Opportunity Gating specific calculations
        if closed_trades:
            # LONG/SHORT distribution
            long_trades_list = [t for t in closed_trades if t.get('side') == 'LONG']
            short_trades_list = [t for t in closed_trades if t.get('side') == 'SHORT']

            metrics.long_trades = len(long_trades_list)
            metrics.short_trades = len(short_trades_list)

            if len(closed_trades) > 0:
                metrics.long_pct = metrics.long_trades / len(closed_trades)
                metrics.short_pct = metrics.short_trades / len(closed_trades)

            # Average position size
            position_sizes = [t.get('position_size_pct', 0) for t in closed_trades if t.get('position_size_pct')]
            if position_sizes:
                metrics.avg_position_size = np.mean(position_sizes)

        # Session duration and trades per day
        session_start = state.get('session_start', '')
        if session_start:
            try:
                start_time = datetime.fromisoformat(session_start)
                metrics.session_start_time = start_time
                duration = datetime.now() - start_time
                metrics.days_running = duration.total_seconds() / (24 * 3600)

                if metrics.days_running > 0:
                    metrics.trades_per_day = metrics.total_trades / metrics.days_running
            except:
                pass

        # Gate effectiveness (from log parsing - will be updated in parse_log_metrics)
        # Placeholder for now - actual data comes from logs

    except Exception as e:
        print(f"Error calculating metrics: {e}")


# ============================================================================
# Visualization
# ============================================================================

def clear_screen():
    """Clear terminal screen (cross-platform)"""
    os.system('cls' if os.name == 'nt' else 'clear')


def create_bar_chart(value: float, max_val: float = 1.0, width: int = 30) -> str:
    """Create ASCII bar chart"""
    if max_val == 0:
        return "[" + " " * width + "]"

    filled = int((value / max_val) * width)
    filled = min(width, max(0, filled))
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}]"


def create_sparkline(values: List[float], width: int = 40) -> str:
    """Create ASCII sparkline"""
    if not values or len(values) < 2:
        return "─" * width

    min_val = min(values)
    max_val = max(values)
    value_range = max_val - min_val

    if value_range == 0:
        return "─" * width

    # Normalize to 0-8 range (for ticks)
    ticks = "▁▂▃▄▅▆▇█"
    sparkline = ""

    step = len(values) / width
    for i in range(width):
        idx = int(i * step)
        normalized = (values[idx] - min_val) / value_range
        tick_idx = int(normalized * (len(ticks) - 1))
        sparkline += ticks[tick_idx]

    return sparkline


def format_percentage(value: float, decimal: int = 2, width: int = 0) -> str:
    """Format percentage with color and optional width"""
    formatted = f"{value*100:+.{decimal}f}%"

    # Add padding if width specified
    if width > 0:
        formatted = f"{formatted:>{width}s}"

    # Apply color
    if value > 0:
        return f"\033[92m{formatted}\033[0m"  # Green
    elif value < 0:
        return f"\033[91m{formatted}\033[0m"  # Red
    else:
        return formatted


def format_metric(value: float, good_threshold: float, bad_threshold: float,
                  higher_is_better: bool = True, width: int = 0) -> str:
    """Format metric with color based on thresholds and optional width"""
    formatted = f"{value:.2f}"

    # Add padding if width specified
    if width > 0:
        formatted = f"{formatted:>{width}s}"

    # Apply color
    if higher_is_better:
        if value >= good_threshold:
            color = "\033[92m"  # Green
        elif value >= bad_threshold:
            color = "\033[93m"  # Yellow
        else:
            color = "\033[91m"  # Red
    else:
        if value <= good_threshold:
            color = "\033[92m"  # Green
        elif value <= bad_threshold:
            color = "\033[93m"  # Yellow
        else:
            color = "\033[91m"  # Red

    return f"{color}{formatted}\033[0m"


def format_signal_probability(prob: float, threshold: float, show_threshold: bool = True) -> str:
    """Format signal probability with color based on threshold proximity

    Args:
        prob: Current probability value
        threshold: Threshold for entry/exit
        show_threshold: Whether to show threshold in output

    Returns:
        Colored formatted string
    """
    # Calculate percentage of threshold
    pct = (prob / threshold * 100) if threshold > 0 else 0

    # Determine color based on threshold proximity
    if prob >= threshold:
        # Above threshold - ready to enter/exit
        color = "\033[1;92m"  # Bright green (bold)
    elif pct >= 85:
        # 85-99% of threshold - very close
        color = "\033[92m"  # Green
    elif pct >= 70:
        # 70-84% of threshold - approaching
        color = "\033[93m"  # Yellow
    elif pct >= 50:
        # 50-69% of threshold - moderate
        color = "\033[0m"   # White (default)
    else:
        # < 50% of threshold - far away
        color = "\033[91m"  # Red

    # Format output
    if show_threshold:
        return f"{color}{prob:.3f}/{threshold:.2f}\033[0m"
    else:
        return f"{color}{prob:.3f}\033[0m"


# ============================================================================
# Opportunity Gating Specific Display Functions
# ============================================================================

def display_strategy_info(metrics: TradingMetrics) -> None:
    """Display strategy information section (matches production bot output)"""
    config = metrics.config
    long_thresh = config.get('long_threshold', 0.80)  # UPDATED 2025-10-30: Default 0.80 (Threshold 0.80)
    short_thresh = config.get('short_threshold', 0.80)  # UPDATED 2025-10-30: Default 0.80 (Threshold 0.80)
    gate_thresh = config.get('gate_threshold', 0.001)
    leverage = config.get('leverage', 4)
    ml_exit_long = config.get('ml_exit_threshold_base_long', 0.70)  # UPDATED 2025-11-02: Default 0.70 (Phase 1 Optimized)
    ml_exit_short = config.get('ml_exit_threshold_base_short', 0.70)  # UPDATED 2025-11-02: Default 0.70 (Phase 1 Optimized)
    stop_loss = config.get('emergency_stop_loss', -0.03)  # UPDATED 2025-10-22: -3% balance-based SL
    max_hold = config.get('emergency_max_hold_hours', 10)  # UPDATED 2025-10-22: 10 hours (120 candles)
    take_profit = config.get('fixed_take_profit', 0.03)
    trailing_activation = config.get('trailing_tp_activation', 0.02)
    trailing_drawdown = config.get('trailing_tp_drawdown', 0.10)
    vol_high_thresh = config.get('ml_threshold_high_vol', 0.65)
    vol_low_thresh = config.get('ml_threshold_low_vol', 0.75)
    exit_strategy = config.get('exit_strategy', 'COMBINED')

    print("\n┌─ STRATEGY: OPPORTUNITY GATING + 4x LEVERAGE " + "─"*57 + "┐")
    print(f"│ Strategy           : Opportunity Gating (SHORT gated by Expected Value)                 │")
    print(f"│ Leverage           : {leverage}x (BOTH mode)  │  Gate Threshold: {gate_thresh:.3f} ({gate_thresh*100:.1f}% opportunity cost)  │")
    print(f"│ Entry Thresholds   : LONG: {long_thresh:.2f}  │  SHORT: {short_thresh:.2f}  │  Gate: EV(SHORT) > EV(LONG) + {gate_thresh:.3f}│")
    print(f"│ Exit Strategy      : ML Exit + Emergency Rules (ML: {ml_exit_long:.2f}/{ml_exit_short:.2f}, SL: {stop_loss*100:+.1f}%, MaxHold: {max_hold:.0f}h)              │")
    print(f"│                                                                                          │")
    print(f"│ Expected Return    : {EXPECTED_RETURN_5D*100:>5.2f}% per 5 days  │  Win Rate: {EXPECTED_WIN_RATE*100:>4.1f}%  │  Sharpe: {EXPECTED_SHARPE:>5.2f}    │")
    print(f"│ Expected Mix       : LONG: {EXPECTED_LONG_PCT*100:>4.1f}%  │  SHORT: {EXPECTED_SHORT_PCT*100:>4.1f}%  │  Trades: {EXPECTED_TRADES_PER_DAY:.1f}/day              │")
    print("└" + "─"*99 + "┘")


def display_gate_effectiveness(metrics: TradingMetrics) -> None:
    """Display gate effectiveness section"""
    print("\n┌─ GATE EFFECTIVENESS " + "─"*76 + "┐")

    # Get SHORT threshold from config
    short_thresh = metrics.config.get('short_threshold', 0.80)  # UPDATED 2025-10-30: Default 0.80

    # Gate blocking stats
    if metrics.short_signals_total > 0:
        block_pct = metrics.gate_block_rate * 100
        block_status = "✅ OPTIMAL" if 0.70 <= metrics.gate_block_rate <= 0.80 else \
                       "🟡 ACCEPTABLE" if 0.60 <= metrics.gate_block_rate <= 0.90 else \
                       "🚨 ABNORMAL"

        print(f"│ SHORT Signals      : {metrics.short_signals_total:>4d} total  │  "
              f"Blocked by gate: {metrics.short_signals_blocked:>4d} ({block_pct:>4.1f}%)  │  {block_status}        │")
    else:
        print(f"│ SHORT Signals      : No data yet (waiting for SHORT signals > {short_thresh:.2f} threshold)       │")

    # Trade distribution (Updated for optimized thresholds: 75.0% LONG, 25.0% SHORT)
    if metrics.total_trades > 0:
        dist_status = "✅ ON TARGET" if 0.20 <= metrics.short_pct <= 0.30 else \
                      "🟡 ACCEPTABLE" if 0.15 <= metrics.short_pct <= 0.35 else \
                      "⚠️  OFF TARGET"

        print(f"│ Trade Distribution : LONG: {metrics.long_trades:>3d} ({metrics.long_pct*100:>4.1f}%)  │  "
              f"SHORT: {metrics.short_trades:>3d} ({metrics.short_pct*100:>4.1f}%)  │  {dist_status}       │")

        # Distribution bars
        long_bar = create_bar_chart(metrics.long_pct, 1.0, 40)
        short_bar = create_bar_chart(metrics.short_pct, 0.30, 40)

        print(f"│ LONG Distribution  : {long_bar} {metrics.long_pct*100:>4.1f}%  │")
        print(f"│ SHORT Distribution : {short_bar} {metrics.short_pct*100:>4.1f}%  │")
    else:
        print(f"│ Trade Distribution : No trades yet (waiting for first entry signal)                │")

    # Average position size (Target range from dynamic sizing)
    if metrics.avg_position_size > 0:
        print(f"│ Avg Position Size  : {metrics.avg_position_size*100:>5.1f}%  │  Target: 40-50% (dynamic 20-95%)                │")

    print("└" + "─"*99 + "┘")


def display_expected_vs_actual(metrics: TradingMetrics) -> None:
    """Display expected vs actual performance comparison"""
    print("\n┌─ EXPECTED vs ACTUAL PERFORMANCE " + "─"*64 + "┐")

    # Helper function for status
    def get_status(actual, expected, higher_is_better=True):
        if expected == 0:
            return "N/A", "─"
        ratio = actual / expected
        if higher_is_better:
            if ratio >= 0.85:
                return f"{ratio*100:>3.0f}%", "✅" if ratio >= 1.0 else "🟡"
            else:
                return f"{ratio*100:>3.0f}%", "🚨"
        else:
            if ratio <= 1.15:
                return f"{ratio*100:>3.0f}%", "✅" if ratio <= 1.0 else "🟡"
            else:
                return f"{ratio*100:>3.0f}%", "🚨"

    print(f"│ {'Metric':<18s} │ {'Expected':>10s} │ {'Actual':>10s} │ {'Ratio':>6s} │ {'Status':>8s} │")
    print(f"│ {'':<18s} │ {'':<10s} │ {'':<10s} │ {'':<6s} │ {'':<8s} │")

    # Return (scaled to 5 days) - only meaningful after 1+ day
    if metrics.days_running >= 1.0:
        actual_return_5d = metrics.total_return / metrics.days_running * 5
        ratio_str, status = get_status(actual_return_5d, EXPECTED_RETURN_5D, True)
        print(f"│ Return (5 days)    │ {EXPECTED_RETURN_5D*100:>9.2f}% │ {actual_return_5d*100:>9.2f}% │ {ratio_str:>6s} │ {status:>8s} │")
    else:
        # Too early to extrapolate to 5 days
        print(f"│ Return (5 days)    │ {EXPECTED_RETURN_5D*100:>9.2f}% │  Too early │    N/A │      ─ │")

    # Win Rate
    if metrics.total_trades >= 2:  # Lowered from 5 to 2 (2 trades = 50% calculable)
        ratio_str, status = get_status(metrics.win_rate, EXPECTED_WIN_RATE, True)
        sample_note = " *" if metrics.total_trades < 10 else "  "  # * = small sample
        print(f"│ Win Rate{sample_note}          │ {EXPECTED_WIN_RATE*100:>9.1f}% │ {metrics.win_rate*100:>9.1f}% │ {ratio_str:>6s} │ {status:>8s} │")
    else:
        print(f"│ Win Rate           │ {EXPECTED_WIN_RATE*100:>9.1f}% │        N/A │    N/A │      ─ │")

    # Trades per day
    if metrics.days_running > 0.1:
        ratio_str, status = get_status(metrics.trades_per_day, EXPECTED_TRADES_PER_DAY, True)
        print(f"│ Trades/day         │ {EXPECTED_TRADES_PER_DAY:>10.1f} │ {metrics.trades_per_day:>10.1f} │ {ratio_str:>6s} │ {status:>8s} │")
    else:
        print(f"│ Trades/day         │ {EXPECTED_TRADES_PER_DAY:>10.1f} │        N/A │    N/A │      ─ │")

    # LONG %
    if metrics.total_trades >= 5:  # Keep 5+ for distribution (needs more samples)
        ratio_str, status = get_status(metrics.long_pct, EXPECTED_LONG_PCT, True)
        sample_note = " *" if metrics.total_trades < 20 else "  "  # * = small sample
        print(f"│ LONG Distribution{sample_note} │ {EXPECTED_LONG_PCT*100:>9.1f}% │ {metrics.long_pct*100:>9.1f}% │ {ratio_str:>6s} │ {status:>8s} │")
    else:
        print(f"│ LONG Distribution  │ {EXPECTED_LONG_PCT*100:>9.1f}% │        N/A │    N/A │      ─ │")

    # SHORT %
    if metrics.total_trades >= 5:  # Keep 5+ for distribution (needs more samples)
        ratio_str, status = get_status(metrics.short_pct, EXPECTED_SHORT_PCT, True)
        sample_note = " *" if metrics.total_trades < 20 else "  "  # * = small sample
        print(f"│ SHORT Distribution{sample_note}│ {EXPECTED_SHORT_PCT*100:>9.1f}% │ {metrics.short_pct*100:>9.1f}% │ {ratio_str:>6s} │ {status:>8s} │")
    else:
        print(f"│ SHORT Distribution │ {EXPECTED_SHORT_PCT*100:>9.1f}% │        N/A │    N/A │      ─ │")

    # Sharpe Ratio
    if metrics.total_trades >= 10:
        ratio_str, status = get_status(metrics.sharpe_ratio, EXPECTED_SHARPE, True)
        print(f"│ Sharpe Ratio       │ {EXPECTED_SHARPE:>10.2f} │ {metrics.sharpe_ratio:>10.2f} │ {ratio_str:>6s} │ {status:>8s} │")
    else:
        print(f"│ Sharpe Ratio       │ {EXPECTED_SHARPE:>10.2f} │        N/A │    N/A │      ─ │")

    print("└" + "─"*99 + "┘")


# ============================================================================
# Alert System
# ============================================================================

def check_alerts(metrics: TradingMetrics) -> List[str]:
    """Check for alert conditions (updated for Opportunity Gating + 4x leverage)"""
    alerts = []

    # Maximum drawdown alert (stricter for 4x leverage)
    if metrics.current_drawdown > ALERT_MAX_DRAWDOWN:
        alerts.append(f"🚨 HIGH DRAWDOWN: {metrics.current_drawdown*100:.1f}% (threshold: {ALERT_MAX_DRAWDOWN*100:.1f}%)")

    # Low Sharpe ratio
    if metrics.sharpe_ratio < ALERT_MIN_SHARPE and metrics.total_trades > 10:
        alerts.append(f"⚠️  LOW SHARPE: {metrics.sharpe_ratio:.2f} (threshold: {ALERT_MIN_SHARPE}, expected: {EXPECTED_SHARPE:.2f})")

    # High position risk
    if metrics.current_exposure > ALERT_POSITION_RISK:
        alerts.append(f"⚠️  HIGH EXPOSURE: {metrics.current_exposure*100:.1f}% (threshold: {ALERT_POSITION_RISK*100:.1f}%)")

    # Poor win rate (if enough trades)
    if metrics.total_trades > 10 and metrics.win_rate < ALERT_MIN_WIN_RATE:
        alerts.append(f"⚠️  LOW WIN RATE: {metrics.win_rate*100:.1f}% (threshold: {ALERT_MIN_WIN_RATE*100:.1f}%, expected: {EXPECTED_WIN_RATE*100:.1f}%)")

    # Negative profit factor
    if metrics.profit_factor < 1.0 and metrics.total_trades > 10:
        alerts.append(f"⚠️  PROFIT FACTOR < 1.0: {metrics.profit_factor:.2f}")

    # SHORT distribution alerts (Opportunity Gating specific)
    if metrics.total_trades >= 10:
        if metrics.short_pct < ALERT_SHORT_RATIO_MIN:
            alerts.append(f"⚠️  SHORT % TOO LOW: {metrics.short_pct*100:.1f}% (threshold: {ALERT_SHORT_RATIO_MIN*100:.1f}%, expected: {EXPECTED_SHORT_PCT*100:.1f}%)")
        elif metrics.short_pct > ALERT_SHORT_RATIO_MAX:
            alerts.append(f"🚨 SHORT % TOO HIGH: {metrics.short_pct*100:.1f}% (threshold: {ALERT_SHORT_RATIO_MAX*100:.1f}%, expected: {EXPECTED_SHORT_PCT*100:.1f}%)")

    # Gate effectiveness alerts
    if metrics.short_signals_total >= 10:
        if metrics.gate_block_rate < ALERT_GATE_BLOCK_MIN:
            alerts.append(f"🚨 GATE UNDERBLOCKING: {metrics.gate_block_rate*100:.1f}% (threshold: {ALERT_GATE_BLOCK_MIN*100:.1f}%, expected: {EXPECTED_GATE_BLOCK_RATE*100:.1f}%)")
        elif metrics.gate_block_rate > ALERT_GATE_BLOCK_MAX:
            alerts.append(f"⚠️  GATE OVERBLOCKING: {metrics.gate_block_rate*100:.1f}% (threshold: {ALERT_GATE_BLOCK_MAX*100:.1f}%, expected: {EXPECTED_GATE_BLOCK_RATE*100:.1f}%)")

    # Trade frequency alert
    if metrics.days_running >= 1.0:
        if metrics.trades_per_day < ALERT_TRADES_PER_DAY_MIN:
            alerts.append(f"⚠️  LOW TRADE FREQUENCY: {metrics.trades_per_day:.1f}/day (threshold: {ALERT_TRADES_PER_DAY_MIN:.1f}, expected: {EXPECTED_TRADES_PER_DAY:.1f})")

    return alerts


# ============================================================================
# Display
# ============================================================================

def display_header(state: Dict, log_file: Path, realtime_data: Optional[Dict] = None) -> None:
    """Display header section"""
    print("\n" + "="*100)
    print(" " * 45 + "🎯 QUANT MONITOR")
    print("="*100)

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if state:
        initial = state.get('initial_balance', 100000)
        timestamp = state.get('timestamp', current_time)
        
        # ✅ FIXED 2025-10-25: Use API equity directly when available (no duplicate calculation)
        if realtime_data and realtime_data.get('source') == 'exchange_api':
            # Use BingX API's equity and balance directly (NO CALCULATION)
            equity = realtime_data.get('equity', 0)
            balance = realtime_data.get('balance', 0)
            position = realtime_data.get('position')
            unrealized_pnl = position.get('unrealized_pnl', 0) if position else 0
            data_source = "API"
        else:
            # Fallback to state file (calculate equity)
            balance = state.get('current_balance', 0)
            unrealized_pnl = state.get('unrealized_pnl', 0)
            equity = balance + unrealized_pnl
            data_source = "State"

        print(f"  ⏰ Time: {current_time} | 📊 Session: {timestamp[:19]} | 💰 Equity (BingX {data_source}): ${equity:,.2f} | Wallet: ${balance:,.2f} | Unrealized: ${unrealized_pnl:+,.2f} | Initial: ${initial:,.2f}")
    else:
        print(f"  ⏰ Time: {current_time} | ⚠️  No trading state available")

    print("="*100)


def display_bot_status(state: Dict, log_file: Path) -> None:
    """Display bot status and session info"""
    print("\n┌─ BOT STATUS & SESSION INFO " + "─"*70 + "┐")

    # Session duration
    if state:
        session_start = state.get('session_start', '')
        if session_start:
            try:
                # Remove timezone info if present (+00:00)
                session_start_clean = session_start.replace('+00:00', '').replace('Z', '')
                start_time = datetime.fromisoformat(session_start_clean)
                # Make start_time timezone naive for comparison
                if start_time.tzinfo is not None:
                    start_time = start_time.replace(tzinfo=None)
                duration = datetime.now() - start_time
                hours = duration.total_seconds() / 3600
                print(f"│ Session Started    : {session_start[:19]}  │  Duration: {hours:>6.2f}h  │")
            except Exception as e:
                print(f"│ Session Started    : {session_start[:19]}  │  Duration: N/A  │")

        # Last update
        timestamp = state.get('timestamp', '')
        if timestamp:
            try:
                last_update = datetime.fromisoformat(timestamp)
                time_since = (datetime.now() - last_update).total_seconds()
                status_color = "\033[92m" if time_since < 120 else "\033[93m" if time_since < 300 else "\033[91m"
                print(f"│ Last Update        : {timestamp[:19]}  │  {status_color}({time_since:.0f}s ago)\033[0m  │")
            except:
                print(f"│ Last Update        : {timestamp[:19]}  │")

    # Log file info
    if log_file:
        log_size = log_file.stat().st_size / 1024  # KB
        log_modified = datetime.fromtimestamp(log_file.stat().st_mtime)
        time_since_log = (datetime.now() - log_modified).total_seconds()
        bot_status = "\033[92m●  ACTIVE\033[0m" if time_since_log < 120 else "\033[93m●  IDLE\033[0m" if time_since_log < 300 else "\033[91m●  STOPPED\033[0m"

        print(f"│ Bot Status         : {bot_status}  │  Log: {log_file.name}  │")
        print(f"│ Log Size           : {log_size:>8.1f} KB  │  Modified: {time_since_log:>.0f}s ago  │")

    # State file info
    if STATE_FILE.exists():
        state_size = STATE_FILE.stat().st_size / 1024
        print(f"│ State File         : {STATE_FILE.name}  │  Size: {state_size:>6.1f} KB  │")

    print("└" + "─"*99 + "┘")


def display_performance_metrics(metrics: TradingMetrics) -> None:
    """Display performance metrics section"""
    print("\n┌─ PERFORMANCE METRICS " + "─"*77 + "┐")

    # Returns - separated by realized (closed) and unrealized (open)
    realized_ret_color = format_percentage(metrics.realized_return, 1, width=8)
    unrealized_ret_color = format_percentage(metrics.unrealized_return, 1, width=8)
    total_ret_color = format_percentage(metrics.total_return, 1, width=8)

    print(f"│ Trading P&L        : {realized_ret_color}  │  Closed trades only       │  Trades: {metrics.total_trades:>4d}  │")
    print(f"│ Unrealized Change  : {unrealized_ret_color}  │  Position P&L vs Reset    │  Win Rate: {metrics.win_rate*100:>5.1f}%  │")

    # Balance Change (Trading P&L + Fees, excludes deposits/withdrawals)
    balance_change_color = format_percentage(metrics.balance_change_pct, 0, width=8)
    print(f"│ Wallet Change      : {balance_change_color}  │  Trades+Fees ONLY         │  ${metrics.balance_change:>+7,.2f}  │")

    # Deposits/Withdrawals detected (auto-detected from balance discrepancy)
    if hasattr(metrics, 'deposits_withdrawals_detected') and abs(metrics.deposits_withdrawals_detected) > 0.01:
        deposit_color = "🔵" if metrics.deposits_withdrawals_detected > 0 else "🔴"
        deposit_type = "Deposits" if metrics.deposits_withdrawals_detected > 0 else "Withdrawals"
        print(f"│ {deposit_type:18s} : {deposit_color} ${abs(metrics.deposits_withdrawals_detected):>+7,.2f}  │  Auto-detected            │                │")

    print(f"│ Total Return       : {total_ret_color}  │  Wallet + Unrealized      │                │")

    # Risk-adjusted metrics
    sharpe_str = format_metric(metrics.sharpe_ratio, 2.0, 1.0, True, width=8)
    sortino_str = format_metric(metrics.sortino_ratio, 2.5, 1.5, True, width=6)
    calmar_str = format_metric(metrics.calmar_ratio, 3.0, 1.5, True, width=6)

    print(f"│ Sharpe Ratio       : {sharpe_str}  │  Sortino: {sortino_str}  │  Calmar: {calmar_str}  │")

    # Drawdown
    max_dd_str = format_metric(metrics.max_drawdown, 0.05, 0.10, False, width=6)
    curr_dd_str = format_metric(metrics.current_drawdown, 0.05, 0.10, False, width=6)

    print(f"│ Max Drawdown       : {max_dd_str}%  │  Current DD: {curr_dd_str}%  │             │")

    # Drawdown bar
    dd_bar = create_bar_chart(metrics.current_drawdown, 0.20, 50)
    print(f"│ DD Progress        : {dd_bar} {metrics.current_drawdown*100:.1f}%  │")

    print("└" + "─"*99 + "┘")


def display_fees_and_costs(metrics: TradingMetrics, state: Dict) -> None:
    """Display fees and costs section"""
    print("\n┌─ FEES & COSTS (📡 Exchange API) " + "─"*64 + "┐")

    # Check if fees were fetched
    if metrics.total_fees == 0 and metrics.entry_fees == 0 and metrics.exit_fees == 0:
        print(f"│ Status             : \033[93mFees not fetched (API unavailable)\033[0m                             │")
        print("└" + "─"*99 + "┘")
        return

    # Total fees
    fee_color = "\033[91m" if metrics.total_fees > 0 else "\033[0m"
    print(f"│ Total Fees         : {fee_color}${metrics.total_fees:>8.2f}\033[0m  │  Entry: ${metrics.entry_fees:>8.2f}  │  Exit: ${metrics.exit_fees:>8.2f}  │")

    # Calculate fee percentage of initial balance
    initial_balance = state.get('initial_balance', 0)
    if initial_balance > 0:
        fee_pct = (metrics.total_fees / initial_balance) * 100
        fee_pct_color = "\033[91m" if fee_pct > 1.0 else "\033[93m" if fee_pct > 0.5 else "\033[0m"
        print(f"│ Fee Impact         : {fee_pct_color}{fee_pct:>5.2f}%\033[0m of initial balance  │  "
              f"Maker/Taker: \033[93mTaker (0.05%)\033[0m  │")

    # Breakdown by trade
    # Include exchange-reconciled trades (accurate statistics)
    total_trades = len([t for t in state.get('trades', [])
                        if t.get('status') in ['OPEN', 'CLOSED']
                        and (not t.get('manual_trade', False) or t.get('exchange_reconciled', False))])
    if total_trades > 0:
        avg_fee_per_trade = metrics.total_fees / total_trades
        print(f"│ Fee Breakdown      : {total_trades} trades × ${avg_fee_per_trade:.2f} avg  │  "
              f"Maker/Taker: \033[93mTaker (0.05%)\033[0m  │")

    # Cost impact on performance
    if initial_balance > 0:
        # Calculate what return would be without fees
        actual_return_pct = metrics.total_return * 100
        return_without_fees = ((initial_balance + metrics.total_fees + state.get('current_balance', initial_balance) - initial_balance) / initial_balance - 1) * 100
        fee_drag = actual_return_pct - return_without_fees

        drag_color = "\033[91m" if fee_drag < -1.0 else "\033[93m" if fee_drag < -0.5 else "\033[0m"
        print(f"│ Performance Impact : {drag_color}{fee_drag:>+6.2f}%\033[0m drag on returns  │  "
              f"Return without fees: {return_without_fees:>+6.2f}%  │")

    print("└" + "─"*99 + "┘")


def display_trading_stats(metrics: TradingMetrics) -> None:
    """Display trading statistics section"""
    print("\n┌─ TRADING STATISTICS " + "─"*76 + "┐")

    # Win/Loss stats
    print(f"│ Win/Loss           : {metrics.winning_trades:>4d}W / {metrics.losing_trades:>4d}L  │  "
          f"Profit Factor: {metrics.profit_factor:>6.2f}  │                │")

    # Average trade stats
    avg_win_str = f"${metrics.avg_win:>8.2f}" if metrics.avg_win > 0 else "  N/A"
    avg_loss_str = f"${metrics.avg_loss:>8.2f}" if metrics.avg_loss > 0 else "  N/A"

    print(f"│ Avg Win/Loss       : \033[92m{avg_win_str}\033[0m / \033[91m{avg_loss_str}\033[0m  │  "
          f"Avg Hold: {metrics.avg_holding_time:>6.2f}h  │                │")

    # Largest win/loss
    large_win_str = f"${metrics.largest_win:>8.2f}" if metrics.largest_win > 0 else "  N/A"
    large_loss_str = f"${metrics.largest_loss:>8.2f}" if metrics.largest_loss > 0 else "  N/A"

    print(f"│ Largest Win/Loss   : \033[92m{large_win_str}\033[0m / \033[91m{large_loss_str}\033[0m  │"
          f"                                       │")

    print("└" + "─"*99 + "┘")


def display_risk_metrics(metrics: TradingMetrics) -> None:
    """Display risk metrics section"""
    print("\n┌─ RISK ANALYTICS " + "─"*80 + "┐")

    # VaR and CVaR
    var_str = format_metric(metrics.var_95, 0.01, 0.02, False, width=6)
    cvar_str = format_metric(metrics.cvar_95, 0.015, 0.03, False, width=6)

    print(f"│ VaR (95%)          : {var_str}%  │  CVaR (95%): {cvar_str}%  │                │")

    # Position exposure
    exposure_str = format_metric(metrics.current_exposure, 0.70, 0.85, False, width=6)
    print(f"│ Position Exposure  : {exposure_str}%  │  Volatility: {metrics.volatility*100:>6.2f}%  │                │")

    # Exposure bar
    exp_bar = create_bar_chart(metrics.current_exposure, 1.0, 50)
    print(f"│ Exposure Level     : {exp_bar} {metrics.current_exposure*100:.1f}%  │")

    print("└" + "─"*99 + "┘")


def display_position_analysis(metrics: TradingMetrics, state: Dict, realtime_data: Optional[Dict] = None) -> None:
    """Display detailed position and exit analysis

    Args:
        metrics: Trading metrics
        state: Bot state from JSON file (fallback)
        realtime_data: Real-time data from exchange API (priority)
    """
    # Determine data source
    using_realtime_api = realtime_data and realtime_data.get('source') == 'exchange_api'

    # PRIORITY 1: Use real-time API data if available
    if using_realtime_api:
        current_position = realtime_data.get('position')
        current_price = realtime_data.get('current_price', 0)
        balance = realtime_data.get('balance', 0)
        has_position = current_position is not None

        # 🔧 FIX 2025-10-30: Fall back to state file when API returns no position
        if not has_position and state:
            # API returned no position - check state file
            state_position = state.get('position')
            if state_position and state_position.get('status') == 'OPEN':
                # Use state file position
                current_position = state_position
                has_position = True
                using_realtime_api = False  # Mark as hybrid source

        # Validate realtime position data - fallback to state if incomplete
        if has_position and current_position.get('entry_price', 0) == 0:
            # Realtime API returned incomplete data - merge with state file
            if state:
                state_position = state.get('position')
                if state_position and state_position.get('status') == 'OPEN':
                    # Merge state data into realtime position
                    current_position = {**current_position, **state_position}
                    using_realtime_api = False  # Mark as hybrid source

        # Validate current price - fallback to metrics if API returned 0
        if current_price == 0 and metrics.prices:
            current_price = metrics.prices[0]
    else:
        # PRIORITY 2: Fallback to state file
        if not state:
            print(f"\n┌─ POSITION & EXIT ANALYSIS (📁 State File) " + "─"*59 + "┐")
            print(f"│ Position           : No data available                                              │")
            print("└" + "─"*99 + "┘")
            return

        # Check 'position' field first (synced from exchange)
        current_position = state.get('position')
        has_position = current_position and current_position.get('status') == 'OPEN'

        # Fallback to trades array if position field doesn't exist
        if not has_position:
            trades = state.get('trades', [])
            # Include exchange-reconciled trades
            open_trades = [t for t in trades
                           if t.get('status') == 'OPEN'
                           and (not t.get('manual_trade', False) or t.get('exchange_reconciled', False))]
            if open_trades:
                current_position = open_trades[-1]
                has_position = True

        # Get price from metrics or state
        current_price = metrics.prices[0] if metrics.prices else 0
        balance = state.get('current_balance', 0)

    # Update data source indicator after validation
    data_source_indicator = "📡 LIVE API" if using_realtime_api else "📁 State File"
    print(f"\n┌─ POSITION & EXIT ANALYSIS ({data_source_indicator}) " + "─"*59 + "┐")

    # Display position information
    if state and has_position:
        latest = current_position
        side = latest.get('side', 'N/A')
        prob = latest.get('probability', 0)
        entry_price = latest.get('entry_price', 0)
        quantity = latest.get('quantity', 0)
        position_pct = latest.get('position_size_pct', 0)
        position_value = latest.get('position_value', 0)

        if isinstance(prob, str):
            prob = float(prob)

        # Calculate actual leverage (notional value / equity)
        # Get current balance for calculations
        current_balance = state.get('current_balance', 0) if not using_realtime_api else balance
        # Current position notional value (quantity × current price)
        current_position_value = quantity * current_price if current_price else quantity * entry_price
        # Leverage = notional value / equity
        actual_leverage = current_position_value / current_balance if current_balance > 0 else 0

        # Show basic position info first (always display)
        # Check if synced position (for Entry Prob display)
        is_synced = latest.get('synced_from_exchange', False)
        prob_display = "N/A (synced)" if is_synced else f"{prob:.3f}"

        print(f"│ Position           : {side:>6s}  │  Leverage: {actual_leverage:>4.2f}x  │  Entry Prob: {prob_display:<13s}  │")
        print(f"│ Entry Price        : ${entry_price:>10,.2f}  │  Quantity: {quantity:.8f}  │                       │")

        # Calculate P&L (if current price available)
        if current_price and entry_price:
            if side == 'LONG':
                pnl_pct = (current_price - entry_price) / entry_price
                pnl_usd = quantity * (current_price - entry_price)
            else:  # SHORT
                pnl_pct = (entry_price - current_price) / entry_price
                pnl_usd = quantity * (entry_price - current_price)

            # P&L calculations for leveraged position
            # Note: pnl_usd is already the actual P&L (quantity was calculated with leverage)
            # Leveraged P&L % = actual P&L / margin used (position_value)
            pnl_pct_on_margin = pnl_usd / position_value if position_value > 0 else 0
            unleveraged_pnl_pct = pnl_pct  # Price change %

            # Current position value (quantity × current price)
            current_position_value = quantity * current_price

            # Get current balance for calculations
            current_balance = state.get('current_balance', 0) if not using_realtime_api else balance
            value_multiplier = current_position_value / current_balance if current_balance > 0 else 0
            roi_on_balance = pnl_usd / current_balance if current_balance > 0 else 0

            pnl_color = "\033[92m" if pnl_usd > 0 else "\033[91m" if pnl_usd < 0 else "\033[0m"

            print(f"│ Current Price      : ${current_price:>10,.2f}  │  Value: ${current_position_value:>10,.2f} ({value_multiplier:.2f}x)  │")
            print(f"│ Position P&L       : {pnl_color}${pnl_usd:>+10,.2f}\033[0m ({pnl_color}{roi_on_balance*100:>+5.2f}%\033[0m of balance)  │  Price Δ: {unleveraged_pnl_pct*100:>+5.2f}%  │")
        else:
            # No current price - show warning
            print(f"│ Current Price      : \033[93mN/A (waiting for price data)\033[0m                                    │")

        # Holding time and exit conditions
        entry_time_str = latest.get('entry_time', '')
        if entry_time_str:
            try:
                # Try ISO format first (2025-10-17T07:05:00)
                entry_time = datetime.fromisoformat(entry_time_str)
            except:
                try:
                    # Try space-separated format (2025-10-17 07:05:00)
                    entry_time = datetime.strptime(entry_time_str, '%Y-%m-%d %H:%M:%S')
                except:
                    entry_time = None

            if entry_time:
                holding_hours = (datetime.now() - entry_time).total_seconds() / 3600
                max_hold = metrics.config.get('emergency_max_hold_hours', 8)  # From config

                # Time-based exit
                time_left = max_hold - holding_hours
                time_color = "\033[92m" if time_left > 4 else "\033[93m" if time_left > 2 else "\033[91m"

                print(f"│ Holding Time       : {holding_hours:>6.2f}h  │  Max Hold: {max_hold:.1f}h  │  {time_color}Time Left: {time_left:.2f}h\033[0m  │")

        # Get exit signals and display (포지션 있을 때는 항상 표시)
        signals = state.get('latest_signals', {})
        exit_signals = signals.get('exit', {})

        # Get exit threshold from config - side별로 다른 threshold 사용
        ml_exit_thresh_long = metrics.config.get('ml_exit_threshold_base_long', 0.70)  # UPDATED 2025-11-02: Default 0.70 (Phase 1 Optimized)
        ml_exit_thresh_short = metrics.config.get('ml_exit_threshold_base_short', 0.70)  # UPDATED 2025-11-02: Default 0.70 (Phase 1 Optimized)
        ml_exit_thresh = ml_exit_thresh_long if side == 'LONG' else ml_exit_thresh_short
        max_hold_display = metrics.config.get('emergency_max_hold_hours', 10)  # UPDATED 2025-10-22: 10 hours

        # Get exit probability (try multiple sources)
        exit_prob = 0.0
        # Always use config threshold (not state file) to display current settings
        exit_thresh = ml_exit_thresh

        # Source 1: latest_signals.exit (most reliable)
        if exit_signals:
            exit_prob = exit_signals.get('exit_prob', 0.0)

        # Source 2: position object itself (fallback)
        if exit_prob == 0.0 and 'exit_prob' in latest:
            exit_prob = latest.get('exit_prob', 0.0)

        # Calculate percentage
        exit_pct = (exit_prob / exit_thresh * 100) if exit_thresh > 0 else 0

        # Format exit probability with color
        if exit_prob > 0:
            exit_prob_str = format_signal_probability(exit_prob, exit_thresh)
            exit_pct_color = "\033[1;92m" if exit_pct >= 100 else "\033[92m" if exit_pct >= 85 else "\033[93m" if exit_pct >= 70 else "\033[0m" if exit_pct >= 50 else "\033[91m"

            print(f"│ Exit Signal ({side:<5s}): {exit_prob_str} ({exit_pct_color}{exit_pct:>4.0f}%\033[0m)  │  Threshold: ML Exit ({exit_thresh:.2f}) │        │")
        else:
            # No exit probability available yet
            print(f"│ Exit Signal ({side:<5s}): \033[93mN/A\033[0m (waiting for signal)  │  Threshold: ML Exit ({exit_thresh:.2f}) │        │")

        print(f"│ Exit Conditions    : Exit Model (prob > {exit_thresh:.2f}) │  Max Hold ({max_hold_display:.1f}h) │  Stop Loss/TP  │")

    else:
        # No open position - show entry signals
        signals = state.get('latest_signals', {})
        entry_signals = signals.get('entry', {})

        # Get entry thresholds from config (Single Source of Truth)
        config_long_thresh = metrics.config.get('long_threshold', 0.80)  # UPDATED 2025-10-30: Default 0.80 (Threshold 0.80)
        config_short_thresh = metrics.config.get('short_threshold', 0.80)  # UPDATED 2025-10-30: Default 0.80 (Threshold 0.80)

        # Get entry signals (already floats with new JSON serializer)
        long_prob = entry_signals.get('long_prob', 0.0)
        short_prob = entry_signals.get('short_prob', 0.0)
        long_thresh = entry_signals.get('long_threshold', config_long_thresh)
        short_thresh = entry_signals.get('short_threshold', config_short_thresh)

        # Calculate signal strength percentages
        long_pct = (long_prob / long_thresh * 100) if long_thresh > 0 else 0
        short_pct = (short_prob / short_thresh * 100) if short_thresh > 0 else 0

        # Format probabilities with color
        long_prob_str = format_signal_probability(long_prob, long_thresh)
        short_prob_str = format_signal_probability(short_prob, short_thresh)

        # Color percentage based on threshold proximity
        long_pct_color = "\033[1;92m" if long_pct >= 100 else "\033[92m" if long_pct >= 85 else "\033[93m" if long_pct >= 70 else "\033[0m" if long_pct >= 50 else "\033[91m"
        short_pct_color = "\033[1;92m" if short_pct >= 100 else "\033[92m" if short_pct >= 85 else "\033[93m" if short_pct >= 70 else "\033[0m" if short_pct >= 50 else "\033[91m"

        print(f"│ Position           : No open position                                               │")
        print(f"│ Status             : Waiting for signal                                             │")
        print(f"│ Entry Signals      : LONG: {long_prob_str} ({long_pct_color}{long_pct:>4.0f}%\033[0m)  │  SHORT: {short_prob_str} ({short_pct_color}{short_pct:>4.0f}%\033[0m)  │")

        # Display threshold adjustment context (V3: ACTUAL ENTRY RATE)
        threshold_context = entry_signals.get('threshold_context', {})
        entry_rate = threshold_context.get('entry_rate')
        entries_count = threshold_context.get('entries_count')
        base_long = threshold_context.get('base_long', config_long_thresh)
        target_rate = threshold_context.get('target_rate', 0.011)
        target_trades_per_week = threshold_context.get('target_trades_per_week', 22.0)

        if entry_rate is not None and abs(long_thresh - base_long) > 0.05:
            # Significant threshold adjustment
            adjustment = long_thresh - base_long
            ratio = entry_rate / target_rate if target_rate > 0 else 1.0

            if adjustment > 0:
                status = f"⚠️ RAISED (+{adjustment:.2f})"
                if entries_count is not None and entries_count > 0:
                    reason = f"High entry rate ({entries_count} entries in 6h vs {target_trades_per_week:.0f}/week target)"
                else:
                    reason = f"High entry rate ({entry_rate*100:.2f}% vs {target_rate*100:.2f}% target, {ratio:.1f}x)"
            else:
                status = f"✓ LOWERED ({adjustment:.2f})"
                if entries_count is not None:
                    reason = f"Low entry rate ({entries_count} entries in 6h vs {target_trades_per_week:.0f}/week target)"
                else:
                    reason = f"Low entry rate ({entry_rate*100:.2f}% vs {target_rate*100:.2f}% target, {ratio:.1f}x)"

            print(f"│ Threshold Status   : {status:<30s} │  {reason:<40s} │")
        else:
            print(f"│ Threshold Status   : ✓ NORMAL (base thresholds)                                    │")

        # Threshold 0.80 backtest: LONG 61.8%, SHORT 38.2%, Total 4.64/day = 32.48/week
        print(f"│ Expected Frequency : LONG: 20.1/week (61.8% of trades) │  SHORT: 12.4/week (38.2% of trades)  │")

    print("└" + "─"*99 + "┘")


def normalize_trade_side(side: str) -> str:
    """
    Normalize trade side to standard format (LONG/SHORT).

    BingX API may return different side values:
    - "BUY" → "LONG" (opening LONG position)
    - "Open-Short" → "SHORT" (opening SHORT position)
    - "LONG", "SHORT" → Keep as-is (already correct)

    Args:
        side: Original side value from trade

    Returns:
        Normalized side: "LONG" or "SHORT"
    """
    side_upper = side.upper()

    mapping = {
        "BUY": "LONG",
        "LONG": "LONG",
        "SHORT": "SHORT",
        "OPEN-SHORT": "SHORT"
    }

    return mapping.get(side_upper, side)  # Return original if not in mapping


def display_recent_trades(state: Dict, api_client=None) -> None:
    """Display recent trade history with fees from exchange API"""
    if not state:
        return

    print("\n┌─ CLOSED POSITIONS (Last 5) - Historical Exit Reasons " + "─"*48 + "┐")

    trades = state.get('trades', [])
    # Show ALL closed trades (including reconciled from exchange)
    # These represent actual positions that occurred on the exchange
    closed_trades = [t for t in trades if t.get('status') == 'CLOSED']

    if not closed_trades:
        print("└" + "─"*99 + "┘")
        return

    # Fetch API trades for fee verification if API client available
    api_trades_by_order = {}
    if api_client:
        try:
            api_trades = api_client.exchange.fetch_my_trades('BTC/USDT:USDT', limit=50)

            # Create lookup dict by order ID
            for trade in api_trades:
                order_id = str(trade['order'])
                if order_id not in api_trades_by_order:
                    api_trades_by_order[order_id] = trade

        except Exception as e:
            # Silently fall back to state file data on API errors
            pass

    # Group trades by position_id_exchange to show position-level P&L
    # One position can have multiple entry/exit orders
    from collections import defaultdict

    positions = defaultdict(lambda: {
        'trades': [],
        'side': 'N/A',
        'total_pnl': 0,
        'total_fees': 0,
        'exit_reason': 'N/A'
    })

    # Group trades by position
    for trade in closed_trades:
        position_id = trade.get('position_id_exchange', trade.get('order_id'))  # Fallback to order_id for legacy

        positions[position_id]['trades'].append(trade)
        # Normalize side to handle API variations (BUY → LONG, etc.)
        raw_side = trade.get('side', 'N/A')
        positions[position_id]['side'] = normalize_trade_side(raw_side) if raw_side != 'N/A' else 'N/A'
        positions[position_id]['total_pnl'] += trade.get('pnl_usd_net', 0)
        positions[position_id]['total_fees'] += trade.get('total_fee', 0)
        positions[position_id]['exit_reason'] = trade.get('exit_reason', 'N/A')

    # Calculate average entry/exit prices for each position
    for position_id, pos_data in positions.items():
        trades_list = pos_data['trades']

        # Average entry price (weighted by quantity if available)
        total_entry_value = 0
        total_quantity = 0
        for t in trades_list:
            qty = t.get('quantity', 0)
            entry_price = t.get('entry_price', 0)
            if qty > 0:
                total_entry_value += entry_price * qty
                total_quantity += qty

        pos_data['avg_entry_price'] = total_entry_value / total_quantity if total_quantity > 0 else trades_list[0].get('entry_price', 0)

        # Average exit price (weighted by quantity if available)
        total_exit_value = 0
        for t in trades_list:
            qty = t.get('quantity', 0)
            exit_price = t.get('exit_price', 0)
            if qty > 0:
                total_exit_value += exit_price * qty

        pos_data['avg_exit_price'] = total_exit_value / total_quantity if total_quantity > 0 else trades_list[0].get('exit_price', 0)

    # Sort positions by close time (last trade in position)
    # 🔧 FIX 2025-11-03: Check both 'close_time' (manual/reconciled) and 'exit_time' (bot trades)
    sorted_positions = sorted(
        positions.items(),
        key=lambda x: max([t.get('close_time') or t.get('exit_time', '') for t in x[1]['trades']])
    )

    # Get initial balance for account-based return calculation
    initial_balance = state.get('initial_balance', 1000.0)

    # Display last 5 positions (newest first, oldest last - reverse chronological order)
    # 🔧 FIX 2025-11-03: Reversed to show newest trades at top (#1 = newest)
    recent_positions = sorted_positions[-5:][::-1]  # Reverse to show newest first
    for position_num, (position_id, pos_data) in enumerate(recent_positions, 1):
        side = pos_data['side']
        entry_price = pos_data['avg_entry_price']
        exit_price = pos_data['avg_exit_price']
        pnl_usd = pos_data['total_pnl']
        total_fee = pos_data['total_fees']
        exit_reason = pos_data['exit_reason']

        # Account-based return: P&L / Initial Balance
        account_return_pct = (pnl_usd / initial_balance) * 100 if initial_balance > 0 else 0

        # Shorten exit reason
        if 'ML Exit' in exit_reason:
            exit_reason = 'ML Exit'
        elif 'Max Hold' in exit_reason:
            exit_reason = 'Max Hold'
        elif 'Manual' in exit_reason:
            exit_reason = 'Manual'
        elif 'Fixed Take Profit' in exit_reason or 'Take Profit' in exit_reason:
            exit_reason = 'Take Profit'
        elif 'Stop Loss' in exit_reason:
            exit_reason = 'Stop Loss'
        elif 'Emergency' in exit_reason:
            exit_reason = 'Emergency'
        elif 'Reconciled' in exit_reason:
            exit_reason = 'Exchange'

        # Format P&L with fee information
        pnl_color = "\033[92m" if pnl_usd > 0 else "\033[91m"
        if total_fee > 0:
            pnl_str = f"{account_return_pct:>+6.2f}% (${pnl_usd:>+8.2f}, fee: ${total_fee:.2f})"
        else:
            pnl_str = f"{account_return_pct:>+6.2f}% (${pnl_usd:>+8.2f})"

        # Display with reverse chronological position number (1 = newest, 5 = oldest)
        print(f"│ #{position_num:>3d}  {side:>5s}  │  ${entry_price:>10,.2f} → ${exit_price:>10,.2f}  │  "
              f"{pnl_color}{pnl_str}\033[0m  │  {exit_reason:<10s}  │")

    print("└" + "─"*99 + "┘")




def display_alerts(alerts: List[str]) -> None:
    """Display alerts section"""
    if not alerts:
        return

    print("\n┌─ ⚠️  ALERTS " + "─"*85 + "┐")
    for alert in alerts:
        print(f"│ {alert:<97s} │")
    print("└" + "─"*99 + "┘")


def display_footer(refresh_interval: int = REFRESH_INTERVAL) -> None:
    """Display footer with API rate info"""
    api_calls_per_min = (60 / refresh_interval) * 3
    print("\n" + "="*100)
    print(f"  ⟳ Auto-refresh: {refresh_interval}s  │  API: {api_calls_per_min:.1f} calls/min  │  Press Ctrl+C to exit")
    print("="*100 + "\n")


# ============================================================================
# Main Monitor
# ============================================================================

def run_monitor(refresh_interval: int = REFRESH_INTERVAL):
    """Main monitoring loop with real-time API integration

    Args:
        refresh_interval: Seconds between refreshes (default: 30s)
                         Minimum: 10s (avoid rate limits)
                         Recommended: 30-60s for safety
    """
    # Validate and adjust refresh interval
    if refresh_interval < 10:
        print(f"⚠️  Warning: {refresh_interval}s refresh too fast, using 10s minimum")
        refresh_interval = 10

    api_calls_per_min = (60 / refresh_interval) * 3  # 3 API calls per refresh

    print("\n🚀 Starting Professional Quantitative Trading Monitor...")
    print(f"📁 State File: {STATE_FILE}")
    print(f"📊 Logs Dir: {LOGS_DIR}")
    print(f"⏱️  Refresh Interval: {refresh_interval}s ({api_calls_per_min:.1f} API calls/min)")
    print("\n" + "="*100)

    # Load initial state and configuration (AUTO-SYNC from state file)
    print("\n📥 Loading initial state and configuration...")
    state, config = load_trading_state()
    if config is None:
        print("❌ Failed to load configuration - cannot start monitor")
        print("   Please check production bot status and state file")
        return

    # Initialize metrics with auto-synced configuration
    metrics = TradingMetrics(config=config)
    print(f"✅ Configuration loaded successfully (source: state file)")
    print(f"   Entry thresholds: LONG {config['long_threshold']:.2f}, SHORT {config['short_threshold']:.2f}")
    print(f"   Exit thresholds: LONG {config['ml_exit_threshold_base_long']:.2f}, SHORT {config['ml_exit_threshold_base_short']:.2f}")
    print("="*100)

    # Initialize API client for real-time data
    api_client = None
    try:
        api_client = init_api_client()
        if api_client:
            print("✅ API client initialized - using real-time exchange data")
        else:
            print("⚠️ API client initialization failed - using state file data")
    except Exception as e:
        print(f"⚠️ Could not initialize API client: {e}")
        print("   Falling back to state file data")

    print("="*100 + "\n")

    try:
        while True:
            # Clear screen (Windows/Linux compatible)
            clear_screen()

            # Load data and configuration
            state, config = load_trading_state()
            log_file = find_latest_log()

            # Update metrics with configuration (Single Source of Truth)
            if config:
                metrics = TradingMetrics(config)

            if log_file:
                parse_log_metrics(log_file, metrics)

            # Fetch real-time data from exchange API
            realtime_data = None
            fee_data = None
            if api_client:
                try:
                    realtime_data = fetch_realtime_data(api_client, SYMBOL)
                except Exception as e:
                    # Silently fall back to state file on API errors
                    realtime_data = None

                # Extract fees from state file (V2 reconciliation compatible)
                # Note: V2 reconciliation already populates 'total_fee' from fetchPositionHistory
                try:
                    fee_data = extract_fees_from_state(state)
                except Exception as e:
                    # Fee extraction is optional - continue without it
                    fee_data = None

            # Calculate metrics (with fees and realtime data if available)
            if state:
                calculate_metrics(state, metrics, fee_data, realtime_data)

            # Display sections
            display_header(state, log_file, realtime_data)
            display_bot_status(state, log_file)
            display_strategy_info(metrics)  # NEW: Strategy information (now uses config)
            display_position_analysis(metrics, state, realtime_data)  # Pass real-time data
            # display_gate_effectiveness(metrics)  # REMOVED: User request - not needed
            display_expected_vs_actual(metrics)  # NEW: Expected vs Actual
            display_recent_trades(state, api_client)  # Pass API client
            display_performance_metrics(metrics)
            # display_fees_and_costs(metrics, state)  # REMOVED: Integrated into PERFORMANCE METRICS
            display_trading_stats(metrics)
            display_risk_metrics(metrics)

            # Check and display alerts
            alerts = check_alerts(metrics)
            display_alerts(alerts)

            display_footer(refresh_interval)

            # Wait for next refresh
            time.sleep(refresh_interval)

    except KeyboardInterrupt:
        print("\n\n✅ Monitor stopped by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys

    # Parse command line arguments
    refresh = REFRESH_INTERVAL
    if len(sys.argv) > 1:
        try:
            refresh = int(sys.argv[1])
            if refresh < 10:
                print(f"⚠️  Minimum refresh interval: 10s (you specified {refresh}s)")
                refresh = 10
        except ValueError:
            print(f"⚠️  Invalid refresh interval: {sys.argv[1]}, using default {REFRESH_INTERVAL}s")

    run_monitor(refresh)