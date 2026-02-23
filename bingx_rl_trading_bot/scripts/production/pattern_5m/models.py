"""
Pattern 5m Bot - Data Models
Dataclasses for type-safe data structures.
"""

import time
from collections import deque
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, TypedDict

from .constants import (
    CACHE_TTL_SECONDS,
    CB_FAILURE_THRESHOLD,
    CB_RESET_TIMEOUT,
    CB_INITIAL_TIMEOUT,
    CB_MAX_TIMEOUT,
    CB_BACKOFF_MULTIPLIER,
    EXPECTED_WIN_RATE,
    EXPECTED_AVG_WIN,
    EXPECTED_AVG_LOSS,
    EXPECTED_EDGE,
    METRICS_WINDOW_SIZE,
    MIN_TRADES_FOR_COMPARISON,
)


@dataclass
class APICache:
    """API response cache with TTL."""

    ticker: Optional[Dict] = None
    ticker_time: float = 0
    balance: Optional[Dict] = None
    balance_time: float = 0
    positions: Optional[List] = None
    positions_time: float = 0

    def get_ticker(self, ttl: float = CACHE_TTL_SECONDS) -> Optional[Dict]:
        """Get cached ticker if not expired."""
        if self.ticker and (time.time() - self.ticker_time) < ttl:
            return self.ticker
        return None

    def set_ticker(self, data: Dict) -> None:
        """Update ticker cache."""
        self.ticker = data
        self.ticker_time = time.time()

    def get_balance(self, ttl: float = CACHE_TTL_SECONDS) -> Optional[Dict]:
        """Get cached balance if not expired."""
        if self.balance and (time.time() - self.balance_time) < ttl:
            return self.balance
        return None

    def set_balance(self, data: Dict) -> None:
        """Update balance cache."""
        self.balance = data
        self.balance_time = time.time()

    def get_positions(self, ttl: float = CACHE_TTL_SECONDS) -> Optional[List]:
        """Get cached positions if not expired."""
        if self.positions is not None and (time.time() - self.positions_time) < ttl:
            return self.positions
        return None

    def set_positions(self, data: List) -> None:
        """Update positions cache."""
        self.positions = data
        self.positions_time = time.time()

    def invalidate_all(self) -> None:
        """Invalidate all cached data."""
        self.ticker = None
        self.balance = None
        self.positions = None


@dataclass
class CircuitBreaker:
    """Circuit breaker for API error handling with exponential backoff."""

    failure_count: int = 0
    failure_threshold: int = CB_FAILURE_THRESHOLD
    reset_timeout: float = CB_RESET_TIMEOUT
    last_failure_time: float = 0
    is_open: bool = False
    backoff_level: int = 0  # Exponential backoff level (0, 1, 2, ...)

    def record_failure(self) -> None:
        """Record an API failure and increment backoff level."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            if not self.is_open:
                # First time opening circuit breaker
                self.is_open = True
            else:
                # Already open, increment backoff level
                self.backoff_level += 1

    def record_success(self) -> None:
        """Record a successful API call and reset backoff."""
        self.failure_count = 0
        self.is_open = False
        self.backoff_level = 0  # Reset backoff on success

    def _get_current_timeout(self) -> float:
        """Calculate current timeout based on backoff level."""
        timeout = CB_INITIAL_TIMEOUT * (CB_BACKOFF_MULTIPLIER ** self.backoff_level)
        return min(timeout, CB_MAX_TIMEOUT)

    def can_execute(self) -> bool:
        """Check if API calls are allowed."""
        if not self.is_open:
            return True
        # Check if reset timeout has passed (with exponential backoff)
        current_timeout = self._get_current_timeout()
        if (time.time() - self.last_failure_time) >= current_timeout:
            self.is_open = False
            self.failure_count = 0
            # Keep backoff_level for next failure (only reset on success)
            return True
        return False

    def get_wait_time(self) -> float:
        """Get remaining wait time before circuit breaker resets."""
        if not self.is_open:
            return 0
        current_timeout = self._get_current_timeout()
        remaining = current_timeout - (time.time() - self.last_failure_time)
        return max(0, remaining)


@dataclass
class PerformanceMetrics:
    """Real-time performance tracking with backtest comparison."""

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl_pct: float = 0.0

    # Backtest comparison (expected values)
    expected_win_rate: float = EXPECTED_WIN_RATE
    expected_avg_win: float = EXPECTED_AVG_WIN
    expected_avg_loss: float = EXPECTED_AVG_LOSS
    expected_edge: float = EXPECTED_EDGE

    # Actual performance
    actual_win_rate: float = 0.0
    actual_avg_win: float = 0.0
    actual_avg_loss: float = 0.0
    actual_edge: float = 0.0

    # Recent trades for rolling stats (deque for O(1) popleft)
    recent_wins: deque = field(default_factory=lambda: deque(maxlen=METRICS_WINDOW_SIZE))
    recent_losses: deque = field(default_factory=lambda: deque(maxlen=METRICS_WINDOW_SIZE))

    # v1.34.0: Complete trade history (survives log rotation)
    trade_history: list = field(default_factory=list)

    # Timing stats
    avg_api_latency_ms: float = 0.0
    api_call_count: int = 0

    # Session info
    session_start: str = ""
    last_updated: str = ""

    def update_trade(self, pnl_pct: float, trade_detail: Optional[Dict[str, Any]] = None) -> None:
        """Update metrics after a trade."""
        self.total_trades += 1
        self.total_pnl_pct += pnl_pct

        if pnl_pct >= 0:
            self.winning_trades += 1
            self.recent_wins.append(pnl_pct)
        else:
            self.losing_trades += 1
            self.recent_losses.append(abs(pnl_pct))

        if trade_detail:
            self.trade_history.append(trade_detail)

        self._recalculate()

    def _recalculate(self) -> None:
        """Recalculate derived metrics."""
        if self.total_trades > 0:
            self.actual_win_rate = (self.winning_trades / self.total_trades) * 100

        if self.recent_wins:
            self.actual_avg_win = sum(self.recent_wins) / len(self.recent_wins)

        if self.recent_losses:
            self.actual_avg_loss = sum(self.recent_losses) / len(self.recent_losses)

        # Calculate edge: per-trade expected PnL% (same unit as EXPECTED_EDGE)
        if self.total_trades >= MIN_TRADES_FOR_COMPARISON:
            self.actual_edge = self.total_pnl_pct / self.total_trades

        self.last_updated = datetime.now().isoformat()

    def update_api_latency(self, latency_ms: float) -> None:
        """Update API latency stats with exponential moving average."""
        self.api_call_count += 1
        alpha = 0.1
        self.avg_api_latency_ms = alpha * latency_ms + (1 - alpha) * self.avg_api_latency_ms

    def get_comparison_report(self) -> str:
        """Generate backtest vs actual comparison report."""
        if self.total_trades < MIN_TRADES_FOR_COMPARISON:
            return f"Insufficient trades for comparison (need >= {MIN_TRADES_FOR_COMPARISON})"

        wr_diff = self.actual_win_rate - self.expected_win_rate

        report = f"""
═══════════════════════════════════════════════════════════════
              BACKTEST vs ACTUAL PERFORMANCE
═══════════════════════════════════════════════════════════════
                    Expected    Actual      Diff
───────────────────────────────────────────────────────────────
Win Rate:           {self.expected_win_rate:.1f}%       {self.actual_win_rate:.1f}%      {wr_diff:+.1f}%
Avg Win:            {self.expected_avg_win:.2f}%      {self.actual_avg_win:.2f}%     {self.actual_avg_win - self.expected_avg_win:+.2f}%
Avg Loss:           {self.expected_avg_loss:.2f}%      {self.actual_avg_loss:.2f}%     {self.actual_avg_loss - self.expected_avg_loss:+.2f}%
Edge/Trade:         {self.expected_edge:.2f}%      {self.actual_edge:.2f}%     {self.actual_edge - self.expected_edge:+.2f}%
───────────────────────────────────────────────────────────────
Total Trades:       {self.total_trades}
Total PnL:          {self.total_pnl_pct:+.2f}%
API Latency:        {self.avg_api_latency_ms:.0f}ms (avg)
═══════════════════════════════════════════════════════════════
"""
        return report

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'total_pnl_pct': self.total_pnl_pct,
            'actual_win_rate': self.actual_win_rate,
            'actual_avg_win': self.actual_avg_win,
            'actual_avg_loss': self.actual_avg_loss,
            'actual_edge': self.actual_edge,
            'avg_api_latency_ms': self.avg_api_latency_ms,
            'api_call_count': self.api_call_count,
            'session_start': self.session_start,
            'last_updated': self.last_updated,
            'recent_wins': list(self.recent_wins),
            'recent_losses': list(self.recent_losses),
            'trade_history': self.trade_history,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceMetrics':
        """Create PerformanceMetrics from dictionary."""
        metrics = cls()
        metrics.total_trades = data.get('total_trades', 0)
        metrics.winning_trades = data.get('winning_trades', 0)
        metrics.losing_trades = data.get('losing_trades', 0)
        metrics.total_pnl_pct = data.get('total_pnl_pct', 0.0)
        metrics.actual_win_rate = data.get('actual_win_rate', 0.0)
        metrics.actual_avg_win = data.get('actual_avg_win', 0.0)
        metrics.actual_avg_loss = data.get('actual_avg_loss', 0.0)
        metrics.actual_edge = data.get('actual_edge', 0.0)
        metrics.avg_api_latency_ms = data.get('avg_api_latency_ms', 0.0)
        metrics.api_call_count = data.get('api_call_count', 0)
        metrics.recent_wins = deque(data.get('recent_wins', []), maxlen=METRICS_WINDOW_SIZE)
        metrics.recent_losses = deque(data.get('recent_losses', []), maxlen=METRICS_WINDOW_SIZE)
        metrics.session_start = data.get('session_start', '')
        metrics.last_updated = data.get('last_updated', '')
        metrics.trade_history = data.get('trade_history', [])
        return metrics


class BotState(TypedDict, total=False):
    """Schema for bot state dictionary — documents required and optional keys."""
    # Required
    positions: Dict[str, Dict[str, Any]]  # v1.29.0: {slot_id: position_dict}
    active_direction: Optional[str]  # v1.29.0: None / 'LONG' / 'SHORT'
    emergency_sl_order_id: Optional[str]  # v1.29.0: One-Way emergency SL
    emergency_sl_orders: Dict[str, Optional[str]]  # v1.30.0: Hedge per-direction SL
    daily_pnl: float
    daily_trades: int
    consecutive_losses: int
    total_trades: int
    total_pnl: float
    winning_trades: int
    last_trade_date: str
    # Optional
    last_signal_time: Optional[str]
    last_signal_candle_timestamp: Optional[Any]
    last_trade: Optional[Dict[str, Any]]
    created_at: str
    updated_at: str
    state_version: int  # v1.30.0: 3 = hedge-capable format


# Keys that MUST exist in a valid state (used by validate_state)
BOT_STATE_REQUIRED_KEYS = {
    'positions', 'active_direction', 'daily_pnl', 'daily_trades',
    'consecutive_losses', 'total_trades', 'total_pnl', 'winning_trades',
    'last_trade_date',
}
