"""
Engulf 5m Bot - Data Models
Dataclasses for type-safe data structures.
"""

import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any

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
        if self.positions and (time.time() - self.positions_time) < ttl:
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

    # Recent trades for rolling stats
    recent_wins: List[float] = field(default_factory=list)
    recent_losses: List[float] = field(default_factory=list)

    # Timing stats
    avg_api_latency_ms: float = 0.0
    api_call_count: int = 0

    # Session info
    session_start: str = ""
    last_updated: str = ""

    def update_trade(self, pnl_pct: float) -> None:
        """Update metrics after a trade."""
        self.total_trades += 1
        self.total_pnl_pct += pnl_pct

        if pnl_pct > 0:
            self.winning_trades += 1
            self.recent_wins.append(pnl_pct)
            if len(self.recent_wins) > METRICS_WINDOW_SIZE:
                self.recent_wins.pop(0)
        else:
            self.losing_trades += 1
            self.recent_losses.append(abs(pnl_pct))
            if len(self.recent_losses) > METRICS_WINDOW_SIZE:
                self.recent_losses.pop(0)

        self._recalculate()

    def _recalculate(self) -> None:
        """Recalculate derived metrics."""
        if self.total_trades > 0:
            self.actual_win_rate = (self.winning_trades / self.total_trades) * 100

        if self.recent_wins:
            self.actual_avg_win = sum(self.recent_wins) / len(self.recent_wins)

        if self.recent_losses:
            self.actual_avg_loss = sum(self.recent_losses) / len(self.recent_losses)

        # Calculate edge (simplified)
        if self.total_trades >= MIN_TRADES_FOR_COMPARISON:
            max_dd = max(self.recent_losses) if self.recent_losses else 1
            self.actual_edge = self.total_pnl_pct / (max_dd + 0.1)

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
Edge:               {self.expected_edge:.1f}       {self.actual_edge:.1f}      {self.actual_edge - self.expected_edge:+.1f}
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
            'recent_wins': self.recent_wins[-20:],  # Keep last 20
            'recent_losses': self.recent_losses[-20:],
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
        metrics.recent_wins = data.get('recent_wins', [])
        metrics.recent_losses = data.get('recent_losses', [])
        return metrics


@dataclass
class Position:
    """Trading position data structure."""

    direction: str  # 'LONG' or 'SHORT'
    entry_price: float
    quantity: float
    remaining_quantity: float
    tp_price: float
    sl_price: float
    vol_mult: float = 1.0
    scale_out_enabled: bool = False
    scale_out_stages: List[Dict] = field(default_factory=list)
    entry_time: str = ""
    reason: str = ""
    order_id: Optional[str] = None
    tp_order_id: Optional[str] = None
    sl_order_id: Optional[str] = None
    estimated_entry: Optional[float] = None
    recovered: bool = False
    needs_tpsl: bool = False
    # Rotation (순환매) fields
    rotation_enabled: bool = False
    is_partial: bool = False  # True after TP1 fill
    avg_entry_price: float = 0.0  # Weighted average entry after refills
    total_entries: int = 1  # Number of entries (original + refills)
    refill_entries: List[Dict] = field(default_factory=list)  # History of refill entries

    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary."""
        return {
            'direction': self.direction,
            'entry_price': self.entry_price,
            'quantity': self.quantity,
            'remaining_quantity': self.remaining_quantity,
            'tp_price': self.tp_price,
            'sl_price': self.sl_price,
            'vol_mult': self.vol_mult,
            'scale_out_enabled': self.scale_out_enabled,
            'scale_out_stages': self.scale_out_stages,
            'entry_time': self.entry_time,
            'reason': self.reason,
            'order_id': self.order_id,
            'tp_order_id': self.tp_order_id,
            'sl_order_id': self.sl_order_id,
            'estimated_entry': self.estimated_entry,
            'recovered': self.recovered,
            'needs_tpsl': self.needs_tpsl,
            # Rotation fields
            'rotation_enabled': self.rotation_enabled,
            'is_partial': self.is_partial,
            'avg_entry_price': self.avg_entry_price,
            'total_entries': self.total_entries,
            'refill_entries': self.refill_entries,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Position':
        """Create Position from dictionary."""
        return cls(
            direction=data.get('direction', ''),
            entry_price=data.get('entry_price', 0),
            quantity=data.get('quantity', 0),
            remaining_quantity=data.get('remaining_quantity', 0),
            tp_price=data.get('tp_price', 0),
            sl_price=data.get('sl_price', 0),
            vol_mult=data.get('vol_mult', 1.0),
            scale_out_enabled=data.get('scale_out_enabled', False),
            scale_out_stages=data.get('scale_out_stages', []),
            entry_time=data.get('entry_time', ''),
            reason=data.get('reason', ''),
            order_id=data.get('order_id'),
            tp_order_id=data.get('tp_order_id'),
            sl_order_id=data.get('sl_order_id'),
            estimated_entry=data.get('estimated_entry'),
            recovered=data.get('recovered', False),
            needs_tpsl=data.get('needs_tpsl', False),
            # Rotation fields
            rotation_enabled=data.get('rotation_enabled', False),
            is_partial=data.get('is_partial', False),
            avg_entry_price=data.get('avg_entry_price', 0.0),
            total_entries=data.get('total_entries', 1),
            refill_entries=data.get('refill_entries', []),
        )


@dataclass
class BotState:
    """Bot state data structure."""

    position: Optional[Position] = None
    last_signal_time: Optional[str] = None
    last_signal_candle_timestamp: Optional[int] = None
    daily_pnl: float = 0.0
    daily_trades: int = 0
    total_trades: int = 0
    total_pnl: float = 0.0
    winning_trades: int = 0
    last_trade: Optional[Dict] = None
    last_trade_date: str = ""
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for JSON serialization."""
        return {
            'position': self.position.to_dict() if self.position else None,
            'last_signal_time': self.last_signal_time,
            'last_signal_candle_timestamp': self.last_signal_candle_timestamp,
            'daily_pnl': self.daily_pnl,
            'daily_trades': self.daily_trades,
            'total_trades': self.total_trades,
            'total_pnl': self.total_pnl,
            'winning_trades': self.winning_trades,
            'last_trade': self.last_trade,
            'last_trade_date': self.last_trade_date,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BotState':
        """Create BotState from dictionary."""
        position_data = data.get('position')
        position = Position.from_dict(position_data) if position_data else None

        return cls(
            position=position,
            last_signal_time=data.get('last_signal_time'),
            last_signal_candle_timestamp=data.get('last_signal_candle_timestamp'),
            daily_pnl=data.get('daily_pnl', 0.0),
            daily_trades=data.get('daily_trades', 0),
            total_trades=data.get('total_trades', 0),
            total_pnl=data.get('total_pnl', 0.0),
            winning_trades=data.get('winning_trades', 0),
            last_trade=data.get('last_trade'),
            last_trade_date=data.get('last_trade_date', ''),
            created_at=data.get('created_at', ''),
            updated_at=data.get('updated_at', ''),
        )
