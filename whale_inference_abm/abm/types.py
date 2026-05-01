"""Core types used across ABM modules.

Design ref: docs/02-design/features/whale_inference_abm.design.md Sections 3.1, 8.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    CANCEL = "CANCEL"


class Role(str, Enum):
    MAKER = "maker"
    TAKER = "taker"


OrderID = str
AgentID = str


@dataclass(frozen=True)
class Order:
    """Submitted order. Immutable after creation; modifications use cancel+resubmit."""

    order_id: OrderID
    agent_id: AgentID
    order_type: OrderType
    side: Side
    size: float
    price: Optional[float] = None  # None for MARKET; required for LIMIT
    sequence_no: int = 0  # assigned by scheduler at submission time

    def __post_init__(self) -> None:
        if self.order_type == OrderType.LIMIT and self.price is None:
            raise ValueError(f"LIMIT order {self.order_id} requires price")
        if self.order_type == OrderType.MARKET and self.price is not None:
            raise ValueError(f"MARKET order {self.order_id} must not have price")
        if self.size <= 0 and self.order_type != OrderType.CANCEL:
            raise ValueError(f"Order {self.order_id} size must be > 0, got {self.size}")


@dataclass(frozen=True)
class OrderIntent:
    """Agent-emitted order intent before simulation assigns order_id + sequence_no.

    Agent.decide() returns list[OrderIntent]; simulation wraps each into a full Order
    by calling scheduler.next_sequence_no() and assigning a unique order_id.
    """

    order_type: OrderType
    side: Side
    size: float
    price: Optional[float] = None

    def __post_init__(self) -> None:
        if self.order_type == OrderType.LIMIT and self.price is None:
            raise ValueError("LIMIT intent requires price")
        if self.order_type == OrderType.MARKET and self.price is not None:
            raise ValueError("MARKET intent must not have price")
        if self.size <= 0:
            raise ValueError(f"OrderIntent size must be > 0, got {self.size}")


@dataclass(frozen=True)
class Trade:
    """A matched trade between two orders."""

    trade_id: str
    timestamp_ns: int
    sequence_no: int
    buyer_agent_id: AgentID
    seller_agent_id: AgentID
    buyer_order_id: OrderID
    seller_order_id: OrderID
    price: float
    size: float
    buyer_role: Role
    seller_role: Role


@dataclass
class PriceLevel:
    """One price level in the book — FIFO queue of resting orders."""

    price: float
    orders: list[Order] = field(default_factory=list)

    @property
    def total_size(self) -> float:
        return sum(o.size for o in self.orders)

    @property
    def is_empty(self) -> bool:
        return len(self.orders) == 0


@dataclass
class OrderbookSnapshot:
    """Read-only view of the book at a moment in time."""

    timestamp_ns: int
    best_bid: Optional[float]
    best_ask: Optional[float]
    bid_depth: list[tuple[float, float]]  # [(price, size), ...] descending price
    ask_depth: list[tuple[float, float]]  # [(price, size), ...] ascending price

    @property
    def mid_price(self) -> Optional[float]:
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2.0

    @property
    def spread(self) -> Optional[float]:
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_ask - self.best_bid
