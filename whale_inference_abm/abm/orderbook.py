"""Continuous Double Auction (CDA) limit order book.

Design ref: docs/02-design/features/whale_inference_abm.design.md Section 3.1.

Implementation pattern: SortedDict per side, price -> PriceLevel (FIFO deque-like list).
- bids: SortedDict, best = peekitem(-1) (highest price)
- asks: SortedDict, best = peekitem(0) (lowest price)
- Order match: walk opposite side, FIFO within price level
"""

from __future__ import annotations

import hashlib
import json
from typing import Optional

from sortedcontainers import SortedDict

from abm.types import (
    AgentID,
    Order,
    OrderbookSnapshot,
    OrderID,
    OrderType,
    PriceLevel,
    Role,
    Side,
    Trade,
)


class Orderbook:
    """Continuous double auction limit order book.

    Invariants (asserted in _assert_invariants):
    - best_bid().price < best_ask().price (no crossed book after match)
    - all PriceLevel sizes > 0 (empty levels removed)
    - FIFO order preserved within each level
    """

    def __init__(self, *, strict: bool = True) -> None:
        # strict=True: assert invariants on every mutation (defensive, slower).
        # strict=False: skip checks (production sim runs). Invariants are guaranteed
        # by construction; defensive checks were a 73% bottleneck at 1k bars.
        self._strict: bool = strict
        # Both SortedDict[float, PriceLevel]; bids accessed via peekitem(-1) for max
        self.bids: SortedDict = SortedDict()
        self.asks: SortedDict = SortedDict()
        # order_id -> (side, price) for cancel lookup
        self._order_index: dict[OrderID, tuple[Side, float]] = {}
        # monotonic trade id counter
        self._trade_counter: int = 0

    # ----- Public read-only API -----

    def best_bid(self) -> Optional[tuple[float, float]]:
        if not self.bids:
            return None
        price, level = self.bids.peekitem(-1)
        return (price, level.total_size)

    def best_ask(self) -> Optional[tuple[float, float]]:
        if not self.asks:
            return None
        price, level = self.asks.peekitem(0)
        return (price, level.total_size)

    def snapshot(self, timestamp_ns: int, depth: int = 10) -> OrderbookSnapshot:
        bid_depth = [
            (price, level.total_size)
            for price, level in reversed(list(self.bids.items()[-depth:]))
        ]
        ask_depth = [
            (price, level.total_size) for price, level in self.asks.items()[:depth]
        ]
        return OrderbookSnapshot(
            timestamp_ns=timestamp_ns,
            best_bid=self.best_bid()[0] if self.best_bid() else None,
            best_ask=self.best_ask()[0] if self.best_ask() else None,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
        )

    def state_hash(self) -> str:
        """SHA256 over canonical book state. Used by determinism tests."""
        state = {
            "bids": [
                (
                    float(price),
                    [
                        (o.order_id, o.agent_id, o.size, o.sequence_no)
                        for o in level.orders
                    ],
                )
                for price, level in self.bids.items()
            ],
            "asks": [
                (
                    float(price),
                    [
                        (o.order_id, o.agent_id, o.size, o.sequence_no)
                        for o in level.orders
                    ],
                )
                for price, level in self.asks.items()
            ],
        }
        canonical = json.dumps(state, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    # ----- Mutating API -----

    def submit(self, order: Order, timestamp_ns: int) -> list[Trade]:
        """Submit an order. Returns list of trades resulting from match.

        - MARKET: walk opposite book until size filled or empty (taker)
        - LIMIT crossing: take liquidity for the crossed quantity, rest the remainder
        - LIMIT non-crossing: rest in book (maker on future match)
        """
        if order.order_type == OrderType.CANCEL:
            raise ValueError(
                "Use orderbook.cancel(order_id) for cancellation, not submit()"
            )

        trades: list[Trade] = []
        remaining_size = order.size

        if order.order_type == OrderType.MARKET:
            trades = self._match_against_book(
                order=order,
                remaining_size=remaining_size,
                timestamp_ns=timestamp_ns,
                taker_is_market=True,
            )
        else:  # LIMIT
            # First, attempt to cross
            trades = self._match_against_book(
                order=order,
                remaining_size=remaining_size,
                timestamp_ns=timestamp_ns,
                taker_is_market=False,
                limit_price=order.price,
            )
            filled = sum(t.size for t in trades)
            unfilled = order.size - filled
            if unfilled > 1e-12:
                # Rest unfilled remainder
                resting_order = Order(
                    order_id=order.order_id,
                    agent_id=order.agent_id,
                    order_type=OrderType.LIMIT,
                    side=order.side,
                    size=unfilled,
                    price=order.price,
                    sequence_no=order.sequence_no,
                )
                self._add_to_book(resting_order)

        self._assert_invariants()
        return trades

    def cancel(self, order_id: OrderID) -> bool:
        if order_id not in self._order_index:
            return False
        side, price = self._order_index[order_id]
        book = self.bids if side == Side.BUY else self.asks
        level = book[price]
        level.orders = [o for o in level.orders if o.order_id != order_id]
        if level.is_empty:
            del book[price]
        del self._order_index[order_id]
        self._assert_invariants()
        return True

    # ----- Internal -----

    def _match_against_book(
        self,
        order: Order,
        remaining_size: float,
        timestamp_ns: int,
        taker_is_market: bool,
        limit_price: Optional[float] = None,
    ) -> list[Trade]:
        """Walk opposite book matching against best prices.

        For LIMIT crossing: stop when best opposite price is no longer crossed by limit_price.
        For MARKET: walk until remaining_size == 0 or opposite book empty.
        """
        trades: list[Trade] = []
        # Opposite book to walk
        opposite = self.asks if order.side == Side.BUY else self.bids

        while remaining_size > 1e-12 and len(opposite) > 0:
            if order.side == Side.BUY:
                best_price, level = opposite.peekitem(0)  # min ask
                if not taker_is_market and limit_price is not None and best_price > limit_price:
                    break  # LIMIT BUY no longer crosses
            else:  # SELL
                best_price, level = opposite.peekitem(-1)  # max bid
                if not taker_is_market and limit_price is not None and best_price < limit_price:
                    break  # LIMIT SELL no longer crosses

            # Match against FIFO at this price level
            i = 0
            while i < len(level.orders) and remaining_size > 1e-12:
                resting = level.orders[i]
                trade_size = min(resting.size, remaining_size)
                trade = self._build_trade(
                    timestamp_ns=timestamp_ns,
                    incoming_order=order,
                    resting_order=resting,
                    trade_size=trade_size,
                    trade_price=best_price,
                )
                trades.append(trade)
                remaining_size -= trade_size

                # Update or remove resting order
                new_resting_size = resting.size - trade_size
                if new_resting_size <= 1e-12:
                    del self._order_index[resting.order_id]
                    level.orders.pop(i)
                else:
                    level.orders[i] = Order(
                        order_id=resting.order_id,
                        agent_id=resting.agent_id,
                        order_type=resting.order_type,
                        side=resting.side,
                        size=new_resting_size,
                        price=resting.price,
                        sequence_no=resting.sequence_no,
                    )
                    i += 1  # only advance if not popped

            if level.is_empty:
                del opposite[best_price]

        return trades

    def _add_to_book(self, order: Order) -> None:
        if order.order_type != OrderType.LIMIT or order.price is None:
            raise ValueError(f"Only LIMIT orders rest in book; got {order.order_type}")
        book = self.bids if order.side == Side.BUY else self.asks
        if order.price not in book:
            book[order.price] = PriceLevel(price=order.price)
        book[order.price].orders.append(order)
        self._order_index[order.order_id] = (order.side, order.price)

    def _build_trade(
        self,
        timestamp_ns: int,
        incoming_order: Order,
        resting_order: Order,
        trade_size: float,
        trade_price: float,
    ) -> Trade:
        # Incoming = taker, resting = maker (always)
        if incoming_order.side == Side.BUY:
            buyer = incoming_order
            seller = resting_order
        else:
            buyer = resting_order
            seller = incoming_order

        buyer_role = Role.TAKER if buyer is incoming_order else Role.MAKER
        seller_role = Role.TAKER if seller is incoming_order else Role.MAKER

        self._trade_counter += 1
        return Trade(
            trade_id=f"t_{self._trade_counter:010d}",
            timestamp_ns=timestamp_ns,
            sequence_no=incoming_order.sequence_no,
            buyer_agent_id=buyer.agent_id,
            seller_agent_id=seller.agent_id,
            buyer_order_id=buyer.order_id,
            seller_order_id=seller.order_id,
            price=trade_price,
            size=trade_size,
            buyer_role=buyer_role,
            seller_role=seller_role,
        )

    def _assert_invariants(self) -> None:
        if not self._strict:
            return
        bb = self.best_bid()
        ba = self.best_ask()
        if bb is not None and ba is not None:
            assert bb[0] < ba[0], f"Crossed book: bid {bb[0]} >= ask {ba[0]}"
        for book in (self.bids, self.asks):
            for price, level in book.items():
                assert not level.is_empty, f"Empty level at price {price}"
                for o in level.orders:
                    assert o.size > 1e-12, f"Zero-size order {o.order_id} at {price}"
