"""Orderbook unit tests.

Coverage targets (per design v0.3 Section 11.1):
- limit add / cancel / match
- market order walks the book
- state invariants (no crossed book, no empty levels, no zero-size)
- state_hash() consistency
- price-time priority within levels
- partial fill on insufficient liquidity
"""

from __future__ import annotations

import pytest

from abm.orderbook import Orderbook
from abm.types import Order, OrderType, Side


def _limit(order_id: str, agent: str, side: Side, price: float, size: float, seq: int) -> Order:
    return Order(
        order_id=order_id,
        agent_id=agent,
        order_type=OrderType.LIMIT,
        side=side,
        size=size,
        price=price,
        sequence_no=seq,
    )


def _market(order_id: str, agent: str, side: Side, size: float, seq: int) -> Order:
    return Order(
        order_id=order_id,
        agent_id=agent,
        order_type=OrderType.MARKET,
        side=side,
        size=size,
        sequence_no=seq,
    )


# ----- Limit add / snapshot -----

def test_empty_book_no_best_quotes() -> None:
    book = Orderbook()
    assert book.best_bid() is None
    assert book.best_ask() is None
    snap = book.snapshot(timestamp_ns=0)
    assert snap.mid_price is None
    assert snap.spread is None


def test_single_limit_bid_creates_best_bid() -> None:
    book = Orderbook()
    trades = book.submit(_limit("o1", "a1", Side.BUY, 100.0, 1.0, 1), timestamp_ns=0)
    assert trades == []
    assert book.best_bid() == (100.0, 1.0)
    assert book.best_ask() is None


def test_single_limit_ask_creates_best_ask() -> None:
    book = Orderbook()
    trades = book.submit(_limit("o1", "a1", Side.SELL, 101.0, 1.0, 1), timestamp_ns=0)
    assert trades == []
    assert book.best_ask() == (101.0, 1.0)
    assert book.best_bid() is None


def test_two_bids_higher_price_is_best() -> None:
    book = Orderbook()
    book.submit(_limit("o1", "a1", Side.BUY, 100.0, 1.0, 1), timestamp_ns=0)
    book.submit(_limit("o2", "a2", Side.BUY, 101.0, 1.0, 2), timestamp_ns=0)
    assert book.best_bid() == (101.0, 1.0)


def test_two_asks_lower_price_is_best() -> None:
    book = Orderbook()
    book.submit(_limit("o1", "a1", Side.SELL, 102.0, 1.0, 1), timestamp_ns=0)
    book.submit(_limit("o2", "a2", Side.SELL, 101.0, 1.0, 2), timestamp_ns=0)
    assert book.best_ask() == (101.0, 1.0)


# ----- Cancel -----

def test_cancel_removes_order() -> None:
    book = Orderbook()
    book.submit(_limit("o1", "a1", Side.BUY, 100.0, 1.0, 1), timestamp_ns=0)
    assert book.cancel("o1") is True
    assert book.best_bid() is None


def test_cancel_unknown_returns_false() -> None:
    book = Orderbook()
    assert book.cancel("nonexistent") is False


def test_cancel_removes_only_that_order_at_level() -> None:
    book = Orderbook()
    book.submit(_limit("o1", "a1", Side.BUY, 100.0, 1.0, 1), timestamp_ns=0)
    book.submit(_limit("o2", "a2", Side.BUY, 100.0, 2.0, 2), timestamp_ns=0)
    book.cancel("o1")
    assert book.best_bid() == (100.0, 2.0)


# ----- Match: limit crossing -----

def test_limit_buy_crosses_existing_ask() -> None:
    book = Orderbook()
    book.submit(_limit("ask1", "seller", Side.SELL, 100.0, 1.0, 1), timestamp_ns=0)
    trades = book.submit(_limit("buy1", "buyer", Side.BUY, 100.0, 1.0, 2), timestamp_ns=10)
    assert len(trades) == 1
    t = trades[0]
    assert t.price == 100.0
    assert t.size == 1.0
    assert t.buyer_agent_id == "buyer"
    assert t.seller_agent_id == "seller"
    assert book.best_bid() is None  # both sides cleared
    assert book.best_ask() is None


def test_limit_buy_partial_cross_then_rest() -> None:
    book = Orderbook()
    book.submit(_limit("ask1", "seller", Side.SELL, 100.0, 1.0, 1), timestamp_ns=0)
    trades = book.submit(_limit("buy1", "buyer", Side.BUY, 100.0, 3.0, 2), timestamp_ns=10)
    assert len(trades) == 1
    assert trades[0].size == 1.0
    # 2.0 unfilled rests as best bid at 100.0
    assert book.best_bid() == (100.0, 2.0)


def test_limit_no_cross_just_rests() -> None:
    book = Orderbook()
    book.submit(_limit("ask1", "seller", Side.SELL, 102.0, 1.0, 1), timestamp_ns=0)
    trades = book.submit(_limit("buy1", "buyer", Side.BUY, 100.0, 1.0, 2), timestamp_ns=10)
    assert trades == []
    assert book.best_bid() == (100.0, 1.0)
    assert book.best_ask() == (102.0, 1.0)


# ----- Match: market walking -----

def test_market_buy_walks_multiple_levels() -> None:
    book = Orderbook()
    book.submit(_limit("ask1", "s1", Side.SELL, 100.0, 1.0, 1), timestamp_ns=0)
    book.submit(_limit("ask2", "s2", Side.SELL, 101.0, 1.0, 2), timestamp_ns=0)
    book.submit(_limit("ask3", "s3", Side.SELL, 102.0, 1.0, 3), timestamp_ns=0)
    trades = book.submit(_market("m1", "buyer", Side.BUY, 2.5, 4), timestamp_ns=10)
    # 1.0 @ 100 + 1.0 @ 101 + 0.5 @ 102 = 2.5 size, 3 trades
    assert len(trades) == 3
    assert [t.price for t in trades] == [100.0, 101.0, 102.0]
    assert [t.size for t in trades] == [1.0, 1.0, 0.5]
    # 102 level has 0.5 left
    assert book.best_ask() == (102.0, 0.5)


def test_market_buy_partial_fill_on_insufficient_liquidity() -> None:
    book = Orderbook()
    book.submit(_limit("ask1", "s1", Side.SELL, 100.0, 1.0, 1), timestamp_ns=0)
    trades = book.submit(_market("m1", "buyer", Side.BUY, 5.0, 2), timestamp_ns=10)
    # Only 1.0 available
    assert len(trades) == 1
    assert trades[0].size == 1.0
    assert book.best_ask() is None


# ----- Price-Time priority -----

def test_fifo_within_level_first_in_first_filled() -> None:
    book = Orderbook()
    book.submit(_limit("a1", "seller_first", Side.SELL, 100.0, 1.0, 1), timestamp_ns=0)
    book.submit(_limit("a2", "seller_second", Side.SELL, 100.0, 1.0, 2), timestamp_ns=0)
    trades = book.submit(_market("m1", "buyer", Side.BUY, 1.0, 3), timestamp_ns=10)
    assert len(trades) == 1
    assert trades[0].seller_agent_id == "seller_first"  # FIFO


# ----- Invariants -----

def test_no_crossed_book_after_match() -> None:
    book = Orderbook()
    book.submit(_limit("ask1", "s", Side.SELL, 100.0, 1.0, 1), timestamp_ns=0)
    book.submit(_limit("buy1", "b", Side.BUY, 100.0, 1.0, 2), timestamp_ns=10)
    # Both cleared, no crossing
    assert book.best_bid() is None and book.best_ask() is None
    book.submit(_limit("ask2", "s", Side.SELL, 101.0, 1.0, 3), timestamp_ns=20)
    book.submit(_limit("buy2", "b", Side.BUY, 100.0, 1.0, 4), timestamp_ns=30)
    bb = book.best_bid()
    ba = book.best_ask()
    assert bb is not None and ba is not None
    assert bb[0] < ba[0]


def test_invalid_limit_without_price_raises() -> None:
    with pytest.raises(ValueError, match="LIMIT order"):
        Order(
            order_id="bad",
            agent_id="a",
            order_type=OrderType.LIMIT,
            side=Side.BUY,
            size=1.0,
            sequence_no=1,
        )


def test_invalid_market_with_price_raises() -> None:
    with pytest.raises(ValueError, match="MARKET order"):
        Order(
            order_id="bad",
            agent_id="a",
            order_type=OrderType.MARKET,
            side=Side.BUY,
            size=1.0,
            price=100.0,
            sequence_no=1,
        )


def test_invalid_zero_size_raises() -> None:
    with pytest.raises(ValueError, match="size must be > 0"):
        Order(
            order_id="bad",
            agent_id="a",
            order_type=OrderType.LIMIT,
            side=Side.BUY,
            size=0.0,
            price=100.0,
            sequence_no=1,
        )


# ----- state_hash determinism -----

def test_state_hash_empty_consistent() -> None:
    b1 = Orderbook()
    b2 = Orderbook()
    assert b1.state_hash() == b2.state_hash()


def test_state_hash_same_orders_same_hash() -> None:
    b1 = Orderbook()
    b2 = Orderbook()
    for book in (b1, b2):
        book.submit(_limit("o1", "a1", Side.BUY, 100.0, 1.0, 1), timestamp_ns=0)
        book.submit(_limit("o2", "a2", Side.SELL, 102.0, 2.0, 2), timestamp_ns=0)
    assert b1.state_hash() == b2.state_hash()


def test_state_hash_different_orders_different_hash() -> None:
    b1 = Orderbook()
    b2 = Orderbook()
    b1.submit(_limit("o1", "a1", Side.BUY, 100.0, 1.0, 1), timestamp_ns=0)
    b2.submit(_limit("o1", "a1", Side.BUY, 100.0, 2.0, 1), timestamp_ns=0)  # diff size
    assert b1.state_hash() != b2.state_hash()


def test_state_hash_after_cancel_matches_pre_submit() -> None:
    book = Orderbook()
    h0 = book.state_hash()
    book.submit(_limit("o1", "a1", Side.BUY, 100.0, 1.0, 1), timestamp_ns=0)
    book.cancel("o1")
    assert book.state_hash() == h0
