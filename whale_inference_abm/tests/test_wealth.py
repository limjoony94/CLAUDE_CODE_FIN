"""WealthTracker tests."""

from __future__ import annotations

import pytest

from abm.constants import BANKRUPTCY_THRESHOLD
from abm.friction import Friction
from abm.types import Role, Trade
from abm.wealth import WealthTracker


def _trade(buyer: str, seller: str, price: float, size: float,
           buyer_role: Role = Role.TAKER, seller_role: Role = Role.MAKER,
           ts: int = 0, seq: int = 1, tid: str = "t_1") -> Trade:
    return Trade(
        trade_id=tid,
        timestamp_ns=ts,
        sequence_no=seq,
        buyer_agent_id=buyer,
        seller_agent_id=seller,
        buyer_order_id=f"{buyer}_o",
        seller_order_id=f"{seller}_o",
        price=price,
        size=size,
        buyer_role=buyer_role,
        seller_role=seller_role,
    )


# ----- Lifecycle -----

def test_initialize_agent() -> None:
    w = WealthTracker()
    w.initialize_agent("a1", initial_cash=1000.0)
    assert w.cash("a1") == 1000.0
    assert w.inventory("a1") == 0.0
    assert w.wealth_at("a1", mid_price=100.0) == 1000.0


def test_duplicate_initialize_raises() -> None:
    w = WealthTracker()
    w.initialize_agent("a1", initial_cash=1000.0)
    with pytest.raises(ValueError, match="already initialized"):
        w.initialize_agent("a1", initial_cash=500.0)


# ----- Trade application -----

def test_buyer_pays_cash_gains_inventory() -> None:
    w = WealthTracker()
    w.initialize_agent("buyer", initial_cash=10000.0)
    w.initialize_agent("seller", initial_cash=10000.0)
    f = Friction()

    t = _trade("buyer", "seller", price=100.0, size=1.0)  # taker buy, maker sell
    w.apply_trade(t, f)

    # Buyer: cash -= 100 + 0.05 (taker fee 0.05% × 100 = 0.05); inventory += 1.0
    assert w.cash("buyer") == pytest.approx(10000.0 - 100.0 - 0.05)
    assert w.inventory("buyer") == pytest.approx(1.0)


def test_seller_receives_cash_loses_inventory() -> None:
    w = WealthTracker()
    w.initialize_agent("buyer", initial_cash=10000.0)
    w.initialize_agent("seller", initial_cash=10000.0)
    f = Friction()

    t = _trade("buyer", "seller", price=100.0, size=1.0)  # taker buy, maker sell
    w.apply_trade(t, f)

    # Seller: cash += 100 - 0.02 (maker fee 0.02% × 100 = 0.02); inventory -= 1.0
    assert w.cash("seller") == pytest.approx(10000.0 + 100.0 - 0.02)
    assert w.inventory("seller") == pytest.approx(-1.0)


def test_apply_trade_with_unknown_agent_silent() -> None:
    """If an agent not in tracker (e.g., test fixture), apply_trade silently skips that side."""
    w = WealthTracker()
    w.initialize_agent("buyer", initial_cash=10000.0)
    f = Friction()
    t = _trade("buyer", "unknown_seller", price=100.0, size=1.0)
    w.apply_trade(t, f)
    assert w.cash("buyer") == pytest.approx(10000.0 - 100.0 - 0.05)


def test_mtm_wealth_includes_inventory() -> None:
    w = WealthTracker()
    w.initialize_agent("a", initial_cash=5000.0)
    f = Friction()
    # Buy 1 BTC @ 50000 (taker) — cash drains
    w.apply_trade(_trade("a", "b", price=50000.0, size=0.1), f)
    # Cash now ~5000 - 5000 - 2.5 = -2.5 (technically negative cash)
    # Inventory = 0.1 BTC; at mid 60000, MTM = -2.5 + 0.1*60000 = 5997.5
    assert w.wealth_at("a", mid_price=60000.0) == pytest.approx(-2.5 + 6000.0)


# ----- Bankruptcy -----

def test_bankruptcy_when_wealth_below_threshold() -> None:
    w = WealthTracker()
    w.initialize_agent("a", initial_cash=0.5)
    assert w.is_bankrupt("a", mid_price=100.0)


def test_not_bankrupt_when_above_threshold() -> None:
    w = WealthTracker()
    w.initialize_agent("a", initial_cash=BANKRUPTCY_THRESHOLD + 1.0)
    assert not w.is_bankrupt("a", mid_price=100.0)


def test_mark_bankrupt_persists_even_if_wealth_recovers() -> None:
    w = WealthTracker()
    w.initialize_agent("a", initial_cash=10000.0)
    w.mark_bankrupt("a")
    # Even though wealth > threshold, marked-bankrupt is sticky
    assert w.is_bankrupt("a", mid_price=100.0)


def test_alive_ids_excludes_bankrupt() -> None:
    w = WealthTracker()
    w.initialize_agent("a", initial_cash=1000.0)
    w.initialize_agent("b", initial_cash=1000.0)
    w.mark_bankrupt("b")
    assert w.alive_ids() == ["a"]


# ----- Snapshot + leaderboard -----

def test_snapshot_records_history() -> None:
    w = WealthTracker()
    w.initialize_agent("a", initial_cash=1000.0)
    w.snapshot(timestamp_ns=0, mid_price=100.0)
    w.snapshot(timestamp_ns=60_000_000_000, mid_price=100.0)
    assert w.history_size() == 2


def test_leaderboard_empty_during_cold_start() -> None:
    w = WealthTracker()
    w.initialize_agent("a", initial_cash=1000.0)
    w.snapshot(timestamp_ns=0, mid_price=100.0)
    # lookback=10 but only 1 snapshot
    leaderboard = w.growth_leaderboard(lookback_bars=10, current_mid=100.0)
    assert leaderboard == []


def test_leaderboard_returns_growth_ratios_sorted_desc() -> None:
    w = WealthTracker()
    w.initialize_agent("winner", initial_cash=1000.0)
    w.initialize_agent("loser", initial_cash=1000.0)
    f = Friction()

    # Snapshot baseline at t=0
    w.snapshot(timestamp_ns=0, mid_price=100.0)

    # Winner buys 1 BTC, then mid moves up
    w.apply_trade(_trade("winner", "loser", price=100.0, size=1.0), f)

    # 11 more snapshots so we have lookback=10 worth of history
    for i in range(1, 12):
        w.snapshot(timestamp_ns=i * 60_000_000_000, mid_price=100.0 + i)

    # Now mid = 110, winner has +1 BTC, loser has -1 BTC
    leaderboard = w.growth_leaderboard(lookback_bars=10, current_mid=110.0)
    assert len(leaderboard) == 2
    # Winner first (higher growth)
    assert leaderboard[0][0] == "winner"
    assert leaderboard[1][0] == "loser"


def test_leaderboard_excludes_bankrupt() -> None:
    w = WealthTracker()
    w.initialize_agent("alive", initial_cash=1000.0)
    w.initialize_agent("dead", initial_cash=1000.0)
    w.snapshot(timestamp_ns=0, mid_price=100.0)
    for i in range(1, 12):
        w.snapshot(timestamp_ns=i * 60_000_000_000, mid_price=100.0)

    w.mark_bankrupt("dead")
    leaderboard = w.growth_leaderboard(lookback_bars=10, current_mid=100.0)
    assert all(aid != "dead" for aid, _ratio in leaderboard)


def test_wealth_at_unknown_raises() -> None:
    w = WealthTracker()
    with pytest.raises(KeyError):
        w.wealth_at("nonexistent", mid_price=100.0)
