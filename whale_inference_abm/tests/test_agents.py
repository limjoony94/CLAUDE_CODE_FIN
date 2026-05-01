"""Canonical agent tests. Design Section 11.1 coverage."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from abm.agents.market_maker import MM_DECISION_PERIOD_NS, MarketMakerAgent
from abm.agents.mean_reversion import MeanReversionAgent
from abm.agents.momentum import MomentumAgent
from abm.agents.piggyback import PiggybackAgent
from abm.agents.random_agent import RandomAgent
from abm.constants import BAR_DURATION_NS, NS_PER_SECOND
from abm.types import OrderbookSnapshot, OrderType, Side


def _snap(mid: float, ts: int = 0) -> OrderbookSnapshot:
    return OrderbookSnapshot(
        timestamp_ns=ts,
        best_bid=mid - 0.5,
        best_ask=mid + 0.5,
        bid_depth=[(mid - 0.5, 1.0)],
        ask_depth=[(mid + 0.5, 1.0)],
    )


def _no_book() -> OrderbookSnapshot:
    return OrderbookSnapshot(
        timestamp_ns=0,
        best_bid=None,
        best_ask=None,
        bid_depth=[],
        ask_depth=[],
    )


# ============= Momentum =============

def _momentum() -> MomentumAgent:
    return MomentumAgent(
        agent_id="m1",
        initial_wealth=1000.0,
        rng=np.random.default_rng(42),
        decision_offset_ns=0,
        N=3,
    )


def test_momentum_warmup_no_signal() -> None:
    a = _momentum()
    for i in range(3):
        assert a.decide(_snap(100.0 + i), {}) == []


def test_momentum_uptrend_buy() -> None:
    a = _momentum()
    for p in [100.0, 101.0, 102.0]:
        a.decide(_snap(p), {})  # warmup
    intents = a.decide(_snap(105.0), {})
    assert len(intents) == 1
    assert intents[0].side == Side.BUY
    assert intents[0].order_type == OrderType.MARKET


def test_momentum_downtrend_sell() -> None:
    a = _momentum()
    for p in [105.0, 104.0, 103.0]:
        a.decide(_snap(p), {})
    intents = a.decide(_snap(100.0), {})
    assert len(intents) == 1
    assert intents[0].side == Side.SELL


def test_momentum_flat_no_signal() -> None:
    a = _momentum()
    for _ in range(4):
        out = a.decide(_snap(100.0), {})
    assert out == []


def test_momentum_no_book_no_action() -> None:
    a = _momentum()
    assert a.decide(_no_book(), {}) == []


def test_momentum_decision_freq_is_bar() -> None:
    assert _momentum().next_decision_delay_ns() == BAR_DURATION_NS


# ============= Mean-Reversion =============

def _meanrev(threshold: float = 0.005) -> MeanReversionAgent:
    return MeanReversionAgent(
        agent_id="mr1",
        initial_wealth=1000.0,
        rng=np.random.default_rng(42),
        decision_offset_ns=0,
        N=5,
        threshold=threshold,
    )


def test_meanrev_warmup_no_signal() -> None:
    a = _meanrev()
    for _ in range(4):
        assert a.decide(_snap(100.0), {}) == []


def test_meanrev_high_deviation_sells() -> None:
    """Build MA = 100 (5 prior prices) then snap to 110 -> deviation = +10% -> SELL.

    Verifies advisor checkpoint fix: MA must be N PRIOR prices, NOT including current.
    Old (buggy) implementation would compute MA over [100,100,100,100,100,110]/6 = 101.67,
    yielding deviation 8.2% instead of 10%.
    """
    a = _meanrev()
    for _ in range(5):
        a.decide(_snap(100.0), {})  # warmup builds price_history = [100,100,100,100,100]
    intents = a.decide(_snap(110.0), {})  # MA still = 100 (current excluded), dev = +10%
    assert len(intents) == 1
    assert intents[0].side == Side.SELL


def test_meanrev_ma_excludes_current_price() -> None:
    """Direct verification: with prior 5 prices = 100, current = 110, MA must be 100 not 101.67."""
    a = _meanrev(threshold=0.09)  # 9% threshold
    for _ in range(5):
        a.decide(_snap(100.0), {})
    # Old buggy MA = 101.67 -> deviation 8.2% < 9% threshold -> NO trade
    # New correct MA = 100 -> deviation 10% > 9% threshold -> SELL
    intents = a.decide(_snap(110.0), {})
    assert len(intents) == 1, "MA off-by-one fix: deviation should be 10%, above 9% threshold"
    assert intents[0].side == Side.SELL


def test_meanrev_low_deviation_buys() -> None:
    a = _meanrev()
    for _ in range(5):
        a.decide(_snap(100.0), {})
    intents = a.decide(_snap(90.0), {})
    assert len(intents) == 1
    assert intents[0].side == Side.BUY


def test_meanrev_in_band_no_signal() -> None:
    a = _meanrev(threshold=0.05)
    for _ in range(5):
        a.decide(_snap(100.0), {})
    intents = a.decide(_snap(101.0), {})  # 1% deviation, below 5% threshold
    assert intents == []


# ============= Market-Maker =============

def _mm() -> MarketMakerAgent:
    return MarketMakerAgent(
        agent_id="mm1",
        initial_wealth=1000.0,
        rng=np.random.default_rng(42),
        decision_offset_ns=0,
        base_spread=0.001,
    )


def test_mm_quotes_both_sides() -> None:
    intents = _mm().decide(_snap(100.0), {})
    assert len(intents) == 2
    sides = {i.side for i in intents}
    assert sides == {Side.BUY, Side.SELL}
    assert all(i.order_type == OrderType.LIMIT for i in intents)


def test_mm_bid_below_ask() -> None:
    intents = _mm().decide(_snap(100.0), {})
    bid = next(i for i in intents if i.side == Side.BUY)
    ask = next(i for i in intents if i.side == Side.SELL)
    assert bid.price is not None and ask.price is not None
    assert bid.price < ask.price


def test_mm_inventory_skew_widens_spread_on_long() -> None:
    a = _mm()
    a.update_inventory(1.0)  # long inventory
    intents = a.decide(_snap(100.0), {})
    bid_ask_spread = next(i.price for i in intents if i.side == Side.SELL) - next(
        i.price for i in intents if i.side == Side.BUY
    )
    assert bid_ask_spread > 0.001 * 100.0  # wider than base spread


def test_mm_decision_freq_10s() -> None:
    assert _mm().next_decision_delay_ns() == MM_DECISION_PERIOD_NS == 10 * NS_PER_SECOND


# ============= Random =============

def _random() -> RandomAgent:
    return RandomAgent(
        agent_id="r1",
        initial_wealth=1000.0,
        rng=np.random.default_rng(42),
        decision_offset_ns=0,
    )


def test_random_emits_one_intent() -> None:
    intents = _random().decide(_snap(100.0), {})
    assert len(intents) == 1


def test_random_action_distribution_balanced() -> None:
    """Over many samples, BUY/SELL should be roughly equal (chi-square sanity)."""
    a = _random()
    sides = []
    for _ in range(200):
        intents = a.decide(_snap(100.0), {})
        sides.append(intents[0].side)
    buy_count = sides.count(Side.BUY)
    # Expect 100 ± reasonable; binomial(200, 0.5) ~ N(100, 7) so 80-120 is safe
    assert 80 <= buy_count <= 120


def test_random_market_limit_mix() -> None:
    a = _random()
    types = []
    for _ in range(200):
        intents = a.decide(_snap(100.0), {})
        types.append(intents[0].order_type)
    market_count = types.count(OrderType.MARKET)
    assert 80 <= market_count <= 120


def test_random_poisson_delay_positive() -> None:
    a = _random()
    delays = [a.next_decision_delay_ns() for _ in range(50)]
    assert all(d >= 1 for d in delays)
    # Mean should be near 120s (lambda = 1/120). Very loose bounds for 50 samples.
    mean_delay_sec = sum(delays) / len(delays) / NS_PER_SECOND
    assert 30 < mean_delay_sec < 300


# ============= Piggyback =============

def _pb(lookback: int = 1000) -> PiggybackAgent:
    return PiggybackAgent(
        agent_id="pb1",
        initial_wealth=1000.0,
        rng=np.random.default_rng(42),
        decision_offset_ns=0,
        lookback_bars=lookback,
    )


def test_piggyback_cold_start_no_trade() -> None:
    """B2 patch: no trades when t < lookback × BAR_DURATION_NS."""
    a = _pb(lookback=10)
    snap = _snap(100.0, ts=5 * BAR_DURATION_NS)  # halfway through cold-start
    intents = a.decide(snap, {"wealth_growth_leaderboard": [("top", 0.5)]})
    assert intents == []


def test_piggyback_post_cold_start_no_context_no_trade() -> None:
    a = _pb(lookback=10)
    snap = _snap(100.0, ts=15 * BAR_DURATION_NS)  # past cold-start
    assert a.decide(snap, {}) == []


def test_piggyback_follows_top_performer() -> None:
    a = _pb(lookback=10)
    snap = _snap(100.0, ts=15 * BAR_DURATION_NS)
    ctx: dict[str, Any] = {
        "wealth_growth_leaderboard": [("top_perf", 0.5), ("worst", -0.1)],
        "last_actions_by_agent": {
            "top_perf": {
                "side": "buy",
                "timestamp_ns": 15 * BAR_DURATION_NS - 2 * BAR_DURATION_NS,
            }
        },
    }
    intents = a.decide(snap, ctx)
    assert len(intents) == 1
    assert intents[0].side == Side.BUY


def test_piggyback_excludes_self() -> None:
    a = _pb(lookback=10)
    snap = _snap(100.0, ts=15 * BAR_DURATION_NS)
    ctx: dict[str, Any] = {
        "wealth_growth_leaderboard": [("pb1", 1.0)],  # self
        "last_actions_by_agent": {
            "pb1": {"side": "buy", "timestamp_ns": 13 * BAR_DURATION_NS},
        },
    }
    assert a.decide(snap, ctx) == []


def test_piggyback_lag_not_yet_elapsed_no_trade() -> None:
    """Action too recent (within delay window) should not be copied."""
    a = _pb(lookback=10)
    snap = _snap(100.0, ts=15 * BAR_DURATION_NS)
    ctx: dict[str, Any] = {
        "wealth_growth_leaderboard": [("top_perf", 0.5)],
        "last_actions_by_agent": {
            "top_perf": {
                "side": "buy",
                "timestamp_ns": 15 * BAR_DURATION_NS - 1,  # 1 ns ago
            }
        },
    }
    assert a.decide(snap, ctx) == []
