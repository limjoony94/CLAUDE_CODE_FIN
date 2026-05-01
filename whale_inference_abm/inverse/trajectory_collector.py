"""Collect per-agent action trajectories from simulation trade tape.

Used by all 3 anchors as input. Each agent participates in trades as either buyer or seller;
this module groups trades by agent and produces sorted action records per agent.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from abm.types import Trade


@dataclass(frozen=True)
class ActionRecord:
    """A single agent participation in a trade."""

    timestamp_ns: int
    side: str  # 'buy' or 'sell'
    role: str  # 'taker' or 'maker'
    size: float
    price: float


def collect_per_agent_actions(trades: Iterable[Trade]) -> dict[str, list[ActionRecord]]:
    """Group trades by agent. Each agent gets a list of ActionRecords sorted by timestamp.

    A trade contributes 2 records: one to buyer's list (side='buy', role=trade.buyer_role),
    one to seller's list (side='sell', role=trade.seller_role).
    """
    per_agent: dict[str, list[ActionRecord]] = defaultdict(list)

    for t in trades:
        per_agent[t.buyer_agent_id].append(
            ActionRecord(
                timestamp_ns=t.timestamp_ns,
                side="buy",
                role=t.buyer_role.value,
                size=t.size,
                price=t.price,
            )
        )
        per_agent[t.seller_agent_id].append(
            ActionRecord(
                timestamp_ns=t.timestamp_ns,
                side="sell",
                role=t.seller_role.value,
                size=t.size,
                price=t.price,
            )
        )

    # Sort each agent's actions by timestamp (deterministic ordering for reproducible features)
    for aid in per_agent:
        per_agent[aid].sort(key=lambda r: r.timestamp_ns)

    return dict(per_agent)


def trade_count_per_agent(per_agent: dict[str, list[ActionRecord]]) -> dict[str, int]:
    return {aid: len(records) for aid, records in per_agent.items()}


def filter_by_min_trades(
    per_agent: dict[str, list[ActionRecord]], min_trades: int
) -> dict[str, list[ActionRecord]]:
    """Filter agents with insufficient trade history. Required for stable feature computation."""
    return {aid: records for aid, records in per_agent.items() if len(records) >= min_trades}
