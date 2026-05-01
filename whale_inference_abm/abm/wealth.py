"""WealthTracker: per-agent wealth ledger + bankruptcy detection + leaderboard.

Design ref: docs/02-design/features/whale_inference_abm.design.md Sections 5, 4.5 (v0.4 patch).

Single owner of agent.current_wealth mutations (per Agent base class docstring).
Per advisor v0.4 patch: leaderboard COMPUTATION lives here, not in Simulation.

Wealth update on trade:
    realized_pnl_per_unit = (executed_price - cost_basis) for sells
    For ABM v1 spot-like dynamics: position tracked as signed BTC inventory per agent;
    wealth changes as inventory marks-to-market on each trade leg.

Simplified v1 model (cash + BTC inventory):
    Each agent has cash_usd + btc_inventory.
    On BUY: cash -= price × size + fee; btc_inventory += size
    On SELL: cash += price × size - fee; btc_inventory -= size
    Wealth (mark-to-market) = cash + btc_inventory × current_mid_price
    For wealth_tracker.apply_trade: we update cash and inventory; wealth() computes MTM.
    For bankruptcy check: use wealth_at(current_mid).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from abm.constants import BANKRUPTCY_THRESHOLD
from abm.friction import Friction
from abm.types import Role, Side, Trade


@dataclass
class WealthState:
    """Per-agent wealth ledger entry."""

    cash: float
    btc_inventory: float = 0.0
    initial_wealth: float = 0.0  # for ratio-based growth

    def wealth_at(self, mid_price: float) -> float:
        """Mark-to-market total wealth."""
        return self.cash + self.btc_inventory * mid_price


class WealthTracker:
    """Single source of truth for agent wealth state across the sim run.

    Data flow:
      - apply_trade(trade, friction): mutate ledger for both buyer + seller
      - wealth_at(agent_id, mid): compute current MTM
      - is_bankrupt(agent_id, mid): bankruptcy check
      - snapshot(mid): record per-bar wealth distribution into history
      - growth_leaderboard(window_bars, mid): compute rolling-window growth ratios
    """

    def __init__(self) -> None:
        self._ledger: dict[str, WealthState] = {}
        # snapshot history: list of (timestamp_ns, dict[agent_id, wealth_at_mid])
        self._history: list[tuple[int, dict[str, float]]] = []
        # tracks bankrupt agents by their last recorded wealth (for tape lookup, NOT leaderboard)
        self._bankrupt_ids: set[str] = set()

    # ----- Lifecycle -----

    def initialize_agent(self, agent_id: str, initial_cash: float) -> None:
        if agent_id in self._ledger:
            raise ValueError(f"Agent {agent_id!r} already initialized in WealthTracker")
        self._ledger[agent_id] = WealthState(
            cash=initial_cash,
            btc_inventory=0.0,
            initial_wealth=initial_cash,
        )

    def mark_bankrupt(self, agent_id: str) -> None:
        """Sim calls this after registry.remove_agent on bankruptcy."""
        if agent_id not in self._ledger:
            raise KeyError(f"Cannot mark bankrupt: {agent_id!r} not in WealthTracker")
        self._bankrupt_ids.add(agent_id)

    # ----- Trade application -----

    def apply_trade(self, trade: Trade, friction: Friction) -> None:
        """Update buyer + seller ledgers from a single trade. Idempotent if called once per trade."""
        notional = trade.price * trade.size

        buyer_state = self._ledger.get(trade.buyer_agent_id)
        seller_state = self._ledger.get(trade.seller_agent_id)

        if buyer_state is not None:
            buyer_fee = friction.fee(trade.buyer_role, notional)
            buyer_state.cash -= notional + buyer_fee
            buyer_state.btc_inventory += trade.size

        if seller_state is not None:
            seller_fee = friction.fee(trade.seller_role, notional)
            seller_state.cash += notional - seller_fee
            seller_state.btc_inventory -= trade.size

    # ----- Read API -----

    def wealth_at(self, agent_id: str, mid_price: float) -> float:
        if agent_id not in self._ledger:
            raise KeyError(f"Agent {agent_id!r} not in WealthTracker")
        return self._ledger[agent_id].wealth_at(mid_price)

    def cash(self, agent_id: str) -> float:
        return self._ledger[agent_id].cash

    def inventory(self, agent_id: str) -> float:
        return self._ledger[agent_id].btc_inventory

    def is_bankrupt(self, agent_id: str, mid_price: float, threshold: float = BANKRUPTCY_THRESHOLD) -> bool:
        if agent_id in self._bankrupt_ids:
            return True
        return self.wealth_at(agent_id, mid_price) <= threshold

    def alive_ids(self) -> list[str]:
        return [aid for aid in self._ledger if aid not in self._bankrupt_ids]

    # ----- Snapshot + leaderboard -----

    def snapshot(self, timestamp_ns: int, mid_price: float) -> None:
        """Record per-agent wealth at this bar for leaderboard history."""
        snapshot = {
            aid: state.wealth_at(mid_price)
            for aid, state in self._ledger.items()
            if aid not in self._bankrupt_ids
        }
        self._history.append((timestamp_ns, snapshot))

    def growth_leaderboard(
        self,
        lookback_bars: int,
        current_mid: float,
    ) -> list[tuple[str, float]]:
        """Rolling lookback wealth-growth ratio leaderboard (per design v0.4 Section 4.5 patch).

        Returns: list[(agent_id, growth_ratio)] sorted descending.
        - growth_ratio = wealth[t] / wealth[t-lookback]
        - Bankrupt agents EXCLUDED.
        - If history shorter than lookback: returns empty list (cold-start signal to caller).
        - If wealth[t-lookback] <= 0: agent excluded (would divide by ~0).
        """
        if len(self._history) < lookback_bars + 1:
            return []

        # current snapshot index = -1; baseline = -1 - lookback_bars
        _ts_now, snapshot_now_recorded = self._history[-1]
        _ts_old, snapshot_old = self._history[-1 - lookback_bars]

        # Use live wealth_at(current_mid) for current side (more accurate than last recorded snapshot)
        ratios: list[tuple[str, float]] = []
        for aid in snapshot_old:
            if aid in self._bankrupt_ids:
                continue
            if aid not in self._ledger:
                continue  # was removed entirely
            old_w = snapshot_old[aid]
            if old_w <= 0:
                continue
            new_w = self._ledger[aid].wealth_at(current_mid)
            ratios.append((aid, new_w / old_w))

        ratios.sort(key=lambda x: x[1], reverse=True)
        return ratios

    def history_size(self) -> int:
        return len(self._history)
