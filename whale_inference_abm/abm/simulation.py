"""Simulation event-dispatch driver.

Design ref: docs/02-design/features/whale_inference_abm.design.md Sections 2.2, 4.6.5.

Event flow per advisor v0.4 checkpoint Section 4.6.5:
  1. step() pops next event from scheduler
  2. Dispatch by EventType:
       AGENT_DECISION → _dispatch_agent_decision()
       BAR_TICK       → _dispatch_bar_tick()
       ADMISSION      → _dispatch_admission()
       AGENT_REMOVED  → _dispatch_agent_removed()
  3. Each dispatch may push downstream events back to scheduler

Determinism event-ordering rule (Section 4.6.5):
  Same scheduler tick triggers downstream events in DETERMINISTIC ORDER:
    (1) wealth update from any trades that resulted
    (2) bankruptcy check + removal
    (3) MM inventory update from any fills
    (4) re-quote / next-decision schedule push
    (5) tape emit (logger)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from abm.constants import BAR_DURATION_NS, BANKRUPTCY_THRESHOLD
from abm.scheduler import Event, EventType, Scheduler
from abm.types import Order, OrderbookSnapshot, OrderIntent, OrderType, Side, Trade

if TYPE_CHECKING:
    from abm.admission import AdmissionScheduler
    from abm.agents.base import Agent
    from abm.friction import Friction
    from abm.orderbook import Orderbook
    from abm.registry import AgentRegistry
    from abm.wealth import WealthTracker


class NullLogger:
    """No-op logger satisfying the Simulation logger contract for tests / G0 smoke without persistence."""

    def trade(self, trade: Trade) -> None: ...
    def bar_snapshot(self, snapshot: OrderbookSnapshot, wealth_dist: dict[str, float]) -> None: ...
    def agent_removed(self, agent_id: str, reason: str) -> None: ...
    def decision(
        self, agent_id: str, family: str, intent_count: int, observed_state: dict, action: dict
    ) -> None: ...
    def orphan_event_dropped(self, event_type: str, agent_id: str) -> None: ...


class Simulation:
    """Drives the ABM event loop. Single-process deterministic dispatch."""

    def __init__(
        self,
        scheduler: Scheduler,
        orderbook: "Orderbook",
        registry: "AgentRegistry",
        friction: "Friction",
        wealth_tracker: "WealthTracker",
        admission_scheduler: "AdmissionScheduler",
        logger: Optional[Any] = None,
        piggyback_lookback_bars: int = 1000,
        shock_scheduler: Optional[Any] = None,  # v2: ShockScheduler or None
    ) -> None:
        # Logger expected methods (Day 11-13 implementation contract):
        #   logger.trade(trade: Trade) -> None
        #   logger.bar_snapshot(snapshot: OrderbookSnapshot, wealth_dist: dict[str, float]) -> None
        #   logger.agent_removed(agent_id: str, reason: str) -> None
        #   logger.decision(agent_id: str, family: str, intent_count: int, observed_state: dict, action: dict) -> None
        #   logger.orphan_event_dropped(event_type: str, agent_id: str) -> None
        self.scheduler = scheduler
        self.orderbook = orderbook
        self.registry = registry
        self.friction = friction
        self.wealth_tracker = wealth_tracker
        self.admission_scheduler = admission_scheduler
        self.shock_scheduler = shock_scheduler  # v2: optional ShockScheduler
        self.logger = logger if logger is not None else NullLogger()
        self.piggyback_lookback_bars = piggyback_lookback_bars
        self._active_orders: dict[str, set[str]] = {}
        self._last_actions: dict[str, dict[str, Any]] = {}
        self._bar_counter: int = 0
        # v2: track shocks applied for diagnostic logging
        self._shock_log: list[dict[str, Any]] = []
        # Leaderboard cache (advisor G1 → G2 binding requirement):
        # growth_leaderboard() is O(N agents) per call; called per agent decision = O(N²)
        # per bar. Cache invalidates on bar boundary. Single-slot (bar_idx, lookback)
        # → leaderboard list. Small memory, large speedup for 10k+ bar runs.
        self._leaderboard_cache_key: tuple[int, int] = (-1, -1)
        self._leaderboard_cache_value: list[tuple[str, float]] = []
        # Piggyback excluded set cache (per bar) — same pattern
        self._pb_excluded_cache_bar: int = -1
        self._pb_excluded_cache_value: set[str] = set()
        # Track whether any piggyback agent exists (skip context build if not)
        self._has_piggyback: bool = False

    # ----- Initialization -----

    def seed_initial_events(self) -> None:
        """Push initial AGENT_DECISION (one per alive agent), BAR_TICK, and ADMISSION events.

        Call AFTER all initial agents are registered + wealth_tracker.initialize_agent for each.
        Determinism: events pushed in agent_id sorted order so sequence_no assignment is reproducible.
        """
        # Initial decision events for each alive agent at offset
        for agent in sorted(self.registry.alive_agents(), key=lambda a: a.agent_id):
            self._push_decision_event(agent, base_time_ns=0)

        # First BAR_TICK at t=0 (snapshot before first decision tick processed if offset > 0)
        self.scheduler.push(
            Event(
                timestamp_ns=0,
                sequence_no=self.scheduler.next_sequence_no(),
                agent_id="__sim__",
                event_type=EventType.BAR_TICK,
            )
        )

        # First admission event during open phase
        if self.admission_scheduler.is_open_phase(0):
            delay = self.admission_scheduler.next_admission_delay_ns(self.scheduler.rng)
            self.scheduler.push(
                Event(
                    timestamp_ns=delay,
                    sequence_no=self.scheduler.next_sequence_no(),
                    agent_id="__admission__",
                    event_type=EventType.ADMISSION,
                )
            )

        # v2: First shock event if shock scheduler enabled
        if self.shock_scheduler is not None and self.shock_scheduler.enabled:
            self.scheduler.push(
                Event(
                    timestamp_ns=self.shock_scheduler.shock_interval_ns,
                    sequence_no=self.scheduler.next_sequence_no(),
                    agent_id="__shock__",
                    event_type=EventType.SHOCK,
                )
            )

    # ----- Top-level loop -----

    def step(self) -> bool:
        """Pop and dispatch the next event. Returns False if scheduler exhausted/past terminal.

        DETERMINISM CONTRACT: Push order within a single dispatch IS the sequence_no order.
        """
        try:
            event = self.scheduler.pop_next()
        except StopIteration:
            return False

        if event.event_type == EventType.AGENT_DECISION:
            self._dispatch_agent_decision(event)
        elif event.event_type == EventType.BAR_TICK:
            self._dispatch_bar_tick(event)
        elif event.event_type == EventType.ADMISSION:
            self._dispatch_admission(event)
        elif event.event_type == EventType.AGENT_REMOVED:
            self._dispatch_agent_removed(event)
        elif event.event_type == EventType.SHOCK:
            self._dispatch_shock(event)
        else:
            raise ValueError(f"Unknown event type: {event.event_type}")
        return True

    def run(self, max_steps: Optional[int] = None) -> int:
        """Run step() until done. Returns number of events processed."""
        steps = 0
        while self.step():
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break
        return steps

    # ----- Dispatch handlers -----

    def _dispatch_agent_decision(self, event: Event) -> None:
        agent_id = event.agent_id
        # Step 1: orphan event check
        if not self.registry.has(agent_id):
            self.logger.orphan_event_dropped(event.event_type.value, agent_id)
            return

        agent = self.registry.get(agent_id)
        now = self.scheduler.now()
        snapshot = self.orderbook.snapshot(now, depth=10)

        # Step 2: build context
        context = self._build_context(agent, now)

        # Step 4: agent decision
        intents = agent.decide(snapshot, context)

        # Step 5: MM cancel-and-requote ONCE before iterating
        if agent.family == "market_maker":
            for oid in list(self._active_orders.get(agent_id, set())):
                self.orderbook.cancel(oid)
            self._active_orders[agent_id] = set()

        # Step 6: per-intent processing
        last_action_recorded: Optional[dict[str, Any]] = None
        for intent in intents:
            order = self._wrap_intent_to_order(intent, agent)
            trades = self.orderbook.submit(order, now)
            filled_size = sum(t.size for t in trades)

            for trade in trades:
                # 6c.1 wealth update
                self.wealth_tracker.apply_trade(trade, self.friction)

                # 6c.2 bankruptcy check (both sides)
                for side_agent_id in (trade.buyer_agent_id, trade.seller_agent_id):
                    if not self.registry.has(side_agent_id):
                        continue  # already removed earlier in this dispatch
                    mid = snapshot.mid_price if snapshot.mid_price is not None else trade.price
                    if self.wealth_tracker.is_bankrupt(side_agent_id, mid):
                        self.wealth_tracker.mark_bankrupt(side_agent_id)
                        self.registry.remove_agent(side_agent_id)
                        # AGENT_REMOVED event for tape (push deterministic order)
                        self.scheduler.push(
                            Event(
                                timestamp_ns=now,
                                sequence_no=self.scheduler.next_sequence_no(),
                                agent_id=side_agent_id,
                                event_type=EventType.AGENT_REMOVED,
                                payload={"reason": "bankruptcy"},
                            )
                        )

                # 6c.3 MM inventory update
                self._maybe_update_mm_inventory(trade)

                # 6c.4 logger
                self.logger.trade(trade)

                # 6c.5 active_orders cleanup: if resting order was fully consumed, remove from sim's map
                # (orderbook already cleaned its own _order_index; we mirror for sim's tracking)
                resting_oid = (
                    trade.seller_order_id
                    if intent.side == Side.BUY
                    else trade.buyer_order_id
                )
                resting_aid = (
                    trade.seller_agent_id
                    if intent.side == Side.BUY
                    else trade.buyer_agent_id
                )
                if resting_oid not in self.orderbook._order_index:  # type: ignore[attr-defined]
                    self._active_orders.get(resting_aid, set()).discard(resting_oid)

            # Step 6d: incoming LIMIT remainder rested?
            if intent.order_type == OrderType.LIMIT and order.size - filled_size > 1e-12:
                if order.order_id in self.orderbook._order_index:  # type: ignore[attr-defined]
                    self._active_orders.setdefault(agent_id, set()).add(order.order_id)

            last_action_recorded = {"side": intent.side.value, "timestamp_ns": now}

        if last_action_recorded is not None:
            self._last_actions[agent_id] = last_action_recorded

        self.logger.decision(
            agent_id=agent_id,
            family=agent.family,
            intent_count=len(intents),
            observed_state={"mid": snapshot.mid_price},
            action={"intents_count": len(intents)},
        )

        # Step 7: schedule next decision (only if agent still alive)
        if self.registry.has(agent_id):
            delay = agent.next_decision_delay_ns()
            self.scheduler.push(
                Event(
                    timestamp_ns=now + delay,
                    sequence_no=self.scheduler.next_sequence_no(),
                    agent_id=agent_id,
                    event_type=EventType.AGENT_DECISION,
                )
            )

    def _dispatch_bar_tick(self, event: Event) -> None:
        now = self.scheduler.now()
        snapshot = self.orderbook.snapshot(now, depth=10)
        mid = snapshot.mid_price if snapshot.mid_price is not None else 0.0

        # wealth snapshot only when mid is computable
        if mid > 0:
            self.wealth_tracker.snapshot(now, mid)

        wealth_dist = {
            aid: self.wealth_tracker.wealth_at(aid, mid) if mid > 0 else self.wealth_tracker.cash(aid)
            for aid in self.wealth_tracker.alive_ids()
        }
        self.logger.bar_snapshot(snapshot, wealth_dist)

        # Schedule next bar tick (always — bar ticks don't depend on agent activity)
        next_bar_ts = now + BAR_DURATION_NS
        if next_bar_ts < self.scheduler.terminal_time_ns:
            self.scheduler.push(
                Event(
                    timestamp_ns=next_bar_ts,
                    sequence_no=self.scheduler.next_sequence_no(),
                    agent_id="__sim__",
                    event_type=EventType.BAR_TICK,
                )
            )
        self._bar_counter += 1

    def _dispatch_admission(self, event: Event) -> None:
        now = self.scheduler.now()
        # Step 1: create new agent
        new_agent = self.admission_scheduler.create_new_agent(
            self.scheduler.rng, self.registry, now
        )
        # Step 2: register + initialize wealth
        self.registry.add_agent(new_agent)
        self.wealth_tracker.initialize_agent(
            new_agent.agent_id, new_agent.initial_wealth
        )
        # Step 3: push first AGENT_DECISION for new agent
        self._push_decision_event(new_agent, base_time_ns=now)

        # Step 4: schedule next admission ONLY if still in open phase
        if self.admission_scheduler.is_open_phase(now):
            delay = self.admission_scheduler.next_admission_delay_ns(self.scheduler.rng)
            next_ts = now + delay
            if self.admission_scheduler.is_open_phase(next_ts):
                self.scheduler.push(
                    Event(
                        timestamp_ns=next_ts,
                        sequence_no=self.scheduler.next_sequence_no(),
                        agent_id="__admission__",
                        event_type=EventType.ADMISSION,
                    )
                )

    def _dispatch_agent_removed(self, event: Event) -> None:
        reason = event.payload.get("reason", "unknown")
        self.logger.agent_removed(event.agent_id, reason)
        # Cancel any remaining active orders for this agent
        for oid in list(self._active_orders.get(event.agent_id, set())):
            self.orderbook.cancel(oid)
        self._active_orders.pop(event.agent_id, None)

    def _dispatch_shock(self, event: Event) -> None:
        """v2: External wealth shock — pick uniform random agent and multiply their wealth.

        Per advisor 2026-05-01: tests whether v1 mechanism amplifies exogenous wealth
        perturbations into emergent persistent whales (vs. just transient noise).
        """
        if self.shock_scheduler is None:
            return
        now = self.scheduler.now()
        alive_ids = self.registry.alive_ids()
        target_id = self.shock_scheduler.select_target_agent(self.scheduler.rng, alive_ids)
        if target_id is None:
            return  # no agents alive, skip shock

        # Apply shock: directly modify wealth tracker's cash for the target agent
        # (cash multiplier; inventory unchanged so MTM scales naturally)
        state = self.wealth_tracker._ledger.get(target_id)
        if state is None:
            return
        wealth_before = state.cash  # using cash as proxy; inventory * mid would be MTM
        new_cash = self.shock_scheduler.apply_shock(state.cash)
        state.cash = new_cash

        shock_idx = self.shock_scheduler.next_shock_index()
        agent_family = self.registry.get(target_id).family if self.registry.has(target_id) else "unknown"
        self._shock_log.append({
            "shock_idx": shock_idx,
            "timestamp_ns": now,
            "bar": now // BAR_DURATION_NS,
            "target_agent_id": target_id,
            "target_family": agent_family,
            "cash_before": wealth_before,
            "cash_after": new_cash,
        })

        # Schedule next shock if still within terminal_time
        next_ts = now + self.shock_scheduler.shock_interval_ns
        if next_ts < self.scheduler.terminal_time_ns:
            self.scheduler.push(
                Event(
                    timestamp_ns=next_ts,
                    sequence_no=self.scheduler.next_sequence_no(),
                    agent_id="__shock__",
                    event_type=EventType.SHOCK,
                )
            )

    # ----- Helpers -----

    def _wrap_intent_to_order(self, intent: OrderIntent, agent: "Agent") -> Order:
        return Order(
            order_id=agent.next_order_id(),
            agent_id=agent.agent_id,
            order_type=intent.order_type,
            side=intent.side,
            size=intent.size,
            price=intent.price,
            sequence_no=self.scheduler.next_sequence_no(),
        )

    def _build_context(self, agent: "Agent", now_ns: int) -> dict[str, Any]:
        # Fast path: only piggyback agents read context fields. Skip all O(N) work otherwise.
        if agent.family != "piggyback":
            return {}

        snapshot = self.orderbook.snapshot(now_ns, depth=1)
        mid = snapshot.mid_price if snapshot.mid_price is not None else 0.0

        # Leaderboard with per-bar cache (advisor G2 prerequisite)
        leaderboard: list[tuple[str, float]] = []
        if mid > 0:
            bar_idx = now_ns // BAR_DURATION_NS
            cache_key = (bar_idx, self.piggyback_lookback_bars)
            if cache_key == self._leaderboard_cache_key:
                leaderboard = self._leaderboard_cache_value
            else:
                leaderboard = self.wealth_tracker.growth_leaderboard(
                    self.piggyback_lookback_bars, mid
                )
                self._leaderboard_cache_key = cache_key
                self._leaderboard_cache_value = leaderboard

        # Piggyback excluded with per-bar cache
        bar_idx = now_ns // BAR_DURATION_NS
        if bar_idx == self._pb_excluded_cache_bar:
            piggyback_excluded_ids = self._pb_excluded_cache_value
        else:
            piggyback_excluded_ids = {
                a.agent_id for a in self.registry.alive_agents() if a.family == "piggyback"
            }
            self._pb_excluded_cache_bar = bar_idx
            self._pb_excluded_cache_value = piggyback_excluded_ids

        return {
            "wealth_growth_leaderboard": leaderboard,
            "last_actions_by_agent": self._last_actions,  # read-only by piggyback; no copy
            "piggyback_excluded_ids": piggyback_excluded_ids,
        }

    def _push_decision_event(self, agent: "Agent", base_time_ns: int) -> None:
        ts = base_time_ns + agent.decision_offset_ns
        # Ensure ts is not in the past (scheduler rejects past events)
        ts = max(ts, base_time_ns)
        self.scheduler.push(
            Event(
                timestamp_ns=ts,
                sequence_no=self.scheduler.next_sequence_no(),
                agent_id=agent.agent_id,
                event_type=EventType.AGENT_DECISION,
            )
        )

    def _maybe_update_mm_inventory(self, trade: Trade) -> None:
        for aid, signed_size in (
            (trade.buyer_agent_id, trade.size),
            (trade.seller_agent_id, -trade.size),
        ):
            if not self.registry.has(aid):
                continue
            agent = self.registry.get(aid)
            if agent.family == "market_maker":
                # MarketMakerAgent has update_inventory method
                update = getattr(agent, "update_inventory", None)
                if callable(update):
                    update(signed_size)

    @property
    def bar_counter(self) -> int:
        return self._bar_counter
