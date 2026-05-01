"""Simulation event-dispatch driver.

Design ref: docs/02-design/features/whale_inference_abm.design.md Sections 2.2, 4.6.5.

INTERFACE SKETCH (Day 8-10) — bodies pending advisor checkpoint approval.

Event flow per advisor v0.4 checkpoint Section 4.6.5:
  1. step() pops next event from scheduler
  2. Dispatch by EventType:
       AGENT_DECISION → _dispatch_agent_decision()
       BAR_TICK       → _dispatch_bar_tick()
       ADMISSION      → _dispatch_admission()
       AGENT_REMOVED  → _dispatch_agent_removed()  (informational only; registry already mutated)
  3. Each dispatch may push downstream events back to scheduler

Determinism event-ordering rule (Section 4.6.5):
  Same scheduler tick triggers downstream events in DETERMINISTIC ORDER:
    (1) wealth update from any trades that resulted
    (2) bankruptcy check + removal
    (3) MM inventory update from any fills
    (4) re-quote / next-decision schedule push
    (5) tape emit (logger)
  Document this ordering in dispatch method docstrings.

NOTHING below has executable bodies yet. Advisor checkpoint REQUIRED before implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from abm.scheduler import Event, EventType, Scheduler
from abm.types import Order, OrderIntent, Trade

if TYPE_CHECKING:
    # Avoid circular imports for modules built later
    from abm.admission import AdmissionScheduler
    from abm.agents.base import Agent
    from abm.friction import Friction
    from abm.orderbook import Orderbook
    from abm.registry import AgentRegistry
    from abm.wealth import WealthTracker


class Simulation:
    """Drives the ABM event loop. Single-process deterministic dispatch.

    Composition (all injected — sim does not construct collaborators):
      - scheduler: time progression + RNG + sequence_no issuance
      - orderbook: order match + state_hash
      - registry: alive agents + sub-seed derivation
      - friction: fee calculation per trade leg
      - wealth_tracker: PnL update, bankruptcy detection, leaderboard
      - admission_scheduler: open-phase Poisson admissions + frozen-window enforcement
      - logger (Day 11-13): NDJSON event emission

    Active-orders tracking (advisor v0.4 #4):
      `_active_orders[agent_id] = set[OrderID]` for cancel-and-requote (MM 10s cycle).
      Sim is sole writer to this map.
    """

    def __init__(
        self,
        scheduler: Scheduler,
        orderbook: "Orderbook",
        registry: "AgentRegistry",
        friction: "Friction",
        wealth_tracker: "WealthTracker",
        admission_scheduler: "AdmissionScheduler",
        logger: Any,  # structlog logger; type tightened in Day 11-13
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
        self.logger = logger
        self._active_orders: dict[str, set[str]] = {}

    # ----- Top-level loop -----

    def step(self) -> bool:
        """Pop and dispatch the next event. Returns False if scheduler exhausted/past terminal.

        DETERMINISM CONTRACT: Push order within a single dispatch IS the sequence_no order.
        Sim MUST push downstream events in deterministic source-code order. Do not reorder
        push() calls thinking it's cosmetic — sequence_no is monotonic and reordering
        breaks reproducibility.

        BODY PENDING ADVISOR CHECKPOINT.
        """
        raise NotImplementedError("Pending advisor checkpoint approval")

    def run(self) -> None:
        """Run step() until done.

        BODY PENDING. Likely:
            while self.step():
                pass
        """
        raise NotImplementedError("Pending advisor checkpoint approval")

    # ----- Dispatch handlers -----

    def _dispatch_agent_decision(self, event: Event) -> None:
        """Agent's decision tick.

        Sequence (deterministic, advisor-corrected):
          1. Look up agent. If agent_id NOT in registry (orphan event from prior bankruptcy):
               - logger.orphan_event_dropped(event_type, agent_id)
               - return  (silent drop, known-and-handled case)
          2. Build context via wealth_tracker.growth_leaderboard(lookback) + last_actions
             (Leaderboard COMPUTATION lives in WealthTracker, NOT here. Sim is dispatch only.)
          3. Snapshot orderbook for agent.decide()
          4. agent.decide(snapshot, context) → list[OrderIntent]
          5. (MM ONLY, ONCE before iterating intents)
               cancel all _active_orders[agent_id] via orderbook.cancel(),
               then clear _active_orders[agent_id]
             (Per advisor patch: cancellation is ONCE per decision tick, not per intent.)
          6. For each intent in deterministic source-code order:
               a. Wrap intent → Order with order_id (agent.next_order_id()) + sequence_no
                  (scheduler.next_sequence_no())
               b. orderbook.submit() → list[Trade]
               c. For each trade in deterministic order:
                  - wealth_tracker.apply_trade(trade, friction)
                  - bankruptcy check (per side); if bankrupt:
                      registry.remove_agent(agent_id) +
                      push AGENT_REMOVED event (deterministic push order)
                  - MM inventory update (if MM is participant — sim detects by agent family)
                  - logger.trade(trade)
                  - For each trade: if resting order WAS CONSUMED (size went to 0),
                    REMOVE its order_id from _active_orders[resting_agent_id]
                    (Per advisor patch: sync _active_orders with orderbook state.)
               d. If intent was LIMIT and remainder rested in book (= submitted size > sum trade sizes):
                    add the new order_id to _active_orders[agent_id]
          7. Push next AGENT_DECISION event at now() + agent.next_decision_delay_ns()
             (only if agent still alive; bankrupt agents already removed in step 6c)

        BODY PENDING ADVISOR CHECKPOINT.
        """
        raise NotImplementedError("Pending advisor checkpoint approval")

    def _dispatch_bar_tick(self, event: Event) -> None:
        """Per-bar snapshot emission.

        Sequence:
          1. orderbook.snapshot(now, depth=10)
          2. wealth_tracker.snapshot() → record per-agent wealth at this bar
          3. logger.bar_snapshot(snapshot, wealth_dist)
          4. Push next BAR_TICK at now + BAR_DURATION_NS

        BODY PENDING.
        """
        raise NotImplementedError("Pending advisor checkpoint approval")

    def _dispatch_admission(self, event: Event) -> None:
        """New agent joins (open phase only).

        Sequence (advisor-corrected):
          1. admission_scheduler.create_new_agent(scheduler.rng, registry, now())
             - family drawn uniform via scheduler.rng (master RNG, NOT per-agent)
             - sub-seed derived via registry.derived_seed(new_agent_id)
             - decision_offset_ns drawn from registry.make_decision_offset()
          2. registry.add_agent(new_agent)
          3. Push first AGENT_DECISION event for new agent at now() + decision_offset_ns
          4. If now() < admission_scheduler.T_open_ns:
               delay_ns = admission_scheduler.next_admission_delay_ns(scheduler.rng)
               push next ADMISSION event at now() + delay_ns
             else:
               # Frozen window reached. NO further ADMISSION events scheduled.
               pass
             (Per advisor patch: do NOT push None; either push event OR don't push.)

        RNG source contract: admission RNG draws use scheduler.rng (master).
        Per-agent RNGs don't exist yet at admission time.

        BODY PENDING.
        """
        raise NotImplementedError("Pending advisor checkpoint approval")

    def _dispatch_agent_removed(self, event: Event) -> None:
        """Tape-only: registry already mutated at bankruptcy detection in _dispatch_agent_decision.

        Sequence:
          1. logger.agent_removed(event.agent_id, reason='bankruptcy')
          2. (Optional Day 8-10) cancel any remaining active orders for this agent

        BODY PENDING.
        """
        raise NotImplementedError("Pending advisor checkpoint approval")

    # ----- Helpers (interface only) -----

    def _wrap_intent_to_order(self, intent: OrderIntent, agent: "Agent") -> Order:
        """Combine OrderIntent (agent-emitted) + sequence_no (scheduler) + order_id (agent counter).

        BODY PENDING — but interface is fixed:
            order_id = agent.next_order_id()
            sequence_no = self.scheduler.next_sequence_no()
            return Order(order_id, agent.agent_id, intent.order_type, intent.side, intent.size, intent.price, sequence_no)
        """
        raise NotImplementedError("Pending advisor checkpoint approval")

    def _build_context(self, agent: "Agent", now_ns: int) -> dict[str, Any]:
        """Build agent.decide() context dict.

        Returns:
            wealth_growth_leaderboard: list[(agent_id, growth_ratio)]
            last_actions_by_agent: dict[agent_id, {"side": str, "timestamp_ns": int}]
            piggyback_excluded_ids: set[str] — all piggyback-family agents

        BODY PENDING.
        """
        raise NotImplementedError("Pending advisor checkpoint approval")
