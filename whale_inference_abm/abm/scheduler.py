"""Deterministic event-driven scheduler.

Design ref: docs/02-design/features/whale_inference_abm.design.md Section 3.2.

- Priority queue (heapq) keyed on (timestamp_ns, sequence_no, agent_id)
- sequence_no = global monotonic counter, assigned at Event creation via scheduler.next_sequence_no()
- Tie-break: earlier ts wins; same ts -> lower seq wins; same both -> lex agent_id wins
- time = sim-internal int64 ns (NOT wall-clock; design F1 patch enforced)
- Single-process. No async, no threading.
- Scheduler is the ONLY consumer of `seed`. Agents get derived sub-seeds from registry.
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np


class EventType(str, Enum):
    AGENT_DECISION = "AGENT_DECISION"
    BAR_TICK = "BAR_TICK"
    ADMISSION = "ADMISSION"
    AGENT_REMOVED = "AGENT_REMOVED"
    SHOCK = "SHOCK"  # v2 ABM external wealth perturbation event


@dataclass(frozen=True)
class Event:
    """Scheduled event. Immutable after creation.

    sort_key = (timestamp_ns, sequence_no, agent_id) — design Section 3.2 total order.
    """

    timestamp_ns: int
    sequence_no: int
    agent_id: str
    event_type: EventType
    payload: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)

    @property
    def sort_key(self) -> tuple[int, int, str]:
        return (self.timestamp_ns, self.sequence_no, self.agent_id)

    def __post_init__(self) -> None:
        if self.timestamp_ns < 0:
            raise ValueError(f"timestamp_ns must be >= 0, got {self.timestamp_ns}")
        if self.sequence_no < 0:
            raise ValueError(f"sequence_no must be >= 0, got {self.sequence_no}")


class Scheduler:
    """Single-process deterministic event scheduler.

    Ownership: scheduler owns time progression and main RNG.
    Events are pushed by simulation; agents schedule their NEXT decision via
    sim-level helper (not scheduler directly), receiving sequence_no from
    scheduler.next_sequence_no() at event creation.

    Determinism: same seed -> same RNG draws. Heap order = total order on sort_key
    plus monotonic tiebreak counter (so Event objects themselves never get compared).
    """

    def __init__(self, seed: int, terminal_time_ns: int) -> None:
        if terminal_time_ns <= 0:
            raise ValueError(
                f"terminal_time_ns must be > 0, got {terminal_time_ns}"
            )
        self.terminal_time_ns: int = terminal_time_ns
        self._seed: int = seed
        self._rng: np.random.Generator = np.random.default_rng(seed)
        # heap items: (sort_key, tiebreak_count, event)
        self._heap: list[tuple[tuple[int, int, str], int, Event]] = []
        self._sequence_counter: int = 0
        self._heap_tiebreak = itertools.count()
        self._current_time_ns: int = 0

    # ----- Sequence number issuance -----

    def next_sequence_no(self) -> int:
        """Issue a new monotonic sequence number for use in Event creation."""
        self._sequence_counter += 1
        return self._sequence_counter

    # ----- Queue operations -----

    def push(self, event: Event) -> None:
        if event.timestamp_ns < self._current_time_ns:
            raise ValueError(
                f"Cannot push event in the past: event_ts={event.timestamp_ns}, "
                f"current_ts={self._current_time_ns}"
            )
        heapq.heappush(
            self._heap,
            (event.sort_key, next(self._heap_tiebreak), event),
        )

    def pop_next(self) -> Event:
        """Pop the next event, advance current_time_ns. Raises StopIteration if past terminal."""
        if not self._heap:
            raise StopIteration("Scheduler heap is empty")
        sort_key, tiebreak, event = heapq.heappop(self._heap)
        if event.timestamp_ns >= self.terminal_time_ns:
            # Restore — no events past terminal are ever popped
            heapq.heappush(self._heap, (sort_key, tiebreak, event))
            raise StopIteration(
                f"Next event ts={event.timestamp_ns} >= terminal={self.terminal_time_ns}"
            )
        self._current_time_ns = event.timestamp_ns
        return event

    def peek_next(self) -> Optional[Event]:
        """Inspect next event without popping. Returns None if heap empty."""
        if not self._heap:
            return None
        _key, _tb, event = self._heap[0]
        return event

    def is_done(self) -> bool:
        """True if no more events before terminal_time_ns."""
        if not self._heap:
            return True
        next_ev = self.peek_next()
        assert next_ev is not None
        return next_ev.timestamp_ns >= self.terminal_time_ns

    # ----- Read-only state -----

    def now(self) -> int:
        return self._current_time_ns

    @property
    def rng(self) -> np.random.Generator:
        return self._rng

    @property
    def queue_size(self) -> int:
        return len(self._heap)

    @property
    def seed(self) -> int:
        return self._seed
