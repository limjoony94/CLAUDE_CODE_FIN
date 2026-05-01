"""Scheduler unit tests.

Coverage targets (per design v0.3 Section 11.1):
- priority order
- tie-break by sequence_no
- tie-break by agent_id
- terminal_time stop
- now() updates on pop
- next_sequence_no monotonic
- queue_size correct
- empty heap behavior
- past-event push rejected (causality enforcement)
"""

from __future__ import annotations

import numpy as np
import pytest

from abm.scheduler import Event, EventType, Scheduler


def _ev(ts: int, seq: int, agent: str, etype: EventType = EventType.AGENT_DECISION) -> Event:
    return Event(timestamp_ns=ts, sequence_no=seq, agent_id=agent, event_type=etype)


# ----- Construction -----

def test_scheduler_invalid_terminal_time_raises() -> None:
    with pytest.raises(ValueError, match="terminal_time_ns must be > 0"):
        Scheduler(seed=42, terminal_time_ns=0)


def test_scheduler_initial_state() -> None:
    s = Scheduler(seed=42, terminal_time_ns=1000)
    assert s.now() == 0
    assert s.queue_size == 0
    assert s.is_done() is True  # empty heap = done
    assert s.peek_next() is None


# ----- Priority order -----

def test_pop_returns_earliest_timestamp() -> None:
    s = Scheduler(seed=42, terminal_time_ns=1_000_000)
    s.push(_ev(300, 3, "a"))
    s.push(_ev(100, 1, "b"))
    s.push(_ev(200, 2, "c"))
    assert s.pop_next().timestamp_ns == 100
    assert s.pop_next().timestamp_ns == 200
    assert s.pop_next().timestamp_ns == 300


def test_now_updates_on_pop() -> None:
    s = Scheduler(seed=42, terminal_time_ns=1000)
    s.push(_ev(100, 1, "a"))
    s.push(_ev(500, 2, "b"))
    assert s.now() == 0
    s.pop_next()
    assert s.now() == 100
    s.pop_next()
    assert s.now() == 500


# ----- Tie-break -----

def test_tiebreak_by_sequence_no_at_same_timestamp() -> None:
    s = Scheduler(seed=42, terminal_time_ns=1000)
    s.push(_ev(100, 5, "z"))
    s.push(_ev(100, 1, "z"))
    s.push(_ev(100, 3, "z"))
    assert [s.pop_next().sequence_no for _ in range(3)] == [1, 3, 5]


def test_tiebreak_by_agent_id_at_same_timestamp_and_sequence() -> None:
    s = Scheduler(seed=42, terminal_time_ns=1000)
    s.push(_ev(100, 1, "charlie"))
    s.push(_ev(100, 1, "alice"))
    s.push(_ev(100, 1, "bob"))
    assert [s.pop_next().agent_id for _ in range(3)] == ["alice", "bob", "charlie"]


# ----- Terminal time -----

def test_pop_past_terminal_raises_stopiteration() -> None:
    s = Scheduler(seed=42, terminal_time_ns=500)
    s.push(_ev(100, 1, "a"))
    s.push(_ev(600, 2, "b"))  # past terminal
    s.pop_next()  # ts=100 OK
    with pytest.raises(StopIteration):
        s.pop_next()  # ts=600 >= 500


def test_is_done_true_when_only_post_terminal_remains() -> None:
    s = Scheduler(seed=42, terminal_time_ns=500)
    s.push(_ev(600, 1, "a"))
    assert s.is_done() is True


def test_is_done_false_when_pre_terminal_remains() -> None:
    s = Scheduler(seed=42, terminal_time_ns=500)
    s.push(_ev(100, 1, "a"))
    assert s.is_done() is False


# ----- Sequence numbers -----

def test_next_sequence_no_monotonic() -> None:
    s = Scheduler(seed=42, terminal_time_ns=1000)
    seq1 = s.next_sequence_no()
    seq2 = s.next_sequence_no()
    seq3 = s.next_sequence_no()
    assert seq1 < seq2 < seq3
    assert seq1 >= 1


# ----- Causality (no past events) -----

def test_push_past_event_rejected() -> None:
    s = Scheduler(seed=42, terminal_time_ns=1000)
    s.push(_ev(500, 1, "a"))
    s.pop_next()  # current_time = 500
    with pytest.raises(ValueError, match="Cannot push event in the past"):
        s.push(_ev(100, 2, "a"))


def test_push_at_current_time_allowed() -> None:
    s = Scheduler(seed=42, terminal_time_ns=1000)
    s.push(_ev(500, 1, "a"))
    s.pop_next()
    # Push at exactly current time = OK (e.g., same-bar follow-up event)
    s.push(_ev(500, 2, "b"))
    assert s.queue_size == 1


# ----- RNG determinism -----

def test_same_seed_same_rng_draws() -> None:
    s1 = Scheduler(seed=42, terminal_time_ns=1000)
    s2 = Scheduler(seed=42, terminal_time_ns=1000)
    draws1 = [s1.rng.uniform() for _ in range(5)]
    draws2 = [s2.rng.uniform() for _ in range(5)]
    assert draws1 == draws2


def test_different_seed_different_rng_draws() -> None:
    s1 = Scheduler(seed=42, terminal_time_ns=1000)
    s2 = Scheduler(seed=43, terminal_time_ns=1000)
    draws1 = [s1.rng.uniform() for _ in range(5)]
    draws2 = [s2.rng.uniform() for _ in range(5)]
    assert draws1 != draws2


# ----- Queue size + peek -----

def test_queue_size_reflects_pushes_and_pops() -> None:
    s = Scheduler(seed=42, terminal_time_ns=1000)
    assert s.queue_size == 0
    s.push(_ev(100, 1, "a"))
    s.push(_ev(200, 2, "b"))
    assert s.queue_size == 2
    s.pop_next()
    assert s.queue_size == 1


def test_peek_next_does_not_consume() -> None:
    s = Scheduler(seed=42, terminal_time_ns=1000)
    s.push(_ev(100, 1, "a"))
    peeked = s.peek_next()
    assert peeked is not None
    assert peeked.timestamp_ns == 100
    assert s.queue_size == 1
    popped = s.pop_next()
    assert popped == peeked


# ----- Empty heap -----

def test_pop_empty_raises_stopiteration() -> None:
    s = Scheduler(seed=42, terminal_time_ns=1000)
    with pytest.raises(StopIteration, match="empty"):
        s.pop_next()


# ----- Event validation -----

def test_event_negative_timestamp_raises() -> None:
    with pytest.raises(ValueError, match="timestamp_ns must be >= 0"):
        Event(
            timestamp_ns=-1,
            sequence_no=1,
            agent_id="a",
            event_type=EventType.AGENT_DECISION,
        )


def test_event_negative_sequence_no_raises() -> None:
    with pytest.raises(ValueError, match="sequence_no must be >= 0"):
        Event(
            timestamp_ns=0,
            sequence_no=-1,
            agent_id="a",
            event_type=EventType.AGENT_DECISION,
        )


# ----- Heavy load: many events -----

def test_many_events_correctly_ordered() -> None:
    """Push 1000 events with random timestamps; pop must return strictly non-decreasing ts."""
    s = Scheduler(seed=42, terminal_time_ns=10_000_000)
    rng = np.random.default_rng(123)
    timestamps = rng.integers(0, 10_000_000 - 1, size=1000).tolist()
    for i, ts in enumerate(timestamps):
        s.push(_ev(int(ts), i + 1, f"agent_{i:04d}"))

    last_ts = -1
    popped_count = 0
    while not s.is_done():
        ev = s.pop_next()
        assert ev.timestamp_ns >= last_ts
        last_ts = ev.timestamp_ns
        popped_count += 1
    assert popped_count == 1000


# ----- Mixed event types -----

def test_mixed_event_types_pop_by_priority_not_type() -> None:
    s = Scheduler(seed=42, terminal_time_ns=1000)
    s.push(_ev(300, 3, "a", EventType.BAR_TICK))
    s.push(_ev(100, 1, "b", EventType.AGENT_DECISION))
    s.push(_ev(200, 2, "c", EventType.ADMISSION))
    types_in_order = [s.pop_next().event_type for _ in range(3)]
    assert types_in_order == [EventType.AGENT_DECISION, EventType.ADMISSION, EventType.BAR_TICK]
