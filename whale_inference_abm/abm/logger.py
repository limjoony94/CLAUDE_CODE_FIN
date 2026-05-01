"""NDJSON event-tape logger.

Design ref: docs/02-design/features/whale_inference_abm.design.md Section 8.

Writes 4 NDJSON files per run (one record per line, newline-delimited JSON):
  - trade_tape.ndjson      : per-trade
  - bar_snapshots.ndjson   : per-bar (orderbook L10 + wealth_dist)
  - agent_decisions.ndjson : per-decision (required for G3 Layer C MI computation)
  - events.ndjson          : sim-level events (AGENT_REMOVED, ORPHAN_DROPPED)

ABM_DATA_DIR enforcement (advisor v0.4 patch + design Section 8):
  - Hard-fail RuntimeError if env var unset
  - Hard-fail RuntimeError if value contains 'onedrive' (case-insensitive)
  - Rationale: BUG#58 trading bot precedent (OneDrive sync lock corrupted state.json).
    Same risk applies to high-frequency NDJSON writes from per-decision logger
    (~9M records per smoke run).

Sim-time only (advisor F1 patch):
  - All emitted records use sim-time `timestamp_ns` from Trade/OrderbookSnapshot
  - NO wall-clock timestamps emitted (would break cross-process determinism hash)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import TracebackType
from typing import Any, Optional, TextIO

from abm.types import OrderbookSnapshot, Trade


def _validate_data_dir(data_dir: Optional[str]) -> str:
    """Apply advisor v0.4 patch: hard-fail if missing or OneDrive."""
    if data_dir is None:
        data_dir = os.environ.get("ABM_DATA_DIR")
    if not data_dir:
        raise RuntimeError(
            "ABM_DATA_DIR not set. Per design v0.2 F2 / v0.4 patch, must point to "
            "NON-OneDrive path. Set env var or pass data_dir kwarg."
        )
    if "onedrive" in data_dir.lower():
        raise RuntimeError(
            f"ABM_DATA_DIR={data_dir} contains 'OneDrive' (case-insensitive). "
            "Use local-only path. BUG#58 precedent: OneDrive sync lock corrupted state.json."
        )
    return data_dir


class NDJSONLogger:
    """NDJSON event-tape logger satisfying Simulation logger contract."""

    def __init__(self, run_id: str, data_dir: Optional[str] = None) -> None:
        validated_dir = _validate_data_dir(data_dir)
        run_dir = Path(validated_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = run_dir

        self._trade_file: TextIO = open(run_dir / "trade_tape.ndjson", "w", encoding="utf-8")
        self._bar_file: TextIO = open(run_dir / "bar_snapshots.ndjson", "w", encoding="utf-8")
        self._decision_file: TextIO = open(run_dir / "agent_decisions.ndjson", "w", encoding="utf-8")
        self._events_file: TextIO = open(run_dir / "events.ndjson", "w", encoding="utf-8")
        self._closed: bool = False

    # ----- Logger contract methods -----

    def trade(self, trade: Trade) -> None:
        record = {
            "event_type": "TRADE",
            "timestamp_ns": trade.timestamp_ns,
            "sequence_no": trade.sequence_no,
            "trade_id": trade.trade_id,
            "buyer_agent_id": trade.buyer_agent_id,
            "seller_agent_id": trade.seller_agent_id,
            "buyer_order_id": trade.buyer_order_id,
            "seller_order_id": trade.seller_order_id,
            "price": trade.price,
            "size": trade.size,
            "buyer_role": trade.buyer_role.value,
            "seller_role": trade.seller_role.value,
        }
        self._write_line(self._trade_file, record)

    def bar_snapshot(
        self, snapshot: OrderbookSnapshot, wealth_dist: dict[str, float]
    ) -> None:
        record = {
            "event_type": "BAR_SNAPSHOT",
            "timestamp_ns": snapshot.timestamp_ns,
            "best_bid": snapshot.best_bid,
            "best_ask": snapshot.best_ask,
            "mid_price": snapshot.mid_price,
            "spread": snapshot.spread,
            "bid_depth_l10": [list(t) for t in snapshot.bid_depth],
            "ask_depth_l10": [list(t) for t in snapshot.ask_depth],
            "wealth_dist": wealth_dist,
        }
        self._write_line(self._bar_file, record)

    def agent_removed(self, agent_id: str, reason: str) -> None:
        self._write_line(
            self._events_file,
            {"event_type": "AGENT_REMOVED", "agent_id": agent_id, "reason": reason},
        )

    def decision(
        self,
        agent_id: str,
        family: str,
        intent_count: int,
        observed_state: dict[str, Any],
        action: dict[str, Any],
    ) -> None:
        self._write_line(
            self._decision_file,
            {
                "event_type": "DECISION",
                "agent_id": agent_id,
                "family": family,
                "intent_count": intent_count,
                "observed_state": observed_state,
                "action": action,
            },
        )

    def orphan_event_dropped(self, event_type: str, agent_id: str) -> None:
        self._write_line(
            self._events_file,
            {
                "event_type": "ORPHAN_DROPPED",
                "dropped_event_type": event_type,
                "agent_id": agent_id,
            },
        )

    # ----- Lifecycle -----

    def close(self) -> None:
        if self._closed:
            return
        for f in (self._trade_file, self._bar_file, self._decision_file, self._events_file):
            f.flush()
            f.close()
        self._closed = True

    def __enter__(self) -> "NDJSONLogger":
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # ----- Internal -----

    @staticmethod
    def _write_line(file: TextIO, record: dict[str, Any]) -> None:
        # sort_keys for deterministic byte ordering across runs (cross-process hash check)
        line = json.dumps(record, sort_keys=True, separators=(",", ":"))
        file.write(line + "\n")
