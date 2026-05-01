"""NDJSONLogger tests including ABM_DATA_DIR enforcement (advisor v0.4 hard-fail)."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from abm.logger import NDJSONLogger
from abm.types import OrderbookSnapshot, Role, Side, Trade


# ----- ABM_DATA_DIR enforcement -----

def test_unset_abm_data_dir_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ABM_DATA_DIR", raising=False)
    with pytest.raises(RuntimeError, match="ABM_DATA_DIR not set"):
        NDJSONLogger(run_id="test_r1")


def test_onedrive_path_raises_uppercase() -> None:
    with pytest.raises(RuntimeError, match="contains 'OneDrive'"):
        NDJSONLogger(run_id="r", data_dir="C:\\Users\\J\\OneDrive\\abm_data")


def test_onedrive_path_raises_lowercase() -> None:
    """Substring match is case-insensitive."""
    with pytest.raises(RuntimeError, match="contains 'OneDrive'"):
        NDJSONLogger(run_id="r", data_dir="/home/user/onedrive/abm_data")


def test_onedrive_path_raises_mixedcase() -> None:
    with pytest.raises(RuntimeError, match="contains 'OneDrive'"):
        NDJSONLogger(run_id="r", data_dir="/path/with/OneDrive/anywhere")


def test_local_path_accepted(tmp_path: Path) -> None:
    logger = NDJSONLogger(run_id="r", data_dir=str(tmp_path))
    try:
        assert logger.run_dir.exists()
    finally:
        logger.close()


def test_env_var_used_when_no_kwarg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ABM_DATA_DIR", str(tmp_path))
    logger = NDJSONLogger(run_id="env_test")
    try:
        assert logger.run_dir == tmp_path / "env_test"
    finally:
        logger.close()


# ----- Output format / NDJSON correctness -----

def _trade(ts: int = 100, price: float = 50000.0, size: float = 0.01) -> Trade:
    return Trade(
        trade_id="t_001",
        timestamp_ns=ts,
        sequence_no=1,
        buyer_agent_id="a1",
        seller_agent_id="a2",
        buyer_order_id="a1_o_001",
        seller_order_id="a2_o_001",
        price=price,
        size=size,
        buyer_role=Role.TAKER,
        seller_role=Role.MAKER,
    )


def _snapshot() -> OrderbookSnapshot:
    return OrderbookSnapshot(
        timestamp_ns=60_000_000_000,
        best_bid=49995.0,
        best_ask=50005.0,
        bid_depth=[(49995.0, 0.5), (49990.0, 1.0)],
        ask_depth=[(50005.0, 0.5), (50010.0, 1.0)],
    )


def test_trade_emits_valid_ndjson(tmp_path: Path) -> None:
    with NDJSONLogger(run_id="t1", data_dir=str(tmp_path)) as logger:
        logger.trade(_trade())
    lines = (tmp_path / "t1" / "trade_tape.ndjson").read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["event_type"] == "TRADE"
    assert rec["timestamp_ns"] == 100
    assert rec["price"] == 50000.0
    assert rec["buyer_role"] == "taker"
    assert rec["seller_role"] == "maker"


def test_bar_snapshot_emits_valid_ndjson(tmp_path: Path) -> None:
    with NDJSONLogger(run_id="t1", data_dir=str(tmp_path)) as logger:
        logger.bar_snapshot(_snapshot(), wealth_dist={"a1": 1000.0, "a2": 2000.0})
    lines = (tmp_path / "t1" / "bar_snapshots.ndjson").read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["event_type"] == "BAR_SNAPSHOT"
    assert rec["best_bid"] == 49995.0
    assert rec["best_ask"] == 50005.0
    assert rec["mid_price"] == 50000.0
    assert rec["spread"] == 10.0
    assert rec["wealth_dist"] == {"a1": 1000.0, "a2": 2000.0}
    assert rec["bid_depth_l10"] == [[49995.0, 0.5], [49990.0, 1.0]]


def test_decision_emits_valid_ndjson(tmp_path: Path) -> None:
    with NDJSONLogger(run_id="t1", data_dir=str(tmp_path)) as logger:
        logger.decision(
            agent_id="a1",
            family="momentum",
            intent_count=1,
            observed_state={"mid": 50000.0},
            action={"side": "buy"},
        )
    lines = (tmp_path / "t1" / "agent_decisions.ndjson").read_text(encoding="utf-8").strip().split("\n")
    rec = json.loads(lines[0])
    assert rec["event_type"] == "DECISION"
    assert rec["family"] == "momentum"
    assert rec["intent_count"] == 1
    assert rec["observed_state"] == {"mid": 50000.0}


def test_agent_removed_emits_to_events(tmp_path: Path) -> None:
    with NDJSONLogger(run_id="t1", data_dir=str(tmp_path)) as logger:
        logger.agent_removed("a1", reason="bankruptcy")
    rec = json.loads((tmp_path / "t1" / "events.ndjson").read_text(encoding="utf-8").strip())
    assert rec["event_type"] == "AGENT_REMOVED"
    assert rec["agent_id"] == "a1"
    assert rec["reason"] == "bankruptcy"


def test_orphan_dropped_emits_to_events(tmp_path: Path) -> None:
    with NDJSONLogger(run_id="t1", data_dir=str(tmp_path)) as logger:
        logger.orphan_event_dropped("AGENT_DECISION", "a1")
    rec = json.loads((tmp_path / "t1" / "events.ndjson").read_text(encoding="utf-8").strip())
    assert rec["event_type"] == "ORPHAN_DROPPED"
    assert rec["dropped_event_type"] == "AGENT_DECISION"


def test_multiple_writes_one_per_line(tmp_path: Path) -> None:
    with NDJSONLogger(run_id="t1", data_dir=str(tmp_path)) as logger:
        for i in range(5):
            logger.trade(_trade(ts=100 * i))
    lines = (tmp_path / "t1" / "trade_tape.ndjson").read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 5
    for line in lines:
        json.loads(line)  # all parseable


def test_canonical_byte_order_for_determinism(tmp_path: Path) -> None:
    """Two loggers writing same records produce byte-identical files (sort_keys + sep)."""
    with NDJSONLogger(run_id="r1", data_dir=str(tmp_path)) as l1:
        l1.trade(_trade(ts=100))
        l1.trade(_trade(ts=200))
    with NDJSONLogger(run_id="r2", data_dir=str(tmp_path)) as l2:
        l2.trade(_trade(ts=100))
        l2.trade(_trade(ts=200))
    bytes1 = (tmp_path / "r1" / "trade_tape.ndjson").read_bytes()
    bytes2 = (tmp_path / "r2" / "trade_tape.ndjson").read_bytes()
    assert bytes1 == bytes2


def test_close_idempotent(tmp_path: Path) -> None:
    logger = NDJSONLogger(run_id="r", data_dir=str(tmp_path))
    logger.close()
    logger.close()  # no error


def test_files_created_on_init(tmp_path: Path) -> None:
    with NDJSONLogger(run_id="r", data_dir=str(tmp_path)):
        for fname in ("trade_tape.ndjson", "bar_snapshots.ndjson", "agent_decisions.ndjson", "events.ndjson"):
            assert (tmp_path / "r" / fname).exists()


# ----- Integration: full smoke with real logger -----

def test_smoke_with_ndjson_logger(tmp_path: Path) -> None:
    """100-bar smoke writes valid parseable NDJSON files."""
    from tests.test_simulation_smoke import _build_smoke_sim

    with NDJSONLogger(run_id="smoke_int", data_dir=str(tmp_path)) as logger:
        sim = _build_smoke_sim(seed=42, terminal_bars=50, logger=logger)
        sim.run()

    run_dir = tmp_path / "smoke_int"
    trades = run_dir / "trade_tape.ndjson"
    bars = run_dir / "bar_snapshots.ndjson"
    decisions = run_dir / "agent_decisions.ndjson"

    # All produced
    assert trades.stat().st_size > 0
    assert bars.stat().st_size > 0
    assert decisions.stat().st_size > 0

    # Parseable
    trade_records = [json.loads(l) for l in trades.read_text(encoding="utf-8").strip().split("\n")]
    bar_records = [json.loads(l) for l in bars.read_text(encoding="utf-8").strip().split("\n")]
    decision_records = [json.loads(l) for l in decisions.read_text(encoding="utf-8").strip().split("\n")]

    # Sanity counts (50 bars; calibrated agents → many trades + decisions)
    assert len(trade_records) > 50, f"Too few trades: {len(trade_records)}"
    assert len(bar_records) == 50
    assert len(decision_records) > 100, f"Too few decisions: {len(decision_records)}"

    # All trades have required fields
    for rec in trade_records:
        for field in ("timestamp_ns", "buyer_agent_id", "seller_agent_id", "price", "size", "buyer_role", "seller_role"):
            assert field in rec, f"Missing {field} in trade record"
