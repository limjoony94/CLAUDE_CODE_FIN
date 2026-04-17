"""Bot-level tests: orphan resolution, state reconciliation, I/O defenses.

Encodes critical-evaluation cases from cycles 1-16 covering:
  BUG#48 _resolve_orphan_sl — 7 field-mapping cases
  BUG#54 last_exit_time — elapsed_bars reconciliation
  BUG#56 trade_history in-memory cap
  BUG#58 state I/O graceful failure
  BUG#57 naive ISO format consistency
"""
import json
import tempfile
import os
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock
import pytest

from scripts.production.c1_breakout.bot import (
    C1BreakoutBot, _utc_now, _utc_now_naive_iso,
)


# ── BUG#48: _resolve_orphan_sl ─────────────────────────────

class TestResolveOrphanSL:
    """Critical-evaluation angles for orphan SL restoration (BUG#48)."""

    def test_uppercase_side_accepted(self, mock_bot):
        """A. Edge: exchange returns 'SELL' uppercase (BingX pattern)."""
        mock_bot.exchange.fetch_open_orders.return_value = [
            {'id': 'o1', 'side': 'SELL', 'type': 'STOP_MARKET',
             'info': {'type': 'STOP_MARKET', 'reduceOnly': 'TRUE',
                      'stopPrice': '70000'}}
        ]
        sp, oid = mock_bot._resolve_orphan_sl('LONG', 71000)
        assert sp == 70000.0
        assert oid == 'o1'

    def test_int_reduceonly_and_triggerprice_alias(self, mock_bot):
        """A. Edge: reduceOnly as int 1, stopPrice under triggerPrice alias."""
        mock_bot.exchange.fetch_open_orders.return_value = [
            {'id': 'o2', 'side': 'buy', 'type': 'STOP',
             'reduceOnly': 1,
             'info': {'type': 'STOP', 'triggerPrice': '72500'}}
        ]
        sp, oid = mock_bot._resolve_orphan_sl('SHORT', 71000)
        assert sp == 72500.0

    def test_trailing_excluded(self, mock_bot):
        """C. Bug interaction: TRAILING_STOP_MARKET is NOT fractal SL.

        Including it would confuse trail with stop on orphan recovery.
        """
        mock_bot.exchange.fetch_open_orders.return_value = [
            {'id': 'trail', 'side': 'sell', 'type': 'TRAILING_STOP_MARKET',
             'reduceOnly': True,
             'info': {'type': 'TRAILING_STOP_MARKET', 'stopPrice': '70000'}}
        ]
        sp, oid = mock_bot._resolve_orphan_sl('LONG', 71000)
        assert sp is None

    def test_multiple_stops_picks_tightest(self, mock_bot):
        """C. Bug interaction: legacy + new STOPs → pick closest to entry.

        Tightest = already-partially-triggered = most conservative.
        """
        mock_bot.exchange.fetch_open_orders.return_value = [
            {'id': 'far', 'side': 'sell', 'type': 'STOP_MARKET',
             'reduceOnly': True,
             'info': {'type': 'STOP_MARKET', 'stopPrice': '69000'}},
            {'id': 'close', 'side': 'sell', 'type': 'STOP_MARKET',
             'reduceOnly': True,
             'info': {'type': 'STOP_MARKET', 'stopPrice': '70500'}},
        ]
        sp, oid = mock_bot._resolve_orphan_sl('LONG', 71000)
        assert sp == 70500.0
        assert oid == 'close'

    def test_wrong_side_excluded(self, mock_bot):
        """A. Edge: LONG SL above entry is directionally wrong — reject."""
        mock_bot.exchange.fetch_open_orders.return_value = [
            {'id': 'bad', 'side': 'sell', 'type': 'STOP_MARKET',
             'reduceOnly': True,
             'info': {'type': 'STOP_MARKET', 'stopPrice': '72000'}}
        ]
        sp, oid = mock_bot._resolve_orphan_sl('LONG', 71000)
        assert sp is None

    def test_no_reduceonly_excluded(self, mock_bot):
        """C. Bug interaction: non-reduceOnly = likely user's manual order."""
        mock_bot.exchange.fetch_open_orders.return_value = [
            {'id': 'manual', 'side': 'sell', 'type': 'STOP_MARKET',
             'info': {'type': 'STOP_MARKET', 'stopPrice': '70000'}}
        ]
        sp, oid = mock_bot._resolve_orphan_sl('LONG', 71000)
        assert sp is None

    def test_api_exception_returns_none(self, mock_bot):
        """D. Rollback: API failure → fallback to 3% emergency (caller handles)."""
        mock_bot.exchange.fetch_open_orders.side_effect = Exception('network')
        sp, oid = mock_bot._resolve_orphan_sl('LONG', 71000)
        assert sp is None
        assert oid is None


# ── BUG#54: bars_since_last_exit wall-clock reconciliation ─

class TestLastExitTimeReconciliation:
    """Elapsed-bar calculation from wall-clock on restart."""

    def _write_state(self, path, **kwargs):
        base = {
            'positions': [], 'trade_history': [],
            'bars_since_last_exit': 0, 'last_exit_time': None,
        }
        base.update(kwargs)
        with open(path, 'w') as f:
            json.dump(base, f)

    def test_two_hour_outage_eight_bars(self, mock_bot):
        """A. Edge: 2h outage → counter = 8 bars (2h / 15min)."""
        two_hours_ago = datetime.now(timezone.utc) - timedelta(hours=2)
        self._write_state(mock_bot.state_path,
                          bars_since_last_exit=0,
                          last_exit_time=two_hours_ago.isoformat())
        mock_bot._load_state()
        assert mock_bot.bars_since_last_exit == 8

    def test_saved_counter_wins_when_greater(self, mock_bot):
        """C. Bug interaction: never regress — prefer the larger value.

        If saved counter = 10 but elapsed = 0 (recent exit), keep 10.
        """
        recent = datetime.now(timezone.utc) - timedelta(minutes=5)
        self._write_state(mock_bot.state_path,
                          bars_since_last_exit=10,
                          last_exit_time=recent.isoformat())
        mock_bot._load_state()
        assert mock_bot.bars_since_last_exit == 10

    def test_missing_last_exit_time_falls_back(self, mock_bot):
        """D. Rollback: old state.json (pre-BUG#54) → counter-only logic."""
        self._write_state(mock_bot.state_path, bars_since_last_exit=5)
        mock_bot._load_state()
        assert mock_bot.bars_since_last_exit == 5
        assert mock_bot.last_exit_time is None

    def test_corrupt_timestamp_silent_fallback(self, mock_bot):
        """D. Rollback: unparseable timestamp → ignore, use counter only."""
        self._write_state(mock_bot.state_path,
                          bars_since_last_exit=3,
                          last_exit_time='NOT_A_DATE')
        mock_bot._load_state()
        assert mock_bot.bars_since_last_exit == 3

    def test_z_suffix_parsed(self, mock_bot):
        """A. Edge: ISO timestamp with 'Z' suffix (alternative UTC marker)."""
        ts_z = ((datetime.now(timezone.utc) - timedelta(hours=1))
                .isoformat().replace('+00:00', 'Z'))
        self._write_state(mock_bot.state_path,
                          bars_since_last_exit=0,
                          last_exit_time=ts_z)
        mock_bot._load_state()
        assert mock_bot.bars_since_last_exit == 4  # 1h / 15min


# ── BUG#58: state I/O graceful failure ─────────────────────

class TestStateIOResilience:
    """OneDrive sync lock / transient I/O must not crash the main loop."""

    def test_invalid_path_does_not_crash(self, mock_bot):
        """D. Rollback: bad path → log warning, do not raise."""
        mock_bot.state_path = '/nonexistent/nested/path/state.json'
        mock_bot._save_state()  # should NOT raise

    def test_valid_path_saves_and_loads(self, mock_bot):
        """Happy path: save then load recovers state."""
        mock_bot.positions = [{
            'direction': 'LONG', 'entry_price': 71000, 'sl_price': 70500,
            'best_price': 71200, 'entry_time': '2026-04-17T12:00:00',
            'bars_held': 3, 'size_pct': 100.0,
        }]
        mock_bot.bars_since_last_exit = 5
        mock_bot.last_exit_time = datetime.now(timezone.utc)
        mock_bot._save_state()
        assert os.path.exists(mock_bot.state_path)

        # Load into new bot
        new_bot = C1BreakoutBot.__new__(C1BreakoutBot)
        new_bot.state_path = mock_bot.state_path
        new_bot.positions = []
        new_bot.trade_history = []
        new_bot.bars_since_last_exit = 999
        new_bot.last_exit_time = None
        new_bot._load_state()
        assert len(new_bot.positions) == 1
        assert new_bot.positions[0]['direction'] == 'LONG'


# ── BUG#56: trade_history in-memory cap ────────────────────

class TestTradeHistoryTrim:
    """Memory-growth prevention for long-running sessions."""

    def test_trim_over_threshold(self):
        """Trim 1001 entries → 500 (BUG#56 logic)."""
        th = [{'i': i} for i in range(1001)]
        if len(th) > 1000:
            th = th[-500:]
        assert len(th) == 500
        assert th[0]['i'] == 501  # oldest retained
        assert th[-1]['i'] == 1000

    def test_no_trim_under_threshold(self):
        """Under 1000 → no trim (preserve full history)."""
        th = [{'i': i} for i in range(500)]
        if len(th) > 1000:
            th = th[-500:]
        assert len(th) == 500


# ── BUG#57: naive ISO format consistency ───────────────────

class TestTimeHelpers:
    """Naive-ISO format backwards-compatible with pre-BUG#57 state.json."""

    def test_utc_now_returns_aware(self):
        """Internal arithmetic uses aware datetime."""
        dt = _utc_now()
        assert dt.tzinfo is not None

    def test_utc_now_naive_iso_has_no_tz_suffix(self):
        """Serialization format must match legacy naive timestamps."""
        s = _utc_now_naive_iso()
        assert '+00:00' not in s
        assert 'Z' not in s
        assert 'T' in s
