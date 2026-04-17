"""process_candles integration tests.

End-to-end flow: exit check → trail update → entry signal. Verifies that
bars_held/best_price/bars_since_last_exit move correctly through a cycle.
"""
import pytest
from unittest.mock import MagicMock, patch


def _uptrend_candles(n=50, base=100):
    """Monotonic uptrend — guarantees ATR/Channel valid."""
    return {
        'open': [base + i for i in range(n)],
        'high': [base + 0.5 + i for i in range(n)],
        'low': [base - 0.5 + i for i in range(n)],
        'close': [base + 0.3 + i for i in range(n)],
    }


class TestBarsHeldIncrement:
    """Invariant: every cycle with open position increments bars_held by 1."""

    def test_bars_held_increments(self, mock_bot):
        mock_bot.positions = [{
            'direction': 'LONG', 'entry_price': 130, 'sl_price': 125,
            'best_price': 130, 'entry_time': '2026-04-17T12:00:00',
            'bars_held': 7, 'size_pct': 100.0,
        }]
        mock_bot.bars_since_last_exit = 10
        candles = _uptrend_candles(n=50, base=100)  # close[bar] = ~148
        # No SL hit (low high all > 125), so position stays
        mock_bot.process_candles(candles)
        assert mock_bot.positions[0]['bars_held'] == 8


class TestBestPriceUpdate:
    """LONG: best_price = max; SHORT: best_price = min."""

    def test_long_best_updates_on_new_high(self, mock_bot):
        mock_bot.positions = [{
            'direction': 'LONG', 'entry_price': 100, 'sl_price': 95,
            'best_price': 110, 'entry_time': '2026-04-17T12:00:00',
            'bars_held': 5, 'size_pct': 100.0,
        }]
        mock_bot.bars_since_last_exit = 10
        candles = _uptrend_candles(n=50, base=100)
        # bar = n-2 = 48, high[48] = 148.5 > 110
        mock_bot.process_candles(candles)
        assert mock_bot.positions[0]['best_price'] > 110

    def test_short_best_updates_on_new_low(self, mock_bot):
        mock_bot.positions = [{
            'direction': 'SHORT', 'entry_price': 200, 'sl_price': 205,
            'best_price': 195, 'entry_time': '2026-04-17T12:00:00',
            'bars_held': 5, 'size_pct': 100.0,
        }]
        mock_bot.bars_since_last_exit = 10
        # Downtrend: low goes below 195
        n = 50
        candles = {
            'open': [200 - i * 0.5 for i in range(n)],
            'high': [200.5 - i * 0.5 for i in range(n)],
            'low': [199 - i * 0.5 for i in range(n)],
            'close': [199.5 - i * 0.5 for i in range(n)],
        }
        mock_bot.process_candles(candles)
        # best_price should drop below 195
        assert mock_bot.positions[0]['best_price'] < 195


class TestBarsSinceLastExit:
    """BUG#16: counter increments every cycle, gates entries."""

    def test_counter_increments_each_cycle(self, mock_bot):
        mock_bot.bars_since_last_exit = 3
        candles = _uptrend_candles(n=50)
        mock_bot.process_candles(candles)
        # +1 at cycle start, no exit (no position) → 4
        assert mock_bot.bars_since_last_exit == 4

    def test_no_entry_during_cooldown(self, mock_bot):
        """BUG#16: bars_since_last_exit < min_bars_between (2) blocks entry."""
        mock_bot.bars_since_last_exit = 0  # just exited
        mock_bot.positions = []
        # Build candles that would normally trigger LONG breakout
        n = 40
        candles = {
            'open': [100] * n + [100],
            'high': [100.5] * n + [102],
            'low': [99.5] * n + [99.5],
            'close': [100] * n + [101.5],  # bar close > channel_high
        }
        # Fix length mismatch
        for k in candles:
            candles[k] = candles[k][:n]
        # Override last bar to break out
        candles['open'][-2] = 100
        candles['close'][-2] = 101.5
        candles['high'][-2] = 102
        candles['low'][-2] = 99.5
        mock_bot.process_candles(candles)
        # +1 → 1, still < min_bars=2 → entry blocked
        assert mock_bot.bars_since_last_exit == 1
        # Exchange mock wasn't called to create MARKET order
        # (can't assert directly without deeper mocking, but position should stay empty)
        assert len(mock_bot.positions) == 0


class TestWarmupGuard:
    """A. Edge: bar < 25 → skip processing entirely."""

    def test_insufficient_bars_short_circuit(self, mock_bot):
        """Only 20 bars → bar=18 < 25 → early return, no state change."""
        mock_bot.bars_since_last_exit = 5
        candles = _uptrend_candles(n=20)
        mock_bot.process_candles(candles)
        # Counter NOT incremented (early return happens before +=1)
        # Actually: re-reading bot.py, +=1 happens AFTER `if bar < 25: return`
        # So counter unchanged
        assert mock_bot.bars_since_last_exit == 5
