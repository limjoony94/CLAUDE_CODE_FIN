"""Trend filter regime gate tests.

Validates entry gate that skips breakout signals when rolling trend magnitude
is below threshold. Reference: regime_filter_trend PDCA (2026-04-19).

Critical-evaluation cases:
  A. Disabled by default (no effect when config flag off)
  B. Warmup bypass (bar < lookback → allow entry)
  C. Zero division / invalid past close → no skip (fail safe)
  D. Trend above threshold → entry proceeds
  E. Trend below threshold → entry skipped
"""
import pytest
from unittest.mock import MagicMock, patch


def _uptrend_candles(n=200, base=100, step=0.5):
    """Strong uptrend: +0.5 per bar = 100% over 200 bars."""
    return {
        'open':  [base + i * step for i in range(n)],
        'high':  [base + 0.5 + i * step for i in range(n)],
        'low':   [base - 0.5 + i * step for i in range(n)],
        'close': [base + 0.3 + i * step for i in range(n)],
    }


def _flat_candles(n=200, price=100.0):
    """Flat/choppy — trend ≈ 0."""
    import random
    rng = random.Random(42)
    closes = [price + rng.uniform(-0.05, 0.05) for _ in range(n)]
    return {
        'open':  closes.copy(),
        'high':  [c + 0.2 for c in closes],
        'low':   [c - 0.2 for c in closes],
        'close': closes,
    }


class TestTrendFilterDisabled:
    """A: When disabled (default), entry gate is bypassed."""

    def test_disabled_flag_allows_entry(self, mock_bot):
        """trend_filter.enabled=false → no skip regardless of trend."""
        mock_bot.config['strategy']['trend_filter'] = {
            'enabled': False,
            'lookback_bars': 192,
            'min_abs_trend_pct': 1.0,
        }
        mock_bot.positions = []
        mock_bot.bars_since_last_exit = 999
        # Flat candles — trend = 0. If filter were active, it would skip.
        candles = _flat_candles(n=200)
        # With filter disabled, entry logic runs normally (may still no-signal
        # due to body filter but we check that the trend skip log is NOT emitted).
        with patch('scripts.production.c1_breakout.bot.logger') as mock_log:
            mock_bot.process_candles(candles)
            skip_logs = [c for c in mock_log.info.call_args_list
                         if 'Trend filter skip' in str(c)]
            assert len(skip_logs) == 0

    def test_missing_config_key_allows_entry(self, mock_bot):
        """No trend_filter key at all → no skip (backward compat)."""
        mock_bot.config['strategy'].pop('trend_filter', None)
        mock_bot.positions = []
        mock_bot.bars_since_last_exit = 999
        candles = _flat_candles(n=200)
        with patch('scripts.production.c1_breakout.bot.logger') as mock_log:
            mock_bot.process_candles(candles)
            skip_logs = [c for c in mock_log.info.call_args_list
                         if 'Trend filter skip' in str(c)]
            assert len(skip_logs) == 0


class TestTrendFilterWarmup:
    """B: bar < lookback → no skip (insufficient history)."""

    def test_warmup_bar_bypass(self, mock_bot):
        mock_bot.config['strategy']['trend_filter'] = {
            'enabled': True,
            'lookback_bars': 192,
            'min_abs_trend_pct': 1.0,
        }
        mock_bot.positions = []
        mock_bot.bars_since_last_exit = 999
        # Only 50 bars < lookback 192 → no skip path; but process_candles
        # requires bar ≥ 25 warmup. With n=50, bar=48 < 192 so bypass.
        candles = _flat_candles(n=50)
        with patch('scripts.production.c1_breakout.bot.logger') as mock_log:
            mock_bot.process_candles(candles)
            skip_logs = [c for c in mock_log.info.call_args_list
                         if 'Trend filter skip' in str(c)]
            # bar=48 < lb=192 → warmup bypass (no skip message)
            assert len(skip_logs) == 0


class TestTrendFilterActive:
    """D/E: When enabled with sufficient history, gate skips low-trend."""

    def test_strong_uptrend_passes(self, mock_bot):
        """Uptrend +100% → trend_pct >> 1% → filter passes (no skip)."""
        mock_bot.config['strategy']['trend_filter'] = {
            'enabled': True,
            'lookback_bars': 50,
            'min_abs_trend_pct': 1.0,
        }
        mock_bot.positions = []
        mock_bot.bars_since_last_exit = 999
        candles = _uptrend_candles(n=200, base=100, step=0.5)
        with patch('scripts.production.c1_breakout.bot.logger') as mock_log:
            mock_bot.process_candles(candles)
            skip_logs = [c for c in mock_log.info.call_args_list
                         if 'Trend filter skip' in str(c)]
            assert len(skip_logs) == 0

    def test_flat_regime_skips_entry(self, mock_bot):
        """Flat candles → |trend| < 1% → filter skips."""
        mock_bot.config['strategy']['trend_filter'] = {
            'enabled': True,
            'lookback_bars': 50,
            'min_abs_trend_pct': 1.0,
        }
        mock_bot.positions = []
        mock_bot.bars_since_last_exit = 999
        candles = _flat_candles(n=200, price=100.0)
        with patch('scripts.production.c1_breakout.bot.logger') as mock_log:
            mock_bot.process_candles(candles)
            skip_logs = [c for c in mock_log.info.call_args_list
                         if 'Trend filter skip' in str(c)]
            assert len(skip_logs) >= 1, \
                f'expected skip log; got calls: {mock_log.info.call_args_list}'


class TestTrendFilterZeroDivision:
    """C: Defensive — past close = 0 should not crash."""

    def test_zero_past_close_no_crash(self, mock_bot):
        mock_bot.config['strategy']['trend_filter'] = {
            'enabled': True,
            'lookback_bars': 50,
            'min_abs_trend_pct': 1.0,
        }
        mock_bot.positions = []
        mock_bot.bars_since_last_exit = 999
        candles = _uptrend_candles(n=200)
        # Corrupt: zero out past close
        candles['close'][200 - 2 - 50] = 0.0  # bar_i - lookback
        # process_candles should not raise
        try:
            mock_bot.process_candles(candles)
        except ZeroDivisionError:
            pytest.fail('trend filter crashed on zero past close')
