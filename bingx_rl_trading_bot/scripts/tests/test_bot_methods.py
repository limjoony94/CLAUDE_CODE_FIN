"""Bot unit-testable methods: fetch_candles, _calc_amount, _exchange_close, _get_balance.

These methods are decomposable from the main loop and deterministic enough
for unit tests. Covers BUG#51 (stale guard), BUG#22 (min balance), BUG#44
(balance cache).
"""
import pytest


class TestFetchCandles:
    """BUG#51: consecutive fetch failures tracked via _candle_fail_streak."""

    def test_success_resets_streak(self, mock_bot):
        """Happy path: successful fetch resets streak to 0."""
        mock_bot._candle_fail_streak = 5
        ohlcv = [[i * 60_000, 100 + i, 101 + i, 99 + i, 100.5 + i, 1.0]
                 for i in range(50)]
        mock_bot.exchange.fetch_ohlcv.return_value = ohlcv
        # First call sets _last_bar_ts
        candles = mock_bot.fetch_candles()
        assert candles is not None
        assert mock_bot._candle_fail_streak == 0

    def test_exception_increments_streak(self, mock_bot):
        mock_bot.exchange.fetch_ohlcv.side_effect = Exception('network')
        assert getattr(mock_bot, '_candle_fail_streak', 0) == 0
        mock_bot.fetch_candles()
        assert mock_bot._candle_fail_streak == 1
        mock_bot.fetch_candles()
        assert mock_bot._candle_fail_streak == 2

    def test_short_data_returns_none(self, mock_bot):
        """A. Edge: <30 bars → None (warmup guard)."""
        ohlcv = [[i * 60_000, 100, 101, 99, 100.5, 1.0] for i in range(5)]
        mock_bot.exchange.fetch_ohlcv.return_value = ohlcv
        assert mock_bot.fetch_candles() is None
        assert mock_bot._candle_fail_streak == 1

    def test_stale_bar_ts_rejected(self, mock_bot):
        """BUG#51: same last_completed_bar timestamp → skip, count as fail."""
        ohlcv = [[i * 60_000, 100 + i, 101 + i, 99 + i, 100.5 + i, 1.0]
                 for i in range(50)]
        mock_bot.exchange.fetch_ohlcv.return_value = ohlcv
        # First call succeeds
        assert mock_bot.fetch_candles() is not None
        assert mock_bot._candle_fail_streak == 0
        # Second call with same data → stale, fail streak increments
        assert mock_bot.fetch_candles() is None
        assert mock_bot._candle_fail_streak == 1

    def test_dry_run_returns_none(self, mock_bot):
        """A. Edge: no exchange → dry run skips fetch."""
        mock_bot.exchange = None
        assert mock_bot.fetch_candles() is None


class TestGetBalance:
    """_get_balance: USDT free extraction."""

    def test_normal_balance(self, mock_bot):
        mock_bot.exchange.fetch_balance.return_value = {'USDT': {'free': 1234.56}}
        assert mock_bot._get_balance() == 1234.56

    def test_missing_usdt_returns_zero(self, mock_bot):
        """A. Edge: no USDT key → 0 (conservative)."""
        mock_bot.exchange.fetch_balance.return_value = {}
        assert mock_bot._get_balance() == 0

    def test_exception_returns_zero(self, mock_bot):
        """D. Rollback: API error → 0 (caller checks min threshold)."""
        mock_bot.exchange.fetch_balance.side_effect = Exception('network')
        assert mock_bot._get_balance() == 0


class TestCalcAmount:
    """_calc_amount: position sizing with min balance + leverage scaling."""

    def test_normal_sizing(self, mock_bot):
        mock_bot.exchange.fetch_balance.return_value = {'USDT': {'free': 1000.0}}
        # trading_leverage=3, size_pct=100 (N=1)
        # qty = 1000 * 0.98 * 1.0 * 100/100 * 3 / 70000 = 0.042
        qty = mock_bot._calc_amount(70000)
        expected = round(1000 * 0.98 * 3 / 70000, 4)
        assert abs(qty - expected) < 1e-4

    def test_scale_parameter_reduces_qty(self, mock_bot):
        """BUG#38: scale<1 (retry) → proportionally smaller qty."""
        mock_bot.exchange.fetch_balance.return_value = {'USDT': {'free': 1000.0}}
        qty_full = mock_bot._calc_amount(70000, scale=1.0)
        qty_95 = mock_bot._calc_amount(70000, scale=0.95)
        qty_90 = mock_bot._calc_amount(70000, scale=0.90)
        assert qty_95 < qty_full
        assert qty_90 < qty_95
        # 0.95× should yield ~95% of full
        assert abs(qty_95 / qty_full - 0.95) < 0.01

    def test_low_balance_returns_zero(self, mock_bot):
        """BUG#22: balance < $10 → 0 (protect against dust trades)."""
        mock_bot.exchange.fetch_balance.return_value = {'USDT': {'free': 5.0}}
        assert mock_bot._calc_amount(70000) == 0


class TestExchangeClose:
    """_exchange_close: market close + cancel all orders."""

    def test_close_places_market_reduce_only(self, mock_bot):
        """Happy path: LONG close = sell market with reduceOnly."""
        mock_bot.exchange.fetch_positions.return_value = [{
            'side': 'LONG', 'contracts': 0.01, 'entryPrice': 70000.0,
        }]
        mock_bot.exchange.fetch_open_orders.return_value = [
            {'id': 'o1'}, {'id': 'o2'},
        ]
        mock_bot._exchange_close('LONG')
        # First call: market sell
        first = mock_bot.exchange.create_order.call_args_list[0]
        assert first[0][1] == 'market'
        assert first[0][2] == 'sell'
        assert first[1]['params'].get('reduceOnly') is True
        # Then cancel each open order
        assert mock_bot.exchange.cancel_order.call_count == 2

    def test_close_short_uses_buy(self, mock_bot):
        """C. Bug interaction: SHORT close = buy market."""
        mock_bot.exchange.fetch_positions.return_value = [{
            'side': 'SHORT', 'contracts': 0.01, 'entryPrice': 70000.0,
        }]
        mock_bot.exchange.fetch_open_orders.return_value = []
        mock_bot._exchange_close('SHORT')
        first = mock_bot.exchange.create_order.call_args_list[0]
        assert first[0][2] == 'buy'

    def test_close_no_live_position_skips_market(self, mock_bot):
        """A. Edge: position already gone → no market order, still cancel orders."""
        mock_bot.exchange.fetch_positions.return_value = []
        mock_bot.exchange.fetch_open_orders.return_value = [{'id': 'orphan_sl'}]
        mock_bot._exchange_close('LONG')
        # No market create_order
        mock_bot.exchange.create_order.assert_not_called()
        # Orphan order still cancelled
        mock_bot.exchange.cancel_order.assert_called()

    def test_close_cancel_failure_does_not_crash(self, mock_bot):
        """D. Rollback: individual cancel fail → continue with others."""
        mock_bot.exchange.fetch_positions.return_value = [{
            'side': 'LONG', 'contracts': 0.01, 'entryPrice': 70000.0,
        }]
        mock_bot.exchange.fetch_open_orders.return_value = [
            {'id': 'bad'}, {'id': 'good'},
        ]
        # First cancel fails, second succeeds
        mock_bot.exchange.cancel_order.side_effect = [
            Exception('cancel failed'),
            None,
        ]
        # Must not raise
        mock_bot._exchange_close('LONG')
        # Both attempted despite first failure
        assert mock_bot.exchange.cancel_order.call_count == 2
