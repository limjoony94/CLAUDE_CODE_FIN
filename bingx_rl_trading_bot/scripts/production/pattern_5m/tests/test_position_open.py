"""Tests for position_open.py — TP/SL calculation, scale-out setup,
position sizing, leverage setting, fill price detection."""

import pytest
import ccxt
from unittest.mock import MagicMock, patch

from bingx_rl_trading_bot.scripts.production.pattern_5m.position_open import (
    calculate_tp_sl,
    setup_scale_out,
    get_position_size,
    _set_leverage,
    _get_actual_fill_price,
)
from bingx_rl_trading_bot.scripts.production.pattern_5m.constants import (
    SLIPPAGE_BUFFER_PCT,
    PRICE_ROUND_DECIMALS,
    QUANTITY_ROUND_DECIMALS,
    PATTERN_OPTIMAL_TPSL,
)
from bingx_rl_trading_bot.scripts.production.pattern_5m.models import (
    APICache,
)


# ── calculate_tp_sl ─────────────────────────────────────────


class TestCalculateTPSL:
    """Test calculate_tp_sl() — 4 TP/SL source modes."""

    @pytest.fixture
    def strategy(self):
        return {'tp_pct': 1.0, 'sl_pct': 1.5}

    # ── Basic direction tests ──

    def test_long_tp_above_sl_below(self, strategy):
        """LONG: TP > entry > SL."""
        tp, sl, _, _ = calculate_tp_sl(100000.0, +1, strategy, 1.0)
        assert tp > 100000.0
        assert sl < 100000.0

    def test_short_tp_below_sl_above(self, strategy):
        """SHORT: TP < entry < SL."""
        tp, sl, _, _ = calculate_tp_sl(100000.0, -1, strategy, 1.0)
        assert tp < 100000.0
        assert sl > 100000.0

    # ── Slippage buffer ──

    def test_slippage_buffer_applied(self, strategy):
        """TP gets slippage added, SL gets slippage subtracted."""
        _, _, tp_adj, sl_adj = calculate_tp_sl(100000.0, +1, strategy, 1.0)
        assert tp_adj == pytest.approx(strategy['tp_pct'] + SLIPPAGE_BUFFER_PCT)
        assert sl_adj == pytest.approx(strategy['sl_pct'] - SLIPPAGE_BUFFER_PCT)

    # ── Volatility multiplier ──

    def test_vol_mult_scales_tp_sl(self, strategy):
        """Volatility multiplier scales TP/SL percentages."""
        _, _, tp1, sl1 = calculate_tp_sl(100000.0, +1, strategy, 1.0)
        _, _, tp2, sl2 = calculate_tp_sl(100000.0, +1, strategy, 1.5)
        assert tp2 > tp1
        assert sl2 > sl1

    # ── SL floor guard ──

    def test_sl_floor_guard(self):
        """SL adjustment should never go below 0.1%."""
        strategy = {'tp_pct': 1.0, 'sl_pct': 0.01}  # tiny SL
        _, _, _, sl_adj = calculate_tp_sl(100000.0, +1, strategy, 1.0)
        assert sl_adj >= 0.1

    # ── Mode 1: Dynamic per-pattern ──

    def test_dynamic_per_pattern_mode(self, strategy):
        """Dynamic per-pattern config overrides all other sources."""
        config = {
            '_dynamic_tpsl_per_pattern': True,
            '_dynamic_patterns_tpsl': {'TEST-PAT': [2.5, 4.0]},
        }
        tp, sl, tp_adj, sl_adj = calculate_tp_sl(
            100000.0, +1, strategy, 1.0, pattern='TEST-PAT', config=config
        )
        assert tp_adj == pytest.approx(2.5 + SLIPPAGE_BUFFER_PCT)
        assert sl_adj == pytest.approx(4.0 - SLIPPAGE_BUFFER_PCT)
        assert tp > 100000.0
        assert sl < 100000.0

    def test_dynamic_per_pattern_missing_falls_to_default(self, strategy):
        """Pattern not in dynamic dict → uses strategy defaults."""
        config = {
            '_dynamic_tpsl_per_pattern': True,
            '_dynamic_patterns_tpsl': {'OTHER-PAT': [2.5, 4.0]},
        }
        _, _, tp_adj, _ = calculate_tp_sl(
            100000.0, +1, strategy, 1.0, pattern='UNKNOWN', config=config
        )
        assert tp_adj == pytest.approx(strategy['tp_pct'] + SLIPPAGE_BUFFER_PCT)

    # ── Mode 2: Dynamic universal ──

    def test_dynamic_universal_mode(self, strategy):
        """Dynamic universal config overrides pattern/regime/defaults."""
        config = {
            '_dynamic_tpsl_universal': True,
            '_dynamic_tp': 2.0,
            '_dynamic_sl': 3.0,
        }
        _, _, tp_adj, sl_adj = calculate_tp_sl(
            100000.0, +1, strategy, 1.0, config=config
        )
        assert tp_adj == pytest.approx(2.0 + SLIPPAGE_BUFFER_PCT)
        assert sl_adj == pytest.approx(3.0 - SLIPPAGE_BUFFER_PCT)

    # ── Mode 3: Regime-specific ──

    def test_regime_tp_sl_overrides_pattern(self, strategy):
        """Regime TP/SL overrides pattern-specific and defaults."""
        regime_tp_sl = (1.8, 2.5)
        _, _, tp_adj, sl_adj = calculate_tp_sl(
            100000.0, +1, strategy, 1.0, pattern=None, regime_tp_sl=regime_tp_sl
        )
        assert tp_adj == pytest.approx(1.8 + SLIPPAGE_BUFFER_PCT)
        assert sl_adj == pytest.approx(2.5 - SLIPPAGE_BUFFER_PCT)

    # ── Mode 4: Static pattern-specific ──

    def test_static_pattern_specific(self, strategy):
        """Pattern in PATTERN_OPTIMAL_TPSL → uses pattern-specific values."""
        if not PATTERN_OPTIMAL_TPSL:
            pytest.skip("No patterns in PATTERN_OPTIMAL_TPSL")
        pattern = next(iter(PATTERN_OPTIMAL_TPSL))
        expected_tp, expected_sl = PATTERN_OPTIMAL_TPSL[pattern]
        _, _, tp_adj, sl_adj = calculate_tp_sl(
            100000.0, +1, strategy, 1.0, pattern=pattern
        )
        assert tp_adj == pytest.approx(expected_tp + SLIPPAGE_BUFFER_PCT)
        assert sl_adj == pytest.approx(expected_sl - SLIPPAGE_BUFFER_PCT)

    def test_unknown_pattern_uses_strategy_defaults(self, strategy):
        """Unknown pattern → uses strategy tp_pct/sl_pct."""
        _, _, tp_adj, sl_adj = calculate_tp_sl(
            100000.0, +1, strategy, 1.0, pattern='NONEXISTENT-PAT'
        )
        assert tp_adj == pytest.approx(strategy['tp_pct'] + SLIPPAGE_BUFFER_PCT)
        assert sl_adj == pytest.approx(strategy['sl_pct'] - SLIPPAGE_BUFFER_PCT)

    # ── Priority: dynamic PP > dynamic universal > regime > pattern > default ──

    def test_per_pattern_beats_universal(self, strategy):
        """Per-pattern mode takes priority over universal."""
        config = {
            '_dynamic_tpsl_per_pattern': True,
            '_dynamic_patterns_tpsl': {'PAT': [3.0, 3.5]},
            '_dynamic_tpsl_universal': True,
            '_dynamic_tp': 2.0,
            '_dynamic_sl': 3.0,
        }
        _, _, tp_adj, _ = calculate_tp_sl(
            100000.0, +1, strategy, 1.0, pattern='PAT', config=config
        )
        assert tp_adj == pytest.approx(3.0 + SLIPPAGE_BUFFER_PCT)

    # ── Rounding ──

    def test_prices_rounded(self, strategy):
        """TP/SL prices should be rounded to PRICE_ROUND_DECIMALS."""
        tp, sl, _, _ = calculate_tp_sl(99999.123456, +1, strategy, 1.0)
        assert tp == round(tp, PRICE_ROUND_DECIMALS)
        assert sl == round(sl, PRICE_ROUND_DECIMALS)


# ── setup_scale_out ──────────────────────────────────────────


class TestSetupScaleOut:
    """Test setup_scale_out() stage creation."""

    def test_disabled_returns_empty(self):
        """Scale-out not enabled → empty list."""
        strategy = {'scale_out': {'enabled': False}}
        result = setup_scale_out(strategy, 100000.0, 0.01, +1, 2.0)
        assert result == []

    def test_no_scale_out_config_returns_empty(self):
        """No scale_out in strategy → empty list."""
        result = setup_scale_out({}, 100000.0, 0.01, +1, 2.0)
        assert result == []

    def test_single_stage(self):
        """Single stage → one entry with correct fields."""
        strategy = {
            'scale_out': {
                'enabled': True,
                'stages': [(1.0, 0.5)],  # 100% at 50% of TP
            }
        }
        stages = setup_scale_out(strategy, 100000.0, 0.01, +1, 2.0)
        assert len(stages) == 1
        assert stages[0]['stage'] == 1
        assert stages[0]['pct'] == 1.0
        assert stages[0]['tp_mult'] == 0.5
        assert stages[0]['filled'] is False
        assert stages[0]['order_id'] is None

    def test_two_stages_quantity_sum(self):
        """Two stages → quantities sum to total."""
        strategy = {
            'scale_out': {
                'enabled': True,
                'stages': [(0.5, 0.5), (0.5, 1.0)],
            }
        }
        stages = setup_scale_out(strategy, 100000.0, 0.01, +1, 2.0)
        assert len(stages) == 2
        total = sum(s['quantity'] for s in stages)
        assert total == pytest.approx(0.01, abs=1e-4)

    def test_long_tp_prices_ascending(self):
        """LONG stages: higher tp_mult → higher TP price."""
        strategy = {
            'scale_out': {
                'enabled': True,
                'stages': [(0.5, 0.5), (0.5, 1.0)],
            }
        }
        stages = setup_scale_out(strategy, 100000.0, 0.01, +1, 2.0)
        assert stages[0]['tp_price'] < stages[1]['tp_price']
        assert stages[0]['tp_price'] > 100000.0

    def test_short_tp_prices_descending(self):
        """SHORT stages: higher tp_mult → lower TP price."""
        strategy = {
            'scale_out': {
                'enabled': True,
                'stages': [(0.5, 0.5), (0.5, 1.0)],
            }
        }
        stages = setup_scale_out(strategy, 100000.0, 0.01, -1, 2.0)
        assert stages[0]['tp_price'] > stages[1]['tp_price']
        assert stages[0]['tp_price'] < 100000.0


# ── get_position_size ────────────────────────────────────────


@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.exchange._interruptible_api_sleep',
       new=MagicMock())
class TestGetPositionSize:
    """Test get_position_size() balance-based sizing."""

    def _make_config(self):
        return {
            'symbol': 'BTC/USDT:USDT',
            'leverage': 3,
            'position_size_pct': 100,
            'risk': {'max_position_size_usd': 10000},
        }

    def test_normal_calculation(self):
        """Normal balance → correct quantity."""
        exchange = MagicMock()
        exchange.fetch_balance.return_value = {'USDT': {'free': 1000, 'total': 1000}}
        exchange.fetch_ticker.return_value = {'last': 100000.0}
        cache = APICache()

        qty, balance, price = get_position_size(exchange, self._make_config(), cache)

        assert balance == 1000.0
        assert price == 100000.0
        # quantity = (min(1000*1.0, 10000) * 3) / 100000 = 3000/100000 = 0.03
        assert qty == pytest.approx(0.03, abs=0.001)

    def test_max_size_cap(self):
        """Large balance capped by max_position_size_usd."""
        exchange = MagicMock()
        exchange.fetch_balance.return_value = {'USDT': {'free': 50000, 'total': 50000}}
        exchange.fetch_ticker.return_value = {'last': 100000.0}
        cache = APICache()
        config = self._make_config()

        qty, _, _ = get_position_size(exchange, config, cache)

        # capped: min(50000*1.0, 10000) = 10000, qty = (10000*3)/100000 = 0.3
        assert qty == pytest.approx(0.3, abs=0.001)

    def test_network_error_returns_none(self):
        """Network error → returns (None, None, None)."""
        exchange = MagicMock()
        exchange.fetch_balance.side_effect = ccxt.NetworkError('timeout')
        cache = APICache()

        qty, bal, price = get_position_size(exchange, self._make_config(), cache)

        assert qty is None
        assert bal is None
        assert price is None

    def test_exchange_error_returns_none(self):
        """Exchange error → returns (None, None, None)."""
        exchange = MagicMock()
        exchange.fetch_balance.side_effect = ccxt.ExchangeError('fail')
        cache = APICache()

        qty, bal, price = get_position_size(exchange, self._make_config(), cache)

        assert qty is None

    def test_zero_balance(self):
        """Zero balance → quantity is 0."""
        exchange = MagicMock()
        exchange.fetch_balance.return_value = {'USDT': {'free': 0, 'total': 0}}
        exchange.fetch_ticker.return_value = {'last': 100000.0}
        cache = APICache()

        qty, _, _ = get_position_size(exchange, self._make_config(), cache)

        assert qty == 0


# ── _set_leverage ────────────────────────────────────────────


class TestSetLeverage:
    """Test _set_leverage() exchange setup."""

    def test_success(self):
        """Normal leverage set → no error."""
        exchange = MagicMock()
        _set_leverage(exchange, 'BTC/USDT:USDT', 3)
        exchange.set_leverage.assert_called_once_with(
            3, 'BTC/USDT:USDT', params={'side': 'BOTH'}
        )

    def test_already_set_no_error(self):
        """'No need to change' → silently accepted."""
        exchange = MagicMock()
        exchange.set_leverage.side_effect = ccxt.ExchangeError('No need to change')
        _set_leverage(exchange, 'BTC/USDT:USDT', 3)  # should not raise

    def test_same_keyword_no_error(self):
        """'same' in error → silently accepted."""
        exchange = MagicMock()
        exchange.set_leverage.side_effect = ccxt.ExchangeError('leverage is the same')
        _set_leverage(exchange, 'BTC/USDT:USDT', 3)  # should not raise

    def test_other_exchange_error_no_raise(self):
        """Other exchange error → logged but doesn't raise."""
        exchange = MagicMock()
        exchange.set_leverage.side_effect = ccxt.ExchangeError('permission denied')
        _set_leverage(exchange, 'BTC/USDT:USDT', 3)  # should not raise

    def test_network_error_no_raise(self):
        """Network error → logged but doesn't raise."""
        exchange = MagicMock()
        exchange.set_leverage.side_effect = ccxt.NetworkError('timeout')
        _set_leverage(exchange, 'BTC/USDT:USDT', 3)  # should not raise


# ── _get_actual_fill_price ───────────────────────────────────


@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_open.time.sleep')
@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.exchange._interruptible_api_sleep',
       new=MagicMock())
class TestGetActualFillPrice:
    """Test _get_actual_fill_price() fill detection."""

    def test_from_order_result(self, mock_sleep):
        """Order has 'average' → uses it directly."""
        exchange = MagicMock()
        order = {'average': 50100.0, 'filled': 0.01}
        price, qty = _get_actual_fill_price(
            exchange, order, 'LONG', 'BTC/USDT:USDT', 50000.0, 0.01,
            APICache(), None, None
        )
        assert price == 50100.0
        assert qty == 0.01

    def test_fallback_to_positions(self, mock_sleep):
        """Order average=0 → falls back to exchange positions."""
        exchange = MagicMock()
        exchange.fetch_positions.return_value = [
            {'side': 'long', 'contracts': 0.01, 'entryPrice': 50200.0},
        ]
        order = {'average': 0, 'filled': 0.01}
        cache = APICache()
        price, qty = _get_actual_fill_price(
            exchange, order, 'LONG', 'BTC/USDT:USDT', 50000.0, 0.01,
            cache, None, None
        )
        assert price == 50200.0

    def test_network_error_uses_estimated(self, mock_sleep):
        """Position fetch fails → uses estimated price."""
        exchange = MagicMock()
        exchange.fetch_positions.side_effect = ccxt.NetworkError('fail')
        order = {'average': 0, 'filled': 0.01}
        cache = APICache()
        price, qty = _get_actual_fill_price(
            exchange, order, 'LONG', 'BTC/USDT:USDT', 50000.0, 0.01,
            cache, None, None
        )
        assert price == 50000.0  # estimated fallback

    def test_order_with_price_key(self, mock_sleep):
        """Order has 'price' but no 'average' → uses 'price'."""
        exchange = MagicMock()
        order = {'price': 50300.0, 'filled': 0.01}
        price, qty = _get_actual_fill_price(
            exchange, order, 'LONG', 'BTC/USDT:USDT', 50000.0, 0.01,
            APICache(), None, None
        )
        assert price == 50300.0
