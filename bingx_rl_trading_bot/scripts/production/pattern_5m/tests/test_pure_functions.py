"""Tests for pure functions — calculate_pnl, extract_pattern_name, setup_scale_out,
get_current_pattern, get_volatility_multiplier, _infer_exit_reason, calculate_indicators."""

import pytest
import pandas as pd
import numpy as np

from bingx_rl_trading_bot.scripts.production.pattern_5m.position_close import calculate_pnl
from bingx_rl_trading_bot.scripts.production.pattern_5m.position_open import setup_scale_out
from bingx_rl_trading_bot.scripts.production.pattern_5m.utils import extract_pattern_name
from bingx_rl_trading_bot.scripts.production.pattern_5m.indicators import (
    calculate_indicators,
    get_current_pattern,
    get_volatility_multiplier,
)
from bingx_rl_trading_bot.scripts.production.pattern_5m.position_monitor import (
    _infer_exit_reason,
)
from bingx_rl_trading_bot.scripts.production.pattern_5m.constants import (
    FEE_PCT,
    PRICE_TOLERANCE_PCT,
    AVG_BODY_WINDOW,
)


# ── calculate_pnl ────────────────────────────────────────────


class TestCalculatePnl:
    """Test calculate_pnl() — leveraged and price-basis PnL."""

    def test_long_profit(self):
        """LONG: exit above entry → positive PnL."""
        pnl, price_pnl = calculate_pnl(50000.0, 51000.0, direction=1, leverage=3)
        assert pnl > 0
        assert price_pnl > 0

    def test_long_loss(self):
        """LONG: exit below entry → negative PnL."""
        pnl, price_pnl = calculate_pnl(50000.0, 49000.0, direction=1, leverage=3)
        assert pnl < 0
        assert price_pnl < 0

    def test_short_profit(self):
        """SHORT: exit below entry → positive PnL."""
        pnl, price_pnl = calculate_pnl(50000.0, 49000.0, direction=-1, leverage=3)
        assert pnl > 0
        assert price_pnl > 0

    def test_short_loss(self):
        """SHORT: exit above entry → negative PnL."""
        pnl, price_pnl = calculate_pnl(50000.0, 51000.0, direction=-1, leverage=3)
        assert pnl < 0
        assert price_pnl < 0

    def test_leverage_multiplier(self):
        """Leveraged PnL should be leverage × price PnL (approximately, fees differ)."""
        pnl_3x, price_pnl = calculate_pnl(50000.0, 51000.0, direction=1, leverage=3)
        pnl_1x, _ = calculate_pnl(50000.0, 51000.0, direction=1, leverage=1)
        # Price move is 2%, fee is 0.05% × 2 = 0.10%
        # pnl_3x ≈ (2% - 0.10%) × 3 = 5.7%, pnl_1x ≈ 2% - 0.10% = 1.9%
        # Ratio should be approximately 3x
        assert abs(pnl_3x / pnl_1x - 3.0) < 0.1

    def test_fee_deduction(self):
        """PnL should be less than raw price movement (fees deducted)."""
        pnl, _ = calculate_pnl(50000.0, 51000.0, direction=1, leverage=1)
        raw_pnl = (51000.0 / 50000.0 - 1) * 100  # 2.0%
        assert pnl < raw_pnl  # Fees reduce PnL

    def test_exact_values(self):
        """Verify exact PnL calculation with known values."""
        # LONG: entry=50000, exit=51000, 3x leverage
        # Price move = 1 × (51000/50000 - 1) × 100 = 2.0%
        # pnl = 2.0 × 3 - 2 × 0.05 × 3 = 6.0 - 0.30 = 5.70%
        # price_pnl = 2.0 - 2 × 0.05 = 2.0 - 0.10 = 1.90%
        pnl, price_pnl = calculate_pnl(50000.0, 51000.0, direction=1, leverage=3)
        assert pnl == pytest.approx(5.70, abs=0.01)
        assert price_pnl == pytest.approx(1.90, abs=0.01)

    def test_breakeven_exit(self):
        """Exit at entry → PnL should be slightly negative (fees only)."""
        pnl, _ = calculate_pnl(50000.0, 50000.0, direction=1, leverage=3)
        expected = -2 * FEE_PCT * 3  # Only fees
        assert pnl == pytest.approx(expected, abs=0.001)

    def test_symmetric_long_short(self):
        """Same move distance: LONG profit ≈ SHORT profit (symmetric)."""
        pnl_long, _ = calculate_pnl(50000.0, 51000.0, direction=1, leverage=3)
        pnl_short, _ = calculate_pnl(50000.0, 49000.0, direction=-1, leverage=3)
        assert pnl_long == pytest.approx(pnl_short, abs=0.1)


# ── extract_pattern_name ─────────────────────────────────────


class TestExtractPatternName:
    """Test extract_pattern_name() — regex pattern extraction."""

    def test_standard_format(self):
        """Standard reason string → extracts pattern."""
        assert extract_pattern_name('Pattern: BD-BD-U (LONG)') == 'BD-BD-U'

    def test_long_suffix(self):
        """Pattern with LONG direction suffix."""
        assert extract_pattern_name('Pattern: U-MU-H (LONG)') == 'U-MU-H'

    def test_short_suffix(self):
        """Pattern with SHORT direction suffix."""
        assert extract_pattern_name('Pattern: DN-BU-BU (SHORT)') == 'DN-BU-BU'

    def test_recovery_reason(self):
        """Recovered position reason."""
        assert extract_pattern_name('Recovered from exchange (BD-BD-BU)') == ''

    def test_pattern_colon_format(self):
        """Pattern: NAME format without parentheses."""
        assert extract_pattern_name('Pattern: MU-BD-ST') == 'MU-BD-ST'

    def test_no_pattern(self):
        """No pattern in reason → empty string."""
        assert extract_pattern_name('Manual trade') == ''

    def test_empty_string(self):
        """Empty string → empty string."""
        assert extract_pattern_name('') == ''

    def test_none_safe(self):
        """Should handle edge cases gracefully."""
        # extract_pattern_name expects str, but check empty
        assert extract_pattern_name('Pattern: ') == ''


# ── setup_scale_out ──────────────────────────────────────────


class TestSetupScaleOut:
    """Test setup_scale_out() — scale-out stage calculation."""

    @pytest.fixture
    def so_strategy(self):
        """Strategy with scale-out enabled."""
        return {
            'scale_out': {
                'enabled': True,
                'stages': [(0.5, 0.5), (0.5, 1.0)],  # 50% at 50% TP, 50% at 100% TP
            }
        }

    @pytest.fixture
    def no_so_strategy(self):
        """Strategy without scale-out."""
        return {'scale_out': {'enabled': False}}

    def test_disabled_returns_empty(self, no_so_strategy):
        """Scale-out disabled → empty list."""
        stages = setup_scale_out(no_so_strategy, 50000.0, 0.01, 1, 2.0)
        assert stages == []

    def test_missing_scale_out_key(self):
        """No scale_out key → empty list."""
        stages = setup_scale_out({}, 50000.0, 0.01, 1, 2.0)
        assert stages == []

    def test_enabled_creates_stages(self, so_strategy):
        """Enabled scale-out → correct number of stages."""
        stages = setup_scale_out(so_strategy, 50000.0, 0.01, 1, 2.0)
        assert len(stages) == 2

    def test_stage_quantities_sum(self, so_strategy):
        """Stage quantities should sum to total position quantity."""
        quantity = 0.01
        stages = setup_scale_out(so_strategy, 50000.0, quantity, 1, 2.0)
        total = sum(s['quantity'] for s in stages)
        assert total == pytest.approx(quantity, abs=0.0001)

    def test_stage_tp_prices_ascending_long(self, so_strategy):
        """LONG: stage TP prices should be ascending."""
        stages = setup_scale_out(so_strategy, 50000.0, 0.01, 1, 2.0)
        assert stages[0]['tp_price'] < stages[1]['tp_price']

    def test_stage_tp_prices_descending_short(self, so_strategy):
        """SHORT: stage TP prices should be descending."""
        stages = setup_scale_out(so_strategy, 50000.0, 0.01, -1, 2.0)
        assert stages[0]['tp_price'] > stages[1]['tp_price']

    def test_stage_fields(self, so_strategy):
        """Each stage should have all required fields."""
        stages = setup_scale_out(so_strategy, 50000.0, 0.01, 1, 2.0)
        for stage in stages:
            assert 'stage' in stage
            assert 'pct' in stage
            assert 'tp_mult' in stage
            assert 'tp_price' in stage
            assert 'quantity' in stage
            assert 'filled' in stage
            assert stage['filled'] is False

    def test_long_tp_above_entry(self, so_strategy):
        """LONG: all TP prices should be above entry."""
        stages = setup_scale_out(so_strategy, 50000.0, 0.01, 1, 2.0)
        for stage in stages:
            assert stage['tp_price'] > 50000.0

    def test_short_tp_below_entry(self, so_strategy):
        """SHORT: all TP prices should be below entry."""
        stages = setup_scale_out(so_strategy, 50000.0, 0.01, -1, 2.0)
        for stage in stages:
            assert stage['tp_price'] < 50000.0


# ── get_current_pattern ──────────────────────────────────────


class TestGetCurrentPattern:
    """Test get_current_pattern() — 3-candle pattern extraction."""

    def _make_df(self, patterns):
        """Helper: create DataFrame with pattern_3 column."""
        n = len(patterns)
        return pd.DataFrame({
            'open': [50000.0] * n,
            'high': [50100.0] * n,
            'low': [49900.0] * n,
            'close': [50050.0] * n,
            'pattern_3': patterns,
        })

    def test_normal_extraction(self):
        """Normal case → returns iloc[-2] pattern."""
        df = self._make_df([None, None, 'U-U-BD', 'BD-U-MU', 'U-MU-H'])
        assert get_current_pattern(df) == 'BD-U-MU'

    def test_less_than_3_bars(self):
        """Fewer than 3 bars → empty string."""
        df = self._make_df([None, 'U-U-BD'])
        assert get_current_pattern(df) == ''

    def test_missing_pattern_3_column(self):
        """No pattern_3 column → empty string."""
        df = pd.DataFrame({'open': [1, 2, 3, 4], 'close': [1, 2, 3, 4]})
        assert get_current_pattern(df) == ''

    def test_none_pattern_at_iloc_minus_2(self):
        """pattern_3=None at iloc[-2] → empty string."""
        df = self._make_df([None, None, None, 'U-U-BD'])
        assert get_current_pattern(df) == ''

    def test_empty_string_pattern(self):
        """pattern_3='' at iloc[-2] → empty string."""
        df = self._make_df([None, None, '', 'U-U-BD'])
        assert get_current_pattern(df) == ''

    def test_exactly_3_bars(self):
        """Exactly 3 bars → reads iloc[-2] = index 1."""
        df = self._make_df([None, 'U-BD-DN', 'BD-DN-U'])
        assert get_current_pattern(df) == 'U-BD-DN'


# ── get_volatility_multiplier ────────────────────────────────


class TestGetVolatilityMultiplier:
    """Test get_volatility_multiplier() — ATR percentile → multiplier."""

    def _make_atr_df(self, n, atr_values):
        """Helper: create DataFrame with ATR column."""
        return pd.DataFrame({
            'open': [50000.0] * n,
            'close': [50050.0] * n,
            'atr': atr_values,
        })

    def _vol_config(self, enabled=True, lookback=10):
        """Helper: vol_adaptive config."""
        return {
            'strategy': {
                'vol_adaptive': {
                    'enabled': enabled,
                    'lookback': lookback,
                    'thresholds': [0.3, 0.6, 0.9],
                    'multipliers': [0.8, 1.0, 1.15, 1.4],
                },
            },
        }

    def test_disabled_returns_1(self):
        """vol_adaptive disabled → always 1.0."""
        df = self._make_atr_df(20, [100.0] * 20)
        config = self._vol_config(enabled=False)
        assert get_volatility_multiplier(df, config) == 1.0

    def test_no_vol_adaptive_key(self):
        """Missing vol_adaptive key → 1.0."""
        df = self._make_atr_df(20, [100.0] * 20)
        assert get_volatility_multiplier(df, {'strategy': {}}) == 1.0

    def test_no_atr_column(self):
        """No ATR column → 1.0."""
        df = pd.DataFrame({'open': [50000.0] * 20, 'close': [50050.0] * 20})
        assert get_volatility_multiplier(df, self._vol_config(lookback=10)) == 1.0

    def test_insufficient_data(self):
        """Fewer bars than lookback → 1.0."""
        df = self._make_atr_df(5, [100.0] * 5)
        assert get_volatility_multiplier(df, self._vol_config(lookback=10)) == 1.0

    def test_low_volatility(self):
        """Current ATR at min → multiplier = 0.8 (lowest bucket)."""
        # ATR range: 100-200, current=100 → percentile=0.0 (< 0.3 threshold)
        atr_vals = list(np.linspace(100, 200, 9)) + [100.0]
        df = self._make_atr_df(10, atr_vals)
        mult = get_volatility_multiplier(df, self._vol_config(lookback=10))
        assert mult == 0.8

    def test_high_volatility(self):
        """Current ATR at max → multiplier = 1.4 (highest bucket)."""
        # ATR range: 100-200, current=200 → percentile=1.0 (> 0.9 threshold)
        atr_vals = list(np.linspace(100, 200, 9)) + [200.0]
        df = self._make_atr_df(10, atr_vals)
        mult = get_volatility_multiplier(df, self._vol_config(lookback=10))
        assert mult == 1.4

    def test_mid_volatility(self):
        """Current ATR at midpoint → multiplier = 1.0 (second bucket)."""
        # ATR range: 100-200, current=140 → percentile=0.4 (0.3 < p ≤ 0.6)
        atr_vals = list(np.linspace(100, 200, 9)) + [140.0]
        df = self._make_atr_df(10, atr_vals)
        mult = get_volatility_multiplier(df, self._vol_config(lookback=10))
        assert mult == 1.0

    def test_flat_atr(self):
        """All ATR values equal → percentile=0.5 → multiplier=1.0."""
        df = self._make_atr_df(10, [150.0] * 10)
        mult = get_volatility_multiplier(df, self._vol_config(lookback=10))
        assert mult == 1.0

    def test_nan_current_atr(self):
        """Current ATR is NaN → 1.0."""
        atr_vals = [100.0] * 9 + [float('nan')]
        df = self._make_atr_df(10, atr_vals)
        mult = get_volatility_multiplier(df, self._vol_config(lookback=10))
        assert mult == 1.0


# ── _infer_exit_reason ───────────────────────────────────────


class TestInferExitReason:
    """Test _infer_exit_reason() — price proximity to TP/SL."""

    def test_tp_hit(self):
        """Price near TP → 'TP'."""
        pos = {'tp_price': 51000.0, 'sl_price': 49000.0}
        assert _infer_exit_reason(51000.0, pos) == 'TP'

    def test_sl_hit(self):
        """Price near SL → 'SL'."""
        pos = {'tp_price': 51000.0, 'sl_price': 49000.0}
        assert _infer_exit_reason(49000.0, pos) == 'SL'

    def test_market_exit(self):
        """Price far from both TP and SL → 'MARKET'."""
        pos = {'tp_price': 51000.0, 'sl_price': 49000.0}
        assert _infer_exit_reason(50000.0, pos) == 'MARKET'

    def test_within_tolerance(self):
        """Price slightly off TP but within tolerance → 'TP'."""
        pos = {'tp_price': 51000.0, 'sl_price': 49000.0}
        # Tolerance is 0.3% → 51000 * 0.003 = 153. So 51100 is within tolerance.
        assert _infer_exit_reason(51100.0, pos) == 'TP'

    def test_outside_tolerance(self):
        """Price beyond TP tolerance → 'MARKET'."""
        pos = {'tp_price': 51000.0, 'sl_price': 49000.0}
        # 51000 * 0.003 = 153. So 51200 is outside tolerance.
        assert _infer_exit_reason(51200.0, pos) == 'MARKET'

    def test_no_tp_price(self):
        """tp_price=0 → can't match TP."""
        pos = {'tp_price': 0, 'sl_price': 49000.0}
        assert _infer_exit_reason(50000.0, pos) == 'MARKET'

    def test_no_sl_price(self):
        """sl_price=0 → can't match SL."""
        pos = {'tp_price': 51000.0, 'sl_price': 0}
        assert _infer_exit_reason(50000.0, pos) == 'MARKET'

    def test_missing_keys(self):
        """Missing tp_price/sl_price keys → defaults to 0 → 'MARKET'."""
        assert _infer_exit_reason(50000.0, {}) == 'MARKET'

    def test_tp_priority_over_sl(self):
        """If price matches both (TP == SL), TP checked first → 'TP'."""
        pos = {'tp_price': 50000.0, 'sl_price': 50000.0}
        assert _infer_exit_reason(50000.0, pos) == 'TP'


# ── calculate_indicators ─────────────────────────────────────


class TestCalculateIndicators:
    """Test calculate_indicators() — DataFrame enrichment."""

    def _make_ohlcv(self, n=50):
        """Helper: create realistic OHLCV DataFrame."""
        np.random.seed(42)
        base = 50000.0
        closes = base + np.cumsum(np.random.randn(n) * 50)
        opens = np.roll(closes, 1)
        opens[0] = base
        highs = np.maximum(opens, closes) + np.abs(np.random.randn(n) * 30)
        lows = np.minimum(opens, closes) - np.abs(np.random.randn(n) * 30)
        return pd.DataFrame({
            'open': opens, 'high': highs, 'low': lows, 'close': closes,
            'volume': np.random.randint(100, 1000, n).astype(float),
        })

    def test_adds_core_columns(self):
        """Should add body, candle_type, type_code, pattern_3 columns."""
        df = self._make_ohlcv(50)
        result = calculate_indicators(df, {'strategy': {}})
        for col in ['body', 'body_abs', 'avg_body_20', 'candle_type', 'type_code', 'pattern_3']:
            assert col in result.columns

    def test_pattern_3_format(self):
        """pattern_3 should have format 'XX-YY-ZZ' from index 2 onward."""
        df = self._make_ohlcv(50)
        result = calculate_indicators(df, {'strategy': {}})
        # First two are None
        assert result['pattern_3'].iloc[0] is None
        assert result['pattern_3'].iloc[1] is None
        # From index 2: 'XX-YY-ZZ' format
        pat = result['pattern_3'].iloc[AVG_BODY_WINDOW + 3]
        assert isinstance(pat, str)
        parts = pat.split('-')
        assert len(parts) == 3

    def test_insufficient_data(self):
        """Too few bars → returns df unchanged."""
        df = self._make_ohlcv(10)  # < AVG_BODY_WINDOW + 5 = 25
        result = calculate_indicators(df, {'strategy': {}})
        assert 'candle_type' not in result.columns

    def test_no_rsi_without_context_filters(self):
        """No context_filters → RSI/ATR not calculated."""
        df = self._make_ohlcv(50)
        result = calculate_indicators(df, {'strategy': {}})
        assert 'rsi' not in result.columns
        assert 'atr' not in result.columns

    def test_rsi_with_context_filters(self):
        """context_filters enabled → RSI/ATR columns added."""
        df = self._make_ohlcv(50)
        config = {'strategy': {'context_filters': {'some_filter': True}}}
        result = calculate_indicators(df, config)
        assert 'rsi' in result.columns
        assert 'atr' in result.columns
        assert 'atr_pct' in result.columns

    def test_rsi_range(self):
        """RSI values should be 0-100."""
        df = self._make_ohlcv(50)
        config = {'strategy': {'context_filters': {'f': True}}}
        result = calculate_indicators(df, config)
        rsi_valid = result['rsi'].dropna()
        assert (rsi_valid >= 0).all()
        assert (rsi_valid <= 100).all()

    def test_does_not_mutate_input(self):
        """Should not modify the original DataFrame."""
        df = self._make_ohlcv(50)
        original_cols = set(df.columns)
        calculate_indicators(df, {'strategy': {}})
        assert set(df.columns) == original_cols

    def test_type_code_valid_values(self):
        """type_code should be valid CandleType values."""
        from bingx_rl_trading_bot.scripts.production.pattern_5m.constants import CandleType
        valid_codes = {ct.value for ct in CandleType}
        df = self._make_ohlcv(50)
        result = calculate_indicators(df, {'strategy': {}})
        for code in result['type_code']:
            assert code in valid_codes
