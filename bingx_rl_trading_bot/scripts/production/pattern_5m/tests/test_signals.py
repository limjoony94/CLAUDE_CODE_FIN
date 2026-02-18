"""Tests for signals.py — context, confidence, entry signal, cooldown, consecutive loss."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch

from bingx_rl_trading_bot.scripts.production.pattern_5m.signals import (
    add_candle_classification,
    calculate_context,
    check_context_filter,
    calculate_candle_clarity,
    calculate_pattern_confidence,
    check_entry_signal,
    check_cooldown,
    check_consecutive_loss_limit,
    check_daily_loss_limit,
    check_early_exit_signal,
)
from bingx_rl_trading_bot.scripts.production.pattern_5m.constants import (
    CandleType,
    RSI_OVERSOLD_THRESHOLD,
    RSI_OVERBOUGHT_THRESHOLD,
    TREND_LOOKBACK_BARS,
    MAX_CONSECUTIVE_LOSSES,
    VALIDATED_LONG_PATTERNS,
    VALIDATED_SHORT_PATTERNS,
    CONFIDENCE_WEIGHT_CLARITY,
    CONFIDENCE_WEIGHT_HISTORICAL,
    CONFIDENCE_WEIGHT_REGIME,
    CONTEXT_PREFERRED_BONUS,
    EARLY_EXIT_CONFIRM_CANDLES,
    EARLY_EXIT_MIN_PROFIT_PCT,
)


# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture
def make_ohlcv_df():
    """Factory to create a realistic OHLCV DataFrame with indicators."""
    def _make(n_bars=50, rsi_value=50.0, trend_up=True, atr_pct_value=0.5):
        base_price = 50000.0
        # Create trending data
        if trend_up:
            prices = [base_price + i * 10 for i in range(n_bars)]
        else:
            prices = [base_price - i * 10 for i in range(n_bars)]

        timestamps = pd.date_range('2025-01-01', periods=n_bars, freq='5min')
        df = pd.DataFrame({
            'open': prices,
            'high': [p + 50 for p in prices],
            'low': [p - 50 for p in prices],
            'close': [p + 20 for p in prices],
            'volume': [100.0] * n_bars,
            'timestamp': (timestamps.astype('int64') // 10**6).tolist(),
        })

        # Add pre-computed indicator columns
        df['rsi'] = rsi_value
        df['atr_pct'] = atr_pct_value
        df['body'] = df['close'] - df['open']
        df['avg_body_20'] = df['body'].abs().rolling(20, min_periods=1).mean()

        # Add classification columns
        df['candle_type'] = CandleType.MED_UP
        df['type_code'] = 'U'
        df['pattern_3'] = None

        return df
    return _make


@pytest.fixture
def make_classified_df(make_ohlcv_df):
    """Factory to create DataFrame with pattern_3 set on the second-to-last row."""
    def _make(pattern='U-U-U', signal_direction='LONG', **kwargs):
        df = make_ohlcv_df(**kwargs)
        df.at[df.index[-2], 'pattern_3'] = pattern
        return df
    return _make


# ── calculate_context ────────────────────────────────────────


class TestCalculateContext:
    """Test calculate_context() RSI zone, volatility, and trend detection."""

    def test_insufficient_data(self):
        """Short DataFrame → returns neutral defaults."""
        df = pd.DataFrame({
            'open': [50000], 'high': [50100], 'low': [49900],
            'close': [50050], 'rsi': [50.0], 'atr_pct': [0.5],
        })
        ctx = calculate_context(df)
        assert ctx == {'rsi_zone': 'N', 'vol': 'M', 'trend': 'N'}

    def test_rsi_oversold(self, make_ohlcv_df):
        """RSI below threshold → 'OS' zone."""
        df = make_ohlcv_df(rsi_value=RSI_OVERSOLD_THRESHOLD - 5)
        ctx = calculate_context(df)
        assert ctx['rsi_zone'] == 'OS'

    def test_rsi_overbought(self, make_ohlcv_df):
        """RSI above threshold → 'OB' zone."""
        df = make_ohlcv_df(rsi_value=RSI_OVERBOUGHT_THRESHOLD + 5)
        ctx = calculate_context(df)
        assert ctx['rsi_zone'] == 'OB'

    def test_rsi_neutral(self, make_ohlcv_df):
        """RSI in normal range → 'N' zone."""
        df = make_ohlcv_df(rsi_value=50.0)
        ctx = calculate_context(df)
        assert ctx['rsi_zone'] == 'N'

    def test_rsi_nan_defaults_neutral(self, make_ohlcv_df):
        """NaN RSI → treated as neutral (50.0)."""
        df = make_ohlcv_df()
        df['rsi'] = float('nan')
        ctx = calculate_context(df)
        assert ctx['rsi_zone'] == 'N'

    def test_trend_up(self, make_ohlcv_df):
        """Uptrend data → trend='UP'."""
        df = make_ohlcv_df(trend_up=True)
        ctx = calculate_context(df)
        assert ctx['trend'] == 'UP'

    def test_trend_down(self, make_ohlcv_df):
        """Downtrend data → trend='DN'."""
        df = make_ohlcv_df(trend_up=False)
        ctx = calculate_context(df)
        assert ctx['trend'] == 'DN'

    def test_context_includes_rsi_value(self, make_ohlcv_df):
        """Context dict should include raw rsi value."""
        df = make_ohlcv_df(rsi_value=42.5)
        ctx = calculate_context(df)
        assert ctx['rsi'] == pytest.approx(42.5)

    def test_vol_low(self, make_ohlcv_df):
        """ATR below low quantile → vol='L'."""
        df = make_ohlcv_df(atr_pct_value=0.01)
        # Set a high baseline so the current value is in the bottom quantile
        df['atr_pct'] = 1.0
        df.iloc[-2, df.columns.get_loc('atr_pct')] = 0.01
        ctx = calculate_context(df)
        assert ctx['vol'] == 'L'

    def test_vol_high(self, make_ohlcv_df):
        """ATR above high quantile → vol='H'."""
        df = make_ohlcv_df(atr_pct_value=0.01)
        # Set a low baseline so the current value is in the top quantile
        df['atr_pct'] = 0.01
        df.iloc[-2, df.columns.get_loc('atr_pct')] = 5.0
        ctx = calculate_context(df)
        assert ctx['vol'] == 'H'

    def test_no_atr_column(self, make_ohlcv_df):
        """Missing atr_pct column → vol defaults gracefully."""
        df = make_ohlcv_df()
        df.drop(columns=['atr_pct'], inplace=True)
        ctx = calculate_context(df)
        assert ctx['vol'] in ('L', 'M', 'H')


# ── check_context_filter ─────────────────────────────────────


class TestCheckContextFilter:
    """Test check_context_filter() with various filter configurations."""

    def test_filter_disabled(self):
        """CONTEXT_FILTER_ENABLED=False → always passes."""
        with patch('bingx_rl_trading_bot.scripts.production.pattern_5m.signals.CONTEXT_FILTER_ENABLED', False):
            passes, bonus, reason = check_context_filter('U-U-U', {'rsi_zone': 'OB'})
            assert passes is True
            assert bonus == 0.0
            assert reason == "context_disabled"

    def test_no_filter_for_pattern(self):
        """Pattern not in PATTERN_CONTEXT_FILTERS → passes with no_filter."""
        # Default: PATTERN_CONTEXT_FILTERS = {} (empty)
        passes, bonus, reason = check_context_filter('NONEXISTENT', {'rsi_zone': 'N'})
        assert passes is True
        assert reason == "no_filter"

    def test_required_filter_passes(self):
        """Pattern passes required context filter."""
        filters = {'U-U-U': {'required': {'rsi_zone': ['N', 'OS']}}}
        with patch('bingx_rl_trading_bot.scripts.production.pattern_5m.signals.PATTERN_CONTEXT_FILTERS', filters):
            passes, bonus, reason = check_context_filter('U-U-U', {'rsi_zone': 'N'})
            assert passes is True

    def test_required_filter_rejects(self):
        """Pattern fails required context filter."""
        filters = {'U-U-U': {'required': {'rsi_zone': ['OS']}}}
        with patch('bingx_rl_trading_bot.scripts.production.pattern_5m.signals.PATTERN_CONTEXT_FILTERS', filters):
            passes, bonus, reason = check_context_filter('U-U-U', {'rsi_zone': 'OB'})
            assert passes is False
            assert 'required_rsi_zone' in reason

    def test_excluded_filter_rejects(self):
        """Pattern fails excluded context filter."""
        filters = {'U-U-U': {'excluded': {'trend': ['DN']}}}
        with patch('bingx_rl_trading_bot.scripts.production.pattern_5m.signals.PATTERN_CONTEXT_FILTERS', filters):
            passes, bonus, reason = check_context_filter('U-U-U', {'trend': 'DN'})
            assert passes is False
            assert 'excluded_trend' in reason

    def test_excluded_filter_passes(self):
        """Pattern passes excluded context filter (value not excluded)."""
        filters = {'U-U-U': {'excluded': {'trend': ['DN']}}}
        with patch('bingx_rl_trading_bot.scripts.production.pattern_5m.signals.PATTERN_CONTEXT_FILTERS', filters):
            passes, bonus, reason = check_context_filter('U-U-U', {'trend': 'UP'})
            assert passes is True

    def test_preferred_filter_bonus(self):
        """Preferred context match → gives confidence bonus."""
        filters = {'U-U-U': {'preferred': {'vol': ['H']}}}
        with patch('bingx_rl_trading_bot.scripts.production.pattern_5m.signals.PATTERN_CONTEXT_FILTERS', filters):
            passes, bonus, reason = check_context_filter('U-U-U', {'vol': 'H'})
            assert passes is True
            assert bonus == pytest.approx(CONTEXT_PREFERRED_BONUS)
            assert 'preferred_match' in reason

    def test_preferred_no_match(self):
        """Preferred context NOT matched → no bonus, still passes."""
        filters = {'U-U-U': {'preferred': {'vol': ['H']}}}
        with patch('bingx_rl_trading_bot.scripts.production.pattern_5m.signals.PATTERN_CONTEXT_FILTERS', filters):
            passes, bonus, reason = check_context_filter('U-U-U', {'vol': 'L'})
            assert passes is True
            assert bonus == 0.0
            assert reason == "passed"


# ── calculate_candle_clarity ─────────────────────────────────


class TestCalculateCandleClarity:
    """Test calculate_candle_clarity() for various candle types."""

    def _make_row(self, o, h, l, c, candle_type):
        return pd.Series({
            'open': o, 'high': h, 'low': l, 'close': c,
            'candle_type': candle_type,
        })

    def test_zero_range(self):
        """Zero-range candle → clarity 0.5."""
        row = self._make_row(100, 100, 100, 100, CandleType.DOJI)
        assert calculate_candle_clarity(row, 10.0) == 0.5

    def test_doji_clarity(self):
        """DOJI with tiny body → high clarity (close to 1.0)."""
        row = self._make_row(100.0, 110.0, 90.0, 100.1, CandleType.DOJI)
        clarity = calculate_candle_clarity(row, 10.0)
        assert 0.3 <= clarity <= 1.0

    def test_marubozu_up_clarity(self):
        """MARUBOZU_UP with no wicks → high clarity."""
        row = self._make_row(100.0, 110.0, 100.0, 110.0, CandleType.MARUBOZU_UP)
        clarity = calculate_candle_clarity(row, 10.0)
        assert clarity >= 0.5

    def test_hammer_clarity(self):
        """HAMMER with long lower wick → high clarity."""
        # body: 100-105, lower wick: 80-100 (20), upper: 105-106 (1)
        row = self._make_row(100.0, 106.0, 80.0, 105.0, CandleType.HAMMER)
        clarity = calculate_candle_clarity(row, 10.0)
        assert clarity > 0.3

    def test_spinning_top_clarity(self):
        """SPINNING_TOP with balanced wicks → moderate-to-high clarity."""
        # small body, balanced wicks
        row = self._make_row(100.0, 108.0, 92.0, 101.0, CandleType.SPINNING_TOP)
        clarity = calculate_candle_clarity(row, 10.0)
        assert 0.0 <= clarity <= 1.0

    def test_big_up_clarity(self):
        """BIG_UP with large body → high clarity when body >> avg."""
        row = self._make_row(100.0, 130.0, 99.0, 129.0, CandleType.BIG_UP)
        clarity = calculate_candle_clarity(row, 5.0)  # avg_body=5, actual=29
        assert clarity > 0.5

    def test_med_up_default(self):
        """MED_UP → returns fixed 0.6."""
        row = self._make_row(100.0, 108.0, 98.0, 106.0, CandleType.MED_UP)
        assert calculate_candle_clarity(row, 10.0) == 0.6

    def test_med_down_default(self):
        """MED_DOWN → returns fixed 0.6."""
        row = self._make_row(106.0, 108.0, 98.0, 100.0, CandleType.MED_DOWN)
        assert calculate_candle_clarity(row, 10.0) == 0.6

    def test_none_candle_type(self):
        """candle_type=None (NaN in pandas) → falls to default 0.6."""
        row = self._make_row(100, 110, 90, 105, None)
        # pandas converts None to NaN, which fails `is None` check → else branch → 0.6
        assert calculate_candle_clarity(row, 10.0) == 0.6

    def test_dragonfly_clarity(self):
        """DRAGONFLY with dominant lower wick → high clarity."""
        # Lower wick dominates: low=80, close=open=100, high=101
        row = self._make_row(100.0, 101.0, 80.0, 100.0, CandleType.DRAGONFLY)
        clarity = calculate_candle_clarity(row, 10.0)
        assert clarity > 0.5

    def test_inv_hammer_clarity(self):
        """INV_HAMMER with long upper wick → high clarity."""
        # body: 100-102, upper wick: 102-120 (18)
        row = self._make_row(100.0, 120.0, 99.0, 102.0, CandleType.INV_HAMMER)
        clarity = calculate_candle_clarity(row, 5.0)
        assert clarity > 0.3


# ── calculate_pattern_confidence ─────────────────────────────


class TestCalculatePatternConfidence:
    """Test calculate_pattern_confidence() weighted score."""

    def test_insufficient_data(self):
        """Short DataFrame → returns default 0.5."""
        df = pd.DataFrame({'open': [1], 'high': [2], 'low': [0], 'close': [1.5]})
        conf, comps = calculate_pattern_confidence(df, 'U-U-U')
        assert conf == 0.5
        assert comps == {"clarity": 0.5, "historical": 0.5, "regime": 0.5}

    def test_confidence_range(self, make_classified_df):
        """Confidence score must be between 0 and 1."""
        df = make_classified_df(pattern='U-U-U')
        conf, comps = calculate_pattern_confidence(df, 'U-U-U')
        assert 0.0 <= conf <= 1.0
        assert 0.0 <= comps['clarity'] <= 1.0
        assert 0.0 <= comps['historical'] <= 1.0

    def test_regime_placeholder(self, make_classified_df):
        """Regime component should be 0.6 (placeholder)."""
        df = make_classified_df(pattern='U-U-U')
        _, comps = calculate_pattern_confidence(df, 'U-U-U')
        assert comps['regime'] == 0.6

    def test_high_wr_pattern_higher_historical(self, make_classified_df):
        """Pattern with high WR → higher historical component."""
        df = make_classified_df(pattern='U-U-U')
        # Mock a high WR pattern
        with patch('bingx_rl_trading_bot.scripts.production.pattern_5m.signals.PATTERN_STATS',
                   {'HIGH-WR': {'wr': 95.0}, 'LOW-WR': {'wr': 55.0}}):
            _, comps_high = calculate_pattern_confidence(df, 'HIGH-WR')
            _, comps_low = calculate_pattern_confidence(df, 'LOW-WR')
            assert comps_high['historical'] > comps_low['historical']

    def test_unknown_pattern_50_historical(self, make_classified_df):
        """Unknown pattern (not in PATTERN_STATS) → historical based on 50% WR."""
        df = make_classified_df(pattern='U-U-U')
        _, comps = calculate_pattern_confidence(df, 'UNKNOWN-PATTERN')
        assert comps['historical'] == 0.0  # (50/100 - 0.50) / 0.20 = 0.0

    def test_dynamic_pattern_stats_used(self, make_classified_df):
        """Dynamic pattern stats should be used when available."""
        df = make_classified_df(pattern='U-U-U')
        config = {
            '_dynamic_pattern_stats': {
                ('DYN-PAT', 'SHORT'): {'wr': 90.0, 'trades': 50, 'edge': 25.0},
            }
        }
        _, comps = calculate_pattern_confidence(
            df, 'DYN-PAT', config=config, direction='SHORT'
        )
        # (90/100 - 0.50) / 0.20 = 2.0, clamped to 1.0
        assert comps['historical'] == 1.0

    def test_dynamic_stats_fallback_to_static(self, make_classified_df):
        """Missing dynamic entry falls back to static PATTERN_STATS."""
        df = make_classified_df(pattern='U-U-U')
        config = {
            '_dynamic_pattern_stats': {}  # no entries
        }
        with patch('bingx_rl_trading_bot.scripts.production.pattern_5m.signals.PATTERN_STATS',
                   {'STATIC-PAT': {'wr': 75.0}}):
            _, comps = calculate_pattern_confidence(
                df, 'STATIC-PAT', config=config, direction='LONG'
            )
            # (75/100 - 0.50) / 0.20 = 1.25, clamped to 1.0
            assert comps['historical'] == 1.0

    def test_dynamic_stats_direction_mismatch(self, make_classified_df):
        """Wrong direction in dynamic stats → fallback to static."""
        df = make_classified_df(pattern='U-U-U')
        config = {
            '_dynamic_pattern_stats': {
                ('DIR-PAT', 'SHORT'): {'wr': 90.0},
            }
        }
        _, comps = calculate_pattern_confidence(
            df, 'DIR-PAT', config=config, direction='LONG'  # LONG ≠ SHORT
        )
        # Not found in dynamic (wrong direction) or static → default 50% → 0.0
        assert comps['historical'] == 0.0

    def test_no_config_backward_compatible(self, make_classified_df):
        """No config/direction args → backward compatible with static lookup."""
        df = make_classified_df(pattern='U-U-U')
        # Without config and direction, should still work (uses PATTERN_STATS only)
        conf, comps = calculate_pattern_confidence(df, 'UNKNOWN-PAT')
        assert 0.0 <= conf <= 1.0
        assert comps['historical'] == 0.0


# ── check_entry_signal ───────────────────────────────────────


class TestCheckEntrySignal:
    """Test check_entry_signal() pattern matching and signal generation."""

    def _make_state(self):
        return {'last_signal_candle_timestamp': None}

    def _make_config(self):
        return {
            'strategy': {},
        }

    def test_insufficient_data(self):
        """DataFrame with < 5 bars → returns (None, None)."""
        df = pd.DataFrame({
            'open': [1, 2, 3], 'high': [2, 3, 4],
            'low': [0, 1, 2], 'close': [1.5, 2.5, 3.5],
        })
        signal, reason = check_entry_signal(df, self._make_state(), self._make_config())
        assert signal is None
        assert reason is None

    def test_long_pattern_detected(self, make_classified_df):
        """Known LONG pattern → returns ('LONG', reason)."""
        if not VALIDATED_LONG_PATTERNS:
            pytest.skip("No validated LONG patterns")
        pattern = VALIDATED_LONG_PATTERNS[0]
        df = make_classified_df(pattern=pattern)
        signal, reason = check_entry_signal(df, self._make_state(), self._make_config())
        assert signal == 'LONG'
        assert pattern in reason

    def test_short_pattern_detected(self, make_classified_df):
        """Known SHORT pattern → returns ('SHORT', reason)."""
        if not VALIDATED_SHORT_PATTERNS:
            pytest.skip("No validated SHORT patterns")
        pattern = VALIDATED_SHORT_PATTERNS[0]
        df = make_classified_df(pattern=pattern)
        signal, reason = check_entry_signal(df, self._make_state(), self._make_config())
        assert signal == 'SHORT'
        assert pattern in reason

    def test_unknown_pattern_no_signal(self, make_classified_df):
        """Unknown pattern → returns (None, None)."""
        df = make_classified_df(pattern='FAKE-FAKE-FAKE')
        signal, reason = check_entry_signal(df, self._make_state(), self._make_config())
        assert signal is None
        assert reason is None

    def test_no_pattern_3_column(self, make_ohlcv_df):
        """DataFrame without pattern_3 → calculates indicators first."""
        df = make_ohlcv_df(n_bars=50)
        df.drop(columns=['pattern_3'], inplace=True)
        # Should not crash — will recalculate indicators
        signal, reason = check_entry_signal(df, self._make_state(), self._make_config())
        # Result depends on calculated pattern, but shouldn't error
        assert signal is None or signal in ('LONG', 'SHORT')

    def test_duplicate_signal_prevention(self, make_classified_df):
        """Same candle timestamp → second call returns (None, None)."""
        if not VALIDATED_LONG_PATTERNS:
            pytest.skip("No validated LONG patterns")
        pattern = VALIDATED_LONG_PATTERNS[0]
        df = make_classified_df(pattern=pattern)
        state = self._make_state()

        # First call → signal
        signal1, _ = check_entry_signal(df, state, self._make_config())
        assert signal1 == 'LONG'

        # Second call with same state → duplicate blocked
        signal2, _ = check_entry_signal(df, state, self._make_config())
        assert signal2 is None

    def test_long_takes_priority_over_short(self, make_classified_df):
        """Pattern in both LONG and SHORT → LONG wins (if/elif ordering)."""
        # This tests the if/elif structure: LONG is checked first
        # Find a pattern in LONG set
        if not VALIDATED_LONG_PATTERNS:
            pytest.skip("No validated LONG patterns")
        pattern = VALIDATED_LONG_PATTERNS[0]

        # Temporarily add it to SHORT too
        df = make_classified_df(pattern=pattern)
        config = {
            'strategy': {
                'long_patterns': [pattern],
                'short_patterns': [pattern],  # same pattern in both
            }
        }
        signal, _ = check_entry_signal(df, self._make_state(), config)
        assert signal == 'LONG'  # LONG takes priority

    def test_config_overrides_default_patterns(self, make_classified_df):
        """Config-provided pattern lists override default constants."""
        df = make_classified_df(pattern='CUSTOM-PAT-LONG')
        config = {
            'strategy': {
                'long_patterns': ['CUSTOM-PAT-LONG'],
                'short_patterns': [],
            }
        }
        signal, _ = check_entry_signal(df, self._make_state(), config)
        assert signal == 'LONG'

    def test_none_pattern_no_crash(self, make_ohlcv_df):
        """pattern_3=None → returns (None, None)."""
        df = make_ohlcv_df(n_bars=10)
        # pattern_3 column exists but all None (default from fixture)
        signal, reason = check_entry_signal(df, self._make_state(), self._make_config())
        assert signal is None


# ── check_cooldown ───────────────────────────────────────────


class TestCheckCooldown:
    """Test check_cooldown() timing logic."""

    def test_no_cooldown_configured(self):
        """cooldown_candles=0 → always returns True."""
        state = {}
        config = {'strategy': {'cooldown_candles': 0}}
        assert check_cooldown(state, config) is True

    def test_no_last_signal_time(self):
        """No last_signal_time in state → returns True."""
        state = {}
        config = {'strategy': {'cooldown_candles': 3}}
        assert check_cooldown(state, config) is True

    def test_cooldown_not_elapsed(self):
        """Signal too recent → returns False."""
        now = datetime.now()
        state = {'last_signal_time': now.isoformat()}
        config = {'strategy': {'cooldown_candles': 3}}  # 15 minutes cooldown
        assert check_cooldown(state, config) is False

    def test_cooldown_elapsed(self):
        """Signal long ago → returns True."""
        past = datetime.now() - timedelta(hours=1)
        state = {'last_signal_time': past.isoformat()}
        config = {'strategy': {'cooldown_candles': 3}}  # 15 minutes cooldown
        assert check_cooldown(state, config) is True

    def test_missing_strategy_key(self):
        """Missing 'strategy' in config → defaults to 0 cooldown → True."""
        state = {}
        config = {}
        assert check_cooldown(state, config) is True


# ── check_consecutive_loss_limit ─────────────────────────────


class TestCheckConsecutiveLossLimit:
    """Test check_consecutive_loss_limit() threshold."""

    def test_zero_losses(self):
        """No consecutive losses → returns False."""
        assert check_consecutive_loss_limit({'consecutive_losses': 0}) is False

    def test_below_threshold(self):
        """Losses below threshold → returns False."""
        assert check_consecutive_loss_limit(
            {'consecutive_losses': MAX_CONSECUTIVE_LOSSES - 1}
        ) is False

    def test_at_threshold(self):
        """Losses at threshold → returns True."""
        assert check_consecutive_loss_limit(
            {'consecutive_losses': MAX_CONSECUTIVE_LOSSES}
        ) is True

    def test_above_threshold(self):
        """Losses above threshold → returns True."""
        assert check_consecutive_loss_limit(
            {'consecutive_losses': MAX_CONSECUTIVE_LOSSES + 5}
        ) is True

    def test_missing_key(self):
        """Missing 'consecutive_losses' key → defaults to 0 → False."""
        assert check_consecutive_loss_limit({}) is False


# ── add_candle_classification (wrapper) ──────────────────────


class TestAddCandleClassification:
    """Test add_candle_classification() delegation to calculate_indicators."""

    def test_delegates_to_calculate_indicators(self, make_ohlcv_df):
        """Should add candle_type column via calculate_indicators."""
        df = make_ohlcv_df(n_bars=30)
        # Remove pre-set columns to verify they get regenerated
        df.drop(columns=['candle_type', 'type_code', 'pattern_3'], inplace=True)
        result = add_candle_classification(df)
        assert 'candle_type' in result.columns

    def test_with_none_config(self, make_ohlcv_df):
        """config=None should not crash."""
        df = make_ohlcv_df(n_bars=30)
        df.drop(columns=['candle_type', 'type_code', 'pattern_3'], inplace=True)
        result = add_candle_classification(df, config=None)
        assert 'candle_type' in result.columns


# ── check_daily_loss_limit ──────────────────────────────────


class TestCheckDailyLossLimit:
    """Test check_daily_loss_limit() — daily loss threshold."""

    def _today(self):
        return datetime.now().strftime('%Y-%m-%d')

    def test_no_loss(self):
        """No daily loss → not limited."""
        state = {'daily_pnl': 0.0, 'last_trade_date': self._today()}
        config = {'risk': {'max_daily_loss_pct': 10.0}}
        assert check_daily_loss_limit(state, config) is False

    def test_profit_day(self):
        """Positive PnL → not limited."""
        state = {'daily_pnl': 5.0, 'last_trade_date': self._today()}
        config = {'risk': {'max_daily_loss_pct': 10.0}}
        assert check_daily_loss_limit(state, config) is False

    def test_below_limit(self):
        """Loss below threshold → not limited."""
        state = {'daily_pnl': -9.9, 'last_trade_date': self._today()}
        config = {'risk': {'max_daily_loss_pct': 10.0}}
        assert check_daily_loss_limit(state, config) is False

    def test_at_limit(self):
        """Loss exactly at threshold → limited."""
        state = {'daily_pnl': -10.0, 'last_trade_date': self._today()}
        config = {'risk': {'max_daily_loss_pct': 10.0}}
        assert check_daily_loss_limit(state, config) is True

    def test_above_limit(self):
        """Loss beyond threshold → limited."""
        state = {'daily_pnl': -15.0, 'last_trade_date': self._today()}
        config = {'risk': {'max_daily_loss_pct': 10.0}}
        assert check_daily_loss_limit(state, config) is True

    def test_default_config(self):
        """Missing risk config → uses default 10%."""
        state = {'daily_pnl': -11.0, 'last_trade_date': self._today()}
        assert check_daily_loss_limit(state, {}) is True

    def test_custom_threshold(self):
        """Custom max_daily_loss_pct = 13%."""
        state = {'daily_pnl': -12.0, 'last_trade_date': self._today()}
        config = {'risk': {'max_daily_loss_pct': 13.0}}
        assert check_daily_loss_limit(state, config) is False


# ── check_early_exit_signal ─────────────────────────────────


class TestCheckEarlyExitSignal:
    """Test check_early_exit_signal() — reversal candle early exit."""

    @pytest.fixture
    def long_position(self):
        """LONG position with profit."""
        return {
            'direction': 'LONG',
            'entry_price': 50000.0,
            'reversal_count': 0,
            'last_counted_candle_ts': None,
        }

    @pytest.fixture
    def short_position(self):
        """SHORT position with profit."""
        return {
            'direction': 'SHORT',
            'entry_price': 50000.0,
            'reversal_count': 0,
            'last_counted_candle_ts': None,
        }

    @pytest.fixture
    def config_3x(self):
        """Standard config with 3x leverage."""
        return {'leverage': 3}

    def _make_df(self, type_codes, timestamps=None):
        """Helper: create DataFrame with type_code column."""
        n = len(type_codes)
        if timestamps is None:
            timestamps = list(range(1000, 1000 + n * 300000, 300000))
        df = pd.DataFrame({
            'open': [50000.0] * n,
            'high': [50100.0] * n,
            'low': [49900.0] * n,
            'close': [50050.0] * n,
            'type_code': type_codes,
            'timestamp': timestamps,
        })
        return df

    def test_disabled_returns_false(self, long_position, config_3x):
        """Early exit disabled → no exit."""
        df = self._make_df(['U', 'BD', 'BD', 'BD', 'BD'])
        with patch(
            'bingx_rl_trading_bot.scripts.production.pattern_5m.signals.EARLY_EXIT_CONFIG',
            {'enabled': False},
        ):
            should_exit, count, reason, ts = check_early_exit_signal(
                long_position, df, 50500.0, config_3x
            )
        assert should_exit is False

    def test_insufficient_data(self, long_position, config_3x):
        """Less than 2 bars → no exit."""
        df = self._make_df(['BD'])
        should_exit, count, reason, ts = check_early_exit_signal(
            long_position, df, 50500.0, config_3x
        )
        assert should_exit is False

    def test_no_type_code_column(self, long_position, config_3x):
        """Missing type_code column → no exit."""
        df = pd.DataFrame({'open': [50000.0, 50000.0], 'close': [50000.0, 50000.0]})
        should_exit, count, reason, ts = check_early_exit_signal(
            long_position, df, 50500.0, config_3x
        )
        assert should_exit is False

    def test_long_reversal_count_increments(self, long_position, config_3x):
        """LONG: BD candle → reversal count +1."""
        df = self._make_df(['U', 'U', 'BD', 'BD'])
        # current_price above entry to ensure profit check doesn't block
        price = 50100.0  # 0.2% profit * 3x = 0.6% > 0.3%
        should_exit, count, reason, ts = check_early_exit_signal(
            long_position, df, price, config_3x
        )
        assert should_exit is False
        assert count == 1  # BD counted

    def test_long_3bd_triggers_exit(self, config_3x):
        """LONG: 3 consecutive BD + profit → early exit."""
        position = {
            'direction': 'LONG',
            'entry_price': 50000.0,
            'reversal_count': 2,  # Already 2 BDs counted
            'last_counted_candle_ts': None,
        }
        # iloc[-2] is the last completed candle → BD must be there
        df = self._make_df(['U', 'U', 'BD', 'U'])
        price = 50100.0  # 0.2% * 3x = 0.6% > 0.3% min profit
        should_exit, count, reason, ts = check_early_exit_signal(
            position, df, price, config_3x
        )
        assert should_exit is True
        assert count == EARLY_EXIT_CONFIRM_CANDLES
        assert reason is not None

    def test_short_3bu_triggers_exit(self, config_3x):
        """SHORT: 3 consecutive BU + profit → early exit."""
        position = {
            'direction': 'SHORT',
            'entry_price': 50000.0,
            'reversal_count': 2,  # Already 2 BUs counted
            'last_counted_candle_ts': None,
        }
        # iloc[-2] is the last completed candle → BU must be there
        df = self._make_df(['U', 'U', 'BU', 'U'])
        price = 49900.0  # 0.2% * 3x = 0.6% > 0.3% min profit
        should_exit, count, reason, ts = check_early_exit_signal(
            position, df, price, config_3x
        )
        assert should_exit is True
        assert count == EARLY_EXIT_CONFIRM_CANDLES

    def test_no_exit_insufficient_profit(self, config_3x):
        """3 BDs but PnL < min_profit_pct → no exit."""
        position = {
            'direction': 'LONG',
            'entry_price': 50000.0,
            'reversal_count': 2,
            'last_counted_candle_ts': None,
        }
        # BD at iloc[-2] (last completed candle)
        df = self._make_df(['U', 'U', 'BD', 'U'])
        price = 50000.0  # breakeven → PnL ≈ 0%
        should_exit, count, reason, ts = check_early_exit_signal(
            position, df, price, config_3x
        )
        assert should_exit is False

    def test_non_reversal_resets_count(self, long_position, config_3x):
        """Non-reversal candle → count resets to 0."""
        long_position['reversal_count'] = 2
        df = self._make_df(['U', 'U', 'U', 'U'])  # U is not BD
        should_exit, count, reason, ts = check_early_exit_signal(
            long_position, df, 50100.0, config_3x
        )
        assert should_exit is False
        assert count == 0

    def test_same_candle_no_double_count(self, long_position, config_3x):
        """Same candle timestamp → don't re-count."""
        long_position['last_counted_candle_ts'] = 1300000
        df = self._make_df(['U', 'U', 'U', 'BD'], timestamps=[1000, 1300000, 1300000, 1300000])
        should_exit, count, reason, ts = check_early_exit_signal(
            long_position, df, 50100.0, config_3x
        )
        assert should_exit is False
        assert count == 0  # Not re-counted

    def test_invalid_direction(self, config_3x):
        """Unknown direction → no exit."""
        position = {
            'direction': 'UNKNOWN',
            'entry_price': 50000.0,
            'reversal_count': 0,
            'last_counted_candle_ts': None,
        }
        df = self._make_df(['U', 'U', 'BD', 'BD'])
        should_exit, count, reason, ts = check_early_exit_signal(
            position, df, 50100.0, config_3x
        )
        assert should_exit is False

    def test_zero_price_no_exit(self, long_position, config_3x):
        """current_price=0 → no exit."""
        df = self._make_df(['U', 'U', 'BD', 'BD'])
        should_exit, count, reason, ts = check_early_exit_signal(
            long_position, df, 0.0, config_3x
        )
        assert should_exit is False
