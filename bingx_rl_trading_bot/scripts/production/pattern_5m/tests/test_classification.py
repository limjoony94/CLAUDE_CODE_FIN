"""Tests for classify_candle() — the canonical 12-type candle classification."""

import pytest
import pandas as pd
import numpy as np

from bingx_rl_trading_bot.scripts.production.pattern_5m.indicators import (
    classify_candle,
    calculate_indicators,
)
from bingx_rl_trading_bot.scripts.production.pattern_5m.constants import CandleType


# ── Helper ──────────────────────────────────────────────────

def _row(o, h, l, c):
    return pd.Series({'open': o, 'high': h, 'low': l, 'close': c})


# ── 12 Candle Types ─────────────────────────────────────────

class TestClassifyCandleTypes:
    """Each of the 12 CandleType values should be reachable."""

    def test_doji(self):
        # body ≈ 0, balanced wicks
        assert classify_candle(_row(100, 105, 95, 100.5), 80.0) == CandleType.DOJI

    def test_dragonfly(self):
        # body ≈ 0, long lower wick (>70% of range)
        assert classify_candle(_row(100, 100.5, 92, 100.3), 80.0) == CandleType.DRAGONFLY

    def test_gravestone(self):
        # body ≈ 0, long upper wick (>70% of range)
        assert classify_candle(_row(100, 108, 99.5, 100.3), 80.0) == CandleType.GRAVESTONE

    def test_marubozu_up(self):
        # Large bullish body, tiny wicks (<15% of range)
        assert classify_candle(_row(100, 110.5, 99.5, 110), 80.0) == CandleType.MARUBOZU_UP

    def test_marubozu_down(self):
        # Large bearish body, tiny wicks
        assert classify_candle(_row(110, 110.5, 99.5, 100), 80.0) == CandleType.MARUBOZU_DOWN

    def test_hammer(self):
        # Small body at top, long lower wick (>2x body), small upper wick (<0.3x body)
        assert classify_candle(_row(104, 105, 95, 106), 80.0) == CandleType.HAMMER

    def test_inv_hammer(self):
        # Small body at bottom, long upper wick (>2x body), small lower wick
        assert classify_candle(_row(96, 105, 95, 94), 80.0) == CandleType.INV_HAMMER

    def test_spinning_top(self):
        # Small normalized body (<0.5x avg), balanced wicks (>0.5x body each)
        # avg_body=80, body=10 → norm=0.125 < 0.5 ✓; wicks 20 each > 0.5*10=5 ✓
        assert classify_candle(_row(95, 125, 75, 105), 80.0) == CandleType.SPINNING_TOP

    def test_big_up(self):
        # body=130, avg=80, norm=1.625>1.5 ✓; wicks present to avoid marubozu
        assert classify_candle(_row(100, 260, 70, 230), 80.0) == CandleType.BIG_UP

    def test_big_down(self):
        assert classify_candle(_row(230, 260, 70, 100), 80.0) == CandleType.BIG_DOWN

    def test_med_up(self):
        # Normal bullish candle, not big, not spinning top, not doji/hammer/marubozu
        # avg_body=80, body=50 → norm=0.625 (0.5..1.5), body_ratio=50/80=0.625>0.1
        # wicks: upper=10, lower=20, total_wick_ratio=30/80=0.375>0.15 (not marubozu)
        # lower/body=20/50=0.4<2.0 (not hammer), upper/body=10/50=0.2<2.0 (not inv_hammer)
        # upper_wick(10) < 0.5*body(25) → not spinning top
        assert classify_candle(_row(100, 160, 80, 150), 80.0) == CandleType.MED_UP

    def test_med_down(self):
        # Normal bearish candle
        assert classify_candle(_row(150, 160, 80, 100), 80.0) == CandleType.MED_DOWN


# ── Edge Cases ──────────────────────────────────────────────

class TestEdgeCases:
    def test_zero_range(self):
        """range_hl == 0 → DOJI."""
        assert classify_candle(_row(100, 100, 100, 100), 80.0) == CandleType.DOJI

    def test_zero_body_nonzero_range(self):
        """body=0 with range → DOJI."""
        assert classify_candle(_row(100, 110, 90, 100), 80.0) == CandleType.DOJI

    def test_avg_body_zero(self):
        """avg_body_20=0 should not crash (norm_body defaults to 1.0)."""
        result = classify_candle(_row(100, 110, 90, 105), 0.0)
        assert isinstance(result, CandleType)

    def test_avg_body_negative(self):
        """Negative avg_body shouldn't crash."""
        result = classify_candle(_row(100, 110, 90, 105), -1.0)
        assert isinstance(result, CandleType)

    def test_tiny_body(self):
        """Very small body with big range → doji family."""
        r = classify_candle(_row(100.0, 120, 80, 100.1), 80.0)
        assert r in (CandleType.DOJI, CandleType.DRAGONFLY, CandleType.GRAVESTONE)


# ── Early Bar Handling ──────────────────────────────────────

class TestEarlyBars:
    """Verify indicators.py and signals.py agree on early-bar classification."""

    def _make_df(self, n=30):
        """Create a small OHLCV DataFrame."""
        np.random.seed(42)
        opens = 100 + np.cumsum(np.random.randn(n) * 0.5)
        closes = opens + np.random.randn(n) * 2
        highs = np.maximum(opens, closes) + np.abs(np.random.randn(n))
        lows = np.minimum(opens, closes) - np.abs(np.random.randn(n))
        return pd.DataFrame({
            'open': opens, 'high': highs, 'low': lows, 'close': closes,
        })

    def test_calculate_indicators_early_bars_use_default(self):
        """First 20 bars should use avg_body_20=1.0 (not NaN)."""
        df = self._make_df(30)
        config = {}
        result = calculate_indicators(df, config)
        # All rows should have a candle_type (no NaN/None)
        assert result['candle_type'].notna().all()
        assert len(result['candle_type']) == 30

    def test_early_bars_not_forced_med(self):
        """Early bars should NOT be forced to MED_UP/MED_DOWN — they should
        go through classify_candle() with avg_body=1.0, allowing DOJI etc."""
        df = self._make_df(30)
        config = {}
        result = calculate_indicators(df, config)
        # With avg_body=1.0, any candle with body_abs>1.5 becomes BIG_UP/BIG_DOWN
        # This verifies early bars aren't just MED_UP/MED_DOWN
        early_types = set(result['candle_type'].iloc[:20].apply(lambda x: x.value))
        # Should have variety, not just 'U' and 'DN'
        assert len(early_types) > 2, f"Early bars only have {early_types}, expected variety"

    def test_signals_matches_indicators(self):
        """signals.add_candle_classification() should produce identical results
        to indicators.calculate_indicators() for the same data."""
        from bingx_rl_trading_bot.scripts.production.pattern_5m.signals import add_candle_classification

        df = self._make_df(30)
        config = {}
        ind_result = calculate_indicators(df, config)
        sig_result = add_candle_classification(df)

        # type_code columns must match exactly
        assert list(ind_result['type_code']) == list(sig_result['type_code']), \
            "signals.py and indicators.py produce different classifications!"

    def test_pattern_3_built_correctly(self):
        """pattern_3 should be None for first 2 rows, then 'X-Y-Z' format."""
        df = self._make_df(30)
        config = {}
        result = calculate_indicators(df, config)
        assert pd.isna(result['pattern_3'].iloc[0])
        assert pd.isna(result['pattern_3'].iloc[1])
        assert '-' in str(result['pattern_3'].iloc[2])
        parts = result['pattern_3'].iloc[5].split('-')
        assert len(parts) == 3
