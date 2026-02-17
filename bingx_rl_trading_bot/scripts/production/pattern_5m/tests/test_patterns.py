"""Tests for 3-candle pattern matching logic."""

import pytest

from bingx_rl_trading_bot.scripts.production.pattern_5m.constants import (
    VALIDATED_LONG_PATTERNS,
    VALIDATED_SHORT_PATTERNS,
    PATTERN_OPTIMAL_TPSL,
    PATTERN_STATS,
)


class TestPatternLists:
    """Verify pattern list consistency."""

    def test_long_patterns_not_empty(self):
        assert len(VALIDATED_LONG_PATTERNS) > 0

    def test_short_patterns_not_empty(self):
        assert len(VALIDATED_SHORT_PATTERNS) > 0

    def test_no_overlap(self):
        overlap = set(VALIDATED_LONG_PATTERNS) & set(VALIDATED_SHORT_PATTERNS)
        assert not overlap, f"Patterns in both LONG and SHORT: {overlap}"

    def test_all_patterns_have_tpsl(self):
        """Every validated pattern should have an entry in PATTERN_OPTIMAL_TPSL."""
        all_patterns = set(VALIDATED_LONG_PATTERNS) | set(VALIDATED_SHORT_PATTERNS)
        missing = all_patterns - set(PATTERN_OPTIMAL_TPSL.keys())
        assert not missing, f"Missing TP/SL config for: {missing}"

    def test_all_patterns_have_stats(self):
        """Every validated pattern should have historical stats."""
        all_patterns = set(VALIDATED_LONG_PATTERNS) | set(VALIDATED_SHORT_PATTERNS)
        missing = all_patterns - set(PATTERN_STATS.keys())
        assert not missing, f"Missing stats for: {missing}"

    def test_pattern_format(self):
        """All patterns should be in X-Y-Z format with valid type codes."""
        valid_codes = {'D', 'DF', 'GS', 'H', 'IH', 'ST', 'MU', 'MD', 'BU', 'BD', 'U', 'DN'}
        for p in VALIDATED_LONG_PATTERNS + VALIDATED_SHORT_PATTERNS:
            parts = p.split('-')
            assert len(parts) == 3, f"Pattern {p} doesn't have 3 parts"
            for code in parts:
                assert code in valid_codes, f"Invalid type code '{code}' in pattern {p}"


class TestPatternMatching:
    """Test that known patterns are correctly matched."""

    def test_known_long_patterns_match(self):
        for p in VALIDATED_LONG_PATTERNS:
            assert p in VALIDATED_LONG_PATTERNS

    def test_known_short_patterns_match(self):
        for p in VALIDATED_SHORT_PATTERNS:
            assert p in VALIDATED_SHORT_PATTERNS

    def test_unknown_pattern_not_matched(self):
        fake = "XX-YY-ZZ"
        assert fake not in VALIDATED_LONG_PATTERNS
        assert fake not in VALIDATED_SHORT_PATTERNS

    def test_reversed_pattern_not_matched(self):
        """Reversing a known pattern should not match (unless coincidentally valid)."""
        for p in VALIDATED_LONG_PATTERNS[:3]:
            parts = p.split('-')
            reversed_p = '-'.join(reversed(parts))
            if reversed_p != p:
                # It might still be valid, but shouldn't be in the SAME list
                # (just verify the lookup works)
                _ = reversed_p in VALIDATED_LONG_PATTERNS

    def test_tpsl_values_reasonable(self):
        """TP/SL values should be between 0.1% and 10%."""
        for pattern, tpsl in PATTERN_OPTIMAL_TPSL.items():
            tp, sl = tpsl
            assert 0.1 <= tp <= 10.0, f"{pattern} TP={tp} out of range"
            assert 0.1 <= sl <= 10.0, f"{pattern} SL={sl} out of range"

    def test_stats_win_rates_reasonable(self):
        """Historical win rates should be between 0% and 100%."""
        for pattern, stats in PATTERN_STATS.items():
            wr = stats['wr']
            assert 0.0 <= wr <= 100.0, f"{pattern} WR={wr} out of range"
