"""Indicator purity + causality tests.

Each test documents one critical-evaluation angle — if an indicator ever
violates these invariants, the strategy's MC/WF/Backtest validation is void.
"""
import math
import pytest
from scripts.production.c1_breakout.indicators import (
    compute_atr, compute_channel, compute_fractal_swings,
)


# ── ATR ────────────────────────────────────────────────────

class TestATR:
    """Wilder-smoothed ATR."""

    def test_warmup_returns_nan_before_period(self):
        """A. Edge: first `period` bars have no ATR yet."""
        h = [100 + i for i in range(20)]
        l = [99 + i for i in range(20)]
        c = [99.5 + i for i in range(20)]
        atr = compute_atr(h, l, c, period=14)
        # Bars 0..12 should be NaN
        for i in range(13):
            assert math.isnan(atr[i]), f"bar {i} should be NaN, got {atr[i]}"
        # Bar 13 (= period-1) is the first valid
        assert not math.isnan(atr[13])

    def test_flat_data_stable_atr(self):
        """C. Bug interaction: flat data with consistent range → TR = 1 → ATR = 1.

        (Uptrend data has gap-ups pushing TR = 2, so test uses flat prices.)
        """
        h = [100.0] * 20
        l = [99.0] * 20
        c = [99.5] * 20
        atr = compute_atr(h, l, c, period=14)
        assert abs(atr[-1] - 1.0) < 1e-9

    def test_insufficient_bars_all_nan(self):
        """A. Edge: fewer bars than period → all NaN."""
        atr = compute_atr([100, 101], [99, 100], [99.5, 100.5], period=14)
        assert all(math.isnan(x) for x in atr)

    def test_causality_no_future_leak(self):
        """B. Parity: ATR at bar N uses only bars 0..N.

        Modifying bar N+1 must not change ATR[N].
        """
        h = [100 + i for i in range(30)]
        l = [99 + i for i in range(30)]
        c = [99.5 + i for i in range(30)]
        atr_full = compute_atr(h, l, c, period=14)
        # Modify bar 20 forward
        h2 = h[:20] + [x * 10 for x in h[20:]]
        l2 = l[:20] + [x * 10 for x in l[20:]]
        c2 = c[:20] + [x * 10 for x in c[20:]]
        atr_mod = compute_atr(h2, l2, c2, period=14)
        # ATR[0..19] should match
        for i in range(20):
            if math.isnan(atr_full[i]) and math.isnan(atr_mod[i]):
                continue
            assert atr_full[i] == atr_mod[i], f"bar {i} leaked future data"


# ── Channel ────────────────────────────────────────────────

class TestChannel:
    """N-bar high/low channel (EXCLUDES current bar)."""

    def test_causality_excludes_current_bar(self):
        """B. Parity: channel at bar N = max/min of bars N-period..N-1.

        Current bar's high/low must NOT be in the window — that would be
        look-ahead bias.
        """
        h = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
             110, 111, 112, 113, 114, 115, 116, 117, 118, 119]
        l = [99]*20
        ch_h, ch_l = compute_channel(h, l, period=15)
        # At bar 15: high[0..14] = 100..114 → max = 114
        # Current bar 15 has high=115, which must NOT be in channel
        assert ch_h[15] == 114, f"expected 114 (excl current), got {ch_h[15]}"

    def test_warmup_nan(self):
        """A. Edge: first `period` bars have no channel."""
        h = list(range(10))
        l = list(range(10))
        ch_h, ch_l = compute_channel(h, l, period=15)
        assert all(math.isnan(x) for x in ch_h)
        assert all(math.isnan(x) for x in ch_l)

    def test_flat_data_channel_equals_value(self):
        """A. Edge: flat data → channel_high = channel_low = constant.

        C1BreakoutSignal (BUG#53) must reject this case to avoid spurious signals.
        """
        h = [100.0] * 20
        l = [100.0] * 20
        ch_h, ch_l = compute_channel(h, l, period=15)
        assert ch_h[15] == 100.0 and ch_l[15] == 100.0


# ── Fractal Swings ─────────────────────────────────────────

class TestFractalSwings:
    """Causal fractal swing detection — only past bars."""

    def test_no_future_lookahead(self):
        """B. Parity: swing at bar N must not change when future bars are modified.

        Causality invariant — this is the defining property of a production
        indicator. Any violation invalidates MC/WF/Backtest.
        """
        # Bar 10 hits new low (80) → marked as swing
        # Bar 11 (low=85) is NOT lowest → cur_sl preserved at 80
        lows = [100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 80, 85, 90]
        highs = [x + 2 for x in lows]
        sw_l, _ = compute_fractal_swings(highs, lows, lookback=10)
        baseline_10 = sw_l[10]
        baseline_11 = sw_l[11]
        assert baseline_10 == 80
        assert baseline_11 == 80  # swing preserved across bars

        # Modify future: replace bar 12 with extreme value
        lows_mod = lows[:12] + [1]
        highs_mod = highs[:12] + [3]
        sw_l_mod, _ = compute_fractal_swings(highs_mod, lows_mod, lookback=10)
        assert sw_l_mod[10] == baseline_10, "bar 10 leaked future"
        assert sw_l_mod[11] == baseline_11, "bar 11 leaked future"

    def test_current_bar_is_swing_when_lowest(self):
        """C. Bug interaction: if current bar IS lowest of lookback window,
        it IS a swing (eager detection).
        """
        # Bar 10 hits a new low
        lows = [100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 80]
        highs = [x + 2 for x in lows]
        sw_l, sw_h = compute_fractal_swings(highs, lows, lookback=10)
        assert sw_l[10] == 80  # current bar is the lowest

    def test_warmup_nan(self):
        """A. Edge: before lookback+1 bars, no swings available."""
        lows = list(range(5))
        highs = list(range(2, 7))
        sw_l, sw_h = compute_fractal_swings(highs, lows, lookback=10)
        assert all(math.isnan(x) for x in sw_l[:10])
        assert all(math.isnan(x) for x in sw_h[:10])

    def test_swing_high_detection(self):
        """C. Bug interaction mirror: swing high update (line 64 coverage)."""
        # Bar 10 hits new HIGH (120) — should be marked as swing high
        highs = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 120]
        lows = [x - 2 for x in highs]
        sw_l, sw_h = compute_fractal_swings(highs, lows, lookback=10)
        assert sw_h[10] == 120  # current bar is highest
