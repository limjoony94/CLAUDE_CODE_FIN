"""Signal generation + exit logic tests.

Encodes critical-evaluation angles for signals.py. Covers all BUG fixes
that introduced behavior assertions (#53 channel sanity, #60 NaN/zero).
"""
import math
import pytest
from scripts.production.c1_breakout.signals import C1BreakoutSignal


# ── check_entry ────────────────────────────────────────────

class TestCheckEntry:
    """Channel breakout entry with body confirmation + SL clamp."""

    def test_no_signal_below_channel(self, default_strategy_config):
        """A. Edge: close inside channel → no signal."""
        sig = C1BreakoutSignal(default_strategy_config)
        # close=102, channel=[101, 103] → inside, no breakout
        r = sig.check_entry(101, 102.5, 100.5, 102, 103, 101, 1.0, 100, 103)
        assert r is None

    def test_long_breakout_with_strong_body(self, default_strategy_config):
        """Happy path: close > channel_high + body > 40% range → LONG."""
        sig = C1BreakoutSignal(default_strategy_config)
        # open=100, high=102, low=99.5, close=101.8 → body=1.8, range=2.5 → 72%
        r = sig.check_entry(100, 102, 99.5, 101.8, 101, 99, 1.0, 99.5, 101)
        assert r is not None
        assert r['direction'] == 'LONG'
        assert r['sl_price'] < r.get('sl_price', 0) + 1  # sl must be set

    def test_short_breakout_with_strong_body(self, default_strategy_config):
        """Happy path mirror: close < channel_low, negative body."""
        sig = C1BreakoutSignal(default_strategy_config)
        r = sig.check_entry(100, 100.5, 98, 98.2, 100, 99, 1.0, 98, 100.5)
        assert r is not None
        assert r['direction'] == 'SHORT'

    def test_body_filter_rejects_weak_bar(self, default_strategy_config):
        """A. Edge: body < 40% of range → rejected (doji/indecision)."""
        sig = C1BreakoutSignal(default_strategy_config)
        # open=100, close=100.2 (tiny body 0.2), range=3 → body=6% < 40%
        r = sig.check_entry(100, 102, 99, 100.2, 99.5, 98, 1.0, 99, 100)
        assert r is None

    def test_direction_must_match_body(self, default_strategy_config):
        """C. Bug interaction: LONG breakout but negative body → reject."""
        sig = C1BreakoutSignal(default_strategy_config)
        # close > channel_high (LONG breakout) but body is negative (close < open)
        r = sig.check_entry(102, 103, 99, 100, 99.5, 98, 1.0, 99, 100)
        # close=100 not > channel_high=99.5 … actually 100 > 99.5 → LONG
        # body = close - open = 100 - 102 = -2 → negative
        # Should reject because LONG needs positive body
        assert r is None

    def test_bug53_flat_channel_rejected(self, default_strategy_config):
        """D. Rollback safety: channel_high == channel_low (flat data) → reject.

        BUG#53 fix. Without this, division/comparison could produce spurious
        signals on data anomalies (e.g. exchange returning same value twice).
        """
        sig = C1BreakoutSignal(default_strategy_config)
        r = sig.check_entry(100, 100.5, 99.5, 100.2, 100.0, 100.0, 1.0, 99.5, 100.5)
        assert r is None

    def test_nan_atr_rejected(self, default_strategy_config):
        """A. Edge: NaN ATR (warmup) → reject."""
        sig = C1BreakoutSignal(default_strategy_config)
        r = sig.check_entry(100, 102, 99, 101.5, 101, 99, float('nan'), 99.5, 101)
        assert r is None

    def test_zero_range_rejected(self, default_strategy_config):
        """A. Edge: high == low (zero range) → division guard."""
        sig = C1BreakoutSignal(default_strategy_config)
        r = sig.check_entry(100, 100, 100, 100, 99, 99.5, 1.0, 99, 100.5)
        # high == low → no range → must reject
        assert r is None

    def test_sl_clamped_by_atr_cap(self, default_strategy_config):
        """B. Parity: SL distance bounded by max_sl_atr × ATR.

        Fractal SL can be far — the cap prevents extreme risk trades.
        """
        sig = C1BreakoutSignal(default_strategy_config)
        # Strong LONG breakout, swing low very far (50) — must be capped.
        # ATR=0.5, max_sl_atr=3.3 → atr_sl = 101.8 - 1.65 = 100.15
        # sl_pct = 1.62% (within [0.15, 3.0] bounds)
        r = sig.check_entry(100, 102, 99.5, 101.8, 101, 99, 0.5, 50, 105)
        assert r is not None, "cap should allow trade (sl_pct 1.62% in bounds)"
        # SL must come from ATR cap, NOT fractal (50 far below)
        assert r['sl_price'] > 99.5
        assert r['sl_price'] < 101

    def test_sl_too_narrow_rejected(self, default_strategy_config):
        """D. Rollback: sl_pct below sl_min_pct (0.15%) → reject trade.

        Too-tight SL causes noise-induced stop-outs.
        """
        sig = C1BreakoutSignal(default_strategy_config)
        # ATR=0.01 → cap very tight; swing also tight → sl_pct very small
        r = sig.check_entry(100, 100.05, 99.98, 100.02, 100.0, 99.99,
                            0.01, 99.99, 100.01)
        # Expected: either direction check fails or sl_min_pct filter
        assert r is None


# ── check_exit ─────────────────────────────────────────────

class TestCheckExit:
    """Exit priority: SL → Emergency → Timeout → Trail."""

    def test_sl_priority_over_emergency(self, default_strategy_config):
        """C. Bug interaction: SL and Emergency both triggered → SL wins.

        Realistic market order: price hits SL before reaching 3% emergency.
        """
        sig = C1BreakoutSignal(default_strategy_config)
        # LONG: low hits both sl_price (99) AND emergency level (97)
        r = sig.check_exit('LONG', 100, 100, 99.5, 96, 96.5, 99, 1.0, 5)
        assert r['reason'] == 'SL'
        assert r['exit_price'] == 99

    def test_emergency_triggers_without_sl(self, default_strategy_config):
        """A. Edge: gap move past SL → Emergency fires (SL didn't catch it)."""
        sig = C1BreakoutSignal(default_strategy_config)
        # LONG with sl=99 but bar has low=96 (far below SL and below emergency 97)
        # Wait, low=96 would hit SL first. For pure emergency: sl wasn't set or different
        # Simulate: sl=95 (below emergency), bar low=96 → SL not hit, emergency 3% triggers
        # Actually emergency_sl_pct=3 → emergency at entry*0.97 = 97. low=96 < 97 → emergency
        # But SL=95, low=96 > 95 → SL not hit
        r = sig.check_exit('LONG', 100, 100, 99, 96, 96.5, 95, 1.0, 5)
        assert r['reason'] == 'EMERGENCY'

    def test_timeout_after_max_hold(self, default_strategy_config):
        """A. Edge: bars_held >= max_hold_bars (192) → timeout exit at close."""
        sig = C1BreakoutSignal(default_strategy_config)
        r = sig.check_exit('LONG', 100, 105, 103, 101, 102, 95, 1.0, 192)
        assert r['reason'] == 'TIMEOUT'
        assert r['exit_price'] == 102

    def test_trail_tp_on_drawdown(self, default_strategy_config):
        """Happy path: best_pnl → drawdown ≥ trail_dist → TRAIL_TP exit."""
        sig = C1BreakoutSignal(default_strategy_config)
        # LONG entry=100, best=110 (+10%), cur_close=102 (+2% from entry)
        # drawdown = 10 - 2 = 8%, trail_dist = 2.5 × 1.0 / 102 × 100 = 2.45%
        # 8% > 2.45% → trigger
        r = sig.check_exit('LONG', 100, 110, 108, 102, 102, 90, 1.0, 5)
        assert r['reason'] == 'TRAIL_TP'

    def test_bug60_nan_close_no_crash(self, default_strategy_config):
        """D. Rollback: current_close = NaN → no ZeroDivisionError, hold.

        BUG#60. Bad candle data must not crash trail calculation.
        """
        sig = C1BreakoutSignal(default_strategy_config)
        r = sig.check_exit('LONG', 100, 110, 108, 102, float('nan'), 90, 1.0, 5)
        assert r is None

    def test_bug60_zero_close_no_crash(self, default_strategy_config):
        """D. Rollback: current_close = 0 → no crash, hold."""
        sig = C1BreakoutSignal(default_strategy_config)
        r = sig.check_exit('LONG', 100, 110, 108, 102, 0, 90, 1.0, 5)
        assert r is None

    def test_bug60_negative_close_no_crash(self, default_strategy_config):
        """D. Rollback: current_close < 0 (impossible but defensive) → hold."""
        sig = C1BreakoutSignal(default_strategy_config)
        r = sig.check_exit('LONG', 100, 110, 108, 102, -1, 90, 1.0, 5)
        assert r is None

    def test_short_sl_priority(self, default_strategy_config):
        """C. Bug interaction mirror: SHORT SL triggers on current_high >= sl."""
        sig = C1BreakoutSignal(default_strategy_config)
        r = sig.check_exit('SHORT', 100, 100, 101.5, 99, 101.2, 101, 1.0, 5)
        assert r['reason'] == 'SL'
        assert r['exit_price'] == 101

    def test_hold_when_no_condition_met(self, default_strategy_config):
        """Happy path: nothing triggers → return None."""
        sig = C1BreakoutSignal(default_strategy_config)
        r = sig.check_exit('LONG', 100, 100.5, 100.6, 99.9, 100.2, 95, 1.0, 5)
        assert r is None
