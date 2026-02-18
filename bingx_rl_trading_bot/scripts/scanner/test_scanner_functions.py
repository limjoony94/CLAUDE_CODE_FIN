"""Tests for scanner pure functions — bt_signals, mc_test, calc_stats, etc."""

import sys
import os

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from scripts.scanner.pattern_scanner import (
    bt_signals,
    build_signal_index,
    calc_stats,
    mc_test,
    portfolio_1pos,
    apply_multiple_testing_correction,
    FEE_PCT,
    LEVERAGE,
    MAX_BARS,
)


# ── build_signal_index ──────────────────────────────────────


class TestBuildSignalIndex:
    """Test build_signal_index() — pattern name to signal bar mapping."""

    def test_basic_3bar(self):
        """3 candles → 1 pattern at index 2."""
        types = ['U', 'DN', 'H']
        idx = build_signal_index(types, len(types))
        assert 'U-DN-H' in idx
        assert idx['U-DN-H'] == [2]

    def test_multiple_signals(self):
        """Repeating types → multiple entries for same pattern."""
        types = ['U', 'U', 'U', 'U']
        idx = build_signal_index(types, len(types))
        assert idx['U-U-U'] == [2, 3]

    def test_unique_patterns(self):
        """All different types → all unique patterns."""
        types = ['U', 'DN', 'H', 'BD', 'MU']
        idx = build_signal_index(types, len(types))
        assert len(idx) == 3  # U-DN-H, DN-H-BD, H-BD-MU

    def test_empty_insufficient(self):
        """Less than 3 candles → empty index."""
        assert build_signal_index(['U', 'DN'], 2) == {}
        assert build_signal_index(['U'], 1) == {}
        assert build_signal_index([], 0) == {}


# ── bt_signals ──────────────────────────────────────────────


class TestBtSignals:
    """Test bt_signals() — backtesting with TP/SL."""

    @pytest.fixture
    def price_data(self):
        """10-bar price data for testing.

        Bar 0: signal bar
        Bar 1: entry at 50000 (next bar open)
        Bar 2-9: price movement bars
        """
        n = 10
        opens = np.full(n, 50000.0)
        highs = np.full(n, 50100.0)
        lows = np.full(n, 49900.0)
        return opens, highs, lows, n

    def test_long_tp_hit(self, price_data):
        """LONG: high reaches TP → win."""
        opens, highs, lows, n = price_data
        # Bar 3 high spikes above TP (2% of 50000 = 51000)
        highs[3] = 51100.0
        trades = bt_signals([0], 'LONG', 2.0, 3.0, opens, highs, lows, n)
        assert len(trades) == 1
        eb, xb, pnl = trades[0]
        assert eb == 1  # entry bar
        assert xb == 3  # exit bar
        fee = FEE_PCT * LEVERAGE
        assert pnl == pytest.approx(2.0 * LEVERAGE - fee, abs=0.01)

    def test_short_tp_hit(self, price_data):
        """SHORT: low reaches TP → win."""
        opens, highs, lows, n = price_data
        # Bar 4 low drops below TP (2% below 50000 = 49000)
        lows[4] = 48900.0
        trades = bt_signals([0], 'SHORT', 2.0, 3.0, opens, highs, lows, n)
        assert len(trades) == 1
        _, _, pnl = trades[0]
        fee = FEE_PCT * LEVERAGE
        assert pnl == pytest.approx(2.0 * LEVERAGE - fee, abs=0.01)

    def test_long_sl_hit(self, price_data):
        """LONG: low reaches SL → loss."""
        opens, highs, lows, n = price_data
        # Bar 2 low drops below SL (3% below 50000 = 48500)
        lows[2] = 48400.0
        trades = bt_signals([0], 'LONG', 2.0, 3.0, opens, highs, lows, n)
        assert len(trades) == 1
        _, _, pnl = trades[0]
        fee = FEE_PCT * LEVERAGE
        assert pnl == pytest.approx(-3.0 * LEVERAGE - fee, abs=0.01)

    def test_short_sl_hit(self, price_data):
        """SHORT: high reaches SL → loss."""
        opens, highs, lows, n = price_data
        # Bar 2 high spikes above SL (3% above 50000 = 51500)
        highs[2] = 51600.0
        trades = bt_signals([0], 'SHORT', 2.0, 3.0, opens, highs, lows, n)
        assert len(trades) == 1
        _, _, pnl = trades[0]
        fee = FEE_PCT * LEVERAGE
        assert pnl == pytest.approx(-3.0 * LEVERAGE - fee, abs=0.01)

    def test_timeout_dropped(self):
        """Trade that never hits TP or SL → dropped (not counted)."""
        n = 5
        opens = np.full(n, 50000.0)
        highs = np.full(n, 50010.0)  # barely moves
        lows = np.full(n, 49990.0)
        trades = bt_signals([0], 'LONG', 2.0, 3.0, opens, highs, lows, n)
        assert len(trades) == 0

    def test_signal_at_end_skipped(self):
        """Signal at last bar → no entry possible, skipped."""
        n = 5
        opens = np.full(n, 50000.0)
        highs = np.full(n, 51000.0)
        lows = np.full(n, 49000.0)
        trades = bt_signals([4], 'LONG', 2.0, 3.0, opens, highs, lows, n)
        assert len(trades) == 0

    def test_zero_entry_skipped(self):
        """Entry bar with open=0 → skipped."""
        n = 5
        opens = np.array([50000.0, 0.0, 50000.0, 51000.0, 50000.0])
        highs = np.full(n, 51000.0)
        lows = np.full(n, 49000.0)
        trades = bt_signals([0], 'LONG', 2.0, 3.0, opens, highs, lows, n)
        assert len(trades) == 0

    def test_same_bar_tp_sl_distance_resolution(self):
        """Both TP and SL hit same bar → resolve by bar open distance."""
        n = 5
        entry = 50000.0
        tp_pct = 2.0  # TP at 51000
        sl_pct = 3.0  # SL at 48500
        opens = np.array([50000.0, entry, 50000.0, 50800.0, 50000.0])
        highs = np.array([50000.0, entry, 50000.0, 52000.0, 50000.0])  # hits TP
        lows = np.array([50000.0, entry, 50000.0, 48000.0, 50000.0])   # hits SL too
        # Bar 3: open=50800, TP=51000 dist=200, SL=48500 dist=2300 → TP closer → win
        trades = bt_signals([0], 'LONG', tp_pct, sl_pct, opens, highs, lows, n)
        assert len(trades) == 1
        _, _, pnl = trades[0]
        fee = FEE_PCT * LEVERAGE
        assert pnl == pytest.approx(tp_pct * LEVERAGE - fee, abs=0.01)

    def test_fee_matches_production(self):
        """Scanner fee = FEE_PCT * LEVERAGE must match production."""
        from scripts.production.pattern_5m.position_close import calculate_pnl

        # TP hit: entry=50000, exit=51000 (2% up)
        n = 5
        entry = 50000.0
        tp_pct = 2.0
        sl_pct = 3.0
        exit_price = entry * (1 + tp_pct / 100)  # 51000

        # Bar 2 must be neutral (between SL and TP)
        opens = np.array([entry, entry, entry, exit_price, entry], dtype=float)
        highs = np.array([entry, entry, entry, exit_price + 100, entry], dtype=float)
        lows = np.array([entry, entry, entry, exit_price - 100, entry], dtype=float)

        trades = bt_signals([0], 'LONG', tp_pct, sl_pct, opens, highs, lows, n)
        assert len(trades) == 1
        scanner_pnl = trades[0][2]

        prod_pnl, _ = calculate_pnl(entry, exit_price, direction=1, leverage=LEVERAGE)
        assert scanner_pnl == pytest.approx(prod_pnl, abs=0.05)

    def test_multiple_signals(self):
        """Multiple signal bars → multiple trades."""
        n = 20
        opens = np.full(n, 50000.0)
        highs = np.full(n, 51100.0)  # always hits 2% TP for LONG
        lows = np.full(n, 49900.0)
        # Signals far apart to avoid overlap
        trades = bt_signals([0, 10], 'LONG', 2.0, 3.0, opens, highs, lows, n)
        assert len(trades) == 2


# ── mc_test ─────────────────────────────────────────────────


class TestMcTest:
    """Test mc_test() — Monte Carlo sign randomization."""

    def test_strong_signal_low_p(self):
        """All positive PnLs → p-value should be very low."""
        pnls = [5.0] * 50  # All wins
        p = mc_test(pnls)
        assert p < 0.01

    def test_random_signal_high_p(self):
        """50/50 win/loss → p-value should be high."""
        pnls = [5.0, -5.0] * 25
        p = mc_test(pnls)
        assert p > 0.1

    def test_too_few_trades(self):
        """Less than 5 trades → returns 1.0."""
        assert mc_test([1.0, 2.0, 3.0]) == 1.0
        assert mc_test([]) == 1.0

    def test_p_value_range(self):
        """P-value should always be between 0 and 1."""
        pnls = [3.0, -1.0, 2.0, -2.0, 5.0, -0.5, 4.0, -3.0, 1.0, -1.0]
        p = mc_test(pnls)
        assert 0.0 <= p <= 1.0

    def test_multi_seed_conservative(self):
        """MC uses max(p_values) across seeds — most conservative."""
        # All negative → p should be ~1.0 (sum is never exceeded by random)
        pnls = [-5.0] * 20
        p = mc_test(pnls)
        assert p > 0.9


# ── calc_stats ──────────────────────────────────────────────


class TestCalcStats:
    """Test calc_stats() — portfolio statistics."""

    def test_empty_trades(self):
        """Empty trades → zero stats."""
        stats = calc_stats([])
        assert stats['pnl'] == 0
        assert stats['trades'] == 0
        assert stats['wr'] == 0
        assert stats['mdd'] == 0

    def test_all_wins(self):
        """All winning trades → WR 100%, no drawdown."""
        trades = [(0, 1, 5.0), (2, 3, 3.0), (4, 5, 7.0)]
        stats = calc_stats(trades)
        assert stats['trades'] == 3
        assert stats['wr'] == 100.0
        assert stats['pnl'] == 15.0
        assert stats['mdd'] == 0

    def test_all_losses(self):
        """All losing trades → WR 0%, MDD equals total loss."""
        trades = [(0, 1, -5.0), (2, 3, -3.0)]
        stats = calc_stats(trades)
        assert stats['wr'] == 0.0
        assert stats['pnl'] == -8.0
        assert stats['mdd'] == 8.0
        assert stats['pf'] == 0

    def test_mixed_trades(self):
        """Mixed wins/losses → correct WR, MDD, PF."""
        trades = [(0, 1, 10.0), (2, 3, -5.0), (4, 5, 8.0)]
        stats = calc_stats(trades)
        assert stats['trades'] == 3
        assert stats['wr'] == pytest.approx(66.7, abs=0.1)
        assert stats['pnl'] == 13.0
        assert stats['mdd'] == 5.0
        assert stats['pf'] == pytest.approx(18.0 / 5.0, abs=0.01)

    def test_drawdown_calculation(self):
        """MDD should be the largest peak-to-trough decline."""
        # Cumulative: 10, 5, 12, 2
        # Peaks: 10, 10, 12, 12
        # DD: 0, 5, 0, 10 → MDD = 10
        trades = [(0, 1, 10.0), (2, 3, -5.0), (4, 5, 7.0), (6, 7, -10.0)]
        stats = calc_stats(trades)
        assert stats['mdd'] == 10.0


# ── portfolio_1pos ──────────────────────────────────────────


class TestPortfolio1Pos:
    """Test portfolio_1pos() — 1-position-at-a-time filter."""

    def test_empty(self):
        """Empty input → empty output."""
        assert portfolio_1pos([]) == []

    def test_no_overlap(self):
        """Non-overlapping trades → all kept."""
        trades = [(0, 5, 3.0), (6, 10, -2.0), (11, 15, 5.0)]
        result = portfolio_1pos(trades)
        assert len(result) == 3

    def test_overlapping_trades(self):
        """Overlapping trades → later ones filtered out."""
        trades = [(0, 10, 3.0), (5, 15, -2.0), (11, 20, 5.0)]
        result = portfolio_1pos(trades)
        assert len(result) == 2
        assert result[0] == (0, 10, 3.0)
        assert result[1] == (11, 20, 5.0)

    def test_unsorted_input(self):
        """Unsorted input → sorted by entry bar first."""
        trades = [(10, 15, 2.0), (0, 5, 3.0), (6, 9, -1.0)]
        result = portfolio_1pos(trades)
        assert len(result) == 3
        assert result[0][0] == 0  # sorted by entry

    def test_adjacent_trades(self):
        """Trade ending at bar N, next starting at bar N → overlap (eb > last_exit)."""
        trades = [(0, 5, 3.0), (5, 10, -2.0), (6, 15, 5.0)]
        result = portfolio_1pos(trades)
        # (5, 10) starts at 5 which is NOT > 5 (last_exit), so filtered
        assert len(result) == 2
        assert result[0] == (0, 5, 3.0)
        assert result[1] == (6, 15, 5.0)


# ── apply_multiple_testing_correction ───────────────────────


class TestMultipleTestingCorrection:
    """Test apply_multiple_testing_correction()."""

    @pytest.fixture
    def sample_selected(self):
        """Sample selected patterns with varying p-values."""
        return {
            'A_LONG': {'pattern': 'A', 'direction': 'LONG', 'mc_p': 0.001},
            'B_SHORT': {'pattern': 'B', 'direction': 'SHORT', 'mc_p': 0.005},
            'C_LONG': {'pattern': 'C', 'direction': 'LONG', 'mc_p': 0.008},
        }

    def test_none_correction(self, sample_selected):
        """No correction → all patterns pass."""
        filtered, meta = apply_multiple_testing_correction(
            sample_selected, n_tested=100, method='none'
        )
        assert len(filtered) == 3
        assert meta['correction_method'] == 'none'

    def test_bonferroni_strict(self, sample_selected):
        """Bonferroni with many tests → strict threshold."""
        filtered, meta = apply_multiple_testing_correction(
            sample_selected, n_tested=1000, method='bonferroni', alpha=0.01
        )
        # Threshold = 0.01/1000 = 0.00001 → none pass
        assert len(filtered) == 0
        assert meta['bonf_threshold'] == pytest.approx(0.00001)

    def test_bonferroni_lenient(self, sample_selected):
        """Bonferroni with few tests → lenient threshold."""
        filtered, meta = apply_multiple_testing_correction(
            sample_selected, n_tested=3, method='bonferroni', alpha=0.01
        )
        # Threshold = 0.01/3 ≈ 0.0033 → A (0.001) passes
        assert len(filtered) >= 1
        assert 'A_LONG' in filtered

    def test_bh_fdr(self, sample_selected):
        """BH FDR correction."""
        filtered, meta = apply_multiple_testing_correction(
            sample_selected, n_tested=100, method='bh', fdr_q=0.05
        )
        assert meta['correction_method'] == 'bh'
        assert meta['fdr_q'] == 0.05
        # All have very low p-values, should pass BH
        assert len(filtered) >= 1

    def test_empty_input(self):
        """Empty selected → empty output."""
        filtered, meta = apply_multiple_testing_correction(
            {}, n_tested=100, method='bh'
        )
        assert len(filtered) == 0

    def test_unknown_method_raises(self, sample_selected):
        """Unknown correction method → ValueError."""
        with pytest.raises(ValueError, match="Unknown correction method"):
            apply_multiple_testing_correction(
                sample_selected, n_tested=100, method='invalid'
            )
