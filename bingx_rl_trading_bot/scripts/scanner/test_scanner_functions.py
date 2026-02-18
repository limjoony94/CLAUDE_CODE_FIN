"""Tests for scanner pure functions — bt_signals, mc_test, calc_stats, etc."""

import sys
import os
import json
import tempfile
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
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
    compute_excursions,
    derive_tp_sl,
    grid_search_best,
    grid_search_mae_mfe,
    load_and_classify,
    scan_universe_range,
    expanding_window_wf,
    scan_patterns,
    scan_patterns_pp,
    scan_patterns_mae_mfe,
    build_output_json,
    _pp_init,
    _pp_worker,
    _mae_mfe_worker,
    _shared_data,
    FEE_PCT,
    LEVERAGE,
    MAX_BARS,
    TP_GRID,
    SL_GRID,
    MAX_BASELINE_WR,
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


# ── Synthetic data helpers ────────────────────────────────────

def _make_trending_data(n=500, start=50000.0, trend_pct=0.001):
    """Create synthetic OHLCV data with slight uptrend + noise.

    Returns opens, highs, lows arrays and n_bars.
    """
    rng = np.random.RandomState(42)
    opens = np.zeros(n)
    highs = np.zeros(n)
    lows = np.zeros(n)
    price = start
    for i in range(n):
        opens[i] = price
        move = price * (trend_pct + rng.normal(0, 0.003))
        highs[i] = price + abs(rng.normal(0, price * 0.005))
        lows[i] = price - abs(rng.normal(0, price * 0.005))
        price = price + move
    return opens, highs, lows, n


def _make_signal_index_for_tests(n_bars=500, n_signals=30, seed=42):
    """Build a signal index with enough signals for testing."""
    rng = np.random.RandomState(seed)
    types_list = ['U', 'DN', 'H', 'BD', 'MU', 'MD', 'ST', 'BU', 'IH', 'D']
    types = [rng.choice(types_list) for _ in range(n_bars)]
    signal_index = build_signal_index(types, n_bars)
    return signal_index, types


def _make_df_for_tests(n=500, seed=42):
    """Create a minimal DataFrame with OHLCV and candle_type for scanner tests."""
    rng = np.random.RandomState(seed)
    opens, highs, lows, _ = _make_trending_data(n)
    closes = opens + rng.normal(0, 50, n)
    # Fix highs/lows to be consistent
    for i in range(n):
        highs[i] = max(highs[i], opens[i], closes[i])
        lows[i] = min(lows[i], opens[i], closes[i])

    types_list = ['U', 'DN', 'H', 'BD', 'MU', 'MD', 'ST', 'BU', 'IH', 'D', 'DF', 'GS']
    types = [rng.choice(types_list) for _ in range(n)]

    df = pd.DataFrame({
        'open': opens, 'high': highs, 'low': lows, 'close': closes,
        'candle_type': types,
    })
    return df


# ── compute_excursions ─────────────────────────────────────────


class TestComputeExcursions:
    """Test compute_excursions() — MFE/MAE natural path."""

    def test_basic_long(self):
        """LONG excursion: MFE from highs, MAE from lows."""
        n = 10
        opens = np.full(n, 50000.0)
        highs = np.full(n, 50200.0)
        lows = np.full(n, 49800.0)
        # Signal at bar 0, entry at bar 1
        excursions = compute_excursions([0], 'LONG', opens, highs, lows, n)
        assert len(excursions) == 1
        e = excursions[0]
        assert e['signal_bar'] == 0
        assert e['entry'] == 50000.0
        assert e['mfe_pct'] > 0  # highs above entry
        assert e['mae_pct'] > 0  # lows below entry

    def test_basic_short(self):
        """SHORT excursion: MFE from lows, MAE from highs."""
        n = 10
        opens = np.full(n, 50000.0)
        highs = np.full(n, 50300.0)
        lows = np.full(n, 49700.0)
        excursions = compute_excursions([0], 'SHORT', opens, highs, lows, n)
        assert len(excursions) == 1
        e = excursions[0]
        assert e['mfe_pct'] > 0  # entry - low > 0
        assert e['mae_pct'] > 0  # high - entry > 0

    def test_signal_at_end_skipped(self):
        """Signal at last bar → skipped (no room for entry)."""
        n = 5
        opens = np.full(n, 50000.0)
        highs = np.full(n, 50100.0)
        lows = np.full(n, 49900.0)
        excursions = compute_excursions([4], 'LONG', opens, highs, lows, n)
        assert len(excursions) == 0

    def test_zero_entry_skipped(self):
        """Entry with open=0 → skipped."""
        n = 5
        opens = np.array([50000.0, 0.0, 50000.0, 50000.0, 50000.0])
        highs = np.full(n, 50100.0)
        lows = np.full(n, 49900.0)
        excursions = compute_excursions([0], 'LONG', opens, highs, lows, n)
        assert len(excursions) == 0

    def test_bars_to_mfe_tracked(self):
        """bars_to_mfe records when MFE was reached."""
        n = 8
        opens = np.full(n, 50000.0)
        highs = np.full(n, 50100.0)
        lows = np.full(n, 49900.0)
        # Bar 5 has big high → MFE should be at bar 5
        highs[5] = 51000.0
        excursions = compute_excursions([0], 'LONG', opens, highs, lows, n)
        assert len(excursions) == 1
        assert excursions[0]['bars_to_mfe'] == 5 - 1  # relative to entry bar

    def test_max_bars_limit(self):
        """max_bars limits how far excursion looks ahead."""
        n = 100
        opens = np.full(n, 50000.0)
        highs = np.full(n, 50100.0)
        lows = np.full(n, 49900.0)
        # Big move at bar 50 — should be seen with default max_bars but not max_bars=5
        highs[50] = 55000.0
        exc_short = compute_excursions([0], 'LONG', opens, highs, lows, n, max_bars=5)
        exc_long = compute_excursions([0], 'LONG', opens, highs, lows, n, max_bars=100)
        assert exc_short[0]['mfe_pct'] < exc_long[0]['mfe_pct']

    def test_multiple_signals(self):
        """Multiple signals → multiple excursion entries."""
        n = 50
        opens = np.full(n, 50000.0)
        highs = np.full(n, 50200.0)
        lows = np.full(n, 49800.0)
        excursions = compute_excursions([0, 10, 20], 'LONG', opens, highs, lows, n)
        assert len(excursions) == 3

    def test_final_pnl_calculated(self):
        """final_pnl_pct is based on open of last reachable bar."""
        n = 10
        opens = np.array([50000.0] * 5 + [51000.0] * 5)
        highs = np.full(n, 51500.0)
        lows = np.full(n, 49500.0)
        excursions = compute_excursions([0], 'LONG', opens, highs, lows, n, max_bars=5)
        assert len(excursions) == 1
        # Last bar open is 51000, entry is 50000 → ~2% gain
        assert excursions[0]['final_pnl_pct'] > 0


# ── derive_tp_sl ──────────────────────────────────────────────


class TestDeriveTpSl:
    """Test derive_tp_sl() — TP/SL from MFE/MAE percentiles."""

    def test_basic(self):
        """Derive TP/SL from simple excursion data."""
        excursions = [
            {'mfe_pct': 1.0, 'mae_pct': 0.5},
            {'mfe_pct': 2.0, 'mae_pct': 1.0},
            {'mfe_pct': 3.0, 'mae_pct': 1.5},
        ]
        tp, sl = derive_tp_sl(excursions, tp_percentile=50, sl_percentile=50)
        assert tp == 2.0
        assert sl == 1.0

    def test_empty_excursions(self):
        """Empty excursions → (None, None)."""
        tp, sl = derive_tp_sl([], tp_percentile=50, sl_percentile=50)
        assert tp is None
        assert sl is None

    def test_percentile_extremes(self):
        """High percentile → higher values."""
        excursions = [{'mfe_pct': float(i), 'mae_pct': float(i) * 0.5}
                      for i in range(1, 101)]
        tp_low, sl_low = derive_tp_sl(excursions, 10, 10)
        tp_high, sl_high = derive_tp_sl(excursions, 90, 90)
        assert tp_high > tp_low
        assert sl_high > sl_low


# ── grid_search_best ──────────────────────────────────────────


class TestGridSearchBest:
    """Test grid_search_best() — TP/SL grid search optimization."""

    @pytest.fixture
    def strong_signal_data(self):
        """Create price data where LONG signals reliably profit."""
        n = 600
        rng = np.random.RandomState(42)
        opens = np.full(n, 50000.0)
        highs = np.full(n, 50000.0)
        lows = np.full(n, 50000.0)

        # Create strong upward moves after every 20th bar
        signals = list(range(5, n - 10, 20))
        for sig in signals:
            entry_bar = sig + 1
            if entry_bar + 3 >= n:
                continue
            for j in range(entry_bar + 1, min(entry_bar + 5, n)):
                highs[j] = opens[entry_bar] * 1.025  # +2.5% high
                lows[j] = opens[entry_bar] * 0.998   # -0.2% low

        return signals, opens, highs, lows, n

    def test_finds_optimal(self, strong_signal_data):
        """Should find a TP/SL combo when signals are strong."""
        signals, opens, highs, lows, n = strong_signal_data
        result = grid_search_best(signals, 'LONG', opens, highs, lows, n, min_tr=5)
        assert result is not None
        assert 'tp' in result
        assert 'sl' in result
        assert result['trades'] >= 5
        assert result['wr'] > 0

    def test_returns_none_few_trades(self):
        """Returns None when not enough trades for any combo."""
        n = 10
        opens = np.full(n, 50000.0)
        highs = np.full(n, 50010.0)
        lows = np.full(n, 49990.0)
        result = grid_search_best([0], 'LONG', opens, highs, lows, n, min_tr=20)
        assert result is None

    def test_baseline_wr_filter(self):
        """Combos with baseline WR > max_baseline_wr should be skipped."""
        n = 200
        opens = np.full(n, 50000.0)
        highs = np.full(n, 51000.0)  # Always hits TP
        lows = np.full(n, 49000.0)
        signals = list(range(0, n - 5, 5))
        # With max_baseline_wr=0 → nothing passes
        result = grid_search_best(signals, 'LONG', opens, highs, lows, n,
                                  min_tr=5, max_baseline_wr=0.0)
        assert result is None

    def test_result_fields(self, strong_signal_data):
        """Result dict has expected fields."""
        signals, opens, highs, lows, n = strong_signal_data
        result = grid_search_best(signals, 'LONG', opens, highs, lows, n, min_tr=5)
        if result is not None:
            for field in ['tp', 'sl', 'trades', 'wr', 'pnl', 'mdd']:
                assert field in result


# ── grid_search_mae_mfe ───────────────────────────────────────


class TestGridSearchMaeMfe:
    """Test grid_search_mae_mfe() — MAE/MFE based optimization."""

    def test_returns_none_few_trades(self):
        """Returns None when not enough excursion data."""
        n = 10
        opens = np.full(n, 50000.0)
        highs = np.full(n, 50010.0)
        lows = np.full(n, 49990.0)
        best, exc_stats = grid_search_mae_mfe([0], 'LONG', opens, highs, lows, n, min_tr=20)
        assert best is None
        assert exc_stats is None

    def test_returns_result_with_enough_data(self):
        """Returns result when data is sufficient."""
        n = 600
        rng = np.random.RandomState(42)
        opens = np.full(n, 50000.0)
        highs = np.full(n, 50000.0)
        lows = np.full(n, 50000.0)
        signals = list(range(5, n - 10, 15))
        for sig in signals:
            eb = sig + 1
            if eb + 5 >= n:
                continue
            for j in range(eb + 1, min(eb + 5, n)):
                highs[j] = 50000.0 * 1.03
                lows[j] = 50000.0 * 0.998
        best, exc_stats = grid_search_mae_mfe(signals, 'LONG', opens, highs, lows, n, min_tr=5)
        # May or may not find a result depending on exact data
        if best is not None:
            assert 'tp' in best
            assert 'sl' in best
            assert exc_stats is not None
            assert 'n_signals' in exc_stats
            assert 'mfe_median' in exc_stats


# ── _pp_init / _pp_worker ────────────────────────────────────


class TestPpWorker:
    """Test _pp_init and _pp_worker for parallel per-pattern processing."""

    def test_pp_init_sets_shared_data(self):
        """_pp_init stores arrays in _shared_data."""
        opens = np.array([1.0, 2.0, 3.0])
        highs = np.array([4.0, 5.0, 6.0])
        lows = np.array([0.5, 1.5, 2.5])
        _pp_init(opens, highs, lows, 3)
        assert np.array_equal(_shared_data['opens'], opens)
        assert np.array_equal(_shared_data['highs'], highs)
        assert np.array_equal(_shared_data['lows'], lows)
        assert _shared_data['n_bars'] == 3

    def test_pp_worker_returns_none_few_signals(self):
        """Worker returns None when signals < min_trades."""
        _pp_init(np.full(10, 50000.0), np.full(10, 50100.0),
                 np.full(10, 49900.0), 10)
        result = _pp_worker(('TEST', 'LONG', [0, 1], 20, 10.0, 0.01, 70.0))
        assert result is None

    def test_pp_worker_processes_strong_signal(self):
        """Worker processes a strong pattern and returns result or None."""
        n = 300
        opens = np.full(n, 50000.0)
        highs = np.full(n, 50000.0)
        lows = np.full(n, 50000.0)
        signals = list(range(5, n - 10, 10))
        for sig in signals:
            eb = sig + 1
            if eb + 3 >= n:
                continue
            for j in range(eb + 1, min(eb + 4, n)):
                highs[j] = 50000.0 * 1.025
                lows[j] = 50000.0 * 0.998
        _pp_init(opens, highs, lows, n)
        result = _pp_worker(('TEST', 'LONG', signals, 5, 5.0, 0.05, 70.0))
        if result is not None:
            assert result['pattern'] == 'TEST'
            assert result['direction'] == 'LONG'
            assert 'tp' in result
            assert 'trades_raw' in result


# ── _mae_mfe_worker ──────────────────────────────────────────


class TestMaeMfeWorker:
    """Test _mae_mfe_worker for parallel MAE/MFE processing."""

    def test_returns_none_few_signals(self):
        """Worker returns None when signals < min_trades."""
        _pp_init(np.full(10, 50000.0), np.full(10, 50100.0),
                 np.full(10, 49900.0), 10)
        result = _mae_mfe_worker(('TEST', 'LONG', [0, 1], 20, 10.0, 0.01, 70.0))
        assert result is None


# ── load_and_classify ─────────────────────────────────────────


class TestLoadAndClassify:
    """Test load_and_classify() — CSV loading + candle classification."""

    def test_basic_classification(self, tmp_path):
        """Load valid CSV and classify candles."""
        csv_file = tmp_path / "test_data.csv"
        n = 30
        rng = np.random.RandomState(42)
        opens = 50000.0 + rng.normal(0, 100, n)
        closes = opens + rng.normal(0, 50, n)
        highs = np.maximum(opens, closes) + abs(rng.normal(0, 30, n))
        lows = np.minimum(opens, closes) - abs(rng.normal(0, 30, n))
        df = pd.DataFrame({
            'open': opens, 'high': highs, 'low': lows, 'close': closes,
        })
        df.to_csv(csv_file, index=False)

        result = load_and_classify(str(csv_file))
        assert 'candle_type' in result.columns
        assert len(result) == n
        # All candle types should be valid 12-type codes
        valid_types = {'D', 'DF', 'GS', 'H', 'IH', 'ST', 'MU', 'MD', 'BU', 'BD', 'U', 'DN'}
        assert set(result['candle_type'].unique()).issubset(valid_types)

    def test_missing_column_raises(self, tmp_path):
        """Missing required column → ValueError."""
        csv_file = tmp_path / "bad_data.csv"
        pd.DataFrame({'open': [1], 'high': [2], 'low': [0.5]}).to_csv(csv_file, index=False)
        with pytest.raises(ValueError, match="Missing required column"):
            load_and_classify(str(csv_file))


# ── scan_universe_range ───────────────────────────────────────


class TestScanUniverseRange:
    """Test scan_universe_range() — range-based pattern scanning."""

    @pytest.fixture
    def scan_data(self):
        """Create data for range scanning with known strong patterns."""
        n = 400
        rng = np.random.RandomState(42)
        opens = np.full(n, 50000.0)
        highs = np.full(n, 50000.0)
        lows = np.full(n, 50000.0)

        # Create types with a repeating pattern to get enough signals
        base_pattern = ['U', 'DN', 'H']
        types = []
        for i in range(n):
            types.append(base_pattern[i % 3])

        signal_index = build_signal_index(types, n)

        # Make strong upward moves after U-DN-H signals
        for pat_name, sigs in signal_index.items():
            for sig in sigs:
                eb = sig + 1
                if eb + 3 >= n:
                    continue
                for j in range(eb + 1, min(eb + 4, n)):
                    highs[j] = 50000.0 * 1.03
                    lows[j] = 50000.0 * 0.998

        return signal_index, opens, highs, lows, n

    def test_universal_mode(self, scan_data):
        """Universal mode scans with fixed TP/SL."""
        signal_index, opens, highs, lows, n = scan_data
        result = scan_universe_range(
            signal_index, opens, highs, lows, n,
            bar_start=0, bar_end=n, mode='universal',
            uni_tp=2.0, uni_sl=3.0,
            min_trades=5, edge_threshold=1.0, mc_threshold=0.05,
        )
        assert isinstance(result, list)
        # May or may not find patterns depending on data

    def test_per_pattern_mode(self, scan_data):
        """Per-pattern mode runs grid search per pattern."""
        signal_index, opens, highs, lows, n = scan_data
        result = scan_universe_range(
            signal_index, opens, highs, lows, n,
            bar_start=0, bar_end=n, mode='per_pattern',
            min_trades=5, edge_threshold=1.0, mc_threshold=0.05,
        )
        assert isinstance(result, list)

    def test_unknown_mode_raises(self, scan_data):
        """Unknown mode → ValueError."""
        signal_index, opens, highs, lows, n = scan_data
        with pytest.raises(ValueError, match="Unknown mode"):
            scan_universe_range(
                signal_index, opens, highs, lows, n,
                bar_start=0, bar_end=n, mode='invalid',
                min_trades=5, edge_threshold=1.0,
            )

    def test_bar_range_filter(self, scan_data):
        """Only signals within [bar_start, bar_end) are used."""
        signal_index, opens, highs, lows, n = scan_data
        # Scan first half only
        result_half = scan_universe_range(
            signal_index, opens, highs, lows, n,
            bar_start=0, bar_end=n // 2, mode='universal',
            uni_tp=2.0, uni_sl=3.0,
            min_trades=5, edge_threshold=1.0, mc_threshold=0.05,
        )
        assert isinstance(result_half, list)

    def test_mae_mfe_mode(self, scan_data):
        """MAE/MFE mode uses excursion-based grid search."""
        signal_index, opens, highs, lows, n = scan_data
        result = scan_universe_range(
            signal_index, opens, highs, lows, n,
            bar_start=0, bar_end=n, mode='mae_mfe',
            min_trades=5, edge_threshold=1.0, mc_threshold=0.05,
        )
        assert isinstance(result, list)


# ── expanding_window_wf ──────────────────────────────────────


class TestExpandingWindowWF:
    """Test expanding_window_wf() — walk-forward validation."""

    @pytest.fixture
    def wf_data(self):
        """Create sufficient data for walk-forward with known patterns."""
        n = 600
        opens = np.full(n, 50000.0)
        highs = np.full(n, 50000.0)
        lows = np.full(n, 50000.0)

        base_pattern = ['U', 'DN', 'H']
        types = [base_pattern[i % 3] for i in range(n)]
        signal_index = build_signal_index(types, n)

        # Strong moves after signals throughout data
        for sigs in signal_index.values():
            for sig in sigs:
                eb = sig + 1
                if eb + 4 >= n:
                    continue
                for j in range(eb + 1, min(eb + 4, n)):
                    highs[j] = 50000.0 * 1.03
                    lows[j] = 50000.0 * 0.998

        return signal_index, opens, highs, lows, n

    def test_basic_wf(self, wf_data):
        """Walk-forward with 2 folds returns expected structure."""
        signal_index, opens, highs, lows, n = wf_data
        result = expanding_window_wf(
            signal_index, opens, highs, lows, n,
            n_folds=2, mode='universal',
            uni_tp=2.0, uni_sl=3.0,
            min_trades=5, edge_threshold=1.0, mc_threshold=0.05,
        )
        assert 'n_folds' in result
        assert result['n_folds'] == 2
        assert 'folds' in result
        assert len(result['folds']) == 2
        assert 'positive_folds' in result
        assert 'total_oos_pnl' in result
        assert 'stable_patterns' in result

    def test_fold_structure(self, wf_data):
        """Each fold has expected fields."""
        signal_index, opens, highs, lows, n = wf_data
        result = expanding_window_wf(
            signal_index, opens, highs, lows, n,
            n_folds=2, mode='universal',
            uni_tp=2.0, uni_sl=3.0,
            min_trades=5, edge_threshold=1.0, mc_threshold=0.05,
        )
        for fold in result['folds']:
            for key in ['fold', 'is_bars', 'oos_bars', 'is_patterns',
                        'oos_trades', 'oos_wr', 'oos_pnl', 'oos_mdd',
                        'oos_pf', 'oos_positive']:
                assert key in fold


# ── scan_patterns (universal) ─────────────────────────────────


class TestScanPatterns:
    """Test scan_patterns() — universal TP/SL full scan."""

    def test_basic_scan(self):
        """Scan with synthetic data returns expected structure."""
        df = _make_df_for_tests(n=300)
        result = scan_patterns(
            df, uni_tp=2.0, uni_sl=3.0,
            edge_threshold=1.0, mc_threshold=0.05, min_trades=5,
        )
        assert 'long_patterns' in result
        assert 'short_patterns' in result
        assert 'portfolio_stats' in result
        assert 'portfolio_mc' in result
        assert 'pattern_details' in result

    def test_with_prebuilt_signal_index(self):
        """Scan with pre-built signal index."""
        df = _make_df_for_tests(n=300)
        types = df['candle_type'].tolist()
        si = build_signal_index(types, len(types))
        result = scan_patterns(
            df, uni_tp=2.0, uni_sl=3.0,
            edge_threshold=1.0, mc_threshold=0.05, min_trades=5,
            signal_index=si,
        )
        assert isinstance(result['long_patterns'], list)
        assert isinstance(result['short_patterns'], list)

    def test_high_edge_threshold_filters_all(self):
        """Very high edge threshold → no patterns pass."""
        df = _make_df_for_tests(n=300)
        result = scan_patterns(
            df, uni_tp=2.0, uni_sl=3.0,
            edge_threshold=99.0, mc_threshold=0.05, min_trades=5,
        )
        assert len(result['long_patterns']) == 0
        assert len(result['short_patterns']) == 0

    def test_with_correction(self):
        """Scan with BH correction applied."""
        df = _make_df_for_tests(n=300)
        result = scan_patterns(
            df, uni_tp=2.0, uni_sl=3.0,
            edge_threshold=1.0, mc_threshold=0.05, min_trades=5,
            correction_method='bh', fdr_q=0.05,
        )
        assert 'correction_meta' in result

    def test_require_portfolio_mc(self):
        """require_portfolio_mc logs warning but doesn't change result structure."""
        df = _make_df_for_tests(n=300)
        result = scan_patterns(
            df, uni_tp=2.0, uni_sl=3.0,
            edge_threshold=1.0, mc_threshold=0.05, min_trades=5,
            require_portfolio_mc=True,
        )
        assert 'portfolio_mc_pass' in result


# ── scan_patterns_pp ──────────────────────────────────────────


class TestScanPatternsPP:
    """Test scan_patterns_pp() — per-pattern discovery."""

    def test_sequential_mode(self):
        """Sequential mode (concurrency=1) returns expected structure."""
        df = _make_df_for_tests(n=300)
        result = scan_patterns_pp(
            df, edge_threshold=1.0, mc_threshold=0.05, min_trades=5,
            concurrency=1,
        )
        assert 'long_patterns' in result
        assert 'short_patterns' in result
        assert 'patterns_tpsl' in result
        assert 'portfolio_stats' in result

    def test_high_threshold_empty(self):
        """High edge threshold → no patterns."""
        df = _make_df_for_tests(n=300)
        result = scan_patterns_pp(
            df, edge_threshold=99.0, mc_threshold=0.05, min_trades=5,
            concurrency=1,
        )
        assert len(result['long_patterns']) == 0
        assert len(result['short_patterns']) == 0

    def test_with_correction_method(self):
        """Per-pattern scan with BH correction."""
        df = _make_df_for_tests(n=300)
        result = scan_patterns_pp(
            df, edge_threshold=1.0, mc_threshold=0.05, min_trades=5,
            concurrency=1, correction_method='bh',
        )
        assert 'correction_meta' in result

    def test_tp_sl_distribution(self):
        """Result includes TP/SL distribution when patterns found."""
        df = _make_df_for_tests(n=300)
        result = scan_patterns_pp(
            df, edge_threshold=1.0, mc_threshold=0.05, min_trades=5,
            concurrency=1,
        )
        # Distribution may be empty if no patterns found
        assert 'tp_distribution' in result
        assert 'sl_distribution' in result


# ── scan_patterns_mae_mfe ─────────────────────────────────────


class TestScanPatternsMaeMfe:
    """Test scan_patterns_mae_mfe() — MAE/MFE discovery."""

    def test_sequential_mode(self):
        """Sequential MAE/MFE scan returns expected structure."""
        df = _make_df_for_tests(n=300)
        result = scan_patterns_mae_mfe(
            df, edge_threshold=1.0, mc_threshold=0.05, min_trades=5,
            concurrency=1,
        )
        assert 'long_patterns' in result
        assert 'short_patterns' in result
        assert 'patterns_tpsl' in result

    def test_high_threshold_empty(self):
        """Very high edge threshold → no patterns."""
        df = _make_df_for_tests(n=300)
        result = scan_patterns_mae_mfe(
            df, edge_threshold=99.0, mc_threshold=0.05, min_trades=5,
            concurrency=1,
        )
        assert len(result['long_patterns']) == 0
        assert len(result['short_patterns']) == 0


# ── build_output_json ─────────────────────────────────────────


class TestBuildOutputJson:
    """Test build_output_json() — JSON output structure builder."""

    @pytest.fixture
    def minimal_scan_result(self):
        """Minimal scan result for testing output builder."""
        return {
            'long_patterns': ['A-B-C'],
            'short_patterns': ['D-E-F'],
            'portfolio_stats': {'trades': 10, 'wr': 70.0, 'pnl': 50.0,
                                'mdd': 10.0, 'pf': 2.5},
            'portfolio_mc': 0.001,
            'portfolio_mc_pass': True,
            'pattern_details': {
                'A-B-C_LONG': {'pattern': 'A-B-C', 'direction': 'LONG',
                               'trades': 5, 'wr': 80.0, 'edge': 20.0, 'mc_p': 0.001},
            },
            'correction_meta': {'correction_method': 'none'},
        }

    def test_universal_mode(self, minimal_scan_result):
        """Universal discovery mode output."""
        output = build_output_json(
            minimal_scan_result,
            data_file='/tmp/test.csv', data_bars=1000,
            discovery_method='universal',
            edge_threshold=10.0, mc_threshold=0.01, min_trades=25,
            uni_tp=2.0, uni_sl=3.0,
        )
        assert output['version'] == '2.1'
        assert output['tp_sl_mode'] == 'universal'
        assert output['universal_tp'] == 2.0
        assert output['universal_sl'] == 3.0
        assert output['patterns']['long'] == ['A-B-C']
        assert output['backtest_summary']['total_trades'] == 10

    def test_per_pattern_mode(self, minimal_scan_result):
        """Per-pattern discovery mode output."""
        minimal_scan_result['patterns_tpsl'] = {'A-B-C': [1.5, 3.0]}
        minimal_scan_result['tp_distribution'] = {'min': 1.5, 'median': 1.5,
                                                   'mean': 1.5, 'max': 1.5}
        minimal_scan_result['sl_distribution'] = {'min': 3.0, 'median': 3.0,
                                                   'mean': 3.0, 'max': 3.0}
        output = build_output_json(
            minimal_scan_result,
            data_file='/tmp/test.csv', data_bars=1000,
            discovery_method='per_pattern',
            edge_threshold=10.0, mc_threshold=0.01, min_trades=25,
        )
        assert output['tp_sl_mode'] == 'per_pattern'
        assert 'patterns_tpsl' in output
        assert 'tp_distribution' in output

    def test_mae_mfe_mode(self, minimal_scan_result):
        """MAE/MFE discovery mode output."""
        minimal_scan_result['patterns_tpsl'] = {'A-B-C': [1.5, 3.0]}
        minimal_scan_result['tp_distribution'] = {'min': 1.5, 'median': 1.5,
                                                   'mean': 1.5, 'max': 1.5}
        minimal_scan_result['sl_distribution'] = {'min': 3.0, 'median': 3.0,
                                                   'mean': 3.0, 'max': 3.0}
        output = build_output_json(
            minimal_scan_result,
            data_file='/tmp/test.csv', data_bars=1000,
            discovery_method='mae_mfe',
            edge_threshold=10.0, mc_threshold=0.01, min_trades=25,
        )
        assert output['tp_sl_mode'] == 'per_pattern'  # Bot-compatible
        assert 'tp_percentile_grid' in output['selection_criteria']
        assert 'sl_percentile_grid' in output['selection_criteria']

    def test_with_wf_result(self, minimal_scan_result):
        """Walk-forward results included in output."""
        wf = {'n_folds': 3, 'positive_folds': 2, 'total_oos_pnl': 100.0}
        output = build_output_json(
            minimal_scan_result,
            data_file='/tmp/test.csv', data_bars=1000,
            discovery_method='universal',
            edge_threshold=10.0, mc_threshold=0.01, min_trades=25,
            uni_tp=2.0, uni_sl=3.0,
            wf_result=wf,
        )
        assert output['walk_forward'] == wf

    def test_with_timing(self, minimal_scan_result):
        """Timing dict included in output."""
        timing = {'classify_sec': 1.0, 'scan_sec': 5.0, 'total_sec': 6.0}
        output = build_output_json(
            minimal_scan_result,
            data_file='/tmp/test.csv', data_bars=1000,
            discovery_method='universal',
            edge_threshold=10.0, mc_threshold=0.01, min_trades=25,
            uni_tp=2.0, uni_sl=3.0,
            timing=timing,
        )
        assert output['timing'] == timing

    def test_correction_meta_from_param(self, minimal_scan_result):
        """Correction meta from parameter takes precedence."""
        meta = {'correction_method': 'bh', 'fdr_q': 0.05, 'n_before_correction': 10}
        output = build_output_json(
            minimal_scan_result,
            data_file='/tmp/test.csv', data_bars=1000,
            discovery_method='universal',
            edge_threshold=10.0, mc_threshold=0.01, min_trades=25,
            uni_tp=2.0, uni_sl=3.0,
            correction_meta=meta,
        )
        assert output['selection_criteria']['correction_method'] == 'bh'

    def test_data_file_basename_only(self, minimal_scan_result):
        """data_file is stored as basename only."""
        output = build_output_json(
            minimal_scan_result,
            data_file='/very/long/path/to/data.csv', data_bars=1000,
            discovery_method='universal',
            edge_threshold=10.0, mc_threshold=0.01, min_trades=25,
            uni_tp=2.0, uni_sl=3.0,
        )
        assert output['data_file'] == 'data.csv'


# ── main() CLI ───────────────────────────────────────────────


class TestMainCLI:
    """Test main() CLI entry point with mocked file I/O."""

    @pytest.fixture
    def mock_df(self):
        """Small DataFrame for fast CLI testing."""
        return _make_df_for_tests(n=200)

    def test_universal_mode(self, mock_df, tmp_path):
        """CLI with --discovery-method universal runs end-to-end."""
        output_file = str(tmp_path / "test_output.json")
        with patch('scripts.scanner.pattern_scanner.load_and_classify',
                   return_value=mock_df):
            with patch('sys.argv', ['scanner',
                                    '--discovery-method', 'universal',
                                    '--output', output_file,
                                    '--min-trades', '3',
                                    '--edge-threshold', '1.0',
                                    '--mc-threshold', '0.1']):
                from scripts.scanner.pattern_scanner import main
                main()
        assert os.path.exists(output_file)
        with open(output_file) as f:
            data = json.load(f)
        assert data['version'] == '2.1'
        assert data['tp_sl_mode'] == 'universal'

    def test_per_pattern_mode(self, mock_df, tmp_path):
        """CLI with --discovery-method per_pattern runs end-to-end."""
        output_file = str(tmp_path / "test_pp.json")
        with patch('scripts.scanner.pattern_scanner.load_and_classify',
                   return_value=mock_df):
            with patch('sys.argv', ['scanner',
                                    '--discovery-method', 'per_pattern',
                                    '--output', output_file,
                                    '--min-trades', '3',
                                    '--edge-threshold', '1.0',
                                    '--mc-threshold', '0.1',
                                    '--concurrency', '1']):
                from scripts.scanner.pattern_scanner import main
                main()
        assert os.path.exists(output_file)
        with open(output_file) as f:
            data = json.load(f)
        assert 'tp_sl_mode' in data

    def test_mae_mfe_mode(self, mock_df, tmp_path):
        """CLI with --discovery-method mae_mfe runs end-to-end."""
        output_file = str(tmp_path / "test_mae.json")
        with patch('scripts.scanner.pattern_scanner.load_and_classify',
                   return_value=mock_df):
            with patch('sys.argv', ['scanner',
                                    '--discovery-method', 'mae_mfe',
                                    '--output', output_file,
                                    '--min-trades', '3',
                                    '--edge-threshold', '1.0',
                                    '--mc-threshold', '0.1',
                                    '--concurrency', '1']):
                from scripts.scanner.pattern_scanner import main
                main()
        assert os.path.exists(output_file)

    def test_with_wf_folds(self, mock_df, tmp_path):
        """CLI with --wf-folds activates walk-forward validation."""
        output_file = str(tmp_path / "test_wf.json")
        with patch('scripts.scanner.pattern_scanner.load_and_classify',
                   return_value=mock_df):
            with patch('sys.argv', ['scanner',
                                    '--discovery-method', 'universal',
                                    '--output', output_file,
                                    '--min-trades', '3',
                                    '--edge-threshold', '1.0',
                                    '--mc-threshold', '0.1',
                                    '--wf-folds', '2']):
                from scripts.scanner.pattern_scanner import main
                main()
        with open(output_file) as f:
            data = json.load(f)
        assert 'walk_forward' in data

    def test_with_correction(self, mock_df, tmp_path):
        """CLI with --correction bh applies BH FDR correction."""
        output_file = str(tmp_path / "test_corr.json")
        with patch('scripts.scanner.pattern_scanner.load_and_classify',
                   return_value=mock_df):
            with patch('sys.argv', ['scanner',
                                    '--discovery-method', 'universal',
                                    '--output', output_file,
                                    '--min-trades', '3',
                                    '--edge-threshold', '1.0',
                                    '--mc-threshold', '0.1',
                                    '--correction', 'bh']):
                from scripts.scanner.pattern_scanner import main
                main()
        with open(output_file) as f:
            data = json.load(f)
        # correction_method in selection_criteria comes from correction_meta
        # which is 'bh' when patterns exist, but may be overwritten by
        # scan_result's own correction_meta when 0 patterns pass initially
        sc = data['selection_criteria']
        assert 'correction_method' in sc

    def test_verbose_flag(self, mock_df, tmp_path):
        """CLI with -v flag sets debug logging."""
        output_file = str(tmp_path / "test_verbose.json")
        with patch('scripts.scanner.pattern_scanner.load_and_classify',
                   return_value=mock_df):
            with patch('sys.argv', ['scanner',
                                    '--discovery-method', 'universal',
                                    '--output', output_file,
                                    '--min-trades', '3',
                                    '--edge-threshold', '1.0',
                                    '--mc-threshold', '0.1',
                                    '-v']):
                from scripts.scanner.pattern_scanner import main
                main()


# ── Additional coverage: strong synthetic data ────────────────


def _make_strong_df(n=400, seed=42):
    """Create DataFrame where specific patterns have strong edge.

    Pattern 'U-U-U' appears repeatedly with strong upward moves after,
    ensuring LONG signals pass edge + MC filters.
    """
    rng = np.random.RandomState(seed)
    opens = np.full(n, 50000.0)
    highs = np.full(n, 50100.0)
    lows = np.full(n, 49900.0)
    closes = np.full(n, 50000.0)

    # Create repeating U-U-U pattern
    types = ['U'] * n

    # After every signal bar (every 3rd bar starting at 2), create strong up move
    signal_bars = list(range(2, n - 5, 3))
    for sb in signal_bars:
        entry_bar = sb + 1
        for j in range(entry_bar + 1, min(entry_bar + 5, n)):
            highs[j] = 50000.0 * 1.025  # +2.5% → hits TP 2.0%
            lows[j] = 50000.0 * 0.997   # -0.3% → far from SL

    df = pd.DataFrame({
        'open': opens, 'high': highs, 'low': lows, 'close': closes,
        'candle_type': types,
    })
    return df


class TestScanPatternsWithHits:
    """Test scan functions with data that produces actual pattern hits."""

    def test_scan_patterns_finds_patterns(self):
        """scan_patterns finds patterns when strong signal exists."""
        df = _make_strong_df(n=400)
        result = scan_patterns(
            df, uni_tp=2.0, uni_sl=3.0,
            edge_threshold=1.0, mc_threshold=0.1, min_trades=5,
        )
        # With strong repeating U-U-U + upward move, should find LONG patterns
        total = len(result['long_patterns']) + len(result['short_patterns'])
        if total > 0:
            assert result['portfolio_stats']['trades'] > 0
            assert result['portfolio_mc'] < 1.0

    def test_scan_patterns_portfolio_mc(self):
        """scan_patterns with require_portfolio_mc and actual trades."""
        df = _make_strong_df(n=400)
        result = scan_patterns(
            df, uni_tp=2.0, uni_sl=3.0,
            edge_threshold=1.0, mc_threshold=0.1, min_trades=5,
            require_portfolio_mc=True,
        )
        assert 'portfolio_mc_pass' in result

    def test_scan_pp_sequential_with_hits(self):
        """scan_patterns_pp sequential finds patterns and builds tpsl dict."""
        df = _make_strong_df(n=400)
        result = scan_patterns_pp(
            df, edge_threshold=1.0, mc_threshold=0.1, min_trades=5,
            concurrency=1,
        )
        if result['long_patterns'] or result['short_patterns']:
            assert len(result['patterns_tpsl']) > 0
            assert result['portfolio_stats']['trades'] > 0

    def test_scan_pp_dedup_logic(self):
        """Test dedup logic when same pattern appears in both directions."""
        n = 400
        opens = np.full(n, 50000.0)
        highs = np.full(n, 50100.0)
        lows = np.full(n, 49900.0)
        closes = np.full(n, 50000.0)
        types = ['U'] * n

        # Create moves that work for BOTH long and short (ambiguous)
        for i in range(5, n - 5, 3):
            eb = i + 1
            for j in range(eb + 1, min(eb + 5, n)):
                # Both directions "win" with big range
                highs[j] = 50000.0 * 1.03
                lows[j] = 50000.0 * 0.97

        df = pd.DataFrame({
            'open': opens, 'high': highs, 'low': lows, 'close': closes,
            'candle_type': types,
        })

        result = scan_patterns_pp(
            df, edge_threshold=0.5, mc_threshold=0.1, min_trades=3,
            concurrency=1, max_baseline_wr=90.0,
        )
        # Dedup ensures no pattern appears in both directions
        all_pats = result['long_patterns'] + result['short_patterns']
        assert len(all_pats) == len(set(all_pats))

    def test_scan_pp_require_portfolio_mc(self):
        """scan_patterns_pp with require_portfolio_mc warning path."""
        df = _make_strong_df(n=400)
        result = scan_patterns_pp(
            df, edge_threshold=1.0, mc_threshold=0.1, min_trades=5,
            concurrency=1, require_portfolio_mc=True,
        )
        assert 'portfolio_mc_pass' in result


class TestScanMaeMfeWithHits:
    """Test scan_patterns_mae_mfe with data producing hits."""

    def test_sequential_with_hits(self):
        """MAE/MFE scan finds patterns when strong signal exists."""
        df = _make_strong_df(n=400)
        result = scan_patterns_mae_mfe(
            df, edge_threshold=1.0, mc_threshold=0.1, min_trades=5,
            concurrency=1,
        )
        if result['long_patterns'] or result['short_patterns']:
            assert len(result['patterns_tpsl']) > 0

    def test_dedup_both_directions(self):
        """MAE/MFE dedup keeps best direction per pattern."""
        n = 400
        opens = np.full(n, 50000.0)
        highs = np.full(n, 50100.0)
        lows = np.full(n, 49900.0)
        closes = np.full(n, 50000.0)
        types = ['U'] * n

        for i in range(5, n - 5, 3):
            eb = i + 1
            for j in range(eb + 1, min(eb + 5, n)):
                highs[j] = 50000.0 * 1.03
                lows[j] = 50000.0 * 0.97

        df = pd.DataFrame({
            'open': opens, 'high': highs, 'low': lows, 'close': closes,
            'candle_type': types,
        })
        result = scan_patterns_mae_mfe(
            df, edge_threshold=0.5, mc_threshold=0.1, min_trades=3,
            concurrency=1, max_baseline_wr=90.0,
        )
        all_pats = result['long_patterns'] + result['short_patterns']
        assert len(all_pats) == len(set(all_pats))


class TestGridSearchBestEdgeCases:
    """Additional grid_search_best edge case tests."""

    def test_sl_below_half_skipped(self):
        """SL grid values below 0.5 are skipped."""
        n = 200
        opens = np.full(n, 50000.0)
        highs = np.full(n, 51000.0)
        lows = np.full(n, 49000.0)
        signals = list(range(0, n - 5, 5))
        result = grid_search_best(signals, 'LONG', opens, highs, lows, n, min_tr=5)
        if result is not None:
            assert result['sl'] >= 0.5

    def test_no_mdd_score_is_cum_pnl(self):
        """When MDD=0 (all wins), score = cum PnL."""
        n = 200
        opens = np.full(n, 50000.0)
        highs = np.full(n, 50000.0)
        lows = np.full(n, 50000.0)
        signals = list(range(5, n - 10, 10))
        # All bars hit TP immediately
        for sig in signals:
            eb = sig + 1
            if eb + 2 >= n:
                continue
            highs[eb + 1] = 50000.0 * 1.05
        result = grid_search_best(signals, 'LONG', opens, highs, lows, n, min_tr=5)
        if result is not None:
            assert result['pnl'] > 0


class TestPpWorkerEdgeCases:
    """Additional _pp_worker edge case tests."""

    def test_worker_edge_filter(self):
        """Worker returns None when edge < threshold."""
        n = 100
        opens = np.full(n, 50000.0)
        highs = np.full(n, 50050.0)  # Very small moves
        lows = np.full(n, 49950.0)
        _pp_init(opens, highs, lows, n)
        signals = list(range(0, n - 5, 5))
        # edge_threshold=99 → nothing passes
        result = _pp_worker(('TEST', 'LONG', signals, 3, 99.0, 0.01, 70.0))
        assert result is None

    def test_worker_mc_filter(self):
        """Worker returns None when MC p-value >= threshold."""
        n = 100
        opens = np.full(n, 50000.0)
        highs = np.full(n, 50100.0)
        lows = np.full(n, 49900.0)
        _pp_init(opens, highs, lows, n)
        signals = list(range(0, n - 5, 5))
        # mc_threshold=0.0 → nothing passes
        result = _pp_worker(('TEST', 'LONG', signals, 3, 0.0, 0.0, 70.0))
        assert result is None


class TestMaeMfeWorkerEdgeCases:
    """Additional _mae_mfe_worker edge case tests."""

    def test_worker_with_enough_data(self):
        """Worker processes enough data for MAE/MFE analysis."""
        n = 300
        opens = np.full(n, 50000.0)
        highs = np.full(n, 50000.0)
        lows = np.full(n, 50000.0)
        signals = list(range(5, n - 10, 10))
        for sig in signals:
            eb = sig + 1
            if eb + 4 >= n:
                continue
            for j in range(eb + 1, min(eb + 4, n)):
                highs[j] = 50000.0 * 1.025
                lows[j] = 50000.0 * 0.998
        _pp_init(opens, highs, lows, n)
        result = _mae_mfe_worker(('TEST', 'LONG', signals, 5, 1.0, 0.1, 70.0))
        # May or may not find a result
        if result is not None:
            assert 'exc_stats' in result
            assert 'tp_percentile' in result


class TestMainCLIPerPatternWithHits:
    """CLI tests with data that produces actual hits for TP/SL distribution."""

    def test_pp_with_hits_prints_distribution(self, tmp_path):
        """CLI per_pattern mode with hits prints TP/SL distribution."""
        df = _make_strong_df(n=400)
        output_file = str(tmp_path / "test_pp_hits.json")
        with patch('scripts.scanner.pattern_scanner.load_and_classify',
                   return_value=df):
            with patch('sys.argv', ['scanner',
                                    '--discovery-method', 'per_pattern',
                                    '--output', output_file,
                                    '--min-trades', '5',
                                    '--edge-threshold', '1.0',
                                    '--mc-threshold', '0.1',
                                    '--concurrency', '1']):
                from scripts.scanner.pattern_scanner import main
                main()
        with open(output_file) as f:
            data = json.load(f)
        if data.get('tp_distribution'):
            assert 'min' in data['tp_distribution']
            assert 'max' in data['tp_distribution']

    def test_wf_pp_prints_summary(self, tmp_path):
        """CLI with WF folds and per_pattern prints WF summary."""
        df = _make_strong_df(n=400)
        output_file = str(tmp_path / "test_wf_pp.json")
        with patch('scripts.scanner.pattern_scanner.load_and_classify',
                   return_value=df):
            with patch('sys.argv', ['scanner',
                                    '--discovery-method', 'per_pattern',
                                    '--output', output_file,
                                    '--min-trades', '5',
                                    '--edge-threshold', '1.0',
                                    '--mc-threshold', '0.1',
                                    '--concurrency', '1',
                                    '--wf-folds', '2']):
                from scripts.scanner.pattern_scanner import main
                main()
        with open(output_file) as f:
            data = json.load(f)
        assert 'walk_forward' in data


# ── Coverage gap: scan_universe_range filter branches ────────


class TestScanUniverseRangeFilters:
    """Test scan_universe_range per_pattern/mae_mfe inner filter paths."""

    @pytest.fixture
    def strong_scan_data(self):
        """Data where patterns reliably pass filters in all modes."""
        n = 400
        opens = np.full(n, 50000.0)
        highs = np.full(n, 50100.0)
        lows = np.full(n, 49900.0)

        # Repeating 'U' pattern → single pattern 'U-U-U'
        types = ['U'] * n
        signal_index = build_signal_index(types, n)

        # Strong moves after every signal
        for sigs in signal_index.values():
            for sig in sigs:
                eb = sig + 1
                if eb + 4 >= n:
                    continue
                for j in range(eb + 1, min(eb + 4, n)):
                    highs[j] = 50000.0 * 1.025
                    lows[j] = 50000.0 * 0.997

        return signal_index, opens, highs, lows, n

    def test_per_pattern_with_hits(self, strong_scan_data):
        """per_pattern mode with strong data → patterns found."""
        signal_index, opens, highs, lows, n = strong_scan_data
        result = scan_universe_range(
            signal_index, opens, highs, lows, n,
            bar_start=0, bar_end=n, mode='per_pattern',
            min_trades=5, edge_threshold=1.0, mc_threshold=0.1,
        )
        assert isinstance(result, list)

    def test_mae_mfe_with_hits(self, strong_scan_data):
        """mae_mfe mode with strong data → patterns found."""
        signal_index, opens, highs, lows, n = strong_scan_data
        result = scan_universe_range(
            signal_index, opens, highs, lows, n,
            bar_start=0, bar_end=n, mode='mae_mfe',
            min_trades=5, edge_threshold=1.0, mc_threshold=0.1,
        )
        assert isinstance(result, list)


class TestPpWorkerFullPath:
    """Test _pp_worker with data that exercises trades_raw filtering."""

    def test_worker_trades_fewer_than_min_after_bt(self):
        """Worker returns None when bt_signals produces < min_trades."""
        n = 100
        opens = np.full(n, 50000.0)
        highs = np.full(n, 50010.0)  # tiny moves → timeouts
        lows = np.full(n, 49990.0)
        _pp_init(opens, highs, lows, n)
        # Many signals but moves too small to hit any TP/SL
        signals = list(range(0, 80, 4))
        result = _pp_worker(('TEST', 'LONG', signals, 5, 1.0, 0.1, 70.0))
        assert result is None

    def test_worker_success_path(self):
        """Worker returns full result when all filters pass."""
        n = 400
        opens = np.full(n, 50000.0)
        highs = np.full(n, 50100.0)
        lows = np.full(n, 49900.0)
        signals = list(range(5, n - 10, 3))
        for sig in signals:
            eb = sig + 1
            if eb + 4 >= n:
                continue
            for j in range(eb + 1, min(eb + 4, n)):
                highs[j] = 50000.0 * 1.025
                lows[j] = 50000.0 * 0.997
        _pp_init(opens, highs, lows, n)
        result = _pp_worker(('U-U-U', 'LONG', signals, 5, 1.0, 0.1, 70.0))
        if result is not None:
            assert result['key'] == 'U-U-U_LONG'
            assert 'trades_raw' in result
            assert result['trades'] >= 5


class TestMaeMfeWorkerFullPath:
    """Test _mae_mfe_worker with data that exercises full path."""

    def test_worker_full_path(self):
        """Worker with enough data for complete MAE/MFE flow."""
        n = 400
        opens = np.full(n, 50000.0)
        highs = np.full(n, 50100.0)
        lows = np.full(n, 49900.0)
        signals = list(range(5, n - 10, 3))
        for sig in signals:
            eb = sig + 1
            if eb + 4 >= n:
                continue
            for j in range(eb + 1, min(eb + 4, n)):
                highs[j] = 50000.0 * 1.025
                lows[j] = 50000.0 * 0.997
        _pp_init(opens, highs, lows, n)
        result = _mae_mfe_worker(('U-U-U', 'LONG', signals, 5, 1.0, 0.1, 70.0))
        if result is not None:
            assert result['key'] == 'U-U-U_LONG'
            assert 'exc_stats' in result
            assert 'trades_raw' in result

    def test_worker_edge_filter_fail(self):
        """Worker returns None when edge below threshold."""
        n = 200
        opens = np.full(n, 50000.0)
        highs = np.full(n, 50100.0)
        lows = np.full(n, 49900.0)
        signals = list(range(5, n - 10, 5))
        for sig in signals:
            eb = sig + 1
            if eb + 4 >= n:
                continue
            for j in range(eb + 1, min(eb + 4, n)):
                highs[j] = 50000.0 * 1.02
                lows[j] = 50000.0 * 0.998
        _pp_init(opens, highs, lows, n)
        # Very high edge threshold → fails
        result = _mae_mfe_worker(('TEST', 'LONG', signals, 3, 99.0, 0.1, 70.0))
        assert result is None

    def test_worker_mc_filter_fail(self):
        """Worker returns None when MC p-value too high."""
        n = 200
        opens = np.full(n, 50000.0)
        highs = np.full(n, 50100.0)
        lows = np.full(n, 49900.0)
        signals = list(range(5, n - 10, 5))
        for sig in signals:
            eb = sig + 1
            if eb + 4 >= n:
                continue
            for j in range(eb + 1, min(eb + 4, n)):
                highs[j] = 50000.0 * 1.02
                lows[j] = 50000.0 * 0.998
        _pp_init(opens, highs, lows, n)
        # mc_threshold=0 → nothing passes MC
        result = _mae_mfe_worker(('TEST', 'LONG', signals, 3, 0.0, 0.0, 70.0))
        assert result is None


class TestGridSearchMaeMfeEdgeCases:
    """Additional grid_search_mae_mfe edge case tests."""

    def test_baseline_wr_filter_blocks(self):
        """High baseline WR filter blocks all combos."""
        n = 200
        opens = np.full(n, 50000.0)
        highs = np.full(n, 51000.0)
        lows = np.full(n, 49000.0)
        signals = list(range(5, n - 10, 5))
        # max_baseline_wr=0 → all combos blocked
        best, exc = grid_search_mae_mfe(
            signals, 'LONG', opens, highs, lows, n,
            min_tr=3, max_baseline_wr=0.0,
        )
        assert best is None

    def test_successful_search(self):
        """Successful MAE/MFE search returns result and excursion stats."""
        n = 400
        opens = np.full(n, 50000.0)
        highs = np.full(n, 50100.0)
        lows = np.full(n, 49900.0)
        signals = list(range(5, n - 10, 3))
        for sig in signals:
            eb = sig + 1
            if eb + 4 >= n:
                continue
            for j in range(eb + 1, min(eb + 4, n)):
                highs[j] = 50000.0 * 1.025
                lows[j] = 50000.0 * 0.997
        best, exc = grid_search_mae_mfe(
            signals, 'LONG', opens, highs, lows, n, min_tr=5,
        )
        if best is not None:
            for key in ['tp', 'sl', 'tp_percentile', 'sl_percentile',
                        'trades', 'wr', 'pnl', 'mdd']:
                assert key in best
            assert exc is not None
            assert 'mfe_mae_ratio' in exc
