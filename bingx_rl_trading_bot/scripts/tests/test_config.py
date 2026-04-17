"""Config loading + validation tests (BUG#52 + general).

Config errors must abort at startup — a silent misconfig (e.g. leverage
mismatch) could cause immediate liquidation.
"""
import os
import tempfile
import yaml
import pytest
from scripts.production.c1_breakout.config import load_config, DEFAULT_CONFIG


def _write_yaml(path, data):
    with open(path, 'w') as f:
        yaml.dump(data, f)


class TestConfigDefaults:
    """Default config has sane production values."""

    def test_defaults_returned_when_file_missing(self, tmp_path):
        """A. Edge: no config file → defaults returned (leverage=1 conservative)."""
        c = load_config(str(tmp_path / 'nonexistent.yaml'))
        assert c['exchange']['leverage'] == 1  # default conservative
        assert c['strategy']['channel_period'] == 15

    def test_default_structure_has_all_sections(self):
        """A. Edge: DEFAULT_CONFIG dict must have strategy/exchange/bot/risk."""
        assert 'strategy' in DEFAULT_CONFIG
        assert 'exchange' in DEFAULT_CONFIG
        assert 'bot' in DEFAULT_CONFIG


class TestLeverageValidation:
    """BUG#52: trading_leverage must not exceed exchange leverage."""

    def test_trading_greater_than_exchange_rejected(self, tmp_path):
        """D. Rollback safety: misconfig → ValueError at load, not runtime."""
        p = tmp_path / 'cfg.yaml'
        _write_yaml(p, {'exchange': {'leverage': 3, 'trading_leverage': 10}})
        with pytest.raises(ValueError, match='trading_leverage'):
            load_config(str(p))

    def test_trading_equal_to_exchange_accepted(self, tmp_path):
        """C. Bug interaction: trading == exchange is valid (single cap)."""
        p = tmp_path / 'cfg.yaml'
        _write_yaml(p, {'exchange': {'leverage': 5, 'trading_leverage': 5}})
        c = load_config(str(p))
        assert c['exchange']['leverage'] == 5

    def test_trading_less_than_exchange_accepted(self, tmp_path):
        """Happy path: current production (10 / 3) is valid."""
        p = tmp_path / 'cfg.yaml'
        _write_yaml(p, {'exchange': {'leverage': 10, 'trading_leverage': 3}})
        c = load_config(str(p))
        assert c['exchange']['leverage'] == 10
        assert c['exchange']['trading_leverage'] == 3

    def test_zero_leverage_rejected(self, tmp_path):
        """A. Edge: 0 leverage meaningless."""
        p = tmp_path / 'cfg.yaml'
        _write_yaml(p, {'exchange': {'leverage': 0, 'trading_leverage': 0}})
        with pytest.raises(ValueError, match='positive'):
            load_config(str(p))

    def test_negative_leverage_rejected(self, tmp_path):
        """A. Edge: negative leverage inverts position — catastrophic."""
        p = tmp_path / 'cfg.yaml'
        # Both negative — 'trading > exchange' check (1 > -1) fires first;
        # if both -1, 'positive' check catches.
        _write_yaml(p, {'exchange': {'leverage': -5, 'trading_leverage': -5}})
        with pytest.raises(ValueError, match='positive'):
            load_config(str(p))


class TestSLBoundsValidation:
    """BUG#52 extension: sl_min_pct < sl_max_pct."""

    def test_reversed_bounds_rejected(self, tmp_path):
        """D. Rollback: sl_min > sl_max would filter all signals."""
        p = tmp_path / 'cfg.yaml'
        _write_yaml(p, {'strategy': {'sl_min_pct': 3.0, 'sl_max_pct': 0.15}})
        with pytest.raises(ValueError, match='sl_min_pct'):
            load_config(str(p))

    def test_equal_bounds_rejected(self, tmp_path):
        """A. Edge: sl_min == sl_max → only exact match allowed (degenerate)."""
        p = tmp_path / 'cfg.yaml'
        _write_yaml(p, {'strategy': {'sl_min_pct': 1.0, 'sl_max_pct': 1.0}})
        with pytest.raises(ValueError):
            load_config(str(p))


class TestDeepMerge:
    """User overrides should merge with defaults (not replace)."""

    def test_partial_override_keeps_other_defaults(self, tmp_path):
        """C. Bug interaction: overriding trail_K should not reset other params."""
        p = tmp_path / 'cfg.yaml'
        _write_yaml(p, {'strategy': {'trail_K': 3.0}})
        c = load_config(str(p))
        assert c['strategy']['trail_K'] == 3.0
        # Other defaults preserved
        assert c['strategy']['channel_period'] == 15
        assert c['strategy']['body_min_ratio'] == 0.4

    def test_empty_yaml_equals_defaults(self, tmp_path):
        """A. Edge: empty YAML → defaults returned."""
        p = tmp_path / 'cfg.yaml'
        p.write_text('')  # empty file
        c = load_config(str(p))
        assert c['strategy']['channel_period'] == 15
