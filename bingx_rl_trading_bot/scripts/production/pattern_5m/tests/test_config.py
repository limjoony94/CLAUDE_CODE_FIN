"""Tests for config.py — YAML parsing, validation, required keys, error handling."""

import pytest
import os
import yaml
from pathlib import Path

from bingx_rl_trading_bot.scripts.production.pattern_5m.config import (
    load_config,
    validate_config,
    _deep_copy_config,
    _merge_config,
    get_strategy_config,
    get_risk_config,
    get_api_config,
)
from bingx_rl_trading_bot.scripts.production.pattern_5m.constants import DEFAULT_CONFIG


# ── Helper Fixtures ───────────────────────────────────────────

@pytest.fixture
def temp_config_file(tmp_path):
    """Create a temporary config file path."""
    return str(tmp_path / "test_config.yaml")


@pytest.fixture
def valid_config():
    """Create a valid configuration dictionary."""
    return {
        'symbol': 'BTC-USDT',
        'timeframe': '5m',
        'leverage': 3,
        'strategy': {
            'tp_pct': 1.0,
            'sl_pct': 1.0,
            'patterns_long': ['U-MU-H'],
            'patterns_short': ['DN-D-BD'],
        },
        'risk': {
            'max_daily_loss_pct': 5.0,
            'max_position_size': 100.0,
        },
        'api': {
            'rate_limit': 1200,
        }
    }


# ── YAML Parsing ──────────────────────────────────────────────

class TestYAMLParsing:
    """Test YAML file loading and parsing."""

    def test_load_valid_yaml(self, temp_config_file, valid_config):
        """Loading valid YAML should succeed."""
        with open(temp_config_file, 'w', encoding='utf-8') as f:
            yaml.dump(valid_config, f)

        config = load_config(temp_config_file)

        assert config['symbol'] == 'BTC-USDT'
        assert config['timeframe'] == '5m'
        assert config['leverage'] == 3

    def test_load_nonexistent_file(self, temp_config_file):
        """Loading nonexistent file should return defaults."""
        config = load_config(temp_config_file)

        # Should return DEFAULT_CONFIG
        assert 'symbol' in config
        assert 'strategy' in config
        assert 'risk' in config

    def test_load_corrupted_yaml(self, temp_config_file):
        """Loading corrupted YAML should return defaults and log warning."""
        with open(temp_config_file, 'w') as f:
            f.write("invalid: yaml: : syntax")

        config = load_config(temp_config_file)

        # Should return default config
        assert 'symbol' in config
        assert 'strategy' in config

    def test_load_empty_yaml(self, temp_config_file):
        """Loading empty YAML should return defaults."""
        with open(temp_config_file, 'w') as f:
            f.write('')

        config = load_config(temp_config_file)

        assert 'symbol' in config

    def test_yaml_with_comments(self, temp_config_file):
        """YAML with comments should parse correctly."""
        with open(temp_config_file, 'w', encoding='utf-8') as f:
            f.write("""
# Trading configuration
symbol: BTC-USDT  # Main trading pair
timeframe: 5m
leverage: 3

strategy:
  tp_pct: 1.0  # Take profit 1%
  sl_pct: 1.0
""")

        config = load_config(temp_config_file)
        assert config['symbol'] == 'BTC-USDT'


# ── Config Validation ─────────────────────────────────────────

class TestConfigValidation:
    """Test validate_config() for required fields and valid ranges."""

    def test_validate_valid_config(self, valid_config):
        """Valid config should pass validation."""
        assert validate_config(valid_config) is True

    def test_missing_required_field_symbol(self):
        """Missing 'symbol' should raise ValueError."""
        config = {
            'timeframe': '5m',
            'leverage': 3,
            'strategy': {'tp_pct': 1.0, 'sl_pct': 1.0},
            'risk': {'max_daily_loss_pct': 5.0}
        }

        with pytest.raises(ValueError, match="Missing required field: symbol"):
            validate_config(config)

    def test_missing_required_field_leverage(self):
        """Missing 'leverage' should raise ValueError."""
        config = {
            'symbol': 'BTC-USDT',
            'timeframe': '5m',
            'strategy': {'tp_pct': 1.0, 'sl_pct': 1.0},
            'risk': {'max_daily_loss_pct': 5.0}
        }

        with pytest.raises(ValueError, match="Missing required field: leverage"):
            validate_config(config)

    def test_missing_strategy_tp_pct(self):
        """Missing strategy.tp_pct should raise ValueError."""
        config = {
            'symbol': 'BTC-USDT',
            'timeframe': '5m',
            'leverage': 3,
            'strategy': {'sl_pct': 1.0},
            'risk': {'max_daily_loss_pct': 5.0}
        }

        with pytest.raises(ValueError, match="Missing strategy.tp_pct"):
            validate_config(config)

    def test_missing_strategy_sl_pct(self):
        """Missing strategy.sl_pct should raise ValueError."""
        config = {
            'symbol': 'BTC-USDT',
            'timeframe': '5m',
            'leverage': 3,
            'strategy': {'tp_pct': 1.0},
            'risk': {'max_daily_loss_pct': 5.0}
        }

        with pytest.raises(ValueError, match="Missing strategy.sl_pct"):
            validate_config(config)

    def test_invalid_leverage_zero(self):
        """Leverage <= 0 should raise ValueError."""
        config = {
            'symbol': 'BTC-USDT',
            'timeframe': '5m',
            'leverage': 0,
            'strategy': {'tp_pct': 1.0, 'sl_pct': 1.0},
            'risk': {'max_daily_loss_pct': 5.0}
        }

        with pytest.raises(ValueError, match="leverage must be positive"):
            validate_config(config)

    def test_invalid_tp_pct_negative(self):
        """Negative tp_pct should raise ValueError."""
        config = {
            'symbol': 'BTC-USDT',
            'timeframe': '5m',
            'leverage': 3,
            'strategy': {'tp_pct': -1.0, 'sl_pct': 1.0},
            'risk': {'max_daily_loss_pct': 5.0}
        }

        with pytest.raises(ValueError, match="strategy.tp_pct must be positive"):
            validate_config(config)

    def test_invalid_sl_pct_zero(self):
        """sl_pct <= 0 should raise ValueError."""
        config = {
            'symbol': 'BTC-USDT',
            'timeframe': '5m',
            'leverage': 3,
            'strategy': {'tp_pct': 1.0, 'sl_pct': 0},
            'risk': {'max_daily_loss_pct': 5.0}
        }

        with pytest.raises(ValueError, match="strategy.sl_pct must be positive"):
            validate_config(config)

    def test_invalid_max_daily_loss_pct(self):
        """max_daily_loss_pct <= 0 should raise ValueError."""
        config = {
            'symbol': 'BTC-USDT',
            'timeframe': '5m',
            'leverage': 3,
            'strategy': {'tp_pct': 1.0, 'sl_pct': 1.0},
            'risk': {'max_daily_loss_pct': 0}
        }

        with pytest.raises(ValueError, match="risk.max_daily_loss_pct must be positive"):
            validate_config(config)


# ── Config Merging ────────────────────────────────────────────

class TestConfigMerging:
    """Test config merging with defaults."""

    def test_deep_copy_config(self):
        """_deep_copy_config should create independent copy."""
        original = {'a': 1, 'b': {'c': 2}, 'list': [1, 2]}
        copy = _deep_copy_config(original)

        # Modify copy
        copy['a'] = 99
        copy['b']['c'] = 99
        copy['list'].append(3)

        # Original should be unchanged
        assert original['a'] == 1
        assert original['b']['c'] == 2
        assert len(original['list']) == 2

    def test_merge_config_override(self):
        """_merge_config should override base with override values."""
        base = {
            'symbol': 'BTC-USDT',
            'leverage': 3,
            'strategy': {'tp_pct': 1.0, 'sl_pct': 1.0}
        }
        override = {
            'leverage': 5,
            'strategy': {'tp_pct': 1.5}
        }

        merged = _merge_config(base, override)

        assert merged['leverage'] == 5
        assert merged['strategy']['tp_pct'] == 1.5
        assert merged['strategy']['sl_pct'] == 1.0  # preserved from base

    def test_merge_config_new_keys(self):
        """_merge_config should add new keys from override."""
        base = {'a': 1}
        override = {'b': 2, 'c': 3}

        merged = _merge_config(base, override)

        assert merged['a'] == 1
        assert merged['b'] == 2
        assert merged['c'] == 3

    def test_merge_preserves_base(self):
        """_merge_config should not modify original base."""
        base = {'a': 1, 'b': {'c': 2}}
        override = {'a': 99}

        merged = _merge_config(base, override)

        assert base['a'] == 1  # base unchanged


# ── Config Getters ────────────────────────────────────────────

class TestConfigGetters:
    """Test get_strategy_config, get_risk_config, get_api_config."""

    def test_get_strategy_config(self, valid_config):
        """get_strategy_config should return strategy section."""
        strategy = get_strategy_config(valid_config)

        assert strategy['tp_pct'] == 1.0
        assert strategy['sl_pct'] == 1.0
        assert 'patterns_long' in strategy

    def test_get_strategy_config_missing_uses_default(self):
        """get_strategy_config should return default if missing."""
        config = {'symbol': 'BTC-USDT'}

        strategy = get_strategy_config(config)

        assert 'tp_pct' in strategy
        assert 'sl_pct' in strategy

    def test_get_risk_config(self, valid_config):
        """get_risk_config should return risk section."""
        risk = get_risk_config(valid_config)

        assert risk['max_daily_loss_pct'] == 5.0

    def test_get_api_config(self, valid_config):
        """get_api_config should return api section."""
        api = get_api_config(valid_config)

        assert api['rate_limit'] == 1200


# ── Integration Tests ─────────────────────────────────────────

class TestConfigIntegration:
    """Test complete load → validate workflow."""

    def test_load_and_validate_valid_config(self, temp_config_file, valid_config):
        """Load and validate valid config should succeed."""
        with open(temp_config_file, 'w', encoding='utf-8') as f:
            yaml.dump(valid_config, f)

        config = load_config(temp_config_file)
        assert validate_config(config) is True

    def test_load_invalid_config_validates_with_error(self, temp_config_file):
        """Config with invalid value should fail validation."""
        invalid = {
            'symbol': 'BTC-USDT',
            'timeframe': '5m',
            'leverage': 0,  # Invalid: must be positive
            'strategy': {'tp_pct': 1.0, 'sl_pct': 1.0}
        }

        with open(temp_config_file, 'w', encoding='utf-8') as f:
            yaml.dump(invalid, f)

        config = load_config(temp_config_file)

        with pytest.raises(ValueError, match="leverage must be positive"):
            validate_config(config)

    def test_partial_config_merges_with_defaults(self, temp_config_file):
        """Partial config file should merge with DEFAULT_CONFIG."""
        partial = {
            'symbol': 'ETH-USDT',
            'leverage': 5
        }

        with open(temp_config_file, 'w', encoding='utf-8') as f:
            yaml.dump(partial, f)

        config = load_config(temp_config_file)

        # Overridden values
        assert config['symbol'] == 'ETH-USDT'
        assert config['leverage'] == 5

        # Default values preserved
        assert 'strategy' in config
        assert 'risk' in config
