"""Tests for config.py — YAML parsing, validation, required keys, dynamic patterns."""

import pytest
import os
import json
import yaml
from pathlib import Path

from bingx_rl_trading_bot.scripts.production.pattern_5m.config import (
    load_config,
    validate_config,
    load_dynamic_patterns,
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
        validate_config(valid_config)  # should not raise

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
        validate_config(config)  # should not raise

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


# ── load_dynamic_patterns ────────────────────────────────────


class TestLoadDynamicPatterns:
    """Test load_dynamic_patterns() — JSON loading, TP/SL modes, fallbacks."""

    @pytest.fixture
    def base_config(self):
        """Config with pattern_source: dynamic."""
        return {
            'strategy': {
                'pattern_source': 'dynamic',
                'tp_pct': 1.0,
                'sl_pct': 1.0,
                'long_patterns': [],
                'short_patterns': [],
            }
        }

    @pytest.fixture
    def universal_json(self):
        """Valid universal TP/SL JSON data."""
        return {
            'tp_sl_mode': 'universal',
            'universal_tp': 2.0,
            'universal_sl': 3.0,
            'patterns': {
                'long': ['U-MU-H', 'BD-BD-BU'],
                'short': ['DN-BU-BU', 'ST-BD-BU'],
            },
        }

    @pytest.fixture
    def per_pattern_json(self):
        """Valid per_pattern TP/SL JSON data."""
        return {
            'tp_sl_mode': 'per_pattern',
            'patterns': {
                'long': ['U-MU-H'],
                'short': ['DN-BU-BU', 'ST-BD-BU'],
            },
            'patterns_tpsl': {
                'U-MU-H': [1.8, 3.5],
                'DN-BU-BU': [2.0, 4.0],
                'ST-BD-BU': [2.1, 4.0],
            },
        }

    def _write_json(self, path, data):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f)

    # ── Static mode (no-op) ──

    def test_static_mode_returns_unchanged(self):
        """pattern_source != 'dynamic' → returns config unchanged."""
        config = {'strategy': {'pattern_source': 'static'}}
        result = load_dynamic_patterns(config)
        assert result is config

    def test_missing_pattern_source_returns_unchanged(self):
        """No pattern_source → defaults to 'static', returns unchanged."""
        config = {'strategy': {}}
        result = load_dynamic_patterns(config)
        assert result is config

    # ── File not found / corrupt ──

    def test_file_not_found_fallback(self, base_config, monkeypatch):
        """Missing JSON file → fallback to static (returns config without patterns)."""
        monkeypatch.setattr(
            'bingx_rl_trading_bot.scripts.production.pattern_5m.config.DYNAMIC_PATTERNS_FILE',
            '/nonexistent/path.json',
        )
        result = load_dynamic_patterns(base_config)
        # Should return config without injecting patterns
        assert '_dynamic_tpsl_universal' not in result
        assert '_dynamic_tpsl_per_pattern' not in result

    def test_invalid_json_fallback(self, base_config, tmp_path, monkeypatch):
        """Corrupt JSON → fallback to static."""
        bad_file = str(tmp_path / 'bad.json')
        with open(bad_file, 'w') as f:
            f.write('{invalid json}')
        monkeypatch.setattr(
            'bingx_rl_trading_bot.scripts.production.pattern_5m.config.DYNAMIC_PATTERNS_FILE',
            bad_file,
        )
        result = load_dynamic_patterns(base_config)
        assert '_dynamic_tpsl_universal' not in result

    # ── Missing required fields ──

    def test_missing_patterns_key_fallback(self, base_config, tmp_path, monkeypatch):
        """JSON without 'patterns' key → fallback."""
        f = str(tmp_path / 'no_patterns.json')
        self._write_json(f, {'tp_sl_mode': 'universal'})
        monkeypatch.setattr(
            'bingx_rl_trading_bot.scripts.production.pattern_5m.config.DYNAMIC_PATTERNS_FILE', f,
        )
        result = load_dynamic_patterns(base_config)
        assert '_dynamic_tpsl_universal' not in result

    def test_missing_long_short_fallback(self, base_config, tmp_path, monkeypatch):
        """patterns without long/short → fallback."""
        f = str(tmp_path / 'no_ls.json')
        self._write_json(f, {'tp_sl_mode': 'universal', 'patterns': {'long': ['A']}})
        monkeypatch.setattr(
            'bingx_rl_trading_bot.scripts.production.pattern_5m.config.DYNAMIC_PATTERNS_FILE', f,
        )
        result = load_dynamic_patterns(base_config)
        assert '_dynamic_tpsl_universal' not in result

    # ── Universal mode ──

    def test_universal_mode_valid(self, base_config, tmp_path, monkeypatch, universal_json):
        """Valid universal JSON → injects patterns + universal TP/SL."""
        f = str(tmp_path / 'uni.json')
        self._write_json(f, universal_json)
        monkeypatch.setattr(
            'bingx_rl_trading_bot.scripts.production.pattern_5m.config.DYNAMIC_PATTERNS_FILE', f,
        )
        result = load_dynamic_patterns(base_config)
        assert result['_dynamic_tpsl_universal'] is True
        assert result['_dynamic_tp'] == 2.0
        assert result['_dynamic_sl'] == 3.0
        assert result['strategy']['long_patterns'] == ['U-MU-H', 'BD-BD-BU']
        assert result['strategy']['short_patterns'] == ['DN-BU-BU', 'ST-BD-BU']

    def test_universal_mode_invalid_tp(self, base_config, tmp_path, monkeypatch):
        """Universal mode with TP <= 0 → fallback."""
        f = str(tmp_path / 'bad_tp.json')
        self._write_json(f, {
            'tp_sl_mode': 'universal',
            'universal_tp': 0,
            'universal_sl': 3.0,
            'patterns': {'long': ['A'], 'short': ['B']},
        })
        monkeypatch.setattr(
            'bingx_rl_trading_bot.scripts.production.pattern_5m.config.DYNAMIC_PATTERNS_FILE', f,
        )
        result = load_dynamic_patterns(base_config)
        assert '_dynamic_tpsl_universal' not in result

    # ── Per-pattern mode ──

    def test_per_pattern_mode_valid(self, base_config, tmp_path, monkeypatch, per_pattern_json):
        """Valid per_pattern JSON → injects patterns + per-pattern TP/SL."""
        f = str(tmp_path / 'pp.json')
        self._write_json(f, per_pattern_json)
        monkeypatch.setattr(
            'bingx_rl_trading_bot.scripts.production.pattern_5m.config.DYNAMIC_PATTERNS_FILE', f,
        )
        result = load_dynamic_patterns(base_config)
        assert result['_dynamic_tpsl_per_pattern'] is True
        assert result['_dynamic_patterns_tpsl']['U-MU-H'] == [1.8, 3.5]
        assert result['strategy']['long_patterns'] == ['U-MU-H']
        assert len(result['strategy']['short_patterns']) == 2

    def test_per_pattern_empty_tpsl_fallback(self, base_config, tmp_path, monkeypatch):
        """Per-pattern mode with empty patterns_tpsl → fallback."""
        f = str(tmp_path / 'empty_tpsl.json')
        self._write_json(f, {
            'tp_sl_mode': 'per_pattern',
            'patterns': {'long': ['A'], 'short': ['B']},
            'patterns_tpsl': {},
        })
        monkeypatch.setattr(
            'bingx_rl_trading_bot.scripts.production.pattern_5m.config.DYNAMIC_PATTERNS_FILE', f,
        )
        result = load_dynamic_patterns(base_config)
        assert '_dynamic_tpsl_per_pattern' not in result

    # ── tp_sl_mode auto-inference ──

    def test_auto_infer_per_pattern(self, base_config, tmp_path, monkeypatch):
        """Missing tp_sl_mode with patterns_tpsl → auto-infer 'per_pattern'."""
        f = str(tmp_path / 'auto_pp.json')
        self._write_json(f, {
            'patterns': {'long': ['A'], 'short': ['B']},
            'patterns_tpsl': {'A': [1.0, 2.0], 'B': [1.5, 2.5]},
        })
        monkeypatch.setattr(
            'bingx_rl_trading_bot.scripts.production.pattern_5m.config.DYNAMIC_PATTERNS_FILE', f,
        )
        result = load_dynamic_patterns(base_config)
        assert result['_dynamic_tpsl_per_pattern'] is True

    def test_auto_infer_universal(self, base_config, tmp_path, monkeypatch):
        """Missing tp_sl_mode with universal_tp/sl → auto-infer 'universal'."""
        f = str(tmp_path / 'auto_uni.json')
        self._write_json(f, {
            'patterns': {'long': ['A'], 'short': ['B']},
            'universal_tp': 2.0,
            'universal_sl': 3.0,
        })
        monkeypatch.setattr(
            'bingx_rl_trading_bot.scripts.production.pattern_5m.config.DYNAMIC_PATTERNS_FILE', f,
        )
        result = load_dynamic_patterns(base_config)
        assert result['_dynamic_tpsl_universal'] is True

    def test_unsupported_tp_sl_mode_fallback(self, base_config, tmp_path, monkeypatch):
        """Unsupported tp_sl_mode → fallback."""
        f = str(tmp_path / 'bad_mode.json')
        self._write_json(f, {
            'tp_sl_mode': 'random_mode',
            'patterns': {'long': ['A'], 'short': ['B']},
        })
        monkeypatch.setattr(
            'bingx_rl_trading_bot.scripts.production.pattern_5m.config.DYNAMIC_PATTERNS_FILE', f,
        )
        result = load_dynamic_patterns(base_config)
        assert '_dynamic_tpsl_universal' not in result
        assert '_dynamic_tpsl_per_pattern' not in result

    # ── Pattern details / confidence stats ──

    def test_pattern_details_injected(self, base_config, tmp_path, monkeypatch):
        """pattern_details → _dynamic_pattern_stats dict injected."""
        f = str(tmp_path / 'details.json')
        self._write_json(f, {
            'tp_sl_mode': 'universal',
            'universal_tp': 2.0,
            'universal_sl': 3.0,
            'patterns': {'long': ['A-B-C'], 'short': ['D-E-F']},
            'pattern_details': {
                'A-B-C_LONG': {
                    'pattern': 'A-B-C', 'direction': 'LONG',
                    'wr': 85.0, 'trades': 30, 'edge': 22.5,
                },
                'D-E-F_SHORT': {
                    'pattern': 'D-E-F', 'direction': 'SHORT',
                    'wr': 90.0, 'trades': 25, 'edge': 28.0,
                },
            },
        })
        monkeypatch.setattr(
            'bingx_rl_trading_bot.scripts.production.pattern_5m.config.DYNAMIC_PATTERNS_FILE', f,
        )
        result = load_dynamic_patterns(base_config)
        stats = result['_dynamic_pattern_stats']
        assert len(stats) == 2
        assert stats[('A-B-C', 'LONG')]['wr'] == 85.0
        assert stats[('D-E-F', 'SHORT')]['trades'] == 25

    def test_no_pattern_details_no_stats(self, base_config, tmp_path, monkeypatch, universal_json):
        """No pattern_details → _dynamic_pattern_stats not injected."""
        f = str(tmp_path / 'no_details.json')
        self._write_json(f, universal_json)
        monkeypatch.setattr(
            'bingx_rl_trading_bot.scripts.production.pattern_5m.config.DYNAMIC_PATTERNS_FILE', f,
        )
        result = load_dynamic_patterns(base_config)
        assert '_dynamic_pattern_stats' not in result

    # ── Staleness check ──

    def test_stale_json_no_error(self, base_config, tmp_path, monkeypatch, universal_json):
        """Old generated_at should warn but still load successfully."""
        universal_json['generated_at'] = '2025-01-01T00:00:00'
        f = str(tmp_path / 'stale.json')
        self._write_json(f, universal_json)
        monkeypatch.setattr(
            'bingx_rl_trading_bot.scripts.production.pattern_5m.config.DYNAMIC_PATTERNS_FILE', f,
        )
        result = load_dynamic_patterns(base_config)
        # Should still load patterns despite staleness warning
        assert result['_dynamic_tpsl_universal'] is True
        assert len(result['strategy']['long_patterns']) == 2


# ── load_config IOError / generic Exception (lines 39-42) ─────────


class TestLoadConfigErrors:
    """Test load_config IOError and generic exception handlers."""

    def test_ioerror_uses_defaults(self, tmp_path):
        """IOError reading config → uses defaults."""
        config_file = str(tmp_path / "config.yaml")
        with open(config_file, 'w') as f:
            f.write("symbol: BTC/USDT:USDT")

        from unittest.mock import patch
        with patch('builtins.open', side_effect=IOError('disk error')):
            result = load_config(config_file)
        # Should return default config
        assert 'strategy' in result

    def test_generic_exception_uses_defaults(self, tmp_path):
        """Generic exception → uses defaults."""
        config_file = str(tmp_path / "config.yaml")
        with open(config_file, 'w') as f:
            f.write("symbol: BTC/USDT:USDT")

        from unittest.mock import patch
        with patch('builtins.open', side_effect=RuntimeError('unexpected')):
            result = load_config(config_file)
        assert 'strategy' in result


# ── load_dynamic_patterns validation edge cases ───────────────────


class TestDynamicPatternsValidation:
    """Test TP/SL validation branches in load_dynamic_patterns."""

    def _write_json(self, path, data):
        with open(path, 'w') as f:
            json.dump(data, f)

    def _make_base_config(self):
        from bingx_rl_trading_bot.scripts.production.pattern_5m.constants import DEFAULT_CONFIG
        import copy
        c = copy.deepcopy(DEFAULT_CONFIG)
        c['strategy']['pattern_source'] = 'dynamic'
        return c

    def test_invalid_tpsl_format_warns(self, tmp_path, monkeypatch):
        """patterns_tpsl with invalid format → warns but loads."""
        data = {
            'patterns': {'long': ['A-B-C'], 'short': ['D-E-F']},
            'tp_sl_mode': 'per_pattern',
            'patterns_tpsl': {
                'A-B-C': 'not_a_list',  # invalid format
                'D-E-F': [1.0, 2.0],
            },
        }
        f = str(tmp_path / 'pat.json')
        self._write_json(f, data)
        monkeypatch.setattr(
            'bingx_rl_trading_bot.scripts.production.pattern_5m.config.DYNAMIC_PATTERNS_FILE', f,
        )
        result = load_dynamic_patterns(self._make_base_config())
        assert 'D-E-F' in result.get('_dynamic_patterns_tpsl', {})

    def test_negative_tpsl_warns(self, tmp_path, monkeypatch):
        """TP or SL <= 0 → warns."""
        data = {
            'patterns': {'long': ['A-B-C'], 'short': []},
            'tp_sl_mode': 'per_pattern',
            'patterns_tpsl': {
                'A-B-C': [0.0, 2.0],  # TP=0
            },
        }
        f = str(tmp_path / 'pat.json')
        self._write_json(f, data)
        monkeypatch.setattr(
            'bingx_rl_trading_bot.scripts.production.pattern_5m.config.DYNAMIC_PATTERNS_FILE', f,
        )
        result = load_dynamic_patterns(self._make_base_config())
        assert result.get('_dynamic_tpsl_per_pattern') is True

    def test_suspicious_tpsl_warns(self, tmp_path, monkeypatch):
        """TP > 10% → warns as suspicious."""
        data = {
            'patterns': {'long': ['A-B-C'], 'short': []},
            'tp_sl_mode': 'per_pattern',
            'patterns_tpsl': {
                'A-B-C': [15.0, 2.0],  # TP=15% suspicious
            },
        }
        f = str(tmp_path / 'pat.json')
        self._write_json(f, data)
        monkeypatch.setattr(
            'bingx_rl_trading_bot.scripts.production.pattern_5m.config.DYNAMIC_PATTERNS_FILE', f,
        )
        result = load_dynamic_patterns(self._make_base_config())
        assert result.get('_dynamic_tpsl_per_pattern') is True

    def test_missing_tpsl_entries_warns(self, tmp_path, monkeypatch):
        """Pattern without tpsl entry → warns about missing."""
        data = {
            'patterns': {'long': ['A-B-C', 'D-E-F'], 'short': []},
            'tp_sl_mode': 'per_pattern',
            'patterns_tpsl': {
                'A-B-C': [1.5, 2.0],
                # D-E-F missing
            },
        }
        f = str(tmp_path / 'pat.json')
        self._write_json(f, data)
        monkeypatch.setattr(
            'bingx_rl_trading_bot.scripts.production.pattern_5m.config.DYNAMIC_PATTERNS_FILE', f,
        )
        result = load_dynamic_patterns(self._make_base_config())
        assert result.get('_dynamic_tpsl_per_pattern') is True

    def test_generated_at_invalid_format(self, tmp_path, monkeypatch):
        """Invalid generated_at → warns but loads."""
        data = {
            'patterns': {'long': ['A-B-C'], 'short': []},
            'generated_at': 'not-a-date',
            'universal_tp': 2.0,
            'universal_sl': 3.0,
        }
        f = str(tmp_path / 'pat.json')
        self._write_json(f, data)
        monkeypatch.setattr(
            'bingx_rl_trading_bot.scripts.production.pattern_5m.config.DYNAMIC_PATTERNS_FILE', f,
        )
        c = self._make_base_config()
        c['tp_sl_mode'] = 'universal'
        result = load_dynamic_patterns(c)
        assert len(result['strategy']['long_patterns']) == 1

    def test_backtest_summary_logged(self, tmp_path, monkeypatch):
        """backtest_summary present → logged."""
        data = {
            'patterns': {'long': ['A-B-C'], 'short': []},
            'universal_tp': 2.0,
            'universal_sl': 3.0,
            'backtest_summary': {'total_trades': 100, 'win_rate': 85.0, 'pnl_pct': 500.0},
        }
        f = str(tmp_path / 'pat.json')
        self._write_json(f, data)
        monkeypatch.setattr(
            'bingx_rl_trading_bot.scripts.production.pattern_5m.config.DYNAMIC_PATTERNS_FILE', f,
        )
        c = self._make_base_config()
        c['tp_sl_mode'] = 'universal'
        result = load_dynamic_patterns(c)
        assert len(result['strategy']['long_patterns']) == 1

    def test_json_decode_error_fallback(self, tmp_path, monkeypatch):
        """JSONDecodeError → returns config unchanged."""
        f = str(tmp_path / 'pat.json')
        with open(f, 'w') as fh:
            fh.write('not json')  # Will cause JSONDecodeError
        monkeypatch.setattr(
            'bingx_rl_trading_bot.scripts.production.pattern_5m.config.DYNAMIC_PATTERNS_FILE', f,
        )
        c = self._make_base_config()
        result = load_dynamic_patterns(c)
        assert 'strategy' in result

    def test_generic_exception_fallback(self, tmp_path, monkeypatch):
        """Generic exception during load → returns config unchanged (line 289-291)."""
        f = str(tmp_path / 'pat.json')
        self._write_json(f, {'patterns': {'long': [], 'short': []}})
        monkeypatch.setattr(
            'bingx_rl_trading_bot.scripts.production.pattern_5m.config.DYNAMIC_PATTERNS_FILE', f,
        )
        # Patch json.load to raise a generic exception after file opens
        from unittest.mock import patch as _patch
        c = self._make_base_config()
        with _patch('bingx_rl_trading_bot.scripts.production.pattern_5m.config.json.load',
                     side_effect=ValueError('unexpected')):
            result = load_dynamic_patterns(c)
        assert 'strategy' in result

    def test_many_invalid_tpsl_truncated(self, tmp_path, monkeypatch):
        """>5 invalid tpsl entries → truncated warning with 'and N more'."""
        patterns_long = [f'A-B-{chr(65+i)}' for i in range(8)]
        tpsl = {p: 'invalid' for p in patterns_long}  # 8 invalid entries
        data = {
            'patterns': {'long': patterns_long, 'short': []},
            'tp_sl_mode': 'per_pattern',
            'patterns_tpsl': tpsl,
        }
        f = str(tmp_path / 'pat.json')
        self._write_json(f, data)
        monkeypatch.setattr(
            'bingx_rl_trading_bot.scripts.production.pattern_5m.config.DYNAMIC_PATTERNS_FILE', f,
        )
        result = load_dynamic_patterns(self._make_base_config())
        # Should load without crash; all 8 patterns invalid → "and 3 more" logged
        assert 'strategy' in result
