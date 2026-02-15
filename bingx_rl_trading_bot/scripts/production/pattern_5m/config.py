"""
Pattern 5m Bot - Configuration Management
Load, validate, and manage bot configuration.
"""

import os
import json
import yaml
import logging
from datetime import datetime
from typing import Dict, Any, List

from .constants import CONFIG_FILE, DEFAULT_CONFIG, DYNAMIC_PATTERNS_FILE

logger = logging.getLogger('pattern_5m')


def load_config(config_file: str = CONFIG_FILE) -> Dict[str, Any]:
    """
    Load configuration from YAML file, merging with defaults.

    Args:
        config_file: Path to YAML configuration file

    Returns:
        Merged configuration dictionary
    """
    config = _deep_copy_config(DEFAULT_CONFIG)

    if os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                file_config = yaml.safe_load(f)
                if file_config:
                    config = _merge_config(config, file_config)
            logger.info(f"Config loaded from {config_file}")
        except yaml.YAMLError as e:
            logger.warning(f"Failed to parse config YAML: {e}, using defaults")
        except (IOError, OSError) as e:
            logger.warning(f"Failed to read config file: {e}, using defaults")
        except Exception as e:
            logger.warning(f"Failed to load config: {e}, using defaults")

    return config


def _deep_copy_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create a deep copy of configuration dictionary."""
    result = {}
    for key, value in config.items():
        if isinstance(value, dict):
            result[key] = _deep_copy_config(value)
        elif isinstance(value, list):
            result[key] = value.copy()
        else:
            result[key] = value
    return result


def _merge_config(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override config into base config."""
    result = _deep_copy_config(base)

    for key, value in override.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = _merge_config(result[key], value)
        else:
            result[key] = value

    return result


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration for required fields and valid values.

    Args:
        config: Configuration dictionary to validate

    Returns:
        True if validation passes

    Raises:
        ValueError: If validation fails
    """
    errors: List[str] = []

    # Required top-level fields
    required_fields = ['symbol', 'timeframe', 'leverage']
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")

    # Validate strategy section
    strategy = config.get('strategy', {})
    if 'tp_pct' not in strategy:
        errors.append("Missing strategy.tp_pct")
    if 'sl_pct' not in strategy:
        errors.append("Missing strategy.sl_pct")

    # Validate numeric ranges
    if config.get('leverage', 0) <= 0:
        errors.append("leverage must be positive")
    if strategy.get('tp_pct', 0) <= 0:
        errors.append("strategy.tp_pct must be positive")
    if strategy.get('sl_pct', 0) <= 0:
        errors.append("strategy.sl_pct must be positive")

    # Validate risk section
    risk = config.get('risk', {})
    if risk.get('max_daily_loss_pct', 0) <= 0:
        errors.append("risk.max_daily_loss_pct must be positive")

    if errors:
        error_msg = "Config validation failed:\n  - " + "\n  - ".join(errors)
        raise ValueError(error_msg)

    logger.info("✅ Config validation passed")
    return True


def get_strategy_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract strategy configuration with defaults (returns a copy)."""
    strategy = config.get('strategy', DEFAULT_CONFIG['strategy'])
    return _deep_copy_config(strategy)


def get_risk_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract risk configuration with defaults."""
    return config.get('risk', DEFAULT_CONFIG['risk'])


def get_api_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract API configuration with defaults."""
    return config.get('api', DEFAULT_CONFIG['api'])


def load_dynamic_patterns(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load dynamic patterns from JSON if pattern_source is 'dynamic'.

    If pattern_source is 'static' (default) or missing, returns config unchanged.
    On any failure in dynamic mode, falls back to static with a warning.

    Injects into config:
      - strategy.long_patterns / short_patterns (from JSON)
      - _dynamic_tpsl_universal: True (flags universal TP/SL mode)
      - _dynamic_tp / _dynamic_sl (universal values from JSON)
    """
    pattern_source = config.get('strategy', {}).get('pattern_source', 'static')

    if pattern_source != 'dynamic':
        return config

    json_path = DYNAMIC_PATTERNS_FILE
    logger.info(f"Dynamic pattern mode: loading from {json_path}")

    try:
        if not os.path.exists(json_path):
            logger.error(f"Dynamic patterns file not found: {json_path} — falling back to static")
            return config

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Validate required fields
        patterns = data.get('patterns')
        if not patterns or 'long' not in patterns or 'short' not in patterns:
            logger.error("Dynamic patterns JSON missing 'patterns.long/short' — falling back to static")
            return config

        tp_sl_mode = data.get('tp_sl_mode')
        if tp_sl_mode not in ('universal', 'per_pattern'):
            logger.error(f"Unsupported tp_sl_mode '{tp_sl_mode}' — falling back to static")
            return config

        # Staleness check (warn if > 30 days old)
        generated_at = data.get('generated_at', '')
        if generated_at:
            try:
                gen_time = datetime.fromisoformat(generated_at)
                age_days = (datetime.now() - gen_time).days
                if age_days > 30:
                    logger.warning(f"Dynamic patterns JSON is {age_days} days old (generated: {generated_at})")
            except (ValueError, TypeError):
                logger.warning(f"Could not parse generated_at: {generated_at}")

        # Inject patterns into config
        long_patterns = patterns['long']
        short_patterns = patterns['short']
        config['strategy']['long_patterns'] = long_patterns
        config['strategy']['short_patterns'] = short_patterns

        total = len(long_patterns) + len(short_patterns)
        logger.info(f"Dynamic patterns loaded: {len(long_patterns)}L + {len(short_patterns)}S = {total}")

        # Mode-specific TP/SL injection
        if tp_sl_mode == 'universal':
            uni_tp = data.get('universal_tp')
            uni_sl = data.get('universal_sl')
            if not uni_tp or not uni_sl or uni_tp <= 0 or uni_sl <= 0:
                logger.error("Invalid universal_tp/universal_sl in JSON — falling back to static")
                return config
            config['_dynamic_tpsl_universal'] = True
            config['_dynamic_tp'] = uni_tp
            config['_dynamic_sl'] = uni_sl
            logger.info(f"Universal TP/SL: {uni_tp}% / {uni_sl}%")
        elif tp_sl_mode == 'per_pattern':
            patterns_tpsl = data.get('patterns_tpsl', {})
            if not patterns_tpsl:
                logger.error("per_pattern mode but no patterns_tpsl — falling back to static")
                return config
            config['_dynamic_tpsl_per_pattern'] = True
            config['_dynamic_patterns_tpsl'] = patterns_tpsl
            logger.info(f"Per-pattern TP/SL loaded: {len(patterns_tpsl)} patterns")

        bs = data.get('backtest_summary', {})
        if bs:
            logger.info(f"Backtest: {bs.get('total_trades', '?')} trades, "
                         f"WR {bs.get('win_rate', '?')}%, PnL {bs.get('pnl_pct', '?')}%")

        return config

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse dynamic patterns JSON: {e} — falling back to static")
        return config
    except Exception as e:
        logger.error(f"Failed to load dynamic patterns: {e} — falling back to static")
        return config