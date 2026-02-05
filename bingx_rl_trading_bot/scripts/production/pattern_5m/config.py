"""
Pattern 5m Bot - Configuration Management
Load, validate, and manage bot configuration.
"""

import os
import yaml
import logging
from typing import Dict, Any, List

from .constants import CONFIG_FILE, DEFAULT_CONFIG

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