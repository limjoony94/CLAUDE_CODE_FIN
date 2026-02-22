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

from .constants import CONFIG_FILE, DEFAULT_CONFIG, DYNAMIC_PATTERNS_FILE, MAX_ALLOWED_POSITIONS, VALID_POSITION_MODES

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


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration for required fields and valid values.

    Args:
        config: Configuration dictionary to validate

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

    # Validate max_positions
    max_pos = config.get('max_positions', 1)
    if not isinstance(max_pos, int) or max_pos < 1 or max_pos > MAX_ALLOWED_POSITIONS:
        errors.append(f"max_positions must be 1~{MAX_ALLOWED_POSITIONS}, got {max_pos}")

    # Validate position_mode
    pos_mode = config.get('position_mode', 'one_way')
    if pos_mode not in VALID_POSITION_MODES:
        errors.append(f"position_mode must be one of {VALID_POSITION_MODES}, got '{pos_mode}'")

    # Validate risk section
    risk = config.get('risk', {})
    if risk.get('max_daily_loss_pct', 0) <= 0:
        errors.append("risk.max_daily_loss_pct must be positive")

    if errors:
        error_msg = "Config validation failed:\n  - " + "\n  - ".join(errors)
        raise ValueError(error_msg)

    logger.info("✅ Config validation passed")


def get_strategy_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract strategy configuration with defaults (returns a copy)."""
    strategy = config.get('strategy', DEFAULT_CONFIG['strategy'])
    return _deep_copy_config(strategy)


def get_risk_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract risk configuration with defaults."""
    return _deep_copy_config(config.get('risk', DEFAULT_CONFIG['risk']))


def get_api_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract API configuration with defaults."""
    return _deep_copy_config(config.get('api', DEFAULT_CONFIG['api']))


def _log_fallback_warning(reason: str) -> None:
    """Log a prominent warning when falling back to static patterns."""
    logger.error(f"{'!'*60}")
    logger.error(f"  DYNAMIC PATTERN LOAD FAILED — FALLING BACK TO STATIC")
    logger.error(f"  Reason: {reason}")
    logger.error(f"  The bot will trade with STATIC patterns from constants.py")
    logger.error(f"{'!'*60}")


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
            _log_fallback_warning(f"File not found: {json_path}")
            return config

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Validate required fields
        patterns = data.get('patterns')
        if not patterns or 'long' not in patterns or 'short' not in patterns:
            _log_fallback_warning("JSON missing 'patterns.long/short'")
            return config

        tp_sl_mode = data.get('tp_sl_mode')

        # Auto-infer tp_sl_mode from JSON structure when field is missing
        if not tp_sl_mode:
            if data.get('patterns_tpsl'):
                tp_sl_mode = 'per_pattern'
                logger.warning(f"tp_sl_mode missing — auto-inferred 'per_pattern' from patterns_tpsl")
            elif data.get('universal_tp') and data.get('universal_sl'):
                tp_sl_mode = 'universal'
                logger.warning(f"tp_sl_mode missing — auto-inferred 'universal' from universal_tp/sl")

        if tp_sl_mode not in ('universal', 'per_pattern'):
            _log_fallback_warning(f"Unsupported tp_sl_mode '{tp_sl_mode}'")
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

        # Extract patterns (but DON'T inject into config yet — validate TP/SL first)
        long_patterns = patterns['long']
        short_patterns = patterns['short']
        total = len(long_patterns) + len(short_patterns)

        # Mode-specific TP/SL validation (BEFORE injecting patterns)
        if tp_sl_mode == 'universal':
            uni_tp = data.get('universal_tp')
            uni_sl = data.get('universal_sl')
            if not uni_tp or not uni_sl or uni_tp <= 0 or uni_sl <= 0:
                _log_fallback_warning("Invalid universal_tp/universal_sl in JSON")
                return config
            config['_dynamic_tpsl_universal'] = True
            config['_dynamic_tp'] = uni_tp
            config['_dynamic_sl'] = uni_sl
            logger.info(f"Universal TP/SL: {uni_tp}% / {uni_sl}%")
        elif tp_sl_mode == 'per_pattern':
            patterns_tpsl = data.get('patterns_tpsl', {})
            if not patterns_tpsl:
                _log_fallback_warning("per_pattern mode but no patterns_tpsl")
                return config

            # Validate per-pattern TP/SL values
            # patterns_tpsl uses plain pattern names (e.g. "BD-BD-BU"), not direction-suffixed
            all_pattern_keys = set()
            for p in long_patterns:
                all_pattern_keys.add(p)
            for p in short_patterns:
                all_pattern_keys.add(p)

            invalid = []
            for key, vals in patterns_tpsl.items():
                if not isinstance(vals, (list, tuple)) or len(vals) < 2:
                    invalid.append(f"{key}: invalid format {vals}")
                elif vals[0] <= 0 or vals[1] <= 0:
                    invalid.append(f"{key}: TP={vals[0]} SL={vals[1]} (must be > 0)")
                elif vals[0] > 10 or vals[1] > 10:
                    invalid.append(f"{key}: TP={vals[0]}% SL={vals[1]}% (> 10% is suspicious)")

            if invalid:
                for msg in invalid[:5]:
                    logger.warning(f"  TP/SL validation: {msg}")
                if len(invalid) > 5:
                    logger.warning(f"  ... and {len(invalid) - 5} more")

            # Check for missing tpsl entries
            missing = all_pattern_keys - set(patterns_tpsl.keys())
            if missing:
                logger.warning(f"  {len(missing)} patterns have no TP/SL entry (will use config defaults): {list(missing)[:5]}")

            config['_dynamic_tpsl_per_pattern'] = True
            config['_dynamic_patterns_tpsl'] = patterns_tpsl
            logger.info(f"Per-pattern TP/SL loaded: {len(patterns_tpsl)} patterns")

        # TP/SL validated — NOW inject patterns into config
        config['strategy']['long_patterns'] = long_patterns
        config['strategy']['short_patterns'] = short_patterns
        logger.info(f"Dynamic patterns loaded: {len(long_patterns)}L + {len(short_patterns)}S = {total}")

        # Inject per-pattern stats for confidence scoring (WR, trades, edge)
        pattern_details = data.get('pattern_details', {})
        if pattern_details:
            # Build {(pattern, direction): {wr, trades, edge}} lookup
            dynamic_stats = {}
            for key, details in pattern_details.items():
                pat = details.get('pattern', '')
                direction = details.get('direction', '')
                if pat and direction:
                    dynamic_stats[(pat, direction)] = {
                        'wr': details.get('wr', 50.0),
                        'trades': details.get('trades', 0),
                        'edge': details.get('edge', 0.0),
                    }
            config['_dynamic_pattern_stats'] = dynamic_stats
            logger.info(f"Dynamic pattern stats loaded: {len(dynamic_stats)} entries for confidence scoring")

        bs = data.get('backtest_summary', {})
        if bs:
            logger.info(f"Backtest: {bs.get('total_trades', '?')} trades, "
                         f"WR {bs.get('win_rate', '?')}%, PnL {bs.get('pnl_pct', '?')}%")

        return config

    except json.JSONDecodeError as e:
        _log_fallback_warning(f"JSON parse error: {e}")
        return config
    except Exception as e:
        _log_fallback_warning(f"Unexpected error: {e}")
        return config