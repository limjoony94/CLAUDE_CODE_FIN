"""
Configuration Auto-Sync Module for Monitoring Program
====================================================

Purpose: Ensure monitoring program always uses production bot's latest configuration
Strategy: State JSON file as Single Source of Truth (SSOT)

Architecture:
1. Production bot writes configuration to state JSON file
2. Monitoring program reads configuration from state JSON file
3. No hardcoded defaults (except emergency fallback if state file missing)
4. Auto-sync: Configuration changes propagate automatically

Created: 2025-10-30
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime


class ConfigurationSyncError(Exception):
    """Raised when configuration synchronization fails"""
    pass


class ConfigurationValidator:
    """Validates and loads configuration from state JSON file"""

    # Emergency fallback values (ONLY used if state file completely missing)
    # DO NOT UPDATE THESE - They are emergency-only fallbacks
    # Real configuration comes from state JSON file written by production bot
    EMERGENCY_FALLBACK_CONFIG = {
        'long_threshold': 0.80,  # Emergency fallback only
        'short_threshold': 0.80,  # Emergency fallback only
        'gate_threshold': 0.001,
        'ml_exit_threshold_base_long': 0.80,
        'ml_exit_threshold_base_short': 0.80,
        'emergency_stop_loss': 0.03,  # Stored as positive in state, negative in display
        'emergency_max_hold_hours': 10.0,
        'leverage': 4,
        'long_avg_return': 0.0041,
        'short_avg_return': 0.0047,
        'fixed_take_profit': 0.03,
        'trailing_tp_activation': 0.02,
        'trailing_tp_drawdown': 0.10,
        'exit_strategy': 'COMBINED'
    }

    # Required configuration keys
    REQUIRED_KEYS = [
        'long_threshold',
        'short_threshold',
        'gate_threshold',
        'ml_exit_threshold_base_long',
        'ml_exit_threshold_base_short',
        'emergency_stop_loss',
        'emergency_max_hold_hours',
        'leverage'
    ]

    @classmethod
    def load_production_config(cls, state_file: Path) -> Tuple[Dict, str]:
        """
        Load configuration from production bot's state JSON file.

        Args:
            state_file: Path to opportunity_gating_bot_4x_state.json

        Returns:
            Tuple of (config dict, source string)

        Raises:
            ConfigurationSyncError: If state file invalid or configuration incomplete
        """
        # Check state file exists
        if not state_file.exists():
            print("="*80)
            print("⚠️ WARNING: State file not found - Using EMERGENCY FALLBACK configuration")
            print(f"   State file expected: {state_file}")
            print("   This should only happen if production bot has never run.")
            print("   EMERGENCY FALLBACK values may be outdated!")
            print("="*80)
            return cls.EMERGENCY_FALLBACK_CONFIG.copy(), "EMERGENCY_FALLBACK"

        # Load state file
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigurationSyncError(
                f"State file corrupted (invalid JSON): {state_file}\n"
                f"Error: {e}\n"
                f"Cannot auto-sync configuration. Please check production bot status."
            )
        except Exception as e:
            raise ConfigurationSyncError(
                f"Failed to read state file: {state_file}\n"
                f"Error: {e}"
            )

        # Extract configuration from state
        config = state.get('configuration')

        if config is None:
            print("="*80)
            print("⚠️ WARNING: State file exists but has no 'configuration' key")
            print("   This should never happen with current production bot.")
            print("   Using EMERGENCY FALLBACK configuration")
            print("="*80)
            return cls.EMERGENCY_FALLBACK_CONFIG.copy(), "EMERGENCY_FALLBACK"

        # Validate configuration completeness
        missing_keys = [key for key in cls.REQUIRED_KEYS if key not in config]
        if missing_keys:
            raise ConfigurationSyncError(
                f"Configuration incomplete - missing required keys: {missing_keys}\n"
                f"State file: {state_file}\n"
                f"Available keys: {list(config.keys())}\n"
                f"This indicates state file corruption or version mismatch."
            )

        # Validate configuration values
        cls._validate_config_values(config)

        # Configuration successfully loaded from state file
        return config, "STATE_FILE"

    @classmethod
    def _validate_config_values(cls, config: Dict) -> None:
        """
        Validate configuration value ranges.

        Raises:
            ConfigurationSyncError: If values are out of valid range
        """
        # Threshold validations
        for threshold_key in ['long_threshold', 'short_threshold',
                              'ml_exit_threshold_base_long', 'ml_exit_threshold_base_short']:
            value = config.get(threshold_key, 0)
            if not (0.0 <= value <= 1.0):
                raise ConfigurationSyncError(
                    f"Invalid {threshold_key}: {value} (must be 0.0-1.0)"
                )

        # Gate threshold
        gate = config.get('gate_threshold', 0)
        if not (0.0 <= gate <= 0.1):
            raise ConfigurationSyncError(
                f"Invalid gate_threshold: {gate} (must be 0.0-0.1)"
            )

        # Stop loss (stored as positive)
        sl = config.get('emergency_stop_loss', 0)
        if not (0.0 <= sl <= 0.2):
            raise ConfigurationSyncError(
                f"Invalid emergency_stop_loss: {sl} (must be 0.0-0.2, stored as positive)"
            )

        # Max hold hours
        max_hold = config.get('emergency_max_hold_hours', 0)
        if not (1 <= max_hold <= 24):
            raise ConfigurationSyncError(
                f"Invalid emergency_max_hold_hours: {max_hold} (must be 1-24)"
            )

        # Leverage
        leverage = config.get('leverage', 0)
        if leverage not in [1, 2, 3, 4, 5, 10, 20]:
            raise ConfigurationSyncError(
                f"Invalid leverage: {leverage} (must be 1, 2, 3, 4, 5, 10, or 20)"
            )

    @classmethod
    def get_config_display_info(cls, config: Dict, source: str) -> str:
        """
        Generate human-readable configuration display.

        Args:
            config: Configuration dictionary
            source: Configuration source ("STATE_FILE" or "EMERGENCY_FALLBACK")

        Returns:
            Formatted string for display
        """
        source_emoji = "✅" if source == "STATE_FILE" else "⚠️"
        source_text = "State File (Production Bot)" if source == "STATE_FILE" else "Emergency Fallback"

        return f"""
{source_emoji} Configuration Source: {source_text}

Entry Thresholds:
  LONG Entry:  {config['long_threshold']:.2f} ({config['long_threshold']*100:.0f}%)
  SHORT Entry: {config['short_threshold']:.2f} ({config['short_threshold']*100:.0f}%)
  Gate:        {config['gate_threshold']:.4f} ({config['gate_threshold']*100:.2f}%)

Exit Thresholds:
  LONG Exit:   {config['ml_exit_threshold_base_long']:.2f} ({config['ml_exit_threshold_base_long']*100:.0f}%)
  SHORT Exit:  {config['ml_exit_threshold_base_short']:.2f} ({config['ml_exit_threshold_base_short']*100:.0f}%)

Risk Parameters:
  Stop Loss:   -{config['emergency_stop_loss']*100:.1f}% (balance-based)
  Max Hold:    {config['emergency_max_hold_hours']:.1f} hours
  Leverage:    {config['leverage']}x

Expected Returns (for gating):
  LONG:  {config.get('long_avg_return', 0)*100:.2f}%
  SHORT: {config.get('short_avg_return', 0)*100:.2f}%
"""


def load_config_with_sync(state_file: Path) -> Tuple[Dict, str]:
    """
    Convenience function to load configuration with auto-sync.

    This is the main entry point for monitoring program.

    Args:
        state_file: Path to state JSON file

    Returns:
        Tuple of (config dict, source string)

    Raises:
        ConfigurationSyncError: If configuration cannot be loaded
    """
    return ConfigurationValidator.load_production_config(state_file)


def print_config_comparison(old_config: Dict, new_config: Dict) -> None:
    """
    Print side-by-side comparison of configuration changes.

    Args:
        old_config: Previous configuration
        new_config: Current configuration
    """
    print("="*80)
    print("📊 Configuration Change Detected")
    print("="*80)

    # Find changed keys
    all_keys = set(old_config.keys()) | set(new_config.keys())
    changed_keys = [k for k in all_keys if old_config.get(k) != new_config.get(k)]

    if not changed_keys:
        print("✅ No changes detected")
        return

    print(f"\n{len(changed_keys)} parameter(s) changed:\n")

    for key in sorted(changed_keys):
        old_val = old_config.get(key, "N/A")
        new_val = new_config.get(key, "N/A")
        print(f"  {key}:")
        print(f"    Before: {old_val}")
        print(f"    After:  {new_val}")

    print("="*80)


if __name__ == "__main__":
    """Test configuration sync"""
    import sys
    from pathlib import Path

    # Test with actual state file
    project_root = Path(__file__).parent.parent.parent
    state_file = project_root / "results" / "opportunity_gating_bot_4x_state.json"

    print("Configuration Auto-Sync Module - Test")
    print("="*80)
    print(f"State file: {state_file}")
    print(f"Exists: {state_file.exists()}")
    print("="*80)

    try:
        config, source = load_config_with_sync(state_file)
        print(ConfigurationValidator.get_config_display_info(config, source))

        print("✅ Configuration loaded successfully!")
        print(f"   Source: {source}")
        print(f"   Keys: {len(config)}")

    except ConfigurationSyncError as e:
        print(f"❌ Configuration sync failed:")
        print(f"   {e}")
        sys.exit(1)
