"""Pytest configuration for pattern_5m tests.

Sets up sys.path to allow imports from bingx_rl_trading_bot package.
Isolates test state/metrics files from production to prevent contamination.
"""
import sys
import pytest
from pathlib import Path

# Add CLAUDE_CODE_FIN to sys.path to enable:
# from bingx_rl_trading_bot.scripts.production.pattern_5m import ...
#
# Path resolution:
# __file__ = .../CLAUDE_CODE_FIN/bingx_rl_trading_bot/scripts/production/pattern_5m/tests/conftest.py
# .parents[4] = bingx_rl_trading_bot/
# .parents[5] = CLAUDE_CODE_FIN/
PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(autouse=True)
def _isolate_state_files(tmp_path, monkeypatch):
    """Redirect STATE_FILE and METRICS_FILE to temp dir during tests.

    Prevents any test (even those that forget to mock save_state) from
    writing to the production state/metrics files.

    Works because save_state/load_state/save_metrics/load_metrics now
    resolve STATE_FILE/METRICS_FILE at call time (default=None → lookup),
    so monkeypatching the module-level variable takes effect.
    """
    test_state = str(tmp_path / "test_state.json")
    test_metrics = str(tmp_path / "test_metrics.json")

    monkeypatch.setattr(
        'bingx_rl_trading_bot.scripts.production.pattern_5m.state.STATE_FILE',
        test_state,
    )
    monkeypatch.setattr(
        'bingx_rl_trading_bot.scripts.production.pattern_5m.state.METRICS_FILE',
        test_metrics,
    )
