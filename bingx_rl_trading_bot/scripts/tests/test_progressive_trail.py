"""Progressive Trail (v4.8.0) unit tests.

Validates signals.py progressive_trail branch:
  enabled=false → behaves like fixed trail_K
  enabled=true + best_pnl < threshold → uses trail_K
  enabled=true + best_pnl >= threshold → uses trail_K_post (tighter)
"""
import math
import pytest

from scripts.production.c1_breakout.signals import C1BreakoutSignal


BASE_CFG = {
    'channel_period': 15,
    'body_min_ratio': 0.60,
    'atr_period': 14,
    'trail_K': 2.5,
    'max_sl_atr': 4.0,
    'emergency_sl_pct': 3.0,
    'max_hold_bars': 192,
    'sl_min_pct': 0.15,
    'sl_max_pct': 3.0,
    'min_bars_between': 2,
    'trail_activation_pct': 0.05,
}


def make_signal(progressive_enabled=False, threshold_pct=0.9, tk_post=0.5):
    cfg = dict(BASE_CFG)
    cfg['progressive_trail'] = {
        'enabled': progressive_enabled,
        'threshold_pct': threshold_pct,
        'trail_K_post': tk_post,
    }
    return C1BreakoutSignal(cfg)


def test_progressive_disabled_uses_base_trail_k():
    """When enabled=false, get_effective_trail_k returns trail_K regardless of best_pnl."""
    sig = make_signal(progressive_enabled=False)
    assert sig.get_effective_trail_k(0.01) == 2.5
    assert sig.get_effective_trail_k(0.5) == 2.5
    assert sig.get_effective_trail_k(1.5) == 2.5
    assert sig.get_effective_trail_k(5.0) == 2.5


def test_progressive_enabled_below_threshold_uses_base_k():
    """best_pnl < threshold_pct → trail_K (2.5)."""
    sig = make_signal(progressive_enabled=True, threshold_pct=0.9, tk_post=0.5)
    assert sig.get_effective_trail_k(0.0) == 2.5
    assert sig.get_effective_trail_k(0.3) == 2.5
    assert sig.get_effective_trail_k(0.89) == 2.5  # just below


def test_progressive_enabled_above_threshold_uses_post_k():
    """best_pnl >= threshold_pct → trail_K_post (0.5)."""
    sig = make_signal(progressive_enabled=True, threshold_pct=0.9, tk_post=0.5)
    assert sig.get_effective_trail_k(0.9) == 0.5  # exact boundary
    assert sig.get_effective_trail_k(1.0) == 0.5
    assert sig.get_effective_trail_k(2.5) == 0.5


def test_progressive_check_exit_long_applies_post_k_after_threshold():
    """LONG: best_pnl > threshold → tighter trail dist → earlier exit possible."""
    sig = make_signal(progressive_enabled=True, threshold_pct=0.9, tk_post=0.5)
    entry = 100.0
    best = 101.2         # best_pnl = 1.2% > threshold
    current_close = 101.05  # drawdown from best 0.148%
    atr = 0.2            # 0.2/101.05 * 0.5 * 100 = 0.099% trail_dist
    # trail_dist_pct with tk=0.5: 0.5 * 0.2 / 101.05 * 100 ≈ 0.0990%
    # drawdown = 1.2 - 1.05 = 0.15% > 0.099% → TRAIL_TP fires
    result = sig.check_exit(
        direction='LONG', entry_price=entry, best_price=best,
        current_high=101.2, current_low=100.9, current_close=current_close,
        sl_price=95.0, atr_val=atr, bars_held=10,
    )
    assert result is not None
    assert result['reason'] == 'TRAIL_TP'


def test_progressive_check_exit_long_baseline_holds_at_same_point():
    """Same scenario with disabled progressive → base K=2.5 → larger trail_dist → HOLD."""
    sig = make_signal(progressive_enabled=False)
    entry = 100.0
    best = 101.2
    current_close = 101.05
    atr = 0.2
    # trail_dist_pct with tk=2.5: 2.5 * 0.2 / 101.05 * 100 ≈ 0.495%
    # drawdown = 0.15% < 0.495% → HOLD
    result = sig.check_exit(
        direction='LONG', entry_price=entry, best_price=best,
        current_high=101.2, current_low=100.9, current_close=current_close,
        sl_price=95.0, atr_val=atr, bars_held=10,
    )
    assert result is None  # Hold — trail not triggered


def test_progressive_short_symmetry():
    """SHORT: best_pnl > threshold → tighter K applied symmetrically."""
    sig = make_signal(progressive_enabled=True, threshold_pct=0.9, tk_post=0.5)
    entry = 100.0
    best = 98.8          # best_pnl = 1.2% for SHORT
    current_close = 98.95  # drawdown 0.15%
    atr = 0.2
    result = sig.check_exit(
        direction='SHORT', entry_price=entry, best_price=best,
        current_high=99.1, current_low=98.8, current_close=current_close,
        sl_price=105.0, atr_val=atr, bars_held=10,
    )
    assert result is not None
    assert result['reason'] == 'TRAIL_TP'


def test_progressive_nan_atr_safety():
    """NaN atr must not trigger trail (no crash, returns None)."""
    sig = make_signal(progressive_enabled=True)
    result = sig.check_exit(
        direction='LONG', entry_price=100, best_price=102,
        current_high=102, current_low=101, current_close=101.5,
        sl_price=95, atr_val=float('nan'), bars_held=10,
    )
    assert result is None


def test_progressive_config_defaults_when_missing():
    """Missing progressive_trail section in config → enabled=false by default."""
    cfg = dict(BASE_CFG)  # no progressive_trail key
    sig = C1BreakoutSignal(cfg)
    assert sig.prog_trail_enabled is False
    assert sig.get_effective_trail_k(5.0) == 2.5
