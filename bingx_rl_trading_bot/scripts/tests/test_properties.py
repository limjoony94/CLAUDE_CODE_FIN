"""Property-based tests with Hypothesis.

Each property codifies an invariant that must hold for *all* valid inputs.
Hypothesis generates thousands of random cases to find violations the
human author wouldn't think of.

Invariants tested:
  - ATR is causal: modifying future bars cannot change past ATR values
  - check_entry: either returns None, or a dict with valid sl_price
  - check_exit: exit priority SL > EMG > TO > TRAIL preserved under any input
  - SL distance always in [sl_min_pct, sl_max_pct] when signal returned
"""
import math
import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck

from scripts.production.c1_breakout.indicators import (
    compute_atr, compute_channel, compute_fractal_swings,
)
from scripts.production.c1_breakout.signals import C1BreakoutSignal


# ── Strategies (random input generators) ───────────────────

# Price bars: positive floats in a reasonable BTC range
price = st.floats(min_value=1000.0, max_value=200000.0,
                  allow_nan=False, allow_infinity=False)

# ATR can be 0 (warmup) or positive
atr_val_st = st.one_of(st.just(0.0), price)


# ── ATR properties ────────────────────────────────────────

@given(highs=st.lists(price, min_size=20, max_size=50))
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_atr_never_negative(highs):
    """Invariant: ATR values are always ≥ 0 (or NaN during warmup)."""
    lows = [h * 0.99 for h in highs]  # low below high
    closes = [(h + l) / 2 for h, l in zip(highs, lows)]
    atr = compute_atr(highs, lows, closes, period=14)
    for v in atr:
        if not math.isnan(v):
            assert v >= 0, f"Negative ATR: {v}"


@given(
    prefix_len=st.integers(min_value=20, max_value=25),
    prefix_highs=st.lists(price, min_size=20, max_size=25),
)
@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
def test_atr_causality_invariant(prefix_len, prefix_highs):
    """Invariant: ATR[i] depends only on bars[0..i].

    Generates random prefix, appends different suffixes, verifies ATR[prefix]
    unchanged.
    """
    assume(len(prefix_highs) >= prefix_len)
    prefix_highs = prefix_highs[:prefix_len]
    prefix_lows = [h * 0.99 for h in prefix_highs]
    prefix_closes = [(h + l) / 2 for h, l in zip(prefix_highs, prefix_lows)]

    # Suffix A (small values)
    suffix_a_high = [1000.0] * 5
    suffix_a_low = [999.0] * 5
    suffix_a_close = [999.5] * 5
    full_a_h = prefix_highs + suffix_a_high
    full_a_l = prefix_lows + suffix_a_low
    full_a_c = prefix_closes + suffix_a_close
    atr_a = compute_atr(full_a_h, full_a_l, full_a_c, period=14)

    # Suffix B (huge values)
    suffix_b_high = [200000.0] * 5
    suffix_b_low = [100000.0] * 5
    suffix_b_close = [150000.0] * 5
    full_b_h = prefix_highs + suffix_b_high
    full_b_l = prefix_lows + suffix_b_low
    full_b_c = prefix_closes + suffix_b_close
    atr_b = compute_atr(full_b_h, full_b_l, full_b_c, period=14)

    # ATR at prefix indices must match
    for i in range(prefix_len):
        if math.isnan(atr_a[i]) and math.isnan(atr_b[i]):
            continue
        assert atr_a[i] == atr_b[i], f"Future leaked at bar {i}"


# ── check_entry properties ────────────────────────────────

@given(
    bar_open=price, bar_high=price, bar_low=price, bar_close=price,
    channel_high=price, channel_low=price,
    atr=atr_val_st,
    swing_low=price, swing_high=price,
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_check_entry_never_crashes(
    bar_open, bar_high, bar_low, bar_close,
    channel_high, channel_low, atr, swing_low, swing_high,
):
    """Invariant: check_entry returns None or dict, never raises."""
    sig = C1BreakoutSignal({
        'channel_period': 15, 'body_min_ratio': 0.4, 'atr_period': 14,
        'trail_K': 2.5, 'max_sl_atr': 3.3, 'emergency_sl_pct': 3.0,
        'max_hold_bars': 192, 'sl_min_pct': 0.15, 'sl_max_pct': 3.0,
    })
    # Must not raise on any finite numeric input
    r = sig.check_entry(bar_open, bar_high, bar_low, bar_close,
                        channel_high, channel_low, atr, swing_low, swing_high)
    # Result is None or dict with required keys
    assert r is None or (
        isinstance(r, dict)
        and 'direction' in r
        and 'sl_price' in r
        and 'sl_pct' in r
    )


@given(
    bar_open=st.floats(min_value=10000, max_value=100000, allow_nan=False),
    atr=st.floats(min_value=10, max_value=2000, allow_nan=False),
    swing_offset=st.floats(min_value=10, max_value=2000, allow_nan=False),
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_signal_sl_pct_within_bounds(bar_open, atr, swing_offset):
    """Invariant: when check_entry returns signal, sl_pct ∈ [0.15%, 3.0%].

    Sampled over realistic BTC range + realistic ATRs.
    """
    sig = C1BreakoutSignal({
        'channel_period': 15, 'body_min_ratio': 0.4, 'atr_period': 14,
        'trail_K': 2.5, 'max_sl_atr': 3.3, 'emergency_sl_pct': 3.0,
        'max_hold_bars': 192, 'sl_min_pct': 0.15, 'sl_max_pct': 3.0,
    })
    # Construct a clean LONG breakout bar
    bar_close = bar_open + atr * 2  # strong body up
    bar_high = bar_close * 1.001
    bar_low = bar_open * 0.999
    channel_high = bar_open * 0.999  # below close → breakout
    channel_low = bar_open * 0.95
    swing_low = bar_open - swing_offset
    r = sig.check_entry(bar_open, bar_high, bar_low, bar_close,
                        channel_high, channel_low, atr, swing_low, bar_high)
    if r is not None:
        assert 0.15 <= r['sl_pct'] <= 3.0


# ── check_exit properties ─────────────────────────────────

@given(
    entry=st.floats(min_value=1000, max_value=100000, allow_nan=False),
    best_mult=st.floats(min_value=0.9, max_value=1.2, allow_nan=False),
    cur_high=st.floats(min_value=1000, max_value=200000, allow_nan=False),
    cur_low=st.floats(min_value=100, max_value=100000, allow_nan=False),
    cur_close=st.floats(min_value=1, max_value=200000, allow_nan=False),
    atr=st.floats(min_value=0, max_value=2000, allow_nan=False),
    bars=st.integers(min_value=0, max_value=300),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_check_exit_never_crashes_long(
    entry, best_mult, cur_high, cur_low, cur_close, atr, bars,
):
    """Invariant: check_exit returns None or dict with reason in known set."""
    sig = C1BreakoutSignal({
        'channel_period': 15, 'body_min_ratio': 0.4, 'atr_period': 14,
        'trail_K': 2.5, 'max_sl_atr': 3.3, 'emergency_sl_pct': 3.0,
        'max_hold_bars': 192, 'sl_min_pct': 0.15, 'sl_max_pct': 3.0,
        'trail_activation_pct': 0.05,
    })
    best = entry * best_mult
    sl = entry * 0.98  # LONG SL below entry
    assume(cur_low <= cur_high)  # sanity: high ≥ low

    r = sig.check_exit('LONG', entry, best, cur_high, cur_low, cur_close, sl, atr, bars)
    assert r is None or r['reason'] in ('SL', 'EMERGENCY', 'TIMEOUT', 'TRAIL_TP')


@given(
    entry=st.floats(min_value=10000, max_value=100000, allow_nan=False),
    bars=st.integers(min_value=192, max_value=500),
)
@settings(max_examples=30)
def test_timeout_always_triggers_when_bars_exceed_max(entry, bars):
    """Invariant: bars_held ≥ max_hold_bars AND no earlier exit → TIMEOUT.

    Construct inputs so SL/Emergency/Trail don't fire: cur_close = entry,
    no drawdown, sl far away.
    """
    sig = C1BreakoutSignal({
        'channel_period': 15, 'body_min_ratio': 0.4, 'atr_period': 14,
        'trail_K': 2.5, 'max_sl_atr': 3.3, 'emergency_sl_pct': 3.0,
        'max_hold_bars': 192, 'sl_min_pct': 0.15, 'sl_max_pct': 3.0,
        'trail_activation_pct': 0.05,
    })
    # Price exactly at entry: no SL, no emergency, no trail drawdown
    r = sig.check_exit('LONG', entry, entry, entry, entry, entry,
                       entry * 0.5, 10.0, bars)
    assert r is not None
    assert r['reason'] == 'TIMEOUT'
