"""Shared fixtures for pattern_5m tests."""

import sys
import os
import pytest
import pandas as pd
import numpy as np

# Add parent package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))


@pytest.fixture
def make_candle():
    """Factory fixture to create a single candle row."""
    def _make(open_: float, high: float, low: float, close: float):
        return pd.Series({
            'open': open_, 'high': high, 'low': low, 'close': close,
        })
    return _make


@pytest.fixture
def make_df():
    """Factory fixture to create a DataFrame with OHLCV data."""
    def _make(rows: list[tuple], columns=('open', 'high', 'low', 'close')):
        df = pd.DataFrame(rows, columns=list(columns))
        return df
    return _make


@pytest.fixture
def avg_body_default():
    """Default avg_body_20 value used for early bars."""
    return 1.0


@pytest.fixture
def avg_body_typical():
    """Typical avg_body_20 for BTC 5m candles (~50-100 USD)."""
    return 80.0
