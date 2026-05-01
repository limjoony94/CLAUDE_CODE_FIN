"""Friction tests."""

from __future__ import annotations

import pytest

from abm.constants import MAKER_FEE, TAKER_FEE
from abm.friction import Friction
from abm.types import Role


def test_default_rates_match_constants() -> None:
    f = Friction()
    assert f.taker_fee == TAKER_FEE == 0.0005
    assert f.maker_fee == MAKER_FEE == 0.0002


def test_taker_fee_correct() -> None:
    f = Friction()
    notional = 50000.0  # 1 BTC at 50000
    assert f.fee(Role.TAKER, notional) == pytest.approx(50000.0 * 0.0005)
    assert f.fee(Role.TAKER, notional) == pytest.approx(25.0)


def test_maker_fee_correct() -> None:
    f = Friction()
    notional = 50000.0
    assert f.fee(Role.MAKER, notional) == pytest.approx(50000.0 * 0.0002)
    assert f.fee(Role.MAKER, notional) == pytest.approx(10.0)


def test_zero_notional_zero_fee() -> None:
    f = Friction()
    assert f.fee(Role.TAKER, 0.0) == 0.0
    assert f.fee(Role.MAKER, 0.0) == 0.0


def test_negative_notional_raises() -> None:
    f = Friction()
    with pytest.raises(ValueError, match="notional must be >= 0"):
        f.fee(Role.TAKER, -1.0)


def test_negative_fee_rate_raises() -> None:
    with pytest.raises(ValueError, match="taker_fee must be >= 0"):
        Friction(taker_fee=-0.001)
    with pytest.raises(ValueError, match="maker_fee must be >= 0"):
        Friction(maker_fee=-0.001)


def test_slippage_positive_when_executed_above_expected() -> None:
    f = Friction()
    s = f.slippage_observed(submitted_size=1.0, executed_avg_price=100.5, expected_price=100.0)
    assert s == pytest.approx(0.005)  # 0.5% slippage


def test_slippage_negative_when_executed_below_expected() -> None:
    f = Friction()
    s = f.slippage_observed(submitted_size=1.0, executed_avg_price=99.5, expected_price=100.0)
    assert s == pytest.approx(-0.005)


def test_slippage_zero_for_zero_size() -> None:
    f = Friction()
    assert f.slippage_observed(submitted_size=0.0, executed_avg_price=100.5, expected_price=100.0) == 0.0


def test_slippage_invalid_expected_price_raises() -> None:
    f = Friction()
    with pytest.raises(ValueError, match="expected_price must be > 0"):
        f.slippage_observed(submitted_size=1.0, executed_avg_price=100.0, expected_price=0.0)


def test_custom_fee_rates() -> None:
    f = Friction(taker_fee=0.001, maker_fee=0.0)
    assert f.fee(Role.TAKER, 1000.0) == 1.0
    assert f.fee(Role.MAKER, 1000.0) == 0.0
