"""Friction model — round-trip cost + funding accrual (separated per advisor #3).

Three taker scenarios (optimistic / realistic / stress) per precommit_amendment_001.
Maker scenarios stubbed per amendment_003 (NotImplementedError until P3 activation).

Convention:
- All percentages stored as % (e.g., 0.045 = 4.5 bps = 0.045%). NOT fractions.
- All cost outputs in USD (notional × pct / 100).
- Funding rate convention: positive rate → longs PAY shorts (Binance perp standard).

Public API:
    get_friction(scenario) -> FrictionParams
    round_trip_cost_pct(params) -> float (% of notional)
    round_trip_cost_usd(notional, params) -> float (USD)
    per_leg_taker_fee_usd(notional, params) -> float (USD, single leg)
    adjust_fill_price(ideal, side, params) -> float (slippage-adjusted price)
    funding_accrual_usd(position_signed_usd, funding_rates) -> float (USD signed)
    apply_friction_to_returns(returns, n_trades_per_period, params) -> np.ndarray
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Union

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FrictionParams:
    """Friction parameters. All values in % (e.g., 0.045 = 4.5 bps)."""
    taker_fee_pct: float          # per side
    slippage_pct: float            # per side
    scenario_name: str = "custom"

    @property
    def per_leg_pct(self) -> float:
        """Single leg cost (taker fee + slippage)."""
        return self.taker_fee_pct + self.slippage_pct

    @property
    def round_trip_pct(self) -> float:
        """Round-trip cost (entry + exit, taker x2 + slip x2)."""
        return self.per_leg_pct * 2


# Pre-committed scenarios (precommit_amendment_001)
_TAKER_SCENARIOS = {
    "optimistic": FrictionParams(taker_fee_pct=0.045, slippage_pct=0.005, scenario_name="optimistic"),
    "realistic":  FrictionParams(taker_fee_pct=0.045, slippage_pct=0.035, scenario_name="realistic"),
    "stress":     FrictionParams(taker_fee_pct=0.045, slippage_pct=0.055, scenario_name="stress"),
}

_VALID_TAKER = list(_TAKER_SCENARIOS.keys())


def get_friction(scenario: str = "realistic") -> FrictionParams:
    """Resolve friction params by scenario name.

    Args:
        scenario: one of {"optimistic", "realistic", "stress"} (taker)
                  or "maker_*" (raises NotImplementedError until P3 activation per amendment_003).

    Raises:
        NotImplementedError if maker_* requested.
        ValueError if unknown scenario.
    """
    if scenario in _TAKER_SCENARIOS:
        return _TAKER_SCENARIOS[scenario]
    if scenario.startswith("maker_"):
        raise NotImplementedError(
            f"Maker scenario '{scenario}' is stubbed per precommit_amendment_003. "
            "Activate via amendment_004 when post-only mechanism identified at P3."
        )
    raise ValueError(f"Unknown friction scenario: {scenario!r}. Valid: {_VALID_TAKER}")


def round_trip_cost_pct(params: FrictionParams) -> float:
    """Round-trip cost as % of notional."""
    return params.round_trip_pct


def round_trip_cost_usd(notional_usd: float, params: FrictionParams) -> float:
    """Round-trip cost in USD."""
    if notional_usd < 0:
        raise ValueError(f"notional_usd must be ≥0, got {notional_usd}")
    return notional_usd * params.round_trip_pct / 100.0


def per_leg_taker_fee_usd(notional_usd: float, params: FrictionParams) -> float:
    """Single leg taker fee (no slippage component)."""
    if notional_usd < 0:
        raise ValueError(f"notional_usd must be ≥0, got {notional_usd}")
    return notional_usd * params.taker_fee_pct / 100.0


def adjust_fill_price(ideal_price: float, side: str, params: FrictionParams) -> float:
    """Apply slippage to ideal fill price.

    Args:
        ideal_price: target price (e.g., bar open, signal price)
        side: 'buy' (paying ask) or 'sell' (paying bid)
        params: FrictionParams

    Returns:
        Slippage-adjusted fill price.
        - buy: ideal × (1 + slip%/100) — fill HIGHER than ideal
        - sell: ideal × (1 − slip%/100) — fill LOWER than ideal
    """
    if ideal_price <= 0:
        raise ValueError(f"ideal_price must be >0, got {ideal_price}")
    slip = params.slippage_pct / 100.0
    if side == "buy":
        return ideal_price * (1 + slip)
    elif side == "sell":
        return ideal_price * (1 - slip)
    else:
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")


def funding_accrual_usd(position_signed_usd: float,
                         funding_rates: Iterable[float]) -> float:
    """Funding accrual over a list of 8h funding intervals.

    Args:
        position_signed_usd: notional × direction (positive = LONG, negative = SHORT)
        funding_rates: list/array of 8h funding rate values (fractions, e.g. 0.0001 = 0.01%)

    Returns:
        USD signed: positive = received, negative = paid.

    Convention (Binance perp):
        - Positive funding rate → LONG pays SHORT
        - Negative funding rate → SHORT pays LONG
        - For a long with rate +0.0001: pays |position| × 0.0001 = small loss
        - For a short with rate +0.0001: receives |position| × 0.0001
        - Formula: received = -position_signed × total_rate
    """
    rates = list(funding_rates)
    if not rates:
        return 0.0
    total_rate = float(sum(rates))
    return -position_signed_usd * total_rate


def apply_friction_to_returns(returns: Union[pd.Series, np.ndarray],
                                n_trades_per_period: float,
                                params: FrictionParams) -> np.ndarray:
    """Apply RT friction to per-period return series.

    Each period's gross return reduced by `n_trades_per_period × round_trip_pct`.

    Args:
        returns: gross per-period returns (fractions, e.g., 0.005 = 0.5%/day)
        n_trades_per_period: avg trades per period (e.g., 2.0 for 2 trades/day)
        params: FrictionParams

    Returns:
        Net returns array.
    """
    if isinstance(returns, pd.Series):
        arr = returns.values
    else:
        arr = np.asarray(returns)
    if n_trades_per_period < 0:
        raise ValueError(f"n_trades_per_period must be ≥0, got {n_trades_per_period}")
    cost_per_period = n_trades_per_period * params.round_trip_pct / 100.0
    return arr - cost_per_period


def list_scenarios() -> list[str]:
    """List active taker scenarios."""
    return list(_TAKER_SCENARIOS.keys())


__all__ = [
    "FrictionParams",
    "get_friction",
    "round_trip_cost_pct",
    "round_trip_cost_usd",
    "per_leg_taker_fee_usd",
    "adjust_fill_price",
    "funding_accrual_usd",
    "apply_friction_to_returns",
    "list_scenarios",
]
