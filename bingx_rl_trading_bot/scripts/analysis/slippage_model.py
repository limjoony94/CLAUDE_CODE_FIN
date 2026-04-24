#!/usr/bin/env python3
"""
Slippage Model — LIVE-realistic slip injection for C1 BT results
==================================================================
Post-hoc slip adjustment helper. Takes a BT trade list (dict format used
across scripts/analysis/), applies directional slippage to entry/exit
prices, and recomputes PnL.

## Empirical basis (2026-04-25)

| Component           | Source                          | Mean    | Notes |
|---------------------|---------------------------------|---------|---|
| Entry MARKET        | slippage_diagnosis_20260422     | 0.058%  | N=26 |
| Exit TRAIL_TP       | state.json exit_slippage_pct    | 0.24%   | N=5, max 0.64% (F v2) |
| Exit SL (STOP)      | slippage_diagnosis STOP+TRAILING| 0.008%  | price-triggered, minimal |
| Exit EMERGENCY      | inferred (volatility spike)     | 0.20%   | conservative |
| Exit TIMEOUT        | inferred (market close)         | 0.06%   | ~ entry MARKET |
| Exit TRAIL_TP (F v2)| state.json first F v2 trade     | 0.64%   | N=1, outlier; HIGH scenario |

## Scenarios

ZERO    — no slip, theoretical BT (current default)
LOW     — 10th percentile, optimistic LIVE
MED     — 50th percentile, pre-F-v2 typical LIVE
HIGH    — 90th percentile, F v2 conservative (MARKET close larger slip)
STRESS  — F v2 first-trade level (MARKET close worst observed)

## Slip direction convention

Adverse to trader:
- LONG entry:  fill at price × (1 + slip)  — paid more
- SHORT entry: fill at price × (1 - slip)  — received less
- LONG exit:   fill at price × (1 - slip)  — received less
- SHORT exit:  fill at price × (1 + slip)  — paid more

All slip values are in % (e.g., 0.058 means 0.058%).

## Usage

    from scripts.analysis.slippage_model import apply_slip_to_trades, SCENARIOS

    # BT trades from any analysis script
    adjusted = apply_slip_to_trades(bt_trades, scenario='MED')

    # Or manual
    for trade in bt_trades:
        new_trade = apply_slip_to_trade(trade, SCENARIOS['MED'])

## Integration note

This is POST-HOC adjustment: takes completed BT trade outcomes and
re-prices them. Does NOT re-simulate with slipped prices (which could
change exit trigger outcomes). For MVP and baseline reference, post-hoc
is sufficient and reusable across all existing analysis scripts.
"""

from __future__ import annotations
from typing import Any

# ═══════════════════════════════════════════════════════════════════════
# Scenarios — all values in % (e.g., 0.058 = 0.058%)
# ═══════════════════════════════════════════════════════════════════════

SCENARIOS: dict[str, dict[str, float]] = {
    'ZERO': {
        'entry':        0.000,
        'exit_trail':   0.000,
        'exit_sl':      0.000,
        'exit_emerg':   0.000,
        'exit_timeout': 0.000,
    },
    'LOW': {   # ~10th percentile optimistic
        'entry':        0.029,
        'exit_trail':   0.050,
        'exit_sl':      0.005,
        'exit_emerg':   0.100,
        'exit_timeout': 0.030,
    },
    'MED': {   # pre-F-v2 typical LIVE (median of observed)
        'entry':        0.058,
        'exit_trail':   0.100,
        'exit_sl':      0.020,
        'exit_emerg':   0.200,
        'exit_timeout': 0.060,
    },
    'HIGH': {  # ~90th percentile conservative, F v2 MARKET close expectation
        'entry':        0.100,
        'exit_trail':   0.300,
        'exit_sl':      0.050,
        'exit_emerg':   0.400,
        'exit_timeout': 0.120,
    },
    'STRESS': {  # F v2 first-trade observed (0.64%), applied broadly
        'entry':        0.150,
        'exit_trail':   0.640,
        'exit_sl':      0.100,
        'exit_emerg':   0.500,
        'exit_timeout': 0.200,
    },
}

# Map exit reason strings (and their _SAMEBAR variants) to scenario key
REASON_MAP = {
    'TRAIL_TP':            'exit_trail',
    'TRAIL_TP_SAMEBAR':    'exit_trail',
    'TRAIL_TP_V2':         'exit_trail',
    'EXCHANGE_TRAIL':      'exit_trail',
    'SL':                  'exit_sl',
    'SL_SAMEBAR':          'exit_sl',
    'EXCHANGE_SL':         'exit_sl',
    'EMERGENCY':           'exit_emerg',
    'EMERGENCY_SAMEBAR':   'exit_emerg',
    'TIMEOUT':             'exit_timeout',
    'TIMEOUT_SAMEBAR':     'exit_timeout',
}


# ═══════════════════════════════════════════════════════════════════════
# Core functions
# ═══════════════════════════════════════════════════════════════════════

def apply_slip_to_trade(trade: dict[str, Any],
                         slip: dict[str, float],
                         fee_rt_pct: float = 0.10) -> dict[str, Any]:
    """
    Apply directional slippage to a single BT trade dict.

    Expected input fields: 'direction', 'entry_price', 'exit_price', 'reason'
    Returns a new dict with additional fields:
      entry_price_slipped, exit_price_slipped, pnl_pct_slipped, slip_entry, slip_exit

    Leaves original 'entry_price', 'exit_price', 'pnl_pct' untouched.
    """
    direction = trade.get('direction') or trade.get('fade_direction')
    entry = float(trade['entry_price'])
    exit_p = float(trade['exit_price'])
    reason = trade.get('reason', 'TRAIL_TP')

    # Entry slip — adverse
    e_slip = slip['entry'] / 100.0
    if direction == 'LONG':
        entry_slipped = entry * (1 + e_slip)
    else:  # SHORT
        entry_slipped = entry * (1 - e_slip)

    # Exit slip — adverse, reason-specific
    reason_key = REASON_MAP.get(reason, 'exit_trail')
    x_slip_pct = slip.get(reason_key, slip['exit_trail'])
    x_slip = x_slip_pct / 100.0
    if direction == 'LONG':
        exit_slipped = exit_p * (1 - x_slip)
    else:
        exit_slipped = exit_p * (1 + x_slip)

    # Recompute PnL net of fee
    if direction == 'LONG':
        pnl_raw = (exit_slipped / entry_slipped - 1) * 100
    else:
        pnl_raw = (1 - exit_slipped / entry_slipped) * 100
    pnl_net = pnl_raw - fee_rt_pct

    return {
        **trade,
        'entry_price_slipped': round(entry_slipped, 4),
        'exit_price_slipped':  round(exit_slipped, 4),
        'pnl_pct_slipped':     round(pnl_net, 4),
        'slip_entry_pct':      round(e_slip * 100, 4),
        'slip_exit_pct':       round(x_slip_pct, 4),
    }


def apply_slip_to_trades(trades: list[dict[str, Any]],
                          scenario: str = 'MED',
                          fee_rt_pct: float = 0.10) -> list[dict[str, Any]]:
    """Apply slip scenario to a list of trades. Returns new list."""
    if scenario not in SCENARIOS:
        raise ValueError(f'Unknown scenario: {scenario}. '
                          f'Choose from {list(SCENARIOS.keys())}')
    slip = SCENARIOS[scenario]
    return [apply_slip_to_trade(t, slip, fee_rt_pct) for t in trades]


def compare_scenarios(trades: list[dict[str, Any]],
                       scenarios: list[str] | None = None,
                       fee_rt_pct: float = 0.10) -> dict[str, dict[str, Any]]:
    """
    Apply all scenarios to a trade list and return side-by-side metrics.

    Returns: {scenario_name: {trades, pnl, WR, RR, MDD}}
    """
    if scenarios is None:
        scenarios = list(SCENARIOS.keys())

    results = {}
    for sc in scenarios:
        adjusted = apply_slip_to_trades(trades, scenario=sc, fee_rt_pct=fee_rt_pct)
        if sc == 'ZERO':
            pnls = [t['pnl_pct'] for t in trades]
        else:
            pnls = [t['pnl_pct_slipped'] for t in adjusted]
        if not pnls:
            results[sc] = {'trades': 0}
            continue
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        wr = len(wins) / len(pnls) * 100
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = abs(sum(losses) / len(losses)) if losses else 0
        rr = avg_win / avg_loss if avg_loss > 0 else float('inf')
        # MDD
        cum = 0.0; peak = 0.0; mdd = 0.0
        for p in pnls:
            cum += p
            peak = max(peak, cum)
            mdd = max(mdd, peak - cum)
        results[sc] = {
            'trades':       len(pnls),
            'total_pnl':    round(sum(pnls), 2),
            'WR':           round(wr, 2),
            'RR':           round(rr, 3) if rr != float('inf') else None,
            'MDD':          round(mdd, 2),
            'avg_win':      round(avg_win, 4),
            'avg_loss':     round(avg_loss, 4),
        }
    return results


# ═══════════════════════════════════════════════════════════════════════
# GO-gate helpers (for reuse across future PDCA validation scripts)
# ═══════════════════════════════════════════════════════════════════════

def slip_adjusted_delta(candidate_trades: list[dict],
                         baseline_trades: list[dict],
                         scenario: str = 'MED',
                         days: float = 332.0) -> dict[str, float]:
    """
    Compute slip-adjusted daily PnL delta (candidate − baseline).

    Returns dict with 'candidate_daily', 'baseline_daily', 'delta_daily',
    'baseline_slip_cost' (how much baseline lost from slip).
    """
    cand_adj = apply_slip_to_trades(candidate_trades, scenario=scenario)
    base_adj = apply_slip_to_trades(baseline_trades, scenario=scenario)

    cand_pnl = sum(t['pnl_pct_slipped'] for t in cand_adj)
    base_pnl_slip = sum(t['pnl_pct_slipped'] for t in base_adj)
    base_pnl_zero = sum(t['pnl_pct'] for t in baseline_trades)

    return {
        'scenario':             scenario,
        'candidate_daily':      round(cand_pnl / days, 4),
        'baseline_daily_slip':  round(base_pnl_slip / days, 4),
        'baseline_daily_zero':  round(base_pnl_zero / days, 4),
        'delta_daily':          round((cand_pnl - base_pnl_slip) / days, 4),
        'baseline_slip_cost':   round((base_pnl_zero - base_pnl_slip) / days, 4),
    }


# ═══════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # Minimal sanity check
    demo_trades = [
        {'direction': 'LONG',  'entry_price': 60000, 'exit_price': 61000,
         'pnl_pct': 1.567, 'reason': 'TRAIL_TP'},
        {'direction': 'SHORT', 'entry_price': 60000, 'exit_price': 59500,
         'pnl_pct': 0.734, 'reason': 'TRAIL_TP'},
        {'direction': 'LONG',  'entry_price': 60000, 'exit_price': 59000,
         'pnl_pct': -1.767, 'reason': 'SL'},
    ]

    print('=== slippage_model.py self-test ===')
    print(f'Scenarios: {list(SCENARIOS.keys())}')
    print()

    for sc_name in ['ZERO', 'LOW', 'MED', 'HIGH', 'STRESS']:
        adj = apply_slip_to_trades(demo_trades, scenario=sc_name)
        print(f'--- {sc_name} ---')
        for i, (orig, slipped) in enumerate(zip(demo_trades, adj)):
            print(f'  Trade {i+1}: {orig["direction"]} {orig["reason"]:10s}  '
                  f'raw PnL = {orig["pnl_pct"]:+.3f}%  '
                  f'slip PnL = {slipped["pnl_pct_slipped"]:+.3f}%  '
                  f'(entry={slipped["slip_entry_pct"]:.3f}%, '
                  f'exit={slipped["slip_exit_pct"]:.3f}%)')
        print()

    print('=== compare_scenarios output ===')
    import json
    result = compare_scenarios(demo_trades)
    print(json.dumps(result, indent=2))
