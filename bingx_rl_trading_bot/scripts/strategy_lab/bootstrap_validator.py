"""Bootstrap Validator — 사용자 success criteria 정형화 (2026-05-01).

Memory ref: research_protocol_3day_bootstrap.md

User-defined success criteria:
  1. Daily avg return >= 0.20% (no leverage, gross of leverage)
  2. Avg per-trade return > taker round-trip friction (0.07%)
  3. Statistically sufficient trade count
  4. 3-day random-window bootstrap shows STABLE positive (not 1-2 trade explosions)

Bootstrap test:
  - Sample N (default 1000) random 3-day windows from trade history
  - For each: cum_return_pct, n_trades, mean_daily_pct, contains_pos
  - Aggregate stats: mean_daily, median_daily, p5_daily, pos_rate, p_beats_baseline
  - Pass = all 4 criteria met

Used by:
  - Reality Filter dim #5 (path-dependent failure mode → upgraded to bootstrap-based)
  - Phase 2/3/4 candidate evaluation
  - Self-validation against R5 (baseline) and 28 rounds
"""
import json
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# User criteria
# ============================================================================
DEFAULT_CRITERIA = {
    'min_daily_pct':          0.20,    # 1일 평균 수익 >= 0.2%
    'min_per_trade_pct':      0.07,    # 1회 거래 평균 > taker RT
    'min_n_trades_window':    3,       # 3-day window에 최소 3 trades (통계적 의미)
    'min_pos_rate':           0.50,    # 50% windows positive
    'min_p5_daily_pct':       0.00,    # 5-tile non-negative (worst 5% non-loss)
    'min_p_beats_baseline':   0.55,    # 55% windows beat fee-only baseline
    'bootstrap_n_iter':       1000,
    'window_days':            3,
}


@dataclass
class BootstrapResult:
    n_iter: int
    window_days: int
    n_trades_total: int
    span_days: float
    avg_per_trade_pct: float
    mean_daily_pct: float
    median_daily_pct: float
    p5_daily_pct: float
    p25_daily_pct: float
    p75_daily_pct: float
    p95_daily_pct: float
    pos_rate: float
    p_beats_baseline: float
    pass_criteria: dict
    overall_pass: bool


def bootstrap_validate(
    trades_df: pd.DataFrame,
    span_start: pd.Timestamp,
    span_end: pd.Timestamp,
    criteria: Optional[dict] = None,
    seed: int = 42,
) -> BootstrapResult:
    """
    trades_df columns required: 'close_ts' (datetime), 'net_pnl_pct' (float, % of balance)
    Optional: 'gross_pct' for per-trade gross test.

    Returns BootstrapResult + pass/fail per criterion.
    """
    crit = {**DEFAULT_CRITERIA, **(criteria or {})}
    span_start = pd.Timestamp(span_start, tz='UTC') if span_start.tzinfo is None else pd.Timestamp(span_start)
    span_end = pd.Timestamp(span_end, tz='UTC') if span_end.tzinfo is None else pd.Timestamp(span_end)
    span_days = (span_end - span_start).total_seconds() / 86400

    if span_days <= crit['window_days']:
        return BootstrapResult(
            n_iter=0, window_days=crit['window_days'], n_trades_total=len(trades_df),
            span_days=span_days, avg_per_trade_pct=0.0,
            mean_daily_pct=0.0, median_daily_pct=0.0,
            p5_daily_pct=0.0, p25_daily_pct=0.0, p75_daily_pct=0.0, p95_daily_pct=0.0,
            pos_rate=0.0, p_beats_baseline=0.0,
            pass_criteria={'span_too_short': True},
            overall_pass=False,
        )

    # Prep trade timestamps
    trades_df = trades_df.copy()
    trades_df['close_ts'] = pd.to_datetime(trades_df['close_ts'], utc=True)
    trades_df = trades_df[(trades_df['close_ts'] >= span_start) & (trades_df['close_ts'] <= span_end)]
    trades_df = trades_df.sort_values('close_ts').reset_index(drop=True)

    n_total = len(trades_df)
    if n_total == 0:
        return BootstrapResult(
            n_iter=0, window_days=crit['window_days'], n_trades_total=0,
            span_days=span_days, avg_per_trade_pct=0.0,
            mean_daily_pct=0.0, median_daily_pct=0.0,
            p5_daily_pct=0.0, p25_daily_pct=0.0, p75_daily_pct=0.0, p95_daily_pct=0.0,
            pos_rate=0.0, p_beats_baseline=0.0,
            pass_criteria={'no_trades': True},
            overall_pass=False,
        )

    avg_per_trade = float(trades_df['net_pnl_pct'].mean())

    # Bootstrap — random window starts
    rng = np.random.default_rng(seed)
    window_days = crit['window_days']
    n_iter = crit['bootstrap_n_iter']
    starts_ts = pd.date_range(span_start, span_end - pd.Timedelta(days=window_days), freq='1h')
    if len(starts_ts) < n_iter:
        # If span small relative to n_iter, sample with replacement
        starts_idx = rng.integers(0, len(starts_ts), size=n_iter)
    else:
        starts_idx = rng.choice(len(starts_ts), size=n_iter, replace=False)
    starts = starts_ts[starts_idx]

    cums = np.zeros(n_iter)
    n_tr_per_window = np.zeros(n_iter, dtype=int)
    for i, s in enumerate(starts):
        e = s + pd.Timedelta(days=window_days)
        in_win = trades_df[(trades_df['close_ts'] >= s) & (trades_df['close_ts'] < e)]
        cums[i] = float(in_win['net_pnl_pct'].sum())
        n_tr_per_window[i] = len(in_win)

    # Daily-equivalent normalization
    daily = cums / window_days

    mean_daily = float(daily.mean())
    median_daily = float(np.median(daily))
    p5_daily = float(np.percentile(daily, 5))
    p25_daily = float(np.percentile(daily, 25))
    p75_daily = float(np.percentile(daily, 75))
    p95_daily = float(np.percentile(daily, 95))
    pos_rate = float((daily > 0).mean())

    # Beats baseline (zero PnL = fee-only buy-hold simplification)
    # baseline daily = 0
    p_beats_baseline = float((daily > 0).mean())   # same as pos_rate when baseline=0

    # Apply criteria
    pass_criteria = {
        'mean_daily_>=_0.20': mean_daily >= crit['min_daily_pct'],
        'p5_daily_>=_0':       p5_daily >= crit['min_p5_daily_pct'],
        'pos_rate_>=_0.50':    pos_rate >= crit['min_pos_rate'],
        'avg_per_trade_>_taker_RT': avg_per_trade > crit['min_per_trade_pct'],
        'sufficient_trades_per_window': float(n_tr_per_window.mean()) >= crit['min_n_trades_window'],
        'p_beats_baseline_>=_0.55': p_beats_baseline >= crit['min_p_beats_baseline'],
    }
    overall_pass = all(pass_criteria.values())

    return BootstrapResult(
        n_iter=n_iter, window_days=window_days, n_trades_total=n_total,
        span_days=span_days, avg_per_trade_pct=avg_per_trade,
        mean_daily_pct=mean_daily, median_daily_pct=median_daily,
        p5_daily_pct=p5_daily, p25_daily_pct=p25_daily,
        p75_daily_pct=p75_daily, p95_daily_pct=p95_daily,
        pos_rate=pos_rate, p_beats_baseline=p_beats_baseline,
        pass_criteria=pass_criteria, overall_pass=overall_pass,
    )


def report(res: BootstrapResult, name: str = ''):
    print(f'\n=== Bootstrap Validator{(" — " + name) if name else ""} ===')
    print(f'  span: {res.span_days:.1f} days, n_trades: {res.n_trades_total}')
    print(f'  iter: {res.n_iter}, window: {res.window_days}d')
    print(f'  avg_per_trade_pct: {res.avg_per_trade_pct:+.4f}%')
    print(f'  daily distribution: mean={res.mean_daily_pct:+.4f}%, median={res.median_daily_pct:+.4f}%')
    print(f'                      p5={res.p5_daily_pct:+.4f}%, p25={res.p25_daily_pct:+.4f}%, '
          f'p75={res.p75_daily_pct:+.4f}%, p95={res.p95_daily_pct:+.4f}%')
    print(f'  pos_rate: {res.pos_rate:.3f}, p_beats_baseline: {res.p_beats_baseline:.3f}')
    print(f'  Pass criteria:')
    for k, v in res.pass_criteria.items():
        mark = '✅' if v else '🔴'
        print(f'    {mark} {k}')
    overall = '✅ PASS' if res.overall_pass else '🔴 FAIL'
    print(f'  Overall: {overall}')


# ============================================================================
# Self-validation: 28 rounds calibration
# ============================================================================
def calibrate_against_kb():
    """Apply bootstrap_validator to known KB rounds with synthetic trade history.

    R5 deployable: APY 3.28% / 365 = ~0.009%/day → should NOT pass criterion (0.20% required).
    Demonstrates: even our best deployable strategy fails user criteria.
    """
    print('=' * 100)
    print('Bootstrap Validator — Calibration against KB rounds')
    print('=' * 100)

    # Synthetic R5 carry trade history: ~365 days, daily 0.009% PnL, 1 trade/day
    rng = np.random.default_rng(42)
    span_start = pd.Timestamp('2025-04-01', tz='UTC')
    n_days = 365
    timestamps = [span_start + pd.Timedelta(days=i) for i in range(n_days)]
    # R5 daily = 0.009% with low variance (carry is stable)
    pnl = rng.normal(loc=0.009, scale=0.005, size=n_days)
    r5_synth = pd.DataFrame({'close_ts': timestamps, 'net_pnl_pct': pnl})

    res = bootstrap_validate(r5_synth, span_start, span_start + pd.Timedelta(days=n_days))
    report(res, 'R5_synthetic (deployable, ~$49/yr)')

    # Synthetic "successful candidate" hypothetical: daily 0.25%, 3 trades/day
    pnl_good = rng.normal(loc=0.083, scale=0.05, size=n_days * 3)
    timestamps_good = [span_start + pd.Timedelta(days=i / 3) for i in range(n_days * 3)]
    good_synth = pd.DataFrame({'close_ts': timestamps_good, 'net_pnl_pct': pnl_good})
    res_good = bootstrap_validate(good_synth, span_start, span_start + pd.Timedelta(days=n_days))
    report(res_good, 'Hypothetical_successful_candidate (daily 0.25%, 3/day)')

    # R26 BT (in-sample +0.21%/day = synthetic)
    pnl_r26 = rng.normal(loc=0.07, scale=0.5, size=n_days * 3)  # 3 trades/day, high variance
    timestamps_r26 = [span_start + pd.Timedelta(days=i / 3) for i in range(n_days * 3)]
    r26_synth = pd.DataFrame({'close_ts': timestamps_r26, 'net_pnl_pct': pnl_r26})
    res_r26 = bootstrap_validate(r26_synth, span_start, span_start + pd.Timedelta(days=n_days))
    report(res_r26, 'R26_BT_synthetic (in-sample +0.21%/day, high variance)')


if __name__ == '__main__':
    calibrate_against_kb()
