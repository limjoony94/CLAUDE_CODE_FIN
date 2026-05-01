"""Mechanism Sweep Standard Framework — pre-registered parameter sweep.

User critique (2026-05-01): single-config falsification은 충분치 않음.
Parameter sweep으로 mechanism potential 측정 의무.

Anti-fishing protection:
  - Parameter space는 pre-reg에서 LOCK
  - 50/25/25 train/val/fresh-OOS split
  - IS sweep → top-K (default 5) → val confirm → fresh OOS 1회만
  - Per-mechanism Bonferroni (mechanism 단위 pre-reg)
  - 모든 config 결과 보고 (cherry-pick 차단)

Usage:
  - Subclass MechanismSweep, implement build_signals(df, params)
  - Define PARAM_GRID dict
  - Call run_sweep(data_path, label)

Outputs:
  - results/{label}_sweep_{ts}.json — 모든 config IS/val/OOS 결과
  - 사용자 criteria PASS/FAIL per config
  - Best-per-stage table
"""
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd

from bootstrap_validator import bootstrap_validate, DEFAULT_CRITERIA


@dataclass
class StageResult:
    """Result for one (config, stage) combination."""
    config: Dict[str, Any]
    stage: str  # 'IS' | 'VAL' | 'OOS'
    span_days: float
    n_trades: int
    avg_gross_pct: float
    avg_net_pct: float
    cum_net_pct: float
    daily_net_pct: float
    wr: float
    bootstrap_mean_daily: float
    bootstrap_pos_rate: float
    bootstrap_p5_daily: float
    bootstrap_avg_per_trade: float
    bootstrap_overall_pass: bool
    bootstrap_pass_criteria: Dict[str, bool]
    f1_avg_gross_pass: bool  # > 0.07%
    f6_full_n_pass: bool     # >= 50 trades
    overall_pass: bool       # F1 + bootstrap


@dataclass
class SweepResult:
    label: str
    mechanism: str
    timestamp: str
    param_grid: Dict[str, List[Any]]
    n_configs: int
    split_ratio: Tuple[float, float, float]
    is_results: List[StageResult]
    val_results: List[StageResult]
    oos_results: List[StageResult]
    is_pass_count: int
    val_pass_count: int
    oos_pass_count: int
    deployable: bool


def _make_config_grid(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Cartesian product of param grid → list of config dicts."""
    keys = list(grid.keys())
    vals_list = [grid[k] for k in keys]
    configs = []
    for combo in product(*vals_list):
        configs.append(dict(zip(keys, combo)))
    return configs


def _split_data(df: pd.DataFrame, ts_col: str, ratio: Tuple[float, float, float]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """50/25/25 split (or arbitrary ratio summing to 1.0) by timestamp."""
    assert abs(sum(ratio) - 1.0) < 1e-6, f'Ratio must sum to 1.0, got {ratio}'
    df = df.sort_values(ts_col).reset_index(drop=True)
    n = len(df)
    i1 = int(n * ratio[0])
    i2 = int(n * (ratio[0] + ratio[1]))
    return df.iloc[:i1].copy(), df.iloc[i1:i2].copy(), df.iloc[i2:].copy()


def _evaluate_trades(trades_df: pd.DataFrame, span_days: float, config: Dict, stage: str) -> StageResult:
    """Compute all metrics + bootstrap for trade history. Returns StageResult."""
    if len(trades_df) == 0:
        return StageResult(
            config=config, stage=stage, span_days=span_days, n_trades=0,
            avg_gross_pct=0.0, avg_net_pct=0.0, cum_net_pct=0.0, daily_net_pct=0.0,
            wr=0.0,
            bootstrap_mean_daily=0.0, bootstrap_pos_rate=0.0, bootstrap_p5_daily=0.0,
            bootstrap_avg_per_trade=0.0,
            bootstrap_overall_pass=False, bootstrap_pass_criteria={},
            f1_avg_gross_pass=False, f6_full_n_pass=False, overall_pass=False,
        )

    avg_gross = float(trades_df['gross_pct'].mean())
    avg_net = float(trades_df['net_pnl_pct'].mean())
    cum_net = float(trades_df['net_pnl_pct'].sum())
    daily_net = cum_net / span_days if span_days > 0 else 0.0
    wr = float((trades_df['net_pnl_pct'] > 0).mean())
    n_trades = len(trades_df)

    df_bt = trades_df.copy()
    df_bt['close_ts'] = pd.to_datetime(df_bt['close_ts'])
    span_min = df_bt['close_ts'].min()
    span_max = df_bt['close_ts'].max()

    res = bootstrap_validate(df_bt, span_min, span_max)

    f1 = avg_gross > 0.07
    f6 = n_trades >= 50

    return StageResult(
        config=config, stage=stage, span_days=span_days, n_trades=n_trades,
        avg_gross_pct=avg_gross, avg_net_pct=avg_net,
        cum_net_pct=cum_net, daily_net_pct=daily_net, wr=wr,
        bootstrap_mean_daily=float(res.mean_daily_pct),
        bootstrap_pos_rate=float(res.pos_rate),
        bootstrap_p5_daily=float(res.p5_daily_pct),
        bootstrap_avg_per_trade=float(res.avg_per_trade_pct),
        bootstrap_overall_pass=bool(res.overall_pass),
        bootstrap_pass_criteria={k: bool(v) for k, v in res.pass_criteria.items()},
        f1_avg_gross_pass=bool(f1),
        f6_full_n_pass=bool(f6),
        overall_pass=bool(f1 and f6 and res.overall_pass),
    )


class MechanismSweep:
    """Subclass and implement build_trades(df_segment, config) → trades DataFrame.

    trades DataFrame columns required:
      'close_ts' (datetime), 'gross_pct' (float), 'net_pnl_pct' (float)
    """

    label = 'mechanism'
    mechanism_description = 'override'

    PARAM_GRID: Dict[str, List[Any]] = {}

    SPLIT_RATIO: Tuple[float, float, float] = (0.50, 0.25, 0.25)

    TS_COL = 'timestamp'

    TOP_K_FOR_VAL = 5  # IS top-K → val
    OOS_REQUIRES_VAL_PASS = True  # val에서 PASS인 것만 OOS로

    def build_trades(self, df_segment: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        raise NotImplementedError

    def run_sweep(self, df: pd.DataFrame, output_dir: Path, label: Optional[str] = None) -> SweepResult:
        label = label or self.label
        configs = _make_config_grid(self.PARAM_GRID)
        n_configs = len(configs)
        print('=' * 100)
        print(f'Mechanism Sweep: {self.mechanism_description}')
        print('=' * 100)
        print(f'Param grid:')
        for k, v in self.PARAM_GRID.items():
            print(f'  {k}: {v}')
        print(f'Total configs: {n_configs}')

        df_is, df_val, df_oos = _split_data(df, self.TS_COL, self.SPLIT_RATIO)
        is_span = (df_is[self.TS_COL].max() - df_is[self.TS_COL].min()).total_seconds() / 86400
        val_span = (df_val[self.TS_COL].max() - df_val[self.TS_COL].min()).total_seconds() / 86400
        oos_span = (df_oos[self.TS_COL].max() - df_oos[self.TS_COL].min()).total_seconds() / 86400

        print(f'\nSplit ({self.SPLIT_RATIO}):')
        print(f'  IS : {is_span:.0f}d ({len(df_is):,} bars)')
        print(f'  VAL: {val_span:.0f}d ({len(df_val):,} bars)')
        print(f'  OOS: {oos_span:.0f}d ({len(df_oos):,} bars)')

        # IS sweep
        print(f'\n[IS] sweeping {n_configs} configs...')
        is_results = []
        for i, cfg in enumerate(configs):
            if (i + 1) % max(1, n_configs // 10) == 0:
                print(f'  [{i+1}/{n_configs}] {cfg}')
            trades = self.build_trades(df_is, cfg)
            is_results.append(_evaluate_trades(trades, is_span, cfg, 'IS'))

        is_pass = [r for r in is_results if r.overall_pass]
        print(f'\n[IS] {len(is_pass)}/{n_configs} configs PASS')

        # Top-K by IS daily_net for val
        is_sorted = sorted(is_results, key=lambda r: r.daily_net_pct, reverse=True)
        top_k = is_sorted[:self.TOP_K_FOR_VAL]
        print(f'\n[VAL] testing IS top-{len(top_k)} configs by daily_net:')
        for r in top_k:
            print(f'  daily_net={r.daily_net_pct:+.4f}%, n={r.n_trades}, IS-overall={r.overall_pass}: {r.config}')

        val_results = []
        for r in top_k:
            trades_val = self.build_trades(df_val, r.config)
            val_results.append(_evaluate_trades(trades_val, val_span, r.config, 'VAL'))

        val_pass = [r for r in val_results if r.overall_pass]
        print(f'\n[VAL] {len(val_pass)}/{len(top_k)} configs PASS')

        # OOS — only val-pass configs
        if self.OOS_REQUIRES_VAL_PASS:
            oos_configs = [r.config for r in val_results if r.overall_pass]
        else:
            oos_configs = [r.config for r in val_results]

        print(f'\n[OOS] {len(oos_configs)} configs proceed to fresh OOS test')
        oos_results = []
        for cfg in oos_configs:
            trades_oos = self.build_trades(df_oos, cfg)
            oos_results.append(_evaluate_trades(trades_oos, oos_span, cfg, 'OOS'))

        oos_pass = [r for r in oos_results if r.overall_pass]
        print(f'\n[OOS] {len(oos_pass)}/{len(oos_configs)} configs PASS — DEPLOYABLE')

        deployable = len(oos_pass) > 0

        sweep_result = SweepResult(
            label=label,
            mechanism=self.mechanism_description,
            timestamp=datetime.now(timezone.utc).isoformat(),
            param_grid=self.PARAM_GRID,
            n_configs=n_configs,
            split_ratio=self.SPLIT_RATIO,
            is_results=is_results,
            val_results=val_results,
            oos_results=oos_results,
            is_pass_count=len(is_pass),
            val_pass_count=len(val_pass),
            oos_pass_count=len(oos_pass),
            deployable=deployable,
        )

        # Save
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = output_dir / f'{label}_sweep_{ts}.json'

        def _serialize(obj):
            if isinstance(obj, StageResult):
                return asdict(obj)
            if isinstance(obj, SweepResult):
                return asdict(obj)
            return obj

        out_dict = asdict(sweep_result)
        with open(out_path, 'w') as f:
            json.dump(out_dict, f, indent=2, default=str)

        # Summary print
        print('\n' + '=' * 100)
        print(f'SWEEP SUMMARY')
        print('=' * 100)
        print(f'IS  PASS: {len(is_pass)}/{n_configs}')
        print(f'VAL PASS: {len(val_pass)}/{len(top_k)}')
        print(f'OOS PASS: {len(oos_pass)}/{len(oos_configs)}')
        print(f'DEPLOYABLE: {"✅ YES" if deployable else "🔴 NO"}')

        if oos_pass:
            print(f'\n--- OOS-passing configs (DEPLOYABLE) ---')
            for r in oos_pass:
                print(f'  daily_net={r.daily_net_pct:+.4f}%, avg_gross={r.avg_gross_pct:+.4f}%, '
                      f'n={r.n_trades}, WR={r.wr:.3f}: {r.config}')

        # Best regardless of pass/fail
        if is_results:
            best_is = max(is_results, key=lambda r: r.daily_net_pct)
            print(f'\n--- Best IS by daily_net ---')
            print(f'  daily={best_is.daily_net_pct:+.4f}%, n={best_is.n_trades}, '
                  f'avg_gross={best_is.avg_gross_pct:+.4f}%, '
                  f'overall={best_is.overall_pass}: {best_is.config}')

        print(f'\nSaved: {out_path}')
        return sweep_result
