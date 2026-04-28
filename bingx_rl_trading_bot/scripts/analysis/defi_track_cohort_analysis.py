"""DeFi-Track Week 1 Reconnaissance — cohort + survivor + tail analysis.

Per advisor 2026-04-29:
  "Pre-reg should require *cohort* analysis (what fraction of protocols in any
   month became zero or near-zero) not just average APY. If you find a 12%
   historical mean with 5% of protocol-months hitting -100%, the survivor
   bias matters."

This script answers (without yet running any strategy):
  Q1. Per-protocol distribution: median APY, p25/p75, p95/p5 (tail thickness).
  Q2. Zero-month rate: fraction of (pool × month) cells where APY ≤ 0.5%.
  Q3. Survivor curve: of pools alive in cohort month T, how many still report
      data 6 / 12 / 24 months later? (operational survivor bias proxy)
  Q4. Catastrophic-week rate: WoW APY drop ≥ 50% AND tvlUsd drop ≥ 25%
      (proxy for depeg / exploit / IL spike).
  Q5. Cross-section dispersion: how different are top-3 vs bottom-3 each month?
      (rotation strategy needs sustained dispersion to harvest)

Inputs:
  data/defi_yields_panel.parquet   (long-format APY history)
  data/defi_yields_cohort.parquet  (per-pool aggregates)
  data/defi_yields_pools.parquet   (current snapshot with metadata)

Outputs:
  results/defi_track_week1_recon_{ts}.json  (numeric findings)
  results/defi_track_week1_recon_{ts}.txt   (human-readable summary)
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'
RESULTS.mkdir(exist_ok=True)

PANEL_FILE = DATA / 'defi_yields_panel.parquet'
COHORT_FILE = DATA / 'defi_yields_cohort.parquet'
POOLS_FILE = DATA / 'defi_yields_pools.parquet'

ZERO_THRESHOLD_PCT = 0.5     # APY ≤ 0.5% counts as "near-zero"
CATASTROPHE_APY_DROP = 0.5   # ≥50% WoW APY drop
CATASTROPHE_TVL_DROP = 0.25  # ≥25% WoW TVL drop


def load_data():
    panel = pd.read_parquet(PANEL_FILE)
    cohort = pd.read_parquet(COHORT_FILE)
    pools = pd.read_parquet(POOLS_FILE)
    panel['date'] = pd.to_datetime(panel['date'])
    panel = panel.sort_values(['pool_id', 'date']).reset_index(drop=True)
    return panel, cohort, pools


def q1_distribution(panel: pd.DataFrame, cohort: pd.DataFrame) -> dict:
    """Per-project distribution stats."""
    by_proj = cohort.groupby('project').agg(
        n_pools=('pool_id', 'count'),
        apy_median_p50=('apy_median', 'median'),
        apy_median_p25=('apy_median', lambda s: float(np.percentile(s.dropna(), 25))),
        apy_median_p75=('apy_median', lambda s: float(np.percentile(s.dropna(), 75))),
        apy_max_p95=('apy_max', lambda s: float(np.percentile(s.dropna(), 95))),
        apy_min_p5=('apy_min', lambda s: float(np.percentile(s.dropna(), 5))),
        lifetime_median=('lifetime_days', 'median'),
        lifetime_p25=('lifetime_days', lambda s: float(np.percentile(s.dropna(), 25))),
    ).round(3)
    return {
        'by_project': by_proj.to_dict(orient='index'),
    }


def q2_zero_month_rate(panel: pd.DataFrame, cohort: pd.DataFrame) -> dict:
    """Fraction of pool-months with APY ≤ ZERO_THRESHOLD_PCT."""
    p = panel.copy()
    p['ym'] = p['date'].dt.to_period('M').astype(str)
    monthly = p.groupby(['pool_id', 'ym']).agg(apy_med=('apy', 'median')).reset_index()
    monthly['is_zero'] = (monthly['apy_med'].fillna(0) <= ZERO_THRESHOLD_PCT).astype(int)
    proj_map = cohort.set_index('pool_id')['project']
    monthly['project'] = monthly['pool_id'].map(proj_map)
    by_proj = monthly.groupby('project').agg(
        pool_months=('is_zero', 'count'),
        zero_months=('is_zero', 'sum'),
        zero_rate=('is_zero', 'mean'),
    ).round(4)
    overall = {
        'pool_months': int(monthly.shape[0]),
        'zero_months': int(monthly['is_zero'].sum()),
        'zero_rate': float(monthly['is_zero'].mean()),
        'threshold_pct': ZERO_THRESHOLD_PCT,
    }
    return {'overall': overall, 'by_project': by_proj.to_dict(orient='index')}


def q3_survivor_curve(panel: pd.DataFrame, cohort: pd.DataFrame) -> dict:
    """Of pools observed in cohort month T, fraction still reporting at T+6/12/24."""
    p = panel.copy()
    p['ym'] = p['date'].dt.to_period('M')
    pool_months = p.groupby('pool_id')['ym'].agg(['min', 'max', 'nunique']).reset_index()
    pool_months.columns = ['pool_id', 'first_ym', 'last_ym', 'n_months']
    pool_months['lifetime_months'] = pool_months.apply(
        lambda r: (r['last_ym'] - r['first_ym']).n + 1, axis=1
    )
    proj_map = cohort.set_index('pool_id')['project']
    pool_months['project'] = pool_months['pool_id'].map(proj_map)

    horizons = [6, 12, 24]
    surv = {}
    for h in horizons:
        survived = (pool_months['lifetime_months'] >= h).sum()
        surv[f'h{h}m_survivors'] = int(survived)
        surv[f'h{h}m_rate'] = float(survived / len(pool_months))

    by_proj_rows = []
    for proj, sub in pool_months.groupby('project'):
        row = {'project': proj, 'n_pools': int(len(sub))}
        for h in horizons:
            row[f'survives_{h}m'] = int((sub['lifetime_months'] >= h).sum())
            row[f'rate_{h}m'] = round(float((sub['lifetime_months'] >= h).mean()), 4)
        by_proj_rows.append(row)
    return {'overall': surv, 'by_project': by_proj_rows}


def q4_catastrophic_weeks(panel: pd.DataFrame, cohort: pd.DataFrame) -> dict:
    """WoW APY drop ≥50% AND TVL drop ≥25% — proxy for depeg/exploit."""
    p = panel.copy()
    p = p.sort_values(['pool_id', 'date'])
    p['apy_prev7'] = p.groupby('pool_id')['apy'].shift(7)
    p['tvl_prev7'] = p.groupby('pool_id')['tvlUsd'].shift(7)
    p['apy_drop'] = (p['apy_prev7'] - p['apy']) / p['apy_prev7'].replace(0, np.nan)
    p['tvl_drop'] = (p['tvl_prev7'] - p['tvlUsd']) / p['tvl_prev7'].replace(0, np.nan)
    cat = p[(p['apy_drop'] >= CATASTROPHE_APY_DROP) &
            (p['tvl_drop'] >= CATASTROPHE_TVL_DROP)].copy()
    proj_map = cohort.set_index('pool_id')['project']
    cat['project'] = cat['pool_id'].map(proj_map)

    overall = {
        'total_pool_weeks': int(p['apy_drop'].notna().sum()),
        'catastrophe_weeks': int(len(cat)),
        'rate': float(len(cat) / max(p['apy_drop'].notna().sum(), 1)),
        'apy_drop_threshold': CATASTROPHE_APY_DROP,
        'tvl_drop_threshold': CATASTROPHE_TVL_DROP,
    }
    by_proj = cat.groupby('project').size().to_dict()
    samples = cat[['date', 'pool_id', 'project', 'apy_prev7', 'apy', 'tvl_prev7', 'tvlUsd',
                   'apy_drop', 'tvl_drop']].head(20).copy()
    samples['date'] = samples['date'].dt.strftime('%Y-%m-%d')
    return {
        'overall': overall,
        'by_project_count': by_proj,
        'sample_events': samples.to_dict(orient='records'),
    }


def q5_dispersion(panel: pd.DataFrame, cohort: pd.DataFrame) -> dict:
    """Monthly cross-sectional APY dispersion: top-3 mean - bottom-3 mean."""
    p = panel.copy()
    p['ym'] = p['date'].dt.to_period('M').astype(str)
    monthly = p.groupby(['pool_id', 'ym']).agg(apy_med=('apy', 'median')).reset_index()

    rows = []
    for ym, g in monthly.groupby('ym'):
        s = g.dropna(subset=['apy_med']).sort_values('apy_med', ascending=False)
        if len(s) < 7:
            continue
        top3 = s['apy_med'].head(3).mean()
        bot3 = s['apy_med'].tail(3).mean()
        rows.append({'ym': ym, 'n_pools': len(s), 'top3': top3, 'bot3': bot3,
                     'spread': top3 - bot3})
    df = pd.DataFrame(rows)
    if df.empty:
        return {'note': 'insufficient pool count per month'}
    return {
        'months': len(df),
        'spread_mean_pp': float(df['spread'].mean()),
        'spread_p25_pp': float(np.percentile(df['spread'], 25)),
        'spread_median_pp': float(df['spread'].median()),
        'spread_p75_pp': float(np.percentile(df['spread'], 75)),
        'spread_min_pp': float(df['spread'].min()),
        'spread_max_pp': float(df['spread'].max()),
        'months_spread_above_3pp': int((df['spread'] > 3).sum()),
        'months_spread_above_5pp': int((df['spread'] > 5).sum()),
        'months_spread_above_10pp': int((df['spread'] > 10).sum()),
    }


def main():
    panel, cohort, pools = load_data()
    print(f'panel: {len(panel):,} rows × {panel.pool_id.nunique()} pools')
    print(f'date range: {panel.date.min().date()} → {panel.date.max().date()}\n')

    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'panel_summary': {
            'rows': int(len(panel)),
            'pools': int(panel.pool_id.nunique()),
            'date_min': str(panel.date.min().date()),
            'date_max': str(panel.date.max().date()),
        },
        'q1_distribution': q1_distribution(panel, cohort),
        'q2_zero_month_rate': q2_zero_month_rate(panel, cohort),
        'q3_survivor_curve': q3_survivor_curve(panel, cohort),
        'q4_catastrophic_weeks': q4_catastrophic_weeks(panel, cohort),
        'q5_dispersion': q5_dispersion(panel, cohort),
    }

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_p = RESULTS / f'defi_track_week1_recon_{ts}.json'
    txt_p = RESULTS / f'defi_track_week1_recon_{ts}.txt'

    with open(json_p, 'w') as f:
        json.dump(out, f, indent=2, default=str)

    lines = []
    lines.append(f'=== DeFi-Track Week 1 Reconnaissance — {ts} ===\n')
    lines.append(f"panel rows={out['panel_summary']['rows']:,}  "
                 f"pools={out['panel_summary']['pools']}  "
                 f"range={out['panel_summary']['date_min']} → {out['panel_summary']['date_max']}\n")

    lines.append('## Q1 — Per-project APY distribution')
    for proj, stats in out['q1_distribution']['by_project'].items():
        lines.append(f'  {proj:18s}  n={stats["n_pools"]:3d}  '
                     f'median_apy=p25:{stats["apy_median_p25"]:.2f}% / '
                     f'p50:{stats["apy_median_p50"]:.2f}% / '
                     f'p75:{stats["apy_median_p75"]:.2f}%  '
                     f'lifetime_p25={stats["lifetime_p25"]:.0f}d')
    lines.append('')

    z = out['q2_zero_month_rate']
    lines.append('## Q2 — Zero-month rate (APY ≤ 0.5%)')
    lines.append(f"  overall: {z['overall']['zero_months']:,}/{z['overall']['pool_months']:,} = "
                 f"{z['overall']['zero_rate']*100:.2f}% of pool-months")
    for proj, stats in z['by_project'].items():
        lines.append(f"  {proj:18s}  zero_rate={stats['zero_rate']*100:6.2f}%  "
                     f"({int(stats['zero_months'])}/{int(stats['pool_months'])})")
    lines.append('')

    s = out['q3_survivor_curve']['overall']
    lines.append('## Q3 — Survivor curve (lifetime months)')
    lines.append(f"  ≥6m: {s['h6m_survivors']} ({s['h6m_rate']*100:.1f}%)  "
                 f"≥12m: {s['h12m_survivors']} ({s['h12m_rate']*100:.1f}%)  "
                 f"≥24m: {s['h24m_survivors']} ({s['h24m_rate']*100:.1f}%)")
    for row in out['q3_survivor_curve']['by_project']:
        lines.append(f"  {row['project']:18s}  n={row['n_pools']:3d}  "
                     f"6m={row['rate_6m']*100:.0f}%  "
                     f"12m={row['rate_12m']*100:.0f}%  "
                     f"24m={row['rate_24m']*100:.0f}%")
    lines.append('')

    c = out['q4_catastrophic_weeks']['overall']
    lines.append(f"## Q4 — Catastrophic weeks (≥{int(c['apy_drop_threshold']*100)}% APY drop "
                 f"AND ≥{int(c['tvl_drop_threshold']*100)}% TVL drop, WoW)")
    lines.append(f"  {c['catastrophe_weeks']:,}/{c['total_pool_weeks']:,} pool-weeks = "
                 f"{c['rate']*100:.3f}%")
    if out['q4_catastrophic_weeks']['by_project_count']:
        lines.append('  by project:')
        for proj, n in sorted(out['q4_catastrophic_weeks']['by_project_count'].items(),
                              key=lambda x: -x[1]):
            lines.append(f'    {proj:18s}  {n}')
    lines.append('')

    d = out['q5_dispersion']
    lines.append('## Q5 — Cross-sectional dispersion (top-3 vs bottom-3 monthly APY median)')
    if 'note' in d:
        lines.append(f"  {d['note']}")
    else:
        lines.append(f"  months={d['months']}  spread_pp: "
                     f"p25={d['spread_p25_pp']:.2f}  med={d['spread_median_pp']:.2f}  "
                     f"p75={d['spread_p75_pp']:.2f}  mean={d['spread_mean_pp']:.2f}")
        lines.append(f"  months >3pp: {d['months_spread_above_3pp']}/{d['months']}  "
                     f">5pp: {d['months_spread_above_5pp']}/{d['months']}  "
                     f">10pp: {d['months_spread_above_10pp']}/{d['months']}")
    lines.append('')

    txt = '\n'.join(lines)
    with open(txt_p, 'w', encoding='utf-8') as f:
        f.write(txt)
    print(txt)
    print(f'saved: {json_p}')
    print(f'saved: {txt_p}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
