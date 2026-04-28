"""M3-R33 — Multi-strategy ensemble portfolio.

Run 3 strategies concurrently:
1. C1 production (15m Donchian + body + production exit)
2. R21 native 5m (pattern reversal at extreme + structural exit)
3. R24 native 5m (pullback continuation)

Compute daily PnL of each, correlations, combined portfolio stats.
Real traders run multiple uncorrelated strategies for diversification.
"""
import sys, json, random
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))

from m3_round29_c1_exact_revalidation import prepare_15m_data
from m3_round30_c1_production_exact import entry_c1_production, run_bt_c1_production
from m3_round20_dynamic_scalping import prepare_5m_data
from m3_round21_pattern_structure import (add_sma200_1h, entry_psi_prime, run_bt_structural)
from m3_round24_pullback_continuation import add_ema_indicators, entry_pullback


def trades_to_daily(trades):
    """Aggregate trades to daily PnL series."""
    if not trades:
        return pd.Series(dtype=float)
    df_t = pd.DataFrame(trades)
    df_t['exit_dt'] = pd.to_datetime(df_t['exit_ts'])
    df_t['date'] = df_t['exit_dt'].dt.date
    daily = df_t.groupby('date')['net_pct'].sum()
    return daily


def main():
    print("=" * 100)
    print("M3-R33 — Multi-strategy ensemble portfolio")
    print("=" * 100)

    # Strategy 1: C1 production on 15m
    print("\n[Strategy 1] C1 production (15m)...")
    df_15m, valid_15m = prepare_15m_data()
    sigs_c1 = entry_c1_production(df_15m, valid_15m)
    trades_c1 = run_bt_c1_production(df_15m, sigs_c1, friction=0.07)
    print(f"  C1 trades: {len(trades_c1)}")
    daily_c1 = trades_to_daily(trades_c1)
    print(f"  Daily series: {len(daily_c1)} days, mean={daily_c1.mean():+.4f}%, std={daily_c1.std():.4f}%")

    # Strategy 2: R21 5m
    print("\n[Strategy 2] R21 native 5m (pattern reversal at extreme)...")
    df_5m, h1, h4, valid_5m = prepare_5m_data()
    df_5m = add_sma200_1h(df_5m)
    df_5m = add_ema_indicators(df_5m)
    valid_5m_full = valid_5m & (~df_5m['sma200_long'].isna()).values & (~pd.isna(df_5m['ema20_5m']).values) & (~pd.isna(df_5m['ema50_5m']).values)
    sigs_r21 = entry_psi_prime(df_5m, h1, h4, valid_5m_full)
    trades_r21 = run_bt_structural(df_5m, sigs_r21, friction_tp=0.04, friction_sl=0.07)
    print(f"  R21 trades: {len(trades_r21)}")
    # Adjust trade keys (R21 uses gross_pct/net_pct already)
    daily_r21 = trades_to_daily(trades_r21)
    print(f"  Daily series: {len(daily_r21)} days, mean={daily_r21.mean():+.4f}%, std={daily_r21.std():.4f}%")

    # Strategy 3: R24 5m
    print("\n[Strategy 3] R24 native 5m (pullback continuation)...")
    sigs_r24 = entry_pullback(df_5m, h1, h4, valid_5m_full)
    # R24 uses run_bt_pullback (similar interface)
    from m3_round24_pullback_continuation import run_bt_pullback
    trades_r24 = run_bt_pullback(df_5m, sigs_r24, friction_tp=0.04, friction_sl=0.07)
    print(f"  R24 trades: {len(trades_r24)}")
    daily_r24 = trades_to_daily(trades_r24)
    print(f"  Daily series: {len(daily_r24)} days, mean={daily_r24.mean():+.4f}%, std={daily_r24.std():.4f}%")

    # Combine into single dataframe with all dates
    all_dates = pd.date_range(
        start=min(daily_c1.index.min(), daily_r21.index.min(), daily_r24.index.min()),
        end=max(daily_c1.index.max(), daily_r21.index.max(), daily_r24.index.max()),
        freq='D'
    ).date
    daily_df = pd.DataFrame(index=all_dates)
    daily_df['C1'] = daily_c1.reindex(all_dates, fill_value=0)
    daily_df['R21'] = daily_r21.reindex(all_dates, fill_value=0)
    daily_df['R24'] = daily_r24.reindex(all_dates, fill_value=0)

    print(f"\n{'='*80}\nIndividual strategy daily stats\n{'='*80}")
    print(f"{'strategy':<8} {'mean':>10} {'std':>10} {'sharpe':>10} {'days_pos':>10} {'days_total':>10}")
    for col in ('C1', 'R21', 'R24'):
        arr = daily_df[col].values
        nz = arr[arr != 0]  # only count active days
        if len(nz) > 0:
            mean = np.mean(arr)
            std = np.std(arr)
            sharpe = mean / std * np.sqrt(365) if std > 0 else 0
            print(f"{col:<8} {mean:>+9.4f}% {std:>9.4f}% {sharpe:>+9.3f} {(arr > 0).sum():>10} {len(arr):>10}")

    # Correlations
    print(f"\n{'='*80}\nCorrelation matrix (daily PnL)\n{'='*80}")
    corr_mat = daily_df.corr()
    print(corr_mat.to_string())

    # Combined portfolio (equal-weight: 1/3 each strategy capital)
    weights = {'C1': 1/3, 'R21': 1/3, 'R24': 1/3}
    daily_df['portfolio_eq'] = sum(daily_df[s] * w for s, w in weights.items())

    # Active-only weighting (each strategy gets full 1× when active)
    daily_df['portfolio_full'] = daily_df['C1'] + daily_df['R21'] + daily_df['R24']

    print(f"\n{'='*80}\nPortfolio comparison\n{'='*80}")
    print(f"{'portfolio':<25} {'mean':>10} {'std':>10} {'sharpe':>10} {'max_dd':>10}")
    for col in ('C1', 'R21', 'R24', 'portfolio_eq', 'portfolio_full'):
        arr = daily_df[col].values
        mean = np.mean(arr)
        std = np.std(arr)
        sharpe = mean / std * np.sqrt(365) if std > 0 else 0
        # Equity curve
        eq = (1 + arr / 100).cumprod()
        running_max = pd.Series(eq).cummax().values
        dd = (eq - running_max) / running_max * 100
        max_dd = dd.min()
        print(f"{col:<25} {mean:>+9.4f}% {std:>9.4f}% {sharpe:>+9.3f} {max_dd:>9.2f}%")

    # 3-day bootstrap on combined portfolio
    print(f"\n{'='*80}\nBootstrap 1000 × 3-day on portfolio_eq (equal weight 1/3 each)\n{'='*80}")
    daily_arr = daily_df['portfolio_eq'].values
    random.seed(42)
    n_dates = len(daily_arr)
    if n_dates >= 3:
        starts = random.sample(range(n_dates - 3), min(1000, n_dates - 3))
        cand_pnls = []
        for st in starts:
            cand_pnls.append(daily_arr[st:st+3].sum())
        mean_p = float(np.mean(cand_pnls))
        pos_rate = float(np.mean(np.array(cand_pnls) > 0))
        p5 = float(np.percentile(cand_pnls, 5))
        print(f"  mean={mean_p:+.4f}%  pos_rate={pos_rate:.4f}  p5={p5:+.4f}%")

    # 333-day window check
    print(f"\n{'='*80}\n333-day window check on portfolio\n{'='*80}")
    if n_dates >= 333:
        for label, slc in [('First 333d', slice(0, 333)), ('Mid 333d', slice((n_dates-333)//2, (n_dates-333)//2 + 333)),
                           ('Last 333d', slice(n_dates-333, n_dates))]:
            sub = daily_df.iloc[slc]
            for col in ('C1', 'portfolio_eq', 'portfolio_full'):
                arr = sub[col].values
                mean = np.mean(arr)
                std = np.std(arr)
                pos = (arr > 0).sum()
                print(f"  {label:<12} {col:<20} mean={mean:+.4f}% std={std:.4f}% pos_days={pos}/{len(arr)}")

    # Verdict (against user strict criteria)
    print(f"\n{'='*80}\nM3-R33 VERDICT (portfolio strategies)\n{'='*80}")
    portfolio_eq_mean = daily_df['portfolio_eq'].mean()
    portfolio_full_mean = daily_df['portfolio_full'].mean()
    print(f"  Portfolio EQ-weight daily: {portfolio_eq_mean:+.4f}% (target ≥+0.2%)")
    print(f"  Portfolio FULL daily: {portfolio_full_mean:+.4f}%")
    print(f"  Strict ≥+0.2%/day: {portfolio_eq_mean >= 0.2}")

    out = {'date': datetime.now(timezone.utc).isoformat(),
           'individual_stats': {
               col: {'mean': float(daily_df[col].mean()), 'std': float(daily_df[col].std()),
                      'pos_days': int((daily_df[col] > 0).sum()),
                      'total_days': int(len(daily_df[col]))}
               for col in ('C1', 'R21', 'R24')},
           'correlation_matrix': corr_mat.to_dict(),
           'portfolio_eq_mean_daily': float(portfolio_eq_mean),
           'portfolio_full_mean_daily': float(portfolio_full_mean),
           'bootstrap': {'mean': mean_p, 'pos_rate': pos_rate, 'p5': p5} if n_dates >= 3 else None,
           'strict_pass_eq_daily': bool(portfolio_eq_mean >= 0.2)}
    p = ROOT / 'results' / f'm3_r33_ensemble_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
