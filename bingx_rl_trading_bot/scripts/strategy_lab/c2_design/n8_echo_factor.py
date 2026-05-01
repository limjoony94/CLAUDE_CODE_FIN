"""N8 echo factor — 8-coin correlation measurement.

가설: 24 nominal signals (8 coins × 3 macro)는 BTC dominance 때문에
실질적으로 1×8 echo에 가깝다. Math gate (+0.20%/day) 도달 가능성 판정.

측정:
  1. 8-coin pairwise daily return correlation matrix
  2. ρ_avg (off-diagonal mean)
  3. Effective independent N = N / (1 + (N-1) × ρ_avg)
  4. Variance reduction factor = sqrt(N_eff / N)
  5. 만약 ρ_avg > 0.7 → echo, expansion 무효
     ρ_avg < 0.4 → diversification 가치, develop 진행
     0.4-0.7 → borderline

Output: 정량 결정 (develop GO/NOGO).
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'


def main():
    print('=' * 100)
    print('N8 ECHO FACTOR — 8-coin correlation measurement')
    print('=' * 100)

    df = pd.read_csv(DATA / 'n7_8coin_4h_close.csv', parse_dates=['timestamp'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

    # Pivot to wide format
    wide = df.pivot(index='timestamp', columns='symbol', values='close')
    print(f'Coins: {list(wide.columns)}')
    print(f'Span: {wide.index.min()} → {wide.index.max()}')
    print(f'Bars (4h): {len(wide):,}')

    # Daily resample (last close)
    daily = wide.resample('1D').last().dropna()
    print(f'Daily bars: {len(daily):,}')

    # Daily returns
    rets = daily.pct_change().dropna()
    print(f'Return rows: {len(rets):,}\n')

    # Correlation matrix
    corr = rets.corr()
    print('=== Pairwise daily-return correlation ===')
    print(corr.round(3).to_string())

    # Off-diagonal mean
    n = len(corr.columns)
    mask = ~np.eye(n, dtype=bool)
    rho_avg = corr.values[mask].mean()
    rho_std = corr.values[mask].std()
    rho_min = corr.values[mask].min()
    rho_max = corr.values[mask].max()
    print(f'\n=== Off-diagonal stats ===')
    print(f'  ρ_avg = {rho_avg:.4f}')
    print(f'  ρ_std = {rho_std:.4f}')
    print(f'  ρ_min = {rho_min:.4f}')
    print(f'  ρ_max = {rho_max:.4f}')

    # BTC pairwise
    btc_corrs = corr['BTC'].drop('BTC')
    print(f'\n=== BTC pairwise correlations ===')
    for sym, c in btc_corrs.sort_values(ascending=False).items():
        print(f'  BTC ↔ {sym}: {c:+.3f}')
    print(f'  BTC mean ρ = {btc_corrs.mean():.4f}')

    # Effective independent N
    N = n
    N_eff = N / (1 + (N - 1) * rho_avg)
    var_reduction = np.sqrt(N_eff / N)
    print(f'\n=== Effective independent samples ===')
    print(f'  Nominal N = {N}')
    print(f'  N_eff = {N_eff:.3f}')
    print(f'  Var reduction = {var_reduction:.4f} (1.0 = no reduction)')
    print(f'  Effective gain over 1 coin = {N_eff:.2f}× independent samples')

    # Math gate
    base_daily_pct = 0.057  # N8 base
    target_daily_pct = 0.20  # User criterion
    needed_multiplier = target_daily_pct / base_daily_pct
    print(f'\n=== Math gate ===')
    print(f'  N8 base: +{base_daily_pct:.3f}%/day')
    print(f'  Target:  +{target_daily_pct:.2f}%/day')
    print(f'  Required multiplier: {needed_multiplier:.2f}×')
    print(f'  Frequency-only path: needs {needed_multiplier:.2f}× more independent trades')
    print(f'  Echo expansion delivers: {N_eff:.2f}× → ', end='')
    if N_eff >= needed_multiplier:
        print(f'✅ POSSIBLE (math allows)')
        verdict = 'POSSIBLE'
    else:
        gap = needed_multiplier / N_eff
        print(f'❌ INSUFFICIENT (still {gap:.2f}× short)')
        verdict = 'INSUFFICIENT'

    # Decision
    print(f'\n=== Decision ===')
    if rho_avg > 0.7:
        print(f'  🔴 ρ_avg = {rho_avg:.3f} > 0.7. Echo confirmed. Same-macro 8-coin expansion = leverage, not new edge.')
        decision = 'ECHO_CONFIRMED_NO_8COIN_EXPANSION'
    elif rho_avg > 0.5:
        print(f'  🟡 ρ_avg = {rho_avg:.3f}. Partial echo. 8-coin path marginal value.')
        decision = 'PARTIAL_ECHO_MARGINAL'
    else:
        print(f'  🟢 ρ_avg = {rho_avg:.3f} < 0.5. Diversification real. 8-coin path valid.')
        decision = 'DIVERSIFIED_VALID'

    if verdict == 'INSUFFICIENT':
        print(f'  → Frequency expansion alone CANNOT reach +0.20%/day. Need per-trade edge increase OR uncorrelated macro substrates.')
        next_action = 'PIVOT_TO_UNCORRELATED_MACRO_SUBSTRATES'
    else:
        next_action = 'PROCEED_WITH_8COIN_EXPANSION'
    print(f'  Next: {next_action}')

    # Save
    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'n_coins': N,
        'rho_avg': float(rho_avg),
        'rho_std': float(rho_std),
        'rho_min': float(rho_min),
        'rho_max': float(rho_max),
        'btc_mean_corr': float(btc_corrs.mean()),
        'btc_pairwise': {sym: float(c) for sym, c in btc_corrs.items()},
        'N_effective': float(N_eff),
        'variance_reduction_factor': float(var_reduction),
        'math_gate': {
            'base_daily_pct': base_daily_pct,
            'target_daily_pct': target_daily_pct,
            'required_multiplier': needed_multiplier,
            'echo_delivers_multiplier': float(N_eff),
            'verdict': verdict,
        },
        'echo_decision': decision,
        'next_action': next_action,
    }
    out_path = RESULTS / f'n8_echo_factor_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
