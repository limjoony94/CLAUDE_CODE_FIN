"""N8b — Cross-asset Macro Regime SWEEP.

N8 surface: +41%/720d, W4 -29.53% borderline (audit BORDERLINE).
Sweep retry — threshold + lookback potential.

Pre-registered grid:
  corr_lookback_days:    [14, 30, 60, 90]
  risk_on_thresh:        [0.3, 0.4, 0.5]
  risk_off_thresh:       [0.2, 0.3, 0.4]
  usd_strong_thresh:     [-0.5, -0.4, -0.3]
= 4×3×3×3 = 108 configs

Friction LOCKED 0.04% maker × 2 = 0.08%/RT
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts' / 'strategy_lab'))
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))
from mechanism_sweep_standard import MechanismSweep
from c2_design.n8_macro_regime_bt import fetch_btc_daily, fetch_macro

DATA = ROOT / 'data'
RESULTS = ROOT / 'results'
FRICTION_RT_PCT = 0.08
CAPITAL = 1000


_DATA_CACHE = None


def get_macro_data():
    global _DATA_CACHE
    if _DATA_CACHE is None:
        print('Fetching BTC + macro data (one-time)...')
        btc = fetch_btc_daily()
        macro = fetch_macro()
        df = pd.DataFrame({'BTC': btc})
        for k, s in macro.items():
            if s.index.tz is None:
                s.index = s.index.tz_localize('UTC')
            df[k] = s.reindex(df.index, method='ffill')
        df['BTC_ret'] = df['BTC'].pct_change()
        for k in macro.keys():
            df[f'{k}_ret'] = df[k].pct_change()
        _DATA_CACHE = df
    return _DATA_CACHE


def run_strategy(df, params):
    df = df.copy()
    lb = params['corr_lookback_days']
    risk_on_thr = params['risk_on_thresh']
    risk_off_thr = params['risk_off_thresh']
    usd_thr = params['usd_strong_thresh']

    macro_keys = ['DXY', 'SPY', 'GLD']
    for k in macro_keys:
        if f'{k}_ret' not in df.columns:
            return pd.DataFrame()
        df[f'corr_BTC_{k}'] = df['BTC_ret'].rolling(lb).corr(df[f'{k}_ret']).shift(1)

    def classify(row):
        if pd.isna(row.get('corr_BTC_SPY')):
            return 'NEUTRAL'
        if row['corr_BTC_SPY'] > risk_on_thr:
            return 'RISK_ON'
        if row.get('corr_BTC_GLD', 0) > risk_off_thr:
            return 'RISK_OFF'
        if row.get('corr_BTC_DXY', 0) < usd_thr:
            return 'USD_STRONG'
        return 'NEUTRAL'

    df['regime'] = df.apply(classify, axis=1)

    fric_per_round = 2 * FRICTION_RT_PCT / 2 / 100 * CAPITAL  # 0.08% RT total
    active = None
    trades = []
    for ts, row in df.iterrows():
        regime = row['regime']
        btc_p = row['BTC']
        if pd.isna(btc_p):
            continue
        target = None
        if regime == 'RISK_ON':
            target = 'LONG'
        elif regime in ('RISK_OFF', 'USD_STRONG'):
            target = 'SHORT'

        if active is None and target is not None:
            active = {'side': target, 'enter_price': btc_p, 'enter_ts': ts}
        elif active is not None:
            if target != active['side']:
                d_ret = (btc_p - active['enter_price']) / active['enter_price']
                pnl = CAPITAL * d_ret if active['side'] == 'LONG' else -CAPITAL * d_ret
                net_pnl = pnl - fric_per_round
                trades.append({
                    'close_ts': ts,
                    'gross_pct': pnl / CAPITAL * 100,
                    'net_pnl_pct': net_pnl / CAPITAL * 100,
                })
                if target is not None:
                    active = {'side': target, 'enter_price': btc_p, 'enter_ts': ts}
                else:
                    active = None
    return pd.DataFrame(trades)


class N8bSweep(MechanismSweep):
    label = 'n8b_macro_regime'
    mechanism_description = 'N8b — Macro Regime BTC vs DXY/SPY/GLD (sweep)'
    TS_COL = 'date'

    PARAM_GRID = {
        'corr_lookback_days':  [14, 30, 60, 90],
        'risk_on_thresh':      [0.3, 0.4, 0.5],
        'risk_off_thresh':     [0.2, 0.3, 0.4],
        'usd_strong_thresh':   [-0.5, -0.4, -0.3],
    }

    def build_trades(self, df_segment, config):
        full = get_macro_data()
        ts_min = df_segment[self.TS_COL].min()
        ts_max = df_segment[self.TS_COL].max()
        seg = full[(full.index >= ts_min) & (full.index <= ts_max)]
        if len(seg) < config['corr_lookback_days'] + 5:
            return pd.DataFrame()
        return run_strategy(seg, config)


def main():
    df = get_macro_data()
    print(f'Macro data: {len(df)} days, {df.index.min()} → {df.index.max()}')
    df_seg = pd.DataFrame({'date': df.index})
    sweep = N8bSweep()
    result = sweep.run_sweep(df_seg, RESULTS)
    if not result.deployable:
        print('\n→ N8b sweep: 0 OOS-passing configs.')


if __name__ == '__main__':
    main()
