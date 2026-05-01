"""N8 — Cross-asset Macro Regime BT.

Mechanism:
  - 30d rolling correlation between BTC daily returns and macro asset returns:
    * BTC vs DXY (US Dollar Index, inverse risk-on)
    * BTC vs SPY (S&P 500, risk-on proxy)
    * BTC vs GLD (Gold, safe-haven)
  - Regime detection:
    * Risk-on: BTC ↔ SPY corr > 0.4 → long BTC
    * Risk-off: BTC ↔ GLD corr > 0.3 → short BTC (BTC behaves as risk asset)
    * USD strong: BTC ↔ DXY corr < -0.4 → short BTC
    * Neutral: idle
  - Position size: $1000 capital × 1× leverage = $1000 perp position
  - Hold until regime flip; daily check
  - Exit on regime change

Friction: maker entry/exit (taker if regime flips suddenly)
  0.04% maker × 2 sides = 0.08% RT per trade

Bootstrap: 사용자 criteria 적용
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bootstrap_validator import bootstrap_validate, report as bootstrap_report


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'


LOCKED = {
    'capital_usd': 1000,
    'corr_lookback_days': 30,
    'risk_on_corr_thresh': 0.4,
    'risk_off_corr_thresh': 0.3,
    'usd_strong_corr_thresh': -0.4,
    'maker_fric_pct': 0.04,
    'slippage_pct': 0.02,
}


def fetch_btc_daily():
    """Resample existing 4h BingX BTC to daily."""
    df = pd.read_csv(DATA / 'n7_8coin_4h_close.csv', parse_dates=['timestamp'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    btc = df[df.symbol == 'BTC'].set_index('timestamp')['close']
    btc_daily = btc.resample('1D').last().dropna()
    return btc_daily


def fetch_macro():
    """yfinance daily DXY, SPY, GLD."""
    end = pd.Timestamp.utcnow().normalize()
    start = end - pd.Timedelta(days=730)
    macro = {}
    for ticker, name in [('DX-Y.NYB', 'DXY'), ('SPY', 'SPY'), ('GLD', 'GLD')]:
        try:
            d = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if d.empty:
                print(f'  {ticker}: empty')
                continue
            close_col = d['Close']
            if isinstance(close_col, pd.DataFrame):
                close_col = close_col.iloc[:, 0]
            macro[name] = close_col
            print(f'  {ticker} ({name}): {len(d):,} bars, {d.index.min().date()} → {d.index.max().date()}')
        except Exception as e:
            print(f'  {ticker} err: {e}')
    return macro


def simulate(btc_daily, macro):
    """Daily check regime, enter/exit BTC position."""
    # Align all to daily
    df = pd.DataFrame({'BTC': btc_daily})
    for k, s in macro.items():
        # Convert macro index to UTC daily
        if s.index.tz is None:
            s.index = s.index.tz_localize('UTC')
        df[k] = s.reindex(df.index, method='ffill')

    df['BTC_ret'] = df['BTC'].pct_change()
    for k in macro.keys():
        df[f'{k}_ret'] = df[k].pct_change()

    # Rolling correlations — SHIFT(1) to avoid look-ahead bias
    # corr at bar i must use information up to bar i-1 (decision made BEFORE bar i price move)
    lb = LOCKED['corr_lookback_days']
    for k in macro.keys():
        df[f'corr_BTC_{k}'] = df['BTC_ret'].rolling(lb).corr(df[f'{k}_ret']).shift(1)

    # Regime classification
    def classify(row):
        if pd.isna(row.get('corr_BTC_SPY')):
            return 'NEUTRAL'
        if row['corr_BTC_SPY'] > LOCKED['risk_on_corr_thresh']:
            return 'RISK_ON'
        if row.get('corr_BTC_GLD', 0) > LOCKED['risk_off_corr_thresh']:
            return 'RISK_OFF'
        if row.get('corr_BTC_DXY', 0) < LOCKED['usd_strong_corr_thresh']:
            return 'USD_STRONG'
        return 'NEUTRAL'

    df['regime'] = df.apply(classify, axis=1)

    # Position by regime
    capital = LOCKED['capital_usd']
    fric_per_side = LOCKED['maker_fric_pct'] + LOCKED['slippage_pct']
    fric_per_round = 2 * fric_per_side / 100 * capital

    active = None
    trades = []
    for ts, row in df.iterrows():
        regime = row['regime']
        btc_price = row['BTC']
        if pd.isna(btc_price):
            continue

        # Determine target position by regime
        target = None
        if regime == 'RISK_ON':
            target = 'LONG'
        elif regime in ('RISK_OFF', 'USD_STRONG'):
            target = 'SHORT'
        # NEUTRAL → idle

        if active is None and target is not None:
            active = {'side': target, 'enter_price': btc_price, 'enter_ts': ts}
        elif active is not None:
            if target != active['side']:   # regime change → close
                d_ret = (btc_price - active['enter_price']) / active['enter_price']
                if active['side'] == 'LONG':
                    pnl = capital * d_ret
                else:
                    pnl = -capital * d_ret
                net_pnl = pnl - fric_per_round
                trades.append({
                    'side': active['side'],
                    'enter_ts': str(active['enter_ts']), 'close_ts': str(ts),
                    'enter_price': active['enter_price'], 'exit_price': btc_price,
                    'gross_pct': pnl / capital * 100,
                    'net_pnl_pct': net_pnl / capital * 100,
                    'duration_days': (ts - active['enter_ts']).days,
                    'next_regime': regime,
                })
                # Open new if target is LONG/SHORT
                if target is not None:
                    active = {'side': target, 'enter_price': btc_price, 'enter_ts': ts}
                else:
                    active = None
    return trades, df


def main():
    print('=' * 100)
    print('N8 — Cross-asset Macro Regime BT')
    print('=' * 100)
    print(f'Locked: corr lookback {LOCKED["corr_lookback_days"]}d, '
          f'thresholds (SPY>{LOCKED["risk_on_corr_thresh"]} on, '
          f'GLD>{LOCKED["risk_off_corr_thresh"]} off, '
          f'DXY<{LOCKED["usd_strong_corr_thresh"]} strong)\n')

    print('Fetching BTC daily (resample 4h → 1d)...')
    btc_daily = fetch_btc_daily()
    print(f'  BTC daily: {len(btc_daily):,} bars, {btc_daily.index.min()} → {btc_daily.index.max()}\n')

    print('Fetching macro (DXY/SPY/GLD via yfinance)...')
    macro = fetch_macro()
    if len(macro) < 3:
        print('Insufficient macro data')
        return
    print()

    trades, df = simulate(btc_daily, macro)
    print(f'\n=== Regime distribution ===')
    print(df['regime'].value_counts().to_string())
    print()

    if not trades:
        print('No trades generated.')
        return

    df_t = pd.DataFrame(trades)
    span_days = (df.index.max() - df.index.min()).days
    cum_pct = float(df_t['net_pnl_pct'].sum())
    apy = cum_pct / span_days * 365
    print(f'=== Aggregate ===')
    print(f'  span: {span_days} days')
    print(f'  n_trades: {len(df_t)}')
    print(f'  cum_net_pct: {cum_pct:+.4f}%')
    print(f'  APY: {apy:+.4f}%')
    print(f'  avg_per_trade_net: {df_t["net_pnl_pct"].mean():+.4f}%')
    print(f'  avg_per_trade_gross: {df_t["gross_pct"].mean():+.4f}%')
    print(f'  WR: {(df_t["net_pnl_pct"] > 0).mean():.3f}')
    print(f'  median duration: {df_t["duration_days"].median():.0f} days')

    print('\n=== Bootstrap (사용자 criteria) ===')
    df_t['close_ts'] = pd.to_datetime(df_t['close_ts'])
    res = bootstrap_validate(df_t, df.index.min(), df.index.max())
    bootstrap_report(res, 'N8 macro regime')

    out_path = RESULTS / f'n8_first_pass_bt_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(out_path, 'w') as f:
        json.dump({
            'date': datetime.now(timezone.utc).isoformat(),
            'locked': LOCKED,
            'aggregate': {
                'span_days': span_days, 'n_trades': len(df_t),
                'cum_pct': cum_pct, 'apy': apy,
                'avg_per_trade_net': float(df_t['net_pnl_pct'].mean()),
                'avg_per_trade_gross': float(df_t['gross_pct'].mean()),
                'wr': float((df_t['net_pnl_pct'] > 0).mean()),
            },
            'bootstrap': {
                'mean_daily': res.mean_daily_pct,
                'p5_daily': res.p5_daily_pct,
                'pos_rate': res.pos_rate,
                'overall_pass': res.overall_pass,
                'pass_criteria': res.pass_criteria,
            },
            'regime_distribution': df['regime'].value_counts().to_dict(),
        }, f, indent=2, default=str)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
