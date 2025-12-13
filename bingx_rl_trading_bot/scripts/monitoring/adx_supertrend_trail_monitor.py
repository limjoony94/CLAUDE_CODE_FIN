#!/usr/bin/env python3
"""
ADX Trend + Supertrend Trail Bot Monitor
Real-time monitoring dashboard
"""

import os
import sys
import json
import time
from datetime import datetime
import yaml

import ccxt
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATE_FILE = os.path.join(PROJECT_ROOT, 'results', 'adx_supertrend_trail_bot_state.json')
CONFIG_FILE = os.path.join(PROJECT_ROOT, 'config', 'adx_supertrend_trail_config.yaml')

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}

def load_api_keys():
    api_keys_file = os.path.join(PROJECT_ROOT, 'config', 'api_keys.yaml')
    if os.path.exists(api_keys_file):
        with open(api_keys_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            return config.get('bingx', {}).get('mainnet', {})
    return {}

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def calculate_indicators(df):
    """Calculate ADX, +DI, -DI, and Supertrend"""
    df = df.copy()

    tr = pd.DataFrame({
        'hl': df['high'] - df['low'],
        'hc': abs(df['high'] - df['close'].shift(1)),
        'lc': abs(df['low'] - df['close'].shift(1))
    }).max(axis=1)

    df['atr'] = tr.ewm(span=14, adjust=False).mean()

    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    atr_14 = tr.ewm(span=14, adjust=False).mean()
    df['plus_di'] = 100 * (plus_dm.ewm(span=14, adjust=False).mean() / atr_14)
    df['minus_di'] = 100 * (minus_dm.ewm(span=14, adjust=False).mean() / atr_14)

    dx = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'] + 1e-10)
    df['adx'] = dx.ewm(span=14, adjust=False).mean()

    hl2 = (df['high'] + df['low']) / 2
    mult = 3.0
    upper = hl2 + mult * df['atr']
    lower = hl2 - mult * df['atr']

    supertrend = np.zeros(len(df))
    direction = np.zeros(len(df))

    for i in range(1, len(df)):
        if df['close'].iloc[i] > upper.iloc[i-1]:
            direction[i] = 1
        elif df['close'].iloc[i] < lower.iloc[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]

        if direction[i] == 1:
            supertrend[i] = max(lower.iloc[i], supertrend[i-1] if direction[i-1] == 1 else lower.iloc[i])
        else:
            supertrend[i] = min(upper.iloc[i], supertrend[i-1] if direction[i-1] == -1 else upper.iloc[i])

    df['supertrend'] = supertrend
    df['st_direction'] = direction

    return df

def main():
    config = load_config()
    api_config = load_api_keys()

    exchange = ccxt.bingx({
        'apiKey': api_config.get('api_key', ''),
        'secret': api_config.get('secret_key', ''),
        'enableRateLimit': True,
        'options': {'defaultType': 'swap'}
    })

    symbol = config.get('trading', {}).get('symbol', 'BTC-USDT')
    timeframe = config.get('trading', {}).get('timeframe', '15m')
    adx_threshold = config.get('strategy', {}).get('adx_threshold', 20)
    tp_pct = config.get('exit', {}).get('take_profit_pct', 2.0)
    leverage = config.get('leverage', {}).get('effective_leverage', 4)

    while True:
        try:
            clear_screen()

            print("=" * 70)
            print("  ADX TREND + SUPERTREND TRAIL BOT MONITOR")
            print("=" * 70)
            print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 70)

            # Fetch data
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = calculate_indicators(df)

            ticker = exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            price_24h = ticker.get('percentage', 0)

            last = df.iloc[-1]
            adx = last['adx']
            plus_di = last['plus_di']
            minus_di = last['minus_di']
            supertrend = last['supertrend']
            st_dist = abs(current_price - supertrend) / current_price * 100

            # Market section
            print("\n[MARKET]")
            print(f"  Price:      ${current_price:,.1f} ({price_24h:+.2f}% 24h)")
            print(f"  ADX:        {adx:.1f} {'(Strong Trend)' if adx > 25 else '(Weak Trend)' if adx < 20 else '(Moderate)'}")
            print(f"  +DI:        {plus_di:.1f}")
            print(f"  -DI:        {minus_di:.1f}")
            print(f"  DI Signal:  {'BULLISH (+DI > -DI)' if plus_di > minus_di else 'BEARISH (-DI > +DI)'}")
            print(f"  Supertrend: ${supertrend:,.1f} ({st_dist:.2f}% from price)")

            # Signal status
            print("\n[SIGNAL STATUS]")
            signal_ready = adx > adx_threshold
            print(f"  ADX > {adx_threshold}: {'YES' if signal_ready else 'NO'}")

            if signal_ready:
                if plus_di > minus_di:
                    print(f"  Next Signal: LONG (if +DI crosses above -DI)")
                else:
                    print(f"  Next Signal: SHORT (if -DI crosses above +DI)")
            else:
                print(f"  Next Signal: NONE (ADX too low)")

            # Load state
            state = load_state()

            # Position section
            print("\n[POSITION]")
            position = state.get('position')
            if position:
                direction = position.get('direction', 'UNKNOWN')
                entry_price = position.get('entry_price', 0)
                tp_price = position.get('tp_price', 0)
                sl_price = position.get('sl_price', 0)
                initial_sl = position.get('initial_sl_price', sl_price)
                entry_time = position.get('entry_time', '')

                if direction == 'LONG':
                    pnl_pct = (current_price - entry_price) / entry_price * 100
                else:
                    pnl_pct = (entry_price - current_price) / entry_price * 100

                pnl_with_leverage = pnl_pct * leverage

                print(f"  Direction:  {direction}")
                print(f"  Entry:      ${entry_price:,.1f}")
                print(f"  Current:    ${current_price:,.1f}")
                print(f"  PnL:        {pnl_pct:+.2f}% (x{leverage} = {pnl_with_leverage:+.2f}%)")
                print(f"  TP:         ${tp_price:,.1f} ({tp_pct}% fixed)")
                print(f"  SL:         ${sl_price:,.1f} (Supertrend Trail)")
                print(f"  Initial SL: ${initial_sl:,.1f}")
                print(f"  Entry Time: {entry_time}")

                # Progress bars
                if direction == 'LONG':
                    tp_progress = max(0, min(100, (current_price - entry_price) / (tp_price - entry_price) * 100))
                    sl_progress = max(0, min(100, (entry_price - current_price) / (entry_price - sl_price) * 100)) if current_price < entry_price else 0
                else:
                    tp_progress = max(0, min(100, (entry_price - current_price) / (entry_price - tp_price) * 100))
                    sl_progress = max(0, min(100, (current_price - entry_price) / (sl_price - entry_price) * 100)) if current_price > entry_price else 0

                tp_bar = '█' * int(tp_progress / 5) + '░' * (20 - int(tp_progress / 5))
                sl_bar = '█' * int(sl_progress / 5) + '░' * (20 - int(sl_progress / 5))

                print(f"\n  TP Progress: [{tp_bar}] {tp_progress:.1f}%")
                print(f"  SL Progress: [{sl_bar}] {sl_progress:.1f}%")
            else:
                print("  No open position")

            # Statistics section
            print("\n[STATISTICS]")
            total_trades = state.get('total_trades', 0)
            winning_trades = state.get('winning_trades', 0)
            total_pnl = state.get('total_pnl', 0)
            win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0

            print(f"  Total Trades:   {total_trades}")
            print(f"  Win Rate:       {win_rate:.1f}%")
            print(f"  Cumulative PnL: {total_pnl:+.2f}%")

            # Account section
            print("\n[ACCOUNT]")
            try:
                balance = exchange.fetch_balance()
                usdt = balance.get('USDT', {})
                total = usdt.get('total', 0)
                free = usdt.get('free', 0)
                used = usdt.get('used', 0)

                initial = state.get('initial_balance', total)
                account_pnl = (total - initial) / initial * 100 if initial > 0 else 0

                print(f"  Balance:      ${total:,.2f}")
                print(f"  Available:    ${free:,.2f}")
                print(f"  In Position:  ${used:,.2f}")
                print(f"  Account PnL:  {account_pnl:+.2f}%")
            except Exception as e:
                print(f"  Error fetching balance: {e}")

            # Config summary
            print("\n[CONFIG]")
            print(f"  Symbol:     {symbol}")
            print(f"  Timeframe:  {timeframe}")
            print(f"  ADX Thres:  {adx_threshold}")
            print(f"  TP:         {tp_pct}% (fixed)")
            print(f"  SL:         Supertrend Trail")
            print(f"  Leverage:   {leverage}x")

            print("\n" + "=" * 70)
            print("  Press Ctrl+C to exit")
            print("=" * 70)

            time.sleep(15)

        except KeyboardInterrupt:
            print("\nMonitor stopped")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()
