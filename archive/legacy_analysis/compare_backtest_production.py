"""Compare ADX Supertrend Trail backtest logic vs production logic
Uses HIGH/LOW for TP/SL hit detection (same as original research script)
"""
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('data/btc_15m_full_2025.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

def add_indicators(df, period=14, multiplier=3.0):
    """Add all indicators matching original research script"""
    high, low, close = df['high'], df['low'], df['close']

    # ADX + DI
    plus_dm = high.diff()
    minus_dm = low.diff().abs() * -1
    plus_dm = np.where((plus_dm > minus_dm.abs()) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm.abs() > plus_dm) & (minus_dm < 0), minus_dm.abs(), 0)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.rolling(period).mean()
    df['adx'], df['plus_di'], df['minus_di'], df['atr'] = adx, plus_di, minus_di, atr

    # Supertrend
    hl2 = (high + low) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr
    st = pd.Series(index=df.index, dtype=float)
    st_dir = pd.Series(index=df.index, dtype=int)
    for i in range(period, len(df)):
        if close.iloc[i] > upper.iloc[i-1]:
            st_dir.iloc[i] = 1
            st.iloc[i] = lower.iloc[i]
        elif close.iloc[i] < lower.iloc[i-1]:
            st_dir.iloc[i] = -1
            st.iloc[i] = upper.iloc[i]
        else:
            st_dir.iloc[i] = st_dir.iloc[i-1] if i > period else 1
            if st_dir.iloc[i] == 1:
                st.iloc[i] = max(lower.iloc[i], st.iloc[i-1]) if not pd.isna(st.iloc[i-1]) else lower.iloc[i]
            else:
                st.iloc[i] = min(upper.iloc[i], st.iloc[i-1]) if not pd.isna(st.iloc[i-1]) else upper.iloc[i]
    df['supertrend'], df['st_direction'] = st, st_dir
    df['st_upper'], df['st_lower'] = upper, lower
    return df

def generate_signals(df, adx_threshold=20):
    """Generate ADX trend signals - EXACTLY matching research script"""
    signals = np.zeros(len(df))
    adx = df['adx'].values
    plus_di = df['plus_di'].values
    minus_di = df['minus_di'].values

    for i in range(1, len(df)):
        if adx[i] > adx_threshold:
            if plus_di[i-1] < minus_di[i-1] and plus_di[i] > minus_di[i]:
                signals[i] = 1  # LONG
            elif plus_di[i-1] > minus_di[i-1] and plus_di[i] < minus_di[i]:
                signals[i] = -1  # SHORT
    return signals

def backtest(df, signals, validate_sl=False, tp_pct=2.0, cooldown=6, leverage=4, fee_pct=0.05):
    """Backtest with optional SL validation (production-style)"""
    trades = []
    position = None
    last_trade_idx = -cooldown - 1

    for i in range(200, len(df)):
        if i - last_trade_idx < cooldown:
            continue

        if position is None and signals[i] != 0:
            direction = signals[i]
            entry_price = df['close'].iloc[i]

            if validate_sl:
                # Production-style: use correct bands with validation
                if direction == 1:  # LONG
                    sl_price = df['st_lower'].iloc[i]
                    if sl_price >= entry_price:
                        continue  # Invalid - skip
                else:  # SHORT
                    sl_price = df['st_upper'].iloc[i]
                    if sl_price <= entry_price:
                        continue  # Invalid - skip
            else:
                # Original backtest: use generic supertrend (BUG)
                sl_price = df['supertrend'].iloc[i]

            sl_distance = abs(entry_price - sl_price) / entry_price * 100
            if sl_distance > 5.0 or sl_distance < 0.3:
                continue

            position = {
                'direction': direction,
                'entry_price': entry_price,
                'entry_idx': i,
                'tp_price': entry_price * (1 + direction * tp_pct / 100),
                'sl_price': sl_price,
                'sl_distance': sl_distance
            }
            continue

        if position is not None:
            high = df['high'].iloc[i]
            low = df['low'].iloc[i]

            # Update trailing SL
            if validate_sl:
                # Production-style: use correct bands with price check
                if position['direction'] == 1:
                    new_sl = df['st_lower'].iloc[i]
                    if new_sl > position['sl_price'] and new_sl < df['close'].iloc[i]:
                        position['sl_price'] = new_sl
                else:
                    new_sl = df['st_upper'].iloc[i]
                    if new_sl < position['sl_price'] and new_sl > df['close'].iloc[i]:
                        position['sl_price'] = new_sl
            else:
                # Original backtest: use generic supertrend
                current_st = df['supertrend'].iloc[i]
                if position['direction'] == 1:
                    position['sl_price'] = max(position['sl_price'], current_st)
                else:
                    position['sl_price'] = min(position['sl_price'], current_st)

            # Check TP/SL using HIGH/LOW (intrabar)
            hit_tp = (position['direction'] == 1 and high >= position['tp_price']) or \
                     (position['direction'] == -1 and low <= position['tp_price'])
            hit_sl = (position['direction'] == 1 and low <= position['sl_price']) or \
                     (position['direction'] == -1 and high >= position['sl_price'])

            if not hit_tp and not hit_sl:
                continue

            # Determine exit
            if hit_tp and hit_sl:
                exit_price = position['sl_price']  # SL takes priority when both hit
                exit_type = 'SL'
            elif hit_tp:
                exit_price = position['tp_price']
                exit_type = 'TP'
            else:
                exit_price = position['sl_price']
                exit_type = 'SL'

            pnl_pct = position['direction'] * (exit_price - position['entry_price']) / position['entry_price'] * 100 * leverage
            pnl_pct -= 2 * fee_pct * leverage  # Entry + exit fees

            trades.append({
                'direction': 'LONG' if position['direction'] == 1 else 'SHORT',
                'entry': position['entry_price'],
                'exit': exit_price,
                'pnl': pnl_pct,
                'sl_distance': position['sl_distance'],
                'exit_type': exit_type
            })

            position = None
            last_trade_idx = i

    return trades

def calc_metrics(trades):
    """Calculate performance metrics"""
    if not trades:
        return {'trades': 0, 'return': 0, 'win_rate': 0, 'mdd': 0, 'long_trades': 0, 'short_trades': 0}

    balance = 100
    peak = 100
    mdd = 0

    for t in trades:
        balance *= (1 + t['pnl'] / 100)
        peak = max(peak, balance)
        dd = (peak - balance) / peak * 100
        mdd = max(mdd, dd)

    wins = sum(1 for t in trades if t['pnl'] > 0)
    return {
        'trades': len(trades),
        'return': balance - 100,
        'win_rate': 100 * wins / len(trades),
        'mdd': mdd,
        'long_trades': sum(1 for t in trades if t['direction'] == 'LONG'),
        'short_trades': sum(1 for t in trades if t['direction'] == 'SHORT')
    }

# Main execution
print("Adding indicators...")
df = add_indicators(df)

print("Generating signals...")
signals = generate_signals(df)
print(f"Total signals: {np.sum(signals != 0)}")

print('\n' + '=' * 70)
print('CORRECTED BACKTEST COMPARISON')
print('Using HIGH/LOW for TP/SL detection (same as original research)')
print('=' * 70)

# Original (with SL bug)
print("\nRunning ORIGINAL backtest (with SL direction bug)...")
trades_orig = backtest(df, signals, validate_sl=False)
m_orig = calc_metrics(trades_orig)
print(f'\nORIGINAL (with SL bug):')
print(f'  Trades: {m_orig["trades"]} (LONG: {m_orig["long_trades"]}, SHORT: {m_orig["short_trades"]})')
print(f'  Return: {m_orig["return"]:+.1f}%')
print(f'  Win Rate: {m_orig["win_rate"]:.1f}%')
print(f'  Max DD: {m_orig["mdd"]:.1f}%')

# Production-style (SL validated)
print("\nRunning PRODUCTION-STYLE backtest (SL validated)...")
trades_prod = backtest(df, signals, validate_sl=True)
m_prod = calc_metrics(trades_prod)
print(f'\nPRODUCTION-STYLE (SL validated):')
print(f'  Trades: {m_prod["trades"]} (LONG: {m_prod["long_trades"]}, SHORT: {m_prod["short_trades"]})')
print(f'  Return: {m_prod["return"]:+.1f}%')
print(f'  Win Rate: {m_prod["win_rate"]:.1f}%')
print(f'  Max DD: {m_prod["mdd"]:.1f}%')

print(f'\n' + '=' * 70)
print('IMPACT OF BUG FIX')
print('=' * 70)
print(f'  Trades: {m_orig["trades"]} -> {m_prod["trades"]} ({m_prod["trades"] - m_orig["trades"]:+d})')
print(f'  Return: {m_orig["return"]:+.1f}% -> {m_prod["return"]:+.1f}% ({m_prod["return"] - m_orig["return"]:+.1f}%p)')
print(f'  Win Rate: {m_orig["win_rate"]:.1f}% -> {m_prod["win_rate"]:.1f}% ({m_prod["win_rate"] - m_orig["win_rate"]:+.1f}%p)')
print(f'  Max DD: {m_orig["mdd"]:.1f}% -> {m_prod["mdd"]:.1f}% ({m_prod["mdd"] - m_orig["mdd"]:+.1f}%p)')

# Analyze invalid trades
print(f'\n' + '=' * 70)
print('ANALYSIS: INVALID SL DIRECTION IN ORIGINAL')
print('=' * 70)

valid_long, invalid_long = 0, 0
valid_short, invalid_short = 0, 0

for i in range(200, len(df)):
    if signals[i] == 1:  # LONG
        entry = df['close'].iloc[i]
        sl = df['supertrend'].iloc[i]
        sl_dist = abs(entry - sl) / entry * 100
        if 0.3 <= sl_dist <= 5.0:
            if sl < entry:
                valid_long += 1
            else:
                invalid_long += 1
    elif signals[i] == -1:  # SHORT
        entry = df['close'].iloc[i]
        sl = df['supertrend'].iloc[i]
        sl_dist = abs(sl - entry) / entry * 100
        if 0.3 <= sl_dist <= 5.0:
            if sl > entry:
                valid_short += 1
            else:
                invalid_short += 1

total = valid_long + invalid_long + valid_short + invalid_short
invalid = invalid_long + invalid_short

print(f'LONG signals (pass filter): {valid_long + invalid_long}')
print(f'  Valid (SL below entry): {valid_long} ({100*valid_long/(valid_long+invalid_long):.1f}%)')
print(f'  Invalid (SL above entry): {invalid_long} ({100*invalid_long/(valid_long+invalid_long):.1f}%)')
print(f'\nSHORT signals (pass filter): {valid_short + invalid_short}')
print(f'  Valid (SL above entry): {valid_short} ({100*valid_short/(valid_short+invalid_short):.1f}%)')
print(f'  Invalid (SL below entry): {invalid_short} ({100*invalid_short/(valid_short+invalid_short):.1f}%)')
print(f'\nTotal potential trades: {total}')
print(f'  Invalid trades: {invalid} ({100*invalid/total:.1f}%)')

print(f'\n' + '=' * 70)
print('CONCLUSION')
print('=' * 70)
print("""
The original ADX Supertrend Trail backtest has a CRITICAL BUG:
- Uses generic 'supertrend' for SL instead of correct bands
- 42.4% of trades have invalid SL direction
- Invalid trades have SL on wrong side (essentially pre-profitable)

This inflates backtest performance. Production bot correctly:
- Uses st_lower for LONG SL
- Uses st_upper for SHORT SL
- Validates SL is on correct side before entry

RECOMMENDATION: The production bot logic is CORRECT.
Backtest results should NOT be trusted as-is.
""")
