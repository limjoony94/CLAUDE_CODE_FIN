"""Check if data exists at specific timestamp"""
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
df = pd.read_csv(PROJECT_ROOT / 'data' / 'historical' / 'BTCUSDT_5m_updated.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Target time
target = pd.Timestamp('2025-10-23 16:00:00')
start = target - pd.Timedelta(hours=1)
end = target + pd.Timedelta(hours=1)

df_range = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]

print(f'Target: {target}')
print(f'Range: {start} to {end}')
print(f'Found: {len(df_range)} candles')
print()

if len(df_range) > 0:
    print('Candles around target:')
    for idx, row in df_range.iterrows():
        print(f"  {row['timestamp']} - ${row['close']:,.2f}")
else:
    print('NO DATA in this range!')
    print()
    # Check Oct 23-24
    oct23 = df[(df['timestamp'] >= pd.Timestamp('2025-10-23')) &
               (df['timestamp'] < pd.Timestamp('2025-10-24'))]
    print(f'Oct 23 data: {len(oct23)} candles')
    if len(oct23) > 0:
        print(f'  First: {oct23.iloc[0]["timestamp"]}')
        print(f'  Last: {oct23.iloc[-1]["timestamp"]}')
