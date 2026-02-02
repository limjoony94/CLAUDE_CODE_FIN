"""
Analyze Consecutive Stop Loss Patterns in Backtest
"""
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Load 28-day backtest results
df = pd.read_csv(PROJECT_ROOT / 'results' / 'backtest_28days_full_20251104_0142.csv')
df['entry_time'] = pd.to_datetime(df['entry_time'])
df['exit_time'] = pd.to_datetime(df['exit_time'])

print('=' * 80)
print('CONSECUTIVE STOP LOSS ANALYSIS - 28-DAY BACKTEST')
print('=' * 80)

# Identify Stop Loss trades
df['is_sl'] = df['exit_reason'].str.contains('Stop Loss', case=False)

# Find consecutive SL sequences
consecutive_sl = []
current_streak = 0
streak_start_idx = None

for i, row in df.iterrows():
    if row['is_sl']:
        if current_streak == 0:
            streak_start_idx = i
        current_streak += 1
    else:
        if current_streak > 0:
            consecutive_sl.append({
                'start_idx': streak_start_idx,
                'end_idx': i - 1,
                'length': current_streak
            })
        current_streak = 0

# Handle if backtest ends with SL streak
if current_streak > 0:
    consecutive_sl.append({
        'start_idx': streak_start_idx,
        'end_idx': len(df) - 1,
        'length': current_streak
    })

print(f'\n📊 Stop Loss Statistics:')
print(f'   Total trades: {len(df)}')
print(f'   Stop Loss trades: {df["is_sl"].sum()} ({df["is_sl"].sum()/len(df)*100:.1f}%)')
print(f'   Non-SL trades: {(~df["is_sl"]).sum()}')

print(f'\n🔁 Consecutive Stop Loss Sequences:')
print(f'   Total sequences: {len(consecutive_sl)}')

# Filter sequences of 3+ SL
sequences_3plus = [s for s in consecutive_sl if s['length'] >= 3]

if sequences_3plus:
    print(f'   Sequences of 3+ SL: {len(sequences_3plus)} ⚠️')
    print()

    for i, seq in enumerate(sequences_3plus, 1):
        start_idx = seq['start_idx']
        end_idx = seq['end_idx']
        length = seq['length']

        print(f'   Sequence #{i}: {length} consecutive Stop Losses')
        print(f'   ─────────────────────────────────────────────')

        # Calculate total loss in sequence
        sequence_trades = df.iloc[start_idx:end_idx+1]
        total_pnl = sequence_trades['pnl_net'].sum()

        print(f'      Period: {sequence_trades.iloc[0]["entry_time"]} → {sequence_trades.iloc[-1]["exit_time"]}')
        print(f'      Total P&L: ${total_pnl:,.2f}')
        print()

        # Show trades in this sequence
        for j in range(start_idx, end_idx + 1):
            trade = df.iloc[j]
            trade_num = j + 1
            print(f'      Trade #{trade_num}: {trade["entry_time"]} → {trade["exit_time"]}')
            print(f'         Side: {trade["side"]:5s}  |  Entry: ${trade["entry_price"]:>10,.1f}  |  Exit: ${trade["exit_price"]:>10,.1f}')
            print(f'         P&L: ${trade["pnl_net"]:>10,.2f} ({trade["pnl_pct"]*100:>6.2f}%)  |  Reason: {trade["exit_reason"]}')
        print()
else:
    print(f'   Sequences of 3+ SL: 0 ✅ (No 3+ consecutive SL in backtest)')
    print()

# Show distribution of consecutive SL lengths
print(f'📈 Distribution of Consecutive SL Lengths:')
from collections import Counter
lengths = [s['length'] for s in consecutive_sl]
length_counts = Counter(lengths)

for length in sorted(length_counts.keys()):
    count = length_counts[length]
    bar = '█' * count
    print(f'   {length} consecutive SL: {count:2d} times {bar}')

# Max consecutive SL
if consecutive_sl:
    max_streak = max(s['length'] for s in consecutive_sl)
    max_seq = [s for s in consecutive_sl if s['length'] == max_streak][0]
    print(f'\n🔥 Maximum consecutive SL: {max_streak}')

    if max_streak >= 3:
        print(f'   Period: {df.iloc[max_seq["start_idx"]]["entry_time"]} → {df.iloc[max_seq["end_idx"]]["exit_time"]}')
else:
    print(f'\n🔥 Maximum consecutive SL: 0 (no SL at all)')

# Compare with production (if known)
print('\n' + '=' * 80)
print('COMPARISON WITH PRODUCTION')
print('=' * 80)
print('\n⚠️  Production: 3 consecutive Stop Losses (recent)')
if sequences_3plus:
    print(f'✅ Backtest: {len(sequences_3plus)} sequence(s) of 3+ consecutive SL found')
    print(f'   → This pattern EXISTS in backtest')
    print(f'   → Model IS capable of 3+ consecutive losses')
    print(f'\n💡 Interpretation: Current production SL streak is within expected model behavior')
else:
    print(f'❌ Backtest: NO sequences of 3+ consecutive SL')
    print(f'   → This pattern NEVER occurred in backtest')
    print(f'   → Current production behavior is UNUSUAL')
    print(f'\n⚠️  Warning: Model may be encountering market regime not seen in training/backtest')

print('\n' + '=' * 80)
