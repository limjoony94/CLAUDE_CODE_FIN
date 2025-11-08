"""
Analyze recent entry signals since bot restart
"""
import re
from datetime import datetime

# Read log file
log_file = 'logs/opportunity_gating_bot_4x_20251022.log'

with open(log_file, 'r') as f:
    lines = f.readlines()

# Extract signals
signals = []
pattern = r'\[([\d-]+ [\d:]+)\] Price: \$([0-9,]+\.[0-9]) \| Balance: \$([0-9,]+\.[0-9]+) \| LONG: ([0-9.]+) \| SHORT: ([0-9.]+)'

for line in lines:
    match = re.search(pattern, line)
    if match:
        timestamp = match.group(1)
        price = float(match.group(2).replace(',', ''))
        balance = float(match.group(3).replace(',', ''))
        long_prob = float(match.group(4))
        short_prob = float(match.group(5))
        signals.append({
            'timestamp': timestamp,
            'price': price,
            'balance': balance,
            'long': long_prob,
            'short': short_prob
        })

print(f'Total signals: {len(signals)}')
print(f'Start time: {signals[0]["timestamp"]}')
print(f'End time: {signals[-1]["timestamp"]}')
print(f'Duration: {len(signals) * 5} minutes ({len(signals) * 5 / 60:.1f} hours)')
print()

# Price movement
start_price = signals[0]['price']
end_price = signals[-1]['price']
price_change = (end_price - start_price) / start_price * 100
print(f'Price: ${start_price:,.1f} -> ${end_price:,.1f} ({price_change:+.2f}%)')
print()

# Signal analysis
long_probs = [s['long'] for s in signals]
short_probs = [s['short'] for s in signals]

print('LONG Signal Analysis:')
print(f'  Max: {max(long_probs):.4f} (threshold: 0.65)')
print(f'  Min: {min(long_probs):.4f}')
print(f'  Avg: {sum(long_probs)/len(long_probs):.4f}')
print(f'  Above 0.50: {sum(1 for p in long_probs if p >= 0.50)} ({sum(1 for p in long_probs if p >= 0.50)/len(long_probs)*100:.1f}%)')
print(f'  Above 0.60: {sum(1 for p in long_probs if p >= 0.60)} ({sum(1 for p in long_probs if p >= 0.60)/len(long_probs)*100:.1f}%)')
print(f'  Above 0.65 (entry): {sum(1 for p in long_probs if p >= 0.65)} (0.0%)')
print()

print('SHORT Signal Analysis:')
print(f'  Max: {max(short_probs):.4f} (threshold: 0.70)')
print(f'  Min: {min(short_probs):.4f}')
print(f'  Avg: {sum(short_probs)/len(short_probs):.4f}')
print(f'  Above 0.50: {sum(1 for p in short_probs if p >= 0.50)} ({sum(1 for p in short_probs if p >= 0.50)/len(short_probs)*100:.1f}%)')
print(f'  Above 0.60: {sum(1 for p in short_probs if p >= 0.60)} ({sum(1 for p in short_probs if p >= 0.60)/len(short_probs)*100:.1f}%)')
print(f'  Above 0.70 (entry): {sum(1 for p in short_probs if p >= 0.70)} (0.0%)')
print()

# Top 5 LONG signals
print('Top 5 LONG Signals:')
long_signals = [(i, s) for i, s in enumerate(signals)]
long_signals.sort(key=lambda x: x[1]['long'], reverse=True)
for i, (idx, s) in enumerate(long_signals[:5], 1):
    print(f'  {i}. {s["timestamp"]} | Price: ${s["price"]:,.1f} | LONG: {s["long"]:.4f} | SHORT: {s["short"]:.4f}')
print()

# Top 5 SHORT signals
print('Top 5 SHORT Signals:')
short_signals = [(i, s) for i, s in enumerate(signals)]
short_signals.sort(key=lambda x: x[1]['short'], reverse=True)
for i, (idx, s) in enumerate(short_signals[:5], 1):
    print(f'  {i}. {s["timestamp"]} | Price: ${s["price"]:,.1f} | LONG: {s["long"]:.4f} | SHORT: {s["short"]:.4f}')
