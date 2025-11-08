"""
Visualize recent entry signals
"""
import re
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
        timestamp = datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S')
        price = float(match.group(2).replace(',', ''))
        long_prob = float(match.group(4))
        short_prob = float(match.group(5))
        signals.append({
            'timestamp': timestamp,
            'price': price,
            'long': long_prob,
            'short': short_prob
        })

# Prepare data
timestamps = [s['timestamp'] for s in signals]
prices = [s['price'] for s in signals]
long_probs = [s['long'] for s in signals]
short_probs = [s['short'] for s in signals]

# Create figure with 3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# Plot 1: Price
ax1.plot(timestamps, prices, 'b-', linewidth=1.5, label='BTC Price')
ax1.set_ylabel('Price (USD)', fontsize=12, fontweight='bold')
ax1.set_title('BTC Price & Entry Signals Analysis (Bot Restart: 2025-10-22 01:20)', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right')
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Plot 2: LONG probability
ax2.plot(timestamps, long_probs, 'g-', linewidth=1.5, label='LONG Probability')
ax2.axhline(y=0.65, color='r', linestyle='--', linewidth=2, label='Entry Threshold (0.65)')
ax2.axhline(y=0.50, color='orange', linestyle=':', linewidth=1, label='50% Level')
ax2.fill_between(timestamps, 0, long_probs, alpha=0.3, color='green')
ax2.set_ylabel('LONG Probability', fontsize=12, fontweight='bold')
ax2.set_ylim(0, 0.75)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right')

# Plot 3: SHORT probability
ax3.plot(timestamps, short_probs, 'r-', linewidth=1.5, label='SHORT Probability')
ax3.axhline(y=0.70, color='r', linestyle='--', linewidth=2, label='Entry Threshold (0.70)')
ax3.axhline(y=0.50, color='orange', linestyle=':', linewidth=1, label='50% Level')
ax3.fill_between(timestamps, 0, short_probs, alpha=0.3, color='red')
ax3.set_ylabel('SHORT Probability', fontsize=12, fontweight='bold')
ax3.set_xlabel('Time', fontsize=12, fontweight='bold')
ax3.set_ylim(0, 0.75)
ax3.grid(True, alpha=0.3)
ax3.legend(loc='upper right')

# Format x-axis
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax3.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('results/signal_analysis_20251022.png', dpi=150, bbox_inches='tight')
print('Chart saved to: results/signal_analysis_20251022.png')
plt.close()

# Create correlation analysis
print('\nPearson Correlation Analysis:')
import numpy as np

price_changes = [prices[i+1] - prices[i] for i in range(len(prices)-1)]
long_changes = [long_probs[i+1] - long_probs[i] for i in range(len(long_probs)-1)]
short_changes = [short_probs[i+1] - short_probs[i] for i in range(len(short_probs)-1)]

corr_long_price = np.corrcoef(price_changes, long_changes)[0, 1]
corr_short_price = np.corrcoef(price_changes, short_changes)[0, 1]

print(f'  LONG signal vs Price change: {corr_long_price:.4f}')
print(f'  SHORT signal vs Price change: {corr_short_price:.4f}')
print()
print('Interpretation:')
if abs(corr_long_price) > 0.5:
    print(f'  - LONG signals have {"strong positive" if corr_long_price > 0 else "strong negative"} correlation with price')
elif abs(corr_long_price) > 0.3:
    print(f'  - LONG signals have {"moderate positive" if corr_long_price > 0 else "moderate negative"} correlation with price')
else:
    print(f'  - LONG signals have weak correlation with price (expected for ML model)')

if abs(corr_short_price) > 0.5:
    print(f'  - SHORT signals have {"strong positive" if corr_short_price > 0 else "strong negative"} correlation with price')
elif abs(corr_short_price) > 0.3:
    print(f'  - SHORT signals have {"moderate positive" if corr_short_price > 0 else "moderate negative"} correlation with price')
else:
    print(f'  - SHORT signals have weak correlation with price (expected for ML model)')
