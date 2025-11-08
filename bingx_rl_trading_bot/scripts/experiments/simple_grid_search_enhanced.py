"""
Simple Grid Search for Enhanced Models
=======================================

Systematically test threshold combinations by modifying and running
temp_backtest_threshold_080.py multiple times.

Grid:
- Entry: [0.60, 0.65, 0.70, 0.75, 0.80]
- Exit:  [0.70, 0.75, 0.80, 0.85]
"""

import subprocess
import pandas as pd
from pathlib import Path
from datetime import datetime
import re
import time

PROJECT_ROOT = Path(__file__).parent.parent.parent
TEMP_BACKTEST = PROJECT_ROOT / "scripts" / "experiments" / "temp_backtest_threshold_080.py"

# Grid parameters
ENTRY_THRESHOLDS = [0.60, 0.65, 0.70, 0.75, 0.80]
EXIT_THRESHOLDS = [0.70, 0.75, 0.80, 0.85]

print("="*80)
print("SIMPLE GRID SEARCH - Enhanced Models")
print("="*80)
print(f"\nGrid:")
print(f"  Entry: {ENTRY_THRESHOLDS}")
print(f"  Exit:  {EXIT_THRESHOLDS}")
print(f"  Total: {len(ENTRY_THRESHOLDS) * len(EXIT_THRESHOLDS)} combinations")
print("="*80)

results = []

for entry_threshold in ENTRY_THRESHOLDS:
    for exit_threshold in EXIT_THRESHOLDS:
        print(f"\n[{len(results)+1}/{len(ENTRY_THRESHOLDS)*len(EXIT_THRESHOLDS)}] Testing Entry={entry_threshold:.2f}, Exit={exit_threshold:.2f}")

        # Read current file
        with open(TEMP_BACKTEST, 'r', encoding='utf-8') as f:
            content = f.read()

        # Modify thresholds
        content_modified = re.sub(
            r'LONG_THRESHOLD = [\d.]+',
            f'LONG_THRESHOLD = {entry_threshold}',
            content
        )
        content_modified = re.sub(
            r'SHORT_THRESHOLD = [\d.]+',
            f'SHORT_THRESHOLD = {entry_threshold}',
            content_modified
        )
        content_modified = re.sub(
            r'ML_EXIT_THRESHOLD_LONG = [\d.]+',
            f'ML_EXIT_THRESHOLD_LONG = {exit_threshold}',
            content_modified
        )
        content_modified = re.sub(
            r'ML_EXIT_THRESHOLD_SHORT = [\d.]+',
            f'ML_EXIT_THRESHOLD_SHORT = {exit_threshold}',
            content_modified
        )

        # Save temporary file
        temp_file = PROJECT_ROOT / "scripts" / "experiments" / "_temp_grid_search.py"
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(content_modified)

        # Run backtest
        try:
            start_time = time.time()
            result = subprocess.run(
                ['python', str(temp_file)],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes max per combination
                cwd=str(PROJECT_ROOT)
            )
            elapsed = time.time() - start_time

            # Parse output
            output = result.stdout

            # Extract key metrics
            mean_return = None
            sharpe = None
            win_rate = None
            total_trades = None

            for line in output.split('\n'):
                if 'Mean Return:' in line:
                    match = re.search(r'Mean Return:\s*([-+]?[\d.]+)%', line)
                    if match:
                        mean_return = float(match.group(1))

                elif 'Sharpe Ratio (annualized):' in line:
                    match = re.search(r'Sharpe Ratio \(annualized\):\s*([\d.]+)', line)
                    if match:
                        sharpe = float(match.group(1))

                elif 'Win Rate:' in line and 'Overall:' in line:
                    match = re.search(r'Win Rate:.*?([\d.]+)%', line)
                    if match:
                        win_rate = float(match.group(1))

                elif 'Total Trades:' in line:
                    match = re.search(r'Total Trades:\s*([\d,]+)', line)
                    if match:
                        total_trades = int(match.group(1).replace(',', ''))

            if mean_return is not None:
                results.append({
                    'entry_threshold': entry_threshold,
                    'exit_threshold': exit_threshold,
                    'mean_return': mean_return,
                    'sharpe_ratio': sharpe if sharpe else 0,
                    'win_rate': win_rate if win_rate else 0,
                    'total_trades': total_trades if total_trades else 0,
                    'elapsed_time': elapsed
                })

                print(f"  → Return={mean_return:+.2f}%, Sharpe={sharpe:.3f}, WR={win_rate:.1f}%, Trades={total_trades} ({elapsed:.1f}s)")
            else:
                print(f"  → Failed to parse results")
                print(f"  Output: {output[:500]}")

        except subprocess.TimeoutExpired:
            print(f"  → Timeout (>5min)")
        except Exception as e:
            print(f"  → Error: {e}")

        # Clean up temp file
        if temp_file.exists():
            temp_file.unlink()

# Save results
if results:
    results_df = pd.DataFrame(results)
    results_df['composite_score'] = (
        results_df['mean_return'] *
        results_df['win_rate'] *
        results_df['sharpe_ratio']
    ) / 10000  # Normalization

    results_df = results_df.sort_values('composite_score', ascending=False)

    output_file = PROJECT_ROOT / "results" / f"simple_grid_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(output_file, index=False)

    print("\n" + "="*80)
    print("GRID SEARCH COMPLETE")
    print("="*80)

    print("\nTop 5 Configurations:")
    for idx, row in results_df.head(5).iterrows():
        rank = list(results_df.index).index(idx) + 1
        print(f"\nRank {rank}: Entry {row['entry_threshold']:.2f} / Exit {row['exit_threshold']:.2f}")
        print(f"  Return: {row['mean_return']:+.2f}%")
        print(f"  Sharpe: {row['sharpe_ratio']:.3f}")
        print(f"  Win Rate: {row['win_rate']:.1f}%")
        print(f"  Trades: {row['total_trades']}")
        print(f"  Composite: {row['composite_score']:.2f}")

    print(f"\n✅ Results saved: {output_file}")
else:
    print("\n⚠️  No results collected")

print("\n" + "="*80)
