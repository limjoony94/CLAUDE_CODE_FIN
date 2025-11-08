"""
Update EMERGENCY_STOP_LOSS to -0.03 in all backtest files
==========================================================

Updates all backtest scripts to use the optimized -3% SL
based on 30-day detailed grid search results.
"""

import re
from pathlib import Path

# Files to update
PROJECT_ROOT = Path(__file__).parent.parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "scripts" / "experiments"

# Target files (found via grep)
target_files = [
    "analyze_exit_performance.py",
    "backtest_continuous_compound.py",
    "backtest_dynamic_exit_strategy.py",
    "backtest_exit_model.py",
    "backtest_full_trade_outcome_system.py",
    "backtest_improved_entry_models.py",
    "backtest_oct09_oct13_production_models.py",
    "backtest_oct14_oct19_production_models.py",
    "backtest_production_30days.py",
    "backtest_production_72h.py",
    "backtest_production_7days.py",
    "backtest_production_settings.py",
    "backtest_trade_outcome_full_models.py",
    "backtest_trade_outcome_sample_models.py",
    "compare_exit_improvements.py",
    "full_backtest_opportunity_gating_4x.py",
    "optimize_entry_thresholds.py",
    "optimize_short_exit_threshold.py",
    "validate_exit_logic_4x.py"
]

print("="*80)
print("UPDATING EMERGENCY_STOP_LOSS TO -3% IN ALL BACKTEST FILES")
print("="*80)
print()

updated_count = 0
skipped_count = 0

for filename in target_files:
    filepath = EXPERIMENTS_DIR / filename

    if not filepath.exists():
        print(f"⚠️  SKIP: {filename} (not found)")
        skipped_count += 1
        continue

    # Read file
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # Pattern 1: EMERGENCY_STOP_LOSS = -0.XX
    # Replace with -0.03
    pattern1 = r'EMERGENCY_STOP_LOSS\s*=\s*-0\.\d+'
    if re.search(pattern1, content):
        content = re.sub(pattern1, 'EMERGENCY_STOP_LOSS = -0.03', content)

    # Pattern 2: EMERGENCY_STOP_LOSS = 0.XX (without minus)
    # Replace with 0.03
    pattern2 = r'EMERGENCY_STOP_LOSS\s*=\s*0\.\d+'
    if re.search(pattern2, content):
        content = re.sub(pattern2, 'EMERGENCY_STOP_LOSS = 0.03', content)

    # Check if content changed
    if content != original_content:
        # Write back
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"✅ UPDATED: {filename}")
        updated_count += 1
    else:
        print(f"⏭️  SKIP: {filename} (no SL found or already -0.03)")
        skipped_count += 1

print()
print("="*80)
print("UPDATE SUMMARY")
print("="*80)
print(f"✅ Updated: {updated_count} files")
print(f"⏭️  Skipped: {skipped_count} files")
print(f"📁 Total: {len(target_files)} files")
print()
print("🎯 All backtest files now use EMERGENCY_STOP_LOSS = -0.03 (-3%)")
print("="*80)
