"""
Automatic Script Categorization

72개 scripts를 체계적으로 분류:
- production/: 최종 production-ready 스크립트
- analysis/: 분석 및 리포트 스크립트
- experiments/: 실험적 스크립트
- data/: 데이터 수집 및 처리
- deprecated/: 더 이상 사용하지 않는 스크립트
- utils/: 유틸리티 및 디버그
"""

import os
from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Categorization rules
CATEGORIES = {
    'production': [
        'train_xgboost_improved_v3_phase2.py',  # Latest training (Phase 2)
        'backtest_hybrid_v4.py',                 # Latest hybrid backtest
        'backtest_regime_specific_v5.py',        # Regime-specific
        'technical_strategy.py',                 # Production strategy
        'optimize_hybrid_thresholds.py',         # Threshold optimization
        'test_ultraconservative.py',             # Ultra-conservative test
    ],

    'analysis': [
        'analyze_technical_debt.py',
        'analyze_all_configs.py',
        'analyze_feature_structure.py',
        'analyze_v5_behavior.py',
        'critical_analysis_ultra5.py',
        'critical_reanalysis_with_risk_metrics.py',
        'critical_regime_comparison.py',
        'deep_analysis.py',
        'market_regime_analysis.py',
        'categorize_scripts.py',  # This file
    ],

    'experiments': [
        # Training experiments
        'train.py',
        'train_v2.py',
        'train_v3.py',
        'train_final.py',
        'train_extended.py',
        'train_conservative.py',
        'train_kfold.py',
        'train_v5_quick.py',
        'train_v6_quick.py',
        'continue_train.py',

        # XGBoost experiments
        'train_xgboost.py',
        'train_xgboost_fixed.py',
        'train_xgboost_improved.py',
        'train_xgboost_improved_v2.py',
        'train_xgboost_regression.py',
        'train_xgboost_with_sequential.py',
        'train_xgboost_with_smote.py',
        'train_simple_xgboost_for_paper_trading.py',

        # LSTM experiments
        'train_lstm_timeseries.py',
        'test_lstm_thresholds.py',
        'verify_lstm_stability.py',
        'optimize_sequence_length.py',

        # Backtest experiments
        'backtest.py',
        'backtest_smote_model.py',
        'backtest_with_stop_loss_take_profit.py',
        'backtest_xgboost_v2.py',
        'backtest_xgboost_v3_phase2.py',
        'regime_filtered_backtest.py',
        'extended_test_with_dynamic_sl_tp.py',
        'walk_forward_validation.py',
        'xgboost_rolling_window_validation.py',

        # Optimization experiments
        'optimize_win_rate.py',
        'threshold_sweep_regression.py',
        'threshold_sweep_sequential.py',
        'timeframe_comparison_1h.py',

        # Comparison experiments
        'compare_models.py',
        'fair_comparison_lstm_xgboost.py',
        'test_ensemble_voting.py',
        'verify_xgboost_stability.py',
        'simple_baselines.py',

        # Strategy experiments
        'hybrid_strategy_manager.py',
        'regime_specific_strategy.py',
        'paper_trading_bot.py',
        'test_paper_trading_bot.py',
        'live_trade.py',

        # Evaluation
        'quick_eval.py',
        'eval_kfold.py',
    ],

    'data': [
        'collect_data.py',
        'collect_max_data.py',
        'collect_more_data.py',
        'collect_public_data.py',
        'create_15min_data.py',
    ],

    'utils': [
        'debug_api.py',
        'debug_backtest_low_trades.py',
        'debug_environment.py',
        'debug_regression_predictions.py',
        'test_connection.py',
    ],

    'deprecated': [
        # Old versions (will be identified programmatically)
    ]
}


def create_directories():
    """Create category subdirectories"""
    print("Creating category directories...\n")

    for category in CATEGORIES.keys():
        target_dir = SCRIPTS_DIR / category
        if not target_dir.exists():
            target_dir.mkdir(parents=True)
            print(f"✅ Created: {category}/")
        else:
            print(f"ℹ️  Exists: {category}/")


def categorize_files(dry_run=True):
    """Categorize and move files"""
    print(f"\n{'DRY RUN' if dry_run else 'EXECUTING'} - File Categorization")
    print("=" * 80)

    all_scripts = list(SCRIPTS_DIR.glob('*.py'))
    categorized_files = set()

    # Collect all categorized files
    for files in CATEGORIES.values():
        categorized_files.update(files)

    # Move files to categories
    moves = []

    for category, files in CATEGORIES.items():
        for filename in files:
            source = SCRIPTS_DIR / filename
            if source.exists() and source.is_file():
                target_dir = SCRIPTS_DIR / category
                target = target_dir / filename
                moves.append((source, target, category))

    # Find uncategorized files
    uncategorized = []
    for script in all_scripts:
        if script.name not in categorized_files:
            # Check if it's already in a subdirectory
            if script.parent == SCRIPTS_DIR:
                uncategorized.append(script)

    # Print categorization plan
    print(f"\n📊 Categorization Summary:")
    print(f"  Total scripts: {len(all_scripts)}")
    print(f"  To be categorized: {len(moves)}")
    print(f"  Uncategorized: {len(uncategorized)}")

    print(f"\n📁 Files by Category:")
    category_counts = {}
    for source, target, category in moves:
        category_counts[category] = category_counts.get(category, 0) + 1

    for category, count in sorted(category_counts.items()):
        print(f"  {category}/: {count} files")

    if uncategorized:
        print(f"\n⚠️  Uncategorized files ({len(uncategorized)}):")
        for script in uncategorized[:10]:  # Show first 10
            print(f"    - {script.name}")
        if len(uncategorized) > 10:
            print(f"    ... and {len(uncategorized) - 10} more")

    # Execute moves if not dry run
    if not dry_run:
        print(f"\n🚀 Executing file moves...")
        for source, target, category in moves:
            try:
                if not target.exists():
                    shutil.move(str(source), str(target))
                    print(f"  ✅ Moved: {source.name} → {category}/")
                else:
                    print(f"  ⚠️  Skip (exists): {source.name}")
            except Exception as e:
                print(f"  ❌ Error moving {source.name}: {e}")

        print(f"\n✅ File reorganization complete!")
    else:
        print(f"\n💡 This was a DRY RUN. Re-run with dry_run=False to execute moves.")


def create_readmes():
    """Create README files for each category"""
    print("\nCreating README files...\n")

    readmes = {
        'production': """# Production Scripts

**Status**: Production-ready, actively maintained

These scripts are the final, production-ready versions used for:
- Training models (Phase 2+)
- Backtesting strategies
- Strategy optimization

## Key Scripts

- `train_xgboost_improved_v3_phase2.py`: Latest XGBoost training (Phase 2)
- `backtest_hybrid_v4.py`: Hybrid strategy backtesting
- `technical_strategy.py`: Technical indicators strategy module
- `optimize_hybrid_thresholds.py`: Threshold optimization

## Usage

See individual script docstrings for detailed usage.
""",

        'analysis': """# Analysis Scripts

Scripts for analyzing results, technical debt, and system behavior.

## Key Scripts

- `analyze_technical_debt.py`: Project technical debt analysis
- `critical_analysis_ultra5.py`: Critical analysis of Ultra-5 results
- `analyze_all_configs.py`: Compare all threshold configurations
- `market_regime_analysis.py`: Market regime analysis

## Usage

These scripts generate analysis reports and insights.
""",

        'experiments': """# Experimental Scripts

**Status**: Experimental, archived

Historical experiments and explorations:
- Various training approaches
- Backtesting experiments
- Strategy variations
- Model comparisons

## Note

These scripts are kept for reference but are not actively maintained.
Most experiments have been superseded by production scripts.
""",

        'data': """# Data Collection Scripts

Scripts for collecting and processing market data.

## Scripts

- `collect_max_data.py`: Collect maximum available historical data
- `create_15min_data.py`: Create 15-minute timeframe data

## Usage

Run these scripts to update or collect market data.
""",

        'utils': """# Utility Scripts

Helper scripts for debugging and testing.

## Scripts

- `test_connection.py`: Test API connection
- `debug_*.py`: Various debugging tools

## Usage

These are utility scripts for development and debugging.
""",

        'deprecated': """# Deprecated Scripts

**Status**: Deprecated, kept for reference only

These scripts are no longer used but are kept for historical reference.
Do not use these in new development.

If you need functionality from these scripts, check the `production/` directory first.
"""
    }

    for category, content in readmes.items():
        readme_path = SCRIPTS_DIR / category / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Created: {category}/README.md")


def main(execute=False):
    """Main reorganization function"""
    print("=" * 80)
    print("SCRIPTS DIRECTORY REORGANIZATION")
    print("=" * 80)
    print(f"Target: {SCRIPTS_DIR}")
    print(f"Mode: {'EXECUTE' if execute else 'DRY RUN'}")
    print()

    # Step 1: Create directories
    create_directories()

    # Step 2: Categorize and move files
    categorize_files(dry_run=not execute)

    # Step 3: Create README files
    if execute:
        create_readmes()

    print("\n" + "=" * 80)
    print(f"Reorganization {'COMPLETE' if execute else 'PLAN READY'}!")
    print("=" * 80)

    if not execute:
        print("\n💡 To execute, run: python categorize_scripts.py --execute")


if __name__ == '__main__':
    import sys
    execute = '--execute' in sys.argv
    main(execute=execute)
