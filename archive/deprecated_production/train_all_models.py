"""
Unified Training Script - Train All 4 Models

목적: 4개 모델을 일관된 데이터로 순차적으로 훈련
- LONG Entry Model (37 features)
- SHORT Entry Model (37 features)
- LONG Exit Model (44 features: 36 base + 8 position)
- SHORT Exit Model (44 features: 36 base + 8 position)

사용법:
    python scripts/production/train_all_models.py
    python scripts/production/train_all_models.py --download-data  # 데이터 다운로드 먼저 수행
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime
from pathlib import Path
from loguru import logger
import json

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SCRIPTS_DIR = PROJECT_ROOT / "scripts" / "production"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Training scripts
DOWNLOAD_SCRIPT = SCRIPTS_DIR / "download_historical_data.py"
LONG_ENTRY_SCRIPT = SCRIPTS_DIR / "train_xgboost_phase4_advanced.py"
SHORT_ENTRY_SCRIPT = SCRIPTS_DIR / "train_xgboost_short_model.py"
EXIT_MODELS_SCRIPT = PROJECT_ROOT / "scripts" / "experiments" / "train_exit_models.py"

# Results directory
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_script(script_path: Path, description: str, timeout: int = 600) -> bool:
    """
    Run a training script and capture output

    Returns:
        True if successful, False otherwise
    """
    logger.info("=" * 80)
    logger.info(f"Running: {description}")
    logger.info(f"Script: {script_path}")
    logger.info("=" * 80)

    start_time = datetime.now()

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(PROJECT_ROOT)
        )

        elapsed = (datetime.now() - start_time).total_seconds()

        if result.returncode == 0:
            logger.success(f"✅ {description} completed successfully ({elapsed:.1f}s)")
            logger.debug(f"Output:\n{result.stdout}")
            return True
        else:
            logger.error(f"❌ {description} failed (return code: {result.returncode})")
            logger.error(f"Error output:\n{result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.error(f"❌ {description} timed out after {elapsed:.1f}s")
        return False

    except Exception as e:
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.error(f"❌ {description} failed with exception ({elapsed:.1f}s): {e}")
        return False


def check_model_exists(model_name: str) -> bool:
    """Check if a model file exists"""
    model_files = list(MODELS_DIR.glob(f"{model_name}*.pkl"))
    return len(model_files) > 0


def get_model_info(model_name: str) -> dict:
    """Get model metadata"""
    metadata_files = list(MODELS_DIR.glob(f"{model_name}*metadata*.json"))

    if not metadata_files:
        return None

    try:
        with open(metadata_files[0], 'r') as f:
            metadata = json.load(f)
        return metadata
    except:
        return None


def print_training_summary(results: dict):
    """Print summary of all training results"""
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 80)

    total_time = sum(r['time'] for r in results.values())
    successful = sum(1 for r in results.values() if r['success'])
    failed = len(results) - successful

    logger.info(f"Total Time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    logger.info(f"Successful: {successful}/{len(results)}")
    logger.info(f"Failed: {failed}/{len(results)}")
    logger.info("")

    for step, result in results.items():
        status = "✅" if result['success'] else "❌"
        logger.info(f"{status} {step}: {result['time']:.1f}s")

    logger.info("=" * 80)

    # Model info
    logger.info("\nMODEL INFORMATION:")
    logger.info("=" * 80)

    models_to_check = [
        ("LONG Entry", "xgboost_v4_phase4_advanced_lookahead3_thresh0"),
        ("SHORT Entry", "xgboost_short_model_lookahead3_thresh0.3"),
        ("LONG Exit", "xgboost_v4_long_exit"),
        ("SHORT Exit", "xgboost_v4_short_exit")
    ]

    for model_desc, model_name in models_to_check:
        exists = check_model_exists(model_name)
        metadata = get_model_info(model_name)

        if exists:
            logger.info(f"✅ {model_desc}: {model_name}.pkl")
            if metadata:
                timestamp = metadata.get('timestamp', 'Unknown')
                logger.info(f"   Trained: {timestamp}")
                if 'scores' in metadata:
                    scores = metadata['scores']
                    logger.info(f"   Accuracy: {scores.get('accuracy', 0):.4f}")
                    logger.info(f"   Recall: {scores.get('recall', 0):.4f}")
        else:
            logger.warning(f"⚠️ {model_desc}: NOT FOUND")

    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Train all 4 models")
    parser.add_argument('--download-data', action='store_true',
                       help='Download latest data before training')
    parser.add_argument('--skip-long-entry', action='store_true',
                       help='Skip LONG Entry Model training')
    parser.add_argument('--skip-short-entry', action='store_true',
                       help='Skip SHORT Entry Model training')
    parser.add_argument('--skip-exit-models', action='store_true',
                       help='Skip Exit Models training')

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("UNIFIED MODEL TRAINING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    overall_start = datetime.now()
    results = {}

    # Step 0: Download data (optional)
    if args.download_data:
        step_start = datetime.now()
        success = run_script(DOWNLOAD_SCRIPT, "Data Download", timeout=300)
        step_time = (datetime.now() - step_start).total_seconds()
        results['Data Download'] = {'success': success, 'time': step_time}

        if not success:
            logger.error("Data download failed! Aborting training.")
            return

    # Step 1: LONG Entry Model
    if not args.skip_long_entry:
        step_start = datetime.now()
        success = run_script(LONG_ENTRY_SCRIPT, "LONG Entry Model Training", timeout=600)
        step_time = (datetime.now() - step_start).total_seconds()
        results['LONG Entry Model'] = {'success': success, 'time': step_time}

        if not success:
            logger.error("LONG Entry Model training failed!")
            logger.warning("Continuing with remaining models...")

    # Step 2: SHORT Entry Model
    if not args.skip_short_entry:
        step_start = datetime.now()
        success = run_script(SHORT_ENTRY_SCRIPT, "SHORT Entry Model Training", timeout=600)
        step_time = (datetime.now() - step_start).total_seconds()
        results['SHORT Entry Model'] = {'success': success, 'time': step_time}

        if not success:
            logger.error("SHORT Entry Model training failed!")
            logger.warning("Continuing with remaining models...")

    # Step 3: Exit Models (LONG + SHORT together)
    if not args.skip_exit_models:
        # Check if Entry models exist (required for Exit model training)
        long_entry_exists = check_model_exists("xgboost_v4_phase4_advanced_lookahead3_thresh0")

        if not long_entry_exists:
            logger.error("LONG Entry Model not found! Cannot train Exit Models.")
            logger.error("Please train LONG Entry Model first.")
            results['Exit Models'] = {'success': False, 'time': 0}
        else:
            step_start = datetime.now()
            success = run_script(EXIT_MODELS_SCRIPT, "Exit Models Training (LONG + SHORT)", timeout=1200)
            step_time = (datetime.now() - step_start).total_seconds()
            results['Exit Models'] = {'success': success, 'time': step_time}

            if not success:
                logger.error("Exit Models training failed!")

    # Print summary
    overall_time = (datetime.now() - overall_start).total_seconds()
    print_training_summary(results)

    logger.info("\n" + "=" * 80)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total Duration: {overall_time:.1f}s ({overall_time/60:.1f} minutes)")
    logger.info("=" * 80)

    # Save training report
    report = {
        'timestamp': datetime.now().isoformat(),
        'overall_time': overall_time,
        'results': results
    }

    report_file = RESULTS_DIR / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"\n✅ Training report saved: {report_file}")

    # Exit code based on success
    all_successful = all(r['success'] for r in results.values())
    sys.exit(0 if all_successful else 1)


if __name__ == "__main__":
    main()
