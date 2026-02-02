"""
Prediction Distribution Collector

Purpose: Collect 24-hour model prediction distributions to identify if
the model is predicting differently in production vs training.

Critical for diagnosing:
- Why signal rate is 19.4% vs expected 6.12%
- Whether model distribution has shifted
- If threshold adjustments are based on accurate assumptions
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

logger.add(LOGS_DIR / "prediction_distribution_collector.log")


class PredictionDistributionCollector:
    """Collect and analyze model prediction distributions over time"""

    def __init__(self):
        self.predictions_file = RESULTS_DIR / "prediction_distributions_24h.json"
        self.data = self._load_existing()

    def _load_existing(self):
        """Load existing prediction data if available"""
        if self.predictions_file.exists():
            with open(self.predictions_file, 'r') as f:
                return json.load(f)
        return {
            'start_time': datetime.now().isoformat(),
            'predictions': {
                'long_entry': [],
                'short_entry': [],
                'long_exit': [],
                'short_exit': []
            },
            'thresholds': {
                'long_entry': [],
                'short_entry': []
            },
            'metadata': []
        }

    def add_prediction(self, model_type, probability, threshold=None, metadata=None):
        """
        Add a prediction to the collection

        Args:
            model_type: 'long_entry', 'short_entry', 'long_exit', 'short_exit'
            probability: Model prediction (0-1)
            threshold: Current threshold (if applicable)
            metadata: Additional info (regime, features, etc.)
        """
        timestamp = datetime.now().isoformat()

        # Store prediction
        self.data['predictions'][model_type].append({
            'timestamp': timestamp,
            'probability': float(probability)
        })

        # Store threshold if provided
        if threshold is not None and model_type in ['long_entry', 'short_entry']:
            self.data['thresholds'][model_type].append({
                'timestamp': timestamp,
                'threshold': float(threshold)
            })

        # Store metadata
        if metadata:
            metadata['timestamp'] = timestamp
            metadata['model_type'] = model_type
            self.data['metadata'].append(metadata)

        # Save immediately (for crash recovery)
        self._save()

    def _save(self):
        """Save current data"""
        with open(self.predictions_file, 'w') as f:
            json.dump(self.data, f, indent=2)

    def analyze(self):
        """Analyze collected predictions and generate report"""

        if not self.predictions_file.exists():
            logger.error("No prediction data collected yet")
            return

        start_time = datetime.fromisoformat(self.data['start_time'])
        duration = (datetime.now() - start_time).total_seconds() / 3600

        logger.info(f"Analyzing {duration:.1f} hours of prediction data...")

        report = []
        report.append("=" * 100)
        report.append("PREDICTION DISTRIBUTION ANALYSIS (24-Hour Collection)")
        report.append("=" * 100)
        report.append("")

        report.append(f"Collection Period:")
        report.append(f"  Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"  Duration: {duration:.1f} hours")
        report.append(f"  Target: 24 hours")
        report.append("")

        # Analyze each model type
        for model_type in ['long_entry', 'short_entry', 'long_exit', 'short_exit']:
            predictions = [p['probability'] for p in self.data['predictions'][model_type]]

            if len(predictions) == 0:
                report.append(f"{model_type.upper()}: No data")
                continue

            predictions = np.array(predictions)

            report.append("=" * 100)
            report.append(f"{model_type.upper()} MODEL PREDICTIONS")
            report.append("=" * 100)
            report.append("")

            report.append(f"Sample Size: {len(predictions)}")
            report.append(f"Collection Rate: {len(predictions)/duration:.1f} predictions/hour")
            report.append("")

            report.append("Distribution Statistics:")
            report.append(f"  Mean: {predictions.mean():.4f}")
            report.append(f"  Median: {np.median(predictions):.4f}")
            report.append(f"  Std Dev: {predictions.std():.4f}")
            report.append(f"  Min: {predictions.min():.4f}")
            report.append(f"  Max: {predictions.max():.4f}")
            report.append("")

            # Percentiles
            percentiles = [10, 25, 50, 75, 90, 95, 99]
            report.append("Percentiles:")
            for p in percentiles:
                val = np.percentile(predictions, p)
                report.append(f"  {p:2d}th: {val:.4f}")
            report.append("")

            # Threshold analysis (for entry models)
            if model_type in ['long_entry', 'short_entry']:
                thresholds = self.data['thresholds'][model_type]
                if thresholds:
                    threshold_values = [t['threshold'] for t in thresholds]
                    avg_threshold = np.mean(threshold_values)

                    # Calculate signal rate at various thresholds
                    report.append("Signal Rates at Different Thresholds:")
                    test_thresholds = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.92, 0.95]
                    for thresh in test_thresholds:
                        signal_rate = (predictions >= thresh).mean()
                        marker = " ← Avg Threshold" if abs(thresh - avg_threshold) < 0.01 else ""
                        report.append(f"  {thresh:.2f}: {signal_rate*100:5.1f}%{marker}")
                    report.append("")

                    report.append(f"Average Threshold (Dynamic): {avg_threshold:.3f}")
                    report.append(f"Signal Rate at Avg Threshold: {(predictions >= avg_threshold).mean()*100:.1f}%")
                    report.append("")

            # Comparison to expected (for entry models)
            if model_type == 'long_entry':
                EXPECTED_SIGNAL_RATE = 0.0612  # From backtest
                BASE_THRESHOLD = 0.70

                actual_signal_rate = (predictions >= BASE_THRESHOLD).mean()

                report.append("Backtest vs Production:")
                report.append(f"  Expected Signal Rate (base 0.70): {EXPECTED_SIGNAL_RATE*100:.1f}%")
                report.append(f"  Actual Signal Rate (base 0.70): {actual_signal_rate*100:.1f}%")

                if actual_signal_rate > EXPECTED_SIGNAL_RATE * 1.5:
                    report.append(f"  ❌ DISTRIBUTION SHIFT: {actual_signal_rate/EXPECTED_SIGNAL_RATE:.2f}x higher")
                elif actual_signal_rate < EXPECTED_SIGNAL_RATE * 0.5:
                    report.append(f"  ⚠️  DISTRIBUTION SHIFT: {EXPECTED_SIGNAL_RATE/actual_signal_rate:.2f}x lower")
                else:
                    report.append(f"  ✅ DISTRIBUTION MATCH: Within expected range")
                report.append("")

        # Overall assessment
        report.append("=" * 100)
        report.append("ASSESSMENT")
        report.append("=" * 100)
        report.append("")

        long_preds = np.array([p['probability'] for p in self.data['predictions']['long_entry']])
        if len(long_preds) > 0:
            signal_rate_70 = (long_preds >= 0.70).mean()
            signal_rate_85 = (long_preds >= 0.85).mean()
            signal_rate_92 = (long_preds >= 0.92).mean()

            report.append("Signal Rate Analysis:")
            report.append(f"  At 0.70 (base): {signal_rate_70*100:.1f}%")
            report.append(f"  At 0.85 (old max): {signal_rate_85*100:.1f}%")
            report.append(f"  At 0.92 (new max): {signal_rate_92*100:.1f}%")
            report.append("")

            if signal_rate_70 > 0.15:  # >15%
                report.append("🚨 CRITICAL: Model predictions significantly shifted")
                report.append("   Possible causes:")
                report.append("   1. Feature calculation bug")
                report.append("   2. Market regime change")
                report.append("   3. Scaler mismatch")
                report.append("")
                report.append("   Recommended actions:")
                report.append("   1. Run feature distribution analysis")
                report.append("   2. Verify scaler min/max ranges")
                report.append("   3. Consider model retraining")
                report.append("")

            elif signal_rate_70 < 0.03:  # <3%
                report.append("⚠️  WARNING: Very low signal rate")
                report.append("   Consider lowering thresholds or checking model")
                report.append("")

            else:
                report.append("✅ Signal rate within reasonable range")
                report.append("")

        # Save report
        report_text = "\n".join(report)
        report_path = PROJECT_ROOT / "claudedocs" / "PREDICTION_DISTRIBUTION_ANALYSIS_20251016.md"

        with open(report_path, 'w') as f:
            f.write(report_text)

        logger.success(f"Report saved: {report_path}")
        print("\n" + report_text)

        return report_text


def main():
    """Run analysis on collected data"""
    collector = PredictionDistributionCollector()
    collector.analyze()


if __name__ == "__main__":
    main()
