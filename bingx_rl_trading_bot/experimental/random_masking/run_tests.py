"""
Test Runner for Random Masking Candle Predictor

Run integration tests with proper package structure.

Usage:
    python run_tests.py
"""

import sys
import os
from pathlib import Path

# Get directories
script_dir = Path(__file__).parent.absolute()
experimental_dir = script_dir.parent.absolute()

# Add experimental directory to path so we can import random_masking as a package
sys.path.insert(0, str(experimental_dir))

# Change to experimental directory
os.chdir(experimental_dir)

# Now import from the package
from random_masking.tests.test_integration import IntegrationTester

if __name__ == '__main__':
    print("=" * 70)
    print("Random Masking Candle Predictor - Integration Tests")
    print("=" * 70)
    print()

    # Create tester (use synthetic data for speed)
    tester = IntegrationTester(use_real_data=False)

    # Run all tests
    all_passed = tester.run_all_tests()

    print()
    print("=" * 70)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 70)

    sys.exit(0 if all_passed else 1)
