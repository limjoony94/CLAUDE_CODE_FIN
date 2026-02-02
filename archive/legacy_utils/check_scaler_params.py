"""
Check scaler parameters (mean, std) to verify normalization
Created: 2025-11-03
Purpose: Investigate feature value distribution
"""

import joblib
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

print("="*80)
print("SCALER PARAMETER ANALYSIS")
print("="*80)

# Load LONG Entry scaler
print("\n1. LONG ENTRY SCALER (Enhanced 5-Fold CV)")
print("-" * 80)

scaler_path = MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_scaler.pkl"
scaler = joblib.load(scaler_path)

print(f"Scaler type: {type(scaler)}")
print(f"Feature count: {len(scaler.mean_)}")

# Get feature names
features_path = MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_features.txt"
with open(features_path, 'r') as f:
    feature_names = [line.strip() for line in f.readlines()]

print(f"Feature names: {len(feature_names)}")

# Show first 20 features with mean/std
print("\nFirst 20 features (mean, std):")
for i in range(min(20, len(feature_names))):
    feat_name = feature_names[i]
    mean = scaler.mean_[i]
    std = scaler.scale_[i]  # StandardScaler stores std in scale_

    print(f"  {feat_name:40s}: mean={mean:10.4f}, std={std:10.4f}")

# Check for suspicious scaler params
print("\n" + "="*80)
print("SUSPICIOUS SCALER PARAMETERS")
print("="*80)

suspicious = []
for i in range(len(feature_names)):
    mean = scaler.mean_[i]
    std = scaler.scale_[i]

    # Check for zero std (would cause division by zero or invalid normalization)
    if std < 1e-10:
        suspicious.append((feature_names[i], f"Zero std: {std}"))

    # Check for very large std (might indicate outliers)
    if std > 10000:
        suspicious.append((feature_names[i], f"Large std: {std:.2f}"))

    # Check for NaN
    if np.isnan(mean) or np.isnan(std):
        suspicious.append((feature_names[i], "NaN in scaler"))

if suspicious:
    print(f"\n⚠️  Found {len(suspicious)} suspicious parameters:")
    for feat, issue in suspicious:
        print(f"  - {feat}: {issue}")
else:
    print("\n✅ All scaler parameters look normal")

# Check SHORT Entry scaler
print("\n" + "="*80)
print("2. SHORT ENTRY SCALER (Enhanced 5-Fold CV)")
print("-" * 80)

scaler_path = MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_scaler.pkl"
scaler = joblib.load(scaler_path)

print(f"Scaler type: {type(scaler)}")
print(f"Feature count: {len(scaler.mean_)}")

# Get feature names
features_path = MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_features.txt"
with open(features_path, 'r') as f:
    feature_names = [line.strip() for line in f.readlines() if line.strip()]

print(f"Feature names: {len(feature_names)}")

# Show first 20 features
print("\nFirst 20 features (mean, std):")
for i in range(min(20, len(feature_names))):
    feat_name = feature_names[i]
    mean = scaler.mean_[i]
    std = scaler.scale_[i]

    print(f"  {feat_name:40s}: mean={mean:10.4f}, std={std:10.4f}")

# Check suspicious params
suspicious = []
for i in range(len(feature_names)):
    mean = scaler.mean_[i]
    std = scaler.scale_[i]

    if std < 1e-10:
        suspicious.append((feature_names[i], f"Zero std: {std}"))
    if std > 10000:
        suspicious.append((feature_names[i], f"Large std: {std:.2f}"))
    if np.isnan(mean) or np.isnan(std):
        suspicious.append((feature_names[i], "NaN in scaler"))

if suspicious:
    print(f"\n⚠️  Found {len(suspicious)} suspicious parameters:")
    for feat, issue in suspicious:
        print(f"  - {feat}: {issue}")
else:
    print("\n✅ All scaler parameters look normal")

print("\n" + "="*80)
print("DONE")
print("="*80)
