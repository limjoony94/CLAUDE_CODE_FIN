"""
Compare feature lists between Entry and Exit models
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Load Entry model features (used as base for Exit training)
entry_features_path = MODELS_DIR / "xgboost_v4_realistic_labels_features.txt"
with open(entry_features_path, 'r') as f:
    entry_features = [line.strip() for line in f.readlines() if line.strip()]

# Load Exit model features
exit_features_path = MODELS_DIR / "xgboost_v4_long_exit_features.txt"
with open(exit_features_path, 'r') as f:
    exit_features = [line.strip() for line in f.readlines() if line.strip()]

print("=" * 80)
print("FEATURE LIST COMPARISON")
print("=" * 80)

print(f"\nEntry model features: {len(entry_features)}")
print(f"Exit model features: {len(exit_features)}")

# Position features (last 8 in exit model)
position_features = exit_features[-8:]
print(f"\nPosition features (last 8 in exit model):")
for i, feat in enumerate(position_features, 1):
    print(f"  {i}. {feat}")

# Base features in exit model (first 36)
exit_base_features = exit_features[:-8]
print(f"\nExit base features: {len(exit_base_features)}")

# Check for duplicates in entry features
from collections import Counter
entry_counts = Counter(entry_features)
duplicates = {k: v for k, v in entry_counts.items() if v > 1}

if duplicates:
    print(f"\n⚠️ DUPLICATES in Entry features:")
    for feat, count in duplicates.items():
        print(f"  {feat}: appears {count} times")

# Get unique entry features
unique_entry_features = list(dict.fromkeys(entry_features))  # Preserves order, removes duplicates
print(f"\nUnique Entry features: {len(unique_entry_features)}")

# Compare exit base features with unique entry features
print(f"\n{'=' * 80}")
print("COMPARISON: Exit Base vs Entry (unique)")
print(f"{'=' * 80}")

# Find features in exit but not in entry
exit_only = set(exit_base_features) - set(unique_entry_features)
if exit_only:
    print(f"\n✅ Features in Exit base but NOT in Entry:")
    for feat in sorted(exit_only):
        print(f"  - {feat}")
else:
    print(f"\n✅ No features in Exit base that aren't in Entry")

# Find features in entry but not in exit base
entry_only = set(unique_entry_features) - set(exit_base_features)
if entry_only:
    print(f"\n⚠️ Features in Entry but NOT in Exit base:")
    for feat in sorted(entry_only):
        print(f"  - {feat}")
else:
    print(f"\n✅ No features in Entry that aren't in Exit base")

# Check if order is the same (for features that exist in both)
print(f"\n{'=' * 80}")
print("ORDER COMPARISON")
print(f"{'=' * 80}")

order_matches = True
for i, (entry_feat, exit_feat) in enumerate(zip(unique_entry_features, exit_base_features), 1):
    if entry_feat != exit_feat:
        if order_matches:
            print(f"\n⚠️ ORDER MISMATCH detected:")
        print(f"  Position {i}: Entry='{entry_feat}' vs Exit='{exit_feat}'")
        order_matches = False

if order_matches and len(unique_entry_features) == len(exit_base_features):
    print(f"\n✅ Feature order MATCHES perfectly!")
elif len(unique_entry_features) != len(exit_base_features):
    print(f"\n⚠️ Different lengths: Entry={len(unique_entry_features)}, Exit base={len(exit_base_features)}")

print(f"\n{'=' * 80}")
