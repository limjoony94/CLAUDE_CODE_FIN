"""
List all zero and low importance features for removal
"""
import sys
from pathlib import Path
import pandas as pd
import joblib

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODELS_DIR = PROJECT_ROOT / "models"

print("=" * 80)
print("ZERO & LOW IMPORTANCE FEATURES - REMOVAL CANDIDATES")
print("=" * 80)

models = {
    'LONG_Entry': {
        'model': joblib.load(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445.pkl"),
        'features_file': MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_features.txt"
    },
    'SHORT_Entry': {
        'model': joblib.load(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445.pkl"),
        'features_file': MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_features.txt"
    },
    'LONG_Exit': {
        'model': joblib.load(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527.pkl"),
        'features_file': MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527_features.txt"
    },
    'SHORT_Exit': {
        'model': joblib.load(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510.pkl"),
        'features_file': MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510_features.txt"
    }
}

# Load feature names
for name, info in models.items():
    with open(info['features_file'], 'r') as f:
        info['features'] = [line.strip() for line in f.readlines() if line.strip()]

all_zero_features = set()
all_low_features = set()

for name, info in models.items():
    model = info['model']
    features = info['features']
    importance = model.feature_importances_

    df_importance = pd.DataFrame({
        'feature': features,
        'importance': importance
    }).sort_values('importance', ascending=False)

    zero_features = df_importance[df_importance['importance'] == 0]['feature'].tolist()
    low_features = df_importance[(df_importance['importance'] > 0) & (df_importance['importance'] < 0.001)]['feature'].tolist()

    print(f"\n{name}:")
    print("-" * 80)
    print(f"Total features: {len(features)}")
    print(f"\nZERO importance ({len(zero_features)}):")
    for f in zero_features:
        print(f"  • {f}")
        all_zero_features.add(f)

    print(f"\nLOW importance <0.001 ({len(low_features)}):")
    for f in low_features:
        print(f"  • {f} (importance: {df_importance[df_importance['feature']==f]['importance'].values[0]:.6f})")
        all_low_features.add(f)

print("\n" + "=" * 80)
print("REMOVAL RECOMMENDATIONS")
print("=" * 80)

print(f"\nOption 1: Remove ZERO-importance features only")
print(f"  Total to remove: {len(all_zero_features)}")
print(f"  Impact: Minimal performance change, reduced overfitting")

print(f"\nOption 2: Remove ZERO + LOW-importance features")
print(f"  Total to remove: {len(all_zero_features) + len(all_low_features)}")
print(f"  Impact: Likely improved generalization")

print(f"\nAll unique features to consider removing:")
print(f"  ZERO: {len(all_zero_features)}")
for f in sorted(all_zero_features):
    print(f"    • {f}")

print(f"\n  LOW (<0.001): {len(all_low_features)}")
for f in sorted(all_low_features):
    print(f"    • {f}")

print("\n" + "=" * 80)
