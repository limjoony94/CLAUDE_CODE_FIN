"""
Compare Production (Walk-Forward Decoupled) vs Phase 1 (Optimized) Features

Identify which critical features were removed during Phase 1 optimization.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Load Production features (Walk-Forward Decoupled - 20251027_194313)
prod_long_path = MODELS_DIR / "xgboost_long_entry_walkforward_decoupled_20251027_194313_features.txt"
prod_short_path = MODELS_DIR / "xgboost_short_entry_walkforward_decoupled_20251027_194313_features.txt"

# Load Phase 1 features (Optimized - 20251031_150234/150417)
phase1_long_path = MODELS_DIR / "features_long_optimized_20251031_150234.txt"
phase1_short_path = MODELS_DIR / "features_short_optimized_20251031_150417.txt"

with open(prod_long_path) as f:
    prod_long = set(line.strip() for line in f if line.strip())
with open(prod_short_path) as f:
    prod_short = set(line.strip() for line in f if line.strip())
with open(phase1_long_path) as f:
    phase1_long = set(line.strip() for line in f if line.strip())
with open(phase1_short_path) as f:
    phase1_short = set(line.strip() for line in f if line.strip())

print("="*80)
print("PRODUCTION vs PHASE 1 FEATURE COMPARISON")
print("="*80)

print("\n📊 LONG FEATURES")
print("-"*80)
print(f"Production (Walk-Forward):  {len(prod_long)} features")
print(f"Phase 1 (Optimized):        {len(phase1_long)} features")
print(f"Removed:                    {len(prod_long - phase1_long)} features ({(len(prod_long - phase1_long)/len(prod_long)*100):.1f}%)")
print(f"Added (new):                {len(phase1_long - prod_long)} features")

removed_long = sorted(prod_long - phase1_long)
kept_long = sorted(prod_long & phase1_long)
added_long = sorted(phase1_long - prod_long)

print("\n❌ REMOVED FEATURES (Production → Phase 1):")
for i, feat in enumerate(removed_long, 1):
    print(f"  {i:2d}. {feat}")

if added_long:
    print("\n➕ NEW FEATURES (not in Production):")
    for i, feat in enumerate(added_long, 1):
        print(f"  {i:2d}. {feat}")

print("\n" + "="*80)
print("📊 SHORT FEATURES")
print("-"*80)
print(f"Production (Walk-Forward):  {len(prod_short)} features")
print(f"Phase 1 (Optimized):        {len(phase1_short)} features")
print(f"Removed:                    {len(prod_short - phase1_short)} features ({(len(prod_short - phase1_short)/len(prod_short)*100):.1f}%)")
print(f"Added (new):                {len(phase1_short - prod_short)} features")

removed_short = sorted(prod_short - phase1_short)
kept_short = sorted(prod_short & phase1_short)
added_short = sorted(phase1_short - prod_short)

print("\n❌ REMOVED FEATURES (Production → Phase 1):")
for i, feat in enumerate(removed_short, 1):
    print(f"  {i:2d}. {feat}")

if added_short:
    print("\n➕ NEW FEATURES (not in Production):")
    for i, feat in enumerate(added_short, 1):
        print(f"  {i:2d}. {feat}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"LONG: Removed {len(removed_long)}/{len(prod_long)} ({len(removed_long)/len(prod_long)*100:.1f}%) features")
print(f"SHORT: Removed {len(removed_short)}/{len(prod_short)} ({len(removed_short)/len(prod_short)*100:.1f}%) features")

print("\n🔍 CRITICAL INSIGHT:")
print("Production models use MORE features (85/79) than Phase 1 (50/50)")
print("Phase 1 optimization removed 35/29 features that Production depends on")
print("\n💡 HYPOTHESIS:")
print("The removed features may be critical for filtering low-quality signals")
print("Even if they have low importance individually, they provide essential")
print("quality control that prevents bad entries.")
