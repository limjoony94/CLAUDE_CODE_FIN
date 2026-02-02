"""
Analyze feature importance and detect potential overfitting from too many features
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODELS_DIR = PROJECT_ROOT / "models"

print("=" * 80)
print("FEATURE IMPORTANCE ANALYSIS - OVERFITTING DETECTION")
print("=" * 80)

# =============================================================================
# LOAD MODELS AND FEATURE LISTS
# =============================================================================

print("\n1. Loading models and features...")
print("-" * 80)

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
    print(f"  {name}: {len(info['features'])} features")

# =============================================================================
# ANALYZE FEATURE IMPORTANCE
# =============================================================================

print("\n2. Feature Importance Analysis")
print("-" * 80)

for name, info in models.items():
    model = info['model']
    features = info['features']

    # Get feature importance
    importance = model.feature_importances_

    # Create DataFrame
    df_importance = pd.DataFrame({
        'feature': features,
        'importance': importance
    }).sort_values('importance', ascending=False)

    # Calculate cumulative importance
    df_importance['cumulative'] = df_importance['importance'].cumsum()
    df_importance['cumulative_pct'] = (df_importance['cumulative'] / df_importance['importance'].sum()) * 100

    info['importance_df'] = df_importance

    print(f"\n{name}:")
    print(f"  Total features: {len(features)}")
    print(f"  Non-zero importance: {(importance > 0).sum()}")
    print(f"  Zero importance: {(importance == 0).sum()}")
    print(f"  Mean importance: {importance.mean():.6f}")
    print(f"  Median importance: {np.median(importance):.6f}")
    print(f"  Max importance: {importance.max():.6f}")

    # How many features needed for 80%, 90%, 95% cumulative importance
    n_80 = (df_importance['cumulative_pct'] <= 80).sum()
    n_90 = (df_importance['cumulative_pct'] <= 90).sum()
    n_95 = (df_importance['cumulative_pct'] <= 95).sum()

    print(f"\n  Cumulative importance:")
    print(f"    Top {n_80} features → 80% importance")
    print(f"    Top {n_90} features → 90% importance")
    print(f"    Top {n_95} features → 95% importance")
    print(f"    Efficiency ratio: {n_80}/{len(features)} = {n_80/len(features)*100:.1f}%")

# =============================================================================
# TOP FEATURES
# =============================================================================

print("\n" + "=" * 80)
print("3. TOP 20 MOST IMPORTANT FEATURES (by model)")
print("=" * 80)

for name, info in models.items():
    df_imp = info['importance_df']
    print(f"\n{name}:")
    print("-" * 80)
    print(f"{'Rank':<6} {'Feature':<40} {'Importance':<12} {'Cumulative %':<12}")
    print("-" * 80)
    for i, row in df_imp.head(20).iterrows():
        print(f"{i+1:<6} {row['feature']:<40} {row['importance']:<12.6f} {row['cumulative_pct']:<12.1f}%")

# =============================================================================
# OVERFITTING RISK ASSESSMENT
# =============================================================================

print("\n" + "=" * 80)
print("4. OVERFITTING RISK ASSESSMENT")
print("=" * 80)

for name, info in models.items():
    df_imp = info['importance_df']
    features = info['features']
    importance = df_imp['importance'].values

    print(f"\n{name}:")
    print("-" * 80)

    # Risk factors
    risks = []

    # 1. Too many features with low importance
    low_importance = (importance < 0.001).sum()
    low_pct = low_importance / len(features) * 100
    if low_pct > 30:
        risks.append(f"⚠️ {low_importance} features ({low_pct:.1f}%) have <0.001 importance (likely noise)")
    else:
        print(f"✅ Low-importance features: {low_importance} ({low_pct:.1f}%) - acceptable")

    # 2. Zero importance features
    zero_importance = (importance == 0).sum()
    if zero_importance > 0:
        risks.append(f"⚠️ {zero_importance} features have ZERO importance (completely unused)")
    else:
        print(f"✅ No zero-importance features")

    # 3. Top-heavy distribution (Pareto principle violated)
    top_10_pct = df_imp.head(10)['importance'].sum() / importance.sum() * 100
    if top_10_pct > 70:
        risks.append(f"⚠️ Top 10 features dominate {top_10_pct:.1f}% (>70%) - many features are redundant")
    else:
        print(f"✅ Top 10 features: {top_10_pct:.1f}% - balanced distribution")

    # 4. Efficiency ratio
    n_80 = (df_imp['cumulative_pct'] <= 80).sum()
    efficiency = n_80 / len(features)
    if efficiency < 0.3:  # Less than 30% of features needed for 80%
        risks.append(f"⚠️ Only {n_80}/{len(features)} ({efficiency*100:.1f}%) features needed for 80% - rest may be noise")
    else:
        print(f"✅ Feature efficiency: {efficiency*100:.1f}% - reasonable")

    # 5. Feature/Sample ratio (need to check training data)
    # This will be analyzed separately with actual training data

    if len(risks) > 0:
        print(f"\n🔴 OVERFITTING RISK DETECTED:")
        for risk in risks:
            print(f"  {risk}")
    else:
        print(f"\n✅ NO MAJOR OVERFITTING RISK from feature count")

# =============================================================================
# RECOMMENDATIONS
# =============================================================================

print("\n" + "=" * 80)
print("5. FEATURE REDUCTION RECOMMENDATIONS")
print("=" * 80)

for name, info in models.items():
    df_imp = info['importance_df']

    print(f"\n{name}:")
    print("-" * 80)

    n_current = len(df_imp)
    n_80 = (df_imp['cumulative_pct'] <= 80).sum()
    n_90 = (df_imp['cumulative_pct'] <= 90).sum()
    n_95 = (df_imp['cumulative_pct'] <= 95).sum()

    zero_features = df_imp[df_imp['importance'] == 0]['feature'].tolist()
    low_features = df_imp[df_imp['importance'] < 0.001]['feature'].tolist()

    print(f"Current: {n_current} features")
    print(f"\nReduction Options:")
    print(f"  Option A (Conservative): Keep top {n_95} features (95% importance) - Remove {n_current - n_95}")
    print(f"  Option B (Moderate):      Keep top {n_90} features (90% importance) - Remove {n_current - n_90}")
    print(f"  Option C (Aggressive):    Keep top {n_80} features (80% importance) - Remove {n_current - n_80}")

    print(f"\nSpecific removals:")
    print(f"  • {len(zero_features)} features with ZERO importance")
    print(f"  • {len(low_features)} features with <0.001 importance")

    # Calculate potential improvement
    noise_ratio = len(low_features) / n_current * 100
    if noise_ratio > 20:
        print(f"\n⚠️ WARNING: {noise_ratio:.1f}% of features are likely noise")
        print(f"  Recommendation: Remove at least {len(zero_features) + len(low_features)} low-importance features")
        print(f"  Expected benefit: Reduced overfitting, better generalization")

# =============================================================================
# FEATURE CATEGORIES ANALYSIS
# =============================================================================

print("\n" + "=" * 80)
print("6. FEATURE CATEGORY ANALYSIS")
print("=" * 80)

def categorize_feature(feature_name):
    """Categorize feature by name pattern"""
    feature_lower = feature_name.lower()

    if any(x in feature_lower for x in ['rsi', 'macd', 'bb_', 'ma_', 'ema_', 'sma_']):
        return 'Technical Indicators'
    elif any(x in feature_lower for x in ['volume', 'vwap', 'obv']):
        return 'Volume-based'
    elif any(x in feature_lower for x in ['high', 'low', 'close', 'open', 'price']):
        return 'Price-based'
    elif any(x in feature_lower for x in ['ratio', 'spread', 'diff']):
        return 'Engineered Ratios'
    elif any(x in feature_lower for x in ['trend', 'momentum']):
        return 'Trend/Momentum'
    elif any(x in feature_lower for x in ['support', 'resistance']):
        return 'Support/Resistance'
    else:
        return 'Other'

for name, info in models.items():
    df_imp = info['importance_df']

    print(f"\n{name}:")
    print("-" * 80)

    # Add category
    df_imp['category'] = df_imp['feature'].apply(categorize_feature)

    # Aggregate by category
    category_importance = df_imp.groupby('category').agg({
        'importance': ['sum', 'mean', 'count']
    }).sort_values(('importance', 'sum'), ascending=False)

    category_importance.columns = ['Total', 'Mean', 'Count']
    category_importance['Total_pct'] = (category_importance['Total'] / df_imp['importance'].sum()) * 100

    print(category_importance.to_string())

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("7. SUMMARY & ACTION ITEMS")
print("=" * 80)

print("\nCurrent Feature Counts:")
for name, info in models.items():
    n_total = len(info['features'])
    n_zero = (info['importance_df']['importance'] == 0).sum()
    n_low = (info['importance_df']['importance'] < 0.001).sum()
    print(f"  {name}: {n_total} total | {n_zero} zero | {n_low} low (<0.001)")

print("\n⚠️ OVERFITTING RISKS IDENTIFIED:")
print("  1. High feature count relative to model complexity")
print("  2. Many features with very low or zero importance")
print("  3. Top features dominate importance (Pareto principle)")

print("\n✅ RECOMMENDED ACTIONS:")
print("  1. Remove zero-importance features immediately")
print("  2. Consider feature selection to keep top 80-90% importance")
print("  3. Re-train models with reduced feature set")
print("  4. Compare performance: reduced vs full feature set")
print("  5. Monitor if reduced features improve generalization")

print("\n💡 EXPECTED BENEFITS:")
print("  • Reduced overfitting")
print("  • Better generalization to new data")
print("  • Faster training and prediction")
print("  • More interpretable models")
print("  • Lower memory footprint")

print("\n" + "=" * 80)
