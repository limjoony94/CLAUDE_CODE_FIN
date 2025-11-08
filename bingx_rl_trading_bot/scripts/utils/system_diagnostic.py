"""
System Diagnostic Tool

Comprehensive check of all bot components and configurations.
"""

import sys
from pathlib import Path
import json
import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def check_models():
    """Check if all required models exist"""
    print("\n📦 Checking Models...")
    print("-" * 60)

    models_dir = PROJECT_ROOT / "models"
    required_models = [
        "xgboost_long_trade_outcome_full_20251018_233146.pkl",
        "xgboost_short_trade_outcome_full_20251018_233146.pkl",
        "xgboost_long_exit_oppgating_improved_20251017_151624.pkl",
        "xgboost_short_exit_oppgating_improved_20251017_152440.pkl",
        "scaler_long_trade_outcome_full_20251018_233146.pkl",
        "scaler_short_trade_outcome_full_20251018_233146.pkl"
    ]

    all_exist = True
    for model in required_models:
        model_path = models_dir / model
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"  ✅ {model} ({size_mb:.2f} MB)")
        else:
            print(f"  ❌ {model} - MISSING!")
            all_exist = False

    return all_exist

def check_config():
    """Check API configuration"""
    print("\n⚙️  Checking Configuration...")
    print("-" * 60)

    api_keys_file = PROJECT_ROOT / "config" / "api_keys.yaml"

    if not api_keys_file.exists():
        print("  ❌ config/api_keys.yaml - NOT FOUND!")
        return False

    try:
        with open(api_keys_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        mainnet_config = config.get('bingx', {}).get('mainnet', {})

        if mainnet_config.get('api_key') and mainnet_config.get('secret_key'):
            print("  ✅ API keys configured (mainnet)")
            print(f"     API Key: {mainnet_config['api_key'][:8]}...")
            return True
        else:
            print("  ❌ API keys not found in config!")
            return False

    except Exception as e:
        print(f"  ❌ Error reading config: {e}")
        return False

def check_state_file():
    """Check state file integrity"""
    print("\n📊 Checking State File...")
    print("-" * 60)

    state_file = PROJECT_ROOT / "results" / "opportunity_gating_bot_4x_state.json"

    if not state_file.exists():
        print("  ❌ State file not found!")
        return False

    try:
        with open(state_file, 'r', encoding='utf-8') as f:
            state = json.load(f)

        print(f"  ✅ State file valid JSON")
        print(f"     Balance: ${state.get('current_balance', 0):,.2f}")
        print(f"     Trades: {len(state.get('trades', []))}")
        print(f"     Position: {state.get('position')}")

        # Check configuration
        config = state.get('configuration', {})
        emergency_sl = config.get('emergency_stop_loss')
        sl_strategy = config.get('sl_strategy')

        if emergency_sl == 0.06 and sl_strategy == 'balance_6pct':
            print(f"  ✅ Balance-Based SL configured correctly")
            print(f"     emergency_stop_loss: {emergency_sl}")
            print(f"     sl_strategy: {sl_strategy}")
            return True
        else:
            print(f"  ⚠️  Configuration mismatch:")
            print(f"     emergency_stop_loss: {emergency_sl} (expected: 0.06)")
            print(f"     sl_strategy: {sl_strategy} (expected: balance_6pct)")
            return False

    except json.JSONDecodeError as e:
        print(f"  ❌ Invalid JSON: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def check_backup():
    """Check if backup exists"""
    print("\n💾 Checking Backup...")
    print("-" * 60)

    results_dir = PROJECT_ROOT / "results"
    backups = list(results_dir.glob("opportunity_gating_bot_4x_state_backup_*.json"))

    if backups:
        latest_backup = max(backups, key=lambda p: p.stat().st_mtime)
        print(f"  ✅ Latest backup: {latest_backup.name}")
        return True
    else:
        print("  ⚠️  No backups found")
        return False

def check_logs():
    """Check log directory"""
    print("\n📝 Checking Logs...")
    print("-" * 60)

    logs_dir = PROJECT_ROOT / "logs"

    if not logs_dir.exists():
        print("  ⚠️  Logs directory not found")
        return False

    log_files = list(logs_dir.glob("opportunity_gating_bot_4x_*.log"))

    if log_files:
        latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
        size_kb = latest_log.stat().st_size / 1024
        print(f"  ✅ Latest log: {latest_log.name} ({size_kb:.1f} KB)")
        return True
    else:
        print("  ⚠️  No log files found")
        return False

def check_documentation():
    """Check recent documentation"""
    print("\n📚 Checking Documentation...")
    print("-" * 60)

    docs = [
        "claudedocs/BALANCE_BASED_SL_DEPLOYMENT_20251021.md",
        "claudedocs/STOP_LOSS_BUG_FIX_20251021.md",
        "SYSTEM_STATUS.md"
    ]

    all_exist = True
    for doc in docs:
        doc_path = PROJECT_ROOT / doc
        if doc_path.exists():
            print(f"  ✅ {doc}")
        else:
            print(f"  ⚠️  {doc} - not found")
            all_exist = False

    return all_exist

def main():
    print("=" * 60)
    print("SYSTEM DIAGNOSTIC TOOL")
    print("=" * 60)
    print(f"Project Root: {PROJECT_ROOT}")

    results = {
        'models': check_models(),
        'config': check_config(),
        'state_file': check_state_file(),
        'backup': check_backup(),
        'logs': check_logs(),
        'documentation': check_documentation()
    }

    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for component, status in results.items():
        icon = "✅" if status else "❌"
        print(f"  {icon} {component.replace('_', ' ').title()}")

    print(f"\n📊 Overall: {passed}/{total} checks passed")

    if passed == total:
        print("\n✅ SYSTEM READY FOR DEPLOYMENT")
    elif passed >= total - 1:
        print("\n⚠️  SYSTEM MOSTLY READY (Minor issues)")
    else:
        print("\n❌ SYSTEM NOT READY (Critical issues found)")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
