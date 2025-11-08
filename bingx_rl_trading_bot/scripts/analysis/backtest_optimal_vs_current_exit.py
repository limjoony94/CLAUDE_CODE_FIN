"""
Backtest Comparison: Optimal Exit vs Current Exit Models

Configuration A (NEW):
- Entry: 90-day Trade Outcome (171 features)
- Exit: Optimal Triple Barrier (171 features, 15% exit rate)

Configuration B (CURRENT):
- Entry: 90-day Trade Outcome (171 features)
- Exit: 52-day (12 features)

Validation Period: Sep 29 - Oct 26, 2025 (28 days)
"""

import sys
from pathlib import Path
BASE_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(BASE_DIR / "scripts" / "production"))

import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import joblib

# Configuration
STARTING_BALANCE = 300.0
LEVERAGE = 4.0
ENTRY_THRESHOLD_LONG = 0.85
ENTRY_THRESHOLD_SHORT = 0.80
EXIT_THRESHOLD = 0.75
STOP_LOSS_PCT = -0.03  # -3% balance-based
MAX_HOLD_CANDLES = 120  # 10 hours

# Paths
DATA_DIR = BASE_DIR / "data" / "features"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Input data
FEATURES_FILE = DATA_DIR / "BTCUSDT_5m_features_90days_complete_20251106_164542.csv"

# Entry models (same for both)
ENTRY_LONG = MODELS_DIR / "xgboost_long_entry_90days_tradeoutcome_20251106_193900.pkl"
ENTRY_SHORT = MODELS_DIR / "xgboost_short_entry_90days_tradeoutcome_20251106_193900.pkl"
ENTRY_SCALER_LONG = MODELS_DIR / "xgboost_long_entry_90days_tradeoutcome_20251106_193900_scaler.pkl"
ENTRY_SCALER_SHORT = MODELS_DIR / "xgboost_short_entry_90days_tradeoutcome_20251106_193900_scaler.pkl"
ENTRY_FEATURES_LONG = MODELS_DIR / "xgboost_long_entry_90days_tradeoutcome_20251106_193900_features.txt"
ENTRY_FEATURES_SHORT = MODELS_DIR / "xgboost_short_entry_90days_tradeoutcome_20251106_193900_features.txt"

# Exit models - OPTIMAL (NEW)
EXIT_LONG_OPTIMAL = MODELS_DIR / "xgboost_long_exit_optimal_20251106_223613.pkl"
EXIT_SHORT_OPTIMAL = MODELS_DIR / "xgboost_short_exit_optimal_20251106_223613.pkl"
EXIT_SCALER_LONG_OPTIMAL = MODELS_DIR / "xgboost_long_exit_optimal_20251106_223613_scaler.pkl"
EXIT_SCALER_SHORT_OPTIMAL = MODELS_DIR / "xgboost_short_exit_optimal_20251106_223613_scaler.pkl"
EXIT_FEATURES_LONG_OPTIMAL = MODELS_DIR / "xgboost_long_exit_optimal_20251106_223613_features.txt"
EXIT_FEATURES_SHORT_OPTIMAL = MODELS_DIR / "xgboost_short_exit_optimal_20251106_223613_features.txt"

# Exit models - CURRENT (52-day, 12 features)
EXIT_LONG_CURRENT = MODELS_DIR / "xgboost_long_exit_52day_20251106_140955.pkl"
EXIT_SHORT_CURRENT = MODELS_DIR / "xgboost_short_exit_52day_20251106_140955.pkl"
EXIT_SCALER_LONG_CURRENT = MODELS_DIR / "xgboost_long_exit_52day_20251106_140955_scaler.pkl"
EXIT_SCALER_SHORT_CURRENT = MODELS_DIR / "xgboost_short_exit_52day_20251106_140955_scaler.pkl"
EXIT_FEATURES_LONG_CURRENT = MODELS_DIR / "xgboost_long_exit_52day_20251106_140955_features.txt"
EXIT_FEATURES_SHORT_CURRENT = MODELS_DIR / "xgboost_short_exit_52day_20251106_140955_features.txt"

# Backtest period
BACKTEST_START = "2025-09-29"
BACKTEST_END = "2025-10-26"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def load_model_set(entry_long, entry_short, exit_long, exit_short,
                   entry_scaler_long, entry_scaler_short,
                   exit_scaler_long, exit_scaler_short,
                   entry_features_long, entry_features_short,
                   exit_features_long, exit_features_short, name):
    """Load a complete set of Entry + Exit models"""
    print(f"\n📦 Loading {name} models...")

    models = {}

    # Entry models
    with open(entry_long, 'rb') as f:
        models['entry_long_model'] = pickle.load(f)
    with open(entry_short, 'rb') as f:
        models['entry_short_model'] = pickle.load(f)

    models['entry_long_scaler'] = joblib.load(entry_scaler_long)
    models['entry_short_scaler'] = joblib.load(entry_scaler_short)

    with open(entry_features_long, 'r') as f:
        models['entry_long_features'] = [line.strip() for line in f]
    with open(entry_features_short, 'r') as f:
        models['entry_short_features'] = [line.strip() for line in f]

    # Exit models
    with open(exit_long, 'rb') as f:
        models['exit_long_model'] = pickle.load(f)
    with open(exit_short, 'rb') as f:
        models['exit_short_model'] = pickle.load(f)

    models['exit_long_scaler'] = joblib.load(exit_scaler_long)
    models['exit_short_scaler'] = joblib.load(exit_scaler_short)

    with open(exit_features_long, 'r') as f:
        models['exit_long_features'] = [line.strip() for line in f]
    with open(exit_features_short, 'r') as f:
        models['exit_short_features'] = [line.strip() for line in f]

    print(f"   ✅ Entry LONG: {len(models['entry_long_features'])} features")
    print(f"   ✅ Entry SHORT: {len(models['entry_short_features'])} features")
    print(f"   ✅ Exit LONG: {len(models['exit_long_features'])} features")
    print(f"   ✅ Exit SHORT: {len(models['exit_short_features'])} features")

    return models


def predict_probabilities(df, models, config_name):
    """Calculate Entry and Exit probabilities for all candles"""
    print(f"\n🔮 Calculating probabilities for {config_name}...")

    # Entry LONG
    X_entry_long = df[models['entry_long_features']]
    X_entry_long_scaled = models['entry_long_scaler'].transform(X_entry_long)
    long_entry_proba = models['entry_long_model'].predict_proba(X_entry_long_scaled)[:, 1]

    # Entry SHORT
    X_entry_short = df[models['entry_short_features']]
    X_entry_short_scaled = models['entry_short_scaler'].transform(X_entry_short)
    short_entry_proba = models['entry_short_model'].predict_proba(X_entry_short_scaled)[:, 1]

    # Exit LONG
    X_exit_long = df[models['exit_long_features']]
    X_exit_long_scaled = models['exit_long_scaler'].transform(X_exit_long)
    long_exit_proba = models['exit_long_model'].predict_proba(X_exit_long_scaled)[:, 1]

    # Exit SHORT
    X_exit_short = df[models['exit_short_features']]
    X_exit_short_scaled = models['exit_short_scaler'].transform(X_exit_short)
    short_exit_proba = models['exit_short_model'].predict_proba(X_exit_short_scaled)[:, 1]

    df['long_entry_proba'] = long_entry_proba
    df['short_entry_proba'] = short_entry_proba
    df['long_exit_proba'] = long_exit_proba
    df['short_exit_proba'] = short_exit_proba

    print(f"   ✅ LONG Entry: max {long_entry_proba.max():.4f}")
    print(f"   ✅ SHORT Entry: max {short_entry_proba.max():.4f}")
    print(f"   ✅ LONG Exit: max {long_exit_proba.max():.4f}")
    print(f"   ✅ SHORT Exit: max {short_exit_proba.max():.4f}")

    return df


def run_backtest(df, config_name):
    """Run backtest simulation"""
    print(f"\n{'='*80}")
    print(f"BACKTESTING: {config_name}")
    print(f"{'='*80}")

    balance = STARTING_BALANCE
    position = None
    trades = []

    for i in range(len(df)):
        current = df.iloc[i]
        timestamp = current['timestamp']
        price = current['close']

        # Check exit first if in position
        if position is not None:
            should_exit = False
            exit_reason = None
            exit_prob = 0.0

            # ML Exit check
            if position['side'] == 'LONG':
                exit_prob = current['long_exit_proba']
                if exit_prob >= EXIT_THRESHOLD:
                    should_exit = True
                    exit_reason = 'ML_Exit'
            else:  # SHORT
                exit_prob = current['short_exit_proba']
                if exit_prob >= EXIT_THRESHOLD:
                    should_exit = True
                    exit_reason = 'ML_Exit'

            # Stop Loss check (balance-based)
            if not should_exit:
                pnl_pct = (price - position['entry_price']) / position['entry_price']
                if position['side'] == 'SHORT':
                    pnl_pct = -pnl_pct

                pnl_balance_pct = pnl_pct * LEVERAGE
                if pnl_balance_pct <= STOP_LOSS_PCT:
                    should_exit = True
                    exit_reason = 'Stop_Loss'

            # Max Hold check
            if not should_exit:
                hold_candles = i - position['entry_idx']
                if hold_candles >= MAX_HOLD_CANDLES:
                    should_exit = True
                    exit_reason = 'Max_Hold'

            # Execute exit
            if should_exit:
                pnl_pct = (price - position['entry_price']) / position['entry_price']
                if position['side'] == 'SHORT':
                    pnl_pct = -pnl_pct

                pnl_balance_pct = pnl_pct * LEVERAGE
                pnl_usd = balance * pnl_balance_pct

                balance += pnl_usd

                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': timestamp,
                    'side': position['side'],
                    'entry_price': position['entry_price'],
                    'exit_price': price,
                    'entry_probability': position['entry_prob'],
                    'exit_probability': exit_prob,
                    'pnl_pct': pnl_pct * 100,
                    'pnl_usd': pnl_usd,
                    'balance': balance,
                    'exit_reason': exit_reason,
                    'hold_candles': i - position['entry_idx']
                })

                position = None

        # Check entry if no position
        if position is None:
            long_entry = current['long_entry_proba']
            short_entry = current['short_entry_proba']

            # Opportunity gating logic
            if long_entry >= ENTRY_THRESHOLD_LONG and short_entry >= ENTRY_THRESHOLD_SHORT:
                # Both signals strong - choose higher EV
                long_ev = long_entry
                short_ev = short_entry

                if short_ev > long_ev + 0.001:
                    side = 'SHORT'
                    entry_prob = short_entry
                else:
                    side = 'LONG'
                    entry_prob = long_entry

            elif long_entry >= ENTRY_THRESHOLD_LONG:
                side = 'LONG'
                entry_prob = long_entry

            elif short_entry >= ENTRY_THRESHOLD_SHORT:
                side = 'SHORT'
                entry_prob = short_entry

            else:
                continue

            # Enter position
            position = {
                'side': side,
                'entry_price': price,
                'entry_time': timestamp,
                'entry_prob': entry_prob,
                'entry_idx': i
            }

    # Close any remaining position at end
    if position is not None:
        final_row = df.iloc[-1]
        price = final_row['close']

        pnl_pct = (price - position['entry_price']) / position['entry_price']
        if position['side'] == 'SHORT':
            pnl_pct = -pnl_pct

        pnl_balance_pct = pnl_pct * LEVERAGE
        pnl_usd = balance * pnl_balance_pct
        balance += pnl_usd

        trades.append({
            'entry_time': position['entry_time'],
            'exit_time': final_row['timestamp'],
            'side': position['side'],
            'entry_price': position['entry_price'],
            'exit_price': price,
            'entry_probability': position['entry_prob'],
            'exit_probability': 0.0,
            'pnl_pct': pnl_pct * 100,
            'pnl_usd': pnl_usd,
            'balance': balance,
            'exit_reason': 'End_of_Period',
            'hold_candles': len(df) - 1 - position['entry_idx']
        })

    return balance, trades


def analyze_results(trades, config_name, starting_balance):
    """Analyze backtest results"""
    if not trades:
        print(f"\n⚠️ {config_name}: No trades executed")
        return None

    df_trades = pd.DataFrame(trades)

    total_return = ((df_trades['balance'].iloc[-1] - starting_balance) / starting_balance) * 100
    total_trades = len(df_trades)
    wins = (df_trades['pnl_usd'] > 0).sum()
    losses = (df_trades['pnl_usd'] < 0).sum()
    breakeven = (df_trades['pnl_usd'] == 0).sum()
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0

    avg_win = df_trades[df_trades['pnl_usd'] > 0]['pnl_usd'].mean() if wins > 0 else 0
    avg_loss = df_trades[df_trades['pnl_usd'] < 0]['pnl_usd'].mean() if losses > 0 else 0

    profit_factor = abs(df_trades[df_trades['pnl_usd'] > 0]['pnl_usd'].sum() /
                       df_trades[df_trades['pnl_usd'] < 0]['pnl_usd'].sum()) if losses > 0 else float('inf')

    # Exit mechanism breakdown
    exit_reasons = df_trades['exit_reason'].value_counts()
    ml_exit_pct = (exit_reasons.get('ML_Exit', 0) / total_trades) * 100 if total_trades > 0 else 0
    sl_pct = (exit_reasons.get('Stop_Loss', 0) / total_trades) * 100 if total_trades > 0 else 0
    max_hold_pct = (exit_reasons.get('Max_Hold', 0) / total_trades) * 100 if total_trades > 0 else 0

    # Direction breakdown
    long_trades = df_trades[df_trades['side'] == 'LONG']
    short_trades = df_trades[df_trades['side'] == 'SHORT']

    results = {
        'config_name': config_name,
        'total_return': total_return,
        'final_balance': df_trades['balance'].iloc[-1],
        'total_trades': total_trades,
        'wins': wins,
        'losses': losses,
        'breakeven': breakeven,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'ml_exit_pct': ml_exit_pct,
        'sl_pct': sl_pct,
        'max_hold_pct': max_hold_pct,
        'long_count': len(long_trades),
        'short_count': len(short_trades),
        'long_wr': (long_trades['pnl_usd'] > 0).mean() * 100 if len(long_trades) > 0 else 0,
        'short_wr': (short_trades['pnl_usd'] > 0).mean() * 100 if len(short_trades) > 0 else 0,
        'avg_hold_candles': df_trades['hold_candles'].mean()
    }

    return results, df_trades


def main():
    print("="*80)
    print("BACKTEST COMPARISON: OPTIMAL EXIT VS CURRENT EXIT")
    print("="*80)

    print(f"\n📊 Period: {BACKTEST_START} to {BACKTEST_END}")
    print(f"💰 Starting Balance: ${STARTING_BALANCE}")
    print(f"📈 Leverage: {LEVERAGE}x")
    print(f"🎯 Entry Thresholds: LONG {ENTRY_THRESHOLD_LONG}, SHORT {ENTRY_THRESHOLD_SHORT}")
    print(f"🚪 Exit Threshold: {EXIT_THRESHOLD}")

    # Load data
    print(f"\n{'='*80}")
    print("LOADING DATA")
    print(f"{'='*80}")

    df = pd.read_csv(FEATURES_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Filter to backtest period
    df = df[(df['timestamp'] >= BACKTEST_START) & (df['timestamp'] <= BACKTEST_END)].copy()
    df = df.reset_index(drop=True)

    print(f"\n✅ Loaded {len(df):,} candles")
    print(f"   Period: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"   Days: {(df['timestamp'].max() - df['timestamp'].min()).days}")

    # Load model sets
    print(f"\n{'='*80}")
    print("LOADING MODELS")
    print(f"{'='*80}")

    models_optimal = load_model_set(
        ENTRY_LONG, ENTRY_SHORT, EXIT_LONG_OPTIMAL, EXIT_SHORT_OPTIMAL,
        ENTRY_SCALER_LONG, ENTRY_SCALER_SHORT,
        EXIT_SCALER_LONG_OPTIMAL, EXIT_SCALER_SHORT_OPTIMAL,
        ENTRY_FEATURES_LONG, ENTRY_FEATURES_SHORT,
        EXIT_FEATURES_LONG_OPTIMAL, EXIT_FEATURES_SHORT_OPTIMAL,
        "OPTIMAL EXIT (171 features, 15% exit rate)"
    )

    models_current = load_model_set(
        ENTRY_LONG, ENTRY_SHORT, EXIT_LONG_CURRENT, EXIT_SHORT_CURRENT,
        ENTRY_SCALER_LONG, ENTRY_SCALER_SHORT,
        EXIT_SCALER_LONG_CURRENT, EXIT_SCALER_SHORT_CURRENT,
        ENTRY_FEATURES_LONG, ENTRY_FEATURES_SHORT,
        EXIT_FEATURES_LONG_CURRENT, EXIT_FEATURES_SHORT_CURRENT,
        "CURRENT EXIT (12 features, 52-day)"
    )

    # Calculate probabilities
    print(f"\n{'='*80}")
    print("CALCULATING PROBABILITIES")
    print(f"{'='*80}")

    df_optimal = predict_probabilities(df.copy(), models_optimal, "OPTIMAL EXIT")
    df_current = predict_probabilities(df.copy(), models_current, "CURRENT EXIT")

    # Run backtests
    print(f"\n{'='*80}")
    print("RUNNING BACKTESTS")
    print(f"{'='*80}")

    balance_optimal, trades_optimal = run_backtest(df_optimal, "OPTIMAL EXIT")
    balance_current, trades_current = run_backtest(df_current, "CURRENT EXIT")

    # Analyze results
    print(f"\n{'='*80}")
    print("ANALYZING RESULTS")
    print(f"{'='*80}")

    results_optimal, df_trades_optimal = analyze_results(trades_optimal, "OPTIMAL EXIT", STARTING_BALANCE)
    results_current, df_trades_current = analyze_results(trades_current, "CURRENT EXIT", STARTING_BALANCE)

    # Save trade logs
    output_optimal = RESULTS_DIR / f"backtest_optimal_exit_{TIMESTAMP}.csv"
    output_current = RESULTS_DIR / f"backtest_current_exit_{TIMESTAMP}.csv"

    df_trades_optimal.to_csv(output_optimal, index=False)
    df_trades_current.to_csv(output_current, index=False)

    print(f"\n💾 Saved: {output_optimal.name}")
    print(f"💾 Saved: {output_current.name}")

    # Print comparison
    print(f"\n{'='*80}")
    print("COMPARISON RESULTS")
    print(f"{'='*80}")

    print(f"\n📊 OPTIMAL EXIT (171 features, 15% exit rate):")
    print(f"   Total Return: {results_optimal['total_return']:+.2f}%")
    print(f"   Final Balance: ${results_optimal['final_balance']:.2f}")
    print(f"   Total Trades: {results_optimal['total_trades']}")
    print(f"   Win Rate: {results_optimal['win_rate']:.2f}%")
    print(f"   Profit Factor: {results_optimal['profit_factor']:.2f}x")
    print(f"\n   Exit Mechanisms:")
    print(f"      ML Exit: {results_optimal['ml_exit_pct']:.1f}%")
    print(f"      Stop Loss: {results_optimal['sl_pct']:.1f}%")
    print(f"      Max Hold: {results_optimal['max_hold_pct']:.1f}%")
    print(f"\n   Direction:")
    print(f"      LONG: {results_optimal['long_count']} ({results_optimal['long_wr']:.1f}% WR)")
    print(f"      SHORT: {results_optimal['short_count']} ({results_optimal['short_wr']:.1f}% WR)")
    print(f"   Avg Hold: {results_optimal['avg_hold_candles']:.1f} candles")

    print(f"\n📊 CURRENT EXIT (12 features, 52-day):")
    print(f"   Total Return: {results_current['total_return']:+.2f}%")
    print(f"   Final Balance: ${results_current['final_balance']:.2f}")
    print(f"   Total Trades: {results_current['total_trades']}")
    print(f"   Win Rate: {results_current['win_rate']:.2f}%")
    print(f"   Profit Factor: {results_current['profit_factor']:.2f}x")
    print(f"\n   Exit Mechanisms:")
    print(f"      ML Exit: {results_current['ml_exit_pct']:.1f}%")
    print(f"      Stop Loss: {results_current['sl_pct']:.1f}%")
    print(f"      Max Hold: {results_current['max_hold_pct']:.1f}%")
    print(f"\n   Direction:")
    print(f"      LONG: {results_current['long_count']} ({results_current['long_wr']:.1f}% WR)")
    print(f"      SHORT: {results_current['short_count']} ({results_current['short_wr']:.1f}% WR)")
    print(f"   Avg Hold: {results_current['avg_hold_candles']:.1f} candles")

    # Comparison summary
    print(f"\n{'='*80}")
    print("WINNER ANALYSIS")
    print(f"{'='*80}")

    return_diff = results_optimal['total_return'] - results_current['total_return']
    wr_diff = results_optimal['win_rate'] - results_current['win_rate']
    ml_exit_diff = results_optimal['ml_exit_pct'] - results_current['ml_exit_pct']

    print(f"\n📈 Return Difference: {return_diff:+.2f}% (OPTIMAL - CURRENT)")
    print(f"🎯 Win Rate Difference: {wr_diff:+.2f}%")
    print(f"🚪 ML Exit Usage Difference: {ml_exit_diff:+.1f}%")

    if return_diff > 0:
        print(f"\n✅ WINNER: OPTIMAL EXIT")
        print(f"   Better by: {return_diff:.2f}% return")
        recommendation = "DEPLOY"
    else:
        print(f"\n✅ WINNER: CURRENT EXIT")
        print(f"   Better by: {-return_diff:.2f}% return")
        recommendation = "KEEP CURRENT"

    print(f"\n{'='*80}")
    print("RECOMMENDATION")
    print(f"{'='*80}")
    print(f"\n🎯 {recommendation}")

    if recommendation == "DEPLOY":
        print(f"\n📋 Next Steps:")
        print(f"   1. Review trade logs for quality")
        print(f"   2. Update production bot configuration")
        print(f"   3. Deploy Optimal Exit models (171 features)")
        print(f"   4. Monitor first week of live trading")
    else:
        print(f"\n📋 Analysis:")
        print(f"   - Current 12-feature Exit models still competitive")
        print(f"   - 171 features did not improve performance significantly")
        print(f"   - Consider alternative labeling approaches")


if __name__ == "__main__":
    main()
