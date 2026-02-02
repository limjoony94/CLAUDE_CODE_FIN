"""
EXIT Threshold Comparison - REALISTIC Backtest with Fees
========================================================
REALISTIC production settings:
- Entry: Enhanced 5-Fold CV (20251024_012445) @ 0.80/0.80 thresholds
- Exit: oppgating_improved (20251024_044510)
- Data: 104 days (Jul 14 - Oct 26, 2025), 108 windows of 5 days
- Fees: 0.05% taker (entry + exit = 0.10% per round trip)
- Slippage: 0.01% per execution
- Window-based: Capital resets every 5 days (no compounding between windows)
- Compare: EXIT 0.15, 0.20, 0.75
"""

import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
from datetime import datetime

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "features"
RESULTS_DIR = BASE_DIR / "results"

def prepare_exit_features(df):
    """Add EXIT-specific features required by oppgating_improved models"""
    print("\nPreparing EXIT features...")

    # Volume features
    df['volume_surge'] = (df['volume'] > df['volume'].rolling(20).mean() * 1.5).astype(float)

    # Price momentum features
    if 'sma_20' in df.columns:
        df['price_vs_ma20'] = (df['close'] - df['sma_20']) / df['sma_20']
    else:
        ma_20 = df['close'].rolling(20).mean()
        df['price_vs_ma20'] = (df['close'] - ma_20) / ma_20

    if 'sma_50' in df.columns:
        df['price_vs_ma50'] = (df['close'] - df['sma_50']) / df['sma_50']
    else:
        ma_50 = df['close'].rolling(50).mean()
        df['price_vs_ma50'] = (df['close'] - ma_50) / ma_50

    # Volatility features
    df['returns'] = df['close'].pct_change()
    df['volatility_20'] = df['returns'].rolling(20).std()

    # RSI dynamics
    df['rsi_slope'] = df['rsi'].diff(3) / 3
    df['rsi_overbought'] = (df['rsi'] > 70).astype(float)
    df['rsi_oversold'] = (df['rsi'] < 30).astype(float)
    df['rsi_divergence'] = 0

    # MACD dynamics
    if 'macd_diff' in df.columns:
        df['macd_histogram_slope'] = df['macd_diff'].diff(3) / 3
    else:
        df['macd_histogram_slope'] = 0
    df['macd_crossover'] = ((df['macd'] > df['macd_signal']) &
                            (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(float)
    df['macd_crossunder'] = ((df['macd'] < df['macd_signal']) &
                             (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(float)

    # Price patterns
    df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(float)
    df['price_acceleration'] = df['close'].diff().diff()

    # Support/Resistance proximity
    df['near_support'] = 0

    # Bollinger Band position
    if 'bb_high' in df.columns and 'bb_low' in df.columns:
        df['bb_position'] = (df['close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'])
    else:
        df['bb_position'] = 0.5

    # Clean NaN
    df = df.ffill().bfill()

    print(f"✓ EXIT features prepared")
    return df

# Backtest parameters (REALISTIC production settings) - MOVED UP
LEVERAGE = 4
INITIAL_CAPITAL = 10000
EMERGENCY_MAX_HOLD = 120  # 10 hours
LONG_THRESHOLD = 0.80  # PRODUCTION setting (was 0.65)
SHORT_THRESHOLD = 0.80  # PRODUCTION setting (was 0.70)
GATE_THRESHOLD = 0.001  # Opportunity gating
TAKER_FEE = 0.0005  # 0.05% taker fee (BingX)
SLIPPAGE = 0.0001  # 0.01% slippage per execution
WINDOW_DAYS = 5  # 5-day windows for realistic compounding
THRESHOLDS_TO_TEST = [0.15, 0.20, 0.75]

print("=" * 80)
print("EXIT THRESHOLD COMPARISON - REALISTIC WITH FEES")
print("=" * 80)
print("Settings:")
print(f"  - Entry Thresholds: LONG {LONG_THRESHOLD:.2f}, SHORT {SHORT_THRESHOLD:.2f}")
print(f"  - Fees: {TAKER_FEE:.2%} taker × 2 = {2*TAKER_FEE:.2%} per round trip")
print(f"  - Slippage: {SLIPPAGE:.2%} × 2 = {2*SLIPPAGE:.2%} per round trip")
print(f"  - Total Cost: {2*(TAKER_FEE+SLIPPAGE):.2%} per trade")
print(f"  - Window-Based: {WINDOW_DAYS} days, capital resets each window")
print("=" * 80)
print()

# Load data
print("Loading market data...")
df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_features.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
print(f"✓ Data loaded: {len(df):,} candles")
print(f"  Period: {df['timestamp'].min()} to {df['timestamp'].max()}")

# Prepare EXIT features (ADD missing 15 features)
df = prepare_exit_features(df)
print()

# Load CURRENT production models
print("Loading CURRENT production models...")

# Entry models
with open(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445.pkl", 'rb') as f:
    long_entry_model = pickle.load(f)
with open(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_scaler.pkl", 'rb') as f:
    long_entry_scaler = joblib.load(f)
with open(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_features.txt", 'r') as f:
    long_entry_features = [line.strip() for line in f.readlines() if line.strip()]

with open(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445.pkl", 'rb') as f:
    short_entry_model = pickle.load(f)
with open(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_scaler.pkl", 'rb') as f:
    short_entry_scaler = joblib.load(f)
with open(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_features.txt", 'r') as f:
    short_entry_features = [line.strip() for line in f.readlines() if line.strip()]

# Exit models
with open(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527.pkl", 'rb') as f:
    long_exit_model = pickle.load(f)
with open(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527_scaler.pkl", 'rb') as f:
    long_exit_scaler = joblib.load(f)
with open(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527_features.txt", 'r') as f:
    long_exit_features = [line.strip() for line in f.readlines() if line.strip()]

with open(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510.pkl", 'rb') as f:
    short_exit_model = pickle.load(f)
with open(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510_scaler.pkl", 'rb') as f:
    short_exit_scaler = joblib.load(f)
with open(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510_features.txt", 'r') as f:
    short_exit_features = [line.strip() for line in f.readlines() if line.strip()]

print(f"✓ LONG Entry: {len(long_entry_features)} features")
print(f"✓ SHORT Entry: {len(short_entry_features)} features")
print(f"✓ LONG Exit: {len(long_exit_features)} features")
print(f"✓ SHORT Exit: {len(short_exit_features)} features")
print()

def simulate_trade(df, entry_idx, side, exit_threshold):
    """Simulate a single trade with given exit threshold"""
    entry_price = df.iloc[entry_idx]['close']

    # Select models
    if side == 'LONG':
        exit_model = long_exit_model
        exit_scaler = long_exit_scaler
        exit_feats = long_exit_features
    else:  # SHORT
        exit_model = short_exit_model
        exit_scaler = short_exit_scaler
        exit_feats = short_exit_features

    # Simulate holding position
    for hold_time in range(1, min(len(df) - entry_idx, EMERGENCY_MAX_HOLD + 1)):
        current_idx = entry_idx + hold_time
        if current_idx >= len(df):
            break

        current_candle = df.iloc[current_idx]
        current_price = current_candle['close']

        # Calculate P&L
        if side == 'LONG':
            pnl_pct = (current_price - entry_price) / entry_price
        else:  # SHORT
            pnl_pct = (entry_price - current_price) / entry_price

        leveraged_pnl = pnl_pct * LEVERAGE

        # Apply fees and slippage (REALISTIC)
        # Entry: fee + slippage, Exit: fee + slippage
        total_cost = 2 * (TAKER_FEE + SLIPPAGE)  # 0.12% total
        net_pnl = leveraged_pnl - total_cost

        # Check ML Exit
        try:
            exit_feat = current_candle[exit_feats].values.reshape(1, -1)
            exit_feat_scaled = exit_scaler.transform(exit_feat)
            exit_prob = exit_model.predict_proba(exit_feat_scaled)[0, 1]

            if exit_prob >= exit_threshold:
                return {
                    'exit_reason': 'ml_exit',
                    'hold_time': hold_time,
                    'pnl': net_pnl,  # Fees and slippage applied
                    'exit_prob': exit_prob
                }
        except:
            pass

        # Emergency Max Hold
        if hold_time >= EMERGENCY_MAX_HOLD:
            return {
                'exit_reason': 'max_hold',
                'hold_time': hold_time,
                'pnl': net_pnl,  # Fees and slippage applied
                'exit_prob': None
            }

    # End of data
    return {
        'exit_reason': 'end_of_data',
        'hold_time': len(df) - entry_idx - 1,
        'pnl': net_pnl if 'net_pnl' in locals() else -total_cost,  # At least pay fees
        'exit_prob': None
    }

def backtest_exit_threshold(df, exit_threshold):
    """
    WINDOW-BASED Backtest with given exit threshold
    - Capital resets every 5 days (no compounding between windows)
    - Realistic for comparing to production performance
    """
    all_results = []
    window_returns = []

    # Calculate window boundaries (5 days = 1440 candles)
    candles_per_window = WINDOW_DAYS * 24 * 60 // 5  # 1440 candles
    num_windows = (len(df) - 200) // candles_per_window

    print(f"  Running {num_windows} windows of {WINDOW_DAYS} days ({candles_per_window} candles each)...")

    for window_idx in range(num_windows):
        # Reset capital for each window
        capital = INITIAL_CAPITAL
        window_start = 200 + window_idx * candles_per_window
        window_end = window_start + candles_per_window

        if window_end > len(df) - EMERGENCY_MAX_HOLD - 10:
            break

        next_available_idx = window_start
        window_trades = []

        for i in range(window_start, window_end):
            # Skip if still holding previous position
            if i < next_available_idx:
                continue

            current_candle = df.iloc[i]

            # Get Entry signals
            try:
                # LONG Entry
                long_feat = current_candle[long_entry_features].values.reshape(1, -1)
                long_feat_scaled = long_entry_scaler.transform(long_feat)
                long_prob = long_entry_model.predict_proba(long_feat_scaled)[0, 1]

                # SHORT Entry
                short_feat = current_candle[short_entry_features].values.reshape(1, -1)
                short_feat_scaled = short_entry_scaler.transform(short_feat)
                short_prob = short_entry_model.predict_proba(short_feat_scaled)[0, 1]

                # Entry decision with opportunity gating
                entered = False

                if long_prob >= LONG_THRESHOLD:
                    # LONG entry (no gating)
                    trade_result = simulate_trade(df, i, 'LONG', exit_threshold)
                    pnl_amount = capital * trade_result['pnl']
                    capital += pnl_amount

                    window_trades.append({
                        'entry_idx': i,
                        'side': 'LONG',
                        'entry_prob': long_prob,
                        'pnl': trade_result['pnl'],
                        'pnl_amount': pnl_amount,
                        'hold_time': trade_result['hold_time'],
                        'exit_reason': trade_result['exit_reason'],
                        'exit_prob': trade_result['exit_prob'],
                        'capital': capital
                    })
                    entered = True

                elif short_prob >= SHORT_THRESHOLD:
                    # SHORT entry (with opportunity gating)
                    long_ev = long_prob * 0.0041
                    short_ev = short_prob * 0.0047
                    opportunity_cost = short_ev - long_ev

                    if opportunity_cost > GATE_THRESHOLD:
                        trade_result = simulate_trade(df, i, 'SHORT', exit_threshold)
                        pnl_amount = capital * trade_result['pnl']
                        capital += pnl_amount

                        window_trades.append({
                            'entry_idx': i,
                            'side': 'SHORT',
                            'entry_prob': short_prob,
                            'pnl': trade_result['pnl'],
                            'pnl_amount': pnl_amount,
                            'hold_time': trade_result['hold_time'],
                            'exit_reason': trade_result['exit_reason'],
                            'exit_prob': trade_result['exit_prob'],
                            'capital': capital
                        })
                        entered = True

                if entered:
                    # Update next available entry index (after this trade completes)
                    next_available_idx = i + trade_result['hold_time'] + 1

            except Exception as e:
                # Missing features, skip this candle
                continue

        # Window complete - calculate window return
        window_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
        window_returns.append(window_return)
        all_results.extend(window_trades)

        if window_idx % 10 == 0:
            print(f"    Window {window_idx}/{num_windows}: {len(window_trades)} trades, Return: {window_return:.4f}")

    # All windows complete - aggregate results
    df_results = pd.DataFrame(all_results)

    if len(df_results) == 0:
        return {
            'threshold': exit_threshold,
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0,
            'total_return': 0,
            'avg_pnl': 0,
            'final_capital': INITIAL_CAPITAL,
            'ml_exits': 0,
            'max_hold_exits': 0,
            'ml_exit_rate': 0,
            'avg_hold_time': 0
        }

    # Calculate WINDOW-BASED metrics
    total_trades = len(df_results)
    wins = len(df_results[df_results['pnl'] > 0])
    losses = len(df_results[df_results['pnl'] < 0])
    win_rate = wins / total_trades if total_trades > 0 else 0

    # Window-based return (realistic, no compounding between windows)
    avg_window_return = np.mean(window_returns) if len(window_returns) > 0 else 0
    std_window_return = np.std(window_returns) if len(window_returns) > 0 else 0
    sharpe_ratio = (avg_window_return / std_window_return * np.sqrt(73)) if std_window_return > 0 else 0  # Annualized (73 windows/year)

    avg_pnl = df_results['pnl'].mean()
    trades_per_window = total_trades / len(window_returns) if len(window_returns) > 0 else 0

    ml_exits = len(df_results[df_results['exit_reason'] == 'ml_exit'])
    max_hold_exits = len(df_results[df_results['exit_reason'] == 'max_hold'])
    ml_exit_rate = ml_exits / total_trades if total_trades > 0 else 0

    long_trades = len(df_results[df_results['side'] == 'LONG'])
    short_trades = len(df_results[df_results['side'] == 'SHORT'])

    return {
        'threshold': exit_threshold,
        'total_trades': total_trades,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'avg_window_return': avg_window_return,  # REALISTIC metric
        'std_window_return': std_window_return,
        'sharpe_ratio': sharpe_ratio,
        'trades_per_window': trades_per_window,
        'num_windows': len(window_returns),
        'avg_pnl': avg_pnl,
        'ml_exits': ml_exits,
        'max_hold_exits': max_hold_exits,
        'ml_exit_rate': ml_exit_rate,
        'avg_hold_time': df_results['hold_time'].mean(),
        'long_trades': long_trades,
        'short_trades': short_trades
    }

# Run backtests for all thresholds
print("Running backtests for EXIT thresholds...")
print("(This may take several minutes...)")
print()

all_results = []

for threshold in THRESHOLDS_TO_TEST:
    print(f"Testing EXIT threshold: {threshold:.2f}...")
    result = backtest_exit_threshold(df, threshold)
    all_results.append(result)

    print(f"  Trades: {result['total_trades']} (LONG: {result['long_trades']}, SHORT: {result['short_trades']})")
    print(f"  Win Rate: {result['win_rate']:.1%}")
    print(f"  Avg Return per 5-day window: {result['avg_window_return']:+.2%} (±{result['std_window_return']:.2%})")
    print(f"  Sharpe Ratio (annualized): {result['sharpe_ratio']:.2f}")
    print(f"  Trades per window: {result['trades_per_window']:.1f}")
    print(f"  ML Exit Rate: {result['ml_exit_rate']:.1%}")
    print(f"  Avg Hold Time: {result['avg_hold_time']:.1f} candles")
    print()

# Create comparison DataFrame
df_comparison = pd.DataFrame(all_results)

# Save results
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file = RESULTS_DIR / f"exit_comparison_104days_{timestamp}.csv"
df_comparison.to_csv(output_file, index=False)

print("=" * 80)
print("COMPARISON SUMMARY - 104 DAYS")
print("=" * 80)
print()

print(df_comparison[['threshold', 'total_trades', 'win_rate', 'avg_window_return', 'sharpe_ratio', 'ml_exit_rate', 'trades_per_window']].to_string(index=False))
print()

# Find winner (by Sharpe Ratio - risk-adjusted return)
if len(df_comparison) > 0 and df_comparison['total_trades'].sum() > 0:
    best_idx = df_comparison['sharpe_ratio'].idxmax()
    best_threshold = df_comparison.loc[best_idx, 'threshold']

    print(f"✅ BEST THRESHOLD (by Sharpe Ratio): {best_threshold:.2f}")
    print(f"   Avg Return per 5-day window: {df_comparison.loc[best_idx, 'avg_window_return']:+.2%}")
    print(f"   Sharpe Ratio: {df_comparison.loc[best_idx, 'sharpe_ratio']:.2f}")
    print(f"   Win Rate: {df_comparison.loc[best_idx, 'win_rate']:.1%}")
    print(f"   ML Exit Rate: {df_comparison.loc[best_idx, 'ml_exit_rate']:.1%}")
    print(f"   Trades per window: {df_comparison.loc[best_idx, 'trades_per_window']:.1f}")
else:
    print("⚠️ No trades generated")

print()
print(f"Results saved: {output_file.name}")
print()
print("=" * 80)
