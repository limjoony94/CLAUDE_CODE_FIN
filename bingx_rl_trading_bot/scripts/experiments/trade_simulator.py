"""
Trade Simulator for Entry Labeling
===================================

Simulate trades from entry to exit using actual strategy logic
to create realistic entry labels based on trade outcomes.

UPDATED 2025-10-24: Added fee calculation (matches backtest logic)
"""

import numpy as np
import pandas as pd
import pickle
import joblib
from pathlib import Path

# Trading Fees (BingX TAKER_FEE - same as backtest)
TAKER_FEE = 0.0005  # 0.05% per trade (market orders)

class TradeSimulator:
    """
    Simulate individual trades for entry labeling

    Uses same logic as backtest to ensure labels reflect actual trading outcomes.
    """

    def __init__(
        self,
        exit_model,
        exit_scaler,
        exit_features,
        leverage=4,
        ml_exit_threshold=0.70,
        emergency_stop_loss=-0.04,
        emergency_max_hold=96  # 8 hours
    ):
        self.exit_model = exit_model
        self.exit_scaler = exit_scaler
        self.exit_features = exit_features
        self.leverage = leverage
        self.ml_exit_threshold = ml_exit_threshold
        self.emergency_stop_loss = emergency_stop_loss
        self.emergency_max_hold = emergency_max_hold

    def simulate_trade(self, df, entry_idx, side):
        """
        Simulate a single trade from entry to exit

        Returns:
            dict: Trade outcome metrics
                - pnl_pct: Unleveraged P&L percentage
                - leveraged_pnl_pct: Leveraged P&L percentage
                - mae: Maximum Adverse Excursion (worst drawdown)
                - mfe: Maximum Favorable Excursion (best profit)
                - hold_time: Candles held
                - exit_reason: Why trade was closed
                - exit_idx: Exit candle index
        """
        if entry_idx >= len(df) - self.emergency_max_hold:
            return None  # Not enough data to simulate

        entry_price = df['close'].iloc[entry_idx]

        # Track MAE and MFE
        mae = 0.0  # Maximum Adverse Excursion
        mfe = 0.0  # Maximum Favorable Excursion

        # Simulate holding
        for i in range(entry_idx + 1, min(entry_idx + self.emergency_max_hold + 1, len(df))):
            current_price = df['close'].iloc[i]
            hold_time = i - entry_idx

            # Calculate P&L
            if side == 'LONG':
                pnl_pct = (current_price - entry_price) / entry_price
            else:  # SHORT
                pnl_pct = (entry_price - current_price) / entry_price

            # Update MAE/MFE
            mae = min(mae, pnl_pct)
            mfe = max(mfe, pnl_pct)

            # Calculate leveraged P&L with fees (matches backtest logic)
            # Normalized capital = 1.0, position_value = 1.0
            # leveraged_value = leverage × position_value = leverage

            # Entry commission (as % of capital)
            entry_commission_pct = self.leverage * TAKER_FEE

            # Exit commission (as % of capital)
            # exit_notional = (leveraged_value / entry_price) × exit_price
            #               = leveraged_value × (exit_price / entry_price)
            #               = leveraged_value × (1 + pnl_pct)
            # exit_commission_pct = exit_notional × TAKER_FEE / position_value
            exit_commission_pct = self.leverage * (1 + pnl_pct) * TAKER_FEE

            # Total commission
            total_commission_pct = entry_commission_pct + exit_commission_pct

            # Net leveraged P&L (after fees)
            gross_leveraged_pnl_pct = pnl_pct * self.leverage
            leveraged_pnl_pct = gross_leveraged_pnl_pct - total_commission_pct

            # Check exit conditions

            # 1. ML Exit (PRIMARY)
            try:
                exit_features_values = df[self.exit_features].iloc[i:i+1].values
                exit_features_scaled = self.exit_scaler.transform(exit_features_values)
                exit_prob = self.exit_model.predict_proba(exit_features_scaled)[0][1]

                if exit_prob >= self.ml_exit_threshold:
                    return {
                        'pnl_pct': pnl_pct,
                        'leveraged_pnl_pct': leveraged_pnl_pct,
                        'mae': mae,
                        'mfe': mfe,
                        'hold_time': hold_time,
                        'exit_reason': 'ml_exit',
                        'exit_idx': i,
                        'exit_price': current_price
                    }
            except Exception as e:
                # If ML Exit fails, continue to emergency exits
                pass

            # 2. Emergency Stop Loss
            if leveraged_pnl_pct <= self.emergency_stop_loss:
                return {
                    'pnl_pct': pnl_pct,
                    'leveraged_pnl_pct': leveraged_pnl_pct,
                    'mae': mae,
                    'mfe': mfe,
                    'hold_time': hold_time,
                    'exit_reason': 'emergency_stop_loss',
                    'exit_idx': i,
                    'exit_price': current_price
                }

            # 3. Emergency Max Hold
            if hold_time >= self.emergency_max_hold:
                return {
                    'pnl_pct': pnl_pct,
                    'leveraged_pnl_pct': leveraged_pnl_pct,
                    'mae': mae,
                    'mfe': mfe,
                    'hold_time': hold_time,
                    'exit_reason': 'emergency_max_hold',
                    'exit_idx': i,
                    'exit_price': current_price
                }

        # Should not reach here (max hold should trigger)
        return {
            'pnl_pct': pnl_pct,
            'leveraged_pnl_pct': leveraged_pnl_pct,
            'mae': mae,
            'mfe': mfe,
            'hold_time': hold_time,
            'exit_reason': 'end_of_data',
            'exit_idx': len(df) - 1,
            'exit_price': current_price
        }

    def batch_simulate(self, df, entry_indices, side, show_progress=True):
        """
        Simulate multiple trades in batch

        Args:
            df: DataFrame with all features
            entry_indices: List of entry candle indices
            side: 'LONG' or 'SHORT'
            show_progress: Show progress bar

        Returns:
            list: Trade results for each entry index
        """
        results = []

        total = len(entry_indices)
        for idx, entry_idx in enumerate(entry_indices):
            if show_progress and idx % 1000 == 0:
                print(f"  Simulating trades: {idx}/{total} ({idx/total*100:.1f}%)")

            result = self.simulate_trade(df, entry_idx, side)
            results.append(result)

        if show_progress:
            print(f"  ✅ Simulated {total} trades")

        return results


def load_exit_models():
    """Load exit models for trade simulation"""
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    MODELS_DIR = PROJECT_ROOT / "models"

    print("Loading Exit models for simulation...")

    # LONG Exit
    long_exit_model_path = MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251017_151624.pkl"
    with open(long_exit_model_path, 'rb') as f:
        long_exit_model = pickle.load(f)

    long_exit_scaler_path = MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251017_151624_scaler.pkl"
    long_exit_scaler = joblib.load(long_exit_scaler_path)

    long_exit_features_path = MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251017_151624_features.txt"
    with open(long_exit_features_path, 'r') as f:
        long_exit_features = [line.strip() for line in f.readlines() if line.strip()]

    # SHORT Exit
    short_exit_model_path = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251017_152440.pkl"
    with open(short_exit_model_path, 'rb') as f:
        short_exit_model = pickle.load(f)

    short_exit_scaler_path = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251017_152440_scaler.pkl"
    short_exit_scaler = joblib.load(short_exit_scaler_path)

    short_exit_features_path = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251017_152440_features.txt"
    with open(short_exit_features_path, 'r') as f:
        short_exit_features = [line.strip() for line in f.readlines() if line.strip()]

    print(f"  ✅ LONG Exit: {len(long_exit_features)} features")
    print(f"  ✅ SHORT Exit: {len(short_exit_features)} features")

    return {
        'long': (long_exit_model, long_exit_scaler, long_exit_features),
        'short': (short_exit_model, short_exit_scaler, short_exit_features)
    }


if __name__ == "__main__":
    # Test trade simulator
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from scripts.experiments.calculate_all_features import calculate_all_features
    from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features

    print("="*80)
    print("Testing Trade Simulator")
    print("="*80)

    # Load data
    print("\nLoading data...")
    DATA_DIR = Path(__file__).parent.parent.parent / "data" / "historical"
    df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_max.csv")
    print(f"  Loaded {len(df):,} candles")

    # Calculate features
    print("\nCalculating features...")
    df = calculate_all_features(df)
    df = prepare_exit_features(df)
    print(f"  ✅ Features ready")

    # Load exit models
    exit_models = load_exit_models()

    # Create simulators
    long_simulator = TradeSimulator(
        exit_model=exit_models['long'][0],
        exit_scaler=exit_models['long'][1],
        exit_features=exit_models['long'][2]
    )

    short_simulator = TradeSimulator(
        exit_model=exit_models['short'][0],
        exit_scaler=exit_models['short'][1],
        exit_features=exit_models['short'][2]
    )

    # Test on a few random entries
    print("\nTesting LONG trades...")
    test_indices = [1000, 2000, 3000, 5000, 10000]

    for idx in test_indices:
        result = long_simulator.simulate_trade(df, idx, 'LONG')
        if result:
            print(f"  Entry {idx}: P&L {result['leveraged_pnl_pct']*100:+.2f}%, "
                  f"Hold {result['hold_time']} candles, Exit: {result['exit_reason']}")

    print("\nTesting SHORT trades...")
    for idx in test_indices:
        result = short_simulator.simulate_trade(df, idx, 'SHORT')
        if result:
            print(f"  Entry {idx}: P&L {result['leveraged_pnl_pct']*100:+.2f}%, "
                  f"Hold {result['hold_time']} candles, Exit: {result['exit_reason']}")

    print("\n✅ Trade Simulator test complete!")
