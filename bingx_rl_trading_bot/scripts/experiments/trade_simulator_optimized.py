"""
Optimized Trade Simulator for Entry Labeling
=============================================

Vectorized and parallelized trade simulation for faster processing.
"""

import numpy as np
import pandas as pd
import pickle
import joblib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from functools import partial

class OptimizedTradeSimulator:
    """
    Optimized trade simulator using vectorization and parallelization
    """

    def __init__(
        self,
        exit_model,
        exit_scaler,
        exit_features,
        leverage=4,
        ml_exit_threshold=0.70,
        emergency_stop_loss=-0.04,
        emergency_max_hold=96
    ):
        self.exit_model = exit_model
        self.exit_scaler = exit_scaler
        self.exit_features = exit_features
        self.leverage = leverage
        self.ml_exit_threshold = ml_exit_threshold
        self.emergency_stop_loss = emergency_stop_loss
        self.emergency_max_hold = emergency_max_hold

    def simulate_trade_batch(self, df, entry_indices, side):
        """
        Simulate multiple trades in batch for better performance

        Args:
            df: DataFrame with all features
            entry_indices: List of entry indices to simulate
            side: 'LONG' or 'SHORT'

        Returns:
            List of trade outcome dicts
        """
        results = []

        for entry_idx in entry_indices:
            result = self._simulate_single_trade(df, entry_idx, side)
            result['entry_idx'] = entry_idx  # Add entry_idx to result
            results.append(result)

        return results

    def _simulate_single_trade(self, df, entry_idx, side):
        """Single trade simulation (same logic as original)"""
        entry_price = df['close'].iloc[entry_idx]
        mae = 0.0
        mfe = 0.0

        # Calculate max possible holding period
        max_hold_end = min(entry_idx + self.emergency_max_hold + 1, len(df))

        # Vectorized P&L calculation for entire potential holding period
        prices = df['close'].iloc[entry_idx+1:max_hold_end].values

        if side == 'LONG':
            pnl_series = (prices - entry_price) / entry_price
        else:  # SHORT
            pnl_series = (entry_price - prices) / entry_price

        # Calculate MAE/MFE vectorized
        cummin = np.minimum.accumulate(pnl_series)
        cummax = np.maximum.accumulate(pnl_series)

        # Check exit conditions sequentially (can't fully vectorize due to dependencies)
        for i, pnl_pct in enumerate(pnl_series):
            current_idx = entry_idx + 1 + i
            hold_time = i + 1

            mae = cummin[i]
            mfe = cummax[i]
            leveraged_pnl = pnl_pct * self.leverage

            # Exit condition 1: Emergency Stop Loss
            if leveraged_pnl <= self.emergency_stop_loss:
                return {
                    'pnl_pct': pnl_pct,
                    'leveraged_pnl_pct': leveraged_pnl,
                    'mae': mae,
                    'mfe': mfe,
                    'hold_time': hold_time,
                    'exit_reason': 'emergency_stop_loss'
                }

            # Exit condition 2: ML Exit
            try:
                exit_feat = df[self.exit_features].iloc[current_idx:current_idx+1].values
                exit_scaled = self.exit_scaler.transform(exit_feat)
                exit_prob = self.exit_model.predict_proba(exit_scaled)[0][1]

                if exit_prob >= self.ml_exit_threshold:
                    return {
                        'pnl_pct': pnl_pct,
                        'leveraged_pnl_pct': leveraged_pnl,
                        'mae': mae,
                        'mfe': mfe,
                        'hold_time': hold_time,
                        'exit_reason': 'ml_exit'
                    }
            except:
                pass

            # Exit condition 3: Max Hold
            if hold_time >= self.emergency_max_hold:
                return {
                    'pnl_pct': pnl_pct,
                    'leveraged_pnl_pct': leveraged_pnl,
                    'mae': mae,
                    'mfe': mfe,
                    'hold_time': hold_time,
                    'exit_reason': 'emergency_max_hold'
                }

        # If we get here, exit at max_hold_end
        final_pnl = pnl_series[-1] if len(pnl_series) > 0 else 0
        return {
            'pnl_pct': final_pnl,
            'leveraged_pnl_pct': final_pnl * self.leverage,
            'mae': cummin[-1] if len(cummin) > 0 else 0,
            'mfe': cummax[-1] if len(cummax) > 0 else 0,
            'hold_time': len(pnl_series),
            'exit_reason': 'data_end'
        }

    def simulate_all_parallel(self, df, side, n_workers=4):
        """
        Simulate all possible entries in parallel

        Args:
            df: DataFrame with all features
            side: 'LONG' or 'SHORT'
            n_workers: Number of parallel workers

        Returns:
            List of trade outcome dicts
        """
        # Generate all valid entry indices
        valid_indices = list(range(100, len(df) - self.emergency_max_hold))

        if len(valid_indices) == 0:
            return []

        # Split indices into chunks for parallel processing
        chunk_size = len(valid_indices) // n_workers
        if chunk_size == 0:
            chunk_size = len(valid_indices)

        chunks = [valid_indices[i:i+chunk_size] for i in range(0, len(valid_indices), chunk_size)]

        # Process in parallel
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            simulate_func = partial(self._simulate_chunk, df=df, side=side)
            chunk_results = list(executor.map(simulate_func, chunks))

        # Flatten results
        all_results = []
        for chunk_result in chunk_results:
            all_results.extend(chunk_result)

        return all_results

    def _simulate_chunk(self, indices, df, side):
        """Simulate a chunk of trades (for parallel processing)"""
        return self.simulate_trade_batch(df, indices, side)


def simulate_trades_optimized(df, exit_model, exit_scaler, exit_features,
                              side, ml_exit_threshold, n_workers=4):
    """
    Convenience function for optimized trade simulation

    Args:
        df: DataFrame with features
        exit_model: Trained exit model
        exit_scaler: Feature scaler
        exit_features: List of feature names
        side: 'LONG' or 'SHORT'
        ml_exit_threshold: ML exit probability threshold
        n_workers: Number of parallel workers

    Returns:
        List of trade outcomes
    """
    simulator = OptimizedTradeSimulator(
        exit_model=exit_model,
        exit_scaler=exit_scaler,
        exit_features=exit_features,
        leverage=4,
        ml_exit_threshold=ml_exit_threshold,
        emergency_stop_loss=-0.04,
        emergency_max_hold=96
    )

    # Try parallel first, fall back to batch if it fails
    try:
        results = simulator.simulate_all_parallel(df, side, n_workers=n_workers)
    except Exception as e:
        print(f"⚠️ Parallel processing failed: {e}")
        print(f"   Falling back to batch processing...")
        valid_indices = list(range(100, len(df) - 96))
        results = simulator.simulate_trade_batch(df, valid_indices, side)

    return results
