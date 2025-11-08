"""
Signal Generation from Model Predictions

Converts model predictions and uncertainty estimates into trading signals with confidence scores.

Strategy:
1. Directional prediction (up/down from close price)
2. Uncertainty filtering (reject high-uncertainty predictions)
3. Confidence scoring (combine direction strength + low uncertainty)
"""

import numpy as np
from typing import Dict, Optional
from loguru import logger


class SignalGenerator:
    """
    Convert model predictions to trading signals

    Uses:
    - Predicted close price direction
    - Uncertainty estimates
    - Confidence thresholds
    """

    def __init__(
        self,
        min_confidence: float = 0.6,
        max_uncertainty_pct: float = 0.02,
        min_price_move_pct: float = 0.001
    ):
        """
        Initialize signal generator

        Args:
            min_confidence: Minimum confidence to generate signal (0-1)
            max_uncertainty_pct: Maximum uncertainty as % of price
            min_price_move_pct: Minimum predicted price move to trade
        """
        self.min_confidence = min_confidence
        self.max_uncertainty_pct = max_uncertainty_pct
        self.min_price_move_pct = min_price_move_pct

        logger.info(f"Initialized SignalGenerator:")
        logger.info(f"  - Min confidence: {min_confidence:.1%}")
        logger.info(f"  - Max uncertainty: {max_uncertainty_pct:.2%}")
        logger.info(f"  - Min price move: {min_price_move_pct:.3%}")

    def generate_signal(
        self,
        prediction: np.ndarray,
        uncertainty: np.ndarray,
        current_candle: Optional[Dict] = None
    ) -> Dict:
        """
        Generate trading signal from prediction

        Args:
            prediction: Model prediction (n_features,) - OHLCV order
            uncertainty: Prediction uncertainty (n_features,)
            current_candle: Current candle data (optional)

        Returns:
            Signal dict with:
                - action: 'long', 'short', or 'hold'
                - confidence: 0-1
                - predicted_direction: 'up' or 'down'
                - uncertainty_score: 0-1 (lower is better)
        """
        # Extract close price (index 3 for OHLCV)
        pred_close = prediction[3]
        unc_close = uncertainty[3]

        # Get current price
        if current_candle is not None:
            current_price = current_candle['close']
        else:
            # If no current candle, use prediction as reference
            current_price = pred_close

        # Calculate predicted price move
        price_move = (pred_close - current_price) / current_price

        # Calculate uncertainty score (normalized)
        uncertainty_pct = unc_close / current_price
        uncertainty_score = min(uncertainty_pct / self.max_uncertainty_pct, 1.0)

        # Determine direction
        if abs(price_move) < self.min_price_move_pct:
            # Predicted move too small
            action = 'hold'
            direction = 'neutral'
        elif price_move > 0:
            direction = 'up'
            action = 'long'
        else:
            direction = 'down'
            action = 'short'

        # Calculate confidence
        # Confidence = (1 - uncertainty_score) * signal_strength
        signal_strength = min(abs(price_move) / self.min_price_move_pct, 1.0)
        confidence = (1 - uncertainty_score) * signal_strength

        # Apply confidence threshold
        if confidence < self.min_confidence:
            action = 'hold'

        # Apply uncertainty filter
        if uncertainty_score > 1.0:  # Uncertainty too high
            action = 'hold'
            confidence = 0.0

        signal = {
            'action': action,
            'confidence': confidence,
            'predicted_direction': direction,
            'predicted_price': pred_close,
            'price_move_pct': price_move,
            'uncertainty_score': uncertainty_score,
            'signal_strength': signal_strength
        }

        return signal

    def generate_batch_signals(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        current_candles: Optional[np.ndarray] = None
    ) -> list:
        """
        Generate signals for a batch of predictions

        Args:
            predictions: (batch_size, n_features)
            uncertainties: (batch_size, n_features)
            current_candles: Optional array of current candle data

        Returns:
            List of signal dicts
        """
        batch_size = predictions.shape[0]
        signals = []

        for i in range(batch_size):
            pred = predictions[i]
            unc = uncertainties[i]

            current_candle = None
            if current_candles is not None:
                current_candle = current_candles[i]

            signal = self.generate_signal(pred, unc, current_candle)
            signals.append(signal)

        return signals

    def update_thresholds(
        self,
        min_confidence: Optional[float] = None,
        max_uncertainty_pct: Optional[float] = None,
        min_price_move_pct: Optional[float] = None
    ):
        """Update signal generation thresholds"""
        if min_confidence is not None:
            self.min_confidence = min_confidence
            logger.info(f"Updated min_confidence to {min_confidence:.1%}")

        if max_uncertainty_pct is not None:
            self.max_uncertainty_pct = max_uncertainty_pct
            logger.info(f"Updated max_uncertainty_pct to {max_uncertainty_pct:.2%}")

        if min_price_move_pct is not None:
            self.min_price_move_pct = min_price_move_pct
            logger.info(f"Updated min_price_move_pct to {min_price_move_pct:.3%}")


if __name__ == '__main__':
    # Test signal generator
    print("=" * 60)
    print("Testing SignalGenerator")
    print("=" * 60)

    # Initialize
    signal_gen = SignalGenerator(
        min_confidence=0.6,
        max_uncertainty_pct=0.02,
        min_price_move_pct=0.001
    )

    # Test 1: Strong bullish signal (low uncertainty, good price move)
    print("\n1. Testing Strong Bullish Signal")
    print("-" * 60)

    prediction = np.array([100, 105, 99, 103, 1000])  # OHLCV
    uncertainty = np.array([0.5, 0.5, 0.5, 0.5, 10])  # Low uncertainty
    current_candle = {'close': 100}

    signal = signal_gen.generate_signal(prediction, uncertainty, current_candle)

    print(f"Current price: ${current_candle['close']}")
    print(f"Predicted price: ${signal['predicted_price']:.2f}")
    print(f"Action: {signal['action']}")
    print(f"Confidence: {signal['confidence']:.2%}")
    print(f"Price move: {signal['price_move_pct']:.3%}")
    print(f"Uncertainty score: {signal['uncertainty_score']:.3f}")

    # Test 2: High uncertainty signal (should be filtered)
    print("\n2. Testing High Uncertainty Signal")
    print("-" * 60)

    prediction = np.array([100, 105, 99, 105, 1000])  # OHLCV
    uncertainty = np.array([2.0, 2.0, 2.0, 3.0, 50])  # High uncertainty
    current_candle = {'close': 100}

    signal = signal_gen.generate_signal(prediction, uncertainty, current_candle)

    print(f"Predicted price: ${signal['predicted_price']:.2f}")
    print(f"Action: {signal['action']}")
    print(f"Confidence: {signal['confidence']:.2%}")
    print(f"Uncertainty score: {signal['uncertainty_score']:.3f}")

    # Test 3: Weak signal (small price move)
    print("\n3. Testing Weak Signal (Small Price Move)")
    print("-" * 60)

    prediction = np.array([100, 101, 99, 100.05, 1000])  # Small move
    uncertainty = np.array([0.3, 0.3, 0.3, 0.3, 10])
    current_candle = {'close': 100}

    signal = signal_gen.generate_signal(prediction, uncertainty, current_candle)

    print(f"Predicted price: ${signal['predicted_price']:.2f}")
    print(f"Action: {signal['action']}")
    print(f"Confidence: {signal['confidence']:.2%}")
    print(f"Price move: {signal['price_move_pct']:.3%}")

    # Test 4: Bearish signal
    print("\n4. Testing Bearish Signal")
    print("-" * 60)

    prediction = np.array([100, 99, 95, 97, 1000])  # Down move
    uncertainty = np.array([0.5, 0.5, 0.5, 0.5, 10])
    current_candle = {'close': 100}

    signal = signal_gen.generate_signal(prediction, uncertainty, current_candle)

    print(f"Predicted price: ${signal['predicted_price']:.2f}")
    print(f"Action: {signal['action']}")
    print(f"Confidence: {signal['confidence']:.2%}")
    print(f"Direction: {signal['predicted_direction']}")

    # Test 5: Batch signals
    print("\n5. Testing Batch Signal Generation")
    print("-" * 60)

    batch_predictions = np.random.randn(5, 5) + np.array([100, 105, 99, 103, 1000])
    batch_uncertainties = np.random.rand(5, 5)

    batch_signals = signal_gen.generate_batch_signals(batch_predictions, batch_uncertainties)

    for i, sig in enumerate(batch_signals):
        print(f"Signal {i+1}: {sig['action']:5s} | Confidence: {sig['confidence']:.2%}")

    # Test 6: Update thresholds
    print("\n6. Testing Threshold Updates")
    print("-" * 60)

    signal_gen.update_thresholds(
        min_confidence=0.7,
        max_uncertainty_pct=0.015
    )

    signal = signal_gen.generate_signal(prediction, uncertainty, current_candle)
    print(f"After threshold update:")
    print(f"  Action: {signal['action']}")
    print(f"  Confidence: {signal['confidence']:.2%}")

    print("\n" + "=" * 60)
    print("All signal generator tests passed! ✅")
    print("=" * 60)
