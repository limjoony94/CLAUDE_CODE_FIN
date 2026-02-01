"""
Peak/Trough Detection Labeling

Inspired by LONG Exit model's success (35.2% precision):
  Labeling: "near_peak_80pct_and_beats_holding_1h"

Concept:
  - TP/SL labeling: 고정된 % 목표 (시장 구조 무시)
  - Peak/Trough labeling: 실제 시장 구조 활용 (고점/저점)

Applications:
  - SHORT Entry: "저점 80% 근처 진입" + "홀딩보다 좋음"
  - LONG Exit: "고점 80% 근처 청산" + "홀딩보다 좋음"
"""

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema


class PeakTroughLabeling:
    """
    Peak/Trough 기반 Labeling

    Philosophy: 시장은 peak와 trough를 만든다
               최적 진입/청산은 trough/peak 근처
    """

    def __init__(self, lookforward=48, peak_window=10, near_threshold=0.80, holding_hours=1):
        """
        Args:
            lookforward: 미래 peak/trough 탐색 기간 (candles)
            peak_window: Peak/trough detection window
            near_threshold: "근처" 판단 기준 (0.80 = 80%)
            holding_hours: 비교 대상 홀딩 기간 (hours)
        """
        self.lookforward = lookforward
        self.peak_window = peak_window
        self.near_threshold = near_threshold
        self.holding_candles = int(holding_hours * 12)  # 5분 candle

    def create_short_entry_labels(self, df):
        """
        SHORT Entry Labels: "저점 80% 근처 진입" + "홀딩보다 좋음"

        Label 1:
          - 현재 가격이 미래 저점(trough)의 80% 이하
          - 지금 SHORT 진입 > 1시간 후 SHORT 진입

        Label 0: Otherwise
        """
        print("\n" + "="*80)
        print("Creating SHORT Entry Labels (Peak/Trough Method)")
        print("="*80)
        print(f"Lookforward: {self.lookforward} candles ({self.lookforward*5/60:.1f} hours)")
        print(f"Peak window: {self.peak_window}")
        print(f"Near threshold: {self.near_threshold*100:.0f}%")
        print(f"Holding comparison: {self.holding_candles} candles ({self.holding_candles*5/60:.1f} hour)")

        labels = []
        trough_info = []

        for i in range(len(df)):
            if i >= len(df) - self.lookforward:
                labels.append(0)
                trough_info.append({'has_trough': False})
                continue

            # Find future trough (low point) within lookforward
            future_window = df.iloc[i:i+self.lookforward]
            future_lows = future_window['low'].values

            # Detect local minima
            minima_indices = argrelextrema(future_lows, np.less_equal, order=self.peak_window)[0]

            if len(minima_indices) == 0:
                # No trough found
                labels.append(0)
                trough_info.append({'has_trough': False})
                continue

            # Get lowest trough
            lowest_trough_idx = minima_indices[np.argmin(future_lows[minima_indices])]
            trough_price = future_lows[lowest_trough_idx]
            trough_candle = i + lowest_trough_idx

            current_price = df['close'].iloc[i]

            # Check if current price is near trough (within threshold)
            # For SHORT: current price should be ABOVE trough (we sell high, it goes low)
            # But "near" trough means we're catching the beginning of drop
            near_trough_upper = trough_price / self.near_threshold  # e.g., trough $100 → upper bound $125 (100/0.8)

            is_near_trough = current_price <= near_trough_upper

            if not is_near_trough:
                labels.append(0)
                trough_info.append({
                    'has_trough': True,
                    'near_trough': False,
                    'trough_price': trough_price
                })
                continue

            # Calculate profit if enter SHORT now vs wait
            # SHORT profit = (entry - exit) / entry
            short_now_profit = (current_price - trough_price) / current_price

            # If wait holding_candles, then enter
            if i + self.holding_candles < len(df):
                delayed_entry_price = df['close'].iloc[i + self.holding_candles]
                short_delayed_profit = (delayed_entry_price - trough_price) / delayed_entry_price
            else:
                short_delayed_profit = 0

            # Label 1 if entering now is better than waiting
            beats_holding = short_now_profit > short_delayed_profit

            labels.append(1 if beats_holding else 0)
            trough_info.append({
                'has_trough': True,
                'near_trough': True,
                'beats_holding': beats_holding,
                'trough_price': trough_price,
                'now_profit': short_now_profit,
                'delayed_profit': short_delayed_profit
            })

        labels = np.array(labels)

        # Analysis
        positive_count = np.sum(labels == 1)
        positive_rate = positive_count / len(labels) * 100

        has_trough_count = sum(1 for info in trough_info if info.get('has_trough', False))
        near_trough_count = sum(1 for info in trough_info if info.get('near_trough', False))

        print(f"\nLabel Statistics:")
        print(f"  Total candles: {len(labels):,}")
        print(f"  Has trough: {has_trough_count:,} ({has_trough_count/len(labels)*100:.1f}%)")
        print(f"  Near trough: {near_trough_count:,} ({near_trough_count/len(labels)*100:.1f}%)")
        print(f"  Beats holding: {positive_count:,} ({positive_rate:.2f}%)")
        print(f"\n✅ Positive rate: {positive_rate:.2f}%")

        return labels

    def create_long_exit_labels(self, df):
        """
        LONG Exit Labels: "고점 80% 근처 청산" + "홀딩보다 좋음"

        Label 1:
          - 현재 가격이 미래 고점(peak)의 80% 이상
          - 지금 청산 > 1시간 더 홀딩

        Label 0: Otherwise
        """
        print("\n" + "="*80)
        print("Creating LONG Exit Labels (Peak/Trough Method)")
        print("="*80)
        print(f"Lookforward: {self.lookforward} candles ({self.lookforward*5/60:.1f} hours)")
        print(f"Peak window: {self.peak_window}")
        print(f"Near threshold: {self.near_threshold*100:.0f}%")
        print(f"Holding comparison: {self.holding_candles} candles ({self.holding_candles*5/60:.1f} hour)")

        labels = []
        peak_info = []

        for i in range(len(df)):
            if i >= len(df) - self.lookforward:
                labels.append(0)
                peak_info.append({'has_peak': False})
                continue

            # Find future peak (high point) within lookforward
            future_window = df.iloc[i:i+self.lookforward]
            future_highs = future_window['high'].values

            # Detect local maxima
            maxima_indices = argrelextrema(future_highs, np.greater_equal, order=self.peak_window)[0]

            if len(maxima_indices) == 0:
                labels.append(0)
                peak_info.append({'has_peak': False})
                continue

            # Get highest peak
            highest_peak_idx = maxima_indices[np.argmax(future_highs[maxima_indices])]
            peak_price = future_highs[highest_peak_idx]
            peak_candle = i + highest_peak_idx

            current_price = df['close'].iloc[i]

            # Check if current price is near peak (within threshold)
            near_peak_lower = peak_price * self.near_threshold  # e.g., peak $100 → lower bound $80

            is_near_peak = current_price >= near_peak_lower

            if not is_near_peak:
                labels.append(0)
                peak_info.append({
                    'has_peak': True,
                    'near_peak': False,
                    'peak_price': peak_price
                })
                continue

            # Calculate profit if exit now vs hold longer
            # Assume entered at some earlier price (use simplified model)
            # For labeling, we compare: exit now at current_price vs exit at peak
            exit_now_profit = current_price  # Simplified
            exit_at_peak_profit = peak_price

            # If hold for holding_candles more
            if i + self.holding_candles < len(df):
                exit_delayed_profit = df['close'].iloc[i + self.holding_candles]
            else:
                exit_delayed_profit = current_price

            # Label 1 if exiting now captures more of the peak than waiting
            # "Near peak now" is better than "maybe past peak later"
            beats_holding = current_price > exit_delayed_profit

            labels.append(1 if beats_holding else 0)
            peak_info.append({
                'has_peak': True,
                'near_peak': True,
                'beats_holding': beats_holding,
                'peak_price': peak_price,
                'now_profit': current_price,
                'delayed_profit': exit_delayed_profit
            })

        labels = np.array(labels)

        # Analysis
        positive_count = np.sum(labels == 1)
        positive_rate = positive_count / len(labels) * 100

        has_peak_count = sum(1 for info in peak_info if info.get('has_peak', False))
        near_peak_count = sum(1 for info in peak_info if info.get('near_peak', False))

        print(f"\nLabel Statistics:")
        print(f"  Total candles: {len(labels):,}")
        print(f"  Has peak: {has_peak_count:,} ({has_peak_count/len(labels)*100:.1f}%)")
        print(f"  Near peak: {near_peak_count:,} ({near_peak_count/len(labels)*100:.1f}%)")
        print(f"  Beats holding: {positive_count:,} ({positive_rate:.2f}%)")
        print(f"\n✅ Positive rate: {positive_rate:.2f}%")

        return labels


def test_labeling():
    """Test peak/trough labeling"""
    print("="*80)
    print("Testing Peak/Trough Labeling")
    print("="*80)

    # Create synthetic data with clear peaks and troughs
    np.random.seed(42)

    # Create wave pattern
    x = np.linspace(0, 4*np.pi, 200)
    price = 100 + 10 * np.sin(x) + np.random.normal(0, 1, 200)

    df = pd.DataFrame({
        'open': price,
        'high': price + abs(np.random.normal(0, 0.5, 200)),
        'low': price - abs(np.random.normal(0, 0.5, 200)),
        'close': price,
        'volume': np.random.normal(1000, 100, 200)
    })

    # Test SHORT Entry labeling
    labeler = PeakTroughLabeling(
        lookforward=48,
        peak_window=5,
        near_threshold=0.80,
        holding_hours=1
    )

    short_labels = labeler.create_short_entry_labels(df)

    print(f"\n📊 SHORT Entry Labels:")
    print(f"  Positive: {np.sum(short_labels):,} ({np.sum(short_labels)/len(short_labels)*100:.1f}%)")

    # Test LONG Exit labeling
    long_exit_labels = labeler.create_long_exit_labels(df)

    print(f"\n📊 LONG Exit Labels:")
    print(f"  Positive: {np.sum(long_exit_labels):,} ({np.sum(long_exit_labels)/len(long_exit_labels)*100:.1f}%)")

    print("\n✅ Labeling test complete!")


if __name__ == "__main__":
    test_labeling()
