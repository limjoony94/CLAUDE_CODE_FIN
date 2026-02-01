"""
Improved Entry Labeling - 2-of-3 Scoring System
================================================

Inspired by Exit labeling success (improved precision from weak labeling)

Philosophy:
    Exit labeling의 2-of-3 scoring system이 성공했으므로
    Entry labeling에도 동일한 철학 적용

Current Problem:
    - LONG Entry Precision: 13.7% (Too many False Positives)
    - Peak/Trough labeling: Single criterion (too简单)
    - Result: 100 signals → only 13.7 profitable

Improved Approach:
    **2-of-3 Scoring System for LONG Entry**

    Criterion 1: **Profit Target** (0.4% within 4 hours)
        - 4시간 내 0.4% 이상 수익 가능한가?
        - 현실적인 수익 기대치

    Criterion 2: **Early Entry Advantage** (Lead-Time)
        - 지금 진입 vs 30분-2시간 후 진입
        - 일찍 진입해야 더 큰 수익

    Criterion 3: **Relative Performance** (Best Future Entry)
        - 현재 진입이 향후 최선의 진입점에 근접
        - Within 0.2% of best entry in next 4 hours

    Label = 1 if **2 or more criteria met**
    Label = 0 otherwise

Same for SHORT Entry (inverted logic)
"""

import numpy as np
import pandas as pd


class ImprovedEntryLabeling:
    """
    2-of-3 Scoring System for Entry Labels

    Similar to Exit labeling's success, but applied to Entry decisions
    """

    def __init__(
        self,
        profit_threshold=0.004,  # 0.4% profit target (realistic for 5min)
        lookforward_min=6,  # 30min (6 candles × 5min)
        lookforward_max=48,  # 4h (48 candles × 5min)
        lead_time_min=6,  # 30min
        lead_time_max=24,  # 2h
        relative_tolerance=0.002,  # 0.2% tolerance
        scoring_threshold=2  # 2 of 3
    ):
        """
        Args:
            profit_threshold: Minimum profit target (e.g., 0.004 = 0.4%)
            lookforward_min: Minimum future window (candles)
            lookforward_max: Maximum future window (candles)
            lead_time_min: Min lead time for early entry advantage
            lead_time_max: Max lead time for early entry advantage
            relative_tolerance: Tolerance for best entry comparison
            scoring_threshold: How many criteria must be met (default: 2 of 3)
        """
        self.profit_threshold = profit_threshold
        self.lookforward_min = lookforward_min
        self.lookforward_max = lookforward_max
        self.lead_time_min = lead_time_min
        self.lead_time_max = lead_time_max
        self.relative_tolerance = relative_tolerance
        self.scoring_threshold = scoring_threshold

    def create_long_entry_labels(self, df):
        """
        Create LONG Entry Labels using 2-of-3 scoring

        Returns:
            labels (np.array): 1 if good entry, 0 otherwise
        """
        print("\n" + "="*80)
        print("Creating LONG Entry Labels (2-of-3 Scoring System)")
        print("="*80)
        print(f"Profit Threshold: {self.profit_threshold*100:.2f}%")
        print(f"Lookforward: {self.lookforward_min}-{self.lookforward_max} candles "
              f"({self.lookforward_min*5/60:.1f}-{self.lookforward_max*5/60:.1f}h)")
        print(f"Lead Time: {self.lead_time_min}-{self.lead_time_max} candles "
              f"({self.lead_time_min*5/60:.1f}-{self.lead_time_max*5/60:.1f}h)")
        print(f"Relative Tolerance: {self.relative_tolerance*100:.2f}%")
        print(f"Scoring: {self.scoring_threshold} of 3 criteria\n")

        labels = []
        stats = {
            'criterion_1': 0,
            'criterion_2': 0,
            'criterion_3': 0,
            'score_0': 0,
            'score_1': 0,
            'score_2': 0,
            'score_3': 0
        }

        for i in range(len(df)):
            # Can't look forward enough
            if i >= len(df) - self.lookforward_max:
                labels.append(0)
                stats['score_0'] += 1
                continue

            entry_price = df['close'].iloc[i]
            score = 0

            # Future window for analysis
            future_start = i + self.lookforward_min
            future_end = min(i + self.lookforward_max + 1, len(df))
            future_prices = df['close'].iloc[future_start:future_end]

            if len(future_prices) == 0:
                labels.append(0)
                stats['score_0'] += 1
                continue

            # ===================================================================
            # Criterion 1: Profit Target (0.4% within 4 hours)
            # ===================================================================
            max_future_price = future_prices.max()
            max_profit = (max_future_price - entry_price) / entry_price

            criterion_1_met = max_profit >= self.profit_threshold
            if criterion_1_met:
                score += 1
                stats['criterion_1'] += 1

            # ===================================================================
            # Criterion 2: Early Entry Advantage (Lead-Time)
            # ===================================================================
            # Compare: enter now vs enter 30min-2h later
            lead_time_start = i + self.lead_time_min
            lead_time_end = min(i + self.lead_time_max + 1, len(df))

            if lead_time_end <= lead_time_start:
                criterion_2_met = False
            else:
                # Find best delayed entry
                lead_time_prices = df['close'].iloc[lead_time_start:lead_time_end]
                best_delayed_entry = lead_time_prices.min()  # Best for LONG = lowest price

                # Profit from entering now
                profit_now = (max_future_price - entry_price) / entry_price

                # Profit from delayed entry
                profit_delayed = (max_future_price - best_delayed_entry) / best_delayed_entry

                # Early entry should be significantly better (or at least not worse)
                criterion_2_met = profit_now >= profit_delayed * (1 - self.relative_tolerance)

            if criterion_2_met:
                score += 1
                stats['criterion_2'] += 1

            # ===================================================================
            # Criterion 3: Relative Performance (Best Future Entry)
            # ===================================================================
            # Is current entry close to the best possible entry in next 4h?
            all_future_prices = df['close'].iloc[i:future_end]
            best_entry_price = all_future_prices.min()  # Best for LONG = lowest

            # How far is current price from best entry?
            price_distance = abs(entry_price - best_entry_price) / best_entry_price

            criterion_3_met = price_distance <= self.relative_tolerance

            if criterion_3_met:
                score += 1
                stats['criterion_3'] += 1

            # ===================================================================
            # Final Label: 2 of 3 criteria
            # ===================================================================
            if score >= self.scoring_threshold:
                labels.append(1)
            else:
                labels.append(0)

            # Track score distribution
            stats[f'score_{score}'] += 1

        labels = np.array(labels)

        # Print statistics
        positive_count = np.sum(labels == 1)
        positive_rate = positive_count / len(labels) * 100

        print(f"Label Statistics:")
        print(f"  Total candles: {len(labels):,}")
        print(f"\n  Criterion Met Rates:")
        print(f"    1. Profit Target: {stats['criterion_1']:,} ({stats['criterion_1']/len(labels)*100:.1f}%)")
        print(f"    2. Early Entry:   {stats['criterion_2']:,} ({stats['criterion_2']/len(labels)*100:.1f}%)")
        print(f"    3. Best Entry:    {stats['criterion_3']:,} ({stats['criterion_3']/len(labels)*100:.1f}%)")
        print(f"\n  Score Distribution:")
        print(f"    Score 0: {stats['score_0']:,} ({stats['score_0']/len(labels)*100:.1f}%)")
        print(f"    Score 1: {stats['score_1']:,} ({stats['score_1']/len(labels)*100:.1f}%)")
        print(f"    Score 2: {stats['score_2']:,} ({stats['score_2']/len(labels)*100:.1f}%)")
        print(f"    Score 3: {stats['score_3']:,} ({stats['score_3']/len(labels)*100:.1f}%)")
        print(f"\n✅ Positive Labels (Score >= {self.scoring_threshold}): {positive_count:,} ({positive_rate:.2f}%)")

        return labels

    def create_short_entry_labels(self, df):
        """
        Create SHORT Entry Labels using 2-of-3 scoring

        Same philosophy as LONG, but inverted logic
        """
        print("\n" + "="*80)
        print("Creating SHORT Entry Labels (2-of-3 Scoring System)")
        print("="*80)
        print(f"Profit Threshold: {self.profit_threshold*100:.2f}%")
        print(f"Lookforward: {self.lookforward_min}-{self.lookforward_max} candles")
        print(f"Lead Time: {self.lead_time_min}-{self.lead_time_max} candles")
        print(f"Relative Tolerance: {self.relative_tolerance*100:.2f}%")
        print(f"Scoring: {self.scoring_threshold} of 3 criteria\n")

        labels = []
        stats = {
            'criterion_1': 0,
            'criterion_2': 0,
            'criterion_3': 0,
            'score_0': 0,
            'score_1': 0,
            'score_2': 0,
            'score_3': 0
        }

        for i in range(len(df)):
            if i >= len(df) - self.lookforward_max:
                labels.append(0)
                stats['score_0'] += 1
                continue

            entry_price = df['close'].iloc[i]
            score = 0

            future_start = i + self.lookforward_min
            future_end = min(i + self.lookforward_max + 1, len(df))
            future_prices = df['close'].iloc[future_start:future_end]

            if len(future_prices) == 0:
                labels.append(0)
                stats['score_0'] += 1
                continue

            # ===================================================================
            # Criterion 1: Profit Target (SHORT: price drops 0.4%)
            # ===================================================================
            min_future_price = future_prices.min()
            max_profit = (entry_price - min_future_price) / entry_price

            criterion_1_met = max_profit >= self.profit_threshold
            if criterion_1_met:
                score += 1
                stats['criterion_1'] += 1

            # ===================================================================
            # Criterion 2: Early Entry Advantage
            # ===================================================================
            lead_time_start = i + self.lead_time_min
            lead_time_end = min(i + self.lead_time_max + 1, len(df))

            if lead_time_end <= lead_time_start:
                criterion_2_met = False
            else:
                lead_time_prices = df['close'].iloc[lead_time_start:lead_time_end]
                best_delayed_entry = lead_time_prices.max()  # Best for SHORT = highest price

                profit_now = (entry_price - min_future_price) / entry_price
                profit_delayed = (best_delayed_entry - min_future_price) / best_delayed_entry

                criterion_2_met = profit_now >= profit_delayed * (1 - self.relative_tolerance)

            if criterion_2_met:
                score += 1
                stats['criterion_2'] += 1

            # ===================================================================
            # Criterion 3: Best Entry (SHORT: highest price)
            # ===================================================================
            all_future_prices = df['close'].iloc[i:future_end]
            best_entry_price = all_future_prices.max()  # Best for SHORT = highest

            price_distance = abs(entry_price - best_entry_price) / best_entry_price

            criterion_3_met = price_distance <= self.relative_tolerance

            if criterion_3_met:
                score += 1
                stats['criterion_3'] += 1

            # Final Label
            if score >= self.scoring_threshold:
                labels.append(1)
            else:
                labels.append(0)

            stats[f'score_{score}'] += 1

        labels = np.array(labels)

        positive_count = np.sum(labels == 1)
        positive_rate = positive_count / len(labels) * 100

        print(f"Label Statistics:")
        print(f"  Total candles: {len(labels):,}")
        print(f"\n  Criterion Met Rates:")
        print(f"    1. Profit Target: {stats['criterion_1']:,} ({stats['criterion_1']/len(labels)*100:.1f}%)")
        print(f"    2. Early Entry:   {stats['criterion_2']:,} ({stats['criterion_2']/len(labels)*100:.1f}%)")
        print(f"    3. Best Entry:    {stats['criterion_3']:,} ({stats['criterion_3']/len(labels)*100:.1f}%)")
        print(f"\n  Score Distribution:")
        print(f"    Score 0: {stats['score_0']:,} ({stats['score_0']/len(labels)*100:.1f}%)")
        print(f"    Score 1: {stats['score_1']:,} ({stats['score_1']/len(labels)*100:.1f}%)")
        print(f"    Score 2: {stats['score_2']:,} ({stats['score_2']/len(labels)*100:.1f}%)")
        print(f"    Score 3: {stats['score_3']:,} ({stats['score_3']/len(labels)*100:.1f}%)")
        print(f"\n✅ Positive Labels (Score >= {self.scoring_threshold}): {positive_count:,} ({positive_rate:.2f}%)")

        return labels


def test_labeling():
    """Test improved entry labeling"""
    print("="*80)
    print("Testing Improved Entry Labeling (2-of-3 Scoring)")
    print("="*80)

    # Create synthetic data
    np.random.seed(42)
    n = 500

    # Trend + noise
    trend = np.linspace(100, 120, n)
    noise = np.random.normal(0, 2, n)
    price = trend + noise

    df = pd.DataFrame({
        'open': price,
        'high': price + abs(np.random.normal(0, 0.5, n)),
        'low': price - abs(np.random.normal(0, 0.5, n)),
        'close': price,
        'volume': np.random.normal(1000, 100, n)
    })

    # Test LONG Entry labeling
    labeler = ImprovedEntryLabeling(
        profit_threshold=0.004,  # 0.4%
        lookforward_min=6,  # 30min
        lookforward_max=48,  # 4h
        lead_time_min=6,
        lead_time_max=24,
        relative_tolerance=0.002,  # 0.2%
        scoring_threshold=2  # 2 of 3
    )

    long_labels = labeler.create_long_entry_labels(df)
    print(f"\n📊 LONG Entry Labels: {np.sum(long_labels):,} positive ({np.sum(long_labels)/len(long_labels)*100:.1f}%)")

    short_labels = labeler.create_short_entry_labels(df)
    print(f"\n📊 SHORT Entry Labels: {np.sum(short_labels):,} positive ({np.sum(short_labels)/len(short_labels)*100:.1f}%)")

    print("\n✅ Improved Entry Labeling test complete!")


if __name__ == "__main__":
    test_labeling()
