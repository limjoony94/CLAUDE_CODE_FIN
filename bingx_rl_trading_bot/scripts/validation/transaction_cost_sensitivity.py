"""
Transaction Cost Sensitivity Analysis

비판적 사고 검증: 실제 거래 비용이 높을 경우 전략 수익성 평가

목적:
    - 현재 가정 (0.02%) vs 실제 가능한 비용 (0.05-0.15%) 영향 분석
    - Slippage, maker/taker fee 변동에 따른 민감도 평가
    - 손익분기점 비용 계산

방법:
    - 다양한 거래 비용 시나리오 테스트
    - 0.01%, 0.02%, 0.05%, 0.08%, 0.10%, 0.15%
    - 각 시나리오별 수익률 계산
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures

# Directories
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "historical"
RESULTS_DIR = PROJECT_ROOT / "results"

# Configuration (90/10 Allocation)
INITIAL_CAPITAL = 10000.0
LONG_ALLOCATION = 0.90
SHORT_ALLOCATION = 0.10

# LONG Configuration
LONG_THRESHOLD = 0.7
LONG_STOP_LOSS = 0.01
LONG_TAKE_PROFIT = 0.03
LONG_MAX_HOLDING_HOURS = 4
LONG_POSITION_SIZE_PCT = 0.95

# SHORT Configuration
SHORT_THRESHOLD = 0.4
SHORT_STOP_LOSS = 0.015
SHORT_TAKE_PROFIT = 0.06
SHORT_MAX_HOLDING_HOURS = 4
SHORT_POSITION_SIZE_PCT = 0.95

CHECK_INTERVAL_MINUTES = 5


class CostSensitivityAnalyzer:
    """Analyze strategy sensitivity to transaction costs"""

    def __init__(self):
        # Load models
        long_model_file = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl"
        with open(long_model_file, 'rb') as f:
            self.long_model = pickle.load(f)

        long_feature_file = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt"
        with open(long_feature_file, 'r') as f:
            self.long_features = [line.strip() for line in f.readlines()]

        short_model_file = MODELS_DIR / "xgboost_v4_phase4_3class_lookahead3_thresh3.pkl"
        with open(short_model_file, 'rb') as f:
            self.short_model = pickle.load(f)

        short_feature_file = MODELS_DIR / "xgboost_v4_phase4_3class_lookahead3_thresh3_features.txt"
        with open(short_feature_file, 'r') as f:
            self.short_features = [line.strip() for line in f.readlines()]

        self.adv_features = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)

        print("✅ Models loaded")

    def load_data(self):
        """Load and prepare data"""
        df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_max.csv")

        if 'timestamp' in df.columns and df['timestamp'].dtype == 'object':
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        df = calculate_features(df)
        df = self.adv_features.calculate_all_features(df)
        df = df.ffill().dropna()

        # Use recent data only for speed (last 5000 candles ~ 17 days)
        df = df.tail(5000).reset_index(drop=True)

        print(f"✅ Data loaded: {len(df)} candles")
        return df

    def backtest_with_cost(self, df, transaction_cost):
        """Run backtest with specific transaction cost"""
        long_capital = INITIAL_CAPITAL * LONG_ALLOCATION
        short_capital = INITIAL_CAPITAL * SHORT_ALLOCATION

        long_position = None
        short_position = None

        long_trades = []
        short_trades = []

        for idx in range(len(df)):
            current_price = df['close'].iloc[idx]

            # === LONG Position Management ===
            if long_position is not None:
                entry_price = long_position['entry_price']
                entry_idx = long_position['entry_idx']
                quantity = long_position['quantity']

                pnl_pct = (current_price - entry_price) / entry_price
                hours_held = (idx - entry_idx) * CHECK_INTERVAL_MINUTES / 60

                exit_reason = None
                if pnl_pct <= -LONG_STOP_LOSS:
                    exit_reason = "Stop Loss"
                elif pnl_pct >= LONG_TAKE_PROFIT:
                    exit_reason = "Take Profit"
                elif hours_held >= LONG_MAX_HOLDING_HOURS:
                    exit_reason = "Max Holding"

                if exit_reason:
                    pnl_usd = pnl_pct * (entry_price * quantity)

                    # Transaction costs (VARIABLE)
                    entry_cost = entry_price * quantity * transaction_cost
                    exit_cost = current_price * quantity * transaction_cost
                    net_pnl = pnl_usd - entry_cost - exit_cost

                    long_capital += net_pnl

                    long_trades.append({
                        'pnl_usd_gross': pnl_usd,
                        'transaction_cost': entry_cost + exit_cost,
                        'pnl_usd_net': net_pnl
                    })

                    long_position = None

            # === SHORT Position Management ===
            if short_position is not None:
                entry_price = short_position['entry_price']
                entry_idx = short_position['entry_idx']
                quantity = short_position['quantity']

                pnl_pct = (entry_price - current_price) / entry_price
                hours_held = (idx - entry_idx) * CHECK_INTERVAL_MINUTES / 60

                exit_reason = None
                if pnl_pct <= -SHORT_STOP_LOSS:
                    exit_reason = "Stop Loss"
                elif pnl_pct >= SHORT_TAKE_PROFIT:
                    exit_reason = "Take Profit"
                elif hours_held >= SHORT_MAX_HOLDING_HOURS:
                    exit_reason = "Max Holding"

                if exit_reason:
                    pnl_usd = pnl_pct * (entry_price * quantity)

                    # Transaction costs (VARIABLE)
                    entry_cost = entry_price * quantity * transaction_cost
                    exit_cost = current_price * quantity * transaction_cost
                    net_pnl = pnl_usd - entry_cost - exit_cost

                    short_capital += net_pnl

                    short_trades.append({
                        'pnl_usd_gross': pnl_usd,
                        'transaction_cost': entry_cost + exit_cost,
                        'pnl_usd_net': net_pnl
                    })

                    short_position = None

            # === Entry Signals ===
            if idx < 100:
                continue

            # LONG entry
            if long_position is None:
                try:
                    features = df[self.long_features].iloc[idx:idx+1].values
                    if not np.isnan(features).any():
                        prob = self.long_model.predict_proba(features)[0][1]

                        if prob >= LONG_THRESHOLD:
                            position_value = long_capital * LONG_POSITION_SIZE_PCT
                            quantity = position_value / current_price

                            long_position = {
                                'entry_idx': idx,
                                'entry_price': current_price,
                                'quantity': quantity
                            }
                except:
                    pass

            # SHORT entry
            if short_position is None:
                try:
                    features = df[self.short_features].iloc[idx:idx+1].values
                    if not np.isnan(features).any():
                        probs = self.short_model.predict_proba(features)[0]
                        short_prob = probs[2]

                        if short_prob >= SHORT_THRESHOLD:
                            position_value = short_capital * SHORT_POSITION_SIZE_PCT
                            quantity = position_value / current_price

                            short_position = {
                                'entry_idx': idx,
                                'entry_price': current_price,
                                'quantity': quantity
                            }
                except:
                    pass

        # Calculate results
        total_capital = long_capital + short_capital
        total_return = ((total_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100

        # Calculate days
        days = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds() / 86400
        monthly_return = (total_return / days) * 30.42

        # Calculate total costs
        total_costs = sum([t['transaction_cost'] for t in long_trades + short_trades])
        total_gross_pnl = sum([t['pnl_usd_gross'] for t in long_trades + short_trades])

        return {
            'transaction_cost_pct': transaction_cost * 100,
            'total_trades': len(long_trades) + len(short_trades),
            'total_capital': total_capital,
            'total_return_pct': total_return,
            'monthly_return_pct': monthly_return,
            'total_costs_usd': total_costs,
            'total_gross_pnl_usd': total_gross_pnl,
            'cost_impact_pct': (total_costs / INITIAL_CAPITAL) * 100
        }

    def analyze_sensitivity(self, df):
        """Run sensitivity analysis across different cost scenarios"""
        # Cost scenarios (in decimal, e.g., 0.0002 = 0.02%)
        cost_scenarios = [
            ('Optimistic', 0.0001),   # 0.01% (best case)
            ('Current Assumption', 0.0002),  # 0.02% (current)
            ('Maker Fee', 0.0004),    # 0.04% (BingX maker)
            ('Taker Fee', 0.0005),    # 0.05% (BingX taker)
            ('With Slippage (Low)', 0.0008),   # 0.08%
            ('With Slippage (Med)', 0.0010),   # 0.10%
            ('With Slippage (High)', 0.0015),  # 0.15%
            ('Worst Case', 0.0020),   # 0.20%
        ]

        results = []

        print("\n🔬 Running sensitivity analysis...")
        for scenario_name, cost in cost_scenarios:
            print(f"   Testing: {scenario_name} ({cost*100:.3f}%)")
            result = self.backtest_with_cost(df, cost)
            result['scenario'] = scenario_name
            results.append(result)

        return pd.DataFrame(results)

    def print_results(self, results_df):
        """Print sensitivity analysis results"""
        print("\n" + "="*100)
        print("TRANSACTION COST SENSITIVITY ANALYSIS")
        print("="*100)

        print(f"\n{'Scenario':<30} {'Cost%':<10} {'Monthly Return':<18} {'Total Costs':<15} {'Cost Impact':<15}")
        print("-"*100)

        baseline_return = results_df[results_df['scenario'] == 'Current Assumption']['monthly_return_pct'].iloc[0]

        for _, row in results_df.iterrows():
            scenario = row['scenario']
            cost_pct = row['transaction_cost_pct']
            monthly = row['monthly_return_pct']
            total_costs = row['total_costs_usd']
            cost_impact = row['cost_impact_pct']

            degradation = monthly - baseline_return
            status = "✅" if monthly > 15 else ("⚠️" if monthly > 10 else "🚨")

            print(f"{scenario:<30} {cost_pct:>6.3f}%   {monthly:>+7.2f}% ({degradation:+6.2f}%)  ${total_costs:>9.2f}      {cost_impact:>6.2f}%       {status}")

        print("="*100)

        # Break-even analysis
        print(f"\n📊 BREAK-EVEN ANALYSIS:")

        # Find where monthly return = 0
        baseline_trades = results_df['total_trades'].iloc[0]
        baseline_gross = results_df['total_gross_pnl_usd'].iloc[0]

        # Break-even cost = gross PnL / (2 * trades * avg position value)
        avg_position_value = INITIAL_CAPITAL * 0.9  # Approximate
        breakeven_cost = baseline_gross / (2 * baseline_trades * avg_position_value)

        print(f"   Gross P&L: ${baseline_gross:.2f}")
        print(f"   Total Trades: {baseline_trades}")
        print(f"   Break-even Cost: {breakeven_cost*100:.3f}%")
        print(f"   Safety Margin: {(breakeven_cost / 0.0002 - 1) * 100:.1f}× current assumption")

        # Profitability thresholds
        print(f"\n🎯 PROFITABILITY THRESHOLDS:")

        monthly_15_row = results_df[results_df['monthly_return_pct'] >= 15].tail(1)
        if not monthly_15_row.empty:
            max_cost_for_15 = monthly_15_row['transaction_cost_pct'].iloc[0]
            print(f"   Max cost for +15% monthly: {max_cost_for_15:.3f}%")
        else:
            print(f"   Max cost for +15% monthly: Not achievable")

        monthly_10_row = results_df[results_df['monthly_return_pct'] >= 10].tail(1)
        if not monthly_10_row.empty:
            max_cost_for_10 = monthly_10_row['transaction_cost_pct'].iloc[0]
            print(f"   Max cost for +10% monthly: {max_cost_for_10:.3f}%")
        else:
            print(f"   Max cost for +10% monthly: Not achievable")

        # Warnings
        print(f"\n⚠️ RISK WARNINGS:")

        taker_row = results_df[results_df['scenario'] == 'Taker Fee']
        taker_return = taker_row['monthly_return_pct'].iloc[0]
        taker_degradation = ((taker_return / baseline_return) - 1) * 100

        if taker_degradation < -20:
            print(f"   🚨 Taker fee causes {abs(taker_degradation):.1f}% degradation!")
            print(f"   🚨 CRITICAL: Use maker orders whenever possible!")
        elif taker_degradation < -10:
            print(f"   ⚠️ Taker fee causes {abs(taker_degradation):.1f}% degradation")
            print(f"   ⚠️ RECOMMEND: Prefer maker orders")

        slippage_high = results_df[results_df['scenario'] == 'With Slippage (High)']['monthly_return_pct'].iloc[0]
        if slippage_high < 10:
            print(f"   🚨 High slippage (0.15%) reduces returns to {slippage_high:+.2f}%!")
            print(f"   🚨 CRITICAL: Monitor actual slippage closely!")

        print("="*100)


def main():
    print("="*100)
    print("TRANSACTION COST SENSITIVITY ANALYSIS")
    print("="*100)
    print("\n목적: 실제 거래 비용이 높을 경우 전략 수익성에 미치는 영향 평가")
    print("\n시나리오:")
    print("  - Optimistic: 0.01% (최선)")
    print("  - Current: 0.02% (현재 가정)")
    print("  - Maker: 0.04% (BingX maker fee)")
    print("  - Taker: 0.05% (BingX taker fee)")
    print("  - Slippage (Low/Med/High): 0.08-0.15% (시장가 주문 영향)")
    print("  - Worst Case: 0.20% (최악)")

    analyzer = CostSensitivityAnalyzer()

    print("\n📥 Loading data...")
    df = analyzer.load_data()

    print("\n🔬 Running sensitivity analysis...")
    results_df = analyzer.analyze_sensitivity(df)

    # Print results
    analyzer.print_results(results_df)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = RESULTS_DIR / f"cost_sensitivity_analysis_{timestamp}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n✅ Results saved: {output_file}")

    # Save report
    report_file = RESULTS_DIR / f"cost_sensitivity_report_{timestamp}.txt"
    with open(report_file, 'w') as f:
        f.write("="*100 + "\n")
        f.write("TRANSACTION COST SENSITIVITY ANALYSIS REPORT\n")
        f.write("="*100 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Configuration: LONG 90% / SHORT 10%\n\n")

        f.write("RESULTS SUMMARY\n")
        f.write("-"*100 + "\n\n")

        baseline = results_df[results_df['scenario'] == 'Current Assumption'].iloc[0]
        f.write(f"Baseline (Current Assumption: 0.02%):\n")
        f.write(f"  Monthly Return: {baseline['monthly_return_pct']:+.2f}%\n")
        f.write(f"  Total Costs: ${baseline['total_costs_usd']:.2f}\n")
        f.write(f"  Cost Impact: {baseline['cost_impact_pct']:.2f}%\n\n")

        for _, row in results_df.iterrows():
            f.write(f"{row['scenario']} ({row['transaction_cost_pct']:.3f}%):\n")
            f.write(f"  Monthly Return: {row['monthly_return_pct']:+.2f}%\n")
            f.write(f"  vs Baseline: {row['monthly_return_pct'] - baseline['monthly_return_pct']:+.2f}%\n")
            f.write(f"  Total Costs: ${row['total_costs_usd']:.2f}\n\n")

        taker = results_df[results_df['scenario'] == 'Taker Fee'].iloc[0]
        degradation = ((taker['monthly_return_pct'] / baseline['monthly_return_pct']) - 1) * 100

        f.write(f"CRITICAL FINDINGS:\n")
        f.write(f"  - Taker fee impact: {degradation:+.1f}%\n")
        f.write(f"  - Recommendation: Use maker orders\n")

    print(f"✅ Report saved: {report_file}")
    print("\n" + "="*100)
    print("COST SENSITIVITY ANALYSIS COMPLETE")
    print("="*100)


if __name__ == "__main__":
    main()
