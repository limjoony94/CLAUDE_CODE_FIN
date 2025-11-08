"""
Hold-out Validation - Out-of-Sample Testing

비판적 사고 검증: 완전히 새로운 데이터에서 전략 성능 확인

목적:
    - 백테스트가 과적합이 아닌지 확인
    - 최근 데이터에서 실제 성능 검증
    - Overfitting vs Generalization 평가

방법:
    - 전체 데이터를 Train/Test로 분할
    - Train: 처음 80% (백테스트 기간)
    - Test: 최근 20% (완전히 새로운 데이터)
    - Test 데이터로 성능 재평가
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
TRANSACTION_COST = 0.0002


class HoldoutValidator:
    """Hold-out validation for out-of-sample testing"""

    def __init__(self, train_pct=0.80):
        """
        Args:
            train_pct: Percentage of data for training (backtest period)
        """
        self.train_pct = train_pct
        self.test_pct = 1.0 - train_pct

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
        print(f"   LONG: {len(self.long_features)} features")
        print(f"   SHORT: {len(self.short_features)} features")

    def load_and_split_data(self):
        """Load data and split into train/test"""
        df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_max.csv")

        # Convert timestamp to datetime if needed
        if 'timestamp' in df.columns and df['timestamp'].dtype == 'object':
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Calculate features
        df = calculate_features(df)
        df = self.adv_features.calculate_all_features(df)
        df = df.ffill().dropna()

        # Split point
        split_idx = int(len(df) * self.train_pct)

        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        print(f"\n📊 Data Split:")
        print(f"   Total: {len(df)} candles")
        print(f"   Train: {len(train_df)} candles ({self.train_pct*100:.0f}%)")
        print(f"   Test: {len(test_df)} candles ({self.test_pct*100:.0f}%)")
        print(f"   Train period: {train_df['timestamp'].iloc[0]} to {train_df['timestamp'].iloc[-1]}")
        print(f"   Test period: {test_df['timestamp'].iloc[0]} to {test_df['timestamp'].iloc[-1]}")

        return train_df, test_df

    def backtest_strategy(self, df, dataset_name="Test"):
        """Run backtest on given dataset"""
        long_capital = INITIAL_CAPITAL * LONG_ALLOCATION
        short_capital = INITIAL_CAPITAL * SHORT_ALLOCATION

        long_position = None
        short_position = None

        long_trades = []
        short_trades = []

        equity_curve = []

        for idx in range(len(df)):
            current_time = df['timestamp'].iloc[idx]
            current_price = df['close'].iloc[idx]

            # Track equity
            total_capital = long_capital + short_capital
            equity_curve.append({
                'timestamp': current_time,
                'capital': total_capital,
                'long_capital': long_capital,
                'short_capital': short_capital
            })

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

                    # Transaction costs
                    entry_cost = entry_price * quantity * TRANSACTION_COST
                    exit_cost = current_price * quantity * TRANSACTION_COST
                    net_pnl = pnl_usd - entry_cost - exit_cost

                    long_capital += net_pnl

                    long_trades.append({
                        'entry_time': df['timestamp'].iloc[entry_idx],
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'quantity': quantity,
                        'pnl_pct': pnl_pct,
                        'pnl_usd': net_pnl,
                        'exit_reason': exit_reason,
                        'hours_held': hours_held
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

                    # Transaction costs
                    entry_cost = entry_price * quantity * TRANSACTION_COST
                    exit_cost = current_price * quantity * TRANSACTION_COST
                    net_pnl = pnl_usd - entry_cost - exit_cost

                    short_capital += net_pnl

                    short_trades.append({
                        'entry_time': df['timestamp'].iloc[entry_idx],
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'quantity': quantity,
                        'pnl_pct': pnl_pct,
                        'pnl_usd': net_pnl,
                        'exit_reason': exit_reason,
                        'hours_held': hours_held
                    })

                    short_position = None

            # === Entry Signals ===
            if idx < 100:  # Need enough history
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
                                'quantity': quantity,
                                'probability': prob
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
                                'quantity': quantity,
                                'probability': short_prob
                            }
                except:
                    pass

        # Calculate results
        total_capital = long_capital + short_capital
        total_return = ((total_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100

        # Calculate days
        days = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds() / 86400
        monthly_return = (total_return / days) * 30.42

        # Win rates
        long_wins = len([t for t in long_trades if t['pnl_usd'] > 0])
        long_win_rate = (long_wins / len(long_trades) * 100) if long_trades else 0

        short_wins = len([t for t in short_trades if t['pnl_usd'] > 0])
        short_win_rate = (short_wins / len(short_trades) * 100) if short_trades else 0

        total_trades = len(long_trades) + len(short_trades)
        total_wins = long_wins + short_wins
        overall_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0

        # Trades per day
        trades_per_day = total_trades / days if days > 0 else 0

        # Calculate Sharpe ratio
        equity_df = pd.DataFrame(equity_curve)
        equity_df['returns'] = equity_df['capital'].pct_change()
        sharpe = (equity_df['returns'].mean() / equity_df['returns'].std() * np.sqrt(288)) if equity_df['returns'].std() > 0 else 0

        # Max drawdown
        equity_df['cummax'] = equity_df['capital'].cummax()
        equity_df['drawdown'] = (equity_df['capital'] - equity_df['cummax']) / equity_df['cummax']
        max_dd = equity_df['drawdown'].min() * 100

        results = {
            'dataset': dataset_name,
            'days': days,
            'initial_capital': INITIAL_CAPITAL,
            'final_capital': total_capital,
            'total_return_pct': total_return,
            'monthly_return_pct': monthly_return,
            'total_trades': total_trades,
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
            'overall_win_rate': overall_win_rate,
            'long_win_rate': long_win_rate,
            'short_win_rate': short_win_rate,
            'trades_per_day': trades_per_day,
            'sharpe_ratio': sharpe,
            'max_drawdown_pct': max_dd,
            'long_trades_list': long_trades,
            'short_trades_list': short_trades,
            'equity_curve': equity_curve
        }

        return results

    def print_results(self, train_results, test_results):
        """Print comparison of train vs test results"""
        print("\n" + "="*80)
        print("HOLD-OUT VALIDATION RESULTS")
        print("="*80)

        print(f"\n{'Metric':<30} {'Train (80%)':<20} {'Test (20%)':<20} {'Degradation':<15}")
        print("-"*85)

        metrics = [
            ('Period (days)', f"{train_results['days']:.1f}", f"{test_results['days']:.1f}", ""),
            ('Total Trades', f"{train_results['total_trades']}", f"{test_results['total_trades']}", ""),
            ('Trades/Day', f"{train_results['trades_per_day']:.2f}", f"{test_results['trades_per_day']:.2f}", ""),
            ('', '', '', ''),
            ('Monthly Return', f"{train_results['monthly_return_pct']:+.2f}%", f"{test_results['monthly_return_pct']:+.2f}%",
             f"{((test_results['monthly_return_pct'] / train_results['monthly_return_pct']) - 1) * 100:+.1f}%"),
            ('Overall Win Rate', f"{train_results['overall_win_rate']:.1f}%", f"{test_results['overall_win_rate']:.1f}%",
             f"{test_results['overall_win_rate'] - train_results['overall_win_rate']:+.1f}pp"),
            ('Sharpe Ratio', f"{train_results['sharpe_ratio']:.2f}", f"{test_results['sharpe_ratio']:.2f}",
             f"{((test_results['sharpe_ratio'] / train_results['sharpe_ratio']) - 1) * 100:+.1f}%"),
            ('Max Drawdown', f"{train_results['max_drawdown_pct']:.2f}%", f"{test_results['max_drawdown_pct']:.2f}%",
             f"{test_results['max_drawdown_pct'] - train_results['max_drawdown_pct']:+.2f}pp"),
            ('', '', '', ''),
            ('LONG Trades', f"{train_results['long_trades']}", f"{test_results['long_trades']}", ""),
            ('LONG Win Rate', f"{train_results['long_win_rate']:.1f}%", f"{test_results['long_win_rate']:.1f}%",
             f"{test_results['long_win_rate'] - train_results['long_win_rate']:+.1f}pp"),
            ('SHORT Trades', f"{train_results['short_trades']}", f"{test_results['short_trades']}", ""),
            ('SHORT Win Rate', f"{train_results['short_win_rate']:.1f}%", f"{test_results['short_win_rate']:.1f}%",
             f"{test_results['short_win_rate'] - train_results['short_win_rate']:+.1f}pp"),
        ]

        for metric, train_val, test_val, deg in metrics:
            print(f"{metric:<30} {train_val:<20} {test_val:<20} {deg:<15}")

        print("="*80)

        # Performance degradation analysis
        monthly_degradation = ((test_results['monthly_return_pct'] / train_results['monthly_return_pct']) - 1) * 100

        print(f"\n📊 PERFORMANCE DEGRADATION ANALYSIS:")
        print(f"   Monthly Return Degradation: {monthly_degradation:+.1f}%")

        if monthly_degradation >= -10:
            print(f"   Status: ✅ EXCELLENT (degradation < 10%)")
        elif monthly_degradation >= -20:
            print(f"   Status: ✅ GOOD (degradation 10-20%)")
        elif monthly_degradation >= -30:
            print(f"   Status: ⚠️ ACCEPTABLE (degradation 20-30%)")
        else:
            print(f"   Status: 🚨 CONCERNING (degradation > 30%)")

        # Generalization assessment
        print(f"\n🎯 GENERALIZATION ASSESSMENT:")

        if test_results['overall_win_rate'] >= 55:
            print(f"   Win Rate: ✅ GOOD ({test_results['overall_win_rate']:.1f}% >= 55%)")
        elif test_results['overall_win_rate'] >= 50:
            print(f"   Win Rate: ⚠️ MARGINAL ({test_results['overall_win_rate']:.1f}% 50-55%)")
        else:
            print(f"   Win Rate: 🚨 POOR ({test_results['overall_win_rate']:.1f}% < 50%)")

        if test_results['monthly_return_pct'] >= 15:
            print(f"   Returns: ✅ GOOD ({test_results['monthly_return_pct']:+.1f}% >= 15%)")
        elif test_results['monthly_return_pct'] >= 10:
            print(f"   Returns: ⚠️ MARGINAL ({test_results['monthly_return_pct']:+.1f}% 10-15%)")
        else:
            print(f"   Returns: 🚨 POOR ({test_results['monthly_return_pct']:+.1f}% < 10%)")

        if abs(test_results['sharpe_ratio'] - train_results['sharpe_ratio']) / train_results['sharpe_ratio'] <= 0.30:
            print(f"   Sharpe Stability: ✅ GOOD (within 30%)")
        else:
            print(f"   Sharpe Stability: ⚠️ UNSTABLE (>30% change)")

        print("="*80)


def main():
    print("="*80)
    print("HOLD-OUT VALIDATION - Out-of-Sample Testing")
    print("="*80)
    print("\n목적: 백테스트가 과적합이 아닌지 완전히 새로운 데이터에서 검증")
    print("\n방법:")
    print("  - Train: 처음 80% (백테스트 기간)")
    print("  - Test: 최근 20% (완전히 새로운 데이터)")
    print("  - Test 데이터에서 성능 재평가")

    validator = HoldoutValidator(train_pct=0.80)

    print("\n📥 Loading and splitting data...")
    train_df, test_df = validator.load_and_split_data()

    print("\n🔬 Running backtest on Train data...")
    train_results = validator.backtest_strategy(train_df, "Train (80%)")

    print("\n🔬 Running backtest on Test data (Out-of-Sample)...")
    test_results = validator.backtest_strategy(test_df, "Test (20%)")

    # Print comparison
    validator.print_results(train_results, test_results)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save train trades
    if train_results['long_trades_list'] or train_results['short_trades_list']:
        all_trades = []
        for t in train_results['long_trades_list']:
            t['side'] = 'LONG'
            all_trades.append(t)
        for t in train_results['short_trades_list']:
            t['side'] = 'SHORT'
            all_trades.append(t)

        df_train = pd.DataFrame(all_trades)
        train_file = RESULTS_DIR / f"holdout_train_trades_{timestamp}.csv"
        df_train.to_csv(train_file, index=False)
        print(f"\n✅ Train trades saved: {train_file}")

    # Save test trades
    if test_results['long_trades_list'] or test_results['short_trades_list']:
        all_trades = []
        for t in test_results['long_trades_list']:
            t['side'] = 'LONG'
            all_trades.append(t)
        for t in test_results['short_trades_list']:
            t['side'] = 'SHORT'
            all_trades.append(t)

        df_test = pd.DataFrame(all_trades)
        test_file = RESULTS_DIR / f"holdout_test_trades_{timestamp}.csv"
        df_test.to_csv(test_file, index=False)
        print(f"✅ Test trades saved: {test_file}")

    # Save summary report
    report_file = RESULTS_DIR / f"holdout_validation_report_{timestamp}.txt"
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("HOLD-OUT VALIDATION REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Configuration: LONG 90% / SHORT 10%\n\n")

        f.write("RESULTS SUMMARY\n")
        f.write("-"*80 + "\n\n")

        f.write(f"Train Period (80%):\n")
        f.write(f"  Days: {train_results['days']:.1f}\n")
        f.write(f"  Monthly Return: {train_results['monthly_return_pct']:+.2f}%\n")
        f.write(f"  Win Rate: {train_results['overall_win_rate']:.1f}%\n")
        f.write(f"  Total Trades: {train_results['total_trades']}\n")
        f.write(f"  Sharpe Ratio: {train_results['sharpe_ratio']:.2f}\n\n")

        f.write(f"Test Period (20% - Out-of-Sample):\n")
        f.write(f"  Days: {test_results['days']:.1f}\n")
        f.write(f"  Monthly Return: {test_results['monthly_return_pct']:+.2f}%\n")
        f.write(f"  Win Rate: {test_results['overall_win_rate']:.1f}%\n")
        f.write(f"  Total Trades: {test_results['total_trades']}\n")
        f.write(f"  Sharpe Ratio: {test_results['sharpe_ratio']:.2f}\n\n")

        monthly_degradation = ((test_results['monthly_return_pct'] / train_results['monthly_return_pct']) - 1) * 100
        f.write(f"Performance Degradation: {monthly_degradation:+.1f}%\n\n")

        if monthly_degradation >= -20:
            f.write("Status: ✅ PASSED (degradation < 20%)\n")
        elif monthly_degradation >= -30:
            f.write("Status: ⚠️ ACCEPTABLE (degradation 20-30%)\n")
        else:
            f.write("Status: 🚨 FAILED (degradation > 30%)\n")

    print(f"✅ Summary report saved: {report_file}")
    print("\n" + "="*80)
    print("HOLD-OUT VALIDATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
