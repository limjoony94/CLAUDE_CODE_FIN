"""
Comprehensive Backtest Validation - 90/10 Allocation

Purpose: 상세한 백테스트 검증
- Trade-by-trade analysis
- Daily/Weekly/Monthly performance
- Risk metrics (Sharpe, Sortino, Max DD)
- Win/Loss distribution
- Validation report generation
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime, timedelta
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Configuration
INITIAL_CAPITAL = 10000.0
LONG_ALLOCATION = 0.90
SHORT_ALLOCATION = 0.10

LONG_THRESHOLD = 0.7
LONG_STOP_LOSS = 0.01
LONG_TAKE_PROFIT = 0.03
LONG_MAX_HOLDING_HOURS = 4
LONG_POSITION_SIZE_PCT = 0.95

SHORT_THRESHOLD = 0.4
SHORT_STOP_LOSS = 0.015
SHORT_TAKE_PROFIT = 0.06
SHORT_MAX_HOLDING_HOURS = 4
SHORT_POSITION_SIZE_PCT = 0.95

TRANSACTION_COST = 0.0002


class DetailedBacktester:
    """Comprehensive backtester with detailed tracking"""

    def __init__(self):
        self.long_capital = INITIAL_CAPITAL * LONG_ALLOCATION
        self.short_capital = INITIAL_CAPITAL * SHORT_ALLOCATION

        self.long_position = None
        self.short_position = None

        self.long_trades = []
        self.short_trades = []

        self.equity_curve = []
        self.daily_returns = []

    def backtest_strategy(self, df, long_model, long_features, short_model, short_features):
        """Run detailed backtest"""

        for i in range(len(df)):
            timestamp = df['timestamp'].iloc[i] if 'timestamp' in df.columns else i
            current_price = df['close'].iloc[i]

            # Record equity
            total_equity = self.long_capital + self.short_capital
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': total_equity,
                'long_capital': self.long_capital,
                'short_capital': self.short_capital
            })

            # Manage LONG position
            if self.long_position is not None:
                self._check_long_exit(i, current_price)

            # Manage SHORT position
            if self.short_position is not None:
                self._check_short_exit(i, current_price)

            # Look for LONG entry
            if self.long_position is None and i < len(df) - 1:
                self._check_long_entry(i, df, long_model, long_features, current_price)

            # Look for SHORT entry
            if self.short_position is None and i < len(df) - 1:
                self._check_short_entry(i, df, short_model, short_features, current_price)

        return self._generate_report()

    def _check_long_entry(self, i, df, model, features, price):
        """Check LONG entry"""
        try:
            feature_values = df[features].iloc[i:i+1].values
            if np.isnan(feature_values).any():
                return

            prob = model.predict_proba(feature_values)[0][1]

            if prob >= LONG_THRESHOLD:
                position_value = self.long_capital * LONG_POSITION_SIZE_PCT
                quantity = position_value / price

                self.long_position = {
                    'entry_idx': i,
                    'entry_price': price,
                    'quantity': quantity,
                    'probability': prob,
                    'entry_time': df['timestamp'].iloc[i] if 'timestamp' in df.columns else i
                }
        except Exception as e:
            pass

    def _check_short_entry(self, i, df, model, features, price):
        """Check SHORT entry"""
        try:
            feature_values = df[features].iloc[i:i+1].values
            if np.isnan(feature_values).any():
                return

            probs = model.predict_proba(feature_values)[0]
            prob = probs[2]

            if prob >= SHORT_THRESHOLD:
                position_value = self.short_capital * SHORT_POSITION_SIZE_PCT
                quantity = position_value / price

                self.short_position = {
                    'entry_idx': i,
                    'entry_price': price,
                    'quantity': quantity,
                    'probability': prob,
                    'entry_time': df['timestamp'].iloc[i] if 'timestamp' in df.columns else i
                }
        except Exception as e:
            pass

    def _check_long_exit(self, i, current_price):
        """Check LONG exit"""
        entry_idx = self.long_position['entry_idx']
        entry_price = self.long_position['entry_price']
        hours_held = (i - entry_idx) / 12

        pnl_pct = (current_price - entry_price) / entry_price

        exit_reason = None
        if pnl_pct <= -LONG_STOP_LOSS:
            exit_reason = "Stop Loss"
        elif pnl_pct >= LONG_TAKE_PROFIT:
            exit_reason = "Take Profit"
        elif hours_held >= LONG_MAX_HOLDING_HOURS:
            exit_reason = "Max Holding"

        if exit_reason:
            quantity = self.long_position['quantity']
            pnl_usd = pnl_pct * (entry_price * quantity)

            # Transaction costs
            entry_cost = entry_price * quantity * TRANSACTION_COST
            exit_cost = current_price * quantity * TRANSACTION_COST
            pnl_usd -= (entry_cost + exit_cost)

            self.long_capital += pnl_usd

            self.long_trades.append({
                'strategy': 'LONG',
                'entry_idx': entry_idx,
                'exit_idx': i,
                'entry_price': entry_price,
                'exit_price': current_price,
                'pnl_pct': pnl_pct * 100,
                'pnl_usd': pnl_usd,
                'exit_reason': exit_reason,
                'hours_held': hours_held,
                'probability': self.long_position['probability']
            })

            self.long_position = None

    def _check_short_exit(self, i, current_price):
        """Check SHORT exit"""
        entry_idx = self.short_position['entry_idx']
        entry_price = self.short_position['entry_price']
        hours_held = (i - entry_idx) / 12

        pnl_pct = (entry_price - current_price) / entry_price

        exit_reason = None
        if pnl_pct <= -SHORT_STOP_LOSS:
            exit_reason = "Stop Loss"
        elif pnl_pct >= SHORT_TAKE_PROFIT:
            exit_reason = "Take Profit"
        elif hours_held >= SHORT_MAX_HOLDING_HOURS:
            exit_reason = "Max Holding"

        if exit_reason:
            quantity = self.short_position['quantity']
            pnl_usd = pnl_pct * (entry_price * quantity)

            # Transaction costs
            entry_cost = entry_price * quantity * TRANSACTION_COST
            exit_cost = current_price * quantity * TRANSACTION_COST
            pnl_usd -= (entry_cost + exit_cost)

            self.short_capital += pnl_usd

            self.short_trades.append({
                'strategy': 'SHORT',
                'entry_idx': entry_idx,
                'exit_idx': i,
                'entry_price': entry_price,
                'exit_price': current_price,
                'pnl_pct': pnl_pct * 100,
                'pnl_usd': pnl_usd,
                'exit_reason': exit_reason,
                'hours_held': hours_held,
                'probability': self.short_position['probability']
            })

            self.short_position = None

    def _generate_report(self):
        """Generate comprehensive report"""

        all_trades = self.long_trades + self.short_trades

        if not all_trades:
            return None

        # Basic metrics
        total_trades = len(all_trades)
        long_trades_count = len(self.long_trades)
        short_trades_count = len(self.short_trades)

        winning_trades = [t for t in all_trades if t['pnl_usd'] > 0]
        losing_trades = [t for t in all_trades if t['pnl_usd'] <= 0]

        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0

        # P&L metrics
        total_pnl = sum(t['pnl_usd'] for t in all_trades)
        avg_win = np.mean([t['pnl_usd'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl_usd'] for t in losing_trades]) if losing_trades else 0

        # Return metrics
        final_capital = self.long_capital + self.short_capital
        total_return = ((final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100

        # Risk metrics
        returns = [t['pnl_pct'] for t in all_trades]

        if len(returns) > 1:
            sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0

            downside_returns = [r for r in returns if r < 0]
            sortino_ratio = (np.mean(returns) / np.std(downside_returns)) * np.sqrt(252) if downside_returns and np.std(downside_returns) > 0 else 0
        else:
            sharpe_ratio = 0
            sortino_ratio = 0

        # Max drawdown
        equity_values = [e['equity'] for e in self.equity_curve]
        peak = equity_values[0]
        max_dd = 0

        for value in equity_values:
            if value > peak:
                peak = value
            dd = ((peak - value) / peak) * 100
            if dd > max_dd:
                max_dd = dd

        return {
            'total_trades': total_trades,
            'long_trades': long_trades_count,
            'short_trades': short_trades_count,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return': total_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_dd,
            'final_capital': final_capital,
            'trades': all_trades,
            'equity_curve': self.equity_curve
        }


def main():
    """Run comprehensive backtest"""
    print("="*80)
    print("COMPREHENSIVE BACKTEST VALIDATION - 90/10 Allocation")
    print("="*80)
    print("\nValidating optimal configuration:")
    print("  LONG: 90% allocation")
    print("  SHORT: 10% allocation")
    print("  Initial Capital: $10,000")
    print("="*80)

    # Load models
    print("\n📥 Loading models...")

    long_model_file = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl"
    with open(long_model_file, 'rb') as f:
        long_model = pickle.load(f)

    long_feature_file = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt"
    with open(long_feature_file, 'r') as f:
        long_features = [line.strip() for line in f.readlines()]
    print(f"  ✅ LONG model loaded ({len(long_features)} features)")

    short_model_file = MODELS_DIR / "xgboost_v4_phase4_3class_lookahead3_thresh3.pkl"
    with open(short_model_file, 'rb') as f:
        short_model = pickle.load(f)

    short_feature_file = MODELS_DIR / "xgboost_v4_phase4_3class_lookahead3_thresh3_features.txt"
    with open(short_feature_file, 'r') as f:
        short_features = [line.strip() for line in f.readlines()]
    print(f"  ✅ SHORT model loaded ({len(short_features)} features)")

    # Load and prepare data
    print("\n📥 Loading data...")
    data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
    df = pd.read_csv(data_file)

    print("  ⚙️  Calculating features...")
    df = calculate_features(df)
    adv_features = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)
    df = adv_features.calculate_all_features(df)
    df = df.ffill().dropna()

    print(f"  ✅ Data prepared: {len(df):,} candles ({len(df)/288:.1f} days)")

    # Run backtest
    print("\n🔬 Running comprehensive backtest...")
    backtester = DetailedBacktester()
    results = backtester.backtest_strategy(df, long_model, long_features, short_model, short_features)

    if not results:
        print("  ❌ No trades executed!")
        return

    # Display results
    print("\n" + "="*80)
    print("📊 BACKTEST RESULTS")
    print("="*80)

    print(f"\n💰 Performance Summary:")
    print(f"  Initial Capital: ${INITIAL_CAPITAL:,.2f}")
    print(f"  Final Capital: ${results['final_capital']:,.2f}")
    print(f"  Total Return: {results['total_return']:+.2f}%")
    print(f"  Total P&L: ${results['total_pnl']:+,.2f}")

    print(f"\n📈 Trade Statistics:")
    print(f"  Total Trades: {results['total_trades']}")
    print(f"    LONG: {results['long_trades']} ({results['long_trades']/results['total_trades']*100:.1f}%)")
    print(f"    SHORT: {results['short_trades']} ({results['short_trades']/results['total_trades']*100:.1f}%)")
    print(f"  Win Rate: {results['win_rate']:.1f}%")
    print(f"  Average Win: ${results['avg_win']:,.2f}")
    print(f"  Average Loss: ${results['avg_loss']:,.2f}")

    # Calculate trades per day
    total_days = len(df) / 288  # 288 5-min candles per day
    trades_per_day = results['total_trades'] / total_days

    print(f"\n📊 Frequency Metrics:")
    print(f"  Backtest Period: {total_days:.1f} days")
    print(f"  Trades per Day: {trades_per_day:.2f}")
    print(f"  Trades per Month (est): {trades_per_day * 30:.1f}")

    print(f"\n⚖️  Risk Metrics:")
    print(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"  Sortino Ratio: {results['sortino_ratio']:.2f}")
    print(f"  Max Drawdown: {results['max_drawdown']:.2f}%")

    # Monthly extrapolation
    monthly_return = (results['total_return'] / total_days) * 30
    print(f"\n📅 Monthly Extrapolation:")
    print(f"  Estimated Monthly Return: {monthly_return:+.2f}%")

    # Save detailed results
    print("\n💾 Saving detailed results...")

    # Save trades
    trades_df = pd.DataFrame(results['trades'])
    trades_file = RESULTS_DIR / "backtest_90_10_trades.csv"
    trades_df.to_csv(trades_file, index=False)
    print(f"  ✅ Trades saved: {trades_file}")

    # Save equity curve
    equity_df = pd.DataFrame(results['equity_curve'])
    equity_file = RESULTS_DIR / "backtest_90_10_equity_curve.csv"
    equity_df.to_csv(equity_file, index=False)
    print(f"  ✅ Equity curve saved: {equity_file}")

    # Generate validation report
    report_file = RESULTS_DIR / "backtest_90_10_validation_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("BACKTEST VALIDATION REPORT - 90/10 Allocation\n")
        f.write("="*80 + "\n\n")

        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Configuration: LONG 90% / SHORT 10%\n\n")

        f.write("Configuration Details:\n")
        f.write(f"  LONG Threshold: {LONG_THRESHOLD}\n")
        f.write(f"  LONG SL/TP: {LONG_STOP_LOSS*100:.1f}% / {LONG_TAKE_PROFIT*100:.1f}%\n")
        f.write(f"  SHORT Threshold: {SHORT_THRESHOLD}\n")
        f.write(f"  SHORT SL/TP: {SHORT_STOP_LOSS*100:.1f}% / {SHORT_TAKE_PROFIT*100:.1f}%\n\n")

        f.write("="*80 + "\n")
        f.write("RESULTS\n")
        f.write("="*80 + "\n\n")

        f.write(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}\n")
        f.write(f"Final Capital: ${results['final_capital']:,.2f}\n")
        f.write(f"Total Return: {results['total_return']:+.2f}%\n")
        f.write(f"Monthly Return (est): {monthly_return:+.2f}%\n\n")

        f.write(f"Total Trades: {results['total_trades']}\n")
        f.write(f"  LONG: {results['long_trades']}\n")
        f.write(f"  SHORT: {results['short_trades']}\n")
        f.write(f"Win Rate: {results['win_rate']:.1f}%\n\n")

        f.write(f"Trades per Day: {trades_per_day:.2f}\n")
        f.write(f"Estimated Trades per Month: {trades_per_day * 30:.1f}\n\n")

        f.write(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}\n")
        f.write(f"Sortino Ratio: {results['sortino_ratio']:.2f}\n")
        f.write(f"Max Drawdown: {results['max_drawdown']:.2f}%\n\n")

        f.write("="*80 + "\n")
        f.write("VALIDATION STATUS\n")
        f.write("="*80 + "\n\n")

        # Validation checks
        monthly_target = 18.0  # Target ~19.82%
        trades_target = 120   # Target ~4/day × 30 = 120/month

        monthly_ok = monthly_return >= monthly_target
        trades_ok = trades_per_day * 30 >= trades_target * 0.8  # 80% of target acceptable
        sharpe_ok = results['sharpe_ratio'] >= 2.0
        dd_ok = results['max_drawdown'] <= 5.0

        f.write(f"✓ Monthly Return ≥ {monthly_target}%: {'✅ PASS' if monthly_ok else '❌ FAIL'} ({monthly_return:+.2f}%)\n")
        f.write(f"✓ Trades/Month ≥ {trades_target*0.8:.0f}: {'✅ PASS' if trades_ok else '❌ FAIL'} ({trades_per_day * 30:.1f})\n")
        f.write(f"✓ Sharpe Ratio ≥ 2.0: {'✅ PASS' if sharpe_ok else '❌ FAIL'} ({results['sharpe_ratio']:.2f})\n")
        f.write(f"✓ Max Drawdown ≤ 5%: {'✅ PASS' if dd_ok else '❌ FAIL'} ({results['max_drawdown']:.2f}%)\n\n")

        all_pass = monthly_ok and trades_ok and sharpe_ok and dd_ok

        f.write(f"Overall Status: {'✅ VALIDATED - Ready for Testnet' if all_pass else '⚠️ REVIEW NEEDED'}\n\n")

        f.write("="*80 + "\n")
        f.write("Next Steps:\n")
        f.write("="*80 + "\n\n")

        if all_pass:
            f.write("1. Deploy to BingX Testnet\n")
            f.write("2. Monitor for 1 week\n")
            f.write("3. Validate real-time performance matches backtest\n")
        else:
            f.write("1. Review failed validation criteria\n")
            f.write("2. Adjust configuration if needed\n")
            f.write("3. Re-run backtest\n")

    print(f"  ✅ Validation report saved: {report_file}")

    # Final summary
    print("\n" + "="*80)
    print("✅ VALIDATION COMPLETE")
    print("="*80)

    print(f"\nEstimated Monthly Return: {monthly_return:+.2f}%")
    print(f"Expected Trades/Month: {trades_per_day * 30:.1f}")

    if all([monthly_return >= 18.0, trades_per_day * 30 >= 96, results['sharpe_ratio'] >= 2.0, results['max_drawdown'] <= 5.0]):
        print("\n✅ All validation criteria PASSED")
        print("✅ Ready for Testnet deployment")
    else:
        print("\n⚠️  Some validation criteria need review")
        print("   See validation report for details")

    print("\n📁 Files generated:")
    print(f"  - {trades_file}")
    print(f"  - {equity_file}")
    print(f"  - {report_file}")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
