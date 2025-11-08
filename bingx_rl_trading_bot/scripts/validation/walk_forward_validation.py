"""
Walk-forward Validation
시간에 따른 전략 강건성 검증 - Rolling window 방식
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

class WalkForwardValidator:
    """
    Walk-forward 검증기
    - 데이터를 순차적으로 분할하여 시간에 따른 성능 추적
    - 각 fold에서 독립적으로 백테스트 실행
    - 시장 regime 변화에 대한 강건성 검증
    """

    def __init__(self, n_folds=5):
        """
        Args:
            n_folds: 분할할 fold 수 (기본 5개)
        """
        self.n_folds = n_folds
        self.adv_features = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)

        # 모델 로드 (LONG, SHORT)
        long_model_file = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl"
        with open(long_model_file, 'rb') as f:
            self.model_long = pickle.load(f)

        long_feature_file = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt"
        with open(long_feature_file, 'r') as f:
            self.long_features = [line.strip() for line in f.readlines()]

        short_model_file = MODELS_DIR / "xgboost_v4_phase4_3class_lookahead3_thresh3.pkl"
        with open(short_model_file, 'rb') as f:
            self.model_short = pickle.load(f)

        short_feature_file = MODELS_DIR / "xgboost_v4_phase4_3class_lookahead3_thresh3_features.txt"
        with open(short_feature_file, 'r') as f:
            self.short_features = [line.strip() for line in f.readlines()]

        print(f"✅ Models loaded: LONG ({len(self.long_features)} features), SHORT ({len(self.short_features)} features)")

    def load_and_prepare_data(self):
        """데이터 로드 및 feature 계산"""
        print("\n" + "="*100)
        print("WALK-FORWARD VALIDATION - DATA PREPARATION")
        print("="*100)

        df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_max.csv")

        # Convert timestamp
        if 'timestamp' in df.columns and df['timestamp'].dtype == 'object':
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        print(f"\n📊 Data loaded: {len(df)} candles")
        print(f"   Period: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"   Duration: {(df['timestamp'].max() - df['timestamp'].min()).days} days")

        # Calculate features
        df = calculate_features(df)
        df = self.adv_features.calculate_all_features(df)
        df = df.ffill().dropna()

        print(f"   After feature calculation: {len(df)} candles")

        return df

    def create_folds(self, df):
        """데이터를 시간순으로 n_folds로 분할"""
        fold_size = len(df) // self.n_folds
        folds = []

        for i in range(self.n_folds):
            start_idx = i * fold_size
            # 마지막 fold는 남은 모든 데이터 포함
            end_idx = (i + 1) * fold_size if i < self.n_folds - 1 else len(df)

            fold_df = df.iloc[start_idx:end_idx].copy()

            fold_info = {
                'fold_num': i + 1,
                'data': fold_df,
                'start_date': fold_df['timestamp'].min(),
                'end_date': fold_df['timestamp'].max(),
                'n_candles': len(fold_df),
                'days': (fold_df['timestamp'].max() - fold_df['timestamp'].min()).days
            }

            folds.append(fold_info)

        return folds

    def backtest_fold(self, fold_df, fold_num):
        """특정 fold에서 백테스트 실행"""

        # 초기 자본
        INITIAL_CAPITAL = 10000
        LONG_CAPITAL = INITIAL_CAPITAL * 0.90  # 90%
        SHORT_CAPITAL = INITIAL_CAPITAL * 0.10  # 10%

        long_capital = LONG_CAPITAL
        short_capital = SHORT_CAPITAL

        # 거래 설정
        LONG_POSITION_SIZE = 0.95
        SHORT_POSITION_SIZE = 0.95

        LONG_SL = 0.01  # 1%
        LONG_TP = 0.03  # 3%
        SHORT_SL = 0.015  # 1.5%
        SHORT_TP = 0.06  # 6%

        MAX_HOLDING_TIME = pd.Timedelta(hours=4)
        TRANSACTION_COST = 0.0002  # 0.02%

        XGB_THRESHOLD_LONG = 0.7
        XGB_THRESHOLD_SHORT = 0.4

        # 포지션 추적
        long_position = None
        short_position = None

        long_trades = []
        short_trades = []

        # 백테스트 루프
        for idx in range(len(fold_df)):
            if idx < 100:  # Need enough history
                continue

            current_time = fold_df['timestamp'].iloc[idx]
            current_price = fold_df['close'].iloc[idx]

            # LONG 전략
            if long_position is None:
                # Entry signal - XGBoost 예측
                try:
                    features = fold_df[self.long_features].iloc[idx:idx+1].values
                    if not np.isnan(features).any():
                        long_prob = self.model_long.predict_proba(features)[0][1]
                    else:
                        long_prob = 0
                except:
                    long_prob = 0

                if long_prob >= XGB_THRESHOLD_LONG:
                    position_value = long_capital * LONG_POSITION_SIZE
                    quantity = position_value / current_price
                    entry_cost = position_value * TRANSACTION_COST

                    long_position = {
                        'entry_time': current_time,
                        'entry_price': current_price,
                        'quantity': quantity,
                        'entry_cost': entry_cost
                    }
            else:
                # Exit conditions
                holding_time = current_time - long_position['entry_time']
                pnl_pct = (current_price - long_position['entry_price']) / long_position['entry_price']

                exit_signal = False
                exit_reason = None

                if pnl_pct <= -LONG_SL:
                    exit_signal = True
                    exit_reason = "Stop Loss"
                elif pnl_pct >= LONG_TP:
                    exit_signal = True
                    exit_reason = "Take Profit"
                elif holding_time >= MAX_HOLDING_TIME:
                    exit_signal = True
                    exit_reason = "Max Holding"

                if exit_signal:
                    exit_value = long_position['quantity'] * current_price
                    exit_cost = exit_value * TRANSACTION_COST

                    pnl_usd = exit_value - (long_position['quantity'] * long_position['entry_price']) - long_position['entry_cost'] - exit_cost

                    long_capital += pnl_usd

                    long_trades.append({
                        'entry_time': long_position['entry_time'],
                        'exit_time': current_time,
                        'entry_price': long_position['entry_price'],
                        'exit_price': current_price,
                        'quantity': long_position['quantity'],
                        'pnl_pct': pnl_pct,
                        'pnl_usd': pnl_usd,
                        'exit_reason': exit_reason,
                        'hours_held': holding_time.total_seconds() / 3600,
                        'side': 'LONG'
                    })

                    long_position = None

            # SHORT 전략
            if short_position is None:
                # Entry signal - XGBoost 예측 (3-class)
                try:
                    features = fold_df[self.short_features].iloc[idx:idx+1].values
                    if not np.isnan(features).any():
                        short_probs = self.model_short.predict_proba(features)[0]
                        short_prob = short_probs[2]  # Class 2 = SHORT
                    else:
                        short_prob = 0
                except:
                    short_prob = 0

                if short_prob >= XGB_THRESHOLD_SHORT:
                    position_value = short_capital * SHORT_POSITION_SIZE
                    quantity = position_value / current_price
                    entry_cost = position_value * TRANSACTION_COST

                    short_position = {
                        'entry_time': current_time,
                        'entry_price': current_price,
                        'quantity': quantity,
                        'entry_cost': entry_cost
                    }
            else:
                # Exit conditions
                holding_time = current_time - short_position['entry_time']
                pnl_pct = (short_position['entry_price'] - current_price) / short_position['entry_price']

                exit_signal = False
                exit_reason = None

                if pnl_pct <= -SHORT_SL:
                    exit_signal = True
                    exit_reason = "Stop Loss"
                elif pnl_pct >= SHORT_TP:
                    exit_signal = True
                    exit_reason = "Take Profit"
                elif holding_time >= MAX_HOLDING_TIME:
                    exit_signal = True
                    exit_reason = "Max Holding"

                if exit_signal:
                    exit_value = short_position['quantity'] * current_price
                    entry_value = short_position['quantity'] * short_position['entry_price']
                    exit_cost = exit_value * TRANSACTION_COST

                    pnl_usd = entry_value - exit_value - short_position['entry_cost'] - exit_cost

                    short_capital += pnl_usd

                    short_trades.append({
                        'entry_time': short_position['entry_time'],
                        'exit_time': current_time,
                        'entry_price': short_position['entry_price'],
                        'exit_price': current_price,
                        'quantity': short_position['quantity'],
                        'pnl_pct': pnl_pct,
                        'pnl_usd': pnl_usd,
                        'exit_reason': exit_reason,
                        'hours_held': holding_time.total_seconds() / 3600,
                        'side': 'SHORT'
                    })

                    short_position = None

        # 결과 계산
        long_trades_df = pd.DataFrame(long_trades) if long_trades else pd.DataFrame()
        short_trades_df = pd.DataFrame(short_trades) if short_trades else pd.DataFrame()

        total_capital = long_capital + short_capital
        total_pnl = total_capital - INITIAL_CAPITAL
        total_return_pct = (total_pnl / INITIAL_CAPITAL) * 100

        days = (fold_df['timestamp'].max() - fold_df['timestamp'].min()).days
        monthly_return_pct = (total_return_pct / days) * 30.42

        # Win rates
        long_wins = len(long_trades_df[long_trades_df['pnl_usd'] > 0]) if len(long_trades_df) > 0 else 0
        long_win_rate = (long_wins / len(long_trades_df) * 100) if len(long_trades_df) > 0 else 0

        short_wins = len(short_trades_df[short_trades_df['pnl_usd'] > 0]) if len(short_trades_df) > 0 else 0
        short_win_rate = (short_wins / len(short_trades_df) * 100) if len(short_trades_df) > 0 else 0

        total_trades = len(long_trades_df) + len(short_trades_df)
        total_wins = long_wins + short_wins
        overall_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0

        # Sharpe ratio (simplified)
        if total_trades > 0:
            all_trades = pd.concat([long_trades_df, short_trades_df])
            returns = all_trades['pnl_pct'].values
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        else:
            sharpe = 0

        results = {
            'fold_num': fold_num,
            'days': days,
            'total_return_pct': total_return_pct,
            'monthly_return_pct': monthly_return_pct,
            'total_trades': total_trades,
            'long_trades': len(long_trades_df),
            'short_trades': len(short_trades_df),
            'overall_win_rate': overall_win_rate,
            'long_win_rate': long_win_rate,
            'short_win_rate': short_win_rate,
            'sharpe_ratio': sharpe,
            'final_capital': total_capital,
            'long_capital': long_capital,
            'short_capital': short_capital
        }

        return results, long_trades_df, short_trades_df

    def run_validation(self):
        """Walk-forward 검증 실행"""
        print("\n" + "="*100)
        print("WALK-FORWARD VALIDATION")
        print("="*100)

        # 데이터 준비
        df = self.load_and_prepare_data()

        # Fold 생성
        folds = self.create_folds(df)

        print(f"\n📊 Created {self.n_folds} folds:")
        for fold in folds:
            print(f"   Fold {fold['fold_num']}: {fold['start_date'].date()} to {fold['end_date'].date()} "
                  f"({fold['days']} days, {fold['n_candles']} candles)")

        # 각 fold에서 백테스트
        all_results = []

        for fold in folds:
            print(f"\n{'='*100}")
            print(f"FOLD {fold['fold_num']}/{self.n_folds}")
            print(f"Period: {fold['start_date'].date()} to {fold['end_date'].date()}")
            print(f"{'='*100}")

            results, long_trades, short_trades = self.backtest_fold(
                fold['data'],
                fold['fold_num']
            )

            # Fold 정보 추가
            results['start_date'] = fold['start_date']
            results['end_date'] = fold['end_date']

            all_results.append(results)

            # 결과 출력
            print(f"\n✅ Fold {fold['fold_num']} Results:")
            print(f"   Monthly Return: {results['monthly_return_pct']:+.2f}%")
            print(f"   Total Trades: {results['total_trades']} (LONG: {results['long_trades']}, SHORT: {results['short_trades']})")
            print(f"   Win Rate: {results['overall_win_rate']:.1f}% (LONG: {results['long_win_rate']:.1f}%, SHORT: {results['short_win_rate']:.1f}%)")
            print(f"   Sharpe Ratio: {results['sharpe_ratio']:.2f}")
            print(f"   Final Capital: ${results['final_capital']:,.2f}")

        # 종합 분석
        self.analyze_results(all_results)

        # 결과 저장
        self.save_results(all_results)

        return all_results

    def analyze_results(self, results):
        """Walk-forward 결과 종합 분석"""
        print(f"\n{'='*100}")
        print("WALK-FORWARD VALIDATION - SUMMARY")
        print(f"{'='*100}")

        results_df = pd.DataFrame(results)

        # 통계
        mean_monthly_return = results_df['monthly_return_pct'].mean()
        std_monthly_return = results_df['monthly_return_pct'].std()
        min_monthly_return = results_df['monthly_return_pct'].min()
        max_monthly_return = results_df['monthly_return_pct'].max()

        mean_win_rate = results_df['overall_win_rate'].mean()
        mean_sharpe = results_df['sharpe_ratio'].mean()

        print(f"\n📊 Performance Across {self.n_folds} Folds:")
        print(f"   Monthly Return:")
        print(f"      Mean: {mean_monthly_return:+.2f}%")
        print(f"      Std Dev: {std_monthly_return:.2f}%")
        print(f"      Range: [{min_monthly_return:+.2f}%, {max_monthly_return:+.2f}%]")
        print(f"\n   Win Rate: {mean_win_rate:.1f}% (average)")
        print(f"   Sharpe Ratio: {mean_sharpe:.2f} (average)")

        # 일관성 분석
        consistency = (results_df['monthly_return_pct'] > 0).sum() / len(results_df) * 100
        print(f"\n   Consistency: {consistency:.1f}% of folds profitable")

        # 시간에 따른 트렌드
        print(f"\n📈 Performance Trend Over Time:")
        for i, row in results_df.iterrows():
            status = "✅" if row['monthly_return_pct'] > 0 else "❌"
            print(f"   Fold {row['fold_num']}: {row['monthly_return_pct']:+7.2f}% monthly  {status}")

        # 강건성 평가
        print(f"\n🎯 Robustness Assessment:")

        if consistency >= 80:
            robustness = "✅ EXCELLENT"
            message = "Strategy is highly consistent across time periods"
        elif consistency >= 60:
            robustness = "✅ GOOD"
            message = "Strategy shows good consistency with some variance"
        elif consistency >= 40:
            robustness = "⚠️ MODERATE"
            message = "Strategy has mixed results, needs improvement"
        else:
            robustness = "🚨 POOR"
            message = "Strategy lacks consistency, high risk"

        print(f"   {robustness}: {message}")
        print(f"   Coefficient of Variation: {(std_monthly_return / abs(mean_monthly_return)):.2f}")

        # 최악 시나리오
        worst_fold = results_df.loc[results_df['monthly_return_pct'].idxmin()]
        print(f"\n⚠️ Worst Performance:")
        print(f"   Fold {worst_fold['fold_num']}: {worst_fold['monthly_return_pct']:+.2f}% monthly")
        print(f"   Period: {worst_fold['start_date'].date()} to {worst_fold['end_date'].date()}")

        # 최고 시나리오
        best_fold = results_df.loc[results_df['monthly_return_pct'].idxmax()]
        print(f"\n🏆 Best Performance:")
        print(f"   Fold {best_fold['fold_num']}: {best_fold['monthly_return_pct']:+.2f}% monthly")
        print(f"   Period: {best_fold['start_date'].date()} to {best_fold['end_date'].date()}")

        print(f"\n{'='*100}")

    def save_results(self, results):
        """결과 저장"""
        results_df = pd.DataFrame(results)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # CSV 저장
        csv_file = RESULTS_DIR / f"walk_forward_validation_{timestamp}.csv"
        results_df.to_csv(csv_file, index=False)

        # 리포트 저장
        report_file = RESULTS_DIR / f"walk_forward_validation_report_{timestamp}.txt"

        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("WALK-FORWARD VALIDATION REPORT\n")
            f.write("="*80 + "\n\n")

            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Configuration: LONG 90% / SHORT 10%\n")
            f.write(f"Number of Folds: {self.n_folds}\n\n")

            f.write("SUMMARY STATISTICS\n")
            f.write("-"*80 + "\n\n")

            mean_monthly = results_df['monthly_return_pct'].mean()
            std_monthly = results_df['monthly_return_pct'].std()

            f.write(f"Monthly Return:\n")
            f.write(f"  Mean: {mean_monthly:+.2f}%\n")
            f.write(f"  Std Dev: {std_monthly:.2f}%\n")
            f.write(f"  Min: {results_df['monthly_return_pct'].min():+.2f}%\n")
            f.write(f"  Max: {results_df['monthly_return_pct'].max():+.2f}%\n\n")

            f.write(f"Win Rate: {results_df['overall_win_rate'].mean():.1f}%\n")
            f.write(f"Sharpe Ratio: {results_df['sharpe_ratio'].mean():.2f}\n\n")

            consistency = (results_df['monthly_return_pct'] > 0).sum() / len(results_df) * 100
            f.write(f"Consistency: {consistency:.1f}% of folds profitable\n\n")

            f.write("\nFOLD-BY-FOLD RESULTS\n")
            f.write("-"*80 + "\n\n")

            for _, row in results_df.iterrows():
                f.write(f"Fold {row['fold_num']}:\n")
                f.write(f"  Period: {row['start_date'].date()} to {row['end_date'].date()}\n")
                f.write(f"  Monthly Return: {row['monthly_return_pct']:+.2f}%\n")
                f.write(f"  Win Rate: {row['overall_win_rate']:.1f}%\n")
                f.write(f"  Total Trades: {row['total_trades']}\n")
                f.write(f"  Sharpe Ratio: {row['sharpe_ratio']:.2f}\n\n")

        print(f"\n✅ Results saved:")
        print(f"   {csv_file}")
        print(f"   {report_file}")

if __name__ == "__main__":
    validator = WalkForwardValidator(n_folds=5)
    results = validator.run_validation()
