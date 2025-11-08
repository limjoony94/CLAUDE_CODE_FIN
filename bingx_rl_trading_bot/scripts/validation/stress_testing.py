"""
Stress Testing - 극단적 시장 상황 시뮬레이션

목적:
    - 급격한 가격 변동 상황에서 전략 성능 확인
    - 최대 손실 한계 파악
    - 리스크 관리 효과성 검증

시나리오:
    1. Flash Crash (-10% / 1시간)
    2. Flash Rally (+10% / 1시간)
    3. High Volatility (±5% / 15분 반복)
    4. Sideways Market (±1% / 24시간)
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

# Configuration
INITIAL_CAPITAL = 10000
LONG_ALLOCATION = 0.90
SHORT_ALLOCATION = 0.10

LONG_THRESHOLD = 0.7
SHORT_THRESHOLD = 0.4

LONG_SL = 0.01
LONG_TP = 0.03
SHORT_SL = 0.015
SHORT_TP = 0.06

LONG_POSITION_SIZE = 0.95
SHORT_POSITION_SIZE = 0.95

MAX_HOLDING_HOURS = 4
TRANSACTION_COST = 0.0002

class StressTester:
    """극단적 시장 상황에서 전략 테스트"""

    def __init__(self):
        self.adv_features = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)

        # Load models
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

        print("✅ Models loaded")
        print(f"   LONG: {len(self.long_features)} features")
        print(f"   SHORT: {len(self.short_features)} features")

    def load_base_data(self):
        """기본 데이터 로드"""
        df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_max.csv")

        if 'timestamp' in df.columns and df['timestamp'].dtype == 'object':
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        df = calculate_features(df)
        df = self.adv_features.calculate_all_features(df)
        df = df.ffill().dropna()

        return df

    def create_flash_crash(self, df, start_idx=1000, crash_pct=-0.10):
        """Flash crash 시나리오 생성: 급격한 하락"""
        scenario_df = df.copy()

        # 1시간 (12 candles) 동안 crash_pct% 하락
        crash_candles = 12
        price_multiplier = 1 + crash_pct

        for i in range(crash_candles):
            idx = start_idx + i
            if idx < len(scenario_df):
                progress = (i + 1) / crash_candles
                current_multiplier = 1 + (crash_pct * progress)

                scenario_df.loc[idx, 'open'] *= current_multiplier
                scenario_df.loc[idx, 'high'] *= current_multiplier
                scenario_df.loc[idx, 'low'] *= current_multiplier
                scenario_df.loc[idx, 'close'] *= current_multiplier

        return scenario_df

    def create_flash_rally(self, df, start_idx=1000, rally_pct=0.10):
        """Flash rally 시나리오 생성: 급격한 상승"""
        scenario_df = df.copy()

        # 1시간 (12 candles) 동안 rally_pct% 상승
        rally_candles = 12
        price_multiplier = 1 + rally_pct

        for i in range(rally_candles):
            idx = start_idx + i
            if idx < len(scenario_df):
                progress = (i + 1) / rally_candles
                current_multiplier = 1 + (rally_pct * progress)

                scenario_df.loc[idx, 'open'] *= current_multiplier
                scenario_df.loc[idx, 'high'] *= current_multiplier
                scenario_df.loc[idx, 'low'] *= current_multiplier
                scenario_df.loc[idx, 'close'] *= current_multiplier

        return scenario_df

    def create_high_volatility(self, df, start_idx=1000, duration_candles=288):
        """High volatility 시나리오: ±5% 변동 반복 (24시간)"""
        scenario_df = df.copy()

        # 15분마다 ±5% 변동
        swing_candles = 3  # 15분 = 3 candles
        swing_pct = 0.05

        for i in range(0, duration_candles, swing_candles):
            direction = 1 if (i // swing_candles) % 2 == 0 else -1
            change = swing_pct * direction

            for j in range(swing_candles):
                idx = start_idx + i + j
                if idx < len(scenario_df):
                    progress = (j + 1) / swing_candles
                    multiplier = 1 + (change * progress)

                    scenario_df.loc[idx, 'open'] *= multiplier
                    scenario_df.loc[idx, 'high'] *= multiplier
                    scenario_df.loc[idx, 'low'] *= multiplier
                    scenario_df.loc[idx, 'close'] *= multiplier

        return scenario_df

    def create_sideways(self, df, start_idx=1000, duration_candles=288):
        """Sideways 시나리오: ±1% 횡보 (24시간)"""
        scenario_df = df.copy()

        base_price = scenario_df.iloc[start_idx]['close']

        for i in range(duration_candles):
            idx = start_idx + i
            if idx < len(scenario_df):
                # Random walk within ±1%
                random_change = np.random.uniform(-0.01, 0.01)
                new_price = base_price * (1 + random_change)

                scenario_df.loc[idx, 'close'] = new_price
                scenario_df.loc[idx, 'open'] = scenario_df.loc[idx-1, 'close'] if idx > 0 else new_price
                scenario_df.loc[idx, 'high'] = new_price * 1.002
                scenario_df.loc[idx, 'low'] = new_price * 0.998

        return scenario_df

    def backtest_scenario(self, df, scenario_name):
        """특정 시나리오에서 백테스트 실행"""
        long_capital = INITIAL_CAPITAL * LONG_ALLOCATION
        short_capital = INITIAL_CAPITAL * SHORT_ALLOCATION

        long_position = None
        short_position = None

        long_trades = []
        short_trades = []

        for idx in range(len(df)):
            if idx < 100:
                continue

            current_time = df['timestamp'].iloc[idx]
            current_price = df['close'].iloc[idx]

            # LONG 전략
            if long_position is None:
                try:
                    features = df[self.long_features].iloc[idx:idx+1].values
                    if not np.isnan(features).any():
                        long_prob = self.model_long.predict_proba(features)[0][1]
                    else:
                        long_prob = 0
                except:
                    long_prob = 0

                if long_prob >= LONG_THRESHOLD:
                    position_value = long_capital * LONG_POSITION_SIZE
                    quantity = position_value / current_price
                    entry_cost = position_value * TRANSACTION_COST

                    long_position = {
                        'entry_idx': idx,
                        'entry_time': current_time,
                        'entry_price': current_price,
                        'quantity': quantity,
                        'entry_cost': entry_cost
                    }
            else:
                pnl_pct = (current_price - long_position['entry_price']) / long_position['entry_price']
                hours_held = (idx - long_position['entry_idx']) * 5 / 60

                exit_signal = False
                exit_reason = None

                if pnl_pct <= -LONG_SL:
                    exit_signal = True
                    exit_reason = "Stop Loss"
                elif pnl_pct >= LONG_TP:
                    exit_signal = True
                    exit_reason = "Take Profit"
                elif hours_held >= MAX_HOLDING_HOURS:
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
                        'pnl_pct': pnl_pct,
                        'pnl_usd': pnl_usd,
                        'exit_reason': exit_reason,
                        'hours_held': hours_held,
                        'side': 'LONG'
                    })

                    long_position = None

            # SHORT 전략
            if short_position is None:
                try:
                    features = df[self.short_features].iloc[idx:idx+1].values
                    if not np.isnan(features).any():
                        short_probs = self.model_short.predict_proba(features)[0]
                        short_prob = short_probs[2]
                    else:
                        short_prob = 0
                except:
                    short_prob = 0

                if short_prob >= SHORT_THRESHOLD:
                    position_value = short_capital * SHORT_POSITION_SIZE
                    quantity = position_value / current_price
                    entry_cost = position_value * TRANSACTION_COST

                    short_position = {
                        'entry_idx': idx,
                        'entry_time': current_time,
                        'entry_price': current_price,
                        'quantity': quantity,
                        'entry_cost': entry_cost
                    }
            else:
                pnl_pct = (short_position['entry_price'] - current_price) / short_position['entry_price']
                hours_held = (idx - short_position['entry_idx']) * 5 / 60

                exit_signal = False
                exit_reason = None

                if pnl_pct <= -SHORT_SL:
                    exit_signal = True
                    exit_reason = "Stop Loss"
                elif pnl_pct >= SHORT_TP:
                    exit_signal = True
                    exit_reason = "Take Profit"
                elif hours_held >= MAX_HOLDING_HOURS:
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
                        'pnl_pct': pnl_pct,
                        'pnl_usd': pnl_usd,
                        'exit_reason': exit_reason,
                        'hours_held': hours_held,
                        'side': 'SHORT'
                    })

                    short_position = None

        # 결과 계산
        total_capital = long_capital + short_capital
        total_pnl = total_capital - INITIAL_CAPITAL
        total_return_pct = (total_pnl / INITIAL_CAPITAL) * 100

        # Win rates
        long_wins = len([t for t in long_trades if t['pnl_usd'] > 0])
        long_win_rate = (long_wins / len(long_trades) * 100) if long_trades else 0

        short_wins = len([t for t in short_trades if t['pnl_usd'] > 0])
        short_win_rate = (short_wins / len(short_trades) * 100) if short_trades else 0

        total_trades = len(long_trades) + len(short_trades)
        total_wins = long_wins + short_wins
        overall_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0

        # Max drawdown
        if long_trades or short_trades:
            all_trades = long_trades + short_trades
            sorted_trades = sorted(all_trades, key=lambda x: x['exit_time'])

            capital_series = [INITIAL_CAPITAL]
            for trade in sorted_trades:
                capital_series.append(capital_series[-1] + trade['pnl_usd'])

            capital_series = pd.Series(capital_series)
            cummax = capital_series.cummax()
            drawdown = (capital_series - cummax) / cummax
            max_dd = drawdown.min() * 100
        else:
            max_dd = 0

        results = {
            'scenario': scenario_name,
            'final_capital': total_capital,
            'total_return_pct': total_return_pct,
            'total_trades': total_trades,
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
            'overall_win_rate': overall_win_rate,
            'long_win_rate': long_win_rate,
            'short_win_rate': short_win_rate,
            'max_drawdown_pct': max_dd,
            'long_trades_list': long_trades,
            'short_trades_list': short_trades
        }

        return results

    def run_all_scenarios(self):
        """모든 스트레스 테스트 시나리오 실행"""
        print("\n" + "="*100)
        print("STRESS TESTING - Extreme Market Scenarios")
        print("="*100)

        # 기본 데이터 로드
        print("\n📥 Loading base data...")
        base_df = self.load_base_data()

        print(f"   Loaded {len(base_df)} candles")
        print(f"   Period: {base_df['timestamp'].min()} to {base_df['timestamp'].max()}")

        scenarios = []

        # Scenario 1: Flash Crash (-10% / 1시간)
        print("\n" + "="*100)
        print("SCENARIO 1: Flash Crash (-10% / 1 hour)")
        print("="*100)

        crash_df = self.create_flash_crash(base_df, start_idx=1000, crash_pct=-0.10)
        crash_df = calculate_features(crash_df)
        crash_df = self.adv_features.calculate_all_features(crash_df)
        crash_df = crash_df.ffill().dropna()

        crash_results = self.backtest_scenario(crash_df, "Flash Crash (-10%)")
        scenarios.append(crash_results)

        print(f"\n📊 Results:")
        print(f"   Final Capital: ${crash_results['final_capital']:,.2f}")
        print(f"   Return: {crash_results['total_return_pct']:+.2f}%")
        print(f"   Win Rate: {crash_results['overall_win_rate']:.1f}%")
        print(f"   Max Drawdown: {crash_results['max_drawdown_pct']:.2f}%")
        print(f"   Total Trades: {crash_results['total_trades']}")

        # Scenario 2: Flash Rally (+10% / 1시간)
        print("\n" + "="*100)
        print("SCENARIO 2: Flash Rally (+10% / 1 hour)")
        print("="*100)

        rally_df = self.create_flash_rally(base_df, start_idx=1000, rally_pct=0.10)
        rally_df = calculate_features(rally_df)
        rally_df = self.adv_features.calculate_all_features(rally_df)
        rally_df = rally_df.ffill().dropna()

        rally_results = self.backtest_scenario(rally_df, "Flash Rally (+10%)")
        scenarios.append(rally_results)

        print(f"\n📊 Results:")
        print(f"   Final Capital: ${rally_results['final_capital']:,.2f}")
        print(f"   Return: {rally_results['total_return_pct']:+.2f}%")
        print(f"   Win Rate: {rally_results['overall_win_rate']:.1f}%")
        print(f"   Max Drawdown: {rally_results['max_drawdown_pct']:.2f}%")
        print(f"   Total Trades: {rally_results['total_trades']}")

        # Scenario 3: High Volatility (±5% / 15분, 24시간)
        print("\n" + "="*100)
        print("SCENARIO 3: High Volatility (±5% swings, 24 hours)")
        print("="*100)

        volatile_df = self.create_high_volatility(base_df, start_idx=1000, duration_candles=288)
        volatile_df = calculate_features(volatile_df)
        volatile_df = self.adv_features.calculate_all_features(volatile_df)
        volatile_df = volatile_df.ffill().dropna()

        volatile_results = self.backtest_scenario(volatile_df, "High Volatility (±5%)")
        scenarios.append(volatile_results)

        print(f"\n📊 Results:")
        print(f"   Final Capital: ${volatile_results['final_capital']:,.2f}")
        print(f"   Return: {volatile_results['total_return_pct']:+.2f}%")
        print(f"   Win Rate: {volatile_results['overall_win_rate']:.1f}%")
        print(f"   Max Drawdown: {volatile_results['max_drawdown_pct']:.2f}%")
        print(f"   Total Trades: {volatile_results['total_trades']}")

        # Scenario 4: Sideways Market (±1%, 24시간)
        print("\n" + "="*100)
        print("SCENARIO 4: Sideways Market (±1%, 24 hours)")
        print("="*100)

        sideways_df = self.create_sideways(base_df, start_idx=1000, duration_candles=288)
        sideways_df = calculate_features(sideways_df)
        sideways_df = self.adv_features.calculate_all_features(sideways_df)
        sideways_df = sideways_df.ffill().dropna()

        sideways_results = self.backtest_scenario(sideways_df, "Sideways Market (±1%)")
        scenarios.append(sideways_results)

        print(f"\n📊 Results:")
        print(f"   Final Capital: ${sideways_results['final_capital']:,.2f}")
        print(f"   Return: {sideways_results['total_return_pct']:+.2f}%")
        print(f"   Win Rate: {sideways_results['overall_win_rate']:.1f}%")
        print(f"   Max Drawdown: {sideways_results['max_drawdown_pct']:.2f}%")
        print(f"   Total Trades: {sideways_results['total_trades']}")

        # Summary
        self.print_summary(scenarios)

        # Save results
        self.save_results(scenarios)

        return scenarios

    def print_summary(self, scenarios):
        """결과 요약 출력"""
        print("\n" + "="*100)
        print("STRESS TESTING SUMMARY")
        print("="*100)

        results_df = pd.DataFrame([
            {
                'Scenario': s['scenario'],
                'Return %': s['total_return_pct'],
                'Trades': s['total_trades'],
                'Win Rate %': s['overall_win_rate'],
                'Max DD %': s['max_drawdown_pct'],
                'Final Capital': s['final_capital']
            }
            for s in scenarios
        ])

        print(f"\n{results_df.to_string(index=False)}")

        # 리스크 평가
        worst_return = min(s['total_return_pct'] for s in scenarios)
        worst_dd = min(s['max_drawdown_pct'] for s in scenarios)

        print(f"\n🚨 Risk Assessment:")
        print(f"   Worst Return: {worst_return:+.2f}%")
        print(f"   Worst Drawdown: {worst_dd:.2f}%")

        if worst_return >= -5:
            print(f"   Status: ✅ ROBUST (worst case ≥ -5%)")
        elif worst_return >= -10:
            print(f"   Status: ⚠️ ACCEPTABLE (worst case -5% to -10%)")
        else:
            print(f"   Status: 🚨 HIGH RISK (worst case < -10%)")

        print("="*100)

    def save_results(self, scenarios):
        """결과 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # CSV 저장
        results_df = pd.DataFrame([
            {
                'scenario': s['scenario'],
                'final_capital': s['final_capital'],
                'return_pct': s['total_return_pct'],
                'total_trades': s['total_trades'],
                'long_trades': s['long_trades'],
                'short_trades': s['short_trades'],
                'win_rate': s['overall_win_rate'],
                'long_win_rate': s['long_win_rate'],
                'short_win_rate': s['short_win_rate'],
                'max_drawdown': s['max_drawdown_pct']
            }
            for s in scenarios
        ])

        csv_file = RESULTS_DIR / f"stress_testing_{timestamp}.csv"
        results_df.to_csv(csv_file, index=False)

        # 리포트 저장
        report_file = RESULTS_DIR / f"stress_testing_report_{timestamp}.txt"

        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("STRESS TESTING REPORT\n")
            f.write("="*80 + "\n\n")

            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Configuration: LONG 90% / SHORT 10%\n\n")

            for s in scenarios:
                f.write(f"\n{s['scenario']}:\n")
                f.write(f"  Return: {s['total_return_pct']:+.2f}%\n")
                f.write(f"  Win Rate: {s['overall_win_rate']:.1f}%\n")
                f.write(f"  Max Drawdown: {s['max_drawdown_pct']:.2f}%\n")
                f.write(f"  Total Trades: {s['total_trades']}\n")

            worst_return = min(s['total_return_pct'] for s in scenarios)
            worst_dd = min(s['max_drawdown_pct'] for s in scenarios)

            f.write(f"\nRisk Assessment:\n")
            f.write(f"  Worst Return: {worst_return:+.2f}%\n")
            f.write(f"  Worst Drawdown: {worst_dd:.2f}%\n")

        print(f"\n✅ Results saved:")
        print(f"   {csv_file}")
        print(f"   {report_file}")

if __name__ == "__main__":
    tester = StressTester()
    results = tester.run_all_scenarios()
