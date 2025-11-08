"""
Stress Testing (Quick Version) - 극단적 시장 상황 시뮬레이션

최적화:
    - 기존 백테스트 결과 활용
    - 시나리오별 손실 시뮬레이션
    - 빠른 리스크 평가
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

# 기존 백테스트 결과 로드
BACKTEST_FILE = RESULTS_DIR / "backtest_90_10_trades.csv"

class QuickStressTester:
    """빠른 스트레스 테스트 - 기존 거래 데이터 활용"""

    def __init__(self):
        print("="*100)
        print("STRESS TESTING (Quick Analysis)")
        print("="*100)

        # 기존 거래 데이터 로드
        self.trades_df = pd.read_csv(BACKTEST_FILE)

        print(f"\n✅ Loaded {len(self.trades_df)} trades from backtest")
        print(f"   Strategies: LONG and SHORT combined")

    def scenario_flash_crash(self):
        """
        Flash Crash 시나리오: -10% 급락

        영향:
        - LONG 포지션: Stop Loss (-1%) 트리거
        - SHORT 포지션: 큰 이익 (하지만 진입 조건 희박)
        """
        print("\n" + "="*100)
        print("SCENARIO 1: Flash Crash (-10% in 1 hour)")
        print("="*100)

        long_trades = self.trades_df[self.trades_df['strategy'] == 'LONG']
        short_trades = self.trades_df[self.trades_df['strategy'] == 'SHORT']

        # LONG: 모든 포지션 Stop Loss로 청산 (-1%)
        long_sl_loss = len(long_trades) * (-0.01)  # 각 포지션 -1% 손실

        # SHORT: 일부는 Take Profit (+6%), 대부분은 진입 안 함
        short_tp_gain = len(short_trades) * 0.06 * 0.3  # 30%만 진입 가정

        # 전체 손실 (LONG 90%, SHORT 10%)
        total_impact = (long_sl_loss * 0.90) + (short_tp_gain * 0.10)

        print(f"\n📊 Impact Analysis:")
        print(f"   LONG Positions: {len(long_trades)} → All Stop Loss (-1%)")
        print(f"   SHORT Positions: Limited entries → Minimal benefit")
        print(f"   Estimated Loss: {total_impact:.2%}")
        print(f"   Final Capital: ${10000 * (1 + total_impact):,.2f}")

        if total_impact >= -0.05:
            status = "✅ ACCEPTABLE"
        elif total_impact >= -0.10:
            status = "⚠️ MODERATE RISK"
        else:
            status = "🚨 HIGH RISK"

        print(f"   Risk Assessment: {status}")

        return {
            'scenario': 'Flash Crash (-10%)',
            'impact_pct': total_impact * 100,
            'final_capital': 10000 * (1 + total_impact),
            'status': status
        }

    def scenario_flash_rally(self):
        """
        Flash Rally 시나리오: +10% 급등

        영향:
        - LONG 포지션: Take Profit (+3%) 달성
        - SHORT 포지션: Stop Loss (-1.5%) 트리거
        """
        print("\n" + "="*100)
        print("SCENARIO 2: Flash Rally (+10% in 1 hour)")
        print("="*100)

        long_trades = self.trades_df[self.trades_df['strategy'] == 'LONG']
        short_trades = self.trades_df[self.trades_df['strategy'] == 'SHORT']

        # LONG: 모든 포지션 Take Profit (+3%)
        long_tp_gain = len(long_trades) * 0.03

        # SHORT: 모든 포지션 Stop Loss (-1.5%)
        short_sl_loss = len(short_trades) * (-0.015)

        # 전체 수익 (LONG 90%, SHORT 10%)
        total_impact = (long_tp_gain * 0.90) + (short_sl_loss * 0.10)

        print(f"\n📊 Impact Analysis:")
        print(f"   LONG Positions: {len(long_trades)} → All Take Profit (+3%)")
        print(f"   SHORT Positions: {len(short_trades)} → All Stop Loss (-1.5%)")
        print(f"   Estimated Gain: {total_impact:.2%}")
        print(f"   Final Capital: ${10000 * (1 + total_impact):,.2f}")

        status = "✅ PROFITABLE"
        print(f"   Risk Assessment: {status}")

        return {
            'scenario': 'Flash Rally (+10%)',
            'impact_pct': total_impact * 100,
            'final_capital': 10000 * (1 + total_impact),
            'status': status
        }

    def scenario_high_volatility(self):
        """
        High Volatility 시나리오: ±5% 변동 반복

        영향:
        - 많은 False signals
        - Stop Loss 연속 발동
        - 낮은 승률
        """
        print("\n" + "="*100)
        print("SCENARIO 3: High Volatility (±5% swings, 24 hours)")
        print("="*100)

        # 승률 하락 시뮬레이션
        baseline_win_rate = 0.598
        volatile_win_rate = baseline_win_rate * 0.7  # 30% 성능 저하

        long_trades = self.trades_df[self.trades_df['strategy'] == 'LONG']
        short_trades = self.trades_df[self.trades_df['strategy'] == 'SHORT']

        # 승률 감소로 인한 손실
        baseline_return = (len(long_trades) * 0.03 * 0.691) + (len(short_trades) * 0.06 * 0.52)
        volatile_return = baseline_return * 0.5  # 50% 성능 저하

        total_impact = (volatile_return - baseline_return) / 100

        print(f"\n📊 Impact Analysis:")
        print(f"   Baseline Win Rate: {baseline_win_rate:.1%}")
        print(f"   Volatile Win Rate: {volatile_win_rate:.1%} (30% degradation)")
        print(f"   Performance Impact: {total_impact:.2%}")
        print(f"   Final Capital: ${10000 * (1 + total_impact):,.2f}")

        if total_impact >= -0.10:
            status = "✅ ACCEPTABLE"
        elif total_impact >= -0.20:
            status = "⚠️ MODERATE RISK"
        else:
            status = "🚨 HIGH RISK"

        print(f"   Risk Assessment: {status}")

        return {
            'scenario': 'High Volatility (±5%)',
            'impact_pct': total_impact * 100,
            'final_capital': 10000 * (1 + total_impact),
            'status': status
        }

    def scenario_sideways(self):
        """
        Sideways 시나리오: ±1% 횡보

        영향:
        - 진입 신호 감소
        - Max Holding으로 청산
        - 소폭 수익 또는 손실
        """
        print("\n" + "="*100)
        print("SCENARIO 4: Sideways Market (±1%, 24 hours)")
        print("="*100)

        long_trades = self.trades_df[self.trades_df['strategy'] == 'LONG']

        # 거래 빈도 감소 (50%)
        reduced_trades = len(long_trades) * 0.5

        # Max Holding 청산 많음 (평균 수익 감소)
        avg_return = 0.005  # 0.5% 평균

        total_impact = (reduced_trades * avg_return * 0.90) / 100

        print(f"\n📊 Impact Analysis:")
        print(f"   Trade Frequency: Reduced by 50%")
        print(f"   Avg Return per Trade: {avg_return:.2%}")
        print(f"   Estimated Impact: {total_impact:.2%}")
        print(f"   Final Capital: ${10000 * (1 + total_impact):,.2f}")

        status = "✅ LOW IMPACT"
        print(f"   Risk Assessment: {status}")

        return {
            'scenario': 'Sideways Market (±1%)',
            'impact_pct': total_impact * 100,
            'final_capital': 10000 * (1 + total_impact),
            'status': status
        }

    def run_all_scenarios(self):
        """모든 시나리오 실행 및 요약"""
        results = []

        results.append(self.scenario_flash_crash())
        results.append(self.scenario_flash_rally())
        results.append(self.scenario_high_volatility())
        results.append(self.scenario_sideways())

        # Summary
        self.print_summary(results)
        self.save_results(results)

        return results

    def print_summary(self, results):
        """결과 요약"""
        print("\n" + "="*100)
        print("STRESS TESTING SUMMARY")
        print("="*100)

        print(f"\n{'Scenario':<30} {'Impact %':>12} {'Final Capital':>15} {'Status':>20}")
        print("-"*100)

        for r in results:
            print(f"{r['scenario']:<30} {r['impact_pct']:>12.2f}% {r['final_capital']:>15,.2f} {r['status']:>20}")

        # 최악 시나리오
        worst = min(results, key=lambda x: x['impact_pct'])

        print(f"\n🚨 Worst Case Scenario:")
        print(f"   {worst['scenario']}: {worst['impact_pct']:+.2f}%")
        print(f"   Final Capital: ${worst['final_capital']:,.2f}")

        if worst['impact_pct'] >= -5:
            overall = "✅ ROBUST"
            msg = "Strategy handles extreme conditions well"
        elif worst['impact_pct'] >= -10:
            overall = "⚠️ ACCEPTABLE"
            msg = "Strategy has moderate risk in extreme conditions"
        else:
            overall = "🚨 HIGH RISK"
            msg = "Strategy vulnerable to extreme market moves"

        print(f"\n🎯 Overall Risk Assessment: {overall}")
        print(f"   {msg}")

        print("="*100)

    def save_results(self, results):
        """결과 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # CSV
        df = pd.DataFrame(results)
        csv_file = RESULTS_DIR / f"stress_testing_quick_{timestamp}.csv"
        df.to_csv(csv_file, index=False)

        # Report
        report_file = RESULTS_DIR / f"stress_testing_report_{timestamp}.txt"

        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("STRESS TESTING REPORT (Quick Analysis)\n")
            f.write("="*80 + "\n\n")

            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Configuration: LONG 90% / SHORT 10%\n")
            f.write(f"Methodology: Scenario-based impact analysis\n\n")

            f.write("RESULTS\n")
            f.write("-"*80 + "\n\n")

            for r in results:
                f.write(f"{r['scenario']}:\n")
                f.write(f"  Impact: {r['impact_pct']:+.2f}%\n")
                f.write(f"  Final Capital: ${r['final_capital']:,.2f}\n")
                f.write(f"  Status: {r['status']}\n\n")

            worst = min(results, key=lambda x: x['impact_pct'])
            f.write(f"\nWorst Case:\n")
            f.write(f"  {worst['scenario']}: {worst['impact_pct']:+.2f}%\n")
            f.write(f"  Final Capital: ${worst['final_capital']:,.2f}\n")

        print(f"\n✅ Results saved:")
        print(f"   {csv_file}")
        print(f"   {report_file}")

if __name__ == "__main__":
    tester = QuickStressTester()
    results = tester.run_all_scenarios()
