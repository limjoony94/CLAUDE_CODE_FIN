"""백테스팅 엔진"""

from typing import Dict, Any, List
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


class BacktestEngine:
    """
    강화학습 모델 백테스팅 엔진
    """

    def __init__(self, env, agent):
        """
        Args:
            env: 거래 환경
            agent: 강화학습 에이전트
        """
        self.env = env
        self.agent = agent

        self.results = {
            'trades': [],
            'equity_curve': [],
            'metrics': {}
        }

    def run_backtest(
        self,
        n_episodes: int = 1,
        deterministic: bool = True,
        render: bool = False
    ) -> Dict[str, Any]:
        """
        백테스트 실행

        Args:
            n_episodes: 실행 에피소드 수
            deterministic: 결정적 행동 여부
            render: 렌더링 여부

        Returns:
            백테스트 결과
        """
        logger.info(f"Running backtest for {n_episodes} episodes...")

        all_trades = []
        all_equity_curves = []

        for episode in range(n_episodes):
            obs, info = self.env.reset()
            done = False
            episode_trades = []
            episode_equity = []

            step = 0
            while not done:
                # 행동 예측
                action, _ = self.agent.predict(obs, deterministic=deterministic)

                # 환경 스텝
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # 거래 기록
                if info['trade_count'] > len(episode_trades):
                    trade_info = {
                        'step': step,
                        'price': info['current_price'],
                        'position': info['position'],
                        'balance': info['balance'],
                        'pnl': info['total_pnl']
                    }
                    episode_trades.append(trade_info)

                # 자산 기록
                episode_equity.append({
                    'step': step,
                    'portfolio_value': info['portfolio_value'],
                    'balance': info['balance'],
                    'unrealized_pnl': info['unrealized_pnl']
                })

                step += 1

                if render:
                    self.env.render()

            all_trades.extend(episode_trades)
            all_equity_curves.extend(episode_equity)

            logger.info(f"Episode {episode + 1} completed - "
                       f"Trades: {len(episode_trades)}, "
                       f"Final PnL: {info['total_pnl']:.2f}")

        # 결과 저장
        self.results['trades'] = all_trades
        self.results['equity_curve'] = all_equity_curves

        # 성과 지표 계산
        self.results['metrics'] = self._calculate_metrics()

        logger.info("Backtest completed")
        return self.results

    def _calculate_metrics(self) -> Dict[str, float]:
        """
        성과 지표 계산

        Returns:
            성과 지표 딕셔너리
        """
        if not self.results['equity_curve']:
            return {}

        equity_df = pd.DataFrame(self.results['equity_curve'])

        # 총 수익률
        initial_value = equity_df['portfolio_value'].iloc[0]
        final_value = equity_df['portfolio_value'].iloc[-1]
        total_return = (final_value - initial_value) / initial_value

        # 최대 낙폭 (Maximum Drawdown)
        rolling_max = equity_df['portfolio_value'].cummax()
        drawdown = (equity_df['portfolio_value'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # 변동성
        returns = equity_df['portfolio_value'].pct_change().dropna()
        volatility = returns.std()

        # 샤프 비율 (무위험 수익률 = 0 가정)
        sharpe_ratio = (returns.mean() / volatility * np.sqrt(252)) if volatility > 0 else 0

        # 승률
        trades_df = pd.DataFrame(self.results['trades'])
        if len(trades_df) > 0:
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            win_rate = winning_trades / len(trades_df)
        else:
            win_rate = 0

        metrics = {
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'total_trades': len(trades_df),
            'final_portfolio_value': final_value
        }

        return metrics

    def print_summary(self) -> None:
        """백테스트 결과 요약 출력"""
        if not self.results['metrics']:
            logger.warning("No backtest results available")
            return

        metrics = self.results['metrics']

        print("\n" + "="*60)
        print("BACKTEST RESULTS SUMMARY")
        print("="*60)
        print(f"Total Return:          {metrics['total_return']*100:>10.2f}%")
        print(f"Max Drawdown:          {metrics['max_drawdown']*100:>10.2f}%")
        print(f"Sharpe Ratio:          {metrics['sharpe_ratio']:>10.2f}")
        print(f"Volatility:            {metrics['volatility']*100:>10.2f}%")
        print(f"Win Rate:              {metrics['win_rate']*100:>10.2f}%")
        print(f"Total Trades:          {metrics['total_trades']:>10}")
        print(f"Final Portfolio Value: ${metrics['final_portfolio_value']:>10.2f}")
        print("="*60 + "\n")

    def save_results(self, filename: str = None) -> None:
        """
        결과 저장

        Args:
            filename: 저장 파일명
        """
        if filename is None:
            filename = "backtest_results.csv"

        project_root = Path(__file__).parent.parent.parent
        save_dir = project_root / 'data' / 'logs'
        save_dir.mkdir(parents=True, exist_ok=True)

        filepath = save_dir / filename

        # 자산 곡선 저장
        equity_df = pd.DataFrame(self.results['equity_curve'])
        equity_df.to_csv(filepath, index=False)

        # 메트릭 저장
        metrics_file = save_dir / filename.replace('.csv', '_metrics.txt')
        with open(metrics_file, 'w') as f:
            for key, value in self.results['metrics'].items():
                f.write(f"{key}: {value}\n")

        logger.info(f"Results saved to {filepath}")

    def plot_results(self) -> None:
        """결과 시각화 (matplotlib 필요)"""
        try:
            import matplotlib.pyplot as plt

            equity_df = pd.DataFrame(self.results['equity_curve'])

            fig, axes = plt.subplots(2, 1, figsize=(12, 8))

            # 자산 곡선
            axes[0].plot(equity_df['step'], equity_df['portfolio_value'])
            axes[0].set_title('Portfolio Value Over Time')
            axes[0].set_xlabel('Step')
            axes[0].set_ylabel('Portfolio Value (USDT)')
            axes[0].grid(True)

            # 손익
            axes[1].plot(equity_df['step'], equity_df['unrealized_pnl'])
            axes[1].set_title('Unrealized PnL Over Time')
            axes[1].set_xlabel('Step')
            axes[1].set_ylabel('Unrealized PnL (USDT)')
            axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
            axes[1].grid(True)

            plt.tight_layout()
            plt.show()

        except ImportError:
            logger.warning("matplotlib not installed, cannot plot results")
