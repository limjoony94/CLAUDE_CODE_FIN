"""
Results Visualization for Random Masking Candle Predictor

Visualizations:
1. Equity curve with drawdowns
2. Trade distribution and P&L
3. Prediction accuracy
4. Attention heatmaps
5. Performance metrics dashboard
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from pathlib import Path
from loguru import logger


class ResultsVisualizer:
    """
    Comprehensive visualization for backtest results

    Creates:
    - Equity curves
    - Trade analysis
    - Prediction analysis
    - Attention visualizations
    - Performance dashboards
    """

    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """
        Initialize visualizer

        Args:
            style: Matplotlib style
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')

        sns.set_palette("husl")

        self.figsize_single = (12, 6)
        self.figsize_multi = (15, 10)

        logger.info(f"Initialized ResultsVisualizer with style: {style}")

    def plot_equity_curve(
        self,
        equity_curve: pd.Series,
        trades: Optional[List[Dict]] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot equity curve with drawdowns and trade markers

        Args:
            equity_curve: Equity over time
            trades: List of trade dicts (optional)
            save_path: Path to save figure (optional)
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize_single, sharex=True)

        # Equity curve
        ax1.plot(equity_curve.index, equity_curve.values, label='Equity', linewidth=2)
        ax1.set_ylabel('Equity ($)')
        ax1.set_title('Equity Curve')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Mark trades
        if trades:
            wins = [t for t in trades if t.get('pnl', 0) > 0]
            losses = [t for t in trades if t.get('pnl', 0) < 0]

            if wins:
                win_times = [t['exit_time'] for t in wins]
                win_equity = [equity_curve.loc[t] for t in win_times if t in equity_curve.index]
                ax1.scatter(win_times[:len(win_equity)], win_equity, color='green', marker='^',
                           s=100, alpha=0.6, label='Wins', zorder=5)

            if losses:
                loss_times = [t['exit_time'] for t in losses]
                loss_equity = [equity_curve.loc[t] for t in loss_times if t in equity_curve.index]
                ax1.scatter(loss_times[:len(loss_equity)], loss_equity, color='red', marker='v',
                           s=100, alpha=0.6, label='Losses', zorder=5)

            ax1.legend()

        # Drawdown
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max

        ax2.fill_between(drawdown.index, drawdown.values, 0,
                         color='red', alpha=0.3, label='Drawdown')
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Time')
        ax2.set_title('Drawdown')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Format y-axis as percentage
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved equity curve to {save_path}")

        plt.show()

    def plot_trade_analysis(
        self,
        trades: List[Dict],
        save_path: Optional[str] = None
    ):
        """
        Plot trade distribution and P&L analysis

        Args:
            trades: List of trade dicts
            save_path: Path to save figure
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figsize_multi)

        # Extract P&L
        pnls = [t.get('pnl', 0) for t in trades]
        returns = [t.get('return_pct', 0) for t in trades]

        # 1. P&L Distribution
        ax1.hist(pnls, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax1.axvline(0, color='red', linestyle='--', linewidth=2)
        ax1.set_xlabel('P&L ($)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('P&L Distribution')
        ax1.grid(True, alpha=0.3)

        # 2. Return Distribution
        ax2.hist(returns, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
        ax2.axvline(0, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Return (%)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Return Distribution')
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))

        # 3. Cumulative P&L
        cumulative_pnl = np.cumsum(pnls)
        ax3.plot(cumulative_pnl, linewidth=2, color='purple')
        ax3.axhline(0, color='red', linestyle='--', linewidth=1)
        ax3.set_xlabel('Trade Number')
        ax3.set_ylabel('Cumulative P&L ($)')
        ax3.set_title('Cumulative P&L')
        ax3.grid(True, alpha=0.3)

        # 4. Win/Loss by Side
        sides = [t.get('side', 'unknown') for t in trades]
        side_pnls = pd.DataFrame({'side': sides, 'pnl': pnls})
        side_stats = side_pnls.groupby('side')['pnl'].sum()

        colors = ['green' if x > 0 else 'red' for x in side_stats.values]
        ax4.bar(side_stats.index, side_stats.values, color=colors, alpha=0.7)
        ax4.set_xlabel('Trade Side')
        ax4.set_ylabel('Total P&L ($)')
        ax4.set_title('P&L by Trade Side')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(0, color='black', linestyle='-', linewidth=1)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved trade analysis to {save_path}")

        plt.show()

    def plot_prediction_analysis(
        self,
        predictions_df: pd.DataFrame,
        save_path: Optional[str] = None
    ):
        """
        Plot prediction accuracy analysis

        Args:
            predictions_df: DataFrame with 'pred_close', 'actual_close', 'uncertainty'
            save_path: Path to save figure
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figsize_multi)

        # 1. Predicted vs Actual
        ax1.scatter(predictions_df['actual_close'], predictions_df['pred_close'],
                   alpha=0.5, s=20)

        # Perfect prediction line
        min_val = min(predictions_df['actual_close'].min(), predictions_df['pred_close'].min())
        max_val = max(predictions_df['actual_close'].max(), predictions_df['pred_close'].max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')

        ax1.set_xlabel('Actual Close')
        ax1.set_ylabel('Predicted Close')
        ax1.set_title('Predicted vs Actual Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Prediction Error
        errors = predictions_df['pred_close'] - predictions_df['actual_close']
        ax2.hist(errors, bins=50, color='orange', edgecolor='black', alpha=0.7)
        ax2.axvline(0, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Prediction Error')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'Prediction Error (Mean: {errors.mean():.2f})')
        ax2.grid(True, alpha=0.3)

        # 3. Uncertainty over time
        ax3.plot(predictions_df.index, predictions_df['uncertainty'], alpha=0.7)
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Uncertainty')
        ax3.set_title('Prediction Uncertainty Over Time')
        ax3.grid(True, alpha=0.3)

        # 4. Error vs Uncertainty
        ax4.scatter(predictions_df['uncertainty'], errors.abs(), alpha=0.5, s=20)
        ax4.set_xlabel('Uncertainty')
        ax4.set_ylabel('Absolute Error')
        ax4.set_title('Prediction Error vs Uncertainty')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved prediction analysis to {save_path}")

        plt.show()

    def plot_metrics_dashboard(
        self,
        metrics: Dict[str, float],
        save_path: Optional[str] = None
    ):
        """
        Plot performance metrics dashboard

        Args:
            metrics: Dict of metrics
            save_path: Path to save figure
        """
        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Title
        fig.suptitle('Performance Metrics Dashboard', fontsize=16, fontweight='bold')

        # Helper function for metric boxes
        def create_metric_box(ax, title, value, format_str='.2f'):
            ax.axis('off')
            ax.text(0.5, 0.6, title, ha='center', va='center',
                   fontsize=12, fontweight='bold')

            # Format value
            if format_str.endswith('%'):
                formatted_value = f"{value:.2%}"
            elif format_str == '.2f':
                formatted_value = f"{value:.2f}"
            elif format_str == '.0f':
                formatted_value = f"{value:.0f}"
            elif format_str.startswith('$'):
                formatted_value = f"${value:,.2f}"
            else:
                formatted_value = f"{value:.2f}"

            # Color based on value
            if 'return' in title.lower() or 'sharpe' in title.lower() or 'profit' in title.lower():
                color = 'green' if value > 0 else 'red'
            elif 'drawdown' in title.lower() or 'loss' in title.lower():
                color = 'red' if abs(value) > 0.1 else 'orange'
            else:
                color = 'blue'

            ax.text(0.5, 0.3, formatted_value, ha='center', va='center',
                   fontsize=20, fontweight='bold', color=color)

        # Return Metrics
        ax1 = fig.add_subplot(gs[0, 0])
        create_metric_box(ax1, 'Total Return', metrics.get('total_return', 0), '%')

        ax2 = fig.add_subplot(gs[0, 1])
        create_metric_box(ax2, 'Annualized Return', metrics.get('annualized_return', 0), '%')

        ax3 = fig.add_subplot(gs[0, 2])
        create_metric_box(ax3, 'Monthly Return', metrics.get('monthly_return', 0), '%')

        # Risk Metrics
        ax4 = fig.add_subplot(gs[1, 0])
        create_metric_box(ax4, 'Sharpe Ratio', metrics.get('sharpe_ratio', 0), '.2f')

        ax5 = fig.add_subplot(gs[1, 1])
        create_metric_box(ax5, 'Max Drawdown', metrics.get('max_drawdown_pct', 0), '%')

        ax6 = fig.add_subplot(gs[1, 2])
        create_metric_box(ax6, 'Volatility', metrics.get('volatility', 0), '%')

        # Trade Metrics
        ax7 = fig.add_subplot(gs[2, 0])
        create_metric_box(ax7, 'Win Rate', metrics.get('win_rate', 0), '%')

        ax8 = fig.add_subplot(gs[2, 1])
        create_metric_box(ax8, 'Profit Factor', metrics.get('profit_factor', 0), '.2f')

        ax9 = fig.add_subplot(gs[2, 2])
        create_metric_box(ax9, 'Total Trades', metrics.get('total_trades', 0), '.0f')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved metrics dashboard to {save_path}")

        plt.show()

    def create_full_report(
        self,
        equity_curve: pd.Series,
        trades: List[Dict],
        predictions_df: pd.DataFrame,
        metrics: Dict[str, float],
        save_dir: str = 'backtest_results'
    ):
        """
        Create full visualization report

        Args:
            equity_curve: Equity curve
            trades: List of trades
            predictions_df: Predictions dataframe
            metrics: Metrics dict
            save_dir: Directory to save all figures
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Creating full visualization report in {save_dir}...")

        # 1. Equity curve
        self.plot_equity_curve(
            equity_curve,
            trades,
            save_path=str(save_path / 'equity_curve.png')
        )

        # 2. Trade analysis
        if trades:
            self.plot_trade_analysis(
                trades,
                save_path=str(save_path / 'trade_analysis.png')
            )

        # 3. Prediction analysis
        if predictions_df is not None:
            self.plot_prediction_analysis(
                predictions_df,
                save_path=str(save_path / 'prediction_analysis.png')
            )

        # 4. Metrics dashboard
        self.plot_metrics_dashboard(
            metrics,
            save_path=str(save_path / 'metrics_dashboard.png')
        )

        logger.info(f"Full report created in {save_dir}/ ✅")


if __name__ == '__main__':
    # Test visualizer
    print("=" * 60)
    print("Testing ResultsVisualizer")
    print("=" * 60)

    # Create sample data
    dates = pd.date_range('2025-01-01', periods=100, freq='D')
    equity = pd.Series(
        10000 * (1 + np.cumsum(np.random.randn(100) * 0.01)),
        index=dates
    )

    trades = [
        {
            'side': 'long',
            'entry_price': 100,
            'exit_price': 105,
            'exit_time': dates[10],
            'pnl': 50,
            'return_pct': 0.05
        },
        {
            'side': 'short',
            'entry_price': 105,
            'exit_price': 103,
            'exit_time': dates[20],
            'pnl': 20,
            'return_pct': 0.019
        },
        {
            'side': 'long',
            'entry_price': 103,
            'exit_price': 100,
            'exit_time': dates[30],
            'pnl': -30,
            'return_pct': -0.029
        }
    ] * 10  # Repeat for more samples

    predictions_df = pd.DataFrame({
        'pred_close': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'actual_close': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'uncertainty': np.random.rand(100) * 2
    }, index=dates)

    metrics = {
        'total_return': 0.25,
        'annualized_return': 0.30,
        'monthly_return': 0.025,
        'sharpe_ratio': 1.8,
        'max_drawdown_pct': -0.15,
        'volatility': 0.20,
        'win_rate': 0.65,
        'profit_factor': 2.5,
        'total_trades': 30
    }

    # Initialize visualizer
    viz = ResultsVisualizer()

    print("\nResultsVisualizer initialized successfully!")
    print("Test plots will be displayed...")

    print("\n" + "=" * 60)
    print("Visualizer test setup complete! ✅")
    print("=" * 60)
    print("(Run individual plot methods to see visualizations)")
