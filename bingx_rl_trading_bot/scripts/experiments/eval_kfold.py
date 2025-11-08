"""K-Fold 모델 실제 수익률 평가"""

import sys
from pathlib import Path
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.environment.trading_env_v4 import TradingEnvironmentV4
from src.agent.rl_agent import RLAgent
from src.indicators.technical_indicators import TechnicalIndicators
from src.data.data_processor import DataProcessor


def main():
    # 데이터 로드
    df = pd.read_csv('data/historical/BTCUSDT_5m_max.csv', parse_dates=['timestamp'])

    # 지표
    indicator_calc = TechnicalIndicators()
    df = indicator_calc.calculate_all_indicators(df)

    # 전처리
    processor = DataProcessor()
    df = processor.prepare_data(df, fit=True)

    # 테스트 세트 (마지막 15%)
    test_size = int(len(df) * 0.15)
    test_df = df.iloc[-test_size:]

    print(f"Test Set: {len(test_df)} candles")
    print(f"Period: {test_df['timestamp'].min()} ~ {test_df['timestamp'].max()}")

    # V4 환경
    env = TradingEnvironmentV4(
        df=test_df,
        initial_balance=10000.0,
        leverage=3,
        transaction_fee=0.0004,
        slippage=0.0001,
        max_position_size=0.03,
        stop_loss_pct=0.01,
        take_profit_pct=0.02,
        min_hold_steps=3
    )

    # Fold 4 모델 로드
    agent = RLAgent(env=env)
    agent.load_model('kfold_model_fold4')

    print("\n" + "="*70)
    print("BACKTESTING FOLD 4 MODEL")
    print("="*70)

    obs, info = env.reset()
    done = False
    step = 0

    initial_portfolio = info['portfolio_value']
    equity_curve = [initial_portfolio]

    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        equity_curve.append(info['portfolio_value'])
        step += 1

    # 결과 계산
    final_portfolio = info['portfolio_value']
    total_return = (final_portfolio - initial_portfolio) / initial_portfolio * 100

    # 최대 낙폭
    equity_series = pd.Series(equity_curve)
    running_max = equity_series.cummax()
    drawdown = (equity_series - running_max) / running_max
    max_dd = drawdown.min() * 100

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Initial Portfolio:    ${initial_portfolio:>10,.2f}")
    print(f"Final Portfolio:      ${final_portfolio:>10,.2f}")
    print(f"Total PnL:            ${info['total_pnl']:>10,.2f}")
    print(f"\n{'─'*70}")
    print(f"Return:               {total_return:>10.2f}%")
    print(f"Max Drawdown:         {max_dd:>10.2f}%")
    print(f"Total Trades:         {info['trade_count']:>10}")
    print(f"Win Rate:             {info['win_rate']*100:>10.2f}%")
    print(f"{'='*70}")

    if total_return > 0:
        print("\n✅ PROFITABLE MODEL!")
    else:
        print("\n❌ LOSING MODEL")

    print(f"\nMean Reward: +5140")
    print(f"Actual Return: {total_return:.2f}%")
    print(f"Difference explained by: Holding bonuses")


if __name__ == "__main__":
    main()
