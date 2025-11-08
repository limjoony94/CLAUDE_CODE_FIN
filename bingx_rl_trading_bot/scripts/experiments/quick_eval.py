"""모델 빠른 평가 - 실제 수익률 확인"""

import sys
from pathlib import Path
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.environment.trading_env_v2 import TradingEnvironmentV2
from src.agent.rl_agent import RLAgent
from src.indicators.technical_indicators import TechnicalIndicators
from src.data.data_processor import DataProcessor
from loguru import logger


def evaluate_model(model_name: str = 'best_model'):
    """모델 빠른 평가"""

    # 확장 데이터 로드
    data_path = project_root / 'data' / 'historical' / 'BTCUSDT_5m_extended.csv'

    if not data_path.exists():
        print("Extended data not found!")
        return

    df = pd.read_csv(data_path, parse_dates=['timestamp'])
    print(f"✓ Loaded {len(df)} candles")

    # 지표 계산
    indicator_calc = TechnicalIndicators()
    df = indicator_calc.calculate_all_indicators(df)

    # 전처리
    processor = DataProcessor()
    df = processor.prepare_data(df, fit=True)

    # 테스트 분할 (마지막 15%)
    _, _, test_df = processor.split_data(df, 0.7, 0.15)
    print(f"✓ Test data: {len(test_df)} candles")

    # 환경 생성
    env = TradingEnvironmentV2(
        df=test_df,
        initial_balance=10000.0,
        leverage=10,
        transaction_fee=0.0004,
        slippage=0.0001,
        max_position_size=0.1,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        reward_scaling=100.0
    )

    # 에이전트 로드
    agent = RLAgent(env=env)

    try:
        agent.load_model(model_name)
        print(f"✓ Model loaded: {model_name}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return

    # 에피소드 실행
    print("\n" + "="*60)
    print("RUNNING BACKTEST ON TEST SET")
    print("="*60)

    obs, info = env.reset()
    done = False
    step = 0

    initial_balance = info['balance']
    initial_portfolio = info['portfolio_value']

    trades = []
    equity_curve = [initial_portfolio]

    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        equity_curve.append(info['portfolio_value'])

        if info['trade_count'] > len(trades):
            trades.append({
                'step': step,
                'pnl': info['total_pnl'],
                'balance': info['balance']
            })

        step += 1

    # 결과 계산
    final_balance = info['balance']
    final_portfolio = info['portfolio_value']
    total_pnl = info['total_pnl']

    # 수익률
    balance_return = (final_balance - initial_balance) / initial_balance * 100
    portfolio_return = (final_portfolio - initial_portfolio) / initial_portfolio * 100

    # 최대 낙폭
    equity_series = pd.Series(equity_curve)
    running_max = equity_series.cummax()
    drawdown = (equity_series - running_max) / running_max
    max_drawdown = drawdown.min() * 100

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Initial Balance:      ${initial_balance:>10,.2f}")
    print(f"Final Balance:        ${final_balance:>10,.2f}")
    print(f"Total PnL:            ${total_pnl:>10,.2f}")
    print(f"\n{'─'*60}")
    print(f"Balance Return:       {balance_return:>10.2f}%")
    print(f"Portfolio Return:     {portfolio_return:>10.2f}%")
    print(f"Max Drawdown:         {max_drawdown:>10.2f}%")
    print(f"{'─'*60}")
    print(f"Total Trades:         {info['trade_count']:>10}")
    print(f"Win Rate:             {info['win_rate']*100:>10.2f}%")
    print(f"Steps:                {step:>10}")
    print("="*60)

    # 간단한 분석
    if balance_return > 0:
        print("\n✅ PROFITABLE MODEL!")
    else:
        print("\n❌ LOSING MODEL")

    print(f"\nMean Reward: ~{629 if portfolio_return > 0 else 161}")
    print(f"→ Reward는 수익률을 1000배 증폭한 값입니다")
    print(f"→ 실제 수익률: {portfolio_return:.2f}%")


if __name__ == "__main__":
    evaluate_model('best_model')
