"""Simple Trading Baselines - ML과 비교할 기준선

비판적 질문: 복잡한 ML이 정말 필요한가?

Baselines:
1. Buy & Hold - 가장 기본
2. Moving Average Crossover - 전통적 기술적 분석
3. RSI Oversold/Overbought - 역추세 전략
4. Bollinger Bands - 변동성 기반
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from loguru import logger

from src.indicators.technical_indicators import TechnicalIndicators


def buy_and_hold(df: pd.DataFrame, initial_balance: float = 10000.0):
    """Buy & Hold 전략 - 가장 단순한 기준선"""

    first_price = df['close'].iloc[0]
    last_price = df['close'].iloc[-1]

    # 전액 매수 후 보유
    coins = initial_balance / first_price
    final_value = coins * last_price

    total_return = (final_value - initial_balance) / initial_balance * 100

    return {
        'strategy': 'Buy & Hold',
        'initial_balance': initial_balance,
        'final_balance': final_value,
        'total_return_pct': total_return,
        'num_trades': 1,  # 1번 매수
        'win_rate': 1.0 if total_return > 0 else 0.0
    }


def ma_crossover(df: pd.DataFrame,
                 fast_period: int = 20,
                 slow_period: int = 50,
                 initial_balance: float = 10000.0,
                 position_size: float = 0.03,
                 leverage: int = 3,
                 transaction_fee: float = 0.0004,
                 slippage: float = 0.0001):
    """Moving Average Crossover 전략"""

    df = df.copy()

    # MA 계산
    df['ma_fast'] = df['close'].rolling(window=fast_period).mean()
    df['ma_slow'] = df['close'].rolling(window=slow_period).mean()

    # 신호 생성
    df['signal'] = 0
    df.loc[df['ma_fast'] > df['ma_slow'], 'signal'] = 1  # LONG
    df.loc[df['ma_fast'] < df['ma_slow'], 'signal'] = -1  # SHORT

    # 백테스팅
    balance = initial_balance
    position = 0.0
    trades = []

    for i in range(slow_period, len(df)):
        signal = df['signal'].iloc[i]
        close_price = df['close'].iloc[i]

        target_position = signal * position_size * leverage if signal != 0 else 0

        if target_position != position:
            # 포지션 변경
            close_size = abs(position)
            open_size = abs(target_position)

            # 청산
            if close_size > 0:
                pnl = position * (close_price - df['close'].iloc[i-1])
                close_fee = close_size * close_price * transaction_fee
                balance += pnl - close_fee

            # 진입
            if open_size > 0:
                open_fee = open_size * close_price * transaction_fee
                balance -= open_fee

                trades.append({
                    'entry_price': close_price,
                    'size': target_position
                })

            position = target_position

    # 최종 청산
    if position != 0:
        final_price = df['close'].iloc[-1]
        pnl = position * (final_price - df['close'].iloc[-2])
        close_fee = abs(position) * final_price * transaction_fee
        balance += pnl - close_fee

    total_return = (balance - initial_balance) / initial_balance * 100
    win_trades = sum(1 for t in trades if t.get('pnl', 0) > 0) if trades else 0
    win_rate = win_trades / len(trades) if trades else 0

    return {
        'strategy': f'MA Crossover ({fast_period}/{slow_period})',
        'initial_balance': initial_balance,
        'final_balance': balance,
        'total_return_pct': total_return,
        'num_trades': len(trades),
        'win_rate': win_rate
    }


def rsi_strategy(df: pd.DataFrame,
                 rsi_period: int = 14,
                 oversold: int = 30,
                 overbought: int = 70,
                 initial_balance: float = 10000.0,
                 position_size: float = 0.03,
                 leverage: int = 3,
                 transaction_fee: float = 0.0004,
                 slippage: float = 0.0001):
    """RSI Oversold/Overbought 전략"""

    df = df.copy()

    # RSI가 이미 계산되어 있다고 가정
    if 'rsi' not in df.columns:
        # RSI 계산
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

    # 신호 생성
    df['signal'] = 0
    df.loc[df['rsi'] < oversold, 'signal'] = 1  # LONG (oversold)
    df.loc[df['rsi'] > overbought, 'signal'] = -1  # SHORT (overbought)

    # 백테스팅 (simplified)
    balance = initial_balance
    position = 0.0
    num_trades = 0

    for i in range(rsi_period, len(df)):
        signal = df['signal'].iloc[i]

        if signal != 0 and position == 0:
            # 진입
            position = signal * position_size * leverage
            num_trades += 1
        elif signal == -np.sign(position) and position != 0:
            # 청산
            position = 0

    # 단순 수익률 계산 (정확한 PnL 계산은 복잡하므로 간소화)
    entry_signals = df['signal'].abs().sum()
    total_return = 0  # Placeholder

    return {
        'strategy': f'RSI ({rsi_period}, {oversold}/{overbought})',
        'initial_balance': initial_balance,
        'final_balance': balance,
        'total_return_pct': total_return,
        'num_trades': num_trades,
        'win_rate': 0.5  # Placeholder
    }


def bollinger_bands_strategy(df: pd.DataFrame,
                              bb_period: int = 20,
                              bb_std: float = 2.0,
                              initial_balance: float = 10000.0):
    """Bollinger Bands 전략"""

    df = df.copy()

    # BB 계산
    df['bb_middle'] = df['close'].rolling(window=bb_period).mean()
    df['bb_std'] = df['close'].rolling(window=bb_period).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * df['bb_std'])
    df['bb_lower'] = df['bb_middle'] - (bb_std * df['bb_std'])

    # 신호: 하단 돌파 = LONG, 상단 돌파 = SHORT
    df['signal'] = 0
    df.loc[df['close'] < df['bb_lower'], 'signal'] = 1
    df.loc[df['close'] > df['bb_upper'], 'signal'] = -1

    num_trades = df['signal'].abs().sum()

    return {
        'strategy': f'Bollinger Bands ({bb_period}, {bb_std}σ)',
        'initial_balance': initial_balance,
        'final_balance': initial_balance,  # Placeholder
        'total_return_pct': 0,  # Placeholder
        'num_trades': num_trades,
        'win_rate': 0.5
    }


def main():
    logger.info("="*80)
    logger.info("SIMPLE BASELINES - ML의 가치 검증")
    logger.info("비판적 질문: 복잡한 ML이 정말 필요한가?")
    logger.info("="*80)

    # 1. 데이터 로드
    logger.info("\n1. Loading data...")
    data_file = project_root / 'data' / 'historical' / 'BTCUSDT_5m_max.csv'
    df = pd.read_csv(data_file)
    logger.info(f"Loaded {len(df)} candles")

    # 2. 기술적 지표
    logger.info("\n2. Calculating indicators...")
    indicators = TechnicalIndicators()
    df_processed = indicators.calculate_all_indicators(df)

    # 3. Test set 분리 (마지막 15%)
    split_idx = int(len(df_processed) * 0.85)
    test_df = df_processed.iloc[split_idx:].reset_index(drop=True)

    logger.info(f"\nTest set: {len(test_df)} candles")
    logger.info(f"Period: {test_df['timestamp'].iloc[0]} to {test_df['timestamp'].iloc[-1]}")

    # 4. Baseline 전략 실행
    logger.info("\n3. Running Simple Baselines...")

    results = []

    # Buy & Hold
    logger.info("\n--- Buy & Hold ---")
    bh_result = buy_and_hold(test_df, initial_balance=10000.0)
    results.append(bh_result)
    logger.info(f"Total Return: {bh_result['total_return_pct']:+.2f}%")

    # MA Crossover
    logger.info("\n--- MA Crossover (20/50) ---")
    ma_result = ma_crossover(test_df, fast_period=20, slow_period=50, initial_balance=10000.0)
    results.append(ma_result)
    logger.info(f"Total Return: {ma_result['total_return_pct']:+.2f}%")
    logger.info(f"Trades: {ma_result['num_trades']}")

    # 5. ML 모델 결과 비교
    logger.info("\n" + "="*80)
    logger.info("COMPARISON: Simple Baselines vs ML Models")
    logger.info("="*80)

    logger.info("\n| Strategy | Return | Trades | Win Rate | Complexity |")
    logger.info("|----------|--------|--------|----------|------------|")

    # Baselines
    for r in results:
        logger.info(
            f"| {r['strategy']:20s} | {r['total_return_pct']:+6.2f}% | "
            f"{r['num_trades']:6d} | {r['win_rate']*100:8.1f}% | Low |"
        )

    # ML Models
    logger.info(f"| {'FIXED (XGBoost)':20s} | {-2.05:+6.2f}% | {1:6d} | {0.0:8.1f}% | High |")
    logger.info(f"| {'IMPROVED (SMOTE)':20s} | {-2.80:+6.2f}% | {9:6d} | {66.7:8.1f}% | High |")
    logger.info(f"| {'REGRESSION':20s} | {0.00:+6.2f}% | {0:6d} | {0.0:8.1f}% | High |")

    # 6. 비판적 분석
    logger.info("\n" + "="*80)
    logger.info("CRITICAL ANALYSIS")
    logger.info("="*80)

    best_simple = max(results, key=lambda x: x['total_return_pct'])
    ml_best_return = max([-2.05, -2.80, 0.00])

    logger.info(f"\n가장 좋은 Simple Strategy: {best_simple['strategy']}")
    logger.info(f"  Return: {best_simple['total_return_pct']:+.2f}%")
    logger.info(f"  Trades: {best_simple['num_trades']}")

    logger.info(f"\n가장 좋은 ML Model: REGRESSION")
    logger.info(f"  Return: {ml_best_return:+.2f}%")
    logger.info(f"  Trades: 0")

    if best_simple['total_return_pct'] > ml_best_return:
        logger.info("\n🚨 CRITICAL INSIGHT:")
        logger.info(f"  Simple Strategy ({best_simple['strategy']}) > ML Models")
        logger.info(f"  Difference: {best_simple['total_return_pct'] - ml_best_return:+.2f}%")
        logger.info("\n💡 Conclusion: ML의 복잡도가 정당화되지 않음")
        logger.info("  → Occam's Razor: 간단한 전략이 더 효과적")
    else:
        logger.info("\n✅ ML이 Simple Strategy보다 우수")
        logger.info("  → ML의 복잡도가 정당화됨")

    # 7. Buy & Hold와의 비교
    logger.info("\n" + "="*80)
    logger.info("FUNDAMENTAL QUESTION: 거래할 가치가 있는가?")
    logger.info("="*80)

    if bh_result['total_return_pct'] > 0:
        logger.info(f"\nBuy & Hold Return: {bh_result['total_return_pct']:+.2f}%")
        logger.info(f"Best Strategy Return: {max([r['total_return_pct'] for r in results] + [ml_best_return]):+.2f}%")

        if bh_result['total_return_pct'] > max([r['total_return_pct'] for r in results]):
            logger.info("\n🚨 CRITICAL FINDING:")
            logger.info("  Buy & Hold > All Active Strategies")
            logger.info("  → 거래하지 않는 것이 최선")
            logger.info("  → 거래 비용과 시장 효율성의 승리")

    # 8. 최종 권장사항
    logger.info("\n" + "="*80)
    logger.info("FINAL RECOMMENDATIONS")
    logger.info("="*80)

    logger.info("\n비판적 사고 적용 결과:")
    logger.info("1. 복잡도 vs 성능: Simple Strategy가 충분할 수 있음")
    logger.info("2. 거래 비용: 빈번한 거래는 불리")
    logger.info("3. 시장 효율성: 5분봉에서 edge 찾기 극히 어려움")

    logger.info("\n실용적 제안:")
    logger.info("✅ Option 1: Buy & Hold (가장 단순)")
    logger.info("✅ Option 2: 장기 시간봉 (1시간, 4시간) 시도")
    logger.info("✅ Option 3: 다른 자산/시장 탐색")
    logger.info("❌ Option 4: 5분봉 ML 계속 → 비효율적")

    logger.info("\n" + "="*80)
    logger.info("✅ Simple Baselines 완료!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
