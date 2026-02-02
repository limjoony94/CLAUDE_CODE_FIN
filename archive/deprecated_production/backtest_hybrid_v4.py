"""
Backtest Hybrid Strategy (XGBoost + Technical Indicators)

Phase 3: Combining the best of both worlds
- XGBoost Phase 2 model (47.6% win rate, but -1.82% vs B&H)
- Technical Strategy (market context filtering)

Expected Outcome:
- False signals reduced by 50%
- Win rate: 55-65%
- Trade frequency: 8-12 per window
- vs B&H: +1.0-2.0%
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.technical_strategy import TechnicalStrategy

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Trading parameters
WINDOW_SIZE = 1440  # 5 days
INITIAL_CAPITAL = 10000.0
POSITION_SIZE_PCT = 0.95
STOP_LOSS = 0.01
TAKE_PROFIT = 0.03
MAX_HOLDING_HOURS = 4
TRANSACTION_COST = 0.0006


def classify_market_regime(df_window):
    start_price = df_window['close'].iloc[0]
    end_price = df_window['close'].iloc[-1]
    return_pct = ((end_price / start_price) - 1) * 100
    if return_pct > 3.0:
        return "Bull"
    elif return_pct < -2.0:
        return "Bear"
    else:
        return "Sideways"


class HybridStrategy:
    """
    Hybrid Strategy combining XGBoost ML predictions with Technical indicators

    Entry Logic:
    - Strong: XGBoost prob > 0.5 AND Technical = LONG
    - Moderate: XGBoost prob > 0.4 AND Technical = LONG with high strength
    - Otherwise: HOLD
    """

    def __init__(self, xgboost_model, feature_columns, technical_strategy,
                 xgb_threshold_strong=0.5,
                 xgb_threshold_moderate=0.4,
                 tech_strength_threshold=0.6):
        self.xgboost = xgboost_model
        self.feature_columns = feature_columns
        self.technical = technical_strategy
        self.xgb_threshold_strong = xgb_threshold_strong
        self.xgb_threshold_moderate = xgb_threshold_moderate
        self.tech_strength_threshold = tech_strength_threshold

    def should_enter(self, df, idx):
        """
        Determine if we should enter a trade

        Returns:
            should_enter (bool)
            confidence (str): 'strong' or 'moderate'
            xgb_prob (float)
            tech_signal (str)
            tech_strength (float)
        """
        # Check if we have enough data
        if idx >= len(df):
            return False, None, 0.0, 'HOLD', 0.0

        # 1. Get XGBoost probability
        features = df[self.feature_columns].iloc[idx:idx+1].values
        if np.isnan(features).any():
            return False, None, 0.0, 'HOLD', 0.0

        xgb_prob = self.xgboost.predict_proba(features)[0][1]

        # 2. Get Technical signal
        tech_signal, tech_strength, tech_reason = self.technical.get_signal(df, idx)

        # 3. Combined decision
        # Strong entry: Both models highly confident
        if xgb_prob > self.xgb_threshold_strong and tech_signal == 'LONG':
            return True, 'strong', xgb_prob, tech_signal, tech_strength

        # Moderate entry: XGBoost moderate + Technical strong
        if (xgb_prob > self.xgb_threshold_moderate and
            tech_signal == 'LONG' and
            tech_strength >= self.tech_strength_threshold):
            return True, 'moderate', xgb_prob, tech_signal, tech_strength

        return False, None, xgb_prob, tech_signal, tech_strength


def backtest_hybrid_strategy(df, hybrid_strategy):
    capital = INITIAL_CAPITAL
    position = None
    trades = []

    for i in range(len(df)):
        current_price = df['close'].iloc[i]

        # Manage existing position
        if position is not None:
            entry_idx = position['entry_idx']
            entry_price = position['entry_price']
            hours_held = (i - entry_idx) / 12

            pnl_pct = (current_price - entry_price) / entry_price

            exit_reason = None
            if pnl_pct <= -STOP_LOSS:
                exit_reason = "Stop Loss"
            elif pnl_pct >= TAKE_PROFIT:
                exit_reason = "Take Profit"
            elif hours_held >= MAX_HOLDING_HOURS:
                exit_reason = "Max Holding"

            if exit_reason:
                quantity = position['quantity']
                pnl_usd = pnl_pct * (entry_price * quantity)

                entry_cost = entry_price * quantity * TRANSACTION_COST
                exit_cost = current_price * quantity * TRANSACTION_COST
                total_cost = entry_cost + exit_cost
                pnl_usd -= total_cost

                capital += pnl_usd

                trades.append({
                    'entry_idx': entry_idx,
                    'exit_idx': i,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'quantity': quantity,
                    'pnl_pct': pnl_pct,
                    'pnl_usd': pnl_usd,
                    'exit_reason': exit_reason,
                    'holding_hours': hours_held,
                    'xgb_prob': position['xgb_prob'],
                    'tech_signal': position['tech_signal'],
                    'tech_strength': position['tech_strength'],
                    'confidence': position['confidence']
                })

                position = None

        # Look for entry
        if position is None and i < len(df) - 1:
            should_enter, confidence, xgb_prob, tech_signal, tech_strength = hybrid_strategy.should_enter(df, i)

            if should_enter:
                position_value = capital * POSITION_SIZE_PCT
                quantity = position_value / current_price

                position = {
                    'entry_idx': i,
                    'entry_price': current_price,
                    'quantity': quantity,
                    'xgb_prob': xgb_prob,
                    'tech_signal': tech_signal,
                    'tech_strength': tech_strength,
                    'confidence': confidence
                }

    # Calculate metrics
    if len(trades) == 0:
        return trades, {
            'total_return_pct': 0.0,
            'num_trades': 0,
            'win_rate': 0.0,
            'avg_pnl_pct': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }

    total_return_pct = ((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    winning_trades = [t for t in trades if t['pnl_usd'] > 0]
    win_rate = (len(winning_trades) / len(trades)) * 100
    avg_pnl_pct = np.mean([t['pnl_pct'] for t in trades]) * 100

    # Drawdown
    cumulative_returns = []
    running_capital = INITIAL_CAPITAL
    for trade in trades:
        running_capital += trade['pnl_usd']
        cumulative_returns.append(running_capital)

    if len(cumulative_returns) > 0:
        peak = cumulative_returns[0]
        max_dd = 0
        for value in cumulative_returns:
            if value > peak:
                peak = value
            dd = ((peak - value) / peak) * 100
            if dd > max_dd:
                max_dd = dd
    else:
        max_dd = 0.0

    # Sharpe
    if len(trades) > 1:
        returns = [t['pnl_pct'] for t in trades]
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0.0
    else:
        sharpe = 0.0

    metrics = {
        'total_return_pct': total_return_pct,
        'num_trades': len(trades),
        'win_rate': win_rate,
        'avg_pnl_pct': avg_pnl_pct,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd
    }

    return trades, metrics


def rolling_window_backtest(df, hybrid_strategy):
    windows = []
    start_idx = 0

    while start_idx + WINDOW_SIZE <= len(df):
        end_idx = start_idx + WINDOW_SIZE
        window_df = df.iloc[start_idx:end_idx].reset_index(drop=True)

        regime = classify_market_regime(window_df)

        trades, metrics = backtest_hybrid_strategy(window_df, hybrid_strategy)

        # Buy & Hold
        bh_start = window_df['close'].iloc[0]
        bh_end = window_df['close'].iloc[-1]
        bh_return = ((bh_end - bh_start) / bh_start) * 100
        bh_cost = 2 * TRANSACTION_COST * 100
        bh_return -= bh_cost

        # Analyze trade confidence distribution
        if len(trades) > 0:
            strong_trades = [t for t in trades if t['confidence'] == 'strong']
            moderate_trades = [t for t in trades if t['confidence'] == 'moderate']
            num_strong = len(strong_trades)
            num_moderate = len(moderate_trades)
        else:
            num_strong = 0
            num_moderate = 0

        windows.append({
            'start_idx': start_idx,
            'end_idx': end_idx,
            'regime': regime,
            'hybrid_return': metrics['total_return_pct'],
            'bh_return': bh_return,
            'difference': metrics['total_return_pct'] - bh_return,
            'num_trades': metrics['num_trades'],
            'num_strong': num_strong,
            'num_moderate': num_moderate,
            'win_rate': metrics['win_rate'],
            'sharpe': metrics['sharpe_ratio'],
            'max_dd': metrics['max_drawdown']
        })

        start_idx += WINDOW_SIZE

    return pd.DataFrame(windows)


if __name__ == "__main__":
    print("=" * 80)
    print("Hybrid Strategy Backtest - Phase 3")
    print("XGBoost Phase 2 + Technical Indicators")
    print("=" * 80)

    # Load XGBoost model
    model_file = MODELS_DIR / "xgboost_v3_lookahead3_thresh1_phase2.pkl"
    with open(model_file, 'rb') as f:
        xgboost_model = pickle.load(f)
    print(f"✅ XGBoost model loaded: {model_file}")

    # Load feature columns
    feature_file = MODELS_DIR / "xgboost_v3_lookahead3_thresh1_phase2_features.txt"
    with open(feature_file, 'r') as f:
        feature_columns = [line.strip() for line in f.readlines()]
    print(f"✅ Features loaded: {len(feature_columns)} features")

    # Load data
    data_file = DATA_DIR / "historical" / "BTCUSDT_5m_max.csv"
    df = pd.read_csv(data_file)
    print(f"✅ Data loaded: {len(df)} candles")

    # Calculate features (Phase 2)
    df = calculate_features(df)

    # Calculate technical indicators
    technical_strategy = TechnicalStrategy()
    df = technical_strategy.calculate_indicators(df)

    df = df.dropna()
    print(f"✅ All features calculated: {len(df)} rows after dropna")

    # Initialize Hybrid Strategy
    hybrid_strategy = HybridStrategy(
        xgboost_model=xgboost_model,
        feature_columns=feature_columns,
        technical_strategy=technical_strategy,
        xgb_threshold_strong=0.5,
        xgb_threshold_moderate=0.4,
        tech_strength_threshold=0.6
    )
    print(f"✅ Hybrid Strategy initialized")

    print(f"\n{'=' * 80}")
    print(f"Running Rolling Window Backtest...")
    print(f"{'=' * 80}")

    results = rolling_window_backtest(df, hybrid_strategy)

    # Summary
    print(f"\n📊 Results ({len(results)} windows):")
    print(f"  Hybrid Return: {results['hybrid_return'].mean():.2f}% ± {results['hybrid_return'].std():.2f}%")
    print(f"  Buy & Hold Return: {results['bh_return'].mean():.2f}% ± {results['bh_return'].std():.2f}%")
    print(f"  Difference: {results['difference'].mean():.2f}% ± {results['difference'].std():.2f}%")
    print(f"  🎯 Avg Trades per Window: {results['num_trades'].mean():.1f}")
    print(f"     - Strong confidence: {results['num_strong'].mean():.1f}")
    print(f"     - Moderate confidence: {results['num_moderate'].mean():.1f}")
    print(f"  Avg Win Rate: {results['win_rate'].mean():.1f}%")
    print(f"  Avg Sharpe: {results['sharpe'].mean():.3f}")
    print(f"  Avg Max DD: {results['max_dd'].mean():.2f}%")

    # By regime
    print(f"\n📈 By Market Regime:")
    for regime in ['Bull', 'Bear', 'Sideways']:
        regime_df = results[results['regime'] == regime]
        if len(regime_df) > 0:
            print(f"  {regime} ({len(regime_df)} windows):")
            print(f"    Hybrid: {regime_df['hybrid_return'].mean():.2f}%")
            print(f"    Buy & Hold: {regime_df['bh_return'].mean():.2f}%")
            print(f"    Difference: {regime_df['difference'].mean():.2f}%")
            print(f"    Trades: {regime_df['num_trades'].mean():.1f}")

    # Statistical test
    from scipy import stats
    t_stat, p_value = stats.ttest_rel(results['hybrid_return'], results['bh_return'])
    print(f"\n🔬 Statistical Test:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Significant: {'✅ Yes' if p_value < 0.05 else '❌ No'}")

    # Save
    output_file = RESULTS_DIR / "backtest_hybrid_v4.csv"
    results.to_csv(output_file, index=False)
    print(f"\n✅ Results saved: {output_file}")

    print(f"\n{'=' * 80}")
    print("Hybrid Strategy Backtest Complete!")
    print(f"{'=' * 80}")

    # Critical analysis
    print(f"\n🎯 비판적 분석:")
    print(f"\n  Phase 1 (XGBoost only, 18 features):")
    print(f"    - vs B&H: -2.01%")
    print(f"    - Trades: 18.5")
    print(f"    - Win Rate: 47.6%")

    print(f"\n  Phase 2 (XGBoost, 33 features):")
    print(f"    - vs B&H: -1.82%")
    print(f"    - Trades: 18.5")
    print(f"    - Win Rate: 45.0%")

    print(f"\n  Phase 3 (Hybrid: XGBoost + Technical):")
    print(f"    - vs B&H: {results['difference'].mean():.2f}%")
    print(f"    - Trades: {results['num_trades'].mean():.1f}")
    print(f"    - Win Rate: {results['win_rate'].mean():.1f}%")

    # Comparison
    improvement_vs_phase2 = results['difference'].mean() - (-1.82)
    improvement_trades = results['num_trades'].mean() - 18.5
    improvement_winrate = results['win_rate'].mean() - 45.0

    print(f"\n  개선 효과 (vs Phase 2):")
    print(f"    - vs B&H: {improvement_vs_phase2:+.2f}%p")
    print(f"    - Trades: {improvement_trades:+.1f}")
    print(f"    - Win Rate: {improvement_winrate:+.1f}%p")

    if results['difference'].mean() > 0:
        print(f"\n  🎉 성공! Buy & Hold 이김! (+{results['difference'].mean():.2f}%)")
        print(f"  ✅ Hybrid Strategy가 XGBoost 단독의 한계를 극복했습니다!")
    elif results['difference'].mean() > -1.82:
        print(f"\n  ✅ 부분 성공: Phase 2 대비 개선 (+{improvement_vs_phase2:.2f}%p)")
        print(f"  ⚠️ Buy & Hold 이기려면 추가 최적화 필요")
    else:
        print(f"\n  ⚠️ Hybrid 전략이 예상보다 성과가 낮습니다.")
        print(f"  📋 Threshold 최적화 또는 Technical 전략 개선 필요")
