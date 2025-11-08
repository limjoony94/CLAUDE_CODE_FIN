"""
테스트넷 실전 거래 봇
- 페이퍼 트레이딩과 다름: 실제 주문 체결
- 실제 슬리피지, 체결 지연 경험
- Mainnet과 동일하지만 실제 돈 없음
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import time
import joblib
import pandas as pd
from datetime import datetime
from loguru import logger

# BingX API (Testnet)
from src.api.bingx_api import BingXAPI

# 로깅 설정
log_file = f"logs/testnet_real_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logger.add(log_file, rotation="100 MB", retention="7 days")

logger.info("="*80)
logger.info("🚀 테스트넷 실전 거래 봇 시작")
logger.info("="*80)
logger.info("⚠️  페이퍼 트레이딩이 아님 - 실제 주문 체결!")
logger.info("✅ 하지만 테스트넷이므로 실제 돈 없음")
logger.info("="*80)

# 설정
CONFIG = {
    'initial_capital': 10000,  # 테스트넷 초기 자본
    'long_capital_pct': 0.70,  # LONG 70%
    'short_capital_pct': 0.30,  # SHORT 30%

    # LONG 설정
    'long_threshold': 0.7,
    'long_sl_pct': 1.0,
    'long_tp_pct': 2.0,  # 1.5% → 2.0% (비용 영향 감소: 13% → 10%)

    # SHORT 설정 (Paper trading 결과 반영)
    'short_threshold': 0.5,  # 0.4 → 0.5 (n=10, 30% 승률 개선 필요)
    'short_sl_pct': 1.5,
    'short_tp_pct': 3.0,

    # 공통
    'max_holding_hours': 4,
    'check_interval_minutes': 5,

    # API
    'use_testnet': True,
    'symbol': 'BTC-USDT',
}

logger.info(f"\n📋 설정:")
logger.info(f"  초기 자본: ${CONFIG['initial_capital']}")
logger.info(f"  LONG Threshold: {CONFIG['long_threshold']}")
logger.info(f"  SHORT Threshold: {CONFIG['short_threshold']} (최적화됨)")
logger.info(f"  네트워크: {'TESTNET' if CONFIG['use_testnet'] else 'MAINNET'}")

# 모델 로드
logger.info(f"\n📦 모델 로딩...")
long_model = joblib.load("models/xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl")

# SHORT model: 3-class 사용 (TechnicalIndicators 호환)
short_model = joblib.load("models/xgboost_v4_phase4_3class_lookahead3_thresh3.pkl")
is_3class_short = hasattr(short_model, 'classes_') and len(short_model.classes_) == 3

logger.info(f"✅ 모델 로드 완료")
logger.info(f"  LONG: Phase 4 Advanced")
logger.info(f"  SHORT: 3-class Phase 4 {'(3-class)' if is_3class_short else ''}")

# API 초기화
api = BingXAPI(use_testnet=CONFIG['use_testnet'])
logger.info(f"✅ BingX API 초기화 완료 (Testnet: {CONFIG['use_testnet']})")

# 상태
class TradingState:
    def __init__(self):
        self.long_capital = CONFIG['initial_capital'] * CONFIG['long_capital_pct']
        self.short_capital = CONFIG['initial_capital'] * CONFIG['short_capital_pct']
        self.long_position = None  # (entry_time, entry_price, quantity, order_id)
        self.short_position = None
        self.trades_completed = []

    @property
    def total_capital(self):
        return self.long_capital + self.short_capital

    @property
    def total_pnl_pct(self):
        return (self.total_capital - CONFIG['initial_capital']) / CONFIG['initial_capital'] * 100

state = TradingState()

def add_indicators(df):
    """기술적 지표 추가 (간단한 버전)"""
    from src.indicators.technical_indicators import TechnicalIndicators

    ti = TechnicalIndicators()
    df = ti.calculate_all_indicators(df)
    return df

def get_market_data():
    """시장 데이터 가져오기"""
    try:
        # BingX에서 최근 500 캔들 가져오기
        klines = api.get_klines(
            symbol=CONFIG['symbol'],
            interval='5m',
            limit=500
        )

        if not klines:
            logger.error("❌ 데이터 가져오기 실패")
            return None

        # 데이터프레임 변환
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])

        logger.info(f"✅ 데이터: {len(df)} candles")
        return df

    except Exception as e:
        logger.error(f"❌ 데이터 가져오기 에러: {e}")
        return None

def check_long_signal(df):
    """LONG 신호 확인"""
    try:
        df = add_indicators(df)
        feature_cols = [col for col in df.columns if col not in
                       ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        X = df[feature_cols].iloc[[-1]]  # 최근 캔들만

        prob = long_model.predict_proba(X)[0, 1]
        return prob >= CONFIG['long_threshold'], prob

    except Exception as e:
        logger.error(f"❌ LONG 신호 에러: {e}")
        return False, 0.0

def check_short_signal(df):
    """SHORT 신호 확인"""
    try:
        df = add_indicators(df)
        feature_cols = [col for col in df.columns if col not in
                       ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        X = df[feature_cols].iloc[[-1]]

        # 3-class model의 경우 class 2 (SHORT) 사용
        if is_3class_short:
            prob = short_model.predict_proba(X)[0, 2]  # Class 0=HOLD, 1=LONG, 2=SHORT
        else:
            prob = short_model.predict_proba(X)[0, 1]

        return prob >= CONFIG['short_threshold'], prob

    except Exception as e:
        logger.error(f"❌ SHORT 신호 에러: {e}")
        logger.error(f"   Features: {len(feature_cols) if 'feature_cols' in locals() else 'unknown'}")
        return False, 0.0

def execute_long_entry(current_price):
    """LONG 포지션 진입 (실제 주문)"""
    try:
        quantity = state.long_capital / current_price

        logger.info("="*80)
        logger.info("🟢 LONG 포지션 진입 시도")
        logger.info(f"  가격: ${current_price}")
        logger.info(f"  수량: {quantity:.6f} BTC")
        logger.info(f"  금액: ${state.long_capital:.2f}")

        # 실제 주문 (테스트넷)
        order = api.place_order(
            symbol=CONFIG['symbol'],
            side='BUY',
            order_type='MARKET',
            quantity=quantity
        )

        if order:
            state.long_position = (datetime.now(), current_price, quantity, order.get('orderId'))
            logger.info(f"✅ LONG 진입 성공!")
            logger.info(f"  Order ID: {order.get('orderId')}")
            logger.info("="*80)
            return True
        else:
            logger.error("❌ LONG 진입 실패")
            return False

    except Exception as e:
        logger.error(f"❌ LONG 진입 에러: {e}")
        return False

def execute_short_entry(current_price):
    """SHORT 포지션 진입 (실제 주문)"""
    try:
        quantity = state.short_capital / current_price

        logger.info("="*80)
        logger.info("🔴 SHORT 포지션 진입 시도")
        logger.info(f"  가격: ${current_price}")
        logger.info(f"  수량: {quantity:.6f} BTC")
        logger.info(f"  금액: ${state.short_capital:.2f}")

        # 실제 주문 (테스트넷에서 SHORT = SELL)
        order = api.place_order(
            symbol=CONFIG['symbol'],
            side='SELL',
            order_type='MARKET',
            quantity=quantity
        )

        if order:
            state.short_position = (datetime.now(), current_price, quantity, order.get('orderId'))
            logger.info(f"✅ SHORT 진입 성공!")
            logger.info(f"  Order ID: {order.get('orderId')}")
            logger.info("="*80)
            return True
        else:
            logger.error("❌ SHORT 진입 실패")
            return False

    except Exception as e:
        logger.error(f"❌ SHORT 진입 에러: {e}")
        return False

def check_long_exit(current_price):
    """LONG 청산 조건 확인"""
    if not state.long_position:
        return False, None

    entry_time, entry_price, quantity, order_id = state.long_position
    hours_held = (datetime.now() - entry_time).total_seconds() / 3600
    pnl_pct = (current_price - entry_price) / entry_price * 100

    if pnl_pct >= CONFIG['long_tp_pct']:
        return True, "TP"
    elif pnl_pct <= -CONFIG['long_sl_pct']:
        return True, "SL"
    elif hours_held >= CONFIG['max_holding_hours']:
        return True, "Max Hold"

    return False, None

def check_short_exit(current_price):
    """SHORT 청산 조건 확인"""
    if not state.short_position:
        return False, None

    entry_time, entry_price, quantity, order_id = state.short_position
    hours_held = (datetime.now() - entry_time).total_seconds() / 3600
    pnl_pct = (entry_price - current_price) / entry_price * 100  # SHORT

    if pnl_pct >= CONFIG['short_tp_pct']:
        return True, "TP"
    elif pnl_pct <= -CONFIG['short_sl_pct']:
        return True, "SL"
    elif hours_held >= CONFIG['max_holding_hours']:
        return True, "Max Hold"

    return False, None

def execute_long_exit(current_price, reason):
    """LONG 포지션 청산 (실제 주문)"""
    try:
        entry_time, entry_price, quantity, order_id = state.long_position
        pnl_pct = (current_price - entry_price) / entry_price * 100
        pnl_usd = state.long_capital * (pnl_pct / 100)

        logger.info("="*80)
        logger.info(f"🟢 LONG 포지션 청산 - {reason}")
        logger.info(f"  진입: ${entry_price:.2f}")
        logger.info(f"  청산: ${current_price:.2f}")
        logger.info(f"  P&L: {pnl_pct:+.2f}% (${pnl_usd:+.2f})")

        # 실제 주문
        order = api.place_order(
            symbol=CONFIG['symbol'],
            side='SELL',
            order_type='MARKET',
            quantity=quantity
        )

        if order:
            state.long_capital *= (1 + pnl_pct / 100)
            state.trades_completed.append({
                'direction': 'LONG',
                'pnl_pct': pnl_pct,
                'reason': reason
            })
            state.long_position = None

            logger.info(f"✅ LONG 청산 성공!")
            logger.info(f"  새 자본: ${state.long_capital:.2f}")
            logger.info(f"  총 자본: ${state.total_capital:.2f} ({state.total_pnl_pct:+.2f}%)")
            logger.info("="*80)
            return True
        else:
            logger.error("❌ LONG 청산 실패")
            return False

    except Exception as e:
        logger.error(f"❌ LONG 청산 에러: {e}")
        return False

def execute_short_exit(current_price, reason):
    """SHORT 포지션 청산 (실제 주문)"""
    try:
        entry_time, entry_price, quantity, order_id = state.short_position
        pnl_pct = (entry_price - current_price) / entry_price * 100
        pnl_usd = state.short_capital * (pnl_pct / 100)

        logger.info("="*80)
        logger.info(f"🔴 SHORT 포지션 청산 - {reason}")
        logger.info(f"  진입: ${entry_price:.2f}")
        logger.info(f"  청산: ${current_price:.2f}")
        logger.info(f"  P&L: {pnl_pct:+.2f}% (${pnl_usd:+.2f})")

        # 실제 주문
        order = api.place_order(
            symbol=CONFIG['symbol'],
            side='BUY',  # SHORT 청산 = BUY
            order_type='MARKET',
            quantity=quantity
        )

        if order:
            state.short_capital *= (1 + pnl_pct / 100)
            state.trades_completed.append({
                'direction': 'SHORT',
                'pnl_pct': pnl_pct,
                'reason': reason
            })
            state.short_position = None

            logger.info(f"✅ SHORT 청산 성공!")
            logger.info(f"  새 자본: ${state.short_capital:.2f}")
            logger.info(f"  총 자본: ${state.total_capital:.2f} ({state.total_pnl_pct:+.2f}%)")
            logger.info("="*80)
            return True
        else:
            logger.error("❌ SHORT 청산 실패")
            return False

    except Exception as e:
        logger.error(f"❌ SHORT 청산 에러: {e}")
        return False

# 메인 루프
logger.info("\n🔄 메인 루프 시작...\n")

iteration = 0
while True:
    try:
        iteration += 1
        logger.info(f"{'='*80}")
        logger.info(f"🔄 Iteration #{iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{'='*80}")

        # 시장 데이터 가져오기
        df = get_market_data()
        if df is None:
            logger.warning("⚠️  데이터 없음, 5분 후 재시도...")
            time.sleep(300)
            continue

        current_price = df['close'].iloc[-1]
        logger.info(f"💰 현재 가격: ${current_price:.2f}")

        # LONG 포지션 체크
        if state.long_position:
            entry_time, entry_price, _, _ = state.long_position
            hours = (datetime.now() - entry_time).total_seconds() / 3600
            pnl_pct = (current_price - entry_price) / entry_price * 100

            logger.info(f"🟢 LONG 보유 중: P&L {pnl_pct:+.2f}% | {hours:.1f}h")

            # 청산 조건 체크
            should_exit, reason = check_long_exit(current_price)
            if should_exit:
                execute_long_exit(current_price, reason)

        else:
            # 신호 체크
            signal, prob = check_long_signal(df)
            logger.info(f"🟢 LONG 확률: {prob:.3f} (Threshold: {CONFIG['long_threshold']})")

            if signal:
                execute_long_entry(current_price)

        # SHORT 포지션 체크
        if state.short_position:
            entry_time, entry_price, _, _ = state.short_position
            hours = (datetime.now() - entry_time).total_seconds() / 3600
            pnl_pct = (entry_price - current_price) / entry_price * 100

            logger.info(f"🔴 SHORT 보유 중: P&L {pnl_pct:+.2f}% | {hours:.1f}h")

            # 청산 조건 체크
            should_exit, reason = check_short_exit(current_price)
            if should_exit:
                execute_short_exit(current_price, reason)

        else:
            # 신호 체크
            signal, prob = check_short_signal(df)
            logger.info(f"🔴 SHORT 확률: {prob:.3f} (Threshold: {CONFIG['short_threshold']})")

            if signal:
                execute_short_entry(current_price)

        # 통계
        logger.info(f"\n📊 현재 상태:")
        logger.info(f"  총 자본: ${state.total_capital:.2f} ({state.total_pnl_pct:+.2f}%)")
        logger.info(f"  완료 거래: {len(state.trades_completed)}")

        if state.trades_completed:
            trades_df = pd.DataFrame(state.trades_completed)
            win_rate = (trades_df['pnl_pct'] > 0).mean() * 100
            logger.info(f"  승률: {win_rate:.1f}%")

        # 대기
        logger.info(f"\n⏳ {CONFIG['check_interval_minutes']}분 대기...\n")
        time.sleep(CONFIG['check_interval_minutes'] * 60)

    except KeyboardInterrupt:
        logger.info("\n\n🛑 사용자 중단")
        break
    except Exception as e:
        logger.error(f"❌ 에러 발생: {e}")
        logger.info("5분 후 재시도...")
        time.sleep(300)

logger.info("\n" + "="*80)
logger.info("✅ 테스트넷 실전 거래 봇 종료")
logger.info("="*80)
