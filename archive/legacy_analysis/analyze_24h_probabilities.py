#!/usr/bin/env python3
"""
Analyze 24-hour probability distribution from bot logs
"""

import re
from datetime import datetime, timedelta
from collections import defaultdict

def parse_log_line(line):
    """Parse a log line to extract timestamp, candle time, price, and probabilities"""
    # Example: 2025-10-20 14:55:42,463 - INFO - [2025-10-20 05:55:00] Price: $110,934.1 | Balance: $5,212.71 | LONG: 0.0754 | SHORT: 0.2096

    pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - INFO - \[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] Price: \$([\d,\.]+) \| Balance: \$([\d,\.]+) \| LONG: ([\d\.]+) \| SHORT: ([\d\.]+)'

    match = re.search(pattern, line)
    if match:
        log_time_str, candle_time_str, price_str, balance_str, long_prob_str, short_prob_str = match.groups()

        log_time = datetime.strptime(log_time_str, '%Y-%m-%d %H:%M:%S')
        candle_time = datetime.strptime(candle_time_str, '%Y-%m-%d %H:%M:%S')
        price = float(price_str.replace(',', ''))
        balance = float(balance_str.replace(',', ''))
        long_prob = float(long_prob_str)
        short_prob = float(short_prob_str)

        return {
            'log_time': log_time,
            'candle_time': candle_time,
            'price': price,
            'balance': balance,
            'long_prob': long_prob,
            'short_prob': short_prob
        }
    return None

def analyze_probabilities(log_files, start_time, end_time):
    """Analyze probabilities from log files within time range"""

    data = []

    for log_file in log_files:
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if 'LONG: ' in line and 'SHORT: ' in line:
                        parsed = parse_log_line(line)
                        if parsed and start_time <= parsed['log_time'] <= end_time:
                            data.append(parsed)
        except Exception as e:
            print(f"Error reading {log_file}: {e}")

    if not data:
        print("No data found in time range")
        return

    # Sort by log time
    data.sort(key=lambda x: x['log_time'])

    # Statistics
    long_probs = [d['long_prob'] for d in data]
    short_probs = [d['short_prob'] for d in data]

    long_above_threshold = sum(1 for p in long_probs if p >= 0.65)
    short_above_threshold = sum(1 for p in short_probs if p >= 0.70)

    print(f"\n{'='*80}")
    print(f"24시간 진입 확률 분석 (10월 19일 15:00 ~ 10월 20일 15:00)")
    print(f"{'='*80}\n")

    print(f"📊 데이터 개수: {len(data)} 캔들 (5분봉)")
    print(f"   예상: 288 캔들 (24h × 12 per hour)")
    print(f"   실제: {len(data)} 캔들 ({len(data)/288*100:.1f}%)\n")

    print(f"💰 Balance 변화:")
    print(f"   시작: ${data[0]['balance']:,.2f}")
    print(f"   종료: ${data[-1]['balance']:,.2f}")
    print(f"   변화: ${data[-1]['balance'] - data[0]['balance']:,.2f} ({(data[-1]['balance']/data[0]['balance']-1)*100:+.2f}%)\n")

    print(f"📈 가격 변화:")
    print(f"   시작: ${data[0]['price']:,.2f}")
    print(f"   종료: ${data[-1]['price']:,.2f}")
    print(f"   변화: ${data[-1]['price'] - data[0]['price']:,.2f} ({(data[-1]['price']/data[0]['price']-1)*100:+.2f}%)\n")

    print(f"🎯 LONG 확률 분석:")
    print(f"   평균: {sum(long_probs)/len(long_probs):.4f}")
    print(f"   최소: {min(long_probs):.4f}")
    print(f"   최대: {max(long_probs):.4f}")
    print(f"   임계값 이상 (≥0.65): {long_above_threshold}개 ({long_above_threshold/len(data)*100:.1f}%)")

    # LONG probability distribution
    long_ranges = {
        '0.00-0.10': sum(1 for p in long_probs if 0.00 <= p < 0.10),
        '0.10-0.20': sum(1 for p in long_probs if 0.10 <= p < 0.20),
        '0.20-0.30': sum(1 for p in long_probs if 0.20 <= p < 0.30),
        '0.30-0.40': sum(1 for p in long_probs if 0.30 <= p < 0.40),
        '0.40-0.50': sum(1 for p in long_probs if 0.40 <= p < 0.50),
        '0.50-0.60': sum(1 for p in long_probs if 0.50 <= p < 0.60),
        '0.60-0.65': sum(1 for p in long_probs if 0.60 <= p < 0.65),
        '0.65+':     sum(1 for p in long_probs if p >= 0.65),
    }

    print(f"\n   분포:")
    for range_name, count in long_ranges.items():
        bar = '█' * int(count / len(data) * 50)
        print(f"   {range_name}: {count:3d} ({count/len(data)*100:5.1f}%) {bar}")

    print(f"\n🎯 SHORT 확률 분석:")
    print(f"   평균: {sum(short_probs)/len(short_probs):.4f}")
    print(f"   최소: {min(short_probs):.4f}")
    print(f"   최대: {max(short_probs):.4f}")
    print(f"   임계값 이상 (≥0.70): {short_above_threshold}개 ({short_above_threshold/len(data)*100:.1f}%)")

    # SHORT probability distribution
    short_ranges = {
        '0.00-0.10': sum(1 for p in short_probs if 0.00 <= p < 0.10),
        '0.10-0.20': sum(1 for p in short_probs if 0.10 <= p < 0.20),
        '0.20-0.30': sum(1 for p in short_probs if 0.20 <= p < 0.30),
        '0.30-0.40': sum(1 for p in short_probs if 0.30 <= p < 0.40),
        '0.40-0.50': sum(1 for p in short_probs if 0.40 <= p < 0.50),
        '0.50-0.60': sum(1 for p in short_probs if 0.50 <= p < 0.60),
        '0.60-0.70': sum(1 for p in short_probs if 0.60 <= p < 0.70),
        '0.70+':     sum(1 for p in short_probs if p >= 0.70),
    }

    print(f"\n   분포:")
    for range_name, count in short_ranges.items():
        bar = '█' * int(count / len(data) * 50)
        print(f"   {range_name}: {count:3d} ({count/len(data)*100:5.1f}%) {bar}")

    # Entry opportunities
    print(f"\n🚀 진입 기회:")
    print(f"   LONG 진입 가능 (≥0.65): {long_above_threshold}개")
    print(f"   SHORT 진입 가능 (≥0.70): {short_above_threshold}개")
    print(f"   총 진입 기회: {long_above_threshold + short_above_threshold}개")
    print(f"   진입 빈도: {(long_above_threshold + short_above_threshold)/len(data)*100:.1f}%")

    # Find high probability periods
    print(f"\n🔥 높은 확률 구간 (LONG ≥ 0.50 또는 SHORT ≥ 0.50):")
    high_prob_count = 0
    for d in data:
        if d['long_prob'] >= 0.50 or d['short_prob'] >= 0.50:
            high_prob_count += 1
            if high_prob_count <= 10:  # Show first 10
                print(f"   {d['log_time'].strftime('%m-%d %H:%M')} | Price: ${d['price']:,.1f} | LONG: {d['long_prob']:.4f} | SHORT: {d['short_prob']:.4f}")

    if high_prob_count > 10:
        print(f"   ... 외 {high_prob_count - 10}개 더")

    print(f"\n   총 {high_prob_count}개 구간 ({high_prob_count/len(data)*100:.1f}%)\n")

    # Show entry threshold moments
    print(f"✅ 진입 임계값 도달 순간:")
    entry_moments = []
    for d in data:
        if d['long_prob'] >= 0.65:
            entry_moments.append((d, 'LONG'))
        if d['short_prob'] >= 0.70:
            entry_moments.append((d, 'SHORT'))

    if entry_moments:
        for d, direction in entry_moments[:20]:  # Show first 20
            print(f"   {d['log_time'].strftime('%m-%d %H:%M')} | {direction:5s} | Price: ${d['price']:,.1f} | LONG: {d['long_prob']:.4f} | SHORT: {d['short_prob']:.4f}")
        if len(entry_moments) > 20:
            print(f"   ... 외 {len(entry_moments) - 20}개 더")
    else:
        print(f"   ❌ 진입 임계값 도달 순간 없음")

    print(f"\n{'='*80}\n")

if __name__ == '__main__':
    # 10월 20일 15:00 기준 과거 24시간
    end_time = datetime(2025, 10, 20, 15, 0, 0)
    start_time = end_time - timedelta(hours=24)

    log_files = [
        'logs/opportunity_gating_bot_4x_20251019.log',
        'logs/opportunity_gating_bot_4x_20251020.log'
    ]

    print(f"분석 기간: {start_time.strftime('%Y-%m-%d %H:%M')} ~ {end_time.strftime('%Y-%m-%d %H:%M')} (24시간)")

    analyze_probabilities(log_files, start_time, end_time)
