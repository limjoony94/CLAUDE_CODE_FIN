"""Compare live ATR regime vs backtest to find regime anomalies."""
import pandas as pd
import numpy as np
import math

df5 = pd.read_csv('data/btc_5m_270days_reclassified.csv')
df5['timestamp'] = pd.to_datetime(df5['timestamp'])
df5 = df5.sort_values('timestamp').reset_index(drop=True)
df5.set_index('timestamp', inplace=True)
df = df5.resample('15min').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna().reset_index()
n = len(df)
H_, L_, C_ = df['high'].values, df['low'].values, df['close'].values

# ATR computation
def atr_calc(h, l, c, p=14):
    tr = [h[0]-l[0]] + [max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1])) for i in range(1, len(c))]
    a = [float('nan')]*len(c)
    if len(c) >= p:
        a[p-1] = sum(tr[:p])/p
        for i in range(p, len(c)): a[i] = (a[i-1]*(p-1)+tr[i])/p
    return a

ATR = atr_calc(H_, L_, C_)

# ATR as % of price
atr_pcts = [(ATR[i] / C_[i] * 100) if not math.isnan(ATR[i]) else np.nan for i in range(len(C_))]
atr_pcts = np.array([x for x in atr_pcts if not np.isnan(x)])

# Live ATR data (from log)
live_atrs = [171.89, 176.76, 204.42, 276.22, 194.17, 199.94, 246.25, 179.59, 171.82, 276.20, 328.42, 207.50, 159.82]
live_prices = [74184, 74321, 75138, 74503, 73976, 74292, 74349, 74670, 74935, 73749, 75001, 74714, 75299]
live_atr_pcts = [a/p*100 for a, p in zip(live_atrs, live_prices)]

print('='*60)
print('  ATR 레짐 비교 — 라이브 vs 백테스트')
print('='*60)
print()
print('백테스트 ATR (가격 대비 %):')
print('  P5:   {:.3f}%'.format(np.percentile(atr_pcts, 5)))
print('  P25:  {:.3f}%'.format(np.percentile(atr_pcts, 25)))
print('  P50:  {:.3f}%'.format(np.percentile(atr_pcts, 50)))
print('  P75:  {:.3f}%'.format(np.percentile(atr_pcts, 75)))
print('  P95:  {:.3f}%'.format(np.percentile(atr_pcts, 95)))
print('  Mean: {:.3f}%'.format(atr_pcts.mean()))
print()

print('라이브 ATR (13거래, 가격 대비 %):')
print('  최소: {:.3f}%'.format(min(live_atr_pcts)))
print('  평균: {:.3f}%'.format(sum(live_atr_pcts)/len(live_atr_pcts)))
print('  최대: {:.3f}%'.format(max(live_atr_pcts)))
print()

# Live ATR percentile vs backtest
for pct in live_atr_pcts:
    # where does this ATR pct fall in backtest distribution?
    pass

live_mean = sum(live_atr_pcts) / len(live_atr_pcts)
percentile_of_live_mean = (atr_pcts <= live_mean).sum() / len(atr_pcts) * 100

print('해석:')
print('  라이브 평균 ATR {:.3f}%는 백테스트의 {:.0f}th percentile'.format(
    live_mean, percentile_of_live_mean))

# Is live ATR abnormally low or high?
bt_median = np.percentile(atr_pcts, 50)
diff = (live_mean - bt_median) / bt_median * 100
print('  백테스트 중위값({:.3f}%) 대비 {:+.1f}% 차이'.format(bt_median, diff))

if abs(diff) < 15:
    print('  → 정상 레짐 (±15% 이내)')
elif diff < 0:
    print('  → 저변동성 레짐 (trail이 평소보다 타이트하게 작동)')
else:
    print('  → 고변동성 레짐 (trail이 평소보다 느슨하게 작동)')

# Price range check
print()
print('='*60)
print('  가격 레짐 비교')
print('='*60)
print('백테스트 BTC: \${:.0f} ~ \${:.0f} (평균 \${:.0f})'.format(C_.min(), C_.max(), C_.mean()))
print('라이브 BTC:   \${:.0f} ~ \${:.0f}'.format(min(live_prices), max(live_prices)))

# Was live price range covered by backtest?
live_min, live_max = min(live_prices), max(live_prices)
bt_in_range = ((C_ >= live_min) & (C_ <= live_max)).sum()
print('  라이브 가격 범위를 포함하는 백테스트 봉: {} / {} ({:.1f}%)'.format(
    bt_in_range, len(C_), bt_in_range/len(C_)*100))

# Analyze ATR within live price range
if bt_in_range > 0:
    mask = (C_ >= live_min) & (C_ <= live_max)
    # Need to filter ATR for valid bars only
    valid_atr_pcts = []
    for i in range(len(C_)):
        if not math.isnan(ATR[i]) and live_min <= C_[i] <= live_max:
            valid_atr_pcts.append(ATR[i]/C_[i]*100)
    if valid_atr_pcts:
        valid_atr_pcts = np.array(valid_atr_pcts)
        print('\n  라이브 가격대에서 백테스트 ATR%:')
        print('    P5: {:.3f}%  P50: {:.3f}%  P95: {:.3f}%'.format(
            np.percentile(valid_atr_pcts, 5), np.percentile(valid_atr_pcts, 50), np.percentile(valid_atr_pcts, 95)))
        print('    평균: {:.3f}%'.format(valid_atr_pcts.mean()))
        same_price_mean = valid_atr_pcts.mean()
        same_diff = (live_mean - same_price_mean) / same_price_mean * 100
        print('    라이브와 차이: {:+.1f}%'.format(same_diff))
