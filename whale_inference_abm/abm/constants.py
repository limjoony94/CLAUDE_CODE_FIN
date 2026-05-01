"""ABM v1 constants. Design ref: docs/02-design/features/whale_inference_abm.design.md Sections 3.3, 5, 6, 7."""

# Time
NS_PER_SECOND = 10**9
BAR_DURATION_SEC = 60
BAR_DURATION_NS = BAR_DURATION_SEC * NS_PER_SECOND

# Order sizing
MIN_ORDER_SIZE = 0.0001  # BTC equivalent
MAX_ORDER_SIZE = 1.0
LOT_STEP = 0.0001

# Friction (BingX rate)
TAKER_FEE = 0.0005  # 0.05%
MAKER_FEE = 0.0002  # 0.02%

# Wealth
INITIAL_WEALTH = 1000.0
BANKRUPTCY_THRESHOLD = 1.0  # 1 USDT equivalent

# Open-system + frozen-admission window (design Section 7)
DEFAULT_T_OPEN_BARS = 7000
DEFAULT_T_EXTRACT_BARS = 3000
DEFAULT_TOTAL_BARS = DEFAULT_T_OPEN_BARS + DEFAULT_T_EXTRACT_BARS
ADMISSION_RATE_LAMBDA = 1.0 / 300.0  # avg 1 new agent per 300 sim-seconds
ADMISSION_INITIAL_WEALTH = 100.0  # smaller than incumbents 1000
