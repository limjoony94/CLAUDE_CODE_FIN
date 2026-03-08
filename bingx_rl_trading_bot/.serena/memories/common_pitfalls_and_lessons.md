# Common Pitfalls and Lessons Learned

## Production Code Pitfalls
- `dict.get(key, {})` vs `dict.get(key) or {}`: When value is explicitly `null/None`, `.get()` returns None (key exists), use `or {}` for None→{} conversion
- OneDrive sync directory: Never put lock/state files in OneDrive-synced dirs. Use `tempfile.gettempdir()` for locks, atomic retry + `.new` fallback for state
- State corruption: `sync_metrics_with_state()` — when state < metrics, state is corrupted, trust metrics
- `_EXCHANGE_MANAGED` sentinel: BingX error 110407/110406/110413 = TP/SL already exists on exchange

## Research Pitfalls
- In-sample optimization ≠ OOS improvement: Subset selection is overfitting
- Small TP/SL fails OOS: TP 0.80/SL 1.20 → WF FAIL (margin too thin)
- IS window 270d suboptimal: IS=135d gives 6x better OOS (edge decay study)
- FC-aware discovery worse than DROP: Timeout PnL = noise
- Individual MC fail ≠ removal: Portfolio diversification > individual significance

## Strategy Design
- R:R unfavorable by design: 92% of patterns have TP < SL (mean R:R 0.478)
- Strategy depends on high WR (>65%) to compensate R:R
- Direction edge is real (+235% pre-overlap OOS), TP/SL partially overfit
- SL scaling is the key driver of ATR adaptation (TP-only FAIL, SL-only PASS)
- Proportional vol_mult cap preserves R:R ratio (hard SL cap distorts up to +65.6%)

## Research Validation
- classify_candle() takes (row, avg_body_20), NOT (open, high, low, close) — always check function signature
- Hedge vs FIFO re-verified (03-08): Hedge PnL/MDD 67.35 vs FIFO 0 — FIFO forced closures destroy strategy
- "DISCRIMINATING" in random test can mean "always loses" not "has edge" — FIFO 0/20 random pass = no signal works
- Smart-OneWay (skip opposite) is viable alternative: PnL/MDD 31.04, WF 3/3 PASS, but 46% of Hedge performance
- Cascade SL removal → PnL +502% to -50%: single most critical mechanism

## Performance Monitoring
- `recent_wins/losses` = rolling buffer, not accurate count — use total_pnl/trades
- Bot PnL includes leverage (3x), TP/SL % are pre-leverage price distances
- Expected: WR 68%, edge 0.27% per trade, daily frequency ~2-4 trades
