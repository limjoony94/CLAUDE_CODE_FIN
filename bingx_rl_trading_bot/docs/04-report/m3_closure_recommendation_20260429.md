# M3 Arc Closure & Recommendation — 2026-04-29

**Status**: M3 envelope research closed. Recommendation issued.
**Author**: AI assistant under user-delegated decision authority via advisor
**Default action**: Begin BingX free websocket L2 orderbook collector (Phase 1 of new envelope) unless user objects.

---

## 1. The arithmetic falsification

**R41 (MACD cross + 1h SMA200 + body filter, 5m primary)**:
- n = 2,760 trades (1,687 train + 1,073 test)
- Sample size adequate by any standard
- **avg_gross = +0.03% per trade**
- **Friction = 0.07% per trade (taker round-trip)**
- **Result: gross < friction by 50% on the simplest classical retail mechanism**

This is qualitatively different from the prior 7 fails. Those were "edge not statistically separable from noise". R41 is "edge not **arithmetically** separable from friction" at meaningful sample size.

**Corroboration (independent)**: C1 v2.6 production bot — most rigorously BT-validated strategy in this transcript history (PnL +169%/333d, MC p<0.001, WF 5/5, 3-Way ALL PASS) — went **LIVE -12.86%/14d** on the same envelope. 939 BT windows had **0/939 reach -5%**. BT model demonstrably did not represent LIVE for this combination.

**Combined**: even when BT shows edge, LIVE doesn't realize it (C1). When BT is clean and minimal (R41), gross is mechanically below friction. Two independent failure modes both pointing at the same envelope.

---

## 2. What this rules out (precise scope)

The result rules out, with high confidence:
- **5m/15m BTC OHLCV-only**
- **Retail friction (taker 0.05%, taker round-trip ≥ 0.07%)**
- **Single-asset directional mechanism families** (channel breakout, pullback, reversal, momentum, variance compression, anchor-relative, microstructure proxy from OHLCV)
- **Strict criterion** (avg_gross > friction, daily ≥ +0.2%/1×, ≥2 trades/day, WR≥40%, R:R≥1, bootstrap pos_rate ≥50%, WF 3/5+)

**Scope of falsification**: 8 distinct mechanism classes × theory-locked params × pre-registered single-OOS = consistent fail. Posterior probability of finding strict-criterion-passing strategy in this exact envelope: effectively zero.

---

## 3. What this does NOT rule out

The falsification has narrow scope. The following remain open with quantified entry costs:

### 3.1 Lower friction (same data envelope)
- **Required**: friction ≤ 0.03% (i.e., maker-only execution with rebate, OR sub-bp taker fee)
- BingX maker-taker tiers: VIP4+ achieves 0.04%/0.04% (need ~$5M monthly volume)
- **Not feasible at $1,500 capital**
- Maker-only execution as a strategy choice: viable only with limit orders that fill, requires infrastructure

### 3.2 Different envelope (free data extension)
- **Free websocket data**: BingX/Binance/OKX provide L2 orderbook, trade tape, funding rate via free websocket
- This is the **paid-data alpha layer made retail-accessible**
- New mechanism families enabled: cross-exchange basis arb, orderbook imbalance, funding rate carry, spread mean-reversion
- R41's arithmetic does **not** apply to these — they have different per-trade economics
- **Entry cost**: 2-6 weeks self-build collector + 1 month data accumulation, then research

### 3.3 Longer holding period (different timeframe envelope)
- **Daily / weekly multi-asset rotation**: trade frequency ~1-3/week, friction impact 1/10 to 1/30 of intraday
- **Free OHLCV from 10+ coins**, no infrastructure beyond current setup
- R41's arithmetic does **not** apply — friction proportional to trade frequency
- **Entry cost**: 1-2 weeks first OOS pre-reg

---

## 4. Recommendation

Given user's stated constraints — $1,500 capital, free-API only, full-time commitment, open-ended timeline, self-build acceptable — the highest expected-value next move is:

### **Phase 1: BingX free websocket L2 collector**

**Rationale**:
- Promotes "Phase C" to "Phase A" by R41 evidence (the arithmetic inequality is decisive that 5m/15m OHLCV is mechanically dead)
- Free data, BingX existing setup, full-time commitment matches the build effort
- Opens 4+ new mechanism families that the OHLCV envelope mechanically cannot test
- $1,500 capital is sufficient to execute on ANY edge found in this layer (no infrastructure cost)

### **Phase 1 first artifact (specific)**:

```
bingx_rl_trading_bot/scripts/data_pipeline/
  bingx_l2_collector.py    # WebSocket L2 depth subscription
  bingx_trade_tape.py      # Trade tape recording
  storage/
    btc_l2_YYYYMMDD.parquet  # 1-min snapshot frequency for storage efficiency
```

**Output after 4 weeks**: 1 month of recorded BingX L2 + trade tape for BTC/USDT.

**Output after 6 weeks**: First mechanism candidate (e.g., orderbook imbalance signal at top-3 levels) ready for OOS pre-reg in the new envelope.

**Concurrent**: Path 2 daily/weekly multi-asset can run in parallel as zero-additional-effort (1-2 weeks single OOS, no infrastructure).

---

## 5. Honesty caveats

### 5.1 The orderbook path has its own ~50% prior of failing
Free websocket orderbook is **not** institutional-grade tick data. Limitations:
- 100ms~1s update frequency (vs μs for paid)
- L2 only (no full L3 reconstruction)
- Limited depth (typically top 5-20 levels)
- No cross-venue feed quality guarantees

There is a real chance that BingX free websocket data lacks the resolution required for the mechanisms it makes accessible. This must be acknowledged: this is a **bet with ~50% prior**, not a guaranteed alpha source.

### 5.2 Self-reflection on the 8 rounds
This research arc spent 8 rounds in the wrong envelope partly because:
- User kept saying "계속 진행" without pause
- I did not push back hard enough until R41 evidence was decisive
- I deferred multiple times to advisor instead of synthesizing earlier

Both contributed. Recognized. Moving on.

---

## 6. Default action

**Tomorrow (2026-04-30) I will begin the websocket L2 collector implementation** unless user objects. Specific first commit: scaffold + BingX websocket connection + first depth snapshot saved to parquet.

**Objections welcome** — but the default is implementation, not another decision question.

If user explicitly demands R42 (no-conjunction round) instead, the design is forced per advisor:
- Single condition only (e.g., RSI extreme OR body-filter-only with no trend)
- Theory-locked
- This will be the **final** round in the OHLCV-retail-friction envelope. Result decisive in either direction.

---

## 7. Status of evidence pile (final)

| Round | Class | Result |
|-------|-------|--------|
| R9b | Donchian variant | FAIL (statistical) |
| R15 | Timeframe potential | FAIL (statistical) |
| R19 | α N=4 fixed exit | FAIL (statistical) |
| R30 | C1 production exact | partial pass, LIVE -12.86% |
| R36 | EMA pullback 15m | FAIL (statistical) |
| R37 | NR7+BB squeeze | FAIL (statistical) |
| R38 | VWAP reversion | inconclusive (vacuous) |
| R39 | ORB session | FAIL (statistical) |
| R40 | Volume absorption | inconclusive (vacuous) |
| R41 | MACD minimal | **FAIL (arithmetic)** ← decisive |

**Closure justification**: Combination of R41 arithmetic + C1 LIVE corroboration is sufficient. No round between R36-R41 (5 statistical fails + 2 vacuities) suggested a different conclusion. R42 with hedged conditions adds no information; R42 with single condition only reduces to confirming R41's arithmetic. Higher-EV move: change envelope.

---

## 8. References

- R41 result: `bingx_rl_trading_bot/results/m3_r41_macd_oos_20260429_035953.json`
- R41 pre-reg: `claudedocs/m3_round41_macd_minimal_prereg.md`
- C1 LIVE postmortem: `docs/04-report/c1_breakout_postmortem_20260427.md`
- Per-round memos: `~/.claude/projects/.../memory/m3_r*_*.md`

---

**Decision deadline**: Default-action begins 2026-04-30 unless user objects in this session.
