# scripts/monitoring/ — 모니터링 스크립트

| 파일 | 설명 | 상태 |
|------|------|------|
| `quant_monitor.py` | 퀀트 모니터 (성과/지표 대시보드) | 레거시 |
| `monitor_bot.py` | 기본 봇 상태 모니터 | 레거시 |
| `config_sync.py` | 설정 파일 동기화 | 레거시 |
| `adx_supertrend_trail_monitor.py` | ADX 봇 모니터 | 미사용 |
| `rsi_trend_filter_monitor.py` | RSI 봇 모니터 | 미사용 |

## monitor 에이전트 참고

현재 모니터링은 직접 파일 접근 방식 사용:

```bash
# 봇 상태
cat results/pattern_5m_bot_state.json | jq .

# 성과 메트릭
cat results/pattern_5m_metrics.json | jq .

# 로그
tail -100 logs/pattern_5m_bot_*.log | grep -E "(TRADE|PROFIT|LOSS|ERROR)"

# 프로세스 확인
ps aux | grep pattern_5m
```
