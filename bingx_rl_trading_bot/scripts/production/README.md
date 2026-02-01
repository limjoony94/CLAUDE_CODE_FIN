# scripts/production/ — 프로덕션 스크립트

## 활성 봇

| 파일 | 설명 |
|------|------|
| `pattern_5m_bot.py` | **현재 운영** — Pattern 5m 봇 엔트리포인트 |
| `pattern_5m/` | 14개 모듈 패키지 ([상세](pattern_5m/README.md)) |

## 실행

```bash
# 직접 실행
python3 scripts/production/pattern_5m_bot.py

# tmux 세션으로 실행
tmux new-session -d -s pattern_5m "python3 scripts/production/pattern_5m_bot.py"
```

## 레거시 (미사용)

이 디렉토리에는 과거 실험/개발 과정의 스크립트가 다수 존재.
현재 운영과 관련된 것은 `pattern_5m_bot.py`와 `pattern_5m/` 패키지만 해당.

주요 레거시:
- `engulf_5m/` — Archived Engulf bot
- `opportunity_gating_bot_4x.py` — 구 전략
- `rsi_trend_filter_bot.py` — 구 전략
- `adx_supertrend_trail_bot.py` — 구 전략
- `train_*.py`, `optimize_*.py` — 구 ML 학습/최적화
