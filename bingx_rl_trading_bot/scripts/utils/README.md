# Utility Scripts

> **Updated**: 2026-02-11

봇 운영 및 디버깅을 위한 유틸리티 스크립트입니다.

## 운영 도구

| 스크립트 | 설명 |
|---------|------|
| `check_status.py` | 봇 상태 확인 (프로세스, 포지션, 메트릭) |
| `monitor_simple.py` | 간단 모니터링 (로그 + 상태) |
| `monitor_bot.py` | 봇 모니터링 |
| `snapshot_monitor.py` | 스냅샷 모니터링 |
| `get_current_balance.py` | 현재 잔고 조회 |
| `verify_state.py` | state.json 상태 검증 |

## 제어 도구

| 스크립트 | 설명 |
|---------|------|
| `stop_bot.py` | 봇 안전 중지 |
| `force_kill_bot.py` | 봇 강제 종료 |
| `close_position.py` | 수동 포지션 청산 |
| `reset_state.py` | 상태 리셋 |
| `safe_reset.py` | 안전 리셋 (포지션 확인 후) |

## 진단 도구

| 스크립트 | 설명 |
|---------|------|
| `system_diagnostic.py` | 시스템 진단 (API, 설정, 상태) |

## Usage

```bash
cd bingx_rl_trading_bot
python scripts/utils/<script_name>.py
```
