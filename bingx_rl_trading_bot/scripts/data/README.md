# Data Collection Scripts

> **Updated**: 2026-02-11

BingX에서 시장 데이터를 수집하는 스크립트입니다.

## Scripts

| 스크립트 | 설명 |
|---------|------|
| `collect_max_data.py` | 최대 기간 히스토리컬 데이터 수집 |
| `collect_mainnet_data.py` | 메인넷 데이터 수집 |
| `collect_public_data.py` | 공개 API 데이터 수집 |
| `collect_data.py` | 기본 데이터 수집 |
| `collect_more_data.py` | 추가 데이터 수집 |
| `fetch_historical.py` | 히스토리컬 데이터 페치 |
| `download_latest_4weeks.py` | 최근 4주 데이터 다운로드 |
| `calculate_features_latest4weeks.py` | 최근 4주 피처 계산 |
| `create_15min_data.py` | 15분봉 데이터 생성 |

## 현재 데이터

- **`data/btc_5m_270days_reclassified.csv`**: 270일 5분봉 데이터 (Ground Truth 분류)
- 프로덕션 봇과 연구 스크립트 모두 이 데이터를 사용합니다.
