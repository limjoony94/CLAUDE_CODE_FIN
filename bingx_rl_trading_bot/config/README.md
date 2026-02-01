# config/ — 설정 파일

| 파일 | 설명 | 민감도 |
|------|------|--------|
| `pattern_5m_config.yaml` | 전략 파라미터 (레버리지, 리스크, 타임프레임) | 일반 |
| `api_keys.yaml` | BingX API 키/시크릿 | ⚠️ **민감** |
| `api_keys.yaml.example` | API 키 템플릿 | 일반 |
| `config.yaml` | 레거시 설정 (미사용) | 일반 |
| `*_config.yaml` | 기타 봇 설정 (미사용) | 일반 |

## 주의사항

- `api_keys.yaml` 절대 커밋/노출 금지
- 패턴별 TP/SL은 `config/`가 아닌 `pattern_5m/constants.py`에 하드코딩
- 설정 변경 후 봇 재시작 필요
