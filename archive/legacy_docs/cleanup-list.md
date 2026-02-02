# 정리 대상 목록 (Cleanup List)

**최종 업데이트**: 2026-02-02

---

## 완료된 정리

### 커밋 49f4083 (02-02)
- [x] experiments/ 419개 → archive/legacy_experiments/
- [x] temp 파일 11개 삭제
- [x] debugging/ 6개 파일 삭제
- [x] requirements.txt 생성

### 커밋 ff331fc (02-02)
- [x] debug/ 21개 → archive/legacy_debug/
- [x] deprecated production bots 60+ → archive/deprecated_production/
- [x] analysis/ 일회성 165개 → archive/legacy_analysis/
- [x] utils/ 일회성 63개 → archive/legacy_utils/
- [x] scripts/ 루트 잡동사니 14+5dirs → archive/legacy_misc_scripts/

### 커밋 (pending, 02-02)
- [x] models/ 메타데이터 377개 → archive/legacy_models/
- [x] deprecated configs (adx, rsi) → archive/deprecated_production/
- [x] data/trained_models/ → archive/legacy_models/
- [x] "Engulf 5m Bot" → "Pattern 5m Bot" docstring 통일
- [x] README.md 구조도 업데이트

## 잔여 (낮은 우선순위)

- [ ] bingx_rl_trading_bot/data/cache/ — 캐시 파일 정리 (gitignored?)
- [ ] bingx_rl_trading_bot/scripts/production/logs/ — 로그 디렉토리 (gitignored?)
- [ ] docs/ 중 outdated 문서 검토 (classification-unification.md, code-review.md 등)
- [ ] .bat 파일이 남아있는지 확인
