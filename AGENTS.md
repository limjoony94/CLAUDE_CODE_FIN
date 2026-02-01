# AGENTS.md - 프로젝트 에이전트 규칙

> 이 프로젝트에서 작업하는 모든 에이전트가 따라야 할 규칙

---

## 필수 규칙

### 1. CLAUDE.md 업데이트
- 코드 수정 시 **반드시** CLAUDE.md의 Version History 업데이트
- 패턴 추가/제거 시 패턴 테이블 업데이트
- 설정 변경 시 핵심 파라미터 테이블 업데이트

### 2. 커밋 메시지 컨벤션
```
v1.XX.X: 간결한 변경 설명

예시:
v1.22.0: Remove DN-D-BD pattern (overfitting MC=0.2390)
v1.21.0: Conservative per-pattern TP/SL optimization
fix: leverage side parameter for BingX API
docs: update CLAUDE.md version history
research: 270-day backtest validation
```

### 3. 테스트 절차
전략 변경 시 반드시:
1. **MC test** (10k sims, p < 0.01)
2. **WF validation** (5-fold, ≥ 4/5 pass)
3. **270일 전체 기간 검증**
4. 결과를 `claudedocs/`에 기록

### 4. 파일 구조 규칙
- 운영 코드: `scripts/production/pattern_5m/` 만 수정
- 연구 스크립트: `scripts/analysis/`에 추가
- 실험: `experimental/`에 추가
- 완료된 연구: `claudedocs/`에 기록
- 레거시/폐기: `archive/`로 이동

### 5. 금지 사항
- `config/api_keys.yaml` 내용 절대 노출/수정 금지
- Look-Ahead Bias: `shift(-1)`, `rolling(center=True)` 금지
- 봇 중지 시 열린 포지션 미확인 금지
- MC/WF 미검증 전략 배포 금지

---

## 에이전트별 권한

| 에이전트 | 코드 수정 | 봇 운영 | 설정 변경 | 문서 수정 |
|---------|----------|---------|----------|----------|
| dev | ✅ | ❌ | ✅ | ✅ |
| automation | ❌ | ✅ | ❌ | ❌ |
| monitor | ❌ | ❌ (읽기만) | ❌ | ❌ |
