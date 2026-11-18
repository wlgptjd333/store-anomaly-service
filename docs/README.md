# 문서 안내

이 프로젝트는 "무엇을 만들었나"보다 **"왜 그렇게 판단했나"** 를 보여주는 데 중점을 둡니다. 읽는 순서:

| 순서 | 문서 | 내용 |
|---|---|---|
| 1 | [프로젝트 README](../README.md) | 문제, 동작 17개, 데이터, Pipeline, 지금까지 결과 |
| 2 | [기술 결정 기록](DESIGN_DECISIONS.md) | 결정 11개마다 다른 방법 / 고른 이유 / 근거 숫자 / 한계 |
| 3 | [정규화 비교 분석](NORMALIZATION_ANALYSIS.md) | 이미지 기준 vs 어깨 폭 vs 몸 크기 — 다른 날 촬영한 사람에서 38% vs 68% |

## 데이터 흐름과 산출물

```text
핸드폰 영상 170개 (5명 × 17동작 × 정면·측면)
  01 Manifest ─ 02 영상 검사 ─ 03 Pose·Feature(원본 Keypoint 보관) ─ 05 자동 QA
  04 라벨 검수 (170 OK) ─ 06 동작 구간 추정 ─ 09 동작별 관절 QA ─ Human QA (PASS 170)
  10 Approved Manifest ─ 11 사람 단위 Split + Leakage 검사 ─ 12 정규화 비교
  → 다음: 고정 길이 Sequence, 동작 분류 모델 (6일차)
```

| 산출물 | 위치 |
|---|---|
| 승인된 학습 목록 (동작 구간 포함) | `configs/approved_clip_manifest.csv` |
| Train / Validation / Test, LOSO 5-fold | `reports/splits/` |
| 품질·검수 표 | `reports/tables/` (`sequence_qa`, `action_keypoint_qa`, `human_qa`, `review_checklist`, `segments_auto`) |
| 정규화 비교 결과 | `reports/tables/normalization_compare.csv`, `normalization_loso.csv` |
| 그림 (얼굴 없이 뼈대만) | `docs/images/` |
