# 영상 추가 · 다시 학습 · 서비스 모델 교체 (Runbook)

영상을 더 찍거나 학습 방법을 바꿀 때 **이 순서대로만 하면** 처음부터 다시 정리하지 않아도 되게 적은 문서입니다.
(마지막 확인 2026-11-24 — 영상 255개, ExtraTrees + 변화량·요약 통계, 서비스 Test S05 78.4%)

두 저장소가 나뉘어 있습니다.

| 저장소 | 하는 일 | 위치 (WSL) |
|---|---|---|
| `unmanned-store-anomaly` (이 저장소, 데이터 저장소) | 영상 정리 → Pose → 검수 → Dataset → 실험 → 서비스용 모델 내보내기 | `~/ai_vision/unmanned-store-anomaly` |
| `store-anomaly-service` | Django + FastAPI 웹 서비스. 모델 파일 3개만 받아 씀 | `~/ai_vision/store-anomaly-service` |

```text
영상 ─01─02─03─05─(04 검수)─06─09─(human_qa 판정)─10─11─13 ─┬─ 19·20 실험 (고르기)
                                                           └─ 15 내보내기 → 서비스 update_model.sh
```

## 0. 지켜야 하는 규칙

- **고르는 건 S01~S04 4-fold 로만.** Test S05 는 고른 뒤 확인만 한다 (S05 숫자를 보고 설정을 바꾸면 Test 가 아니게 됨).
- 같은 사람은 한 세트에만 (사람 단위 Split, `11` 이 검사하고 어기면 멈춤).
- **원본 영상·실명 대응표(`data/`)는 Git 에 안 올림.** 참가자는 S01~ 가명으로만. 팀원 얼굴이 나오는 그림은 올리지 않음 (뼈대 그림만).
- 차이가 난수 변동(4-fold 3번 평균에서 ±1%p 안팎) 안이면 "좋아졌다"고 쓰지 않고 바꾸지 않는다.
- 기본 출력(`reports/sequences`, `reports/dataset`)을 덮어쓰지 않는 실험은 `--out-dir` 로 따로 (`reports/experiments/`, `reports/pose_compare/` 는 Git 제외).

## 1. 영상 추가

### 1-1. 영상 올리기 — 팀 통일 이름으로

팀 드라이브 `D:\3팀드라이브\공유 자료\S0x\` 에 **통일 번호** 이름으로 둡니다 (11/23 뒷모습부터 이 방식).

| 이름 | 뜻 |
|---|---|
| `S04_A005_T03.mp4` | 통일 사람 S04, 동작 A005(밀치기), 방향 T03(뒤) |
| `S04_A005_T03_2.mp4` | 같은 것을 다시 찍은 두 번째 Take → 이 저장소에서 `T02` |

- 통일 번호 ↔ 이 저장소 번호는 [`configs/unified_label_map.csv`](../configs/unified_label_map.csv) (사람 S01↔S03 등, 동작 A001↔A01, 방향 T01=F·T02=S·T03=B).
- **대응표에 없는 새 사람(S06 ..)** 은 번호를 그대로 씁니다 (S06 → S06).
- **새 방향(예: 대각선)** 은 대응표에 `view,D,T04,,대각선,` 한 줄을 더하고 `--unified-views T03,T04` 로 읽습니다.
  (`15` 의 `--views` 와 `VIEW_NAMES` 에도 `D` 를 더해야 학습에 들어감)
- 처음 촬영분(11/13·11/16, 핸드폰 원래 이름 `YYYYMMDD_HHMMSS.mp4`)은 `교과10 동작 영상/` 에서 촬영 시각으로 동작을 정합니다. 새로 찍는 건 이 방식 말고 통일 이름으로.

### 1-1b. 웹 서비스에서 찍힌 영상 가져오기 (IP Webcam 녹화)

서비스(store-anomaly-service)가 분석한 영상 원본은 Docker 볼륨 안에 있습니다. 새 장소·각도에서 찍힌 것이라 다음 학습에 가장 쓸모 있습니다.

```bash
cd ~/ai_vision/store-anomaly-service
docker compose exec django python manage.py export_clips --labeled-only --since 2026-11-24   # → exports/<시각>/
```

1. `exports/<시각>/manifest.csv` 를 열어 영상마다 **사람이 직접 보고** 동작이 맞는지 `label_checked` 칸에 표시 (`expected_action` 은 시연자가 고른 값이라 틀릴 수 있음).
2. 한 사람이 한 동작만 하는 영상만 골라 팀 통일 이름으로 바꿔 팀 드라이브에 둠 — 예: `S04_A005_T04.mp4` (대각선 등 새 방향은 위 '새 방향' 규칙).
   여러 동작이 섞인 긴 녹화는 동작 구간만 잘라서. 다른 사람이 나오면 쓰지 않음.
3. 그다음은 아래 1-2 부터 같음. 자동 감지(`LIVE_AUTO`)로 저장된 영상은 모델이 이미 맞힌 것 위주라(선택 편향) 평가용으로는 쓰지 말고 학습에만.

### 1-2. 목록·복사·검사

```bash
cd ~/ai_vision/unmanned-store-anomaly && source .venv/bin/activate
python scripts/01_build_manifest.py --copy                       # 목록 + data/rgb/videos/<clip_id>.mp4 복사 (바뀐 것만)
python scripts/01_build_manifest.py --copy --unified-views T03,T04   # 새 방향도 읽을 때
python scripts/02_check_videos.py                                # 열림·FPS·길이·끝까지 디코딩
```

- 출력 끝의 `[없음]` 줄 = 빠진 사람·동작·방향. 다시 찍을 목록으로 씀.
- 기존 영상은 같은 clip_id 를 유지합니다 (2026-11-24 확인: 바꾼 뒤 다시 돌려도 Manifest·`source_map` 이 바이트 단위로 같음).

### 1-3. Pose · 자동 QA · 검수

```bash
python scripts/03_build_sequences.py --all --skip-existing        # 새 영상만 YOLO (이미 만든 CSV 는 건너뜀)
python scripts/05_summarize_sequences.py                          # 영상별 PASS / REVIEW
python scripts/04_make_review_sheets.py --action A05              # 라벨 검수 그림 (동작별, 사람이 봄)
python scripts/06_estimate_segments.py --sheets                   # 동작 시작·끝 자동 추정
python scripts/09_action_keypoint_qa.py                           # 동작별 중요 관절 QA + human_qa.csv 에 새 행
```

**사람이 직접 적는 곳 (스크립트를 다시 돌려도 적은 값은 남음)**

| 파일 | 열 | 적는 것 |
|---|---|---|
| `reports/tables/review_checklist.csv` | `review_result` | 라벨 검수 결과 (그림 보고 OK / 틀림) |
| `reports/tables/segments_auto.csv` | `start_sec_final`, `end_sec_final` | 자동 구간이 틀렸을 때만 고친 초 |
| `reports/tables/human_qa.csv` | `human_decision`, `human_note` | **새 영상마다 `PASS` / `RESHOOT` / `EXCLUDE`** — 비어 있으면 10번이 멈춤 |

- 판단 기준과 지난 검수 사례는 [기술 결정 기록](DESIGN_DECISIONS.md) 10·11번, README "지금까지 결과".
- 이름과 실제 촬영 모습이 다르면 `01_build_manifest.py` 의 `FILMED_AS` 에 적음.

### 1-4. 승인 목록 · Split · Dataset

```bash
python scripts/10_build_approved_manifest.py     # PASS 만 → configs/approved_clip_manifest.csv
python scripts/11_build_subject_split.py         # 사람 단위 Split + Leakage 검사 → reports/splits/
python scripts/13_build_sequence_dataset.py      # 30단계 고정 길이 → reports/dataset/ + 입력 점검표
```

- **새 사람을 더했으면** [`configs/subject_split.json`](../configs/subject_split.json) 의 `main` 에 그 사람을 넣어야 11번이 통과합니다
  (보통 Train 에 넣음. Validation S04·Test S05 는 그대로 둬야 예전 숫자와 비교 가능).
- `15`·`19`·`20` 은 `reports/dataset/sequence_dataset_whole.csv` 에 있는 사람을 그대로 씀.

## 2. 학습 방법을 바꿔 보고 고르기

| 스크립트 | 비교하는 것 | 결과 표 | 시간 (CPU) |
|---|---|---|---|
| `18_back_view_experiments.py` | 방향별 학습·좌우 반전·얼굴 관절 (LOSO) | `back_view_*.csv` | 약 4분 |
| `19_model_improvement.py` | 덧붙이는 Feature × RF·ExtraTrees·SVM | `model_improvement_cv.csv` | 약 3분 |
| `21_temporal_models.py [--test]` | 1D-CNN·GRU 와 ExtraTrees 섞기 (지금 서비스 = ExtraTrees + 1D-CNN) | `temporal_models_cv.csv`, `_per_action.csv` | GPU 약 4분 |
| `20_more_improvements.py [이름 일부]` | Pose 모델 크기, 단계 수, 떨림 줄이기, TTA, 나무 설정, 시간 자르기 증강 | `more_improvements_cv.csv` | 설정 하나에 1~2분 |
| `22_gpu_search.py [--only 이름] [--test "a,b"]` | 1D-CNN 크기·평균 개수·mixup·TTA·TCN·Transformer·Pose 11s/11m·섞는 비율 | `gpu_search_*.csv` | GPU, 설정 하나에 1~4분 |
| `23_mocap_synthetic.py [--views 8,16] [--test]` | 교과12 VICON 3D 를 가상 카메라로 2D 투영한 합성 데이터를 학습에 더하기 | `mocap_synthetic_*.csv` | GPU, 설정 하나에 약 7분 |
| `24_geo_augment.py [--only rot,all] [--test all]` | Keypoint 회전·시간 늘이기·앞뒤 자르기·화면 위치 이동 증강 | `geo_augment_*.csv` | GPU, 설정 하나에 약 8분 |

- 새 방법은 `20` 의 `configs()` 에 한 줄(또는 `DATASETS` 에 데이터 하나) 더하고 `python scripts/20_more_improvements.py <이름 일부>` 로 그것만 돌리면 기존 표에 합쳐집니다.
- 입력을 바꾸는 실험은 기본 데이터를 건드리지 않게 만듭니다:

```bash
# Pose 모델 크기 (서비스처럼 30fps 로 줄여서)
python scripts/03_build_sequences.py --all --approved --max-fps 30 --pose-model yolo11s-pose.pt --out-dir reports/pose_compare/yolo11s
python scripts/13_build_sequence_dataset.py --seq-dir reports/pose_compare/yolo11s/sequences --out-dir reports/pose_compare/yolo11s/dataset
# 단계 수 · 떨림 줄이기
python scripts/13_build_sequence_dataset.py --steps 45 --out-dir reports/experiments/steps45
python scripts/13_build_sequence_dataset.py --smooth 5 --out-dir reports/experiments/smooth5
```

- 고른 결과와 이유는 [Baseline 성능평가](BASELINE_EVALUATION.md)와 [기술 결정 기록](DESIGN_DECISIONS.md)에 **다른 방법 / 고른 이유 / 숫자 / 한계** 로 이어 적습니다.

## 3. 서비스용 모델 내보내기

```bash
python scripts/15_export_service_model.py                    # 기본: 정면·측면·뒤 + 좌우 반전, ExtraTrees, velocity·stats
python scripts/15_export_service_model.py --model rf --extras ""            # 11/24 오전 모델 (비교용)
python scripts/15_export_service_model.py --model rf --extras "" --views FS --no-flip   # 11/22 모델
```

- 출력 `artifacts/service/`: `action_model.joblib`(ExtraTrees), `action_cnn.pt`(1D-CNN 3개), `feature_columns.json`, `training_info.json`
  - 1D-CNN 학습은 GPU 면 약 1분, CPU 만 있으면 수십 분. ExtraTrees 만 내보내려면 `--cnn-seeds ""` (이때 예전 `action_cnn.pt` 는 지움)
- 화면에 Train S01~S03 → Validation S04 / Test S05 정확도가 방향별로 나옴 (기록용, `training_info.json` 에도 저장). 최종 모델은 5명 전부로 학습.
- 서비스 화면·대시보드에 보이는 번호·이름은 **팀 통일 표기**(A001 `kiosk_tap` 키오스크 두드리기 …): `15` 가 [`configs/unified_label_map.csv`](../configs/unified_label_map.csv) 로 Class 를 바꿔 `training_info.json` 의 `class_info` 에 넣음 (분류는 `configs/action_classes.csv`). 이름을 바꾸려면 `unified_label_map.csv` 를 고치고, 서비스 `backend/analyses/actions.py` 도 같이 (Django 는 그 목록으로 화면·대시보드를 그림). 예전 번호로 내보내기는 `--repo-ids`.

## 4. 서비스에 넣기 (store-anomaly-service)

```bash
cd ~/ai_vision/store-anomaly-service
bash scripts/update_model.sh ~/ai_vision/unmanned-store-anomaly ~/ai_vision/unmanned-store-anomaly/data/rgb/videos/T03_S04_A05_T01_F.mp4
```

`update_model.sh` 가 하는 일: 모델 파일 복사(`action_cnn.pt` 포함, 없으면 서비스 쪽도 지움) → Feature·모델 코드(`src/feature_utils.py`, `team_feature_utils.py`, `sequence_extras.py`, `temporal_cnn.py`)가 두 저장소에서 같은지 확인·맞춤 →
Pose 모델이 서비스 설정과 같은지 확인 → `docker compose up -d --build fastapi` + celery 재시작 → `/health` 확인 → (영상을 주면) E2E 업로드·분석·저장 확인.

그다음 손으로:

1. 화면 캡처 다시: `scripts/capture_screens.py --success <job> --failure <job> --multi <job>` (Playwright, 얼굴은 S04 만)
2. 서비스 README: 모델 파일 표, 6장 정확도 표, 7장 T01~T10 표, 한계(약한 동작)
3. 이 저장소 README "지금까지 결과", `BASELINE_EVALUATION.md`
4. 두 저장소 커밋 (서비스의 모델 파일은 Git 에 포함, 약 10 MB. `action_cnn.pt` 는 .gitignore 의 `*.pt` 예외로 올라감)
5. 모델 종류를 바꿨으면 `판단 불확실` 기준(30%)도 다시 확인: 학습에 없던 사람에서 30% 위쪽 정확도·표시 비율 (Baseline 성능평가 5-5 참고)

**Pose 모델을 바꿀 때만 추가로**: 서비스 `ai_service/project.json` 의 `pose_model`, `ai_service/Dockerfile` 의 가중치 받는 줄(`yolo11n-pose.pt`),
이 저장소 `configs/project.json` 의 `pose_model` 을 같이 바꾸고 03 부터 다시 (학습 Keypoint 와 서비스 Keypoint 가 같은 모델이어야 함).
CPU 에서 Frame 당 11n 65 ms · 11s 123 ms · 11m 266 ms (2026-11-24 fastapi 컨테이너 측정) → 분석 시간이 2~4배.

## 5. 자주 막히는 곳

| 증상 | 원인 · 해결 |
|---|---|
| `10` 이 "Human QA 판정이 없거나" 로 멈춤 | 새 영상의 `human_qa.csv` `human_decision` 이 빈칸 |
| `11` 이 "subject_split.json 의 사람 … 와 다릅니다" | 새 사람을 `subject_split.json` 에 안 넣음 |
| 서비스 `/health` 가 `ready: false` | `feature_columns.json` 과 코드가 만드는 Feature 이름이 다름 → Feature 코드 복사 누락. `update_model.sh` 다시 |
| 모델을 바꿨는데 서비스 결과가 그대로 | 모델이 이미지 안에 있음 → `--build fastapi` 안 함 |
| WSL 에서 `nohup` 으로 돌린 03 이 멈춤 | 창을 닫으면 WSL 이 잠듦 → 터미널을 열어 두거나 `tmux` 안에서 실행 |
| `01` 에서 "본 촬영일이 아닌 영상" | 핸드폰 원래 이름 영상이 `교과10 동작 영상/` 에 새로 들어옴 → 통일 이름으로 `공유 자료/` 에 두기 |
