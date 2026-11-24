#!/usr/bin/env bash
# 데이터 저장소(unmanned-store-anomaly)에서 새로 내보낸 모델을 서비스에 넣고, 다시 빌드한 뒤 확인까지 한 번에.
#
#   bash scripts/update_model.sh [데이터 저장소 경로] [확인용 영상.mp4]
#     기본 경로: ~/ai_vision/unmanned-store-anomaly
#     DRY_RUN=1 bash scripts/update_model.sh  → 복사·확인까지만 (Docker 빌드 안 함)
#     영상을 주면 e2e_check.py 로 업로드 → 분석 → 저장까지 확인 (.env 의 DEMO_* 계정 사용)
#
# 하는 일
#   1) 모델 파일 3개 복사: artifacts/service/{action_model.joblib, feature_columns.json, training_info.json} → ai_service/models/
#   2) Feature 코드 3개가 두 저장소에서 같은지 확인, 다르면 데이터 저장소 쪽으로 맞춤
#      (학습과 추론이 같은 코드로 Feature 를 만들어야 해서 — 다르면 정확도가 소리 없이 떨어짐)
#   3) training_info.json 의 Pose 모델이 project.json 과 같은지 확인 (다르면 멈춤 — Pose 모델을 바꿨으면 project.json·Dockerfile 도 같이)
#   4) docker compose up -d --build fastapi, celery 재시작 (모델이 이미지 안에 들어가므로 빌드가 필요)
#   5) /health 에서 모델 이름·Feature 수·Class 수 확인
set -euo pipefail
cd "$(dirname "$0")/.."
DATA="${1:-$HOME/ai_vision/unmanned-store-anomaly}"
VIDEO="${2:-}"
ART="$DATA/artifacts/service"

for f in action_model.joblib feature_columns.json training_info.json; do
  test -f "$ART/$f" || { echo "없음: $ART/$f — 데이터 저장소에서 scripts/15_export_service_model.py 를 먼저 실행"; exit 1; }
done

python3 - "$ART/training_info.json" ai_service/project.json <<'PY'
import json, sys
info, proj = (json.load(open(p, encoding="utf-8")) for p in sys.argv[1:])
from pathlib import PurePosixPath as P
if P(info["pose"]["model"]).name != P(proj["pose_model"]).name:   # 서비스는 models/ 아래 경로로 적음
    sys.exit(f"Pose 모델이 다릅니다: 학습 {info['pose']['model']} / 서비스 {proj['pose_model']} — project.json·Dockerfile 을 먼저 맞추세요")
print(f"새 모델: {info['model']} · Feature {info['n_features']} · Class {len(info['classes'])} · {info['trained_on']}")
print(f"  기록용 평가: {info.get('evaluation_train_S01_S03')}")
PY

cp "$ART"/action_model.joblib "$ART"/feature_columns.json "$ART"/training_info.json ai_service/models/
for f in feature_utils team_feature_utils sequence_extras; do
  if ! cmp -s "$DATA/src/$f.py" "ai_service/src/$f.py"; then
    cp "$DATA/src/$f.py" "ai_service/src/$f.py"
    echo "Feature 코드 갱신: ai_service/src/$f.py (git diff 로 확인)"
  fi
done

if [ -n "${DRY_RUN:-}" ]; then echo "DRY_RUN: 복사·확인까지만 (빌드 안 함)"; git status --short ai_service; exit 0; fi
docker compose up -d --build fastapi
docker compose restart celery
for i in $(seq 1 60); do
  if H=$(curl -sf http://127.0.0.1:8001/health); then break; fi
  sleep 3
done
echo "$H" | python3 -c "import json,sys; m=json.load(sys.stdin)['model']; print('health:', 'ready' if m['ready'] else 'ERROR ' + m['error'], m['model'], m['n_features'], 'features', m['n_classes'], 'classes'); sys.exit(0 if m['ready'] else 1)"

if [ -n "$VIDEO" ]; then
  backend/.venv/bin/python scripts/e2e_check.py "$VIDEO"
fi
echo "끝. README 의 정확도 표·T01~T10 표와 docs/images 화면 캡처(scripts/capture_screens.py)를 새 숫자로 고친 뒤 커밋하세요."
