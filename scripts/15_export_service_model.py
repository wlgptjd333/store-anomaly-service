"""서비스용 모델 내보내기 — 교과10 최종 프로젝트(웹 서비스)에서 쓸 Joblib·Feature 순서·학습 정보.

왜 body40_whole 인가
  - 서비스에 들어오는 영상(업로드·IP Webcam 짧은 촬영)에는 '동작 구간' 표시가 없다.
    segment 모델은 사람이 표시한 구간만 보고 배웠으므로, 구간을 모르는 영상에는 학습 때와 다른 입력이 들어간다.
  - whole 모델은 영상 전체(동작 전후 서 있는 부분 포함)를 30단계로 편 값으로 배웠다 → 서비스 입력과 같은 모양.
  - 14번 비교에서 body40_seg Val 76.5% vs body40_whole Val 73.5% (5-seed 평균 79.4% vs 77.6%),
    차이는 난수 변동(±4%p) 안이다 → 입력 모양이 맞는 쪽을 고른다.

하는 일
  1) 확인: Train S01~S03 → Validation S04, Test S05 정확도 (14번과 같은 설정, 기록용)
  2) 서비스용 최종 학습: 5명(S01~S05) 170개 전부. 평가는 1)의 수치를 README 에 적는다
  3) 저장: artifacts/service/
       store_anomaly_rf.joblib       Random Forest (17 Class)
       rf_feature_columns.json       1,200개 Feature 이름·순서 (t00_bn_nose_x …)
       rf_training_info.json         Class·이름·단계 수·Frame Feature·창 길이·평가 수치

실행: .venv/bin/python scripts/15_export_service_model.py
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs" / "baseline_config.json").read_text(encoding="utf-8"))
SPLIT = json.loads((ROOT / "configs" / "subject_split.json").read_text(encoding="utf-8"))
PROJECT = json.loads((ROOT / "configs" / "project.json").read_text(encoding="utf-8"))
DATA = ROOT / "reports" / "dataset" / "sequence_dataset_whole.csv"
OUT = ROOT / "artifacts" / "service"
STEPS = CFG["sequence_steps"]

KP = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
      "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
      "left_knee", "right_knee", "left_ankle", "right_ankle"]
ANGLES = ["left_elbow_angle", "right_elbow_angle", "left_knee_angle", "right_knee_angle"]
POS = ["hip_x_img", "hip_y_img"]
FRAME_FEATURES = [f"bn_{k}_{a}" for k in KP for a in "xy"] + ANGLES + POS  # body40
COLUMNS = [f"t{i:02d}_{f}" for i in range(STEPS) for f in FRAME_FEATURES]


def make_rf() -> RandomForestClassifier:
    m = CFG["model"]
    return RandomForestClassifier(n_estimators=m["n_estimators"], max_depth=m["max_depth"],
                                  random_state=m["random_state"], class_weight=m["class_weight"], n_jobs=-1)


def subjects(key: str) -> list[str]:
    return SPLIT["main"][key]


def clip_seconds() -> list[float]:
    """학습 영상(whole) 길이 — 서비스에서 긴 영상을 나눌 창 길이의 근거"""
    secs = []
    for r in csv.DictReader((ROOT / "configs" / "approved_clip_manifest.csv").open(encoding="utf-8-sig")):
        p = ROOT / r["sequence_file"]
        if p.exists():
            t = pd.read_csv(p, usecols=["time_sec"])["time_sec"]
            secs.append(float(t.iloc[-1] - t.iloc[0]))
    return secs


def main() -> None:
    df = pd.read_csv(DATA)
    acts = {r["action_id"]: r for r in csv.DictReader((ROOT / "configs" / "action_classes.csv").open(encoding="utf-8-sig"))}
    X, y = df[COLUMNS].to_numpy(float), df["action_id"].to_numpy()

    # 1) 기록용 평가 (Train S01~S03)
    tr = df.subject_id.isin(subjects("train")).to_numpy()
    va = df.subject_id.isin(subjects("validation")).to_numpy()
    te = df.subject_id.isin(subjects("test")).to_numpy()
    rf = make_rf().fit(X[tr], y[tr])
    ev = {}
    for name, m in (("validation_S04", va), ("test_S05", te)):
        p = rf.predict(X[m])
        ev[name] = {"accuracy": round(float(accuracy_score(y[m], p)), 4),
                    "macro_f1": round(float(f1_score(y[m], p, average="macro")), 4), "clips": int(m.sum())}
    print("평가 (Train S01~S03):", ev)

    # 2) 서비스용: 5명 전부
    final = make_rf().fit(X, y)
    secs = clip_seconds()

    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, OUT / "store_anomaly_rf.joblib", compress=3)
    (OUT / "rf_feature_columns.json").write_text(json.dumps(COLUMNS, ensure_ascii=False, indent=1), encoding="utf-8")
    classes = [str(c) for c in final.classes_]
    info = {
        "model": "RandomForestClassifier",
        "feature_set": "body40",
        "window": "whole",
        "sequence_steps": STEPS,
        "frame_features": FRAME_FEATURES,
        "n_features": len(COLUMNS),
        "classes": classes,
        "class_info": {c: {"label": acts[c]["action_label"], "name_ko": acts[c]["name_ko"],
                           "category": acts[c].get("category", "")} for c in classes},
        "rf_params": {k: v for k, v in CFG["model"].items() if k != "type"},
        "pose": {"model": PROJECT["pose_model"], "image_size": PROJECT["image_size"],
                 "person_confidence": PROJECT["person_confidence"],
                 "keypoint_confidence": PROJECT["keypoint_confidence"]},
        "valid_frame_rule": {"min_valid_frames": CFG["min_valid_frames"], "min_valid_ratio": CFG["min_valid_ratio"]},
        "body_scale": {"shoulder_ratio": 0.70, "min_px": 5.0},
        "train_clip_seconds": {"min": round(min(secs), 2), "median": round(float(np.median(secs)), 2),
                               "max": round(max(secs), 2)},
        "trained_on": "S01~S05 (170 clips, 17 actions x 2 views x 5 people)",
        "evaluation_train_S01_S03": ev,
        "sklearn_version": sklearn.__version__,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (OUT / "rf_training_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    size = (OUT / "store_anomaly_rf.joblib").stat().st_size / 1e6
    print(f"saved {OUT.relative_to(ROOT)}: joblib {size:.1f} MB, {len(COLUMNS)} features, {len(classes)} classes")
    print("학습 영상 길이(s):", info["train_clip_seconds"])


if __name__ == "__main__":
    main()
