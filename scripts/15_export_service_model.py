"""서비스용 모델 내보내기 — 교과10 최종 프로젝트(웹 서비스)에서 쓸 Joblib·Feature 순서·학습 정보.

왜 body40_whole 인가
  - 서비스에 들어오는 영상(업로드·IP Webcam 짧은 촬영)에는 '동작 구간' 표시가 없다.
    segment 모델은 사람이 표시한 구간만 보고 배웠으므로, 구간을 모르는 영상에는 학습 때와 다른 입력이 들어간다.
  - whole 모델은 영상 전체(동작 전후 서 있는 부분 포함)를 30단계로 편 값으로 배웠다 → 서비스 입력과 같은 모양.
  - 14번 비교에서 body40_seg Val 76.5% vs body40_whole Val 73.5% (5-seed 평균 79.4% vs 77.6%),
    차이는 난수 변동(±4%p) 안이다 → 입력 모양이 맞는 쪽을 고른다.

하는 일
  1) 확인: Train S01~S03 → Validation S04, Test S05 정확도 (방향별로도, 기록용)
  2) 서비스용 최종 학습: 5명(S01~S05) 전부. 평가는 1)의 수치를 README 에 적는다
  3) 저장: artifacts/service/
       action_model.joblib     분류 모델 (17 Class, ExtraTrees)
       action_cnn.pt           시간 순서 1D-CNN (난수 3개, src/temporal_cnn.py) — 서비스는 ExtraTrees 확률과 평균
       feature_columns.json    입력 Feature 이름·순서 (t00_bn_nose_x … d00_… mean_…)
       training_info.json      Class·이름·단계 수·Frame Feature·덧붙인 Feature·창 길이·평가 수치

갱신 기록
  - 11/22 정면·측면 170개, Random Forest 300그루, 1,200 Feature
  - 11/24 오전 정면·측면·뒤 255개 + 좌우 반전 (18번 실험: LOSO 뒷모습 37% → 68%)
  - 11/24 오후 ExtraTrees 1000그루 + 변화량·요약 통계 Feature (19번 실험: S01~S04 4-fold 71.9% → 82.4%)
    · 좌우 반전 복사본은 학습 데이터에만 더한다 (평가는 원본).
  - 11/24 밤 ExtraTrees + 1D-CNN 확률 평균 (21번 실험: 4-fold 82.4 → 88.7, Test S05 78.4 → 83.0)

실행: .venv/bin/python scripts/15_export_service_model.py                                   # 지금 서비스 모델 (기본)
      .venv/bin/python scripts/15_export_service_model.py --no-pos --view-aug 2 --extra-csv reports/extra/s05_newcam/sequence_dataset_whole.csv
                                                                                                # 구도 강건 옵션 (26번 결과로 고름)
      .venv/bin/python scripts/15_export_service_model.py --cnn-seeds ""                       # 11/24 오후 모델 (ExtraTrees 만)
      .venv/bin/python scripts/15_export_service_model.py --model rf --extras "" --cnn-seeds ""   # 11/24 오전 모델
      .venv/bin/python scripts/15_export_service_model.py --model rf --extras "" --views FS --no-flip   # 11/22 모델
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.temporal_cnn import CnnEnsemble  # noqa: E402
from src.sequence_extras import add_extras, extra_names  # noqa: E402

CFG = json.loads((ROOT / "configs" / "baseline_config.json").read_text(encoding="utf-8"))
SPLIT = json.loads((ROOT / "configs" / "subject_split.json").read_text(encoding="utf-8"))
PROJECT = json.loads((ROOT / "configs" / "project.json").read_text(encoding="utf-8"))
DATA = ROOT / "reports" / "dataset" / "sequence_dataset_whole.csv"
OUT = ROOT / "artifacts" / "service"
UNIFIED_MAP = ROOT / "configs" / "unified_label_map.csv"
STEPS = CFG["sequence_steps"]

KP = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
      "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
      "left_knee", "right_knee", "left_ankle", "right_ankle"]
ANGLES = ["left_elbow_angle", "right_elbow_angle", "left_knee_angle", "right_knee_angle"]
POS = ["hip_x_img", "hip_y_img"]
FRAME_FEATURES = [f"bn_{k}_{a}" for k in KP for a in "xy"] + ANGLES + POS  # body40
COLUMNS = [f"t{i:02d}_{f}" for i in range(STEPS) for f in FRAME_FEATURES]
VIEW_NAMES = {"F": "front", "S": "side", "B": "back"}

_spec = importlib.util.spec_from_file_location("t17", ROOT / "scripts" / "17_flip_and_view_check.py")
t17 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t17)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", default="FSB", help="학습에 쓸 방향 (F 정면, S 측면, B 뒤)")
    ap.add_argument("--no-flip", action="store_true", help="좌우 반전 복사본을 더하지 않음")
    ap.add_argument("--model", choices=["et", "rf"], default="et", help="et = ExtraTrees 1000, rf = RandomForest 300")
    ap.add_argument("--extras", default="velocity,stats", help='덧붙일 Feature (velocity, stats). 없으면 ""')
    ap.add_argument("--cnn-seeds", default="42,0,1",
                    help='1D-CNN 난수 (쉼표). 여러 개면 확률 평균. 없으면 "" → ExtraTrees 만 (GPU 있으면 약 1분)')
    ap.add_argument("--cnn-width", type=int, default=128,
                    help="1D-CNN 채널 폭 (22번: 64 → 128 이 4-fold 90.2 → 91.7, Test S05 86.3 → 88.2). 예전 모델은 64")
    ap.add_argument("--repo-ids", action="store_true",
                    help="Class 를 이 저장소 번호(A01..)로. 기본은 팀 통일 번호(A001.., unified_label_map.csv)")
    ap.add_argument("--no-pos", action="store_true",
                    help="골반 화면 위치(hip_x_img, hip_y_img) 빼기 — 카메라 구도가 바뀌면 값이 달라지는 Feature (26번)")
    ap.add_argument("--view-aug", type=int, default=0, help="시점 증강 사본 수 (26번 view_aug, 0 = 안 함)")
    ap.add_argument("--crop-aug", type=int, default=0, help="앞뒤 자르기 증강 사본 수 (24번 crop, 0 = 안 함)")
    ap.add_argument("--extra-csv", default="",
                    help="최종 학습에만 더할 Sequence CSV (쉼표). 예: reports/extra/s05_newcam/sequence_dataset_whole.csv "
                         "— 기록용 평가(Train S01~S03)에는 넣지 않음")
    return ap.parse_args()


def load_s26():
    spec = importlib.util.spec_from_file_location("s26", ROOT / "scripts" / "26_camera_robust.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_model(kind: str):
    m = CFG["model"]
    if kind == "rf":
        return RandomForestClassifier(n_estimators=m["n_estimators"], max_depth=m["max_depth"],
                                      random_state=m["random_state"], class_weight=m["class_weight"], n_jobs=-1)
    return ExtraTreesClassifier(n_estimators=1000, random_state=m["random_state"], class_weight="balanced", n_jobs=-1)


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
    args = parse_args()
    views, flip = list(args.views), not args.no_flip
    extras = [e for e in args.extras.split(",") if e]
    df = pd.read_csv(DATA)
    df = df[df.view.isin(views)].reset_index(drop=True)
    extra = pd.concat([pd.read_csv(ROOT / p) for p in args.extra_csv.split(",") if p.strip()] or [df.iloc[:0]],
                      ignore_index=True)
    feats = [f for f in FRAME_FEATURES if not (args.no_pos and f in POS)]
    cols = [f"t{i:02d}_{f}" for i in range(STEPS) for f in feats]
    s26 = load_s26() if (args.view_aug or args.crop_aug) else None
    acts = {r["action_id"]: r for r in csv.DictReader((ROOT / "configs" / "action_classes.csv").open(encoding="utf-8-sig"))}
    if not args.repo_ids:
        # 서비스·대시보드는 팀 통일 번호·이름(A001 kiosk_tap 키오스크 두드리기 ..)으로 보여 준다 (11/24).
        # 순서가 같아(A01→A001 …) 학습 결과·예측은 이 저장소 번호로 학습한 것과 같다.
        umap = [r for r in csv.DictReader(UNIFIED_MAP.open(encoding="utf-8-sig")) if r["kind"] == "action"]
        to_u = {r["team_repo_id"]: r["unified_id"] for r in umap}
        acts = {r["unified_id"]: {"action_label": r["unified_label"], "name_ko": r["unified_name_ko"],
                                  "category": acts[r["team_repo_id"]].get("category", ""),
                                  "repo_action_id": r["team_repo_id"]} for r in umap}
        df["action_id"] = df["action_id"].astype(str).map(to_u)
        assert df["action_id"].notna().all()
        if len(extra):
            extra["action_id"] = extra["action_id"].astype(str).map(to_u)
            assert extra["action_id"].notna().all()
    y = df["action_id"].astype(str).to_numpy()
    names = cols + extra_names(feats, STEPS, extras)
    assert add_extras(df[cols].to_numpy(float), STEPS, len(feats), extras).shape[1] == len(names)

    cnn_seeds = [int(x) for x in args.cnn_seeds.split(",") if x.strip()]

    def augment(part: pd.DataFrame) -> pd.DataFrame:
        """26번과 같은 순서: 원본 + 시점 사본 + 자르기 사본 → (뒤에서) 전체 좌우 반전"""
        if s26 is None:
            return part
        X0 = part[COLUMNS].to_numpy(float)
        rng = np.random.default_rng(CFG["model"]["random_state"])
        blocks = [X0]
        if args.view_aug:
            blocks.append(s26.view_aug(X0, args.view_aug, rng))
        if args.crop_aug:
            tmp = pd.DataFrame(X0, columns=COLUMNS).assign(subject_id="x", action_id="x", view="x")
            blocks.append(s26.aug24.augment(tmp, ("crop",), args.crop_aug, STEPS,
                                            CFG["model"]["random_state"])[COLUMNS].to_numpy(float))
        n_copy = len(blocks)
        return pd.DataFrame(np.vstack(blocks), columns=COLUMNS).assign(
            action_id=np.tile(part["action_id"].astype(str).to_numpy(), n_copy))

    def train(part: pd.DataFrame):
        """(ExtraTrees, 1D-CNN 묶음 또는 None)"""
        part = augment(part)
        if flip:
            part = pd.concat([part, t17.flip(part)], ignore_index=True)
        Xb = part[cols].to_numpy(float)
        yp = part["action_id"].astype(str).to_numpy()
        tree = make_model(args.model).fit(add_extras(Xb, STEPS, len(feats), extras), yp)
        cnn = None
        if cnn_seeds:
            idx = {c: i for i, c in enumerate(tree.classes_)}      # CNN 출력 순서 = ExtraTrees classes_ 순서
            cnn = CnnEnsemble.fit(Xb, np.array([idx[a] for a in yp]), STEPS, len(feats),
                                  len(tree.classes_), cnn_seeds, width=args.cnn_width)
        return tree, cnn

    def predict(models, Xb: np.ndarray) -> np.ndarray:
        tree, cnn = models
        p = tree.predict_proba(add_extras(Xb, STEPS, len(feats), extras))
        if cnn is not None:
            p = (p + cnn.predict_proba(Xb)) / 2
        return tree.classes_[p.argmax(1)]

    Xbase = df[cols].to_numpy(float)

    # 1) 기록용 평가 (Train S01~S03) — 방향별로도 적음
    tr = df.subject_id.isin(subjects("train")).to_numpy()
    model = train(df[tr])
    ev = {}
    for name, key in (("validation_S04", "validation"), ("test_S05", "test")):
        m = df.subject_id.isin(subjects(key)).to_numpy()
        p = predict(model, Xbase[m])
        ev[name] = {"accuracy": round(float(accuracy_score(y[m], p)), 4),
                    "macro_f1": round(float(f1_score(y[m], p, average="macro")), 4), "clips": int(m.sum())}
        for v in views:
            mv = (df.view.to_numpy()[m] == v)
            ev[name][f"accuracy_{VIEW_NAMES[v]}"] = round(float(accuracy_score(y[m][mv], p[mv])), 4)
    print("평가 (Train S01~S03):", ev)

    # 2) 서비스용: 5명 전부 (+ --extra-csv 새 구도 영상)
    final, final_cnn = train(pd.concat([df, extra], ignore_index=True))
    secs = clip_seconds()

    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, OUT / "action_model.joblib", compress=3)
    if final_cnn is not None:
        final_cnn.save(OUT / "action_cnn.pt")
    elif (OUT / "action_cnn.pt").exists():
        (OUT / "action_cnn.pt").unlink()          # ExtraTrees 만 내보낼 땐 예전 CNN 을 남기지 않음
    (OUT / "feature_columns.json").write_text(json.dumps(names, ensure_ascii=False, indent=1), encoding="utf-8")
    classes = [str(c) for c in final.classes_]
    params = {k: v for k, v in final.get_params().items() if k in ("n_estimators", "max_depth", "class_weight",
                                                                    "random_state", "max_features")}
    info = {
        "model": type(final).__name__,
        "feature_set": "body40",
        "window": "whole",
        "sequence_steps": STEPS,
        "frame_features": feats,
        "sequence_extras": extras,
        "n_features": len(names),
        "classes": classes,
        "class_info": {c: {"label": acts[c]["action_label"], "name_ko": acts[c]["name_ko"],
                           "category": acts[c].get("category", ""),
                           **({"repo_action_id": acts[c]["repo_action_id"]} if "repo_action_id" in acts[c] else {})}
                       for c in classes},
        "class_ids": "repo" if args.repo_ids else "unified",
        "ensemble": ({"cnn_file": "action_cnn.pt", "cnn_seeds": cnn_seeds, "cnn_width": args.cnn_width,
                      "combine": "mean(ExtraTrees, 1D-CNN)",
                      "code": "src/temporal_cnn.py"} if final_cnn is not None else None),
        "model_params": params,
        "pose": {"model": PROJECT["pose_model"], "image_size": PROJECT["image_size"],
                 "person_confidence": PROJECT["person_confidence"],
                 "keypoint_confidence": PROJECT["keypoint_confidence"]},
        "valid_frame_rule": {"min_valid_frames": CFG["min_valid_frames"], "min_valid_ratio": CFG["min_valid_ratio"]},
        "body_scale": {"shoulder_ratio": 0.70, "min_px": 5.0},
        "train_clip_seconds": {"min": round(min(secs), 2), "median": round(float(np.median(secs)), 2),
                               "max": round(max(secs), 2)},
        "trained_on": f"{', '.join(sorted(df.subject_id.unique()))} ({len(df)} clips, {df.action_id.nunique()} actions x {len(views)} views x {df.subject_id.nunique()} people)"
                      + (f" + new camera {len(extra)} clips" if len(extra) else ""),
        "views": [VIEW_NAMES[v] for v in views],
        "flip_augmentation": flip,
        "camera_augmentation": {"view_copies": args.view_aug, "crop_copies": args.crop_aug,
                                "code": "scripts/26_camera_robust.py view_aug, scripts/24_geo_augment.py crop"},
        "extra_training_clips": int(len(extra)),
        "evaluation_train_S01_S03": ev,
        "sklearn_version": sklearn.__version__,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (OUT / "training_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    size = (OUT / "action_model.joblib").stat().st_size / 1e6
    print(f"saved {OUT.relative_to(ROOT)}: {info['model']} joblib {size:.1f} MB, {len(names)} features, {len(classes)} classes")
    print("학습 영상 길이(s):", info["train_clip_seconds"])


if __name__ == "__main__":
    main()
