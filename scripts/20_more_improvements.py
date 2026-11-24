"""서비스 모델 정확도 더 올리기 (2026-11-24 저녁) — 19번에서 고른 모델(ExtraTrees + 변화량·요약 통계)을 기준으로
입력 쪽(Pose 모델 크기, 단계 수, 떨림 줄이기)과 예측 쪽(좌우 반전 평균, 나무 설정, 두 모델 섞기)을 하나씩 바꿔 본다.

고르는 규칙은 19번과 같다: S01~S04 사람 단위 4-fold, 난수 3개 평균. Test S05 는 고른 뒤 확인만.
학습 데이터: 정면·측면·뒤 + 좌우 반전, whole 창.

비교 항목
  A. Pose 모델 크기 — yolo11n / 11s / 11m 로 Keypoint 를 다시 뽑음 (03번 --pose-model --max-fps 30).
     서비스가 30fps 로 줄여 처리하므로 세 모델 모두 30fps 로 같은 조건에서 비교. 기준(지금 데이터)은 11n 모든 Frame.
  B. 단계 수 20 / 30 / 45 (13번 --steps)
  C. 보간 전 5 Frame 이동 평균으로 관절 떨림 줄이기 (13번 --smooth 5)
  D. 좌우 반전 평균 (TTA): 예측할 때 원본과 좌우 반전본의 확률을 평균
  E. ExtraTrees 설정: max_features sqrt(기본) / 0.1 / 0.25, min_samples_leaf 2
  F. ExtraTrees + SVM 확률 평균
  G. 시간 자르기 증강: 학습 Sequence 의 앞뒤를 0~15% 무작위로 잘라 다시 30단계로 편 복사본을 더함
     (whole 창에서 동작 시작 시점이 사람마다 다른 문제를 직접 겨냥)

준비 (데이터 만들기):
  python scripts/03_build_sequences.py --all --approved --max-fps 30 --pose-model yolo11{n,s,m}-pose.pt --out-dir reports/pose_compare/yolo11{n,s,m}
  python scripts/13_build_sequence_dataset.py --seq-dir reports/pose_compare/yolo11m/sequences --out-dir reports/pose_compare/yolo11m/dataset   (n·s 도)
  python scripts/13_build_sequence_dataset.py --steps 20 --out-dir reports/experiments/steps20   (45 도)
  python scripts/13_build_sequence_dataset.py --smooth 5 --out-dir reports/experiments/smooth5
실행: .venv/bin/python scripts/20_more_improvements.py
결과: reports/tables/more_improvements_cv.csv  (부분 실행: .venv/bin/python scripts/20_more_improvements.py "Pose 11m" → 그 행만 바꿔 합침)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.sequence_extras import add_extras  # noqa: E402

SPLIT = json.loads((ROOT / "configs" / "subject_split.json").read_text(encoding="utf-8"))["main"]
TABLES = ROOT / "reports" / "tables"
KP = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
      "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
      "left_knee", "right_knee", "left_ankle", "right_ankle"]
ANGLES = ["left_elbow_angle", "right_elbow_angle", "left_knee_angle", "right_knee_angle"]
FRAME = [f"bn_{k}_{a}" for k in KP for a in "xy"] + ANGLES + ["hip_x_img", "hip_y_img"]
EXTRAS = ["velocity", "stats"]
SEEDS = (42, 0, 1)

DATASETS = {   # 이름: (파일, 단계 수)
    "지금 (11n, 모든 Frame, 30단계)": ("reports/dataset/sequence_dataset_whole.csv", 30),
    "Pose 11n · 30fps": ("reports/pose_compare/yolo11n/dataset/sequence_dataset_whole.csv", 30),
    "Pose 11s · 30fps": ("reports/pose_compare/yolo11s/dataset/sequence_dataset_whole.csv", 30),
    "Pose 11m · 30fps": ("reports/pose_compare/yolo11m/dataset/sequence_dataset_whole.csv", 30),
    "20단계": ("reports/experiments/steps20/sequence_dataset_whole.csv", 20),
    "45단계": ("reports/experiments/steps45/sequence_dataset_whole.csv", 45),
    "5 Frame 이동 평균": ("reports/experiments/smooth5/sequence_dataset_whole.csv", 30),
}


def swap(n: str) -> str:
    return n.replace("left_", "TMP_").replace("right_", "left_").replace("TMP_", "right_")


def flip_matrix(X: np.ndarray, steps: int) -> np.ndarray:
    """(n, steps × 40) 좌우 반전 — 17번 flip 과 같은 규칙 (x → -x, 왼쪽·오른쪽 관절·각도 바꿈, 골반 x → 1 - x)"""
    seq = X.reshape(len(X), steps, len(FRAME))
    idx = {f: i for i, f in enumerate(FRAME)}
    out = np.empty_like(seq)
    for f, i in idx.items():
        src = seq[:, :, idx[swap(f)]]
        if f.startswith("bn_") and f.endswith("_x"):
            out[:, :, i] = -src
        elif f == "hip_x_img":
            out[:, :, i] = 1 - src
        else:
            out[:, :, i] = src
    return out.reshape(len(X), -1)


def time_crop(X: np.ndarray, steps: int, rng, copies: int, cut: float = 0.15) -> np.ndarray:
    seq = X.reshape(len(X), steps, -1)
    grid = np.linspace(0, 1, steps)
    out = []
    for _ in range(copies):
        for s in seq:
            a, b = rng.uniform(0, cut), rng.uniform(1 - cut, 1)
            tx = a + (b - a) * grid
            out.append(np.stack([np.interp(tx, grid, s[:, j]) for j in range(s.shape[1])], 1).reshape(-1))
    return np.array(out)


def et(seed, max_features="sqrt", leaf=1):
    return ExtraTreesClassifier(1000, class_weight="balanced", random_state=seed, n_jobs=-1,
                                max_features=max_features, min_samples_leaf=leaf)


def svm(seed):
    return make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", probability=True, class_weight="balanced",
                                               random_state=seed))


def run(Xtr, ytr, Xte, steps, seed, model="et", tta=False, crop=0, **kw):
    Xtr_all = np.vstack([Xtr, flip_matrix(Xtr, steps)])
    ytr_all = np.concatenate([ytr, ytr])
    if crop:
        rng = np.random.default_rng(seed)
        Xtr_all = np.vstack([Xtr_all, time_crop(Xtr_all, steps, rng, crop)])
        ytr_all = np.concatenate([ytr_all] * (crop + 1))
    F = lambda X: add_extras(X, steps, len(FRAME), EXTRAS)  # noqa: E731
    models = [et(seed, **kw)] if model == "et" else [et(seed), svm(seed)]
    probs = []
    for m in models:
        m.fit(F(Xtr_all), ytr_all)
        p = m.predict_proba(F(Xte))
        if tta:
            p = (p + m.predict_proba(F(flip_matrix(Xte, steps)))) / 2
        probs.append(p)
    return models[0].classes_[np.mean(probs, 0).argmax(1)]


def load(name):
    file, steps = DATASETS[name]
    path = ROOT / file
    if not path.exists():
        return None, steps
    df = pd.read_csv(path)
    return df, steps


def cv(df, steps, subjects, seed, **kw):
    cols = [f"t{i:02d}_{f}" for i in range(steps) for f in FRAME]
    out = []
    for s in subjects:
        tr, te = df[df.subject_id.isin([x for x in subjects if x != s])], df[df.subject_id == s]
        p = run(tr[cols].to_numpy(float), tr.action_id.astype(str).to_numpy(), te[cols].to_numpy(float), steps, seed, **kw)
        out.append(te[["subject_id", "view", "action_id"]].assign(correct=(p == te.action_id.astype(str).to_numpy())))
    return pd.concat(out)


def score(P):
    return {"acc_all": P.correct.mean() * 100, "acc_FS": P[P.view != "B"].correct.mean() * 100,
            "acc_B": P[P.view == "B"].correct.mean() * 100,
            "subj_min": P.groupby("subject_id").correct.mean().min() * 100}


def configs():
    """(이름, 데이터, 설정)"""
    out = [(n, n, {}) for n in DATASETS]
    base = "지금 (11n, 모든 Frame, 30단계)"
    out += [("좌우 반전 평균 (TTA)", base, {"tta": True}),
            ("max_features 0.1", base, {"max_features": 0.1}),
            ("max_features 0.25", base, {"max_features": 0.25}),
            ("min_samples_leaf 2", base, {"leaf": 2}),
            ("ExtraTrees + SVM 평균", base, {"model": "et+svm"}),
            ("시간 자르기 증강 ×1", base, {"crop": 1}),
            ("시간 자르기 증강 ×2", base, {"crop": 2})]
    return out


def main() -> None:
    dev = SPLIT["train"] + SPLIT["validation"]
    only = sys.argv[1:]          # 이름 일부를 주면 그 설정만 (Pose 데이터가 늦게 끝날 때)
    rows = []
    for name, data, kw in configs():
        if only and not any(o in name for o in only):
            continue
        df, steps = load(data)
        if df is None:
            print(f"  (건너뜀) {name}: 데이터 없음")
            continue
        t = time.time()
        sc = [score(cv(df, steps, dev, seed, **kw)) for seed in SEEDS]
        r = {"config": name, "clips": len(df), **{k: round(float(np.mean([s[k] for s in sc])), 1) for k in sc[0]},
             "acc_all_std": round(float(np.std([s["acc_all"] for s in sc])), 1), "sec": round(time.time() - t)}
        rows.append(r)
        print(f"  {name:28s} 전체 {r['acc_all']:5.1f} ± {r['acc_all_std']:3.1f} | 정면·측면 {r['acc_FS']:5.1f} 뒤 {r['acc_B']:5.1f}"
              f" | 가장 낮은 사람 {r['subj_min']:5.1f} ({r['clips']}개, {r['sec']}s)", flush=True)
    C = pd.DataFrame(rows)
    out = TABLES / "more_improvements_cv.csv"
    if only and out.exists():   # 부분 실행(이름 일부를 줬을 때)은 기존 표에서 그 행만 바꿈
        old = pd.read_csv(out)
        C = pd.concat([old[~old.config.isin(C.config)], C], ignore_index=True)
    C.to_csv(out, index=False, encoding="utf-8-sig")
    print(C.sort_values("acc_all", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
