"""좌우 반전 증강과 촬영 각도 일반화 확인 (2026-11-23).

왜
  - 촬영 때 사람마다 왼쪽/오른쪽을 보거나 쓰는 손이 달랐다. 좌표 숫자로 배우는 모델에는 방향이 반대면 다른 예로 보인다.
    → 학습 데이터만 좌우로 뒤집은 복사본을 더하면(2배) 처음 보는 사람에게 더 강해지는지 확인한다.
  - "대각선·뒤에서 더 찍어야 하나?" → 한 방향만 학습했을 때 처음 보는 방향을 얼마나 맞히는지 잰다.

뒤집는 법 (Feature 단계, YOLO 다시 안 돌림)
  - bn_* 좌표는 골반 중심이 원점 → x 부호 반전 + 왼쪽↔오른쪽 관절 이름 교환 (6쌍·12개)
  - 팔꿈치·무릎 각도 왼쪽↔오른쪽 교환, 골반 화면 위치 hip_x_img → 1 - x
  - Validation·Test 는 원본 그대로 평가 (뒤집은 복사본은 원본과 같은 사람 쪽에만)

결과 → reports/tables/flip_augmentation.csv, reports/tables/view_generalization.csv
실행: .venv/bin/python scripts/17_flip_and_view_check.py
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("t14", ROOT / "scripts" / "14_train_baseline.py")
t14 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t14)
SPLIT = json.loads((ROOT / "configs" / "subject_split.json").read_text(encoding="utf-8"))["main"]
TABLES = ROOT / "reports" / "tables"
STEPS = 30
KP = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow",
      "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee",
      "left_ankle", "right_ankle"]
COLS = t14.cols("body40", "sequence")
SEEDS = [None, 0, 1, 2, 3, 4]  # None = 기준 난수 42


def swap_name(n: str) -> str:
    return n.replace("left_", "TMP_").replace("right_", "left_").replace("TMP_", "right_")


def flip(df: pd.DataFrame) -> pd.DataFrame:
    f = df.copy()
    for t in range(STEPS):
        p = f"t{t:02d}_"
        for k in KP:
            src = swap_name(k)
            f[p + f"bn_{k}_x"] = -df[p + f"bn_{src}_x"]
            f[p + f"bn_{k}_y"] = df[p + f"bn_{src}_y"]
        for side in ("elbow", "knee"):
            f[p + f"left_{side}_angle"] = df[p + f"right_{side}_angle"]
            f[p + f"right_{side}_angle"] = df[p + f"left_{side}_angle"]
        f[p + "hip_x_img"] = 1 - df[p + "hip_x_img"]
    return f


def acc(train: pd.DataFrame, test: pd.DataFrame, aug: bool, seed=None) -> float:
    if aug:
        train = pd.concat([train, flip(train)], ignore_index=True)
    rf = t14.model(seed).fit(train[COLS].to_numpy(float), train.action_id.astype(str))
    return round(accuracy_score(test.action_id.astype(str), rf.predict(test[COLS].to_numpy(float))) * 100, 1)


def main() -> None:
    rows, views = [], []
    for window in ("segment", "whole"):
        df = t14.load_dataset(window)
        assert np.allclose(flip(flip(df))[COLS].to_numpy(float), df[COLS].to_numpy(float), equal_nan=True)
        subs = sorted(df.subject_id.unique())
        tr = df[df.subject_id.isin(SPLIT["train"])]
        for aug in (False, True):
            for name, part in (("validation_S04", SPLIT["validation"]), ("test_S05", SPLIT["test"])):
                vals = [acc(tr, df[df.subject_id.isin(part)], aug, s) for s in SEEDS]
                rows.append(dict(window=window, flip=aug, eval=name, acc_seed42=vals[0],
                                 acc_6seed_mean=round(np.mean(vals), 1), acc_6seed_std=round(np.std(vals), 1)))
            loso = [acc(df[df.subject_id != s], df[df.subject_id == s], aug) for s in subs]
            rows.append(dict(window=window, flip=aug, eval="loso_mean", acc_seed42=round(np.mean(loso), 1),
                             **{f"loso_{s}": a for s, a in zip(subs, loso)}))
            for test_view in ("F", "S"):
                other = "S" if test_view == "F" else "F"
                for train_name, tv in (("both", ["F", "S"]), ("same_only", [test_view]), ("other_only", [other])):
                    a = [acc(df[(df.subject_id != s) & df.view.isin(tv)], df[(df.subject_id == s) & (df.view == test_view)], aug)
                         for s in subs]
                    views.append(dict(window=window, flip=aug, test_view=test_view, train_views=train_name,
                                      loso_acc=round(np.mean(a), 1)))
    R, V = pd.DataFrame(rows), pd.DataFrame(views)
    R.to_csv(TABLES / "flip_augmentation.csv", index=False, encoding="utf-8-sig")
    V.to_csv(TABLES / "view_generalization.csv", index=False, encoding="utf-8-sig")
    print(R[["window", "flip", "eval", "acc_seed42", "acc_6seed_mean", "acc_6seed_std"]].to_string(index=False))
    print(V.pivot_table(index=["window", "test_view", "train_views"], columns="flip", values="loso_acc", sort=False).to_string())


if __name__ == "__main__":
    main()
