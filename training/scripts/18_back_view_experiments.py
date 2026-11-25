"""뒷모습(B) 추가 실험 — 2026-11-23 재촬영 85개 (5명 × 17동작, 통일 번호 T03).

왜
  - 17번: 학습에 없던 촬영 각도는 정확도가 약 25% 로 무너졌다.
    무인 매장 카메라는 키오스크·뽑기 기계·냉동고 앞에 선 사람의 등을 보는 경우가 많다.
  - 그래서 뒷모습을 찍어 넣었을 때
      (1) 뒷모습을 얼마나 맞히는지  (2) 정면·측면 성능이 떨어지지는 않는지
      (3) 좌우 반전 증강·얼굴 관절 빼기가 도움이 되는지 를 숫자로 확인한다.

비교 (body40 = 몸 크기 정규화 17관절 + 각도 4 + 골반 위치 2, 30단계, Random Forest 기준 설정)
  A. LOSO 5-fold (처음 보는 사람): 학습 방향 FS(기존) / FSB(뒤 추가) / B만 × 좌우 반전 없음·있음
     → 평가는 정면·측면·뒤 따로. 난수 3개(42, 0, 1) 평균.
  B. 얼굴 관절(코·눈·귀 5개 = 10열/단계) 빼기 (body30): 뒤에서는 얼굴이 안 보여 YOLO 가 추정한 값이 흔들린다.
  C. 기본 Split: Train S01~S03 → Validation S04 / Test S05, 난수 6개 평균.
  D. 가장 좋은 설정의 동작별 정확도와 뒷모습에서 자주 헷갈린 쌍.

결과 → reports/tables/back_view_loso.csv, back_view_split.csv, back_view_per_action.csv,
        back_view_confusions.csv, back_view_predictions.csv, docs/images/back_view_summary.png
실행: .venv/bin/python scripts/18_back_view_experiments.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import load_script  # noqa: E402



t14 = load_script("14_train_baseline.py")
t17 = load_script("17_flip_and_view_check.py")
SPLIT = json.loads((ROOT / "configs" / "subject_split.json").read_text(encoding="utf-8"))["main"]
TABLES = ROOT / "reports" / "tables"
IMG = ROOT / "docs" / "images"
STEPS = t14.STEPS
FACE = ["nose", "left_eye", "right_eye", "left_ear", "right_ear"]
VIEW_KO = {"F": "정면", "S": "측면", "B": "뒤"}
TRAIN_SETS = {"FS": ["F", "S"], "FSB": ["F", "S", "B"], "B": ["B"]}
LOSO_SEEDS = [None, 0, 1]          # None = 기준 난수 42
SPLIT_SEEDS = [None, 0, 1, 2, 3, 4]


def feature_cols(face: bool) -> list[str]:
    fs = t14.FSETS["body40"]
    if not face:
        fs = [c for c in fs if not any(c.startswith(f"bn_{k}_") for k in FACE)]
    return [f"t{i:02d}_{x}" for i in range(STEPS) for x in fs]


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], flip: bool, seed):
    if flip:
        train = pd.concat([train, t17.flip(train)], ignore_index=True)
    rf = t14.model(seed).fit(train[cols].to_numpy(float), train.action_id.astype(str))
    proba = rf.predict_proba(test[cols].to_numpy(float))
    return rf.classes_[proba.argmax(1)], proba.max(1)


def configs():
    """(이름, 학습 방향, 좌우 반전, 얼굴 관절 사용)"""
    out = [(f"{ts}{'+flip' if fl else ''}", ts, fl, True) for ts in TRAIN_SETS for fl in (False, True)]
    out += [("FSB-face", "FSB", False, False), ("FSB+flip-face", "FSB", True, False)]
    return out


def main() -> None:
    loso_rows, split_rows, pred_rows = [], [], []
    for window in ("segment", "whole"):
        df = t14.load_dataset(window, views=None)
        subs = sorted(df.subject_id.unique())
        print(f"[{window}] 영상 {len(df)}개 {df.groupby('view').size().to_dict()}")
        for name, ts, fl, face in configs():
            cols = feature_cols(face)
            # A·B. LOSO
            rows = []
            for seed in LOSO_SEEDS:
                for s in subs:
                    tr = df[(df.subject_id != s) & df.view.isin(TRAIN_SETS[ts])]
                    te = df[df.subject_id == s]
                    pred, conf = fit_predict(tr, te, cols, fl, seed)
                    for r, p, c in zip(te.itertuples(), pred, conf):
                        rows.append(dict(window=window, config=name, seed=42 if seed is None else seed,
                                         clip_id=r.clip_id, subject_id=s, view=r.view, true=r.action_id,
                                         pred=p, correct=int(p == r.action_id), confidence=round(float(c), 3)))
            pred_rows += rows
            P = pd.DataFrame(rows)
            row = dict(window=window, config=name, train_views=ts, flip=fl, face=face)
            for v in ("F", "S", "B"):
                row[f"acc_{v}"] = round(P[P.view == v].correct.mean() * 100, 1)
            row["acc_FS"] = round(P[P.view.isin(["F", "S"])].correct.mean() * 100, 1)
            row["acc_all"] = round(P.correct.mean() * 100, 1)
            by_s = P[P.view == "B"].groupby("subject_id").correct.mean() * 100
            row.update({f"B_{s}": round(a, 1) for s, a in by_s.items()})
            loso_rows.append(row)
            print(f"  LOSO {name:14s} 정면 {row['acc_F']:5.1f} 측면 {row['acc_S']:5.1f} 뒤 {row['acc_B']:5.1f}"
                  f" | 정면·측면 {row['acc_FS']:5.1f} 전체 {row['acc_all']:5.1f}", flush=True)
            # C. 기본 Split (얼굴 있는 FS / FSB 만)
            if face and ts != "B":
                tr = df[df.subject_id.isin(SPLIT["train"]) & df.view.isin(TRAIN_SETS[ts])]
                for split_name, part in (("validation_S04", SPLIT["validation"]), ("test_S05", SPLIT["test"])):
                    te = df[df.subject_id.isin(part)]
                    accs = {v: [] for v in ("F", "S", "B", "all")}
                    for seed in SPLIT_SEEDS:
                        pred, _ = fit_predict(tr, te, cols, fl, seed)
                        ok = pred == te.action_id.to_numpy(str)
                        for v in ("F", "S", "B"):
                            accs[v].append(ok[(te.view == v).to_numpy()].mean() * 100)
                        accs["all"].append(ok.mean() * 100)
                    split_rows.append(dict(window=window, config=name, eval=split_name,
                                           **{f"acc_{k}_6seed_mean": round(np.mean(a), 1) for k, a in accs.items()},
                                           **{f"acc_{k}_6seed_std": round(np.std(a), 1) for k, a in accs.items()}))

    L, S, P = pd.DataFrame(loso_rows), pd.DataFrame(split_rows), pd.DataFrame(pred_rows)
    L.to_csv(TABLES / "back_view_loso.csv", index=False, encoding="utf-8-sig")
    S.to_csv(TABLES / "back_view_split.csv", index=False, encoding="utf-8-sig")
    print(S.to_string(index=False))

    # D. 가장 좋은 설정 = 뒤를 넣은 설정 중 LOSO 전체 정확도가 가장 높은 것
    best = L[L.train_views == "FSB"].sort_values("acc_all", ascending=False).iloc[0]
    Pb = P[(P.window == best.window) & (P.config == best.config)]
    print(f"가장 좋은 설정: {best.window} {best.config} (전체 {best.acc_all}, 뒤 {best.acc_B})")
    acts = pd.read_csv(ROOT / "configs" / "action_classes.csv", encoding="utf-8-sig").set_index("action_id")
    per = (Pb.pivot_table(index="true", columns="view", values="correct", aggfunc="mean") * 100).round(1)
    per = per.reindex(columns=["F", "S", "B"]).rename(columns=lambda v: f"acc_{v}")
    per.insert(0, "name_ko", acts.loc[per.index, "name_ko"])
    per.index.name = "action_id"
    per.reset_index().assign(window=best.window, config=best.config).to_csv(
        TABLES / "back_view_per_action.csv", index=False, encoding="utf-8-sig")
    wrong = Pb[(Pb.view == "B") & (Pb.correct == 0)]
    conf = wrong.groupby(["true", "pred"]).size().rename("count").reset_index().sort_values("count", ascending=False)
    conf["true_ko"] = conf["true"].map(acts["name_ko"])
    conf["pred_ko"] = conf["pred"].map(acts["name_ko"])
    conf.to_csv(TABLES / "back_view_confusions.csv", index=False, encoding="utf-8-sig")
    # 영상별: 난수 3개 중 몇 번 맞혔나 (라벨 확인용)
    clip = Pb.groupby(["clip_id", "subject_id", "view", "true"]).agg(
        correct_of_3=("correct", "sum"), pred=("pred", lambda s: s.mode().iloc[0]),
        mean_conf=("confidence", "mean")).reset_index()
    clip["mean_conf"] = clip["mean_conf"].round(3)
    clip.assign(window=best.window, config=best.config).to_csv(
        TABLES / "back_view_predictions.csv", index=False, encoding="utf-8-sig")
    print(per.to_string())
    print(conf.head(12).to_string(index=False))

    # 그림
    t14.setup_font()
    order = ["FS", "FS+flip", "B", "FSB", "FSB+flip", "FSB+flip-face"]
    labels = ["정면·측면\n(기존)", "정면·측면\n+반전", "뒤만", "정면·측면·뒤", "정면·측면·뒤\n+반전", "정면·측면·뒤\n+반전 -얼굴"]
    colors = {"F": "#2f6fd6", "S": "#3a9e5c", "B": "#c23b7a"}
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.2), sharey=True)
    for ax, window in zip(axes, ("segment", "whole")):
        g = L[L.window == window].set_index("config").reindex(order)
        x = np.arange(len(order))
        for i, v in enumerate("FSB"):
            bars = ax.bar(x + (i - 1) * 0.27, g[f"acc_{v}"], 0.27, label=f"{VIEW_KO[v]} 영상으로 평가", color=colors[v])
            for b, val in zip(bars, g[f"acc_{v}"]):
                ax.text(b.get_x() + b.get_width() / 2, val + 1, f"{val:.0f}", ha="center", fontsize=7)
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_title(f"{'동작 구간' if window == 'segment' else '영상 전체'} 30단계", fontsize=11)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("LOSO 정확도 % (처음 보는 사람, 난수 3개 평균)")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("뒷모습을 학습에 넣으면? — 가로축: 학습에 쓴 방향, 색: 평가한 방향", fontsize=12)
    fig.tight_layout()
    fig.savefig(IMG / "back_view_summary.png", dpi=100, facecolor="white", bbox_inches="tight")
    print("Back view experiments: COMPLETE")


if __name__ == "__main__":
    main()
