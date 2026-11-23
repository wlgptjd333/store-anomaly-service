"""Action Recognition Baseline — 수업 6일차 (Random Forest, 사람 단위 평가, 한 번에 조건 하나씩 비교).

규칙
  - 고르는 데 쓰는 것: Validation(S04, Train S01~S03 으로 학습) + S01~S04 사람 단위 4-fold
  - Test(S05, 다른 날 촬영)는 고른 뒤 마지막에 한 번만 본다 (수업 PART 8)
  - 모델·하이퍼파라미터는 configs/baseline_config.json 하나로 고정 → 실험마다 바꾸는 건 Feature·창·Static 여부 하나뿐

실험 (기준 = 3팀 body40 · 동작 구간 · Sequence 30단계)
  course38_whole   수업 그대로: 어깨 폭 38-D, 영상 전체
  course38_seg     수업 38-D + 동작 구간
  course34_seg     수업 38-D 에서 각도 제외 (수업 실험 32)
  course38_static  수업 38-D 가운데 Frame 1개 (수업 실험 38)
  body40_seg       3팀 기준 모델
  body40_whole     기준에서 창만 영상 전체로
  body40_static    기준에서 가운데 Frame 1개로
  body38_seg       기준에서 골반 화면 위치 제외
  body36_seg       기준에서 각도 제외

출력
  reports/tables/baseline_experiments.csv     실험별 Validation·4-fold 결과
  reports/tables/baseline_predictions.csv     기준·수업 모델의 Validation/Test 영상별 예측·확신도
  reports/tables/baseline_per_action.csv      동작별 Precision·Recall·F1 (Validation, Test)
  reports/tables/baseline_wrong_clips.csv     틀린 영상과 원인 추적용 정보
  reports/tables/baseline_test.csv            마지막 Test 결과
  reports/tables/baseline_importance.csv      관절별·시간대별 Feature 중요도
  docs/images/baseline_confusion.png          혼동 행렬 (Validation S04 / Test S05)
  docs/images/baseline_importance.png         어느 관절·어느 시간대를 보고 판단했나
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs" / "baseline_config.json").read_text(encoding="utf-8"))
SPLIT = json.loads((ROOT / "configs" / "subject_split.json").read_text(encoding="utf-8"))
DATA = ROOT / "reports" / "dataset"
TABLES = ROOT / "reports" / "tables"
IMG = ROOT / "docs" / "images"
STEPS = CFG["sequence_steps"]
# 기준 모델·문서 숫자는 정면·측면 170개 기준. 뒷모습(B, 11/23 재촬영)은 18번에서 따로 비교한다.
BASE_VIEWS = ("F", "S")

KP = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
      "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
      "left_knee", "right_knee", "left_ankle", "right_ankle"]
ANGLES = ["left_elbow_angle", "right_elbow_angle", "left_knee_angle", "right_knee_angle"]
POS = ["hip_x_img", "hip_y_img"]
FSETS = {
    "course38": [f"{k}_{a}" for k in KP for a in "xy"] + ANGLES,
    "course34": [f"{k}_{a}" for k in KP for a in "xy"],
    "body40": [f"bn_{k}_{a}" for k in KP for a in "xy"] + ANGLES + POS,
    "body38": [f"bn_{k}_{a}" for k in KP for a in "xy"] + ANGLES,
    "body36": [f"bn_{k}_{a}" for k in KP for a in "xy"] + POS,
}
# (이름, Feature 조합, 창, 방식, 기준과 다른 점)
EXPERIMENTS = [
    ("course38_whole", "course38", "whole", "sequence", "수업 그대로 (어깨 폭 38-D, 영상 전체)"),
    ("course38_seg", "course38", "segment", "sequence", "수업 38-D + 동작 구간"),
    ("course34_seg", "course34", "segment", "sequence", "수업 38-D 에서 각도 4개 제외"),
    ("course38_static", "course38", "segment", "static", "수업 38-D, 가운데 Frame 1개"),
    ("body40_seg", "body40", "segment", "sequence", "3팀 기준 (몸 크기 + 각도 + 골반 위치, 동작 구간)"),
    ("body40_whole", "body40", "whole", "sequence", "기준에서 창만 영상 전체"),
    ("body40_static", "body40", "segment", "static", "기준에서 가운데 Frame 1개"),
    ("body38_seg", "body38", "segment", "sequence", "기준에서 골반 화면 위치 2개 제외"),
    ("body36_seg", "body36", "segment", "sequence", "기준에서 각도 4개 제외"),
]
MAIN = "body40_seg"
COURSE = "course38_whole"


def cols(fset: str, mode: str) -> list[str]:
    f = FSETS[fset]
    return [f"mid_{x}" for x in f] if mode == "static" else [f"t{i:02d}_{x}" for i in range(STEPS) for x in f]


SEEDS = [0, 1, 2, 3, 4]  # Random Forest 자체의 우연 변동을 재기 위한 추가 난수 (기준값 42 와 별개)


def model(seed: int | None = None) -> RandomForestClassifier:
    m = CFG["model"]
    return RandomForestClassifier(n_estimators=m["n_estimators"], max_depth=m["max_depth"],
                                  random_state=m["random_state"] if seed is None else seed,
                                  class_weight=m["class_weight"], n_jobs=-1)


def fit_predict(X, y, tr, te, seed: int | None = None):
    rf = model(seed).fit(X[tr], y[tr])
    proba = rf.predict_proba(X[te])
    return rf, rf.classes_[proba.argmax(1)], proba


def load_dataset(window: str, views=BASE_VIEWS) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"sequence_dataset_{window}.csv")
    return df if views is None else df[df.view.isin(views)].reset_index(drop=True)


def setup_font() -> None:
    f = Path("/mnt/c/Windows/Fonts/malgun.ttf")
    if f.exists():
        font_manager.fontManager.addfont(str(f))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(f)).get_name()
    plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    frames = {w: load_dataset(w) for w in ("segment", "whole")}
    base = frames["segment"]
    assert (base["clip_id"] == frames["whole"]["clip_id"]).all()
    y = base["action_id"].to_numpy()
    subj = base["subject_id"].to_numpy()
    sp = SPLIT["main"]
    tr = np.isin(subj, sp["train"])
    va = np.isin(subj, sp["validation"])
    te = np.isin(subj, sp["test"])
    dev = ~te  # S01~S04 — 고르는 데만 사용
    actions = sorted(set(y))
    acts = {r["action_id"]: r for r in csv.DictReader((ROOT / "configs" / "action_classes.csv").open(encoding="utf-8-sig"))}
    approved = {r["clip_id"]: r for r in csv.DictReader((ROOT / "configs" / "approved_clip_manifest.csv").open(encoding="utf-8-sig"))}

    results, preds, keep = [], [], {}
    for name, fset, win, mode, desc in EXPERIMENTS:
        X = frames[win][cols(fset, mode)].to_numpy(float)
        rf, p, proba = fit_predict(X, y, tr, va)
        f_acc, f_f1 = [], []
        for s in sorted(set(subj[dev])):
            _, pf, _ = fit_predict(X, y, dev & (subj != s), subj == s)
            f_acc.append(accuracy_score(y[subj == s], pf))
            f_f1.append(f1_score(y[subj == s], pf, average="macro"))
        seed_acc = [accuracy_score(y[va], fit_predict(X, y, tr, va, s)[1]) for s in SEEDS]
        conf = proba.max(1)
        ok = p == y[va]
        results.append({
            "experiment": name, "feature_set": fset, "window": win, "mode": mode, "description": desc,
            "n_features": X.shape[1],
            "val_accuracy": round(accuracy_score(y[va], p), 4), "val_macro_f1": round(f1_score(y[va], p, average="macro"), 4),
            "val_accuracy_5seed_mean": round(float(np.mean(seed_acc)), 4),
            "val_accuracy_5seed_std": round(float(np.std(seed_acc)), 4),
            "val_mean_conf_correct": round(float(conf[ok].mean()), 3) if ok.any() else "",
            "val_mean_conf_wrong": round(float(conf[~ok].mean()), 3) if (~ok).any() else "",
            "dev4fold_accuracy_mean": round(float(np.mean(f_acc)), 4), "dev4fold_accuracy_std": round(float(np.std(f_acc)), 4),
            "dev4fold_macro_f1_mean": round(float(np.mean(f_f1)), 4),
            "dev4fold_by_subject": " ".join(f"{s}:{a:.3f}" for s, a in zip(sorted(set(subj[dev])), f_acc)),
        })
        keep[name] = (X, rf, p, proba)
        print(f"{name:16s} val acc {results[-1]['val_accuracy']:.3f} f1 {results[-1]['val_macro_f1']:.3f} | "
              f"4-fold {np.mean(f_acc):.3f}±{np.std(f_acc):.3f} | 5 seeds {np.mean(seed_acc):.3f}±{np.std(seed_acc):.3f}")
    pd.DataFrame(results).to_csv(TABLES / "baseline_experiments.csv", index=False, encoding="utf-8-sig")

    # ── 마지막 Test (S05) — 기준 모델과 수업 그대로 모델만, 각각 두 가지 학습 범위로 ──
    test_rows, per_action = [], []
    cms = {}
    for name in (MAIN, COURSE):
        X, rf, p_val, proba_val = keep[name]
        _, p_te, proba_te = fit_predict(X, y, tr, te)
        _, p_te4, _ = fit_predict(X, y, dev, te)
        for label, trn, pp in (("train S01~S03", tr, p_te), ("train S01~S04", dev, p_te4)):
            test_rows.append({"experiment": name, "trained_on": label,
                              "test_accuracy": round(accuracy_score(y[te], pp), 4),
                              "test_macro_f1": round(f1_score(y[te], pp, average="macro"), 4)})
        for split_name, mask, pp, pr in (("validation_S04", va, p_val, proba_val), ("test_S05", te, p_te, proba_te)):
            order = np.argsort(-pr, 1)
            for i, idx in enumerate(np.where(mask)[0]):
                cid = base["clip_id"][idx]
                preds.append({"experiment": name, "split": split_name, "clip_id": cid, "subject_id": subj[idx],
                              "view": base["view"][idx], "true": y[idx], "pred": pp[i], "correct": int(pp[i] == y[idx]),
                              "confidence": round(float(pr[i].max()), 3),
                              "second": rf.classes_[order[i, 1]], "second_conf": round(float(pr[i, order[i, 1]]), 3),
                              "true_conf": round(float(pr[i, list(rf.classes_).index(y[idx])]), 3)})
            if name == MAIN:
                pr_, rc_, f1_, _ = precision_recall_fscore_support(y[mask], pp, labels=actions, zero_division=0)
                for a, a1, a2, a3 in zip(actions, pr_, rc_, f1_):
                    per_action.append({"split": split_name, "action_id": a, "name_ko": acts[a]["name_ko"],
                                       "precision": round(a1, 3), "recall": round(a2, 3), "f1": round(a3, 3)})
                cms[split_name] = confusion_matrix(y[mask], pp, labels=actions)
        print(f"TEST {name}: " + " / ".join(f"{r['trained_on']} acc {r['test_accuracy']:.3f}" for r in test_rows if r["experiment"] == name))
    pd.DataFrame(test_rows).to_csv(TABLES / "baseline_test.csv", index=False, encoding="utf-8-sig")
    P = pd.DataFrame(preds)
    P.to_csv(TABLES / "baseline_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(per_action).to_csv(TABLES / "baseline_per_action.csv", index=False, encoding="utf-8-sig")

    # ── 틀린 영상 추적 (기준 모델) ──
    W = P[(P.experiment == MAIN) & (P.correct == 0)].copy()
    W["true_ko"] = W["true"].map(lambda a: acts[a]["name_ko"])
    W["pred_ko"] = W["pred"].map(lambda a: acts[a]["name_ko"])
    W["filmed_as"] = W["true"].map(lambda a: acts[a].get("filmed_as", ""))
    W["segment_sec"] = W["clip_id"].map(lambda c: round(float(approved[c]["seg_end"]) - float(approved[c]["seg_start"]), 2))
    W["human_note"] = W["clip_id"].map(lambda c: approved[c]["human_note"])
    other_view = {(r.clip_id[:-2], r.view): r.correct for r in P[P.experiment == MAIN].itertuples()}
    W["other_view_correct"] = [other_view.get((c[:-2], "S" if v == "F" else "F"), "") for c, v in zip(W.clip_id, W.view)]
    W.to_csv(TABLES / "baseline_wrong_clips.csv", index=False, encoding="utf-8-sig")
    print(f"틀린 영상 (기준 모델): Validation {int((W.split == 'validation_S04').sum())} · Test {int((W.split == 'test_S05').sum())}")

    # ── Feature 중요도: 관절별·시간대별 (기준 모델, Train 으로 학습한 것) ──
    X, rf, _, _ = keep[MAIN]
    names = cols("body40", "sequence")
    imp = pd.DataFrame({"feature": names, "importance": rf.feature_importances_})
    imp["step"] = imp.feature.str[1:3].astype(int)
    base_name = imp.feature.str[4:]
    imp["group"] = np.select(
        [base_name.isin(ANGLES), base_name.isin(POS)],
        ["각도:" + base_name.str.replace("_angle", ""), "골반 화면 위치"],
        default=base_name.str.replace("bn_", "").str.replace(r"_(x|y)$", "", regex=True))
    by_group = imp.groupby("group").importance.sum().sort_values(ascending=False)
    by_step = imp.groupby("step").importance.sum()
    pd.concat([by_group.rename("importance").reset_index().assign(kind="group"),
               by_step.rename("importance").reset_index().rename(columns={"step": "group"}).assign(kind="step")]
              ).to_csv(TABLES / "baseline_importance.csv", index=False, encoding="utf-8-sig")

    # ── 그림 ──
    setup_font()
    short = [f"{a} {acts[a]['name_ko']}" for a in actions]
    res = {r["experiment"]: r for r in results}
    tst = {(r["experiment"], r["trained_on"]): r for r in test_rows}
    fig, axes = plt.subplots(1, 2, figsize=(19, 8.6))
    for ax, (sname, title) in zip(axes, (("validation_S04", "Validation S04 (같은 날 촬영)"), ("test_S05", "Test S05 (다른 날 촬영)"))):
        cm = cms[sname]
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=2)
        for i in range(len(actions)):
            for j in range(len(actions)):
                if cm[i, j]:
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9,
                            color="white" if i == j else "#b3261e")
        ax.set_xticks(range(len(actions)), actions, fontsize=8, rotation=90)
        ax.set_yticks(range(len(actions)), short if ax is axes[0] else actions, fontsize=8)
        ax.set_xlabel("예측")
        ax.set_ylabel("정답")
        if sname == "validation_S04":
            acc, f1 = res[MAIN]["val_accuracy"], res[MAIN]["val_macro_f1"]
        else:
            acc, f1 = tst[(MAIN, "train S01~S03")]["test_accuracy"], tst[(MAIN, "train S01~S03")]["test_macro_f1"]
        ax.set_title(f"{title}\n정확도 {acc:.1%} · macro F1 {f1:.3f} (영상 34개 = 동작 17 × 정면·측면)", fontsize=11)
    fig.suptitle("3팀 기준 모델 (몸 크기 정규화 + 각도 + 골반 위치, 동작 구간 30단계, Random Forest) — Train S01~S03 로 학습\n"
                 "대각선(흰 글자)이 맞힌 수, 빨간 글자가 틀린 칸", fontsize=12)
    fig.tight_layout()
    fig.savefig(IMG / "baseline_confusion.png", dpi=100, facecolor="white", bbox_inches="tight")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [1.2, 1]})
    g = by_group.iloc[::-1]
    axes[0].barh(g.index, g.values, color=["#c23b7a" if ("각도" in k or "위치" in k) else "#2f6fd6" for k in g.index])
    axes[0].set_title("어느 값을 보고 판단했나 (관절·값 종류별 중요도 합)", fontsize=11)
    axes[0].tick_params(labelsize=8)
    axes[1].bar(by_step.index, by_step.values, color="#3a9e5c")
    axes[1].set_title("동작 구간의 어느 시점을 봤나 (30단계별 중요도 합)", fontsize=11)
    axes[1].set_xlabel("구간 시작 0 → 끝 29")
    fig.suptitle("3팀 기준 모델 Feature 중요도 (Random Forest, Train S01~S03)", fontsize=12)
    fig.tight_layout()
    fig.savefig(IMG / "baseline_importance.png", dpi=100, facecolor="white", bbox_inches="tight")
    print("figures saved")


if __name__ == "__main__":
    main()
