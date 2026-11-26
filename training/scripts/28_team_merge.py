"""팀원 모델 장점 섞기 — 카메라 각도 증강 · 움직임 요약 86개 · 확률 보정을 하나씩 넣어 본다 (2026-11-26).

왜: 11/26 팀 모델 비교(같은 현장 영상 15개)에서 지금 모델과 한 팀원 모델이 14/15 로 공동 1위였다.
    그 팀원 모델은 다른 카메라 구도(묶음 A) 9/10 으로 강했고(카메라 각도 증강 + 움직임 요약 86개 + 손 규칙),
    다른 팀원 모델은 '이 예측이 맞을 확률' 로 보정한 확신도를 보여 줬다. 지금 모델은 맞혀도 확률이 0.3~0.6 이라
    '불확실' 로 표시되는 일이 많다.
    → 장점을 하나씩 더해 효과를 따로 잰다 (ablation). 도움 되는 것만 남긴다. 손 규칙은 가져오지 않는다.

후보 (모두 Pose 11m · whole · 현장식 증강 2 · 좌우 반전 · ExtraTrees 1000 + 1D-CNN(폭 128) 확률 평균 = 지금 서비스)
  base         : 지금 서비스와 같음
  +cam         : 영상마다 카메라 각도 변형 1개 추가 (src/camera_augment.py, yaw 0~60° · pitch 0~45° 무작위)
  +kin86       : ExtraTrees 입력에 움직임 요약 86개 추가 (src/kinematic_summary.py) — 1D-CNN 입력은 그대로
  +cam+kin86   : 둘 다
  +cam2        : 카메라 각도 변형 2개 (+cam 이 도움 될 때 양을 늘려 봄)
  +cam2+kin86  : 둘 다 (카메라 변형 2개)
평가 (S01~S04 사람 단위 4-fold, 난수 42·0·1): 평가 사람 영상을 바꿔서 본다
  orig·cut·late·double·pad : 27번과 같음 (원래 / 바로 시작 / 중간부터 / 두 번 / 오래 서 있기)
  cam      : 원래 영상을 yaw 45° · pitch 30° 로 줄임 (다른 카메라 흉내 — 학습 증강과 같은 방식이라 +cam 에 유리, 참고만)
고르는 규칙 (미리 정함): orig 가 base 보다 1%p 넘게 떨어지면 탈락. 남은 것 중 현장4(cut·late·double·pad 평균)와
  orig 의 평균이 가장 높은 것. base 와의 차이가 난수 3개 표준편차보다 커야 '효과 있음'. cam 열·Test S05·현장 영상은 확인만.
확률 보정 (--calib): 고른 설정의 4-fold 예측(처음 보는 사람)으로 로지스틱 회귀(p1, p2, margin, log_ratio, entropy)
  → '맞을 확률'. 사람 단위로 다시 나눠 (S01~S04 중 3명 학습 → 1명 확인) ECE·Brier·AUC 를 원래 1위 확률과 비교.

실행: .venv/bin/python scripts/28_team_merge.py [--only base,+cam] [--test] [--calib 설정] [--calib-out 폴더]
결과: reports/tables/team_merge_cv.csv (요약), team_merge_oof.csv.gz (예측 확률), team_merge_test.csv (--test),
      team_merge_calibration.txt (--calib)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.camera_augment import camera_view  # noqa: E402
from src.common import load_script  # noqa: E402
from src.schema import BODY40  # noqa: E402
from src.sequence_extras import add_extras  # noqa: E402
from src.temporal_cnn import CNN, fit_cnn, predict_cnn, to_seq  # noqa: E402

fr = load_script("27_field_robust.py")
SPLIT = json.loads((ROOT / "configs" / "subject_split.json").read_text(encoding="utf-8"))["main"]
TABLES = ROOT / "reports" / "tables"
STEPS, SEEDS, NF = 30, (42, 0, 1), len(BODY40)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
VARIANTS = ("orig", "cut", "late", "double", "pad", "cam")
TEST_CAM = dict(yaw=45.0, pitch=30.0)
AUG_KINDS = ["cut", "late", "double", "pad"]

CONFIGS = {
    "base": dict(cam=0, extras=["velocity", "stats"]),
    "+cam": dict(cam=1, extras=["velocity", "stats"]),
    "+kin86": dict(cam=0, extras=["velocity", "stats", "kin86"]),
    "+cam+kin86": dict(cam=1, extras=["velocity", "stats", "kin86"]),
    "+cam2": dict(cam=2, extras=["velocity", "stats"]),
    "+cam2+kin86": dict(cam=2, extras=["velocity", "stats", "kin86"]),
}


def build_train(clips, cfg, seed):
    """영상마다 orig + 현장식 2개 (+ 카메라 각도 cam 개) → 좌우 반전 사본까지"""
    rng = np.random.default_rng(seed)
    seqs, ys = [], []
    for c in clips:
        kinds = ["orig", *rng.choice(AUG_KINDS, 2, replace=False)]
        for k in kinds:
            seqs.append(fr.window(*fr.variant(c, k, rng if k != "orig" else None), "whole"))
            ys.append(c["uid"])
        for _ in range(cfg["cam"]):
            k = str(rng.choice(kinds))
            t, v = fr.variant(c, k, rng if k != "orig" else None)
            seqs.append(fr.window(*camera_view(t, v, rng), "whole"))
            ys.append(c["uid"])
    X = np.vstack([fr.to_matrix(seqs, False), fr.to_matrix(seqs, False, flip=True)])
    return X, np.array(ys * 2)


def test_mats(clips):
    m = {k: fr.to_matrix([fr.window(*fr.variant(c, k), "whole") for c in clips], False) for k in VARIANTS[:-1]}
    m["cam"] = fr.to_matrix([fr.window(*camera_view(c["t"], c["v"], **TEST_CAM), "whole") for c in clips], False)
    return m


def fit_predict(Xtr, ytr, Xte_list, seed, n_cls, extras, ens=1):
    et = fr.more.et(seed).fit(add_extras(Xtr, STEPS, NF, extras), ytr)
    Str = to_seq(Xtr, STEPS, NF)
    cnns = [fit_cnn(Str, ytr, seed + 1000 * k, n_cls, epochs=120, device=DEV,
                    make_model=lambda c, n: CNN(c, n, w=128)) for k in range(ens)]
    out = []
    for Xte in Xte_list:
        p_et = et.predict_proba(add_extras(Xte, STEPS, NF, extras))
        p_nn = np.mean([predict_cnn(mm, mu, sd, to_seq(Xte, STEPS, NF)) for mm, mu, sd in cnns], 0)
        out.append((p_et + p_nn) / 2)
    return out


def rows_for(name, seed, set_name, subject, te, probs, classes):
    rows = []
    y = np.array([c["uid"] for c in te])
    for k, p in zip(VARIANTS, probs, strict=True):
        for c, pi, yi in zip(te, p, y, strict=True):
            rows.append({"config": name, "seed": seed, "set": set_name, "variant": k, "subject_id": subject,
                         "clip_id": c["clip_id"], "view": c["view"], "action_id": classes[yi],
                         "pred": classes[int(pi.argmax())], "correct": int(pi.argmax() == yi),
                         **{f"p_{cl}": round(float(v), 5) for cl, v in zip(classes, pi, strict=True)}})
    return rows


def summarize(df: pd.DataFrame) -> dict:
    acc = df.groupby(["variant", "seed"]).correct.mean().unstack(0) * 100
    o = df[df.variant == "orig"]
    f1 = o.groupby("seed").apply(lambda g: f1_score(g.action_id, g.pred, average="macro"), include_groups=False) * 100
    f4 = acc[AUG_KINDS].mean(1)
    return {**{k: round(acc[k].mean(), 1) for k in VARIANTS},
            "orig_std": round(acc["orig"].std(ddof=0), 1), "macro_f1": round(f1.mean(), 1),
            "field4": round(f4.mean(), 1), "field4_std": round(f4.std(ddof=0), 1),
            "score": round(((acc["orig"] + f4) / 2).mean(), 1),
            "score_std": round(((acc["orig"] + f4) / 2).std(ddof=0), 1),
            "B_orig": round(o[o.view == "B"].correct.mean() * 100, 1)}


def run_cv(names, clips, classes):
    dev = [c for c in clips if c["subject_id"] in SPLIT["train"] + SPLIT["validation"]]
    subjects = sorted({c["subject_id"] for c in dev})
    out_sum, out_oof = TABLES / "team_merge_cv.csv", TABLES / "team_merge_oof.csv.gz"
    summary = pd.read_csv(out_sum).to_dict("records") if out_sum.exists() else []
    oof = [pd.read_csv(out_oof)] if out_oof.exists() else []
    for name in names:
        cfg, t0, rows = CONFIGS[name], time.time(), []
        for seed in SEEDS:
            for s in subjects:
                tr = [c for c in dev if c["subject_id"] != s]
                te = [c for c in dev if c["subject_id"] == s]
                Xtr, ytr = build_train(tr, cfg, seed)
                mats = test_mats(te)
                probs = fit_predict(Xtr, ytr, [mats[k] for k in VARIANTS], seed, len(classes), cfg["extras"])
                rows += rows_for(name, seed, "cv4", s, te, probs, classes)
        df = pd.DataFrame(rows)
        r = {"config": name, **summarize(df), "train_rows_per_fold": len(ytr), "sec": round(time.time() - t0)}
        print(f"  {name:12s} orig {r['orig']:5.1f}±{r['orig_std']} F1 {r['macro_f1']:5.1f} | cut {r['cut']:5.1f} "
              f"late {r['late']:5.1f} double {r['double']:5.1f} pad {r['pad']:5.1f} | 현장4 {r['field4']:5.1f}±"
              f"{r['field4_std']} | 점수 {r['score']:5.1f}±{r['score_std']} | cam {r['cam']:5.1f} | {r['sec']}s", flush=True)
        summary = [x for x in summary if x["config"] != name] + [r]
        oof = [o[o.config != name] for o in oof] + [df]
        pd.DataFrame(summary).to_csv(out_sum, index=False, encoding="utf-8-sig")
        pd.concat(oof).to_csv(out_oof, index=False)


def run_test(names, clips, classes):
    """S01~S04 로 학습 → Test S05 (고른 뒤 확인용, 서비스처럼 CNN 난수 3개 묶음)"""
    tr = [c for c in clips if c["subject_id"] in SPLIT["train"] + SPLIT["validation"]]
    te = [c for c in clips if c["subject_id"] in SPLIT["test"]]
    mats, res = test_mats(te), []
    for name in names:
        rows = []
        for seed in SEEDS:
            Xtr, ytr = build_train(tr, CONFIGS[name], seed)
            probs = fit_predict(Xtr, ytr, [mats[k] for k in VARIANTS], seed, len(classes), CONFIGS[name]["extras"], ens=3)
            rows += rows_for(name, seed, "test_S05", "S05", te, probs, classes)
        r = {"config": name, **summarize(pd.DataFrame(rows))}
        res.append(r)
        print(f"  {name:12s} Test S05 orig {r['orig']:5.1f}±{r['orig_std']} F1 {r['macro_f1']:5.1f} | 현장4 {r['field4']:5.1f} "
              f"| cam {r['cam']:5.1f}", flush=True)
        pd.DataFrame(res).to_csv(TABLES / "team_merge_test.csv", index=False, encoding="utf-8-sig")


# ── 확률 보정 (다른 팀원 모델 52번과 같은 입력) ──
CAL_FEATURES = ["p1", "p2", "margin", "log_ratio", "entropy"]


def cal_features(proba: np.ndarray) -> np.ndarray:
    srt = -np.sort(-proba, axis=1)
    p1, p2, eps = srt[:, 0], srt[:, 1], 1e-6
    entropy = -(proba * np.log(proba + eps)).sum(1) / np.log(proba.shape[1])
    return np.c_[p1, p2, p1 - p2, np.log((p1 + eps) / (p2 + eps)), entropy]


def ece(prob, correct, bins=10):
    edges, total = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (prob >= lo) & ((prob < hi) if hi < 1 else (prob <= hi))
        if m.any():
            total += m.mean() * abs(prob[m].mean() - correct[m].mean())
    return total


def run_calib(name, out_dir: str):
    df = pd.read_csv(TABLES / "team_merge_oof.csv.gz")
    df = df[(df.config == name) & (df.variant != "cam")].reset_index(drop=True)     # cam 은 흉내라 빼고 실제 촬영 변형만
    pcols = [c for c in df.columns if c.startswith("p_")]
    x, correct, subj = cal_features(df[pcols].to_numpy()), df.correct.to_numpy(), df.subject_id.to_numpy()
    cal = np.zeros(len(df))
    for s in np.unique(subj):
        tr, te = subj != s, subj == s
        cal[te] = LogisticRegression(C=1.0, max_iter=1000).fit(x[tr], correct[tr]).predict_proba(x[te])[:, 1]
    raw = x[:, 0]
    lines = [f"설정 {name} · 4-fold 예측 {len(df)}건 (난수 3 × 변형 5: orig·cut·late·double·pad), 정답률 {correct.mean():.3f}",
             "보정기 평가: S01~S04 중 3명으로 학습 → 남은 1명 (4회)", "",
             "방법            평균표시값  실제정답률    ECE   Brier    AUC"]
    for label, p in (("모델 1위 확률", raw), ("보정 후", cal)):
        lines.append(f"{label:14s} {p.mean():10.3f} {correct.mean():10.3f} {ece(p, correct):7.3f} "
                     f"{brier_score_loss(correct, p):7.3f} {roc_auc_score(correct, p):7.3f}")
    lines += ["", "변형별 (보정 후 평균 / 실제 정답률 / 원래 1위 확률 평균)"]
    for k in VARIANTS[:-1]:
        m = df.variant.to_numpy() == k
        lines.append(f"  {k:6s} {cal[m].mean():.3f} / {correct[m].mean():.3f} / {raw[m].mean():.3f}")
    lines += ["", "'불확실' 기준 (값 < 기준이면 불확실): 불확실 비율 · 확실로 보인 것의 정답률 · 틀린 것 중 걸러진 비율"]
    for label, p, ths in (("원래 1위 확률", raw, (0.25, 0.3, 0.4, 0.5)), ("보정 후", cal, (0.5, 0.6, 0.7, 0.8, 0.9))):
        for th in ths:
            unc = p < th
            lines.append(f"  {label:10s} < {th:.2f}: {unc.mean():.2f} · {correct[~unc].mean():.3f} · "
                         f"{(unc & (correct == 0)).sum() / max(1, (correct == 0).sum()):.2f}")
    final = LogisticRegression(C=1.0, max_iter=1000).fit(x, correct)
    text = "\n".join(lines)
    print(text)
    (TABLES / "team_merge_calibration.txt").write_text(text + "\n", encoding="utf-8")
    if out_dir:
        out = ROOT / out_dir
        out.mkdir(parents=True, exist_ok=True)
        (out / "calibration.json").write_text(json.dumps({
            "method": "logistic_regression", "features": CAL_FEATURES,
            "coef": [round(float(c), 6) for c in final.coef_[0]], "intercept": round(float(final.intercept_[0]), 6),
            "n_samples": int(len(df)), "base_accuracy": round(float(correct.mean()), 4),
            "ece_raw": round(ece(raw, correct), 4), "ece_calibrated": round(ece(cal, correct), 4),
            "source": f"scripts/28_team_merge.py --calib {name} (S01~S04 4-fold 예측, 난수 3 × 변형 5)",
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print("저장:", (out / "calibration.json").relative_to(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="설정 이름 (쉼표)")
    ap.add_argument("--test", action="store_true", help="Test S05 확인 (S01~S04 로 학습)")
    ap.add_argument("--calib", default="", help="확률 보정기를 맞출 설정 이름")
    ap.add_argument("--calib-out", default="", help="calibration.json 을 저장할 폴더 (training 기준)")
    args = ap.parse_args()
    if args.calib:
        return run_calib(args.calib, args.calib_out)
    names = [n for n in CONFIGS if not args.only or n in args.only.split(",")]
    clips = fr.load_clips("11m")
    classes = sorted({str(c["action_id"]) for c in clips})
    for c in clips:
        c["uid"] = classes.index(str(c["action_id"]))
    print(f"장치 {DEV} · 영상 {len(clips)} · 설정 {names}", flush=True)
    (run_test if args.test else run_cv)(names, clips, classes)


if __name__ == "__main__":
    main()
