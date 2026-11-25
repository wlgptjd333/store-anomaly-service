"""현장 영상처럼 잘린·반복된 영상에도 맞히게 — 움직임 구간·현장식 증강·관계 Feature·Pose 11m (2026-11-25).

왜: 현장 밀치기 영상(옆에서, 걸어 들어와 팔을 뻗은 채 두 번 밀기, 8초)을 서비스가 '두드리기 19%'로 답함.
    학습 영상은 '서 있기 → 동작 한 번 → 서 있기'로 찍었는데, 현장은 바로 시작하거나 반복하거나 오래 서 있다.
    지금 4-fold 는 학습과 같은 방식으로 찍은 영상만 평가해서 이 차이를 못 잡는다.

평가 (S01~S04 사람 단위 4-fold, 난수 3개): 평가 사람의 영상을 다섯 가지로 바꿔서 본다 — 정답 라벨이 있는 '현장 흉내'
  orig   : 원래 영상 그대로
  cut    : 동작 구간(06번, 사람이 검수)만 — 바로 시작해서 바로 끊음
  late   : 동작 구간의 25% 지점부터 — 동작 중간에 녹화 시작
  double : 동작 구간을 두 번 이어 붙임 — 같은 동작 반복
  pad    : 앞뒤로 2초씩 가만히 서 있기(첫·끝 Frame 유지) — 긴 녹화 속 짧은 동작
후보:
  window : whole(지금, 영상 전체) / motion(src/motion_window — 움직인 구간 ± 0.5초만)
  aug    : 학습 영상에도 같은 종류의 변형을 무작위 길이로 더함 (평가와 같은 값이 아니라 무작위 → 외워서 맞히기 방지)
  rel    : 관계 Feature 10개 — 손 높이(어깨 기준)·손 뻗은 거리·무릎 높이(골반 기준)·두 손 간격·보폭·상체 기울기
  data   : Pose 11n(지금) / 11m (03번 --pose-model yolo11m-pose.pt --max-fps 30 결과)
고르는 규칙 (미리 정함): orig 가 지금보다 1%p 넘게 떨어지면 안 됨. 그 안에서 변형 넷(cut·late·double·pad) 평균이 가장 높은 것.
  이 차이가 난수 3개 편차보다 커야 '효과 있음'. Test S05 와 현장 영상은 고른 뒤 확인만.

실행: .venv/bin/python scripts/27_field_robust.py [--only 이름,…] [--final 이름]
결과: reports/tables/field_robust_cv.csv, field_robust_per_action.csv, (--final) field_robust_final.csv
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import load_script  # noqa: E402
from src.motion_window import active_window  # noqa: E402
from src.schema import BODY40, KP, flip_matrix  # noqa: E402
from src.sequence_extras import add_extras  # noqa: E402
from src.temporal_cnn import CNN, fit_cnn, predict_cnn, to_seq  # noqa: E402

s13 = load_script("13_build_sequence_dataset.py")
more = load_script("20_more_improvements.py")
SPLIT = json.loads((ROOT / "configs" / "subject_split.json").read_text(encoding="utf-8"))["main"]
TABLES = ROOT / "reports" / "tables"
APPROVED = ROOT / "configs" / "approved_clip_manifest.csv"
SEQ_DIRS = {"11n": ROOT / "reports" / "sequences", "11m": ROOT / "reports" / "pose_compare" / "yolo11m" / "sequences"}
EXTRA = ROOT / "reports" / "extra"
FIELD = {"11n": EXTRA / "field_1125" / "sequences", "11m": EXTRA / "field_1125_11m" / "sequences"}      # 현장 영상 (Git 제외)
NEWCAM = {"11n": EXTRA / "s05_newcam" / "sequences", "11m": EXTRA / "s05_newcam_11m" / "sequences"}    # 19번 새 구도 5개
STEPS, SEEDS, EXTRAS = 30, (42, 0, 1), ["velocity", "stats"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
VARIANTS = ("orig", "cut", "late", "double", "pad")

IX = {f: i for i, f in enumerate(BODY40)}
P = {k: (IX[f"bn_{k}_x"], IX[f"bn_{k}_y"]) for k in KP}
REL_NAMES = ["hand_up_l", "hand_up_r", "reach_l", "reach_r", "knee_up_l", "knee_up_r", "hands_gap", "stance",
             "lean_x", "chest_y"]

CONFIGS = {
    "지금 (whole)": dict(window="whole"),
    "whole + 현장식 증강": dict(window="whole", aug=2),
    "motion": dict(window="motion"),
    "motion + 현장식 증강": dict(window="motion", aug=2),
    "whole + 관계 Feature": dict(window="whole", rel=True),
    "motion + 증강 + 관계": dict(window="motion", aug=2, rel=True),
    "Pose 11m · whole": dict(window="whole", data="11m"),
    "Pose 11m · motion + 증강": dict(window="motion", aug=2, data="11m"),
    # 2차: 1차에서 현장4 가 크게 오른 증강을 원래 영상이 덜 떨어지게 (변형 1개) / 11m 과 조합
    "whole + 증강 1": dict(window="whole", aug=1),
    "motion + 증강 1": dict(window="motion", aug=1),
    "Pose 11m · whole + 증강": dict(window="whole", aug=2, data="11m"),
    "Pose 11m · motion + 증강 1": dict(window="motion", aug=1, data="11m"),
}


# ── 데이터: Frame 별 → (시간, BODY40) ──
def read_frames(path: Path):
    df = pd.read_csv(path)
    t = pd.to_numeric(df["time_sec"], errors="coerce").to_numpy(float)
    v = df[BODY40].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    ok = s13.ok(df["pose_ok"]) & s13.ok(df["feature_ok"]) & s13.ok(df["body_feature_ok"]) & np.isfinite(v).all(1)
    return t[ok], v[ok]


def load_clips(data: str) -> list[dict]:
    clips = pd.read_csv(APPROVED, encoding="utf-8-sig").to_dict("records")
    out = []
    for c in clips:
        p = SEQ_DIRS[data] / Path(c["sequence_file"]).name
        t, v = read_frames(p)
        if len(t) >= 10:
            out.append({**c, "t": t, "v": v})
    return out


# ── 변형 (평가: 고정 값, 학습 증강: 무작위) ──
def variant(c: dict, kind: str, rng=None) -> tuple[np.ndarray, np.ndarray]:
    t, v = c["t"], c["v"]
    a, z = float(c["seg_start"]), float(c["seg_end"])
    r = (lambda lo, hi: rng.uniform(lo, hi)) if rng is not None else (lambda lo, hi: (lo + hi) / 2)
    if kind == "orig":
        return t, v
    if kind in ("cut", "late", "double"):
        if rng is not None:                               # 학습: 구간 경계를 조금씩 흔듦
            a, z = a + r(-0.3, 0.3), z + r(-0.3, 0.3)
        if kind == "late":
            a = a + (r(0.1, 0.4) if rng is not None else 0.25) * (z - a)
        m = (t >= a) & (t <= z)
        if m.sum() < 10:
            return t, v
        ts, vs = t[m] - t[m][0], v[m]
        if kind == "double":
            gap = r(0.2, 0.8)
            ts, vs = np.r_[ts, ts + ts[-1] + gap], np.vstack([vs, vs])
        return ts, vs
    if kind == "pad":
        hold = r(1.0, 3.0)
        n = max(2, int(hold * 30))
        pre_t = np.linspace(-hold, -1 / 30, n)
        post_t = np.linspace(t[-1] + 1 / 30, t[-1] + hold, n)
        t = t - t[0]
        post_t = np.linspace(t[-1] + 1 / 30, t[-1] + hold, n)
        return (np.r_[pre_t, t, post_t] + hold,
                np.vstack([np.repeat(v[:1], n, 0), v, np.repeat(v[-1:], n, 0)]))
    raise ValueError(kind)


def window(t, v, mode):
    if mode == "motion":
        a, z = active_window(t, v)
        m = (t >= a) & (t <= z)
        if m.sum() >= 10:
            t, v = t[m], v[m]
    return s13.resample(v, t, STEPS)                      # (30, 40)


def relations(seq: np.ndarray) -> np.ndarray:
    """(n, steps, 40) → (n, steps, 10). 정규화 좌표는 y 가 아래로 커짐."""
    def g(k, a):
        return seq[:, :, P[k][a]]
    sh_y = (g("left_shoulder", 1) + g("right_shoulder", 1)) / 2
    sh_x = (g("left_shoulder", 0) + g("right_shoulder", 0)) / 2
    hip_x = (g("left_hip", 0) + g("right_hip", 0)) / 2
    hip_y = (g("left_hip", 1) + g("right_hip", 1)) / 2
    cols = [g("left_shoulder", 1) - g("left_wrist", 1), g("right_shoulder", 1) - g("right_wrist", 1),
            np.abs(g("left_wrist", 0) - g("left_shoulder", 0)), np.abs(g("right_wrist", 0) - g("right_shoulder", 0)),
            hip_y - g("left_knee", 1), hip_y - g("right_knee", 1),
            np.hypot(g("left_wrist", 0) - g("right_wrist", 0), g("left_wrist", 1) - g("right_wrist", 1)),
            np.abs(g("left_ankle", 0) - g("right_ankle", 0)),
            np.abs(sh_x - hip_x), sh_y - hip_y]
    return np.stack(cols, 2)


def to_matrix(seqs: list[np.ndarray], rel: bool, flip: bool = False) -> np.ndarray:
    X = np.stack(seqs).reshape(len(seqs), -1)
    if flip:
        X = flip_matrix(X, STEPS)
    S = X.reshape(len(seqs), STEPS, len(BODY40))
    if rel:
        S = np.concatenate([S, relations(S)], 2)
    return S.reshape(len(seqs), -1)


def build_train(clips, cfg, seed):
    rng = np.random.default_rng(seed)
    seqs, ys = [], []
    for c in clips:
        kinds = ["orig"] + (list(rng.choice(["cut", "late", "double", "pad"], cfg["aug"], replace=False)) if cfg.get("aug") else [])
        for k in kinds:
            t, v = variant(c, k, rng if k != "orig" else None)
            seqs.append(window(t, v, cfg["window"]))
            ys.append(c["uid"])
    X = np.vstack([to_matrix(seqs, cfg.get("rel", False)), to_matrix(seqs, cfg.get("rel", False), flip=True)])
    return X, np.array(ys * 2)


def fit_predict(Xtr, ytr, Xte_list, seed, n_frame, n_cls, ens=1):
    m = more.et(seed).fit(add_extras(Xtr, STEPS, n_frame, EXTRAS), ytr)
    Str = to_seq(Xtr, STEPS, n_frame)
    cnns = [fit_cnn(Str, ytr, seed + 1000 * k, n_cls, epochs=120, device=DEV,
                    make_model=lambda c, n: CNN(c, n, w=128)) for k in range(ens)]
    out = []
    for Xte in Xte_list:
        p_et = m.predict_proba(add_extras(Xte, STEPS, n_frame, EXTRAS))
        p_nn = np.mean([predict_cnn(mm, mu, sd, to_seq(Xte, STEPS, n_frame)) for mm, mu, sd in cnns], 0)
        out.append((p_et + p_nn) / 2)
    return out


def test_mats(clips, cfg):
    return {k: to_matrix([window(*variant(c, k), cfg["window"]) for c in clips], cfg.get("rel", False)) for k in VARIANTS}


def run_cv(name, cfg, classes):
    clips = [c for c in CLIPS[cfg.get("data", "11n")] if c["subject_id"] in DEV_SUBJ]
    n_frame = len(BODY40) + (len(REL_NAMES) if cfg.get("rel") else 0)
    rows = []
    for seed in SEEDS:
        for s in DEV_SUBJ:
            tr = [c for c in clips if c["subject_id"] != s]
            te = [c for c in clips if c["subject_id"] == s]
            Xtr, ytr = build_train(tr, cfg, seed)
            mats = test_mats(te, cfg)
            probs = fit_predict(Xtr, ytr, [mats[k] for k in VARIANTS], seed, n_frame, len(classes))
            for k, p in zip(VARIANTS, probs, strict=True):
                y = np.array([c["uid"] for c in te])
                for c, pi, yi in zip(te, p, y, strict=True):
                    rows.append({"config": name, "seed": seed, "variant": k, "subject_id": s, "view": c["view"],
                                 "action_id": classes[yi], "correct": int(pi.argmax() == yi), "p_true": pi[yi]})
    return pd.DataFrame(rows)


def main() -> None:
    global CLIPS, DEV_SUBJ
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--final", default="", help="고른 설정 (쉼표) → Test S05 + 현장 영상 확인")
    args = ap.parse_args()
    DEV_SUBJ = SPLIT["train"] + SPLIT["validation"]
    names = [n for n in CONFIGS if not args.only or any(k in n for k in args.only.split(","))]
    if args.final:
        names = ["지금 (whole)", *args.final.split(",")]
    need = {CONFIGS[n].get("data", "11n") for n in names}
    CLIPS = {d: load_clips(d) for d in need}
    classes = sorted({str(c["action_id"]) for c in next(iter(CLIPS.values()))})
    for d in CLIPS.values():
        for c in d:
            c["uid"] = classes.index(str(c["action_id"]))
    print(f"장치 {DEV} · 영상 {[len(v) for v in CLIPS.values()]} · 설정 {len(names)}개", flush=True)

    if args.final:
        return final(names, classes)

    allrows = []
    out = TABLES / "field_robust_cv.csv"
    summary = []
    for name in names:
        t0 = time.time()
        df = run_cv(name, CONFIGS[name], classes)
        allrows.append(df)
        acc = df.groupby(["variant", "seed"]).correct.mean().unstack(0) * 100
        r = {"config": name, **{k: round(acc[k].mean(), 1) for k in VARIANTS},
             "field4": round(acc[["cut", "late", "double", "pad"]].mean(1).mean(), 1),
             "field4_std": round(acc[["cut", "late", "double", "pad"]].mean(1).std(ddof=0), 1),
             "orig_std": round(acc["orig"].std(ddof=0), 1),
             "B_orig": round(df[(df.view == "B") & (df.variant == "orig")].correct.mean() * 100, 1),
             "subj_min_orig": round(df[df.variant == "orig"].groupby("subject_id").correct.mean().min() * 100, 1),
             "sec": round(time.time() - t0)}
        summary.append(r)
        print(f"  {name:28s} orig {r['orig']:5.1f}±{r['orig_std']} | cut {r['cut']:5.1f} late {r['late']:5.1f} "
              f"double {r['double']:5.1f} pad {r['pad']:5.1f} | 현장4 {r['field4']:5.1f}±{r['field4_std']} | {r['sec']}s", flush=True)
        pd.DataFrame(summary).to_csv(out, index=False, encoding="utf-8-sig")
    pa = pd.concat(allrows).groupby(["config", "variant", "action_id"]).correct.mean().unstack(0).mul(100).round(1)
    pa.to_csv(TABLES / "field_robust_per_action.csv", encoding="utf-8-sig")


def final(names, classes):
    """Train S01~S03 → Validation S04 / Test S05 (변형 포함), 5명 + 새 구도로 학습 → 현장 영상"""
    rows, field_rows = [], []
    for name in names:
        cfg = CONFIGS[name]
        data = cfg.get("data", "11n")
        clips = CLIPS[data]
        field = [{"clip": f.name.split("_")[2], **dict(zip("tv", read_frames(f), strict=True))}
                 for f in sorted(FIELD[data].glob("*_sequence.csv"))]
        newcam = []
        for f in sorted(NEWCAM[data].glob("*_sequence.csv")):
            t, v = read_frames(f)
            aid = f.name.split("_")[2]
            newcam.append({"clip_id": f.stem, "subject_id": "S05", "action_id": aid, "uid": classes.index(aid),
                           "t": t, "v": v, "seg_start": t[0], "seg_end": t[-1], "view": "D"})
        n_frame = len(BODY40) + (len(REL_NAMES) if cfg.get("rel") else 0)
        tr = [c for c in clips if c["subject_id"] in SPLIT["train"]]
        for set_name, key in (("validation_S04", "validation"), ("test_S05", "test")):
            te = [c for c in clips if c["subject_id"] in SPLIT[key]]
            mats = test_mats(te, cfg)
            for seed in SEEDS:
                Xtr, ytr = build_train(tr, cfg, seed)
                probs = fit_predict(Xtr, ytr, [mats[k] for k in VARIANTS], seed, n_frame, len(classes), ens=3)
                y = np.array([c["uid"] for c in te])
                for k, p in zip(VARIANTS, probs, strict=True):
                    rows.append({"config": name, "set": set_name, "seed": seed, "variant": k,
                                 "acc": (p.argmax(1) == y).mean() * 100})
        Xtr, ytr = build_train(clips + newcam, cfg, 42)
        Xf = to_matrix([window(c["t"], c["v"], cfg["window"]) for c in field], cfg.get("rel", False))
        p = fit_predict(Xtr, ytr, [Xf], 42, n_frame, len(classes), ens=3)[0]
        for c, pi in zip(field, p, strict=True):
            top = np.argsort(pi)[::-1][:3]
            field_rows.append({"config": name, "clip": c["clip"], "A05_push": round(pi[classes.index("A05")] * 100, 1),
                               "top3": " ".join(f"{classes[i]}:{pi[i] * 100:.0f}" for i in top)})
    r = pd.DataFrame(rows).groupby(["config", "set", "variant"]).acc.mean().unstack(2).round(1)
    r["field4"] = r[["cut", "late", "double", "pad"]].mean(1).round(1)
    print(r.to_string())
    r.to_csv(TABLES / "field_robust_final.csv", encoding="utf-8-sig")
    fr = pd.DataFrame(field_rows)
    print(fr.to_string())
    fr.to_csv(TABLES / "field_robust_field_clips.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
