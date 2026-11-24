"""카메라 구도가 바뀌어도 맞히게 — 새 구도 테스트로 함께 고르기 (2026-11-24 밤).

왜: S05 가 새 구도(가로 화면, 높은 곳 대각선)로 찍은 5개를 지금 서비스가 2/5 만 맞힘 (같은 사람인데도).
    4-fold(같은 촬영 구도) 숫자는 91.7 로 높지만, 실제 웹캠 설치 위치는 매번 다르다 → 구도 변화에 강한 방법을 찾는다.
후보 (모두 지금 서비스 모양 = ExtraTrees + 1D-CNN 폭 128 · 난수 3개, 좌우 반전 포함):
  - nopos  : 골반 화면 위치(hip_x_img, hip_y_img) 2개를 뺌 — 화면 구도가 바뀌면 값 자체가 달라지는 Feature
  - view   : 시점 증강 — 몸 좌표를 가로 0.6~1.1배(사람이 돌아선 것처럼), 세로 0.8~1.05배(위에서 내려다본 것처럼) 늘이고
             몸 크기 정규화·관절 각도를 다시 계산한 사본 2개 (24번 회전과 달리 '다른 각도에서 본 모양'을 흉내)
  - crop   : 앞뒤 0~15% 자르기 사본 2개 (24번, 4-fold 최고였음)
  - 조합, 그리고 23번 VICON 합성(16방향)을 nopos 와 같이
평가 (미리 정한 규칙):
  1) 4-fold S01~S04 (난수 3개) — 지금보다 0.5%p 넘게 떨어지면 안 되고, 가장 낮은 사람·뒤도 1%p 넘게 떨어지면 안 됨
  2) 새 구도 5개 (Train S01~S04 → S05 새 구도, 난수 3개 평균) — 맞힌 수와 '정답 확률 평균'이 올라야 함
  3) 둘 다 만족하는 것 중 새 구도 정답 확률이 가장 높은 것. Test S05(원래 구도 51개)는 고른 뒤 확인 (2%p = 영상 1개 넘게 떨어지면 안 씀)
  새 구도 5개는 영상이 적어 숫자가 크게 흔들림 → 정답 확률 평균(연속값)을 같이 본다.

실행: .venv/bin/python scripts/26_camera_robust.py [--only nopos,view] [--extra reports/extra/s05_newcam]
결과: reports/tables/camera_robust_cv.csv, camera_robust_newcam.csv (영상별), camera_robust_test.csv
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.feature_utils import selected_joint_angles  # noqa: E402
from src.sequence_extras import add_extras  # noqa: E402
from src.team_feature_utils import BODY_SHOULDER_RATIO  # noqa: E402
from src.temporal_cnn import fit_cnn, predict_cnn, to_seq  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


g = _load("g", "22_gpu_search.py")
aug24 = _load("aug24", "24_geo_augment.py")
FRAME, STEPS = g.FRAME, 30
IX = {f: i for i, f in enumerate(FRAME)}
KP = aug24.KP
XS, YS = [IX[f"bn_{k}_x"] for k in KP], [IX[f"bn_{k}_y"] for k in KP]
POS = [IX["hip_x_img"], IX["hip_y_img"]]
ANG = [IX[a] for a in g.more.ANGLES]
LS, RS, LH, RH = (KP.index(k) for k in ("left_shoulder", "right_shoulder", "left_hip", "right_hip"))

CONFIGS = {
    "지금 서비스": dict(),
    "nopos": dict(nopos=True),
    "view": dict(view=2),
    "crop": dict(crop=2),
    "view+crop": dict(view=2, crop=2),
    "nopos+view": dict(nopos=True, view=2),
    "nopos+view+crop": dict(nopos=True, view=2, crop=2),
    "nopos+VICON 합성": dict(nopos=True, syn=True),
    # 새 구도 영상 자체를 학습에 더하면 기존 4-fold 가 떨어지지 않는지만 본다 (S05 라 S01~S04 평가와 겹치지 않음).
    # 새 구도 정확도는 그 영상으로 배웠으니 잴 수 없음 → TEST 는 건너뜀
    "+새 구도 5개 학습": dict(extra=True),
    "view+새 구도 5개 학습": dict(view=2, extra=True),
}


def view_aug(X: np.ndarray, copies: int, rng) -> np.ndarray:
    """(n, steps×40) → 시점 바꾼 사본 (copies×n). 클립마다 같은 배율 (카메라는 클립 내내 고정)."""
    seq = X.reshape(len(X), STEPS, len(FRAME))
    out = []
    for _ in range(copies):
        A = seq.copy()
        for n in range(len(A)):
            sx, sy = rng.uniform(0.6, 1.1), rng.uniform(0.8, 1.05)
            P = np.stack([A[n][:, XS] * sx, A[n][:, YS] * sy], -1)            # (T, 17, 2), 골반 중심 그대로 0
            torso = np.linalg.norm((P[:, LS] + P[:, RS]) / 2 - (P[:, LH] + P[:, RH]) / 2, axis=1)
            sw = np.linalg.norm(P[:, LS] - P[:, RS], axis=1) / BODY_SHOULDER_RATIO
            P = P / np.maximum(np.maximum(torso, sw), 1e-6)[:, None, None]
            A[n][:, XS], A[n][:, YS] = P[..., 0], P[..., 1]
            for t in range(STEPS):
                ang = selected_joint_angles(P[t])
                A[n][t, ANG] = [float(ang[a]) for a in g.more.ANGLES]
        out.append(A.reshape(len(X), -1))
    return np.vstack(out)


def make_train(tr: pd.DataFrame, cfg: dict, seed: int, syn: pd.DataFrame | None, cols, lab):
    X = tr[cols].to_numpy(float)
    y = np.array([lab[a] for a in tr.action_id.astype(str)])
    if cfg.get("syn") and syn is not None:
        X = np.vstack([X, syn[cols].to_numpy(float)])
        y = np.concatenate([y, [lab[a] for a in syn.action_id.astype(str)]])
    rng = np.random.default_rng(seed)
    parts, ys = [X], [y]
    if cfg.get("view"):
        parts.append(view_aug(X, cfg["view"], rng))
        ys.append(np.tile(y, cfg["view"]))
    if cfg.get("crop"):
        df = pd.DataFrame(X, columns=cols).assign(subject_id="x", action_id="x", view="x")
        parts.append(aug24.augment(df, ("crop",), cfg["crop"], STEPS, seed)[cols].to_numpy(float))
        ys.append(np.tile(y, cfg["crop"]))
    X, y = np.vstack(parts), np.concatenate(ys)
    return np.vstack([X, g.flip_matrix(X, STEPS)]), np.concatenate([y, y])


def keep_cols(X: np.ndarray, nopos: bool) -> tuple[np.ndarray, int]:
    if not nopos:
        return X, len(FRAME)
    keep = [i for i in range(len(FRAME)) if i not in POS]
    return X.reshape(len(X), STEPS, len(FRAME))[:, :, keep].reshape(len(X), -1), len(keep)


def fit_predict(cfg, tr, tests, classes, seed, syn):
    """학습 한 번 → 여러 평가 세트 확률. ExtraTrees + 1D-CNN 폭 128 난수 3개 평균 (서비스와 같은 계산)."""
    cols = [f"t{i:02d}_{f}" for i in range(STEPS) for f in FRAME]
    lab = {c: i for i, c in enumerate(classes)}
    Xtr, ytr = make_train(tr, cfg, seed, syn, cols, lab)
    Xtr, nf = keep_cols(Xtr, cfg.get("nopos", False))
    Xte = [keep_cols(t[cols].to_numpy(float), cfg.get("nopos", False))[0] for t in tests]
    m = g.et(seed).fit(add_extras(Xtr, STEPS, nf, g.EXTRAS), ytr)
    p_et = [m.predict_proba(add_extras(x, STEPS, nf, g.EXTRAS)) for x in Xte]
    Str = to_seq(Xtr, STEPS, nf)
    p_nn = [np.zeros_like(p) for p in p_et]
    for k in range(3):
        model, mu, sd = fit_cnn(Str, ytr, seed + 1000 * k, len(classes), epochs=120, device=g.DEV, width=128)
        for i, x in enumerate(Xte):
            p_nn[i] += predict_cnn(model, mu, sd, to_seq(x, STEPS, nf)) / 3
    return [(a + b) / 2 for a, b in zip(p_et, p_nn)]


def load_syn():
    path = ROOT / "reports" / "dataset" / "mocap_synthetic_whole.csv"
    return pd.read_csv(path) if path.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=",".join(CONFIGS))
    ap.add_argument("--extra", default="reports/extra/s05_newcam")
    ap.add_argument("--skip-cv", action="store_true")
    args = ap.parse_args()
    df, _ = g.load(g.BASE)
    classes = sorted(df.action_id.astype(str).unique())
    new = pd.read_csv(ROOT / args.extra / "sequence_dataset_whole.csv")
    syn = load_syn()
    dev = g.SPLIT["train"] + g.SPLIT["validation"]
    cv_rows, new_rows, test_rows = [], [], []
    for name in args.only.split(","):
        cfg = CONFIGS[name]
        t0 = time.time()
        if not args.skip_cv:
            scs, subj = [], []
            for seed in g.SEEDS:
                parts = []
                for s in dev:
                    tr, te = df[df.subject_id.isin([x for x in dev if x != s])], df[df.subject_id == s]
                    if cfg.get("extra"):
                        tr = pd.concat([tr, new], ignore_index=True)
                    sy = None if (syn is None or s == "S03") else syn          # 23번과 같음: 연기자 본인 평가 때는 뺌
                    parts.append((te, fit_predict(cfg, tr, [te], classes, seed, sy)[0].argmax(1)))
                sc, P = g.score(pd.concat([t for t, _ in parts]), np.concatenate([p for _, p in parts]), classes)
                scs.append(sc)
                subj.append(P.groupby("subject_id").correct.mean() * 100)
            cv = {"config": name, **{k: round(float(np.mean([x[k] for x in scs])), 1) for k in scs[0]},
                  "acc_all_std": round(float(np.std([x["acc_all"] for x in scs])), 1),
                  **{f"acc_{k}": round(float(v), 1) for k, v in pd.concat(subj, axis=1).mean(axis=1).items()}}
            cv_rows.append(cv)
            print("CV", cv, flush=True)
        if cfg.get("extra"):
            continue
        # Train S01~S04 → Test S05(원래 구도) + S05 새 구도
        tr = df[df.subject_id.isin(dev)]
        s05 = df[df.subject_id.isin(g.SPLIT["test"])]
        probs_new, accs = [], []
        for seed in g.SEEDS:
            p_s05, p_new = fit_predict(cfg, tr, [s05, new], classes, seed, syn)
            accs.append(g.score(s05, p_s05.argmax(1), classes)[0])
            probs_new.append(p_new)
        pn = np.mean(probs_new, 0)
        truth = new.action_id.astype(str).to_numpy()
        ti = np.array([classes.index(a) for a in truth])
        pred = np.array(classes)[pn.argmax(1)]
        true_p = pn[np.arange(len(new)), ti]
        rank = [int((pn[i] > pn[i, ti[i]]).sum()) + 1 for i in range(len(new))]
        new_rows += [{"config": name, "clip_id": c, "truth": t, "pred": p, "conf": round(float(pn[i].max()) * 100, 1),
                      "true_prob": round(float(true_p[i]) * 100, 1), "true_rank": rank[i]}
                     for i, (c, t, p) in enumerate(zip(new.clip_id, truth, pred))]
        tst = {"config": name, **{f"s05_{k}": round(float(np.mean([x[k] for x in accs])), 1) for k in accs[0]},
               "new_correct": int((pred == truth).sum()), "new_n": len(new),
               "new_true_prob": round(float(true_p.mean()) * 100, 1),
               "new_top3": int((np.array(rank) <= 3).sum()), "sec": round(time.time() - t0)}
        test_rows.append(tst)
        print("TEST", tst, flush=True)
    T = g.TABLES
    for rows, fname in ((cv_rows, "camera_robust_cv.csv"), (new_rows, "camera_robust_newcam.csv"),
                        (test_rows, "camera_robust_test.csv")):
        if not rows:
            continue
        out = T / fname
        new_df = pd.DataFrame(rows)
        if out.exists():
            old = pd.read_csv(out)
            old = old[~old.config.isin(new_df.config)]
            new_df = pd.concat([old, new_df], ignore_index=True)
        new_df.to_csv(out, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
