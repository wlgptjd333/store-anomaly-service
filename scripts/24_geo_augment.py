"""Keypoint 기하 증강 — 학습 데이터에만 '살짝 바꾼 사본'을 더해 본다 (2026-11-24 밤).

왜: 좌우 반전은 이미 쓰고 있고(효과 있음), 남은 차이는 (1) 카메라가 살짝 기울어진 것, (2) 사람마다 동작 빠르기,
    (3) 서비스는 8초 창이라 동작 앞뒤가 잘린 채 들어오는 것. 이 셋을 학습 데이터에서 흉내 낸다.
  - rot  : 몸 좌표(bn_*, 골반 중심·몸 크기로 나눈 값)를 ±10° 회전. 관절 각도·몸 크기는 회전해도 그대로라 정확한 변환
  - warp : 시간 축을 고르지 않게 늘이고 줄임 (t' = t + a·sin(πt), |a| ≤ 0.15) → 같은 동작을 빠르게/느리게
  - crop : 앞이나 뒤를 0~15% 잘라 다시 30단계로 (창 경계 흉내)
  - rotcrop : rot + crop (단독으로 효과 있던 둘)
  - all  : 넷을 한꺼번에
  - hip  : hip_x_img 를 ±0.08 이동 (화면 안 위치)
규칙: 평가 사람 데이터는 절대 안 바꿈. 사본은 그 fold 의 학습 사람 것만. 4-fold S01~S04, 난수 3개, 22번과 같은 모델(폭 128 · 3개).

실행: .venv/bin/python scripts/24_geo_augment.py [--only rot,all] [--copies 2] [--test "all"]
결과: reports/tables/geo_augment_cv.csv (+ _test)
"""
from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("g", ROOT / "scripts" / "22_gpu_search.py")
g = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g)

KP = [f[3:-2] for f in g.FRAME if f.startswith("bn_") and f.endswith("_x")]
AUGS = {"rot": ("rot",), "warp": ("warp",), "crop": ("crop",), "hip": ("hip",), "rotcrop": ("rot", "crop"), "all": ("rot", "warp", "crop", "hip")}


def augment(tr: pd.DataFrame, kinds, copies: int, steps: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    F = len(g.FRAME)
    cols = [f"t{i:02d}_{f}" for i in range(steps) for f in g.FRAME]
    X = tr[cols].to_numpy(float).reshape(len(tr), steps, F)
    ix = {f: j for j, f in enumerate(g.FRAME)}
    xs, ys = [ix[f"bn_{k}_x"] for k in KP], [ix[f"bn_{k}_y"] for k in KP]
    out = []
    t = np.linspace(0, 1, steps)
    for _ in range(copies):
        A = X.copy()
        for n in range(len(A)):
            if "rot" in kinds:
                th = np.radians(rng.uniform(-10, 10))
                x, y = A[n][:, xs].copy(), A[n][:, ys].copy()
                A[n][:, xs], A[n][:, ys] = x * np.cos(th) - y * np.sin(th), x * np.sin(th) + y * np.cos(th)
            src = t.copy()
            if "warp" in kinds:
                src = np.clip(t + rng.uniform(-0.15, 0.15) * np.sin(np.pi * t), 0, 1)
            if "crop" in kinds:
                a, b = (rng.uniform(0, 0.15), 1.0) if rng.random() < 0.5 else (0.0, 1 - rng.uniform(0, 0.15))
                src = a + src * (b - a)
            if "warp" in kinds or "crop" in kinds:
                A[n] = np.stack([np.interp(src, t, A[n][:, j]) for j in range(F)], 1)
            if "hip" in kinds:
                A[n][:, ix["hip_x_img"]] += rng.uniform(-0.08, 0.08)
        meta = tr[["subject_id", "action_id", "view"]].reset_index(drop=True)
        out.append(pd.concat([meta, pd.DataFrame(A.reshape(len(tr), -1), columns=cols)], axis=1))
    return pd.concat(out, ignore_index=True)


def fold_aug(cfg, tr, te, classes, seed, tag, kinds, copies, steps):
    if kinds:
        tr = pd.concat([tr, augment(tr, kinds, copies, steps, seed)], ignore_index=True)
        tag = f"{tag}+{'-'.join(kinds)}x{copies}"
    return g.fold("", cfg, tr, te, classes, seed, tag)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=",".join(AUGS))
    ap.add_argument("--copies", type=int, default=2)
    ap.add_argument("--baseline", action="store_true", help="증강 없는 것도 같이 (23번과 같은 숫자라 기본은 생략)")
    ap.add_argument("--test", default="", help='고른 뒤 확인: "all" 처럼 쉼표로')
    args = ap.parse_args()
    cfg = g.CONFIGS["1D-CNN 폭 128 · 난수 3개"]
    df, steps = g.load(cfg["data"])
    classes = sorted(df.action_id.astype(str).unique())

    if args.test:
        rows = []
        tr = df[df.subject_id.isin(g.SPLIT["train"])]
        for name in ["none", *args.test.split(",")]:
            kinds = AUGS.get(name, ())
            for set_name, key in (("validation_S04", "validation"), ("test_S05", "test")):
                te = df[df.subject_id.isin(g.SPLIT[key])]
                sc = [g.score(te, fold_aug(cfg, tr, te, classes, s, set_name, kinds, args.copies, steps).argmax(1),
                              classes)[0] for s in g.SEEDS]
                rows.append({"set": set_name, "config": name, **{k: round(float(np.mean([x[k] for x in sc])), 1)
                                                                  for k in sc[0]}})
                print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(g.TABLES / "geo_augment_test.csv", index=False, encoding="utf-8-sig")
        return

    dev = g.SPLIT["train"] + g.SPLIT["validation"]
    names = (["none"] if args.baseline else []) + args.only.split(",")
    rows = []
    for name in names:
        kinds = AUGS.get(name, ())
        t0, scs, subj = time.time(), [], []
        for seed in g.SEEDS:
            parts = []
            for s in dev:
                tr, te = df[df.subject_id.isin([x for x in dev if x != s])], df[df.subject_id == s]
                parts.append((te, fold_aug(cfg, tr, te, classes, seed, s, kinds, args.copies, steps).argmax(1)))
            sc, P = g.score(pd.concat([t for t, _ in parts]), np.concatenate([p for _, p in parts]), classes)
            scs.append(sc)
            subj.append(P.groupby("subject_id").correct.mean() * 100)
        r = {"config": f"{name} ×{args.copies}" if kinds else "증강 없음",
             **{k: round(float(np.mean([x[k] for x in scs])), 1) for k in scs[0]},
             "acc_all_std": round(float(np.std([x["acc_all"] for x in scs])), 1),
             **{f"acc_{k}": round(float(v), 1) for k, v in pd.concat(subj, axis=1).mean(axis=1).items()},
             "sec": round(time.time() - t0)}
        rows.append(r)
        print(r, flush=True)
    out = g.TABLES / "geo_augment_cv.csv"
    old = pd.read_csv(out) if out.exists() else pd.DataFrame()
    new = pd.DataFrame(rows)
    if len(old):
        old = old[~old.config.isin(new.config)]
    pd.concat([old, new], ignore_index=True).to_csv(out, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
