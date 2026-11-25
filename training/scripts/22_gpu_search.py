"""GPU 로 더 올릴 수 있나 — 시간 모델·입력·섞는 방법 한꺼번에 비교 (2026-11-24 밤).

왜: 21번에서 ExtraTrees + 1D-CNN 이 4-fold 87.9 로 가장 좋았다. 그 뒤 안 해 본 것들을 같은 규칙으로 확인한다.
  ① 서비스는 1D-CNN 을 난수 3개로 학습해 평균하는데, 21번 4-fold 는 난수 하나짜리 → 서비스 모양 그대로의 점수
  ② 1D-CNN 크기·학습 길이·mixup·Dropout·좌우 반전 TTA
  ③ 다른 시간 모델: TCN(팽창 합성곱), 작은 Transformer
  ④ 입력: 45단계, 큰 Pose 모델(11s·11m) — ExtraTrees 에서는 +0.4~1.1 이었는데 1D-CNN 에서는?
  ⑤ ExtraTrees : 신경망 섞는 비율

규칙 (19~21번과 같음): S01~S04 사람 단위 4-fold, 난수 3개(42, 0, 1) 평균. Test S05 는 고른 뒤 확인만 (--test).
설정이 많을수록 "4-fold 에서 우연히 높은 것"을 고를 위험이 커진다 → 난수 3개 편차보다 확실히 크고(+1%p 넘게),
가장 낮은 사람·뒷모습도 같이 오를 때만 바꾼다. 결과 표에 모든 설정을 남긴다.

결과 (11/24 밤): 서비스 모양(1D-CNN 3개 평균) 90.2 → 폭 128 · 3개 91.7 (Test S05 86.3 → 88.2) → 서비스에 씀 (결정 17번).
  Pose 11m · 3개 94.0 이지만 Test 85.0 · CPU 4배라 보류. 나머지는 변동 안이거나 낮음. 자세히: BASELINE_EVALUATION 5-6.
실행: .venv/bin/python scripts/22_gpu_search.py [--only 이름,…] [--test 이름]   (RTX 3060 약 20분)
결과: reports/tables/gpu_search_cv.csv (+ gpu_search_per_action.csv, --test 면 gpu_search_test.csv)
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
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.schema import BODY40, flip_matrix, seq_columns  # noqa: E402
from src.common import load_script  # noqa: E402
from src.sequence_extras import add_extras  # noqa: E402
from src.temporal_cnn import CNN, fit_cnn, predict_cnn, to_seq  # noqa: E402  (서비스와 같은 코드)

more = load_script("20_more_improvements.py")
et, DATASETS = more.et, more.DATASETS

SPLIT = json.loads((ROOT / "configs" / "subject_split.json").read_text(encoding="utf-8"))["main"]
TABLES = ROOT / "reports" / "tables"
SEEDS = (42, 0, 1)
EXTRAS = ["velocity", "stats"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
BASE = "지금 (11n, 모든 Frame, 30단계)"


class TCN(nn.Module):
    """팽창(dilation) 1·2·4·8 합성곱 — 30단계 전체를 4층 만에 본다"""

    def __init__(self, c_in, n_cls, w=64, p=0.3):
        super().__init__()
        layers, c = [], c_in
        for d in (1, 2, 4, 8):
            layers += [nn.Conv1d(c, w, 3, padding=d, dilation=d), nn.BatchNorm1d(w), nn.ReLU(), nn.Dropout(0.1)]
            c = w
        self.f = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(2 * w, n_cls))

    def forward(self, x):
        h = self.f(x)
        return self.head(torch.cat([h.mean(2), h.amax(2)], 1))


class TinyTransformer(nn.Module):
    def __init__(self, c_in, n_cls, d=64, p=0.3, max_t=64):
        super().__init__()
        self.inp = nn.Linear(c_in, d)
        self.pos = nn.Parameter(torch.zeros(1, max_t, d))
        layer = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.2, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(2 * d, n_cls))

    def forward(self, x):
        h = self.inp(x.transpose(1, 2))
        h = self.enc(h + self.pos[:, : h.shape[1]])
        return self.head(torch.cat([h.mean(1), h.amax(1)], 1))


ARCH = {
    "cnn": None,
    "cnn_w128": lambda c, n: CNN(c, n, w=128),
    "cnn_p05": lambda c, n: CNN(c, n, p=0.5),
    "tcn": lambda c, n: TCN(c, n),
    "transformer": lambda c, n: TinyTransformer(c, n),
}

# 이름: 데이터, 신경망 설정, 섞기   (ens = 한 fold 안에서 난수 몇 개 평균, w_et = ExtraTrees 비율)
CONFIGS = {
    "기준: ExtraTrees + 1D-CNN (21번)": dict(data=BASE),
    "1D-CNN 난수 3개 평균 (서비스 모양)": dict(data=BASE, ens=3),
    "1D-CNN 난수 5개 평균": dict(data=BASE, ens=5),
    "1D-CNN 폭 128": dict(data=BASE, arch="cnn_w128"),
    "1D-CNN Dropout 0.5": dict(data=BASE, arch="cnn_p05"),
    "1D-CNN 200 epoch": dict(data=BASE, epochs=200),
    "1D-CNN mixup 0.2": dict(data=BASE, mixup=0.2),
    "1D-CNN 좌우 반전 TTA": dict(data=BASE, tta=True),
    "TCN (팽창 합성곱)": dict(data=BASE, arch="tcn"),
    "작은 Transformer": dict(data=BASE, arch="transformer"),
    "ExtraTrees 0.3 : 1D-CNN 0.7": dict(data=BASE, w_et=0.3),
    "ExtraTrees 0.7 : 1D-CNN 0.3": dict(data=BASE, w_et=0.7),
    "ExtraTrees + 1D-CNN + TCN": dict(data=BASE, extra_arch="tcn"),
    "45단계": dict(data="45단계"),
    "Pose 11s · 30fps": dict(data="Pose 11s · 30fps"),
    "Pose 11m · 30fps": dict(data="Pose 11m · 30fps"),
    # 2차: 1차에서 난수 편차보다 크게 오른 것끼리 (서비스처럼 난수 3개 평균)
    "1D-CNN 폭 128 · 난수 3개": dict(data=BASE, arch="cnn_w128", ens=3),
    "Pose 11s · 난수 3개": dict(data="Pose 11s · 30fps", ens=3),
    "Pose 11m · 난수 3개": dict(data="Pose 11m · 30fps", ens=3),
    "Pose 11m · 폭 128 · 난수 3개": dict(data="Pose 11m · 30fps", arch="cnn_w128", ens=3),
}

_et_cache: dict = {}
_data_cache: dict = {}


def load(name):
    if name not in _data_cache:
        df, steps = more.load(name)
        if df is None:
            raise SystemExit(f"데이터 없음: {name} (20번 실행 순서 참고)")
        _data_cache[name] = (df, steps)
    return _data_cache[name]


def nn_probs(arch, Xtr, ytr, Xte, seed, steps, cfg):
    n_cls = int(ytr.max()) + 1
    Str, Ste = to_seq(Xtr, steps, len(BODY40)), to_seq(Xte, steps, len(BODY40))
    Ste_f = to_seq(flip_matrix(Xte, steps), steps, len(BODY40)) if cfg.get("tta") else None
    out = []
    for k in range(cfg.get("ens", 1)):
        s = seed + 1000 * k                               # k=0 은 21번과 같은 난수, 나머지는 겹치지 않게
        model, mu, sd = fit_cnn(Str, ytr, s, n_cls, epochs=cfg.get("epochs", 120), device=DEV,
                                make_model=ARCH[arch], mixup=cfg.get("mixup", 0.0))
        p = predict_cnn(model, mu, sd, Ste)
        if Ste_f is not None:
            p = (p + predict_cnn(model, mu, sd, Ste_f)) / 2
        out.append(p)
    return np.mean(out, 0)


def fold(name, cfg, tr, te, classes, seed, tag):
    df, steps = load(cfg["data"])
    cols = seq_columns(steps)
    lab = {c: i for i, c in enumerate(classes)}
    Xtr = tr[cols].to_numpy(float)
    Xtr = np.vstack([Xtr, flip_matrix(Xtr, steps)])
    ytr = np.array([lab[a] for a in tr.action_id.astype(str)] * 2)
    Xte = te[cols].to_numpy(float)
    key = (cfg["data"], seed, tag)
    if key not in _et_cache:
        m = et(seed).fit(add_extras(Xtr, steps, len(BODY40), EXTRAS), ytr)
        _et_cache[key] = m.predict_proba(add_extras(Xte, steps, len(BODY40), EXTRAS))
    p_et = _et_cache[key]
    p_nn = nn_probs(cfg.get("arch", "cnn"), Xtr, ytr, Xte, seed, steps, cfg)
    if cfg.get("extra_arch"):
        p_nn = (p_nn + nn_probs(cfg["extra_arch"], Xtr, ytr, Xte, seed, steps, {})) / 2
    w = cfg.get("w_et", 0.5)
    return w * p_et + (1 - w) * p_nn


def score(te, pred, classes):
    ok = np.array(classes)[pred] == te.action_id.astype(str).to_numpy()
    P = te[["subject_id", "view", "action_id"]].assign(correct=ok)
    return {"acc_all": ok.mean() * 100, "acc_FS": P[P.view != "B"].correct.mean() * 100,
            "acc_B": P[P.view == "B"].correct.mean() * 100,
            "subj_min": P.groupby("subject_id").correct.mean().min() * 100}, P


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="쉼표로 설정 이름 일부")
    ap.add_argument("--test", default="", help="고른 설정 이름(정확히, 쉼표로 여러 개) → 지금 서비스 모양과 함께 Train S01~S03 / Validation S04 / Test S05 확인")
    args = ap.parse_args()
    names = [n for n in CONFIGS if not args.only or any(k in n for k in args.only.split(","))]

    if args.test:
        rows = []
        for name in ["1D-CNN 난수 3개 평균 (서비스 모양)", *args.test.split(",")]:
            cfg = CONFIGS[name]
            df, _ = load(cfg["data"])
            classes = sorted(df.action_id.astype(str).unique())
            tr = df[df.subject_id.isin(SPLIT["train"])]
            for set_name, key in (("validation_S04", "validation"), ("test_S05", "test")):
                te = df[df.subject_id.isin(SPLIT[key])]
                sc = [score(te, fold(name, cfg, tr, te, classes, s, set_name).argmax(1), classes)[0] for s in SEEDS]
                rows.append({"set": set_name, "config": name,
                             **{k: round(float(np.mean([x[k] for x in sc])), 1) for k in sc[0]}})
                print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(TABLES / "gpu_search_test.csv", index=False, encoding="utf-8-sig")
        return

    dev = SPLIT["train"] + SPLIT["validation"]
    rows, per_action = [], []
    print(f"장치 {DEV} · 설정 {len(names)}개", flush=True)
    for name in names:
        cfg = CONFIGS[name]
        df, _ = load(cfg["data"])
        classes = sorted(df.action_id.astype(str).unique())
        t0 = time.time()
        scs = []
        for seed in SEEDS:
            parts = []
            for s in dev:
                tr, te = df[df.subject_id.isin([x for x in dev if x != s])], df[df.subject_id == s]
                parts.append((te, fold(name, cfg, tr, te, classes, seed, s).argmax(1)))
            te_all = pd.concat([t for t, _ in parts])
            sc, P = score(te_all, np.concatenate([p for _, p in parts]), classes)
            scs.append(sc)
            per_action.append(P.assign(config=name, seed=seed))
        r = {"config": name, **{k: round(float(np.mean([x[k] for x in scs])), 1) for k in scs[0]},
             "acc_all_std": round(float(np.std([x["acc_all"] for x in scs])), 1), "sec": round(time.time() - t0)}
        rows.append(r)
        print(f"  {name:34s} 전체 {r['acc_all']:5.1f} ± {r['acc_all_std']:3.1f} | 정면·측면 {r['acc_FS']:5.1f}"
              f" 뒤 {r['acc_B']:5.1f} | 가장 낮은 사람 {r['subj_min']:5.1f} | {r['sec']}s", flush=True)
        pd.DataFrame(rows).to_csv(TABLES / "gpu_search_cv.csv", index=False, encoding="utf-8-sig")
    pa = pd.concat(per_action).groupby(["config", "action_id"]).correct.mean().unstack(0).mul(100).round(1)
    pa.to_csv(TABLES / "gpu_search_per_action.csv", encoding="utf-8-sig")


if __name__ == "__main__":
    main()
