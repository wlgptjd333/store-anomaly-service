"""시간 순서 모델 (1D-CNN · GRU) vs ExtraTrees — 2026-11-24 밤.

왜: ExtraTrees 는 30단계를 펼친 2,560개 값을 "자리"마다 따로 본다. 같은 동작이 조금 빠르거나 늦게 시작하면
같은 자리에 다른 자세가 와서 약하다 (5-3·5-4 에서 남은 약점: 서서 팔만 움직이는 A13·A06·A04 끼리 헷갈림).
1D-CNN 은 시간 방향으로 미끄러지는 필터라 "언제"보다 "어떤 움직임 모양"을 보고, GRU 는 순서대로 읽는다.

규칙은 19·20번과 같다: S01~S04 사람 단위 4-fold, 난수 3개 평균. Test S05 는 고른 뒤 확인만.
입력: reports/dataset/sequence_dataset_whole.csv 의 body40 × 30단계 (+ 변화량 40 = 80 채널).
학습: 정면·측면·뒤 + 좌우 반전 (20번 flip_matrix 와 같은 규칙), 매 epoch 시간 자르기(0~15%)·작은 잡음.
섞기: ExtraTrees 확률과 신경망 확률의 평균도 본다.

결과 (11/24 밤, 서비스와 같은 fit_cnn = cudnn 고정, 두 번 돌려 같은 값):
  4-fold ExtraTrees 82.4 / 1D-CNN 85.0 / GRU 79.7 / ExtraTrees + 1D-CNN 87.9 → 서비스에 씀 (결정 16번)
  고른 뒤 확인 Test S05: ExtraTrees 78.4 → ExtraTrees + 1D-CNN 84.3
실행: .venv/bin/python scripts/21_temporal_models.py [--test]   (GPU 있으면 GPU, 약 4분)
결과: reports/tables/temporal_models_cv.csv  (--test 면 temporal_models_test.csv 도)
"""
from __future__ import annotations

import argparse
import importlib.util
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
from src.sequence_extras import add_extras  # noqa: E402
from src.temporal_cnn import fit_cnn, predict_cnn, to_seq as _to_seq  # noqa: E402  (서비스와 같은 코드)

_spec = importlib.util.spec_from_file_location("more", ROOT / "scripts" / "20_more_improvements.py")
more = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(more)
FRAME, flip_matrix, et = more.FRAME, more.flip_matrix, more.et

SPLIT = json.loads((ROOT / "configs" / "subject_split.json").read_text(encoding="utf-8"))["main"]
DATA = ROOT / "reports" / "dataset" / "sequence_dataset_whole.csv"
TABLES = ROOT / "reports" / "tables"
STEPS = 30
SEEDS = (42, 0, 1)
EXTRAS = ["velocity", "stats"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
COLS = [f"t{i:02d}_{f}" for i in range(STEPS) for f in FRAME]


def to_seq(X: np.ndarray) -> np.ndarray:
    return _to_seq(X, STEPS, len(FRAME))


class GRU(nn.Module):
    def __init__(self, c_in, n_cls, h=64, p=0.3):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(c_in, h), nn.ReLU())
        self.rnn = nn.GRU(h, h, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(4 * h, n_cls))

    def forward(self, x):
        o, _ = self.rnn(self.inp(x.transpose(1, 2)))
        return self.head(torch.cat([o.mean(1), o.amax(1)], 1))


def fit_nn(kind, Xtr, ytr, Xte, seed):
    """학습 규칙(정규화·시간 자르기·잡음·120 epoch·cudnn 고정)은 서비스와 같은 src/temporal_cnn.fit_cnn 하나만 쓴다.
    GRU 는 모델 모양만 바꿔 같은 규칙으로 학습 → 모델 차이만 비교된다."""
    model, mu, sd = fit_cnn(Xtr, ytr, seed, int(ytr.max()) + 1, device=DEV,
                            make_model=GRU if kind == "gru" else None)
    return predict_cnn(model, mu, sd, Xte)


def fold_probs(tr: pd.DataFrame, te: pd.DataFrame, classes, seed, kinds):
    Xtr = tr[COLS].to_numpy(float)
    Xtr = np.vstack([Xtr, flip_matrix(Xtr, STEPS)])
    lab = {c: i for i, c in enumerate(classes)}
    ytr = np.array([lab[a] for a in tr.action_id.astype(str)] * 2)
    Xte = te[COLS].to_numpy(float)
    out = {}
    if "et" in kinds:
        m = et(seed).fit(add_extras(Xtr, STEPS, len(FRAME), EXTRAS), ytr)
        out["et"] = m.predict_proba(add_extras(Xte, STEPS, len(FRAME), EXTRAS))
    for k in ("cnn", "gru"):
        if k in kinds:
            out[k] = fit_nn(k, to_seq(Xtr), ytr, to_seq(Xte), seed)
    return out


COMBOS = {"ExtraTrees (지금)": ["et"], "1D-CNN": ["cnn"], "GRU": ["gru"],
          "ExtraTrees + 1D-CNN 평균": ["et", "cnn"], "ExtraTrees + GRU 평균": ["et", "gru"],
          "ExtraTrees + 1D-CNN + GRU 평균": ["et", "cnn", "gru"]}


def score(te, pred, classes):
    ok = (np.array(classes)[pred] == te.action_id.astype(str).to_numpy())
    P = te[["subject_id", "view", "action_id"]].assign(correct=ok)
    return {"acc_all": ok.mean() * 100, "acc_FS": P[P.view != "B"].correct.mean() * 100,
            "acc_B": P[P.view == "B"].correct.mean() * 100,
            "subj_min": P.groupby("subject_id").correct.mean().min() * 100}, P


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="고른 뒤 확인: Train S01~S03 → Validation S04, Test S05")
    args = ap.parse_args()
    df = pd.read_csv(DATA)
    classes = sorted(df.action_id.astype(str).unique())
    kinds = {"et", "cnn", "gru"}
    print(f"장치 {DEV} · 영상 {len(df)}개", flush=True)

    dev = SPLIT["train"] + SPLIT["validation"]
    res = {c: [] for c in COMBOS}
    per_action = []
    t0 = time.time()
    for seed in SEEDS:
        preds = {c: [] for c in COMBOS}
        for s in dev:
            tr, te = df[df.subject_id.isin([x for x in dev if x != s])], df[df.subject_id == s]
            pr = fold_probs(tr, te, classes, seed, kinds)
            for c, ks in COMBOS.items():
                preds[c].append((te, np.mean([pr[k] for k in ks], 0).argmax(1)))
        for c in COMBOS:
            te_all = pd.concat([t for t, _ in preds[c]])
            p_all = np.concatenate([p for _, p in preds[c]])
            sc, P = score(te_all, p_all, classes)
            res[c].append(sc)
            per_action.append(P.assign(config=c, seed=seed))
        print(f"  난수 {seed} 끝 ({time.time() - t0:.0f}s)", flush=True)

    rows = []
    for c, sc in res.items():
        r = {"config": c, **{k: round(float(np.mean([s[k] for s in sc])), 1) for k in sc[0]},
             "acc_all_std": round(float(np.std([s["acc_all"] for s in sc])), 1)}
        rows.append(r)
        print(f"  {c:30s} 전체 {r['acc_all']:5.1f} ± {r['acc_all_std']:3.1f} | 정면·측면 {r['acc_FS']:5.1f} 뒤 {r['acc_B']:5.1f}"
              f" | 가장 낮은 사람 {r['subj_min']:5.1f}")
    pd.DataFrame(rows).to_csv(TABLES / "temporal_models_cv.csv", index=False, encoding="utf-8-sig")
    pa = pd.concat(per_action).groupby(["config", "action_id"]).correct.mean().unstack(0).mul(100).round(1)
    pa.to_csv(TABLES / "temporal_models_per_action.csv", encoding="utf-8-sig")
    print(pa.to_string())

    if args.test:
        tr = df[df.subject_id.isin(SPLIT["train"])]
        out = []
        for name, key in (("validation_S04", "validation"), ("test_S05", "test")):
            te = df[df.subject_id.isin(SPLIT[key])]
            accs = {c: [] for c in COMBOS}
            for seed in SEEDS:
                pr = fold_probs(tr, te, classes, seed, kinds)
                for c, ks in COMBOS.items():
                    accs[c].append(score(te, np.mean([pr[k] for k in ks], 0).argmax(1), classes)[0])
            for c, sc in accs.items():
                out.append({"set": name, "config": c,
                            **{k: round(float(np.mean([s[k] for s in sc])), 1) for k in sc[0]}})
        T = pd.DataFrame(out)
        T.to_csv(TABLES / "temporal_models_test.csv", index=False, encoding="utf-8-sig")
        print(T.to_string(index=False))


if __name__ == "__main__":
    main()
