"""서비스 모델 정확도 올리기 실험 (2026-11-24) — 같은 body40 whole 입력에서 Feature·모델을 바꿔 본다.

고르는 규칙 (baseline_config.json selection_rule 그대로)
  - 비교·선택은 S01~S04 사람 단위 4-fold (한 명씩 빼고 학습, 그 사람으로 평가) 로만 한다.
  - Test S05 는 고른 뒤 확인용으로만 본다. 5-fold LOSO(S05 포함)도 참고로 적는다.
  - 학습 데이터: 정면·측면·뒤 + 좌우 반전 (18번 실험의 가장 좋은 설정), 창 = whole (서비스 입력과 같은 모양)

바꿔 보는 것
  Feature  base   = 30단계 × body40 = 1,200 (지금 서비스)
           +vel   = 단계 사이 변화량 29 × 40 = 1,160 을 더함 (움직임 속도·방향)
           +stats = Feature 40개마다 30단계의 평균·표준편차·최솟값·최댓값·범위 200 을 더함 (시간 위치에 덜 민감)
           -pos   = 골반 화면 위치 2개(hip_x_img, hip_y_img)를 뺌 (카메라 구도가 바뀌면 흔들리는 값)
  모델     RF300(지금) · RF1000 · ExtraTrees1000 · SVM(RBF)
           (HistGradientBoosting 은 너무 느려 뺌. LogisticRegression 은 base 에서 4-fold 63.2% 로 RF 보다 낮고
            한 번에 150초씩 걸려 뺌)

실행: .venv/bin/python scripts/19_model_improvement.py
결과: reports/tables/model_improvement_cv.csv, model_improvement_test.csv
"""
from __future__ import annotations

import json
import time
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.schema import BODY40  # noqa: E402
from src.common import load_script  # noqa: E402
SPLIT = json.loads((ROOT / "configs" / "subject_split.json").read_text(encoding="utf-8"))["main"]
TABLES = ROOT / "reports" / "tables"



t15 = load_script("15_export_service_model.py")
STEPS, COLUMNS = t15.STEPS, t15.COLUMNS
F = len(BODY40)
POS_IDX = [BODY40.index("hip_x_img"), BODY40.index("hip_y_img")]


def features(X: np.ndarray, kind: str) -> np.ndarray:
    """X: (n, 1200) t-major → 실험용 Feature"""
    seq = X.reshape(len(X), STEPS, F)
    if "-pos" in kind:
        keep = [i for i in range(F) if i not in POS_IDX]
        seq = seq[:, :, keep]
    parts = [seq.reshape(len(X), -1)]
    if "+vel" in kind:
        parts.append(np.diff(seq, axis=1).reshape(len(X), -1))
    if "+stats" in kind:
        parts += [seq.mean(1), seq.std(1), seq.min(1), seq.max(1), seq.max(1) - seq.min(1)]
    return np.hstack(parts)


def models(seed: int) -> dict:
    return {
        "RF300": lambda: RandomForestClassifier(300, class_weight="balanced", random_state=seed, n_jobs=-1),
        "RF1000": lambda: RandomForestClassifier(1000, class_weight="balanced", random_state=seed, n_jobs=-1),
        "ET1000": lambda: ExtraTreesClassifier(1000, class_weight="balanced", random_state=seed, n_jobs=-1),
        "SVM": lambda: make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", probability=True,  # sklearn 1.9 경고: 1.11 에서 바뀜
                                                           class_weight="balanced", random_state=seed)),
    }


def fit_predict(make, tr: pd.DataFrame, te: pd.DataFrame, kind: str):
    tr = pd.concat([tr, t15.t17.flip(tr)], ignore_index=True)
    m = make().fit(features(tr[COLUMNS].to_numpy(float), kind), tr.action_id.astype(str))
    return m.predict(features(te[COLUMNS].to_numpy(float), kind))


def cv(df: pd.DataFrame, subjects: list[str], make, kind: str) -> pd.DataFrame:
    out = []
    for s in subjects:
        tr = df[df.subject_id.isin([x for x in subjects if x != s])]
        te = df[df.subject_id == s]
        p = fit_predict(make, tr, te, kind)
        out.append(te[["clip_id", "subject_id", "view", "action_id"]].assign(pred=p))
    P = pd.concat(out)
    P["correct"] = (P.pred == P.action_id.astype(str)).astype(int)
    return P


def summary(P: pd.DataFrame) -> dict:
    r = {f"acc_{v}": round(P[P.view == v].correct.mean() * 100, 1) for v in ("F", "S", "B")}
    r["acc_all"] = round(P.correct.mean() * 100, 1)
    r["subj_min"] = round(P.groupby("subject_id").correct.mean().min() * 100, 1)
    return r


def main() -> None:
    df = pd.read_csv(t15.DATA)
    dev = SPLIT["train"] + SPLIT["validation"]                    # S01~S04
    print(f"영상 {len(df)}개, 선택용 4-fold = {dev}")
    kinds = ["base", "base+vel", "base+stats", "base+vel+stats", "base-pos", "base+stats-pos"]
    rows = []
    # 1단계: 모든 모델 × Feature, seed 42
    for kind in kinds:
        for name, make in models(42).items():
            t = time.time()
            r = dict(feature=kind, model=name, seeds=1, **summary(cv(df, dev, make, kind)),
                     sec=round(time.time() - t, 1))
            rows.append(r)
            print(f"  {kind:16s} {name:7s} 전체 {r['acc_all']:5.1f} | 정면 {r['acc_F']:5.1f} 측면 {r['acc_S']:5.1f} "
                  f"뒤 {r['acc_B']:5.1f} | 가장 낮은 사람 {r['subj_min']:5.1f} ({r['sec']}s)", flush=True)
    C = pd.DataFrame(rows)
    # 2단계: 상위 4개 + 지금 모델을 난수 3개로 다시
    top = C.sort_values("acc_all", ascending=False).head(4)[["feature", "model"]].values.tolist()
    if ["base", "RF300"] not in top:
        top.append(["base", "RF300"])
    rows3 = []
    for kind, name in top:
        Ps = [cv(df, dev, models(seed)[name], kind) for seed in (42, 0, 1)]
        accs = [summary(P) for P in Ps]
        r = dict(feature=kind, model=name, seeds=3,
                 **{k: round(float(np.mean([a[k] for a in accs])), 1) for k in accs[0]},
                 acc_all_std=round(float(np.std([a["acc_all"] for a in accs])), 1))
        rows3.append(r)
        print(f"  [3 seeds] {kind:16s} {name:7s} 전체 {r['acc_all']:5.1f} ± {r['acc_all_std']} 뒤 {r['acc_B']:5.1f}")
    C = pd.concat([C, pd.DataFrame(rows3)], ignore_index=True)
    C.to_csv(TABLES / "model_improvement_cv.csv", index=False, encoding="utf-8-sig")

    # 3단계: 확인 — 상위 후보와 지금 모델의 Test S05, 5-fold LOSO
    R3 = pd.DataFrame(rows3).sort_values("acc_all", ascending=False)
    trows = []
    for kind, name in R3[["feature", "model"]].values.tolist():
        tr = df[df.subject_id.isin(SPLIT["train"])]
        for split_name, part in (("validation_S04", SPLIT["validation"]), ("test_S05", SPLIT["test"])):
            te = df[df.subject_id.isin(part)]
            accs = []
            for seed in (42, 0, 1):
                ok = fit_predict(models(seed)[name], tr, te, kind) == te.action_id.to_numpy(str)
                accs.append([ok.mean() * 100] + [ok[(te.view == v).to_numpy()].mean() * 100 for v in "FSB"])
            a = np.mean(accs, axis=0)
            trows.append(dict(feature=kind, model=name, eval=split_name, acc_all=round(a[0], 1), acc_F=round(a[1], 1),
                              acc_S=round(a[2], 1), acc_B=round(a[3], 1)))
        loso = summary(cv(df, sorted(df.subject_id.unique()), models(42)[name], kind))
        trows.append(dict(feature=kind, model=name, eval="loso5_seed42", **{k: loso[k] for k in
                                                                            ("acc_all", "acc_F", "acc_S", "acc_B")}))
    T = pd.DataFrame(trows)
    T.to_csv(TABLES / "model_improvement_test.csv", index=False, encoding="utf-8-sig")
    print(T.to_string(index=False))


if __name__ == "__main__":
    main()
