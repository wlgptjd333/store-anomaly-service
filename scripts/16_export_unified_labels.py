"""팀 통일 라벨로 산출물 다시 내보내기 (2026-11-23).

왜
  - 3팀이 공동 데이터 저장소(motion-capture)에서 사람 번호·동작 번호·동작 이름을 하나로 통일했다.
    (S01~S05 = 팀 공통 번호, A001~A017 = 3자리 동작 번호, T01/T02 = 정면/측면)
  - 이 저장소는 그 전에 만든 번호(T03_S0x_A0x_T01_F/S)로 영상·구간 검수·학습을 끝냈다.
    영상·구간·Feature 는 그대로이고 '이름표'만 다르므로, 파이프라인을 다시 돌리지 않고
    번호만 바꿔 산출물을 다시 쓴다. 대응표: configs/unified_label_map.csv

하는 일
  1) 대응표 검사 (사람 5·방향 2·동작 17, 1:1)
  2) 통일 번호로 다시 쓰기 → reports/unified/
       clip_manifest_unified.csv        영상 1개 = 1행 (통일 clip_id, 구간, 검수 메모, split, 옛 clip_id)
       splits_loso_unified.csv          LOSO fold (통일 번호)
       baseline_predictions_unified.csv 기준 실험 영상별 예측 (정답·예측·2순위 동작 번호도 통일)
       baseline_per_action_unified.csv  동작별 Precision·Recall·F1
       human_qa_unified.csv             사람 검수 기록
     Sequence Dataset 통일본(용량이 커서 Git 제외): reports/dataset/sequence_dataset_{segment,whole}_unified.csv
  3) 확인: 통일 번호 Dataset 으로 다시 학습 → 옛 번호와 예측이 영상마다 같은지, 정확도가 같은지
       → reports/unified/verify_same_results.csv

실행: .venv/bin/python scripts/16_export_unified_labels.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import load_script  # noqa: E402
MAP = pd.read_csv(ROOT / "configs" / "unified_label_map.csv", encoding="utf-8-sig", dtype=str).fillna("")
SPLIT = json.loads((ROOT / "configs" / "subject_split.json").read_text(encoding="utf-8"))
TABLES = ROOT / "reports" / "tables"
SPLITS = ROOT / "reports" / "splits"
DATASET = ROOT / "reports" / "dataset"
OUT = ROOT / "reports" / "unified"

train14 = load_script("14_train_baseline.py")


def table(kind: str, col: str = "unified_id") -> dict[str, str]:
    part = MAP[MAP.kind == kind]
    return dict(zip(part.team_repo_id, part[col]))


SUBJ, VIEW, ACT = table("subject"), table("view"), table("action")
ACT_LABEL, ACT_KO = table("action", "unified_label"), table("action", "unified_name_ko")


def check_map() -> None:
    assert len(SUBJ) == 5 and len(set(SUBJ.values())) == 5, "사람 번호 대응이 1:1 이 아님"
    assert len(VIEW) == 3 and len(ACT) == 17 and len(set(ACT.values())) == 17, "방향·동작 대응 확인"


def new_clip_id(old: str) -> str:
    # T03_S04_A12_T01_F → S04_A012_T01  (take 는 모두 T01 이라 방향(F/S)이 통일 번호의 T01/T02 가 된다)
    _, s, a, _take, v = old.split("_")
    return f"{SUBJ[s]}_{ACT[a]}_{VIEW[v]}"


def convert(df: pd.DataFrame, action_cols=("action_id",)) -> pd.DataFrame:
    df = df.copy()
    if "clip_id" in df:
        df.insert(0, "team_repo_clip_id", df["clip_id"])
        df["clip_id"] = df["clip_id"].map(new_clip_id)
    if "subject_id" in df:
        df["subject_id"] = df["subject_id"].map(SUBJ)
    if "view" in df:
        df["take"] = df["view"].map(VIEW)
    for c in action_cols:
        if c in df:
            df[c] = df[c].map(lambda x: ACT.get(x, x))
    if "action_label" in df:
        df["action_label"] = df["action_id"].map({v: ACT_LABEL[k] for k, v in ACT.items()})
    if "name_ko" in df:
        df["name_ko"] = df["action_id"].map({v: ACT_KO[k] for k, v in ACT.items()})
    return df


def write(df: pd.DataFrame, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / name, index=False, encoding="utf-8-sig")
    print(f"  {name}: {len(df)}행")


def split_of(subject: str) -> str:
    for key in ("train", "validation", "test"):
        if subject in SPLIT["main"][key]:
            return key
    raise KeyError(subject)


def verify() -> pd.DataFrame:
    rows = []
    for window in ("segment", "whole"):
        old = train14.load_dataset(window)
        new = convert(old)
        new.to_csv(DATASET / f"sequence_dataset_{window}_unified.csv", index=False, encoding="utf-8-sig")
        X = old[train14.cols("body40", "sequence")].to_numpy(float)
        for name, df, tr_s, va_s, te_s in (
            ("옛 번호", old, SPLIT["main"]["train"], SPLIT["main"]["validation"], SPLIT["main"]["test"]),
            ("통일 번호", new, [SUBJ[s] for s in SPLIT["main"]["train"]],
             [SUBJ[s] for s in SPLIT["main"]["validation"]], [SUBJ[s] for s in SPLIT["main"]["test"]]),
        ):
            y = df["action_id"].to_numpy(str)
            tr = df.subject_id.isin(tr_s).to_numpy()
            for split_name, part in (("validation", va_s), ("test", te_s)):
                te = df.subject_id.isin(part).to_numpy()
                _, pred, _ = train14.fit_predict(X, y, tr, te)
                rows.append(dict(window=window, labels=name, split=split_name,
                                 accuracy=round(accuracy_score(y[te], pred) * 100, 1),
                                 macro_f1=round(f1_score(y[te], pred, average="macro"), 3),
                                 preds=[p if name == "통일 번호" else ACT[p] for p in pred]))
    R = pd.DataFrame(rows)
    out = []
    for (window, split_name), g in R.groupby(["window", "split"], sort=False):
        a, b = g.iloc[0], g.iloc[1]
        out.append(dict(window=window, split=split_name, acc_team_repo_ids=a.accuracy, acc_unified_ids=b.accuracy,
                        f1_team_repo_ids=a.macro_f1, f1_unified_ids=b.macro_f1,
                        same_prediction_every_clip=a.preds == b.preds))
    return pd.DataFrame(out)


def main() -> None:
    check_map()
    print("통일 라벨로 다시 쓰기 →", OUT.relative_to(ROOT))

    m = convert(pd.read_csv(ROOT / "configs" / "approved_clip_manifest.csv", encoding="utf-8-sig"))
    m["split"] = m["subject_id"].map({SUBJ[s]: split_of(s) for s in SUBJ})
    keep = ["clip_id", "subject_id", "action_id", "action_label", "take", "seg_start", "seg_end",
            "human_note", "split", "team_repo_clip_id"]
    write(m[keep].sort_values("clip_id"), "clip_manifest_unified.csv")

    folds = convert(pd.read_csv(SPLITS / "loso_folds.csv", encoding="utf-8-sig"))
    write(folds.drop(columns=["take"], errors="ignore"), "splits_loso_unified.csv")

    pred = convert(pd.read_csv(TABLES / "baseline_predictions.csv", encoding="utf-8-sig"),
                   action_cols=("true", "pred", "second"))
    write(pred.drop(columns=["view"]), "baseline_predictions_unified.csv")

    per = convert(pd.read_csv(TABLES / "baseline_per_action.csv", encoding="utf-8-sig"))
    write(per, "baseline_per_action_unified.csv")

    qa = convert(pd.read_csv(TABLES / "human_qa.csv", encoding="utf-8-sig"))
    write(qa.drop(columns=["view"]), "human_qa_unified.csv")

    print("확인: 통일 번호로 다시 학습 (body40, 기준 설정)")
    v = verify()
    write(v, "verify_same_results.csv")
    print(v.to_string(index=False))
    if not v.same_prediction_every_clip.all():
        raise SystemExit("통일 번호로 바꾸자 예측이 달라졌습니다 — 대응표를 확인하세요")
    print("Unified label export: COMPLETE (예측·정확도 모두 같음)")


if __name__ == "__main__":
    main()
