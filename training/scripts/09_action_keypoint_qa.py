"""동작별 중요 관절 QA (수업 5일차 27번과 같은 기준) + Human QA 작업표.

수업 기준 (하나라도 걸리면 CHECK, 아니면 AUTO_PASS)
  pose_ratio           < 0.90   Pose 를 찾은 Frame 비율
  single_person_ratio  < 0.90   정확히 한 명만 잡힌 Frame 비율
  mean_keypoint_conf   < 0.50   전체 관절 Confidence 평균
  low_critical_ratio   > 0.20   이 동작의 중요 관절이 Confidence 0.5 미만인 비율

3팀에 맞게 바꾼 점 (docs/DESIGN_DECISIONS.md 11번)
  1) 06 에서 추정한 동작 구간(start ~ end_rule, 사람이 고친 *_final 우선) 안의 Frame 만 본다.
     동작 전후에 서 있는 시간은 판정에 넣지 않는다.
  2) 좌우가 있는 관절은 "둘 중 더 잘 보이는 쪽"의 Confidence 를 쓴다.
     측면에서 카메라 반대쪽 팔·다리가 몸에 가려지는 건 촬영으로 피할 수 없고,
     동작은 보이는 쪽으로도 판단할 수 있기 때문이다.

출력
  reports/tables/action_keypoint_qa.csv   Clip 마다 지표·qa_status
  reports/tables/human_qa.csv             사람 판정표 (human_decision: PASS / RESHOOT / EXCLUDE, human_note)
                                          — 다시 돌려도 이미 적은 판정은 그대로 둔다
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEQ = ROOT / "reports" / "sequences"
TABLES = ROOT / "reports" / "tables"
MANIFEST = ROOT / "configs" / "clip_manifest.csv"
SEGMENTS = TABLES / "segments_auto.csv"
OUT_QA = TABLES / "action_keypoint_qa.csv"
OUT_HUMAN = TABLES / "human_qa.csv"

KP_CONF = 0.50
LIMITS = {"pose_ratio": 0.90, "single_person_ratio": 0.90, "mean_keypoint_conf": 0.50, "low_critical_ratio": 0.20}

ARMS = ["shoulder", "elbow", "wrist"]
LEGS = ["hip", "knee", "ankle"]
BODY = ["shoulder", "hip", "knee"]
# 동작마다 판단에 꼭 필요한 관절 (좌우 묶음 이름, "nose" 는 단일)
CRITICAL = {
    "A01": ARMS, "A02": ARMS, "A04": ARMS, "A05": ARMS, "A06": ARMS, "A13": ARMS,   # 팔로 하는 동작
    "A03": LEGS,                                                                    # 발차기
    "A07": LEGS, "A08": LEGS,                                                       # 앉기
    "A09": BODY, "A10": BODY, "A15": BODY, "A16": BODY, "A17": BODY,                # 눕기·쓰러짐
    "A11": ["shoulder", "hip"],                                                     # 몸통 박치기
    "A12": ["shoulder", "hip", "knee", "wrist"],                                    # 무릎 꿇고 손 넣기
    "A14": ["nose", "shoulder", "hip"],                                             # 몸을 숙여 구토
}
QA_FIELDS = ["clip_id", "subject_id", "action_id", "view", "seg_start", "seg_end", "frame_count",
             "pose_ratio", "single_person_ratio", "mean_keypoint_conf", "low_critical_ratio",
             "missing_pose_frames", "critical_keypoints", "check_reasons", "qa_status"]
HUMAN_FIELDS = ["clip_id", "subject_id", "action_id", "view", "qa_status", "check_reasons",
                "human_decision", "human_note"]


def segment_table() -> dict[str, tuple[float, float]]:
    seg = {}
    with SEGMENTS.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            s = float(r["start_sec_final"] or r["start_sec"])
            e = float(r["end_sec_final"] or r["end_sec_rule"])
            seg[r["clip_id"]] = (s, e)
    return seg


def critical_conf(df: pd.DataFrame, names: list[str]) -> np.ndarray:
    """(T, 관절 묶음 수) — 좌우 중 높은 Confidence."""
    cols = []
    for n in names:
        if n == "nose":
            cols.append(pd.to_numeric(df["raw_nose_c"], errors="coerce").to_numpy())
        else:
            l = pd.to_numeric(df[f"raw_left_{n}_c"], errors="coerce").to_numpy()
            r = pd.to_numeric(df[f"raw_right_{n}_c"], errors="coerce").to_numpy()
            cols.append(np.fmax(l, r))
    return np.stack(cols, axis=1)


def qa_clip(row: dict, seg: tuple[float, float]) -> dict:
    df = pd.read_csv(SEQ / f"{row['clip_id']}_sequence.csv")
    t = pd.to_numeric(df["time_sec"], errors="coerce")
    df = df[(t >= seg[0]) & (t <= seg[1])]
    n = len(df)
    pose = pd.to_numeric(df["pose_ok"], errors="coerce").fillna(0).to_numpy()
    people = pd.to_numeric(df["people_count"], errors="coerce").fillna(0).to_numpy()
    names = CRITICAL[row["action_id"]]
    out = {k: row[k] for k in ("clip_id", "subject_id", "action_id", "view")}
    out.update(seg_start=seg[0], seg_end=seg[1], frame_count=n, critical_keypoints="|".join(names))
    if n == 0:
        out.update(pose_ratio=0.0, single_person_ratio=0.0, mean_keypoint_conf=0.0, low_critical_ratio=1.0,
                   missing_pose_frames=0, check_reasons="no_frames", qa_status="CHECK")
        return out
    ok = df[pose == 1]
    mean_conf = float(pd.to_numeric(ok["mean_keypoint_conf"], errors="coerce").mean()) if len(ok) else 0.0
    crit = critical_conf(ok, names) if len(ok) else np.zeros((0, len(names)))
    low = float(np.mean(crit < KP_CONF)) if crit.size else 1.0
    out.update(pose_ratio=round(float(pose.mean()), 4), single_person_ratio=round(float(np.mean(people == 1)), 4),
               mean_keypoint_conf=round(mean_conf, 4), low_critical_ratio=round(low, 4),
               missing_pose_frames=int((pose == 0).sum()))
    reasons = []
    for k, lim in LIMITS.items():
        bad = out[k] > lim if k == "low_critical_ratio" else out[k] < lim
        if bad:
            reasons.append(k)
    out["check_reasons"] = "|".join(reasons)
    out["qa_status"] = "CHECK" if reasons else "AUTO_PASS"
    return out


def main() -> None:
    seg = segment_table()
    manifest = list(csv.DictReader(MANIFEST.open(encoding="utf-8-sig")))
    rows = [qa_clip(r, seg[r["clip_id"]]) for r in manifest]
    with OUT_QA.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=QA_FIELDS, lineterminator="\r\n")
        w.writeheader()
        w.writerows(rows)

    old = {}
    if OUT_HUMAN.exists():
        old = {r["clip_id"]: r for r in csv.DictReader(OUT_HUMAN.open(encoding="utf-8-sig"))}
    human = []
    for r in rows:
        prev = old.get(r["clip_id"], {})
        human.append({k: r[k] for k in ("clip_id", "subject_id", "action_id", "view", "qa_status", "check_reasons")}
                     | {"human_decision": prev.get("human_decision", ""), "human_note": prev.get("human_note", "")})
    with OUT_HUMAN.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HUMAN_FIELDS, lineterminator="\r\n")
        w.writeheader()
        w.writerows(human)

    q = pd.DataFrame(rows)
    print(f"{OUT_QA.relative_to(ROOT)}: {len(q)}개 · {q.qa_status.value_counts().to_dict()}")
    for _, r in q[q.qa_status == "CHECK"].iterrows():
        print(f"  CHECK {r.clip_id}  {r.check_reasons}  pose {r.pose_ratio} single {r.single_person_ratio} "
              f"conf {r.mean_keypoint_conf} low {r.low_critical_ratio} ({r.critical_keypoints})")
    todo = sum(1 for h in human if not h["human_decision"])
    print(f"{OUT_HUMAN.relative_to(ROOT)}: 사람 판정 빈칸 {todo}개")


if __name__ == "__main__":
    main()
