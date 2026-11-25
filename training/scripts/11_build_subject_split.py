"""Approved Manifest 를 사람 단위로 Train / Validation / Test 로 나누고 Data Leakage 를 검사 (수업 5일차 30·31번).

입력: configs/approved_clip_manifest.csv, configs/subject_split.json
출력:
  reports/splits/train.csv, validation.csv, test.csv        기본 Split (subject_split.json 의 main)
  reports/splits/loso_folds.csv                              사람 5명이 한 번씩 Test 가 되는 5-fold 목록
검사 (하나라도 실패하면 멈춤):
  - 한 사람이 두 세트 이상에 들어가지 않음
  - 한 clip_id 가 두 세트 이상에 들어가지 않음
  - Approved 영상이 하나도 빠지지 않음
  - 세트마다 동작 17개가 모두 있음
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APPROVED = ROOT / "configs" / "approved_clip_manifest.csv"
CONFIG = ROOT / "configs" / "subject_split.json"
OUT = ROOT / "reports" / "splits"


def write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\r\n")
        w.writeheader()
        w.writerows(rows)


def check(sets: dict[str, list[dict]], approved: list[dict], actions: set[str]) -> list[str]:
    errors = []
    subj = {k: {r["subject_id"] for r in v} for k, v in sets.items()}
    clips = {k: {r["clip_id"] for r in v} for k, v in sets.items()}
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if subj[a] & subj[b]:
                errors.append(f"사람 겹침 {a}·{b}: {sorted(subj[a] & subj[b])}")
            if clips[a] & clips[b]:
                errors.append(f"영상 겹침 {a}·{b}: {len(clips[a] & clips[b])}개")
    used = set().union(*clips.values())
    lost = {r["clip_id"] for r in approved} - used
    if lost:
        errors.append(f"어느 세트에도 없는 영상 {len(lost)}개")
    for k, v in sets.items():
        miss = actions - {r["action_id"] for r in v}
        if miss:
            errors.append(f"{k} 에 없는 동작: {sorted(miss)}")
    return errors


def main() -> None:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    approved = list(csv.DictReader(APPROVED.open(encoding="utf-8-sig")))
    actions = {r["action_id"] for r in approved}
    subjects = sorted({r["subject_id"] for r in approved})
    listed = [s for v in cfg["main"].values() for s in v]
    if sorted(listed) != subjects:
        sys.exit(f"subject_split.json 의 사람 {sorted(listed)} 와 Approved 사람 {subjects} 가 다릅니다")

    sets = {k: [r for r in approved if r["subject_id"] in v] for k, v in cfg["main"].items()}
    errors = check(sets, approved, actions)
    if errors:
        sys.exit("Leakage 검사 실패:\n  " + "\n  ".join(errors))
    OUT.mkdir(parents=True, exist_ok=True)
    for k, v in sets.items():
        write(OUT / f"{k}.csv", [{**r, "split": k} for r in v])
        c = Counter(r["view"] for r in v)
        print(f"{k:10s} 사람 {sorted({r['subject_id'] for r in v})} · 영상 {len(v)} (정면 {c['F']} 측면 {c['S']} 뒤 {c['B']})")
    print("Leakage 검사 통과: 사람·영상 겹침 없음, 빠진 영상 없음, 세트마다 동작 17개")

    if cfg.get("leave_one_subject_out"):
        folds = []
        for i, s in enumerate(subjects, 1):
            fold_sets = {"train": [r for r in approved if r["subject_id"] != s],
                         "test": [r for r in approved if r["subject_id"] == s]}
            errors = check(fold_sets, approved, actions)
            if errors:
                sys.exit(f"LOSO fold {i} 실패: {errors}")
            folds += [{"fold": i, "clip_id": r["clip_id"], "subject_id": r["subject_id"], "action_id": r["action_id"],
                       "split": k} for k, v in fold_sets.items() for r in v]
        write(OUT / "loso_folds.csv", folds)
        print(f"LOSO {len(subjects)}-fold: 사람마다 한 번씩 Test, 모든 fold 검사 통과 → {OUT.name}/loso_folds.csv")


if __name__ == "__main__":
    main()
