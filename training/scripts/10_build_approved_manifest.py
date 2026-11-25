"""Human QA 결과로 학습에 쓸 영상만 고른 Approved Manifest (수업 5일차 28·29번).

입력: configs/clip_manifest.csv, reports/tables/human_qa.csv, reports/tables/segments_auto.csv
검사: 모든 영상에 human_decision(PASS / RESHOOT / EXCLUDE)이 빠짐없이 적혔는지 — 빈칸이면 멈춤
출력: configs/approved_clip_manifest.csv  (PASS 만, 학습에 쓸 동작 구간 seg_start / seg_end 포함)
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "clip_manifest.csv"
HUMAN = ROOT / "reports" / "tables" / "human_qa.csv"
SEGMENTS = ROOT / "reports" / "tables" / "segments_auto.csv"
OUT = ROOT / "configs" / "approved_clip_manifest.csv"
DECISIONS = {"PASS", "RESHOOT", "EXCLUDE"}


def read(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return {r["clip_id"]: r for r in csv.DictReader(f)}


def main() -> None:
    manifest, human, seg = read(MANIFEST), read(HUMAN), read(SEGMENTS)
    missing = [c for c in manifest if c not in human or human[c]["human_decision"] not in DECISIONS]
    if missing:
        sys.exit(f"Human QA 판정이 없거나 잘못된 영상 {len(missing)}개: {missing[:10]}")

    rows = []
    for cid, m in manifest.items():
        if human[cid]["human_decision"] != "PASS":
            continue
        s = seg[cid]
        rows.append({**{k: m[k] for k in ("clip_id", "subject_id", "action_id", "action_label", "take_id", "view")},
                     "sequence_file": f"reports/sequences/{cid}_sequence.csv",
                     "seg_start": s["start_sec_final"] or s["start_sec"],
                     "seg_end": s["end_sec_final"] or s["end_sec_rule"],
                     "human_note": human[cid]["human_note"]})
    with OUT.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\r\n")
        w.writeheader()
        w.writerows(rows)

    dec = Counter(h["human_decision"] for h in human.values())
    print(f"Human QA: {dict(dec)}")
    print(f"{OUT.relative_to(ROOT)}: {len(rows)}개 / 사람 {len({r['subject_id'] for r in rows})}명 / "
          f"동작 {len({r['action_id'] for r in rows})}개")
    short = Counter((r["subject_id"], r["action_id"]) for r in rows)
    lack = [(s, a) for s in sorted({r["subject_id"] for r in rows}) for a in sorted({r["action_id"] for r in rows})
            if short[(s, a)] == 0]
    if lack:
        print(f"  사람·동작 조합이 빈 곳 (RESHOOT 후보): {lack}")


if __name__ == "__main__":
    main()
