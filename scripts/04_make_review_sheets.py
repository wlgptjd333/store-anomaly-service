from __future__ import annotations

# ==================================================
# 04_make_review_sheets.py
# 3팀 영상 라벨 사람 확인용 그림 + 자동 점검표 (4일차 STEP 3 라벨 검수)
# ==================================================
# - 동작마다 1장: 행 = 영상 8개 (S01~S04 × 정면/측면), 열 = 영상 안 6개 시점 (5%~95%)
#   → 한 장만 보고 "이 8개가 모두 같은 동작인가 / 잘못 찍힌 영상은 없는가" 판단
# - 6개 시점마다 YOLO Pose 로 사람 수를 세어 자동 경고를 붙임
#     no_person     : 사람이 안 잡힌 시점이 있음
#     multi_person  : 2명 이상 잡힌 시점이 있음 (옆 사람·거울·포스터)
#     short / long  : 같은 동작 영상들의 중간 길이보다 0.5배 미만 / 2배 초과
# - 출력
#     reports/review/<action_id>_<label>.jpg   (Git 제외: 얼굴이 나옴)
#     reports/tables/review_checklist.csv      (사람이 review_result 칸을 채움)
# 실행: python scripts/04_make_review_sheets.py [--action A07]

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import load_config, resolve_device  # noqa: E402

MANIFEST = ROOT / "configs" / "clip_manifest.csv"
CLASSES = ROOT / "configs" / "action_classes.csv"
INVENTORY = ROOT / "reports" / "tables" / "video_inventory.csv"
VIDEOS = ROOT / "data" / "rgb" / "videos"
SHEET_DIR = ROOT / "reports" / "review"
CHECKLIST = ROOT / "reports" / "tables" / "review_checklist.csv"

POSITIONS = [0.05, 0.20, 0.40, 0.60, 0.80, 0.95]   # 영상 길이 대비 시점
THUMB_W, THUMB_H = 180, 320                        # 세로 영상 (1080×1920) 비율
LABEL_W, HEADER_H = 260, 70


def read_csv(path):
    with path.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def grab_frames(path: Path):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for p in POSITIONS:
        idx = max(0, min(total - 1, int(total * p)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        frames.append((idx + 1, frame if ok else None))
    cap.release()
    return frames


def thumb(frame, text, people):
    if frame is None:
        img = np.full((THUMB_H, THUMB_W, 3), 40, np.uint8)
        cv2.putText(img, "read fail", (20, THUMB_H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return img
    img = cv2.resize(frame, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)
    color = (0, 255, 0) if people == 1 else (0, 0, 255)
    cv2.rectangle(img, (0, 0), (THUMB_W - 1, 22), (0, 0, 0), -1)
    cv2.putText(img, f"{text}  p={people}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    if people != 1:
        cv2.rectangle(img, (0, 0), (THUMB_W - 1, THUMB_H - 1), (0, 0, 255), 3)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--action", action="append", default=[], help="action_id 만 (여러 번 가능, 기본 전체)")
    args = ap.parse_args()

    manifest = read_csv(MANIFEST)
    names = {r["action_id"]: r for r in read_csv(CLASSES)}
    inv = {r["clip_id"]: r for r in read_csv(INVENTORY)}
    by_action = defaultdict(list)
    for r in manifest:
        by_action[r["action_id"]].append(r)
    actions = sorted(args.action) if args.action else sorted(by_action)

    cfg = load_config()
    device = resolve_device(cfg["device"])
    model = YOLO(str(ROOT / cfg["pose_model"]))

    # 이미 있는 점검표의 사람 입력(review_result, review_note)은 지우지 않음
    old = {r["clip_id"]: r for r in read_csv(CHECKLIST)} if CHECKLIST.exists() else {}
    out_rows = {cid: r for cid, r in old.items()}
    SHEET_DIR.mkdir(parents=True, exist_ok=True)

    for aid in actions:
        clips = sorted(by_action[aid], key=lambda r: (r["subject_id"], r["view"], r["take_id"]))
        durs = [float(inv[c["clip_id"]]["duration_sec"]) for c in clips if inv.get(c["clip_id"], {}).get("duration_sec")]
        med = statistics.median(durs) if durs else 0
        label = names[aid]["action_label"]
        rows_img = []
        for c in clips:
            cid = c["clip_id"]
            dur = float(inv[cid]["duration_sec"]) if inv.get(cid, {}).get("duration_sec") else 0.0
            frames = grab_frames(VIDEOS / c["video_file"])
            counts, tiles = [], []
            for fid, fr in frames:
                n = 0
                if fr is not None:
                    res = model.predict(source=fr, imgsz=cfg["image_size"], conf=cfg["person_confidence"],
                                        device=device, verbose=False)[0]
                    n = 0 if res.keypoints is None else len(res.keypoints.xy)
                counts.append(n)
                tiles.append(thumb(fr, f"f{fid}", n))
            flags = []
            if min(counts) == 0:
                flags.append("no_person")
            if max(counts) > 1:
                flags.append("multi_person")
            if med and dur < med * 0.5:
                flags.append("short")
            if med and dur > med * 2.0:
                flags.append("long")

            lab = np.full((THUMB_H, LABEL_W, 3), 255, np.uint8)
            for i, t in enumerate([cid, f"{c['subject_id']}  {dict(F='FRONT', S='SIDE', B='BACK')[c['view']]}",
                                   f"{dur:.2f}s (med {med:.2f})", " ".join(flags) or "auto OK"]):
                col = (0, 0, 200) if (i == 3 and flags) else (0, 0, 0)
                cv2.putText(lab, t, (8, 40 + i * 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1 + (i == 0))
            rows_img.append(np.hstack([lab] + tiles))

            prev = old.get(cid, {})
            out_rows[cid] = {"clip_id": cid, "subject_id": c["subject_id"], "action_id": aid,
                             "action_label": label, "view": c["view"], "duration_sec": round(dur, 2),
                             "action_median_sec": round(med, 2), "people_counts": "/".join(map(str, counts)),
                             "auto_flags": " ".join(flags),
                             "review_result": prev.get("review_result", ""),
                             "review_note": prev.get("review_note", "")}

        header = np.full((HEADER_H, rows_img[0].shape[1], 3), 230, np.uint8)
        cv2.putText(header, f"{aid}  {label}   ({len(clips)} clips, columns = "
                    + " / ".join(f"{int(p * 100)}%" for p in POSITIONS) + ")",
                    (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        sheet = np.vstack([header] + rows_img)
        out = SHEET_DIR / f"{aid}_{label}.jpg"
        cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        flagged = sum(1 for c in clips if out_rows[c["clip_id"]]["auto_flags"])
        print(f"{aid} {label}: {len(clips)} clips, 자동 경고 {flagged}  → {out.relative_to(ROOT)}")

    CHECKLIST.parent.mkdir(parents=True, exist_ok=True)
    fields = ["clip_id", "subject_id", "action_id", "action_label", "view", "duration_sec", "action_median_sec",
              "people_counts", "auto_flags", "review_result", "review_note"]
    with CHECKLIST.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(sorted(out_rows.values(), key=lambda r: r["clip_id"]))
    print(f"Saved: {CHECKLIST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
