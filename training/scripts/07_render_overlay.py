from __future__ import annotations

# ==================================================
# 07_render_overlay.py
# Sequence CSV 의 관절 좌표·Confidence 를 원본 영상 위에 그려서 재생용 영상으로 만든다
# ==================================================
# YOLO 를 다시 돌리지 않는다 — 03_build_sequences 가 저장한 raw_<관절>_x/y/c 를 그대로 그린다.
#   - 관절 점·뼈대 색: 초록 conf >= 0.50 · 주황 0.30~0.50 · 빨강 < 0.30 (숫자도 관절 옆에 표시)
#   - 위쪽: Frame 번호·시간·평균 conf·LOW 관절 수·사람 수·feature_ok / 실패 사유, LOW 관절 이름
#   - 사람 상자: 회색, 화면 끝(1%)에 닿으면 빨강
#   - 아래쪽: 영상 전체의 평균 conf 그래프 + 지금 위치(세로선), Pose 놓친 구간은 빨강
#
# 입력: data/rgb/videos/<clip>.mp4 + reports/sequences/<clip>_sequence.csv
# 출력: reports/videos/<clip>_overlay.mp4 (얼굴이 나오므로 Git 에 올리지 않음)
# 실행:
#   python scripts/07_render_overlay.py --clip T03_S04_A10_T01_F
#   python scripts/07_render_overlay.py --subject S04          # 한 사람의 34개
#   python scripts/07_render_overlay.py --all --height 720

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "clip_manifest.csv"
VIDEOS = ROOT / "data" / "rgb" / "videos"
SEQ_DIR = ROOT / "reports" / "sequences"
OUT_DIR = ROOT / "reports" / "videos"

KP_NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"]
SHORT = ["nose", "L_eye", "R_eye", "L_ear", "R_ear", "L_sho", "R_sho", "L_elb", "R_elb",
         "L_wri", "R_wri", "L_hip", "R_hip", "L_knee", "R_knee", "L_ank", "R_ank"]
EDGES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
         (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]

KP_CONF, KP_WARN = 0.50, 0.30
EDGE_MARGIN = 0.01
GREEN, ORANGE, RED = (80, 220, 80), (0, 165, 255), (60, 60, 255)
GRAPH_H = 90


def conf_color(c):
    if not np.isfinite(c):
        return None
    return GREEN if c >= KP_CONF else ORANGE if c >= KP_WARN else RED


def parse_args():
    p = argparse.ArgumentParser(description="관절 Confidence 를 그린 재생용 영상 만들기")
    p.add_argument("--clip", action="append", default=[], help="clip_id (여러 번 쓸 수 있음)")
    p.add_argument("--subject", help="S01~S04 중 한 사람의 영상 전부")
    p.add_argument("--all", action="store_true", help="Manifest 의 모든 영상")
    p.add_argument("--height", type=int, default=960, help="출력 영상 세로 길이 px (기본 960, 원본보다 크면 원본)")
    return p.parse_args()


def graph_strip(conf, pose_ok, width):
    """영상 전체 평균 conf 그래프(아래쪽 띠) — 매 Frame 다시 그리지 않도록 한 번만 만든다."""
    strip = np.full((GRAPH_H, width, 3), 30, np.uint8)
    n = len(conf)
    xs = (np.arange(n) * (width - 1) / max(n - 1, 1)).astype(int)
    top, bot = 8, GRAPH_H - 8

    def y_of(v):
        return int(bot - (bot - top) * v)

    for v, col in ((KP_CONF, (0, 120, 0)), (KP_WARN, (0, 90, 150))):
        cv2.line(strip, (0, y_of(v)), (width - 1, y_of(v)), col, 1)
    for i in range(n):
        if not pose_ok[i]:
            cv2.line(strip, (xs[i], top), (xs[i], bot), (40, 40, 160), 1)
    pts = [(xs[i], y_of(conf[i])) for i in range(n) if np.isfinite(conf[i])]
    for a, b in zip(pts, pts[1:]):
        cv2.line(strip, a, b, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(strip, "mean conf", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return strip, xs


def text(img, s, org, scale, color=(255, 255, 255), thick=1, outline=False):
    if outline:                                    # 영상 위 숫자만 검은 테두리 (위쪽 띠는 이미 어두움)
        cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 1, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def render(clip_id, video_file, out_h):
    seq_path = SEQ_DIR / f"{clip_id}_sequence.csv"
    if not seq_path.exists():
        raise FileNotFoundError(f"Sequence CSV 가 없습니다 (03_build_sequences 먼저): {seq_path}")
    df = pd.read_csv(seq_path, encoding="utf-8-sig")
    cap = cv2.VideoCapture(str(VIDEOS / video_file))
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {VIDEOS / video_file}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w0, h0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    s = min(1.0, out_h / h0)
    w, h = int(round(w0 * s / 2) * 2), int(round(h0 * s / 2) * 2)
    fs = h / 960                                   # 글자 크기 기준

    xy = np.stack([df[[f"raw_{n}_x", f"raw_{n}_y"]].to_numpy(float) for n in KP_NAMES], 1) * s
    cf = df[[f"raw_{n}_c" for n in KP_NAMES]].to_numpy(float)
    box = df[["box_x1", "box_y1", "box_x2", "box_y2"]].to_numpy(float)
    mean_c = pd.to_numeric(df["mean_keypoint_conf"], errors="coerce").to_numpy(float)
    strip, gx = graph_strip(mean_c, df["pose_ok"].to_numpy() == 1, w)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{clip_id}_overlay.mp4"
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h + GRAPH_H))
    n_rows, i = len(df), 0
    while True:
        ok, frame = cap.read()
        if not ok or i >= n_rows:
            break
        img = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA) if s < 1 else frame
        r = df.iloc[i]

        if np.isfinite(box[i]).all():
            x1, y1, x2, y2 = box[i]
            edge = (x1 <= EDGE_MARGIN * w0 or y1 <= EDGE_MARGIN * h0
                    or x2 >= (1 - EDGE_MARGIN) * w0 or y2 >= (1 - EDGE_MARGIN) * h0)
            cv2.rectangle(img, (int(x1 * s), int(y1 * s)), (int(x2 * s), int(y2 * s)),
                          RED if edge else (170, 170, 170), max(1, int(2 * fs)))
        for a, b in EDGES:
            c = min(cf[i, a], cf[i, b])
            col = conf_color(c)
            if col is not None and np.isfinite(xy[i, [a, b]]).all():
                cv2.line(img, tuple(xy[i, a].astype(int)), tuple(xy[i, b].astype(int)), col,
                         max(2, int(4 * fs)), cv2.LINE_AA)
        for k in range(17):
            col = conf_color(cf[i, k])
            if col is None or not np.isfinite(xy[i, k]).all():
                continue
            p = tuple(xy[i, k].astype(int))
            cv2.circle(img, p, max(3, int(6 * fs)), col, -1, cv2.LINE_AA)
            if k >= 5:                             # 얼굴 5개는 겹쳐서 숫자 생략 (색만)
                text(img, f"{cf[i, k]:.2f}", (p[0] + int(8 * fs), p[1] - int(4 * fs)), 0.45 * fs, col,
                     outline=True)

        panel_h = int(118 * fs)
        img[:panel_h] = (img[:panel_h] * 0.35).astype(np.uint8)
        mc = "-" if not np.isfinite(mean_c[i]) else f"{mean_c[i]:.3f}"
        state = "OK" if r["feature_ok"] == 1 else f"FAIL {r['fail_reason']}"
        people = "" if pd.isna(r["people_count"]) else int(r["people_count"])
        text(img, f"{clip_id}  {r['action_label']}", (10, int(28 * fs)), 0.62 * fs, (255, 255, 255), max(1, int(2 * fs)))
        text(img, f"frame {i + 1}/{n_rows}  {i / fps:5.2f}s", (10, int(56 * fs)), 0.55 * fs)
        low_n = int(r["low_keypoint_count"]) if pd.notna(r["low_keypoint_count"]) else "-"
        text(img, f"mean conf {mc}   low {low_n}"
                  f"   people {people}   {state}", (10, int(82 * fs)), 0.55 * fs,
             (255, 255, 255) if r["feature_ok"] == 1 else RED)
        low = [SHORT[k] for k in range(17) if np.isfinite(cf[i, k]) and cf[i, k] < KP_CONF]
        if low:
            text(img, "LOW: " + " ".join(low), (10, int(108 * fs)), 0.5 * fs, ORANGE)

        bar = strip.copy()
        cv2.line(bar, (gx[i], 0), (gx[i], GRAPH_H - 1), (0, 255, 255), 2)
        writer.write(np.vstack([img, bar]))
        i += 1
    cap.release()
    writer.release()
    return out, i, n_rows


def main():
    args = parse_args()
    man = pd.read_csv(MANIFEST, encoding="utf-8-sig")
    if args.all:
        sel = man
    elif args.subject:
        sel = man[man["subject_id"] == args.subject]
    elif args.clip:
        sel = man[man["clip_id"].isin(args.clip)]
    else:
        sys.exit("--clip, --subject, --all 중 하나를 주세요.")
    if sel.empty:
        sys.exit("고른 영상이 Manifest 에 없습니다.")
    for _, c in sel.iterrows():
        out, done, total = render(c["clip_id"], c["video_file"], args.height)
        note = "" if done == total else f"  (영상 {done} / CSV {total} Frame — 길이 다름)"
        print(f"{c['clip_id']}: {done} Frame → {out.relative_to(ROOT)}{note}", flush=True)


if __name__ == "__main__":
    main()
