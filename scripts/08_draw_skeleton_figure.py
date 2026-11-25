"""README·발표용 동작 그림 — 얼굴 없이 관절 뼈대만 그린다.

원본 영상은 쓰지 않고 Sequence CSV 의 원본 Keypoint(raw_<관절>_x/y/c)만 읽는다.
동작마다 06 에서 추정한 구간(start_sec ~ end_sec_rule, 사람이 고친 *_final 이 있으면 그것)을
4 시점으로 나눠 뼈대를 나란히 그린다. 발 높이를 맞추고 같은 배율로 그려서 앉기·눕기가 보인다.

출력: docs/images/actions_skeleton.png  (얼굴·영상이 없어 Git 에 올림)
실행:
  python scripts/08_draw_skeleton_figure.py                    # 기본: S04 측면
  python scripts/08_draw_skeleton_figure.py --subject S02 --view F --out docs/images/actions_S02_F.png
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.schema import KP  # noqa: E402
SEQ = ROOT / "reports" / "sequences"
SEGMENTS = ROOT / "reports" / "tables" / "segments_auto.csv"
CLASSES = ROOT / "configs" / "action_classes.csv"

# (a, b, 색) — 왼쪽 파랑, 오른쪽 주황, 몸통 회색
EDGES = [(5, 7, "L"), (7, 9, "L"), (11, 13, "L"), (13, 15, "L"),
         (6, 8, "R"), (8, 10, "R"), (12, 14, "R"), (14, 16, "R"),
         (5, 6, "C"), (11, 12, "C"), (5, 11, "C"), (6, 12, "C")]
COLORS = {"L": "#2f6fd6", "R": "#e8833a", "C": "#7a7a7a"}
CONF = 0.3
N_POSES = 4


def setup_font() -> None:
    f = Path("/mnt/c/Windows/Fonts/malgun.ttf")
    if f.exists():
        font_manager.fontManager.addfont(str(f))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(f)).get_name()


def segment_of(clip_id: str) -> tuple[float, float]:
    with SEGMENTS.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r["clip_id"] == clip_id:
                s = float(r["start_sec_final"] or r["start_sec"])
                e = float(r["end_sec_final"] or r["end_sec_rule"])
                return s, e
    raise KeyError(clip_id)


def poses_at(df: pd.DataFrame, times: list[float]) -> list[np.ndarray]:
    """시점마다 가장 가까운 Pose 성공 Frame 의 (17, 3) [x, y, conf]."""
    ok = df[pd.to_numeric(df["pose_ok"], errors="coerce") == 1]
    t = pd.to_numeric(ok["time_sec"], errors="coerce").to_numpy()
    cols = [f"raw_{k}_{a}" for k in KP for a in ("x", "y", "c")]
    arr = ok[cols].apply(pd.to_numeric, errors="coerce").to_numpy().reshape(len(ok), 17, 3)
    return [arr[int(np.argmin(np.abs(t - x)))] for x in times]


def draw_tile(ax, poses: list[np.ndarray], title: str) -> None:
    body = list(range(5, 17))
    boxes = []
    for p in poses:
        v = p[body][p[body, 2] >= CONF]
        boxes.append((v[:, 0].min(), v[:, 0].max(), v[:, 1].min(), v[:, 1].max()) if len(v) else None)
    unit = max(max(b[1] - b[0], b[3] - b[2]) for b in boxes if b) * 1.1
    for i, (p, b) in enumerate(zip(poses, boxes)):
        if b is None:
            continue
        cx = (b[0] + b[1]) / 2
        foot = b[3]
        # 칸 i: x ∈ [i, i+1], 발 = y 0, 위로 갈수록 +
        X = i + 0.5 + (p[:, 0] - cx) / unit
        Y = (foot - p[:, 1]) / unit + 0.05
        alpha = 0.45 + 0.55 * i / (len(poses) - 1)
        for a, c, side in EDGES:
            if p[a, 2] >= CONF and p[c, 2] >= CONF:
                ax.plot([X[a], X[c]], [Y[a], Y[c]], color=COLORS[side], lw=2.2, alpha=alpha,
                        solid_capstyle="round")
        # 머리는 점 하나로만 (얼굴 관절 위치 = 코·귀 평균)
        head = [k for k in (0, 3, 4) if p[k, 2] >= CONF]
        if head:
            ax.scatter(X[head].mean(), Y[head].mean(), s=60, color="#444444", alpha=alpha, zorder=3)
        vis = [k for k in body if p[k, 2] >= CONF]
        ax.scatter(X[vis], Y[vis], s=8, color="#222222", alpha=alpha, zorder=3)
    ax.axhline(0.05, color="#cccccc", lw=0.8)
    for i in range(1, len(poses)):
        ax.axvline(i, color="#eeeeee", lw=0.6)
    ax.set_xlim(0, len(poses))
    ax.set_ylim(-0.02, 1.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=10, loc="left")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--subject", default="S04")
    ap.add_argument("--view", default="S", choices=["F", "S"])
    ap.add_argument("--out", default="docs/images/actions_skeleton.png")
    args = ap.parse_args()
    setup_font()

    classes = list(csv.DictReader(CLASSES.open(encoding="utf-8-sig")))
    # 그림 제목은 팀 통일 번호·이름 (서비스 화면과 같게, 결정 15번). 영상 파일 이름은 이 저장소 번호 그대로
    unified = {r["team_repo_id"]: f"{r['unified_id']} {r['unified_name_ko']}"
               for r in csv.DictReader((ROOT / "configs" / "unified_label_map.csv").open(encoding="utf-8-sig"))
               if r["kind"] == "action"}
    ncol = 3
    nrow = int(np.ceil(len(classes) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 5.2, nrow * 1.7))
    for ax in axes.flat[len(classes):]:
        ax.axis("off")
    for ax, c in zip(axes.flat, classes):
        clip = f"T03_{args.subject}_{c['action_id']}_T01_{args.view}"
        df = pd.read_csv(SEQ / f"{clip}_sequence.csv")
        s, e = segment_of(clip)
        times = list(np.linspace(s, e, N_POSES))
        draw_tile(ax, poses_at(df, times), unified[c["action_id"]])
    view = "측면" if args.view == "S" else "정면"
    fig.suptitle(f"동작 17개 — 2D Pose 뼈대 ({args.subject} {view}, 동작 구간을 4 시점으로, 왼쪽 → 오른쪽 시간 순)",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight", facecolor="white")
    print(out.relative_to(ROOT))


if __name__ == "__main__":
    main()
