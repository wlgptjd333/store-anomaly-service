"""동작 시작·끝 구간 자동 추정 (사람 검수용 초안).

공식
  자세 벡터 P_t = 몸 크기 정규화 관절 12개(어깨·팔꿈치·손목·골반·무릎·발목, bn_*)
                 (골반 기준이라 걸어서 자리만 옮기는 건 변화로 안 잡음)
  1) 결측 Frame 은 선형 보간, 0.2초 이동평균으로 떨림 제거
  2) 기준 자세 R = 처음 0.5초 P 의 중앙값 (동작 전 서 있는 자세)
  3) 자세 변화 D_t = 관절별 |P_t - R| 평균  (몸 크기 단위)
  4) 움직임 E_t = |P_t - P_{t-0.1초}| 평균 / 0.1초  (몸 크기 / 초)
  5) 문턱 θ = max(0.15, 기준 구간 D 의 평균 + 4·표준편차)
     시작 = D 가 θ 를 처음 넘는 Frame 에서, E 가 낮아지는 곳까지 거꾸로 되짚은 Frame
     끝   = D 가 θ 를 마지막으로 넘는 Frame 에서, E 가 낮아지는 곳까지 앞으로 간 Frame
            단, 마지막 0.5초에도 D > θ (원래 자세로 안 돌아옴: 누움·엎드림 등) 이면
            → 끝 = 영상 끝 (자세 유지가 동작의 일부), flag 'ends_changed'
  6) 누움·앉음 동작(A07 A09 A10 A15 A16 A17)은 일어나는 장면을 뺌 → end_sec_rule
     H_t = 사람 상자 높이 / 처음 0.5초 상자 높이  (누우면·앉으면 작아짐)
     낮은 자세 = H < 0.65, H 최저가 0.65 이상이면 H < H최저 + 0.5·(1 − H최저)
                (정면에서 카메라 쪽으로 누우면 상자 높이가 덜 줄어서)
     마지막 낮은 자세 구간의 바닥 높이 + 0.08 이하를 마지막으로 유지한 Frame = 일어나기 직전
     나머지 동작은 end_sec_rule = end_sec
  학습에 쓰는 구간 = start_sec_final/end_sec_final(사람이 고친 값)이 있으면 그것, 없으면 start_sec ~ end_sec_rule

출력: reports/tables/segments_auto.csv  (사람이 고칠 start_sec_final / end_sec_final 열 포함)
      reports/segments/<aid>_<label>.jpg  (확인 그림, --sheets)
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEQ = ROOT / "reports" / "sequences"
TABLES = ROOT / "reports" / "tables"
OUT_CSV = TABLES / "segments_auto.csv"
SHEETS = ROOT / "reports" / "segments"
VIDEOS = ROOT / "data" / "rgb" / "videos"
CLASSES = ROOT / "configs" / "action_classes.csv"

JOINTS = ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
          "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
BASE_SEC = 0.5
SMOOTH_SEC = 0.2
LAG_SEC = 0.1
THETA_MIN = 0.15
E_LOW_RATIO = 0.25   # E 가 동작 중 최대값의 25% 아래면 "멈춤"으로 봄
# 누움·앉음 동작: 일어나는 장면은 구간에서 뺌 (2026-11-17 팀 결정 — 여러 동작에 공통이라 헷갈림)
LOW_ACTIONS = {"A07", "A09", "A10", "A15", "A16", "A17"}
LOW_H = 0.65         # 상자 높이가 서 있을 때의 65% 미만이면 낮은 자세
H_SMOOTH_SEC = 0.3
MIN_DROP = 0.15      # 상자 높이가 15% 도 안 줄면 낮은 자세를 못 찾은 것으로 봄

FIELDS = ["clip_id", "subject_id", "action_id", "view", "duration_sec", "theta", "start_sec", "end_sec",
          "end_sec_rule", "segment_sec", "flags", "start_sec_final", "end_sec_final", "segment_note"]


def pose_matrix(df: pd.DataFrame) -> np.ndarray:
    """(T, 24) 자세 벡터. 결측은 NaN. 골반 기준 좌표라 걸어서 자리만 옮기는 건 변화로 안 잡음."""
    cols = [f"bn_{j}_{a}" for j in JOINTS for a in ("x", "y")]
    P = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    return P


def fill_smooth(P: np.ndarray, win: int) -> np.ndarray:
    df = pd.DataFrame(P).interpolate(limit_direction="both")
    return df.rolling(win, center=True, min_periods=1).mean().to_numpy()


def per_joint_dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    d = (A - B).reshape(len(A), -1, 2)
    return np.linalg.norm(d, axis=2).mean(axis=1)


def estimate(df: pd.DataFrame) -> dict:
    t = pd.to_numeric(df["time_sec"], errors="coerce").to_numpy(float)
    fps = float(pd.to_numeric(df["fps"], errors="coerce").median())
    step = float(np.median(np.diff(t))) if len(t) > 1 else 1 / fps
    n_base = max(3, int(round(BASE_SEC / step)))
    lag = max(1, int(round(LAG_SEC / step)))
    P = fill_smooth(pose_matrix(df), max(1, int(round(SMOOTH_SEC / step))))

    R = np.median(P[:n_base], axis=0)
    D = per_joint_dist(P, np.broadcast_to(R, P.shape))
    E = np.zeros(len(P))
    E[lag:] = per_joint_dist(P[lag:], P[:-lag]) / (lag * step)

    theta = max(THETA_MIN, float(D[:n_base].mean() + 4 * D[:n_base].std()))
    flags = []
    if E[:n_base].max() > 1.0:
        flags.append("moving_at_start")
    above = np.where(D > theta)[0]
    if len(above) == 0:
        return dict(theta=theta, start=t[0], end=t[-1], end_rule=t[-1], flags="no_motion",
                    D=D, E=E, H=np.ones(len(t)), t=t)

    e_low = E_LOW_RATIO * float(E[above[0]:above[-1] + 1].max())
    i0 = above[0]
    while i0 > 0 and E[i0 - 1] > e_low:
        i0 -= 1
    tail = D[-n_base:]
    if (tail > theta).mean() > 0.5:
        i1 = len(t) - 1
        flags.append("ends_changed")
    else:
        i1 = above[-1]
        while i1 < len(t) - 1 and E[i1 + 1] > e_low:
            i1 += 1
    if i0 <= n_base:
        flags.append("starts_early")

    # 사람 상자 높이 H = 상자 높이 / 처음 0.5초 상자 높이 (누움·앉음이면 작아짐, 정면·측면 모두)
    box = df[["box_y1", "box_y2"]].apply(pd.to_numeric, errors="coerce")
    hgt = (box["box_y2"] - box["box_y1"]).interpolate(limit_direction="both")
    hgt = hgt.rolling(max(1, int(round(H_SMOOTH_SEC / step))), center=True, min_periods=1).mean().to_numpy()
    H = hgt / max(1e-6, float(np.median(hgt[:n_base])))

    ie = i1
    if df["action_id"].iloc[0] in LOW_ACTIONS:
        seg = H[i0:i1 + 1]
        hmin = float(seg.min())
        # 정면에서 카메라 쪽으로 누우면 상자 높이가 덜 줄어듦 → 문턱을 그 영상의 최저값 기준으로 낮춤
        thr = LOW_H if hmin < LOW_H else hmin + 0.5 * (1.0 - hmin)
        low = seg < thr
        if 1.0 - hmin < MIN_DROP:
            flags.append("no_low_posture")
        else:
            # 마지막 낮은 자세 구간 [a, b] 와 그 구간의 바닥 높이
            b = int(np.where(low)[0][-1])
            a = b
            while a > 0 and low[a - 1]:
                a -= 1
            level = float(np.median(seg[a:b + 1]))
            # 바닥 높이 근처(+0.08)를 마지막으로 유지한 Frame = 일어나기 직전
            near = np.where(seg[a:b + 1] <= level + 0.08)[0]
            ie = i0 + a + int(near[-1])
            if thr > LOW_H:
                flags.append("low_threshold_adapted")
            if ie >= len(t) - 2:
                flags.append("no_getup")
    return dict(theta=theta, start=t[i0], end=t[i1], end_rule=t[ie], flags=";".join(flags),
                D=D, E=E, H=H, t=t)


def read_existing() -> dict[str, dict]:
    if not OUT_CSV.exists():
        return {}
    with OUT_CSV.open(encoding="utf-8-sig", newline="") as f:
        return {r["clip_id"]: r for r in csv.DictReader(f)}


def grab(clip_id: str, sec: float, size=(135, 240)):
    import cv2
    cap = cv2.VideoCapture(str(VIDEOS / f"{clip_id}.mp4"))
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, sec) * 1000)
    ok, img = cap.read()
    cap.release()
    if not ok:
        return None
    h, w = img.shape[:2]
    s = min(size[0] / w, size[1] / h)
    return cv2.cvtColor(cv2.resize(img, (int(w * s), int(h * s))), cv2.COLOR_BGR2RGB)


def make_sheet(aid: str, label: str, name_ko: str, items: list[tuple[dict, dict]]) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font = Path("/mnt/c/Windows/Fonts/malgun.ttf")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()

    n = len(items)
    fig = plt.figure(figsize=(16, 2.4 * n))
    gs = fig.add_gridspec(n, 5, width_ratios=[3.2, 1, 1, 1, 1], wspace=0.05, hspace=0.35)
    for i, (row, res) in enumerate(items):
        ax = fig.add_subplot(gs[i, 0])
        ax.plot(res["t"], res["D"], label="자세 변화 D", color="tab:blue")
        ax2 = ax.twinx()
        ax2.plot(res["t"], res["E"], label="움직임 E", color="tab:orange", alpha=0.6)
        ax.axhline(res["theta"], color="tab:blue", ls=":", lw=1)
        s, e = float(row["start_sec"]), float(row["end_sec"])
        ax.plot(res["t"], res["H"], label="상자 높이 H", color="tab:gray", lw=1)
        h = float(row["end_sec_rule"])
        ax.axvspan(s, h, color="tab:green", alpha=0.18)
        if h < e:
            ax.axvspan(h, e, color="tab:red", alpha=0.08)
        ax.set_title(f'{row["clip_id"]}  {s:.2f}s → {h:.2f}s  {row["flags"]}', fontsize=9, loc="left")
        ax.tick_params(labelsize=7)
        ax2.tick_params(labelsize=7)
        mid = (s + h) / 2
        for j, (sec, cap) in enumerate([(s, "시작"), (mid, "가운데"), (h, "끝"), (h + 0.7, "끝+0.7s")]):
            a = fig.add_subplot(gs[i, j + 1])
            img = grab(row["clip_id"], sec)
            if img is not None:
                a.imshow(img)
            a.set_title(f"{cap} {max(0, sec):.2f}s", fontsize=8)
            a.axis("off")
    fig.suptitle(f"{aid} {name_ko} ({label}) — 초록 = 추정 구간, 연빨강 = 뺀 일어나기, 회색 = 상자 높이 H, "
                 f"점선 = 문턱 θ", fontsize=12)
    SHEETS.mkdir(parents=True, exist_ok=True)
    out = SHEETS / f"{aid}_{label}.jpg"
    fig.savefig(out, dpi=80, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sheets", action="store_true", help="동작별 확인 그림도 만들기")
    ap.add_argument("--action", action="append", help="특정 동작만 그림 (예: A07)")
    args = ap.parse_args()

    files = sorted(SEQ.glob("*_sequence.csv"))
    if not files:
        sys.exit(f"Sequence 없음: {SEQ}")
    old = read_existing()
    rows, results = [], {}
    for f in files:
        df = pd.read_csv(f)
        res = estimate(df)
        cid = df["clip_id"].iloc[0]
        prev = old.get(cid, {})
        t = res["t"]
        row = dict(clip_id=cid, subject_id=df["subject_id"].iloc[0], action_id=df["action_id"].iloc[0],
                   view=df["view"].iloc[0], duration_sec=round(float(t[-1]), 2), theta=round(res["theta"], 3),
                   start_sec=round(float(res["start"]), 2), end_sec=round(float(res["end"]), 2),
                   end_sec_rule=round(float(res["end_rule"]), 2),
                   segment_sec=round(float(res["end_rule"] - res["start"]), 2), flags=res["flags"],
                   start_sec_final=prev.get("start_sec_final", ""), end_sec_final=prev.get("end_sec_final", ""),
                   segment_note=prev.get("segment_note", ""))
        rows.append(row)
        results[cid] = res

    TABLES.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\r\n")
        w.writeheader()
        w.writerows(rows)
    print(f"{OUT_CSV.relative_to(ROOT)}: {len(rows)}개")

    if args.sheets:
        classes = {r["action_id"]: r for r in csv.DictReader(CLASSES.open(encoding="utf-8-sig"))}
        for aid in sorted({r["action_id"] for r in rows}):
            if args.action and aid not in args.action:
                continue
            items = [(r, results[r["clip_id"]]) for r in rows if r["action_id"] == aid]
            c = classes[aid]
            print(make_sheet(aid, c["action_label"], c["name_ko"], items).relative_to(ROOT))


if __name__ == "__main__":
    main()
