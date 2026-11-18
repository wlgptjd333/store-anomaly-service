"""정규화 방식 비교 — 픽셀 / 이미지 기준(5일차) / 어깨 폭(3일차) / 몸 크기(3팀) / 몸 크기 + 화면 위치.

YOLO 를 다시 돌리지 않고 Sequence CSV 의 원본 Keypoint 로 다섯 가지 좌표를 만든 뒤 세 가지로 비교한다.
  1) 크기·위치 차이를 얼마나 없애나: 동작 전 서 있는 자세(처음 0.5초)의 "키"와 "골반 위치"가 영상마다 얼마나 다른가
  2) 값이 튀는가: 관절 좌표가 기준 길이의 5배를 넘는 Frame 비율 (측면·누운 자세에서 기준 길이가 0 에 가까워질 때)
  3) 동작을 얼마나 잘 구분하나: 동작 구간의 좌표 통계(평균·표준편차·최소·최대)로 Random Forest 를 학습,
     사람 단위 Leave-One-Subject-Out 5-fold 정확도 (같은 사람은 학습과 시험에 동시에 들어가지 않음)

출력
  reports/tables/normalization_compare.csv     방법별 지표
  reports/tables/normalization_loso.csv        방법 × fold 정확도
  docs/images/normalization_overlay.png        같은 "서 있는 자세"를 방법별로 겹쳐 그린 그림 (얼굴 없음)
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEQ = ROOT / "reports" / "sequences"
APPROVED = ROOT / "configs" / "approved_clip_manifest.csv"
TABLES = ROOT / "reports" / "tables"
FIG = ROOT / "docs" / "images" / "normalization_overlay.png"

JOINTS = ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
          "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
EDGES = [(0, 2), (2, 4), (1, 3), (3, 5), (0, 1), (6, 7), (0, 6), (1, 7), (6, 8), (8, 10), (7, 9), (9, 11)]
METHODS = {
    "pixel": "픽셀 그대로",
    "image": "이미지 기준 (x/W, y/H) — 5일차",
    "shoulder": "골반 중심·어깨 폭 — 3일차 38-D",
    "body": "골반 중심·몸 크기 — 3팀",
    "body_pos": "몸 크기 + 골반 화면 위치",
}
BASE_SEC = 0.5
SEED = 0


def coords(df: pd.DataFrame, method: str) -> np.ndarray:
    """(T, 24) — method 별 관절 12개 좌표. body_pos 는 (T, 26)."""
    num = lambda c: pd.to_numeric(df[c], errors="coerce").to_numpy(float)  # noqa: E731
    if method in ("pixel", "image"):
        xs = np.stack([num(f"raw_{j}_x") for j in JOINTS], 1)
        ys = np.stack([num(f"raw_{j}_y") for j in JOINTS], 1)
        if method == "image":
            xs, ys = xs / num("img_w")[:, None], ys / num("img_h")[:, None]
    else:
        pre = {"shoulder": "", "body": "bn_", "body_pos": "bn_"}[method]
        xs = np.stack([num(f"{pre}{j}_x") for j in JOINTS], 1)
        ys = np.stack([num(f"{pre}{j}_y") for j in JOINTS], 1)
    out = np.stack([xs, ys], 2).reshape(len(df), -1)
    if method == "body_pos":
        hip = np.stack([num("hip_center_x") / num("img_w"), num("hip_center_y") / num("img_h")], 1)
        out = np.hstack([out, hip])
    return out


def base_shape(P: np.ndarray, n: int) -> tuple[float, float, float]:
    """처음 n Frame 의 (키, 골반 x, 골반 y) — 키 = 어깨~발목 세로 길이."""
    J = P[:n, :24].reshape(-1, 12, 2)
    height = np.nanmedian(np.nanmean(J[:, 10:12, 1], 1) - np.nanmean(J[:, 0:2, 1], 1))
    hip = np.nanmedian(np.nanmean(J[:, 6:8, :], 1), 0)
    return float(height), float(hip[0]), float(hip[1])


def clip_features(P: np.ndarray) -> np.ndarray:
    P = pd.DataFrame(P).interpolate(limit_direction="both").to_numpy()
    return np.concatenate([np.nanmean(P, 0), np.nanstd(P, 0), np.nanmin(P, 0), np.nanmax(P, 0)])


def setup_font() -> None:
    f = Path("/mnt/c/Windows/Fonts/malgun.ttf")
    if f.exists():
        font_manager.fontManager.addfont(str(f))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(f)).get_name()
    plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    clips = list(csv.DictReader(APPROVED.open(encoding="utf-8-sig")))
    data = {m: {"X": [], "shape": [], "blow": [], "base": []} for m in METHODS}
    meta = []
    for c in clips:
        df = pd.read_csv(ROOT / c["sequence_file"])
        t = pd.to_numeric(df["time_sec"], errors="coerce").to_numpy()
        step = float(np.median(np.diff(t)))
        n_base = max(3, int(round(BASE_SEC / step)))
        seg = (t >= float(c["seg_start"])) & (t <= float(c["seg_end"]))
        orient = "landscape" if int(df["img_w"].iloc[0]) > int(df["img_h"].iloc[0]) else "portrait"
        meta.append({"clip_id": c["clip_id"], "subject_id": c["subject_id"], "action_id": c["action_id"],
                     "view": c["view"], "orient": orient})
        for m in METHODS:
            P = coords(df, m)
            data[m]["X"].append(clip_features(P[seg]))
            data[m]["shape"].append(base_shape(P, n_base))
            data[m]["base"].append(P[:n_base, :24].reshape(-1, 12, 2))
            body = P[seg, :24]
            data[m]["blow"].append(np.nanmean(np.nanmax(np.abs(body), 1) > 5) if m in ("shoulder", "body") else np.nan)
    meta = pd.DataFrame(meta)
    y = meta["action_id"].to_numpy()
    subj = meta["subject_id"].to_numpy()

    summary, folds = [], []
    for m, name in METHODS.items():
        X = np.nan_to_num(np.array(data[m]["X"]), nan=0.0)
        shape = np.array(data[m]["shape"])
        accs, f1s = [], []
        for s in sorted(set(subj)):
            tr, te = subj != s, subj == s
            rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1).fit(X[tr], y[tr])
            p = rf.predict(X[te])
            accs.append(accuracy_score(y[te], p))
            f1s.append(f1_score(y[te], p, average="macro"))
            folds.append({"method": m, "test_subject": s, "accuracy": round(accs[-1], 4), "macro_f1": round(f1s[-1], 4)})
        h = shape[:, 0]
        blow = np.array(data[m]["blow"], float)
        side = (meta["view"] == "S").to_numpy()
        summary.append({
            "method": m, "name": name,
            "standing_height_cv": round(float(np.nanstd(h) / abs(np.nanmean(h))), 4),
            "hip_position_std": round(float(np.nanmean(np.nanstd(shape[:, 1:3], 0))), 4),
            "hip_position_std_rel": round(float(np.nanmean(np.nanstd(shape[:, 1:3], 0)) / abs(np.nanmean(h))), 4),
            "blowup_ratio_side": round(float(np.nanmean(blow[side])), 4) if m in ("shoulder", "body") else "",
            "loso_accuracy_mean": round(float(np.mean(accs)), 4), "loso_accuracy_std": round(float(np.std(accs)), 4),
            "loso_macro_f1_mean": round(float(np.mean(f1s)), 4),
            "portrait_height_mean": round(float(np.nanmean(h[(meta.orient == "portrait").to_numpy()])), 4),
            "landscape_height_mean": round(float(np.nanmean(h[(meta.orient == "landscape").to_numpy()])), 4),
        })
    pd.DataFrame(summary).to_csv(TABLES / "normalization_compare.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(folds).to_csv(TABLES / "normalization_loso.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(summary).drop(columns="name").to_string(index=False))

    # 그림: 정면 영상들의 "서 있는 자세"를 방법별로 겹쳐 그림 (사람마다 색, 세로 영상 F + 가로 영상 S)
    setup_font()
    pick = meta[(meta.view == "F") & meta.action_id.isin(["A01", "A05", "A11"])].index
    colors = {"S01": "#2f6fd6", "S02": "#e8833a", "S03": "#3a9e5c", "S04": "#c23b7a", "S05": "#7a5cc2"}
    show = ["pixel", "image", "shoulder", "body"]
    fig, axes = plt.subplots(1, 4, figsize=(17, 5.2))
    for ax, m in zip(axes, show):
        for i in pick:
            J = np.nanmedian(data[m]["base"][i], 0)
            ls = "--" if meta.orient[i] == "landscape" else "-"
            for a, b in EDGES:
                ax.plot(J[[a, b], 0], J[[a, b], 1], ls, color=colors[meta.subject_id[i]], lw=1.3, alpha=0.75)
        ax.invert_yaxis()
        ax.set_aspect("equal", adjustable="datalim")
        r = next(s for s in summary if s["method"] == m)
        ax.set_title(f"{METHODS[m]}\n키 변동 {r['standing_height_cv']:.0%} · 위치 변동(키 대비) {r['hip_position_std_rel']:.2f}",
                     fontsize=10)
        ax.tick_params(labelsize=7)
    for s, c in colors.items():
        axes[0].plot([], [], color=c, label=s)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("동작 전 서 있는 자세 (정면 A01·A05·A11 의 처음 0.5초, 5명 15개) — 정규화 방식마다 사람·영상 차이가 얼마나 남나\n"
                 "제목의 변동 수치는 정면·측면 170개 전체 기준", fontsize=11)
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=110, facecolor="white", bbox_inches="tight")
    print(FIG.relative_to(ROOT))


if __name__ == "__main__":
    main()
