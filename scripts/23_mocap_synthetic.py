"""VICON 3D 모션캡처 → 여러 가상 카메라로 2D 투영 → 학습 데이터 늘리기 (2026-11-24 밤).

왜: 실제 영상은 5명 × 17동작 × 3방향(정면·측면·뒤) 255개뿐이다. 교과12 에서 같은 17동작을 VICON 3D 로 찍었으니(1명),
    카메라를 원하는 각도·높이·거리에 놓고 2D Keypoint 를 "만들어" 볼 수 있다. 특히 실제로 안 찍은 대각선 각도.
    DESIGN_DECISIONS 9·10번에서 "다음에 해 볼 것"으로 남겨 둔 실험.

입력: 교과12 프로젝트(subject12_vicon_web)가 GLB 에서 뽑은 3D JSON 17개 (30fps, 73 Bone, 미터, Y-up).
      D001~D017 = 이 저장소 A01~A17 (같은 순서·같은 동작, 교과12 촬영 기록 문서로 확인). 원본은 Git 에 없음.
처리:
  1) 앞뒤 VICON 보정 자세(T-pose, 양손 사이 1.2 m 넘게)를 잘라냄 — 실제 영상에는 없는 자세
  2) Bone → COCO 17 관절 (어깨=Arm, 팔꿈치=ForeArm, 손목=Hand, 골반=UpLeg, 무릎=Leg, 발목=Foot,
     얼굴 5점은 Head·HeadEnd 와 몸 방향으로 근사)
  3) 가상 휴대폰 카메라(세로 1080×1920, 세로 화각 60°)를 사람 둘레 yaw 0~360°, 높이 1.0~1.6 m, 거리 3~4.5 m 에 두고 투영
  4) Keypoint 에 작은 잡음(몸 높이의 0.5%)을 더해 YOLO 떨림 흉내 → 실제와 같은 함수로 Frame Feature 40개 → 30단계
규칙: 4-fold 에서 연기자와 같은 사람(통일 S02 = 이 저장소 S03, 사용자 확인)이 평가 쪽이면 합성 데이터를 학습에서 뺀다.
      합성 데이터는 학습에만 넣고, 평가는 언제나 실제 영상만. Test S05 는 고른 뒤 확인만.

실행: .venv/bin/python scripts/23_mocap_synthetic.py [--json-dir 경로] [--test]
결과: reports/dataset/mocap_synthetic_whole.csv (Git 제외), reports/tables/mocap_synthetic_cv.csv (+ _test)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.feature_utils import selected_joint_angles  # noqa: E402
from src.team_feature_utils import normalize_by_body_scale, team03_features  # noqa: E402

_spec = importlib.util.spec_from_file_location("g", ROOT / "scripts" / "22_gpu_search.py")
g = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g)

JSON_DIR = Path.home() / "ai_vision/subject12_vicon_web/output/json"
OUT_CSV = ROOT / "reports" / "dataset" / "mocap_synthetic_whole.csv"
PERFORMER_SUBJECT = "S03"          # 이 저장소 번호 (통일 S02). 이 사람이 평가 쪽이면 합성 데이터 제외
STEPS = 30
W, H, VFOV = 1080, 1920, 60.0
COCO = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"]
BONE = {"left_shoulder": "LeftArm", "right_shoulder": "RightArm", "left_elbow": "LeftForeArm",
        "right_elbow": "RightForeArm", "left_wrist": "LeftHand", "right_wrist": "RightHand",
        "left_hip": "LeftUpLeg", "right_hip": "RightUpLeg", "left_knee": "LeftLeg", "right_knee": "RightLeg",
        "left_ankle": "LeftFoot", "right_ankle": "RightFoot"}
UP = np.array([0.0, 1.0, 0.0])


def load_clip(path: Path) -> tuple[str, np.ndarray]:
    """JSON → (A01.., (T, 17, 3) COCO 3D 점, T-pose 잘라낸 뒤)"""
    d = json.loads(path.read_text(encoding="utf-8"))
    num = int(re.search(r"_D(\d{3})_", path.name)[1])
    frames = []
    for fr in d["frames"]:
        j = {b["name"]: np.array([b["x"], b["y"], b["z"]]) for b in fr["joints"]}
        p = {k: j[v] for k, v in BONE.items()}
        right = p["right_shoulder"] - p["left_shoulder"]
        fwd = np.cross(UP, right)
        fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
        side = right / (np.linalg.norm(right) + 1e-9)
        head, head_end = j["Head"], j["HeadEnd"]
        mid = head + 0.45 * (head_end - head)
        p["nose"] = mid + 0.10 * fwd
        p["left_eye"], p["right_eye"] = p["nose"] + 0.03 * UP - 0.03 * side, p["nose"] + 0.03 * UP + 0.03 * side
        p["left_ear"], p["right_ear"] = mid - 0.075 * side, mid + 0.075 * side
        frames.append(np.stack([p[k] for k in COCO]))
    X = np.stack(frames)
    span = np.linalg.norm(X[:, 9] - X[:, 10], axis=1)          # 양손 사이
    arms_level = np.abs(X[:, 9, 1] - X[:, 5, 1]) < 0.15         # 손목이 어깨 높이
    tpose = (span > 1.2) & arms_level
    keep = np.where(~tpose)[0]
    a, b = keep[0], keep[-1]
    return f"A{num:02d}", X[a:b + 1]


def project(X: np.ndarray, yaw_deg: float, height: float, dist: float, rng) -> np.ndarray | None:
    """(T,17,3) → (T,17,2) px. 카메라는 클립 내내 고정, 사람 처음 방향 기준 yaw (0 = 정면)."""
    f0 = X[: min(15, len(X))]
    right = (f0[:, 6] - f0[:, 5]).mean(0)
    fwd = np.cross(UP, right)
    fwd[1] = 0
    fwd /= np.linalg.norm(fwd) + 1e-9
    side = np.cross(fwd, UP)
    yaw = np.radians(yaw_deg)
    view = np.cos(yaw) * fwd + np.sin(yaw) * side               # 사람 → 카메라 방향
    hip = X[:, 11:13].mean(1)
    target = hip.mean(0) + rng.uniform(-0.3, 0.3) * np.cross(view, UP)   # 화면 안 위치(hip_x_img)를 조금씩 다르게
    target[1] = 0.8
    for _ in range(4):
        cam = target + dist * view
        cam[1] = height
        fz = target - cam
        fz /= np.linalg.norm(fz)
        rx = np.cross(fz, UP)
        rx /= np.linalg.norm(rx)
        uy = np.cross(rx, fz)
        rel = X - cam
        z = rel @ fz
        if np.any(z < 0.5):
            dist += 1.0
            continue
        focal = (H / 2) / np.tan(np.radians(VFOV / 2))
        u = W / 2 + focal * (rel @ rx) / z
        v = H / 2 - focal * (rel @ uy) / z
        P = np.stack([u, v], -1)
        if P[..., 0].min() < 0 or P[..., 0].max() > W or P[..., 1].min() < 0 or P[..., 1].max() > H:
            dist += 0.75                                         # 몸이 화면 밖이면 뒤로
            continue
        body_h = np.ptp(P[..., 1], axis=1).mean()
        return P + rng.normal(0, 0.005 * body_h, P.shape)       # YOLO 떨림 흉내
    return None


def frame_values(xy: np.ndarray) -> np.ndarray | None:
    """서비스 pose_pipeline.frame_features 와 같은 계산 (Frame 하나, 40개)"""
    try:
        body, _, _ = normalize_by_body_scale(xy)
        ang = selected_joint_angles(xy)
    except (ValueError, ZeroDivisionError, FloatingPointError):
        return None
    team = team03_features(xy, np.ones(17), W, H, 0.5)
    vals = np.concatenate([np.asarray(body, float).reshape(-1), [float(ang[n]) for n in g.more.ANGLES],
                           [team["hip_x_img"], team["hip_y_img"]]])
    return vals if np.all(np.isfinite(vals)) else None


def resample(seq: np.ndarray, steps: int = STEPS) -> np.ndarray:
    t = np.linspace(0, 1, len(seq))
    tt = np.linspace(0, 1, steps)
    return np.stack([np.interp(tt, t, seq[:, k]) for k in range(seq.shape[1])], 1)


def build(json_dir: Path, views: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for path in sorted(json_dir.glob("*_3d.json")):
        action, X = load_clip(path)
        for k in range(views):
            yaw = (360 / views) * k + rng.uniform(-15, 15)
            P = project(X, yaw, rng.uniform(1.0, 1.6), rng.uniform(3.0, 4.5), rng)
            if P is None:
                continue
            vals = [frame_values(p) for p in P]
            ok = np.array([v is not None for v in vals])
            if ok.mean() < 0.8:
                continue
            seq = resample(np.stack([v for v in vals if v is not None]))
            row = {"clip_id": f"MOCAP_{action}_Y{int(yaw) % 360:03d}", "subject_id": "MOCAP", "action_id": action,
                   "view": "B" if 135 <= yaw % 360 <= 225 else ("F" if yaw % 360 <= 45 or yaw % 360 >= 315 else "S"),
                   "yaw": round(yaw % 360, 1)}
            row.update({f"t{i:02d}_{f}": seq[i, j] for i in range(STEPS) for j, f in enumerate(g.FRAME)})
            rows.append(row)
    return pd.DataFrame(rows)


def check_left_right(json_dir: Path) -> None:
    """정면 카메라에서 사람의 왼쪽 어깨는 화면 오른쪽(u 가 큼)에 와야 한다 (투영 방향 확인)"""
    _, X = load_clip(sorted(json_dir.glob("*_3d.json"))[0])
    P = project(X[:5], 0.0, 1.3, 3.5, np.random.default_rng(0))
    assert P is not None and P[0, 5, 0] > P[0, 6, 0], "투영 좌우가 뒤집힘"
    P = project(X[:5], 180.0, 1.3, 3.5, np.random.default_rng(0))
    assert P[0, 5, 0] < P[0, 6, 0], "뒤 카메라 좌우 확인 실패"


def fold_with_syn(cfg, tr, te, classes, seed, tag, syn, held_out):
    """22번 fold 와 같되, 학습에 합성 데이터를 더함 (연기자 본인이 평가 쪽이면 뺌)"""
    if syn is not None and PERFORMER_SUBJECT not in held_out:
        tr = pd.concat([tr, syn[tr.columns.intersection(syn.columns)]], ignore_index=True)
        tag = f"{tag}+syn{len(syn)}"
    return g.fold("", cfg, tr, te, classes, seed, tag)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-dir", type=Path, default=JSON_DIR)
    ap.add_argument("--views", default="4,8", help="클립마다 가상 카메라 수 (쉼표로 여러 설정)")
    ap.add_argument("--test", action="store_true", help="고른 뒤 확인: Train S01~S03 → Validation S04 / Test S05")
    args = ap.parse_args()
    if not args.json_dir.exists():
        raise SystemExit(f"3D JSON 폴더 없음: {args.json_dir} (교과12 subject12_vicon_web 에서 '전체 추출')")
    check_left_right(args.json_dir)
    cfg = g.CONFIGS["1D-CNN 폭 128 · 난수 3개"]           # 지금 서비스 모양
    df, _ = g.load(cfg["data"])
    classes = sorted(df.action_id.astype(str).unique())
    syns = {}
    for v in [int(x) for x in args.views.split(",")]:
        s = build(args.json_dir, v)
        syns[f"합성 {v}방향 × 17 ({len(s)}개)"] = s
        if v == max(int(x) for x in args.views.split(",")):
            OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
            s.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print({k: len(v) for k, v in syns.items()}, flush=True)
    settings = {"실제 영상만 (지금 서비스)": None, **syns}

    if args.test:
        rows = []
        tr = df[df.subject_id.isin(g.SPLIT["train"])]
        for name, syn in settings.items():
            for set_name, key in (("validation_S04", "validation"), ("test_S05", "test")):
                te = df[df.subject_id.isin(g.SPLIT[key])]
                sc = [g.score(te, fold_with_syn(cfg, tr, te, classes, s, set_name, syn, g.SPLIT[key]).argmax(1),
                              classes)[0] for s in g.SEEDS]
                rows.append({"set": set_name, "config": name,
                             **{k: round(float(np.mean([x[k] for x in sc])), 1) for k in sc[0]}})
                print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(g.TABLES / "mocap_synthetic_test.csv", index=False, encoding="utf-8-sig")
        return

    dev = g.SPLIT["train"] + g.SPLIT["validation"]
    rows, per_subj = [], []
    for name, syn in settings.items():
        t0, scs = time.time(), []
        for seed in g.SEEDS:
            parts = []
            for s in dev:
                tr, te = df[df.subject_id.isin([x for x in dev if x != s])], df[df.subject_id == s]
                parts.append((te, fold_with_syn(cfg, tr, te, classes, seed, s, syn, [s]).argmax(1)))
            sc, P = g.score(pd.concat([t for t, _ in parts]), np.concatenate([p for _, p in parts]), classes)
            scs.append(sc)
            per_subj.append(P.groupby("subject_id").correct.mean().mul(100).rename(name).to_frame().T.assign(seed=seed))
        r = {"config": name, **{k: round(float(np.mean([x[k] for x in scs])), 1) for k in scs[0]},
             "acc_all_std": round(float(np.std([x["acc_all"] for x in scs])), 1), "sec": round(time.time() - t0)}
        rows.append(r)
        print(f"  {name:28s} 전체 {r['acc_all']:5.1f} ± {r['acc_all_std']:3.1f} | 정면·측면 {r['acc_FS']:5.1f}"
              f" 뒤 {r['acc_B']:5.1f} | 가장 낮은 사람 {r['subj_min']:5.1f} | {r['sec']}s", flush=True)
    pd.DataFrame(rows).to_csv(g.TABLES / "mocap_synthetic_cv.csv", index=False, encoding="utf-8-sig")
    ps = pd.concat(per_subj).groupby(level=0).mean(numeric_only=True).drop(columns="seed").round(1)
    print("사람별 (평가 사람, 4-fold):\n" + ps.to_string(), flush=True)
    ps.to_csv(g.TABLES / "mocap_synthetic_per_subject.csv", encoding="utf-8-sig")


if __name__ == "__main__":
    main()
