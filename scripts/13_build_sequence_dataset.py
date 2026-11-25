"""고정 길이 Sequence Dataset — 수업 6일차 Sequence Resampling.

영상마다 길이(Frame 수)가 다르므로, 정해진 구간을 시간 0~1 로 펴서 30단계로 다시 뽑는다
(앞 30 Frame 을 자르는 게 아니라 구간 전체를 고르게 30개로 — 수업 resample_sequence 와 같은 np.interp 방식).

입력
  configs/approved_clip_manifest.csv   Human QA 를 통과한 170개 + 동작 구간
  configs/baseline_config.json         단계 수, 최소 유효 Frame 기준
  reports/sequences/*_sequence.csv     Frame별 Keypoint (03번)

처리
  1) 입력 점검: 파일 존재, 구간 안 유효 Frame 수·비율 (수업 6일차 입력 확인 단계)
  2) 창(window) 두 가지로 자름: segment(06번 동작 구간) / whole(영상 전체, 수업 방식)
  3) 유효 Frame(pose·feature·body_feature OK, 값이 모두 숫자)만 남기고 시간축 기준 30단계로 보간
  4) Frame 하나의 값(74개)을 모두 저장 → 14번에서 Feature 조합(course38, body40 …)을 골라 씀
     - 수업 38-D: 어깨 폭 정규화 17관절 x,y + 각도 4
     - 3팀: 몸 크기 정규화 bn_ 17관절 x,y + 골반 화면 위치(hip_x_img, hip_y_img)
     - 가운데 Frame 값(mid_*)도 함께 저장 → Static vs Sequence 비교용

출력
  reports/dataset/sequence_dataset_{segment,whole}.csv   (자동 생성, Git 제외)
  reports/tables/sequence_dataset_check.csv              (입력 점검 결과)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.schema import ANGLES, KP, POS  # noqa: E402
APPROVED = ROOT / "configs" / "approved_clip_manifest.csv"
CONFIG = ROOT / "configs" / "baseline_config.json"
OUT = ROOT / "reports" / "dataset"
CHECK = ROOT / "reports" / "tables" / "sequence_dataset_check.csv"

COURSE = [f"{k}_{a}" for k in KP for a in "xy"]
BODY = [f"bn_{k}_{a}" for k in KP for a in "xy"]
FRAME_FEATURES = COURSE + BODY + ANGLES + POS  # 74
META = ["clip_id", "subject_id", "action_id", "action_label", "view", "take_id"]


def ok(s: pd.Series) -> np.ndarray:
    return s.astype(str).str.strip().str.lower().isin(["1", "true", "1.0"]).to_numpy()


def resample(values: np.ndarray, times: np.ndarray, steps: int) -> np.ndarray:
    """(T, F) → (steps, F). 시간을 0~1 로 펴고 고른 간격으로 보간 (수업 resample_sequence 와 같음)."""
    span = times[-1] - times[0]
    x = (times - times[0]) / span
    tx = np.linspace(0.0, 1.0, steps)
    return np.stack([np.interp(tx, x, values[:, j]) for j in range(values.shape[1])], 1)


def smooth(values: np.ndarray, k: int) -> np.ndarray:
    """유효 Frame 만 모은 (T, F) 를 시간 방향 이동 평균 (k Frame, 가장자리는 있는 만큼만). k<=1 이면 그대로."""
    if k <= 1 or len(values) < 2:
        return values
    out = np.empty_like(values)
    h = k // 2
    for i in range(len(values)):
        out[i] = values[max(0, i - h): i + h + 1].mean(0)
    return out


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-dir", default=None, help="03번 결과 폴더 (기본: Manifest 의 sequence_file 경로)")
    ap.add_argument("--out-dir", default=None, help="출력 폴더 (기본: reports/dataset)")
    ap.add_argument("--steps", type=int, default=None, help="단계 수 (기본: baseline_config 30)")
    ap.add_argument("--smooth", type=int, default=0, help="보간 전 이동 평균 Frame 수 (0 = 안 함)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    steps, min_n, min_r = cfg["sequence_steps"], cfg["min_valid_frames"], cfg["min_valid_ratio"]
    steps = args.steps or steps
    out_dir = (ROOT / args.out_dir) if args.out_dir else OUT
    other = bool(args.seq_dir or args.out_dir or args.steps or args.smooth)   # 실험용이면 점검표는 안 덮어씀
    clips = list(csv.DictReader(APPROVED.open(encoding="utf-8-sig")))
    names = [f"t{i:02d}_{f}" for i in range(steps) for f in FRAME_FEATURES] + [f"mid_{f}" for f in FRAME_FEATURES]
    rows = {"segment": [], "whole": []}
    checks = []
    for c in clips:
        path = (ROOT / args.seq_dir / Path(c["sequence_file"]).name) if args.seq_dir else ROOT / c["sequence_file"]
        chk = {"clip_id": c["clip_id"], "file_exists": path.exists()}
        if not path.exists():
            checks.append({**chk, "status": "MISSING"})
            continue
        df = pd.read_csv(path)
        t = pd.to_numeric(df["time_sec"], errors="coerce").to_numpy(float)
        vals = df[FRAME_FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        valid = ok(df["pose_ok"]) & ok(df["feature_ok"]) & ok(df["body_feature_ok"]) & np.isfinite(vals).all(1)
        windows = {"segment": (t >= float(c["seg_start"])) & (t <= float(c["seg_end"])),
                   "whole": np.ones(len(df), bool)}
        status = "OK"
        for w, m in windows.items():
            n_all, n_valid = int(m.sum()), int((m & valid).sum())
            chk[f"{w}_frames"], chk[f"{w}_valid"] = n_all, n_valid
            chk[f"{w}_valid_ratio"] = round(n_valid / n_all, 4) if n_all else 0.0
            if n_valid < min_n or n_valid / max(n_all, 1) < min_r:
                status = f"LOW_{w.upper()}"
                continue
            sel = m & valid
            seq = resample(smooth(vals[sel], args.smooth), t[sel], steps)
            mid = vals[sel][int(sel.sum()) // 2]
            rows[w].append([c[k] for k in META] + [n_valid] + list(np.round(seq.reshape(-1), 5)) + list(np.round(mid, 5)))
        chk["segment_sec"] = round(float(c["seg_end"]) - float(c["seg_start"]), 2)
        checks.append({**chk, "status": status})

    out_dir.mkdir(parents=True, exist_ok=True)
    for w, r in rows.items():
        pd.DataFrame(r, columns=META + ["n_valid_frames"] + names).to_csv(out_dir / f"sequence_dataset_{w}.csv", index=False)
        print(f"{w}: {len(r)} clips × {len(names)} features ({steps} × {len(FRAME_FEATURES)} + 가운데 Frame {len(FRAME_FEATURES)})")
    ck = pd.DataFrame(checks)
    if not other:
        ck.to_csv(CHECK, index=False, encoding="utf-8-sig")
    print(ck["status"].value_counts().to_string())
    print("segment 유효 Frame: 최소 %d, 중앙값 %d, 유효 비율 최소 %.3f" % (
        ck["segment_valid"].min(), ck["segment_valid"].median(), ck["segment_valid_ratio"].min()))


if __name__ == "__main__":
    main()
