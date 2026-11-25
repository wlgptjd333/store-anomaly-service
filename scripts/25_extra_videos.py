"""학습 목록 밖 영상(새 장소·새 카메라 구도)을 학습과 똑같은 방식으로 Sequence 로 만든다 (2026-11-24 밤).

왜: 11/24 S05 가 다른 구도(가로 1920×1080, 높은 곳에서 대각선)로 찍은 테스트 영상 5개를 서비스가 2/5 만 맞힘.
    같은 사람(S05)인데도 틀렸으니 원인은 카메라 구도. 이런 영상을 '새 구도 테스트(view D)'로 따로 평가하려면
    학습 데이터와 같은 03번(Frame 별 Pose·Feature) → 13번(whole 30단계) 계산이 필요하다.

입력: 영상 폴더 + 파일 이름 → 동작 번호 대응 (--labels "꺼내기=A12,밀치기=A05,...")
출력: reports/extra/<이름>/sequences/*.csv, reports/extra/<이름>/sequence_dataset_whole.csv (Git 제외 — reports/extra 는 .gitignore)
      clip_id = X_<사람>_<동작>_<번호>_<방향>, subject_id 는 --subject (기본 S05), view 는 --view (기본 D = 새 구도)

실행:
  .venv/bin/python scripts/25_extra_videos.py --src "/mnt/d/3팀드라이브/교과10 동작 영상/<사람 폴더>/테스트영상" --name s05_newcam
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import load_config, load_script, resolve_device  # noqa: E402
from src.schema import seq_columns  # noqa: E402



s03 = load_script("03_build_sequences.py")
s13 = load_script("13_build_sequence_dataset.py")
DEFAULT_LABELS = "꺼내기=A12,밀치기=A05,앞으로 눕기=A09,키오스크치기=A01,흔들기=A04"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="영상 폴더")
    ap.add_argument("--name", required=True, help="출력 이름 (reports/extra/<name>)")
    ap.add_argument("--labels", default=DEFAULT_LABELS, help="파일 이름(확장자 빼고)=동작번호, 쉼표로")
    ap.add_argument("--subject", default="S05")
    ap.add_argument("--view", default="D", help="D = 학습에 없던 새 구도")
    args = ap.parse_args()

    labels = dict(x.split("=") for x in args.labels.split(","))
    actions = {r["action_id"]: r["action_label"] for r in pd.read_csv(ROOT / "configs" / "action_classes.csv", encoding="utf-8-sig").to_dict("records")}
    cfg = load_config()
    base = json.loads(s13.CONFIG.read_text(encoding="utf-8"))
    steps, min_n, min_r = base["sequence_steps"], base["min_valid_frames"], base["min_valid_ratio"]
    model = YOLO(str(ROOT / cfg["pose_model"]) if (ROOT / cfg["pose_model"]).exists() else cfg["pose_model"])
    device = resolve_device(cfg["device"])

    out_dir = ROOT / "reports" / "extra" / args.name
    s03.VIDEOS = Path(args.src)
    s03.OUT_DIR = out_dir
    names = seq_columns(steps, s13.FRAME_FEATURES) + [f"mid_{f}" for f in s13.FRAME_FEATURES]
    rows, seen = [], {}
    for video in sorted(Path(args.src).glob("*.mp4")):
        aid = labels.get(video.stem)
        if aid is None:
            print(f"  건너뜀 (라벨 없음): {video.name}")
            continue
        seen[aid] = seen.get(aid, 0) + 1
        clip = {"clip_id": f"X_{args.subject}_{aid}_T{seen[aid]:02d}_{args.view}", "subject_id": args.subject,
                "action_id": aid, "action_label": actions.get(aid, ""), "view": args.view, "source_type": "RGB",
                "video_file": video.name, "take_id": f"T{seen[aid]:02d}"}
        _, fr = s03.build(clip, model, cfg, device, cfg["keypoint_confidence"], 0, False)
        df = pd.DataFrame(fr)
        t = pd.to_numeric(df["time_sec"], errors="coerce").to_numpy(float)
        vals = df[s13.FRAME_FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        valid = s13.ok(df["pose_ok"]) & s13.ok(df["feature_ok"]) & s13.ok(df["body_feature_ok"]) & np.isfinite(vals).all(1)
        n_valid = int(valid.sum())
        if n_valid < min_n or n_valid / max(len(df), 1) < min_r:
            print(f"  유효 Frame 부족: {video.name} ({n_valid}/{len(df)})")
            continue
        seq = s13.resample(vals[valid], t[valid], steps)
        mid = vals[valid][n_valid // 2]
        rows.append([clip[k] for k in s13.META] + [n_valid] + list(np.round(seq.reshape(-1), 5)) + list(np.round(mid, 5)))
        print(f"  {video.name} → {clip['clip_id']} ({n_valid}/{len(df)} Frame)")
    out = out_dir / "sequence_dataset_whole.csv"
    pd.DataFrame(rows, columns=s13.META + ["n_valid_frames"] + names).to_csv(out, index=False)
    print(f"{len(rows)}개 → {out}")


if __name__ == "__main__":
    main()
