"""학습·서비스가 같이 쓰는 Feature 이름과 좌우 반전 — 한 곳에서만 정의한다.

서비스(store-anomaly-service ai_service/src/schema.py)는 이 파일을 그대로 복사해 쓴다 (scripts/update_model.sh 가 같은지 확인).
"""
from __future__ import annotations

import numpy as np

from src.pose_utils import COCO_KEYPOINT_NAMES

KP = list(COCO_KEYPOINT_NAMES)                                                  # 관절 17개
ANGLES = ["left_elbow_angle", "right_elbow_angle", "left_knee_angle", "right_knee_angle"]
POS = ["hip_x_img", "hip_y_img"]                                                # 골반 화면 위치 (0~1)
BODY40 = [f"bn_{k}_{a}" for k in KP for a in "xy"] + ANGLES + POS             # Frame 하나의 Feature 40개


def seq_columns(steps: int, frame: list[str] = BODY40) -> list[str]:
    """30단계 Sequence 열 이름: t00_bn_nose_x … t29_hip_y_img"""
    return [f"t{i:02d}_{f}" for i in range(steps) for f in frame]


def swap_side(name: str) -> str:
    """left_ ↔ right_ (좌우 반전 때 짝 관절)"""
    return name.replace("left_", "TMP_").replace("right_", "left_").replace("TMP_", "right_")


_FLIP_SRC = [BODY40.index(swap_side(f)) for f in BODY40]
_FLIP_NEG = np.array([f.startswith("bn_") and f.endswith("_x") for f in BODY40])
_FLIP_ONE_MINUS = np.array([f == "hip_x_img" for f in BODY40])


def flip_matrix(X: np.ndarray, steps: int) -> np.ndarray:
    """(n, steps × 40) 좌우 반전: 왼쪽·오른쪽 관절과 각도를 바꾸고, x → -x, 골반 화면 x → 1 - x"""
    seq = X.reshape(len(X), steps, len(BODY40))[:, :, _FLIP_SRC]
    out = np.where(_FLIP_NEG, -seq, seq)
    out = np.where(_FLIP_ONE_MINUS, 1 - seq, out)
    return out.reshape(len(X), -1)
