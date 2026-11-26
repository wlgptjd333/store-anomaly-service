"""카메라 각도 증강 — 위·옆에서 찍은 것처럼 관절 좌표를 줄인다 (2026-11-26, 28번 실험).

생각은 팀 모델 비교(11/26)의 공동 1위 팀원 모델에서 가져왔다 (그 저장소에 증강 코드는 없어 여기서 새로 만듦).

어떻게: Frame 마다 골반 중심 기준 좌표(bn_*)에
  - yaw(옆으로 돈 카메라)   → x 를 cos(yaw) 배로 줄임 — 몸이 비스듬히 보이면 좌우 폭이 좁아짐
  - pitch(위에서 내려다봄) → y 를 cos(pitch) 배로 줄임 — 위에서 보면 키가 짧아 보임
  그다음 서비스와 같은 규칙으로 다시 정규화한다:
  - 몸 크기 = max(몸통 길이, 어깨 폭 / 0.70) (src/team_feature_utils.body_scale) → 그 값으로 다시 나눔
  - 팔꿈치·무릎 각도는 줄인 좌표로 다시 계산 (2D 각도는 한쪽 방향만 줄이면 바뀜)
  - 골반 화면 위치(hip_x_img, hip_y_img)는 영상 평균 기준으로 같은 배율만큼 움직임 폭을 줄임
한계: 진짜 카메라 회전은 몸 앞뒤(깊이)로 뻗은 팔·다리가 옆으로 보이게도 만든다 — 2D 만 있어 그건 흉내 내지 못한다.
"""
from __future__ import annotations

import numpy as np

from src.pose_utils import COCO_KEYPOINT_NAMES
from src.schema import BODY40
from src.team_feature_utils import BODY_SHOULDER_RATIO

KP = list(COCO_KEYPOINT_NAMES)
_I = {k: i for i, k in enumerate(KP)}
_ANG = {"left_elbow_angle": ("left_shoulder", "left_elbow", "left_wrist"),
        "right_elbow_angle": ("right_shoulder", "right_elbow", "right_wrist"),
        "left_knee_angle": ("left_hip", "left_knee", "left_ankle"),
        "right_knee_angle": ("right_hip", "right_knee", "right_ankle")}
_COORD = [BODY40.index(f"bn_{k}_{a}") for k in KP for a in "xy"]          # 0..33 (관절 순서 x, y)
_HX, _HY = BODY40.index("hip_x_img"), BODY40.index("hip_y_img")

YAW_MAX, PITCH_MAX = 60.0, 45.0          # 학습 증강 범위 (도). 무작위 U(0, 최대)


def joint_angles(kp: np.ndarray) -> dict[str, np.ndarray]:
    """kp (T, 17, 2) → 각도(도) 4개. 학습 feature_utils.calculate_angle 과 같은 식 (B 를 꼭짓점으로)."""
    out = {}
    for name, (a, b, c) in _ANG.items():
        ba, bc = kp[:, _I[a]] - kp[:, _I[b]], kp[:, _I[c]] - kp[:, _I[b]]
        cos = (ba * bc).sum(1) / np.maximum(np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1), 1e-8)
        out[name] = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    return out


def camera_view(t: np.ndarray, v: np.ndarray, rng=None, yaw: float | None = None, pitch: float | None = None):
    """(시간, (T, 40)) → 같은 모양. yaw·pitch(도)를 주지 않으면 rng 로 무작위."""
    if yaw is None:
        yaw = rng.uniform(0, YAW_MAX)
    if pitch is None:
        pitch = rng.uniform(0, PITCH_MAX)
    sx, sy = np.cos(np.radians(yaw)), np.cos(np.radians(pitch))
    out = v.copy()
    kp = v[:, _COORD].reshape(len(v), 17, 2) * np.array([sx, sy])
    hip = (kp[:, _I["left_hip"]] + kp[:, _I["right_hip"]]) / 2
    sh = (kp[:, _I["left_shoulder"]] + kp[:, _I["right_shoulder"]]) / 2
    torso = np.linalg.norm(sh - hip, axis=1)
    from_sh = np.linalg.norm(kp[:, _I["left_shoulder"]] - kp[:, _I["right_shoulder"]], axis=1) / BODY_SHOULDER_RATIO
    scale = np.maximum(np.maximum(torso, from_sh), 1e-3)
    kp = (kp - hip[:, None]) / scale[:, None, None]
    out[:, _COORD] = kp.reshape(len(v), -1)
    for name, val in joint_angles(kp).items():
        out[:, BODY40.index(name)] = val
    for j, s in ((_HX, sx), (_HY, sy)):
        m = v[:, j].mean()
        out[:, j] = m + (v[:, j] - m) * s
    return t, out
