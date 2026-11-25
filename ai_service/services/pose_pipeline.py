"""영상 → YOLO Pose + ByteTrack → 사람(Track ID)별 Frame Feature 목록.

Frame 하나의 Feature 는 학습 때(training/scripts/03_build_sequences.py)와 같은 계산을 쓴다.
  body40 = 몸 크기 정규화 17관절 x,y (34) + 팔꿈치·무릎 각도 (4) + 골반 화면 위치 (2)
유효 Frame 규칙도 학습 때와 같다:
  Pose 있음 · 어깨 폭 정규화 성공(feature_ok) · 몸 크기 정규화 성공(body_feature_ok) · 40개 값이 모두 숫자

학습 때와 다른 점 하나: 학습 영상에는 사람이 한 명이라 가장 확실한 사람 1명만 썼다.
서비스에서는 ByteTrack 으로 사람마다 Track ID 를 붙여 사람별로 따로 모은다 (교과10 8일차 방식).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from src.feature_utils import normalize_by_shoulder_width, selected_joint_angles
from src.schema import ANGLES, BODY40          # 학습과 같은 파일 (ai_service/src → training/src 링크)
from src.team_feature_utils import normalize_by_body_scale, team03_features

# YOLO 모델·Tracker 는 요청끼리 공유하면 Track ID 가 섞이므로 영상 한 건씩 처리한다
_TRACK_LOCK = threading.Lock()


class VideoReadError(Exception):
    """영상 파일을 열 수 없거나 Frame 이 없음 → 400 응답"""


@dataclass
class TrackData:
    """사람 한 명의 Frame 별 값 (업로드 분석은 영상 전체, 실시간 보기는 최근 8초만)"""
    times: list[float] = field(default_factory=list)
    features: list[np.ndarray] = field(default_factory=list)
    valid: list[bool] = field(default_factory=list)
    keypoint_conf: list[float] = field(default_factory=list)

    def add(self, t: float, person: Person) -> None:
        self.times.append(t)
        self.features.append(person.values)
        self.valid.append(person.valid)
        self.keypoint_conf.append(float(np.mean(person.conf)))


@dataclass
class Person:
    """Frame 하나에서 찾은 사람 한 명"""
    track_id: int
    xy: np.ndarray            # 17×2 px
    conf: np.ndarray          # 17
    box: np.ndarray           # x1,y1,x2,y2 (실시간 화면에 그릴 때)
    values: np.ndarray        # Frame Feature 40
    valid: bool


@dataclass
class VideoResult:
    tracks: dict[int, TrackData]
    frames: int
    fps: float
    width: int
    height: int
    pose_seconds: float

    @property
    def seconds(self) -> float:
        return round(self.frames / self.fps, 2) if self.fps else 0.0


def frame_features(xy: np.ndarray, conf: np.ndarray, img_w: int, img_h: int, kp_conf: float):
    """한 사람의 Keypoint 17×2(px) → (40개 값, 유효 여부)"""
    values = np.full(len(BODY40), np.nan, dtype=np.float64)
    try:
        body_norm, _, _ = normalize_by_body_scale(xy)
        body_ok = bool(np.all(np.isfinite(body_norm)))
    except ValueError:
        return values, False
    try:
        shoulder_norm, _, _ = normalize_by_shoulder_width(xy)
        angles = selected_joint_angles(xy)
    except (ValueError, ZeroDivisionError, FloatingPointError):
        return values, False
    angle_vals = [float(angles[n]) for n in ANGLES]
    feature_ok = bool(np.all(np.isfinite(shoulder_norm)) and np.all(np.isfinite(angle_vals)))
    team = team03_features(xy, conf, img_w, img_h, kp_conf)
    values[:34] = np.round(body_norm.reshape(-1), 6)
    values[34:38] = np.round(angle_vals, 6)
    values[38] = team["hip_x_img"]
    values[39] = team["hip_y_img"]
    return values, body_ok and feature_ok and bool(np.all(np.isfinite(values)))


def track_people(pose_model, frame: np.ndarray, cfg: dict, device, persist: bool = True) -> list[Person]:
    """Frame 하나 → YOLO Pose + ByteTrack → 사람별 Feature. 업로드 분석과 실시간 보기가 같이 쓴다 (규칙이 갈라지지 않게)."""
    h, w = frame.shape[:2]
    res = pose_model.track(source=frame, persist=persist, tracker=cfg["tracker"], imgsz=cfg["image_size"],
                           conf=cfg["person_confidence"], device=device, verbose=False)[0]
    boxes, kps = res.boxes, res.keypoints
    if boxes is None or boxes.id is None or kps is None or kps.xy is None:
        return []
    ids = boxes.id.int().cpu().tolist()
    xys = kps.xy.cpu().numpy()
    confs = kps.conf.cpu().numpy() if kps.conf is not None else np.ones((len(ids), 17), np.float32)
    people = []
    for tid, xy, conf, box in zip(ids, xys, confs, boxes.xyxy.cpu().numpy(), strict=True):
        values, valid = frame_features(xy, conf, w, h, cfg["keypoint_confidence"])
        people.append(Person(int(tid), xy, conf, box, values, valid))
    return people


def extract_tracks(video_path: str, pose_model, cfg: dict, device) -> VideoResult:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoReadError("영상 파일을 열 수 없습니다. MP4 등 정상적인 영상인지 확인하세요.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if not fps or fps != fps or fps > 240:
        fps = 30.0
    max_frames = int(cfg["max_video_seconds"] * fps)
    # 60fps 영상은 2 Frame 마다 1개만 처리 (30단계로 보간하므로 결과 영향은 작고 처리 시간은 절반)
    step = max(1, round(fps / cfg.get("max_process_fps", 30)))
    tracks: dict[int, TrackData] = {}
    n, used, w, h = 0, 0, 0, 0
    t0 = time.perf_counter()
    with _TRACK_LOCK:
        while n < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            n += 1
            if (n - 1) % step:
                continue
            used += 1
            h, w = frame.shape[:2]
            t = (n - 1) / fps
            for person in track_people(pose_model, frame, cfg, device, persist=used > 1):
                tracks.setdefault(person.track_id, TrackData()).add(t, person)
    cap.release()
    if n == 0:
        raise VideoReadError("영상에서 Frame 을 읽지 못했습니다. 파일이 손상됐거나 지원하지 않는 형식입니다.")
    return VideoResult(tracks=tracks, frames=n, fps=round(fps, 3), width=w, height=h,
                       pose_seconds=round(time.perf_counter() - t0, 2))
