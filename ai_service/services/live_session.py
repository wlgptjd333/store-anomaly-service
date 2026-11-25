"""실시간 화면: 카메라 → YOLO Pose + ByteTrack → 사람별 Skeleton 을 그린 JPEG + 최근 8초 기준 동작 예측.

저장(8초 수집 → MP4 → Celery)과 따로 도는 '보기용' 경로다. 결과는 DB 에 남기지 않는다.

스레드 두 개로 나눈 이유
  - 받기(grabber): 휴대폰 스트림을 쉬지 않고 읽어 '가장 최근 Frame' 하나만 들고 있는다.
    읽기를 멈추면 OpenCV 버퍼에 Frame 이 쌓여 화면이 몇 초씩 늦어진다.
  - 처리(worker): 가장 최근 Frame 만 골라 Pose 를 돌린다. CPU 가 느리면 Frame 을 건너뛰지, 밀리지 않는다.

예측 규칙은 업로드 분석과 같은 predict_track 을 쓴다. 사람마다 최근 window_seconds(8초)만 들고 있다가
live_predict_every_seconds(1초)마다 그 구간을 창 하나로 예측한다 → 학습(whole, 30단계)과 같은 모양.

Pose 모델은 세션마다 따로 불러온다. 업로드 분석과 같은 YOLO 객체를 쓰면 ByteTrack 상태(Track ID)가 섞인다.
"""
from __future__ import annotations

import bisect
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from services.action_inference import predict_track
from services.model_loader import APP_ROOT
from services.pose_pipeline import Person, TrackData, track_people
from src.pose_utils import COCO_SKELETON

# 사람(Track ID)마다 다른 색 (BGR)
PALETTE = [(255, 140, 0), (60, 200, 60), (40, 40, 230), (230, 60, 200), (0, 200, 230), (200, 200, 40),
           (140, 80, 255), (80, 160, 255)]


# predict_track 결과 중 실시간 화면이 쓰는 것 (구간별 windows 등은 저장 분석에서만)
LIVE_KEYS = ("track_seconds", "action_id", "action_label", "action_name_ko", "category", "confidence",
             "uncertain", "top3", "valid_frames")

class LiveBusy(Exception):
    """다른 카메라로 실시간 보기가 이미 돌고 있음 (CPU 하나로 두 개를 돌리면 둘 다 느려진다)"""


@dataclass
class _Track(TrackData):
    """실시간: TrackData + 처음·마지막으로 보인 시각, 최근 window 만 남기기"""
    first_seen: float = 0.0
    last_seen: float = 0.0

    def trim(self, keep_from: float) -> None:
        i = bisect.bisect_left(self.times, keep_from)      # times 는 늘 오름차순
        if i:
            for lst in (self.times, self.features, self.valid, self.keypoint_conf):
                del lst[:i]

    def as_track_data(self) -> TrackData:
        return TrackData(times=list(self.times), features=list(self.features), valid=list(self.valid),
                         keypoint_conf=list(self.keypoint_conf))


ROTATE = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def rotate(frame: np.ndarray, degrees: int) -> np.ndarray:
    """휴대폰을 세로로 들어도 IP Webcam 은 가로 영상을 보낸다 → 사람이 누운 것처럼 보이면 모델이 눕기 동작으로 답한다."""
    code = ROTATE.get(degrees)
    return cv2.rotate(frame, code) if code is not None else frame


class LiveSession:
    def __init__(self, camera: str, url: str, bundle, device, rotation: int = 0):
        from ultralytics import YOLO
        self.id = str(uuid.uuid4())
        self.camera, self.url, self.rotation = camera, url, rotation
        # '이 장면 저장'용: 최근 몇 초의 원본 Frame(회전 후, Skeleton 없음)을 JPEG 로 들고 있는다
        self._clip: deque[tuple[float, bytes]] = deque()
        self._clip_lock = threading.Lock()
        self._reset = False
        self.bundle, self.cfg, self.device = bundle, bundle.config, device
        self.pose = YOLO(str(APP_ROOT / bundle.config["pose_model"]))
        self.state, self.message = "connecting", ""
        self.started = self.last_access = time.monotonic()
        self.frame_size = (0, 0)
        self.fps = 0.0
        self.tracks: dict[int, _Track] = {}
        self.results: dict[int, dict] = {}
        self._raw: tuple[float, np.ndarray] | None = None
        self._raw_seq = 0
        self.jpeg: bytes | None = None
        self.jpeg_seq = 0
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._threads = [threading.Thread(target=self._grab, daemon=True, name=f"live-grab-{self.id[:8]}"),
                         threading.Thread(target=self._work, daemon=True, name=f"live-work-{self.id[:8]}")]
        for t in self._threads:
            t.start()

    # ── 상태 ──
    @property
    def alive(self) -> bool:
        return not self._stop.is_set()

    def touch(self) -> None:
        self.last_access = time.monotonic()

    def stop(self, reason: str = "") -> None:
        if self.state != "error":
            self.state, self.message = "stopped", reason or self.message
        self._stop.set()
        with self._cond:
            self._cond.notify_all()

    def _fail(self, message: str) -> None:
        self.state, self.message = "error", message
        self._stop.set()
        with self._cond:
            self._cond.notify_all()

    def snapshot(self) -> dict:
        now = time.monotonic()
        results = dict(self.results)                       # 처리 스레드가 바꾸는 중이어도 안전하게 복사
        people = [{"track_id": tid, **results[tid]} for tid in sorted(results)]
        return {"session_id": self.id, "camera": self.camera, "rotation": self.rotation,
                "state": self.state, "message": self.message,
                "fps": round(self.fps, 1), "width": self.frame_size[0], "height": self.frame_size[1],
                "running_seconds": round(now - self.started, 1), "window_seconds": self.cfg["window_seconds"],
                "uncertain_below": self.cfg["uncertain_below"], "people": people}

    # ── 받기 스레드 ──
    def _grab(self) -> None:
        cap = cv2.VideoCapture(self.url)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.cfg["live_connect_timeout_seconds"] * 1000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
        if not cap.isOpened():
            self._fail(f"카메라 '{self.camera}'에 연결하지 못했습니다. 휴대폰 IP Webcam 앱이 켜져 있고 같은 네트워크인지 확인하세요.")
            return
        misses = 0
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    misses += 1
                    if misses > 30:
                        self._fail(f"카메라 '{self.camera}' 연결이 끊겼습니다.")
                        return
                    time.sleep(0.05)
                    continue
                misses = 0
                frame = rotate(frame, self.rotation)
                t = time.monotonic()
                with self._cond:
                    self._raw = (t, frame)
                    self._raw_seq += 1
                    self._cond.notify_all()
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if ok:
                    keep = self.cfg["live_clip_keep_seconds"]
                    with self._clip_lock:
                        self._clip.append((t, buf.tobytes()))
                        while self._clip and self._clip[0][0] < t - keep:
                            self._clip.popleft()
        finally:
            cap.release()

    def set_rotation(self, degrees: int) -> None:
        """같은 카메라를 다른 회전으로 다시 열면: 쌓아 둔 사람·영상은 모양이 달라지므로 비운다."""
        if degrees == self.rotation:
            return
        self.rotation = degrees
        with self._clip_lock:
            self._clip.clear()
        self._reset = True                                 # 처리 스레드가 다음 Frame 전에 비운다 (동시 수정 방지)

    def clip_mp4(self, seconds: float) -> tuple[bytes, float, int]:
        """최근 seconds 초를 MP4 로. 실시간 예측이 본 구간(최근 8초)과 같은 영상 → 저장 결과가 화면과 맞는다."""
        import os
        import tempfile
        with self._clip_lock:
            if not self._clip:
                return b"", 0.0, 0
            end = self._clip[-1][0]
            frames = [(t, b) for t, b in self._clip if t >= end - seconds]
        span = frames[-1][0] - frames[0][0] if len(frames) > 1 else 0.0
        if span < 1.0:
            return b"", span, len(frames)
        imgs = [cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR) for _, b in frames]
        h, w = imgs[0].shape[:2]
        fps = max(1.0, (len(imgs) - 1) / span)          # 휴대폰 스트림은 FPS 표시가 부정확해 실제 받은 수로
        fd, path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        try:
            wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            for img in imgs:
                if img.shape[:2] == (h, w):                # 회전을 바꾼 직후 섞인 Frame 은 버림
                    wr.write(img)
            wr.release()
            with open(path, "rb") as f:
                return f.read(), span, len(imgs)
        finally:
            os.remove(path)

    # ── 처리 스레드 ──
    def _work(self) -> None:
        cfg = self.cfg
        last_seq, last_predict, fps_t, fps_n = 0, 0.0, time.monotonic(), 0
        while not self._stop.is_set():
            now = time.monotonic()
            if now - self.last_access > cfg["live_idle_seconds"]:
                self.stop("화면을 보는 사람이 없어 멈췄습니다.")
                return
            if now - self.started > cfg["live_max_seconds"]:
                self.stop(f"최대 {cfg['live_max_seconds'] // 60}분이 지나 멈췄습니다. 다시 시작하세요.")
                return
            with self._cond:
                self._cond.wait_for(lambda seq=last_seq: self._raw_seq != seq or self._stop.is_set(), timeout=1.0)
                if self._stop.is_set() or self._raw is None or self._raw_seq == last_seq:
                    continue
                last_seq = self._raw_seq
                t, frame = self._raw
            if self._reset:
                self.tracks.clear()
                self.results.clear()
                self._reset = False
            try:
                people = self._pose(frame, t)
            except Exception as exc:                       # Pose 오류로 스레드가 조용히 죽지 않게
                self._fail(f"실시간 Pose 처리 오류: {type(exc).__name__}")
                return
            if self.state == "connecting":
                self.state = "running"
            if t - last_predict >= cfg["live_predict_every_seconds"]:
                self._predict(t)
                last_predict = t
            self._draw(frame, people)
            fps_n += 1
            if t - fps_t >= 1.0:
                self.fps, fps_t, fps_n = fps_n / (t - fps_t), t, 0

    def _pose(self, frame: np.ndarray, t: float) -> list[Person]:
        h, w = frame.shape[:2]
        self.frame_size = (w, h)
        people = track_people(self.pose, frame, self.cfg, self.device)   # 업로드 분석과 같은 함수
        for person in people:
            tr = self.tracks.get(person.track_id)
            if tr is None:
                tr = self.tracks[person.track_id] = _Track(first_seen=t)
            tr.add(t, person)
            tr.last_seen = t
            tr.trim(t - self.cfg["window_seconds"])
        return people

    def _predict(self, t: float) -> None:
        cfg = self.cfg
        # 3초 넘게 안 보인 사람은 정리 (화면 밖으로 나감)
        for tid in [k for k, tr in self.tracks.items() if t - tr.last_seen > 3.0]:
            self.tracks.pop(tid, None)
            self.results.pop(tid, None)
        for tid, tr in self.tracks.items():
            if t - tr.last_seen > 0.5 or len(tr.times) < 2:
                continue
            seen = round(t - tr.first_seen, 1)
            r = predict_track(tr.as_track_data(), self.bundle)
            if r.pop("skipped"):
                prev = self.results.get(tid)
                if prev and prev.get("status") == "ok":   # 잠깐 가려졌으면 직전 예측을 유지
                    prev["seen_seconds"] = seen
                    continue
                self.results[tid] = {"status": "collecting", "reason": r["reason"], "seen_seconds": seen,
                                     "track_seconds": r["track_seconds"]}
                continue
            # 같은 동작이 몇 초째 이어지는지 (자동 기록 조건: 순간 하나가 튄 것과 구분)
            prev = self.results.get(tid) or {}
            if prev.get("status") == "ok" and prev.get("action_id") == r["action_id"]:
                streak = prev["streak_seconds"] + cfg["live_predict_every_seconds"]
                streak_min = min(prev["streak_min_confidence"], r["confidence"])
            else:
                streak, streak_min = 0.0, r["confidence"]
            self.results[tid] = {"status": "ok", "seen_seconds": seen, **{k: r[k] for k in LIVE_KEYS},
                                 "streak_seconds": round(streak, 1), "streak_min_confidence": round(streak_min, 4)}

    def _draw(self, frame: np.ndarray, people: list[Person]) -> None:
        kp_min = self.cfg["keypoint_confidence"]
        img = frame.copy()
        s = max(1, round(img.shape[1] / 640))            # 큰 화면에서도 선·글자가 보이게
        for person in people:
            tid, xy, conf, box = person.track_id, person.xy, person.conf, person.box
            color = PALETTE[tid % len(PALETTE)]
            for a, b in COCO_SKELETON:
                if conf[a] >= kp_min and conf[b] >= kp_min:
                    cv2.line(img, tuple(map(int, xy[a])), tuple(map(int, xy[b])), color, 2 * s, cv2.LINE_AA)
            for (x, y), c in zip(xy, conf, strict=True):
                if c >= kp_min:
                    cv2.circle(img, (int(x), int(y)), 3 * s, (255, 255, 255), -1, cv2.LINE_AA)
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, s)
            r = self.results.get(tid, {})
            if r.get("status") == "ok":
                label = f"#{tid} {r['action_id']} {r['confidence'] * 100:.0f}%" + ("?" if r["uncertain"] else "")
            else:
                label = f"#{tid} ..."
            # 한글은 cv2 로 못 쓰므로 Frame 에는 ID·동작 번호만, 한글 이름은 화면 옆 표에
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, 2 * s)
            ty = max(th + 6, y1)
            cv2.rectangle(img, (x1, ty - th - 6), (x1 + tw + 6, ty), color, -1)
            cv2.putText(img, label, (x1 + 3, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, (255, 255, 255), 2 * s, cv2.LINE_AA)
        cv2.putText(img, f"LIVE {self.fps:.1f} fps", (10, 24 * s), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, (0, 0, 255), 2 * s,
                    cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.cfg["live_jpeg_quality"]])
        if ok:
            with self._cond:
                self.jpeg = buf.tobytes()
                self.jpeg_seq += 1
                self._cond.notify_all()

    # ── 내보내기 ──
    def mjpeg(self):
        """multipart/x-mixed-replace 스트림. <img src> 하나로 브라우저가 계속 새 Frame 을 그린다."""
        last = 0
        while not self._stop.is_set():
            with self._cond:
                self._cond.wait_for(lambda seq=last: self.jpeg_seq != seq or self._stop.is_set(), timeout=1.0)
                if self.jpeg_seq == last or self.jpeg is None:
                    continue
                last, jpg = self.jpeg_seq, self.jpeg
            self.touch()
            yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode()
                   + b"\r\n\r\n" + jpg + b"\r\n")


class LiveManager:
    """한 번에 실시간 세션 하나만 (같은 카메라를 또 열면 같은 세션에 붙는다)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._sessions: dict[str, LiveSession] = {}

    def start(self, camera: str, url: str, bundle, device, rotation: int = 0) -> LiveSession:
        with self._lock:
            for sid in [k for k, s in self._sessions.items() if not s.alive]:
                del self._sessions[sid]
            for s in self._sessions.values():
                if s.url == url:
                    s.touch()
                    s.set_rotation(rotation)
                    return s
                raise LiveBusy(f"카메라 '{s.camera}' 실시간 보기가 이미 실행 중입니다. 먼저 멈추세요.")
            s = LiveSession(camera, url, bundle, device, rotation)
            self._sessions[s.id] = s
            return s

    def get(self, sid: str) -> LiveSession | None:
        return self._sessions.get(sid)

    def stop(self, sid: str) -> bool:
        with self._lock:
            s = self._sessions.pop(sid, None)
        if s:
            s.stop("사용자가 멈췄습니다.")
        return s is not None

    def stop_all(self) -> None:
        with self._lock:
            sessions, self._sessions = list(self._sessions.values()), {}
        for s in sessions:
            s.stop()
        for s in sessions:                         # 카메라 읽기가 끝날 때까지 잠깐 기다림 (종료 중 강제 abort 방지)
            for t in s._threads:
                t.join(timeout=6)
