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

import threading
import time
import uuid
from dataclasses import dataclass, field

import cv2
import numpy as np

from services.action_inference import predict_track
from services.model_loader import APP_ROOT
from services.pose_pipeline import TrackData, frame_features
from src.pose_utils import COCO_SKELETON

# 사람(Track ID)마다 다른 색 (BGR)
PALETTE = [(255, 140, 0), (60, 200, 60), (40, 40, 230), (230, 60, 200), (0, 200, 230), (200, 200, 40),
           (140, 80, 255), (80, 160, 255)]


class LiveBusy(Exception):
    """다른 카메라로 실시간 보기가 이미 돌고 있음 (CPU 하나로 두 개를 돌리면 둘 다 느려진다)"""


@dataclass
class _Track:
    times: list[float] = field(default_factory=list)
    features: list[np.ndarray] = field(default_factory=list)
    valid: list[bool] = field(default_factory=list)
    keypoint_conf: list[float] = field(default_factory=list)
    first_seen: float = 0.0
    last_seen: float = 0.0

    def trim(self, keep_from: float) -> None:
        i = 0
        while i < len(self.times) and self.times[i] < keep_from:
            i += 1
        if i:
            for lst in (self.times, self.features, self.valid, self.keypoint_conf):
                del lst[:i]

    def as_track_data(self) -> TrackData:
        return TrackData(times=list(self.times), features=list(self.features), valid=list(self.valid),
                         keypoint_conf=list(self.keypoint_conf))


class LiveSession:
    def __init__(self, camera: str, url: str, bundle, device):
        from ultralytics import YOLO
        self.id = str(uuid.uuid4())
        self.camera, self.url = camera, url
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
        people = []
        for tid in sorted(self.results):
            r = self.results[tid]
            people.append({"track_id": tid, **r})
        return {"session_id": self.id, "camera": self.camera, "state": self.state, "message": self.message,
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
                with self._cond:
                    self._raw = (time.monotonic(), frame)
                    self._raw_seq += 1
                    self._cond.notify_all()
        finally:
            cap.release()

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
                self._cond.wait_for(lambda: self._raw_seq != last_seq or self._stop.is_set(), timeout=1.0)
                if self._stop.is_set() or self._raw is None or self._raw_seq == last_seq:
                    continue
                last_seq = self._raw_seq
                t, frame = self._raw
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

    def _pose(self, frame: np.ndarray, t: float) -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
        cfg = self.cfg
        h, w = frame.shape[:2]
        self.frame_size = (w, h)
        res = self.pose.track(source=frame, persist=True, tracker=cfg["tracker"], imgsz=cfg["image_size"],
                              conf=cfg["person_confidence"], device=self.device, verbose=False)[0]
        boxes, kps = res.boxes, res.keypoints
        if boxes is None or boxes.id is None or kps is None or kps.xy is None:
            return []
        ids = boxes.id.int().cpu().tolist()
        xys = kps.xy.cpu().numpy()
        confs = kps.conf.cpu().numpy() if kps.conf is not None else np.ones((len(ids), 17), np.float32)
        bxs = boxes.xyxy.cpu().numpy()
        people = []
        for tid, xy, conf, box in zip(ids, xys, confs, bxs):
            vals, valid = frame_features(xy, conf, w, h, cfg["keypoint_confidence"])
            tr = self.tracks.get(int(tid))
            if tr is None:
                tr = self.tracks[int(tid)] = _Track(first_seen=t)
            tr.times.append(t)
            tr.features.append(vals)
            tr.valid.append(valid)
            tr.keypoint_conf.append(float(np.mean(conf)))
            tr.last_seen = t
            tr.trim(t - cfg["window_seconds"])
            people.append((int(tid), xy, conf, box))
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
            self.results[tid] = {"status": "ok", "seen_seconds": seen, "track_seconds": r["track_seconds"],
                                 "action_id": r["action_id"], "action_label": r["action_label"],
                                 "action_name_ko": r["action_name_ko"], "category": r["category"],
                                 "confidence": r["confidence"], "uncertain": r["uncertain"], "top3": r["top3"],
                                 "valid_frames": r["valid_frames"]}

    def _draw(self, frame: np.ndarray, people) -> None:
        kp_min = self.cfg["keypoint_confidence"]
        img = frame.copy()
        s = max(1, round(img.shape[1] / 640))            # 큰 화면에서도 선·글자가 보이게
        for tid, xy, conf, box in people:
            color = PALETTE[tid % len(PALETTE)]
            for a, b in COCO_SKELETON:
                if conf[a] >= kp_min and conf[b] >= kp_min:
                    cv2.line(img, tuple(map(int, xy[a])), tuple(map(int, xy[b])), color, 2 * s, cv2.LINE_AA)
            for (x, y), c in zip(xy, conf):
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
                self._cond.wait_for(lambda: self.jpeg_seq != last or self._stop.is_set(), timeout=1.0)
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

    def start(self, camera: str, url: str, bundle, device) -> LiveSession:
        with self._lock:
            for sid in [k for k, s in self._sessions.items() if not s.alive]:
                del self._sessions[sid]
            for s in self._sessions.values():
                if s.url == url:
                    s.touch()
                    return s
                raise LiveBusy(f"카메라 '{s.camera}' 실시간 보기가 이미 실행 중입니다. 먼저 멈추세요.")
            s = LiveSession(camera, url, bundle, device)
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
