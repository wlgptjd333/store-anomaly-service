"""FastAPI 추론 서버 호출. 영상 파일을 HTTP multipart 로 보낸다 (컨테이너끼리 파일 경로를 공유한다고 가정하지 않음)."""
from pathlib import Path

import requests
from django.conf import settings


class InferenceError(Exception):
    """사용자에게 그대로 보여 줄 수 있는 실패 사유"""


def _url(path: str) -> str:
    return settings.FASTAPI_URL.rstrip("/") + path


def _detail(resp) -> str:
    """FastAPI HTTPException 의 detail, JSON 이 아니면 본문 앞부분"""
    try:
        return str(resp.json().get("detail", resp.text))
    except ValueError:
        return resp.text[:200]


def predict_video(path: str) -> dict:
    url = _url("/predict/video")
    try:
        with open(path, "rb") as f:
            resp = requests.post(url, files={"file": (Path(path).name, f, "video/mp4")},
                                 timeout=settings.FASTAPI_TIMEOUT_SECONDS)
    except FileNotFoundError as exc:
        raise InferenceError("분석할 영상 파일을 찾지 못했습니다.") from exc
    except requests.Timeout as exc:
        raise InferenceError(f"AI 서버 응답 시간이 {settings.FASTAPI_TIMEOUT_SECONDS}초를 넘었습니다.") from exc
    except requests.ConnectionError as exc:
        raise InferenceError("AI 서버에 연결하지 못했습니다. 잠시 후 다시 시도하세요.") from exc
    if resp.status_code != 200:
        raise InferenceError(f"AI 서버 오류 ({resp.status_code}): {_detail(resp)}")
    return resp.json()


class LiveError(Exception):
    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status


def _live(method: str, path: str, **kw):
    url = _url("/live" + path)
    try:
        resp = requests.request(method, url, timeout=kw.pop("timeout", 10), **kw)
    except requests.RequestException as exc:
        raise LiveError("AI 서버에 연결하지 못했습니다. 잠시 후 다시 시도하세요.") from exc
    if resp.status_code >= 400 and not kw.get("stream"):
        raise LiveError(_detail(resp), resp.status_code if resp.status_code in (404, 409, 503) else 502)
    return resp


def live_start(camera: str, url: str, rotation: int = 0) -> dict:
    return _live("POST", "/start", json={"camera": camera, "url": url, "rotation": rotation}).json()


def live_clip(sid: str, seconds: float) -> bytes:
    """실시간 화면이 본 최근 seconds 초 원본(회전 후) MP4"""
    return _live("GET", f"/{sid}/clip", params={"seconds": seconds}, timeout=30).content


def live_state(sid: str) -> dict:
    return _live("GET", f"/{sid}").json()


def live_stop(sid: str) -> None:
    _live("DELETE", f"/{sid}")


def live_stream(sid: str):
    """MJPEG 를 FastAPI 에서 받아 그대로 넘긴다. 연결은 브라우저가 끊을 때 닫힌다."""
    resp = _live("GET", f"/{sid}/stream", stream=True, timeout=(5, 30))
    if resp.status_code != 200:
        resp.close()
        raise LiveError("실시간 세션이 없습니다. 다시 시작하세요.", 404)
    return resp


def health() -> dict:
    try:
        return requests.get(_url("/health"), timeout=3).json()
    except (requests.RequestException, ValueError):
        return {"status": "unreachable"}
