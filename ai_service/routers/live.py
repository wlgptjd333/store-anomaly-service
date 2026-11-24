"""실시간 보기 — Django 가 '허용된 카메라 주소'를 넘겨 시작하고, 화면·상태를 중계한다.

  POST   /live/start          {"camera": 이름, "url": 주소} → 세션 ID (같은 주소면 기존 세션에 붙음)
  GET    /live/{id}           상태 + 사람별 최근 8초 예측 (JSON)
  GET    /live/{id}/stream    Skeleton 을 그린 MJPEG
  DELETE /live/{id}           멈춤

주소 검사(내부망만, 관리자가 등록한 카메라만)는 Django 가 한다. 이 서버 포트는 127.0.0.1 에만 열려 있다.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from services.live_session import LiveBusy

router = APIRouter(prefix="/live", tags=["live"])


class LiveStart(BaseModel):
    camera: str = Field(max_length=50)
    url: str = Field(max_length=200)


def _session(request: Request, sid: str):
    s = request.app.state.live.get(sid)
    if s is None:
        raise HTTPException(404, "실시간 세션이 없습니다. 다시 시작하세요.")
    return s


@router.post("/start")
def start(request: Request, body: LiveStart):
    bundle = request.app.state.bundle
    if not bundle.ready:
        raise HTTPException(503, f"모델이 준비되지 않았습니다: {bundle.error}")
    try:
        s = request.app.state.live.start(body.camera, body.url, bundle, request.app.state.device)
    except LiveBusy as exc:
        raise HTTPException(409, str(exc)) from exc
    return s.snapshot()


@router.get("/{sid}")
def state(request: Request, sid: str):
    s = _session(request, sid)
    s.touch()
    return s.snapshot()


@router.get("/{sid}/stream")
def stream(request: Request, sid: str):
    s = _session(request, sid)
    s.touch()
    return StreamingResponse(s.mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame",
                             headers={"Cache-Control": "no-store"})


@router.delete("/{sid}")
def stop(request: Request, sid: str):
    if not request.app.state.live.stop(sid):
        raise HTTPException(404, "실시간 세션이 없습니다.")
    return {"ok": True}
