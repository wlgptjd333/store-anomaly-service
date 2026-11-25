"""라우터 공통 의존성."""
from fastapi import HTTPException, Request


def ready_bundle(request: Request):
    """모델이 준비됐을 때만 (아니면 503 — 이유는 /health 의 model.error)"""
    bundle = request.app.state.bundle
    if not bundle.ready:
        raise HTTPException(503, f"모델이 준비되지 않았습니다: {bundle.error}")
    return bundle
