"""IP Webcam 카메라 주소 — 정리·검사·연결 확인.

왜 이렇게 막는가
  - 서버가 사용자가 준 주소로 직접 접속하면(SSRF) 서버 안쪽(postgres·redis·fastapi)이나 바깥 아무 곳이나 찔러 볼 수 있다.
    과제 안내서도 "허용한 카메라만, 임의 URL 무제한 입력 금지"를 요구한다.
  - 그래서 ① 주소는 관리자(is_staff)만 웹에서 등록·수정하고, 일반 사용자는 이름만 고른다.
    ② 주소는 숫자 IP 이면서 CAMERA_ALLOWED_NETWORKS(기본 192.168.0.0/16, 10.0.0.0/8) 안이어야 한다.
       Docker 내부망(172.16.0.0/12)·localhost 는 기본값에서 빠져 있어 서비스 컨테이너를 가리킬 수 없다.
  - 휴대폰 IP 는 와이파이를 다시 잡을 때마다 바뀌므로 .env 를 고치고 재시작하는 대신 화면에서 바로 바꾼다.
"""
import ipaddress
import time
from urllib.parse import urlsplit, urlunsplit

from django.conf import settings
from django.core.exceptions import ValidationError


def normalize_camera_url(raw: str) -> str:
    """'192.168.0.10:8080' 처럼 앱 화면에 보이는 그대로 적어도 http://…/video 로 맞춘다."""
    s = (raw or "").strip()
    if not s:
        return s
    if "://" not in s:
        s = "http://" + s
    p = urlsplit(s)
    path = p.path if p.path not in ("", "/") else "/video"
    return urlunsplit((p.scheme.lower(), p.netloc, path, p.query, ""))


def allowed_networks_text() -> str:
    return ", ".join(str(n) for n in settings.CAMERA_ALLOWED_NETWORKS)


def validate_camera_url(url: str) -> None:
    p = urlsplit(url)
    if p.scheme not in ("http", "https"):
        raise ValidationError("http:// 주소만 쓸 수 있습니다 (예: 192.168.0.10:8080).")
    try:
        _ = p.port                      # 잘못된 포트면 여기서 ValueError
    except ValueError as exc:
        raise ValidationError("포트 번호가 올바르지 않습니다.") from exc
    try:
        ip = ipaddress.ip_address(p.hostname or "")
    except ValueError as exc:
        raise ValidationError("휴대폰 IP 를 숫자로 적어 주세요 (IP Webcam 앱 화면 아래 주소, 예: 192.168.0.10:8080).") from exc
    if not any(ip in net for net in settings.CAMERA_ALLOWED_NETWORKS):
        raise ValidationError(f"같은 와이파이(내부망) 주소만 쓸 수 있습니다. 허용 범위: {allowed_networks_text()}")


def probe(url: str, rotation: int = 0) -> dict:
    """연결 확인: 접속해서 Frame 하나를 받아 본다 (관리자 '연결 확인' 버튼)."""
    from .camera_capture import open_camera
    t0 = time.monotonic()
    cap = open_camera(url, 5)
    try:
        if not cap.isOpened():
            return {"ok": False, "message": "연결하지 못했습니다. 휴대폰 IP Webcam 앱에서 '서버 시작'을 눌렀는지, 같은 와이파이인지, 주소가 맞는지 확인하세요."}
        ok, frame = cap.read()
        if not ok:
            return {"ok": False, "message": "연결은 됐지만 화면을 받지 못했습니다. 주소 끝이 /video 인지 확인하세요."}
        h, w = frame.shape[:2]
        if rotation in (90, 270):
            w, h = h, w
        return {"ok": True, "width": w, "height": h, "seconds": round(time.monotonic() - t0, 2),
                "message": f"연결됨 · {w}×{h}"}
    finally:
        cap.release()
