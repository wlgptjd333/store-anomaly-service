"""카메라 관리 API + 실시간 보기 중계.

  GET    /api/cameras/               카메라 목록 (일반 사용자는 이름만, 관리자는 주소까지)
  POST   /api/cameras/               등록        — 관리자만
  PATCH  /api/cameras/{id}/          주소·사용 수정 — 관리자만
  DELETE /api/cameras/{id}/          삭제        — 관리자만
  POST   /api/cameras/{id}/check/    연결 확인 (Frame 하나 받아 봄) — 관리자만

  POST   /api/live/                  {"camera_name"} → FastAPI 실시간 세션 시작
  GET    /api/live/{id}/             상태 + 사람별 최근 8초 예측
  DELETE /api/live/{id}/             멈춤
  GET    /live/{id}/stream/          Skeleton 을 그린 MJPEG (<img src>)

실시간 보기는 '보기용'이라 DB 에 남기지 않는다. 기록이 필요하면 'AI 분석 시작'(8초 수집 → Celery)을 쓴다.
FastAPI 포트는 127.0.0.1 에만 열려 있어서, 브라우저는 로그인 확인을 거치는 Django 를 통해서만 화면을 받는다.
"""
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse, StreamingHttpResponse
from rest_framework import status, viewsets
from rest_framework.decorators import action, api_view, permission_classes
from rest_framework.permissions import SAFE_METHODS, BasePermission, IsAuthenticated
from rest_framework.response import Response

from . import services
from .cameras import probe
from .models import Camera
from .serializers import CameraPublicSerializer, CameraSerializer


class ReadOrStaff(BasePermission):
    def has_permission(self, request, view):
        return bool(request.user and request.user.is_authenticated
                    and (request.method in SAFE_METHODS and view.action != "check" or request.user.is_staff))


class CameraViewSet(viewsets.ModelViewSet):
    permission_classes = [ReadOrStaff]
    http_method_names = ["get", "post", "patch", "delete", "head", "options"]
    pagination_class = None

    def get_queryset(self):
        qs = Camera.objects.all()
        return qs if self.request.user.is_staff else qs.filter(enabled=True)

    def get_serializer_class(self):
        return CameraSerializer if self.request.user.is_staff else CameraPublicSerializer

    @action(detail=True, methods=["post"])
    def check(self, request, pk=None):
        result = probe(self.get_object().url)
        return Response(result, status=status.HTTP_200_OK if result["ok"] else status.HTTP_400_BAD_REQUEST)


def _error(exc: services.LiveError):
    return Response({"detail": str(exc)}, status=exc.status)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def live_start(request):
    name = request.data.get("camera_name", "")
    cam = Camera.objects.filter(name=name, enabled=True).first()
    if cam is None:
        return Response({"camera_name": ["허용된 카메라만 쓸 수 있습니다."]}, status=400)
    try:
        return Response(services.live_start(cam.name, cam.url), status=201)
    except services.LiveError as exc:
        return _error(exc)


@api_view(["GET", "DELETE"])
@permission_classes([IsAuthenticated])
def live_session(request, sid):
    try:
        if request.method == "DELETE":
            services.live_stop(str(sid))
            return Response(status=204)
        return Response(services.live_state(str(sid)))
    except services.LiveError as exc:
        return _error(exc)


@login_required
def live_stream(request, sid):
    try:
        upstream = services.live_stream(str(sid))
    except services.LiveError as exc:
        return JsonResponse({"detail": str(exc)}, status=exc.status)

    def relay():
        try:
            yield from upstream.iter_content(chunk_size=None)   # FastAPI 가 보낸 조각을 받는 대로
        finally:
            upstream.close()

    resp = StreamingHttpResponse(relay(), content_type=upstream.headers.get(
        "content-type", "multipart/x-mixed-replace; boundary=frame"))
    resp["Cache-Control"] = "no-store"
    resp["X-Accel-Buffering"] = "no"
    return resp
