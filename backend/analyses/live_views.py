"""카메라 관리 API + 실시간 보기 중계.

  GET    /api/cameras/               카메라 목록 (일반 사용자는 이름만, 관리자는 주소까지)
  POST   /api/cameras/               등록        — 관리자만
  PATCH  /api/cameras/{id}/          주소·사용 수정 — 관리자만
  DELETE /api/cameras/{id}/          삭제        — 관리자만
  POST   /api/cameras/{id}/check/    연결 확인 (Frame 하나 받아 봄) — 관리자만

  POST   /api/live/                  {"camera_name"} → FastAPI 실시간 세션 시작
  GET    /api/live/{id}/             상태 + 사람별 최근 8초 예측
  DELETE /api/live/{id}/             멈춤
  POST   /api/live/{id}/save/        지금 화면의 최근 8초를 저장 → Celery 분석 ('이 장면 저장', auto=true 면 자동 감지)
  GET    /live/{id}/stream/          Skeleton 을 그린 MJPEG (<img src>)

실시간 예측 자체는 1초마다 바뀌는 값이라 DB 에 남기지 않는다. 기록은 '이 장면 저장'·자동 감지·'AI 분석 시작'(8초 수집)으로
남기고, 셋 다 같은 Celery → FastAPI /predict/video 를 탄다. 저장 당시 실시간 예측은 live_snapshot 에 같이 둔다.
FastAPI 포트는 127.0.0.1 에만 열려 있어서, 브라우저는 로그인 확인을 거치는 Django 를 통해서만 화면을 받는다.
"""
import uuid
from datetime import timedelta

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.core.files.base import ContentFile
from django.db import transaction
from django.http import JsonResponse, StreamingHttpResponse
from django.utils import timezone
from rest_framework import status, viewsets
from rest_framework.decorators import action, api_view, permission_classes
from rest_framework.permissions import SAFE_METHODS, BasePermission, IsAuthenticated
from rest_framework.response import Response

from . import services
from .actions import ACTION_BY_ID
from .cameras import probe
from .models import AnalysisJob, Camera
from .serializers import CameraPublicSerializer, CameraSerializer
from .views import AnalysisJobViewSet


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
        cam = self.get_object()
        result = probe(cam.url, cam.rotation)
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
        return Response(services.live_start(cam.name, cam.url, cam.rotation), status=201)
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


def _auto_hit(people: list[dict], min_conf: float) -> dict | None:
    """자동 감지 조건: 같은 동작이 LIVE_AUTO_MIN_SECONDS 이상 이어졌고, 그동안 최저 Confidence 도 기준 이상"""
    ok = [p for p in people if p.get("status") == "ok"
          and p.get("streak_seconds", 0) >= settings.LIVE_AUTO_MIN_SECONDS
          and p.get("streak_min_confidence", 0) >= min_conf]
    return max(ok, key=lambda p: p["confidence"], default=None)


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def live_save(request, sid):
    """실시간 화면이 본 최근 8초를 그대로 저장 → 업로드·8초 수집과 같은 Celery → FastAPI 분석으로 기록.

    auto=true 면 서버가 실시간 상태를 다시 보고 조건(이어진 시간·최저 Confidence·저장 간격)을 확인한다.
    브라우저가 보낸 값만 믿지 않기 위해서다.
    """
    auto = str(request.data.get("auto", "")).lower() in {"1", "true", "yes"}
    expected = request.data.get("expected_action", "") or ""
    if expected and expected not in ACTION_BY_ID:
        return Response({"expected_action": ["알 수 없는 동작입니다."]}, status=400)
    try:
        state = services.live_state(str(sid))
    except services.LiveError as exc:
        return _error(exc)
    if state.get("state") != "running":
        return Response({"detail": "실시간 보기가 실행 중일 때만 저장할 수 있습니다."}, status=409)
    people = [p for p in state.get("people", []) if p.get("status") == "ok"]
    snapshot = [{k: p.get(k) for k in ("track_id", "action_id", "action_name_ko", "confidence", "uncertain",
                                       "streak_seconds")} for p in people]
    title = (request.data.get("title") or "")[:100]
    if auto:
        try:
            min_conf = max(settings.UNCERTAIN_BELOW,
                           float(request.data.get("min_confidence", settings.LIVE_AUTO_MIN_CONFIDENCE)))
        except (TypeError, ValueError):
            return Response({"min_confidence": ["숫자로 보내 주세요."]}, status=400)
        hit = _auto_hit(people, min_conf)
        if hit is None:
            return Response({"detail": "자동 감지 조건이 아닙니다."}, status=409)
        since = timezone.now() - timedelta(seconds=settings.LIVE_AUTO_COOLDOWN_SECONDS)
        if AnalysisJob.objects.filter(user=request.user, source=AnalysisJob.Source.LIVE_AUTO,
                                      created_at__gte=since).exists():
            return Response({"detail": f"자동 감지는 {settings.LIVE_AUTO_COOLDOWN_SECONDS:.0f}초에 한 번만 저장합니다."},
                            status=429)
        title = f"자동 감지 {hit['action_id']} {hit['action_name_ko']} {hit['confidence'] * 100:.0f}%"[:100]
    try:
        clip = services.live_clip(str(sid), settings.LIVE_CLIP_SECONDS)
    except services.LiveError as exc:
        return _error(exc)

    task_id = str(uuid.uuid4())
    job = AnalysisJob(user=request.user, input_type=AnalysisJob.InputType.IPWEBCAM, camera_name=state["camera"],
                      source=AnalysisJob.Source.LIVE_AUTO if auto else AnalysisJob.Source.LIVE,
                      live_snapshot=snapshot, expected_action=expected, title=title, celery_task_id=task_id)
    job.video.save(f"live_{timezone.now():%Y%m%d_%H%M%S}.mp4", ContentFile(clip), save=False)
    with transaction.atomic():
        job.save()
        transaction.on_commit(lambda: AnalysisJobViewSet._enqueue(job, task_id))
    job.refresh_from_db()
    return Response({"id": job.pk, "task_id": task_id, "status": job.status, "source": job.source,
                     "title": job.title, "status_url": f"/api/analyses/{job.pk}/status/"}, status=202)


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
