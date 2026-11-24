"""DRF API — /api/analyses/

  POST   /api/analyses/              분석 요청 → DB 저장 → (저장 확정 후) Celery 등록 → 작업 ID 즉시 반환(202)
  GET    /api/analyses/              내 분석 목록
  GET    /api/analyses/{id}/         상세 + 사람별 결과
  PATCH  /api/analyses/{id}/         제목·메모만 수정
  DELETE /api/analyses/{id}/         기록 삭제 (사람별 예측은 CASCADE, 원본·재생용 영상 파일도 함께 지움)
  영상 보기는 HTML 쪽 /analyses/{id}/video/ (권한 확인 + 구간 이동용 Range, page_views.video)
  GET    /api/analyses/{id}/status/  상태만 (브라우저가 주기적으로 조회)
"""
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from . import jobs
from .models import AnalysisJob
from .permissions import IsOwner
from .serializers import (AnalysisJobCreateSerializer, AnalysisJobSerializer, AnalysisJobStatusSerializer,
                          AnalysisJobUpdateSerializer)


class AnalysisJobViewSet(viewsets.ModelViewSet):
    permission_classes = [IsAuthenticated, IsOwner]
    http_method_names = ["get", "post", "patch", "delete", "head", "options"]   # PUT(전체 수정) 막음

    def get_queryset(self):
        qs = AnalysisJob.objects.select_related("user").prefetch_related("predictions")
        # 관리자는 대시보드에서 누른 다른 회원 기록의 상세·상태만 읽을 수 있다. 목록·수정·삭제는 관리자도 본인 것만
        if self.request.user.is_staff and self.action in ("retrieve", "status"):
            return qs
        return qs.filter(user=self.request.user)

    def get_serializer_class(self):
        if self.action == "create":
            return AnalysisJobCreateSerializer
        if self.action == "partial_update":
            return AnalysisJobUpdateSerializer
        if self.action == "status":
            return AnalysisJobStatusSerializer
        return AnalysisJobSerializer

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        # 저장 확정 뒤 Celery 등록 (jobs.py — 실시간 장면 저장도 같은 함수)
        job = jobs.submit(lambda task_id: serializer.save(user=request.user, celery_task_id=task_id))
        return Response(jobs.accepted(job), status=status.HTTP_202_ACCEPTED)

    def update(self, request, *args, **kwargs):
        kwargs["partial"] = True
        super().update(request, *args, **kwargs)
        return Response(AnalysisJobSerializer(self.get_object(), context=self.get_serializer_context()).data)

    def perform_destroy(self, instance):
        # PersonPrediction 은 on_delete=CASCADE, 원본·재생용 영상 파일은 models.delete_video_files(post_delete)가 지움
        instance.delete()

    @action(detail=True, methods=["get"])
    def status(self, request, pk=None):
        return Response(AnalysisJobStatusSerializer(self.get_object()).data)
