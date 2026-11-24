"""분석 작업 등록 — '저장 확정 뒤 Celery 에 넣기'를 한 곳에서.

영상 업로드·IP Webcam 8초 수집(views.create)과 실시간 장면 저장·자동 감지(live_views.live_save)가 같은 함수를 쓴다.
  1. Task ID 를 미리 만든다 → 사용자에게 바로 돌려줄 수 있음 (202)
  2. DB 저장(commit)이 끝난 뒤에 Celery 에 등록 → Worker 가 아직 없는 행을 찾는 일이 없다
  3. Redis 가 꺼져 등록하지 못하면 작업을 FAILURE 와 사유로 남긴다 (대기 상태로 영원히 남지 않게)
"""
from __future__ import annotations

import logging
import uuid
from typing import Callable

from django.db import transaction
from django.utils import timezone

from .models import AnalysisJob
from .tasks import run_analysis

log = logging.getLogger(__name__)


def enqueue(job: AnalysisJob, task_id: str) -> None:
    try:
        run_analysis.apply_async(args=[job.pk], task_id=task_id)
    except Exception as exc:                     # noqa: BLE001  Redis 연결 오류 종류가 여러 가지
        log.exception("enqueue failed for job %s", job.pk)
        AnalysisJob.objects.filter(pk=job.pk).update(
            status=AnalysisJob.Status.FAILURE, finished_at=timezone.now(),
            error_message=f"작업 대기열(Redis)에 등록하지 못했습니다: {type(exc).__name__}")


def submit(save: Callable[[str], AnalysisJob]) -> AnalysisJob:
    """save(task_id) 가 작업을 DB 에 저장해 돌려주면, commit 뒤 Celery 에 등록하고 최신 상태로 돌려준다."""
    task_id = str(uuid.uuid4())
    with transaction.atomic():
        job = save(task_id)
        transaction.on_commit(lambda: enqueue(job, task_id))
    job.refresh_from_db()
    return job


def accepted(job: AnalysisJob, **extra) -> dict:
    """202 응답 본문 (브라우저는 status_url 을 2초마다 조회)"""
    return {"id": job.pk, "task_id": job.celery_task_id, "status": job.status,
            "status_url": f"/api/analyses/{job.pk}/status/", **extra}
