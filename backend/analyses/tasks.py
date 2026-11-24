"""Celery 작업: 영상 확보(업로드 파일 또는 IP Webcam 수집) → FastAPI 추론 → Django ORM 으로 결과·상태 저장."""
import logging
from pathlib import Path

from celery import shared_task
from django.conf import settings
from django.db import transaction
from django.utils import timezone

from .camera_capture import CaptureError, capture_clip
from .models import AnalysisJob, PersonPrediction
from .services import InferenceError, predict_video

log = logging.getLogger(__name__)
CAPTURING = "카메라 녹화 중"            # analyses.js 가 이 문구를 보고 '지금 동작하세요'를 띄운다
ANALYZING = "녹화 끝, AI 분석 중"


def _fail(job: AnalysisJob, message: str) -> None:
    job.status = AnalysisJob.Status.FAILURE
    job.error_message = message
    job.result_message = ""                      # '녹화 중' 같은 단계 안내가 남지 않게
    job.finished_at = timezone.now()
    job.save(update_fields=["status", "error_message", "result_message", "finished_at"])


@shared_task(bind=True)
def run_analysis(self, job_id: int) -> str:
    try:
        job = AnalysisJob.objects.get(pk=job_id)
    except AnalysisJob.DoesNotExist:
        return "missing"                         # 그 사이 사용자가 삭제함
    job.status = AnalysisJob.Status.STARTED
    job.started_at = timezone.now()
    job.save(update_fields=["status", "started_at"])

    try:
        # 실시간 장면 저장·자동 감지는 영상을 이미 들고 오므로 수집을 건너뛴다
        if job.input_type == AnalysisJob.InputType.IPWEBCAM and not job.video:
            # 화면이 '지금 녹화 중'을 알 수 있게 단계 안내를 남긴다 (대기열에서 기다린 만큼 버튼보다 늦게 시작할 수 있음)
            job.result_message = CAPTURING
            job.save(update_fields=["result_message"])
            rel = Path("videos") / "ipwebcam" / f"job{job.pk}_{timezone.now():%Y%m%d_%H%M%S}.mp4"
            capture_clip(job.camera_name, Path(settings.MEDIA_ROOT) / rel)
            job.video.name = str(rel)
            job.result_message = ANALYZING
            job.save(update_fields=["video", "result_message"])
        result = predict_video(job.video.path)
    except (CaptureError, InferenceError) as exc:
        _fail(job, str(exc))
        return "FAILURE"
    except Exception as exc:                     # 예상 못 한 오류도 상태는 반드시 남긴다
        log.exception("analysis job %s failed", job_id)
        _fail(job, f"처리 중 오류가 발생했습니다: {type(exc).__name__}")
        return "FAILURE"

    with transaction.atomic():
        job.predictions.all().delete()
        PersonPrediction.objects.bulk_create([
            PersonPrediction(job=job, track_id=p["track_id"], action_id=p["action_id"],
                             action_label=p["action_label"], confidence=p["confidence"],
                             uncertain=p["uncertain"], window_count=p["window_count"],
                             valid_frames=p["valid_frames"], track_seconds=p["track_seconds"],
                             mean_keypoint_conf=p.get("mean_keypoint_conf"), top3=p.get("top3", []),
                             windows=p.get("windows", []))
            for p in result["predictions"]
        ])
        job.status = AnalysisJob.Status.SUCCESS
        job.result_message = result.get("message", "")[:200]
        job.video_seconds = result["video"]["seconds"]
        job.processing_seconds = result["processing_seconds"]
        job.finished_at = timezone.now()
        job.save(update_fields=["status", "result_message", "video_seconds", "processing_seconds", "finished_at"])
    return "SUCCESS"
