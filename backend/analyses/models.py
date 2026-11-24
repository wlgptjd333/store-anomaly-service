from django.conf import settings
from django.db import models

from .actions import ACTION_CHOICES
from .cameras import normalize_camera_url, validate_camera_url


class Camera(models.Model):
    """허용한 IP Webcam. 관리자만 등록·수정하고, 일반 사용자는 이름만 고른다 (이유는 cameras.py)."""

    name = models.SlugField("이름", max_length=50, unique=True, help_text="화면에 보이는 이름 (영문·숫자·-_)")
    url = models.CharField("영상 주소", max_length=200, validators=[validate_camera_url],
                           help_text="IP Webcam 앱 화면의 주소. 예: 192.168.0.10:8080 → http://192.168.0.10:8080/video")
    enabled = models.BooleanField("사용", default=True)
    updated_at = models.DateTimeField("수정 시각", auto_now=True)

    class Meta:
        ordering = ["name"]
        verbose_name = verbose_name_plural = "카메라"

    def __str__(self):
        return self.name

    def clean_fields(self, exclude=None):
        self.url = normalize_camera_url(self.url)      # Admin 에서 '192.168.x.x:8080' 만 적어도 됨
        super().clean_fields(exclude)


class AnalysisJob(models.Model):
    """분석 요청 한 건. 영상 파일 자체는 Media 폴더에 두고 DB 에는 경로만 저장한다."""

    class InputType(models.TextChoices):
        UPLOAD = "UPLOAD", "영상 업로드"
        IPWEBCAM = "IPWEBCAM", "IP Webcam"

    class Status(models.TextChoices):
        PENDING = "PENDING", "대기"
        STARTED = "STARTED", "처리 중"
        SUCCESS = "SUCCESS", "완료"
        FAILURE = "FAILURE", "실패"

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="analysis_jobs")
    title = models.CharField("제목", max_length=100, blank=True)
    memo = models.TextField("메모", blank=True)
    input_type = models.CharField("입력 방식", max_length=10, choices=InputType.choices)
    video = models.FileField("영상", upload_to="videos/%Y/%m/%d/", blank=True)
    camera_name = models.CharField("카메라", max_length=50, blank=True)
    # 사용자가 '이 동작을 했다'고 고른 값. 검증된 정답이 아니다
    expected_action = models.CharField("예상 동작(사용자 선택)", max_length=3, choices=ACTION_CHOICES, blank=True)
    status = models.CharField("상태", max_length=10, choices=Status.choices, default=Status.PENDING, db_index=True)
    celery_task_id = models.CharField("Celery Task ID", max_length=50, blank=True)
    error_message = models.TextField("오류 사유", blank=True)
    result_message = models.CharField("결과 안내", max_length=200, blank=True)
    video_seconds = models.FloatField("영상 길이(초)", null=True, blank=True)
    processing_seconds = models.FloatField("AI 처리 시간(초)", null=True, blank=True)
    created_at = models.DateTimeField("요청 시각", auto_now_add=True)
    started_at = models.DateTimeField("시작 시각", null=True, blank=True)
    finished_at = models.DateTimeField("완료 시각", null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = verbose_name_plural = "분석 작업"

    def __str__(self):
        return f"#{self.pk} {self.get_input_type_display()} {self.status}"

    @property
    def is_finished(self) -> bool:
        return self.status in {self.Status.SUCCESS, self.Status.FAILURE}


class PersonPrediction(models.Model):
    """한 작업 안에서 추적한 사람 한 명의 예측. track_id 는 그 영상 안에서만 쓰는 번호다."""

    job = models.ForeignKey(AnalysisJob, on_delete=models.CASCADE, related_name="predictions")
    track_id = models.PositiveIntegerField("Track ID")
    action_id = models.CharField("동작 ID", max_length=3, choices=ACTION_CHOICES, db_index=True)
    action_label = models.CharField("Action Label", max_length=50)
    confidence = models.FloatField("Confidence")
    uncertain = models.BooleanField("판단 불확실", default=False)
    window_count = models.PositiveIntegerField("처리한 Window 수")
    valid_frames = models.PositiveIntegerField("유효 Frame 수", default=0)
    track_seconds = models.FloatField("화면에 보인 시간(초)", default=0)
    mean_keypoint_conf = models.FloatField("평균 관절 Confidence", null=True, blank=True)
    top3 = models.JSONField("상위 3개", default=list, blank=True)
    windows = models.JSONField("창별 결과", default=list, blank=True)
    predicted_at = models.DateTimeField("예측 시각", auto_now_add=True)

    class Meta:
        ordering = ["job", "track_id"]
        constraints = [models.UniqueConstraint(fields=["job", "track_id"], name="unique_track_per_job")]
        verbose_name = verbose_name_plural = "사람별 예측"

    def __str__(self):
        return f"job {self.job_id} · track {self.track_id} · {self.action_id}"
