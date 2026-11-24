"""HTML 화면. 데이터는 JavaScript(analyses.js)가 DRF API 로 가져오고, 대시보드는 ORM 집계 결과를 Template 에 넘긴다."""
from pathlib import Path

from django.conf import settings
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.auth.decorators import login_required
from django.http import Http404
from django.shortcuts import get_object_or_404, render

from . import dashboard as dash
from .actions import ACTIONS
from .cameras import allowed_networks_text
from .models import AnalysisJob, Camera
from .playback import ranged_file_response
from .services import health


def _live_config():
    return {"min_confidence": settings.LIVE_AUTO_MIN_CONFIDENCE, "min_seconds": settings.LIVE_AUTO_MIN_SECONDS,
            "cooldown_seconds": settings.LIVE_AUTO_COOLDOWN_SECONDS, "clip_seconds": settings.LIVE_CLIP_SECONDS,
            "capture_seconds": settings.CAPTURE_SECONDS}


def _page_context():
    live = _live_config()
    return {
        "actions": ACTIONS,
        "action_names": {a[0]: a[2] for a in ACTIONS},             # 구간별 예측에 이름 붙이기 (json_script)
        "cameras": list(Camera.objects.filter(enabled=True).values_list("name", flat=True)),  # 이름만
        "camera_networks": allowed_networks_text(),
        "live_auto": live,                      # Template 문구용
        "live_config": live,                    # analyses.js 가 읽는 같은 값 (json_script)
        "capture_seconds": settings.CAPTURE_SECONDS,
        "uncertain_below": settings.UNCERTAIN_BELOW,
        "max_upload_mb": settings.MAX_UPLOAD_MB,
    }


def _job_for(request, pk) -> AnalysisJob:
    """본인 기록, 또는 관리자(대시보드에서 들어온 경우)는 다른 회원 기록도 읽기만. 그 밖에는 있는지조차 알려 주지 않음(404)"""
    job = get_object_or_404(AnalysisJob.objects.select_related("user"), pk=pk)
    if job.user_id != request.user.id and not request.user.is_staff:
        raise Http404
    return job


@login_required
def index(request):
    return render(request, "analyses/index.html", _page_context())


@login_required
def detail(request, pk):
    job = _job_for(request, pk)
    return render(request, "analyses/detail.html",
                  {**_page_context(), "job": job, "can_edit": job.user_id == request.user.id})


@login_required
def video(request, pk):
    """분석한 영상 보기. 기본은 재생용 사본(H.264), 없으면 원본. ?original=1 원본, ?download=1 내려받기.

    /media/ 를 그대로 열지 않는 이유: 다른 회원 영상 주소를 짐작해 볼 수 없게 매번 권한을 확인한다.
    """
    job = _job_for(request, pk)
    use_original = request.GET.get("original") == "1" or not job.playback
    field = job.video if use_original else job.playback
    if not field:
        raise Http404("영상이 없습니다.")
    path = Path(field.path)
    if not path.exists():
        raise Http404("영상 파일이 없습니다.")
    name = Path(job.video.name).name if use_original else f"job{job.pk}_playback.mp4"
    return ranged_file_response(request, path, name, as_attachment=request.GET.get("download") == "1")


@staff_member_required
def dashboard(request):
    params = request.GET
    period = params.get("period", "all")
    if period not in dict(dash.PERIODS):
        period = "all"
    jobs = dash.period_jobs({"period": period})
    listed = dash.filter_jobs(jobs, params)
    if params.get("export") == "csv":
        return dash.csv_response(listed)
    page, query = dash.page_of(listed, params)
    filters = {k: params.get(k, "") for k in dash.FILTER_KEYS}
    return render(request, "analyses/dashboard.html", {
        "chart_data": dash.chart_data(jobs, period),
        "summary": dash.summary(jobs),
        "confusions": dash.confusions(jobs),
        "page": page, "query": query, "listed_count": page.paginator.count,
        "period": period, "periods": dash.PERIODS, "filters": filters,
        "filtered": any(filters.values()),
        "actions": ACTIONS,
        "statuses": AnalysisJob.Status.choices, "input_types": AnalysisJob.InputType.choices,
        "sources": AnalysisJob.Source.choices,
        "ai_health": health(),
    })
