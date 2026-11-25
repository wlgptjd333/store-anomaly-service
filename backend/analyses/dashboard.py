"""관리자 대시보드 집계 (page_views.dashboard 가 부른다). 모두 Django ORM — FastAPI 는 DB 를 모른다.

화면 규칙
  · '기간'은 위 숫자(KPI)·차트·아래 표 모두에 적용
  · 나머지 조건(상태·입력·동작·회원·검색 …)은 아래 '작업 목록' 표와 CSV 에만 적용
    (차트까지 거르면 '상태별 건수' 같은 그림이 한 칸만 남아 의미가 없어서)
  · '수행한 동작과 일치'는 사용자가 고른 값과 비교한 참고 숫자. 검증된 정답이 아니고,
    자동 감지(LIVE_AUTO)는 Confidence 높은 장면만 골라 저장한 것이라 뺀다 (선택 편향)
"""
from __future__ import annotations

import csv
from datetime import timedelta

from django.core.paginator import Paginator
from django.db.models import Avg, Count, F, Q
from django.db.models.functions import TruncDate
from django.http import HttpResponse
from django.utils import timezone

from .actions import ACTIONS, CATEGORY_KO, action_name
from .models import AnalysisJob, PersonPrediction

PERIODS = [("all", "전체"), ("today", "오늘"), ("7d", "최근 7일"), ("30d", "최근 30일")]
PAGE_SIZE = 20
TREND_MAX_DAYS = 60
FILTER_KEYS = ["status", "input_type", "source", "action", "uncertain", "match", "has_video", "user", "q"]


def period_start(period: str):
    now = timezone.localtime()
    if period == "today":
        return now.replace(hour=0, minute=0, second=0, microsecond=0)
    if period == "7d":
        return now - timedelta(days=7)
    if period == "30d":
        return now - timedelta(days=30)
    return None


def period_jobs(params):
    jobs = AnalysisJob.objects.all()
    start = period_start(params.get("period", "all"))
    return jobs.filter(created_at__gte=start) if start else jobs


def filter_jobs(jobs, params):
    """표·CSV 용 조건. 잘못된 값은 결과가 0건이 될 뿐 오류를 내지 않는다."""
    for field in ("status", "input_type", "source"):
        if params.get(field):
            jobs = jobs.filter(**{field: params[field]})
    if params.get("action"):
        jobs = jobs.filter(predictions__action_id=params["action"])
    if params.get("uncertain") == "1":
        jobs = jobs.filter(predictions__uncertain=True)
    if params.get("has_video") == "1":
        jobs = jobs.exclude(video="")
    match = params.get("match")
    if match in ("ok", "bad"):
        jobs = jobs.exclude(expected_action="").filter(predictions__isnull=False)
        hit = Q(predictions__action_id=F("expected_action"))
        jobs = jobs.filter(hit) if match == "ok" else jobs.exclude(hit)
    if params.get("user"):
        jobs = jobs.filter(user__username__icontains=params["user"].strip())
    q = (params.get("q") or "").strip()
    if q:
        cond = Q(title__icontains=q) | Q(memo__icontains=q) | Q(camera_name__icontains=q) | Q(error_message__icontains=q)
        if q.lstrip("#").isdigit():
            cond |= Q(pk=int(q.lstrip("#")))
        jobs = jobs.filter(cond)
    return jobs.distinct()


def _pct(a, b):
    return round(100 * a / b, 1) if b else None


def summary(jobs):
    preds = PersonPrediction.objects.filter(job__in=jobs)
    n_jobs = jobs.count()
    n_fail = jobs.filter(status=AnalysisJob.Status.FAILURE).count()
    n_pred = preds.count()
    n_unc = preds.filter(uncertain=True).count()
    avg = jobs.filter(status=AnalysisJob.Status.SUCCESS).aggregate(proc=Avg("processing_seconds"), video=Avg("video_seconds"))
    graded = preds.graded()
    n_graded = graded.count()
    n_match = graded.filter(action_id=F("job__expected_action")).count()
    return {
        "jobs": n_jobs,
        "users": jobs.values("user").distinct().count(),
        "predictions": n_pred,
        "uncertain": n_unc, "uncertain_pct": _pct(n_unc, n_pred),
        "failure": n_fail, "failure_pct": _pct(n_fail, n_jobs),
        "avg_processing": round(avg["proc"], 1) if avg["proc"] is not None else None,
        "avg_video": round(avg["video"], 1) if avg["video"] is not None else None,
        "graded": n_graded, "matched": n_match, "match_pct": _pct(n_match, n_graded),
        "running": AnalysisJob.objects.running().count(),
        "with_video": jobs.exclude(video="").count(),
    }


def chart_data(jobs, period: str):
    preds = PersonPrediction.objects.filter(job__in=jobs)
    # ① 동작별 예측 건수 — 한 번도 안 나온 동작도 0 으로
    by_action = dict(preds.values_list("action_id").annotate(n=Count("id")))
    uncertain = dict(preds.filter(uncertain=True).values_list("action_id").annotate(n=Count("id")))
    actions = [{"id": a[0], "name_ko": a[2], "category": a[3], "category_ko": CATEGORY_KO[a[3]],
                "count": by_action.get(a[0], 0), "uncertain": uncertain.get(a[0], 0)} for a in ACTIONS]
    # ② 작업 상태별
    by_status = dict(jobs.values_list("status").annotate(n=Count("id")))
    statuses = [{"status": s.value, "label": s.label, "count": by_status.get(s.value, 0)} for s in AnalysisJob.Status]
    # ③ 입력 방식별 건수·평균 처리 시간
    by_input = {r["input_type"]: r for r in jobs.values("input_type").annotate(n=Count("id"), avg_sec=Avg("processing_seconds"))}
    inputs = [{"type": t.value, "label": t.label, "count": by_input.get(t.value, {}).get("n", 0),
               "avg_seconds": round(by_input.get(t.value, {}).get("avg_sec") or 0, 2)} for t in AnalysisJob.InputType]
    return {"actions": actions, "categories": CATEGORY_KO, "statuses": statuses, "inputs": inputs,
            "trend": trend(jobs, period)}


def trend(jobs, period: str):
    """날짜별 작업 수 (완료·실패·그 밖). 빈 날도 0 으로 채운다."""
    tz = timezone.get_current_timezone()
    rows = {r["day"]: r for r in jobs.annotate(day=TruncDate("created_at", tzinfo=tz)).values("day").annotate(
        total=Count("id"), success=Count("id", filter=Q(status=AnalysisJob.Status.SUCCESS)),
        failure=Count("id", filter=Q(status=AnalysisJob.Status.FAILURE)))}
    today = timezone.localdate()
    start = period_start(period)
    first = start.date() if start else (min(rows) if rows else today)
    first = max(first, today - timedelta(days=TREND_MAX_DAYS - 1))
    out = []
    d = first
    while d <= today:
        r = rows.get(d, {})
        out.append({"day": d.strftime("%m-%d"), "success": r.get("success", 0), "failure": r.get("failure", 0),
                    "other": r.get("total", 0) - r.get("success", 0) - r.get("failure", 0)})
        d += timedelta(days=1)
    return out


def confusions(jobs, limit: int = 6):
    """수행한 동작(사용자 선택) → 예측이 달랐던 쌍. 어떤 동작끼리 헷갈리는지 보는 참고용."""
    rows = (PersonPrediction.objects.filter(job__in=jobs).graded().exclude(action_id=F("job__expected_action"))
            .values("job__expected_action", "action_id").annotate(n=Count("id")).order_by("-n")[:limit])
    return [{"expected": r["job__expected_action"], "expected_ko": action_name(r["job__expected_action"]),
             "predicted": r["action_id"], "predicted_ko": action_name(r["action_id"]), "count": r["n"]} for r in rows]


def page_of(jobs, params):
    qs = jobs.with_results().order_by("-created_at")
    page = Paginator(qs, PAGE_SIZE).get_page(params.get("page"))
    keep = params.copy()
    keep.pop("page", None)
    return page, keep.urlencode()


CSV_HEADER = ["작업 번호", "회원", "제목", "입력 방식", "영상 확보 방법", "카메라", "상태", "수행한 동작(사용자 선택)",
              "사람 수", "예측(사람별)", "판단 불확실 포함", "수행한 동작과 일치", "영상 길이(초)", "AI 처리 시간(초)",
              "요청 시각", "완료 시각", "오류 사유", "메모"]


def _cell(text) -> str:
    """회원이 적은 글이 =, +, -, @ 로 시작하면 Excel 이 수식으로 실행할 수 있어 앞에 ' 를 붙인다 (CSV injection)"""
    text = str(text or "")
    return "'" + text if text[:1] in ("=", "+", "-", "@", "\t", "\r") else text


def csv_response(jobs):
    resp = HttpResponse(content_type="text/csv; charset=utf-8")
    stamp = timezone.localtime().strftime("%Y%m%d_%H%M")
    resp["Content-Disposition"] = f'attachment; filename="analysis_jobs_{stamp}.csv"'
    resp.write("﻿")                                   # Excel 에서 한글이 깨지지 않게 (BOM)
    w = csv.writer(resp)
    w.writerow(CSV_HEADER)
    local = lambda t: timezone.localtime(t).strftime("%Y-%m-%d %H:%M:%S") if t else ""  # noqa: E731
    for j in jobs.with_results().order_by("-created_at"):
        preds = list(j.predictions.all())
        match = ""
        if j.expected_action and preds:
            match = "일치" if any(p.action_id == j.expected_action for p in preds) else "불일치"
        w.writerow([j.pk, _cell(j.user.username), _cell(j.title), j.get_input_type_display(),
                    j.get_source_display() if j.source else "", _cell(j.camera_name), j.get_status_display(), j.expected_action, len(preds),
                    " / ".join(f"#{p.track_id} {p.action_id} {action_name(p.action_id)} "
                               f"{p.confidence:.2f}" for p in preds),
                    "예" if any(p.uncertain for p in preds) else "", match, j.video_seconds or "",
                    j.processing_seconds or "", local(j.created_at), local(j.finished_at), _cell(j.error_message), _cell(j.memo)])
    return resp
