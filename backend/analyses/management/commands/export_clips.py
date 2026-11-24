"""분석에 쓴 영상 원본을 폴더 하나로 내보낸다 — 재학습용 데이터 모으기.

왜: 녹화본·업로드 원본은 Docker 볼륨(media) 안에 있어 탐색기로 바로 못 꺼낸다.
    새 장소·카메라 각도에서 찍힌 IP Webcam 녹화는 모델을 더 올릴 때 가장 필요한 데이터라,
    파일 이름만 봐도 무엇인지 알 수 있게 + 목록(manifest.csv)과 함께 꺼낸다.

    docker compose exec django python manage.py export_clips                  # IP Webcam 녹화 전부
    docker compose exec django python manage.py export_clips --source all --labeled-only
    → 컨테이너 /app/exports/<시각>/  =  이 저장소 exports/<시각>/ (compose 에서 연결, Git 제외)

주의
- '수행한 동작'은 사용자가 고른 값이지 검수한 정답이 아니다 → 학습 전에 사람이 영상을 보고 확인 (manifest 의 label_checked 칸).
- 다른 사람 얼굴이 나올 수 있어 공개 저장소에 올리지 않는다.
- 녹화본은 카메라 '화면 회전'을 적용한 뒤 저장한 것이다 (사람이 똑바로 서 보이는 방향).
"""
from __future__ import annotations

import contextlib
import csv
import os
import shutil
from datetime import date, datetime
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from analyses.models import AnalysisJob

SOURCES = {
    "ipwebcam": {"input_type": AnalysisJob.InputType.IPWEBCAM},
    "upload": {"input_type": AnalysisJob.InputType.UPLOAD},
    "all": {},
}
FIELDS = ["file", "job_id", "input_type", "source", "camera", "expected_action", "expected_name",
          "predicted_action", "predicted_name", "confidence", "uncertain", "people", "matches_expected",
          "video_seconds", "created_at", "user", "title", "memo", "original_name", "label_checked"]


def slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in text).strip("-")[:40]


class Command(BaseCommand):
    help = "분석 영상 원본을 폴더로 내보내기 (파일 이름 = 작업·동작·입력 방식, manifest.csv 포함)"

    def add_arguments(self, parser):
        parser.add_argument("--source", choices=list(SOURCES), default="ipwebcam",
                            help="ipwebcam(기본: 녹화·장면 저장·자동 감지) / upload / all")
        parser.add_argument("--labeled-only", action="store_true", help="'수행한 동작'을 고른 기록만")
        parser.add_argument("--since", type=date.fromisoformat, help="이 날짜(YYYY-MM-DD)부터")
        parser.add_argument("--include-failed", action="store_true", help="실패한 작업도 (기본은 완료만)")
        parser.add_argument("--dest", default=str(getattr(settings, "EXPORT_ROOT", "/app/exports")),
                            help="내보낼 상위 폴더 (아래에 시각 이름 폴더를 만든다)")
        parser.add_argument("--dry-run", action="store_true", help="복사하지 않고 몇 개인지만")

    def handle(self, *args, **opts):
        qs = (AnalysisJob.objects.filter(**SOURCES[opts["source"]]).exclude(video="")
              .select_related("user").prefetch_related("predictions").order_by("id"))
        if not opts["include_failed"]:
            qs = qs.filter(status=AnalysisJob.Status.SUCCESS)
        if opts["labeled_only"]:
            qs = qs.exclude(expected_action="")
        if opts["since"]:
            qs = qs.filter(created_at__date__gte=opts["since"])
        jobs = list(qs)
        if not jobs:
            self.stdout.write("내보낼 기록이 없습니다 (조건을 확인하세요).")
            return
        total = sum(Path(j.video.path).stat().st_size for j in jobs if Path(j.video.path).exists())
        self.stdout.write(f"{len(jobs)}개 · {total / 1e6:.1f} MB")
        if opts["dry_run"]:
            return

        out = Path(opts["dest"]) / timezone.localtime().strftime("%Y%m%d_%H%M%S")
        try:
            out.mkdir(parents=True, exist_ok=False)
        except OSError as exc:
            raise CommandError(f"폴더를 만들 수 없습니다: {out} ({exc})") from exc

        rows, missing = [], 0
        for job in jobs:
            src = Path(job.video.path)
            if not src.exists():
                missing += 1
                self.stderr.write(f"  #{job.id}: 파일 없음 {job.video.name}")
                continue
            preds = sorted(job.predictions.all(), key=lambda p: -p.confidence)
            top = preds[0] if preds else None
            kind = job.source or job.input_type
            name = (f"job{job.id:04d}_{job.expected_action or 'NA'}_{kind}_"
                    f"{timezone.localtime(job.created_at):%Y%m%d_%H%M%S}{src.suffix.lower()}")
            shutil.copy2(src, out / name)
            rows.append({
                "file": name, "job_id": job.id, "input_type": job.input_type, "source": job.source,
                "camera": job.camera_name, "expected_action": job.expected_action,
                "expected_name": job.get_expected_action_display().split(" ", 1)[-1] if job.expected_action else "",
                "predicted_action": top.action_id if top else "",
                "predicted_name": top.get_action_id_display().split(" ", 1)[-1] if top else "",
                "confidence": round(top.confidence, 4) if top else "", "uncertain": top.uncertain if top else "",
                "people": len(preds),
                "matches_expected": (top.action_id == job.expected_action) if top and job.expected_action else "",
                "video_seconds": job.video_seconds or "", "created_at": timezone.localtime(job.created_at).isoformat(),
                "user": job.user.username, "title": job.title, "memo": job.memo,
                "original_name": Path(job.video.name).name, "label_checked": "",
            })
        owner = Path(opts["dest"]).stat()                 # 컨테이너는 root 로 돌아서, 받은 폴더 주인에게 넘겨 줘야 WSL·탐색기에서 지울 수 있음
        with (out / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        (out / "README.txt").write_text(
            f"내보낸 시각 {datetime.now():%Y-%m-%d %H:%M} · 조건 source={opts['source']} "
            f"labeled_only={opts['labeled_only']} since={opts['since'] or '-'}\n"
            "파일 이름: job번호_수행한동작(NA=안 고름)_입력방식_촬영시각\n"
            "expected_action 은 사용자가 고른 값(검수 전). 학습에 쓰기 전 영상을 보고 label_checked 칸에 확인 표시.\n"
            "다른 사람 얼굴이 나올 수 있으니 공개 저장소에 올리지 말 것.\n", encoding="utf-8")
        for p in [out, *out.iterdir()]:
            with contextlib.suppress(OSError):
                os.chown(p, owner.st_uid, owner.st_gid)
        self.stdout.write(self.style.SUCCESS(f"{len(rows)}개 내보냄 → {out}" + (f" (파일 없음 {missing})" if missing else "")))
