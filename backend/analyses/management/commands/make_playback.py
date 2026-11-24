"""이미 있는 분석 기록에 재생용 사본(H.264) 만들기.

새 작업은 Celery 가 분석 직후 자동으로 만든다. 이 명령은 기능을 넣기 전 기록이나 사본이 실패한 기록용.
  docker compose exec celery python manage.py make_playback          # 사본 없는 기록만
  docker compose exec celery python manage.py make_playback --force  # 전부 다시
  docker compose exec celery python manage.py make_playback --job 21 27
"""
from django.core.management.base import BaseCommand

from analyses.models import AnalysisJob
from analyses.playback import make_playback


class Command(BaseCommand):
    help = "분석 기록 영상의 브라우저 재생용 사본(H.264)을 만든다"

    def add_arguments(self, parser):
        parser.add_argument("--force", action="store_true", help="이미 있어도 다시 만든다")
        parser.add_argument("--job", type=int, nargs="*", help="이 작업 번호만")

    def handle(self, *args, force=False, job=None, **options):
        jobs = AnalysisJob.objects.exclude(video="").order_by("pk")
        if job:
            jobs = jobs.filter(pk__in=job)
        elif not force:
            jobs = jobs.filter(playback="")
        ok = fail = 0
        for j in jobs:
            if make_playback(j, force=force):
                ok += 1
                self.stdout.write(f"#{j.pk} OK  {j.playback.name}")
            else:
                fail += 1
                self.stdout.write(self.style.WARNING(f"#{j.pk} 건너뜀 (원본 없음·ffmpeg 없음·변환 실패)"))
        self.stdout.write(self.style.SUCCESS(f"완료 {ok}개, 건너뜀 {fail}개"))
