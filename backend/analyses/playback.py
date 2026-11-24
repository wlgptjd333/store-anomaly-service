"""분석한 영상을 브라우저에서 다시 보기 — 재생용 사본(H.264) 만들기 + 구간 이동(Range) 응답.

왜 사본을 따로 만드나
  IP Webcam 8초 수집(camera_capture.py)과 실시간 장면 저장(FastAPI live_session.clip_mp4)은 OpenCV 'mp4v'
  (MPEG-4 Part 2)로 저장한다. 분석에는 문제없지만 Chrome·Edge 의 <video> 는 이 코덱을 못 튼다.
  휴대폰 업로드도 HEVC(H.265)면 브라우저마다 다르다. 원본은 그대로 두고(분석·재분석 근거) ffmpeg 로
  H.264 + yuv420p + faststart 사본을 하나 더 만든다 — 대부분 브라우저가 바로 튼다.
  사본은 최대 720p·소리 없음 (사람 동작 확인용이라 충분, 용량은 원본의 1/3~1/10).

왜 Range 를 직접 처리하나
  Chrome 은 서버가 'Accept-Ranges: bytes' 와 206 응답을 주지 않으면 영상 중간으로 건너뛰지 못한다.
  Django FileResponse 는 Range 를 모른다. 영상은 본인·관리자만 봐야 해서 /media/ 를 그대로 열 수도 없다
  → 권한 확인 뒤 이 파일의 ranged_file_response 로 보낸다.
"""
from __future__ import annotations

import logging
import mimetypes
import re
import shutil
import subprocess
import time
from pathlib import Path

from django.conf import settings
from django.http import FileResponse, HttpResponse, StreamingHttpResponse
from django.utils.http import content_disposition_header

log = logging.getLogger(__name__)

PLAYBACK_DIR = "playback"
FFMPEG_TIMEOUT = 180                     # 초. 100 MB 업로드도 CPU 에서 보통 1분 안
RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")
CHUNK = 256 * 1024


def ffmpeg_command(src: Path, dst: Path) -> list[str]:
    return [
        settings.FFMPEG_BINARY, "-y", "-v", "error", "-i", str(src),
        "-map", "0:v:0", "-an",                                   # 첫 영상 트랙만, 소리 없음
        "-vf", "scale=-2:'trunc(min(720,ih)/2)*2'",                # 최대 720p, 가로세로 짝수 (H.264 조건)
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",                                 # 앞부분만 받아도 바로 재생
        str(dst),
    ]


def make_playback(job, force: bool = False) -> bool:
    """job.video → MEDIA_ROOT/playback/job{pk}.mp4. 실패해도 분석 결과에는 영향 없음 (원본으로 대신 보냄)."""
    if not job.video:
        return False
    src = Path(job.video.path)
    if not src.exists():
        return False
    if job.playback and not force and Path(job.playback.path).exists():
        return True
    if not shutil.which(settings.FFMPEG_BINARY):
        log.warning("ffmpeg 가 없어 재생용 사본을 만들지 않습니다 (job %s)", job.pk)
        return False
    rel = Path(PLAYBACK_DIR) / f"job{job.pk}.mp4"
    dst = Path(settings.MEDIA_ROOT) / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part.mp4")
    t0 = time.perf_counter()
    try:
        subprocess.run(ffmpeg_command(src, tmp), check=True, capture_output=True, timeout=FFMPEG_TIMEOUT)
        tmp.replace(dst)
    except (subprocess.SubprocessError, OSError) as exc:
        err = getattr(exc, "stderr", b"") or b""
        log.warning("재생용 사본 실패 job %s: %s %s", job.pk, type(exc).__name__, err[-300:].decode(errors="ignore"))
        tmp.unlink(missing_ok=True)
        return False
    job.playback.name = str(rel)
    job.save(update_fields=["playback"])
    log.info("재생용 사본 job %s: %.1fs, %.1f MB → %.1f MB", job.pk, time.perf_counter() - t0,
             src.stat().st_size / 1e6, dst.stat().st_size / 1e6)
    return True


def _chunks(f, length: int):
    try:
        while length > 0:
            data = f.read(min(CHUNK, length))
            if not data:
                break
            length -= len(data)
            yield data
    finally:
        f.close()


def ranged_file_response(request, path: Path, filename: str, as_attachment: bool = False):
    """Range 요청이면 206 + 요청한 부분만, 아니면 200 전체. 잘못된 범위는 416."""
    size = path.stat().st_size
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    m = RANGE_RE.match(request.headers.get("Range", "").strip())
    if m and (m[1] or m[2]) and not as_attachment:
        if m[1]:
            start, end = int(m[1]), int(m[2]) if m[2] else size - 1
        else:                                                      # bytes=-500 : 끝에서 500바이트
            start, end = max(0, size - int(m[2])), size - 1
        end = min(end, size - 1)
        if start >= size or start > end:
            resp = HttpResponse(status=416)
            resp["Content-Range"] = f"bytes */{size}"
            return resp
        f = open(path, "rb")                                       # noqa: SIM115  _chunks 가 다 보낸 뒤 닫음
        f.seek(start)
        resp = StreamingHttpResponse(_chunks(f, end - start + 1), status=206, content_type=content_type)
        resp["Content-Range"] = f"bytes {start}-{end}/{size}"
        resp["Content-Length"] = str(end - start + 1)
        resp["Content-Disposition"] = content_disposition_header(False, filename)
    else:
        resp = FileResponse(open(path, "rb"), content_type=content_type,  # noqa: SIM115  FileResponse 가 닫음
                            as_attachment=as_attachment, filename=filename)
    resp["Accept-Ranges"] = "bytes"
    resp["Cache-Control"] = "private, max-age=600"
    resp["X-Content-Type-Options"] = "nosniff"
    return resp
