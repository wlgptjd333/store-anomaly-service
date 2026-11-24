# Django(웹·API)와 Celery Worker 가 같은 이미지를 쓴다 — 실행 명령만 compose.yaml 에서 다르게 준다
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 STATIC_MANIFEST=1
WORKDIR /app

# ffmpeg: 분석한 영상의 브라우저 재생용 사본(H.264) — OpenCV mp4v 녹화본은 Chrome·Edge 가 못 틀어서
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN DJANGO_SECRET_KEY=build-only python manage.py collectstatic --noinput

EXPOSE 8000
CMD ["sh", "-c", "python manage.py migrate --noinput && gunicorn config.wsgi:application -b 0.0.0.0:8000 -w 3 -k gthread --threads 8 --timeout 120"]
