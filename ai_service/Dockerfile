FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics
WORKDIR /app

# OpenCV 가 쓰는 시스템 라이브러리
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# PyTorch (PyPI 기본 = CUDA 판, 학습 환경과 같은 2.15). GPU 가 없으면 같은 이미지가 CPU 로 돈다 (project.json device auto)
# Pose 11m 은 CPU 에서 Frame 당 약 0.27초라 GPU 로 돌린다 — compose.gpu.yaml
RUN pip install torch==2.15.0 torchvision
COPY ai_service/requirements.txt .
RUN pip install -r requirements.txt

COPY ai_service/ .
# Feature·모델 코드는 학습과 같은 파일 하나만 쓴다 (training/src — 다르면 정확도가 소리 없이 떨어짐)
COPY training/src/ src/
# Pose 가중치가 없으면 이미지 만들 때 한 번 받아 둔다 (실행 중에는 인터넷이 필요 없게)
RUN test -f models/yolo11m-pose.pt || python -c "from ultralytics import YOLO; import shutil; m=YOLO('yolo11m-pose.pt'); shutil.move('yolo11m-pose.pt','models/yolo11m-pose.pt')"

EXPOSE 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
