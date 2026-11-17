from __future__ import annotations

# ① 필요한 라이브러리 불러오기
import json
from pathlib import Path
import torch


# ② 프로젝트 최상위 폴더와 설정 파일의 경로 지정
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "project.json"


# ③ JSON 설정 파일 읽기
def load_config() -> dict:
    """project.json을 읽어 Python 딕셔너리로 반환합니다."""
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


# ④ GPU 또는 CPU 선택
def resolve_device(requested: str):
    """auto이면 CUDA 사용 가능 여부에 따라 GPU 또는 CPU를 선택합니다."""

    # auto가 아니면 사용자가 지정한 장치를 그대로 사용
    if requested != "auto":
        return requested

    # CUDA GPU를 사용할 수 있으면 GPU 0번 선택
    if torch.cuda.is_available():
        return 0

    # CUDA GPU를 사용할 수 없으면 CPU 선택
    return "cpu"
