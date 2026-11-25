from __future__ import annotations

# ① 필요한 라이브러리 불러오기
import importlib.util
import json
import sys
from pathlib import Path


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
    import torch                    # 장치 고를 때만 (load_script 만 쓰는 스크립트가 torch 를 안 불러오게)

    # CUDA GPU를 사용할 수 있으면 GPU 0번 선택
    if torch.cuda.is_available():
        return 0

    # CUDA GPU를 사용할 수 없으면 CPU 선택
    return "cpu"


# ⑤ 번호 붙은 스크립트(14_train_baseline.py 등)를 모듈로 불러오기 — 이름이 숫자로 시작해 import 문을 못 씀
def load_script(file: str):
    """scripts/<file> 을 한 번만 실행해 모듈로 돌려줌 (여러 스크립트가 같은 파일을 불러도 다시 실행 안 함)"""
    name = "script_" + Path(file).stem
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / file)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]
