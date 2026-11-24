"""모델과 학습 설정을 서버 시작 때 한 번 불러와 모든 요청에서 재사용한다.

세 파일이 서로 맞는지 먼저 확인한다 (틀리면 서버가 '준비 안 됨'으로 뜨고 이유를 /health 에 보여 준다).
  - Joblib 의 n_features_in_  == Feature 순서 JSON 길이 == sequence_steps × Frame Feature 수 + 덧붙인 Feature (src/sequence_extras.py)
  - Joblib 의 classes_        == 학습 정보 JSON 의 classes (순서까지)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib

from src.sequence_extras import extra_names

APP_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = APP_ROOT / "project.json"


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


@dataclass
class ModelBundle:
    config: dict
    model: object = None
    feature_columns: list[str] = field(default_factory=list)
    info: dict = field(default_factory=dict)
    pose_model: object = None
    error: str = ""

    @property
    def ready(self) -> bool:
        return self.model is not None and self.pose_model is not None and not self.error

    @property
    def classes(self) -> list[str]:
        return [str(c) for c in self.model.classes_] if self.model is not None else []

    def summary(self) -> dict:
        return {
            "ready": self.ready,
            "error": self.error,
            "model": self.info.get("model"),
            "feature_set": self.info.get("feature_set"),
            "sequence_extras": self.info.get("sequence_extras", []),
            "window": self.info.get("window"),
            "sequence_steps": self.info.get("sequence_steps"),
            "n_features": len(self.feature_columns),
            "n_classes": len(self.classes),
            "classes": self.classes,
            "trained_on": self.info.get("trained_on"),
            "pose_model": Path(self.config["pose_model"]).name,
        }


def _check(model, columns: list[str], info: dict) -> None:
    steps, frame_feats = info["sequence_steps"], info["frame_features"]
    if model.n_features_in_ != len(columns):
        raise ValueError(f"모델 입력 {model.n_features_in_}개 ≠ Feature 순서 {len(columns)}개")
    # 30단계 × Frame Feature 뒤에 학습 때 덧붙인 Feature(변화량·요약 통계)가 같은 순서로 와야 한다
    extras = info.get("sequence_extras", [])
    expected = [f"t{i:02d}_{f}" for i in range(steps) for f in frame_feats] + extra_names(frame_feats, steps, extras)
    if len(columns) != len(expected):
        raise ValueError(f"Feature {len(columns)}개 ≠ {steps}단계 × {len(frame_feats)} + 덧붙인 {extras}")
    if columns != expected:
        raise ValueError("Feature 순서 JSON 이 학습 정보(frame_features·sequence_extras)와 다릅니다.")
    if [str(c) for c in model.classes_] != info["classes"]:
        raise ValueError("모델 classes_ 와 학습 정보 classes 가 다릅니다.")


def load_bundle(load_pose: bool = True) -> ModelBundle:
    cfg = load_config()
    bundle = ModelBundle(config=cfg)
    try:
        # 본인이 학습해 저장한 파일만 joblib.load 한다 (Pickle 은 임의 코드를 실행할 수 있음)
        bundle.model = joblib.load(APP_ROOT / cfg["model"])
        bundle.feature_columns = json.loads((APP_ROOT / cfg["feature_columns"]).read_text(encoding="utf-8"))
        bundle.info = json.loads((APP_ROOT / cfg["training_info"]).read_text(encoding="utf-8"))
        _check(bundle.model, bundle.feature_columns, bundle.info)
        if load_pose:
            from ultralytics import YOLO
            bundle.pose_model = YOLO(str(APP_ROOT / cfg["pose_model"]))
    except Exception as exc:  # 서버는 띄우고 /health 로 원인을 알린다
        bundle.error = f"{type(exc).__name__}: {exc}"
    return bundle
