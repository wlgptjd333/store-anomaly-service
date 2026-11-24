"""30단계 Sequence 에 덧붙이는 Feature (2026-11-24, 19번 실험에서 고름).

학습(15번)과 웹 서비스(ai_service/src/sequence_extras.py, 같은 파일 복사)가 같은 계산을 써야 한다.
입력 X 는 (n, steps × F) — t00 의 F개, t01 의 F개 … 순서 (rf_feature_columns 와 같음).

  velocity : 이웃한 단계 사이 변화량 (steps-1) × F — 움직임의 속도·방향
  stats    : Feature 마다 30단계의 평균·표준편차·최솟값·최댓값·범위 5 × F — 동작이 '언제' 일어났는지에 덜 민감

왜 필요했나: 영상 전체(whole)를 30단계로 펴면 같은 동작도 사람마다 시작 시점이 달라 같은 단계 번호에 다른 자세가 온다.
단계 번호와 무관한 요약값을 더하니 S01~S04 4-fold 가 71.9% → 82.4% 로 올랐다 (docs/BASELINE_EVALUATION.md 5-3절).
"""
from __future__ import annotations

import numpy as np

KINDS = ("velocity", "stats")
STATS = ("mean", "std", "min", "max", "range")


def extra_names(frame_features: list[str], steps: int, extras: list[str]) -> list[str]:
    names: list[str] = []
    if "velocity" in extras:
        names += [f"d{i:02d}_{f}" for i in range(steps - 1) for f in frame_features]
    if "stats" in extras:
        names += [f"{s}_{f}" for s in STATS for f in frame_features]
    return names


def add_extras(X: np.ndarray, steps: int, n_frame: int, extras: list[str]) -> np.ndarray:
    unknown = set(extras) - set(KINDS)
    if unknown:
        raise ValueError(f"모르는 Feature 종류: {sorted(unknown)}")
    X = np.asarray(X, dtype=float)
    if not extras:
        return X
    seq = X.reshape(len(X), steps, n_frame)
    parts = [X]
    if "velocity" in extras:
        parts.append(np.diff(seq, axis=1).reshape(len(X), -1))
    if "stats" in extras:
        lo, hi = seq.min(1), seq.max(1)
        parts += [seq.mean(1), seq.std(1), lo, hi, hi - lo]
    return np.hstack(parts)
