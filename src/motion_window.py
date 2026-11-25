"""움직이는 구간만 골라 보기 — 학습과 서비스가 같이 쓴다 (2026-11-25, 27번 실험).

왜: 학습 영상은 '서 있기 → 동작 → 서 있기'로 찍었고 30단계는 영상 전체(whole)를 편다.
    현장 영상은 바로 동작부터 시작하거나, 걸어 들어오거나, 두 번 반복하거나, 오래 서 있다가 끝난다.
    그러면 같은 동작도 30단계 안에서 놓이는 자리가 학습과 달라진다.
    → 몸 관절이 실제로 움직인 구간(앞뒤 여유 pad 초)만 30단계로 펴면, 영상을 어디서 자르든 같은 모양이 된다.

움직임 크기: 몸 크기로 정규화한 관절 17개(bn_*) 좌표의 초당 변화량 평균 → 0.3초 이동 평균.
    정규화 좌표라 카메라 거리·화면 위치와 무관하고, 사람이 화면에서 걸어가도 몸 전체 이동은 빠진다.
구간: 그 영상 안에서 조용한 수준(10%)과 큰 움직임(95%) 사이의 rel 지점을 넘는 첫 Frame ~ 마지막 Frame, 앞뒤 pad 초.
"""
from __future__ import annotations

import numpy as np

from src.schema import BODY40

_BN = [i for i, f in enumerate(BODY40) if f.startswith("bn_")]
REL, PAD, SMOOTH_SEC, MIN_SEC = 0.3, 0.5, 0.3, 1.0


def motion_energy(times: np.ndarray, values: np.ndarray) -> np.ndarray:
    """(T,) 초당 관절 이동량. values 는 유효 Frame 만 모은 (T, 40) BODY40."""
    t = np.asarray(times, float)
    xy = np.asarray(values, float)[:, _BN].reshape(len(t), -1, 2)
    if len(t) < 3:
        return np.zeros(len(t))
    dt = np.maximum(np.diff(t), 1e-3)
    v = np.linalg.norm(np.diff(xy, axis=0), axis=2).mean(1) / dt
    v = np.r_[v[0], v]
    out = np.empty_like(v)
    j0 = j1 = 0
    for i, ti in enumerate(t):                      # 시간 기준 이동 평균 (FPS 가 달라도 같은 폭)
        while t[j0] < ti - SMOOTH_SEC / 2:
            j0 += 1
        while j1 < len(t) and t[j1] <= ti + SMOOTH_SEC / 2:
            j1 += 1
        out[i] = v[j0:j1].mean()
    return out


def active_window(times: np.ndarray, values: np.ndarray, rel: float = REL, pad: float = PAD) -> tuple[float, float]:
    """움직인 구간 (시작 초, 끝 초). 움직임이 거의 없으면 전체."""
    t = np.asarray(times, float)
    if len(t) < 3:
        return float(t[0]), float(t[-1])
    e = motion_energy(t, values)
    lo, hi = np.percentile(e, 10), np.percentile(e, 95)
    if hi - lo < 1e-6:
        return float(t[0]), float(t[-1])
    on = np.flatnonzero(e > lo + rel * (hi - lo))
    a, z = max(t[0], t[on[0]] - pad), min(t[-1], t[on[-1]] + pad)
    if z - a < MIN_SEC:                               # 너무 짧으면 가운데 기준으로 MIN_SEC 만큼
        c = (a + z) / 2
        a, z = max(t[0], c - MIN_SEC / 2), min(t[-1], c + MIN_SEC / 2)
    return float(a), float(z)
