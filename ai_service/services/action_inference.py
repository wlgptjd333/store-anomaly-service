"""사람별 Frame Feature → 30단계 Sequence (+ 변화량·요약 통계) → ExtraTrees → Track ID 별 Action·Confidence.

창(Window) 규칙
  - 모델은 영상 전체(whole)를 30단계로 편 값으로 학습했다. 학습 영상 길이는 2.2~26.8초, 중앙값 5.5초.
  - 사람이 화면에 있던 시간이 window_seconds(8초) 이하면 그 전체를 창 하나로 본다 (학습과 같은 모양).
  - 더 길면 8초 창을 4초씩 밀며 여러 개 만들고, 창별 확률을 평균해 사람 한 명의 결과로 정한다.
    → 7일차 Sliding Window 와 같은 생각. 창 하나가 우연히 튀어도 평균이 흔들림을 줄인다.
  - 창 안 유효 Frame 이 10개 미만이거나 50% 미만이면 그 창은 쓰지 않는다 (학습 때 13번 기준과 같음).
"""
from __future__ import annotations

import numpy as np

from services.pose_pipeline import TrackData
from src.sequence_extras import add_extras


def resample(values: np.ndarray, times: np.ndarray, steps: int) -> np.ndarray:
    """(T, F) → (steps, F). 시간을 0~1 로 펴고 고른 간격으로 보간 (학습 13번 resample 과 같음)."""
    span = times[-1] - times[0]
    x = (times - times[0]) / span
    tx = np.linspace(0.0, 1.0, steps)
    return np.stack([np.interp(tx, x, values[:, j]) for j in range(values.shape[1])], 1)


def make_windows(t_start: float, t_end: float, size: float, stride: float) -> list[tuple[float, float]]:
    if t_end - t_start <= size:
        return [(t_start, t_end)]
    starts = list(np.arange(t_start, t_end - size, stride))
    starts.append(t_end - size)                      # 마지막 창은 끝에 맞춤
    return [(float(s), float(s) + size) for s in starts]


def predict_track(track: TrackData, bundle) -> dict:
    cfg, info = bundle.config, bundle.info
    rule = info["valid_frame_rule"]
    steps = info["sequence_steps"]
    times = np.asarray(track.times, dtype=float)
    feats = np.stack(track.features)
    valid = np.asarray(track.valid, dtype=bool)
    span = float(times[-1] - times[0]) if len(times) > 1 else 0.0
    quality = {"frames": int(len(times)), "valid_frames": int(valid.sum()),
               "valid_ratio": round(float(valid.mean()), 3) if len(valid) else 0.0,
               "track_seconds": round(span, 2),
               "mean_keypoint_conf": round(float(np.mean(track.keypoint_conf)), 3)}
    if span < cfg["min_track_seconds"]:
        return {"skipped": True, "reason": f"화면에 {span:.1f}초만 보여 판단하지 않음", **quality}

    probs, windows = [], []
    for s, e in make_windows(float(times[0]), float(times[-1]), cfg["window_seconds"], cfg["window_stride_seconds"]):
        m = (times >= s - 1e-9) & (times <= e + 1e-9)
        sel = m & valid
        n_all, n_valid = int(m.sum()), int(sel.sum())
        if n_valid < rule["min_valid_frames"] or n_valid / max(n_all, 1) < rule["min_valid_ratio"]:
            continue
        seq = resample(feats[sel], times[sel], steps).reshape(1, -1)
        seq = add_extras(seq, steps, feats.shape[1], info.get("sequence_extras", []))  # 학습 때와 같은 덧붙인 Feature
        p = bundle.model.predict_proba(seq)[0]
        probs.append(p)
        k = int(np.argmax(p))
        windows.append({"start_sec": round(s, 2), "end_sec": round(e, 2), "action_id": bundle.classes[k],
                        "confidence": round(float(p[k]), 4)})
    if not probs:
        return {"skipped": True, "reason": "관절이 잘 보이는 Frame 이 부족함 (창마다 유효 Frame 10개·50% 미만)",
                **quality}

    mean_p = np.mean(probs, axis=0)
    order = np.argsort(mean_p)[::-1]
    top = order[0]
    cls = info["class_info"]

    def describe(i: int) -> dict:
        aid = bundle.classes[i]
        return {"action_id": aid, "action_label": cls[aid]["label"], "action_name_ko": cls[aid]["name_ko"],
                "probability": round(float(mean_p[i]), 4)}

    best = describe(top)
    return {
        "skipped": False,
        "action_id": best["action_id"],
        "action_label": best["action_label"],
        "action_name_ko": best["action_name_ko"],
        "category": cls[best["action_id"]].get("category", ""),
        "confidence": best["probability"],
        "uncertain": best["probability"] < cfg["uncertain_below"],
        "window_count": len(probs),
        "top3": [describe(int(i)) for i in order[:3]],
        "windows": windows,
        **quality,
    }


def predict_tracks(video, bundle) -> tuple[list[dict], list[dict]]:
    """(predictions, skipped). Track ID 순서로 정렬."""
    preds, skipped = [], []
    for tid in sorted(video.tracks):
        r = predict_track(video.tracks[tid], bundle)
        (skipped if r.pop("skipped") else preds).append({"track_id": tid, **r})
    return preds, skipped
