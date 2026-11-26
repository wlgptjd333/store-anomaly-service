"""움직임 요약 86개 Feature — 팀 모델 비교(11/26)의 공동 1위 팀원 모델에서 가져옴 (2026-11-26, 28번 실험).

원본: 팀원 vision_action_platform ai_service/src/feature_utils.py 의 extract_kinematic_summary_86d (계산은 그대로).
입력은 Frame 하나의 앞 38개 = 골반 중심·몸 크기로 정규화한 관절 34개(bn_*) + 팔꿈치·무릎 각도 4개(도).
이 저장소 BODY40 의 앞 38개와 순서·뜻이 같다 (몸 크기 기준만 조금 다름 — 86개는 다시 몸통 길이로 나누므로 영향이 작다).

왜: 30단계 좌표·변화량·요약 통계(지금 2,560개)는 관절 값 하나하나를 본다.
    86개는 두 손이 같이 움직이는지(상관), 팔을 몇 번 폈다 굽혔는지(피크 수), 손이 코 위로 갔는지처럼
    동작을 가르는 관계·횟수를 요약한다. 팀원 모델은 이것과 손으로 만든 규칙으로 다른 카메라 구도에서 강했다.
    규칙은 가져오지 않고 Feature 만 더해 모델이 배우게 한다.

numpy·표준 라이브러리만 쓴다 (팀 저장소 추론 파일에 그대로 옮길 수 있게). 원본의 scipy.signal.find_peaks 는
아래 find_peaks 로 바꿨고, 28번 학습 행 전부(+ 평평한 값 무작위 8만 개)에서 scipy 와 결과가 같음을 확인함.
"""
from __future__ import annotations

import numpy as np

KIN86_NAMES = [f"kin_summary_{i:02d}" for i in range(86)]


def kinematic_summary_batch(X: np.ndarray, steps: int, n_frame: int) -> np.ndarray:
    """(n, steps × n_frame) 펼친 Sequence → (n, 86). 원본이 30단계 기준이라 steps 는 30 이어야 한다."""
    if steps != 30 or n_frame < 38:
        raise ValueError("kin86 은 30단계 · Frame Feature 앞 38개(관절 34 + 각도 4)가 필요합니다.")
    seq = np.asarray(X, dtype=np.float32).reshape(len(X), steps, n_frame)[:, :, :38]
    if not len(seq):
        return np.zeros((0, 86), dtype=np.float32)
    return np.stack([extract_kinematic_summary_86d(s.reshape(-1)) for s in seq])


def find_peaks(x, height=None, prominence=None):
    """scipy.signal.find_peaks(x, height=h, prominence=p) 와 같은 결과 (numpy·표준 라이브러리만, 1D, 최솟값 조건만).

    scipy 순서 그대로: 국소 최댓값(평평한 꼭대기는 가운데) → height 이상 → prominence 이상 (wlen 없음 = 전체 구간).
    돌려주는 값도 scipy 처럼 (peak 위치 배열, dict). 학습 데이터 전체에서 scipy 와 같은지 28번에서 확인함.
    """
    x = np.asarray(x)
    n = len(x)
    peaks = []
    i = 1
    while i < n - 1:
        if x[i - 1] < x[i]:
            ahead = i + 1
            while ahead < n - 1 and x[ahead] == x[i]:
                ahead += 1
            if x[ahead] < x[i]:
                peaks.append((i + ahead - 1) // 2)
                i = ahead
        i += 1
    if height is not None:
        peaks = [p for p in peaks if x[p] >= height]
    if prominence is not None:
        keep = []
        for p in peaks:
            left_min, j = x[p], p
            while j >= 0 and x[j] <= x[p]:
                left_min = min(left_min, x[j])
                j -= 1
            right_min, j = x[p], p
            while j <= n - 1 and x[j] <= x[p]:
                right_min = min(right_min, x[j])
                j += 1
            if x[p] - max(left_min, right_min) >= prominence:
                keep.append(p)
        peaks = keep
    return np.asarray(peaks, dtype=np.intp), {}


def extract_kinematic_summary_86d(seq_1140: np.ndarray) -> np.ndarray:
    """
    Day 03 정규화 관절 좌표 및 4대 관절 각도 시퀀스(30x38=1,140차원)에서
    동작 발생 시점(Shift) 및 좌우 방향(Flip)·카메라 각도에 불변인 86차원 물리 운동학 요약 특징을 추출합니다.
    """
    frames = np.asarray(seq_1140, dtype=np.float32).reshape((30, 38))
    coords = frames[:, :34].reshape((30, 17, 2))
    sh_mid = (coords[:, 5, :] + coords[:, 6, :]) / 2.0
    torso_s = np.maximum(np.linalg.norm(sh_mid, axis=1), 0.40)[:, None, None]
    tc = coords / torso_s

    l_elb, r_elb = frames[:, 34], frames[:, 35]
    l_knee, r_knee = frames[:, 36], frames[:, 37]
    avg_elb = (l_elb + r_elb) / 2.0
    min_elb_lr = np.minimum(l_elb, r_elb)
    max_elb_lr = np.maximum(l_elb, r_elb)
    avg_knee = (l_knee + r_knee) / 2.0
    min_knee_lr = np.minimum(l_knee, r_knee)
    max_knee_lr = np.maximum(l_knee, r_knee)

    sp = (tc[:, 5, :] + tc[:, 6, :]) / 2.0
    sp_tilt = np.degrees(np.arctan2(np.abs(sp[:, 0]), np.maximum(-sp[:, 1], 1e-3)))
    sh_y, sh_x = sp[:, 1], sp[:, 0]

    nose_x, nose_y = tc[:, 0, 0], tc[:, 0, 1]
    nose_rel_sh_y = nose_y - sh_y

    l_wx, l_wy = tc[:, 9, 0], tc[:, 9, 1]
    r_wx, r_wy = tc[:, 10, 0], tc[:, 10, 1]
    w_mid_x, w_mid_y = (l_wx + r_wx) / 2.0, (l_wy + r_wy) / 2.0
    w_dist = np.hypot(l_wx - r_wx, l_wy - r_wy)

    l_wy_sh, r_wy_sh = l_wy - sh_y, r_wy - sh_y
    w_rel_nose_y = w_mid_y - nose_y
    min_single_w_rel_nose = np.minimum(l_wy, r_wy) - nose_y

    act_wy_span = max(float(np.ptp(l_wy)), float(np.ptp(r_wy)))
    sec_wy_span = min(float(np.ptp(l_wy)), float(np.ptp(r_wy)))
    two_hand_y_ratio = sec_wy_span / max(act_wy_span, 1e-4)

    act_wx_span = max(float(np.ptp(l_wx)), float(np.ptp(r_wx)))
    sec_wx_span = min(float(np.ptp(l_wx)), float(np.ptp(r_wx)))
    two_hand_x_ratio = sec_wx_span / max(act_wx_span, 1e-4)

    wy_corr = float(np.corrcoef(l_wy, r_wy)[0, 1]) if (np.std(l_wy) > 1e-3 and np.std(r_wy) > 1e-3) else 0.0
    wx_corr = float(np.corrcoef(l_wx, r_wx)[0, 1]) if (np.std(l_wx) > 1e-3 and np.std(r_wx) > 1e-3) else 0.0

    l_ky = tc[:, 13, 1]
    r_ky = tc[:, 14, 1]
    l_ax, l_ay = tc[:, 15, 0], tc[:, 15, 1]
    r_ax, r_ay = tc[:, 16, 0], tc[:, 16, 1]
    ank_mid_y = (l_ay + r_ay) / 2.0
    knee_mid_y = (l_ky + r_ky) / 2.0
    ank_asym_y = np.abs(l_ay - r_ay)
    ank_sep_2d = np.hypot(l_ax - r_ax, l_ay - r_ay)
    knee_asym_y = np.abs(l_ky - r_ky)

    # 측면 낙상(fall_sideways) vs 전후방 낙상 구분용 좌우 롤(Roll) 비대칭 특징
    sh_roll_y = np.abs(tc[:, 5, 1] - tc[:, 6, 1])
    hip_roll_y = np.abs(tc[:, 11, 1] - tc[:, 12, 1])
    ear_roll_y = np.abs(tc[:, 3, 1] - tc[:, 4, 1])
    sh_width_tc = np.hypot(tc[:, 5, 0] - tc[:, 6, 0], tc[:, 5, 1] - tc[:, 6, 1])

    d_tc = np.diff(tc, axis=0)
    w_speed = np.maximum(np.linalg.norm(d_tc[:, 9, :], axis=1), np.linalg.norm(d_tc[:, 10, :], axis=1))
    a_speed = np.maximum(np.linalg.norm(d_tc[:, 15, :], axis=1), np.linalg.norm(d_tc[:, 16, :], axis=1))
    n_speed = np.linalg.norm(d_tc[:, 0, :], axis=1)
    d_elb = np.abs(np.diff(avg_elb))
    d_wy_mid = np.diff(w_mid_y)
    slam_3step = w_mid_y[3:] - w_mid_y[:-3]  # 양수 = 3스텝 동안 급격한 양손 하강(샷건!)

    peaks_elb, _ = find_peaks(avg_elb, prominence=12.0, height=95.0)
    valleys_elb, _ = find_peaks(-avg_elb, prominence=12.0)
    d_elb_signed = np.diff(avg_elb)
    elb_reversals = int(np.sum((d_elb_signed[:-1] * d_elb_signed[1:]) < -4.0))
    wy_reversals = int(np.sum((d_wy_mid[:-1] * d_wy_mid[1:]) < -0.002))

    sk_w = np.max(tc[:, :, 0], axis=1) - np.min(tc[:, :, 0], axis=1)
    sk_h = np.max(tc[:, :, 1], axis=1) - np.min(tc[:, :, 1], axis=1)
    wh_ratio = sk_w / np.maximum(sk_h, 0.2)

    s_feats = [
        # 1. 팔꿈치 각도 및 진동 (12)
        float(np.min(avg_elb)), float(np.max(avg_elb)), float(np.mean(avg_elb)), float(np.std(avg_elb)),
        float(np.min(min_elb_lr)), float(np.max(max_elb_lr)),
        float(np.sum(d_elb)), float(np.max(d_elb)),
        float(len(peaks_elb)), float(len(valleys_elb)), float(elb_reversals),
        float(np.sum(avg_elb > 110.0)),

        # 2. 무릎 각도 및 하체 비대칭 (14)
        float(np.min(min_knee_lr)), float(np.percentile(min_knee_lr, 10)), float(np.mean(avg_knee)), float(np.std(avg_knee)),
        float(np.median(max_knee_lr)), float(np.max(max_knee_lr - min_knee_lr)),
        float(np.min(ank_mid_y)), float(np.max(ank_mid_y)), float(np.mean(ank_mid_y)), float(np.ptp(ank_mid_y)),
        float(np.min(np.minimum(l_ay, r_ay))), float(np.max(ank_asym_y)), float(np.max(ank_sep_2d)), float(np.max(knee_asym_y)),

        # 3. 손목 궤적, 양손 대칭성, 머리 위 타격 충격량 (24)
        float(np.min(w_mid_y)), float(np.max(w_mid_y)), float(np.mean(w_mid_y)), float(np.ptp(w_mid_y)),
        float(np.ptp(w_mid_x)), float(np.std(w_mid_x)),
        act_wy_span, sec_wy_span, two_hand_y_ratio,
        act_wx_span, sec_wx_span, two_hand_x_ratio,
        wy_corr, wx_corr,
        float(np.min(np.minimum(l_wy_sh, r_wy_sh))), float(np.min(np.maximum(l_wy_sh, r_wy_sh))),
        float(np.max(np.maximum(l_wy, r_wy))), float(np.min(w_dist)), float(np.max(w_dist)),
        float(wy_reversals),
        float(np.min(w_rel_nose_y)), float(np.min(min_single_w_rel_nose)),
        float(np.max(slam_3step)), float(np.min(slam_3step)),

        # 4. 척추 기울기, 좌우 롤 비대칭, 종횡비 및 관절 속도 (26)
        float(np.min(nose_y)), float(np.max(nose_y)), float(np.mean(nose_y)), float(np.ptp(nose_y)),
        float(np.ptp(nose_x)), float(np.max(nose_rel_sh_y)),
        float(np.max(sp_tilt)), float(np.percentile(sp_tilt, 85)), float(np.mean(sp_tilt)),
        float(np.max(wh_ratio)), float(np.percentile(wh_ratio, 85)), float(np.min(wh_ratio)),
        float(np.max(w_speed)), float(np.mean(w_speed)),
        float(np.max(a_speed)), float(np.mean(a_speed)),
        float(np.max(n_speed)), float(np.max(d_wy_mid)),
        float(np.max(sh_roll_y)), float(np.mean(sh_roll_y)),
        float(np.max(hip_roll_y)), float(np.mean(hip_roll_y)),
        float(np.max(ear_roll_y)), float(np.min(sh_width_tc)),
        float(np.max(sh_width_tc)), float(np.ptp(sh_x)),

        # 5. 시작-중반-종료 자세 변화량 (10)
        float(np.mean(nose_y[10:20]) - np.mean(nose_y[:5])),
        float(np.mean(nose_y[24:]) - np.mean(nose_y[:5])),
        float(np.mean(ank_mid_y[10:20]) - np.mean(ank_mid_y[:5])),
        float(np.mean(ank_mid_y[24:]) - np.mean(ank_mid_y[:5])),
        float(np.mean(avg_knee[10:20]) - np.mean(avg_knee[:5])),
        float(np.mean(avg_knee[24:]) - np.mean(avg_knee[:5])),
        float(np.mean(sp_tilt[10:20]) - np.mean(sp_tilt[:5])),
        float(np.mean(sp_tilt[24:]) - np.mean(sp_tilt[:5])),
        float(np.mean(knee_mid_y[10:20])),
        float(np.max(np.abs(sh_x))),
    ]
    return np.nan_to_num(np.asarray(s_feats, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
