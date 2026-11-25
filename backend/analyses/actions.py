"""모델이 예측하는 17개 동작 — 3팀 통일 번호·이름 (2026-11-24 부터).

AI 서버의 training_info.json(class_info)과 같은 목록이다. Django 는 화면 표시·예상 동작 선택·대시보드 집계에만 쓴다.
번호·이름 원본: training/configs/unified_label_map.csv.
그 저장소 안(영상 이름·실험 표)은 예전 번호(A01..A17)를 그대로 쓰고, 대응은 REPO_ID 로 남긴다.
11/24 전에 저장된 기록은 migration 0004 가 새 번호·이름으로 바꿨다.
"""
# (통일 번호, action_label, 한글 이름, 분류, 학습 데이터 번호)
ACTIONS = [
    ("A001", "kiosk_tap", "키오스크 두드리기", "property_damage", "A01"),
    ("A002", "throw_one_hand", "한손 물건 던지기", "property_damage", "A02"),
    ("A003", "front_kick", "앞차기", "property_damage", "A03"),
    ("A004", "grab_shake", "잡고 흔들기", "property_damage", "A04"),
    ("A005", "push", "밀치기", "violence", "A05"),
    ("A006", "swing", "휘두르기", "violence", "A06"),
    ("A007", "sit_floor", "바닥에 앉기", "loitering", "A07"),
    ("A008", "sit_on_object", "위에 앉기", "loitering", "A08"),
    ("A009", "lie_forward", "앞으로 눕기", "loitering", "A09"),
    ("A010", "lie_backward", "뒤로 눕기", "loitering", "A10"),
    ("A011", "body_slam", "몸통 박치기", "violence", "A11"),
    ("A012", "crouch_take_item", "쪼그려 물건 빼내기", "theft_suspect", "A12"),
    ("A013", "shotgun", "샷건", "property_damage", "A13"),
    ("A014", "vomit", "구토", "emergency", "A14"),
    ("A015", "fall_sideways", "옆으로 쓰러지기", "emergency", "A15"),
    ("A016", "fall_backward", "뒤로 넘어지기", "emergency", "A16"),
    ("A017", "fall_forward", "앞으로 넘어지기", "emergency", "A17"),
]

CATEGORY_KO = {"property_damage": "기물 파손", "violence": "폭력", "loitering": "장시간 머묾",
               "theft_suspect": "도난 의심", "emergency": "응급"}
ACTION_CHOICES = [(a[0], f"{a[0]} {a[2]}") for a in ACTIONS]
ACTION_BY_ID = {a[0]: {"id": a[0], "label": a[1], "name_ko": a[2], "category": a[3], "repo_id": a[4]} for a in ACTIONS}
ACTION_NAMES = {a[0]: a[2] for a in ACTIONS}                      # 번호 → 한글 이름 (화면 json·CSV·대시보드)


def action_name(action_id: str) -> str:
    return ACTION_NAMES.get(action_id, "")


def category_ko(action_id: str) -> str:
    return CATEGORY_KO.get(ACTION_BY_ID.get(action_id, {}).get("category"), "")
