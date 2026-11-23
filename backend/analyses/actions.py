"""모델이 예측하는 17개 동작 (unmanned-store-anomaly configs/action_classes.csv 와 같은 번호).

AI 서버의 rf_training_info.json 과 같은 목록이다. Django 는 화면 표시·예상 동작 선택·대시보드 집계에만 쓴다.
"""
ACTIONS = [
    ("A01", "kiosk_slam", "키오스크 내려치기", "property_damage"),
    ("A02", "one_arm_throw", "한 팔로 물건 던지기", "property_damage"),
    ("A03", "freezer_front_kick", "냉동고 앞차기", "property_damage"),
    ("A04", "claw_machine_shake", "뽑기 기계 흔들기", "property_damage"),
    ("A05", "two_hand_shove", "양손으로 밀치기", "violence"),
    ("A06", "two_hand_weapon_swing", "도구 들고 양손 휘두르기", "violence"),
    ("A07", "sit_on_floor_mat", "바닥 매트에 주저앉기", "loitering"),
    ("A08", "jump_onto_chair_phone", "의자에 뛰어올라 앉아 휴대폰 보기", "loitering"),
    ("A09", "lie_prone_phone", "엎드려 휴대폰 보기", "loitering"),
    ("A10", "lie_supine_roll_side", "누워서 휴대폰 보다 옆으로 돌아눕기", "loitering"),
    ("A11", "body_check", "몸통 박치기", "violence"),
    ("A12", "kneel_grab_prize", "무릎 꿇고 뽑기 인형 꺼내기", "theft_suspect"),
    ("A13", "table_slam", "테이블 내려치기", "property_damage"),
    ("A14", "repeated_vomiting", "연속 구토하기", "emergency"),
    ("A15", "sleep_against_wall", "벽에 기대어 잠들기", "emergency"),
    ("A16", "faint_backward", "뒤로 기절하기", "emergency"),
    ("A17", "trip_fall_forward", "걸려 넘어지기 (앞으로)", "emergency"),
]
CATEGORY_KO = {"property_damage": "기물 파손", "violence": "폭력", "loitering": "장시간 머묾",
               "theft_suspect": "도난 의심", "emergency": "응급"}
ACTION_IDS = [a[0] for a in ACTIONS]
ACTION_CHOICES = [(a[0], f"{a[0]} {a[2]}") for a in ACTIONS]
ACTION_BY_ID = {a[0]: {"id": a[0], "label": a[1], "name_ko": a[2], "category": a[3]} for a in ACTIONS}
