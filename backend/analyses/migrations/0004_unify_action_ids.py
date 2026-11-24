"""동작 번호·이름을 3팀 통일 번호로 (A01 → A001, kiosk_slam → kiosk_tap, 키오스크 내려치기 → 키오스크 두드리기 ...).

1) 번호 칸을 4글자로 늘리고  2) 이미 저장된 기록을 새 번호·이름으로 바꾼다:
   PersonPrediction.action_id·action_label·top3·windows(JSON), AnalysisJob.expected_action,
   AnalysisJob.live_snapshot(실시간 저장 때 사람별 예측), "자동 감지 A03 냉동고 앞차기 .." 제목.
대응표는 이 파일 안에 고정해 둔다 (actions.py 가 나중에 바뀌어도 이 migration 은 그대로 돌게).
되돌리기(migrate analyses 0003)도 된다.
"""

from django.db import migrations, models

# 데이터 저장소 번호 → (통일 번호, 예전 label, 새 label, 예전 이름, 새 이름)
MAP = {
    "A01": ("A001", "kiosk_slam", "kiosk_tap", "키오스크 내려치기", "키오스크 두드리기"),
    "A02": ("A002", "one_arm_throw", "throw_one_hand", "한 팔로 물건 던지기", "한손 물건 던지기"),
    "A03": ("A003", "freezer_front_kick", "front_kick", "냉동고 앞차기", "앞차기"),
    "A04": ("A004", "claw_machine_shake", "grab_shake", "뽑기 기계 흔들기", "잡고 흔들기"),
    "A05": ("A005", "two_hand_shove", "push", "양손으로 밀치기", "밀치기"),
    "A06": ("A006", "two_hand_weapon_swing", "swing", "도구 들고 양손 휘두르기", "휘두르기"),
    "A07": ("A007", "sit_on_floor_mat", "sit_floor", "바닥 매트에 주저앉기", "바닥에 앉기"),
    "A08": ("A008", "jump_onto_chair_phone", "sit_on_object", "의자에 뛰어올라 앉아 휴대폰 보기", "위에 앉기"),
    "A09": ("A009", "lie_prone_phone", "lie_forward", "엎드려 휴대폰 보기", "앞으로 눕기"),
    "A10": ("A010", "lie_supine_roll_side", "lie_backward", "누워서 휴대폰 보다 옆으로 돌아눕기", "뒤로 눕기"),
    "A11": ("A011", "body_check", "body_slam", "몸통 박치기", "몸통 박치기"),
    "A12": ("A012", "kneel_grab_prize", "crouch_take_item", "무릎 꿇고 뽑기 인형 꺼내기", "쪼그려 물건 빼내기"),
    "A13": ("A013", "table_slam", "shotgun", "테이블 내려치기", "샷건"),
    "A14": ("A014", "repeated_vomiting", "vomit", "연속 구토하기", "구토"),
    "A15": ("A015", "sleep_against_wall", "fall_sideways", "벽에 기대어 잠들기", "옆으로 쓰러지기"),
    "A16": ("A016", "faint_backward", "fall_backward", "뒤로 기절하기", "뒤로 넘어지기"),
    "A17": ("A017", "trip_fall_forward", "fall_forward", "걸려 넘어지기 (앞으로)", "앞으로 넘어지기"),
}


def fix_json(value, table) -> bool:
    """JSON 안의 {"action_id": .., "action_label": .., "action_name_ko": ..} 를 모두 바꿈 (안쪽 목록까지). 바뀌었으면 True."""
    changed = False
    if isinstance(value, list):
        for v in value:
            changed |= fix_json(v, table)
    elif isinstance(value, dict):
        if value.get("action_id") in table:
            dst, label, _, name = table[value["action_id"]]
            value["action_id"] = dst
            if "action_label" in value:
                value["action_label"] = label
            if "action_name_ko" in value:
                value["action_name_ko"] = name
            changed = True
        for v in value.values():
            if isinstance(v, (list, dict)):
                changed |= fix_json(v, table)
    return changed


def convert(apps, forward: bool):
    Job = apps.get_model("analyses", "AnalysisJob")
    Pred = apps.get_model("analyses", "PersonPrediction")
    # (찾을 번호 → (바꿀 번호, 바꿀 label, 찾을 이름, 바꿀 이름))
    if forward:
        table = {old: (new, nl, on, nn) for old, (new, ol, nl, on, nn) in MAP.items()}
    else:
        table = {new: (old, ol, nn, on) for old, (new, ol, nl, on, nn) in MAP.items()}
    for src, (dst, label, _, _) in table.items():
        Pred.objects.filter(action_id=src).update(action_id=dst, action_label=label)
        Job.objects.filter(expected_action=src).update(expected_action=dst)
    for pred in Pred.objects.iterator():
        a, b = fix_json(pred.top3, table), fix_json(pred.windows, table)
        if a or b:
            pred.save(update_fields=["top3", "windows"])
    for job in Job.objects.iterator():
        if fix_json(job.live_snapshot, table):
            job.save(update_fields=["live_snapshot"])
    for job in Job.objects.filter(title__startswith="자동 감지 A").iterator():
        for src, (dst, _, src_name, dst_name) in table.items():
            head = f"자동 감지 {src} {src_name} "
            if job.title.startswith(head):
                job.title = (f"자동 감지 {dst} {dst_name} " + job.title[len(head):])[:100]
                job.save(update_fields=["title"])
                break


def forward(apps, schema_editor):
    convert(apps, True)


def backward(apps, schema_editor):
    convert(apps, False)


class Migration(migrations.Migration):

    dependencies = [
        ('analyses', '0003_camera_rotation_job_source'),
    ]

    operations = [
        migrations.AlterField(
            model_name='analysisjob',
            name='expected_action',
            field=models.CharField(blank=True, choices=[('A001', 'A001 키오스크 두드리기'), ('A002', 'A002 한손 물건 던지기'), ('A003', 'A003 앞차기'), ('A004', 'A004 잡고 흔들기'), ('A005', 'A005 밀치기'), ('A006', 'A006 휘두르기'), ('A007', 'A007 바닥에 앉기'), ('A008', 'A008 위에 앉기'), ('A009', 'A009 앞으로 눕기'), ('A010', 'A010 뒤로 눕기'), ('A011', 'A011 몸통 박치기'), ('A012', 'A012 쪼그려 물건 빼내기'), ('A013', 'A013 샷건'), ('A014', 'A014 구토'), ('A015', 'A015 옆으로 쓰러지기'), ('A016', 'A016 뒤로 넘어지기'), ('A017', 'A017 앞으로 넘어지기')], max_length=4, verbose_name='예상 동작(사용자 선택)'),
        ),
        migrations.AlterField(
            model_name='personprediction',
            name='action_id',
            field=models.CharField(choices=[('A001', 'A001 키오스크 두드리기'), ('A002', 'A002 한손 물건 던지기'), ('A003', 'A003 앞차기'), ('A004', 'A004 잡고 흔들기'), ('A005', 'A005 밀치기'), ('A006', 'A006 휘두르기'), ('A007', 'A007 바닥에 앉기'), ('A008', 'A008 위에 앉기'), ('A009', 'A009 앞으로 눕기'), ('A010', 'A010 뒤로 눕기'), ('A011', 'A011 몸통 박치기'), ('A012', 'A012 쪼그려 물건 빼내기'), ('A013', 'A013 샷건'), ('A014', 'A014 구토'), ('A015', 'A015 옆으로 쓰러지기'), ('A016', 'A016 뒤로 넘어지기'), ('A017', 'A017 앞으로 넘어지기')], db_index=True, max_length=4, verbose_name='동작 ID'),
        ),
        migrations.RunPython(forward, backward),
    ]
