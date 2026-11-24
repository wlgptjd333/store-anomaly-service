import analyses.cameras
from django.conf import settings
from django.db import migrations, models


def seed_from_env(apps, schema_editor):
    """처음 한 번: .env 의 IPWEBCAM_CAMERAS 를 카메라 표에 옮긴다. 이후 주소는 웹(관리자)에서 바꾼다."""
    Camera = apps.get_model("analyses", "Camera")
    for name, url in getattr(settings, "IPWEBCAM_CAMERAS", {}).items():
        Camera.objects.get_or_create(name=name[:50], defaults={"url": url[:200]})


class Migration(migrations.Migration):

    dependencies = [
        ("analyses", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="Camera",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("name", models.SlugField(help_text="화면에 보이는 이름 (영문·숫자·-_)", unique=True, verbose_name="이름")),
                ("url", models.CharField(help_text="IP Webcam 앱 화면의 주소. 예: 192.168.0.10:8080 → http://192.168.0.10:8080/video", max_length=200, validators=[analyses.cameras.validate_camera_url], verbose_name="영상 주소")),
                ("enabled", models.BooleanField(default=True, verbose_name="사용")),
                ("updated_at", models.DateTimeField(auto_now=True, verbose_name="수정 시각")),
            ],
            options={"verbose_name": "카메라", "verbose_name_plural": "카메라", "ordering": ["name"]},
        ),
        migrations.RunPython(seed_from_env, migrations.RunPython.noop),
    ]
