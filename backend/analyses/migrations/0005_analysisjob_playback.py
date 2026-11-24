from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("analyses", "0004_unify_action_ids"),
    ]

    operations = [
        migrations.AddField(
            model_name="analysisjob",
            name="playback",
            field=models.FileField(blank=True, upload_to="playback/", verbose_name="재생용 영상(H.264)"),
        ),
    ]
