"""과제 테스트 중 서버 없이 확인할 수 있는 것 (T02·T04·T05·T06·T07·T08·T09 일부).

FastAPI 와 카메라는 가짜(mock)로 바꾸고, Celery 작업은 on_commit 콜백을 직접 실행해 같은 흐름을 탄다.
실제 서버·Docker 를 거친 확인은 README 의 테스트 표에 따로 기록한다.
"""
import csv
import io
import shutil
import tempfile
from datetime import timedelta
from pathlib import Path
from unittest import mock

from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.utils import timezone
from rest_framework.test import APIClient

from .camera_capture import CaptureError
from .models import AnalysisJob, Camera, PersonPrediction
from .services import InferenceError
from .tasks import run_analysis

TMP_MEDIA = tempfile.mkdtemp()


def fake_result(n_people=2):
    preds = [{"track_id": i + 1, "action_id": aid, "action_label": label, "action_name_ko": "x", "confidence": conf,
              "uncertain": conf < 0.3, "window_count": 1, "valid_frames": 150, "track_seconds": 5.0,
              "mean_keypoint_conf": 0.9, "top3": [{"action_id": aid, "probability": conf}], "windows": []}
             for i, (aid, label, conf) in enumerate([("A001", "kiosk_tap", 0.77), ("A014", "vomit", 0.25)][:n_people])]
    return {"ok": True, "message": "" if preds else "영상에서 사람을 찾지 못했습니다.",
            "video": {"seconds": 5.2, "frames": 156}, "processing_seconds": 3.1, "predictions": preds}


def video_file(name="clip.mp4"):
    return SimpleUploadedFile(name, b"\x00\x00\x00\x18ftypmp42fake", content_type="video/mp4")


class CleanTestCase(TestCase):
    """migration 0002 가 .env 의 IPWEBCAM_CAMERAS 를 테스트 DB 에도 옮겨 넣는다.
    그대로 두면 Docker 안(.env 있음)과 밖(없음)에서 결과가 달라지므로, 모든 테스트는 빈 카메라 표에서 시작한다."""

    @classmethod
    def setUpTestData(cls):
        Camera.objects.all().delete()


@override_settings(MEDIA_ROOT=TMP_MEDIA)
class AnalysisApiTests(CleanTestCase):
    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        shutil.rmtree(TMP_MEDIA, ignore_errors=True)

    def setUp(self):
        self.a = User.objects.create_user("user_a", password="pw-A-12345")
        self.b = User.objects.create_user("user_b", password="pw-B-12345")
        self.api = APIClient()
        self.api.force_authenticate(self.a)
        Camera.objects.create(name="phone", url="http://192.168.40.10:8080/video")

    def create(self, data, result=None, error=None):
        """POST 후 transaction.on_commit 으로 등록된 Celery 작업을 그 자리에서 실행"""
        side = error if error else None
        with mock.patch("analyses.tasks.predict_video", return_value=result or fake_result(),
                        side_effect=side) as pv, \
                mock.patch("analyses.jobs.run_analysis.apply_async",
                           side_effect=lambda args, task_id: run_analysis(*args)):
            with self.captureOnCommitCallbacks(execute=True):
                r = self.api.post("/api/analyses/", data, format="multipart")
        return r, pv

    # T02·T05: 업로드 → 작업 ID 즉시 반환(202) → Worker 가 결과 저장
    def test_upload_returns_job_id_and_saves_result(self):
        r, pv = self.create({"input_type": "UPLOAD", "video": video_file(), "expected_action": "A001"})
        self.assertEqual(r.status_code, 202)
        self.assertIn("task_id", r.data)
        job = AnalysisJob.objects.get(pk=r.data["id"])
        self.assertEqual(job.status, "SUCCESS")
        self.assertEqual(job.celery_task_id, r.data["task_id"])
        self.assertEqual(job.user, self.a)
        pv.assert_called_once()

    # T04: 두 사람 → 한 작업에 Track ID 별 결과 2개
    def test_two_people_saved_separately(self):
        r, _ = self.create({"input_type": "UPLOAD", "video": video_file()})
        job = AnalysisJob.objects.get(pk=r.data["id"])
        self.assertEqual(list(job.predictions.values_list("track_id", "action_id")), [(1, "A001"), (2, "A014")])
        self.assertTrue(job.predictions.get(track_id=2).uncertain)

    # T06: 새로 조회해도(새로고침) 같은 결과
    def test_result_persists_in_detail_and_status(self):
        r, _ = self.create({"input_type": "UPLOAD", "video": video_file()})
        d = self.api.get(f"/api/analyses/{r.data['id']}/").data
        self.assertEqual(len(d["predictions"]), 2)
        s = self.api.get(f"/api/analyses/{r.data['id']}/status/").data
        self.assertEqual((s["status"], s["is_finished"], s["prediction_count"]), ("SUCCESS", True, 2))

    def test_no_person_is_success_with_message(self):
        r, _ = self.create({"input_type": "UPLOAD", "video": video_file()}, result=fake_result(0))
        job = AnalysisJob.objects.get(pk=r.data["id"])
        self.assertEqual((job.status, job.predictions.count()), ("SUCCESS", 0))
        self.assertIn("사람", job.result_message)

    # T09: 잘못된 파일 · AI 서버 오류 · 카메라 끊김
    def test_wrong_extension_rejected(self):
        r, pv = self.create({"input_type": "UPLOAD", "video": SimpleUploadedFile("a.txt", b"hello")})
        self.assertEqual(r.status_code, 400)
        self.assertIn("video", r.data)
        pv.assert_not_called()

    def test_inference_error_saved_as_failure(self):
        r, _ = self.create({"input_type": "UPLOAD", "video": video_file()},
                           error=InferenceError("AI 서버 오류 (400): 영상 파일을 열 수 없습니다."))
        job = AnalysisJob.objects.get(pk=r.data["id"])
        self.assertEqual(job.status, "FAILURE")
        self.assertIn("열 수 없습니다", job.error_message)
        self.assertIsNotNone(job.finished_at)

    def test_queue_down_saved_as_failure(self):
        with mock.patch("analyses.jobs.run_analysis.apply_async", side_effect=ConnectionError("redis down")):
            with self.captureOnCommitCallbacks(execute=True):
                r = self.api.post("/api/analyses/", {"input_type": "UPLOAD", "video": video_file()}, format="multipart")
        self.assertEqual(r.status_code, 202)
        job = AnalysisJob.objects.get(pk=r.data["id"])
        self.assertEqual(job.status, "FAILURE")
        self.assertIn("Redis", job.error_message)

    def test_ipwebcam_only_allowed_camera(self):
        r, _ = self.create({"input_type": "IPWEBCAM", "camera_name": "http://evil.example/video"})
        self.assertEqual(r.status_code, 400)
        self.assertIn("camera_name", r.data)

    def test_ipwebcam_disconnect_saved_as_failure(self):
        with mock.patch("analyses.tasks.capture_clip", side_effect=CaptureError("카메라 'phone'에 연결하지 못했습니다.")):
            r, pv = self.create({"input_type": "IPWEBCAM", "camera_name": "phone"})
        job = AnalysisJob.objects.get(pk=r.data["id"])
        self.assertEqual((job.status, job.input_type), ("FAILURE", "IPWEBCAM"))
        self.assertIn("연결하지 못했습니다", job.error_message)
        pv.assert_not_called()

    # T07: B 는 A 의 기록을 조회·수정·삭제할 수 없음
    def test_other_user_cannot_access(self):
        r, _ = self.create({"input_type": "UPLOAD", "video": video_file()})
        url = f"/api/analyses/{r.data['id']}/"
        other = APIClient()
        other.force_authenticate(self.b)
        self.assertEqual(other.get(url).status_code, 404)
        self.assertEqual(other.patch(url, {"title": "hack"}).status_code, 404)
        self.assertEqual(other.delete(url).status_code, 404)
        self.assertEqual(other.get(url + "status/").status_code, 404)
        self.assertEqual(other.get("/api/analyses/").data["count"], 0)
        self.assertTrue(AnalysisJob.objects.filter(pk=r.data["id"]).exists())

    def test_patch_changes_only_title_and_memo(self):
        r, _ = self.create({"input_type": "UPLOAD", "video": video_file()})
        url = f"/api/analyses/{r.data['id']}/"
        res = self.api.patch(url, {"title": "1차 시연", "memo": "정면", "status": "FAILURE"}, format="json")
        self.assertEqual(res.status_code, 200)
        job = AnalysisJob.objects.get(pk=r.data["id"])
        self.assertEqual((job.title, job.memo, job.status), ("1차 시연", "정면", "SUCCESS"))
        self.assertEqual(self.api.put(url, {"title": "x"}, format="json").status_code, 405)

    def test_delete_removes_predictions_and_file(self):
        r, _ = self.create({"input_type": "UPLOAD", "video": video_file()})
        job = AnalysisJob.objects.get(pk=r.data["id"])
        storage, name = job.video.storage, job.video.name
        self.assertTrue(storage.exists(name))
        self.assertEqual(self.api.delete(f"/api/analyses/{job.pk}/").status_code, 204)
        self.assertFalse(PersonPrediction.objects.filter(job_id=job.pk).exists())
        self.assertFalse(storage.exists(name))

    def test_delete_blocked_while_running(self):
        for state in (AnalysisJob.Status.PENDING, AnalysisJob.Status.STARTED):
            job = AnalysisJob.objects.create(user=self.a, input_type="UPLOAD", status=state)
            res = self.api.delete(f"/api/analyses/{job.pk}/")
            self.assertEqual(res.status_code, 409)
            self.assertIn("분석 중", res.data["detail"])
            self.assertTrue(AnalysisJob.objects.filter(pk=job.pk).exists())
            job.status = AnalysisJob.Status.FAILURE
            job.save(update_fields=["status"])
            self.assertEqual(self.api.delete(f"/api/analyses/{job.pk}/").status_code, 204)


@override_settings(MEDIA_ROOT=TMP_MEDIA)
class DashboardTests(CleanTestCase):
    def setUp(self):
        self.staff = User.objects.create_user("boss", password="pw-boss-123", is_staff=True)
        self.user = User.objects.create_user("normal", password="pw-normal-123")
        job = AnalysisJob.objects.create(user=self.user, input_type="UPLOAD", status="SUCCESS")
        AnalysisJob.objects.create(user=self.user, input_type="IPWEBCAM", status="FAILURE")
        PersonPrediction.objects.create(job=job, track_id=1, action_id="A005", action_label="push",
                                        confidence=0.6, window_count=1)

    # T08: 관리자만 /dashboard/, 차트 2개용 ORM 집계
    @mock.patch("analyses.page_views.health", return_value={"status": "ok"})
    def test_dashboard_staff_only_with_aggregates(self, _):
        self.client.force_login(self.user)
        self.assertEqual(self.client.get("/dashboard/").status_code, 302)
        self.client.force_login(self.staff)
        r = self.client.get("/dashboard/")
        self.assertEqual(r.status_code, 200)
        data = r.context["chart_data"]
        self.assertEqual(len(data["actions"]), 17)
        self.assertEqual(next(a["count"] for a in data["actions"] if a["id"] == "A005"), 1)
        self.assertEqual({s["status"]: s["count"] for s in data["statuses"]},
                         {"PENDING": 0, "STARTED": 0, "SUCCESS": 1, "FAILURE": 1})

    def test_admin_lists_jobs_for_staff(self):
        self.staff.is_superuser = True
        self.staff.save()
        self.client.force_login(self.staff)
        r = self.client.get("/admin/analyses/analysisjob/?status__exact=FAILURE")
        self.assertEqual(r.status_code, 200)
        self.assertContains(r, "IP Webcam")


class CameraTests(CleanTestCase):
    """카메라 주소: 관리자만 웹에서 바꾸고, 내부망 숫자 IP 만 받는다 (임의 URL 호출 방지)."""

    def setUp(self):
        self.staff = User.objects.create_user("boss", password="pw-boss-123", is_staff=True)
        self.user = User.objects.create_user("normal", password="pw-normal-123")
        self.cam = Camera.objects.create(name="phone", url="http://192.168.0.10:8080/video")
        Camera.objects.create(name="off", url="http://192.168.0.11:8080/video", enabled=False)
        self.api = APIClient()

    def test_normalize_and_validate(self):
        from django.core.exceptions import ValidationError
        from .cameras import normalize_camera_url, validate_camera_url
        self.assertEqual(normalize_camera_url(" 192.168.0.10:8080 "), "http://192.168.0.10:8080/video")
        self.assertEqual(normalize_camera_url("http://10.0.0.5:8080/shot.jpg"), "http://10.0.0.5:8080/shot.jpg")
        validate_camera_url("http://192.168.0.10:8080/video")
        for bad in ["http://8.8.8.8/video", "http://127.0.0.1:8001/live", "http://localhost:8080/video",
                    "http://172.18.0.3:5432/", "http://fastapi:8001/", "ftp://192.168.0.2/video",
                    "http://192.168.0.2:99999/video"]:
            with self.assertRaises(ValidationError, msg=bad):
                validate_camera_url(bad)

    def test_user_sees_names_only_and_cannot_edit(self):
        self.api.force_authenticate(self.user)
        r = self.api.get("/api/cameras/")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.data, [{"id": self.cam.id, "name": "phone"}])          # 주소·꺼진 카메라 안 보임
        self.assertEqual(self.api.post("/api/cameras/", {"name": "x", "url": "192.168.0.9:8080"}).status_code, 403)
        self.assertEqual(self.api.patch(f"/api/cameras/{self.cam.id}/", {"url": "192.168.0.9:8080"}).status_code, 403)
        self.assertEqual(self.api.post(f"/api/cameras/{self.cam.id}/check/").status_code, 403)

    def test_staff_changes_address_from_web(self):
        self.api.force_authenticate(self.staff)
        r = self.api.patch(f"/api/cameras/{self.cam.id}/", {"url": "192.168.0.10:8080"}, format="json")
        self.assertEqual(r.status_code, 200)
        self.cam.refresh_from_db()
        self.assertEqual(self.cam.url, "http://192.168.0.10:8080/video")
        r = self.api.patch(f"/api/cameras/{self.cam.id}/", {"url": "http://172.18.0.5:6379/"}, format="json")
        self.assertEqual(r.status_code, 400)
        self.assertIn("url", r.data)
        r = self.api.post("/api/cameras/", {"name": "phone2", "url": "10.1.2.3:8080"}, format="json")
        self.assertEqual((r.status_code, r.data["url"]), (201, "http://10.1.2.3:8080/video"))

    @mock.patch("analyses.live_views.probe", return_value={"ok": True, "width": 640, "height": 480, "seconds": 0.3,
                                                          "message": "연결됨 · 640×480"})
    def test_staff_check_connection(self, probe):
        self.api.force_authenticate(self.staff)
        r = self.api.post(f"/api/cameras/{self.cam.id}/check/")
        self.assertEqual((r.status_code, r.data["width"]), (200, 640))
        probe.assert_called_once_with("http://192.168.0.10:8080/video", 0)

    def test_disabled_camera_cannot_be_analyzed(self):
        self.api.force_authenticate(self.user)
        r = self.api.post("/api/analyses/", {"input_type": "IPWEBCAM", "camera_name": "off"}, format="multipart")
        self.assertEqual(r.status_code, 400)


class LiveTests(CleanTestCase):
    """실시간 보기: Django 는 카메라 이름 → 주소를 바꿔 FastAPI 에 넘기고, 로그인한 사람만 화면을 받는다."""

    SID = "0f8e6a52-3b1c-4c55-9a2f-2b8f4f7d1a10"

    def setUp(self):
        self.user = User.objects.create_user("normal", password="pw-normal-123")
        Camera.objects.create(name="phone", url="http://192.168.0.10:8080/video", rotation=90)
        self.api = APIClient()
        self.api.force_authenticate(self.user)

    def fake_response(self, code=200, data=None):
        r = mock.Mock(status_code=code)
        r.json.return_value = data or {}
        r.text = ""
        return r

    def test_start_sends_registered_url_only(self):
        with mock.patch("analyses.services.requests.request",
                        return_value=self.fake_response(200, {"session_id": self.SID, "state": "connecting"})) as req:
            r = self.api.post("/api/live/", {"camera_name": "phone"}, format="json")
        self.assertEqual((r.status_code, r.data["session_id"]), (201, self.SID))
        self.assertEqual(req.call_args.kwargs["json"],
                         {"camera": "phone", "url": "http://192.168.0.10:8080/video", "rotation": 90})
        with mock.patch("analyses.services.requests.request") as req:
            r = self.api.post("/api/live/", {"camera_name": "http://evil.example/"}, format="json")
        self.assertEqual(r.status_code, 400)
        req.assert_not_called()

    def test_busy_and_ai_down_messages(self):
        with mock.patch("analyses.services.requests.request",
                        return_value=self.fake_response(409, {"detail": "이미 실행 중"})):
            r = self.api.post("/api/live/", {"camera_name": "phone"}, format="json")
        self.assertEqual((r.status_code, r.data["detail"]), (409, "이미 실행 중"))
        import requests
        with mock.patch("analyses.services.requests.request", side_effect=requests.ConnectionError()):
            r = self.api.get(f"/api/live/{self.SID}/")
        self.assertEqual(r.status_code, 502)
        self.assertIn("AI 서버", r.data["detail"])

    def test_stream_requires_login(self):
        self.assertEqual(self.client.get(f"/live/{self.SID}/stream/").status_code, 302)
        self.assertEqual(APIClient().post("/api/live/", {"camera_name": "phone"}).status_code, 403)


@override_settings(MEDIA_ROOT=TMP_MEDIA, LIVE_AUTO_MIN_CONFIDENCE=0.5, LIVE_AUTO_MIN_SECONDS=3,
                   LIVE_AUTO_COOLDOWN_SECONDS=15)
class LiveSaveTests(CleanTestCase):
    """이 장면 저장·자동 감지: 실시간이 본 최근 8초를 저장 → 같은 Celery·FastAPI 분석으로 기록."""

    SID = "0f8e6a52-3b1c-4c55-9a2f-2b8f4f7d1a10"

    def setUp(self):
        self.user = User.objects.create_user("normal", password="pw-normal-123")
        Camera.objects.create(name="phone", url="http://192.168.0.10:8080/video")
        self.api = APIClient()
        self.api.force_authenticate(self.user)

    def state(self, conf=0.62, streak=4.0, streak_min=0.55, running=True):
        person = {"track_id": 3, "status": "ok", "action_id": "A003", "action_name_ko": "앞차기",
                  "confidence": conf, "uncertain": conf < 0.3, "streak_seconds": streak,
                  "streak_min_confidence": streak_min}
        return {"camera": "phone", "state": "running" if running else "stopped", "people": [person]}

    def save(self, body, state):
        with mock.patch("analyses.live_views.services.live_state", return_value=state), \
                mock.patch("analyses.live_views.services.live_clip", return_value=b"\x00\x00\x00\x18ftypmp42clip") as clip, \
                mock.patch("analyses.tasks.predict_video", return_value=fake_result(1)) as pv, \
                mock.patch("analyses.tasks.capture_clip") as cap, \
                mock.patch("analyses.jobs.run_analysis.apply_async",
                           side_effect=lambda args, task_id: run_analysis(*args)):
            with self.captureOnCommitCallbacks(execute=True):
                r = self.api.post(f"/api/live/{self.SID}/save/", body, format="json")
        return r, clip, pv, cap

    def test_manual_save_uses_live_clip_not_new_capture(self):
        r, clip, pv, cap = self.save({"expected_action": "A003", "title": "정면"}, self.state(conf=0.2, streak=0))
        self.assertEqual(r.status_code, 202)
        job = AnalysisJob.objects.get(pk=r.data["id"])
        self.assertEqual((job.input_type, job.source, job.status, job.expected_action, job.title),
                         ("IPWEBCAM", "LIVE", "SUCCESS", "A003", "정면"))
        self.assertTrue(job.video.name.endswith(".mp4"))
        self.assertEqual(job.live_snapshot[0]["action_id"], "A003")
        cap.assert_not_called()                        # 이미 들고 온 영상 → 다시 녹화하지 않음
        pv.assert_called_once()
        clip.assert_called_once_with(self.SID, 8.0)

    def test_auto_save_needs_lasting_confident_action(self):
        r, clip, *_ = self.save({"auto": True}, self.state(streak=2.0))             # 아직 2초
        self.assertEqual(r.status_code, 409)
        r, clip, *_ = self.save({"auto": True}, self.state(streak_min=0.45))        # 중간에 기준 아래로
        self.assertEqual(r.status_code, 409)
        clip.assert_not_called()
        r, *_ = self.save({"auto": True, "min_confidence": 0.1}, self.state(streak_min=0.29))  # 기준은 30% 아래로 못 내림
        self.assertEqual(r.status_code, 409)
        r, *_ = self.save({"auto": True}, self.state())
        self.assertEqual(r.status_code, 202)
        job = AnalysisJob.objects.get(pk=r.data["id"])
        self.assertEqual(job.source, "LIVE_AUTO")
        self.assertIn("자동 감지 A003", job.title)

    def test_auto_save_cooldown_and_stopped_session(self):
        self.assertEqual(self.save({"auto": True}, self.state())[0].status_code, 202)
        self.assertEqual(self.save({"auto": True}, self.state())[0].status_code, 429)   # 15초 안에 또
        self.assertEqual(self.save({}, self.state())[0].status_code, 202)               # 직접 저장은 간격 없음
        self.assertEqual(self.save({}, self.state(running=False))[0].status_code, 409)
        self.assertEqual(self.save({"expected_action": "Z99"}, self.state())[0].status_code, 400)

    def test_capture_job_marks_source_and_recording_step(self):
        seen = []

        def fake_capture(name, path, seconds=None):
            seen.append(AnalysisJob.objects.get(camera_name=name).result_message)   # 녹화 중 안내가 먼저 저장됨
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"\x00\x00\x00\x18ftypmp42cap")
        with mock.patch("analyses.tasks.capture_clip", side_effect=fake_capture), \
                mock.patch("analyses.tasks.predict_video", return_value=fake_result(1)), \
                mock.patch("analyses.jobs.run_analysis.apply_async",
                           side_effect=lambda args, task_id: run_analysis(*args)):
            with self.captureOnCommitCallbacks(execute=True):
                r = self.api.post("/api/analyses/", {"input_type": "IPWEBCAM", "camera_name": "phone"}, format="multipart")
        job = AnalysisJob.objects.get(pk=r.data["id"])
        self.assertEqual((job.source, job.status), ("CAPTURE", "SUCCESS"))
        self.assertEqual(seen, ["카메라 녹화 중"])

    def test_staff_sets_rotation(self):
        staff = User.objects.create_user("boss", password="pw-boss-123", is_staff=True)
        self.api.force_authenticate(staff)
        cam = Camera.objects.get(name="phone")
        r = self.api.patch(f"/api/cameras/{cam.id}/", {"rotation": 90}, format="json")
        self.assertEqual((r.status_code, r.data["rotation"]), (200, 90))
        self.assertEqual(self.api.patch(f"/api/cameras/{cam.id}/", {"rotation": 45}, format="json").status_code, 400)


def fake_ffmpeg(cmd, **kwargs):
    """ffmpeg 대신: 마지막 인자(출력 경로)에 가짜 H.264 파일을 쓴다"""
    from pathlib import Path as _P
    _P(cmd[-1]).write_bytes(b"h264-playback-copy")
    return mock.Mock(returncode=0)


@override_settings(MEDIA_ROOT=TMP_MEDIA)
class VideoPlaybackTests(CleanTestCase):
    """분석한 영상 다시 보기: 재생용 사본·권한·구간 이동(Range)·삭제."""

    def setUp(self):
        self.a = User.objects.create_user("user_a", password="pw-A-12345")
        self.b = User.objects.create_user("user_b", password="pw-B-12345")
        self.staff = User.objects.create_user("boss", password="pw-boss-123", is_staff=True)
        self.job = AnalysisJob.objects.create(user=self.a, input_type="UPLOAD", status="SUCCESS",
                                              video=SimpleUploadedFile("clip.mp4", bytes(range(100))))

    def get(self, user, url, **headers):
        self.client.force_login(user)
        return self.client.get(url, headers=headers)

    def body(self, r):
        return b"".join(r.streaming_content) if r.streaming else r.content

    def test_owner_streams_whole_and_range(self):
        url = f"/analyses/{self.job.pk}/video/"
        r = self.get(self.a, url)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r["Accept-Ranges"], "bytes")
        self.assertEqual(self.body(r), bytes(range(100)))
        r = self.get(self.a, url, Range="bytes=10-19")       # 영상 중간으로 건너뛸 때 브라우저가 보내는 요청
        self.assertEqual(r.status_code, 206)
        self.assertEqual(r["Content-Range"], "bytes 10-19/100")
        self.assertEqual(self.body(r), bytes(range(10, 20)))
        r = self.get(self.a, url, Range="bytes=-5")
        self.assertEqual(self.body(r), bytes(range(95, 100)))
        self.assertEqual(self.get(self.a, url, Range="bytes=200-").status_code, 416)
        r = self.get(self.a, url + "?original=1&download=1")
        self.assertIn("attachment", r["Content-Disposition"])

    def test_other_user_404_staff_can_read(self):
        url = f"/analyses/{self.job.pk}/"
        self.assertEqual(self.get(self.b, url + "video/").status_code, 404)
        self.assertEqual(self.get(self.b, url).status_code, 404)
        self.assertEqual(self.get(self.staff, url + "video/").status_code, 200)
        r = self.get(self.staff, url)
        self.assertEqual(r.status_code, 200)
        self.assertFalse(r.context["can_edit"])               # 관리자는 남의 기록을 읽기만
        self.assertNotContains(r, 'id="edit-form"')
        # API: 관리자는 상세 조회만, 목록·수정·삭제는 본인 것만
        api = APIClient()
        api.force_authenticate(self.staff)
        r = api.get(f"/api/analyses/{self.job.pk}/")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.data["owner"], "user_a")
        self.assertFalse(r.data["can_edit"])
        self.assertEqual(api.get("/api/analyses/").data["count"], 0)
        self.assertEqual(api.patch(f"/api/analyses/{self.job.pk}/", {"title": "x"}).status_code, 404)
        self.assertEqual(api.delete(f"/api/analyses/{self.job.pk}/").status_code, 404)

    def test_no_video_is_404(self):
        job = AnalysisJob.objects.create(user=self.a, input_type="IPWEBCAM", status="FAILURE")
        self.assertEqual(self.get(self.a, f"/analyses/{job.pk}/video/").status_code, 404)

    @mock.patch("analyses.playback.shutil.which", return_value="/usr/bin/ffmpeg")
    @mock.patch("analyses.playback.subprocess.run", side_effect=fake_ffmpeg)
    def test_playback_copy_served_first_and_deleted_with_job(self, run, _):
        from .playback import make_playback
        self.assertTrue(make_playback(self.job))
        cmd = run.call_args[0][0]
        self.assertIn("libx264", cmd)
        self.assertIn("yuv420p", cmd)
        self.job.refresh_from_db()
        self.assertEqual(self.job.playback.name, f"playback/job{self.job.pk}.mp4")
        r = self.get(self.a, f"/analyses/{self.job.pk}/video/")
        self.assertEqual(self.body(r), b"h264-playback-copy")
        r = self.get(self.a, f"/analyses/{self.job.pk}/video/?original=1")
        self.assertEqual(self.body(r), bytes(range(100)))
        api = APIClient()
        api.force_authenticate(self.a)
        data = api.get(f"/api/analyses/{self.job.pk}/").data
        self.assertTrue(data["playback_ready"])
        self.assertEqual(data["video_url"], f"/analyses/{self.job.pk}/video/")
        self.assertEqual(data["video_size"], 100)
        paths = [self.job.video.path, self.job.playback.path]
        self.assertEqual(api.delete(f"/api/analyses/{self.job.pk}/").status_code, 204)
        self.assertFalse(any(__import__("os").path.exists(p) for p in paths))

    @mock.patch("analyses.playback.shutil.which", return_value="/usr/bin/ffmpeg")
    @mock.patch("analyses.playback.subprocess.run", side_effect=__import__("subprocess").CalledProcessError(1, "ffmpeg"))
    def test_ffmpeg_failure_keeps_analysis_result(self, *_):
        api = APIClient()
        api.force_authenticate(self.a)
        with mock.patch("analyses.tasks.predict_video", return_value=fake_result()), \
                mock.patch("analyses.jobs.run_analysis.apply_async", side_effect=lambda args, task_id: run_analysis(*args)):
            with self.captureOnCommitCallbacks(execute=True):
                r = api.post("/api/analyses/", {"input_type": "UPLOAD", "video": video_file()}, format="multipart")
        job = AnalysisJob.objects.get(pk=r.data["id"])
        self.assertEqual(job.status, "SUCCESS")                 # 사본 실패는 분석 결과에 영향 없음
        self.assertFalse(job.playback)
        self.assertEqual(job.predictions.count(), 2)

    @mock.patch("analyses.playback.shutil.which", return_value="/usr/bin/ffmpeg")
    @mock.patch("analyses.playback.subprocess.run", side_effect=fake_ffmpeg)
    def test_backfill_command(self, *_):
        from io import StringIO
        from django.core.management import call_command
        out = StringIO()
        call_command("make_playback", stdout=out)
        self.job.refresh_from_db()
        self.assertTrue(self.job.playback)
        self.assertIn("완료 1개", out.getvalue())


@override_settings(MEDIA_ROOT=TMP_MEDIA)
class DashboardToolsTests(CleanTestCase):
    """대시보드 편의 기능: 기간·조건·검색·쪽 나누기·CSV."""

    def setUp(self):
        self.staff = User.objects.create_user("boss", password="pw-boss-123", is_staff=True)
        self.a = User.objects.create_user("user_a", password="pw-A-12345")
        self.b = User.objects.create_user("user_b", password="pw-B-12345")
        ok = AnalysisJob.objects.create(user=self.a, input_type="UPLOAD", status="SUCCESS", title="=cmd 계산대",
                                        expected_action="A001")
        PersonPrediction.objects.create(job=ok, track_id=1, action_id="A001", action_label="kiosk_tap",
                                        confidence=0.8, window_count=1)
        bad = AnalysisJob.objects.create(user=self.b, input_type="UPLOAD", status="SUCCESS", expected_action="A003")
        PersonPrediction.objects.create(job=bad, track_id=1, action_id="A005", action_label="push",
                                        confidence=0.2, uncertain=True, window_count=1)
        AnalysisJob.objects.create(user=self.b, input_type="IPWEBCAM", status="FAILURE", error_message="카메라 연결 실패")
        old = AnalysisJob.objects.create(user=self.a, input_type="UPLOAD", status="SUCCESS")
        AnalysisJob.objects.filter(pk=old.pk).update(created_at=timezone.now() - timedelta(days=40))
        self.client.force_login(self.staff)

    def ids(self, r):
        return [j.pk for j in r.context["page"]]

    @mock.patch("analyses.page_views.health", return_value={"status": "ok"})
    def test_period_filters_search_and_match(self, _):
        r = self.client.get("/dashboard/")
        self.assertEqual(r.context["summary"]["jobs"], 4)
        r = self.client.get("/dashboard/?period=30d")
        self.assertEqual(r.context["summary"]["jobs"], 3)                 # 40일 전 기록은 빠짐
        self.assertEqual(r.context["summary"]["graded"], 2)
        self.assertEqual(r.context["summary"]["match_pct"], 50.0)
        self.assertEqual(len(r.context["chart_data"]["trend"]), 31)
        self.assertEqual(r.context["confusions"][0]["expected"], "A003")
        self.assertEqual(len(self.ids(self.client.get("/dashboard/?status=FAILURE"))), 1)
        self.assertEqual(len(self.ids(self.client.get("/dashboard/?uncertain=1"))), 1)
        self.assertEqual(len(self.ids(self.client.get("/dashboard/?match=bad"))), 1)
        self.assertEqual(len(self.ids(self.client.get("/dashboard/?match=ok"))), 1)
        self.assertEqual(len(self.ids(self.client.get("/dashboard/?user=user_b"))), 2)
        self.assertEqual(len(self.ids(self.client.get("/dashboard/?q=카메라"))), 1)
        self.assertEqual(len(self.ids(self.client.get("/dashboard/?action=A001"))), 1)
        r = self.client.get("/dashboard/?period=nonsense&status=WRONG")     # 이상한 값은 오류 없이 0건
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.context["period"], "all")

    @mock.patch("analyses.page_views.health", return_value={"status": "ok"})
    def test_pagination(self, _):
        for _ in range(25):
            AnalysisJob.objects.create(user=self.a, input_type="UPLOAD", status="PENDING")
        r = self.client.get("/dashboard/?status=PENDING&page=2")
        self.assertEqual(r.context["page"].number, 2)
        self.assertEqual(len(self.ids(r)), 5)
        self.assertIn("status=PENDING", r.context["query"])
        self.assertNotIn("page=", r.context["query"])

    def test_csv_export_follows_filters_and_escapes_formulas(self):
        r = self.client.get("/dashboard/?user=user_a&export=csv")
        self.assertEqual(r.status_code, 200)
        self.assertIn("attachment", r["Content-Disposition"])
        text = r.content.decode("utf-8-sig")
        lines = text.strip().splitlines()
        self.assertEqual(len(lines), 3)                                    # 머리글 + user_a 2건
        self.assertIn("'=cmd 계산대", text)                                # Excel 수식 실행 방지
        self.assertIn("일치", text)

    def test_csv_staff_only(self):
        self.client.force_login(self.a)
        self.assertEqual(self.client.get("/dashboard/?export=csv").status_code, 302)


@override_settings(MEDIA_ROOT=TMP_MEDIA)
class ExportClipsTests(CleanTestCase):
    """manage.py export_clips: 원본 복사 + 파일 이름 규칙 + manifest.csv, 조건 거르기."""

    def setUp(self):
        self.u = User.objects.create_user("exporter", password="pw-E-12345")
        self.dest = tempfile.mkdtemp()
        mk = AnalysisJob.objects.create
        self.cap = mk(user=self.u, input_type="IPWEBCAM", source="CAPTURE", camera_name="phone", status="SUCCESS",
                      expected_action="A005", video=SimpleUploadedFile("rec.mp4", b"clip-1"))
        PersonPrediction.objects.create(job=self.cap, track_id=1, action_id="A005", action_label="push",
                                        confidence=0.8, window_count=1)
        mk(user=self.u, input_type="IPWEBCAM", source="CAPTURE", status="SUCCESS", video=SimpleUploadedFile("r2.mp4", b"x"))
        mk(user=self.u, input_type="IPWEBCAM", status="FAILURE", video=SimpleUploadedFile("bad.mp4", b"x"))
        mk(user=self.u, input_type="UPLOAD", status="SUCCESS", expected_action="A001",
           video=SimpleUploadedFile("up.mp4", b"x"))

    def tearDown(self):
        shutil.rmtree(self.dest, ignore_errors=True)

    def run_export(self, *args):
        from django.core.management import call_command
        call_command("export_clips", "--dest", self.dest, *args, stdout=io.StringIO())
        (out,) = list(Path(self.dest).iterdir())
        with (out / "manifest.csv").open(encoding="utf-8-sig") as f:
            return out, list(csv.DictReader(f))

    def test_ipwebcam_success_only_by_default(self):
        out, rows = self.run_export()
        self.assertEqual([int(r["job_id"]) for r in rows], [self.cap.id, self.cap.id + 1])   # 실패·업로드 제외
        r = rows[0]
        self.assertTrue(r["file"].startswith(f"job{self.cap.id:04d}_A005_CAPTURE_"))
        self.assertEqual((out / r["file"]).read_bytes(), b"clip-1")                            # 원본 그대로
        self.assertEqual((r["predicted_action"], r["matches_expected"], r["label_checked"]), ("A005", "True", ""))
        self.assertIn("_NA_", rows[1]["file"])                                                    # 동작 안 고름

    def test_filters(self):
        _, rows = self.run_export("--source", "all", "--labeled-only")
        self.assertEqual(sorted(r["expected_action"] for r in rows), ["A001", "A005"])
