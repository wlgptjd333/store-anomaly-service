"""과제 테스트 중 서버 없이 확인할 수 있는 것 (T02·T04·T05·T06·T07·T08·T09 일부).

FastAPI 와 카메라는 가짜(mock)로 바꾸고, Celery 작업은 on_commit 콜백을 직접 실행해 같은 흐름을 탄다.
실제 서버·Docker 를 거친 확인은 README 의 테스트 표에 따로 기록한다.
"""
import shutil
import tempfile
from unittest import mock

from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
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


@override_settings(MEDIA_ROOT=TMP_MEDIA)
class AnalysisApiTests(TestCase):
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
                mock.patch("analyses.views.run_analysis.apply_async",
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
        with mock.patch("analyses.views.run_analysis.apply_async", side_effect=ConnectionError("redis down")):
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


@override_settings(MEDIA_ROOT=TMP_MEDIA)
class DashboardTests(TestCase):
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


class CameraTests(TestCase):
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


class LiveTests(TestCase):
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
class LiveSaveTests(TestCase):
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
                mock.patch("analyses.views.run_analysis.apply_async",
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
                mock.patch("analyses.views.run_analysis.apply_async",
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
