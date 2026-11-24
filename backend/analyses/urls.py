from django.urls import include, path
from rest_framework.routers import DefaultRouter

from . import live_views, page_views
from .views import AnalysisJobViewSet

app_name = "analyses"

router = DefaultRouter()
router.register("analyses", AnalysisJobViewSet, basename="analysis")
router.register("cameras", live_views.CameraViewSet, basename="camera")

urlpatterns = [
    path("", page_views.index, name="index"),
    path("analyses/<int:pk>/", page_views.detail, name="detail"),
    path("analyses/<int:pk>/video/", page_views.video, name="video"),
    path("dashboard/", page_views.dashboard, name="dashboard"),
    path("api/live/", live_views.live_start, name="live-start"),
    path("api/live/<uuid:sid>/", live_views.live_session, name="live-session"),
    path("api/live/<uuid:sid>/save/", live_views.live_save, name="live-save"),
    path("live/<uuid:sid>/stream/", live_views.live_stream, name="live-stream"),
    path("api/", include(router.urls)),
]
