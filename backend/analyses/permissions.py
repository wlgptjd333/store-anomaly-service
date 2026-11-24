from rest_framework.permissions import SAFE_METHODS, BasePermission


class IsOwner(BasePermission):
    """본인이 만든 분석 작업만 조회·수정·삭제. 관리자는 읽기(GET)만 다른 회원 것도.
    (목록은 views.get_queryset 에서 관리자도 본인 것만 거른다)"""

    message = "본인의 분석 기록만 볼 수 있습니다."

    def has_object_permission(self, request, view, obj):
        if request.method in SAFE_METHODS and request.user.is_staff:
            return True
        return obj.user_id == request.user.id
