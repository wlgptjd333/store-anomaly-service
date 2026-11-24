#!/usr/bin/env bash
# 커밋 전 한 번에 확인: 린트 · JS 문법 · Django 설정 · migration 누락 · 단위 테스트 (Docker 없이, SQLite · 가짜 AI 서버)
# 실행: scripts/check.sh   (backend/.venv 가 있으면 그걸 씀. CI 도 같은 순서 — .github/workflows/ci.yml)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-backend/.venv/bin/python}
[ -x "$PY" ] || PY=python3
case $PY in */*) PY="$PWD/$PY";; esac          # 아래에서 backend/ 로 들어가도 찾게
RUFF=${RUFF:-$(command -v ruff || echo "$PY -m ruff")}

echo "== ruff (ruff.toml)";               $RUFF check backend ai_service scripts
if command -v node >/dev/null; then
  echo "== JS 문법";                       for f in backend/static/js/*.js; do node --check "$f"; done
fi
cd backend
export POSTGRES_HOST= IPWEBCAM_CAMERAS=   # 로컬 .env 값과 상관없이 같은 결과
echo "== Django check";                    $PY manage.py check
echo "== migration 누락";                  $PY manage.py makemigrations --check --dry-run
echo "== 단위 테스트";                     $PY manage.py test
echo "모두 통과"
