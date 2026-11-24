"""README 용 화면 캡처 (Playwright). 비밀번호·카메라 주소는 화면에 나오지 않는다.

  pip install playwright requests
  python scripts/capture_screens.py --success 53 --failure 52 --multi 27 --preview <S04 영상.mp4> --channel msedge
결과: docs/images/screen_*.png

  · --channel msedge 권장: Playwright 기본 Chromium 은 H.264 를 못 틀어 영상 칸이 검게 나온다
  · 영상이 화면에 나오는 캡처(--success, --preview)는 본인(S04) 영상만. 팀원·공개 데이터셋 사람 영상은
    --multi 처럼 영상 칸을 숨기고 찍는다 (얼굴·초상권)
"""
from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

from e2e_check import BASE, load_env

OUT = Path(__file__).resolve().parents[1] / "docs" / "images"

SEEK_JS = """async ([sel, t]) => {
  const v = document.querySelector(sel);
  if (!v) return 'no video';
  if (v.readyState < 1) await new Promise((r) => v.addEventListener('loadedmetadata', r, { once: true }));
  v.currentTime = t;
  await new Promise((r) => v.addEventListener('seeked', r, { once: true }));
  return `${v.videoWidth}x${v.videoHeight} @ ${v.currentTime.toFixed(1)}s`;
}"""


def login(page, user, pw):
    page.goto(f"{BASE}/accounts/login/")
    page.fill("input[name=username]", user)
    page.fill("input[name=password]", pw)
    page.click("button[type=submit]")
    page.wait_for_load_state("networkidle")


def shot(page, url, name, wait="networkidle", full=True, before=None):
    page.goto(BASE + url)
    page.wait_for_load_state(wait)
    if before:
        before(page)
    page.wait_for_timeout(800)
    page.screenshot(path=str(OUT / name), full_page=full)
    print("saved", name)


def seek(sel, t):
    def run(page):
        page.wait_for_selector(sel, state="attached")
        print("  video", page.evaluate(SEEK_JS, [sel, t]))
    return run


def hide_video(page):
    page.wait_for_selector("#detail-meta dt")
    page.add_style_tag(content="#video-box, #video-windows { display: none !important; }")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--success", type=int, required=True, help="본인(S04) 영상 작업 — 영상·구간 버튼이 보임")
    ap.add_argument("--failure", type=int, required=True)
    ap.add_argument("--multi", type=int, help="두 사람 작업 (영상 칸은 숨김)")
    ap.add_argument("--preview", help="업로드 미리보기에 넣을 본인(S04) 영상 경로")
    ap.add_argument("--seek", type=float, default=5.0, help="결과 화면 영상을 몇 초 지점으로")
    ap.add_argument("--channel", default=None, help="설치된 브라우저 사용 (예: msedge, chrome)")
    a = ap.parse_args()
    env = load_env()
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch(channel=a.channel) if a.channel else p.chromium.launch()
        ctx = b.new_context(viewport={"width": 1280, "height": 860}, device_scale_factor=1, locale="ko-KR",
                            color_scheme="light")
        page = ctx.new_page()
        login(page, env["DEMO_USER_A"], env["DEMO_USER_PASSWORD"])
        shot(page, "/", "screen_index_upload.png")
        if a.preview:
            def preview(pg):
                pg.set_input_files("#video-input", a.preview)
                pg.select_option("select[name=expected_action]", index=10)
                pg.wait_for_function("document.querySelector('#preview-info').textContent.includes('×')")
                seek("#preview-video", 3.0)(pg)
            shot(page, "/", "screen_upload_preview.png", full=False, before=preview)
        shot(page, "/", "screen_index_ipwebcam.png", full=False,
             before=lambda pg: pg.click("button[data-input=IPWEBCAM]"))
        shot(page, f"/analyses/{a.success}/", "screen_result_success.png", before=seek("#job-video", a.seek))
        shot(page, f"/analyses/{a.failure}/", "screen_result_failure.png")
        if a.multi:
            shot(page, f"/analyses/{a.multi}/", "screen_result_two_people.png", before=hide_video)
        page.goto(f"{BASE}/dashboard/")
        print("normal user /dashboard/ ->", page.url)
        ctx.clear_cookies()
        login(page, env["DEMO_ADMIN_USERNAME"], env["DEMO_ADMIN_PASSWORD"])
        shot(page, "/dashboard/", "screen_dashboard.png")
        shot(page, "/dashboard/?period=all&status=SUCCESS&uncertain=1#jobs", "screen_dashboard_filter.png", full=False,
             before=lambda pg: pg.locator("#jobs").scroll_into_view_if_needed())
        shot(page, "/admin/analyses/analysisjob/", "screen_admin_jobs.png", full=False)
        b.close()


if __name__ == "__main__":
    main()
