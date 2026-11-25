/* 분석 화면 (index.html · detail.html)
 * 흐름: (업로드 미리보기) → AI 분석 시작 → POST /api/analyses/ (작업 ID 즉시 받음) → /status/ 를 2초마다 조회
 *       → 완료되면 상세 GET 으로 사람별 결과 + 분석한 영상 표시 → 최근 이력 새로고침
 * 구성: Config · Util · Data(API) · View · Player · Preview · Handlers · Live · Cameras · Init · Entry
 */
(function () {
  "use strict";

  // ── Util ──
  const Util = {
    $(sel, root = document) { return root.querySelector(sel); },
    $$(sel, root = document) { return Array.from(root.querySelectorAll(sel)); },
    json(id, fallback) {                          // Django json_script 값 읽기
      const el = document.getElementById(id);
      return el ? JSON.parse(el.textContent) : fallback;
    },
    csrf() {
      const m = document.cookie.match(/(?:^|;\s*)csrftoken=([^;]+)/);
      return m ? decodeURIComponent(m[1]) : "";
    },
    esc(s) {
      return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
    },
    pct(x) { return x == null ? "-" : `${(x * 100).toFixed(0)}%`; },
    sec(x) { return x == null ? "-" : `${Number(x).toFixed(1)}초`; },
    mb(bytes) { return bytes == null ? "-" : `${(bytes / 1024 / 1024).toFixed(1)} MB`; },
    clock(s) {                                    // 75.3 → "1:15"
      const t = Math.max(0, Math.floor(Number(s) || 0));
      return `${Math.floor(t / 60)}:${String(t % 60).padStart(2, "0")}`;
    },
    time(iso) {
      if (!iso) return "-";
      const d = new Date(iso);
      return `${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")} ${d.toTimeString().slice(0, 5)}`;
    },
    errorText(err) {
      const data = err?.response?.data;
      if (!data) return "서버에 연결하지 못했습니다.";
      if (typeof data === "string") return data.slice(0, 200);
      if (data.detail || data.message) return data.detail || data.message;
      return Object.values(data).flat().join(" ");
    },
    flash(sel, text, tone = "error") {            // 한 줄 안내: ok(초록) · error(빨강) · info(회색). 빈 글이면 숨김
      const el = Util.$(sel);
      if (!el) return;
      el.textContent = text || "";
      el.classList.remove("ok-text", "error", "muted");
      el.classList.add({ ok: "ok-text", error: "error", info: "muted" }[tone]);
      el.hidden = !text;
    },
  };

  // ── Config ──
  // 서버 설정값은 page_views._page_context 의 json_script "page-config" 한 곳에서 (상태 이름은 API 의 status_display)
  const page = Util.json("page-config", {});
  const POLL_MS = 2000;
  const Config = {
    api: { analyses: "/api/analyses/", cameras: "/api/cameras/", live: "/api/live/" },
    pollMs: POLL_MS,
    pollLimit: Math.ceil(((page.poll_limit_seconds ?? 900) * 1000) / POLL_MS),   // Celery 시간 제한까지만 조회
    live: page.live ?? { min_seconds: 3, cooldown_seconds: 15, capture_seconds: 8 },
    uncertainBelow: page.uncertain_below ?? 0.3,
    actionNames: page.action_names ?? {},
    preview: { longSeconds: 60, shortSeconds: 3, cpuFactor: 2.5 },   // 미리보기 안내 기준 (Docker CPU 실측: 영상 1초 ≈ 처리 2~2.5초)
  };

  // ── Data (DRF API) ──
  const http = axios.create({ headers: { "X-CSRFToken": Util.csrf() } });
  const Api = Config.api;
  const Data = {
    create: (formData, onUploadProgress) => http.post(Api.analyses, formData, { onUploadProgress }).then((r) => r.data),
    list: () => http.get(Api.analyses).then((r) => r.data.results ?? r.data),
    detail: (id) => http.get(`${Api.analyses}${id}/`).then((r) => r.data),
    status: (id) => http.get(`${Api.analyses}${id}/status/`).then((r) => r.data),
    update: (id, body) => http.patch(`${Api.analyses}${id}/`, body).then((r) => r.data),
    remove: (id) => http.delete(`${Api.analyses}${id}/`),
    // 카메라 (일반 사용자는 이름만, 관리자는 주소까지)
    cameras: () => http.get(Api.cameras).then((r) => r.data),
    addCamera: (body) => http.post(Api.cameras, body).then((r) => r.data),
    saveCamera: (id, body) => http.patch(`${Api.cameras}${id}/`, body).then((r) => r.data),
    deleteCamera: (id) => http.delete(`${Api.cameras}${id}/`),
    checkCamera: (id) => http.post(`${Api.cameras}${id}/check/`).then((r) => r.data),
    // 실시간 보기 (보기용, DB 에 남기지 않음)
    liveStart: (camera_name) => http.post(Api.live, { camera_name }).then((r) => r.data),
    liveState: (sid) => http.get(`${Api.live}${sid}/`).then((r) => r.data),
    liveStop: (sid) => http.delete(`${Api.live}${sid}/`),
    liveSave: (sid, body) => http.post(`${Api.live}${sid}/save/`, body).then((r) => r.data),
  };

  // ── View ──
  const View = {
    badge(job) {                                  // 상태 이름은 서버(models.Status)가 보낸 status_display
      return `<span class="badge s-${job.status}">${Util.esc(job.status_display ?? job.status)}</span>`;
    },
    uncertainBadge(p) {
      return p.uncertain ? '<span class="badge warn">판단 불확실</span>' : "";
    },
    actionText(a, prob = a.confidence) {          // "A001 키오스크 두드리기 77%"
      return `${Util.esc(a.action_id)} ${Util.esc(a.action_name_ko)} ${Util.pct(prob)}`;
    },
    confidenceCell(p) {
      return `<div class="bar"><span style="width:${(p.confidence * 100).toFixed(0)}%"></span></div>${Util.pct(p.confidence)}`
        + (p.uncertain ? `<br>${View.uncertainBadge(p)}` : "");
    },
    predictionLine(p) {
      return `#${p.track_id} ${View.actionText(p)} ${View.uncertainBadge(p)}`;
    },
    resultTable(job) {
      if (job.status === "FAILURE") return `<p class="error">실패 사유: ${Util.esc(job.error_message)}</p>`;
      if (job.status !== "SUCCESS") return "";
      if (!job.predictions.length) return `<p class="warn-text">${Util.esc(job.result_message || "예측 결과가 없습니다.")}</p>`;
      const rows = job.predictions.map((p) => {
        const match = job.expected_action
          ? (job.expected_action === p.action_id ? '<span class="badge ok">일치</span>' : '<span class="badge bad">불일치</span>')
          : "-";
        const top3 = p.top3.map((t) => View.actionText(t, t.probability)).join("<br>");
        return `<tr class="${p.uncertain ? "uncertain" : ""}">
          <td>#${p.track_id}</td>
          <td><strong>${Util.esc(p.action_id)} ${Util.esc(p.action_name_ko)}</strong><br><span class="muted">${Util.esc(p.action_label)} · ${Util.esc(p.category_ko)}</span></td>
          <td>${View.confidenceCell(p)}</td>
          <td>${match}</td>
          <td class="small">${top3}</td>
          <td class="small">창 ${p.window_count}개<br>유효 Frame ${p.valid_frames}<br>${Util.sec(p.track_seconds)} 보임<br>관절 Conf ${p.mean_keypoint_conf ?? "-"}</td>
        </tr>`;
      }).join("");
      return `<table class="table">
        <thead><tr><th>Track</th><th>예측 동작</th><th>Confidence</th><th>수행한 동작과</th><th>상위 3개</th><th>입력 품질</th></tr></thead>
        <tbody>${rows}</tbody></table>
        <p class="muted small">Confidence ${Util.pct(Config.uncertainBelow)} 미만은 '판단 불확실'로 표시합니다. 17개 동작 중 하나를 고르는 모델이라, 학습하지 않은 동작에도 가장 가까운 동작을 답합니다.</p>`;
    },
    steps(status) {
      const order = ["PENDING", "STARTED", "DONE"];
      const now = status === "SUCCESS" || status === "FAILURE" ? "DONE" : status;
      Util.$$("#status-steps li").forEach((li) => {
        const i = order.indexOf(li.dataset.step);
        li.className = i < order.indexOf(now) ? "done" : i === order.indexOf(now) ? `now s-${status}` : "";
      });
    },
    current(job) {
      Util.$("#current").hidden = false;
      Util.$("#current-id").innerHTML = `#${job.id} ${View.badge(job)}`;
      View.steps(job.status);
      let msg = { PENDING: "작업을 대기열에 넣었습니다. 앞 작업이 끝나면 시작합니다.", STARTED: "AI 서버가 영상을 분석하고 있습니다…" }[job.status];
      const recording = job.is_recording;            // 서버가 판단 (models.AnalysisJob.is_recording)
      if (job.status === "STARTED" && job.result_message) msg = job.result_message;
      if (recording) msg = `● 지금 녹화 중 — 카메라 앞에서 동작하세요 (${Config.live.capture_seconds}초)`;
      Util.$("#current-message").className = recording ? "recording" : "muted";
      Util.$("#current-message").textContent = msg ?? (job.processing_seconds ? `AI 처리 ${Util.sec(job.processing_seconds)} · 영상 ${Util.sec(job.video_seconds)}` : "");
      if (!job.predictions) { Util.$("#current-result").innerHTML = ""; return; }
      // 끝난 작업: 결과표 + 분석한 영상(녹화본 포함)을 바로 확인
      const video = job.video_url
        ? `<video class="mini" controls preload="metadata" playsinline muted src="${Player.src(job)}"></video>` : "";
      Util.$("#current-result").innerHTML = `<div class="current-grid">${video}<div>${View.resultTable(job)}</div></div>`
        + `<p><a href="/analyses/${job.id}/">상세 보기 (구간별 예측·영상 이동) →</a></p>`;
    },
    history(jobs) {
      const body = Util.$("#history tbody");
      if (!body) return;
      body.innerHTML = jobs.length ? jobs.map((j) => `<tr>
          <td><a href="/analyses/${j.id}/">${j.id}</a></td>
          <td class="nowrap">${Util.esc(j.input_type_display)}${j.source_display ? `<br><span class="${j.source === "LIVE_AUTO" ? "badge" : "muted small"}">${Util.esc(j.source_display)}</span>` : ""}${j.title ? `<br><span class="muted small cell-title" title="${Util.esc(j.title)}">${Util.esc(j.title)}</span>` : ""}</td>
          <td>${j.expected_action ? Util.esc(j.expected_action_display) : "-"}</td>
          <td>${j.predictions.map(View.predictionLine).join("<br>") || (j.status === "FAILURE" ? `<span class="error">${Util.esc(j.error_message)}</span>` : Util.esc(j.result_message) || "-")}</td>
          <td>${View.badge(j)}</td>
          <td class="nowrap">${Player.playable(j) ? `<a href="/analyses/${j.id}/#video">▶ 보기</a>` : "-"}</td>
          <td class="nowrap">${Util.sec(j.processing_seconds)}</td>
          <td class="nowrap">${Util.time(j.created_at)}</td></tr>`).join("")
        : '<tr><td colspan="8" class="muted">아직 분석 기록이 없습니다.</td></tr>';
    },
    detail(job) {
      Util.$("#detail-status").outerHTML = `<span id="detail-status">${View.badge(job)}</span>`;
      const meta = [
        ...(job.can_edit ? [] : [["회원", Util.esc(job.owner)]]),
        ["입력 방식", job.input_type_display + (job.camera_name ? ` (${Util.esc(job.camera_name)})` : "")
          + (job.source_display ? ` · ${Util.esc(job.source_display)}` : "")],
        ...(job.live_snapshot && job.live_snapshot.length ? [["저장 당시 실시간 예측",
          job.live_snapshot.map((p) => `#${p.track_id} ${View.actionText(p)}`
            + (p.streak_seconds ? ` (${Util.sec(p.streak_seconds)} 이어짐)` : "")).join("<br>")
          + '<br><span class="muted small">아래 결과는 저장한 영상을 업로드와 같은 방법으로 다시 분석한 값입니다 (Track 번호는 다시 매김).</span>']] : []),
        ["영상", job.video_name ? `${Util.esc(job.video_name)} (${Util.mb(job.video_size)})`
          + (job.video_url && !Player.playable(job) ? ` · <a href="${job.download_url}">원본 내려받기</a>` : "") : "-"],
        ["수행한 동작(사용자 선택)", job.expected_action ? Util.esc(job.expected_action_display) : "-"],
        ["영상 길이 / AI 처리 시간", `${Util.sec(job.video_seconds)} / ${Util.sec(job.processing_seconds)}`],
        ["요청 / 완료", `${Util.time(job.created_at)} / ${Util.time(job.finished_at)}`],
        ["Celery Task ID", `<code>${Util.esc(job.celery_task_id)}</code>`],
      ];
      Util.$("#detail-meta").innerHTML = meta.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
      Util.$("#detail-result").innerHTML = View.resultTable(job) || `<p class="muted">${Util.esc(job.status_display)}…</p>`;
      Player.show(job);
    },
  };

  // ── Player: 결과 화면의 분석 영상 + 구간별 예측 (누르면 그 시점으로) ──
  const Player = {
    playable(job) {                               // 실패 + 사본 없음 = 열리지 않는 파일일 가능성 → 플레이어 숨김
      return Boolean(job.video_url) && (job.status !== "FAILURE" || job.playback_ready);
    },
    src(job) {                                    // 재생용 사본이 생기면 주소가 바뀌어 새로 받는다
      return `${job.video_url}?v=${job.playback_ready ? "p" : "o"}`;
    },
    show(job) {
      const box = Util.$("#video-box");
      if (!box) return;
      box.hidden = !Player.playable(job);
      if (box.hidden) { Util.$("#video-windows").innerHTML = ""; return; }
      const v = Util.$("#job-video");
      const want = Player.src(job);
      if (v.dataset.src !== want) {               // 상태 조회 때마다 다시 넣으면 재생이 처음으로 돌아감
        v.dataset.src = want;
        v.src = want;
      }
      const kind = job.playback_ready ? "브라우저 재생용 사본 (H.264 · 최대 720p · 소리 없음)" : "원본 영상";
      Util.$("#video-note").innerHTML = `${kind} · <a href="${job.download_url}">원본 내려받기</a>`;
      v.onerror = () => {
        Util.$("#video-note").innerHTML = `<span class="warn-text">이 브라우저에서 재생할 수 없는 형식입니다${job.playback_ready ? "" : " (재생용 사본을 만드는 중이거나 실패)"}.</span>`
          + ` <a href="${job.download_url}">원본 내려받기</a>`;
      };
      Player.windows(job);
    },
    windows(job) {
      const el = Util.$("#video-windows");
      const tracks = (job.predictions || []).filter((p) => p.windows && p.windows.length);
      if (!tracks.length) { el.innerHTML = ""; return; }
      const names = Config.actionNames;
      const limit = Config.uncertainBelow;
      el.innerHTML = `<h2>구간별 예측 <span class="muted small">버튼을 누르면 영상이 그 시점으로 이동합니다 · 재생 중인 구간은 진하게</span></h2>`
        + tracks.map((p) => `<div class="win-row"><span class="win-track">#${p.track_id}</span>${p.windows.map((w) =>
          `<button type="button" class="win ${w.action_id === p.action_id ? "same" : ""} ${w.confidence < limit ? "low" : ""}"
             data-start="${w.start_sec}" data-end="${w.end_sec}"
             title="${Util.esc(w.action_id)} ${Util.esc(names[w.action_id] || "")} · Confidence ${Util.pct(w.confidence)}">
             <span>${Util.clock(w.start_sec)}–${Util.clock(w.end_sec)}</span> ${Util.esc(w.action_id)} ${Util.esc(names[w.action_id] || "")} ${Util.pct(w.confidence)}</button>`).join("")}</div>`).join("")
        + '<p class="muted small">창마다 따로 예측한 값입니다. 사람별 최종 예측은 이 창들의 확률 평균이며, 최종과 같은 동작인 창은 테두리로 표시합니다.</p>';
    },
    seek(e) {
      const btn = e.target.closest("button.win");
      if (!btn) return;
      const v = Util.$("#job-video");
      v.currentTime = Number(btn.dataset.start);
      v.play().catch(() => {});
      v.scrollIntoView({ block: "nearest", behavior: "smooth" });
    },
    highlight() {
      const t = Util.$("#job-video").currentTime;
      Util.$$("#video-windows button.win").forEach((b) =>
        b.classList.toggle("playing", t >= Number(b.dataset.start) && t < Number(b.dataset.end)));
    },
  };

  // ── Preview: 업로드 전 영상 미리보기 + 길이·용량 안내 ──
  const Preview = {
    url: null,
    clear() {
      if (Preview.url) URL.revokeObjectURL(Preview.url);
      Preview.url = null;
      const box = Util.$("#upload-preview");
      if (!box) return;
      box.hidden = true;
      Util.$("#preview-video").removeAttribute("src");
      Util.$("#preview-video").load();
    },
    show(e) {
      Preview.clear();
      const file = e.currentTarget.files[0];
      if (!file) return;
      const maxMb = Number(e.currentTarget.dataset.maxMb || 100);
      const v = Util.$("#preview-video");
      const info = Util.$("#preview-info");
      const warn = Util.$("#preview-warn");
      Preview.url = URL.createObjectURL(file);
      Util.$("#upload-preview").hidden = false;
      info.textContent = `${file.name} · ${Util.mb(file.size)} · 길이 확인 중…`;
      warn.textContent = file.size > maxMb * 1024 * 1024 ? `${maxMb} MB 를 넘어 올릴 수 없습니다. 짧게 자르거나 해상도를 낮추세요.` : "";
      v.onloadedmetadata = () => {
        const d = v.duration;
        const c = Config.preview;
        info.textContent = `${file.name} · ${Util.mb(file.size)} · ${Util.sec(d)} · ${v.videoWidth}×${v.videoHeight}`
          + ` · 예상 분석 시간 약 ${Math.max(5, Math.round(d * c.cpuFactor))}초 (CPU)`;
        const notes = [];
        if (warn.textContent) notes.push(warn.textContent);
        if (d > c.longSeconds) notes.push(`${c.longSeconds}초보다 길면 여러 동작이 섞여 사람별 평균이 흐려질 수 있습니다. 동작 하나가 담긴 구간을 추천합니다.`);
        if (d < c.shortSeconds) notes.push("너무 짧아 사람을 충분히 추적하지 못할 수 있습니다 (8초 안팎 추천).");
        if (v.videoHeight > v.videoWidth) notes.push("세로 영상입니다. 사람이 똑바로 서 보이면 괜찮습니다.");
        warn.textContent = notes.join(" ");
      };
      v.onerror = () => {
        info.textContent = `${file.name} · ${Util.mb(file.size)} — 이 브라우저에서는 미리보기를 재생할 수 없는 형식입니다 (분석은 서버에서 가능).`;
      };
      v.src = Preview.url;
    },
  };

  // ── Handlers ──
  const state = { input: "UPLOAD", polling: null };

  async function poll(id, onDone, count = 0) {
    try {
      const s = await Data.status(id);
      if (s.is_finished) return onDone(await Data.detail(id));
      if (Util.$("#current")) View.current(s);
      if (count < Config.pollLimit) state.polling = setTimeout(() => poll(id, onDone, count + 1), Config.pollMs);
    } catch (err) {
      state.polling = setTimeout(() => poll(id, onDone, count + 1), Config.pollMs * 2);
    }
  }

  const Handlers = {
    switchTab(e) {
      state.input = e.currentTarget.dataset.input;
      Util.$$(".tab").forEach((t) => t.classList.toggle("active", t === e.currentTarget));
      Util.$$(".pane").forEach((p) => p.classList.toggle("hidden", p.dataset.pane !== state.input));
    },
    async submit(e) {
      e.preventDefault();
      const form = e.currentTarget;
      const btn = Util.$("#start-btn");
      Util.flash("#form-error", "");
      const fd = new FormData();
      fd.append("input_type", state.input);
      if (state.input === "UPLOAD") {
        const file = form.video.files[0];
        if (!file) { Util.flash("#form-error", "영상 파일을 선택하세요."); return; }
        fd.append("video", file);
      } else if (form.camera_name) {
        fd.append("camera_name", form.camera_name.value);
      }
      ["expected_action", "title"].forEach((k) => fd.append(k, form[k].value));
      btn.disabled = true;
      btn.textContent = "요청 중…";
      const progress = (ev) => { if (ev.total) btn.textContent = `올리는 중 ${Math.round((ev.loaded / ev.total) * 100)}%`; };
      try {
        const created = await Data.create(fd, state.input === "UPLOAD" ? progress : undefined);   // 202: 작업 ID 만 받음
        Handlers.track(created);
        form.reset();
        Preview.clear();
      } catch (ex) {
        Util.flash("#form-error", Util.errorText(ex));
      } finally {
        btn.disabled = false;
        btn.textContent = "AI 분석 시작";
      }
    },
    finished(job) {
      View.current(job);
      Handlers.refreshHistory();
    },
    track(created) {                              // 작업 ID 를 받은 뒤 상태 조회 시작 (분석 시작·장면 저장·자동 감지 공통)
      View.current(created);
      clearTimeout(state.polling);
      poll(created.id, Handlers.finished);
      Handlers.refreshHistory();
    },
    async refreshHistory() {
      try { View.history(await Data.list()); } catch (ex) { /* 다음 새로고침 때 다시 */ }
    },
    async saveEdit(e) {
      e.preventDefault();
      const id = Util.$("#detail").dataset.jobId;
      const f = e.currentTarget;
      try {
        await Data.update(id, { title: f.title.value, memo: f.memo.value });
        Util.flash("#edit-message", "저장했습니다.", "ok");
      } catch (ex) {
        Util.flash("#edit-message", Util.errorText(ex));
      }
    },
    async remove() {
      const id = Util.$("#detail").dataset.jobId;
      if (!confirm(`분석 #${id} 기록과 사람별 결과, 영상 파일을 삭제할까요? 되돌릴 수 없습니다.`)) return;
      try {
        await Data.remove(id);
        location.href = "/";
      } catch (ex) {                              // 409: 아직 분석 중
        Util.flash("#edit-message", Util.errorText(ex));
      }
    },
  };

  // ── 실시간 보기: <img> 는 MJPEG 를 계속 받고, 사람별 예측은 1초마다 JSON 으로 ──
  const Live = {
    sid: null,
    timer: null,
    pollMs: 1000,
    stateKo: { connecting: "연결 중", running: "실시간", stopped: "멈춤", error: "오류" },
    message: (text, isError = true) => Util.flash("#live-message", text, isError ? "error" : "info"),
    badge(state) {
      const cls = { running: "s-SUCCESS", connecting: "s-STARTED", error: "s-FAILURE" }[state] || "";
      Util.$("#live-badge").innerHTML = state ? `<span class="badge ${cls}">${Live.stateKo[state] ?? state}</span>` : "";
    },
    people(snap) {
      const body = Util.$("#live-people tbody");
      if (!snap.people.length) {
        body.innerHTML = `<tr><td colspan="3" class="muted">${snap.state === "running" ? "화면에 사람이 없습니다." : "카메라에 연결하는 중…"}</td></tr>`;
        return;
      }
      body.innerHTML = snap.people.map((p) => {
        if (p.status !== "ok") {
          return `<tr><td>#${p.track_id}</td><td class="muted">판단 중… (${Util.sec(p.seen_seconds)} 보임)<br><span class="small">${Util.esc(p.reason)}</span></td><td>-</td></tr>`;
        }
        const next = p.top3.slice(1).map((t) => View.actionText(t, t.probability)).join(" · ");
        const streak = p.streak_seconds ? ` · ${Util.sec(p.streak_seconds)}째 같은 동작` : "";
        return `<tr class="${p.uncertain ? "uncertain" : ""}">
          <td>#${p.track_id}</td>
          <td><strong>${Util.esc(p.action_id)} ${Util.esc(p.action_name_ko)}</strong><br><span class="muted small">다음 후보: ${next}${streak}</span></td>
          <td>${View.confidenceCell(p)}</td></tr>`;
      }).join("");
    },
    async start() {
      const select = Util.$("#camera-select");
      if (!select || !select.value) { Live.message("카메라를 먼저 고르세요."); return; }
      const btn = Util.$("#live-btn");
      btn.disabled = true;
      Live.message("");
      try {
        const snap = await Data.liveStart(select.value);
        Live.sid = snap.session_id;
        Util.$("#live-img").src = `/live/${Live.sid}/stream/?t=${Date.now()}`;
        Util.$("#live-body").hidden = false;
        btn.textContent = "실시간 보기 멈춤";
        Live.render(snap);
        Live.timer = setInterval(Live.poll, Live.pollMs);
      } catch (ex) {
        Live.message(Util.errorText(ex));
      } finally {
        btn.disabled = false;
      }
    },
    async poll() {
      if (!Live.sid) return;
      try {
        Live.render(await Data.liveState(Live.sid));
      } catch (ex) {
        Live.end(ex?.response?.status === 404 ? "실시간 세션이 끝났습니다. 다시 시작하세요." : Util.errorText(ex));
      }
    },
    render(snap) {
      Live.badge(snap.state);
      Live.people(snap);
      Util.$("#live-meta").textContent = snap.width
        ? `${snap.camera} · ${snap.width}×${snap.height} · 처리 ${snap.fps} fps · ${Util.sec(snap.running_seconds)} 경과`
        : `${snap.camera} · 연결 중…`;
      if (snap.state === "error" || snap.state === "stopped") { Live.end(snap.message, snap.state === "error"); return; }
      Live.autoCheck(snap);
    },
    // ── 장면 저장 · 자동 감지 ──
    lastAuto: 0,
    saving: false,
    async save(auto = false) {
      if (!Live.sid || Live.saving) return;
      const form = Util.$("#analysis-form");
      Live.saving = true;
      const btn = Util.$("#live-save-btn");
      if (!auto) btn.disabled = true;
      try {
        const created = await Data.liveSave(Live.sid, {
          auto, min_confidence: Util.$("#live-auto-threshold").value,
          expected_action: form.expected_action.value, title: auto ? "" : form.title.value,
        });
        if (auto) Live.lastAuto = Date.now();
        Util.flash("#live-save-message", `${auto ? "자동 감지로" : ""} 저장했습니다 → 분석 #${created.id}${created.title ? ` (${created.title})` : ""}`, "ok");
        Handlers.track(created);
      } catch (ex) {
        const code = ex?.response?.status;
        if (auto && (code === 409 || code === 429)) { if (code === 429) Live.lastAuto = Date.now(); return; }  // 조건 아님·간격 — 조용히
        Util.flash("#live-save-message", Util.errorText(ex));
      } finally {
        Live.saving = false;
        btn.disabled = false;
      }
    },
    autoCheck(snap) {
      // 브라우저는 '조건이 된 것 같다'만 보고 요청하고, 실제 판단은 서버(live_views._auto_hit)가 다시 한다
      if (!Util.$("#live-auto").checked || snap.state !== "running") return;
      const cfg = Config.live;
      if (Date.now() - Live.lastAuto < cfg.cooldown_seconds * 1000) return;
      const min = Number(Util.$("#live-auto-threshold").value);
      const hit = snap.people.some((p) => p.status === "ok" && p.streak_seconds >= cfg.min_seconds && p.streak_min_confidence >= min);
      if (hit) Live.save(true);
    },
    end(text, isError = true) {
      clearInterval(Live.timer);
      Live.timer = null;
      Live.sid = null;
      Util.$("#live-img").removeAttribute("src");        // 스트림 연결 끊기
      Util.$("#live-btn").textContent = "실시간 보기 시작";
      if (text) Live.message(text, isError);
    },
    async stop() {
      const sid = Live.sid;
      Live.end("");
      Live.badge("stopped");
      if (sid) { try { await Data.liveStop(sid); } catch (ex) { /* 이미 멈춤 */ } }
    },
    toggle() { return Live.sid ? Live.stop() : Live.start(); },
    leave() {                                            // 페이지를 떠나면 서버 세션도 멈춤
      if (!Live.sid) return;
      fetch(`${Api.live}${Live.sid}/`, { method: "DELETE", keepalive: true, headers: { "X-CSRFToken": Util.csrf() } });
    },
  };

  // ── 카메라 주소 관리 (관리자만 화면에 있음) ──
  const Cameras = {
    rotations: [[0, "0°"], [90, "시계 90°"], [180, "180°"], [270, "반시계 90°"]],
    message: (text, tone = "ok") => Util.flash("#camera-message", text, tone),
    async refresh() {
      const list = await Data.cameras();
      // 분석용 선택 목록 (사용 중인 카메라만)
      const select = Util.$("#camera-select");
      const enabled = list.filter((c) => c.enabled !== false);
      const keep = select.value;
      select.innerHTML = enabled.map((c) => `<option value="${Util.esc(c.name)}">${Util.esc(c.name)}</option>`).join("");
      if (enabled.some((c) => c.name === keep)) select.value = keep;
      Util.$("#no-camera").hidden = enabled.length > 0;
      const body = Util.$("#camera-table tbody");
      if (!body) return;
      body.innerHTML = list.length ? list.map((c) => `<tr data-id="${c.id}">
          <td><strong>${Util.esc(c.name)}</strong></td>
          <td><input type="text" class="cam-url" value="${Util.esc(c.url)}" maxlength="200"></td>
          <td><select class="cam-rotation">${Cameras.rotations.map(([v, label]) =>
            `<option value="${v}" ${c.rotation === v ? "selected" : ""}>${label}</option>`).join("")}</select></td>
          <td><input type="checkbox" class="cam-enabled" ${c.enabled ? "checked" : ""}></td>
          <td class="nowrap"><button type="button" class="primary small-btn" data-act="save">저장 + 연결 확인</button>
            <button type="button" class="danger small-btn" data-act="delete">삭제</button></td></tr>`).join("")
        : '<tr><td colspan="5" class="muted">등록된 카메라가 없습니다.</td></tr>';
    },
    async onClick(e) {
      const btn = e.target.closest("button[data-act]");
      if (!btn) return;
      const row = btn.closest("tr");
      const id = row.dataset.id;
      const name = row.querySelector("strong").textContent;
      btn.disabled = true;
      try {
        if (btn.dataset.act === "delete") {
          if (!confirm(`카메라 '${name}'를 지울까요? 지난 분석 기록의 카메라 이름은 그대로 남습니다.`)) return;
          await Data.deleteCamera(id);
          Cameras.message(`'${name}'를 지웠습니다.`);
        } else {
          const saved = await Data.saveCamera(id, { url: row.querySelector(".cam-url").value,
                                                    rotation: Number(row.querySelector(".cam-rotation").value),
                                                    enabled: row.querySelector(".cam-enabled").checked });
          Cameras.message(`저장했습니다: ${saved.url} · 회전 ${saved.rotation}° — 연결 확인 중…`);
          if (Live.sid) Live.message("카메라 설정이 바뀌었습니다. 실시간 보기를 멈췄다가 다시 시작하면 적용됩니다.", false);
          await Cameras.check(id, name);
        }
        await Cameras.refresh();
      } catch (ex) {
        Cameras.message(Util.errorText(ex), "error");
      } finally {
        btn.disabled = false;
      }
    },
    async check(id, name) {
      try {
        const r = await Data.checkCamera(id);
        Cameras.message(`'${name}' ${r.message} (${r.seconds}초)`, "ok");
      } catch (ex) {
        Cameras.message(`저장은 됐지만 '${name}' 연결 확인 실패: ${Util.errorText(ex)}`, "error");
      }
    },
    async add() {
      const name = Util.$("#new-camera-name");
      const url = Util.$("#new-camera-url");
      try {
        const c = await Data.addCamera({ name: name.value.trim(), url: url.value, enabled: true });
        name.value = url.value = "";
        Cameras.message(`'${c.name}' 추가: ${c.url} — 연결 확인 중…`);
        await Cameras.refresh();
        await Cameras.check(c.id, c.name);
      } catch (ex) {
        Cameras.message(Util.errorText(ex), "error");
      }
    },
  };

  // ── Init ──
  const Init = {
    index() {
      Util.$$(".tab").forEach((t) => t.addEventListener("click", Handlers.switchTab));
      Util.$("#analysis-form").addEventListener("submit", Handlers.submit);
      Util.$("#video-input").addEventListener("change", Preview.show);
      Util.$("#live-btn").addEventListener("click", Live.toggle);
      Util.$("#live-save-btn").addEventListener("click", () => Live.save(false));
      window.addEventListener("pagehide", Live.leave);
      if (Util.$("#camera-admin")) {
        Util.$("#camera-table").addEventListener("click", Cameras.onClick);
        Util.$("#camera-add").addEventListener("click", Cameras.add);
        Cameras.refresh().catch((ex) => Cameras.message(Util.errorText(ex), "error"));
      }
      // 이력 표시 + 새로고침 전에 진행 중이던 작업이 있으면 이어서 조회 (목록 요청 한 번)
      Data.list().then((jobs) => {
        View.history(jobs);
        const running = jobs.find((j) => !j.is_finished);
        if (running) {
          View.current(running);
          poll(running.id, Handlers.finished);
        }
      }).catch(() => {});
    },
    detail() {
      const id = Util.$("#detail").dataset.jobId;
      Data.detail(id).then((job) => {
        View.detail(job);
        if (!job.is_finished) poll(id, View.detail);
      });
      Util.$("#video-windows").addEventListener("click", Player.seek);
      Util.$("#job-video").addEventListener("timeupdate", Player.highlight);
      if (Util.$("#edit-form")) {                          // 관리자가 남의 기록을 볼 때는 없음
        Util.$("#edit-form").addEventListener("submit", Handlers.saveEdit);
        Util.$("#delete-btn").addEventListener("click", Handlers.remove);
      }
    },
  };

  // ── Entry ──
  document.addEventListener("DOMContentLoaded", () => {
    if (Util.$("#analysis-form")) Init.index();
    if (Util.$("#detail")) Init.detail();
  });
})();
