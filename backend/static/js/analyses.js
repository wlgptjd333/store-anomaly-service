/* 분석 화면 (index.html · detail.html)
 * 흐름: AI 분석 시작 → POST /api/analyses/ (작업 ID 즉시 받음) → /status/ 를 2초마다 조회
 *       → 완료되면 상세 GET 으로 사람별 결과 표시 → 최근 이력 새로고침
 * 구성: Config · Util · Data(API) · View · Handlers · Init · Entry (수업 DRF 문제의 JS 구조와 같음)
 */
(function () {
  "use strict";

  // ── Config ──
  const Config = {
    api: "/api/analyses/",
    pollMs: 2000,
    pollLimit: 450,                       // 2초 × 450 = 15분 (Celery 시간 제한과 같음)
    statusKo: { PENDING: "대기", STARTED: "처리 중", SUCCESS: "완료", FAILURE: "실패" },
  };

  // ── Util ──
  const Util = {
    $(sel, root = document) { return root.querySelector(sel); },
    $$(sel, root = document) { return Array.from(root.querySelectorAll(sel)); },
    csrf() {
      const m = document.cookie.match(/(?:^|;\s*)csrftoken=([^;]+)/);
      return m ? decodeURIComponent(m[1]) : "";
    },
    esc(s) {
      return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
    },
    pct(x) { return x == null ? "-" : `${(x * 100).toFixed(0)}%`; },
    sec(x) { return x == null ? "-" : `${Number(x).toFixed(1)}초`; },
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
    uncertainBelow() {
      const el = document.getElementById("uncertain-below");
      return el ? JSON.parse(el.textContent) : 0.3;
    },
  };

  // ── Data (DRF API) ──
  const http = axios.create({ headers: { "X-CSRFToken": Util.csrf() } });
  const Data = {
    create: (formData) => http.post(Config.api, formData).then((r) => r.data),
    list: () => http.get(Config.api).then((r) => r.data.results ?? r.data),
    detail: (id) => http.get(`${Config.api}${id}/`).then((r) => r.data),
    status: (id) => http.get(`${Config.api}${id}/status/`).then((r) => r.data),
    update: (id, body) => http.patch(`${Config.api}${id}/`, body).then((r) => r.data),
    remove: (id) => http.delete(`${Config.api}${id}/`),
    // 카메라 (일반 사용자는 이름만, 관리자는 주소까지)
    cameras: () => http.get("/api/cameras/").then((r) => r.data),
    addCamera: (body) => http.post("/api/cameras/", body).then((r) => r.data),
    saveCamera: (id, body) => http.patch(`/api/cameras/${id}/`, body).then((r) => r.data),
    deleteCamera: (id) => http.delete(`/api/cameras/${id}/`),
    checkCamera: (id) => http.post(`/api/cameras/${id}/check/`).then((r) => r.data),
    // 실시간 보기 (보기용, DB 에 남기지 않음)
    liveStart: (camera_name) => http.post("/api/live/", { camera_name }).then((r) => r.data),
    liveState: (sid) => http.get(`/api/live/${sid}/`).then((r) => r.data),
    liveStop: (sid) => http.delete(`/api/live/${sid}/`),
  };

  // ── View ──
  const View = {
    badge(status) {
      return `<span class="badge s-${status}">${Config.statusKo[status] ?? status}</span>`;
    },
    predictionLine(p) {
      const flag = p.uncertain ? ' <span class="badge warn">판단 불확실</span>' : "";
      return `#${p.track_id} ${Util.esc(p.action_id)} ${Util.esc(p.action_name_ko)} ${Util.pct(p.confidence)}${flag}`;
    },
    resultTable(job) {
      if (job.status === "FAILURE") {
        return `<p class="error">실패 사유: ${Util.esc(job.error_message)}</p>`;
      }
      if (job.status !== "SUCCESS") return "";
      if (!job.predictions.length) {
        return `<p class="warn-text">${Util.esc(job.result_message || "예측 결과가 없습니다.")}</p>`;
      }
      const limit = Util.uncertainBelow();
      const rows = job.predictions.map((p) => {
        const match = job.expected_action
          ? (job.expected_action === p.action_id ? '<span class="badge ok">일치</span>' : '<span class="badge bad">불일치</span>')
          : "-";
        const top3 = p.top3.map((t) => `${Util.esc(t.action_id)} ${Util.esc(t.action_name_ko)} ${Util.pct(t.probability)}`).join("<br>");
        return `<tr class="${p.uncertain ? "uncertain" : ""}">
          <td>#${p.track_id}</td>
          <td><strong>${Util.esc(p.action_id)} ${Util.esc(p.action_name_ko)}</strong><br><span class="muted">${Util.esc(p.action_label)} · ${Util.esc(p.category_ko)}</span></td>
          <td><div class="bar"><span style="width:${(p.confidence * 100).toFixed(0)}%"></span></div>${Util.pct(p.confidence)}${p.uncertain ? '<br><span class="badge warn">판단 불확실</span>' : ""}</td>
          <td>${match}</td>
          <td class="small">${top3}</td>
          <td class="small">창 ${p.window_count}개<br>유효 Frame ${p.valid_frames}<br>${Util.sec(p.track_seconds)} 보임<br>관절 Conf ${p.mean_keypoint_conf ?? "-"}</td>
        </tr>`;
      }).join("");
      return `<table class="table">
        <thead><tr><th>Track</th><th>예측 동작</th><th>Confidence</th><th>수행한 동작과</th><th>상위 3개</th><th>입력 품질</th></tr></thead>
        <tbody>${rows}</tbody></table>
        <p class="muted small">Confidence ${Util.pct(limit)} 미만은 '판단 불확실'로 표시합니다. 17개 동작 중 하나를 고르는 모델이라, 학습하지 않은 동작에도 가장 가까운 동작을 답합니다.</p>`;
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
      Util.$("#current-id").innerHTML = `#${job.id} ${View.badge(job.status)}`;
      View.steps(job.status);
      const msg = { PENDING: "작업을 대기열에 넣었습니다.", STARTED: "AI 서버가 영상을 분석하고 있습니다…" }[job.status];
      Util.$("#current-message").textContent = msg ?? (job.processing_seconds ? `AI 처리 ${Util.sec(job.processing_seconds)} · 영상 ${Util.sec(job.video_seconds)}` : "");
      Util.$("#current-result").innerHTML = job.predictions ? View.resultTable(job) + `<p><a href="/analyses/${job.id}/">상세 보기 →</a></p>` : "";
    },
    history(jobs) {
      const body = Util.$("#history tbody");
      if (!body) return;
      body.innerHTML = jobs.length ? jobs.map((j) => `<tr>
          <td><a href="/analyses/${j.id}/">${j.id}</a></td>
          <td class="nowrap">${Util.esc(j.input_type_display)}</td>
          <td>${j.expected_action ? Util.esc(j.expected_action_display) : "-"}</td>
          <td>${j.predictions.map(View.predictionLine).join("<br>") || (j.status === "FAILURE" ? `<span class="error">${Util.esc(j.error_message)}</span>` : Util.esc(j.result_message) || "-")}</td>
          <td>${View.badge(j.status)}</td>
          <td class="nowrap">${Util.sec(j.processing_seconds)}</td>
          <td class="nowrap">${Util.time(j.created_at)}</td></tr>`).join("")
        : '<tr><td colspan="7" class="muted">아직 분석 기록이 없습니다.</td></tr>';
    },
    detail(job) {
      Util.$("#detail-status").outerHTML = `<span id="detail-status">${View.badge(job.status)}</span>`;
      const meta = [
        ["입력 방식", job.input_type_display + (job.camera_name ? ` (${Util.esc(job.camera_name)})` : "")],
        ["영상", Util.esc(job.video_name) || "-"],
        ["수행한 동작(사용자 선택)", job.expected_action ? Util.esc(job.expected_action_display) : "-"],
        ["영상 길이 / AI 처리 시간", `${Util.sec(job.video_seconds)} / ${Util.sec(job.processing_seconds)}`],
        ["요청 / 완료", `${Util.time(job.created_at)} / ${Util.time(job.finished_at)}`],
        ["Celery Task ID", `<code>${Util.esc(job.celery_task_id)}</code>`],
      ];
      Util.$("#detail-meta").innerHTML = meta.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
      Util.$("#detail-result").innerHTML = View.resultTable(job) || `<p class="muted">${Config.statusKo[job.status]}…</p>`;
    },
  };

  // ── Handlers ──
  const state = { input: "UPLOAD", polling: null };

  async function poll(id, onDone, count = 0) {
    try {
      const s = await Data.status(id);
      if (s.is_finished) return onDone(await Data.detail(id));
      if (Util.$("#current")) View.current({ id, status: s.status });
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
      const err = Util.$("#form-error");
      const btn = Util.$("#start-btn");
      err.hidden = true;
      const fd = new FormData();
      fd.append("input_type", state.input);
      if (state.input === "UPLOAD") {
        const file = form.video.files[0];
        if (!file) { err.textContent = "영상 파일을 선택하세요."; err.hidden = false; return; }
        fd.append("video", file);
      } else if (form.camera_name) {
        fd.append("camera_name", form.camera_name.value);
      }
      ["expected_action", "title"].forEach((k) => fd.append(k, form[k].value));
      btn.disabled = true;
      btn.textContent = "요청 중…";
      try {
        const created = await Data.create(fd);   // 202: 분석은 아직, 작업 ID 만 받음
        View.current({ id: created.id, status: created.status });
        clearTimeout(state.polling);
        poll(created.id, (job) => { View.current(job); Handlers.refreshHistory(); });
        Handlers.refreshHistory();
        form.reset();
      } catch (ex) {
        err.textContent = Util.errorText(ex);
        err.hidden = false;
      } finally {
        btn.disabled = false;
        btn.textContent = "AI 분석 시작";
      }
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
        Util.$("#edit-message").textContent = "저장했습니다.";
      } catch (ex) {
        Util.$("#edit-message").textContent = Util.errorText(ex);
      }
    },
    async remove() {
      const id = Util.$("#detail").dataset.jobId;
      if (!confirm(`분석 #${id} 기록과 사람별 결과, 영상 파일을 삭제할까요? 되돌릴 수 없습니다.`)) return;
      await Data.remove(id);
      location.href = "/";
    },
  };

  // ── 실시간 보기: <img> 는 MJPEG 를 계속 받고, 사람별 예측은 1초마다 JSON 으로 ──
  const Live = {
    sid: null,
    timer: null,
    pollMs: 1000,
    stateKo: { connecting: "연결 중", running: "실시간", stopped: "멈춤", error: "오류" },
    message(text, isError = true) {
      const el = Util.$("#live-message");
      el.textContent = text || "";
      el.className = isError ? "error" : "muted";
      el.hidden = !text;
    },
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
        const top3 = p.top3.slice(1).map((t) => `${Util.esc(t.action_id)} ${Util.esc(t.action_name_ko)} ${Util.pct(t.probability)}`).join(" · ");
        return `<tr class="${p.uncertain ? "uncertain" : ""}">
          <td>#${p.track_id}</td>
          <td><strong>${Util.esc(p.action_id)} ${Util.esc(p.action_name_ko)}</strong><br><span class="muted small">다음 후보: ${top3}</span></td>
          <td><div class="bar"><span style="width:${(p.confidence * 100).toFixed(0)}%"></span></div>${Util.pct(p.confidence)}${p.uncertain ? '<br><span class="badge warn">판단 불확실</span>' : ""}</td></tr>`;
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
      if (snap.state === "error" || snap.state === "stopped") Live.end(snap.message, snap.state === "error");
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
      fetch(`/api/live/${Live.sid}/`, { method: "DELETE", keepalive: true, headers: { "X-CSRFToken": Util.csrf() } });
    },
  };

  // ── 카메라 주소 관리 (관리자만 화면에 있음) ──
  const Cameras = {
    message(text, ok) {
      const el = Util.$("#camera-message");
      el.textContent = text;
      el.className = `small ${ok ? "ok-text" : "error"}`;
      el.hidden = !text;
    },
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
          <td><input type="checkbox" class="cam-enabled" ${c.enabled ? "checked" : ""}></td>
          <td class="nowrap"><button type="button" class="primary small-btn" data-act="save">저장 + 연결 확인</button>
            <button type="button" class="danger small-btn" data-act="delete">삭제</button></td></tr>`).join("")
        : '<tr><td colspan="4" class="muted">등록된 카메라가 없습니다.</td></tr>';
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
          Cameras.message(`'${name}'를 지웠습니다.`, true);
        } else {
          const saved = await Data.saveCamera(id, { url: row.querySelector(".cam-url").value,
                                                    enabled: row.querySelector(".cam-enabled").checked });
          Cameras.message(`저장했습니다: ${saved.url} — 연결 확인 중…`, true);
          await Cameras.check(id, name);
        }
        await Cameras.refresh();
      } catch (ex) {
        Cameras.message(Util.errorText(ex), false);
      } finally {
        btn.disabled = false;
      }
    },
    async check(id, name) {
      try {
        const r = await Data.checkCamera(id);
        Cameras.message(`'${name}' ${r.message} (${r.seconds}초)`, true);
      } catch (ex) {
        Cameras.message(`저장은 됐지만 '${name}' 연결 확인 실패: ${Util.errorText(ex)}`, false);
      }
    },
    async add() {
      const name = Util.$("#new-camera-name");
      const url = Util.$("#new-camera-url");
      try {
        const c = await Data.addCamera({ name: name.value.trim(), url: url.value, enabled: true });
        name.value = url.value = "";
        Cameras.message(`'${c.name}' 추가: ${c.url} — 연결 확인 중…`, true);
        await Cameras.refresh();
        await Cameras.check(c.id, c.name);
      } catch (ex) {
        Cameras.message(Util.errorText(ex), false);
      }
    },
  };

  // ── Init ──
  const Init = {
    index() {
      Util.$$(".tab").forEach((t) => t.addEventListener("click", Handlers.switchTab));
      Util.$("#analysis-form").addEventListener("submit", Handlers.submit);
      Util.$("#live-btn").addEventListener("click", Live.toggle);
      window.addEventListener("pagehide", Live.leave);
      if (Util.$("#camera-admin")) {
        Util.$("#camera-table").addEventListener("click", Cameras.onClick);
        Util.$("#camera-add").addEventListener("click", Cameras.add);
        Cameras.refresh().catch((ex) => Cameras.message(Util.errorText(ex), false));
      }
      Handlers.refreshHistory();
      // 새로고침 전에 진행 중이던 작업이 있으면 이어서 조회
      Data.list().then((jobs) => {
        const running = jobs.find((j) => !j.is_finished);
        if (running) {
          View.current(running);
          poll(running.id, (job) => { View.current(job); Handlers.refreshHistory(); });
        }
      }).catch(() => {});
    },
    detail() {
      const id = Util.$("#detail").dataset.jobId;
      Data.detail(id).then((job) => {
        View.detail(job);
        if (!job.is_finished) poll(id, View.detail);
      });
      Util.$("#edit-form").addEventListener("submit", Handlers.saveEdit);
      Util.$("#delete-btn").addEventListener("click", Handlers.remove);
    },
  };

  // ── Entry ──
  document.addEventListener("DOMContentLoaded", () => {
    if (Util.$("#analysis-form")) Init.index();
    if (Util.$("#detail")) Init.detail();
  });
})();
