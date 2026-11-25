/* 관리자 대시보드: Django 가 ORM 으로 집계해 json_script 로 넘긴 값으로 Chart.js 차트를 그린다.
 * (FastAPI 는 DB 를 모른다 — 집계는 Django 몫, analyses/dashboard.py)
 * 구성: Config · Data · View · Refresh · Entry
 */
(function () {
  "use strict";

  // ── Config ──
  const css = getComputedStyle(document.documentElement);
  const color = (name, fallback) => css.getPropertyValue(name).trim() || fallback;
  const Config = {
    categoryColor: {                              // 분류 코드(actions.CATEGORY_KO 의 키) → 색. 이름은 서버가 보냄
      property_damage: "#c2410c", violence: "#b91c1c", loitering: "#2563eb", theft_suspect: "#7c3aed", emergency: "#0f766e",
    },
    statusColor: { SUCCESS: "#15803d", FAILURE: "#b91c1c", STARTED: "#d97706", PENDING: "#64748b" },
    refreshSeconds: 30,
    refreshKey: "dashboard-auto-refresh",          // 보는 사람 브라우저에만 저장하는 편의 설정
  };

  // ── Data ──
  const Data = { chart: JSON.parse(document.getElementById("chart-data").textContent) };

  // ── View ──
  const draw = (id, config) => new Chart(document.getElementById(id), config);
  const countAxis = (extra = {}) => ({ beginAtZero: true, ticks: { precision: 0 }, ...extra });

  const View = {
    base() {
      Chart.defaults.color = color("--muted", "#555");
      Chart.defaults.borderColor = color("--line", "#ddd");
      Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
    },
    actionChart() {
      const rows = Data.chart.actions;
      const fill = rows.map((r) => Config.categoryColor[r.category]);
      draw("actionChart", {
        type: "bar",
        data: {
          labels: rows.map((r) => `${r.id} ${r.name_ko}`),
          datasets: [
            { label: "예측 건수", data: rows.map((r) => r.count - r.uncertain), backgroundColor: fill },
            { label: "그중 판단 불확실", data: rows.map((r) => r.uncertain), backgroundColor: fill.map((c) => c + "55") },
          ],
        },
        options: {
          indexAxis: "y", maintainAspectRatio: false,
          scales: { x: countAxis({ stacked: true }), y: { stacked: true } },
          plugins: {
            tooltip: { callbacks: { afterTitle: (items) => rows[items[0].dataIndex].category_ko } },
            legend: { display: false },            // 막대 색이 분류별이라 아래 HTML 범례로 대신
          },
          // 막대를 누르면 그 동작으로 예측된 작업 목록으로
          onClick: (_, els) => { if (els.length) View.goFilter({ action: rows[els[0].index].id }); },
        },
      });
      const legend = document.createElement("div");
      legend.className = "legend";
      legend.innerHTML = Object.entries(Data.chart.categories)
        .map(([code, name]) => `<span><i style="background:${Config.categoryColor[code]}"></i>${name}</span>`).join("")
        + '<span><i style="background:#9ca3af55"></i>옅은 색 = 판단 불확실</span><span>막대를 누르면 아래 목록이 그 동작으로 걸러집니다</span>';
      document.getElementById("actionChart").parentElement.after(legend);
    },
    statusChart() {
      const rows = Data.chart.statuses;
      draw("statusChart", {
        type: "doughnut",
        data: {
          labels: rows.map((r) => `${r.label} (${r.status})`),
          datasets: [{ data: rows.map((r) => r.count), backgroundColor: rows.map((r) => Config.statusColor[r.status]) }],
        },
        options: {
          maintainAspectRatio: false, plugins: { legend: { position: "right" } },
          onClick: (_, els) => { if (els.length) View.goFilter({ status: rows[els[0].index].status }); },
        },
      });
    },
    inputChart() {
      const rows = Data.chart.inputs;
      draw("inputChart", {
        data: {
          labels: rows.map((r) => r.label),
          datasets: [
            { type: "bar", label: "작업 수", data: rows.map((r) => r.count), backgroundColor: "#2563eb", yAxisID: "y" },
            { type: "bar", label: "평균 AI 처리 시간(초)", data: rows.map((r) => r.avg_seconds), backgroundColor: "#94a3b8", yAxisID: "y1" },
          ],
        },
        options: {
          maintainAspectRatio: false,
          scales: {
            y: countAxis({ title: { display: true, text: "작업 수" } }),
            y1: { beginAtZero: true, position: "right", grid: { drawOnChartArea: false }, title: { display: true, text: "초" } },
          },
          plugins: { legend: { position: "bottom" } },
        },
      });
    },
    trendChart() {
      const rows = Data.chart.trend;
      const set = (label, key, c) => ({ label, data: rows.map((r) => r[key]), backgroundColor: c, stack: "jobs" });
      draw("trendChart", {
        type: "bar",
        data: {
          labels: rows.map((r) => r.day),
          datasets: [set("완료", "success", Config.statusColor.SUCCESS), set("실패", "failure", Config.statusColor.FAILURE),
                     set("대기·처리 중", "other", Config.statusColor.PENDING)],
        },
        options: { maintainAspectRatio: false, scales: { x: { stacked: true }, y: countAxis({ stacked: true }) },
                   plugins: { legend: { position: "bottom" } } },
      });
    },
    goFilter(extra) {                              // 기간은 그대로 두고 표 조건만 바꿔 이동
      const q = new URLSearchParams(location.search);
      ["page", "export"].forEach((k) => q.delete(k));
      Object.entries(extra).forEach(([k, v]) => q.set(k, v));
      location.href = `?${q}#jobs`;
    },
  };

  // ── Refresh: '30초마다 새로고침' (체크 상태는 이 브라우저에만 기억) ──
  const Refresh = {
    timer: null,
    get() { try { return localStorage.getItem(Config.refreshKey) === "1"; } catch (e) { return false; } },
    set(on) { try { localStorage.setItem(Config.refreshKey, on ? "1" : "0"); } catch (e) { /* 저장 못 해도 동작은 함 */ } },
    apply(on) {
      clearTimeout(Refresh.timer);
      if (on) Refresh.timer = setTimeout(() => location.reload(), Config.refreshSeconds * 1000);
    },
    init() {
      const box = document.getElementById("auto-refresh");
      if (!box) return;
      box.checked = Refresh.get();
      Refresh.apply(box.checked);
      box.addEventListener("change", () => { Refresh.set(box.checked); Refresh.apply(box.checked); });
    },
  };

  // ── Entry ──
  document.addEventListener("DOMContentLoaded", () => {
    View.base();
    View.actionChart();
    View.statusChart();
    View.inputChart();
    View.trendChart();
    Refresh.init();
    // 조건 고르면 바로 적용 (검색 칸은 Enter)
    document.querySelectorAll("#filter-form select, #filter-form input[type=checkbox]")
      .forEach((el) => el.addEventListener("change", () => el.form.requestSubmit()));
  });
})();
