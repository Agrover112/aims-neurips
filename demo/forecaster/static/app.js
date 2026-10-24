const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

const state = {
  tasks: [],
  checkpoints: null,
  validation: [],
  session: null,
  busy: false,
  view: "benchmark",
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function pct(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "–";
  return `${Math.round(Number(value) * 100)}%`;
}

function brier(value) {
  return Number(value).toFixed(3);
}

function brier4(value) {
  return Number(value).toFixed(4);
}

function deltaText(value, reference) {
  const diff = (Number(value) - Number(reference)) * 100;
  if (Math.abs(diff) < 0.05) return "in line with Beta";
  return `${diff > 0 ? "+" : ""}${diff.toFixed(1)} pts vs Beta`;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  let payload = null;
  try {
    payload = await response.json();
  } catch (_) {
    payload = null;
  }
  if (!response.ok) {
    throw new Error(payload?.detail || `Request failed (${response.status})`);
  }
  return payload;
}

let toastTimer = null;
function toast(message) {
  const el = $("#toast");
  el.textContent = message;
  el.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove("show"), 3200);
}

function setBusy(value) {
  state.busy = value;
  document.body.classList.toggle("is-busy", value);
  if (value) {
    $("#reveal-btn").disabled = true;
    $("#score-btn").disabled = true;
  }
}

function renderTasks() {
  const list = $("#task-list");
  const active = state.session?.task?.task_id;
  $("#task-count").textContent = state.tasks.length;
  list.innerHTML = state.tasks
    .map((task) => {
      return `
        <button class="task-card ${task.task_id === active ? "active" : ""}" data-task-id="${escapeHtml(task.task_id)}">
          <span class="task-card-top">
            <span class="task-bench">${escapeHtml(task.benchmark.name)}</span>
            <span class="task-items">${task.n_evaluation} items</span>
          </span>
          <span class="task-subject">${escapeHtml(task.subject.display_name)}</span>
          <span class="task-meta">
            <span>${task.n_acquisition} labels</span>
            <span>·</span>
            <span>${task.n_evaluation} held out</span>
          </span>
        </button>`;
    })
    .join("");
}

function renderCheckpoints() {
  const data = state.checkpoints;
  if (!data) return;
  $("#checkpoint-list").innerHTML = ["cnp", "airt"]
    .map((key) => {
      const item = data[key];
      return `
        <div class="checkpoint-row">
          <div class="checkpoint-row-top">
            <b>${key === "cnp" ? "CNP-AIRT" : "Online-AIRT"}</b>
            <code>${escapeHtml(item.file)}</code>
          </div>
          <small>${escapeHtml(item.sha256)} · ${Math.round(item.bytes / 1024)} KB · ${escapeHtml(item.mode)}</small>
        </div>`;
    })
    .join("");
}

function renderValidation() {
  $("#validation-list").innerHTML = state.validation
    .map(
      (row) => `
      <div class="validation-row">
        <span>${escapeHtml(row.model)}</span>
        <b>${brier4(row.brier_alc)}</b>
      </div>`
    )
    .join("");
}

function renderBudget(data) {
  const count = data.state.revealed_count;
  const max = data.state.budget_max;
  const grid = [0, 1, 3, 7, 15, 31];
  const current = [...grid].reverse().find((value) => value <= count) ?? 0;
  $("#budget-stepper").innerHTML = grid
    .map(
      (value) => `
      <button class="budget-step ${value === current ? "active" : ""}"
        data-budget="${value}" ${value < count ? "disabled" : ""}
        aria-label="Reveal to ${value} labels">${value}</button>`
    )
    .join("");
  $("#budget-ring").style.setProperty("--progress", `${Math.min(100, (count / max) * 100)}`);
  $("#budget-ring-value").textContent = String(count);
  $("#history-count").textContent = `${count} / ${max}`;
  $("#score-btn").disabled = state.busy || count === 0;
  $("#score-items").textContent = `${data.forecast.evaluation_count} items`;
}

function renderHistory(data) {
  const list = $("#history-list");
  if (!data.history.length) {
    list.innerHTML = `<div class="empty-state">No labels revealed yet. The forecast starts from the task prior.</div>`;
    return;
  }
  const wasAtBottom = list.scrollHeight - list.scrollTop - list.clientHeight < 32;
  list.innerHTML = data.history
    .map(
      (row) => `
      <div class="history-row">
        <span class="history-rank">${String(row.rank).padStart(2, "0")}</span>
        <span class="history-content" title="${escapeHtml(row.content)}">${escapeHtml(row.content)}</span>
        <span class="history-p">${pct(row.cnp)}</span>
        <span class="outcome ${row.outcome ? "ok" : "bad"}">${row.outcome ? "✓" : "✕"}</span>
      </div>`
    )
    .join("");
  if (wasAtBottom) list.scrollTop = list.scrollHeight;
}

function renderNext(data) {
  const next = data.next_item;
  const button = $("#reveal-btn");
  if (!next) {
    $("#next-rank").textContent = "complete";
    $("#next-content").textContent = "All acquisition labels have been revealed. Score the run to compare the forecasts.";
    $("#next-cnp").textContent = "–";
    $("#next-airt").textContent = "–";
    button.disabled = true;
    return;
  }
  $("#next-rank").textContent = `#${String(next.index + 1).padStart(2, "0")} / ${data.task.n_acquisition}`;
  $("#next-content").textContent = next.content;
  $("#next-cnp").textContent = pct(next.cnp);
  $("#next-airt").textContent = pct(next.airt);
  button.disabled = state.busy;
}

function renderForecast(data) {
  const f = data.forecast;
  $("#cnp-value").textContent = pct(f.cnp);
  $("#beta-value").textContent = pct(f.beta);
  $("#airt-value").textContent = pct(f.airt);
  $("#eval-count").textContent = f.evaluation_count;
  $("#cnp-delta").innerHTML = `<b>${deltaText(f.cnp, f.beta)}</b>`;
  renderBudget(data);
  renderChart(f);
  const caption = $("#chart-caption");
  const how = f.curve_x.length <= 1
    ? "Reveal a label to extend the curve."
    : window.matchMedia("(max-width: 640px)").matches
      ? "Reveal labels to trace the forecast."
      : "Hover the chart to inspect a budget.";
  caption.innerHTML = `Mean predicted probability over the <b>${f.evaluation_count}</b> held-out items. ${how}`;
}

function svgEl(tag, attrs = {}) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
  return node;
}

function renderChart(f) {
  const host = $("#chart");
  host.innerHTML = "";
  const width = 760;
  const height = 246;
  const pad = { left: 42, right: 16, top: 14, bottom: 30 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const maxX = 31;
  const count = Math.max(0, f.curve_x.length - 1);
  const x = (value) => pad.left + (value / maxX) * plotW;
  const y = (value) => pad.top + (1 - Math.max(0, Math.min(1, value))) * plotH;
  const svg = svgEl("svg", {
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    "aria-label": "Forecast curves across acquisition budgets",
  });

  const defs = svgEl("defs");
  const gradient = svgEl("linearGradient", { id: "cnp-area", x1: "0", x2: "0", y1: "0", y2: "1" });
  gradient.append(
    svgEl("stop", { offset: "0%", "stop-color": "#d95d39", "stop-opacity": "0.18" }),
    svgEl("stop", { offset: "100%", "stop-color": "#d95d39", "stop-opacity": "0" })
  );
  defs.append(gradient);
  svg.append(defs);

  [0, 0.25, 0.5, 0.75, 1].forEach((value) => {
    const line = svgEl("line", {
      x1: pad.left,
      x2: width - pad.right,
      y1: y(value),
      y2: y(value),
      stroke: value === 0.5 ? "#cfc7b7" : "#e6e0d4",
      "stroke-width": value === 0.5 ? 1.2 : 1,
      "stroke-dasharray": value === 0.5 ? "4 5" : "0",
    });
    svg.append(line);
    const label = svgEl("text", {
      x: pad.left - 9,
      y: y(value) + 3,
      "text-anchor": "end",
      fill: "#9ba3a5",
      "font-size": "9",
      "font-family": "SFMono-Regular, Consolas, monospace",
    });
    label.textContent = `${Math.round(value * 100)}%`;
    svg.append(label);
  });

  const ticks = [0, 1, 3, 7, 15, 31].filter((value) => value <= maxX);
  ticks.forEach((value) => {
    const label = svgEl("text", {
      x: x(value),
      y: height - 10,
      "text-anchor": "middle",
      fill: "#9ba3a5",
      "font-size": "9",
      "font-family": "SFMono-Regular, Consolas, monospace",
    });
    label.textContent = value;
    svg.append(label);
    if (value > 0) {
      svg.append(
        svgEl("line", {
          x1: x(value),
          x2: x(value),
          y1: pad.top,
          y2: height - pad.bottom,
          stroke: "#e6e0d4",
          "stroke-width": "1",
        })
      );
    }
  });

  const pathFor = (values) =>
    values
      .map((value, index) => `${index === 0 ? "M" : "L"} ${x(index).toFixed(2)} ${y(value).toFixed(2)}`)
      .join(" ");

  const cnpValues = f.cnp_curve;
  if (cnpValues.length > 1) {
    const area = svgEl("path", {
      d: `${pathFor(cnpValues)} L ${x(count)} ${y(0)} L ${x(0)} ${y(0)} Z`,
      fill: "url(#cnp-area)",
    });
    svg.append(area);
  }

  const series = [
    { values: f.beta_curve, color: "#8b8f90", width: 1.7, dash: "5 5" },
    { values: f.airt_curve, color: "#2e6f68", width: 2.1 },
    { values: f.cnp_curve, color: "#d95d39", width: 2.5 },
  ];
  series.forEach((serie) => {
    const path = svgEl("path", {
      d: pathFor(serie.values),
      fill: "none",
      stroke: serie.color,
      "stroke-width": serie.width,
      "stroke-linejoin": "round",
      "stroke-linecap": "round",
      "stroke-dasharray": serie.dash || "0",
    });
    svg.append(path);
    const final = serie.values[serie.values.length - 1];
    svg.append(
      svgEl("circle", {
        cx: x(serie.values.length - 1),
        cy: y(final),
        r: serie === series[2] ? 4.2 : 3.1,
        fill: "#fffdf8",
        stroke: serie.color,
        "stroke-width": 2,
      })
    );
  });

  const hoverLine = svgEl("line", {
    x1: 0,
    x2: 0,
    y1: pad.top,
    y2: height - pad.bottom,
    stroke: "#15212b",
    "stroke-width": "1",
    "stroke-opacity": "0.22",
    "stroke-dasharray": "3 3",
    visibility: "hidden",
  });
  svg.append(hoverLine);
  const overlay = svgEl("rect", {
    x: pad.left,
    y: pad.top,
    width: plotW,
    height: plotH,
    fill: "transparent",
    style: "cursor: crosshair",
  });
  svg.append(overlay);
  if (count === 0) {
    const hint = svgEl("text", {
      x: pad.left + plotW / 2,
      y: pad.top + plotH / 2 + 3,
      "text-anchor": "middle",
      fill: "#9ba3a5",
      "font-size": "10",
      "font-family": "Inter, Segoe UI, sans-serif",
    });
    hint.textContent = "Reveal a label to trace the forecast";
    svg.append(hint);
  }
  host.append(svg);

  const tooltip = document.createElement("div");
  tooltip.className = "chart-tooltip";
  tooltip.innerHTML = `<span class="tooltip-budget"></span><i class="dot-cnp"></i><b class="tip-cnp"></b><i class="dot-airt"></i><b class="tip-airt"></b><i class="dot-beta"></i><b class="tip-beta"></b>`;
  host.append(tooltip);

  let hoverIndex = null;
  const showAt = (index) => {
    const clamped = Math.max(0, Math.min(count, index));
    hoverIndex = clamped;
    hoverLine.setAttribute("x1", x(clamped));
    hoverLine.setAttribute("x2", x(clamped));
    hoverLine.setAttribute("visibility", "visible");
    const rect = host.getBoundingClientRect();
    const left = (x(clamped) / width) * rect.width;
    const top = (y(Math.max(f.cnp_curve[clamped], f.airt_curve[clamped], f.beta_curve[clamped])) / height) * rect.height;
    tooltip.style.left = `${Math.min(rect.width - 118, Math.max(6, left + 12))}px`;
    tooltip.style.top = `${Math.max(4, top - 14)}px`;
    tooltip.querySelector(".tooltip-budget").textContent = `${clamped} label${clamped === 1 ? "" : "s"}`;
    tooltip.querySelector(".tip-cnp").textContent = pct(f.cnp_curve[clamped]);
    tooltip.querySelector(".tip-airt").textContent = pct(f.airt_curve[clamped]);
    tooltip.querySelector(".tip-beta").textContent = pct(f.beta_curve[clamped]);
    tooltip.classList.add("show");
  };
  const hide = () => {
    hoverLine.setAttribute("visibility", "hidden");
    tooltip.classList.remove("show");
  };
  overlay.addEventListener("mousemove", (event) => {
    const rect = svg.getBoundingClientRect();
    const vx = ((event.clientX - rect.left) / rect.width) * width;
    const index = Math.round(((vx - pad.left) / plotW) * maxX);
    showAt(index);
  });
  overlay.addEventListener("mouseleave", hide);
  overlay.addEventListener("click", () => {
    if (hoverIndex !== null && hoverIndex > count) hide();
  });
}

function renderScore(data) {
  const empty = $("#score-empty");
  const results = $("#score-results");
  if (!data.score) {
    empty.classList.remove("hidden");
    results.classList.add("hidden");
    return;
  }
  empty.classList.add("hidden");
  results.classList.remove("hidden");
  const score = data.score;
  $("#actual-success").textContent = pct(score.actual_success);
  const rows = [
    ["CNP-AIRT", score.brier.cnp, "cnp"],
    ["Online-AIRT", score.brier.airt, "airt"],
    ["Beta", score.brier.beta, "beta"],
  ];
  const max = Math.max(0.35, ...rows.map((row) => row[1]));
  $("#score-bars").innerHTML = rows
    .map(([label, value, className]) => {
      const width = Math.max(3, Math.min(100, (Number(value) / max) * 100));
      return `
        <div class="score-bar">
          <span class="score-bar-label">${label}</span>
          <span class="score-bar-track"><span class="score-bar-fill ${className}" style="width:${width}%"></span></span>
          <span class="score-bar-value">${brier(value)}</span>
        </div>`;
    })
    .join("");
}

function renderSession(data) {
  state.session = data;
  const task = data.task;
  const policy = data.state.policy;
  $("#policy-select").value = policy;
  renderTasks();
  renderForecast(data);
  renderNext(data);
  renderHistory(data);
  renderScore(data);
}

function renderPolicySelect() {
  const select = $("#policy-select");
  select.addEventListener("change", async () => {
    if (!state.session) return;
    try {
      const data = await api(`/api/sessions/${state.session.session_id}/policy`, {
        method: "POST",
        body: JSON.stringify({ policy: select.value }),
      });
      renderSession(data);
    } catch (error) {
      toast(error.message);
    }
  });
}

async function createSession(taskId) {
  if (state.busy) return;
  setBusy(true);
  try {
    const data = await api("/api/sessions", {
      method: "POST",
      body: JSON.stringify({ task_id: taskId, policy: $("#policy-select").value }),
    });
    renderSession(data);
    toast(`${data.task.benchmark.name} loaded. 60 held-out items.`);
  } catch (error) {
    toast(error.message);
  } finally {
    setBusy(false);
    if (state.session) {
      $("#reveal-btn").disabled = !state.session.next_item;
      $("#score-btn").disabled = state.session.state.revealed_count === 0;
    }
  }
}

async function reveal() {
  if (!state.session || state.busy) return;
  setBusy(true);
  try {
    const data = await api(`/api/sessions/${state.session.session_id}/reveal`, { method: "POST" });
    renderSession(data);
  } catch (error) {
    toast(error.message);
  } finally {
    setBusy(false);
    if (state.session) {
      $("#reveal-btn").disabled = !state.session.next_item;
      $("#score-btn").disabled = state.session.state.revealed_count === 0;
    }
  }
}

async function advanceTo(budget) {
  if (!state.session || state.busy) return;
  if (budget < state.session.state.revealed_count) {
    toast("Reset the run before lowering the budget.");
    return;
  }
  if (budget === state.session.state.revealed_count) return;
  setBusy(true);
  try {
    const data = await api(`/api/sessions/${state.session.session_id}/advance`, {
      method: "POST",
      body: JSON.stringify({ budget }),
    });
    renderSession(data);
  } catch (error) {
    toast(error.message);
  } finally {
    setBusy(false);
    if (state.session) {
      $("#reveal-btn").disabled = !state.session.next_item;
      $("#score-btn").disabled = state.session.state.revealed_count === 0;
    }
  }
}

async function resetRun() {
  if (!state.session || state.busy) return;
  setBusy(true);
  try {
    const data = await api(`/api/sessions/${state.session.session_id}/reset`, { method: "POST" });
    renderSession(data);
    toast("Run reset.");
  } catch (error) {
    toast(error.message);
  } finally {
    setBusy(false);
    if (state.session) {
      $("#reveal-btn").disabled = !state.session.next_item;
      $("#score-btn").disabled = state.session.state.revealed_count === 0;
    }
  }
}

async function scoreRun() {
  if (!state.session || state.busy) return;
  setBusy(true);
  try {
    const data = await api(`/api/sessions/${state.session.session_id}/score`, { method: "POST" });
    renderSession(data);
    $("#score-card").scrollIntoView({ behavior: "smooth", block: "center" });
  } catch (error) {
    toast(error.message);
  } finally {
    setBusy(false);
    if (state.session) {
      $("#reveal-btn").disabled = !state.session.next_item;
      $("#score-btn").disabled = state.session.state.revealed_count === 0;
    }
  }
}

async function runCustom() {
  const subject = $("#custom-subject").value.trim();
  const item = $("#custom-item").value.trim();
  if (!item) {
    toast("Add an item prompt first.");
    return;
  }
  const result = $("#custom-result");
  result.innerHTML = `<div class="empty-state">Running the probe…</div>`;
  try {
    const data = await api("/api/custom", {
      method: "POST",
      body: JSON.stringify({ subject, item }),
    });
    const support = Math.max(0, Math.min(1, Number(data.support)));
    result.innerHTML = `
      <div class="probe-result">
        <div class="probe-big"><b>${pct(data.cnp)}</b><span>CNP-AIRT · prior-only</span></div>
        <div class="probe-row"><span>Online-AIRT</span><b>${pct(data.airt)}</b></div>
        <div class="probe-row"><span>Nearest demo-item support</span><b>${support.toFixed(2)} · ${escapeHtml(data.support_label)}</b></div>
        <div class="support-track"><div class="support-fill" style="width:${support * 100}%"></div></div>
        <p class="probe-note">Support is a nearest-neighbour check against the 455 demo items — not a calibrated OOD detector. Subject features are masked in this checkpoint.</p>
      </div>`;
  } catch (error) {
    result.innerHTML = `<div class="empty-state">${escapeHtml(error.message)}</div>`;
  }
}

function setView(view) {
  state.view = view;
  $$(".navlink[data-view]").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === view);
  });
  $("#benchmark-hero").classList.toggle("hidden", view !== "benchmark");
  $("#benchmark-view").classList.toggle("hidden", view !== "benchmark");
  $("#custom-view").classList.toggle("hidden", view !== "custom");
  if (view === "custom") {
    $("#custom-view").scrollIntoView({ behavior: "smooth", block: "start" });
  } else {
    window.scrollTo({ top: 0, behavior: "smooth" });
  }
}

function bindEvents() {
  $("#task-list").addEventListener("click", (event) => {
    const card = event.target.closest("[data-task-id]");
    if (card) createSession(card.dataset.taskId);
  });
  $("#custom-cta").addEventListener("click", () => setView("custom"));
  $$(".navlink[data-view]").forEach((button) => {
    button.addEventListener("click", () => setView(button.dataset.view));
  });
  $$('a[href="#method"]').forEach((link) => {
    link.addEventListener("click", (event) => {
      event.preventDefault();
      setView("benchmark");
      setTimeout(() => document.querySelector("#method")?.scrollIntoView({ behavior: "smooth", block: "start" }), 50);
    });
  });
  $("#budget-stepper").addEventListener("click", (event) => {
    const button = event.target.closest("[data-budget]");
    if (button) advanceTo(Number(button.dataset.budget));
  });
  $("#reveal-btn").addEventListener("click", reveal);
  $("#reset-btn").addEventListener("click", resetRun);
  $("#score-btn").addEventListener("click", scoreRun);
  $("#custom-run").addEventListener("click", runCustom);
  $("#protocol-btn").addEventListener("click", () => $("#protocol-dialog").showModal());
  $("#dialog-close").addEventListener("click", () => $("#protocol-dialog").close());
  $("#protocol-dialog").addEventListener("click", (event) => {
    if (event.target === $("#protocol-dialog")) $("#protocol-dialog").close();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") $("#protocol-dialog").close();
    if (event.key === "Enter" && !event.shiftKey && !["TEXTAREA", "INPUT", "SELECT", "BUTTON"].includes(event.target.tagName)) {
      if (state.view === "benchmark" && state.session?.next_item) reveal();
    }
  });
}

async function boot() {
  bindEvents();
  renderPolicySelect();
  if (location.hash === "#custom") setView("custom");
  try {
    const meta = await api("/api/tasks");
    state.tasks = meta.tasks;
    state.checkpoints = meta.checkpoints;
    state.validation = meta.validation;
    renderTasks();
    renderCheckpoints();
    renderValidation();
    if (state.tasks.length) await createSession(state.tasks[0].task_id);
  } catch (error) {
    toast(error.message);
    $("#task-list").innerHTML = `<div class="empty-state">${escapeHtml(error.message)}</div>`;
  }
}

boot();
