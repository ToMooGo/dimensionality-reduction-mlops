/* PCA Lab - vanilla JS front-end for the model service. */
"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
// Escape any server-provided text before it goes into innerHTML.
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

async function api(path, options = {}) {
  const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail ? (typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail)) : res.statusText);
  return body;
}

const dark = () => document.documentElement.dataset.theme === "dark" ||
  (document.documentElement.dataset.theme !== "light" && matchMedia("(prefers-color-scheme: dark)").matches);
// eight categorical hues in fixed order + two neutrals for digits 8 and 9 (digits are also written on the map)
const DIGIT_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948", "#52514e", "#a8a7a0"];
const DIGIT_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767", "#c3c2b7", "#6f6e69"];
const digitColors = () => (dark() ? DIGIT_DARK : DIGIT_LIGHT);

/* ---------- 28x28 rendering (0 = white background, 255 = black ink) ---------- */
function drawDigit(canvas, pixels) {
  canvas.width = 28; canvas.height = 28;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(28, 28);
  pixels.forEach((v, i) => {
    const g = 255 - Math.round(Math.max(0, Math.min(255, v)));
    img.data.set([g, g, g, 255], i * 4);
  });
  ctx.putImageData(img, 0, 0);
  canvas.classList.add("pixelated");
}

/* ---------- tabs ---------- */
$$(".tab[data-tab]").forEach((btn) => btn.addEventListener("click", () => {
  $$(".tab[data-tab]").forEach((b) => { b.classList.toggle("active", b === btn); b.setAttribute("aria-selected", b === btn); });
  $$(".panel").forEach((p) => { const on = p.id === `tab-${btn.dataset.tab}`; p.hidden = !on; p.classList.toggle("active", on); });
  if (btn.dataset.tab === "maps") loadMaps();
  if (btn.dataset.tab === "monitor") loadMonitoring();
}));

/* ---------- health ---------- */
async function refreshHealth() {
  const el = $("#status");
  try {
    const h = await api("/health");
    const v = h.model_versions || {};
    el.className = "status " + (h.status === "ok" ? "ok" : "bad");
    $("#status-text").textContent = h.status === "ok"
      ? `@${h.alias}: PCA v${v.compressor} · SVMs v${v.classifier_raw}/v${v.classifier_pca} · detector v${v.detector} · maps v${v.atlas}`
      : "no models deployed yet - run the pipeline";
  } catch (e) {
    el.className = "status bad";
    $("#status-text").textContent = "service unreachable";
  }
}

/* ---------- drawing pad -> 28x28 MNIST format ---------- */
const pad = $("#pad");
const pctx = pad.getContext("2d");
let drawing = false, hasInk = false;
function resetPad() {
  pctx.clearRect(0, 0, pad.width, pad.height); hasInk = false; drawDigit($("#preview"), new Array(784).fill(0));
  $("#pad").classList.add("empty-pad");
}
function pos(e) { const r = pad.getBoundingClientRect(); return [(e.clientX - r.left) * pad.width / r.width, (e.clientY - r.top) * pad.height / r.height]; }
pad.addEventListener("pointerdown", (e) => { $("#pad").classList.remove("empty-pad"); $$(".chip[data-kind]").forEach((c) => c.classList.remove("on")); drawing = true; pad.setPointerCapture(e.pointerId); const [x, y] = pos(e); pctx.beginPath(); pctx.moveTo(x, y); pctx.lineTo(x + 0.1, y + 0.1); stroke(); });
pad.addEventListener("pointermove", (e) => { if (!drawing) return; const [x, y] = pos(e); pctx.lineTo(x, y); stroke(); });
pad.addEventListener("pointerup", () => { drawing = false; const px = padTo28(); if (px) drawDigit($("#preview"), px); });
function stroke() { pctx.lineWidth = 20; pctx.lineCap = "round"; pctx.lineJoin = "round"; pctx.strokeStyle = "#111"; pctx.stroke(); hasInk = true; }

// MNIST format: the digit's bounding box is scaled to fit 20x20 (aspect kept, anti-aliased),
// then placed on a 28x28 grid with its centre of mass at the centre.
function padTo28() {
  if (!hasInk) return null;
  const { width: w, height: h } = pad;
  const data = pctx.getImageData(0, 0, w, h).data;
  let x0 = w, y0 = h, x1 = -1, y1 = -1;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    if (data[(y * w + x) * 4 + 3] > 30) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
  }
  if (x1 < 0) return null;
  const bw = x1 - x0 + 1, bh = y1 - y0 + 1;
  const scale = 20 / Math.max(bw, bh);
  const dw = Math.max(1, Math.round(bw * scale)), dh = Math.max(1, Math.round(bh * scale));
  // downscale in two steps for smoother anti-aliasing
  const mid = document.createElement("canvas"); mid.width = dw * 4; mid.height = dh * 4;
  const mctx = mid.getContext("2d"); mctx.imageSmoothingQuality = "high";
  mctx.drawImage(pad, x0, y0, bw, bh, 0, 0, mid.width, mid.height);
  const small = document.createElement("canvas"); small.width = 28; small.height = 28;
  const sctx = small.getContext("2d"); sctx.imageSmoothingQuality = "high";
  sctx.drawImage(mid, 0, 0, mid.width, mid.height, Math.round((28 - dw) / 2), Math.round((28 - dh) / 2), dw, dh);
  const sd = sctx.getImageData(0, 0, 28, 28).data;
  const g = new Array(784).fill(0);
  let mass = 0, cx = 0, cy = 0;
  for (let i = 0; i < 784; i++) { const v = sd[i * 4 + 3]; g[i] = v; mass += v; cx += v * (i % 28); cy += v * Math.floor(i / 28); }
  if (!mass) return null;
  const sx = Math.round(13.5 - cx / mass), sy = Math.round(13.5 - cy / mass);
  const out = new Array(784).fill(0);
  for (let y = 0; y < 28; y++) for (let x = 0; x < 28; x++) {
    const nx = x + sx, ny = y + sy;
    if (nx >= 0 && nx < 28 && ny >= 0 && ny < 28) out[ny * 28 + nx] = g[y * 28 + x];
  }
  return out;
}

/* ---------- current input ---------- */
const state = { pixels: null, source: null, trueLabel: null, kind: null, positions: null, d: 154, marks: null };

$("#clear").addEventListener("click", resetPad);
$("#use-draw").addEventListener("click", async () => {
  const px = padTo28();
  if (!px) { showError("Draw a digit first."); return; }
  await setInput(px, "canvas", null, "drawing");
});
$$(".chip[data-kind]").forEach((chip) => chip.addEventListener("click", async () => {
  $$(".chip[data-kind]").forEach((c) => c.classList.toggle("on", c === chip));
  try {
    const s = await api(`/samples?kind=${chip.dataset.kind}`);
    resetPad();
    drawDigit($("#preview"), s.pixels);
    await setInput(s.pixels, `sample-${s.kind}`, s.kind === "clean" ? s.true_label : null, s.kind);
  } catch (e) { showError(e.message); }
}));

async function setInput(pixels, source, trueLabel, kind) {
  Object.assign(state, { pixels, source, trueLabel, kind });
  drawDigit($("#c-orig"), pixels);
  await Promise.all([runCompress(), runPrediction()]);
}

/* ---------- compress ---------- */
async function loadCompressInfo() {
  try {
    const info = await api("/compress/info");
    state.marks = info.marks;
    $("#d-marks").innerHTML = Object.entries(info.marks).map(([v, d]) =>
      `<button class="chip${d === state.d ? " on" : ""}" data-d="${d}">${esc(v)} &middot; d=${d}</button>`).join("");
    $$("#d-marks .chip").forEach((c) => c.addEventListener("click", () => setD(+c.dataset.d)));
    if (info.marks["95%"]) setD(info.marks["95%"], false);
  } catch (e) { /* models not deployed yet */ }
}
function setD(d, run = true) {
  state.d = d; $("#d-slider").value = d; $("#d-value").textContent = d;
  $$("#d-marks .chip").forEach((c) => c.classList.toggle("on", +c.dataset.d === d));
  if (run) runCompress();
}
let compressTimer = null;
$("#d-slider").addEventListener("input", (e) => {
  state.d = +e.target.value; $("#d-value").textContent = state.d;
  $$("#d-marks .chip").forEach((c) => c.classList.toggle("on", +c.dataset.d === state.d));
  clearTimeout(compressTimer); compressTimer = setTimeout(runCompress, 90);
});
async function runCompress() {
  if (!state.pixels) return;
  try {
    const r = await api("/compress", { method: "POST", body: JSON.stringify({ pixels: state.pixels, n_components: state.d }) });
    $("#compress-empty").hidden = true; $("#compress").hidden = false;
    drawDigit($("#c-rec"), r.reconstruction);
    $("#c-rec-cap").textContent = `rebuilt from ${r.n_components} values`;
    $("#s-var").textContent = `${(100 * r.variance_kept).toFixed(1)}%`;
    $("#s-feat").innerHTML = `${r.n_components} of 784<small>${r.pct_of_features.toFixed(1)}% of the features</small>`;
    $("#s-rmse").textContent = `${r.rmse_grey_levels.toFixed(1)} grey levels`;
    $("#s-bytes").innerHTML = `PCA ${r.bytes_float32_code} B<small>raw image ${r.bytes_uint8_pixels} B (uint8)</small>`;
    $("#s-bytes").title = "float32 PCA code vs the raw uint8 pixels";
  } catch (e) { $("#compress-empty").hidden = false; $("#compress").hidden = true; $("#compress-empty").textContent = e.message; }
}

/* ---------- classify ---------- */
function renderBars(el, scores, top, alt) {
  el.className = "bars" + (alt ? " alt" : "");
  const max = Math.max(9, ...scores);
  el.innerHTML = scores.map((s, d) =>
    `<div class="bar ${d === top ? "top" : ""}" title="digit ${d}: ${s.toFixed(2)} votes"><i style="height:${Math.max(2, 100 * Math.max(0, s + 0.5) / (max + 0.5))}%"></i><span>${d}</span></div>`).join("");
}
function showError(msg) {
  $("#result-empty").hidden = false; $("#result").hidden = true;
  $("#result-empty").textContent = msg;
}
async function runPrediction() {
  try {
    const r = await api("/predict", { method: "POST", body: JSON.stringify({ pixels: state.pixels, source: state.source }) });
    $("#result-empty").hidden = true; $("#result").hidden = false;
    const u = r.unusual;
    const v = $("#verdict");
    v.className = "verdict " + (u.is_unusual ? "anom" : "ok");
    // gauge on a log scale: from 10 to 100x the threshold
    const lo = Math.log10(10), hi = Math.log10(u.threshold * 100);
    const at = (x) => Math.max(0, Math.min(100, 100 * (Math.log10(Math.max(x, 10)) - lo) / (hi - lo)));
    v.innerHTML = `<span class="icon">${u.is_unusual ? "&#9888;" : "&#10003;"}</span><div style="flex:1">
      <strong>${u.is_unusual ? "Unusual input" : "Looks like a training digit"}</strong>
      <small>${esc(u.message)} Reconstruction error ${u.reconstruction_error.toLocaleString()} vs threshold ${u.threshold.toLocaleString()} (${u.error_percentile >= 99.95 ? "higher than every held-out clean digit" : `higher than ${u.error_percentile}% of held-out clean digits`}).</small>
      <div class="gauge" aria-hidden="true"><i style="width:${at(u.reconstruction_error)}%"></i><b style="left:${at(u.threshold)}%"></b></div>
      <div class="gauge-labels"><span>error, log scale</span><span class="thr" style="left:${at(u.threshold)}%">threshold</span></div></div>`;
    for (const [key, alt] of [["raw", false], ["pca", true]]) {
      const m = r[key];
      $(`#${key}-digit`).textContent = m.digit;
      $(`#${key}-meta`).innerHTML = `${m.n_features} features · ${m.latency_ms.toFixed(1)} ms<br>vote margin ${m.margin.toFixed(2)}`;
      renderBars($(`#${key}-bars`), m.scores, m.digit, alt);
    }
    $("#pca-tag").textContent = `${r.pca.n_features} PCA features`;
    const truth = state.trueLabel !== null ? ` · true label ${state.trueLabel}` : "";
    const disagree = r.raw.digit !== r.pca.digit;
    $("#agree").className = "agree " + (disagree ? "no" : "yes");
    $("#agree").textContent = disagree ? `The two classifiers disagree (${r.raw.digit} vs ${r.pca.digit})` : `Both classifiers say ${r.pca.digit}`;
    $("#meta").textContent = `Bars: one-vs-one votes per digit (max 9). Request #${r.prediction_id ?? "-"} logged · ${r.latency_ms.toFixed(1)} ms in total (both SVMs, detector, map positions)${truth}`;
    state.positions = r.map_positions;
    if (mapState.data) drawMap();
  } catch (e) { showError(e.message); }
}

/* ---------- 2D maps ---------- */
const mapState = { data: null, method: "t-SNE", thumbs: null, hover: -1 };
const mapCanvas = $("#map");
const PAD = 26;

async function loadMaps() {
  if (mapState.data) { drawMap(); return; }
  try {
    const d = await api("/maps");
    mapState.data = d;
    const bin = atob(d.thumbnails_b64);
    mapState.thumbs = Uint8Array.from(bin, (c) => c.charCodeAt(0));
    const order = ["t-SNE", "PCA", "Kernel PCA", "LLE", "Isomap", "MDS"];
    const methods = d.methods.map((m) => m.name).sort((a, b) => order.indexOf(a) - order.indexOf(b));
    if (!methods.includes(mapState.method)) mapState.method = methods[0];
    $("#method-seg").innerHTML = methods.map((m) => `<button data-m="${esc(m)}" class="${m === mapState.method ? "on" : ""}">${esc(m)}</button>`).join("");
    $$("#method-seg button").forEach((b) => b.addEventListener("click", () => {
      mapState.method = b.dataset.m; $$("#method-seg button").forEach((x) => x.classList.toggle("on", x === b)); drawMap();
    }));
    const cols = digitColors();
    $("#map-legend").innerHTML = cols.map((c, i) => `<span><i style="background:${c}"></i>${i}</span>`).join("");
    drawMap();
  } catch (e) { $("#maps-hint").textContent = e.message; }
}

function mapPoint(xy, w, h) { return [PAD + xy[0] * (w - 2 * PAD), PAD + (1 - xy[1]) * (h - 2 * PAD)]; }

function drawMap() {
  const d = mapState.data; if (!d) return;
  const m = d.methods.find((x) => x.name === mapState.method);
  const dpr = window.devicePixelRatio || 1;
  const w = 760, h = 560;
  mapCanvas.width = w * dpr; mapCanvas.height = h * dpr;
  const ctx = mapCanvas.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const css = getComputedStyle(document.documentElement);
  ctx.fillStyle = css.getPropertyValue("--surface").trim() || "#fff"; ctx.fillRect(0, 0, w, h);
  const cols = digitColors();
  m.coords.forEach((xy, i) => {
    const [x, y] = mapPoint(xy, w, h);
    ctx.fillStyle = cols[d.labels[i]]; ctx.globalAlpha = 0.75;
    ctx.beginPath(); ctx.arc(x, y, 2.6, 0, 2 * Math.PI); ctx.fill();
  });
  ctx.globalAlpha = 1;
  // digit written at each class's median position (identity is never colour alone);
  // labels closer than 30 px are pushed apart and joined to their median by a thin line
  const ink = css.getPropertyValue("--ink").trim() || "#000", surf = css.getPropertyValue("--surface").trim() || "#fff";
  const meds = [];
  for (let digit = 0; digit < 10; digit++) {
    const pts = m.coords.filter((_, i) => d.labels[i] === digit);
    if (!pts.length) continue;
    const med = (k) => { const v = pts.map((p) => p[k]).sort((a, b) => a - b); return v[Math.floor(v.length / 2)]; };
    const [x, y] = mapPoint([med(0), med(1)], w, h);
    meds.push({ digit, x, y, lx: x, ly: y });
  }
  for (let it = 0; it < 300; it++) {
    let moved = false;
    for (let i = 0; i < meds.length; i++) for (let j = i + 1; j < meds.length; j++) {
      const a = meds[i], b = meds[j];
      let dx = b.lx - a.lx, dy = b.ly - a.ly, dist = Math.hypot(dx, dy);
      if (dist < 30) {
        if (dist < 1e-6) { dx = 1; dy = 0; dist = 1; }
        const push = (30 - dist) / 2 + 0.1;
        a.lx -= dx / dist * push; a.ly -= dy / dist * push; b.lx += dx / dist * push; b.ly += dy / dist * push;
        moved = true;
      }
    }
    if (!moved) break;
  }
  ctx.font = "700 18px Inter, system-ui, sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
  for (const p of meds) {
    if (Math.hypot(p.lx - p.x, p.ly - p.y) > 15) {
      ctx.strokeStyle = ink; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(p.lx, p.ly); ctx.stroke();
      ctx.fillStyle = ink; ctx.beginPath(); ctx.arc(p.x, p.y, 2.5, 0, 2 * Math.PI); ctx.fill();
    }
  }
  for (const p of meds) {
    if (Math.hypot(p.lx - p.x, p.ly - p.y) <= 15) { p.lx = p.x; p.ly = p.y; }
    ctx.lineWidth = 4; ctx.strokeStyle = surf; ctx.strokeText(p.digit, p.lx, p.ly);
    ctx.fillStyle = ink; ctx.fillText(p.digit, p.lx, p.ly);
  }
  if (mapState.hover >= 0) {  // hovered dot: enlarged, in its own colour, with a white rim
    const [x, y] = mapPoint(m.coords[mapState.hover], w, h);
    ctx.fillStyle = cols[d.labels[mapState.hover]]; ctx.strokeStyle = surf; ctx.lineWidth = 2.5;
    ctx.beginPath(); ctx.arc(x, y, 6, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
  }
  // the latest input, if this method can place new points
  const pos = state.positions && state.positions[m.name];
  if (pos) {
    const [x, y] = mapPoint(pos.map((v) => Math.max(-0.02, Math.min(1.02, v))), w, h);
    ctx.lineWidth = 3; ctx.strokeStyle = css.getPropertyValue("--surface").trim();
    ctx.beginPath(); ctx.arc(x, y, 11, 0, 2 * Math.PI); ctx.stroke();
    ctx.lineWidth = 2.5; ctx.strokeStyle = css.getPropertyValue("--ink").trim();
    ctx.beginPath(); ctx.arc(x, y, 11, 0, 2 * Math.PI); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x - 16, y); ctx.lineTo(x - 7, y); ctx.moveTo(x + 7, y); ctx.lineTo(x + 16, y);
    ctx.moveTo(x, y - 16); ctx.lineTo(x, y - 7); ctx.moveTo(x, y + 7); ctx.lineTo(x, y + 16); ctx.stroke();
    ctx.font = "600 13px Inter, system-ui, sans-serif"; ctx.textAlign = "left"; ctx.textBaseline = "middle";
    ctx.lineWidth = 4; ctx.strokeStyle = surf; ctx.strokeText("your input", x + 20, y);
    ctx.fillStyle = ink; ctx.fillText("your input", x + 20, y);
  }
  $("#m-knn").textContent = m.knn_accuracy_2d != null ? `${(100 * m.knn_accuracy_2d).toFixed(1)}%` : "-";
  $("#m-time").textContent = m.seconds != null ? `${m.seconds.toFixed(1)} s` : "-";
  $("#m-n").textContent = `${d.n_points.toLocaleString()} digits shown`;
  if (!state.pixels) { $("#m-you").textContent = "-"; $("#m-you-sub").textContent = "pick an input on the first tab"; }
  else if (!m.projects_new_points) {
    $("#m-you").textContent = "n/a";
    $("#m-you-sub").textContent = m.name === "Isomap"
      ? "Isomap could place it, but only by storing its geodesic distance matrix; it is not deployed"
      : `${m.name} has no transform(): it can only map the points it was fitted on`;
  } else { $("#m-you").textContent = "placed \u2295"; $("#m-you-sub").textContent = `projected with ${m.name}.transform()`; }
}

mapCanvas.addEventListener("mousemove", (e) => {
  const d = mapState.data; if (!d) return;
  const m = d.methods.find((x) => x.name === mapState.method);
  const r = mapCanvas.getBoundingClientRect();
  const mx = (e.clientX - r.left) * 760 / r.width, my = (e.clientY - r.top) * 560 / r.height;
  let best = -1, bd = 64;
  m.coords.forEach((xy, i) => { const [x, y] = mapPoint(xy, 760, 560); const dd = (x - mx) ** 2 + (y - my) ** 2; if (dd < bd) { bd = dd; best = i; } });
  const tip = $("#map-tip");
  if (best !== mapState.hover) { mapState.hover = best; drawMap(); }
  if (best < 0) { tip.hidden = true; return; }
  drawDigit($("#tip-img"), Array.from(mapState.thumbs.subarray(best * 784, best * 784 + 784)));
  $("#tip-text").textContent = `digit ${d.labels[best]}`;
  tip.hidden = false;
  const wrap = mapCanvas.parentElement.getBoundingClientRect();
  tip.style.left = `${Math.min(e.clientX - wrap.left + 14, r.width - 120)}px`;
  tip.style.top = `${e.clientY - wrap.top + 14}px`;
});
mapCanvas.addEventListener("mouseleave", () => { $("#map-tip").hidden = true; mapState.hover = -1; drawMap(); });

/* ---------- monitoring ---------- */
async function loadMonitoring() {
  try {
    const m = await api("/monitoring/summary");
    $("#mon-window").textContent = `last ${m.window_hours >= 24 ? `${Math.round(m.window_hours / 24)} days` : `${m.window_hours} h`}`;
    $("#k-n").textContent = m.n_predictions.toLocaleString();
    $("#k-unusual").textContent = m.unusual_rate_pct == null ? "-" : `${m.unusual_rate_pct.toFixed(1)}%`;
    const high = m.unusual_rate_pct != null && m.unusual_rate_pct > 3 * m.expected_unusual_rate_pct;
    $("#k-unusual-sub").textContent = `expected about ${m.expected_unusual_rate_pct}% on normal digits${high ? " - inputs look different" : ""}`;
    $("#k-unusual-sub").className = "kpi-sub" + (high ? " alert" : "");
    $("#k-agree").textContent = m.models_agree_pct == null ? "-" : `${m.models_agree_pct.toFixed(1)}%`;
    $("#k-lat").textContent = m.mean_latency_raw_ms == null ? "-" : `${m.mean_latency_raw_ms.toFixed(2)} / ${m.mean_latency_pca_ms.toFixed(2)} ms`;
    const max = Math.max(1, ...Object.values(m.class_counts));
    $("#class-chart").innerHTML = Object.entries(m.class_counts).map(([d, c]) =>
      `<div class="col" title="digit ${d}: ${c}"><em>${c}</em><i style="height:${Math.max(1, 100 * c / max)}%"></i><span>${d}</span></div>`).join("");
    $("#source-table tbody").innerHTML = Object.entries(m.by_source).map(([s, v]) =>
      `<tr><td>${esc(s)}</td><td class="num">${v.n}</td><td class="num">${v.unusual}</td><td class="num">${(100 * v.unusual / v.n).toFixed(0)}%</td></tr>`).join("")
      || `<tr><td colspan="4" class="muted">no requests yet</td></tr>`;
    $("#recent").innerHTML = m.recent.map((r) => `<div class="rec"><canvas width="28" height="28" data-id="${r.id}"></canvas><div>
      <b>${r.digit_pca}</b>${r.digit_raw !== r.digit_pca ? ` <span class="muted">(raw: ${r.digit_raw})</span>` : ""} ${r.is_unusual ? '<span class="flag">unusual</span>' : ""}
      <small>${esc(r.source)}</small><small>recon. error ${Math.round(r.reconstruction_error).toLocaleString()}</small></div></div>`).join("") || '<span class="muted">no requests yet</span>';
    m.recent.forEach((r) => drawDigit($(`#recent canvas[data-id="${r.id}"]`), r.pixels));
  } catch (e) { $("#mon-window").textContent = e.message; }
}
$("#refresh-mon").addEventListener("click", loadMonitoring);

/* ---------- start ---------- */
resetPad();
refreshHealth().then(loadCompressInfo);
setInterval(refreshHealth, 15000);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { if (mapState.data) loadMaps(); });
