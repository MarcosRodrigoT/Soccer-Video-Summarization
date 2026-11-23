// Vista 2 — «Planos y juego».
// En una ventana de un partido de LaLiga se anotan TODAS las transiciones entre
// planos (cortes y graduales) y después se etiqueta cada plano como Juego / No juego.
// Documento de sesión (servidor): {session_id, match_id, start_frame, end_frame (exclusivo),
//   revision, complete, transitions: [{id, type: "cut", frame} | {id, type: "gradual", start, end}],
//   shot_labels: {"<primer fotograma del plano>": "play" | "no_play"}, reviewed: [[a, b], ...]}

const VIEW = "planos";
const CONTEXT_S = 2;            // segundos visibles fuera de la ventana (contexto, en gris)
const THUMB_W = 192;            // ancho de miniaturas (tira y lista de planos)
const SAVE_DEBOUNCE = 600;
const COV_DEBOUNCE = 5000;      // el progreso de revisión se guarda con menos prisa
const RETRY_MS = 4000;
const MAX_UNDO = 200;
const COV_STEP_S = 1;           // pasos de hasta 1 s cuentan como recorrido continuo
const CURVE_CHUNK_S = 30 * 60;  // el servidor admite hasta 30 min por petición
const LABEL_TEXT = { play: "Juego", no_play: "No juego" };
const LS_LAST = "sbd.lastSession";
const LS_PREFS = "sbd.prefs";
const LS_MATCH = "sbd.lastMatch";
const lsBackup = (sid) => `sbd.backup.${sid}`;
const lsPos = (sid) => `sbd.pos.${sid}`;

let C = null;        // ctx de common.js
let root = null;     // contenedor de la vista
let matches = [];
let matchById = {};
let sessions = [];
let P = null;        // estado del selector de sesión
let W = null;        // estado de la sesión abierta
let prefs = { showCurve: true, bigStrip: false, autoAdvance: false };

// ------------------------------------------------------------------ utilidades
function lsGet(k) { try { return localStorage.getItem(k); } catch { return null; } }
function lsSet(k, v) { try { localStorage.setItem(k, v); } catch { /* sin almacenamiento */ } }
function lsDel(k) { try { localStorage.removeItem(k); } catch { /* sin almacenamiento */ } }
function lsGetJSON(k) { try { const v = lsGet(k); return v ? JSON.parse(v) : null; } catch { return null; } }
function savePrefs() { lsSet(LS_PREFS, JSON.stringify(prefs)); }

const esc = (s) => C.escapeHtml(s);
const clone = (x) => JSON.parse(JSON.stringify(x));
const plural = (n, one, many) => `${n} ${n === 1 ? one : many}`;
function secs(frames, fps) { return (frames / fps).toFixed(1).replace(".", ",") + " s"; }
function trFirst(t) { return t.type === "cut" ? t.frame : t.start; }
function trLast(t) { return t.type === "cut" ? t.frame : t.end; }
function trNext(t) { return t.type === "cut" ? t.frame : t.end + 1; } // primer fotograma del plano siguiente
function sortTrs(trs) { return [...trs].sort((a, b) => trFirst(a) - trFirst(b)); }
function fmtDate(iso) {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleString("es-ES", { day: "2-digit", month: "2-digit", year: "numeric", hour: "2-digit", minute: "2-digit" });
  } catch { return iso; }
}
function blurActive() { const a = document.activeElement; if (a && a !== document.body && a.blur) a.blur(); }

function shotsOf(trs, S, E) {
  const shots = [];
  let cur = S;
  for (const t of sortTrs(trs)) {
    const first = trFirst(t);
    if (first > cur) shots.push({ start: cur, end: first - 1 });
    cur = trNext(t);
  }
  if (cur < E) shots.push({ start: cur, end: E - 1 });
  return shots;
}

// ------------------------------------------------------------------ montaje
export async function mount(container, ctx) {
  C = ctx;
  root = container;
  prefs = { ...prefs, ...(lsGetJSON(LS_PREFS) || {}) };
  root.innerHTML = `<div class="empty">Cargando partidos y sesiones…</div>`;
  C.helpPanel.set(helpHtml());
  try {
    [matches, sessions] = await Promise.all([C.api.get("/api/matches?view=sbd"), C.api.get("/api/sbd/sessions")]);
  } catch (e) {
    root.innerHTML = `<div class="error-box">No se pudo cargar la lista de partidos: ${esc(e.message)}
      <div style="margin-top:10px"><button class="btn" data-retry>Reintentar</button></div></div>`;
    root.querySelector("[data-retry]").onclick = () => mount(container, ctx);
    return;
  }
  if (root !== container) return;
  matchById = Object.fromEntries(matches.map((m) => [m.id, m]));
  document.addEventListener("visibilitychange", onVisibility);
  const last = lsGet(LS_LAST);
  if (last && sessions.some((s) => s.session_id === last)) await openSession(last);
  else showPicker();
}

export async function unmount() {
  document.removeEventListener("visibilitychange", onVisibility);
  await closeWork();
  if (P) P.destroy();
  P = null;
  C.keys.unbind(VIEW);
  root = null;
}

function onVisibility() {
  if (document.visibilityState === "hidden" && W && (W.seq !== W.savedSeq || W.covDirty)) doSave(W);
}

// ================================================================== SELECTOR
function showPicker() {
  C.keys.unbind(VIEW);
  C.shortcuts.set([["?", "ayuda"], ["Intro", "abrir / crear la sesión (en el campo de inicio)"]]);
  C.saveStatus.set("saved");
  const lastMatch = lsGet(LS_MATCH);
  const m0 = matchById[lastMatch] || matches.find((m) => !m.error) || matches[0];
  if (!m0) { root.innerHTML = `<div class="error-box">No hay partidos disponibles.</div>`; return; }
  const fps = m0.fps || 25;
  P = { matchId: m0.id, start: 20 * 60 * fps, durMin: 5, wp: null, previewTimer: null, destroy() { if (this.wp) this.wp.destroy(); clearTimeout(this.previewTimer); } };
  root.innerHTML = `
  <div class="sbd-picker">
    <section class="card sbd-new">
      <div class="card-head"><span class="sbd-step">1</span><h3>Elige partido y ventana de vídeo</h3></div>
      <div class="card-body stack sbd-new-body">
        <p class="muted sbd-intro">En cada sesión anotarás <b>todas</b> las transiciones entre planos de un fragmento del
          partido y luego dirás si cada plano es <b>Juego</b> o <b>No juego</b>. Por defecto, 5 minutos a partir del minuto 20.</p>
        <div class="sbd-form">
          <label class="field sbd-f-match">Partido
            <select class="input" data-f="match">${matches.map((m) =>
              `<option value="${esc(m.id)}" ${m.error ? "disabled" : ""}>${esc(m.label)}${m.error ? " — no disponible" : ""}</option>`).join("")}</select>
          </label>
          <label class="field">Inicio (mm:ss)
            <input class="input mono" data-f="start" inputmode="numeric" autocomplete="off" placeholder="20:00">
          </label>
          <label class="field">Duración
            <select class="input mono" data-f="dur">${Array.from({ length: 60 }, (_, i) => `<option value="${i + 1}">${i + 1} min</option>`).join("")}</select>
          </label>
        </div>
        <div class="row">
          <button class="btn" data-act="preset20">⏱ Desde el minuto 20</button>
          <button class="btn" data-act="random">🎲 Ventana aleatoria</button>
          <span class="muted small">o haz clic / arrastra la ventana azul sobre el partido:</span>
        </div>
        <div class="sbd-ov">
          <canvas class="sbd-ov-canvas"></canvas>
          <div class="sbd-ov-legend small"><span class="mono">00:00</span>
            <span class="sbd-ov-key"><span><i class="k-win"></i>Ventana elegida</span><span><i class="k-other"></i>Otras sesiones de este partido</span><span><i class="k-done"></i>Terminadas</span></span>
            <span class="mono" data-ov-end></span></div>
        </div>
        <div class="sbd-sum">
          <div class="sbd-sum-thumb"><img alt="Primer fotograma de la ventana" data-preview><span>Primer fotograma</span></div>
          <div class="stack sbd-sum-text">
            <div class="sbd-sum-main" data-sum></div>
            <div class="small" data-sum-note></div>
            <div><button class="btn btn-primary btn-lg" data-act="create">Crear sesión</button></div>
          </div>
        </div>
      </div>
    </section>
    <section class="card sbd-saved">
      <div class="card-head"><span class="sbd-step">2</span><h3>…o continúa una sesión guardada</h3><span class="badge" data-nsess></span></div>
      <div class="sbd-sess-list" data-sessions></div>
    </section>
  </div>`;
  const q = (s) => root.querySelector(s);
  q('[data-f="match"]').value = P.matchId;
  q('[data-f="dur"]').value = P.durMin;
  P.wp = new WindowPicker(q(".sbd-ov-canvas"), (start) => { P.start = start; updatePicker({ fromDrag: true }); });
  q('[data-f="match"]').addEventListener("change", (e) => {
    P.matchId = e.target.value; lsSet(LS_MATCH, P.matchId); clampPicker(); updatePicker();
  });
  const startIn = q('[data-f="start"]');
  const applyStart = () => {
    const m = matchById[P.matchId], fps = m.fps || 25;
    const n = C.fmt.parseTime(startIn.value, fps);
    if (n === null) { C.toast("No entiendo ese inicio. Escribe minutos y segundos, por ejemplo 20:00 o 47:30.", "warn"); updatePicker(); return false; }
    P.start = n - (n % fps);
    clampPicker(true);
    updatePicker();
    return true;
  };
  startIn.addEventListener("change", applyStart);
  startIn.addEventListener("focus", () => setTimeout(() => startIn.select(), 0));
  startIn.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); if (applyStart()) createSession(); } });
  q('[data-f="dur"]').addEventListener("change", (e) => {
    let v = parseInt(e.target.value, 10);
    if (!Number.isFinite(v)) v = 5;
    v = Math.max(1, Math.min(60, v));
    P.durMin = v; clampPicker(true); updatePicker();
  });
  root.querySelector(".sbd-new").addEventListener("click", (e) => {
    const b = e.target.closest("[data-act]");
    if (!b) return;
    blurActive();
    const m = matchById[P.matchId], fps = m.fps || 25;
    if (b.dataset.act === "preset20") { P.start = 20 * 60 * fps; clampPicker(true); updatePicker(); }
    if (b.dataset.act === "random") randomWindow();
    if (b.dataset.act === "create") createSession();
  });
  root.querySelector("[data-sessions]").addEventListener("click", onSessionListClick);
  clampPicker();
  updatePicker();
  renderSessionList();
  refreshSessions();
}

function pickerLen() { const m = matchById[P.matchId]; return P.durMin * 60 * (m.fps || 25); }

function clampPicker(warn = false) {
  const m = matchById[P.matchId];
  const len = pickerLen();
  const maxStart = Math.max(0, m.total_frames - len);
  const before = P.start;
  P.start = Math.max(0, Math.min(maxStart, P.start));
  if (len > m.total_frames) { P.durMin = Math.max(1, Math.floor(m.total_frames / ((m.fps || 25) * 60))); }
  if (warn && before !== P.start) C.toast("La ventana se salía del partido; la he ajustado.", "warn");
}

function randomWindow() {
  const m = matchById[P.matchId], fps = m.fps || 25;
  const len = pickerLen();
  let lo = 5 * 60 * fps, hi = m.total_frames - 5 * 60 * fps - len;
  if (hi < lo) { lo = 0; hi = m.total_frames - len; }
  const nSec = Math.max(0, Math.floor((hi - lo) / fps));
  P.start = lo + Math.floor(Math.random() * (nSec + 1)) * fps;
  clampPicker();
  updatePicker();
}

function pickerSid() { return `${P.matchId}_${P.start}_${P.start + pickerLen()}`; }

function updatePicker({ fromDrag = false } = {}) {
  if (!P || !root) return;
  const q = (s) => root.querySelector(s);
  const m = matchById[P.matchId], fps = m.fps || 25;
  const len = pickerLen(), end = P.start + len;
  const startIn = q('[data-f="start"]');
  if (document.activeElement !== startIn || fromDrag) startIn.value = C.fmt.clock(P.start, fps);
  q('[data-f="dur"]').value = P.durMin;
  const others = sessions.filter((s) => s.match_id === P.matchId)
    .map((s) => ({ start: s.start_frame, end: s.end_frame, complete: s.complete }));
  P.wp.set({ total: m.total_frames, fps, start: P.start, len, others });
  q("[data-ov-end]").textContent = C.fmt.clock(m.total_frames, fps);
  q("[data-sum]").innerHTML = `Ventana <b class="mono">${C.fmt.clock(P.start, fps)} → ${C.fmt.clock(end, fps)}</b>
    <span class="muted">· ${P.durMin} min · ${C.fmt.frame(len)} fotogramas</span>`;
  const sid = pickerSid();
  const exists = sessions.find((s) => s.session_id === sid);
  const overl = sessions.filter((s) => s.match_id === P.matchId && s.session_id !== sid && s.start_frame < end && s.end_frame > P.start);
  const note = q("[data-sum-note]");
  if (exists) note.innerHTML = `<span class="badge info">Ya existe</span> Esta ventana ya tiene una sesión guardada: se abrirá para continuarla.`;
  else if (overl.length) note.innerHTML = `<span class="badge warn">Aviso</span> Se solapa con ${plural(overl.length, "sesión guardada", "sesiones guardadas")} del mismo partido.`;
  else note.innerHTML = `<span class="muted">Una vez creada la sesión, su ventana queda fija.</span>`;
  q('[data-act="create"]').textContent = exists ? "Continuar esta sesión →" : "Crear sesión →";
  clearTimeout(P.previewTimer);
  P.previewTimer = setTimeout(() => {
    const img = root && root.querySelector("[data-preview]");
    if (img) img.src = C.frameUrl(P.matchId, P.start, 320);
  }, fromDrag ? 250 : 0);
}

async function refreshSessions() {
  try {
    sessions = await C.api.get("/api/sbd/sessions");
  } catch (e) {
    C.toast("No se pudo cargar la lista de sesiones: " + e.message, "error");
    return;
  }
  if (P && root) { renderSessionList(); updatePicker(); }
}

function renderSessionList() {
  const box = root.querySelector("[data-sessions]");
  root.querySelector("[data-nsess]").textContent = sessions.length;
  if (!sessions.length) {
    box.innerHTML = `<div class="empty"><div class="sbd-empty-ico">🗂️</div>Todavía no hay sesiones guardadas.<br>
      <span class="small">Crea la primera con el formulario de la izquierda.</span></div>`;
    return;
  }
  box.innerHTML = sessions.map((s) => {
    const m = matchById[s.match_id];
    const fps = (m && m.fps) || 25;
    const pct = s.n_shots ? Math.round((100 * s.n_labelled) / s.n_shots) : 0;
    const state = s.complete ? `<span class="badge ok">✓ Terminada</span>`
      : (s.n_transitions || s.n_labelled) ? `<span class="badge warn">En curso</span>` : `<span class="badge">Sin empezar</span>`;
    return `<div class="sbd-sess ${s.complete ? "done" : ""}" data-sid="${esc(s.session_id)}">
      <div class="sbd-sess-top">
        <div class="sbd-sess-title">${esc(m ? m.label : s.match_id)}</div>${state}
      </div>
      <div class="sbd-sess-meta">
        <span class="mono">${C.fmt.clock(s.start_frame, fps)} → ${C.fmt.clock(s.end_frame, fps)}</span>
        <span class="muted">· ${Math.round((s.end_frame - s.start_frame) / fps / 60 * 10) / 10} min</span>
        <span class="muted">· ${plural(s.n_transitions, "transición", "transiciones")}</span>
      </div>
      <div class="sbd-sess-prog">
        <div class="sbd-bar-track"><i style="width:${pct}%"></i></div>
        <span class="small muted">Planos etiquetados: ${s.n_labelled} de ${s.n_shots}</span>
      </div>
      <div class="sbd-sess-foot">
        <span class="small muted">Actualizada ${esc(fmtDate(s.updated))}</span>
        <span class="grow"></span>
        <button class="btn btn-sm btn-danger" data-del title="Eliminar la sesión (se guarda una copia en la papelera)">🗑 Eliminar</button>
        <button class="btn btn-sm btn-primary" data-open>${s.complete ? "Revisar" : "Continuar"} →</button>
      </div>
    </div>`;
  }).join("");
}

async function onSessionListClick(e) {
  const item = e.target.closest("[data-sid]");
  if (!item) return;
  const sid = item.dataset.sid;
  if (e.target.closest("[data-del]")) {
    blurActive();
    const s = sessions.find((x) => x.session_id === sid);
    const m = s && matchById[s.match_id];
    const fps = (m && m.fps) || 25;
    const ok = await C.confirmDialog("¿Eliminar esta sesión?",
      `${m ? m.label : sid}, ventana ${s ? C.fmt.clock(s.start_frame, fps) + " → " + C.fmt.clock(s.end_frame, fps) : ""}. ` +
      `Desaparecerá de la lista (${s ? plural(s.n_transitions, "transición", "transiciones") + " y " + plural(s.n_labelled, "etiqueta", "etiquetas") : ""}). ` +
      `Se guarda una copia en la papelera del servidor por si hiciera falta recuperarla.`,
      { ok: "Eliminar", cancel: "Cancelar", danger: true });
    if (!ok) return;
    try {
      await C.api.del(`/api/sbd/${encodeURIComponent(sid)}`);
      lsDel(lsBackup(sid)); lsDel(lsPos(sid));
      if (lsGet(LS_LAST) === sid) lsDel(LS_LAST);
      C.toast("Sesión eliminada (hay copia en la papelera).", "ok");
    } catch (err) {
      C.toast("No se pudo eliminar: " + err.message, "error");
    }
    refreshSessions();
    return;
  }
  if (e.target.closest("[data-open]") || !e.target.closest("button")) openSession(sid);
}

async function createSession() {
  const m = matchById[P.matchId];
  if (!m || m.error) { C.toast("Este partido no está disponible.", "error"); return; }
  const body = { match_id: P.matchId, start_frame: P.start, end_frame: P.start + pickerLen() };
  const btn = root.querySelector('[data-act="create"]');
  btn.disabled = true;
  try {
    const doc = await C.api.post("/api/sbd/sessions", body);
    lsSet(LS_MATCH, P.matchId);
    await openSession(doc.session_id, doc);
  } catch (e) {
    C.toast("No se pudo crear la sesión: " + e.message, "error");
    if (btn.isConnected) btn.disabled = false;
  }
}

// Mini línea de tiempo del partido entero para elegir la ventana.
class WindowPicker {
  constructor(canvas, onChange) {
    this.c = canvas;
    this.onChange = onChange;
    this.s = { total: 1, fps: 25, start: 0, len: 1, others: [] };
    this.hover = null;
    this.drag = null;
    this._ro = new ResizeObserver(() => this.draw());
    this._ro.observe(canvas);
    canvas.addEventListener("pointerdown", (e) => {
      const f = this.frameAt(e);
      const { start, len } = this.s;
      this.drag = { off: f >= start && f < start + len ? f - start : len / 2 };
      canvas.setPointerCapture(e.pointerId);
      this.move(f);
    });
    canvas.addEventListener("pointermove", (e) => {
      const f = this.frameAt(e);
      this.hover = f;
      if (this.drag) this.move(f); else this.draw();
    });
    canvas.addEventListener("pointerup", () => { this.drag = null; this.draw(); });
    canvas.addEventListener("pointerleave", () => { this.hover = null; this.draw(); });
  }
  frameAt(e) {
    const r = this.c.getBoundingClientRect();
    return ((e.clientX - r.left) / r.width) * this.s.total;
  }
  move(f) {
    const { total, fps, len } = this.s;
    let st = Math.round((f - this.drag.off) / fps) * fps;
    st = Math.max(0, Math.min(total - len, st));
    if (st !== this.s.start) { this.s.start = st; this.onChange(st); }
    this.draw();
  }
  set(s) { this.s = { ...this.s, ...s }; this.draw(); }
  draw() {
    const c = this.c, dpr = window.devicePixelRatio || 1;
    const Wd = c.clientWidth, H = c.clientHeight;
    if (!Wd || !H) return;
    if (c.width !== Math.round(Wd * dpr) || c.height !== Math.round(H * dpr)) { c.width = Math.round(Wd * dpr); c.height = Math.round(H * dpr); }
    const g = c.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, Wd, H);
    const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
    const { total, fps, start, len, others } = this.s;
    const x = (f) => (f / total) * Wd;
    const top = 8, bot = H - 16;
    g.fillStyle = css("--surface-2"); g.fillRect(0, top, Wd, bot - top);
    g.strokeStyle = css("--border-strong"); g.lineWidth = 1; g.strokeRect(0.5, top + 0.5, Wd - 1, bot - top - 1);
    // marcas cada 15 min
    g.font = "11px " + css("--mono"); g.fillStyle = css("--text-2");
    for (let mnt = 15; mnt * 60 * fps < total; mnt += 15) {
      const xx = Math.round(x(mnt * 60 * fps)) + 0.5;
      g.strokeStyle = css("--border"); g.beginPath(); g.moveTo(xx, top); g.lineTo(xx, bot); g.stroke();
      g.fillText(`${mnt}'`, xx - 8, H - 3);
    }
    // otras sesiones del partido
    for (const o of others) {
      g.fillStyle = o.complete ? "rgba(22,163,74,.35)" : "rgba(100,116,139,.35)";
      g.fillRect(x(o.start), bot - 8, Math.max(2, x(o.end) - x(o.start)), 8);
    }
    // ventana elegida
    const x0 = x(start), x1 = Math.max(x0 + 4, x(start + len));
    g.fillStyle = "rgba(37,99,235,.22)"; g.fillRect(x0, top, x1 - x0, bot - top);
    g.strokeStyle = css("--primary"); g.lineWidth = 2; g.strokeRect(x0 + 1, top + 1, x1 - x0 - 2, bot - top - 2);
    g.fillStyle = css("--primary"); g.fillRect(x0, top, 3, bot - top); g.fillRect(x1 - 3, top, 3, bot - top);
    const wl = `${C.fmt.clock(start, fps)}–${C.fmt.clock(start + len, fps)}`;
    g.font = "700 11px " + css("--font");
    const wlw = g.measureText(wl).width;
    const lx = Math.min(Wd - wlw - 4, Math.max(4, (x0 + x1) / 2 - wlw / 2));
    if (this.hover === null || this.drag) {
      g.fillStyle = "rgba(255,255,255,.9)"; g.fillRect(lx - 3, top + 3, wlw + 6, 15);
      g.fillStyle = css("--primary-2"); g.fillText(wl, lx, top + 14);
    }
    // hover
    if (this.hover !== null && !this.drag) {
      const xx = Math.round(x(this.hover)) + 0.5;
      g.strokeStyle = css("--text-2"); g.setLineDash([3, 3]); g.beginPath(); g.moveTo(xx, top); g.lineTo(xx, bot); g.stroke(); g.setLineDash([]);
      const label = C.fmt.clock(Math.max(0, this.hover), fps);
      g.fillStyle = css("--text"); g.font = "600 11px " + css("--mono");
      const tw = g.measureText(label).width;
      g.fillText(label, Math.min(Wd - tw - 2, Math.max(2, xx + 4)), top + 13);
    }
  }
  destroy() { this._ro.disconnect(); }
}

// ================================================================== SESIÓN
async function openSession(sid, doc = null) {
  await closeWork();
  if (!root) return;
  if (!doc) {
    root.innerHTML = `<div class="empty">Abriendo la sesión…</div>`;
    try {
      doc = await C.api.get(`/api/sbd/${encodeURIComponent(sid)}`);
    } catch (e) {
      C.toast("No se pudo abrir la sesión: " + e.message, "error");
      lsDel(LS_LAST);
      if (root) showPicker();
      return;
    }
  }
  if (!root) return;
  const m = matchById[doc.match_id];
  if (!m || m.error) { C.toast("El partido de esta sesión no está disponible.", "error"); lsDel(LS_LAST); showPicker(); return; }
  if (P) { P.destroy(); P = null; }
  const fps = m.fps || 25;
  const st = W = {
    sid: doc.session_id, doc, m, fps, S: doc.start_frame, E: doc.end_frame,
    trs: clone(doc.transitions || []), labels: { ...(doc.shot_labels || {}) }, complete: !!doc.complete,
    cov: normCov(doc.reviewed, doc.start_frame, doc.end_frame), revision: doc.revision,
    pendingA: Number.isInteger(doc.open_gradual_start) && doc.open_gradual_start >= doc.start_frame && doc.open_gradual_start < doc.end_frame ? doc.open_gradual_start : null,
    editId: null, undo: [], seq: 0, savedSeq: 0, covDirty: false,
    saving: null, again: false, blocked: false, timer: null, covTimer: null, retryTimer: null, posTimer: null,
    tab: "trs", lastPos: null, curve: null, curveState: "loading", shots: [], curShot: -1, curTr: null, closed: false,
  };
  lsSet(LS_LAST, st.sid);
  C.saveStatus.set("saved");
  root.innerHTML = workHtml(st);
  st.el = (s) => root.querySelector(s);
  buildFrameView(st);
  st.tl = new C.Timeline(st.el(".sbd-tl"), { total: m.total_frames, fps, start: st.S, end: st.E, height: 76 });
  st.tl.on("seek", (n) => { st.fv.pause(); st.fv.goTo(n); });
  bindWorkEvents(st);
  bindKeys(st);
  C.shortcuts.set([
    ["← →", "fotograma"], ["Mayús+←/→", "segundo"], ["Espacio", "reproducir"],
    ["C", "corte"], ["A B", "gradual"], ["J K", "transiciones"], ["Supr", "eliminar"],
    ["1 2", "Juego/No juego"], ["Tab", "sin etiquetar"], ["Ctrl+Z", "deshacer"], ["?", "ayuda"],
  ]);
  C.helpPanel.set(helpHtml());
  recompute(st);
  renderAll(st);
  const pos = parseInt(lsGet(lsPos(st.sid)) || "", 10);
  st.fv.goTo(Number.isFinite(pos) && pos >= st.S && pos < st.E ? pos : st.S);
  loadCurve(st);
  await offerBackup(st);
}

function normCov(cov, S, E) {
  if (!Array.isArray(cov)) return [];
  const out = [];
  for (const iv of cov) {
    if (!Array.isArray(iv) || iv.length !== 2) continue;
    const a = Math.max(S, iv[0] | 0), b = Math.min(E - 1, iv[1] | 0);
    if (a <= b) out.push([a, b]);
  }
  return mergeIntervals(out);
}
function mergeIntervals(list) {
  list.sort((x, y) => x[0] - y[0]);
  const out = [];
  for (const [a, b] of list) {
    const last = out[out.length - 1];
    if (last && a <= last[1] + 1) last[1] = Math.max(last[1], b); else out.push([a, b]);
  }
  return out;
}

async function closeWork() {
  if (!W) return;
  const st = W;
  await flushSave(st);
  st.closed = true;
  clearTimeout(st.timer); clearTimeout(st.covTimer); clearTimeout(st.retryTimer); clearTimeout(st.posTimer);
  if (st.lastPos !== null) lsSet(lsPos(st.sid), String(st.lastPos));
  try { st.fv.destroy(); } catch { /* ya destruido */ }
  try { st.tl.destroy(); } catch { /* ya destruido */ }
  if (W === st) W = null;
  C.keys.unbind(VIEW);
}

function workHtml(st) {
  const m = st.m;
  return `
  <div class="sbd-work">
    <div class="sbd-head card">
      <button class="btn btn-sm" data-act="close" title="Volver a la lista de sesiones">← Sesiones</button>
      <div class="sbd-head-title">
        <div class="sbd-head-match">${esc(m.label)}</div>
        <div class="small muted">Ventana <span class="mono">${C.fmt.clock(st.S, st.fps)} → ${C.fmt.clock(st.E, st.fps)}</span>
          · ${Math.round((st.E - st.S) / st.fps / 60 * 10) / 10} min · fotogramas ${C.fmt.frame(st.S)}–${C.fmt.frame(st.E - 1)}</div>
      </div>
      <div class="sbd-meters">
        <div class="sbd-meter" title="Tramo de la ventana recorrido de forma continua (fotograma a fotograma, en pasos de hasta 1 s o reproduciendo)">
          <div class="sbd-meter-txt" data-m-rev></div><div class="sbd-bar-track rev"><i data-m-rev-bar></i></div>
        </div>
        <div class="sbd-meter">
          <div class="sbd-meter-txt" data-m-lab></div><div class="sbd-bar-track"><i data-m-lab-bar></i></div>
        </div>
      </div>
      <div class="sbd-head-actions">
        <span data-sstate></span>
        <button class="btn btn-ok" data-act="finish"></button>
      </div>
    </div>
    <div class="sbd-main">
      <div class="sbd-left">
        <div class="sbd-fv"></div>
        <div class="sbd-tlbar">
          <label class="sbd-check" title="Ayuda visual: cuánto cambia la imagen entre fotogramas consecutivos"><input type="checkbox" data-pref="showCurve"> Mostrar curva de cambios</label>
          <span class="small muted" data-curve-state></span>
          <span class="grow"></span>
          <span class="sbd-tl-key small muted">
            <span><i class="k-cut"></i>Corte</span><span><i class="k-grad"></i>Gradual</span>
            <span><i class="k-play"></i>Juego</span><span><i class="k-noplay"></i>No juego</span><span><i class="k-none"></i>Sin etiquetar</span>
          </span>
          <span class="sep"></span>
          <span class="small muted">Zoom</span>
          <span class="sbd-seg">
            <button class="btn btn-sm" data-zoom="all" title="Ver toda la ventana">Toda</button><button class="btn btn-sm" data-zoom="750" title="±30 s alrededor del cursor">±30 s</button><button class="btn btn-sm" data-zoom="125" title="±5 s alrededor del cursor">±5 s</button>
          </span>
        </div>
        <div class="sbd-tl"></div>
      </div>
      <aside class="sbd-side">
        <section class="card sbd-tools">
          <div class="sbd-sec-head"><span class="sbd-step">1</span><b>Marcar transiciones</b></div>
          <div class="sbd-tool-grid">
            <button class="btn sbd-big sbd-b-cut" data-act="cut" title="Corte: el fotograma actual es el primero del plano nuevo (C)">
              <span class="sbd-big-ico">✂</span><span class="sbd-big-txt">Corte aquí</span><kbd>C</kbd></button>
            <button class="btn sbd-big sbd-b-grad" data-act="gstart" title="Primer fotograma de la transición gradual (A)">
              <span class="sbd-big-ico">◐</span><span class="sbd-big-txt">Inicio gradual</span><kbd>A</kbd></button>
            <button class="btn sbd-big sbd-b-grad" data-act="gend" title="Último fotograma de la transición gradual (B)">
              <span class="sbd-big-ico">◑</span><span class="sbd-big-txt">Fin gradual</span><kbd>B</kbd></button>
          </div>
          <div class="sbd-msg" data-msg></div>
          <div class="sbd-tool-row sbd-tool-row4">
            <button class="btn btn-sm" data-act="prevTr" title="Ir a la transición anterior (J)">‹ Anterior <kbd>J</kbd></button>
            <button class="btn btn-sm" data-act="nextTr" title="Ir a la transición siguiente (K)">Siguiente › <kbd>K</kbd></button>
            <button class="btn btn-sm btn-danger" data-act="delNear" title="Eliminar la transición más cercana al cursor (Supr)">Eliminar <kbd>Supr</kbd></button>
            <button class="btn btn-sm" data-act="undo" title="Deshacer el último cambio (Ctrl+Z)">↶ Deshacer <kbd>Ctrl+Z</kbd></button>
          </div>
        </section>
        <section class="card sbd-labeler">
          <div class="sbd-sec-head"><span class="sbd-step">2</span><b>Etiquetar el plano actual</b><span class="grow"></span><span class="small muted" data-shotinfo></span></div>
          <div class="sbd-lab-grid">
            <button class="sbd-lab play" data-setcur="play" title="Juego: el partido en directo (1)"><span>▶ Juego</span><kbd>1</kbd></button>
            <button class="sbd-lab noplay" data-setcur="no_play" title="No juego: repeticiones, primeros planos, público… (2)"><span>■ No juego</span><kbd>2</kbd></button>
          </div>
          <p class="sbd-def"><b class="t-play">Juego</b> = el partido en directo (no repeticiones).
            <b class="t-noplay">No juego</b> = repeticiones, primeros planos, público, banquillo, gráficos, pausas.</p>
          <div class="sbd-tool-row">
            <button class="btn btn-sm" data-act="nextUnl" title="Ir al siguiente plano sin etiquetar (Tab o ↓)">Siguiente sin etiquetar <kbd>Tab</kbd></button>
            <label class="sbd-check small" title="Después de pulsar 1 o 2, saltar al siguiente plano sin etiquetar"><input type="checkbox" data-pref="autoAdvance"> Avanzar al etiquetar</label>
          </div>
        </section>
        <section class="card sbd-lists">
          <div class="sbd-tabs" role="tablist">
            <button class="sbd-tab" data-tab="trs" role="tab">Transiciones <span class="badge" data-n-trs></span></button>
            <button class="sbd-tab" data-tab="shots" role="tab">Planos <span class="badge" data-n-shots></span></button>
          </div>
          <div class="sbd-list" data-list></div>
        </section>
      </aside>
    </div>
  </div>`;
}

function buildFrameView(st) {
  const el = st.el(".sbd-fv");
  let cur = null;
  if (st.fv) { cur = st.fv.current; st.fv.destroy(); }
  const ctxF = CONTEXT_S * st.fps;
  st.fv = new C.FrameView(el, {
    matchId: st.m.id, totalFrames: st.m.total_frames, fps: st.fps, width: 960,
    stripRadius: prefs.bigStrip ? 7 : 4, stripWidth: THUMB_W,
  });
  el.classList.toggle("big-strip", !!prefs.bigStrip);
  const ctr = el.querySelector(".fv-controls");
  if (ctr) {
    ctr.insertAdjacentHTML("beforeend", `<span class="grow"></span><label class="sbd-check small" title="Más miniaturas en la tira de abajo (más contexto alrededor del fotograma actual)">
      <input type="checkbox" data-pref="bigStrip" ${prefs.bigStrip ? "checked" : ""}> Tira ampliada</label>`);
  }
  st.fv.setBounds(st.S - ctxF, st.E - 1 + ctxF);
  st.fv.setMarkers((n) => frameBadges(st, n));
  if (st.fv.setStripClass) st.fv.setStripClass((f) => stripClass(st, f));
  st.fv.on("change", (n) => onFrame(st, n));
  if (cur !== null) st.fv.goTo(cur);
}

function frameBadges(st, n) {
  const out = [];
  if (n < st.S || n >= st.E) out.push({ label: "Fuera de la ventana · solo contexto", cls: "end" });
  for (const t of st.trs) {
    if (t.type === "cut" && t.frame === n) out.push({ label: "✂ Corte · primer fotograma del plano nuevo", cls: "cut" });
    else if (t.type === "cut" && t.frame === n + 1) out.push({ label: "Último fotograma antes de un corte", cls: "" });
    else if (t.type === "gradual" && n >= t.start && n <= t.end) out.push({ label: `Gradual · fotograma ${n - t.start + 1} de ${t.end - t.start + 1}`, cls: "gradual" });
  }
  if (st.pendingA !== null) out.push({ label: n === st.pendingA ? "◐ Inicio de la gradual abierta" : "Gradual abierta · falta el fin (B)", cls: "gradual" });
  const i = shotIndexAt(st, n);
  if (i >= 0) {
    const lab = st.labels[String(st.shots[i].start)];
    out.push({ label: `Plano ${i + 1} · ${lab ? LABEL_TEXT[lab] : "sin etiquetar"}`, cls: lab === "play" ? "play" : lab === "no_play" ? "noplay" : "" });
  }
  return out;
}

function stripClass(st, f) {
  const cls = [];
  if (f < st.S || f >= st.E) cls.push("sbd-out");
  for (const t of st.trs) {
    if (t.type === "cut" && t.frame === f) cls.push("sbd-th-cut");
    else if (t.type === "gradual" && f >= t.start && f <= t.end) cls.push("sbd-th-grad");
  }
  if (st.pendingA === f) cls.push("sbd-th-pend");
  return cls.join(" ");
}

function bindWorkEvents(st) {
  const box = root.querySelector(".sbd-work");
  box.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b || !box.contains(b)) return;
    if (b.closest(".fv-controls") || b.closest(".fv-strip")) { setTimeout(blurActive, 0); return; }
    blurActive();
    const act = b.dataset.act;
    if (act === "close") return backToPicker();
    if (act === "finish") return toggleFinish(st);
    if (act === "cut") return markCut(st);
    if (act === "gstart") return markGradualStart(st);
    if (act === "gend") return markGradualEnd(st);
    if (act === "cancelGrad") return cancelGradual(st);
    if (act === "prevTr") return jumpTr(st, -1);
    if (act === "nextTr") return jumpTr(st, 1);
    if (act === "delNear") return deleteNearest(st);
    if (act === "undo") return undo(st);
    if (act === "nextUnl") return nextUnlabelled(st, 1);
    if (b.dataset.setcur) return labelCurrent(st, b.dataset.setcur);
    if (b.dataset.tab) { st.tab = b.dataset.tab; renderLists(st); return; }
    if (b.dataset.zoom) return zoom(st, b.dataset.zoom);
    if (b.dataset.trGo) { const t = st.trs.find((x) => x.id === b.dataset.trGo); if (t) goFrame(st, trFirst(t)); return; }
    if (b.dataset.trGoEnd) { const t = st.trs.find((x) => x.id === b.dataset.trGoEnd); if (t) goFrame(st, t.end); return; }
    if (b.dataset.trEdit) { st.editId = st.editId === b.dataset.trEdit ? null : b.dataset.trEdit; renderLists(st); return; }
    if (b.dataset.trDel) return deleteTr(st, b.dataset.trDel);
    if (b.dataset.mv) return moveTr(st, b.dataset.mvId, b.dataset.mv);
    if (b.dataset.shotGo !== undefined) { const s = st.shots[+b.dataset.shotGo]; if (s) goFrame(st, b.dataset.f ? +b.dataset.f : Math.floor((s.start + s.end) / 2)); return; }
    if (b.dataset.setlab) return setShotLabel(st, +b.dataset.shot, b.dataset.setlab, { toggle: true });
  });
  box.addEventListener("change", (e) => {
    const inp = e.target.closest("[data-pref]");
    if (inp) {
      prefs[inp.dataset.pref] = inp.checked;
      savePrefs();
      if (inp.dataset.pref === "showCurve") applyCurve(st);
      if (inp.dataset.pref === "bigStrip") buildFrameView(st);
    }
    if (e.target.matches("select, input[type=checkbox]")) e.target.blur();
  });
  for (const k of ["showCurve", "bigStrip", "autoAdvance"]) {
    const inp = root.querySelector(`[data-pref="${k}"]`);
    if (inp) inp.checked = !!prefs[k];
  }
}

function bindKeys(st) {
  const fvk = () => st.fv;
  const map = {
    "ArrowLeft": () => fvk().step(-1), "ArrowRight": () => fvk().step(1),
    "Shift+ArrowLeft": () => fvk().step(-st.fps), "Shift+ArrowRight": () => fvk().step(st.fps),
    "Ctrl+ArrowLeft": () => fvk().step(-10 * st.fps), "Ctrl+ArrowRight": () => fvk().step(10 * st.fps),
    "PageUp": () => fvk().step(-60 * st.fps), "PageDown": () => fvk().step(60 * st.fps),
    " ": () => fvk().toggle(),
    "Home": () => goFrame(st, st.S), "End": () => goFrame(st, st.E - 1),
    "g": () => fvk().askGoTo(),
    "c": () => markCut(st), "a": () => markGradualStart(st), "b": () => markGradualEnd(st),
    "Escape": () => { if (st.pendingA !== null) cancelGradual(st); else if (st.editId) { st.editId = null; renderLists(st); } },
    "j": () => jumpTr(st, -1), "k": () => jumpTr(st, 1),
    "Delete": () => deleteNearest(st), "Backspace": () => deleteNearest(st),
    "1": () => labelCurrent(st, "play"), "2": () => labelCurrent(st, "no_play"),
    "Tab": () => nextUnlabelled(st, 1), "Shift+Tab": () => nextUnlabelled(st, -1),
    "ArrowDown": () => nextUnlabelled(st, 1), "ArrowUp": () => nextUnlabelled(st, -1),
    "Ctrl+z": () => undo(st),
  };
  C.keys.bind(VIEW, map);
}

async function backToPicker() {
  await closeWork();
  lsDel(LS_LAST);
  if (!root) return;
  root.innerHTML = `<div class="empty">Cargando sesiones…</div>`;
  try { sessions = await C.api.get("/api/sbd/sessions"); } catch { /* se reintenta al mostrar el selector */ }
  if (!root) return;
  showPicker();
}

// ------------------------------------------------------------------ navegación
function goFrame(st, n) { st.fv.pause(); st.fv.goTo(n); }

function onFrame(st, n) {
  if (st.closed) return;
  st.tl.setCursor(n);
  trackCoverage(st, n);
  st.el(".sbd-fv").classList.toggle("outside", n < st.S || n >= st.E);
  updateCurrent(st, n);
  if (st.pendingA !== null) { updateTimeline(st); renderBanner(st); }
  clearTimeout(st.posTimer);
  st.posTimer = setTimeout(() => lsSet(lsPos(st.sid), String(n)), 400);
}

function zoom(st, z) {
  const n = st.fv.current;
  if (z === "all") st.tl.setRange(st.S, st.E);
  else { const h = parseInt(z, 10); st.tl.setRange(n - h, n + h); }
}

function jumpTr(st, dir) {
  const n = st.fv.current;
  const order = sortTrs(st.trs);
  if (!order.length) { C.toast("Todavía no hay transiciones marcadas.", "info"); return; }
  let t;
  if (dir > 0) t = order.find((x) => trFirst(x) > n);
  else t = [...order].reverse().find((x) => trFirst(x) < n && !(x.type === "gradual" && n >= x.start && n <= x.end)) ||
    [...order].reverse().find((x) => trFirst(x) < n);
  if (!t) { C.toast(dir > 0 ? "No hay más transiciones después del cursor." : "No hay transiciones antes del cursor.", "info", 2000); return; }
  st.tab = "trs";
  goFrame(st, trFirst(t));
  renderLists(st);
}

// ------------------------------------------------------------------ cobertura («revisado hasta»)
function trackCoverage(st, n) {
  const prev = st.lastPos;
  st.lastPos = n;
  let a = n, b = n;
  if (prev !== null && Math.abs(n - prev) <= COV_STEP_S * st.fps) { a = Math.min(prev, n); b = Math.max(prev, n); }
  a = Math.max(st.S, a); b = Math.min(st.E - 1, b);
  if (a > b) return;
  const before = reviewedUntil(st);
  const inside = st.cov.some(([x, y]) => x <= a && b <= y);
  if (inside) return;
  st.cov = mergeIntervals([...st.cov, [a, b]]);
  st.covDirty = true;
  if (reviewedUntil(st) !== before) { renderMeters(st); st.tl.setProgress(reviewedUntil(st) + 1); }
  if (!st.covTimer) st.covTimer = setTimeout(() => { st.covTimer = null; if (!st.closed) doSave(st); }, COV_DEBOUNCE);
}
function reviewedUntil(st) {
  const first = st.cov[0];
  return first && first[0] <= st.S ? first[1] : st.S - 1;
}

// ------------------------------------------------------------------ modelo
function recompute(st) {
  st.trs = sortTrs(st.trs);
  st.shots = shotsOf(st.trs, st.S, st.E);
}

function shotIndexAt(st, n) {
  const sh = st.shots;
  let lo = 0, hi = sh.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (n < sh[mid].start) hi = mid - 1;
    else if (n > sh[mid].end) lo = mid + 1;
    else return mid;
  }
  return -1;
}

function describeTr(st, t) {
  return t.type === "cut" ? `el corte de ${C.fmt.time(t.frame, st.fps)}`
    : `la gradual ${C.fmt.time(t.start, st.fps)} → ${C.fmt.time(t.end, st.fps)}`;
}

// Misma validación que el servidor (store.validate_session), con mensajes para Carlos.
function validate(st, trs) {
  const { S, E, fps } = st;
  for (const t of trs) {
    if (t.type === "cut") {
      if (t.frame < S || t.frame >= E) return `El corte debe estar dentro de la ventana (${C.fmt.time(S + 1, fps)} a ${C.fmt.time(E - 1, fps)}). Ahora estás fuera de ella, en la zona gris de contexto.`;
      if (t.frame === S) return `Un corte en el primer fotograma de la ventana (${C.fmt.time(S, fps)}) no hace falta: la ventana ya empieza con un plano nuevo.`;
    } else {
      if (t.start > t.end) return `El fin de la gradual (${C.fmt.time(t.end, fps)}) no puede ser anterior a su inicio (${C.fmt.time(t.start, fps)}). Avanza hasta el último fotograma de la transición y pulsa B.`;
      if (t.start < S || t.end >= E) return `La transición gradual debe quedar dentro de la ventana (${C.fmt.time(S, fps)} a ${C.fmt.time(E - 1, fps)}).`;
    }
  }
  const order = sortTrs(trs);
  for (let i = 0; i + 1 < order.length; i++) {
    const a = order[i], b = order[i + 1];
    const fb = trFirst(b), next = trNext(a);
    if (a.type === "cut" && b.type === "cut" && a.frame === b.frame) return `Ya hay un corte en este fotograma (${C.fmt.time(a.frame, fps)}).`;
    if (fb < next || (a.type === "gradual" && fb <= a.end)) {
      const cut = [a, b].find((x) => x.type === "cut"), grad = [a, b].find((x) => x.type === "gradual");
      if (cut && grad) return `El corte de ${C.fmt.time(cut.frame, fps)} cae dentro de la transición gradual ${C.fmt.time(grad.start, fps)} → ${C.fmt.time(grad.end, fps)}. Una transición no puede solaparse con otra.`;
      return `${cap(describeTr(st, a))} y ${describeTr(st, b)} se solapan. Una transición no puede solaparse con otra.`;
    }
    if (fb === next) return `${cap(describeTr(st, a))} y ${describeTr(st, b)} quedan pegadas: entre dos transiciones debe quedar al menos un fotograma de plano.`;
  }
  return null;
}
const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);

function nextId(st) {
  let mx = 0;
  for (const t of st.trs) { const k = parseInt(String(t.id).replace(/^\D+/, ""), 10); if (Number.isFinite(k)) mx = Math.max(mx, k); }
  return "t" + (mx + 1);
}

function snapshot(st) { return { trs: clone(st.trs), labels: { ...st.labels }, complete: st.complete }; }
function pushUndo(st, what) { st.undo.push({ ...snapshot(st), what }); if (st.undo.length > MAX_UNDO) st.undo.shift(); }

// Aplica un nuevo conjunto de transiciones (validado), con re-etiquetado opcional y aviso de etiquetas perdidas.
function applyTrs(st, newTrs, what, { rekey = null, toastOk = null } = {}) {
  const err = validate(st, newTrs);
  if (err) { showError(st, err); return false; }
  pushUndo(st, what);
  st.trs = sortTrs(newTrs);
  const oldLabels = st.labels;
  st.shots = shotsOf(st.trs, st.S, st.E);
  const valid = new Set(st.shots.map((s) => String(s.start)));
  const labels = {};
  for (const [k, v] of Object.entries(oldLabels)) if (valid.has(k)) labels[k] = v;
  if (rekey) {
    for (const [from, to] of rekey) {
      if (oldLabels[from] && !valid.has(from) && valid.has(to) && !labels[to]) labels[to] = oldLabels[from];
    }
  }
  const lost = Object.keys(oldLabels).filter((k) => !(k in labels)).length;
  st.labels = labels;
  if (lost) {
    const txt = lost === 1 ? "Se ha quitado 1 etiqueta porque cambió el plano." : `Se han quitado ${lost} etiquetas porque cambió el plano.`;
    C.toast(txt, "warn", 5000);
    flash(st, (toastOk ? toastOk + ". " : "") + txt, "warn", 7000);
  } else if (toastOk) flash(st, `✓ ${toastOk}. Puedes deshacerlo con Ctrl+Z.`, "ok", 4000);
  commit(st);
  return true;
}

function showError(st, msg) {
  flash(st, msg, "error", 8000);
  const tools = st.el(".sbd-tools");
  if (tools) { tools.classList.remove("shake"); void tools.offsetWidth; tools.classList.add("shake"); }
}

// Línea de mensajes de la tarjeta «Marcar transiciones»: gradual abierta > último mensaje > consejo.
function flash(st, html, kind = "ok", ms = 4000, { raw = false } = {}) {
  st.msg = { html: raw ? html : esc(html), kind, until: Date.now() + ms };
  clearTimeout(st.msgTimer);
  st.msgTimer = setTimeout(() => renderBanner(st), ms + 20);
  renderBanner(st);
}

// ------------------------------------------------------------------ acciones de transiciones
function markCut(st) {
  const n = st.fv.current;
  if (st.pendingA !== null) { showError(st, `Hay una transición gradual abierta (desde ${C.fmt.time(st.pendingA, st.fps)}). Marca su fin con B o cancélala con Esc antes de añadir un corte.`); return; }
  st.tab = "trs";
  applyTrs(st, [...st.trs, { id: nextId(st), type: "cut", frame: n }], `añadir el corte de ${C.fmt.time(n, st.fps)}`,
    { toastOk: `Corte añadido en ${C.fmt.time(n, st.fps)}` });
}

function markGradualStart(st) {
  const n = st.fv.current;
  if (n < st.S || n >= st.E) { showError(st, "La transición gradual debe empezar dentro de la ventana."); return; }
  const inside = st.trs.find((t) => (t.type === "gradual" && n >= t.start && n <= t.end) || (t.type === "cut" && t.frame === n));
  if (inside) { showError(st, `Este fotograma ya pertenece a ${describeTr(st, inside)}.`); return; }
  const moved = st.pendingA !== null;
  st.pendingA = n;
  st.tab = "trs";
  if (moved) flash(st, `Inicio de la gradual movido a ${C.fmt.time(n, st.fps)}.`, "ok", 2500);
  metaChanged(st);
  renderAll(st);
}

function markGradualEnd(st) {
  const n = st.fv.current;
  if (st.pendingA === null) { showError(st, "Primero marca el inicio de la gradual con A (su primer fotograma)."); return; }
  const s = st.pendingA;
  if (n < s) { showError(st, `El fin de la gradual (${C.fmt.time(n, st.fps)}) no puede ser anterior a su inicio (${C.fmt.time(s, st.fps)}). Avanza hasta el último fotograma de la transición.`); return; }
  const ok = applyTrs(st, [...st.trs, { id: nextId(st), type: "gradual", start: s, end: n }],
    `añadir la gradual de ${C.fmt.time(s, st.fps)}`, { toastOk: `Gradual añadida: ${C.fmt.time(s, st.fps)} → ${C.fmt.time(n, st.fps)} (${n - s + 1} fot.)` });
  if (ok) { st.pendingA = null; renderAll(st); }
}

function cancelGradual(st) {
  if (st.pendingA === null) return;
  st.pendingA = null;
  flash(st, "Gradual abierta cancelada.", "info", 2500);
  metaChanged(st);
  renderAll(st);
}

function nearestTr(st, n) {
  let best = null, bd = Infinity;
  for (const t of st.trs) {
    const d = n < trFirst(t) ? trFirst(t) - n : n > trLast(t) ? n - trLast(t) : 0;
    if (d < bd) { bd = d; best = t; }
  }
  return best ? { t: best, d: bd } : null;
}

async function deleteNearest(st) {
  if (st.pendingA !== null) { cancelGradual(st); return; }
  const r = nearestTr(st, st.fv.current);
  if (!r) { C.toast("No hay transiciones que eliminar.", "info"); return; }
  await deleteTr(st, r.t.id, r.d);
}

async function deleteTr(st, id, dist = null) {
  const t = st.trs.find((x) => x.id === id);
  if (!t) return;
  const where = dist === null ? "" : dist === 0 ? " (está en el cursor)" : ` (a ${plural(dist, "fotograma", "fotogramas")} del cursor)`;
  const ok = await C.confirmDialog("¿Eliminar esta transición?", `Vas a eliminar ${describeTr(st, t)}${where}. Los dos planos que separa se unirán en uno. Puedes deshacerlo con Ctrl+Z.`,
    { ok: "Eliminar", cancel: "Cancelar", danger: true });
  if (!ok || st.closed) return;
  if (st.editId === id) st.editId = null;
  applyTrs(st, st.trs.filter((x) => x.id !== id), `eliminar ${describeTr(st, t)}`, { toastOk: "Transición eliminada" });
}

function moveTr(st, id, how) {
  const t = st.trs.find((x) => x.id === id);
  if (!t) return;
  const n = st.fv.current;
  const nt = { ...t };
  const oldNext = trNext(t);
  if (t.type === "cut") {
    if (how === "here") nt.frame = n; else if (how === "m1") nt.frame -= 1; else if (how === "p1") nt.frame += 1;
  } else {
    if (how === "sHere") nt.start = n; else if (how === "sM1") nt.start -= 1; else if (how === "sP1") nt.start += 1;
    else if (how === "eHere") nt.end = n; else if (how === "eM1") nt.end -= 1; else if (how === "eP1") nt.end += 1;
  }
  if (JSON.stringify(nt) === JSON.stringify(t)) return;
  const ok = applyTrs(st, st.trs.map((x) => (x.id === id ? nt : x)), `mover ${describeTr(st, t)}`,
    { rekey: [[String(oldNext), String(trNext(nt))]] });
  if (ok) goFrame(st, how.startsWith("e") ? nt.end : trFirst(nt));
}

// ------------------------------------------------------------------ etiquetado
function labelCurrent(st, lab) {
  const n = st.fv.current;
  const i = shotIndexAt(st, n);
  if (i < 0) {
    if (n < st.S || n >= st.E) showError(st, "Estás fuera de la ventana: aquí no hay planos que etiquetar.");
    else showError(st, "El cursor está dentro de una transición gradual. Muévete a un plano para etiquetarlo.");
    return;
  }
  setShotLabel(st, i, lab);
  if (prefs.autoAdvance) nextUnlabelled(st, 1, { quietDone: true });
}

function setShotLabel(st, i, lab, { toggle = false } = {}) {
  const s = st.shots[i];
  if (!s) return;
  const key = String(s.start);
  const cur = st.labels[key];
  if (cur === lab && !toggle) return;
  pushUndo(st, `etiquetar el plano ${i + 1}`);
  if (cur === lab) delete st.labels[key]; else st.labels[key] = lab;
  st.tab = "shots";
  commit(st);
}

function nextUnlabelled(st, dir, { quietDone = false } = {}) {
  const sh = st.shots;
  if (!sh.length) return;
  const n = st.fv.current;
  let i = shotIndexAt(st, n);
  if (i < 0) i = dir > 0 ? sh.findIndex((s) => s.start > n) - 1 : sh.findIndex((s) => s.start > n);
  if (i < -1) i = dir > 0 ? -1 : sh.length;
  for (let k = 1; k <= sh.length; k++) {
    const j = (((i + dir * k) % sh.length) + sh.length) % sh.length;
    if (!st.labels[String(sh[j].start)]) {
      st.tab = "shots";
      goFrame(st, Math.floor((sh[j].start + sh[j].end) / 2));
      renderLists(st);
      return;
    }
  }
  if (!quietDone) C.toast("¡Todos los planos están etiquetados! ✓", "ok");
  else C.toast("Último plano etiquetado: ya no quedan planos sin etiquetar ✓", "ok");
}

function undo(st) {
  const u = st.undo.pop();
  if (!u) { flash(st, "No hay nada que deshacer.", "info", 2500); return; }
  st.trs = u.trs; st.labels = u.labels; st.complete = u.complete;
  recompute(st);
  flash(st, `↶ Deshecho: ${u.what}.`, "info", 3000);
  commit(st, { fromUndo: true });
}

async function toggleFinish(st) {
  if (st.complete) {
    const ok = await C.confirmDialog("¿Reabrir la sesión?", "La sesión volverá a figurar como «en curso». No se pierde nada.", { ok: "Reabrir" });
    if (!ok) return;
    pushUndo(st, "reabrir la sesión");
    st.complete = false;
    commit(st);
    return;
  }
  if (st.pendingA !== null) {
    await C.confirmDialog("Falta cerrar una transición gradual",
      `Hay una gradual abierta desde ${C.fmt.time(st.pendingA, st.fps)}. Ve a su último fotograma y pulsa B (Fin gradual), o cancélala con Esc.`,
      { ok: "Entendido", cancel: "Cerrar" });
    return;
  }
  const missing = st.shots.filter((s) => !st.labels[String(s.start)]).length;
  if (missing) {
    const go = await C.confirmDialog("Faltan planos por etiquetar",
      `${missing === 1 ? "Falta 1 plano" : `Faltan ${missing} planos`} de ${st.shots.length} por etiquetar como Juego o No juego. ` +
      "Pulsa Tab para ir al siguiente sin etiquetar.", { ok: "Ir al siguiente sin etiquetar", cancel: "Cerrar" });
    if (go) nextUnlabelled(st, 1);
    return;
  }
  const ru = reviewedUntil(st);
  if (ru < st.E - 1) {
    const ok = await C.confirmDialog("¿Has revisado toda la ventana?",
      `Según el recorrido registrado, has revisado hasta ${C.fmt.clock(ru + 1 - st.S, st.fps)} de ${C.fmt.clock(st.E - st.S, st.fps)} ` +
      `(${C.fmt.time(ru + 1, st.fps)} del partido). Si ya has comprobado el resto, puedes terminar igualmente.`,
      { ok: "Terminar igualmente", cancel: "Seguir revisando" });
    if (!ok) return;
  }
  pushUndo(st, "terminar la sesión");
  st.complete = true;
  commit(st);
  C.toast("Sesión marcada como terminada. ¡Gracias!", "ok", 4000);
}

// ------------------------------------------------------------------ guardado
function commit(st, { fromUndo = false } = {}) {
  if (st.complete && st.shots.some((s) => !st.labels[String(s.start)])) {
    st.complete = false;
    if (!fromUndo) C.toast("La sesión vuelve a estar «en curso» porque hay planos sin etiquetar.", "warn", 4000);
  }
  st.seq++;
  st.blocked = false;
  writeBackup(st);
  C.saveStatus.set("dirty");
  scheduleSave(st, SAVE_DEBOUNCE);
  renderAll(st);
}

function writeBackup(st) {
  lsSet(lsBackup(st.sid), JSON.stringify({
    base: st.revision, at: Date.now(), transitions: st.trs, shot_labels: st.labels, complete: st.complete, reviewed: st.cov,
  }));
}

// Cambios que no son anotación (gradual abierta): se guardan pronto pero sin marcar «cambios sin guardar».
function metaChanged(st) {
  st.covDirty = true;
  clearTimeout(st.covTimer);
  st.covTimer = setTimeout(() => { st.covTimer = null; if (!st.closed) doSave(st); }, SAVE_DEBOUNCE);
}

function scheduleSave(st, ms) {
  clearTimeout(st.timer);
  st.timer = setTimeout(() => doSave(st), ms);
}

function buildBody(st) {
  const body = { ...st.doc };
  for (const k of Object.keys(body)) if (k.startsWith("_") || k === "current" || k === "error") delete body[k];
  return { ...body, revision: st.revision, transitions: sortTrs(st.trs), shot_labels: { ...st.labels }, complete: st.complete, reviewed: st.cov.map((x) => [...x]), open_gradual_start: st.pendingA };
}

function doSave(st) {
  if (st.saving) { st.again = true; return st.saving; }
  clearTimeout(st.timer); clearTimeout(st.retryTimer);
  if (st.blocked) return Promise.resolve();
  if (st.seq === st.savedSeq && !st.covDirty) return Promise.resolve();
  const seq = st.seq;
  const body = buildBody(st);
  const covWas = st.covDirty;
  st.covDirty = false;
  clearTimeout(st.covTimer); st.covTimer = null;
  const mine = () => W === st && !st.closed;
  const pr = (async () => {
    if (mine() && seq !== st.savedSeq) C.saveStatus.set("saving");
    try {
      const res = await C.api.put(`/api/sbd/${encodeURIComponent(st.sid)}`, body);
      st.revision = res.revision;
      const pruned = res._pruned_labels || [];
      delete res._pruned_labels;
      st.doc = res;
      st.savedSeq = Math.max(st.savedSeq, seq);
      if (st.seq === seq) {
        lsDel(lsBackup(st.sid));
        // El servidor es la referencia: si ha quitado etiquetas (no debería), lo reflejamos.
        if (pruned.length) {
          for (const k of pruned) delete st.labels[k];
          if (mine()) { C.toast(`El servidor ha quitado ${plural(pruned.length, "etiqueta", "etiquetas")} de planos que ya no existen.`, "warn"); renderAll(st); }
        }
        if (mine()) C.saveStatus.set("saved", `Guardado ${new Date().toLocaleTimeString("es-ES", { hour: "2-digit", minute: "2-digit" })}`);
      } else if (mine()) {
        C.saveStatus.set("dirty");
        scheduleSave(st, SAVE_DEBOUNCE);
      }
    } catch (e) {
      st.covDirty = st.covDirty || covWas;
      if (e.status === 409) {
        st.blocked = true;
        if (mine()) { C.saveStatus.set("error", "Conflicto de versiones"); setTimeout(() => onConflict(st, e), 0); }
      } else if (e.status === 400 || e.status === 404 || e.status === 413) {
        st.blocked = true;
        if (mine()) {
          C.saveStatus.set("error", "No se pudo guardar");
          C.confirmDialog("No se ha podido guardar", `${e.message} Tus cambios siguen guardados en este navegador. Puedes deshacer el último cambio (Ctrl+Z) e intentarlo de nuevo, o avisar a Marcos.`, { ok: "Entendido", cancel: "Cerrar" });
        }
      } else {
        if (mine()) C.saveStatus.set("error", "Sin conexión · reintentando…");
        if (!st.closed) st.retryTimer = setTimeout(() => doSave(st), RETRY_MS);
      }
    } finally {
      st.saving = null;
      if (st.again) { st.again = false; if (!st.closed || st.seq !== st.savedSeq) doSave(st); }
    }
  })();
  st.saving = pr;
  return pr;
}

async function flushSave(st) {
  clearTimeout(st.timer); clearTimeout(st.covTimer); st.covTimer = null; clearTimeout(st.retryTimer);
  if (st.saving) { try { await st.saving; } catch { /* gestionado */ } }
  if (st.seq !== st.savedSeq || st.covDirty) { try { await doSave(st); } catch { /* gestionado */ } }
  if (st.saving) { try { await st.saving; } catch { /* gestionado */ } }
}

async function onConflict(st, e) {
  if (W !== st || st.closed) return;
  await C.confirmDialog("Esta sesión ha cambiado en otra ventana",
    "Se han guardado cambios de esta misma sesión desde otra pestaña o ventana del navegador. Para no mezclar versiones, " +
    "voy a cargar la versión guardada en el servidor. Si aquí tenías cambios sin guardar, a continuación podrás recuperarlos.",
    { ok: "Cargar la versión guardada", cancel: "Cerrar" });
  if (W !== st) return;
  const cur = e.body && (e.body.current || (e.body.session_id ? e.body : null));
  st.savedSeq = st.seq; st.covDirty = false; // la copia local queda en localStorage
  await openSession(st.sid, cur && cur.session_id === st.sid ? clone(cur) : null);
}

async function offerBackup(st) {
  const bk = lsGetJSON(lsBackup(st.sid));
  if (!bk) return;
  const same = JSON.stringify(sortTrs(bk.transitions || [])) === JSON.stringify(sortTrs(st.trs)) &&
    JSON.stringify(Object.entries(bk.shot_labels || {}).sort()) === JSON.stringify(Object.entries(st.labels).sort()) &&
    !!bk.complete === st.complete;
  if (same) { lsDel(lsBackup(st.sid)); return; }
  const nl = Object.keys(bk.shot_labels || {}).length, sl = Object.keys(st.labels).length;
  const when = new Date(bk.at || Date.now()).toLocaleString("es-ES", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" });
  const older = (bk.base || 0) < st.revision;
  const ok = await C.confirmDialog("Hay cambios sin guardar de esta sesión",
    `Este navegador conserva una copia de cambios que no llegaron a guardarse (${when}): ${plural((bk.transitions || []).length, "transición", "transiciones")} y ${plural(nl, "etiqueta", "etiquetas")}. ` +
    `La versión guardada tiene ${plural(st.trs.length, "transición", "transiciones")} y ${plural(sl, "etiqueta", "etiquetas")}.` +
    (older ? " Atención: la versión guardada se modificó después (quizá en otra pestaña)." : "") + " ¿Recuperar la copia?",
    { ok: "Recuperar mis cambios", cancel: "Descartar la copia" });
  if (W !== st) return;
  if (!ok) { lsDel(lsBackup(st.sid)); return; }
  const trs = (bk.transitions || []).filter((t) => t && (t.type === "cut" || t.type === "gradual"));
  const err = validate(st, trs);
  if (err) { C.toast("La copia no es válida y no se puede recuperar: " + err, "error", 7000); return; }
  pushUndo(st, "recuperar la copia del navegador");
  st.trs = trs;
  recompute(st);
  const valid = new Set(st.shots.map((s) => String(s.start)));
  st.labels = Object.fromEntries(Object.entries(bk.shot_labels || {}).filter(([k, v]) => valid.has(k) && LABEL_TEXT[v]));
  st.complete = !!bk.complete;
  if (Array.isArray(bk.reviewed)) { st.cov = mergeIntervals([...st.cov, ...normCov(bk.reviewed, st.S, st.E)]); st.covDirty = true; }
  C.toast("Cambios recuperados. Guardando…", "ok");
  commit(st);
}

// ------------------------------------------------------------------ curva de cambios
async function loadCurve(st) {
  const chunk = CURVE_CHUNK_S * st.fps;
  const parts = [];
  for (let a = st.S; a < st.E; a += chunk) parts.push([a, Math.min(st.E, a + chunk)]);
  st.curveState = "loading";
  renderCurveState(st);
  try {
    const res = await Promise.all(parts.map(([a, b]) => C.api.get(`/api/diffcurve/${encodeURIComponent(st.m.id)}?start=${a}&end=${b}`)));
    if (st.closed) return;
    const raw = [];
    for (const r of res) { const mx = r.raw_max || 1; for (const v of r.values) raw.push(v * mx); }
    // Escala robusta: el percentil 99,8 marca el «1» (los cortes muy fuertes se recortan).
    const sorted = raw.filter((v) => v > 0).sort((x, y) => x - y);
    const ref = sorted.length ? sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.998))] : 1;
    st.curve = raw.map((v) => Math.min(1, v / (ref || 1)));
    st.curveState = "ok";
  } catch (e) {
    if (st.closed) return;
    st.curveState = "error";
    st.curveError = e.message;
  }
  renderCurveState(st);
  applyCurve(st);
}

function applyCurve(st) { st.tl.setCurve(prefs.showCurve && st.curve ? st.curve : null, st.S); }

function renderCurveState(st) {
  const el = st.el("[data-curve-state]");
  if (!el) return;
  el.innerHTML = st.curveState === "loading" ? `<span class="sbd-spin"></span> calculando la curva (la primera vez puede tardar un minuto)…`
    : st.curveState === "error" ? `<span class="t-danger">No se pudo calcular la curva: ${esc(st.curveError || "")}</span>` : "";
}

// ------------------------------------------------------------------ render
function renderAll(st) {
  if (st.closed) return;
  recompute(st);
  renderHead(st);
  renderMeters(st);
  renderBanner(st);
  updateTimeline(st);
  renderLists(st);
  st.curShot = -2; st.curTr = undefined;
  updateCurrent(st, st.fv.current);
  st.fv.refresh();
  if (st.fv.setStripClass) st.fv.setStripClass((f) => stripClass(st, f));
}

function renderHead(st) {
  const stEl = st.el("[data-sstate]");
  stEl.innerHTML = st.complete ? `<span class="badge ok">✓ Terminada</span>` : `<span class="badge warn">En curso</span>`;
  const fin = st.el('[data-act="finish"]');
  fin.innerHTML = st.complete ? "↺ Reabrir sesión" : "✓ Marcar sesión como terminada";
  fin.classList.toggle("btn-ok", !st.complete);
}

function renderMeters(st) {
  const len = st.E - st.S;
  const ru = reviewedUntil(st);
  const done = Math.max(0, ru + 1 - st.S);
  const rev = st.el("[data-m-rev]");
  if (!rev) return;
  rev.innerHTML = `Revisado hasta <b class="mono">${C.fmt.clock(done, st.fps)}</b> de <span class="mono">${C.fmt.clock(len, st.fps)}</span>`;
  st.el("[data-m-rev-bar]").style.width = `${(100 * done) / len}%`;
  const nl = st.shots.filter((s) => st.labels[String(s.start)]).length;
  st.el("[data-m-lab]").innerHTML = `Planos etiquetados <b>${nl}</b> de ${st.shots.length}`;
  st.el("[data-m-lab-bar]").style.width = `${st.shots.length ? (100 * nl) / st.shots.length : 0}%`;
  st.tl.setProgress(ru >= st.S ? ru + 1 : null);
}

function renderBanner(st) {
  const el = st.el("[data-msg]");
  if (!el || st.closed) return;
  const msgLive = st.msg && st.msg.until > Date.now();
  if (st.pendingA !== null && !(msgLive && st.msg.kind === "error")) {
    const len = st.fv.current - st.pendingA + 1;
    el.className = "sbd-msg pending";
    el.innerHTML = `<div class="grow"><b>Gradual abierta desde ${C.fmt.time(st.pendingA, st.fps)}</b>
        <div class="small">${len >= 1 ? `${plural(len, "fotograma", "fotogramas")} hasta el cursor. ` : "El cursor está antes del inicio. "}Ve a su <b>último</b> fotograma y pulsa <kbd>B</kbd>.</div></div>
      <button class="btn btn-sm" data-act="cancelGrad" title="Cancelar la gradual abierta (Esc)">Cancelar <kbd>Esc</kbd></button>`;
    return;
  }
  if (msgLive) {
    el.className = `sbd-msg ${st.msg.kind}`;
    el.innerHTML = `<div>${st.msg.html}</div>`;
    return;
  }
  el.className = "sbd-msg tip";
  el.innerHTML = `<div><b>Corte:</b> sitúate en el <b>primer fotograma del plano nuevo</b>.
    <b>Gradual</b> (fundido, cortinilla, logo): marca su primer y su último fotograma.</div>`;
}

function updateTimeline(st) {
  const marks = st.trs.filter((t) => t.type === "cut").map((t) => ({ frame: t.frame, cls: "cut", title: "Corte" }));
  const spans = [];
  for (const t of st.trs) if (t.type === "gradual") spans.push({ start: t.start, end: t.end, cls: "gradual", alpha: 0.6 });
  const cssv = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  for (const s of st.shots) {
    const lab = st.labels[String(s.start)];
    spans.push(lab ? { start: s.start, end: s.end, cls: lab === "play" ? "play" : "noplay", row: "bottom", alpha: 0.85 }
      : { start: s.start, end: s.end, row: "bottom", color: cssv("--border-strong"), alpha: 0.9 });
  }
  if (st.pendingA !== null) {
    const n = st.fv ? st.fv.current : st.pendingA;
    spans.push({ start: Math.min(st.pendingA, n), end: Math.max(st.pendingA, n), cls: "gradual", alpha: 0.25 });
    marks.push({ frame: st.pendingA, cls: "ok", label: "A", width: 2 });
  }
  if (st.editId) {
    const t = st.trs.find((x) => x.id === st.editId);
    if (t && t.type === "cut") marks.push({ frame: t.frame, color: cssv("--primary"), width: 4 });
  }
  st.tl.setMarks(marks);
  st.tl.setSpans(spans);
}

function renderLists(st) {
  for (const b of root.querySelectorAll(".sbd-tab")) b.classList.toggle("active", b.dataset.tab === st.tab);
  st.el("[data-n-trs]").textContent = st.trs.length;
  const nl = st.shots.filter((s) => st.labels[String(s.start)]).length;
  st.el("[data-n-shots]").textContent = `${nl}/${st.shots.length}`;
  const list = st.el("[data-list]");
  const keepScroll = list.dataset.tab === st.tab ? list.scrollTop : 0;
  list.dataset.tab = st.tab;
  list.innerHTML = st.tab === "trs" ? trListHtml(st) : shotListHtml(st);
  list.scrollTop = keepScroll;
  st.curShot = -2; st.curTr = undefined;
  updateCurrent(st, st.fv.current);
}

function trListHtml(st) {
  if (!st.trs.length) {
    return `<div class="empty small"><div class="sbd-empty-ico">✂</div>Aún no hay transiciones.<br>
      Recorre la ventana (→ o Espacio) y pulsa <kbd>C</kbd> en el <b>primer fotograma de cada plano nuevo</b>.</div>`;
  }
  const fps = st.fps;
  return st.trs.map((t, i) => {
    const editing = st.editId === t.id;
    const main = t.type === "cut"
      ? `<b>Corte</b> <span class="mono">${C.fmt.time(t.frame, fps)}</span> <span class="mono muted small">#${C.fmt.frame(t.frame)}</span>`
      : `<b>Gradual</b> <span class="mono">${C.fmt.time(t.start, fps)} → ${C.fmt.time(t.end, fps)}</span> <span class="muted small">${t.end - t.start + 1} fot.</span>`;
    const mv = (how, txt, title) => `<button class="btn btn-sm" data-mv="${how}" data-mv-id="${esc(t.id)}" title="${title}">${txt}</button>`;
    const edit = !editing ? "" : t.type === "cut"
      ? `<div class="sbd-tr-edit"><span class="small muted">Fotograma del corte:</span>
          ${mv("m1", "‹ −1", "Un fotograma antes")}${mv("here", "Al cursor", "Mover el corte al fotograma actual")}${mv("p1", "+1 ›", "Un fotograma después")}</div>`
      : `<div class="sbd-tr-edit"><span class="small muted">Inicio:</span>
          ${mv("sM1", "‹ −1", "Inicio un fotograma antes")}${mv("sHere", "Al cursor", "Poner el inicio en el fotograma actual")}${mv("sP1", "+1 ›", "Inicio un fotograma después")}</div>
         <div class="sbd-tr-edit"><span class="small muted">Fin:</span>
          ${mv("eM1", "‹ −1", "Fin un fotograma antes")}${mv("eHere", "Al cursor", "Poner el fin en el fotograma actual")}${mv("eP1", "+1 ›", "Fin un fotograma después")}
          <button class="btn btn-sm" data-tr-go-end="${esc(t.id)}" title="Ir al último fotograma de la gradual">Ir al fin</button></div>`;
    return `<div class="sbd-tr ${t.type} ${editing ? "editing" : ""}" data-tid="${esc(t.id)}">
      <div class="sbd-tr-row">
        <span class="sbd-tr-n">${i + 1}</span>
        <span class="sbd-tr-ico ${t.type}">${t.type === "cut" ? "✂" : "◐"}</span>
        <span class="sbd-tr-main">${main}</span>
        <span class="sbd-tr-acts">
          <button class="btn btn-sm btn-ghost" data-tr-go="${esc(t.id)}" title="Ir a esta transición">Ir</button>
          <button class="btn btn-sm btn-ghost ${editing ? "on" : ""}" data-tr-edit="${esc(t.id)}" title="Ajustar la posición">${editing ? "Cerrar" : "Ajustar"}</button>
          <button class="btn btn-sm btn-ghost btn-danger" data-tr-del="${esc(t.id)}" title="Eliminar">✕</button>
        </span>
      </div>${edit}
    </div>`;
  }).join("");
}

function shotListHtml(st) {
  if (!st.shots.length) return `<div class="empty small">No hay planos (la ventana está cubierta por transiciones).</div>`;
  const fps = st.fps, mid = st.m.id;
  const head = st.trs.length ? "" : `<div class="sbd-note small">Aún no has marcado transiciones: toda la ventana cuenta como un único plano.
    Lo normal es marcar primero los cortes y después etiquetar.</div>`;
  return head + st.shots.map((s, i) => {
    const lab = st.labels[String(s.start)];
    const m = Math.floor((s.start + s.end) / 2);
    const th = (f, t) => `<button class="sbd-shot-th" data-shot-go="${i}" data-f="${f}" title="${t} · ${C.fmt.time(f, fps)}"><img loading="lazy" alt="" src="${C.frameUrl(mid, f, THUMB_W)}"></button>`;
    return `<div class="sbd-shot ${lab === "play" ? "play" : lab === "no_play" ? "noplay" : "none"}" data-si="${i}">
      <div class="sbd-shot-head">
        <b>Plano ${i + 1}</b>
        <span class="mono small">${C.fmt.time(s.start, fps)} – ${C.fmt.time(s.end, fps)}</span>
        <span class="muted small">${secs(s.end - s.start + 1, fps)}</span>
        <span class="grow"></span>
        <button class="btn btn-sm btn-ghost" data-shot-go="${i}" title="Ir al centro del plano">Ir</button>
      </div>
      <div class="sbd-shot-ths">${th(s.start, "Primer fotograma")}${th(m, "Fotograma central")}${th(s.end, "Último fotograma")}</div>
      <div class="sbd-shot-labs">
        <button class="sbd-chip play ${lab === "play" ? "on" : ""}" data-setlab="play" data-shot="${i}" title="${lab === "play" ? "Pulsa otra vez para quitar la etiqueta" : "Marcar como Juego"}">▶ Juego</button>
        <button class="sbd-chip noplay ${lab === "no_play" ? "on" : ""}" data-setlab="no_play" data-shot="${i}" title="${lab === "no_play" ? "Pulsa otra vez para quitar la etiqueta" : "Marcar como No juego"}">■ No juego</button>
      </div>
    </div>`;
  }).join("");
}

// Actualizaciones baratas en cada cambio de fotograma.
function updateCurrent(st, n) {
  const i = shotIndexAt(st, n);
  const shotInfo = st.el("[data-shotinfo]");
  const lab = i >= 0 ? st.labels[String(st.shots[i].start)] : null;
  for (const b of root.querySelectorAll("[data-setcur]")) {
    b.classList.toggle("on", lab === b.dataset.setcur);
    b.disabled = i < 0;
  }
  if (i >= 0) {
    const s = st.shots[i];
    shotInfo.innerHTML = `Plano ${i + 1} de ${st.shots.length} · ${secs(s.end - s.start + 1, st.fps)}`;
  } else shotInfo.textContent = n < st.S || n >= st.E ? "fuera de la ventana" : "dentro de una gradual";
  const list = st.el("[data-list]");
  if (st.tab === "shots" && i !== st.curShot) {
    const old = list.querySelector(".sbd-shot.current");
    if (old) old.classList.remove("current");
    const el = i >= 0 ? list.querySelector(`.sbd-shot[data-si="${i}"]`) : null;
    if (el) { el.classList.add("current"); scrollIntoList(list, el); }
  }
  st.curShot = i;
  if (st.tab === "trs") {
    const t = st.trs.find((x) => (x.type === "cut" ? x.frame === n : n >= x.start && n <= x.end));
    const id = t ? t.id : null;
    if (id !== st.curTr) {
      const old = list.querySelector(".sbd-tr.current");
      if (old) old.classList.remove("current");
      const el = id ? list.querySelector(`.sbd-tr[data-tid="${CSS.escape(id)}"]`) : null;
      if (el) { el.classList.add("current"); scrollIntoList(list, el); }
    }
    st.curTr = id;
  }
}

function scrollIntoList(list, el) {
  const lr = list.getBoundingClientRect(), er = el.getBoundingClientRect();
  if (er.top < lr.top) list.scrollTop -= lr.top - er.top + 6;
  else if (er.bottom > lr.bottom) list.scrollTop += er.bottom - lr.bottom + 6;
}

// ------------------------------------------------------------------ ayuda
function helpHtml() {
  return `
  <p>En esta sección se anotan, dentro de un fragmento («ventana») de un partido, <b>todas las transiciones entre planos</b>
  y después se indica si cada plano es <b>Juego</b> o <b>No juego</b>.</p>
  <h3>Definiciones</h3>
  <table>
    <tr><td><b>Plano</b></td><td>Fragmento continuo grabado por una misma cámara, sin cortes.</td></tr>
    <tr><td><b style="color:var(--cut)">Corte</b></td><td>Cambio instantáneo de un plano a otro. Se marca en el
      <b>primer fotograma del plano nuevo</b> (el anterior termina en el fotograma justo antes).</td></tr>
    <tr><td><b style="color:#b45309">Transición gradual</b></td><td>Cambio progresivo: fundido, encadenado, cortinilla o
      animación con el logo (típica antes y después de las repeticiones). Se marcan su <b>primer</b> y su <b>último</b> fotograma.
      No hace falta obsesionarse con el fotograma exacto, pero sí que la gradual cubra toda la transición.</td></tr>
    <tr><td><b style="color:var(--play)">Juego</b></td><td>El partido en directo (no repeticiones).</td></tr>
    <tr><td><b style="color:var(--noplay)">No juego</b></td><td>Repeticiones, primeros planos, público, banquillo, gráficos, pausas.</td></tr>
  </table>
  <h3>Cómo trabajar</h3>
  <ol>
    <li>Elige partido y ventana (por defecto 5 min desde el minuto 20) o continúa una sesión guardada.</li>
    <li>Recorre la ventana fotograma a fotograma (<kbd>→</kbd>) o reproduciendo (<kbd>Espacio</kbd>). La <b>tira de miniaturas</b>
      bajo la imagen muestra los fotogramas vecinos: un corte se ve como un cambio brusco entre dos miniaturas.</li>
    <li>En cada corte, sitúate en el primer fotograma del plano nuevo y pulsa <kbd>C</kbd> (o el botón «Corte aquí»).</li>
    <li>En cada gradual, pulsa <kbd>A</kbd> en su primer fotograma y <kbd>B</kbd> en el último. Mientras falte el fin verás
      el aviso «Gradual abierta»; <kbd>Esc</kbd> la cancela.</li>
    <li>Etiqueta cada plano con <kbd>1</kbd> (Juego) o <kbd>2</kbd> (No juego). <kbd>Tab</kbd> salta al siguiente plano sin etiquetar.
      También puedes etiquetar desde la pestaña «Planos», que muestra 3 miniaturas de cada plano (inicio, centro y fin).</li>
    <li>Al acabar, pulsa «Marcar sesión como terminada».</li>
  </ol>
  <h3>Ayudas</h3>
  <ul>
    <li><b>Curva de cambios</b> (gris, en la línea de tiempo): mide cuánto cambia la imagen entre fotogramas consecutivos.
      Los picos suelen indicar cortes, pero también aparecen con movimientos rápidos de cámara o flashes; es solo una ayuda
      para orientarse, la decisión es tuya. Se puede ocultar.</li>
    <li><b>Revisado hasta</b>: tramo de la ventana recorrido de forma continua desde su inicio (fotograma a fotograma,
      en pasos de hasta 1 s o reproduciendo). Los saltos grandes no cuentan como revisado.</li>
    <li>Puedes ver hasta 2 s antes y después de la ventana como contexto (en gris), pero ahí no se anota nada.</li>
    <li>Si mueves o borras una transición, las etiquetas de los planos afectados pueden perderse; se avisa siempre.</li>
  </ul>
  <h3>Atajos de teclado</h3>
  <table>
    <tr><td><kbd>←</kbd> <kbd>→</kbd></td><td>±1 fotograma</td></tr>
    <tr><td><kbd>Mayús+←</kbd> <kbd>Mayús+→</kbd></td><td>±1 segundo</td></tr>
    <tr><td><kbd>Ctrl+←</kbd> <kbd>Ctrl+→</kbd></td><td>±10 segundos</td></tr>
    <tr><td><kbd>Re Pág</kbd> <kbd>Av Pág</kbd></td><td>±1 minuto</td></tr>
    <tr><td><kbd>Espacio</kbd></td><td>Reproducir / pausar</td></tr>
    <tr><td><kbd>Inicio</kbd> <kbd>Fin</kbd></td><td>Principio / final de la ventana</td></tr>
    <tr><td><kbd>G</kbd></td><td>Ir a un tiempo o fotograma</td></tr>
    <tr><td><kbd>C</kbd></td><td>Corte aquí (primer fotograma del plano nuevo)</td></tr>
    <tr><td><kbd>A</kbd> / <kbd>B</kbd></td><td>Inicio / fin de una transición gradual</td></tr>
    <tr><td><kbd>Esc</kbd></td><td>Cancelar la gradual abierta</td></tr>
    <tr><td><kbd>J</kbd> / <kbd>K</kbd></td><td>Transición anterior / siguiente</td></tr>
    <tr><td><kbd>Supr</kbd></td><td>Eliminar la transición más cercana al cursor (pide confirmación)</td></tr>
    <tr><td><kbd>1</kbd> / <kbd>2</kbd></td><td>Plano actual: Juego / No juego</td></tr>
    <tr><td><kbd>Tab</kbd> o <kbd>↓</kbd></td><td>Siguiente plano sin etiquetar (<kbd>Mayús+Tab</kbd> o <kbd>↑</kbd>: anterior)</td></tr>
    <tr><td><kbd>Ctrl+Z</kbd></td><td>Deshacer</td></tr>
    <tr><td><kbd>?</kbd></td><td>Mostrar / ocultar esta ayuda</td></tr>
  </table>
  <h3>Guardado</h3>
  <p>Todo se guarda solo, en el servidor, un instante después de cada cambio (indicador arriba a la derecha). Si se corta la
  conexión, se reintenta automáticamente y se guarda una copia en este navegador que se ofrece recuperar al volver.
  Al eliminar una sesión, se guarda una copia en la papelera del servidor.</p>`;
}
