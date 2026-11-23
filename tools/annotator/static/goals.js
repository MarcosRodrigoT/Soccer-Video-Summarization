// Vista «Goles»: para cada gol de los partidos de otras ligas (PL1, SA1) se anotan
// tres fotogramas, con la misma definición que la ground truth de LaLiga:
//   start_frame           primer fotograma del plano en que se juega el gol
//   end_frame_shot        último fotograma de ese plano
//   end_frame_scoreboard  primer fotograma con el nuevo resultado en el marcador
//
// Guardado: automático (600 ms tras cada cambio, inmediato al cambiar de gol o de
// partido y al salir), con control de revisión (409 = otra pestaña guardó antes) y
// copia local en localStorage hasta que el servidor confirma.

const FIELDS = [
  {
    key: "start_frame", title: "Inicio del plano del gol", short: "Inicio del plano", key1: "S", cls: "start", color: "--ok",
    desc: "Primer fotograma del plano (cámara) en el que se juega la jugada del gol, justo después de un corte o transición.",
  },
  {
    key: "end_frame_shot", title: "Fin del plano del gol", short: "Fin del plano", key1: "E", cls: "end", color: "--danger",
    desc: "Último fotograma de ese mismo plano, justo antes del corte a la repetición, la celebración, etc.",
  },
  {
    key: "end_frame_scoreboard", title: "Cambio en el marcador", short: "Cambio de marcador", key1: "M", cls: "score", color: "--primary",
    desc: "Primer fotograma en el que el marcador de la pantalla ya muestra el nuevo resultado.",
  },
];
const FIELD = Object.fromEntries(FIELDS.map((f) => [f.key, f]));

const LS = {
  lastMatch: "annotator.goals.lastMatch",
  lastGoal: (m) => `annotator.goals.lastGoal.${m}`,
  backup: (m) => `annotator.goals.pending.${m}`,
  conflict: (m) => `annotator.goals.conflict.${m}`,
};
const LEAD_SECONDS = 8;      // al abrir un gol sin marcar, el visor se coloca 8 s antes del cambio detectado
const SAVE_DEBOUNCE = 600;
const RETRY_MS = 5000;

let S = null; // estado de la vista montada

// ------------------------------------------------------------------ utilidades
function lsGet(k) { try { return localStorage.getItem(k); } catch { return null; } }
function lsSet(k, v) { try { localStorage.setItem(k, v); return true; } catch { return false; } }
function lsDel(k) { try { localStorage.removeItem(k); } catch { /* nada */ } }
function lsJson(k) { try { const v = lsGet(k); return v ? JSON.parse(v) : null; } catch { return null; } }
const enc = encodeURIComponent;
const clone = (o) => JSON.parse(JSON.stringify(o));
const isNum = (v) => typeof v === "number" && Number.isFinite(v);
function secs(frames, fps) { return (frames / fps).toLocaleString("es-ES", { minimumFractionDigits: 1, maximumFractionDigits: 1 }); }
function withTimeout(p, ms) { return Promise.race([p, new Promise((r) => setTimeout(r, ms))]); }

function normalizeGoal(g, i) {
  const out = { ...g };
  out.id = typeof g.id === "string" && g.id ? g.id : `g${i + 1}`;
  for (const f of FIELDS) out[f.key] = isNum(g[f.key]) ? g[f.key] : null;
  out.notes = typeof g.notes === "string" ? g.notes : "";
  out.skipped = !!g.skipped;
  return out;
}
function normalizeDoc(raw, matchId) {
  const d = raw && typeof raw === "object" ? { ...raw } : {};
  d.match_id = matchId;
  d.schema = d.schema || 1;
  d.revision = isNum(d.revision) ? d.revision : 0;
  d.complete = !!d.complete;
  d.goals = Array.isArray(d.goals) ? d.goals.map(normalizeGoal) : [];
  return d;
}
function contentKey(doc) {
  return JSON.stringify({ c: !!doc.complete, g: (doc.goals || []).map((g) => [g.id, g.start_frame, g.end_frame_shot, g.end_frame_scoreboard, g.notes || "", !!g.skipped]) });
}
function goalStatus(g) {
  const n = FIELDS.filter((f) => isNum(g[f.key])).length;
  return n === 3 ? "complete" : n === 0 ? "pending" : "partial";
}
function sortKey(g) {
  for (const k of ["start_frame", "hint_frame", "added_at_frame", "end_frame_shot", "end_frame_scoreboard"]) if (isNum(g[k])) return g[k];
  return Number.MAX_SAFE_INTEGER;
}

// ------------------------------------------------------------------ guardado
class Saver {
  constructor(st, matchId, doc) {
    this.st = st; this.id = matchId; this.doc = doc;
    this.rev = doc.revision || 0;
    this.seq = 0; this.savedSeq = 0;
    this.timer = null; this.retry = null; this.inflight = null;
    this.blocked = false; this.failures = 0; this.attached = true;
  }
  get dirty() { return this.seq !== this.savedSeq; }
  status(state, text) { if (this.attached || state === "error") this.st.ctx.saveStatus.set(state, text); }

  changed() {
    this.seq++;
    this.blocked = false;
    this.backup();
    this.status("dirty");
    clearTimeout(this.timer);
    this.timer = setTimeout(() => this.flush(), SAVE_DEBOUNCE);
  }
  backup() {
    lsSet(LS.backup(this.id), JSON.stringify({ doc: this.doc, baseRevision: this.rev, ts: Date.now() }));
  }

  flush() {
    clearTimeout(this.timer); this.timer = null;
    clearTimeout(this.retry); this.retry = null;
    if (this.inflight) return this.inflight.then(() => (this.dirty && !this.blocked ? this.flush() : undefined));
    if (!this.dirty || this.blocked) return Promise.resolve();
    const seq = this.seq;
    const body = clone(this.doc);
    body.match_id = this.id;
    body.revision = this.rev;
    this.status("saving");
    this.inflight = this.st.ctx.api.put(`/api/goals/${enc(this.id)}`, body)
      .then((saved) => {
        this.rev = isNum(saved && saved.revision) ? saved.revision : this.rev + 1;
        this.doc.revision = this.rev;
        if (saved && saved.updated) this.doc.updated = saved.updated;
        this.savedSeq = seq;
        if (this.failures) this.st.ctx.toast("Conexión recuperada: cambios guardados.", "ok");
        this.failures = 0;
        if (this.dirty) { // hubo más cambios mientras se guardaba
          this.backup();
          this.status("dirty");
          this.timer = setTimeout(() => this.flush(), 250);
        } else {
          lsDel(LS.backup(this.id));
          this.status("saved");
        }
        if (this.attached) onSaved(this.st);
      }, (e) => this.onError(e))
      .finally(() => { this.inflight = null; });
    return this.inflight;
  }

  onError(e) {
    const { ctx } = this.st;
    const status = e && e.status;
    if (status === 409) return this.conflict(e.body);
    if (!status || status >= 500) {
      this.failures++;
      this.status("error", "Sin conexión · reintentando");
      if (this.failures === 1) {
        ctx.toast("No se pudo guardar (sin conexión con el servidor). Tus cambios están a salvo en este navegador y se reintentará automáticamente.", "error", 7000);
      }
      this.retry = setTimeout(() => this.flush(), RETRY_MS);
      return;
    }
    // 400 y similares: el servidor rechaza el contenido; no tiene sentido reintentar igual.
    this.blocked = true;
    this.status("error", "No se pudo guardar");
    const msg = (e.body && e.body.error) || e.message || `Error ${status}`;
    ctx.toast(`El servidor no aceptó los cambios: ${msg}`, "error", 9000);
  }

  async conflict(body) {
    const { ctx } = this.st;
    this.blocked = true;
    lsSet(LS.conflict(this.id), JSON.stringify({ doc: this.doc, baseRevision: this.rev, ts: Date.now() }));
    let server = body && (body.current || body.document || body.doc || (Array.isArray(body.goals) ? body : null));
    if (!server) {
      try { server = await ctx.api.get(`/api/goals/${enc(this.id)}`); } catch { server = null; }
    }
    this.status("error", "Versión desactualizada");
    await infoDialog(
      "Este partido se ha modificado en otra ventana",
      `<p>Mientras trabajabas aquí se guardó otra versión de las anotaciones de este partido, seguramente desde
       <b>otra pestaña o ventana</b> del navegador.</p>
       <p>Para no mezclar versiones se va a cargar la versión guardada en el servidor. Los últimos cambios de esta
       pestaña que no llegaron a guardarse se han conservado como copia de seguridad en este navegador
       (avisa a Marcos si los necesitas).</p>
       <p class="muted">Consejo: trabaja con una sola pestaña del anotador abierta.</p>`,
      "Cargar la versión guardada",
    );
    if (!server) {
      try { server = await ctx.api.get(`/api/goals/${enc(this.id)}`); } catch (e2) {
        ctx.toast("No se pudo descargar la versión del servidor. Recarga la página.", "error", 8000);
        return;
      }
    }
    this.doc = normalizeDoc(server, this.id);
    this.rev = this.doc.revision;
    this.savedSeq = this.seq;
    this.blocked = false;
    lsDel(LS.backup(this.id));
    this.status("saved");
    if (this.attached) onDocReplaced(this.st);
  }

  detach() { this.attached = false; }
}

function infoDialog(title, html, okText = "Aceptar") {
  return new Promise((resolve) => {
    const root = document.getElementById("dialog-root");
    const wrap = document.createElement("div");
    wrap.className = "dialog-backdrop";
    wrap.innerHTML = `<div class="dialog gl-dialog" role="alertdialog" aria-modal="true">
        <h3>${S ? S.ctx.escapeHtml(title) : title}</h3><div class="gl-dialog-body">${html}</div>
        <div class="row"><button class="btn btn-primary" data-a="1">${okText}</button></div></div>`;
    const done = () => { wrap.remove(); document.removeEventListener("keydown", onKey, true); resolve(true); };
    const onKey = (e) => { if (e.key === "Enter" || e.key === "Escape") { e.preventDefault(); e.stopPropagation(); done(); } };
    wrap.querySelector("[data-a]").addEventListener("click", done);
    document.addEventListener("keydown", onKey, true);
    root.appendChild(wrap);
    wrap.querySelector("[data-a]").focus();
  });
}

// ------------------------------------------------------------------ montaje
export async function mount(container, ctx) {
  const st = (S = {
    ctx, container, matches: [], matchId: null, info: null, hints: {}, fps: 25, total: 1,
    saver: null, selId: null, fv: null, tl: null, cuts: [], undo: [], colors: {},
    otherProgress: {}, sb: { want: null, busy: false, token: 0 }, notesUndoPushed: false, loading: false,
  });
  readColors(st);
  container.innerHTML = skeleton();
  bindStatic(st);
  setHelp(st);
  setShortcuts(st);
  st.onVis = () => { if (document.visibilityState === "hidden" && st.saver) st.saver.flush(); };
  document.addEventListener("visibilitychange", st.onVis);

  let list;
  try {
    list = await ctx.api.get("/api/matches?view=goals");
  } catch (e) {
    container.querySelector(".gl-body").innerHTML = `<div class="error-box">No se pudo obtener la lista de partidos: ${ctx.escapeHtml(e.message)}</div>`;
    return;
  }
  if (S !== st) return;
  st.matches = Array.isArray(list) ? list : [];
  if (!st.matches.length) {
    container.querySelector(".gl-body").innerHTML = `<div class="empty">No hay partidos configurados para anotar goles.</div>`;
    return;
  }
  const last = lsGet(LS.lastMatch);
  const id = st.matches.some((m) => m.id === last) ? last : st.matches[0].id;
  renderChips(st);
  loadOtherProgress(st);
  await openMatch(st, id);
}

export async function unmount() {
  const st = S;
  if (!st) return;
  S = null;
  document.removeEventListener("visibilitychange", st.onVis);
  st.ctx.keys.unbind("goles");
  clearTimeout(st.sbTimer);
  if (st.saver) {
    const sv = st.saver;
    sv.detach();
    await withTimeout(sv.flush(), 3000);
  }
  if (st.fv) st.fv.destroy();
  if (st.tl) st.tl.destroy();
}

function readColors(st) {
  const cs = getComputedStyle(document.documentElement);
  const v = (n, d) => (cs.getPropertyValue(n).trim() || d);
  st.colors = {
    start: v("--ok", "#16a34a"), end: v("--danger", "#dc2626"), score: v("--primary", "#2563eb"),
    hint: v("--hint", "#0ea5e9"), cut: v("--cut", "#e11d48"),
  };
}

function skeleton() {
  const fieldHtml = FIELDS.map((f, i) => `
    <div class="gl-field" data-field="${f.key}" data-cls="${f.cls}">
      <div class="gl-field-head">
        <span class="gl-field-num">${i + 1}</span>
        <div class="gl-field-titles"><div class="gl-field-title">${f.title}</div>
          <div class="gl-field-desc">${f.desc}</div></div>
      </div>
      <div class="gl-field-row">
        <div class="gl-field-val" data-val>—</div>
        <div class="gl-field-actions">
          <button class="btn btn-sm gl-mark" data-act="mark" data-k="${f.key}" title="Marcar el fotograma actual como «${f.short}» (tecla ${f.key1})">📍 Marcar aquí <kbd>${f.key1}</kbd></button>
          <button class="btn btn-sm" data-act="go" data-k="${f.key}" title="Ir al fotograma marcado">↪ Ir</button>
          <button class="btn btn-sm btn-ghost gl-clear" data-act="clear" data-k="${f.key}" title="Borrar este valor">✕</button>
        </div>
      </div>
    </div>`).join("");

  return `
  <div class="gl">
    <section class="card gl-bar">
      <div class="gl-chips" data-chips><span class="muted">Cargando partidos…</span></div>
      <div class="gl-bar-right">
        <div class="gl-progress" data-progress>
          <div class="gl-progress-txt"><b data-pdone>0</b> de <b data-ptotal>0</b> goles completos</div>
          <div class="gl-progress-bar"><span data-pbar></span></div>
        </div>
        <button class="btn gl-complete" data-act="complete" title="Indica que ya has revisado todos los goles de este partido">☐ Marcar partido como revisado</button>
      </div>
    </section>

    <div class="gl-body">
      <aside class="card gl-list">
        <div class="card-head"><h3>Goles</h3><span class="grow"></span>
          <span class="badge" data-count>—</span></div>
        <div class="gl-goals" data-goals></div>
        <div class="gl-list-foot">
          <div class="gl-list-nav">
            <button class="btn btn-sm" data-act="prevgoal" title="Gol anterior (P)">Anterior <kbd>P</kbd></button>
            <button class="btn btn-sm" data-act="nextgoal" title="Gol siguiente (N)">Siguiente <kbd>N</kbd></button>
          </div>
          <button class="btn btn-sm" data-act="add" title="Añade un gol que no aparece en la lista, en la posición actual del vídeo">＋ Añadir gol</button>
          <button class="btn btn-sm btn-danger" data-act="del" title="Elimina el gol seleccionado (pide confirmación)">🗑 Eliminar gol</button>
        </div>
      </aside>

      <section class="gl-main">
        <div class="card gl-viewer">
          <div class="gl-viewer-head">
            <span class="gl-cut-lab" title="Cambios de plano detectados automáticamente: solo son una ayuda">✂ Cortes sugeridos</span>
            <button class="btn btn-sm" data-act="prevcut" title="Ir al corte sugerido anterior ([)">⟨ Anterior <kbd>[</kbd></button>
            <button class="btn btn-sm" data-act="nextcut" title="Ir al corte sugerido siguiente (])">Siguiente <kbd>]</kbd> ⟩</button>
            <div class="gl-cutinfo" data-cutinfo></div>
            <button class="btn btn-sm" data-act="hint" title="Vuelve al momento del gol detectado automáticamente (${LEAD_SECONDS} s antes del cambio de marcador)">🎯 Ir al gol detectado</button>
          </div>
          <div class="gl-fv" data-fv></div>
        </div>
        <section class="card gl-tl">
          <div class="gl-tl-head">
            <h3>Línea de tiempo del partido</h3>
            <div class="gl-legend">
              <span><i class="lg lg-hint"></i>Gol detectado (pista)</span>
              <span><i class="lg lg-start"></i>Inicio del plano</span>
              <span><i class="lg lg-end"></i>Fin del plano</span>
              <span><i class="lg lg-score"></i>Cambio de marcador</span>
              <span><i class="lg lg-cut"></i>Cortes sugeridos</span>
            </div>
            <span class="grow"></span>
            <button class="btn btn-sm" data-act="zoomgoal" title="Acercar la línea de tiempo al gol seleccionado">🔍 Acercar al gol</button>
            <button class="btn btn-sm" data-act="zoomall" title="Ver el partido completo">↔ Partido completo</button>
          </div>
          <div data-tl></div>
        </section>
      </section>

      <aside class="gl-side">
        <div class="card gl-score">
          <div class="card-head"><h3>Marcador ampliado</h3></div>
          <div class="gl-score-body" data-sb>
            <div class="gl-sb-row"><span class="gl-sb-lab">Anterior</span><div class="gl-sb-img"><img data-sbprev alt=""></div></div>
            <div class="gl-sb-row cur"><span class="gl-sb-lab">Actual</span><div class="gl-sb-img"><img data-sbcur alt=""></div></div>
          </div>
        </div>
        <div class="card gl-panel">
          <div class="card-head"><h3 data-ptitle>Gol</h3><span class="grow"></span>
            <span data-pstatus></span></div>
          <div class="gl-fields">${fieldHtml}</div>
          <div class="gl-derived" data-derived></div>
          <div class="gl-warns" data-warns></div>
          <label class="field gl-notes">Notas (opcional)
            <textarea class="input" data-notes rows="2" placeholder="Cualquier duda o comentario sobre este gol…"></textarea></label>
          <div class="gl-panel-foot">
            <button class="btn btn-sm" data-act="undo" title="Deshacer el último cambio (Ctrl+Z)">↶ Deshacer <kbd>Ctrl+Z</kbd></button>
          </div>
        </div>
      </aside>
    </div>

  </div>`;
}

function $(st, sel) { return st.container.querySelector(sel); }

function bindStatic(st) {
  const c = st.container;
  c.addEventListener("click", (e) => {
    const b = e.target.closest("[data-act]");
    if (b && !b.disabled) {
      const act = b.dataset.act, k = b.dataset.k;
      if (act === "mark") mark(st, k);
      else if (act === "go") goField(st, k);
      else if (act === "clear") clearField(st, k);
      else if (act === "prevgoal") stepGoal(st, -1);
      else if (act === "nextgoal") stepGoal(st, 1);
      else if (act === "add") addGoal(st);
      else if (act === "del") deleteGoal(st);
      else if (act === "hint") goHint(st);
      else if (act === "prevcut") jumpCut(st, -1);
      else if (act === "nextcut") jumpCut(st, 1);
      else if (act === "complete") toggleComplete(st);
      else if (act === "undo") undo(st);
      else if (act === "zoomgoal") zoomGoal(st);
      else if (act === "zoomall") st.tl && st.tl.setRange(0, st.total);
      else if (act === "match") { if (b.dataset.id !== st.matchId) openMatch(st, b.dataset.id); }
    }
    const card = e.target.closest("[data-goal]");
    if (card) selectGoal(st, card.dataset.goal);
  });
  // Tras un clic con el ratón, quitar el foco del botón para que Espacio/Intro no lo repitan.
  c.addEventListener("mouseup", (e) => {
    const b = e.target.closest("button, select");
    if (b && b.tagName === "BUTTON") setTimeout(() => b.blur(), 0);
  });
  const notes = $(st, "[data-notes]");
  notes.addEventListener("focus", () => { st.notesUndoPushed = false; });
  notes.addEventListener("input", () => {
    const g = selGoal(st);
    if (!g) return;
    if (!st.notesUndoPushed) { pushUndo(st); st.notesUndoPushed = true; }
    g.notes = notes.value;
    st.saver.changed();
    renderGoalList(st);
  });
  notes.addEventListener("keydown", (e) => { if (e.key === "Escape") notes.blur(); });
  notes.addEventListener("blur", () => { if (st.saver) st.saver.flush(); });
}

// ------------------------------------------------------------------ partidos
function matchTitle(m) {
  return `${m.home || ""} ${m.score || ""} ${m.away || ""}`.trim() || m.label || m.id;
}
function renderChips(st) {
  const box = $(st, "[data-chips]");
  box.innerHTML = `<span class="gl-chips-lab">Partido</span>` + st.matches.map((m) => {
    const p = m.id === st.matchId && st.saver ? progressOf(st.saver.doc) : st.otherProgress[m.id];
    const prog = p ? (p.reviewed ? `<span class="badge ok">✓ Revisado</span>`
      : p.total ? `<span class="badge ${p.done === p.total ? "ok" : p.done ? "warn" : ""}">${p.done}/${p.total}</span>`
        : `<span class="badge">sin empezar</span>`) : "";
    return `<button class="gl-chip${m.id === st.matchId ? " active" : ""}" data-act="match" data-id="${st.ctx.escapeHtml(m.id)}"
        title="${st.ctx.escapeHtml(m.label || "")}">
        <span class="gl-chip-id">${st.ctx.escapeHtml(m.id)}</span>
        <span class="gl-chip-txt"><span class="gl-chip-title">${st.ctx.escapeHtml(matchTitle(m))}</span>
        <span class="gl-chip-sub">${st.ctx.escapeHtml([m.league, m.season].filter(Boolean).join(" · "))}</span></span>
        ${prog}</button>`;
  }).join("");
}
function progressOf(doc) {
  const goals = doc.goals || [];
  return { done: goals.filter((g) => goalStatus(g) === "complete").length, total: goals.length, reviewed: !!doc.complete };
}
async function loadOtherProgress(st) {
  for (const m of st.matches) {
    try {
      const d = await st.ctx.api.get(`/api/goals/${enc(m.id)}`);
      if (S !== st) return;
      st.otherProgress[m.id] = progressOf(normalizeDoc(d, m.id));
    } catch { /* sin datos todavía */ }
  }
  if (S === st) renderChips(st);
}

async function openMatch(st, id) {
  if (st.loading) return;
  st.loading = true;
  const { ctx } = st;
  try {
    if (st.saver) {
      const old = st.saver;
      await withTimeout(old.flush(), 5000);
      if (old.dirty) ctx.toast(`Los cambios de ${old.id} se guardarán en cuanto vuelva la conexión (hay copia en este navegador).`, "warn", 7000);
      st.otherProgress[old.id] = progressOf(old.doc);
      old.detach();
      st.saver = null;
    }
    st.matchId = id;
    lsSet(LS.lastMatch, id);
    renderChips(st);
    $(st, ".gl-body").classList.add("is-loading");

    let info, raw;
    try {
      info = await ctx.api.get(`/api/match/${enc(id)}`);
    } catch (e) {
      ctx.toast(`No se pudo abrir el partido ${id}: ${e.message}`, "error", 8000);
      return;
    }
    try {
      raw = await ctx.api.get(`/api/goals/${enc(id)}`);
    } catch (e) {
      if (e.status === 404) raw = null;
      else { ctx.toast(`No se pudieron leer las anotaciones de ${id}: ${e.message}`, "error", 8000); return; }
    }
    if (S !== st) return;
    st.info = info;
    st.hints = info.hints || {};
    st.fps = info.fps || 25;
    st.total = info.total_frames || 1;
    st.cuts = (st.hints.shot_starts || []).filter(isNum).slice().sort((a, b) => a - b);
    st.events = (st.hints.score_events || []).filter((e) => isNum(e.frame)).slice().sort((a, b) => a.frame - b.frame);

    let doc = normalizeDoc(raw, id);
    let pendingSave = false;

    // ¿Cambios locales sin confirmar de una sesión anterior?
    const bk = lsJson(LS.backup(id));
    if (bk && bk.doc) {
      const base = isNum(bk.baseRevision) ? bk.baseRevision : -1;
      if (base >= doc.revision && contentKey(normalizeDoc(bk.doc, id)) !== contentKey(doc)) {
        const when = new Date(bk.ts || Date.now()).toLocaleString("es-ES", { dateStyle: "short", timeStyle: "short" });
        const ok = await ctx.confirmDialog(
          "Hay cambios sin guardar",
          `En este navegador quedaron cambios de ${id} (del ${when}) que no llegaron a guardarse en el servidor. ¿Quieres recuperarlos?`,
          { ok: "Recuperar mis cambios", cancel: "Descartarlos" });
        if (S !== st) return;
        if (ok) { doc = normalizeDoc({ ...bk.doc, revision: doc.revision }, id); pendingSave = true; }
        else lsDel(LS.backup(id));
      } else lsDel(LS.backup(id));
    }

    // Primera vez: un gol por cada cambio de marcador detectado por el pipeline.
    if (!doc.goals.length && doc.revision === 0 && st.events.length) {
      doc.goals = st.events.map((ev, i) => normalizeGoal({ id: `g${i + 1}`, hint_frame: ev.frame }, i));
      pendingSave = true;
    }

    st.saver = new Saver(st, id, doc);
    st.undo = [];
    if (pendingSave) st.saver.changed();

    buildViewer(st);
    const lastGoal = lsGet(LS.lastGoal(id));
    const goals = ordered(st);
    const g = goals.find((x) => x.id === lastGoal) || goals.find((x) => goalStatus(x) !== "complete") || goals[0];
    st.selId = null;
    renderAll(st);
    if (g) selectGoal(st, g.id, { save: false });
    else { st.fv.goTo(0); st.tl.setRange(0, st.total); }
    setupScoreboard(st);
  } finally {
    st.loading = false;
    if (S === st) $(st, ".gl-body").classList.remove("is-loading");
  }
}

function buildViewer(st) {
  const { ctx } = st;
  if (st.fv) st.fv.destroy();
  if (st.tl) st.tl.destroy();
  const fvEl = $(st, "[data-fv]");
  st.fv = new ctx.FrameView(fvEl, { matchId: st.matchId, totalFrames: st.total, fps: st.fps, width: 960, stripRadius: 4, stripWidth: 160 });
  st.fv.setMarkers((n) => markersAt(st, n));
  st.fv.on("change", (n) => onFrame(st, n));
  st.tl = new ctx.Timeline($(st, "[data-tl]"), { total: st.total, fps: st.fps, height: 72 });
  st.tl.on("seek", (n) => { st.fv.pause(); st.fv.goTo(n); });
  const map = {
    ...st.fv.standardKeys(),
    s: () => mark(st, "start_frame"),
    e: () => mark(st, "end_frame_shot"),
    m: () => mark(st, "end_frame_scoreboard"),
    "[": () => jumpCut(st, -1), "]": () => jumpCut(st, 1),
    // En teclados españoles «[» y «]» se escriben con AltGr (que el navegador puede ver como Ctrl+Alt).
    "Ctrl+[": () => jumpCut(st, -1), "Ctrl+]": () => jumpCut(st, 1),
    n: () => stepGoal(st, 1), p: () => stepGoal(st, -1),
    "Ctrl+z": () => undo(st),
  };
  ctx.keys.bind("goles", map);
}

// ------------------------------------------------------------------ goles
function doc(st) { return st.saver.doc; }
function ordered(st) {
  return doc(st).goals.map((g, i) => [g, i]).sort((a, b) => sortKey(a[0]) - sortKey(b[0]) || a[1] - b[1]).map((x) => x[0]);
}
function selGoal(st) { return st.saver ? doc(st).goals.find((g) => g.id === st.selId) || null : null; }

function hintInfo(st, g) {
  if (!isNum(g.hint_frame)) return null;
  const idx = st.events.findIndex((e) => e.frame === g.hint_frame);
  if (idx < 0) return { frame: g.hint_frame, before: null, after: null, team: null };
  const ev = st.events[idx], prev = st.events[idx - 1];
  const before = prev ? `${prev.home}-${prev.away}` : "0-0";
  const team = ev.team === "home" ? st.info.home : ev.team === "away" ? st.info.away : null;
  return { frame: ev.frame, before, after: `${ev.home}-${ev.away}`, team };
}
function goalLabel(st, g, num) {
  const h = hintInfo(st, g);
  if (h && h.after) return { title: `Gol ${num}`, score: `${h.before} → ${h.after}`, sub: [st.ctx.fmt.clock(h.frame, st.fps), h.team].filter(Boolean).join(" · ") };
  const f = isNum(g.start_frame) ? g.start_frame : isNum(g.added_at_frame) ? g.added_at_frame : null;
  return { title: `Gol ${num}`, score: "añadido a mano", sub: f !== null ? st.ctx.fmt.clock(f, st.fps) : "sin posición" };
}

function selectGoal(st, id, { save = true } = {}) {
  const g = doc(st).goals.find((x) => x.id === id);
  if (!g) return;
  if (save && st.selId !== id) st.saver.flush();
  const changed = st.selId !== id;
  st.selId = id;
  lsSet(LS.lastGoal(st.matchId), id);
  if (changed) {
    st.fv.pause();
    st.fv.goTo(goalAnchor(st, g));
    zoomGoal(st);
  }
  renderAll(st);
  const card = $(st, `[data-goal="${CSS.escape(id)}"]`);
  if (card) card.scrollIntoView({ block: "nearest" });
}
function goalAnchor(st, g) {
  if (isNum(g.start_frame)) return g.start_frame;
  if (isNum(g.hint_frame)) return Math.max(0, g.hint_frame - LEAD_SECONDS * st.fps);
  if (isNum(g.added_at_frame)) return g.added_at_frame;
  if (isNum(g.end_frame_shot)) return g.end_frame_shot;
  if (isNum(g.end_frame_scoreboard)) return g.end_frame_scoreboard;
  return st.fv.current;
}
function zoomGoal(st) {
  const g = selGoal(st);
  if (!g || !st.tl) return;
  const fr = [g.start_frame, g.end_frame_shot, g.end_frame_scoreboard, g.hint_frame, g.added_at_frame].filter(isNum);
  if (!fr.length) fr.push(st.fv.current);
  const pad = 45 * st.fps;
  st.tl.setRange(Math.min(...fr) - pad, Math.max(...fr) + pad);
  st.tl.setCursor(st.fv.current);
}
function stepGoal(st, d) {
  const goals = ordered(st);
  if (!goals.length) return;
  const i = goals.findIndex((g) => g.id === st.selId);
  const j = i < 0 ? 0 : i + d;
  if (j < 0 || j >= goals.length) { st.ctx.toast(d > 0 ? "Ya estás en el último gol." : "Ya estás en el primer gol.", "info", 1800); return; }
  selectGoal(st, goals[j].id);
}
function goHint(st) {
  const g = selGoal(st);
  if (!g) return;
  if (!isNum(g.hint_frame)) { st.ctx.toast("Este gol se añadió a mano: no tiene gol detectado.", "info"); return; }
  st.fv.pause();
  st.fv.goTo(Math.max(0, g.hint_frame - LEAD_SECONDS * st.fps));
}

function pushUndo(st) {
  st.undo.push(JSON.stringify({ goals: doc(st).goals, complete: doc(st).complete, sel: st.selId }));
  if (st.undo.length > 200) st.undo.shift();
}
function undo(st) {
  if (!st.saver) return;
  const s = st.undo.pop();
  if (!s) { st.ctx.toast("No hay nada que deshacer.", "info", 1800); return; }
  const u = JSON.parse(s);
  doc(st).goals = u.goals;
  doc(st).complete = u.complete;
  if (u.sel && doc(st).goals.some((g) => g.id === u.sel)) st.selId = u.sel;
  st.saver.changed();
  renderAll(st);
  st.ctx.toast("Cambio deshecho.", "info", 1800);
}

function mark(st, key) {
  const { ctx } = st;
  const g = selGoal(st);
  if (!g) { ctx.toast("Primero selecciona un gol en la lista de la izquierda.", "warn"); return; }
  const n = st.fv.current, f = FIELD[key];
  if (g[key] === n) { ctx.toast(`«${f.short}» ya estaba marcado en este fotograma.`, "info", 1800); return; }
  if (key === "end_frame_shot" && isNum(g.start_frame) && n < g.start_frame) {
    ctx.toast(`El fin del plano no puede ser anterior al inicio (${ctx.fmt.time(g.start_frame, st.fps)}). Avanza hasta el final del plano o, si el inicio está mal, vuelve a marcarlo primero (S).`, "error", 7000);
    flashField(st, key);
    return;
  }
  pushUndo(st);
  let extra = "";
  if (key === "start_frame" && isNum(g.end_frame_shot) && n > g.end_frame_shot) {
    g.end_frame_shot = null;
    extra = " Se ha borrado el fin del plano porque quedaba antes del nuevo inicio: vuelve a marcarlo.";
  }
  g[key] = n;
  st.saver.changed();
  renderAll(st);
  flashField(st, key);
  ctx.toast(`${f.short}: ${ctx.fmt.time(n, st.fps)} (fotograma ${ctx.fmt.frame(n)}).${extra}`, extra ? "warn" : "ok", extra ? 6000 : 2200);
}
function clearField(st, key) {
  const g = selGoal(st);
  if (!g || !isNum(g[key])) return;
  pushUndo(st);
  g[key] = null;
  st.saver.changed();
  renderAll(st);
  st.ctx.toast(`«${FIELD[key].short}» borrado. Puedes deshacerlo con Ctrl+Z.`, "info", 2500);
}
function goField(st, key) {
  const g = selGoal(st);
  if (!g || !isNum(g[key])) { st.ctx.toast("Ese valor aún no está marcado.", "info", 1800); return; }
  st.fv.pause();
  st.fv.goTo(g[key]);
}
function flashField(st, key) {
  const el = $(st, `.gl-field[data-field="${key}"]`);
  if (!el) return;
  el.classList.remove("flash");
  void el.offsetWidth;
  el.classList.add("flash");
}

function addGoal(st) {
  const d = doc(st);
  let k = d.goals.length + 1;
  while (d.goals.some((g) => g.id === `g${k}`)) k++;
  pushUndo(st);
  const g = normalizeGoal({ id: `g${k}`, added_at_frame: st.fv.current }, k - 1);
  d.goals.push(g);
  st.saver.changed();
  st.selId = null;
  selectGoal(st, g.id);
  st.ctx.toast("Gol añadido en la posición actual. Marca ahora sus tres fotogramas.", "ok", 4000);
}
async function deleteGoal(st) {
  const g = selGoal(st);
  if (!g) { st.ctx.toast("Selecciona primero el gol que quieres eliminar.", "warn"); return; }
  const goals = ordered(st);
  const i = goals.findIndex((x) => x.id === g.id);
  const lab = goalLabel(st, g, i + 1);
  const ok = await st.ctx.confirmDialog(
    "¿Eliminar este gol?",
    `Se eliminará «${lab.title} · ${lab.score}» con todo lo marcado en él. Úsalo solo si no es realmente un gol (por ejemplo, una detección errónea). Puedes deshacerlo con Ctrl+Z.`,
    { ok: "Eliminar gol", cancel: "Cancelar", danger: true });
  if (!ok || S !== st || !st.saver) return;
  pushUndo(st);
  doc(st).goals = doc(st).goals.filter((x) => x.id !== g.id);
  st.saver.changed();
  const rest = ordered(st);
  st.selId = null;
  if (rest.length) selectGoal(st, rest[Math.min(i, rest.length - 1)].id);
  else renderAll(st);
  st.ctx.toast("Gol eliminado.", "info");
}
async function toggleComplete(st) {
  const d = doc(st);
  if (!d.complete) {
    const p = progressOf(d);
    if (p.done < p.total) {
      const ok = await st.ctx.confirmDialog("Aún faltan goles por completar",
        `Solo ${p.done} de ${p.total} goles tienen los tres fotogramas marcados. ¿Marcar el partido como revisado de todas formas?`,
        { ok: "Sí, marcar como revisado", cancel: "Cancelar" });
      if (!ok || S !== st) return;
    }
  }
  pushUndo(st);
  d.complete = !d.complete;
  st.saver.changed();
  st.saver.flush();
  renderAll(st);
  st.ctx.toast(d.complete ? "Partido marcado como revisado. ¡Gracias!" : "El partido vuelve a estar pendiente de revisión.", d.complete ? "ok" : "info");
}

function jumpCut(st, dir) {
  const n = st.fv.current, c = st.cuts;
  if (!c.length) { st.ctx.toast("No hay cortes sugeridos para este partido.", "info"); return; }
  let target = null;
  if (dir > 0) { const i = upperBound(c, n); if (i < c.length) target = c[i]; }
  else { const i = lowerBound(c, n) - 1; if (i >= 0) target = c[i]; }
  if (target === null) { st.ctx.toast(dir > 0 ? "No hay más cortes sugeridos después." : "No hay cortes sugeridos antes.", "info", 1800); return; }
  st.fv.pause();
  st.fv.goTo(target);
}
function lowerBound(a, x) { let lo = 0, hi = a.length; while (lo < hi) { const m = (lo + hi) >> 1; if (a[m] < x) lo = m + 1; else hi = m; } return lo; }
function upperBound(a, x) { let lo = 0, hi = a.length; while (lo < hi) { const m = (lo + hi) >> 1; if (a[m] <= x) lo = m + 1; else hi = m; } return lo; }

// ------------------------------------------------------------------ eventos
function onFrame(st, n) {
  if (st.tl) st.tl.setCursor(n);
  updateHere(st, n);
  updateCutInfo(st, n);
  requestScoreboard(st, n);
}
function onSaved(st) {
  if (S !== st) return;
  renderChips(st);
}
function onDocReplaced(st) {
  if (S !== st) return;
  st.undo = [];
  const goals = ordered(st);
  if (!goals.some((g) => g.id === st.selId)) st.selId = goals.length ? goals[0].id : null;
  renderAll(st);
}

function markersAt(st, n) {
  const out = [];
  const g = selGoal(st);
  if (g) {
    for (const f of FIELDS) if (g[f.key] === n) out.push({ label: `${f.short} ✓`, cls: f.cls });
  }
  if (st.cuts.length) {
    const i = lowerBound(st.cuts, n);
    if (st.cuts[i] === n) out.push({ label: "Corte sugerido", cls: "gl-suggest" });
  }
  return out;
}

// ------------------------------------------------------------------ render
function renderAll(st) {
  if (!st.saver) return;
  renderProgress(st);
  renderChips(st);
  renderGoalList(st);
  renderPanel(st);
  renderTimeline(st);
  if (st.fv) { st.fv.refresh(); updateHere(st, st.fv.current); updateCutInfo(st, st.fv.current); }
}

function renderProgress(st) {
  const p = progressOf(doc(st));
  $(st, "[data-pdone]").textContent = p.done;
  $(st, "[data-ptotal]").textContent = p.total;
  $(st, "[data-pbar]").style.width = p.total ? `${(100 * p.done) / p.total}%` : "0";
  $(st, "[data-progress]").classList.toggle("all", p.total > 0 && p.done === p.total);
  const b = $(st, '[data-act="complete"]');
  b.classList.toggle("on", p.reviewed);
  b.innerHTML = p.reviewed ? "✓ Partido revisado" : "☐ Marcar partido como revisado";
  b.title = p.reviewed ? "Clic para volver a marcarlo como pendiente" : "Indica que ya has revisado todos los goles de este partido";
  $(st, "[data-count]").textContent = p.total === 1 ? "1 gol" : `${p.total} goles`;
}

const STATUS_BADGE = {
  pending: '<span class="badge">Pendiente</span>',
  partial: '<span class="badge warn">A medias</span>',
  complete: '<span class="badge ok">Completo ✓</span>',
};

function renderGoalList(st) {
  const { escapeHtml } = st.ctx;
  const box = $(st, "[data-goals]");
  const goals = ordered(st);
  if (!goals.length) {
    box.innerHTML = `<div class="empty small">No hay goles. Usa «＋ Añadir gol» para crear uno en la posición actual del vídeo.</div>`;
  } else {
    box.innerHTML = goals.map((g, i) => {
      const lab = goalLabel(st, g, i + 1), stt = goalStatus(g);
      const dots = FIELDS.map((f) => `<i class="dot ${f.cls}${isNum(g[f.key]) ? " on" : ""}" title="${f.short}: ${isNum(g[f.key]) ? "marcado" : "sin marcar"}"></i>`).join("");
      const tip = `${lab.title} · ${lab.score} · ${lab.sub}`;
      return `<button class="gl-goal st-${stt}${g.id === st.selId ? " sel" : ""}" data-goal="${escapeHtml(g.id)}" title="${escapeHtml(tip)}">
          <span class="gl-goal-num">${i + 1}</span>
          <span class="gl-goal-score">${escapeHtml(lab.score)}</span>
          <span class="gl-dots">${dots}</span>
          <span class="gl-goal-row2"><span class="gl-goal-sub">${escapeHtml(lab.sub)}${g.notes ? ' · <span title="Tiene notas">📝</span>' : ""}</span>
            ${STATUS_BADGE[stt]}</span>
        </button>`;
    }).join("");
  }
  $(st, '[data-act="del"]').disabled = !selGoal(st);
  const i = goals.findIndex((g) => g.id === st.selId);
  $(st, '[data-act="prevgoal"]').disabled = i <= 0;
  $(st, '[data-act="nextgoal"]').disabled = i < 0 || i >= goals.length - 1;
}

function renderPanel(st) {
  const { fmt, escapeHtml } = st.ctx;
  const g = selGoal(st);
  const panel = $(st, ".gl-panel");
  panel.classList.toggle("disabled", !g);
  const goals = ordered(st);
  const idx = g ? goals.findIndex((x) => x.id === g.id) : -1;
  const lab = g ? goalLabel(st, g, idx + 1) : null;
  $(st, "[data-ptitle]").textContent = g ? `${lab.title} · ${lab.score}` : "Ningún gol seleccionado";
  $(st, "[data-pstatus]").innerHTML = g ? STATUS_BADGE[goalStatus(g)] : "";
  $(st, '[data-act="hint"]').disabled = !g || !isNum(g.hint_frame);

  for (const f of FIELDS) {
    const el = $(st, `.gl-field[data-field="${f.key}"]`);
    const v = g ? g[f.key] : null;
    el.classList.toggle("set", isNum(v));
    el.querySelector("[data-val]").innerHTML = isNum(v)
      ? `<span class="mono gl-t">${fmt.time(v, st.fps)}</span><span class="mono muted gl-n">#${fmt.frame(v)}</span>`
      : `<span class="muted">Sin marcar</span>`;
    el.querySelectorAll("button").forEach((b) => {
      b.disabled = !g || (b.dataset.act !== "mark" && !isNum(v));
    });
  }

  // Duraciones derivadas
  const der = $(st, "[data-derived]");
  if (g && isNum(g.start_frame) && isNum(g.end_frame_shot) && g.end_frame_shot >= g.start_frame) {
    const parts = [`Plano del gol: <b>${secs(g.end_frame_shot - g.start_frame + 1, st.fps)} s</b>`];
    if (isNum(g.end_frame_scoreboard)) {
      const d = g.end_frame_scoreboard - g.end_frame_shot;
      parts.push(`marcador ${d >= 0 ? "+" : "−"}${secs(Math.abs(d), st.fps)} s ${d >= 0 ? "después del" : "antes del"} fin del plano`);
    }
    der.innerHTML = parts.join(" · ");
  } else der.innerHTML = "";

  // Avisos
  const w = [];
  if (g) {
    const s = g.start_frame, e = g.end_frame_shot, m = g.end_frame_scoreboard;
    if (isNum(s) && isNum(e) && e < s) w.push(["error", "El fin del plano es anterior al inicio. Vuelve a marcar uno de los dos."]);
    if (isNum(m) && isNum(e) && m < e) w.push(["warn", "El marcador cambia antes del fin del plano. Es poco habitual: compruébalo (puede ser correcto)."]);
    else if (isNum(m) && isNum(s) && m < s) w.push(["warn", "El cambio de marcador es anterior al inicio del plano. Compruébalo."]);
    if (isNum(s) && isNum(e) && e - s > 90 * st.fps) w.push(["warn", `El plano dura más de 90 s (${secs(e - s + 1, st.fps)} s). ¿Seguro que no hay un corte en medio?`]);
    if (isNum(m) && isNum(e) && m - e > 180 * st.fps) w.push(["warn", "El marcador cambia más de 3 min después del fin del plano. Compruébalo."]);
    if (isNum(g.hint_frame)) {
      const ref = isNum(m) ? m : isNum(s) ? s : null;
      if (ref !== null && Math.abs(ref - g.hint_frame) > 180 * st.fps) w.push(["warn", "Lo marcado está a más de 3 min del gol detectado. ¿Es el mismo gol?"]);
    }
  }
  $(st, "[data-warns]").innerHTML = w.map(([k, t]) => `<div class="gl-warn ${k}">${k === "error" ? "⛔" : "⚠️"} ${escapeHtml(t)}</div>`).join("");

  const notes = $(st, "[data-notes]");
  notes.disabled = !g;
  if (document.activeElement !== notes || notes.dataset.goal !== (g && g.id)) {
    notes.value = g ? g.notes || "" : "";
    notes.dataset.goal = g ? g.id : "";
  }
  $(st, '[data-act="undo"]').disabled = !st.undo.length;
}

function updateHere(st, n) {
  const g = selGoal(st);
  for (const f of FIELDS) {
    const el = $(st, `.gl-field[data-field="${f.key}"]`);
    if (el) el.classList.toggle("here", !!g && g[f.key] === n);
  }
}
function updateCutInfo(st, n) {
  const el = $(st, "[data-cutinfo]");
  if (!el) return;
  const c = st.cuts;
  if (!c.length) { el.innerHTML = `<span class="muted">Sin cortes sugeridos</span>`; return; }
  const i = lowerBound(c, n);
  if (c[i] === n) { el.innerHTML = `<span class="gl-cut-here">✂ Corte sugerido aquí</span>`; return; }
  const prev = i > 0 ? n - c[i - 1] : null, next = i < c.length ? c[i] - n : null;
  const f = (d) => (d < st.fps ? `${d} fot.` : `${secs(d, st.fps)} s`);
  el.innerHTML = `<span class="muted" title="Distancia al corte sugerido anterior">◂ ${prev !== null ? f(prev) : "—"}</span><span class="gl-cut-sep">·</span>` +
    `<span class="muted" title="Distancia al corte sugerido siguiente">${next !== null ? f(next) : "—"} ▸</span>`;
}

function renderTimeline(st) {
  if (!st.tl) return;
  const C = st.colors;
  const marks = [];
  const spans = [];
  const sel = selGoal(st);
  // Cortes sugeridos cerca del gol seleccionado (para no saturar la vista del partido entero).
  if (sel) {
    const center = goalAnchor(st, sel);
    const lo = center - 4 * 60 * st.fps, hi = center + 4 * 60 * st.fps;
    for (let i = lowerBound(st.cuts, lo); i < st.cuts.length && st.cuts[i] <= hi; i++) {
      marks.push({ frame: st.cuts[i], color: "rgba(225,29,72,.35)", width: 1 });
    }
  }
  for (const ev of st.events) marks.push({ frame: ev.frame, cls: "hint", color: C.hint, width: 2, label: `${ev.home}-${ev.away}` });
  for (const g of doc(st).goals) {
    const isSel = sel && g.id === sel.id;
    if (isNum(g.start_frame) && isNum(g.end_frame_shot) && g.end_frame_shot >= g.start_frame) {
      spans.push({ start: g.start_frame, end: g.end_frame_shot, color: isSel ? C.start : C.score, alpha: isSel ? 0.28 : 0.18 });
    }
    for (const f of FIELDS) if (isNum(g[f.key])) marks.push({ frame: g[f.key], color: C[f.cls], width: isSel ? 3 : 2 });
  }
  st.tl.setSpans(spans);
  st.tl.setMarks(marks);
}

// ------------------------------------------------------------------ marcador ampliado
function setupScoreboard(st) {
  const box = $(st, "[data-sb]");
  const has = !!(st.hints && st.hints.scoreboard_region);
  box.classList.toggle("none", !has);
  box.querySelectorAll(".gl-sb-none").forEach((x) => x.remove());
  if (!has) {
    box.insertAdjacentHTML("beforeend", `<div class="gl-sb-none muted small">Este partido no tiene detectada la región del marcador. Fíjate en el marcador directamente en el fotograma.</div>`);
    return;
  }
  st.sb.want = null;
  requestScoreboard(st, st.fv.current);
}
function sbUrl(st, n) { return `/api/scoreboard/${enc(st.matchId)}/${n}?scale=2`; }
function preload(url) {
  return new Promise((res, rej) => { const im = new Image(); im.onload = () => res(url); im.onerror = rej; im.src = url; });
}
function requestScoreboard(st, n) {
  if (!st.hints || !st.hints.scoreboard_region) return;
  st.sb.want = n;
  if (st.sb.busy) return;
  st.sb.busy = true;
  clearTimeout(st.sbTimer);
  st.sbTimer = setTimeout(() => loadScoreboard(st), 90);
}
async function loadScoreboard(st) {
  const n = st.sb.want, mid = st.matchId;
  const prevN = Math.max(0, n - 1);
  const box = $(st, "[data-sb]");
  try {
    const [uc, up] = await Promise.all([preload(sbUrl(st, n)), n > 0 ? preload(sbUrl(st, prevN)) : Promise.resolve(null)]);
    if (S !== st || st.matchId !== mid) return;
    box.querySelector("[data-sbcur]").src = uc;
    const pi = box.querySelector("[data-sbprev]");
    if (up) pi.src = up; else pi.removeAttribute("src");
    box.querySelector(".gl-sb-row.cur .gl-sb-lab").textContent = `Actual #${st.ctx.fmt.frame(n)}`;
    box.querySelector(".gl-sb-row:not(.cur) .gl-sb-lab").textContent = `Anterior #${st.ctx.fmt.frame(prevN)}`;
    box.classList.remove("err");
  } catch {
    if (S === st) box.classList.add("err");
  } finally {
    if (S === st) {
      st.sb.busy = false;
      if (st.sb.want !== n && st.matchId === mid) requestScoreboard(st, st.sb.want);
    }
  }
}

// ------------------------------------------------------------------ ayuda y atajos
function setShortcuts(st) {
  st.ctx.shortcuts.set([
    ["S", "Inicio del plano"], ["E", "Fin del plano"], ["M", "Cambio de marcador"],
    ["[ ]", "Corte sugerido ant./sig."], ["P N", "Gol ant./sig."],
    ["← →", "±1 fotograma"], ["Mayús+←/→", "±1 s"], ["Ctrl+←/→", "±10 s"],
    ["Espacio", "Reproducir/pausa"], ["G", "Ir a…"], ["Ctrl+Z", "Deshacer"], ["?", "Ayuda"],
  ]);
}
function setHelp(st) {
  st.ctx.helpPanel.set(`
    <p>En esta pestaña se anotan los goles de dos partidos de otras ligas
    (<b>Tottenham 5-3 Chelsea</b> y <b>Inter 4-4 Juventus</b>) exactamente igual que en la anotación de referencia de los
    25 partidos de LaLiga. Para cada gol hay que marcar <b>tres fotogramas</b>:</p>
    <ol class="gl-help-defs">
      <li><b class="c-start">Inicio del plano del gol</b> <kbd>S</kbd><br>
        El <b>primer fotograma</b> del plano (de la cámara) en el que se juega la jugada del gol, justo
        <b>después del corte</b> o transición que lo precede. Es decir, el primer fotograma de la toma en directo en la que el balón acaba entrando en la portería.</li>
      <li><b class="c-end">Fin del plano del gol</b> <kbd>E</kbd><br>
        El <b>último fotograma de ese mismo plano</b>, justo antes del corte (o transición) a la repetición, la celebración,
        un primer plano, etc. Truco: ve al primer fotograma del plano siguiente y retrocede uno con <kbd>←</kbd>.</li>
      <li><b class="c-score">Cambio en el marcador</b> <kbd>M</kbd><br>
        El <b>primer fotograma</b> en el que el marcador de la pantalla muestra ya el <b>nuevo resultado</b>. Suele
        llegar unos segundos después del fin del plano (a veces cuando el marcador reaparece tras la repetición).
        Usa el recuadro «Marcador ampliado» (fotograma anterior y actual) para ver el cambio de dígito.</li>
    </ol>

    <h3>Cómo trabajar</h3>
    <ol>
      <li>Elige el partido arriba y un gol en la lista de la izquierda. El vídeo se coloca unos ${LEAD_SECONDS} s antes del
        cambio de marcador detectado automáticamente.</li>
      <li>Busca el comienzo del plano del gol: reproduce (<kbd>Espacio</kbd>) o avanza/retrocede con los botones o las
        flechas, y ajusta fotograma a fotograma con <kbd>←</kbd> <kbd>→</kbd> o con la tira de miniaturas.</li>
      <li>En el fotograma exacto, pulsa «📍 Marcar aquí» (o la tecla). Repite para el fin del plano y el marcador.</li>
      <li>Cuando el gol tenga los tres valores aparece como <span class="badge ok">Completo ✓</span>. Pasa al siguiente
        con <kbd>N</kbd>.</li>
      <li>Al terminar todos, pulsa «Marcar partido como revisado».</li>
    </ol>

    <h3>Cortes sugeridos</h3>
    <p>Los botones «Corte sugerido anterior / siguiente» (<kbd>[</kbd> <kbd>]</kbd>) saltan a los cambios de plano que
    detectó el sistema automático (aparecen en rosa en la línea de tiempo). Son <b>solo una ayuda</b> para llegar antes:
    pueden faltar o sobrar cortes, y el fotograma correcto lo decides tú. En teclados españoles, si
    <kbd>[</kbd> <kbd>]</kbd> no funcionan, usa los botones.</p>

    <h3>Lista de goles</h3>
    <p>La lista sale de los cambios de marcador detectados automáticamente. Si falta un gol, colócate cerca de él y pulsa
    «＋ Añadir gol». Si alguno no es un gol de verdad, selecciónalo y pulsa «🗑 Eliminar gol». Usa las <b>Notas</b> para
    cualquier duda (p. ej. «no se ve el corte», «gol anulado»).</p>

    <h3>Guardado</h3>
    <p>Todo se guarda <b>automáticamente</b> (indicador arriba a la derecha). Si se corta la conexión, los cambios se
    conservan en este navegador y se reintenta solo. Puedes deshacer el último cambio con «↶ Deshacer» (<kbd>Ctrl+Z</kbd>).
    Trabaja con <b>una sola pestaña</b> del anotador abierta.</p>

    <h3>Tiempos</h3>
    <p>Los tiempos son del <b>vídeo</b> (no del reloj del partido): <span class="mono">mm:ss.ff</span>, donde
    <span class="mono">ff</span> es el fotograma dentro del segundo (25 por segundo). También se muestra siempre el número
    de fotograma (#).</p>

    <h3>Atajos de teclado</h3>
    <table>
      <tr><td><kbd>S</kbd> / <kbd>E</kbd> / <kbd>M</kbd></td><td>Marcar inicio del plano / fin del plano / cambio de marcador</td></tr>
      <tr><td><kbd>[</kbd> / <kbd>]</kbd></td><td>Corte sugerido anterior / siguiente</td></tr>
      <tr><td><kbd>P</kbd> / <kbd>N</kbd></td><td>Gol anterior / siguiente</td></tr>
      <tr><td><kbd>←</kbd> / <kbd>→</kbd></td><td>Retroceder / avanzar 1 fotograma</td></tr>
      <tr><td><kbd>Mayús+←</kbd> / <kbd>Mayús+→</kbd></td><td>±1 segundo</td></tr>
      <tr><td><kbd>Ctrl+←</kbd> / <kbd>Ctrl+→</kbd></td><td>±10 segundos</td></tr>
      <tr><td><kbd>Re Pág</kbd> / <kbd>Av Pág</kbd></td><td>±1 minuto</td></tr>
      <tr><td><kbd>Espacio</kbd></td><td>Reproducir / pausar</td></tr>
      <tr><td><kbd>Inicio</kbd> / <kbd>Fin</kbd></td><td>Principio / final del vídeo</td></tr>
      <tr><td><kbd>G</kbd></td><td>Ir a un tiempo o fotograma concreto</td></tr>
      <tr><td><kbd>Ctrl+Z</kbd></td><td>Deshacer el último cambio</td></tr>
      <tr><td><kbd>?</kbd></td><td>Abrir / cerrar esta ayuda</td></tr>
    </table>
    <p class="muted small">Línea de tiempo: rueda del ratón = zoom, arrastrar = desplazar, clic = ir a ese punto.</p>`);
}
