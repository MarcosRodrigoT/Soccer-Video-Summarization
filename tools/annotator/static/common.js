// Armazón común del anotador: router, API, guardado, diálogos, atajos,
// visor de fotogramas (FrameView) y línea de tiempo (Timeline).

// ---------------------------------------------------------------- API
export class ApiError extends Error {
  constructor(status, body) {
    super((body && body.error) || `Error ${status}`);
    this.status = status;
    this.body = body;
  }
}

async function request(method, path, body) {
  let res;
  try {
    res = await fetch(path, {
      method,
      headers: body !== undefined ? { "Content-Type": "application/json" } : {},
      body: body !== undefined ? JSON.stringify(body) : undefined,
      cache: "no-store",
    });
  } catch (e) {
    throw new ApiError(0, { error: "No hay conexión con el servidor. Comprueba que sigue abierto." });
  }
  let data = null;
  const text = await res.text();
  try { data = text ? JSON.parse(text) : null; } catch { data = { error: text }; }
  if (!res.ok) throw new ApiError(res.status, data);
  return data;
}

export const api = {
  get: (p) => request("GET", p),
  put: (p, b) => request("PUT", p, b),
  post: (p, b) => request("POST", p, b),
  del: (p) => request("DELETE", p),
};

// ---------------------------------------------------------------- save status
const SAVE_TEXT = {
  saved: "Todo guardado", saving: "Guardando…", dirty: "Cambios sin guardar", error: "Error al guardar",
};
export const saveStatus = {
  state: "saved",
  set(state, text) {
    this.state = state;
    const el = document.getElementById("save-status");
    if (!el) return;
    el.dataset.state = state;
    el.querySelector(".txt").textContent = text || SAVE_TEXT[state] || state;
  },
};
window.addEventListener("beforeunload", (e) => {
  if (saveStatus.state === "dirty" || saveStatus.state === "saving" || saveStatus.state === "error") {
    e.preventDefault();
    e.returnValue = "";
  }
});

// ---------------------------------------------------------------- toasts & dialogs
export function toast(msg, kind = "info", ms = 3500) {
  const box = document.getElementById("toasts");
  const t = document.createElement("div");
  t.className = `toast ${kind}`;
  t.textContent = msg;
  box.appendChild(t);
  setTimeout(() => t.remove(), ms);
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
export { escapeHtml };

export function confirmDialog(title, text, { ok = "Aceptar", cancel = "Cancelar", danger = false } = {}) {
  return new Promise((resolve) => {
    const root = document.getElementById("dialog-root");
    const wrap = document.createElement("div");
    wrap.className = "dialog-backdrop";
    wrap.innerHTML = `<div class="dialog" role="dialog" aria-modal="true">
        <h3>${escapeHtml(title)}</h3><p>${escapeHtml(text)}</p>
        <div class="row"><button class="btn" data-a="0">${escapeHtml(cancel)}</button>
        <button class="btn ${danger ? "btn-danger" : "btn-primary"}" data-a="1">${escapeHtml(ok)}</button></div></div>`;
    const done = (v) => { wrap.remove(); document.removeEventListener("keydown", onKey, true); resolve(v); };
    const onKey = (e) => {
      if (e.key === "Escape") { e.stopPropagation(); e.preventDefault(); done(false); }
      if (e.key === "Enter") { e.stopPropagation(); e.preventDefault(); done(true); }
    };
    wrap.addEventListener("click", (e) => {
      const a = e.target.closest("[data-a]");
      if (a) done(a.dataset.a === "1");
      else if (e.target === wrap) done(false);
    });
    document.addEventListener("keydown", onKey, true);
    root.appendChild(wrap);
    wrap.querySelector('[data-a="1"]').focus();
  });
}

export function promptDialog(title, text, initial = "", { ok = "Aceptar", cancel = "Cancelar", placeholder = "" } = {}) {
  return new Promise((resolve) => {
    const root = document.getElementById("dialog-root");
    const wrap = document.createElement("div");
    wrap.className = "dialog-backdrop";
    wrap.innerHTML = `<div class="dialog" role="dialog" aria-modal="true">
        <h3>${escapeHtml(title)}</h3><p>${escapeHtml(text)}</p>
        <input class="input" style="width:100%" placeholder="${escapeHtml(placeholder)}" value="${escapeHtml(initial)}">
        <div class="row"><button class="btn" data-a="0">${escapeHtml(cancel)}</button>
        <button class="btn btn-primary" data-a="1">${escapeHtml(ok)}</button></div></div>`;
    const input = wrap.querySelector("input");
    const done = (v) => { wrap.remove(); resolve(v); };
    wrap.addEventListener("click", (e) => {
      const a = e.target.closest("[data-a]");
      if (a) done(a.dataset.a === "1" ? input.value : null);
      else if (e.target === wrap) done(null);
    });
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") { e.preventDefault(); done(input.value); }
      if (e.key === "Escape") { e.preventDefault(); done(null); }
      e.stopPropagation();
    });
    root.appendChild(wrap);
    input.focus();
    input.select();
  });
}

// ---------------------------------------------------------------- keyboard
// Combos: "ArrowLeft", "Shift+ArrowLeft", "Ctrl+ArrowLeft", "Alt+ArrowLeft", "s", "Delete", "PageUp", " " (espacio).
// Las letras no distinguen mayúsculas. Ctrl y Alt se tratan igual ("Ctrl+...").
const keyMaps = new Map();
export const keys = {
  bind(view, map) { keyMaps.set(view, map); },
  unbind(view) { keyMaps.delete(view); },
};
function comboOf(e) {
  let k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
  if (k === "Spacebar") k = " ";
  const mods = [];
  if (e.ctrlKey || e.metaKey || e.altKey) mods.push("Ctrl");
  if (e.shiftKey && e.key.length !== 1) mods.push("Shift");
  return [...mods, k].join("+");
}
document.addEventListener("keydown", (e) => {
  const tag = (e.target && e.target.tagName) || "";
  const typing = ["INPUT", "TEXTAREA", "SELECT"].includes(tag) || (e.target && e.target.isContentEditable);
  if (document.querySelector(".dialog-backdrop")) return;
  if (e.key === "?" && !typing) { e.preventDefault(); helpPanel.toggle(); return; }
  if (e.key === "Escape" && !document.getElementById("help-panel").hidden) { helpPanel.toggle(false); return; }
  if (typing) return;
  const combo = comboOf(e);
  for (const map of keyMaps.values()) {
    const fn = map[combo] || (e.shiftKey && e.key.length === 1 ? map[e.key.toLowerCase()] : undefined);
    if (fn) { e.preventDefault(); fn(e); return; }
  }
});

// ---------------------------------------------------------------- help panel & shortcut bar
export const helpPanel = {
  html: "",
  set(html) { this.html = html; document.getElementById("help-content").innerHTML = html; },
  toggle(force) {
    const p = document.getElementById("help-panel");
    const show = force !== undefined ? force : p.hidden;
    p.hidden = !show;
  },
};
document.getElementById("help-btn").addEventListener("click", () => helpPanel.toggle(true));
document.getElementById("help-panel").addEventListener("click", (e) => {
  if (e.target.id === "help-panel" || e.target.closest("[data-close]")) helpPanel.toggle(false);
});

export const shortcuts = {
  el: null,
  set(items) { // items: [["←/→", "±1 fotograma"], ...]
    if (!this.el) { this.el = document.createElement("div"); this.el.className = "shortcut-bar"; document.body.appendChild(this.el); }
    this.el.innerHTML = items.map(([k, l]) =>
      `<span>${k.split(" ").map((x) => `<kbd>${escapeHtml(x)}</kbd>`).join(" ")} ${escapeHtml(l)}</span>`).join("");
    this.el.hidden = items.length === 0;
  },
};

// ---------------------------------------------------------------- formatting
export const fmt = {
  time(frame, fps = 25) {
    if (frame === null || frame === undefined || Number.isNaN(frame)) return "—";
    const total = Math.floor(frame / fps);
    const ff = frame - total * fps;
    const h = Math.floor(total / 3600), m = Math.floor((total % 3600) / 60), s = total % 60;
    const mm = String(m).padStart(2, "0"), ss = String(s).padStart(2, "0"), f2 = String(ff).padStart(2, "0");
    return h > 0 ? `${h}:${mm}:${ss}.${f2}` : `${mm}:${ss}.${f2}`;
  },
  clock(frame, fps = 25) { // sin fotograma: "12:34"
    if (frame === null || frame === undefined) return "—";
    const total = Math.floor(frame / fps);
    const h = Math.floor(total / 3600), m = Math.floor((total % 3600) / 60), s = total % 60;
    return h > 0 ? `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}` : `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  },
  frame(n) {
    if (n === null || n === undefined) return "—";
    return String(n).replace(/\B(?=(\d{3})+(?!\d))/g, " ");
  },
  // "12:34", "1:02:03", "12:34.05" -> fotograma
  parseTime(str, fps = 25) {
    const s = String(str).trim();
    if (/^\d+$/.test(s)) return parseInt(s, 10) * 60 * fps; // minutos enteros
    const m = s.match(/^(?:(\d+):)?(\d{1,2}):(\d{1,2})(?:[.,](\d{1,2}))?$/);
    if (!m) return null;
    const [, h, mi, se, ff] = m;
    return ((parseInt(h || "0", 10) * 60 + parseInt(mi, 10)) * 60 + parseInt(se, 10)) * fps + parseInt(ff || "0", 10);
  },
};

// ---------------------------------------------------------------- tiny event emitter
class Emitter {
  constructor() { this._h = {}; }
  on(ev, fn) { (this._h[ev] ||= []).push(fn); return () => this.off(ev, fn); }
  off(ev, fn) { this._h[ev] = (this._h[ev] || []).filter((f) => f !== fn); }
  emit(ev, ...a) { for (const f of this._h[ev] || []) f(...a); }
}

// ---------------------------------------------------------------- image cache
const imgCache = new Map(); // url -> Promise<HTMLImageElement>
const IMG_CACHE_MAX = 600;
function loadImage(url) {
  if (imgCache.has(url)) {
    const p = imgCache.get(url);
    imgCache.delete(url); imgCache.set(url, p); // LRU touch
    return p;
  }
  const p = new Promise((resolve, reject) => {
    const im = new Image();
    im.decoding = "async";
    im.onload = () => resolve(im);
    im.onerror = () => { imgCache.delete(url); reject(new Error("No se pudo cargar " + url)); };
    im.src = url;
  });
  imgCache.set(url, p);
  while (imgCache.size > IMG_CACHE_MAX) imgCache.delete(imgCache.keys().next().value);
  return p;
}
export function frameUrl(matchId, n, w = 960) {
  return `/api/frame/${encodeURIComponent(matchId)}/${n}?w=${w}`;
}

// ---------------------------------------------------------------- FrameView
export class FrameView extends Emitter {
  constructor(el, { matchId, totalFrames, fps = 25, width = 960, stripRadius = 4, stripWidth = 160, controls = true }) {
    super();
    this.el = el;
    this.matchId = matchId;
    this.total = totalFrames;
    this.fps = fps;
    this.width = width;
    this.radius = stripRadius;
    this.stripWidth = stripWidth;
    this.min = 0;
    this.max = totalFrames - 1;
    this.current = 0;
    this.playing = false;
    this._markersFn = null;
    this._token = 0;
    el.classList.add("frameview");
    el.innerHTML = `
      <div class="fv-stage"><img alt="Fotograma"><div class="fv-hud"></div><div class="fv-markers"></div>
        <div class="fv-loading" hidden></div></div>
      ${controls ? `<div class="fv-controls">
        <button class="btn btn-sm" data-step="-1500" title="−1 minuto (Re Pág)">« 1 min</button>
        <button class="btn btn-sm" data-step="-250" title="−10 s (Ctrl+←)">« 10 s</button>
        <button class="btn btn-sm" data-step="-25" title="−1 s (Mayús+←)">‹ 1 s</button>
        <button class="btn btn-sm" data-step="-1" title="−1 fotograma (←)">‹ 1 fot.</button>
        <button class="btn btn-sm btn-primary" data-play title="Reproducir / pausar (Espacio)">▶ Reproducir</button>
        <button class="btn btn-sm" data-step="1" title="+1 fotograma (→)">1 fot. ›</button>
        <button class="btn btn-sm" data-step="25" title="+1 s (Mayús+→)">1 s ›</button>
        <button class="btn btn-sm" data-step="250" title="+10 s (Ctrl+→)">10 s »</button>
        <button class="btn btn-sm" data-step="1500" title="+1 minuto (Av Pág)">1 min »</button>
        <span class="sep"></span>
        <select class="input btn-sm" data-speed title="Velocidad de reproducción">
          <option value="0.25">×0,25</option><option value="0.5">×0,5</option>
          <option value="1" selected>×1</option><option value="2">×2</option><option value="4">×4</option></select>
        <button class="btn btn-sm" data-goto title="Ir a un fotograma o tiempo (G)">Ir a…</button>
      </div>` : ""}
      <div class="fv-strip"></div>`;
    this.img = el.querySelector(".fv-stage img");
    this.hud = el.querySelector(".fv-hud");
    this.markersEl = el.querySelector(".fv-markers");
    this.loadingEl = el.querySelector(".fv-loading");
    this.strip = el.querySelector(".fv-strip");
    this.strip.style.gridTemplateColumns = `repeat(${2 * this.radius + 1}, 1fr)`;
    this.strip.innerHTML = Array.from({ length: 2 * this.radius + 1 }, (_, i) =>
      `<div class="th${i === this.radius ? " current" : ""}" data-off="${i - this.radius}"><img alt=""><span></span></div>`).join("");
    this.strip.addEventListener("click", (e) => {
      const th = e.target.closest(".th");
      if (th) this.step(parseInt(th.dataset.off, 10));
    });
    if (controls) {
      el.querySelector(".fv-controls").addEventListener("click", (e) => {
        const b = e.target.closest("button");
        if (!b) return;
        if (b.dataset.step) this.step(parseInt(b.dataset.step, 10));
        else if (b.hasAttribute("data-play")) this.toggle();
        else if (b.hasAttribute("data-goto")) this.askGoTo();
      });
      this.speedSel = el.querySelector("[data-speed]");
    }
  }

  setBounds(min, max) {
    this.min = Math.max(0, min);
    this.max = Math.min(this.total - 1, max);
    if (this.current < this.min || this.current > this.max) this.goTo(this.current);
  }

  setMarkers(fn) { this._markersFn = fn; this._renderOverlay(); }
  // fn(f) -> clases CSS extra para la miniatura del fotograma f de la tira (opcional).
  setStripClass(fn) { this._stripClassFn = fn; if (this._shownOnce) this._updateStrip(this.current); }
  refresh() { this._renderOverlay(); }

  clamp(n) { return Math.max(this.min, Math.min(this.max, Math.round(n))); }

  goTo(n, { emit = true } = {}) {
    n = this.clamp(n);
    const changed = n !== this.current;
    this.current = n;
    this._renderOverlay();
    this._show(n);
    if (emit && (changed || !this._shownOnce)) this.emit("change", n);
    this._shownOnce = true;
  }

  step(k) { this.pause(); this.goTo(this.current + k); }

  async askGoTo() {
    const v = await promptDialog("Ir a…", "Escribe un tiempo (mm:ss, mm:ss.ff o h:mm:ss) o un número de fotograma precedido de #, p. ej. #37811.", "", { placeholder: "12:34  ·  #37811" });
    if (v === null) return;
    const s = v.trim();
    let n = null;
    if (s.startsWith("#")) n = parseInt(s.slice(1).replace(/\s/g, ""), 10);
    else n = fmt.parseTime(s, this.fps);
    if (n === null || Number.isNaN(n)) { toast("No entiendo ese tiempo. Ejemplos: 12:34, 12:34.05, #37811", "warn"); return; }
    if (n < this.min || n > this.max) toast("Fuera del rango disponible; voy al extremo más cercano.", "warn");
    this.pause();
    this.goTo(n);
  }

  _renderOverlay() {
    const n = this.current;
    this.hud.innerHTML = `<span class="pill">Fotograma ${fmt.frame(n)}</span><span class="pill">${fmt.time(n, this.fps)}</span>`;
    const mk = this._markersFn ? this._markersFn(n) || [] : [];
    this.markersEl.innerHTML = mk.map((m) => `<span class="mk ${m.cls || ""}">${escapeHtml(m.label)}</span>`).join("");
  }

  async _show(n) {
    const token = ++this._token;
    const url = frameUrl(this.matchId, n, this.width);
    const t = setTimeout(() => { if (token === this._token) this.loadingEl.hidden = false; }, 120);
    try {
      const im = await loadImage(url);
      if (token !== this._token) return;
      this.img.src = im.src;
    } catch (e) {
      if (token === this._token) toast("No se pudo cargar el fotograma " + fmt.frame(n), "error");
    } finally {
      clearTimeout(t);
      if (token === this._token) this.loadingEl.hidden = true;
    }
    if (token !== this._token) return;
    this._updateStrip(n);
    if (!this.playing) this._prefetch(n);
  }

  _updateStrip(n) {
    this.strip.querySelectorAll(".th").forEach((th) => {
      const f = n + parseInt(th.dataset.off, 10);
      const img = th.querySelector("img");
      const lab = th.querySelector("span");
      if (f < 0 || f >= this.total) { img.removeAttribute("src"); lab.textContent = ""; th.style.visibility = "hidden"; return; }
      th.style.visibility = "";
      th.style.opacity = (f < this.min || f > this.max) ? ".35" : "";
      const extra = this._stripClassFn ? this._stripClassFn(f) || "" : "";
      th.className = `th${th.dataset.off === "0" ? " current" : ""}${extra ? " " + extra : ""}`;
      lab.textContent = (f - n === 0 ? "▶ " : (f > n ? "+" : "")) + (f - n === 0 ? fmt.frame(f) : String(f - n));
      const url = frameUrl(this.matchId, f, this.stripWidth);
      const want = f;
      loadImage(url).then((im) => { if (this.current + parseInt(th.dataset.off, 10) === want) img.src = im.src; }).catch(() => {});
    });
  }

  _prefetch(n) {
    for (const d of [1, 2, 3, -1, 25, -25]) {
      const f = n + d;
      if (f >= this.min && f <= this.max) loadImage(frameUrl(this.matchId, f, this.width)).catch(() => {});
    }
    for (let d = -this.radius - 2; d <= this.radius + 2; d++) {
      const f = n + d;
      if (f >= 0 && f < this.total) loadImage(frameUrl(this.matchId, f, this.stripWidth)).catch(() => {});
    }
  }

  toggle() { this.playing ? this.pause() : this.play(); }

  play() {
    if (this.playing) return;
    this.playing = true;
    const btn = this.el.querySelector("[data-play]");
    if (btn) btn.textContent = "⏸ Pausa";
    const speed = () => parseFloat(this.speedSel ? this.speedSel.value : "1") || 1;
    const loop = async () => {
      while (this.playing) {
        const t0 = performance.now();
        if (this.current >= this.max) { this.pause(); break; }
        const next = this.current + 1;
        // Precarga unos fotogramas por delante para una reproducción fluida.
        for (let d = 1; d <= 6; d++) if (next + d <= this.max) loadImage(frameUrl(this.matchId, next + d, this.width)).catch(() => {});
        try { await loadImage(frameUrl(this.matchId, next, this.width)); } catch { this.pause(); break; }
        if (!this.playing) break;
        this.goTo(next);
        const wait = 1000 / (this.fps * speed()) - (performance.now() - t0);
        if (wait > 0) await new Promise((r) => setTimeout(r, wait));
      }
    };
    loop();
  }

  pause() {
    if (!this.playing) return;
    this.playing = false;
    const btn = this.el.querySelector("[data-play]");
    if (btn) btn.textContent = "▶ Reproducir";
    this._prefetch(this.current);
    this._updateStrip(this.current);
  }

  // Atajos estándar del visor (para usar en keys.bind de cada vista).
  standardKeys() {
    return {
      "ArrowLeft": () => this.step(-1), "ArrowRight": () => this.step(1),
      "Shift+ArrowLeft": () => this.step(-this.fps), "Shift+ArrowRight": () => this.step(this.fps),
      "Ctrl+ArrowLeft": () => this.step(-10 * this.fps), "Ctrl+ArrowRight": () => this.step(10 * this.fps),
      "PageUp": () => this.step(-60 * this.fps), "PageDown": () => this.step(60 * this.fps),
      " ": () => this.toggle(), "Home": () => { this.pause(); this.goTo(this.min); }, "End": () => { this.pause(); this.goTo(this.max); },
      "g": () => this.askGoTo(),
    };
  }

  destroy() { this.pause(); this._token++; this.el.innerHTML = ""; this._h = {}; }
}

// ---------------------------------------------------------------- Timeline
function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "#888"; }

export class Timeline extends Emitter {
  constructor(el, { total, fps = 25, start = 0, end = null, height = 84 }) {
    super();
    this.el = el;
    this.total = total;
    this.fps = fps;
    this.lo = start;               // límites absolutos
    this.hi = end === null ? total : end;
    this.a = this.lo; this.b = this.hi; // rango visible
    this.cursor = start;
    this.marks = []; this.spans = []; this.curve = null; this.curveStart = 0;
    this.progress = null;
    el.classList.add("timeline");
    el.innerHTML = `<canvas class="tl-canvas" style="height:${height}px"></canvas>
      <div class="tl-legend"><span data-l></span><span class="muted small">Rueda: zoom · Arrastrar: desplazar · Clic: ir</span><span data-r></span></div>
      <div class="tl-tip" hidden></div>`;
    this.canvas = el.querySelector("canvas");
    this.tip = el.querySelector(".tl-tip");
    this._ro = new ResizeObserver(() => this.draw());
    this._ro.observe(this.canvas);
    this._bind();
  }

  setBounds(lo, hi) { this.lo = lo; this.hi = hi; this.setRange(lo, hi); }
  setRange(a, b) {
    const span = Math.max(50, b - a);
    a = Math.max(this.lo, Math.min(a, this.hi - span));
    this.a = a; this.b = Math.min(this.hi, a + span);
    this.draw();
  }
  setCursor(n) {
    this.cursor = n;
    if (n < this.a || n > this.b) { const w = this.b - this.a; this.setRange(n - w / 2, n + w / 2); } else this.draw();
  }
  setMarks(m) { this.marks = m || []; this.draw(); }
  setSpans(s) { this.spans = s || []; this.draw(); }
  setCurve(values, start = 0) { this.curve = values; this.curveStart = start; this.draw(); }
  setProgress(frame) { this.progress = frame; this.draw(); }

  _x(f, W) { return ((f - this.a) / (this.b - this.a)) * W; }
  _f(x, W) { return this.a + (x / W) * (this.b - this.a); }

  draw() {
    const c = this.canvas, dpr = window.devicePixelRatio || 1;
    const W = c.clientWidth, H = c.clientHeight;
    if (!W || !H) return;
    if (c.width !== Math.round(W * dpr) || c.height !== Math.round(H * dpr)) { c.width = Math.round(W * dpr); c.height = Math.round(H * dpr); }
    const g = c.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    const top = 6, bot = H - 18;
    // progreso revisado
    if (this.progress !== null && this.progress > this.a) {
      g.fillStyle = "rgba(37,99,235,.07)";
      g.fillRect(0, 0, Math.min(W, this._x(this.progress, W)), H);
    }
    // curva
    if (this.curve && this.curve.length) {
      const i0 = Math.max(0, Math.floor(this.a - this.curveStart)), i1 = Math.min(this.curve.length - 1, Math.ceil(this.b - this.curveStart));
      if (i1 > i0) {
        const step = Math.max(1, Math.floor((i1 - i0) / (W * 2)));
        g.beginPath(); g.moveTo(this._x(this.curveStart + i0, W), bot);
        for (let i = i0; i <= i1; i += step) {
          let v = 0; for (let j = i; j < Math.min(i + step, i1 + 1); j++) v = Math.max(v, this.curve[j] || 0);
          g.lineTo(this._x(this.curveStart + i, W), bot - Math.min(1, v) * (bot - top));
        }
        g.lineTo(this._x(this.curveStart + i1, W), bot); g.closePath();
        g.fillStyle = "rgba(100,116,139,.28)"; g.fill();
      }
    }
    // franjas
    for (const s of this.spans) {
      if (s.end < this.a || s.start > this.b) continue;
      g.fillStyle = s.color || cssVar(s.cls === "gradual" ? "--gradual" : s.cls === "play" ? "--play" : s.cls === "noplay" ? "--noplay" : "--primary");
      g.globalAlpha = s.alpha || (s.cls === "gradual" ? 0.55 : 0.35);
      const x0 = this._x(s.start, W), x1 = Math.max(x0 + 2, this._x(s.end + 1, W));
      const y0 = s.row === "bottom" ? bot - 10 : top, y1 = s.row === "bottom" ? bot : bot;
      g.fillRect(x0, y0, x1 - x0, y1 - y0);
      g.globalAlpha = 1;
    }
    // marcas
    for (const m of this.marks) {
      if (m.frame < this.a || m.frame > this.b) continue;
      const x = Math.round(this._x(m.frame, W)) + 0.5;
      g.strokeStyle = m.color || cssVar(m.cls === "cut" ? "--cut" : m.cls === "hint" ? "--hint" : m.cls === "ok" ? "--ok" : m.cls === "score" ? "--primary" : "--text-2");
      g.lineWidth = m.width || 2;
      if (m.cls === "hint") g.setLineDash([3, 3]);
      g.beginPath(); g.moveTo(x, top); g.lineTo(x, bot); g.stroke(); g.setLineDash([]);
      if (m.label) { g.fillStyle = g.strokeStyle; g.font = "600 11px system-ui"; g.fillText(m.label, x + 3, top + 10); }
    }
    // eje
    g.fillStyle = cssVar("--text-2"); g.font = "11px monospace";
    const secs = (this.b - this.a) / this.fps;
    const stepS = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800].find((s) => secs / s <= 10) || 3600;
    const first = Math.ceil(this.a / (stepS * this.fps)) * stepS * this.fps;
    g.strokeStyle = cssVar("--border-strong"); g.lineWidth = 1;
    for (let f = first; f <= this.b; f += stepS * this.fps) {
      const x = Math.round(this._x(f, W)) + 0.5;
      g.beginPath(); g.moveTo(x, bot); g.lineTo(x, bot + 4); g.stroke();
      g.fillText(fmt.clock(f, this.fps), x + 2, H - 4);
    }
    // cursor
    if (this.cursor >= this.a && this.cursor <= this.b) {
      const x = Math.round(this._x(this.cursor, W)) + 0.5;
      g.strokeStyle = "#0f172a"; g.lineWidth = 2;
      g.beginPath(); g.moveTo(x, 0); g.lineTo(x, H); g.stroke();
      g.fillStyle = "#0f172a"; g.beginPath(); g.moveTo(x - 5, 0); g.lineTo(x + 5, 0); g.lineTo(x, 7); g.fill();
    }
    this.el.querySelector("[data-l]").textContent = fmt.clock(this.a, this.fps);
    this.el.querySelector("[data-r]").textContent = fmt.clock(this.b, this.fps);
  }

  _bind() {
    const c = this.canvas;
    let drag = null;
    c.addEventListener("mousedown", (e) => { drag = { x: e.clientX, a: this.a, b: this.b, moved: false }; });
    window.addEventListener("mousemove", (this._mm = (e) => {
      const r = c.getBoundingClientRect();
      if (drag) {
        const dx = e.clientX - drag.x;
        if (Math.abs(dx) > 4) drag.moved = true;
        if (drag.moved) {
          const df = -dx / r.width * (drag.b - drag.a);
          this.setRange(drag.a + df, drag.b + df);
        }
      }
      if (e.target === c) {
        const f = Math.round(this._f(e.clientX - r.left, r.width));
        this.tip.hidden = false;
        this.tip.style.left = `${e.clientX - r.left}px`;
        this.tip.textContent = `${fmt.time(f, this.fps)} · #${fmt.frame(f)}`;
      } else this.tip.hidden = true;
    }));
    window.addEventListener("mouseup", (this._mu = (e) => {
      if (!drag) return;
      const wasDrag = drag.moved; drag = null;
      if (!wasDrag && e.target === c) {
        const r = c.getBoundingClientRect();
        const f = Math.round(this._f(e.clientX - r.left, r.width));
        this.emit("seek", Math.max(this.lo, Math.min(this.hi - 1, f)));
      }
    }));
    c.addEventListener("mouseleave", () => { this.tip.hidden = true; });
    c.addEventListener("wheel", (e) => {
      e.preventDefault();
      const r = c.getBoundingClientRect();
      const fx = this._f(e.clientX - r.left, r.width);
      const k = e.deltaY > 0 ? 1.25 : 0.8;
      const a = fx - (fx - this.a) * k, b = fx + (this.b - fx) * k;
      if (b - a >= this.hi - this.lo) this.setRange(this.lo, this.hi); else this.setRange(a, b);
    }, { passive: false });
  }

  destroy() {
    this._ro.disconnect();
    window.removeEventListener("mousemove", this._mm);
    window.removeEventListener("mouseup", this._mu);
    this.el.innerHTML = ""; this._h = {};
  }
}

// ---------------------------------------------------------------- router
const VIEWS = { goles: "./goals.js", planos: "./sbd.js" };
const ctx = { api, ApiError, saveStatus, toast, confirmDialog, promptDialog, keys, helpPanel, shortcuts, fmt, FrameView, Timeline, frameUrl, escapeHtml };
let current = null; // {name, mod}

async function route() {
  const name = (location.hash || "#goles").slice(1);
  const viewName = VIEWS[name] ? name : "goles";
  if (current && current.name === viewName) return;
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.view === viewName));
  if (current && current.mod.unmount) {
    try { await current.mod.unmount(); } catch (e) { console.error(e); }
  }
  keys.unbind(current && current.name);
  shortcuts.set([]);
  helpPanel.set("");
  const container = document.getElementById("view");
  container.innerHTML = `<div class="empty">Cargando…</div>`;
  try {
    const mod = await import(VIEWS[viewName]);
    current = { name: viewName, mod };
    container.innerHTML = "";
    await mod.mount(container, ctx);
  } catch (e) {
    console.error(e);
    container.innerHTML = `<div class="error-box">No se pudo abrir esta sección: ${escapeHtml(e.message || e)}</div>`;
  }
}
window.addEventListener("hashchange", route);
if (!location.hash) history.replaceState(null, "", "#goles");
route();
