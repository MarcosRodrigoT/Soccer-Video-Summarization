// E2E de la vista «Planos y juego». Uso: node test_sbd.js  (servidor en :8792 con data-root limpio)
const puppeteer = require("puppeteer-core");
const fs = require("fs");
const path = require("path");

const BASE = process.env.BASE || "http://localhost:8792";
const DATA = process.env.DATA_ROOT;
const OUT = path.join(__dirname, "shots");
fs.mkdirSync(OUT, { recursive: true });
const SID = "Match1_30000_37500";

let passed = 0, failed = 0;
function check(cond, msg) { if (cond) { passed++; console.log("  ✓ " + msg); } else { failed++; console.log("  ✗ " + msg); } }
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function waitFor(fn, { timeout = 15000, every = 100, label = "condición" } = {}) {
  const t0 = Date.now();
  let last;
  while (Date.now() - t0 < timeout) {
    try { last = await fn(); if (last) return last; } catch (e) { last = e; }
    await sleep(every);
  }
  if (global.__page) console.log("toasts:", await global.__page.evaluate(() => window.__toasts));
  throw new Error("Timeout esperando " + label + " (último: " + JSON.stringify(last) + ")");
}
function diskDoc() {
  const p = path.join(DATA, "sbd", SID + ".json");
  return fs.existsSync(p) ? JSON.parse(fs.readFileSync(p, "utf8")) : null;
}

(async () => {
  const browser = await puppeteer.launch({ executablePath: "/usr/bin/google-chrome", headless: "new", args: ["--no-sandbox", "--window-size=1920,1080"] });
  const page = await browser.newPage();
  global.__page = page;
  await page.setViewport({ width: 1366, height: 768 });
  const errors = [];
  let expectNetErrors = false; // en la fase 9 los fallos de red y el 409 son intencionados
  page.on("console", (m) => { if (m.type() === "error" && !expectNetErrors && !(m.location() && /favicon/.test(m.location().url || ""))) errors.push(m.text() + " @ " + ((m.location() || {}).url || "")); });
  page.on("pageerror", (e) => errors.push("pageerror: " + e.message));
  page.on("requestfailed", (r) => { if ((r.failure() || {}).errorText === "net::ERR_ABORTED") return; /* recargas o cambios de vista */ if (!expectNetErrors && !r.url().includes("/api/frame/")) errors.push("requestfailed: " + r.url() + " " + (r.failure() || {}).errorText); });

  const installToastSpy = () => page.evaluate(() => {
    window.__toasts = [];
    new MutationObserver((ms) => { for (const m of ms) for (const n of m.addedNodes) if (n.classList && n.classList.contains("toast")) window.__toasts.push(n.className + "|" + n.textContent); })
      .observe(document.getElementById("toasts"), { childList: true });
  });
  const toasts = () => page.evaluate(() => {
    const m = document.querySelector("[data-msg]");
    const inline = m ? [m.className.replace("sbd-msg", "").trim() + "|" + m.innerText] : [];
    return window.__toasts.slice().concat(inline);
  });
  const clearToasts = () => page.evaluate(() => { window.__toasts = []; });
  const curFrame = () => page.evaluate(() => parseInt(document.querySelector(".fv-hud .pill").textContent.replace(/\D/g, ""), 10));
  const nTrs = () => page.evaluate(() => document.querySelector("[data-n-trs]").textContent);
  const shotsBadge = () => page.evaluate(() => document.querySelector("[data-n-shots]").textContent);
  const saveState = () => page.evaluate(() => document.getElementById("save-status").dataset.state);
  async function goto(n) {
    await page.keyboard.press("g");
    await page.waitForSelector(".dialog input");
    await page.type(".dialog input", "#" + n);
    await page.keyboard.press("Enter");
    await waitFor(async () => (await curFrame()) === n, { label: "ir a " + n });
  }
  async function dialogClick(okText) {
    await page.waitForSelector(".dialog-backdrop .dialog");
    const txt = await page.evaluate(() => document.querySelector(".dialog-backdrop .dialog").innerText);
    await page.evaluate((ok) => {
      const bs = [...document.querySelectorAll(".dialog-backdrop button")];
      const b = ok ? bs.find((x) => x.textContent.includes(ok)) : bs.find((x) => x.dataset.a === "1");
      b.click();
    }, okText || null);
    await sleep(150);
    return txt;
  }
  async function settle() {
    await waitFor(() => page.evaluate(() => {
      const l = document.querySelector(".fv-loading");
      if (l && !l.hidden) return false;
      const imgs = [...document.querySelectorAll(".fv-stage img, .fv-strip .th img, .sbd-shot-th img, [data-preview]")]
        .filter((i) => i.getBoundingClientRect().bottom > 0 && i.getBoundingClientRect().top < innerHeight && i.closest(".th, .fv-stage, .sbd-shot-th, .sbd-sum-thumb").style.visibility !== "hidden");
      return imgs.every((i) => i.complete && i.naturalWidth > 0);
    }), { label: "imágenes cargadas", timeout: 20000 }).catch((e) => console.log("  (aviso) " + e.message));
    await sleep(150);
  }
  const waitSaved = () => waitFor(async () => (await saveState()) === "saved", { label: "guardado" });

  // ----------------------------------------------------------------- 1. selector
  console.log("1. Selector de sesión");
  await page.goto(BASE + "/#planos", { waitUntil: "networkidle0" });
  await installToastSpy();
  await page.waitForSelector(".sbd-picker");
  check(await page.$eval('[data-f="start"]', (e) => e.value) === "20:00", "inicio por defecto 20:00");
  check(await page.$eval('[data-f="dur"]', (e) => e.value) === "5", "duración por defecto 5 min");
  check((await page.$eval("[data-sessions]", (e) => e.innerText)).includes("Todavía no hay sesiones"), "lista de sesiones vacía");
  // manual
  await page.click('[data-f="start"]');
  await sleep(100);
  await page.type('[data-f="start"]', "47:30");
  await page.keyboard.press("Tab");
  check((await page.$eval("[data-sum]", (e) => e.innerText)).includes("47:30 → 52:30"), "inicio manual 47:30 → resumen 47:30 → 52:30");
  await page.select('[data-f="dur"]', "10");
  check((await page.$eval("[data-sum]", (e) => e.innerText)).includes("47:30 → 57:30"), "duración 10 min");
  await page.select('[data-f="dur"]', "60");
  check(await page.$eval('[data-f="dur"]', (e) => e.options.length) === 60, "duración: opciones de 1 a 60 min");
  // aleatoria
  const starts = new Set();
  for (let i = 0; i < 5; i++) {
    await page.click('[data-act="random"]');
    const v = await page.$eval('[data-f="start"]', (e) => e.value);
    starts.add(v);
    const [mm, ss] = v.split(":").map(Number);
    const s = mm * 60 + ss;
    check(s >= 300 && s + 3600 <= 6265 - 300 + 1, `aleatoria ${v} dentro de [5 min, fin−5 min−ventana]`);
  }
  check(starts.size >= 2, "aleatoria produce inicios distintos");
  // overview: clic en el centro
  await page.select('[data-f="dur"]', "5");
  const box = await (await page.$(".sbd-ov-canvas")).boundingBox();
  await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2);
  let v = await page.$eval('[data-f="start"]', (e) => e.value);
  let [mm, ss] = v.split(":").map(Number);
  check(Math.abs(mm * 60 + ss - (6265.48 / 2 - 150)) < 15, `clic en el centro del partido centra la ventana (${v})`);
  // arrastre
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width / 4, box.y + box.height / 2, { steps: 5 });
  await page.mouse.up();
  v = await page.$eval('[data-f="start"]', (e) => e.value);
  [mm, ss] = v.split(":").map(Number);
  check(Math.abs(mm * 60 + ss - (6265.48 / 4 - 150)) < 15, `arrastrar mueve la ventana (${v})`);
  await page.click('[data-act="preset20"]');
  check(await page.$eval('[data-f="start"]', (e) => e.value) === "20:00", "«Desde el minuto 20» → 20:00");
  check(await page.$eval('[data-f="match"]', (e) => e.options.length) === 25, "25 partidos en el selector");
  await sleep(600);
  await settle();
  await page.screenshot({ path: path.join(OUT, "picker_1366.png") });
  await page.setViewport({ width: 1920, height: 1080 });
  await sleep(500);
  await settle();
  await page.screenshot({ path: path.join(OUT, "picker_1920.png") });
  await page.setViewport({ width: 1366, height: 768 });

  // ----------------------------------------------------------------- 2. crear sesión
  console.log("2. Crear sesión Match1 20:00 + 5 min");
  await page.click('[data-act="create"]');
  await page.waitForSelector(".sbd-work");
  await waitFor(async () => (await curFrame()) === 30000, { label: "frame 30000" });
  check(diskDoc() && diskDoc().revision === 1, "sesión creada en disco (revision 1)");
  check((await page.$eval(".sbd-head", (e) => e.innerText)).includes("20:00 → 25:00"), "cabecera muestra la ventana");

  // ----------------------------------------------------------------- 3. transiciones
  console.log("3. Cortes y graduales");
  await goto(30100);
  await page.keyboard.press("c");
  await waitFor(async () => (await nTrs()) === "1", { label: "1 transición" });
  check(true, "corte con tecla C en 30100");
  await goto(30200);
  await page.keyboard.press("c");
  await goto(30300);
  await page.click('[data-act="cut"]');
  await waitFor(async () => (await nTrs()) === "3", { label: "3 transiciones" });
  check(true, "corte con botón «Corte aquí» en 30300");
  await goto(30400);
  await page.keyboard.press("a");
  await sleep(100);
  const banner = await page.$eval("[data-msg]", (e) => (e.classList.contains("pending") ? e.innerText : ""));
  check(banner.includes("Gradual abierta desde 20:16.00"), "banner «Gradual abierta desde 20:16.00»");
  for (let i = 0; i < 10; i++) await page.keyboard.press("ArrowRight");
  await waitFor(async () => (await curFrame()) === 30410, { label: "30410" });
  await settle();
  await page.screenshot({ path: path.join(OUT, "work_gradual_open_1366.png") });
  // Cut while gradual open → error
  await clearToasts();
  await page.keyboard.press("c");
  await sleep(150);
  check((await toasts()).some((t) => t.includes("error") && t.includes("gradual abierta")), "corte con gradual abierta → error en español");
  await page.keyboard.press("b");
  await waitFor(async () => (await nTrs()) === "4", { label: "4 transiciones" });
  check(await page.$eval("[data-msg]", (e) => !e.classList.contains("pending")), "banner oculto tras B");
  // gradual por botones
  await goto(30500);
  await page.click('[data-act="gstart"]');
  await goto(30505);
  await page.click('[data-act="gend"]');
  await waitFor(async () => (await nTrs()) === "5", { label: "5 transiciones" });
  check(true, "gradual 30500–30505 con botones");
  // Escape cancela
  await goto(31000);
  await page.keyboard.press("a");
  await page.keyboard.press("Escape");
  await sleep(100);
  check(await page.$eval("[data-msg]", (e) => !e.classList.contains("pending")), "Esc cancela la gradual abierta");

  // ----------------------------------------------------------------- 4. inválidos
  console.log("4. Validación");
  await clearToasts();
  await goto(30405);
  await page.keyboard.press("c");
  await sleep(200);
  let ts = await toasts();
  check(ts.some((t) => t.includes("error") && t.includes("cae dentro de la transición gradual")), "corte dentro de gradual → error: " + (ts.find((t) => t.includes("error")) || "").split("|")[1]);
  await clearToasts();
  await goto(30100);
  await page.keyboard.press("c");
  await sleep(200);
  ts = await toasts();
  check(ts.some((t) => t.includes("Ya hay un corte")), "corte duplicado → error");
  await clearToasts();
  await goto(30395);
  await page.keyboard.press("a");
  await goto(30402);
  await page.keyboard.press("b");
  await sleep(200);
  ts = await toasts();
  check(ts.some((t) => t.includes("error") && t.includes("se solapan")), "gradual solapada → error: " + (ts.find((t) => t.includes("error")) || "").split("|")[1]);
  await page.keyboard.press("Escape");
  await clearToasts();
  await goto(30411);
  await page.keyboard.press("c");
  await sleep(200);
  ts = await toasts();
  check(ts.some((t) => t.includes("error") && t.includes("pegadas")), "corte pegado al fin de gradual → error");
  await clearToasts();
  await page.keyboard.press("Home");
  await page.keyboard.press("ArrowLeft");
  await page.keyboard.press("ArrowLeft");
  await waitFor(async () => (await curFrame()) === 29998, { label: "29998 (contexto)" });
  check((await page.$eval(".sbd-fv", (e) => e.className)).includes("outside"), "fuera de la ventana se ve en gris");
  await page.keyboard.press("c");
  await sleep(200);
  ts = await toasts();
  check(ts.some((t) => t.includes("error")), "corte fuera de la ventana → error");
  check((await nTrs()) === "5", "sigue habiendo 5 transiciones");
  await waitSaved();
  let d = diskDoc();
  check(d.transitions.length === 5, "en disco: 5 transiciones");

  // ----------------------------------------------------------------- 5. etiquetas
  console.log("5. Etiquetado");
  await goto(30050);
  await page.keyboard.press("1");
  const labs = ["2", "1", "2", "1", "2"];
  for (const k of labs) { await page.keyboard.press("Tab"); await sleep(60); await page.keyboard.press(k); }
  await waitFor(async () => (await shotsBadge()) === "6/6", { label: "6/6 etiquetados" });
  check(true, "6/6 planos etiquetados (1, Tab, 2…)");
  await clearToasts();
  await page.keyboard.press("Tab");
  await sleep(150);
  check((await toasts()).some((t) => t.includes("Todos los planos están etiquetados")), "Tab sin pendientes → aviso");
  await waitSaved();
  await sleep(300);
  d = diskDoc();
  const expLabels = { 30000: "play", 30100: "no_play", 30200: "play", 30300: "no_play", 30411: "play", 30506: "no_play" };
  check(JSON.stringify(d.shot_labels) === JSON.stringify(Object.fromEntries(Object.entries(expLabels).map(([k, v]) => [String(k), v]))), "en disco: etiquetas correctas " + JSON.stringify(d.shot_labels));
  check(JSON.stringify(d.transitions.map((t) => t.type === "cut" ? [t.frame] : [t.start, t.end])) === JSON.stringify([[30100], [30200], [30300], [30400, 30410], [30500, 30505]]), "en disco: transiciones correctas");
  // cambiar una etiqueta desde la lista de planos
  await page.click('[data-tab="shots"]');
  await page.waitForSelector('.sbd-shot[data-si="1"]');
  await page.click('.sbd-shot[data-si="1"] [data-setlab="play"]');
  await waitSaved();
  await sleep(200);
  check(diskDoc().shot_labels["30100"] === "play", "cambiar etiqueta desde la lista (plano 2 → Juego)");
  await page.click('.sbd-shot[data-si="1"] [data-setlab="no_play"]');
  await goto(30150);
  await sleep(800);
  await settle();
  await page.screenshot({ path: path.join(OUT, "work_shots_1366.png") });
  await page.click('[data-tab="trs"]');
  await sleep(300);
  await settle();
  await page.screenshot({ path: path.join(OUT, "work_trs_1366.png") });
  await page.setViewport({ width: 1920, height: 1080 });
  await sleep(900);
  await settle();
  await page.screenshot({ path: path.join(OUT, "work_trs_1920.png") });
  await page.click('[data-tab="shots"]');
  await sleep(900);
  await settle();
  await page.screenshot({ path: path.join(OUT, "work_shots_1920.png") });
  await page.setViewport({ width: 1366, height: 768 });
  await waitSaved();

  // ----------------------------------------------------------------- 6. recarga
  console.log("6. Recarga y persistencia");
  await page.reload({ waitUntil: "networkidle0" });
  await installToastSpy();
  await page.waitForSelector(".sbd-work");
  await sleep(400);
  check((await nTrs()) === "5", "tras recargar: 5 transiciones");
  check((await shotsBadge()) === "6/6", "tras recargar: 6/6 etiquetados");
  check(await curFrame() === 30150, "tras recargar: se recupera la posición (30150)");
  check(!(await page.$(".dialog-backdrop")), "sin diálogo de copia de seguridad (todo guardado)");

  // ----------------------------------------------------------------- 7. borrar transición y poda de etiquetas
  console.log("7. Eliminar transición → poda de etiquetas");
  await goto(30203);
  await clearToasts();
  await page.keyboard.press("Delete");
  const dtxt = await dialogClick("Eliminar");
  check(dtxt.includes("corte de 20:08.00"), "Supr pide confirmación del corte más cercano (20:08.00)");
  await sleep(200);
  ts = await toasts();
  check(ts.some((t) => t.includes("warn") && t.includes("Se ha quitado 1 etiqueta porque cambió el plano")), "aviso de etiqueta quitada");
  check((await nTrs()) === "4" && (await shotsBadge()) === "5/5", "4 transiciones y 5/5 planos");
  await waitSaved();
  await sleep(200);
  d = diskDoc();
  check(!("30200" in d.shot_labels) && Object.keys(d.shot_labels).length === 5, "en disco: etiqueta 30200 eliminada");
  // deshacer
  await page.keyboard.down("Control"); await page.keyboard.press("z"); await page.keyboard.up("Control");
  await waitFor(async () => (await nTrs()) === "5" && (await shotsBadge()) === "6/6", { label: "deshacer" });
  check(true, "Ctrl+Z restaura el corte y su etiqueta");
  await waitSaved();
  await sleep(200);
  check(diskDoc().shot_labels["30200"] === "play", "en disco: restaurado tras deshacer");
  // borrar desde la lista
  await page.click('[data-tab="trs"]');
  await page.waitForSelector('[data-tr-del="t1"]');
  await page.click('[data-tr-del="t1"]');
  await dialogClick("Eliminar");
  await waitFor(async () => (await nTrs()) === "4", { label: "4 tras borrar t1" });
  await waitSaved(); await sleep(200);
  check(!("30100" in diskDoc().shot_labels), "borrar desde la lista poda la etiqueta 30100");
  // mover un corte: re-etiquetado
  await page.click('[data-tr-edit="t3"]');
  await page.click('[data-mv="p1"][data-mv-id="t3"]');
  await waitSaved(); await sleep(200);
  d = diskDoc();
  check(d.transitions.find((t) => t.id === "t3").frame === 30301 && d.shot_labels["30301"] === "no_play" && !("30300" in d.shot_labels), "ajustar corte +1 conserva la etiqueta del plano siguiente");
  // J/K
  await goto(30250);
  await page.keyboard.press("k");
  await waitFor(async () => (await curFrame()) === 30301, { label: "K → 30301" });
  await page.keyboard.press("k");
  await waitFor(async () => (await curFrame()) === 30400, { label: "K → 30400" });
  await page.keyboard.press("j");
  await waitFor(async () => (await curFrame()) === 30301, { label: "J → 30301" });
  check(true, "J/K navegan entre transiciones");

  // ----------------------------------------------------------------- 8. terminar sesión
  console.log("8. Terminar sesión");
  await goto(33000);
  await page.keyboard.press("c");
  await waitFor(async () => (await shotsBadge()) === "5/6", { label: "5/6" });
  await page.click('[data-act="finish"]');
  let ft = await dialogClick("Ir al siguiente sin etiquetar");
  check(ft.includes("Falta 1 plano"), "terminar con planos sin etiquetar → explica lo que falta");
  await page.keyboard.press("2");
  await waitFor(async () => (await shotsBadge()) === "6/6", { label: "6/6" });
  await page.keyboard.press("a");
  await page.click('[data-act="finish"]');
  ft = await dialogClick("Entendido");
  check(ft.includes("gradual abierta"), "terminar con gradual abierta → aviso");
  await page.keyboard.press("Escape");
  await page.click('[data-act="finish"]');
  ft = await dialogClick("Terminar igualmente");
  check(ft.includes("revisado"), "aviso de ventana no revisada entera");
  await waitSaved(); await sleep(200);
  check(diskDoc().complete === true, "en disco: complete = true");
  check((await page.$eval("[data-sstate]", (e) => e.innerText)).includes("Terminada"), "insignia «Terminada»");
  // un cambio que deja planos sin etiquetar reabre la sesión
  await goto(31000);
  await page.keyboard.press("c");
  await waitSaved(); await sleep(200);
  check(diskDoc().complete === false, "añadir un corte (plano nuevo sin etiquetar) vuelve a «en curso»");

  // ----------------------------------------------------------------- 9. sin conexión y copia local
  console.log("9. Errores de red, copia local y conflicto");
  expectNetErrors = true;
  let blockPut = true;
  await page.setRequestInterception(true);
  page.on("request", (req) => { if (blockPut && req.method() === "PUT") req.abort("internetdisconnected"); else req.continue(); });
  await page.keyboard.press("1");
  await waitFor(async () => (await saveState()) === "error", { label: "estado error sin conexión" });
  check(true, "sin conexión → indicador de error");
  const bk = await page.evaluate((sid) => localStorage.getItem("sbd.backup." + sid), SID);
  check(bk && JSON.parse(bk).shot_labels["31000"] === "play", "copia local en localStorage");
  blockPut = false;
  await waitFor(async () => (await saveState()) === "saved", { label: "reintento → guardado", timeout: 12000 });
  await sleep(200);
  check(diskDoc().shot_labels["31000"] === "play", "tras reconectar se guarda (reintento automático)");
  // conflicto: otra «pestaña» guarda
  const other = diskDoc();
  other.shot_labels["31000"] = "no_play";
  const r = await page.evaluate(async (sid, doc) => (await fetch("/api/sbd/" + sid, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(doc) })).status, SID, other);
  check(r === 200, "PUT externo (simula otra pestaña)");
  await goto(31100);
  await page.keyboard.press("c");
  const ct = await dialogClick("Cargar la versión guardada");
  check(ct.includes("ha cambiado en otra"), "409 → diálogo de conflicto en español");
  const bt = await dialogClick("Recuperar mis cambios");
  check(bt.includes("no llegaron a guardarse"), "tras recargar, ofrece recuperar la copia local");
  await waitSaved(); await sleep(300);
  d = diskDoc();
  check(d.transitions.some((t) => t.frame === 31100), "copia recuperada y guardada (corte 31100)");

  // ----------------------------------------------------------------- 10. ayuda y vuelta a la lista
  await sleep(300);
  expectNetErrors = false;
  console.log("10. Ayuda, lista de sesiones y borrado de sesión");
  await page.keyboard.press("?");
  await sleep(300);
  await settle();
  await page.screenshot({ path: path.join(OUT, "help_1366.png") });
  await page.keyboard.press("Escape");
  await page.click('[data-act="close"]');
  await page.waitForSelector(".sbd-sess");
  const sessTxt = await page.$eval(".sbd-sess", (e) => e.innerText);
  check(sessTxt.includes("20:00 → 25:00") && /Planos etiquetados: \d+ de \d+/.test(sessTxt), "la sesión aparece en la lista con progreso");
  // crear otra sesión y borrarla
  await page.click('[data-act="random"]');
  await page.click('[data-act="create"]');
  await page.waitForSelector(".sbd-work");
  await sleep(300);
  await page.click('[data-act="close"]');
  await page.waitForSelector(".sbd-sess");
  const nBefore = await page.$$eval(".sbd-sess", (e) => e.length);
  check(nBefore === 2, "dos sesiones en la lista");
  await sleep(300);
  await settle();
  await page.screenshot({ path: path.join(OUT, "picker_sessions_1366.png") });
  const victim = await page.$$eval(".sbd-sess", (e) => e.map((x) => x.dataset.sid).find((s) => s !== "Match1_30000_37500"));
  await page.click(`.sbd-sess[data-sid="${victim}"] [data-del]`);
  const delTxt = await dialogClick("Eliminar");
  check(delTxt.includes("papelera"), "borrar sesión pide confirmación");
  await waitFor(async () => (await page.$$eval(".sbd-sess", (e) => e.length)) === 1, { label: "1 sesión" });
  check(fs.readdirSync(path.join(DATA, "trash")).some((f) => f.startsWith(victim)), "sesión movida a trash/");
  await page.setViewport({ width: 1920, height: 1080 });
  await sleep(400);
  await settle();
  await page.screenshot({ path: path.join(OUT, "picker_sessions_1920.png") });

  // ----------------------------------------------------------------- 11. Navegación fluida
  console.log("11. Rendimiento de navegación");
  await page.click('.sbd-sess [data-open]');
  await page.waitForSelector(".sbd-work");
  await goto(32000);
  await sleep(1000);
  const t0 = Date.now();
  for (let i = 0; i < 25; i++) { await page.keyboard.press("ArrowRight"); }
  await waitFor(async () => (await curFrame()) === 32025, { label: "32025" });
  await waitFor(() => page.evaluate(() => document.querySelector(".fv-loading").hidden), { label: "imagen cargada" });
  console.log(`  25 pasos de fotograma en ${Date.now() - t0} ms`);
  await settle();
  await page.screenshot({ path: path.join(OUT, "work_1920_after.png") });

  // revisado hasta: recorrer desde el inicio
  await page.keyboard.press("Home");
  await waitFor(async () => (await curFrame()) === 30000, { label: "Home" });
  for (let i = 0; i < 40; i++) await page.keyboard.press("ArrowRight");
  for (let i = 0; i < 4; i++) await page.keyboard.press("ArrowRight");
  await page.keyboard.down("Shift"); for (let i = 0; i < 4; i++) await page.keyboard.press("ArrowRight"); await page.keyboard.up("Shift");
  await waitFor(async () => (await curFrame()) === 30144, { label: "30144" });
  const rev = await page.$eval("[data-m-rev]", (e) => e.innerText);
  check(rev.includes("00:05"), "revisado hasta avanza al recorrer (" + rev + ")");
  await page.keyboard.down("Control"); await page.keyboard.press("ArrowRight"); await page.keyboard.up("Control");
  await sleep(100);
  check((await page.$eval("[data-m-rev]", (e) => e.innerText)).includes("00:05"), "un salto de 10 s no cuenta como revisado");
  // tira ampliada
  await page.click('[data-pref="bigStrip"]');
  await waitFor(() => page.evaluate(() => document.querySelectorAll(".fv-strip .th").length === 15), { label: "15 miniaturas" });
  check(true, "tira ampliada → 15 miniaturas");
  // gradual abierta persiste tras recargar
  await goto(35000);
  await page.keyboard.press("a");
  await sleep(1500);
  check(diskDoc().open_gradual_start === 35000, "gradual abierta guardada (open_gradual_start)");
  await page.reload({ waitUntil: "networkidle0" });
  await page.waitForSelector(".sbd-work");
  await sleep(300);
  check((await page.$eval("[data-msg]", (e) => e.innerText)).includes("Gradual abierta desde 23:20.00"), "tras recargar sigue la gradual abierta");
  check(await page.evaluate(() => document.querySelectorAll(".fv-strip .th").length) === 15, "preferencia de tira ampliada recordada");
  check(diskDoc().reviewed && diskDoc().reviewed[0][0] === 30000 && diskDoc().reviewed[0][1] >= 30144, "progreso de revisión guardado en disco (reviewed)");
  await page.keyboard.press("Escape");
  await settle();
  await page.screenshot({ path: path.join(OUT, "work_bigstrip_1920.png") });
  check(errors.length === 0, "sin errores de consola: " + JSON.stringify(errors));
  console.log(`\n${passed} OK, ${failed} fallos`);
  await browser.close();
  process.exit(failed ? 1 : 0);
})().catch(async (e) => { console.error("FALLO:", e); process.exit(2); });
