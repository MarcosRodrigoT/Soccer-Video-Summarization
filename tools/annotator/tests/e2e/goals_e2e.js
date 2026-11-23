// Test end-to-end de la vista «Goles» del anotador (Chrome headless + puppeteer-core).
// Uso: node goals_e2e.js   (servidor en http://localhost:8791 con --data-root $DATA)
const puppeteer = require("puppeteer-core");
const fs = require("fs");
const path = require("path");

const BASE = "http://localhost:8791";
// Directorio de datos del servidor de PRUEBA (no usar variables de entorno: DATA ya existe en este sistema).
const DATA = "/tmp/annotator_e2e/annot_test_goals";
if (!DATA.includes("/annotator_e2e/annot_test_goals")) throw new Error("DATA no es el directorio de prueba");
const OUT = path.join(__dirname, "out");
fs.mkdirSync(OUT, { recursive: true });

const results = [];
function check(name, cond, info = "") {
  results.push({ name, ok: !!cond, info });
  console.log(`${cond ? "PASS" : "FAIL"}  ${name}${info ? "  — " + info : ""}`);
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function api(method, p, body) {
  const r = await fetch(BASE + p, { method, headers: body ? { "Content-Type": "application/json" } : {}, body: body ? JSON.stringify(body) : undefined });
  let json = null; try { json = await r.json(); } catch { /* */ }
  return { status: r.status, json };
}
const readDoc = (id) => JSON.parse(fs.readFileSync(path.join(DATA, "goals", `${id}.json`), "utf8"));
const readCsv = (id) => fs.readFileSync(path.join(DATA, "goals", `${id}_goals.csv`), "utf8");

(async () => {
  // Reinicia solo los datos de goles del directorio de PRUEBA.
  fs.rmSync(path.join(DATA, "goals"), { recursive: true, force: true });
  fs.mkdirSync(path.join(DATA, "goals"), { recursive: true });
  for (const f of fs.existsSync(path.join(DATA, "history")) ? fs.readdirSync(path.join(DATA, "history")) : []) {
    if (f.startsWith("goals_")) fs.rmSync(path.join(DATA, "history", f));
  }

  const pl1 = (await api("GET", "/api/match/PL1")).json;
  const sa1 = (await api("GET", "/api/match/SA1")).json;
  const ev = pl1.hints.score_events;
  const cuts = pl1.hints.shot_starts;

  const browser = await puppeteer.launch({ executablePath: "/usr/bin/google-chrome", headless: "new", args: ["--no-sandbox"] });
  const page = await browser.newPage();
  const errors = [];
  let allowNetErrors = false;
  page.on("console", (m) => {
    if (m.type() !== "error") return;
    const url = (m.location() && m.location().url) || "";
    if (/favicon\.ico/.test(url)) return;
    if (allowNetErrors && /api\/goals/.test(url)) return;
    errors.push(`${m.text()} ${url} ${JSON.stringify(m.stackTrace().slice(0, 4))}`);
  });
  page.on("pageerror", (e) => errors.push("PAGEERROR " + (e.stack || e.message)));
  page.on("dialog", (d) => d.accept()); // beforeunload

  let blockPut = false;
  await page.setRequestInterception(true);
  page.on("request", (req) => {
    if (blockPut && req.method() === "PUT" && req.url().includes("/api/goals/")) req.abort("internetdisconnected");
    else req.continue();
  });

  const frame = () => page.$eval(".fv-hud .pill", (el) => parseInt(el.textContent.replace(/\D/g, ""), 10));
  const waitFrame = (n) => page.waitForFunction((n) => {
    const el = document.querySelector(".fv-hud .pill");
    return el && parseInt(el.textContent.replace(/\D/g, ""), 10) === n;
  }, { timeout: 8000 }, n);
  const blur = () => page.evaluate(() => document.activeElement && document.activeElement.blur());
  async function goTo(n) {
    await blur();
    await page.keyboard.press("g");
    await page.waitForSelector(".dialog input", { timeout: 4000 });
    await page.keyboard.type(`#${n}`);
    await page.keyboard.press("Enter");
    await waitFrame(n);
  }
  const saveState = () => page.$eval("#save-status", (el) => el.dataset.state);
  const waitState = (s, timeout = 10000) => page.waitForFunction((s) => document.getElementById("save-status").dataset.state === s, { timeout }, s);
  const fieldText = (k) => page.$eval(`.gl-field[data-field="${k}"] [data-val]`, (el) => el.textContent.replace(/[\s\u00a0\u202f]+/g, " ").trim());
  const lastToast = () => page.evaluate(() => { const t = [...document.querySelectorAll(".toast")]; return t.length ? { cls: t[t.length - 1].className, text: t[t.length - 1].textContent } : null; });
  const nGoals = () => page.$$eval(".gl-goal", (els) => els.length);
  const noHScroll = () => page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth);

  // ---------------------------------------------------------------- 1366x768
  await page.setViewport({ width: 1366, height: 768 });
  await page.goto(BASE + "/#goles", { waitUntil: "networkidle2" });
  await page.waitForSelector(".gl-goal", { timeout: 20000 });
  check("Carga #goles con 8 goles de PL1", (await nGoals()) === 8, `${await nGoals()} goles`);
  check("Chip PL1 activo", await page.$eval(".gl-chip.active", (el) => el.dataset.id) === "PL1");
  await waitState("saved");
  let d = readDoc("PL1");
  check("Inicialización desde pistas guardada en disco (8 goles con hint_frame)",
    d.goals.length === 8 && d.goals.every((g, i) => g.hint_frame === ev[i].frame) && d.revision === 1, `rev ${d.revision}`);
  const label1 = await page.$eval(".gl-goal", (el) => el.textContent.replace(/\s+/g, " ").trim());
  check("Etiqueta del gol 1 «0-0 → 0-1 · 18:39 · Chelsea»", /0-0 → 0-1/.test(label1) && /18:39/.test(label1) && /Chelsea/.test(label1), label1);
  check("Visor 8 s antes del gol detectado", (await frame()) === ev[0].frame - 200, `frame ${await frame()}`);
  check("Sin scroll horizontal a 1366", await noHScroll());
  await sleep(1200);
  await page.screenshot({ path: path.join(OUT, "01_inicial_1366.png") });

  // Marcar con teclas S / E y botón M
  await goTo(27800);
  await page.keyboard.press("s");
  check("S marca inicio (27 800)", (await fieldText("start_frame")).includes("27 800"), await fieldText("start_frame"));
  const mk = await page.$eval(".fv-markers", (el) => el.textContent);
  check("Insignia «Inicio del plano ✓» sobre el fotograma", mk.includes("Inicio del plano"), mk);
  await page.keyboard.press("]");
  const nextCut = cuts.find((c) => c > 27800);
  await waitFrame(nextCut);
  check("] salta al siguiente corte sugerido", (await frame()) === nextCut, `${nextCut}`);
  await page.keyboard.press("[");
  const prevCut = [...cuts].reverse().find((c) => c < nextCut);
  await waitFrame(prevCut);
  check("[ salta al corte sugerido anterior", (await frame()) === prevCut, `${prevCut}`);
  await goTo(27950);
  await page.keyboard.press("e");
  check("E marca fin del plano (27 950)", (await fieldText("end_frame_shot")).includes("27 950"));
  await goTo(28010);
  await page.click('.gl-field[data-field="end_frame_scoreboard"] [data-act="mark"]');
  check("Botón «Marcar aquí» marca el cambio de marcador (28 010)", (await fieldText("end_frame_scoreboard")).includes("28 010"));
  // Validación: fin antes del inicio → rechazado
  await goTo(27700);
  await page.keyboard.press("e");
  const t = await lastToast();
  check("Fin anterior al inicio se rechaza con aviso", t && t.cls.includes("error") && (await fieldText("end_frame_shot")).includes("27 950"), t && t.text);
  // Estado y progreso
  check("Gol 1 «Completo ✓»", (await page.$eval(".gl-goal.sel", (el) => el.textContent)).includes("Completo"));
  check("Progreso «1 de 8»", (await page.$eval("[data-progress]", (el) => el.textContent.replace(/[\s\u202f]+/g, " "))).includes("1 de 8"));
  // Notas
  await page.$eval("[data-notes]", (el) => el.scrollIntoView({ block: "center" }));
  await page.focus("[data-notes]");
  check("Foco en las notas", await page.evaluate(() => document.activeElement.matches("[data-notes]")));
  await page.keyboard.type("Prueba e2e");
  await page.keyboard.press("Escape");
  await waitState("saved");
  await sleep(300);
  d = readDoc("PL1");
  const g1 = d.goals.find((g) => g.id === "g1");
  check("JSON en disco con los 3 fotogramas y notas", g1.start_frame === 27800 && g1.end_frame_shot === 27950 && g1.end_frame_scoreboard === 28010 && g1.notes === "Prueba e2e", JSON.stringify(g1));
  const csv = readCsv("PL1");
  check("CSV exportado", csv === "goal_id,start_frame,end_frame_shot,end_frame_scoreboard,notes\n1,27800,27950,28010,Prueba e2e\n", JSON.stringify(csv));
  check("Copia local borrada tras confirmar el servidor", await page.evaluate(() => localStorage.getItem("annotator.goals.pending.PL1") === null));
  await goTo(27950);
  await sleep(1200);
  await page.screenshot({ path: path.join(OUT, "02_marcado_1366.png") });
  // Se ve scroll a la línea de tiempo
  await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
  await sleep(400);
  await page.screenshot({ path: path.join(OUT, "02b_marcado_1366_scroll.png") });
  await page.evaluate(() => window.scrollTo(0, 0));

  // Recarga: persistencia
  await page.reload({ waitUntil: "networkidle2" });
  await page.waitForSelector(".gl-goal.sel", { timeout: 20000 });
  check("Tras recargar: gol 1 seleccionado y valores persistidos",
    (await fieldText("start_frame")).includes("27 800") && (await fieldText("end_frame_shot")).includes("27 950") && (await fieldText("end_frame_scoreboard")).includes("28 010"));
  check("Tras recargar: visor en el inicio marcado", (await frame()) === 27800);
  check("Tras recargar: notas", (await page.$eval("[data-notes]", (el) => el.value)) === "Prueba e2e");

  // ------------------------------------------------------------ conflicto 409
  await blur();
  await page.keyboard.press("n");
  await waitFrame(ev[1].frame - 200);
  check("N pasa al gol 2 (8 s antes)", true);
  const cur = (await api("GET", "/api/goals/PL1")).json;
  cur.goals.find((g) => g.id === "g2").notes = "Editado en otra pestaña";
  const put = await api("PUT", "/api/goals/PL1", cur);
  check("Edición concurrente por API", put.status === 200, `status ${put.status}`);
  allowNetErrors = true;
  await page.keyboard.press("s");
  await page.waitForSelector(".gl-dialog", { timeout: 8000 });
  const dlg = await page.$eval(".gl-dialog", (el) => el.textContent);
  check("Diálogo de conflicto en español", /otra pestaña o ventana/.test(dlg), dlg.slice(0, 80));
  await page.screenshot({ path: path.join(OUT, "03_conflicto_1366.png") });
  await page.click('.gl-dialog [data-a="1"]');
  await waitState("saved");
  await sleep(300);
  check("Tras 409: se carga la versión del servidor (notas)", (await page.$eval("[data-notes]", (el) => el.value)) === "Editado en otra pestaña");
  check("Tras 409: el cambio local no se aplica", (await fieldText("start_frame")).includes("Sin marcar"));
  check("Tras 409: copia de seguridad del conflicto en localStorage", await page.evaluate(() => !!localStorage.getItem("annotator.goals.conflict.PL1")));
  d = readDoc("PL1");
  check("Tras 409: disco intacto", d.goals.find((g) => g.id === "g2").start_frame === null && d.goals.find((g) => g.id === "g2").notes === "Editado en otra pestaña");

  // ------------------------------------------------- error de red y reintento
  blockPut = true;
  await blur();
  await page.keyboard.press("s");
  await waitState("error", 8000);
  check("Sin red: indicador de error", (await saveState()) === "error", await page.$eval("#save-status", (el) => el.textContent.trim()));
  check("Sin red: copia pendiente en localStorage", await page.evaluate(() => !!localStorage.getItem("annotator.goals.pending.PL1")));
  await page.screenshot({ path: path.join(OUT, "04_sin_red_1366.png") });
  blockPut = false;
  await waitState("saved", 12000);
  await sleep(300);
  check("Reintento automático guarda al volver la red", readDoc("PL1").goals.find((g) => g.id === "g2").start_frame === ev[1].frame - 200);

  // ------------------------------------ recarga con cambios sin guardar
  blockPut = true;
  await goTo(45500);
  await page.keyboard.press("e");
  await waitState("error", 8000);
  await page.reload({ waitUntil: "networkidle2" });
  await page.waitForSelector(".dialog-backdrop", { timeout: 15000 });
  const rd = await page.$eval(".dialog-backdrop", (el) => el.textContent);
  check("Al recargar se ofrece recuperar cambios sin guardar", /Recuperar mis cambios/.test(rd), rd.replace(/\s+/g, " ").slice(0, 90));
  await page.screenshot({ path: path.join(OUT, "05_recuperar_1366.png") });
  blockPut = false;
  await page.click('.dialog-backdrop [data-a="1"]');
  await waitState("saved", 12000);
  await sleep(300);
  check("Cambios recuperados y guardados", readDoc("PL1").goals.find((g) => g.id === "g2").end_frame_shot === 45500);
  allowNetErrors = false;

  // ------------------------------------------------ añadir / eliminar / deshacer
  await page.waitForSelector(".gl-goal.sel");
  await goTo(100000);
  await page.click('[data-act="add"]');
  check("Añadir gol → 9 goles", (await nGoals()) === 9);
  await blur();
  await page.keyboard.press("s");
  check("Gol añadido: inicio marcado", (await fieldText("start_frame")).includes("100 000"));
  check("Etiqueta «añadido a mano»", (await page.$eval(".gl-goal.sel", (el) => el.textContent)).includes("añadido a mano"));
  await page.click('[data-act="del"]');
  await page.waitForSelector(".dialog-backdrop");
  await page.screenshot({ path: path.join(OUT, "06_eliminar_1366.png") });
  await page.click('.dialog-backdrop [data-a="1"]');
  await sleep(200);
  check("Eliminar gol (confirmado) → 8", (await nGoals()) === 8);
  await blur();
  await page.keyboard.down("Control"); await page.keyboard.press("z"); await page.keyboard.up("Control");
  check("Ctrl+Z recupera el gol eliminado → 9", (await nGoals()) === 9);
  await page.click('[data-act="del"]');
  await page.waitForSelector(".dialog-backdrop");
  await page.click('.dialog-backdrop [data-a="0"]');
  check("Cancelar eliminación mantiene 9", (await nGoals()) === 9);
  await page.click('[data-act="del"]');
  await page.waitForSelector(".dialog-backdrop");
  await page.click('.dialog-backdrop [data-a="1"]');
  await waitState("saved");
  await sleep(300);
  check("Disco: 8 goles tras eliminar", readDoc("PL1").goals.length === 8);

  // ------------------------------------------------------------- SA1
  await page.click('.gl-chip[data-id="SA1"]');
  await page.waitForFunction(() => document.querySelector(".gl-chip.active")?.dataset.id === "SA1", { timeout: 15000 });
  await page.waitForSelector(".gl-body:not(.is-loading) .gl-goal.sel", { timeout: 15000 });
  const sev = sa1.hints.score_events;
  await waitFrame(sev[0].frame - 200);
  check("SA1: 8 goles y visor 8 s antes del primero", (await nGoals()) === 8);
  await blur();
  await page.keyboard.press("s");
  await waitState("saved");
  await sleep(300);
  check("SA1: inicio guardado", readDoc("SA1").goals.find((g) => g.id === "g1").start_frame === sev[0].frame - 200);
  await page.waitForFunction(() => { const i = document.querySelector("[data-sbcur]"); return i && i.complete && i.naturalWidth > 0; }, { timeout: 10000 });
  check("SA1: recorte del marcador cargado", true);
  await sleep(800);
  await page.screenshot({ path: path.join(OUT, "07_sa1_1366.png") });
  await page.click('.gl-chip[data-id="PL1"]');
  await page.waitForFunction(() => document.querySelector(".gl-chip.active")?.dataset.id === "PL1", { timeout: 15000 });
  await page.waitForSelector(".gl-body:not(.is-loading) .gl-goal.sel");
  check("Volver a PL1 conserva el progreso", (await page.$eval("[data-progress]", (el) => el.textContent.replace(/[\s\u202f]+/g, " "))).includes("1 de 8"));

  // ------------------------------------------------------ revisado
  await page.click('[data-act="complete"]');
  await page.waitForSelector(".dialog-backdrop");
  await page.click('.dialog-backdrop [data-a="1"]');
  await waitState("saved");
  await sleep(300);
  check("Marcar partido como revisado → complete=true", readDoc("PL1").complete === true);
  check("Botón muestra «Partido revisado»", (await page.$eval('[data-act="complete"]', (el) => el.textContent)).includes("Partido revisado"));
  await page.click('[data-act="complete"]');
  await waitState("saved");
  await sleep(300);
  check("Desmarcar revisado → complete=false", readDoc("PL1").complete === false);

  // ------------------------------------------- cambiar de pestaña y volver
  await page.click('.gl-goal[data-goal="g1"]');
  await waitFrame(27800);
  await goTo(27400);
  await page.keyboard.press("m");   // cambio pendiente: debe guardarse al desmontar la vista
  await page.evaluate(() => { location.hash = "#planos"; });
  await page.waitForFunction(() => !document.querySelector(".gl"), { timeout: 10000 });
  await sleep(500);
  check("Al salir de la vista se guarda el cambio pendiente", readDoc("PL1").goals.find((g) => g.id === "g1").end_frame_scoreboard === 27400);
  await page.evaluate(() => { location.hash = "#goles"; });
  await page.waitForSelector(".gl-body:not(.is-loading) .gl-goal.sel", { timeout: 15000 });
  check("Al volver a #goles se recupera el estado", (await fieldText("end_frame_scoreboard")).includes("27 400"));
  await blur();
  await page.keyboard.down("Control"); await page.keyboard.press("z"); await page.keyboard.up("Control");
  check("Ctrl+Z sin historial (tras remontar) no rompe nada", (await fieldText("end_frame_scoreboard")).includes("27 400"));
  await goTo(28010);
  await page.keyboard.press("m");
  await waitState("saved");
  await sleep(300);
  check("Valor restaurado a 28 010", readDoc("PL1").goals.find((g) => g.id === "g1").end_frame_scoreboard === 28010);

  // ---------------------------------------------------------------- 1920x1080
  await page.setViewport({ width: 1920, height: 1080 });
  await page.reload({ waitUntil: "networkidle2" });
  await page.waitForSelector(".gl-goal.sel", { timeout: 20000 });
  await blur();
  await page.click('.gl-goal[data-goal="g1"]');
  await waitFrame(27800);
  await sleep(1500);
  check("Sin scroll horizontal a 1920", await noHScroll());
  const fits = await page.evaluate(() => {
    const tl = document.querySelector(".gl-tl").getBoundingClientRect();
    const bar = document.querySelector(".shortcut-bar").getBoundingClientRect();
    return tl.bottom <= bar.top + 1;
  });
  check("A 1920×1080 la línea de tiempo cabe sin scroll (no tapada por la barra de atajos)", fits);
  await page.screenshot({ path: path.join(OUT, "08_1920.png") });
  await page.keyboard.press("?");
  await sleep(300);
  await page.screenshot({ path: path.join(OUT, "09_ayuda_1920.png") });
  await page.keyboard.press("Escape");
  await page.setViewport({ width: 1366, height: 768 });
  await sleep(600);
  await page.screenshot({ path: path.join(OUT, "10_final_1366.png") });
  await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
  await sleep(300);
  await page.screenshot({ path: path.join(OUT, "11_final_1366_abajo.png") });
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.setViewport({ width: 1920, height: 1080 });
  await sleep(400);

  // Comprobación de solapes: elementos de la vista que se salgan de su tarjeta.
  const overlaps = await page.evaluate(() => {
    const bad = [];
    document.querySelectorAll(".gl .card").forEach((card) => {
      const r = card.getBoundingClientRect();
      card.querySelectorAll("button, .badge, .gl-field-val, kbd").forEach((el) => {
        const e = el.getBoundingClientRect();
        if (e.width && (e.right > r.right + 1 || e.left < r.left - 1)) bad.push(el.textContent.trim().slice(0, 30));
      });
    });
    return bad;
  });
  check("Ningún botón se sale de su tarjeta (1920)", overlaps.length === 0, overlaps.join(" | "));
  await page.setViewport({ width: 1366, height: 768 });
  await sleep(500);
  const overlaps2 = await page.evaluate(() => {
    const bad = [];
    document.querySelectorAll(".gl .card").forEach((card) => {
      const r = card.getBoundingClientRect();
      card.querySelectorAll("button, .badge, .gl-field-val, kbd").forEach((el) => {
        const e = el.getBoundingClientRect();
        if (e.width && (e.right > r.right + 1 || e.left < r.left - 1)) bad.push(el.textContent.trim().slice(0, 30));
      });
    });
    return bad;
  });
  check("Ningún botón se sale de su tarjeta (1366)", overlaps2.length === 0, overlaps2.join(" | "));

  check("Sin errores en consola", errors.length === 0, errors.join(" || "));
  await browser.close();
  const failed = results.filter((r) => !r.ok);
  console.log(`\n${results.length - failed.length}/${results.length} comprobaciones superadas`);
  process.exit(failed.length ? 1 : 0);
})().catch((e) => { console.error("ERROR", e); process.exit(2); });
