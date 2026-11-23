// Mide solapes de la vista de trabajo a distintas resoluciones (con la sesión que dejó el e2e).
const puppeteer = require("puppeteer-core");
(async () => {
  const browser = await puppeteer.launch({ executablePath: "/usr/bin/google-chrome", headless: "new", args: ["--no-sandbox"] });
  const page = await browser.newPage();
  for (const [w, h] of [[1366, 768], [1920, 1080], [1536, 864], [1280, 720]]) {
    await page.setViewport({ width: w, height: h });
    await page.goto("http://localhost:8792/#planos", { waitUntil: "networkidle0" });
    await page.evaluate(() => localStorage.setItem("sbd.lastSession", "Match1_30000_37500"));
    await page.reload({ waitUntil: "networkidle0" });
    await page.waitForSelector(".sbd-work");
    await new Promise((r) => setTimeout(r, 500));
    const r = await page.evaluate(() => {
      const R = (s) => { const e = document.querySelector(s); if (!e) return null; const b = e.getBoundingClientRect(); return [Math.round(b.top), Math.round(b.bottom), Math.round(b.left), Math.round(b.right)]; };
      const row4 = document.querySelector(".sbd-tool-row4");
      return { stage: R(".fv-stage"), tl: R(".sbd-tl"), bar: R(".shortcut-bar"), side: R(".sbd-side"), list: R(".sbd-list"), labeler: R(".sbd-labeler"),
        row4overflow: row4.scrollWidth - row4.clientWidth, docH: document.documentElement.scrollHeight, vh: innerHeight,
        sideScroll: document.querySelector(".sbd-side").scrollHeight - document.querySelector(".sbd-side").clientHeight,
        ctrlOverflow: (() => { const c = document.querySelector(".fv-controls"); return c.scrollWidth - c.clientWidth; })() };
    });
    console.log(w + "x" + h, JSON.stringify(r));
    await page.screenshot({ path: `shots/measure_${w}x${h}.png` });
  }
  await browser.close();
})();
