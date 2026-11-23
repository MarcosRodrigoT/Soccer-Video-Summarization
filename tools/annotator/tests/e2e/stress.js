const puppeteer = require('puppeteer-core');
(async () => {
  const b = await puppeteer.launch({ executablePath: '/usr/bin/google-chrome', headless: 'new', args: ['--no-sandbox'] });
  const p = await b.newPage();
  await p.setViewport({ width: 1366, height: 768 });
  const errs = [];
  p.on('pageerror', e => errs.push('PAGEERROR ' + (e.stack || e.message)));
  p.on('console', m => { if (m.type()==='error' && !/favicon/.test(m.text()+JSON.stringify(m.stackTrace()))) errs.push(m.text() + ' ' + JSON.stringify(m.stackTrace().slice(0,5))); });
  await p.goto('http://localhost:8791/#goles', { waitUntil: 'networkidle2' });
  await p.waitForSelector('.gl-goal.sel');
  for (const d of [0, 30, 80, 150, 300, 700, 1500]) {
    await p.evaluate(() => { location.hash = '#planos'; }); await new Promise(r => setTimeout(r, d));
    await p.evaluate(() => { location.hash = '#goles'; }); await new Promise(r => setTimeout(r, d));
  }
  // cambio rápido de partido
  await p.waitForSelector('.gl-chip[data-id="SA1"]');
  await p.click('.gl-chip[data-id="SA1"]'); await p.click('.gl-chip[data-id="PL1"]');
  await p.evaluate(() => { location.hash = '#planos'; }); await new Promise(r => setTimeout(r, 50));
  await p.evaluate(() => { location.hash = '#goles'; });
  await new Promise(r => setTimeout(r, 4000));
  console.log('view has .gl:', await p.evaluate(() => !!document.querySelector('.gl-goal.sel')), 'tabs active:', await p.evaluate(() => document.querySelector('.tab.active').dataset.view));
  console.log(errs.length ? errs.join('\n') : 'sin errores');
  await b.close();
})();
