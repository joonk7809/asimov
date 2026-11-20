// The web export plays its bundled world in a browser with no engine behind it (G7).
//   node client/godot/tests/web_smoke.js build/web [shot.png]
// Serves the export's files from this process (static files only: no websocket, no Python), opens the page in
// headless Chromium and waits for the client's own lines: "playback: ready" (the bundled pack loaded) and
// "playback: playing" (the scrubber is moving). Any request off this host fails the test.
// WEB_SMOKE_CHANNEL=chrome uses an installed Chrome (CI: npm i playwright-core); default is Playwright's Chromium.
const http = require('http');
const fs = require('fs');
const path = require('path');
let pw;
try { pw = require('playwright'); } catch (e) { pw = require('playwright-core'); }

const root = path.resolve(process.argv[2] || 'build/web');
const shot = process.argv[3];
const TYPES = {'.html': 'text/html', '.js': 'text/javascript', '.wasm': 'application/wasm', '.png': 'image/png',
  '.pck': 'application/octet-stream'};

(async () => {
  const server = http.createServer((req, res) => {
    const file = path.join(root, path.normalize(decodeURIComponent(req.url.split('?')[0])).replace(/^(\.\.[/\\])+/, ''));
    const f = req.url === '/' ? path.join(root, 'index.html') : file;
    fs.readFile(f, (err, data) => {
      if (err) { res.writeHead(404); res.end(); return; }
      res.writeHead(200, {'Content-Type': TYPES[path.extname(f)] || 'application/octet-stream'});
      res.end(data);
    });
  });
  await new Promise(r => server.listen(0, '127.0.0.1', r));
  const origin = `http://127.0.0.1:${server.address().port}`;
  const args = ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'];
  const channel = process.env.WEB_SMOKE_CHANNEL;
  const browser = await pw.chromium.launch(channel ? {channel, args} : {args});
  const page = await browser.newPage({viewport: {width: 1280, height: 800}});
  const lines = [], outside = [], failed = [];
  page.on('console', m => { lines.push(m.text()); });
  page.on('pageerror', e => { lines.push('pageerror: ' + e.message); });
  page.on('request', r => { if (!r.url().startsWith(origin) && !r.url().startsWith('data:') && !r.url().startsWith('blob:')) outside.push(r.url()); });
  page.on('websocket', w => { outside.push(w.url()); });
  page.on('response', r => { if (r.status() >= 400) failed.push(`${r.status()} ${r.url()}`); });
  let ok = true;
  const seen = (re, ms) => new Promise(resolve => {
    const t0 = Date.now();
    const poll = () => lines.some(l => re.test(l)) ? resolve(true) : Date.now() - t0 > ms ? resolve(false) : setTimeout(poll, 100);
    poll();
  });
  await page.goto(origin + '/');
  for (const re of [/^playback: ready, engine \S+, \d+ villagers, view ticks 0-\d+/, /^playback: playing, view tick \d+/]) {
    if (!(await seen(re, 120000))) { console.error('never printed ' + re); ok = false; break; }
  }
  if (shot) await page.screenshot({path: shot});
  if (outside.length) { console.error('reached outside its own files: ' + outside.join(', ')); ok = false; }
  if (failed.length) { console.error('failed requests: ' + failed.join(', ')); ok = false; }
  console.log(lines.filter(l => /playback|rror/.test(l)).join('\n'));
  await browser.close();
  server.close();
  process.exit(ok ? 0 : 1);
})();
