// Screenshots of the web app from links, in headless Chrome.
//
//   node shot.mjs <link> <out.png> [<link> <out.png> ...] [options]
//
// A link is what "Copy link" gives (or just its "#..." part): it holds the model,
// the options, the camera and the layer cut. Add run=1 to process before the shot.
// The dev server must be running (node serve.mjs).
//
//   --panel out|src|both|page   what to capture (default: out, the output view)
//   --size 800x600              size of the picture in pixels. For out and src the
//                               view fills it; for both and page it is the window.
//   --scale 1                   pixel density: 2 gives a picture twice as large (sharper)
//   --ui                        keep the overlays (sliders, labels) in the picture
//   --base http://localhost:8000/   where "#..." links are opened
//   --timeout 300               seconds to wait for a run
//   CHROME=/path/to/chrome      browser to use (default: installed Google Chrome)
import { spawn } from "node:child_process";
import { existsSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const opts = { panel: "out", size: "", scale: "1", base: "http://localhost:8000/", timeout: "300", ui: false };
const jobs = [];
const argv = process.argv.slice(2);
for (let i = 0; i < argv.length; i++) {
  const a = argv[i];
  if (a === "--ui") opts.ui = true;
  else if (a.startsWith("--") && a.slice(2) in opts) opts[a.slice(2)] = argv[++i];
  else if (a.startsWith("--")) fail(`unknown option ${a}`);
  else jobs.push(a);
}
if (!jobs.length || jobs.length % 2) fail("usage: node shot.mjs <link> <out.png> [<link> <out.png> ...] [--panel out|src|both|page] [--size WxH] [--scale N] [--ui]");
if (!["out", "src", "both", "page"].includes(opts.panel)) fail(`--panel must be out, src, both or page`);
const single = opts.panel === "out" || opts.panel === "src";
const [width, height] = (opts.size || (single ? "800x600" : "1600x1000")).split("x").map(Number);
if (!(width > 0 && height > 0)) fail("--size must look like 800x600");

function fail(msg) {
  console.error(msg);
  process.exit(1);
}

const chromePath = [process.env.CHROME, "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser"]
  .find((p) => p && existsSync(p));
if (!chromePath) fail("Chrome not found: set CHROME=/path/to/chrome");

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const port = 9400 + Math.floor(Math.random() * 500);
const profile = mkdtempSync(join(tmpdir(), "filadeep-shot-"));
const chrome = spawn(chromePath, ["--headless=new", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", `--remote-debugging-port=${port}`,
  `--user-data-dir=${profile}`, `--window-size=${width},${height}`, "about:blank"], { stdio: "ignore" });
const quit = (code) => {
  chrome.kill();
  try { rmSync(profile, { recursive: true, force: true }); } catch {}
  process.exit(code);
};

let ws;
for (let i = 0; i < 75 && !ws; i++) {
  await sleep(200);
  try {
    const page = (await (await fetch(`http://127.0.0.1:${port}/json`)).json()).find((t) => t.type === "page");
    if (page) ws = new WebSocket(page.webSocketDebuggerUrl);
  } catch {}
}
if (!ws) {
  console.error("could not start Chrome");
  quit(1);
}
await new Promise((r) => (ws.onopen = r));
let id = 0;
const pending = new Map();
ws.onmessage = (m) => {
  const d = JSON.parse(m.data);
  if (d.id && pending.has(d.id)) pending.get(d.id)(d), pending.delete(d.id);
};
const send = (method, params = {}) => new Promise((r) => {
  pending.set(++id, r);
  ws.send(JSON.stringify({ id, method, params }));
});
const evalJs = async (e) => (await send("Runtime.evaluate", { expression: e, awaitPromise: true, returnByValue: true })).result?.result?.value;

await send("Page.enable");
await send("Runtime.enable");
await send("Emulation.setDeviceMetricsOverride", { width, height, deviceScaleFactor: +opts.scale, mobile: false });

let failed = 0;
for (let j = 0; j < jobs.length; j += 2) {
  const link = jobs[j], out = jobs[j + 1];
  const url = /^[a-z]+:\/\//i.test(link) ? link : opts.base + (link.startsWith("#") ? link : `#${link}`);
  const wantRun = /[#&]run=1(&|$)/.test(url);
  await send("Page.navigate", { url: "about:blank" }); // a changed hash alone would not reload the page
  await send("Page.navigate", { url });
  const t0 = Date.now();
  let state = "";
  while ((Date.now() - t0) / 1000 < +opts.timeout) {
    await sleep(300);
    state = await evalJs(`(() => {
      const s = document.getElementById("status")?.textContent || "", run = document.getElementById("run");
      if (/could not|failed|error/i.test(s)) return "error: " + s;
      if (!run || run.disabled) return "";
      return ${wantRun} ? (s.startsWith("done in") ? "ok: " + s : "") : "ok: loaded";
    })()`) || "";
    if (state) break;
  }
  if (!state.startsWith("ok")) {
    console.error(`${out}: ${state || "timed out"}`);
    failed++;
    continue;
  }
  let css = opts.ui ? "" : ".zslider,.tag,.logpanel,#tip{display:none!important}";
  // One view: stretch it over the whole window, so the picture is exactly --size.
  if (single) css += `#${opts.panel === "out" ? "right" : "left"} .view{position:fixed!important;inset:0;z-index:100;border:0}html,body{overflow:hidden!important}`;
  await evalJs(`document.head.insertAdjacentHTML("beforeend", ${JSON.stringify(`<style>${css}</style>`)})`);
  await evalJs(`new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(() => setTimeout(r, 300))))`);
  const sel = { out: "#cvOut", src: "#cvSrc", both: "main > .panel .view", page: "" }[opts.panel];
  const clip = sel && (await evalJs(`(() => {
    const rs = [...document.querySelectorAll(${JSON.stringify(sel)})].map((e) => e.getBoundingClientRect());
    const x = Math.min(...rs.map((r) => r.left)), y = Math.min(...rs.map((r) => r.top));
    return { x, y, width: Math.max(...rs.map((r) => r.right)) - x, height: Math.max(...rs.map((r) => r.bottom)) - y, scale: 1 };
  })()`));
  const shot = await send("Page.captureScreenshot", { format: "png", ...(clip ? { clip } : {}) });
  writeFileSync(out, Buffer.from(shot.result.data, "base64"));
  console.log(`${out}  (${state.slice(4)})`);
}
quit(failed ? 1 : 0);
