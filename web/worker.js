// Runs the pipeline off the main thread. A fresh worker is used per run: WASM
// memory never shrinks, so this is the only way to give it back.
import Module from "./node_modules/manifold-3d/manifold.js";
import { load } from "./src/threemf.js";
import { resetClock } from "./src/depth.js";
import { run } from "./src/pipeline.js";
import { setProgressSink } from "./src/progress.js";

console.log = (...a) => postMessage({ type: "log", text: a.join(" ") });

// Progress: at most ~10 messages/s, but always when the step label changes.
let lastT = 0, lastLabel = "";
setProgressSink((f, label) => {
  const now = performance.now();
  if (label === lastLabel && now - lastT < 100) return;
  lastT = now;
  lastLabel = label;
  postMessage({ type: "progress", f, label });
});

const ready = Module().then((w) => {
  w.setup();
  return w;
});

onmessage = async ({ data }) => {
  try {
    const wasm = await ready;
    resetClock();
    const t0 = performance.now();
    const proj = load(new Uint8Array(data.bytes), data.name);
    const { bytes, parts } = run(wasm, proj, data.opts);
    const out = parts.map((p) => ({ ...p, vertices: Float32Array.from(p.vertices), faces: Uint32Array.from(p.faces) }));
    const transfer = [bytes.buffer, ...out.flatMap((p) => [p.vertices.buffer, p.faces.buffer])];
    postMessage({ type: "done", bytes, parts: out, seconds: (performance.now() - t0) / 1000 }, transfer);
  } catch (e) {
    postMessage({ type: "error", text: String(e?.stack || e) });
  }
};
