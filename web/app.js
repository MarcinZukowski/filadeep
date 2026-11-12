// LayerFun web UI: source and output views with a shared camera and layer slice,
// options <-> URL hash, processing in a Web Worker.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { load, applyTransform, matmul, partExtruder, filamentColor } from "./src/threemf.js";
import { tessellate } from "./src/paint.js";

const $ = (id) => document.getElementById(id);

// ------------------------------------------------------------------ viewers

const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 5000);
camera.up.set(0, 0, 1);
const target = new THREE.Vector3();
const clip = new THREE.Plane(new THREE.Vector3(0, 0, -1), 0); // keeps z <= constant

class View {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, stencil: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.setClearColor(0x16181c);
    this.scene = new THREE.Scene();
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x444450, 2.2));
    this.sun = new THREE.DirectionalLight(0xffffff, 1.6); // follows the camera, see render()
    this.scene.add(this.sun, this.sun.target);
    this.content = new THREE.Group();
    this.scene.add(this.content);
    this.grid = new THREE.GridHelper(256, 32, 0x3a404a, 0x2a2e35);
    this.grid.rotation.x = Math.PI / 2;
    this.scene.add(this.grid);
    this.controls = new OrbitControls(camera, canvas);
    this.controls.target = target; // shared
    this.controls.addEventListener("change", renderAll);
  }
  setContent(objs) {
    for (const c of this.content.children) c.traverse((o) => { o.geometry?.dispose(); o.material?.dispose?.(); });
    this.content.clear();
    for (const o of objs) this.content.add(o);
  }
  render() {
    const { clientWidth: w, clientHeight: h } = this.canvas;
    if (!w || !h) return;
    if (this.canvas.width !== Math.round(w * devicePixelRatio) || this.canvas.height !== Math.round(h * devicePixelRatio)) this.renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    // Headlight, slightly from above.
    this.sun.position.copy(camera.position).add(new THREE.Vector3(0, 0, camera.position.distanceTo(target) * 0.5));
    this.sun.target.position.copy(target);
    const on = $("sliceOn").checked;
    this.renderer.clippingPlanes = on ? [clip] : [];
    const c = bounds.getCenter(new THREE.Vector3()), size = bounds.getSize(new THREE.Vector3());
    this.content.traverse((o) => {
      if (!o.userData.sliceOnly) return;
      o.visible = on;
      if (o.userData.cap) {
        o.position.set(c.x, c.y, clip.constant - 1e-3);
        o.scale.set(size.x * 2 + 10, size.y * 2 + 10, 1);
      }
    });
    this.renderer.render(this.scene, camera);
  }
}

const views = { src: new View($("cvSrc")), out: new View($("cvOut")) };
let frame = 0;
function renderAll() {
  if (frame) return;
  frame = requestAnimationFrame(() => {
    frame = 0;
    views.src.render();
    views.out.render();
  });
}
new ResizeObserver(renderAll).observe(document.body);

function parseColor(hex) {
  const c = (hex || "#808080").replace("#", "");
  const color = new THREE.Color(`#${c.slice(0, 6)}`);
  const alpha = c.length >= 8 ? parseInt(c.slice(6, 8), 16) / 255 : 1;
  return { color, alpha };
}

// A part = lit front faces + a cross-section cap for the layer cut. The cap uses
// the stencil buffer (as in three.js's clipping example): back faces increment,
// front faces decrement, so the stencil is non-zero exactly where the cut plane
// lies inside this part; a plane in the part's color is drawn only there.
// Each part gets its own render-order slot, so nested parts (skin around core)
// produce correct rings.
let slot = 0;
const capGeometry = new THREE.PlaneGeometry(1, 1);

function partMesh(geometry, hex, vertexColors = false) {
  const { color, alpha } = parseColor(hex);
  const group = new THREE.Group();
  const front = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({
    vertexColors, color: vertexColors ? 0xffffff : color, roughness: 0.75, metalness: 0, flatShading: true,
    transparent: alpha < 1, opacity: alpha, depthWrite: alpha >= 1,
  }));
  front.renderOrder = alpha < 1 ? 1e6 : 0;
  const k = ++slot * 3;
  const stencil = (side, op) => {
    const m = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({
      side, depthWrite: false, depthTest: false, colorWrite: false,
      stencilWrite: true, stencilFunc: THREE.AlwaysStencilFunc, stencilFail: op, stencilZFail: op, stencilZPass: op,
    }));
    m.renderOrder = k;
    m.userData.sliceOnly = true;
    return m;
  };
  const cap = new THREE.Mesh(capGeometry, new THREE.MeshBasicMaterial({
    // Opaque even for translucent filaments: transparent objects render in a
    // later pass, after the stencil state of this part is gone.
    color: color.clone().multiplyScalar(0.9),
    stencilWrite: true, stencilRef: 0, stencilFunc: THREE.NotEqualStencilFunc,
    stencilFail: THREE.ReplaceStencilOp, stencilZFail: THREE.ReplaceStencilOp, stencilZPass: THREE.ReplaceStencilOp,
  }));
  cap.renderOrder = k + 1;
  cap.userData.cap = true;
  cap.userData.sliceOnly = true;
  cap.onAfterRender = (r) => r.clearStencil();
  group.add(front, stencil(THREE.BackSide, THREE.IncrementWrapStencilOp), stencil(THREE.FrontSide, THREE.DecrementWrapStencilOp), cap);
  return group;
}

let bounds = new THREE.Box3();

function frameView(name = "iso") {
  if (bounds.isEmpty()) return;
  const c = bounds.getCenter(new THREE.Vector3());
  const r = bounds.getSize(new THREE.Vector3()).length() * 0.5;
  const dirs = { iso: [1, -1.3, 0.9], front: [0, -1, 0.05], back: [0, 1, 0.05], left: [-1, 0, 0.05], right: [1, 0, 0.05], top: [0, -0.001, 1] };
  const d = new THREE.Vector3(...dirs[name]).normalize();
  target.copy(c);
  camera.position.copy(c).addScaledVector(d, r / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2)) * 1.05);
  camera.near = r / 100;
  camera.far = r * 20;
  views.src.controls.update();
  views.out.controls.update();
  renderAll();
}
document.querySelectorAll("[data-view]").forEach((b) => (b.onclick = () => frameView(b.dataset.view)));

// ------------------------------------------------------------------ slicing

function updateSlice() {
  const z0 = bounds.min.z, z1 = bounds.max.z;
  const t = +$("slice").value / 100;
  const z = z0 + (z1 - z0) * t;
  clip.constant = z;
  $("sliceZ").textContent = bounds.isEmpty() ? "–" : `z ${z.toFixed(2)} mm`;
  renderAll();
}
$("slice").oninput = () => {
  $("sliceOn").checked = true;
  updateSlice();
};
$("sliceOn").onchange = updateSlice;

// ------------------------------------------------------------------ source

let source = null; // { proj, bytes, name, url }

function legend(el, entries, onToggle) {
  el.innerHTML = "";
  for (const { key, hex, label } of entries) {
    const l = document.createElement("label");
    l.innerHTML = `${onToggle ? '<input type="checkbox" checked>' : ""}<span class="sw" style="background:${parseColor(hex).color.getStyle()};opacity:${Math.max(parseColor(hex).alpha, 0.35)}"></span>${label}`;
    if (onToggle) l.querySelector("input").onchange = (e) => onToggle(key, e.target.checked);
    el.appendChild(l);
  }
}

function showSource(proj) {
  const meshes = [];
  const box = new THREE.Box3();
  const areas = new Map();
  let tris = 0, painted = 0;
  for (const obj of proj.objects)
    for (const part of obj.parts) {
      if (part.subtype !== "normal_part") continue;
      const { tris: t, states } = tessellate(part.mesh.vertices, part.mesh.faces, part.mesh.paint);
      const world = applyTransform(t, matmul(part.transform, obj.transform));
      const pos = Float32Array.from(world);
      const col = new Float32Array(pos.length);
      const pe = partExtruder(obj, part);
      for (let i = 0; i < states.length; i++) {
        const f = states[i] || pe;
        const c = parseColor(filamentColor(proj, f)).color;
        for (let v = 0; v < 3; v++) col.set([c.r, c.g, c.b], 9 * i + 3 * v);
        const o = 9 * i;
        const ax = pos[o + 3] - pos[o], ay = pos[o + 4] - pos[o + 1], az = pos[o + 5] - pos[o + 2];
        const bx = pos[o + 6] - pos[o], by = pos[o + 7] - pos[o + 1], bz = pos[o + 8] - pos[o + 2];
        const a = 0.5 * Math.hypot(ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx);
        areas.set(f, (areas.get(f) || 0) + a);
      }
      tris += part.mesh.faces.length / 3;
      painted += part.mesh.painted;
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      g.setAttribute("color", new THREE.BufferAttribute(col, 3));
      g.computeBoundingBox();
      box.union(g.boundingBox);
      meshes.push(partMesh(g, filamentColor(proj, pe), true));
    }
  views.src.setContent(meshes);
  views.out.setContent([]);
  bounds = box;
  const c = box.getCenter(new THREE.Vector3()), size = box.getSize(new THREE.Vector3());
  for (const v of Object.values(views)) {
    v.grid.position.set(Math.round(c.x / 10) * 10, Math.round(c.y / 10) * 10, box.min.z);
    v.grid.scale.setScalar(Math.max(size.x, size.y) * 1.6 / 256);
  }
  const n = proj.filamentColors.length;
  $("srcInfo").textContent =
    `${proj.name}  ·  ${proj.objects.length} object(s), ${tris.toLocaleString()} triangles, ${painted.toLocaleString()} painted\n` +
    `${proj.application || "unknown app"}  ·  ${n} filament slot(s)`;
  legend($("srcLegend"), [...areas.keys()].sort((a, b) => a - b).map((f) => ({
    key: f, hex: filamentColor(proj, f), label: `${f} ${proj.filamentTypes[f - 1] || ""} · ${(areas.get(f) / 100).toFixed(1)} cm²`,
  })), null);
  // Filament pickers.
  for (const sel of document.querySelectorAll("select.fil")) {
    const prev = sel.value || sel.dataset.want;
    sel.innerHTML = proj.filamentColors.map((c, i) => `<option value="${i + 1}">${i + 1} ${proj.filamentTypes[i] || ""} ${c}</option>`).join("");
    if (prev && +prev <= n) sel.value = prev;
  }
  if (!$("shellF").dataset.want && !$("shellF").dataset.touched) $("shellF").value = String(n);
  $("run").disabled = false;
  updateSlice();
  frameView("iso");
}

async function setSource(bytes, name, url = null) {
  $("status").textContent = "loading…";
  try {
    const proj = load(new Uint8Array(bytes), name);
    source = { proj, bytes, name, url };
    $("srcUrl").value = url || "";
    showSource(proj);
    $("status").textContent = "";
    result = null;
    $("save").disabled = true;
    $("outLegend").innerHTML = "";
  } catch (e) {
    $("status").textContent = `could not read ${name}: ${e.message}`;
    console.error(e);
  }
}

async function loadUrl(url) {
  $("status").textContent = `fetching ${url}…`;
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const name = decodeURIComponent(new URL(url, location.href).pathname.split("/").pop() || "model.3mf");
    await setSource(await res.arrayBuffer(), name, url);
  } catch (e) {
    $("status").textContent = `could not fetch (${e.message}). The server must allow CORS; otherwise download it and open the file.`;
  }
}

$("loadUrl").onclick = () => $("srcUrl").value && loadUrl($("srcUrl").value.trim());
$("srcUrl").onkeydown = (e) => e.key === "Enter" && $("loadUrl").click();
$("srcFile").onchange = async (e) => {
  const f = e.target.files[0];
  if (f) await setSource(await f.arrayBuffer(), f.name);
};
document.addEventListener("dragover", (e) => { e.preventDefault(); document.body.classList.add("drop"); });
document.addEventListener("dragleave", () => document.body.classList.remove("drop"));
document.addEventListener("drop", async (e) => {
  e.preventDefault();
  document.body.classList.remove("drop");
  const f = e.dataTransfer.files[0];
  if (f) await setSource(await f.arrayBuffer(), f.name);
});

// ------------------------------------------------------------------ options <-> URL

function readOpts() {
  const num = (id) => +$(id).value;
  const opts = {
    dmin: num("dmin"), dmax: num("dmax"), voxel: num("voxel"),
    depths: Object.fromEntries($("depths").value.split(",").map((s) => s.trim()).filter(Boolean).map((kv) => kv.split("=").map(Number))),
    object: $("obj").value.trim() || null, core: null, shell: null,
  };
  if ($("coreOn").checked)
    opts.core = { filament: num("coreF"), skin: num("skin"), primer: $("primerOn").checked ? { filament: num("primerF"), thickness: num("pt") } : null };
  if ($("shellOn").checked) opts.shell = { filament: num("shellF"), thickness: num("st"), keepBottom: $("kb").checked };
  return opts;
}

const FIELDS = { dmin: "dmin", dmax: "dmax", depths: "depths", voxel: "voxel", obj: "obj", skin: "skin", pt: "pt", st: "st" };

function toHash() {
  const p = new URLSearchParams();
  if (source?.url) p.set("src", source.url);
  for (const [k, id] of Object.entries(FIELDS)) if ($(id).value !== $(id).defaultValue) p.set(k, $(id).value);
  if ($("coreOn").checked) p.set("core", $("coreF").value);
  if ($("coreOn").checked && $("primerOn").checked) p.set("primer", $("primerF").value);
  if ($("shellOn").checked) p.set("shell", $("shellF").value);
  if ($("shellOn").checked && $("kb").checked) p.set("kb", "1");
  return p.toString().replace(/%2F/g, "/").replace(/%3A/g, ":");
}

function fromHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  for (const [k, id] of Object.entries(FIELDS)) if (p.has(k)) $(id).value = p.get(k);
  const pick = (on, sel, key) => {
    $(on).checked = p.has(key);
    if (p.has(key)) $(sel).dataset.want = p.get(key);
  };
  pick("coreOn", "coreF", "core");
  pick("primerOn", "primerF", "primer");
  pick("shellOn", "shellF", "shell");
  $("kb").checked = p.get("kb") === "1";
  return p;
}

function syncHash() {
  history.replaceState(null, "", "#" + toHash());
}
document.querySelectorAll(".controls input, .controls select").forEach((el) => el.addEventListener("change", () => {
  if (el.id === "shellF") el.dataset.touched = "1";
  if (el.id !== "srcFile" && el.id !== "srcUrl") syncHash();
}));

$("copyLink").onclick = async () => {
  syncHash();
  await navigator.clipboard.writeText(location.href);
  $("status").textContent = source && !source.url
    ? "link copied — note: the model was opened from a file, so the link carries only the options"
    : "link copied";
};

// ------------------------------------------------------------------ run

let worker = null, result = null;

function showOutput(parts, proj) {
  const byF = new Map();
  for (const p of parts) {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(p.vertices, 3));
    g.setIndex(new THREE.BufferAttribute(p.faces, 1));
    const m = partMesh(g, filamentColor(proj, p.extruder));
    m.userData.extruder = p.extruder;
    (byF.get(p.extruder) ?? byF.set(p.extruder, []).get(p.extruder)).push({ mesh: m, part: p });
  }
  views.out.setContent([...byF.values()].flat().map((x) => x.mesh));
  legend($("outLegend"), [...byF.keys()].sort((a, b) => a - b).map((f) => {
    const names = [...new Set(byF.get(f).map((x) => x.part.name.replace(/^.*? (core|primer|shell|color \d+)$/, "$1")))].join(", ");
    return { key: f, hex: filamentColor(proj, f), label: `${f} · ${names}` };
  }), (f, on) => {
    for (const m of views.out.content.children) if (m.userData.extruder === f) m.visible = on;
    renderAll();
  });
  renderAll();
}

function runJob() {
  if (!source) return;
  worker?.terminate();
  syncHash();
  $("log").textContent = "";
  $("run").disabled = true;
  $("cancel").hidden = false;
  $("save").disabled = true;
  $("status").textContent = "";
  const t0 = performance.now();
  let frac = 0, label = "starting";
  const fmt = (s) => (s >= 60 ? `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}` : `${Math.round(s)} s`);
  const paint = () => {
    const el = (performance.now() - t0) / 1000;
    // Remaining time from the average rate so far; only once there's some signal.
    const eta = frac > 0.05 && el > 2 ? ` · ~${fmt(el / frac - el)} left` : "";
    $("progress").querySelector(".fill").style.width = `${(frac * 100).toFixed(1)}%`;
    $("progress").querySelector(".ptext").textContent = `${Math.floor(frac * 100)}% · ${label} · ${fmt(el)}${eta}`;
  };
  $("progress").hidden = false;
  paint();
  const tick = setInterval(paint, 500);
  const done = () => {
    clearInterval(tick);
    $("progress").hidden = true;
    $("run").disabled = false;
    $("cancel").hidden = true;
    worker.terminate();
    worker = null;
  };
  worker = new Worker("worker.js", { type: "module" });
  worker.onmessage = ({ data }) => {
    if (data.type === "progress") {
      frac = Math.max(frac, data.f);
      label = data.label;
      paint();
    } else if (data.type === "log") {
      $("log").textContent += data.text + "\n";
      $("log").scrollTop = 1e9;
    } else if (data.type === "done") {
      done();
      result = { bytes: data.bytes, name: source.name.replace(/\.3mf$/i, "") + "-layerfun.3mf" };
      $("save").disabled = false;
      $("status").textContent = `done in ${data.seconds.toFixed(1)} s · ${(data.bytes.length / 1e6).toFixed(1)} MB`;
      showOutput(data.parts, source.proj);
    } else if (data.type === "error") {
      done();
      $("status").textContent = "failed (see log)";
      $("log").textContent += data.text + "\n";
      $("logBox").open = true;
    }
  };
  worker.onerror = (e) => {
    done();
    $("status").textContent = `worker error: ${e.message || "out of memory?"}`;
  };
  worker.postMessage({ bytes: source.bytes.slice(0), name: source.name, opts: readOpts() });
}

$("run").onclick = runJob;
$("cancel").onclick = () => {
  worker?.terminate();
  $("progress").hidden = true;
  worker = null;
  $("run").disabled = false;
  $("cancel").hidden = true;
  $("status").textContent = "cancelled";
};
$("save").onclick = () => {
  if (!result) return;
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([result.bytes], { type: "application/vnd.ms-package.3dmanufacturing-3dmodel+xml" }));
  a.download = result.name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 10000);
};

// ------------------------------------------------------------------ start

async function start() {
  const p = fromHash();
  if (p.has("src")) {
    await loadUrl(p.get("src"));
    if (p.get("run") === "1" && source) runJob();
  }
  renderAll();
}
window.addEventListener("hashchange", () => {
  const src = new URLSearchParams(location.hash.slice(1)).get("src");
  if (src && src !== source?.url) start();
});
start();
