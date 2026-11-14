// FilaDeep web UI: source and output views with a shared camera and layer slice,
// options <-> URL hash, processing in a Web Worker.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { LineSegments2 } from "three/addons/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/addons/lines/LineSegmentsGeometry.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import { load, applyTransform, matmul, objectBox, partExtruder, filamentColor, scaleProject } from "./src/threemf.js";
import { tessellate } from "./src/paint.js";

const $ = (id) => document.getElementById(id);

// ------------------------------------------------------------------ viewers

const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 5000);
camera.up.set(0, 0, 1);
const target = new THREE.Vector3();
// Layer cut: everything between a bottom and a top height stays visible (like the
// slicer's preview range). An inactive plane sits far away, so it clips nothing.
const FAR = 1e6;
const clipTop = new THREE.Plane(new THREE.Vector3(0, 0, -1), FAR); // keeps z <= top
const clipBot = new THREE.Plane(new THREE.Vector3(0, 0, 1), FAR); // keeps z >= bottom
const BOTH = [clipTop, clipBot];
const cut = { lo: 0, hi: 1 }; // slider positions as fractions of sliceBox's height
const cutTop = () => cut.hi < 1;
const cutBot = () => cut.lo > 0;

class View {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, stencil: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.setClearColor(0x16181c);
    this.renderer.localClippingEnabled = true; // planes are set per material (see partMesh)
    this.scene = new THREE.Scene();
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x444450, 2.2));
    this.sun = new THREE.DirectionalLight(0xffffff, 1.6); // follows the camera, see render()
    this.scene.add(this.sun, this.sun.target);
    this.content = new THREE.Group();
    this.scene.add(this.content);
    // Outline of the cut pieces at the layer height (see updateOutline).
    // Fat lines (plain WebGL lines are always 1 px).
    this.outline = new LineSegments2(new LineSegmentsGeometry(), new LineMaterial({ vertexColors: true, linewidth: 2.5, worldUnits: false }));
    this.outline.renderOrder = 5e5; // after the caps, before translucent parts
    this.outline.frustumCulled = false;
    this.scene.add(this.outline);
    this.outlineKey = "";
    this.version = 0;
    this.grid = new THREE.GridHelper(256, 32, 0x3a404a, 0x2a2e35);
    this.grid.rotation.x = Math.PI / 2;
    this.scene.add(this.grid);
    this.controls = new OrbitControls(camera, canvas);
    this.controls.target = target; // shared
    this.controls.addEventListener("change", renderAll);
    this.controls.addEventListener("end", () => saveCam());
  }
  setContent(objs) {
    for (const c of this.content.children) c.traverse((o) => { o.geometry?.dispose(); o.material?.dispose?.(); });
    this.content.clear();
    for (const o of objs) this.content.add(o);
    this.version++;
  }
  /** Line segments where the active cut planes cross each part's triangles. */
  updateOutline(on) {
    // [cut height, height to draw at: just inside the slab, above/below the cap]
    const cuts = [];
    if (on && cutTop()) cuts.push([clipTop.constant, clipTop.constant - 5e-4]);
    if (on && cutBot()) cuts.push([-clipBot.constant, -clipBot.constant + 5e-4]);
    const key = `${cuts.map((c) => c[0]).join(",")}|${this.version}`;
    if (key === this.outlineKey) return;
    this.outlineKey = key;
    const pos = [], col = [];
    for (const [z, zl] of cuts) {
      for (const group of this.content.children) {
        if (!group.visible) continue;
        const g = group.children[0].geometry;
        const p = g.attributes.position.array, idx = g.index?.array;
        const n = idx ? idx.length / 3 : p.length / 9;
        const { r, g: gg, b } = outlineColor;
        for (let t = 0; t < n; t++) {
          const a = 3 * (idx ? idx[3 * t] : 3 * t), bb = 3 * (idx ? idx[3 * t + 1] : 3 * t + 1), c = 3 * (idx ? idx[3 * t + 2] : 3 * t + 2);
          const za = p[a + 2], zb = p[bb + 2], zc = p[c + 2];
          if ((za > z && zb > z && zc > z) || (za <= z && zb <= z && zc <= z)) continue;
          // Two crossing edges -> one segment.
          let k = 0;
          for (const [u, w] of [[a, bb], [bb, c], [c, a]]) {
            const zu = p[u + 2], zw = p[w + 2];
            if ((zu > z) === (zw > z)) continue;
            const f = (z - zu) / (zw - zu);
            pos.push(p[u] + (p[w] - p[u]) * f, p[u + 1] + (p[w + 1] - p[u + 1]) * f, zl);
            col.push(r, gg, b);
            if (++k === 2) break;
          }
          if (k === 1) pos.length -= 3, col.length -= 3;
        }
      }
    }
    this.outline.geometry.dispose();
    const geo = new LineSegmentsGeometry();
    if (pos.length) {
      geo.setPositions(pos);
      geo.setColors(col);
    }
    this.outline.geometry = geo;
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
    const top = cutTop(), bot = cutBot(), on = top || bot;
    const hl = $("hl").value;
    this.updateOutline(on && hl !== "off");
    this.outline.material.resolution.set(w, h);
    this.outline.visible = on && hl !== "off";
    // "dim": darken the model's surfaces so the cut layer stands out.
    const dim = on && hl === "dim" ? 0.35 : 1;
    for (const group of this.content.children) {
      const m = group.children[0].material;
      if (m.userData.dim !== dim) {
        m.userData.base ??= m.color.clone();
        m.color.copy(m.userData.base).multiplyScalar(dim);
        m.userData.dim = dim;
      }
    }
    // Wireframe: triangle edges over each part, created on first use.
    const wire = $("wire").checked;
    for (const group of this.content.children) {
      if (wire && !group.userData.wire) {
        const front = group.children[0];
        const w = new THREE.Mesh(front.geometry, new THREE.MeshBasicMaterial({ wireframe: true, color: 0x8a8f98, transparent: true, opacity: 0.4, depthWrite: false, clippingPlanes: BOTH }));
        w.renderOrder = front.renderOrder + 0.5;
        group.userData.wire = w;
        group.add(w);
      }
      if (group.userData.wire) group.userData.wire.visible = wire;
    }
    const c = sliceBox.getCenter(new THREE.Vector3()), size = sliceBox.getSize(new THREE.Vector3());
    this.content.traverse((o) => {
      const side = o.userData.cut;
      if (!side) return;
      o.visible = side === "top" ? top : bot;
      if (o.userData.cap) {
        o.position.set(c.x, c.y, side === "top" ? clipTop.constant - 1e-3 : -clipBot.constant + 1e-3);
        o.scale.set(size.x * 2 + 10, size.y * 2 + 10, 1);
      }
    });
    this.renderer.render(this.scene, camera);
  }
}

// Cut-outline color: from a few vivid colors, the one perceptually furthest
// (CIELAB distance) from every filament color in the model.
const OUTLINE_PALETTE = ["#ff00ff", "#00ffff", "#9b30ff", "#ffe600", "#ff7a00", "#ffffff"];
let outlineColor = new THREE.Color("#ff00ff");

function lab(hex) {
  const c = parseColor(hex).color; // linear sRGB
  const x = (0.4124 * c.r + 0.3576 * c.g + 0.1805 * c.b) / 0.9505;
  const y = 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
  const z = (0.0193 * c.r + 0.1192 * c.g + 0.9505 * c.b) / 1.089;
  const f = (t) => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116);
  return [116 * f(y) - 16, 500 * (f(x) - f(y)), 200 * (f(y) - f(z))];
}

function pickOutlineColor(hexes) {
  const used = [...new Set(hexes)].map(lab);
  let best = OUTLINE_PALETTE[0], bestD = -1;
  for (const cand of OUTLINE_PALETTE) {
    const L = lab(cand);
    const d = Math.min(...used.map((u) => Math.hypot(L[0] - u[0], L[1] - u[1], L[2] - u[2])), 1e9);
    if (d > bestD) (bestD = d), (best = cand);
  }
  outlineColor = new THREE.Color(best);
  for (const v of Object.values(views)) v.outlineKey = ""; // recolor
  $("hl").dataset.color = best;
  renderAll();
  return best;
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
const resized = new ResizeObserver(renderAll);
resized.observe(document.body);
for (const v of Object.values(views)) resized.observe(v.canvas);

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
    transparent: alpha < 1, opacity: alpha, depthWrite: alpha >= 1, clippingPlanes: BOTH,
    polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1, // keeps the wireframe lines in front
  }));
  front.renderOrder = alpha < 1 ? 1e6 : 0;
  group.add(front);
  // One stencil pass + cap per cut plane. As in three.js's clipping example, the
  // stencil meshes are clipped by their own plane only, the cap by the other one.
  const k = ++slot * 4;
  for (const [i, side, own, other] of [[0, "top", clipTop, clipBot], [2, "bot", clipBot, clipTop]]) {
    const stencil = (faces, op) => {
      const m = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({
        side: faces, depthWrite: false, depthTest: false, colorWrite: false, clippingPlanes: [own],
        stencilWrite: true, stencilFunc: THREE.AlwaysStencilFunc, stencilFail: op, stencilZFail: op, stencilZPass: op,
      }));
      m.renderOrder = k + i;
      m.userData.cut = side;
      return m;
    };
    const cap = new THREE.Mesh(capGeometry, new THREE.MeshBasicMaterial({
      // Opaque even for translucent filaments: transparent objects render in a
      // later pass, after the stencil state of this part is gone.
      color: color.clone().multiplyScalar(0.9), clippingPlanes: [other],
      stencilWrite: true, stencilRef: 0, stencilFunc: THREE.NotEqualStencilFunc,
      stencilFail: THREE.ReplaceStencilOp, stencilZFail: THREE.ReplaceStencilOp, stencilZPass: THREE.ReplaceStencilOp,
    }));
    if (side === "bot") cap.rotation.x = Math.PI; // faces down, seen from below
    cap.renderOrder = k + i + 1;
    cap.userData.cap = true;
    cap.userData.cut = side;
    cap.onAfterRender = (r) => r.clearStencil();
    group.add(stencil(THREE.BackSide, THREE.IncrementWrapStencilOp), stencil(THREE.FrontSide, THREE.DecrementWrapStencilOp), cap);
  }
  return group;
}

let bounds = new THREE.Box3();

// View state kept in the link (see toHash): camera when moved away from the
// default view, as "px,py,pz,tx,ty,tz"; the cut heights and highlight mode.
let camSaved = null;
let pendingCam = null, wantCut = null; // from the link, applied once the model is shown
function saveCam(on = true) {
  camSaved = on ? [...camera.position.toArray(), ...target.toArray()].map((x) => +x.toFixed(2)).join(",") : null;
  syncHash();
}
function applyCam(str) {
  const v = str.split(",").map(Number);
  if (v.length !== 6 || v.some(Number.isNaN)) return;
  camera.position.set(v[0], v[1], v[2]);
  target.set(v[3], v[4], v[5]);
  views.src.controls.update();
  views.out.controls.update();
  camSaved = str;
  renderAll();
}

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
document.querySelectorAll("[data-view]").forEach((b) => (b.onclick = () => {
  frameView(b.dataset.view);
  saveCam(b.dataset.view !== "iso"); // iso is the default: nothing to store
}));

// ------------------------------------------------------------------ slicing

let sliceBox = new THREE.Box3(); // source and output together

/** Set the sliders' range (e.g. a shell makes the output taller), keeping the cuts at their heights. */
function setSliceRange(box, reset = false) {
  const old = sliceBox;
  sliceBox = box.clone();
  if (reset || old.isEmpty()) {
    cut.lo = 0;
    cut.hi = 1;
  } else {
    const h = sliceBox.max.z - sliceBox.min.z || 1;
    const conv = (f) => Math.min(1, Math.max(0, (old.min.z + f * (old.max.z - old.min.z) - sliceBox.min.z) / h));
    if (cutBot()) cut.lo = conv(cut.lo);
    if (cutTop()) cut.hi = conv(cut.hi);
  }
  if (wantCut) {
    const h = sliceBox.max.z - sliceBox.min.z || 1;
    const frac = (z) => (z - sliceBox.min.z) / h;
    if (wantCut[0] != null) cut.lo = Math.min(1 - GAP, Math.max(0, frac(wantCut[0])));
    if (wantCut[1] != null) cut.hi = Math.min(1, Math.max(cut.lo + GAP, frac(wantCut[1])));
    wantCut = null;
  }
  updateSlice();
}

const zAt = (f) => sliceBox.min.z + (sliceBox.max.z - sliceBox.min.z) * f;

function updateSlice() {
  clipTop.constant = cutTop() ? zAt(cut.hi) : FAR;
  clipBot.constant = cutBot() ? -zAt(cut.lo) : FAR;
  const empty = sliceBox.isEmpty();
  for (const s of sliders) s.update(empty);
  $("sliceZ").textContent = empty || !(cutTop() || cutBot()) ? "no cut" : `z ${zAt(cut.lo).toFixed(2)} – ${zAt(cut.hi).toFixed(2)} mm`;
  $("cutReset").hidden = empty || !(cutTop() || cutBot());
  renderAll();
}

/** Vertical two-handle slider over the right edge of a view; all instances share `cut`. */
class CutSlider {
  constructor(view) {
    const el = document.createElement("div");
    el.className = "zslider";
    el.dataset.tip = "Layer range. Drag the top handle down to cut from above, the bottom handle up to cut from below; the cut faces show each part's filament color. Both views follow. Double-click a handle to reset it; arrow keys move the focused handle (Shift: faster).";
    el.innerHTML = '<div class="track"><div class="range"></div></div>' +
      '<div class="handle" data-k="hi" tabindex="0"><span></span></div><div class="handle" data-k="lo" tabindex="0"><span></span></div>';
    view.append(el);
    this.el = el;
    this.track = el.querySelector(".track");
    this.range = el.querySelector(".range");
    this.handles = { hi: el.querySelector('[data-k="hi"]'), lo: el.querySelector('[data-k="lo"]') };
    const fracAt = (y) => {
      const r = this.track.getBoundingClientRect();
      return Math.min(1, Math.max(0, 1 - (y - r.top) / r.height));
    };
    const drag = (k, e, off = 0) => {
      el.setPointerCapture(e.pointerId);
      el.classList.add("dragging");
      const move = (ev) => setCut(k, fracAt(ev.clientY - off));
      const up = () => {
        el.removeEventListener("pointermove", move);
        el.classList.remove("dragging");
      };
      el.addEventListener("pointermove", move);
      el.addEventListener("pointerup", up, { once: true });
      el.addEventListener("pointercancel", up, { once: true });
      move(e);
    };
    // Double-click is detected here: pointer capture during the drag sends the
    // browser's dblclick event to the container, not the handle.
    let last = { k: "", t: 0 };
    for (const [k, h] of Object.entries(this.handles)) {
      h.addEventListener("pointerdown", (e) => {
        e.stopPropagation();
        e.preventDefault();
        h.focus();
        const now = performance.now();
        if (last.k === k && now - last.t < 400) {
          last = { k: "", t: 0 };
          setCut(k, k === "hi" ? 1 : 0);
          return;
        }
        last = { k, t: now };
        // A plain click shouldn't nudge the handle: drag relative to where it was grabbed.
        const r = this.track.getBoundingClientRect();
        const off = e.clientY - (r.bottom - cut[k] * r.height);
        drag(k, e, off);
      });
      h.addEventListener("keydown", (e) => {
        const step = (e.shiftKey ? 0.05 : 0.005) * (e.key === "ArrowUp" ? 1 : e.key === "ArrowDown" ? -1 : 0);
        if (!step) return;
        e.preventDefault();
        setCut(k, cut[k] + step);
      });
    }
    // Clicking the track moves the nearer handle there.
    this.track.parentElement.addEventListener("pointerdown", (e) => {
      if (e.target.closest(".handle")) return;
      e.preventDefault();
      const f = fracAt(e.clientY);
      drag(Math.abs(f - cut.hi) <= Math.abs(f - cut.lo) ? "hi" : "lo", e);
    });
  }
  update(empty) {
    this.el.hidden = empty;
    if (empty) return;
    this.range.style.bottom = `${cut.lo * 100}%`;
    this.range.style.top = `${(1 - cut.hi) * 100}%`;
    for (const [k, h] of Object.entries(this.handles)) {
      h.style.bottom = `${cut[k] * 100}%`;
      h.firstChild.textContent = zAt(cut[k]).toFixed(2);
      h.classList.toggle("active", k === "hi" ? cutTop() : cutBot());
    }
  }
}

const GAP = 0.002; // handles can't cross
function setCut(k, f) {
  f = Math.min(1, Math.max(0, f));
  cut[k] = k === "hi" ? Math.max(f, cut.lo + GAP) : Math.min(f, cut.hi - GAP);
  updateSlice();
  syncHash();
}

const sliders = [...document.querySelectorAll(".view")].map((v) => new CutSlider(v));
$("hl").onchange = () => {
  updateSlice();
  syncHash();
};
$("cutReset").onclick = () => {
  cut.lo = 0;
  cut.hi = 1;
  updateSlice();
  syncHash();
};

// ------------------------------------------------------------------ source

let source = null; // { proj, bytes, name, url }
let srcAreas = new Map(); // filament -> surface area (mm²)

// Rough cost model (seconds on an Apple M4 Max, GB peak), fitted to measured
// JS runs at voxel 0.25: black kitsune depth 41 s / 2.9 GB, light kitsune +
// shell 53 s / 3.1 GB, test plate with core+primer+shell 20 s, kitsune.3mf
// (2.4M triangles) 73 s / 5.3 GB. Area terms scale with (0.25 / voxel)².
// Areas in cm².
let srcTris = 0;
function estimate() {
  const el = $("estimate");
  if (!srcAreas.size) return (el.textContent = "");
  const o = readOpts();
  const v = o.voxel > 0 ? o.voxel : 0.25;
  const cm2 = new Map([...srcAreas].map(([f, a]) => [f, a / 100]));
  const total = [...cm2.values()].reduce((a, b) => a + b, 0);
  // Colors that get their own volumes: everything but the core filament
  // (without a core option, the largest area is usually the parts' default).
  const coreF = o.core ? o.core.filament : [...cm2].sort((a, b) => b[1] - a[1])[0][0];
  const colored = [...cm2].filter(([f]) => f !== coreF).reduce((a, [, x]) => a + x, 0);
  const mtris = srcTris / 1e6;
  const cost = (r) => {
    let t = 4 + 14 * mtris + 0.012 * total * r + 0.09 * colored * r;
    if (o.core?.primer) t += 0.1 * colored * r;
    if (o.shell) t += 0.1 * total * r;
    const gb = 1.0 + 1.3 * mtris + 0.0035 * total * r + (o.shell ? 0.8 : 0);
    return { t, gb };
  };
  const { t, gb } = cost((0.25 / v) ** 2);
  const fmt = (s) => (s >= 90 ? `${Math.round(s / 60)} min` : `${Math.round(s)} s`);
  // Heavy: over ~2 min, or near the browser's 4 GB WebAssembly limit.
  const ok = (c) => c.t <= 120 && c.gb <= 3.5;
  const heavy = !ok({ t, gb });
  let text = `est. ~${fmt(t)} · ~${gb.toFixed(1)} GB`;
  if (heavy) {
    // Smallest voxel up to 0.5 mm (coarser loses the thin dark colors) that fixes it.
    let sv = null;
    for (let x = Math.ceil(v * 20 + 1e-9) / 20; x <= 0.5 + 1e-9; x += 0.05) if (ok(cost((0.25 / x) ** 2))) { sv = x; break; }
    if (sv) text += ` — heavy: voxel ${sv.toFixed(2)} would be ~${fmt(cost((0.25 / sv) ** 2).t)}, ~${cost((0.25 / sv) ** 2).gb.toFixed(1)} GB`;
    else if (mtris > 1) text += ` — heavy: the mesh is large (${mtris.toFixed(1)}M triangles); simplify it first`;
    else text += " — heavy: try a larger voxel";
  }
  el.textContent = text;
  el.classList.toggle("warn", heavy);
}

function legend(el, entries, onToggle) {
  el.innerHTML = "";
  for (const { key, hex, label, tip } of entries) {
    const l = document.createElement("label");
    if (tip) l.dataset.tip = tip;
    l.innerHTML = `${onToggle ? '<input type="checkbox" checked>' : ""}<span class="sw" style="background:${parseColor(hex).color.getStyle()};opacity:${Math.max(parseColor(hex).alpha, 0.35)}"></span>${label}`;
    if (onToggle) l.querySelector("input").onchange = (e) => onToggle(key, e.target.checked);
    el.appendChild(l);
  }
}

const dims = (b) => {
  const s = b.getSize(new THREE.Vector3());
  return `${s.x.toFixed(1)} × ${s.y.toFixed(1)} × ${s.z.toFixed(1)} mm`;
};
function sizes(objBoxes) {
  const named = objBoxes.filter(([, b]) => !b.isEmpty());
  if (named.length <= 4) return named.map(([n, b]) => `${n}: ${dims(b)}`).join("\n");
  const all = new THREE.Box3();
  for (const [, b] of named) all.union(b);
  return `${named.length} objects, plate area used: ${dims(all)}`;
}

function showSource(proj) {
  const meshes = [];
  const box = new THREE.Box3();
  const areas = new Map();
  let tris = 0, painted = 0;
  const objBoxes = [];
  for (const obj of proj.objects) {
    const ob = new THREE.Box3();
    objBoxes.push([obj.name, ob]);
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
      ob.union(g.boundingBox);
      meshes.push(partMesh(g, filamentColor(proj, pe), true));
    }
  }
  srcAreas = areas;
  pickOutlineColor([...areas.keys()].map((f) => filamentColor(proj, f)));
  srcTris = tris;
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
    `${proj.application || "unknown app"}  ·  ${n} filament slot(s)` +
    "\n" + sizes(objBoxes) +
    (tris > 1e6 ? `\n⚠ ${(tris / 1e6).toFixed(1)}M triangles: expect slower runs, more memory and a large output file. Simplifying the model first helps (e.g. in the slicer).` : "");
  legend($("srcLegend"), [...areas.keys()].sort((a, b) => a - b).map((f) => ({
    key: f, hex: filamentColor(proj, f), label: `${f} ${proj.filamentTypes[f - 1] || ""} · ${(areas.get(f) / 100).toFixed(1)} cm²`,
    tip: `Filament slot ${f} (${proj.filamentTypes[f - 1] || "?"}, ${filamentColor(proj, f)}): ${(areas.get(f) / 100).toFixed(1)} cm² of the surface uses it, painted or as a part's default.`,
  })), null);
  // Filament pickers.
  for (const sel of document.querySelectorAll("select.fil")) {
    const prev = sel.value || sel.dataset.want;
    sel.innerHTML = proj.filamentColors.map((c, i) => `<option value="${i + 1}">${i + 1} ${proj.filamentTypes[i] || ""} ${c}</option>`).join("");
    if (prev && +prev <= n) sel.value = prev;
  }
  if (!$("shellF").dataset.want && !$("shellF").dataset.touched) $("shellF").value = String(n);
  $("run").disabled = false;
  estimate();
  setSliceRange(box, true);
  const cam = pendingCam;
  pendingCam = camSaved = null;
  frameView("iso");
  if (cam) applyCam(cam);
}

function clearResult() {
  $("status").textContent = "";
  result = null;
  $("save").disabled = true;
  $("outLegend").innerHTML = "";
}

async function setSource(bytes, name, url = null) {
  $("status").textContent = "loading…";
  try {
    const proj = load(new Uint8Array(bytes), name);
    source = { proj, bytes, name, url };
    // The mm fields of Scale show the largest object.
    scaleBase = null;
    for (const obj of proj.objects) {
      const b = objectBox(obj);
      if (!b) continue;
      const size = [b[3] - b[0], b[4] - b[1], b[5] - b[2]];
      if (!scaleBase || size[0] * size[1] * size[2] > scaleBase.size[0] * scaleBase.size[1] * scaleBase.size[2]) scaleBase = { name: obj.name, size };
    }
    scaleProject(proj, scale);
    showScale();
    showSource(proj);
    clearResult();
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

$("openUrl").onclick = async () => {
  const url = prompt("Web address of a .3mf file (the server must allow CORS):", source?.url ?? "")?.trim();
  if (!url) return;
  resetScale();
  await loadUrl(url);
  syncHash();
};
$("srcFile").onchange = async (e) => {
  const f = e.target.files[0];
  if (f) resetScale(), await setSource(await f.arrayBuffer(), f.name);
};
document.addEventListener("dragover", (e) => { e.preventDefault(); document.body.classList.add("drop"); });
document.addEventListener("dragleave", () => document.body.classList.remove("drop"));
document.addEventListener("drop", async (e) => {
  e.preventDefault();
  document.body.classList.remove("drop");
  const f = e.dataTransfer.files[0];
  if (f) resetScale(), await setSource(await f.arrayBuffer(), f.name);
});

// ------------------------------------------------------------------ scale

let scale = [1, 1, 1]; // factors along x, y, z
let scaleBase = null; // { name, size: [x, y, z] in mm }: the largest object as saved in the file
const AXES = ["x", "y", "z"];
const scaleKey = () => scale.map((s) => +(s * 100).toFixed(3)).join(",");

function showScale() {
  for (const [k, a] of AXES.entries()) {
    $("sp" + a).value = +(scale[k] * 100).toFixed(2);
    $("sm" + a).value = scaleBase ? +(scaleBase.size[k] * scale[k]).toFixed(2) : "";
  }
  $("sreset").hidden = scale.every((s) => s === 1);
  $("smTip").dataset.tip = "The resulting size along X, Y and Z, in mm" +
    (source?.proj.objects.length > 1 && scaleBase ? `: of '${scaleBase.name}', the largest object. The others are scaled by the same percentages.` : ".");
}

/** Apply `scale` to the loaded model and show it again; the output is cleared. */
function rescale(keepView = false) {
  showScale();
  if (!source) return;
  const V = THREE.Vector3;
  const c0 = bounds.getCenter(new V()), r0 = bounds.getSize(new V()).length();
  const pos = camera.position.clone(), tgt = target.clone(), moved = camSaved != null;
  scaleProject(source.proj, scale);
  stopJob?.();
  clearResult();
  showSource(source.proj);
  if (keepView && moved && r0 > 0) {
    // Same view of the resized model.
    const k = bounds.getSize(new V()).length() / r0;
    target.copy(bounds.getCenter(new V())).addScaledVector(tgt.clone().sub(c0), k);
    camera.position.copy(target).addScaledVector(pos.sub(tgt), k);
    views.src.controls.update();
    views.out.controls.update();
    saveCam();
    renderAll();
  }
}

/** A model the user opens starts at its own size. */
function resetScale() {
  scale = [1, 1, 1];
  $("suni").checked = true;
}

function editScale(k, f) {
  if (!(f > 0) || !Number.isFinite(f)) return showScale(); // not a size: put the old value back
  const r = f / scale[k];
  scale = $("suni").checked ? scale.map((s) => s * r) : scale.map((s, i) => (i === k ? f : s));
  scale = scale.map((s) => +s.toFixed(5)); // what the link can hold
  rescale(true);
  syncHash();
}
for (const [k, a] of AXES.entries()) {
  $("sp" + a).onchange = (e) => editScale(k, +e.target.value / 100);
  $("sm" + a).onchange = (e) => (scaleBase ? editScale(k, +e.target.value / scaleBase.size[k]) : showScale());
}
$("sreset").onclick = () => {
  scale = [1, 1, 1];
  $("suni").checked = true;
  rescale(true);
  syncHash();
};

// ------------------------------------------------------------------ options <-> URL

// Output mode: "layers" (depth / core / shell) or "inlays"; the tabs next to OUTPUT.
let mode = "layers";
function setMode(m) {
  mode = m === "inlays" ? "inlays" : "layers";
  $("right").querySelector(".controls").dataset.mode = mode;
  for (const b of document.querySelectorAll(".tabs button")) b.classList.toggle("active", b.dataset.mode === mode);
}
/** Filament slots typed into the inlays box; null = all painted filaments. */
function inlayFilaments() {
  const f = $("inlayF").value.split(",").map((s) => +s.trim()).filter((n) => Number.isInteger(n) && n > 0);
  return f.length ? f : null;
}

function readOpts() {
  const num = (id) => +$(id).value;
  const opts = {
    dmin: num("dmin"), dmax: num("dmax"), voxel: num("voxel"),
    depths: Object.fromEntries($("depths").value.split(",").map((s) => s.trim()).filter(Boolean).map((kv) => kv.split("=").map(Number))),
    object: $("obj").value.trim() || null, core: null, shell: null, inlay: null, fewer: $("fewer").checked,
    scale: scale.every((s) => s === 1) ? null : scale,
  };
  if (mode === "inlays") {
    // Inlays replace the layer options (a core or shell would bring color changes back or cover the pieces).
    opts.inlay = { filaments: inlayFilaments(), depth: num("idepth"), clearance: num("iclear"), leftover: $("ileft").value };
    return opts;
  }
  if ($("coreOn").checked)
    opts.core = { filament: num("coreF"), skin: num("skin"), primer: $("primerOn").checked ? { filament: num("primerF"), thickness: num("pt") } : null };
  if ($("shellOn").checked) opts.shell = { filament: num("shellF"), thickness: num("st"), keepBottom: $("kb").checked };
  return opts;
}

const FIELDS = { dmin: "dmin", dmax: "dmax", depths: "depths", voxel: "voxel", obj: "obj", skin: "skin", pt: "pt", st: "st", idepth: "idepth", iclear: "iclear" };

let autoRun = false; // run=1 in the link

function toHash() {
  const p = new URLSearchParams();
  if (source?.url) p.set("src", source.url);
  for (const [k, id] of Object.entries(FIELDS)) if ($(id).value !== $(id).defaultValue) p.set(k, $(id).value);
  // Scale in percent: one number, or "x,y,z" when the axes differ.
  if (!scale.every((s) => s === 1)) p.set("scale", scale.every((s) => s === scale[0]) ? scaleKey().split(",")[0] : scaleKey());
  if (mode === "inlays") {
    p.set("inlay", inlayFilaments()?.join(",") ?? "all");
    if ($("ileft").value !== "paint") p.set("ileft", $("ileft").value);
  } else {
    if ($("coreOn").checked) p.set("core", $("coreF").value);
    if ($("coreOn").checked && $("primerOn").checked) p.set("primer", $("primerF").value);
    if ($("shellOn").checked) p.set("shell", $("shellF").value);
    if ($("shellOn").checked && $("kb").checked) p.set("kb", "1");
  }
  // View: camera, cut heights in mm ("bottom,top", empty = no cut), highlight.
  if (camSaved) p.set("cam", camSaved);
  if (!sliceBox.isEmpty() && (cutTop() || cutBot())) p.set("cut", `${cutBot() ? zAt(cut.lo).toFixed(2) : ""},${cutTop() ? zAt(cut.hi).toFixed(2) : ""}`);
  if ($("hl").value !== "outline") p.set("hl", $("hl").value);
  if ($("bg").value !== $("bg").defaultValue) p.set("bg", $("bg").value.slice(1));
  if ($("fewer").checked) p.set("fewer", "1");
  if ($("wire").checked) p.set("wire", "1");
  if (autoRun) p.set("run", "1"); // kept as given, so a reload or an edited address runs again
  return p.toString().replace(/%2F/g, "/").replace(/%3A/g, ":").replace(/%2C/g, ",");
}

function setBackground() {
  for (const v of Object.values(views)) v.renderer.setClearColor($("bg").value);
  renderAll();
}
$("bg").oninput = setBackground;
$("wire").onchange = () => {
  renderAll();
  syncHash();
};
$("bg").onchange = () => syncHash();

/** The link's parameters; when one is repeated, the last one wins (handy for appending overrides). */
function hashParams() {
  const last = new Map(new URLSearchParams(location.hash.slice(1)));
  return new URLSearchParams([...last]);
}

function fromHash() {
  const p = hashParams();
  for (const [k, id] of Object.entries(FIELDS)) if (p.has(k)) $(id).value = p.get(k);
  const pick = (on, sel, key) => {
    $(on).checked = p.has(key);
    if (!p.has(key)) return;
    $(sel).dataset.want = p.get(key);
    if ([...$(sel).options].some((o) => o.value === p.get(key))) $(sel).value = p.get(key); // model already loaded
  };
  pick("coreOn", "coreF", "core");
  pick("primerOn", "primerF", "primer");
  pick("shellOn", "shellF", "shell");
  setMode(p.has("inlay") ? "inlays" : "layers");
  if (p.has("inlay")) $("inlayF").value = p.get("inlay") === "all" ? "" : p.get("inlay");
  if (["paint", "body"].includes(p.get("ileft"))) $("ileft").value = p.get("ileft");
  $("kb").checked = p.get("kb") === "1";
  $("fewer").checked = p.get("fewer") === "1";
  const sc = (p.get("scale") ?? "100").split(",").map((x) => +x / 100);
  scale = sc.length === 3 ? sc : [sc[0], sc[0], sc[0]];
  if (!scale.every((s) => s > 0 && Number.isFinite(s))) scale = [1, 1, 1];
  $("suni").checked = scale.every((s) => s === scale[0]);
  showScale();
  pendingCam = p.get("cam");
  if (p.has("cut")) {
    const [lo, hi] = p.get("cut").split(",").map((x) => (x === "" || x == null || Number.isNaN(+x) ? null : +x));
    wantCut = [lo ?? null, hi ?? null];
  }
  if (["outline", "dim", "off"].includes(p.get("hl"))) $("hl").value = p.get("hl");
  if (/^#?[0-9a-f]{6}$/i.test(p.get("bg") ?? "")) $("bg").value = "#" + p.get("bg").replace("#", "").toLowerCase();
  $("wire").checked = p.get("wire") === "1";
  autoRun = p.get("run") === "1";
  setBackground();
  return p;
}

function syncHash() {
  history.replaceState(null, "", "#" + toHash());
}
for (const b of document.querySelectorAll(".tabs button")) b.onclick = () => {
  setMode(b.dataset.mode);
  syncHash();
  estimate();
};
document.querySelectorAll(".controls input, .controls select").forEach((el) => el.addEventListener("change", () => {
  if (el.id === "shellF") el.dataset.touched = "1";
  if (el.id !== "srcFile") syncHash();
  estimate();
}));

$("copyLink").onclick = async () => {
  syncHash();
  await navigator.clipboard.writeText(location.href);
  $("status").textContent = source && !source.url
    ? "link copied — note: the model was opened from a file, so the link carries only the options"
    : "link copied";
};

// QR code of the current link (the library is loaded on first use).
async function showQr() {
  syncHash();
  const { default: qrcode } = await import("./node_modules/qrcode-generator/dist/qrcode.mjs");
  const qr = qrcode(0, "L");
  qr.addData(location.href);
  qr.make();
  const n = qr.getModuleCount(), m = 3; // m: quiet margin in modules
  let d = "";
  for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) if (qr.isDark(r, c)) d += `M${c + m} ${r + m}h1v1h-1z`;
  $("qrImg").innerHTML = `<svg viewBox="0 0 ${n + 2 * m} ${n + 2 * m}" shape-rendering="crispEdges"><path d="${d}" fill="#000"/></svg>`;
  $("qrNote").textContent = ["localhost", "127.0.0.1"].includes(location.hostname) ? "This is a local address: it only opens on this computer."
    : source && !source.url ? "The model was opened from a file, so the link carries only the options." : "";
  $("qr").hidden = false;
}
$("qrBtn").onclick = (e) => {
  e.stopPropagation();
  if ($("qr").hidden) showQr().catch((err) => ($("status").textContent = `could not make the QR code: ${err.message}`));
  else $("qr").hidden = true;
};
document.addEventListener("click", (e) => {
  if (!$("qr").contains(e.target)) $("qr").hidden = true;
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") $("qr").hidden = true;
});

// ------------------------------------------------------------------ run

let worker = null, result = null;
let stopJob = null; // cleanup for the running job (timer, worker, buttons)

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
  const box = bounds.clone();
  for (const group of views.out.content.children) {
    const g = group.children[0].geometry; // the lit mesh; stencil and cap meshes share it
    g.computeBoundingBox();
    box.union(g.boundingBox);
  }
  setSliceRange(box);
  pickOutlineColor([...new Set([...srcAreas.keys(), ...byF.keys()])].map((f) => filamentColor(proj, f)));
  legend($("outLegend"), [...byF.keys()].sort((a, b) => a - b).map((f) => {
    const names = [...new Set(byF.get(f).map((x) => x.part.name.replace(/^.*? (core|primer|shell|color \d+|inlay)( [\d.]+)?$/, "$1")))].join(", ");
    return { key: f, hex: filamentColor(proj, f), label: `${f} · ${names}`, tip: `Filament ${f}: ${names}. Untick to hide these parts in the output view.` };
  }), (f, on) => {
    for (const m of views.out.content.children) if (m.userData.extruder === f) m.visible = on;
    views.out.version++;
    renderAll();
  });
  renderAll();
}

function runJob() {
  if (!source) return;
  stopJob?.();
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
  $("progress").classList.add("running");
  paint();
  const tick = setInterval(paint, 500);
  const me = new Worker("worker.js", { type: "module" });
  const done = () => {
    clearInterval(tick);
    me.terminate();
    if (worker !== me) return; // a newer job owns the UI
    worker = null;
    stopJob = null;
    $("progress").classList.remove("running");
    $("run").disabled = false;
    $("cancel").hidden = true;
  };
  stopJob = done;
  worker = me;
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
      result = { bytes: data.bytes, name: source.name.replace(/\.3mf$/i, "") + "-filadeep.3mf" };
      $("save").disabled = false;
      showOutput(data.parts, source.proj);
      const outBox = new THREE.Box3();
      for (const group of views.out.content.children) outBox.union(group.children[0].geometry.boundingBox);
      $("status").textContent = `done in ${data.seconds.toFixed(1)} s · ${(data.bytes.length / 1e6).toFixed(1)} MB · output ${dims(outBox)}, ${data.parts.reduce((n, q) => n + q.faces.length / 3, 0).toLocaleString("en-US")} triangles`;
    } else if (data.type === "error") {
      done();
      $("status").textContent = "failed (see log)";
      $("log").textContent += data.text + "\n";
      showLog(true);
    }
  };
  worker.onerror = (e) => {
    done();
    $("status").textContent = `worker error: ${e.message || "out of memory?"}`;
  };
  worker.postMessage({ bytes: source.bytes.slice(0), name: source.name, opts: readOpts() });
}

$("run").onclick = runJob;
function showLog(on = $("logPanel").hidden) {
  $("logPanel").hidden = !on;
  $("logBtn").classList.toggle("active", on);
  if (on) $("log").scrollTop = 1e9;
}
$("logBtn").onclick = () => showLog();
$("closeLog").onclick = () => showLog(false);
$("copyLog").onclick = async () => {
  await navigator.clipboard.writeText($("log").textContent);
  $("copyLog").textContent = "Copied";
  setTimeout(() => ($("copyLog").textContent = "Copy"), 1200);
};
$("cancel").onclick = () => {
  stopJob?.();
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

// ------------------------------------------------------------------ tooltips
// Elements carry their help text in data-tip; one floating box shows it after
// a short hover (or on keyboard focus). The 3D views only show their tip while
// the pointer rests, not while dragging.

const tip = $("tip");
let tipTimer = 0, tipFor = null, mouseX = 0;
document.addEventListener("pointermove", (e) => (mouseX = e.clientX), { passive: true });
/** "Default: …" for a tip on an option that is stored in the link (empty otherwise). */
function defaultNote(el) {
  if (!el.closest("#right .controls")) return "";
  const c = el.querySelector("input, select");
  if (!c) return "";
  if (c.type === "checkbox") return `Default: ${c.defaultChecked ? "on" : "off"}.`;
  if (c.type === "number") return `Default: ${c.defaultValue} mm.`;
  if (c.type === "text") return `Default: ${c.defaultValue || "empty"}.`;
  if (c.classList.contains("fil")) return c.id === "shellF" ? "Default: the last filament slot." : "Default: filament slot 1.";
  return "";
}
function showTip(el) {
  const note = defaultNote(el);
  tip.textContent = el.dataset.tip + (note ? `\n${note}` : "");
  tip.hidden = false;
  const r = el.getBoundingClientRect(), t = tip.getBoundingClientRect();
  const view = el.hasAttribute("data-tip-view");
  // Narrow elements: centred under them. Wide ones (headings, views): near the pointer.
  const wide = view || r.width > 360;
  let x = wide && mouseX ? mouseX - 20 : r.left + r.width / 2 - t.width / 2;
  let y = view ? r.top + 32 : r.bottom + 6;
  if (y + t.height > innerHeight - 4) y = r.top - t.height - 6;
  x = Math.max(4, Math.min(x, innerWidth - t.width - 4));
  tip.style.left = `${x}px`;
  tip.style.top = `${y}px`;
}
function hideTip() {
  clearTimeout(tipTimer);
  tip.hidden = true;
  tipFor = null;
}
document.addEventListener("pointerover", (e) => {
  const el = e.target.closest("[data-tip]");
  if (el === tipFor) return;
  hideTip();
  if (!el || e.buttons) return;
  tipFor = el;
  tipTimer = setTimeout(() => showTip(el), el.hasAttribute("data-tip-view") ? 1200 : 450);
});
document.addEventListener("pointerdown", hideTip);
document.addEventListener("wheel", hideTip, { passive: true });
document.addEventListener("focusin", (e) => {
  const el = e.target.closest("[data-tip]");
  if (el && !el.hasAttribute("data-tip-view")) showTip(el);
});
document.addEventListener("focusout", hideTip);

// ------------------------------------------------------------------ start

const DEFAULT_SRC = "models/testshapes.3mf";

async function start() {
  const p = fromHash();
  if (p.has("src")) await loadUrl(p.get("src"));
  else if (!source) await loadUrl(DEFAULT_SRC); // something to look at on a first visit
  if (p.get("run") === "1" && source) runJob();
  renderAll();
}
// The address was edited by hand (our own updates don't fire this): apply it.
window.addEventListener("hashchange", () => {
  const p = hashParams(), src = p.get("src");
  if (src && src !== source?.url) return start();
  const was = scaleKey();
  fromHash();
  if (!source) return;
  if (scaleKey() !== was) rescale(); // shows the model again, with the link's camera and cut
  else {
    setSliceRange(sliceBox, !wantCut); // no cut in the link: show the whole model
    const cam = pendingCam;
    pendingCam = camSaved = null;
    if (cam) applyCam(cam);
    else frameView("iso");
  }
  estimate();
  renderAll();
  if (p.get("run") === "1") runJob();
});
start();
