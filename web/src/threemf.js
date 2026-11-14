// Read painted 3MF projects (Bambu Studio / OrcaSlicer). Port of threemf.py.
//
// Parsing is regex-based: the mesh files are large (100+ MB of XML) and a
// DOM parser would be slow and memory-hungry; the layout Bambu writes is regular.

import { unzipSync } from "../node_modules/fflate/esm/browser.js";

const dec = new TextDecoder();

// ------------------------------------------------------------- transforms
// 4x4, row-vector convention (p' = [p,1] @ M), stored row-major in Float64Array(16).

export function identity() {
  const m = new Float64Array(16);
  m[0] = m[5] = m[10] = m[15] = 1;
  return m;
}

export function parseTransform(s) {
  const m = identity();
  if (!s) return m;
  const v = s.trim().split(/\s+/).map(Number);
  if (v.length !== 12) throw new Error(`bad transform: ${s}`);
  for (let r = 0; r < 4; r++) for (let c = 0; c < 3; c++) m[r * 4 + c] = v[r * 3 + c];
  return m;
}

export function matmul(a, b) {
  const m = new Float64Array(16);
  for (let r = 0; r < 4; r++)
    for (let c = 0; c < 4; c++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[r * 4 + k] * b[k * 4 + c];
      m[r * 4 + c] = s;
    }
  return m;
}

export function invert(a) {
  // Gauss-Jordan on [A | I].
  const n = 4, m = new Float64Array(32);
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) m[r * 8 + c] = a[r * 4 + c];
    m[r * 8 + 4 + r] = 1;
  }
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(m[r * 8 + c]) > Math.abs(m[p * 8 + c])) p = r;
    if (p !== c) for (let k = 0; k < 8; k++) [m[c * 8 + k], m[p * 8 + k]] = [m[p * 8 + k], m[c * 8 + k]];
    const d = m[c * 8 + c];
    for (let k = 0; k < 8; k++) m[c * 8 + k] /= d;
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = m[r * 8 + c];
      if (f) for (let k = 0; k < 8; k++) m[r * 8 + k] -= f * m[c * 8 + k];
    }
  }
  const out = new Float64Array(16);
  for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) out[r * 4 + c] = m[r * 8 + 4 + c];
  return out;
}

/** Transform a flat xyz Float64Array; returns a new array. */
export function applyTransform(pts, m) {
  const out = new Float64Array(pts.length);
  for (let i = 0; i < pts.length; i += 3) {
    const x = pts[i], y = pts[i + 1], z = pts[i + 2];
    out[i] = x * m[0] + y * m[4] + z * m[8] + m[12];
    out[i + 1] = x * m[1] + y * m[5] + z * m[9] + m[13];
    out[i + 2] = x * m[2] + y * m[6] + z * m[10] + m[14];
  }
  return out;
}

// ------------------------------------------------------------------ parsing

function attr(tag, name) {
  const m = tag.match(new RegExp(`\\s${name.replace(":", "\\:")}="([^"]*)"`));
  return m ? m[1] : null;
}

function metadata(block) {
  const out = {};
  for (const m of block.matchAll(/<metadata key="([^"]*)" value="([^"]*)"\s*\/>/g)) out[m[1]] = m[2];
  return out;
}

function parseMesh(xml) {
  // Vertices: Bambu always writes x, y, z in this order.
  const vs = [];
  const vre = /<vertex\s+x="([^"]+)"\s+y="([^"]+)"\s+z="([^"]+)"/g;
  let m;
  while ((m = vre.exec(xml))) vs.push(+m[1], +m[2], +m[3]);
  const fs = [];
  const paint = [];
  let painted = 0;
  const tre = /<triangle\s+v1="(\d+)"\s+v2="(\d+)"\s+v3="(\d+)"([^>]*)\/>/g;
  while ((m = tre.exec(xml))) {
    fs.push(+m[1], +m[2], +m[3]);
    let p = null;
    if (m[4]) {
      const pm = m[4].match(/(?:paint_color|mmu_segmentation)="([^"]*)"/);
      if (pm && pm[1]) {
        p = pm[1];
        painted++;
      }
    }
    paint.push(p);
  }
  return { vertices: Float64Array.from(vs), faces: Uint32Array.from(fs), paint, painted };
}

function objectsIn(xml) {
  const out = new Map();
  for (const m of xml.matchAll(/<object\b([^>]*)>([\s\S]*?)<\/object>/g)) out.set(attr(m[1], "id"), { tag: m[1], body: m[2] });
  return out;
}

/** Load a 3MF from bytes (Uint8Array). Keeps the raw zip entries for writing. */
export function load(bytes, name = "model.3mf") {
  const files = unzipSync(bytes);
  const text = (n) => dec.decode(files[n]);
  let rootPath = "3D/3dmodel.model";
  if (files["_rels/.rels"]) {
    const m = text("_rels/.rels").match(/Target="\/?([^"]+)"[^>]*Type="[^"]*\/3dmodel"|Type="[^"]*\/3dmodel"[^>]*Target="\/?([^"]+)"/);
    if (m) rootPath = m[1] || m[2];
  }
  const rootXml = text(rootPath);
  const app = (rootXml.match(/<metadata name="Application">([^<]*)<\/metadata>/) || [, ""])[1].trim();
  const cache = new Map([[rootPath, objectsIn(rootXml)]]);
  const objs = (path) => {
    path = path.replace(/^\//, "");
    if (!cache.has(path)) cache.set(path, objectsIn(text(path)));
    return cache.get(path);
  };

  const settings = new Map();
  if (files["Metadata/model_settings.config"]) {
    const cfg = text("Metadata/model_settings.config");
    for (const m of cfg.matchAll(/<object id="(\d+)">([\s\S]*?)<\/object>/g)) {
      const parts = new Map();
      for (const p of m[2].matchAll(/<part id="(\d+)"([^>]*)>([\s\S]*?)<\/part>/g))
        parts.set(p[1], { meta: metadata(p[3]), subtype: attr(p[2], "subtype") || "normal_part" });
      settings.set(m[1], { meta: metadata(m[2].replace(/<part[\s\S]*?<\/part>/g, "")), parts });
    }
  }

  const objects = [];
  const build = rootXml.match(/<build\b[^>]*>([\s\S]*?)<\/build>/)[1];
  for (const im of build.matchAll(/<item\b([^>]*)\/>/g)) {
    const oid = attr(im[1], "objectid");
    const oel = objs(rootPath).get(oid);
    const st = settings.get(oid) || { meta: {}, parts: new Map() };
    const parts = [];
    const mesh = oel.body.match(/<mesh>[\s\S]*<\/mesh>/);
    if (mesh) parts.push({ id: oid, name: attr(oel.tag, "name") || oid, mesh: parseMesh(mesh[0]), transform: identity(), extruder: null, subtype: "normal_part" });
    for (const cm of oel.body.matchAll(/<component\b([^>]*)\/>/g)) {
      const cid = attr(cm[1], "objectid");
      const cpath = attr(cm[1], "p:path") || rootPath;
      const cel = objs(cpath).get(cid);
      const cmesh = cel.body.match(/<mesh>[\s\S]*<\/mesh>/);
      if (!cmesh) throw new Error("nested components are not supported yet");
      const ps = st.parts.get(cid) || { meta: {}, subtype: "normal_part" };
      parts.push({
        id: cid,
        name: ps.meta.name || attr(cel.tag, "name") || cid,
        mesh: parseMesh(cmesh[0]),
        transform: parseTransform(attr(cm[1], "transform")),
        extruder: ps.meta.extruder ? +ps.meta.extruder : null,
        subtype: ps.subtype,
      });
    }
    objects.push({
      id: oid,
      name: st.meta.name || attr(oel.tag, "name") || `object ${oid}`,
      parts,
      transform: parseTransform(attr(im[1], "transform")),
      extruder: +(st.meta.extruder || 1) || 1,
    });
    objects.at(-1).transform0 = objects.at(-1).transform; // as saved, see scaleProject
  }

  let colors = [], types = [], layerHeight = null;
  if (files["Metadata/project_settings.config"]) {
    try {
      const ps = JSON.parse(text("Metadata/project_settings.config"));
      colors = ps.filament_colour || [];
      types = ps.filament_type || [];
      layerHeight = +ps.layer_height || null;
    } catch {}
  }
  return { name, files, rootPath, objects, filamentColors: colors, filamentTypes: types, layerHeight, application: app };
}

/** World-space box of an object's printed parts: [minx, miny, minz, maxx, maxy, maxz], or null. */
export function objectBox(obj, transform = obj.transform) {
  const b = [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];
  for (const part of obj.parts) {
    if (part.subtype !== "normal_part") continue;
    const v = applyTransform(part.mesh.vertices, matmul(part.transform, transform));
    for (let i = 0; i < v.length; i++) {
      const k = i % 3;
      if (v[i] < b[k]) b[k] = v[i];
      if (v[i] > b[k + 3]) b[k + 3] = v[i];
    }
  }
  return b[0] <= b[3] ? b : null;
}

/**
 * Scale every object by [sx, sy, sz] (factors along the plate's axes), about the
 * center of its own footprint, so it stays where it is and on the bed. Only the
 * objects' placement changes (written back by write3mf); meshes and paint are
 * untouched. Always starts from the file's own placement, so it can be repeated.
 */
export function scaleProject(proj, scale) {
  const [sx, sy, sz] = scale;
  if (![sx, sy, sz].every((s) => s > 0 && Number.isFinite(s))) throw new Error(`bad scale: ${scale}`);
  for (const obj of proj.objects) {
    obj.transform = obj.transform0;
    obj.scaled = sx !== 1 || sy !== 1 || sz !== 1;
    const b = obj.scaled && objectBox(obj);
    if (!b) continue;
    const cx = (b[0] + b[3]) / 2, cy = (b[1] + b[4]) / 2, cz = b[2];
    const S = identity();
    (S[0] = sx), (S[5] = sy), (S[10] = sz);
    (S[12] = cx * (1 - sx)), (S[13] = cy * (1 - sy)), (S[14] = cz * (1 - sz));
    obj.transform = matmul(obj.transform0, S);
  }
}

export const partExtruder = (obj, part) => (part.extruder != null ? part.extruder : obj.extruder);
export const filamentColor = (proj, n) => proj.filamentColors[n - 1] || "#808080";
