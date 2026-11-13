// Per-color depth: turn surface paint into solid color volumes. Port of depth.py
// (see there for the method). Plain loops instead of numpy vectorization.

import { planInlays } from "./inlay.js";
import { KDTree } from "./kdtree.js";
import { Span } from "./progress.js";
import { tessellate } from "./paint.js";
import { applyTransform, filamentColor, invert, matmul, partExtruder } from "./threemf.js";

let T0 = performance.now();
const mem = () => {
  const m = globalThis.process?.memoryUsage?.();
  return m ? ` [rss ${(m.rss / 1e9).toFixed(1)} GB, js heap ${(m.heapUsed / 1e9).toFixed(1)} GB, wasm+buffers ${(m.arrayBuffers / 1e9).toFixed(1)} GB]` : "";
};
export const log = (msg) => console.log(`[${((performance.now() - T0) / 1000).toFixed(1).padStart(6)}s] ${msg}${mem()}`);
export const resetClock = () => (T0 = performance.now());

// ------------------------------------------------------------------- colors

export function luminance(hex) {
  const c = hex.replace("#", "").slice(0, 6);
  const rgb = [0, 2, 4].map((i) => parseInt(c.slice(i, i + 2), 16) / 255);
  if (rgb.some(Number.isNaN)) return 0.5;
  const lin = rgb.map((x) => (x <= 0.04045 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4));
  return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2];
}

export const defaultDepth = (hex, dmin = 0.45, dmax = 1.6) => dmin + (dmax - dmin) * luminance(hex) ** 0.5;

// ------------------------------------------------------------------ helpers

class Grow {
  constructor(Type, cap = 1 << 16) {
    this.Type = Type;
    this.a = new Type(cap);
    this.n = 0;
  }
  push(v) {
    if (this.n === this.a.length) {
      const b = new this.Type(this.a.length * 2);
      b.set(this.a);
      this.a = b;
    }
    this.a[this.n++] = v;
  }
  get() {
    return this.a.subarray(0, this.n);
  }
}

// Small deterministic PRNG (mulberry32).
function rng(seed) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Points ~spacing apart on the surface, grouped by label: Map(label -> Float64Array xyz). */
export function sampleSurface(tris, labels, spacing) {
  const rand = rng(1);
  const out = new Map();
  const get = (l) => {
    if (!out.has(l)) out.set(l, new Grow(Float64Array, 1 << 20));
    return out.get(l);
  };
  const seen = new Set();
  for (let t = 0; t < labels.length; t++) {
    const o = 9 * t;
    const ax = tris[o], ay = tris[o + 1], az = tris[o + 2];
    const e1x = tris[o + 3] - ax, e1y = tris[o + 4] - ay, e1z = tris[o + 5] - az;
    const e2x = tris[o + 6] - ax, e2y = tris[o + 7] - ay, e2z = tris[o + 8] - az;
    const cx = e1y * e2z - e1z * e2y, cy = e1z * e2x - e1x * e2z, cz = e1x * e2y - e1y * e2x;
    const area = 0.5 * Math.hypot(cx, cy, cz);
    const n = Math.max(1, Math.ceil((2 * area) / (spacing * spacing)));
    const g = get(labels[t]);
    for (let i = 0; i < n; i++) {
      let r1 = rand(), r2 = rand();
      if (r1 + r2 > 1) { r1 = 1 - r1; r2 = 1 - r2; }
      g.push(ax + e1x * r1 + e2x * r2);
      g.push(ay + e1y * r1 + e2y * r2);
      g.push(az + e1z * r1 + e2z * r2);
    }
    // Vertices too (tiny triangles, sharp corners), once per (vertex, label).
    for (let v = 0; v < 3; v++) {
      const key = `${tris[o + 3 * v]},${tris[o + 3 * v + 1]},${tris[o + 3 * v + 2]},${labels[t]}`;
      if (seen.has(key)) continue;
      seen.add(key);
      g.push(tris[o + 3 * v]); g.push(tris[o + 3 * v + 1]); g.push(tris[o + 3 * v + 2]);
    }
  }
  const res = new Map();
  for (const [l, g] of out) res.set(l, g.get());
  return res;
}

export function toManifold(wasm, verts, faces) {
  const mesh = new wasm.Mesh({ numProp: 3, vertProperties: Float32Array.from(verts), triVerts: Uint32Array.from(faces) });
  const m = new wasm.Manifold(mesh);
  if (m.status() !== "NoError") throw new Error(`mesh is not manifold: ${m.status()}`);
  return m;
}

export function fromManifold(m) {
  const mesh = m.getMesh();
  const np = mesh.numProp;
  const nv = mesh.vertProperties.length / np;
  const v = new Float64Array(nv * 3);
  for (let i = 0; i < nv; i++) for (let k = 0; k < 3; k++) v[3 * i + k] = mesh.vertProperties[i * np + k];
  return { vertices: v, faces: Uint32Array.from(mesh.triVerts) };
}

// --------------------------------------------------------------- isosurface
// Cube corners as bit offsets (x=1, y=2, z=4); Freudenthal split into 6 tets
// along the main diagonal (conforming between neighbouring cubes).
const CORNER = [...Array(8)].map((_, c) => [c & 1, (c >> 1) & 1, (c >> 2) & 1]);
const TETS = [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]].map(([a, b]) => [0, 1 << a, (1 << a) | (1 << b), 7]);
const CASES = { 1: [[[0, 1], [0, 2], [0, 3]]], 3: [[[0, 3], [1, 3], [2, 3]]], 2: [[[0, 2], [0, 3], [1, 3]], [[0, 2], [1, 3], [1, 2]]] };

/**
 * Marching tetrahedra of g = 0 over the blocks of a virtual grid; g > 0 inside.
 * evalBlock(i0, dims, out) fills out (Float64Array, x-major) with g for the
 * block's grid points. Vertices are keyed by global grid edge, so blocks stitch
 * exactly and the result is watertight.
 */
export function isosurface(evalBlock, lo, voxel, shape, blocks, block, onProgress = null, xRange = null) {
  const [SX, SY, SZ] = shape;
  const vkey = new Map();
  const vpos = new Grow(Float64Array, 1 << 20);
  const tri = new Grow(Uint32Array, 1 << 20);
  const eps = 1e-3 * voxel;
  const B = block + 1;
  const g = new Float64Array(B * B * B);
  const cv = new Float64Array(8), ci = new Int32Array(8);
  const tv = new Float64Array(4), tc = new Int32Array(4);
  const order = new Int32Array(4);

  let done = 0;
  for (const [bx, by, bz] of blocks) {
    if (onProgress && (done++ & 15) === 0) onProgress(done / blocks.length);
    const i0 = [bx * block, by * block, bz * block];
    const nx = Math.min(B, SX - i0[0]), ny = Math.min(B, SY - i0[1]), nz = Math.min(B, SZ - i0[2]);
    if (nx < 2 || ny < 2 || nz < 2) continue;
    evalBlock(i0, [nx, ny, nz], g);
    let pos = false, neg = false;
    for (let ix = 0; ix < nx; ix++)
      for (let iy = 0; iy < ny; iy++)
        for (let iz = 0; iz < nz; iz++) {
          const k = (ix * ny + iy) * nz + iz;
          let v = g[k];
          const gx = i0[0] + ix, gy = i0[1] + iy, gz = i0[2] + iz;
          if (gx === 0 || gy === 0 || gz === 0 || gx === SX - 1 || gy === SY - 1 || gz === SZ - 1) v = -voxel;
          if (xRange && (gx < xRange[0] || gx > xRange[1])) v = -voxel; // outside this chunk
          if (Math.abs(v) < eps) v = v < 0 ? -eps : eps;
          g[k] = v;
          if (v > 0) pos = true; else neg = true;
        }
    if (!pos || !neg) continue;

    for (let ix = 0; ix < nx - 1; ix++)
      for (let iy = 0; iy < ny - 1; iy++)
        for (let iz = 0; iz < nz - 1; iz++) {
          let np = 0;
          for (let c = 0; c < 8; c++) {
            const k = ((ix + CORNER[c][0]) * ny + iy + CORNER[c][1]) * nz + iz + CORNER[c][2];
            cv[c] = g[k];
            if (cv[c] > 0) np++;
          }
          if (np === 0 || np === 8) continue;
          for (const tet of TETS) {
            let tp = 0;
            for (let j = 0; j < 4; j++) if (cv[tet[j]] > 0) tp++;
            if (tp === 0 || tp === 4) continue;
            // Order the tet's corners: positives first (stable).
            let o = 0;
            for (let j = 0; j < 4; j++) if (cv[tet[j]] > 0) order[o++] = tet[j];
            for (let j = 0; j < 4; j++) if (!(cv[tet[j]] > 0)) order[o++] = tet[j];
            for (const t of CASES[tp]) {
              const vid = [0, 0, 0];
              const p = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
              let ox = 0, oy = 0, oz = 0; // sum of (B - A): outward direction
              for (let q = 0; q < 3; q++) {
                const ca = order[t[q][0]], cb = order[t[q][1]];
                const ga = cv[ca], gb = cv[cb];
                const ax = i0[0] + ix + CORNER[ca][0], ay = i0[1] + iy + CORNER[ca][1], az = i0[2] + iz + CORNER[ca][2];
                const bx2 = i0[0] + ix + CORNER[cb][0], by2 = i0[1] + iy + CORNER[cb][1], bz2 = i0[2] + iz + CORNER[cb][2];
                const tt = ga / (ga - gb);
                p[q][0] = ax + (bx2 - ax) * tt;
                p[q][1] = ay + (by2 - ay) * tt;
                p[q][2] = az + (bz2 - az) * tt;
                ox += bx2 - ax; oy += by2 - ay; oz += bz2 - az;
                // Edge key: lower corner's linear id * 8 + direction bits (edges
                // of these tets always go "up" in every axis that changes).
                const la = (ax * SY + ay) * SZ + az, lb = (bx2 * SY + by2) * SZ + bz2;
                const key = la < lb
                  ? la * 8 + ((bx2 - ax) | ((by2 - ay) << 1) | ((bz2 - az) << 2))
                  : lb * 8 + ((ax - bx2) | ((ay - by2) << 1) | ((az - bz2) << 2));
                let id = vkey.get(key);
                if (id === undefined) {
                  id = vkey.size;
                  vkey.set(key, id);
                  vpos.push(lo[0] + p[q][0] * voxel);
                  vpos.push(lo[1] + p[q][1] * voxel);
                  vpos.push(lo[2] + p[q][2] * voxel);
                }
                vid[q] = id;
              }
              const ux = p[1][0] - p[0][0], uy = p[1][1] - p[0][1], uz = p[1][2] - p[0][2];
              const wx = p[2][0] - p[0][0], wy = p[2][1] - p[0][1], wz = p[2][2] - p[0][2];
              const nxx = uy * wz - uz * wy, nyy = uz * wx - ux * wz, nzz = ux * wy - uy * wx;
              if (nxx * ox + nyy * oy + nzz * oz < 0) { tri.push(vid[2]); tri.push(vid[1]); tri.push(vid[0]); }
              else { tri.push(vid[0]); tri.push(vid[1]); tri.push(vid[2]); }
            }
          }
        }
  }
  if (tri.n === 0) return null;
  return { vertices: vpos.get(), faces: tri.get() };
}

/**
 * Isosurface -> simplified Manifold, built in chunks so WASM memory stays
 * bounded (a raw marching-tets mesh costs ~350 bytes/triangle in manifold, and
 * big models produce 5-10M triangles; browsers cap WASM memory at 4 GB).
 *
 * Blocks are grouped into slabs along x. Each slab's field is forced to
 * "outside" beyond its grid range [x0, x1], which closes it with a cap just
 * past its boundary. In the overlap layer each slab's region is contained in
 * its neighbour's (forced values are <= the true ones, and the interpolation is
 * linear), so the union of the slabs is exactly the unchunked isosurface; only
 * the per-slab simplification differs (within its tolerance).
 */
export function isoManifold(wasm, evalBlock, lo, voxel, shape, blocks, block, tol, onProgress = null, maxBlocks = 300) {
  const byX = new Map();
  for (const b of blocks) (byX.get(b[0]) ?? byX.set(b[0], []).get(b[0])).push(b);
  const xs = [...byX.keys()].sort((a, b) => a - b);
  const groups = [];
  let cur = [], n = 0;
  for (const x of xs) {
    cur.push(x);
    n += byX.get(x).length;
    if (n >= maxBlocks) {
      groups.push(cur);
      cur = [];
      n = 0;
    }
  }
  if (cur.length) groups.push(cur);
  const parts = [];
  let rawTris = 0;
  for (const [gi, g] of groups.entries()) {
    const first = gi === 0, last = gi === groups.length - 1;
    const xa = g[0], xb = g[g.length - 1];
    const range = groups.length === 1 ? null : [first ? -Infinity : xa * block, last ? Infinity : (xb + 1) * block];
    // Neighbouring block columns hold the cap cubes.
    const bl = blocks.filter((b) => b[0] >= xa - (first ? 0 : 1) && b[0] <= xb + (last ? 0 : 1));
    const step = (f) => onProgress?.((gi + f) / groups.length);
    const iso = isosurface(evalBlock, lo, voxel, shape, bl, block, (f) => step(0.6 * f), range);
    if (!iso) continue;
    rawTris += iso.faces.length / 3;
    step(0.6);
    const raw = toManifold(wasm, iso.vertices, iso.faces);
    step(0.8);
    parts.push(raw.simplify(tol));
    raw.delete();
  }
  onProgress?.(1);
  if (!parts.length) return null;
  if (parts.length === 1) return { manifold: parts[0], rawTris, chunks: 1 };
  const manifold = wasm.Manifold.union(parts);
  for (const p of parts) p.delete();
  return { manifold, rawTris, chunks: groups.length };
}

// ------------------------------------------------------------------- driver

const ZFLAT = 8;
const CAP = 0.7; // |normal z| above which a face is a cap (within about 45 degrees of level)
const MIN_AREA = 0.5; // mm² of a filament's cross-section for it to count as used in a layer

/**
 * Split a solid into per-filament volumes. verts/faces: world mm. subTris/subFil:
 * painted surface (Float64Array N*9) and filament per sub-triangle.
 * Returns { core: {vertices, faces}|null, plugs: Map(filament -> {vertices, faces}) }.
 */
export function build(wasm, verts, faces, subTris, subFil, depths, coreFilament,
  { voxel = 0.25, block = 24, simplify = 0.3, overlap = 1.0, primer = null, span = new Span() } = {}) {
  const { Manifold } = wasm;
  // Rough cost model: prep (samples, trees) ~12%, color shells ~80% (primer
  // shells as much again), core + conversion ~8%.
  const [spPrep, spShells, spCore] = span.split([12, 80, 8]);
  spPrep.at(0, "sampling the painted surface");
  const body = toManifold(wasm, verts, faces);
  log(`body: ${body.numTri()} tris, volume ${body.volume().toFixed(0)} mm³`);
  // Sampled per filament and kind of surface: walls (0) and caps (1: faces that
  // point mostly up or down), which are compared differently (see makeShell).
  const kinds = Int32Array.from(subFil, (f, t) => {
    const o = 9 * t, e = (a, b) => subTris[o + a] - subTris[o + b];
    const nx = e(4, 1) * e(8, 2) - e(5, 2) * e(7, 1), ny = e(5, 2) * e(6, 0) - e(3, 0) * e(8, 2), nz = e(3, 0) * e(7, 1) - e(4, 1) * e(6, 0);
    return 2 * f + (Math.abs(nz) > CAP * Math.hypot(nx, ny, nz) ? 1 : 0);
  });
  const byKind = sampleSurface(subTris, kinds, voxel * 0.4);
  const samples = new Map();
  for (const f of new Set(subFil)) {
    const a = byKind.get(2 * f) ?? new Float64Array(0), b = byKind.get(2 * f + 1) ?? new Float64Array(0);
    const all = new Float64Array(a.length + b.length);
    all.set(a);
    all.set(b, a.length);
    samples.set(f, all);
  }
  let ns = 0;
  for (const s of samples.values()) ns += s.length / 3;
  log(`surface samples: ${ns}`);
  spPrep.at(0.4, "building search trees");
  const trees = new Map();
  // For deciding between colors (see makeShell): wall points with z stretched, cap points as they are.
  const treesFlat = new Map();
  for (const [l, s] of samples) {
    trees.set(l, new KDTree(s));
    const w = byKind.get(2 * l), c = byKind.get(2 * l + 1);
    if (w) for (let i = 2; i < w.length; i += 3) w[i] *= ZFLAT;
    const wall = w?.length ? new KDTree(w) : null, cap = c?.length ? new KDTree(c) : null;
    treesFlat.set(l, (x, y, z, ub) => Math.min(wall ? wall.query(x, y, z * ZFLAT, ub) : Infinity, cap ? cap.query(x, y, z, ub) : Infinity));
  }
  log(`trees built for ${trees.size} filaments`);

  const loAll = [Infinity, Infinity, Infinity], hiAll = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < subTris.length; i++) {
    const k = i % 3;
    if (subTris[i] < loAll[k]) loAll[k] = subTris[i];
    if (subTris[i] > hiAll[k]) hiAll[k] = subTris[i];
  }
  const order = [...depths.keys()].sort((a, b) => depths.get(a) - depths.get(b)).filter((f) => f !== coreFilament && trees.has(f));
  // Per-shell weight: its own samples (area) plus a fixed share for scanning the grid.
  const shellW = order.map((f) => trees.get(f).n * 0.5 + ns * 0.1);
  const shellSpans = spShells.split(primer ? [...shellW, ...shellW] : shellW);

  const makeShell = (k, d, restOff = 0, sp = new Span()) => {
    const f = order[k];
    const [spMask, spIso, spBool] = sp.split([5, 90, 5]);
    const what = `filament ${f}${restOff ? " (primer)" : ""}`;
    spMask.at(0, `${what}: narrow band`);
    const earlier = new Set(order.slice(0, k)), later = new Set(order.slice(k + 1));
    const treeF = trees.get(f);
    // Which color a point belongs to is decided by the nearest surface, but for
    // walls heights count ZFLAT times: a wall's color then reaches mostly level
    // into the model. So under a sloping wall the border between two colors runs
    // level from the paint edge, and a color stays out of layers where it doesn't
    // show. Caps (tops and bottoms) are measured plainly: their color goes straight down or up.
    const treeS = treesFlat.get(f);
    const others = [...treesFlat].filter(([l]) => l !== f).map(([l, t]) => [t, earlier.has(l) ? overlap * voxel : later.has(l) ? 0 : restOff]);
    const reach = d + 2 * voxel, margin = d + 3 * voxel;
    const lo = loAll.map((v) => v - margin);
    const shape = loAll.map((v, i) => Math.ceil((hiAll[i] + margin - lo[i]) / voxel) + 1);

    // Narrow band on a coarse cell grid (CELL^3 grid points per cell).
    const CELL = 4;
    const cs = shape.map((n) => Math.floor(n / CELL) + 1);
    const cellmask = new Uint8Array(cs[0] * cs[1] * cs[2]);
    const half = Math.sqrt(3) * ((CELL - 1) / 2) * voxel;
    const nb = shape.map((n) => Math.ceil(n / block));
    const blockSet = new Set();
    for (let cx = 0; cx < cs[0]; cx++)
      for (let cy = 0; cy < cs[1]; cy++)
        for (let cz = 0; cz < cs[2]; cz++) {
          const px = lo[0] + (cx * CELL + (CELL - 1) / 2) * voxel;
          const py = lo[1] + (cy * CELL + (CELL - 1) / 2) * voxel;
          const pz = lo[2] + (cz * CELL + (CELL - 1) / 2) * voxel;
          if (treeF.query(px, py, pz, reach + half) === Infinity) continue;
          cellmask[(cx * cs[1] + cy) * cs[2] + cz] = 1;
          // Blocks whose cubes can touch this cell's grid points.
          const r = (c, n) => [Math.max(0, Math.floor((c * CELL - 1) / block)), Math.min(n - 1, Math.floor((c * CELL + CELL - 1) / block))];
          const [x0, x1] = r(cx, nb[0]), [y0, y1] = r(cy, nb[1]), [z0, z1] = r(cz, nb[2]);
          for (let x = x0; x <= x1; x++) for (let y = y0; y <= y1; y++) for (let z = z0; z <= z1; z++) blockSet.add((x * nb[1] + y) * nb[2] + z);
        }
    const blocks = [...blockSet].sort((a, b) => a - b).map((id) => [Math.floor(id / (nb[1] * nb[2])), Math.floor(id / nb[2]) % nb[1], id % nb[2]]);
    log(`filament ${f}: depth ${d.toFixed(2)} mm, ${treeF.n} samples, ${blocks.length} blocks`);

    const evalBlock = (i0, [nx, ny, nz], out) => {
      for (let ix = 0; ix < nx; ix++) {
        const gx = i0[0] + ix, px = lo[0] + gx * voxel, cx = Math.floor(gx / CELL);
        for (let iy = 0; iy < ny; iy++) {
          const gy = i0[1] + iy, py = lo[1] + gy * voxel, cy = Math.floor(gy / CELL);
          for (let iz = 0; iz < nz; iz++) {
            const gz = i0[2] + iz, k = (ix * ny + iy) * nz + iz;
            out[k] = -voxel;
            if (!cellmask[(cx * cs[1] + cy) * cs[2] + Math.floor(gz / CELL)]) continue;
            const pz = lo[2] + gz * voxel;
            const df = treeF.query(px, py, pz, reach);
            if (df === Infinity) continue;
            // Inside shell f: within depth d of f's surface and nearer to it than
            // to other surfaces (reaching `overlap` further toward earlier colors).
            let gn = d - df;
            const dfs = Math.min(treeS(px, py, pz, ZFLAT * reach), ZFLAT * reach);
            for (const [t, off] of others) {
              const ub = ZFLAT * reach + Math.max(off, 0);
              const dd = Math.min(t(px, py, pz, ub), ub);
              gn = Math.min(gn, dd + off - dfs);
            }
            out[k] = Math.max(gn, -voxel);
          }
        }
      }
    };
    const res = isoManifold(wasm, evalBlock, lo, voxel, shape, blocks, block, simplify * voxel, (x) => spIso.at(x, `${what}: depth field + mesh`));
    if (!res) return null;
    const shell = res.manifold, n0 = res.rawTris;
    spBool.at(0, `${what}: booleans`);
    log(`filament ${f}: shell ${n0} -> ${shell.numTri()} tris (${res.chunks} chunk${res.chunks > 1 ? "s" : ""})`);
    return shell;
  };

  // Booleans only ever cut with raw shells (see depth.py).
  // Every manifold made here lives in WASM memory until .delete(): track them.
  const owned = [body];
  const own = (m) => (owned.push(m), m);
  const plugs = new Map();
  const shells = [];
  for (let k = 0; k < order.length; k++) {
    const f = order[k];
    const shell = makeShell(k, depths.get(f), 0, shellSpans[k]);
    if (!shell) continue;
    own(shell);
    let plug = own(body.intersect(shell));
    if (shells.length) plug = own(plug.subtract(own(Manifold.union(shells))));
    shells.push(shell);
    log(`filament ${f}: plug ${plug.numTri()} tris, volume ${plug.volume().toFixed(0)} mm³`);
    plugs.set(f, plug);
  }
  const colors = shells.length ? own(Manifold.union(shells)) : null;
  let inner = colors;
  if (primer && shells.length) {
    const [pf, pt] = primer;
    const ext = order.map((f, k) => makeShell(k, depths.get(f) + pt, -overlap * voxel, shellSpans[order.length + k])).filter(Boolean).map(own);
    inner = own(Manifold.union([...ext, colors]));
    const layer = own(own(body.intersect(inner)).subtract(colors));
    log(`primer (filament ${pf}, ${pt.toFixed(2)} mm): ${layer.numTri()} tris, volume ${layer.volume().toFixed(0)} mm³`);
    plugs.set(pf, plugs.has(pf) ? own(plugs.get(pf).add(layer)) : layer);
  }
  spCore.at(0, "core");
  const core = inner ? own(body.subtract(inner)) : body;
  log(`core: ${core.numTri()} tris, volume ${core.volume().toFixed(0)} mm³`);
  const res = { core: core.isEmpty() ? null : fromManifold(core), plugs: new Map() };
  for (const [f, m] of plugs) if (!m.isEmpty()) res.plugs.set(f, fromManifold(m));
  for (const m of new Set(owned)) m.delete();
  span.at(1, "part done");
  return res;
}

// ------------------------------------------------ fewer filament changes

const layerOf = (z, h) => Math.max(0, Math.floor(z / h + 1e-6));

/**
 * Where each filament shows on a surface, per print layer (height h, from z = 0),
 * over all objects: Map(filament -> Float64Array of painted area per layer).
 * A triangle's area is spread over the layers it spans, so 0 means "not in this layer".
 */
function surfaceLayers(proj, h) {
  const per = new Map();
  for (const obj of proj.objects)
    for (const part of obj.parts) {
      if (part.subtype !== "normal_part") continue;
      const { tris, states } = tessellate(part.mesh.vertices, part.mesh.faces, part.mesh.paint);
      const w = applyTransform(tris, matmul(part.transform, obj.transform)), pe = partExtruder(obj, part);
      for (let t = 0; t < states.length; t++) {
        const o = t * 9, f = states[t] === 0 ? pe : states[t];
        const ux = w[o + 3] - w[o], uy = w[o + 4] - w[o + 1], uz = w[o + 5] - w[o + 2], vx = w[o + 6] - w[o], vy = w[o + 7] - w[o + 1], vz = w[o + 8] - w[o + 2];
        const area = 0.5 * Math.hypot(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx);
        const a = layerOf(Math.min(w[o + 2], w[o + 5], w[o + 8]), h), b = Math.max(a, layerOf(Math.max(w[o + 2], w[o + 5], w[o + 8]) - 1e-4, h));
        let arr = per.get(f);
        if (!arr || arr.length <= b) per.set(f, (arr = Float64Array.from({ length: Math.max(b + 1, (arr?.length ?? 0) * 2) }, (_, i) => arr?.[i] ?? 0)));
        for (let l = a; l <= b; l++) arr[l] += area / (b - a + 1) + 1e-12;
      }
    }
  return per;
}

/**
 * Cross-section area (mm²) of a closed mesh at the middle of each print layer:
 * for every triangle the cut passes through, the shoelace term of its cut segment.
 */
function layerAreas(mesh, h, n) {
  const out = new Float64Array(n), v = mesh.vertices, f = mesh.faces;
  for (let i = 0; i < f.length; i += 3) {
    const p = [f[i] * 3, f[i + 1] * 3, f[i + 2] * 3];
    const zs = p.map((k) => v[k + 2]);
    const a = Math.max(0, Math.ceil(Math.min(...zs) / h - 0.5)), b = Math.min(n - 1, Math.floor(Math.max(...zs) / h - 0.5));
    for (let l = a; l <= b; l++) {
      const z = (l + 0.5) * h;
      let up = null, down = null; // where the triangle's outline crosses the cut, going up and going down
      for (let e = 0; e < 3; e++) {
        const k0 = p[e], k1 = p[(e + 1) % 3], z0 = v[k0 + 2], z1 = v[k1 + 2];
        if (z0 < z === z1 < z) continue;
        const t = (z - z0) / (z1 - z0), pt = [v[k0] + t * (v[k1] - v[k0]), v[k0 + 1] + t * (v[k1 + 1] - v[k0 + 1])];
        if (z0 < z) up = pt;
        else down = pt;
      }
      if (up && down) out[l] += 0.5 * (up[0] * down[1] - down[0] * up[1]);
    }
  }
  return out.map(Math.abs);
}

/**
 * Give each print layer's inside a filament that layer uses anyway: the
 * brightest one (a bright inside keeps the colors over it bright), which costs
 * no filament change, and saves one where the inside's own filament is not used.
 *   fixed = false: the inside is free (every surface color, also coreF, has its
 *     own skin in `plugs`), so the brightest filament of the layer always wins.
 *   fixed = true: coreF was chosen as a core. It stays where the layer uses it
 *     anyway and within `keep` mm above and below its own surfaces (tris/fil:
 *     this part's painted surface), and everywhere if it is on no surface at all.
 * layers: surfaceLayers(); plugs: Map(filament -> mesh) generated for this part;
 * bright(filament): 0..1. Returns [[filament, mesh]], coreF's piece first.
 */
function splitInside(wasm, mesh, coreF, { fixed, tris, fil, keep, layers, plugs, h, bright }) {
  let zmin = Infinity, zmax = -Infinity, xy = [Infinity, Infinity, -Infinity, -Infinity];
  for (let i = 0; i < mesh.vertices.length; i += 3) {
    const [x, y, z] = [mesh.vertices[i], mesh.vertices[i + 1], mesh.vertices[i + 2]];
    if (z < zmin) zmin = z;
    if (z > zmax) zmax = z;
    xy = [Math.min(xy[0], x), Math.min(xy[1], y), Math.max(xy[2], x), Math.max(xy[3], y)];
  }
  const l0 = layerOf(zmin, h), l1 = layerOf(zmax - 1e-4, h);
  if (fixed && !layers.has(coreF)) {
    // A core chosen for its material (e.g. PLA-CF): replacing it would remove it.
    log(`fewer changes: filament ${coreF} is not on any surface, so the inside is kept as chosen`);
    return [[coreF, mesh]];
  }
  // Where each filament is printed anyway: surfaces of all objects, plus this part's generated parts.
  // A sliver doesn't count (MIN_AREA): borders between colors are a little ragged.
  const cut = new Map([...plugs].map(([f, m]) => [f, layerAreas(m, h, l1 + 1)]));
  const on = (f, l) => layers.get(f)?.[l] > 0 || cut.get(f)?.[l] > MIN_AREA;
  const need = new Uint8Array(l1 + 1);
  if (fixed) {
    for (let l = l0; l <= l1; l++) if (on(coreF, l)) need[l] = 1;
    for (let t = 0; t < fil.length; t++) {
      if (fil[t] !== coreF) continue;
      const z = [tris[t * 9 + 2], tris[t * 9 + 5], tris[t * 9 + 8]];
      for (let l = layerOf(Math.min(...z) - keep, h); l <= Math.min(l1, layerOf(Math.max(...z) + keep, h)); l++) need[l] = 1;
    }
  }
  const order = [...new Set([...layers.keys(), ...cut.keys()])].sort((a, b) => bright(b) - bright(a) || a - b);
  const runs = new Map(); // filament -> [[layer from, layer to)]
  let saved = 0, changed = 0;
  for (let l = l0; l <= l1; l++) {
    const best = need[l] ? null : order.find((f) => on(f, l));
    if (best == null || best === coreF) continue;
    changed++;
    if (!on(coreF, l)) saved++;
    const r = runs.get(best) ?? runs.set(best, []).get(best);
    if (r.length && r[r.length - 1][1] === l) r[r.length - 1][1] = l + 1;
    else r.push([l, l + 1]);
  }
  if (!runs.size) {
    log(`fewer changes: filament ${coreF} stays the inside in every layer`);
    return [[coreF, mesh]];
  }
  let rest = toManifold(wasm, mesh.vertices, mesh.faces);
  const out = [];
  for (const [f, rs] of [...runs].sort((a, b) => a[0] - b[0])) {
    const boxes = rs.map(([a, b]) => wasm.Manifold.cube([xy[2] - xy[0] + 2, xy[3] - xy[1] + 2, (b - a) * h + (a === l0 ? 1 : 0) + (b === l1 + 1 ? 1 : 0)], false)
      .translate([xy[0] - 1, xy[1] - 1, a * h - (a === l0 ? 1 : 0)]));
    const slab = boxes.length > 1 ? wasm.Manifold.union(boxes) : boxes[0];
    const piece = rest.intersect(slab), next = rest.subtract(slab);
    rest.delete();
    rest = next;
    if (!piece.isEmpty()) out.push([f, fromManifold(piece)]);
    log(`fewer changes: inside in filament ${f} for z ${rs.map(([a, b]) => `${(a * h).toFixed(1)}-${(b * h).toFixed(1)}`).join(", ")} mm, ${piece.volume().toFixed(0)} mm³`);
    piece.delete();
    if (boxes.length > 1) slab.delete();
    for (const b of boxes) b.delete();
  }
  if (!rest.isEmpty()) out.unshift([coreF, fromManifold(rest)]);
  rest.delete();
  log(`fewer changes: inside changed in ${changed} of ${l1 - l0 + 1} layers (${h} mm); about ${saved} filament change${saved === 1 ? "" : "s"} saved`);
  return out;
}

// ------------------------------------------------------------------ project

export function selectObjects(proj, name = null) {
  let objs = proj.objects;
  if (name) return objs.filter((o) => o.name.toLowerCase().includes(name.toLowerCase()));
  if (objs.some((o) => o.parts.some((p) => p.mesh.painted))) objs = objs.filter((o) => o.parts.some((p) => p.mesh.painted));
  return objs;
}

/**
 * Build per-filament parts for every painted part of the selected objects.
 * Returns Map("objId/partId" -> [{name, extruder, vertices, faces}]) in the
 * part's own mesh coordinates.
 */
export function process(wasm, proj, { depthOverrides = new Map(), core = null, voxel = 0.25, dmin = 0.45, dmax = 1.6, uniform = null, objectName = null, primer = null, inlay = null, fewer = false, span = new Span() } = {}) {
  const out = new Map();
  const layerH = proj.layerHeight ?? 0.2;
  const layers = fewer ? surfaceLayers(proj, layerH) : null;
  out.inlays = []; // [{object, pieces}] when `inlay` is set (see inlay.js)
  const todo = [];
  for (const obj of selectObjects(proj, objectName))
    for (const part of obj.parts) if (part.subtype === "normal_part" && (part.mesh.painted || core != null)) todo.push([obj, part]);
  const spans = span.split(todo.map(([, p]) => p.mesh.faces.length + 3 * p.mesh.painted));
  for (const [i, [obj, part]] of todo.entries()) {
    {
      const W = matmul(part.transform, obj.transform);
      const { tris, states } = tessellate(part.mesh.vertices, part.mesh.faces, part.mesh.paint);
      const world = applyTransform(tris, W);
      const pe = partExtruder(obj, part);
      const fil = states.map((s) => (s === 0 ? pe : s));
      const coreF = core != null ? core : pe;
      const V = applyTransform(part.mesh.vertices, W);
      // Inlays: chosen painted regions become separate pieces; on the model
      // they turn into body, and their pockets are cut out of all parts below.
      let pockets = null;
      if (inlay && part.mesh.painted) {
        const body = toManifold(wasm, V, part.mesh.faces);
        const plan = planInlays(wasm, body, world, fil, { ...inlay, bodyFilament: pe });
        body.delete();
        for (const k of plan.relabel) fil[k] = pe;
        pockets = plan.pockets;
        if (plan.pieces.length) out.inlays.push({ object: obj, pieces: plan.pieces });
      }
      const depths = new Map();
      for (const f of new Set(fil)) depths.set(f, uniform != null ? uniform : defaultDepth(filamentColor(proj, f), dmin, dmax));
      for (const [f, d] of depthOverrides) depths.set(f, d);
      log(`object '${obj.name}' part '${part.name}': core filament ${coreF}, ${tris.length / 9} sub-triangles`);
      // Without a chosen core the inside is free: give the body's own color a skin
      // too (filament 0 = none is the core), then pick the inside per layer below.
      const free = fewer && core == null;
      const res = build(wasm, V, part.mesh.faces, world, fil, depths, free ? 0 : coreF, { voxel, primer, span: spans[i] });
      const Winv = invert(W);
      const parts = [];
      const add = (name, extruder, mesh) => {
        if (pockets) {
          const m = toManifold(wasm, mesh.vertices, mesh.faces), cut = m.subtract(pockets);
          mesh = cut.isEmpty() ? null : fromManifold(cut);
          m.delete();
          cut.delete();
        }
        if (mesh) parts.push({ name, extruder, vertices: applyTransform(mesh.vertices, Winv), faces: mesh.faces });
      };
      if (res.core && fewer) {
        const split = splitInside(wasm, res.core, coreF, { fixed: !free, tris: world, fil, keep: depths.get(coreF) ?? dmax, layers, plugs: res.plugs, h: layerH, bright: (f) => luminance(filamentColor(proj, f)) });
        for (const [f, m] of split) {
          // One part per filament: an inside piece joins that filament's color part.
          const plug = res.plugs.get(f);
          if (!plug) add(`${part.name} core${f === coreF ? "" : ` (filament ${f})`}`, f, m);
          else {
            const a = toManifold(wasm, plug.vertices, plug.faces), b = toManifold(wasm, m.vertices, m.faces), u = a.add(b);
            res.plugs.set(f, fromManifold(u));
            for (const x of [a, b, u]) x.delete();
          }
        }
      } else if (res.core) add(`${part.name} core`, coreF, res.core);
      for (const f of [...res.plugs.keys()].sort((a, b) => a - b)) add(`${part.name} ${primer && f === primer[0] ? "primer" : `color ${f}`}`, f, res.plugs.get(f));
      pockets?.delete();
      out.set(`${obj.id}/${part.id}`, parts);
    }
  }
  return out;
}
