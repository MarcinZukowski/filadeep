// Inlays: painted regions of chosen filaments become separate pieces that are
// printed on their own (lying flat) and pushed into matching pockets.
//
// For each connected painted region:
//   - the insertion direction d is the region's average outward normal;
//   - the region's outline, seen along d, is extruded into a straight prism from
//     a flat floor (`depth` below the region's lowest point along d) to just
//     above the surface;
//   - pocket = prism (cut out of every part of the model);
//     piece  = prism shrunk sideways by `clearance`, intersected with the model.
// So a piece has a flat back to print on, straight sides to slide in along d,
// and the model's own surface as its face.

import { fromManifold, log } from "./depth.js";

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => {
  const l = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
};

/** Connected groups (sharing a vertex) among the triangles `idx` of a soup `tris` (9 numbers each). */
function components(tris, idx) {
  const ids = new Map();
  const parent = [];
  const find = (a) => {
    while (parent[a] !== a) a = parent[a] = parent[parent[a]];
    return a;
  };
  const vert = new Int32Array(idx.length * 3);
  for (let n = 0; n < idx.length; n++)
    for (let k = 0; k < 3; k++) {
      const o = idx[n] * 9 + k * 3;
      const key = `${Math.round(tris[o] * 1e4)},${Math.round(tris[o + 1] * 1e4)},${Math.round(tris[o + 2] * 1e4)}`;
      let id = ids.get(key);
      if (id === undefined) ids.set(key, (id = parent.length)), parent.push(id);
      vert[n * 3 + k] = id;
    }
  for (let n = 0; n < idx.length; n++) {
    const a = find(vert[n * 3]);
    parent[find(vert[n * 3 + 1])] = a;
    parent[find(vert[n * 3 + 2])] = a;
  }
  const groups = new Map();
  for (let n = 0; n < idx.length; n++) {
    const r = find(vert[n * 3]);
    (groups.get(r) ?? groups.set(r, []).get(r)).push(idx[n]);
  }
  return [...groups.values()];
}

/**
 * Solid under a painted region, in the region's frame (x, y on the floor, z up):
 * everything from z = 0 up to the region's surface (+ lift), over a box around it.
 * It keeps a pocket from cutting other parts of the model that stand in front
 * of the region (e.g. feet in front of a belly).
 * tri2: [[x, y, z] * 3] per triangle, z measured from the floor.
 */
function underSurface(wasm, tri2, step, lift) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const t of tri2) for (const p of t) {
    if (p[0] < x0) x0 = p[0];
    if (p[0] > x1) x1 = p[0];
    if (p[1] < y0) y0 = p[1];
    if (p[1] > y1) y1 = p[1];
  }
  const pad = 4; // grid cells around the region, filled from their neighbours
  x0 -= pad * step, y0 -= pad * step;
  const nx = Math.ceil((x1 - x0) / step) + pad + 1, ny = Math.ceil((y1 - y0) / step) + pad + 1;
  const h = new Float64Array(nx * ny).fill(-1);
  for (const [a, b, c] of tri2) {
    const den = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1]);
    if (Math.abs(den) < 1e-12) continue;
    const i0 = Math.max(0, Math.floor((Math.min(a[0], b[0], c[0]) - x0) / step)), i1 = Math.min(nx - 1, Math.ceil((Math.max(a[0], b[0], c[0]) - x0) / step));
    const j0 = Math.max(0, Math.floor((Math.min(a[1], b[1], c[1]) - y0) / step)), j1 = Math.min(ny - 1, Math.ceil((Math.max(a[1], b[1], c[1]) - y0) / step));
    for (let j = j0; j <= j1; j++)
      for (let i = i0; i <= i1; i++) {
        const x = x0 + i * step, y = y0 + j * step;
        const w0 = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / den;
        const w1 = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / den;
        const w2 = 1 - w0 - w1;
        if (w0 < -1e-3 || w1 < -1e-3 || w2 < -1e-3) continue;
        const z = w0 * a[2] + w1 * b[2] + w2 * c[2];
        if (z > h[j * nx + i]) h[j * nx + i] = z;
      }
  }
  // Grow the known heights outward, so the surface continues past the region's edge.
  for (let ring = 0; ring < pad + 2; ring++) {
    const prev = h.slice();
    for (let j = 0; j < ny; j++)
      for (let i = 0; i < nx; i++) {
        if (prev[j * nx + i] >= 0) continue;
        let s = 0, n = 0;
        for (let dj = -1; dj <= 1; dj++)
          for (let di = -1; di <= 1; di++) {
            const ii = i + di, jj = j + dj;
            if (ii < 0 || jj < 0 || ii >= nx || jj >= ny || prev[jj * nx + ii] < 0) continue;
            s += prev[jj * nx + ii];
            n++;
          }
        if (n) h[j * nx + i] = s / n;
      }
  }
  // Closed mesh: top at the heights, flat bottom at z = 0, four walls.
  const N = nx * ny, verts = new Float32Array(N * 6), tris = [];
  for (let j = 0; j < ny; j++)
    for (let i = 0; i < nx; i++) {
      const k = j * nx + i;
      verts.set([x0 + i * step, y0 + j * step, Math.max(h[k], 0) + lift], k * 3);
      verts.set([x0 + i * step, y0 + j * step, 0], (N + k) * 3);
    }
  const quad = (a, b, c, d) => tris.push(a, b, c, a, c, d);
  for (let j = 0; j < ny - 1; j++)
    for (let i = 0; i < nx - 1; i++) {
      const k = j * nx + i;
      quad(k, k + 1, k + nx + 1, k + nx); // top, facing +z
      quad(N + k, N + k + nx, N + k + nx + 1, N + k + 1); // bottom
    }
  for (let i = 0; i < nx - 1; i++) {
    quad(N + i, N + i + 1, i + 1, i); // y = y0
    const k = (ny - 1) * nx + i;
    quad(N + k + 1, N + k, k, k + 1); // y = max
  }
  for (let j = 0; j < ny - 1; j++) {
    const k = j * nx;
    quad(N + k + nx, N + k, k, k + nx); // x = x0
    const m = k + nx - 1;
    quad(N + m, N + m + nx, m + nx, m); // x = max
  }
  const solid = new wasm.Manifold(new wasm.Mesh({ numProp: 3, vertProperties: verts, triVerts: Uint32Array.from(tris) }));
  if (solid.status() !== "NoError") throw new Error(`inlay: surface solid is ${solid.status()}`);
  return solid;
}

/**
 * Plan the inlays of one part.
 *   body: Manifold of the part; tris/fil: painted sub-triangles (world coords) and their filament.
 *   opts: { filaments: Set | null (all but the body's), depth, clearance, minWidth,
 *           bodyFilament (the part's own filament: never an inlay),
 *           leftover: what regions that can't be a piece become: "paint" (stay painted) or "body" }
 * Returns { pieces: [{name, extruder, vertices, faces}] (each lying flat: floor at z = 0, centered in x/y),
 *           pockets: Manifold | null (to subtract from every part), relabel: [triangle indices now body] }.
 */
export function planInlays(wasm, body, tris, fil, { filaments = null, depth = 1.5, clearance = 0.1, minWidth = 2, bodyFilament = null, leftover = "paint" }) {
  const { Manifold, CrossSection } = wasm;
  const pieces = [], pocketList = [], relabel = [];
  const smooth = 0.25; // rounds off the triangle-sized steps of a paint outline
  const todo = [];
  const bodyVolume = body.volume();
  for (const f of [...(filaments ?? new Set(fil))].sort((a, b) => a - b)) {
    if (f === bodyFilament) {
      if (filaments) log(`inlay: filament ${f} is the part's own filament, so it stays the body`);
      continue;
    }
    const idx = [];
    for (let i = 0; i < fil.length; i++) if (fil[i] === f) idx.push(i);
    if (!idx.length) continue;
    for (const g of components(tris, idx)) todo.push({ f, g });
  }
  // Area and normal of each region; larger regions first (a later, smaller piece gives way where two overlap).
  for (const t of todo) {
    let area = 0;
    const ns = [0, 0, 0];
    t.normals = new Float64Array(t.g.length * 3);
    t.g.forEach((i, n) => {
      const o = i * 9;
      const e1 = [tris[o + 3] - tris[o], tris[o + 4] - tris[o + 1], tris[o + 5] - tris[o + 2]];
      const e2 = [tris[o + 6] - tris[o], tris[o + 7] - tris[o + 1], tris[o + 8] - tris[o + 2]];
      const c = cross(e1, e2);
      t.normals.set(c, n * 3); // length = 2 * area
      area += Math.hypot(...c) / 2;
      for (let k = 0; k < 3; k++) ns[k] += c[k];
    });
    t.area = area;
    t.d = norm(ns);
  }
  todo.sort((a, b) => b.area - a.area);

  const count = new Map();
  let skipped = 0;
  for (const { f, g, area, d, normals } of todo) {
    const why = (msg) => {
      if (area >= 2) log(`inlay: filament ${f} region of ${area.toFixed(0)} mm² ${leftover === "body" ? "dropped (body color)" : "left as paint"}: ${msg}`);
      if (leftover === "body") for (const i of g) relabel.push(i);
      skipped++;
    };
    if (area < 3) { why("too small"); continue; }
    // A single push-in direction needs the whole region to face it.
    let away = 0;
    for (let n = 0; n < g.length; n++) {
      const c = [normals[n * 3], normals[n * 3 + 1], normals[n * 3 + 2]];
      const l = Math.hypot(...c);
      if (l > 0 && dot(c, d) / l < -0.3) away += l / 2;
    }
    if (away > 0.08 * area) { why("it wraps around the model, so it has no single push-in direction"); continue; }

    // Frame: u, v span the floor, d is up. v follows the model's z where possible.
    const up = Math.abs(d[2]) < 0.95 ? [0, 0, 1] : [0, 1, 0];
    const v = norm([up[0] - dot(up, d) * d[0], up[1] - dot(up, d) * d[1], up[2] - dot(up, d) * d[2]]);
    const u = cross(v, d);
    let tmin = Infinity, tmax = -Infinity;
    const polys = [], tri2 = [];
    for (const i of g) {
      const o = i * 9, p = [], p3 = [];
      for (let k = 0; k < 3; k++) {
        const q = [tris[o + k * 3], tris[o + k * 3 + 1], tris[o + k * 3 + 2]];
        const t = dot(q, d);
        if (t < tmin) tmin = t;
        if (t > tmax) tmax = t;
        p.push([dot(q, u), dot(q, v)]);
        p3.push([p[k][0], p[k][1], t]);
      }
      const a2 = (p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[2][0] - p[0][0]) * (p[1][1] - p[0][1]);
      if (Math.abs(a2) < 1e-9) continue;
      tri2.push(p3);
      if (a2 < 0) p.reverse();
      polys.push(p);
    }
    const tmp = [];
    const keep = (x) => (tmp.push(x), x);
    try {
      const raw = keep(new CrossSection(polys, "NonZero"));
      // Close small gaps, then drop slivers; the result is the pocket outline.
      const outline = keep(keep(keep(raw.offset(smooth, "Round")).offset(-2 * smooth, "Round")).offset(smooth, "Round"));
      // Thin shapes (lines, rims) break and are hard to place: most of the outline must survive
      // removing everything narrower than minWidth.
      const wide = keep(keep(outline.offset(-minWidth / 2, "Round")).offset(minWidth / 2, "Round"));
      if (wide.area() < 0.7 * outline.area()) { why(`mostly narrower than ${minWidth} mm`); continue; }
      const inner = keep(outline.offset(-clearance, "Round"));
      const t0 = tmin - depth, H = tmax + 1 - t0;
      // Local (x, y, z) -> world: x*u + y*v + (t0 + z)*d.
      const toWorld = [...u, 0, ...v, 0, ...d, 0, t0 * d[0], t0 * d[1], t0 * d[2], 1];
      const toLocal = [u[0], v[0], d[0], 0, u[1], v[1], d[1], 0, u[2], v[2], d[2], 0, 0, 0, -t0, 1];
      // Only what lies under the painted surface belongs to the pocket and the piece.
      for (const t of tri2) for (const q of t) q[2] -= t0;
      const under = keep(underSurface(wasm, tri2, 0.3, 0.3));
      const pocket = keep(keep(outline.extrude(H)).intersect(under)).transform(toWorld);
      let piece = keep(keep(keep(keep(inner.extrude(H)).intersect(under)).transform(toWorld)).intersect(body));
      for (const p of pocketList) piece = keep(piece.subtract(p));
      // Anything else of the model that sticks into the prism is not part of the piece.
      const bodies = piece.decompose().map(keep).sort((a, b) => b.volume() - a.volume());
      if (!bodies.length || bodies[0].volume() < 0.5) { pocket.delete(); why("nothing left to print"); continue; }
      if (bodies[0].volume() > 0.35 * bodyVolume) { pocket.delete(); why("the piece would be most of the model"); continue; }
      const flat = keep(bodies[0].transform(toLocal));
      const bb = flat.boundingBox();
      const mesh = fromManifold(keep(flat.translate(-(bb.min[0] + bb.max[0]) / 2, -(bb.min[1] + bb.max[1]) / 2, -bb.min[2])));
      const n = (count.get(f) ?? 0) + 1;
      count.set(f, n);
      pieces.push({ name: `inlay ${f}.${n}`, extruder: f, vertices: mesh.vertices, faces: mesh.faces,
        size: [bb.max[0] - bb.min[0], bb.max[1] - bb.min[1], bb.max[2] - bb.min[2]] });
      pocketList.push(pocket);
      for (const i of g) relabel.push(i);
      log(`inlay ${f}.${n}: ${area.toFixed(0)} mm² -> piece ${(bb.max[0] - bb.min[0]).toFixed(1)} × ${(bb.max[1] - bb.min[1]).toFixed(1)} × ${(bb.max[2] - bb.min[2]).toFixed(1)} mm, ${mesh.faces.length / 3} tris`);
    } finally {
      for (const x of tmp) x.delete();
    }
  }
  if (skipped) log(`inlay: ${skipped} region(s) ${leftover === "body" ? "dropped (body color)" : "left as paint"}`);
  let pockets = null;
  if (pocketList.length) {
    pockets = pocketList.length === 1 ? pocketList[0] : Manifold.union(pocketList);
    if (pocketList.length > 1) for (const p of pocketList) p.delete();
  }
  return { pieces, pockets, relabel };
}

/** Lay pieces out in rows in front (-y) of a box [minx, miny, maxx]; sets piece.at = [x, y]. */
export function arrange(pieces, box, gap = 3) {
  const rowW = Math.max(box.maxx - box.minx, 60);
  let x = box.minx, y = box.miny - gap - 2, rowH = 0;
  for (const p of [...pieces].sort((a, b) => b.size[1] - a.size[1])) {
    if (x > box.minx && x + p.size[0] > box.minx + rowW) {
      x = box.minx;
      y -= rowH + gap;
      rowH = 0;
    }
    p.at = [x + p.size[0] / 2, y - p.size[1] / 2];
    x += p.size[0] + gap;
    rowH = Math.max(rowH, p.size[1]);
  }
}
