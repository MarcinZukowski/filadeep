// Outer shell: a layer of another filament around the model. Port of shell.py.
//   band = { x : distance(x, surface) < t };  shell = (band ∪ body) − body
// (booleans only involve the band surface, which never coincides with the body).

import { KDTree } from "./kdtree.js";
import { fromManifold, isoManifold, log, sampleSurface, toManifold } from "./depth.js";
import { Span } from "./progress.js";

/** bodies: [{vertices, faces}] in world mm (parts of one object). Returns {vertices, faces} or null. */
export function buildShell(wasm, bodies, thickness, { voxel = 0.25, block = 24, simplify = 0.3, trimBottom = true, span = new Span() } = {}) {
  const { Manifold } = wasm;
  const [spPrep, spIso, spBool] = span.split([15, 75, 10]);
  spPrep.at(0, "shell: sampling the surface");
  const pieces = bodies.map((b) => toManifold(wasm, b.vertices, b.faces));
  const body = Manifold.union(pieces);
  for (const m of pieces) m.delete();
  const { vertices: v, faces: f } = fromManifold(body);
  log(`shell body: ${body.numTri()} tris, volume ${body.volume().toFixed(0)} mm³`);
  const tris = new Float64Array(f.length * 3);
  for (let i = 0; i < f.length; i++) for (let k = 0; k < 3; k++) tris[3 * i + k] = v[3 * f[i] + k];
  const pts = sampleSurface(tris, new Int32Array(f.length / 3), voxel * 0.4).get(0);
  const tree = new KDTree(pts);
  const reach = thickness + 2 * voxel, margin = thickness + 3 * voxel;
  const loAll = [Infinity, Infinity, Infinity], hiAll = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < v.length; i++) {
    const k = i % 3;
    if (v[i] < loAll[k]) loAll[k] = v[i];
    if (v[i] > hiAll[k]) hiAll[k] = v[i];
  }
  const lo = loAll.map((x) => x - margin);
  const shape = loAll.map((x, i) => Math.ceil((hiAll[i] + margin - lo[i]) / voxel) + 1);

  const CELL = 4;
  const cs = shape.map((n) => Math.floor(n / CELL) + 1);
  const cellmask = new Uint8Array(cs[0] * cs[1] * cs[2]);
  const half = Math.sqrt(3) * ((CELL - 1) / 2) * voxel;
  const nb = shape.map((n) => Math.ceil(n / block));
  const blockSet = new Set();
  for (let cx = 0; cx < cs[0]; cx++)
    for (let cy = 0; cy < cs[1]; cy++)
      for (let cz = 0; cz < cs[2]; cz++) {
        const c = [cx, cy, cz].map((ci, k) => lo[k] + (ci * CELL + (CELL - 1) / 2) * voxel);
        if (tree.query(c[0], c[1], c[2], reach + half) === Infinity) continue;
        cellmask[(cx * cs[1] + cy) * cs[2] + cz] = 1;
        const r = (ci, n) => [Math.max(0, Math.floor((ci * CELL - 1) / block)), Math.min(n - 1, Math.floor((ci * CELL + CELL - 1) / block))];
        const [x0, x1] = r(cx, nb[0]), [y0, y1] = r(cy, nb[1]), [z0, z1] = r(cz, nb[2]);
        for (let x = x0; x <= x1; x++) for (let y = y0; y <= y1; y++) for (let z = z0; z <= z1; z++) blockSet.add((x * nb[1] + y) * nb[2] + z);
      }
  const blocks = [...blockSet].sort((a, b) => a - b).map((id) => [Math.floor(id / (nb[1] * nb[2])), Math.floor(id / nb[2]) % nb[1], id % nb[2]]);
  log(`shell ${thickness.toFixed(2)} mm: ${tree.n} samples, ${blocks.length} blocks`);

  const evalBlock = (i0, [nx, ny, nz], out) => {
    for (let ix = 0; ix < nx; ix++) {
      const gx = i0[0] + ix, px = lo[0] + gx * voxel, cx = Math.floor(gx / CELL);
      for (let iy = 0; iy < ny; iy++) {
        const gy = i0[1] + iy, py = lo[1] + gy * voxel, cy = Math.floor(gy / CELL);
        for (let iz = 0; iz < nz; iz++) {
          const gz = i0[2] + iz, k = (ix * ny + iy) * nz + iz;
          out[k] = -voxel;
          if (!cellmask[(cx * cs[1] + cy) * cs[2] + Math.floor(gz / CELL)]) continue;
          const d = tree.query(px, py, lo[2] + gz * voxel, reach);
          if (d !== Infinity) out[k] = Math.max(thickness - d, -voxel);
        }
      }
    }
  };
  const res = isoManifold(wasm, evalBlock, lo, voxel, shape, blocks, block, simplify * voxel, (x) => spIso.at(x, "shell: distance field + mesh"));
  if (!res) return null;
  const band = res.manifold, n0 = res.rawTris;
  log(`band: ${n0} -> ${band.numTri()} tris (${res.chunks} chunks)`);
  spBool.at(0, "shell: booleans");
  const outer = band.add(body);
  let shell = outer.subtract(body);
  if (trimBottom) {
    const untrimmed = shell;
    shell = untrimmed.trimByPlane([0, 0, 1], loAll[2]); // keep the model's footprint
    untrimmed.delete();
  }
  for (const m of [band, outer, body]) m.delete();
  const empty = shell.isEmpty();
  if (!empty) log(`shell: ${shell.numTri()} tris, volume ${shell.volume().toFixed(0)} mm³`);
  const out = empty ? null : fromManifold(shell);
  shell.delete();
  return out;
}
