// The whole job: paint -> solid color parts (depth / core / primer), optional
// shell, written into a copy of the 3MF. Shared by the web worker and bench.mjs.

import { log, process as depthProcess, selectObjects } from "./depth.js";
import { Span } from "./progress.js";
import { arrange } from "./inlay.js";
import { buildShell } from "./shell.js";
import { applyTransform, invert, matmul, partExtruder, scaleProject } from "./threemf.js";
import { replaceParts } from "./write3mf.js";

/**
 * opts: {
 *   depths: {filament: mm} overrides, dmin, dmax, voxel, object,
 *   core: {filament, skin, primer: {filament, thickness}} | null,
 *   shell: {filament, thickness, keepBottom} | null,
 *   scale: [sx, sy, sz] factors applied to every object first (see scaleProject),
 *   fewer: true to fill the inside with a filament the layer uses anyway where its own is absent,
 *   inlay: {filaments: [n] | null (all painted), depth, clearance, leftover: "paint" | "body"} | null }  (ignored with a shell: it would cover them)
 * Returns { bytes (3MF), parts: [{name, extruder, vertices (world), faces}] }.
 */
export function run(wasm, proj, opts = {}) {
  const voxel = opts.voxel ?? 0.25;
  if (opts.scale) {
    scaleProject(proj, opts.scale);
    log(`scale: ${opts.scale.map((s) => +(s * 100).toFixed(2)).join(" × ")} %`);
  }
  const depthOverrides = new Map(Object.entries(opts.depths ?? {}).map(([k, v]) => [+k, +v]));
  const core = opts.core ?? null;
  const shellObjs = opts.shell ? selectObjects(proj, opts.object ?? null) : [];
  const [spDepth, spShell, spWrite] = new Span().split([80, shellObjs.length ? 40 : 0, 5]);
  const shellSpans = spShell.split(shellObjs.map(() => 1));
  let inlay = opts.inlay ? { ...opts.inlay, filaments: opts.inlay.filaments?.length ? new Set(opts.inlay.filaments.map(Number)) : null } : null;
  if (inlay && opts.shell) {
    log("inlays skipped: a shell would cover them");
    inlay = null;
  }
  const repl = depthProcess(wasm, proj, {
    inlay,
    fewer: !!opts.fewer,
    span: spDepth,
    depthOverrides,
    weights: opts.weights ? new Map(Object.entries(opts.weights).map(([k, v]) => [+k, +v])) : null,
    voxel,
    dmin: opts.dmin ?? 0.5,
    dmax: opts.dmax ?? 1.6,
    objectName: opts.object ?? null,
    core: core ? core.filament : null,
    uniform: core ? core.skin ?? 1.2 : null,
    primer: core?.primer ? [core.primer.filament, core.primer.thickness ?? 0.8] : null,
  });

  if (opts.shell) {
    const { filament, thickness = 1.0, keepBottom = false } = opts.shell;
    for (const [i, obj] of shellObjs.entries()) {
      const parts = obj.parts.filter((p) => p.subtype === "normal_part");
      if (!parts.length) continue;
      const bodies = parts.map((p) => ({ vertices: applyTransform(p.mesh.vertices, matmul(p.transform, obj.transform)), faces: p.mesh.faces }));
      log(`object '${obj.name}': shell filament ${filament}`);
      const res = buildShell(wasm, bodies, thickness, { voxel, trimBottom: !keepBottom, span: shellSpans[i] });
      if (!res) continue;
      const first = parts[0];
      const Winv = invert(matmul(first.transform, obj.transform));
      const key = `${obj.id}/${first.id}`;
      // Painted parts were already turned into solid parts above (paint under a
      // shell is ignored by slicers); an unpainted part is kept as it is.
      const keep = repl.get(key) ?? [{ name: first.name, extruder: partExtruder(obj, first), vertices: first.mesh.vertices, faces: first.mesh.faces }];
      repl.set(key, [...keep, { name: `${obj.name} shell`, extruder: filament, vertices: applyTransform(res.vertices, Winv), faces: res.faces }]);
    }
  }
  if (!repl.size) throw new Error("nothing to do: no painted parts found");

  // World-space copies for display.
  const parts = [];
  for (const [key, ps] of repl) {
    const [oid, pid] = key.split("/");
    const obj = proj.objects.find((o) => o.id === oid);
    const part = obj.parts.find((p) => p.id === pid);
    const W = matmul(part.transform, obj.transform);
    for (const p of ps) parts.push({ name: p.name, extruder: p.extruder, object: obj.name, vertices: applyTransform(p.vertices, W), faces: p.faces });
  }
  // Untouched parts of untouched objects, so the output view shows the whole plate.
  for (const obj of proj.objects)
    for (const part of obj.parts) {
      if (repl.has(`${obj.id}/${part.id}`) || part.subtype !== "normal_part") continue;
      parts.push({ name: part.name, extruder: partExtruder(obj, part), object: obj.name, vertices: applyTransform(part.mesh.vertices, matmul(part.transform, obj.transform)), faces: part.mesh.faces, untouched: true });
    }
  // Inlay pieces: new objects lying flat on the plate, in front of their model.
  const extra = [];
  for (const { object: obj, pieces } of repl.inlays) {
    let minx = Infinity, miny = Infinity, maxx = -Infinity;
    for (const part of obj.parts) {
      const v = applyTransform(part.mesh.vertices, matmul(part.transform, obj.transform));
      for (let i = 0; i < v.length; i += 3) {
        if (v[i] < minx) minx = v[i];
        if (v[i] > maxx) maxx = v[i];
        if (v[i + 1] < miny) miny = v[i + 1];
      }
    }
    arrange(pieces, { minx, miny, maxx });
    for (const p of pieces) {
      const name = `${obj.name} ${p.name}`;
      extra.push({ name, extruder: p.extruder, vertices: p.vertices, faces: p.faces, at: [p.at[0], p.at[1], 0], plateOf: obj.id });
      const v = Float64Array.from(p.vertices);
      for (let i = 0; i < v.length; i += 3) (v[i] += p.at[0]), (v[i + 1] += p.at[1]);
      parts.push({ name, extruder: p.extruder, object: obj.name, vertices: v, faces: p.faces, inlay: true });
    }
  }
  spWrite.at(0, "writing 3MF");
  const bytes = replaceParts(proj, repl, { extra });
  spWrite.at(1, "done");
  log(`3MF written: ${(bytes.length / 1e6).toFixed(1)} MB`);
  return { bytes, parts };
}
