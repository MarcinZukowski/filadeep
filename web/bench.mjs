// Run the JS port in Node and time each step.
//   node bench.mjs in.3mf out.3mf [--core N --skin S [--primer N --primer-thickness T]]
//                                 [--shell N --thickness T] [--depth 3=0.4,1=1.5] [--object NAME]
import { readFileSync, writeFileSync } from "node:fs";
import Module from "manifold-3d";
import { load } from "./src/threemf.js";
import { log, resetClock } from "./src/depth.js";
import { run } from "./src/pipeline.js";

const a = { _: [] };
const argv = globalThis.process.argv.slice(2);
for (let i = 0; i < argv.length; i++) argv[i].startsWith("--") ? (a[argv[i].slice(2)] = argv[++i]) : a._.push(argv[i]);

resetClock();
const wasm = await Module();
wasm.setup();
const proj = load(readFileSync(a._[0]), a._[0]);
log(`loaded ${proj.objects.length} objects`);
const opts = { object: a.object, voxel: a.voxel ? +a.voxel : undefined, depths: Object.fromEntries((a.depth ?? "").split(",").filter(Boolean).map((kv) => kv.split("="))) };
if (a.core) opts.core = { filament: +a.core, skin: +(a.skin ?? 1.2), primer: a.primer ? { filament: +a.primer, thickness: +(a["primer-thickness"] ?? 0.8) } : null };
if (a.shell) opts.shell = { filament: +a.shell, thickness: +(a.thickness ?? 1.0) };
const { bytes } = run(wasm, proj, opts);
writeFileSync(a._[1], bytes);
log(`wrote ${a._[1]}`);
