# Ideas and open issues for later

Things we found or thought of but parked. Newest findings first within each section.

## Open issues

### More filament changes than the original (black kitsune: 1295 vs 1170)
Bambu reports "saved nozzle purges" 1279 vs 1163; same thing.
- Cause: green is painted on surfaces spanning **507 layers**, but Bambu's own
  slice of the painted original prints green in only **400**. It silently drops
  thin painted slivers, mostly on near-horizontal surfaces (the "shallow plastic"
  problem). We print everything that's painted (518 layers; ~11 more from depth
  reaching past the paint). Each extra layer with green = one more filament change
  + purge.
- Tiny disconnected fragments are *not* the main driver: dropping components
  < 0.5 mm³ only goes 545 → 524 layers.
- Options:
  - `--min-feature`: drop color volumes below some size (per component, or per
    layer cross-section area), falling back to the surrounding color. Lets the
    user trade fidelity for fewer changes.
  - Anisotropic depth: measure depth with z scaled up (layers are solid and
    opaque stacked vertically, so top/bottom-facing colors may need less
    vertical depth than walls need horizontally). Also trims the ~11 extra layers.
  - Report per-color "layers used" and estimated purges in `filadeep info`/`depth`
    so the trade-off is visible before slicing.

### Longer print time (+1.5–2 h)
Light kitsune 16h 13m vs 14h 32m; black 27h 13m vs 25h 17m.
- G-code shows much more core-color sparse infill and inner wall (core filament
  inner wall 9.8 m → 18.3 m on the black one): every part gets full wall loops
  (4 in this profile), including on the hidden boundaries between parts.
- Also the extra filament changes above.
- Ideas: per-part settings in model_settings.config (fewer wall loops on color
  parts and/or core; color parts could be all solid infill), or make thin color
  parts "wall only". Needs experiments with Bambu's per-part overrides.

### Bambu CLI segfault, once
`BambuStudio --slice` crashed once on a depth output, then succeeded on every
rerun. Probably Bambu; keep an eye on it (bigger meshes?).

### Not yet verified by a real print
Is 0.45 mm of black enough to cover? Is 1.5 mm of light green enough over a blue
core? A small test piece (cut section of the head with eyes + a swirl) would answer
this and calibrate default depths.

## Speed (depth: black kitsune 93 s -> 20 s, light ~45 s -> 16 s; output identical)
Done:
- `np.unique(axis=0)` sorted rows as byte records: 60 s of the 93. Replaced by
  `unique_rows` (pack int rows into one int64 key).
- Narrow band precomputed once per shell on a 1 mm cell grid (lookup per point).
- One KD-tree per filament, built once (`balanced_tree=False`: 2x faster build,
  same query speed); duplicate vertex samples removed.
- Isosurface batches run in 3 threads (KD queries overlap numpy marching tets).
Remaining (black kitsune, ~20 s): black isosurface ~6 s, green ~3 s, manifold +
simplify ~3 s, load ~2 s (paint tessellation), write ~3 s (zlib level 6; level 1
would save ~1 s for +24% file size).
Ideas left:
- Narrow-band cell scan covers the whole bounding box (1 mm cells, one KD query
  each, per color): ~1M cells for the kitsune, but ~15M for a 250 mm model
  (~15 s per color, all empty space). Only scan cells near that color's samples
  (dilate the sample cells by reach) — same result, cost follows the surface.
- Grid-native distances (EDT per block) instead of KD queries on 0.1 mm samples.
- Fewer surface samples (spacing 0.1 -> 0.15 mm), check accuracy impact.
- Marching tets vectorized over a whole batch of blocks (now ~2000 small calls).
- Output size: ~2.5M triangles / 18 MB zipped; could simplify shells more.

## Features from IDEA.md
- **#1 core** and **#3 shell** have first versions (`filadeep core`, `filadeep shell`).
  Open points:
  - Shell bakes paint into solid color parts first (as `depth`): Bambu drops paint
    on faces that aren't the object's outside, so under a shell it vanished.
  - Print time: test plate 5h 51m → core 8h 58m, shell 11h 33m (with baked colors). Both add one more
    filament to nearly every layer (= one more change + purge per layer). Inherent
    to the idea, but worth reporting up front, and per-part settings (e.g. core as
    sparse infill only, no walls) could claw some back.
  - Interlocking between core/skin or shell/body for poorly bonding pairs (we only
    warn today, from a small hard-coded list).
  - Shell: option to leave openings, or to skip faces pointing down (supports).
  - Primer (`core --primer N`): currently sits behind every color, black included.
    Option to skip it behind dark colors (`--primer-skip 3`, or by luminance) to
    save material; geometry already supports it (just don't extend those shells).
    Test plate with core + primer: 14h 55m vs 5h 51m (two extra filaments/layer).
  - Core: the skin follows every surface including the bottom; maybe allow
    "no skin on the bottom face" (invisible anyway, saves changes on early layers).
- **Web interface**: prototype JS port in `web/` (`node web/bench.mjs in.3mf out.3mf`),
  same results as Python (Bambu: identical tool changes, print time within seconds).
  - Black kitsune: 36 s single-threaded JS vs 20 s Python (16 threads); test plate 4.1 s vs 3.8 s.
  - Bottleneck: manifold's WASM build is single-threaded and the 8.1M-triangle
    marching-tets shell costs 10 s (build + simplify) and ~4.5 GB of WASM memory,
    over the browser's 4 GB limit. Fix: fewer triangles before manifold (chunked
    build/simplify, or own pre-simplification).
  - Field: Web Workers (like Python's threads), ~8 s -> ~2 s expected.
  - KD-tree: own plain-JS tree is on par with scipy's C++ per thread. If it ever
    becomes the bottleneck, there are WASM KD-tree libraries (batch the queries;
    per-query JS<->WASM calls would eat the gain).
  - manifold-3d JS takes float32 vertices only; no visible effect in Bambu.

## Heavy input meshes
For very dense inputs (scans/sculpts, millions of triangles; e.g. kitsune.3mf), the
booleans against the full body would dominate. Idea: simplify the body within
same-paint regions (paint boundaries fixed, small tolerance), do the work, then
restore full detail on the outer surface. Manifold tracks which output triangles
come from which input face, but restoring means re-cutting the original patches
along the color walls (a boolean on the full mesh again); details to decide.
Not useful for the current bottleneck (the isosurface size depends on the grid
and painted area, not on the input triangle count).

Measured (JS, kitsune.3mf): the same kitsune at 1.8M triangles vs 282k takes ~37 s
vs ~26 s for the depth step (+40%: body/booleans 2x, more samples work), writes a
60 MB 3MF, and peaks at 5.3 GB (JS side: tessellated paint arrays and the
string-keyed vertex dedupe in sampleSurface; worth replacing with typed arrays).
A paint-aware simplifier (half-edge collapse inside same-paint regions, paint
boundaries fixed) would help such inputs; the web UI warns above 1M triangles.

## Test plate
`filadeep testmodel -o web/models/testshapes.3mf --template models/template-h2c.3mf`
uses 5 slots (H2C: 4 AMS + side feed): 1 green and 2 light blue PLA Silk+,
3 black PLA-CF (paint + core), 4 white PLA Basic (paint + primer), 5 red
translucent PLA-PVB (shell). Derived files:
`depth`, `core --filament 3`, `core --filament 3 --primer 4 --skin 1.0`,
`shell --filament 5`. It writes a cube/pyramid/sphere plate covering dark-on-light, light-on-dark, thin
lines, sloped/near-horizontal paint. `--template` copies the slicer settings,
Application tag and nozzle map from a real Bambu project (without it Bambu slices
with its defaults and ignores the paint). ~12–27 s per command vs ~45–90 s for the
kitsune.

## Tooling ideas
- `filadeep gcode-layer`: render a layer of sliced G-code colored by filament
  (exists as a scratch script; was the most useful verification tool).
- `filadeep check`: skin-color mismatch vs paint, flipped faces, films (also
  scratch scripts today).
- Paint-edge smoothing (`--smooth`) exists, off by default; revisit if paint
  sawtooth matters visually.

## Format support gaps
- Nested components, parts stored inline in the root model (some Orca files).
- PrusaSlicer per-volume settings (`Slic3r_PE_model.config`).
- Multiple painted parts per object are handled, but untested.

## Notes
- dither3d (github.com/mpetito/dither3d) is AGPL-3.0: fine to learn the format
  from, copying code would make FilaDeep AGPL (including the website).
