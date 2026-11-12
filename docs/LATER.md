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
  - Report per-color "layers used" and estimated purges in `layerfun info`/`depth`
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

## Speed (depth step: ~45 s light kitsune, ~80 s black)
Profiled: ~80% is KD-tree distance queries over the voxel grid.
- Done: batch queries, narrow-band prefilter (−15%).
- One pass for all colors: query each grid point once against an all-samples tree
  (nearest label + distance) instead of re-evaluating overlapping blocks per color.
- Grid-native distances: rasterize labeled surface into the voxel grid and use an
  exact EDT (`scipy.ndimage.distance_transform_edt` with indices) per block with a
  halo, instead of KD queries on 0.1 mm surface samples.
- Fewer surface samples (spacing 0.1 → 0.15 mm), check accuracy impact.
- numba / compiled inner loop for field + marching tets.
- Output size: ~2.5M triangles / 30 MB; could simplify the shells more, or
  decimate interior-only faces harder.

## Features from IDEA.md
- **#1 core** and **#3 shell** have first versions (`layerfun core`, `layerfun shell`).
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
- **Web interface**: manifold3d has a WASM build; keep the core a library.

## Test plate
`layerfun testmodel -o models/testshapes.3mf --template models/template-h2c.3mf`
uses 5 slots (H2C: 4 AMS + side feed): 1 black PLA-CF (paint + core), 2 green,
3 light blue, 4 white (paint + primer), 5 translucent PLA (shell). Derived files:
`depth`, `core --filament 1`, `core --filament 1 --primer 4 --skin 1.0`,
`shell --filament 5`. It writes a cube/pyramid/sphere plate covering dark-on-light, light-on-dark, thin
lines, sloped/near-horizontal paint. `--template` copies the slicer settings,
Application tag and nozzle map from a real Bambu project (without it Bambu slices
with its defaults and ignores the paint). ~12–27 s per command vs ~45–90 s for the
kitsune.

## Tooling ideas
- `layerfun gcode-layer`: render a layer of sliced G-code colored by filament
  (exists as a scratch script; was the most useful verification tool).
- `layerfun check`: skin-color mismatch vs paint, flipped faces, films (also
  scratch scripts today).
- Paint-edge smoothing (`--smooth`) exists, off by default; revisit if paint
  sawtooth matters visually.

## Format support gaps
- Nested components, parts stored inline in the root model (some Orca files).
- PrusaSlicer per-volume settings (`Slic3r_PE_model.config`).
- Multiple painted parts per object are handled, but untested.

## Notes
- dither3d (github.com/mpetito/dither3d) is AGPL-3.0: fine to learn the format
  from, copying code would make LayerFun AGPL (including the website).
