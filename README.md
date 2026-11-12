# LayerFun

Tools for painted multi-color 3MF files (Bambu Studio / OrcaSlicer). See `IDEA.md`.

```bash
uv run layerfun info model.3mf                         # objects, parts, painted area per filament
uv run layerfun render model.3mf --views iso,front     # quick PNG previews

uv run layerfun depth model.3mf                        # #2 per-color depth: dark shallow, light deep
uv run layerfun depth model.3mf --depth 3=0.4,1=1.8    #    override depths (mm) per filament slot
uv run layerfun core  model.3mf --filament 3 --skin 1.2     # #1 colored skin + core in another filament
uv run layerfun core  model.3mf --filament 3 --primer 4     #    ...with a primer layer between (e.g. white over black PLA-CF)
uv run layerfun shell model.3mf --filament 5 --thickness 1  # #3 wrap in another filament (e.g. PVB / translucent PLA)

uv run layerfun testmodel -o web/models/testshapes.3mf --template models/template-h2c.3mf
uv run layerfun testmodel --model penguin -o web/models/penguin.3mf --template models/template-h2c.3mf   # organic example
```

Each command writes `<file>-<command>.3mf` (or `-o`). Filament slots must already exist
in the Bambu project. Parked ideas and open issues: `docs/LATER.md`.

Tests: `uv run pytest`

## Web version (prototype)

A browser port of the same pipeline (JavaScript + manifold's WASM build), with
side-by-side source/output views, a shared camera and a layer slicer.

```bash
cd web && npm install && cd ..
node web/serve.mjs            # http://localhost:8000/
```

- Load a model from a URL (the server must allow CORS), a file, or drag & drop.
  The demos live in `web/models/` and are referenced relative to the page: `#src=models/testshapes.3mf`.
  The dev server also serves the repo's untracked `models/` under the same path, for private models.
- Options live in the URL hash; "Copy link" gives a link with the source URL and
  all settings, e.g. `#src=models/testshapes.3mf&core=3&primer=4&skin=1.0&shell=5`
  (add `&run=1` to start processing on load).
- `node web/bench.mjs in.3mf out.3mf [--core 3 --skin 1 --primer 4] [--shell 5]` runs it in Node.
