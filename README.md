# LayerFun

Tools for painted multi-color 3MF files (Bambu Studio / OrcaSlicer). See `IDEA.md`.

```bash
uv run layerfun info model.3mf                         # objects, parts, painted area per filament
uv run layerfun render model.3mf --views iso,front     # quick PNG previews

uv run layerfun depth model.3mf                        # #2 per-color depth: dark shallow, light deep
uv run layerfun depth model.3mf --depth 3=0.4,1=1.8    #    override depths (mm) per filament slot
uv run layerfun core  model.3mf --filament 5 --skin 1.2     # #1 colored skin + core in another filament
uv run layerfun shell model.3mf --filament 6 --thickness 1  # #3 wrap in another filament (e.g. PVB)

uv run layerfun testmodel -o models/testshapes.3mf --template some-bambu-project.3mf
```

Each command writes `<file>-<command>.3mf` (or `-o`). Filament slots must already exist
in the Bambu project. Parked ideas and open issues: `docs/LATER.md`.

Tests: `uv run pytest`
