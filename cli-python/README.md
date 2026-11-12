# FilaDeep command-line tool (Python)

The same pipeline as the [web app](../README.md), as a command-line tool. The
demo models are generated here too. Needs [uv](https://docs.astral.sh/uv/); run the commands
from this folder.

```bash
uv run filadeep info model.3mf                              # objects, parts, painted area per filament
uv run filadeep render model.3mf --views iso,front          # quick PNG previews

uv run filadeep depth model.3mf                             # per-color depth: dark shallow, light deep
uv run filadeep depth model.3mf --depth 3=0.4,1=1.8         # override depths (mm) per filament slot
uv run filadeep core  model.3mf --filament 3 --skin 1.2     # colored skin + core in another filament
uv run filadeep core  model.3mf --filament 3 --primer 4     # ...with a primer layer between
uv run filadeep shell model.3mf --filament 5 --thickness 1  # wrap in another filament

uv run filadeep testmodel -o ../web/models/testshapes.3mf --template template-h2c.3mf
uv run filadeep testmodel --model penguin -o ../web/models/penguin.3mf --template template-h2c.3mf
```

Each command writes `<file>-<command>.3mf` (or `-o`).

Tests: `uv run pytest`

`template-h2c.3mf` is a Bambu Studio project (H2C, 5 filament slots) whose slicer
settings the demo models copy.
