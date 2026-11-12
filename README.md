# FilaDeep

FilaDeep takes a color-painted Bambu Studio / OrcaSlicer project (`.3mf`) and
turns the paint into real solid parts, so you control how deep each color goes.
It runs entirely in the browser; nothing is uploaded.

What it does:

- **Color depth.** Dark colors stay shallow, so black eyes don't show through
  the light plastic around them. Light colors go deeper, so white details on a
  dark body stay bright instead of being one thin layer.
- **Core.** Fill the inside with another filament (e.g. strong or cheap PLA-CF)
  and keep the colors as a skin of fixed thickness.
- **Primer.** Add a layer between the colors and the core, e.g. white over a
  black core.
- **Shell.** Wrap the model in a thin layer of another filament, e.g. clear PVB
  to smooth with alcohol.

The output is a copy of your project with the new parts; all slicer settings are
kept. Filament slots must already exist in the project. The original idea and
the problem photos are in `IDEA.md` and `img/`.

## Run it locally

Needs a recent [Node.js](https://nodejs.org/).

```bash
cd web
npm install          # once: downloads three.js, manifold-3d and fflate
node serve.mjs       # then open http://localhost:8000/
```

There is no build step: the browser loads the source files as they are, so
edit a file and reload the page.

The page opens with the `testshapes` demo. Pick options, press **RUN**, then
**Save 3MF** and open the result in your slicer.

- **Models:** open a file, drop one on the page, or use Open URL (the server
  must allow CORS). The demos are in `web/models/`.
- **Private models:** put them in the repo's `models/` folder (not in git). The
  dev server serves them too, e.g. `http://localhost:8000/#src=models/my.3mf`.
- **Links:** all options live in the address, and "Copy link" copies it, e.g.
  `#src=models/penguin.3mf&core=3&primer=4&shell=5`. Add `&run=1` to start
  processing on load.
- **Layer view:** the sliders on the right edge of both views cut the model
  from above and below, to see what's inside.

Without a browser:

```bash
node web/bench.mjs in.3mf out.3mf --core 3 --skin 1.2 --primer 4 --shell 5
```

## Hosting

`web/` is a static site: any static host works once `npm install` has run
there. It must be served over http(s); opening `index.html` from disk does not
work.

## Python version

The same pipeline as a command-line tool, which is also where the demo models
are generated: see [`cli-python/`](cli-python/README.md).

Parked ideas and open issues: `docs/LATER.md`.
