#!/bin/bash
# Assemble the static site for hosting (GitHub Pages):  web/make-site.sh <outdir>
# The app itself needs no build; this only copies it together with the few
# library files the browser loads from node_modules (a few MB instead of all of it).
set -eu
here="$(cd "$(dirname "$0")" && pwd)"
out="${1:?usage: make-site.sh <outdir>}"
[ -d "$here/node_modules/three" ] || { echo "run 'npm ci' in web/ first" >&2; exit 1; }
rm -rf "$out"
mkdir -p "$out"
cp "$here"/index.html "$here"/style.css "$here"/app.js "$here"/worker.js "$out"/
cp -R "$here"/src "$here"/models "$out"/
nm="$out/node_modules"
mkdir -p "$nm/three/build" "$nm/three/examples/jsm" "$nm/manifold-3d" "$nm/fflate/esm"
cp "$here"/node_modules/three/build/three.module.js "$here"/node_modules/three/LICENSE "$nm/three/build/"
cp -R "$here"/node_modules/three/examples/jsm/controls "$here"/node_modules/three/examples/jsm/lines "$nm/three/examples/jsm/"
cp "$here"/node_modules/manifold-3d/manifold.js "$here"/node_modules/manifold-3d/manifold.wasm "$here"/node_modules/manifold-3d/LICENSE "$nm/manifold-3d/"
cp "$here"/node_modules/fflate/esm/browser.js "$here"/node_modules/fflate/LICENSE "$nm/fflate/esm/"
touch "$out/.nojekyll"
du -sh "$out" | sed 's/^/site: /'
