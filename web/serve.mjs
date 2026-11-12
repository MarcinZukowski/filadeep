// Minimal static server for local development:  node web/serve.mjs [port]
// Serves web/ at /, and the repo's models/ at /models/ (for ?src=/models/...).
import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { extname, join, normalize, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = fileURLToPath(new URL(".", import.meta.url));
const roots = [["/models/", resolve(here, "../models")], ["/", here]];
const types = {
  ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css",
  ".wasm": "application/wasm", ".json": "application/json", ".3mf": "application/octet-stream", ".png": "image/png",
};
const port = +(process.argv[2] ?? 8000);

createServer(async (req, res) => {
  const path = decodeURIComponent(new URL(req.url, "http://x").pathname);
  for (const [prefix, dir] of roots) {
    if (!path.startsWith(prefix)) continue;
    let file = normalize(join(dir, path.slice(prefix.length)));
    if (!file.startsWith(dir)) break;
    try {
      if ((await stat(file)).isDirectory()) file = join(file, "index.html");
      const body = await readFile(file);
      res.writeHead(200, { "Content-Type": types[extname(file)] ?? "application/octet-stream", "Cache-Control": "no-cache" });
      return res.end(body);
    } catch {
      break;
    }
  }
  res.writeHead(404).end("not found");
}).listen(port, () => console.log(`LayerFun web: http://localhost:${port}/`));
