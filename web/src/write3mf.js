// Write a modified copy of a Bambu/Orca 3MF, replacing parts with new meshes.
// Port of write3mf.py: the small XML files are edited textually so everything
// else stays byte-identical.

import { zipSync } from "../node_modules/fflate/esm/browser.js";

const enc = new TextEncoder();
const dec = new TextDecoder();
const NS_CORE = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02";

const uuid = () =>
  globalThis.crypto?.randomUUID?.() ??
  "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => ((Math.random() * 16) | (c === "x" ? 0 : 8)).toString(16));

// 9 significant digits, like Python's "%.9g".
const num = (x) => String(+x.toPrecision(9));

function meshXml(id, part) {
  const v = part.vertices, f = part.faces;
  const out = [`  <object id="${id}" p:UUID="${uuid()}" type="model">\n   <mesh>\n    <vertices>\n`];
  for (let i = 0; i < v.length; i += 3) out.push(`     <vertex x="${num(v[i])}" y="${num(v[i + 1])}" z="${num(v[i + 2])}"/>\n`);
  out.push("    </vertices>\n    <triangles>\n");
  for (let i = 0; i < f.length; i += 3) out.push(`     <triangle v1="${f[i]}" v2="${f[i + 1]}" v3="${f[i + 2]}"/>\n`);
  out.push("    </triangles>\n   </mesh>\n  </object>\n");
  return out.join("");
}

function objectFile(objs) {
  return (
    '<?xml version="1.0" encoding="UTF-8"?>\n' +
    `<model unit="millimeter" xml:lang="en-US" xmlns="${NS_CORE}" ` +
    'xmlns:BambuStudio="http://schemas.bambulab.com/package/2021" ' +
    'xmlns:p="http://schemas.microsoft.com/3dmanufacturing/production/2015/06" requiredextensions="p">\n' +
    ' <metadata name="BambuStudio:3mfVersion">1</metadata>\n' +
    ` <resources>\n${objs.map(([i, p]) => meshXml(i, p)).join("")} </resources>\n <build/>\n</model>\n`
  );
}

const esc = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const attr = (tag, name) => (tag.match(new RegExp(`\\s${esc(name)}="([^"]*)"`)) || [])[1] ?? null;

function findBlock(xml, tag, id) {
  const m = new RegExp(`<${tag} id="${esc(id)}"[^>]*>[\\s\\S]*?</${tag}>`).exec(xml);
  if (!m) throw new Error(`<${tag} id=${id}> not found`);
  return [m.index, m.index + m[0].length];
}

/** replacements: Map("objId/partId" -> [{name, extruder, vertices, faces}]). Returns zip bytes. */
export function replaceParts(proj, replacements, { level = 6 } = {}) {
  const files = { ...proj.files };
  const text = (n) => dec.decode(files[n]);
  const rootPath = proj.rootPath;
  let root = text(rootPath);
  let settings = files["Metadata/model_settings.config"] ? text("Metadata/model_settings.config") : "";

  let maxId = 0;
  for (const [n, b] of Object.entries(files)) {
    if (!/\.(model|config)$/.test(n)) continue;
    for (const m of dec.decode(b).matchAll(/(?:objectid|id)="(\d+)"/g)) maxId = Math.max(maxId, +m[1]);
  }
  let nextId = maxId + 1;

  for (const [key, parts] of replacements) {
    const [oid, pid] = key.split("/");
    // ---- root model: one component per new part
    const [o0, o1] = findBlock(root, "object", oid);
    let block = root.slice(o0, o1);
    const comp = (block.match(new RegExp(`<component\\b[^>]*\\bobjectid="${esc(pid)}"[^>]*/>`)) || [])[0];
    if (!comp) throw new Error(`component ${pid} of object ${oid} not found`);
    const path = attr(comp, "p:path");
    if (!path) throw new Error("parts stored inline in the root model are not supported yet");
    const ids = parts.map(() => nextId++);
    const comps = ids.map((i) => comp.replace(/objectid="\d+"/, `objectid="${i}"`).replace(/p:UUID="[^"]*"/, `p:UUID="${uuid()}"`));
    block = block.replace(comp, comps.join("\n    "));
    root = root.slice(0, o0) + block + root.slice(o1);

    // ---- object file: new meshes, keeping other objects in it
    const objfile = path.replace(/^\//, "");
    const old = text(objfile);
    const keep = [...old.matchAll(/<object\b[\s\S]*?<\/object>/g)].map((m) => m[0]).filter((o) => attr(o.split(">")[0], "id") !== pid);
    let xml = objectFile(ids.map((i, k) => [i, parts[k]]));
    if (keep.length) xml = xml.replace(" </resources>", keep.map((k) => "  " + k + "\n").join("") + " </resources>");
    files[objfile] = enc.encode(xml);

    // ---- model_settings.config: one <part> per new part, metadata copied
    if (settings) {
      const [s0, s1] = findBlock(settings, "object", oid);
      let sblock = settings.slice(s0, s1);
      const pm = sblock.match(new RegExp(`<part id="${esc(pid)}"[^>]*>[\\s\\S]*?</part>`));
      if (pm) {
        const tmpl = pm[0];
        const np = ids.map((i, k) => {
          const p = parts[k];
          let t = tmpl.replace(`<part id="${pid}"`, `<part id="${i}"`);
          t = t.replace(/uuid="[^"]*"/, `uuid="${uuid()}"`);
          t = t.replace(/<metadata key="name" value="[^"]*"\/>/, `<metadata key="name" value="${p.name}"/>`);
          t = t.replace(/\s*<metadata key="extruder" value="\d+"\/>/g, "");
          t = t.replace('<metadata key="name"', `<metadata key="extruder" value="${p.extruder}"/>\n      <metadata key="name"`);
          t = t.replace(/(<mesh_stat face_count=")\d+(")/, `$1${p.faces.length / 3}$2`);
          return t;
        });
        sblock = sblock.replace(tmpl, np.join("\n    "));
        const total = parts.reduce((s, p) => s + p.faces.length / 3, 0);
        sblock = sblock.replace(/<metadata face_count="\d+"\/>/, `<metadata face_count="${total}"/>`);
        settings = settings.slice(0, s0) + sblock + settings.slice(s1);
      }
    }
  }
  files[rootPath] = enc.encode(root);
  if (settings) files["Metadata/model_settings.config"] = enc.encode(settings);
  return zipSync(files, { level });
}
