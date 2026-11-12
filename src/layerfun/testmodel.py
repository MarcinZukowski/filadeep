"""Generate a small painted test plate: cube, pyramid and sphere.

Covers the cases LayerFun cares about, in ~1/10 of the kitsune's processing time:
  - dark details on a light body (eyes bleeding through)       cube front, sphere
  - light details on a dark body (too shallow, 1 layer)         cube right, pyramid
  - thin lines (0.8 mm) next to thick ones (4 mm)               cube right, sphere
  - near-horizontal and sloped painted surfaces                 cube top, pyramid
  - light-on-light (white cap on green)                         sphere

Palette = 5 slots (what an H2C feeds easily: 4 AMS + 1 side feed). A real Bambu
project with the same slots should be used as a settings template (`--template`):
  1 green PLA Silk+       paint
  2 light blue PLA Silk+  paint
  3 black PLA-CF          paint + core     (core --filament 3)
  4 white PLA Basic       paint + primer   (core --primer 4)
  5 translucent PLA-PVB   shell            (shell --filament 5)
The template's colours are kept (e.g. a #RRGGBBAA translucent shell); COLORS are
used only without a template.
"""

from __future__ import annotations

import json
import uuid
import zipfile
from pathlib import Path

import numpy as np

from .depth import unique_rows

GREEN, BLUE, BLACK, WHITE, SHELL = 1, 2, 3, 4, 5
# The shell slot is 50% transparent (#RRGGBBAA), so Bambu shows what is underneath.
COLORS = ["#A1FFAC", "#A4DBE8", "#000000", "#FFFFFF", "#FB000780"]
TYPES = ["PLA", "PLA", "PLA-CF", "PLA", "PLA"]
PAINT = {1: "4", 2: "8", 3: "0C", 4: "1C", 5: "2C", 6: "3C"}  # whole-triangle paint codes


# ------------------------------------------------------------------ meshes


def subdivide(v: np.ndarray, f: np.ndarray, max_edge: float, project=None):
    """Midpoint-subdivide (1 -> 4) until every edge is shorter than max_edge."""
    while True:
        e = np.linalg.norm(v[f[:, [1, 2, 0]]] - v[f], axis=2).max()
        if e <= max_edge:
            return v, f
        edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
        uniq, inv = unique_rows(edges, return_inverse=True)
        mids = (v[uniq[:, 0]] + v[uniq[:, 1]]) / 2
        if project is not None:
            mids = project(mids)
        m = inv.reshape(3, -1).T + len(v)  # midpoint ids of edges (01, 12, 20)
        a, b, c = f[:, 0], f[:, 1], f[:, 2]
        m01, m12, m20 = m[:, 0], m[:, 1], m[:, 2]
        f = np.concatenate([np.stack(t, 1) for t in ((a, m01, m20), (m01, b, m12), (m20, m12, c), (m01, m12, m20))])
        v = np.concatenate([v, mids])


def cube_mesh(size: float):
    v = np.array([[x, y, z] for x in (0, size) for y in (0, size) for z in (0, size)], float) - [size / 2, size / 2, 0]
    f = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])  # fmt: skip
    return v, f


def pyramid_mesh(base: float, height: float):
    h = base / 2
    v = np.array([[-h, -h, 0], [h, -h, 0], [h, h, 0], [-h, h, 0], [0, 0, height]], float)
    f = np.array([[0, 2, 1], [0, 3, 2], [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
    return v, f


def sphere_mesh(r: float):
    t = (1 + 5**0.5) / 2
    v = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t],
                  [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)  # fmt: skip
    f = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4],
                  [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8],
                  [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]])  # fmt: skip
    v = v / np.linalg.norm(v, axis=1, keepdims=True) * r
    return v, f


# ------------------------------------------------------------------ paint


def paint_cube(c: np.ndarray, n: np.ndarray, s: float) -> np.ndarray:
    """Light blue body. Front: black eyes + green smile. Right: black face with
    thin and thick green lines. Top: black disc with green dot. Left: white dot, thin black line."""
    lab = np.full(len(c), BLUE)
    x, y, z = c.T
    front, right, top, left = n[:, 1] < -0.9, n[:, 0] > 0.9, n[:, 2] > 0.9, n[:, 0] < -0.9
    for ex in (-s * 0.2, s * 0.2):
        lab[front & (np.hypot(x - ex, z - s * 0.65) < s * 0.1)] = BLACK
    lab[front & (np.abs(z - s * 0.3) < s * 0.05) & (np.abs(x) < s * 0.3)] = GREEN
    lab[right] = BLACK
    lab[right & (np.abs(z - s * 0.35) < 0.4)] = GREEN  # 0.8 mm line
    lab[right & (np.abs(z - s * 0.7) < 2.0)] = GREEN  # 4 mm band
    lab[top & (np.hypot(x, y) < s * 0.3)] = BLACK
    lab[top & (np.hypot(x, y) < s * 0.1)] = GREEN
    lab[left & (np.hypot(y, z - s * 0.6) < s * 0.08)] = WHITE
    lab[left & (np.abs(z - s * 0.3) < 0.4)] = BLACK
    return lab


def paint_pyramid(c: np.ndarray, n: np.ndarray, height: float) -> np.ndarray:
    """Black body: green and white horizontal bands, a light blue triangle on one face."""
    lab = np.full(len(c), BLACK)
    z = c[:, 2]
    side = n[:, 2] > -0.9
    lab[side & (np.abs(z - height * 0.25) < 1.5)] = GREEN
    lab[side & (np.abs(z - height * 0.5) < 0.6)] = WHITE
    lab[side & (z > height * 0.8)] = GREEN  # tip
    face = side & (n[:, 1] < -0.5)
    lab[face & (z > height * 0.32) & (z < height * 0.45) & (np.abs(c[:, 0]) < (height * 0.45 - z) * 0.8)] = BLUE
    return lab


def paint_sphere(c: np.ndarray, r: float) -> np.ndarray:
    """Light green body: black eyes, thin black ring, white cap, light blue meridian stripe."""
    lab = np.full(len(c), GREEN)
    u = c / np.linalg.norm(c, axis=1, keepdims=True)
    lat = np.degrees(np.arcsin(np.clip(u[:, 2], -1, 1)))
    lon = np.degrees(np.arctan2(u[:, 1], u[:, 0]))
    lab[lat > 60] = WHITE
    lab[np.abs(lat - 30) < np.degrees(0.4 / r)] = BLACK  # 0.8 mm ring
    lab[(np.abs(lon - 90) < 6) & (lat < 25) & (lat > -60)] = BLUE
    for elon in (-110, -70):
        lab[np.hypot(lon - elon, lat - 5) < 9] = BLACK
    return lab


# ------------------------------------------------------------------ 3MF


def _shapes(scale: float = 1.0, max_edge: float = 0.8):
    s = 30.0 * scale
    shapes = []
    v, f = subdivide(*cube_mesh(s), max_edge)
    shapes.append(("cube", v, f, BLUE, lambda c, n: paint_cube(c, n, s)))
    hgt = 25.0 * scale
    v, f = subdivide(*pyramid_mesh(s, hgt), max_edge)
    shapes.append(("pyramid", v, f, BLACK, lambda c, n: paint_pyramid(c, n, hgt)))
    r = 15.0 * scale
    v, f = sphere_mesh(r)
    v, f = subdivide(v, f, max_edge, project=lambda p: p / np.linalg.norm(p, axis=1, keepdims=True) * r)
    # Flat base so it prints without support (cut 2 mm off the bottom).
    from .depth import from_manifold, to_manifold

    v, f = from_manifold(to_manifold(v + [0, 0, r], f).trim_by_plane((0.0, 0.0, 1.0), 2.0 * scale).translate((0, 0, -2.0 * scale)))
    shapes.append(("sphere", v, f, GREEN, lambda c, n: paint_sphere(c - [0, 0, r - 2.0 * scale], r)))
    return shapes


def _template_bits(template) -> tuple[str, str, str]:
    """(project_settings.config, Application string, extra <plate> metadata) from a Bambu project."""
    import re

    with zipfile.ZipFile(template) as z:
        settings = z.read("Metadata/project_settings.config").decode("utf-8")
        root = z.read("3D/3dmodel.model").decode("utf-8")
        ms = z.read("Metadata/model_settings.config").decode("utf-8") if "Metadata/model_settings.config" in z.namelist() else ""
    m = re.search(r'<metadata name="Application">([^<]*)</metadata>', root)
    app = m.group(1) if m else "BambuStudio-02.00.00.00"
    # Plate-level filament/nozzle mapping (needed by multi-nozzle printers like the H2 series).
    keys = ("filament_map_mode", "filament_maps", "filament_volume_maps")
    plate = "".join(
        f"\n    {m.group(0)}" for k in keys for m in [re.search(rf'<metadata key="{k}" value="[^"]*"/>', ms)] if m
    )
    return settings, app, plate


def write(path: str | Path, template: str | Path | None = None, scale: float = 1.0, max_edge: float = 0.8) -> Path:
    path = Path(path)
    if template:
        tmpl_settings, app, plate_extra = _template_bits(template)
    else:
        tmpl_settings, app, plate_extra = None, "LayerFun", ""
    ns = 'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" xmlns:BambuStudio="http://schemas.bambulab.com/package/2021" xmlns:p="http://schemas.microsoft.com/3dmanufacturing/production/2015/06" requiredextensions="p"'
    root_objs, items, settings, rels, files = [], [], [], [], {}
    for i, (name, v, f, default, painter) in enumerate(_shapes(scale, max_edge)):
        mesh_id, obj_id = 2 * i + 1, 2 * i + 2
        t = v[f]
        n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        lab = painter(t.mean(1), n)
        vs = "\n".join(f'     <vertex x="{a:.6f}" y="{b:.6f}" z="{c:.6f}"/>' for a, b, c in v.tolist())
        ts = "\n".join(
            f'     <triangle v1="{a}" v2="{b}" v3="{c}"' + (f' paint_color="{PAINT[l]}"' if l != default else "") + "/>"
            for (a, b, c), l in zip(f.tolist(), lab.tolist())
        )
        objpath = f"3D/Objects/{name}.model"
        files[objpath] = (
            f'<?xml version="1.0" encoding="UTF-8"?>\n<model unit="millimeter" xml:lang="en-US" {ns}>\n'
            f' <resources>\n  <object id="{mesh_id}" p:UUID="{uuid.uuid4()}" type="model">\n   <mesh>\n'
            f"    <vertices>\n{vs}\n    </vertices>\n    <triangles>\n{ts}\n    </triangles>\n"
            f"   </mesh>\n  </object>\n </resources>\n <build/>\n</model>\n"
        )
        rels.append(f'<Relationship Target="/{objpath}" Id="rel-{i + 1}" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>')
        root_objs.append(
            f'  <object id="{obj_id}" p:UUID="{uuid.uuid4()}" type="model">\n   <components>\n'
            f'    <component p:path="/{objpath}" objectid="{mesh_id}" p:UUID="{uuid.uuid4()}" transform="1 0 0 0 1 0 0 0 1 0 0 0"/>\n'
            "   </components>\n  </object>"
        )
        x = 175 + (i - 1) * 45 * scale
        items.append(f'  <item objectid="{obj_id}" p:UUID="{uuid.uuid4()}" transform="1 0 0 0 1 0 0 0 1 {x:.3f} 160 0" printable="1"/>')
        settings.append(
            f'  <object id="{obj_id}">\n    <metadata key="name" value="{name}"/>\n    <metadata key="extruder" value="{default}"/>\n'
            f'    <metadata face_count="{len(f)}"/>\n'
            f'    <part id="{mesh_id}" subtype="normal_part">\n      <metadata key="name" value="{name}"/>\n'
            '      <metadata key="matrix" value="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"/>\n'
            f'      <mesh_stat face_count="{len(f)}" edges_fixed="0" degenerate_facets="0" facets_removed="0" facets_reversed="0" backwards_edges="0"/>\n'
            "    </part>\n  </object>"
        )
    files["3D/3dmodel.model"] = (
        f'<?xml version="1.0" encoding="UTF-8"?>\n<model unit="millimeter" xml:lang="en-US" {ns}>\n'
        f' <metadata name="Application">{app}</metadata>\n <metadata name="BambuStudio:3mfVersion">1</metadata>\n'
        f' <resources>\n{chr(10).join(root_objs)}\n </resources>\n <build>\n{chr(10).join(items)}\n </build>\n</model>\n'
    )
    files["3D/_rels/3dmodel.model.rels"] = (
        '<?xml version="1.0" encoding="UTF-8"?>\n<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n '
        + "\n ".join(rels) + "\n</Relationships>\n"
    )
    plate = "\n".join(
        f'    <model_instance>\n      <metadata key="object_id" value="{2 * i + 2}"/>\n      <metadata key="instance_id" value="0"/>\n    </model_instance>'
        for i in range(3)
    )
    files["Metadata/model_settings.config"] = (
        '<?xml version="1.0" encoding="UTF-8"?>\n<config>\n' + "\n".join(settings)
        + f'\n  <plate>\n    <metadata key="plater_id" value="1"/>\n    <metadata key="locked" value="false"/>{plate_extra}\n{plate}\n  </plate>\n</config>\n'
    )
    if tmpl_settings:
        n = len(json.loads(tmpl_settings).get("filament_colour", []))
        if n != len(COLORS):
            from .depth import log

            log(f"warning: template has {n} filament slots, the test plate expects {len(COLORS)} (see module docstring)")
        files["Metadata/project_settings.config"] = tmpl_settings
    else:
        files["Metadata/project_settings.config"] = json.dumps({"filament_colour": COLORS, "filament_type": TYPES}, indent=4)
    files["[Content_Types].xml"] = (
        '<?xml version="1.0" encoding="UTF-8"?>\n<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
        ' <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>\n'
        ' <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>\n'
        ' <Default Extension="png" ContentType="image/png"/>\n <Default Extension="gcode" ContentType="text/x.gcode"/>\n</Types>\n'
    )
    files["_rels/.rels"] = (
        '<?xml version="1.0" encoding="UTF-8"?>\n<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
        ' <Relationship Target="/3D/3dmodel.model" Id="rel-1" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>\n</Relationships>\n'
    )
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ["[Content_Types].xml", "_rels/.rels", "3D/3dmodel.model", "3D/_rels/3dmodel.model.rels"] + sorted(
            k for k in files if k.startswith("3D/Objects/")
        ) + ["Metadata/model_settings.config", "Metadata/project_settings.config"]:
            z.writestr(name, files[name])
    return path
