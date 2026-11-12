"""Read painted 3MF projects (Bambu Studio / OrcaSlicer, with basic PrusaSlicer support)."""

from __future__ import annotations

import json
import re
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

NS_CORE = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
NS_PROD = "http://schemas.microsoft.com/3dmanufacturing/production/2015/06"
NS_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
C = f"{{{NS_CORE}}}"
P_PATH = f"{{{NS_PROD}}}path"


@dataclass
class Mesh:
    vertices: np.ndarray  # (V, 3) float64
    faces: np.ndarray  # (F, 3) int64
    paint: list[str | None]  # per face, raw paint string or None

    @property
    def painted_count(self) -> int:
        return sum(1 for p in self.paint if p)


@dataclass
class Part:
    id: str  # component object id (matches <part id> in model_settings.config)
    name: str
    mesh: Mesh
    transform: np.ndarray  # 4x4, row-vector convention, part -> object
    extruder: int | None = None  # None = inherit from object
    subtype: str = "normal_part"


@dataclass
class Object:
    id: str
    name: str
    parts: list[Part]
    transform: np.ndarray  # 4x4 build transform, object -> world
    extruder: int = 1

    def part_extruder(self, part: Part) -> int:
        return part.extruder if part.extruder is not None else self.extruder


@dataclass
class Project:
    path: Path
    objects: list[Object]
    filament_colors: list[str] = field(default_factory=list)  # "#RRGGBB", index 0 = filament 1
    filament_types: list[str] = field(default_factory=list)  # "PLA", "PETG", ... index 0 = filament 1
    application: str = ""

    def filament_color(self, n: int) -> str:
        if 1 <= n <= len(self.filament_colors):
            return self.filament_colors[n - 1]
        return "#808080"

    def filament_type(self, n: int) -> str:
        if 1 <= n <= len(self.filament_types):
            return self.filament_types[n - 1]
        return "?"


# ------------------------------------------------------------------ helpers


def parse_transform(s: str | None) -> np.ndarray:
    """3MF 3x4 transform string -> 4x4 matrix (row-vector convention: p' = [p,1] @ M)."""
    m = np.eye(4)
    if s:
        v = [float(x) for x in s.split()]
        if len(v) != 12:
            raise ValueError(f"bad transform: {s!r}")
        m[:4, :3] = np.array(v).reshape(4, 3)
    return m


def apply_transform(points: np.ndarray, m: np.ndarray) -> np.ndarray:
    return points @ m[:3, :3] + m[3, :3]


def _paint_attr(el: ET.Element) -> str | None:
    p = el.get("paint_color")
    if p:
        return p
    for k, v in el.attrib.items():
        if k.endswith("mmu_segmentation") and v:
            return v
    return None


def _parse_mesh(mesh_el: ET.Element) -> Mesh:
    verts = mesh_el.find(f"{C}vertices")
    tris = mesh_el.find(f"{C}triangles")
    vs = np.array(
        [(float(v.get("x")), float(v.get("y")), float(v.get("z"))) for v in verts],
        dtype=np.float64,
    ).reshape(-1, 3)
    faces = []
    paint = []
    for t in tris:
        faces.append((int(t.get("v1")), int(t.get("v2")), int(t.get("v3"))))
        paint.append(_paint_attr(t))
    return Mesh(vs, np.array(faces, dtype=np.int64).reshape(-1, 3), paint)


def _root_model_path(z: zipfile.ZipFile) -> str:
    try:
        rels = ET.fromstring(z.read("_rels/.rels"))
        for r in rels:
            if r.get("Type", "").endswith("/3dmodel"):
                return r.get("Target").lstrip("/")
    except KeyError:
        pass
    return "3D/3dmodel.model"


def _metadata(el: ET.Element) -> dict[str, str]:
    return {m.get("key"): m.get("value") for m in el.findall("metadata") if m.get("key")}


# --------------------------------------------------------------------- load


def load(path: str | Path) -> Project:
    path = Path(path)
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        root_path = _root_model_path(z)
        model_cache: dict[str, dict[str, ET.Element]] = {}

        def objects_in(model_path: str) -> dict[str, ET.Element]:
            model_path = model_path.lstrip("/")
            if model_path not in model_cache:
                root = ET.fromstring(z.read(model_path))
                res = root.find(f"{C}resources")
                model_cache[model_path] = {o.get("id"): o for o in res.findall(f"{C}object")}
            return model_cache[model_path]

        root = ET.fromstring(z.read(root_path))
        app = ""
        for m in root.findall(f"{C}metadata"):
            if m.get("name") == "Application":
                app = (m.text or "").strip()
        root_objs = objects_in(root_path)

        # Per-object / per-part settings (Bambu / Orca).
        settings: dict[str, dict] = {}
        if "Metadata/model_settings.config" in names:
            cfg = ET.fromstring(z.read("Metadata/model_settings.config"))
            for o in cfg.findall("object"):
                settings[o.get("id")] = {
                    "meta": _metadata(o),
                    "parts": {p.get("id"): (_metadata(p), p.get("subtype", "normal_part")) for p in o.findall("part")},
                }

        objects: list[Object] = []
        build = root.find(f"{C}build")
        for item in build.findall(f"{C}item"):
            oid = item.get("objectid")
            oel = root_objs[oid]
            st = settings.get(oid, {"meta": {}, "parts": {}})
            meta = st["meta"]
            parts: list[Part] = []
            mesh_el = oel.find(f"{C}mesh")
            if mesh_el is not None:
                parts.append(Part(oid, oel.get("name") or meta.get("name", oid), _parse_mesh(mesh_el), np.eye(4)))
            comps = oel.find(f"{C}components")
            if comps is not None:
                for comp in comps.findall(f"{C}component"):
                    cpath = comp.get(P_PATH) or root_path
                    cid = comp.get("objectid")
                    cel = objects_in(cpath)[cid]
                    cmesh = cel.find(f"{C}mesh")
                    if cmesh is None:
                        raise NotImplementedError("nested components are not supported yet")
                    pmeta, subtype = st["parts"].get(cid, ({}, "normal_part"))
                    ext = pmeta.get("extruder")
                    parts.append(
                        Part(
                            id=cid,
                            name=pmeta.get("name") or cel.get("name") or cid,
                            mesh=_parse_mesh(cmesh),
                            transform=parse_transform(comp.get("transform")),
                            extruder=int(ext) if ext else None,
                            subtype=subtype,
                        )
                    )
            objects.append(
                Object(
                    id=oid,
                    name=meta.get("name") or oel.get("name") or f"object {oid}",
                    parts=parts,
                    transform=parse_transform(item.get("transform")),
                    extruder=int(meta.get("extruder", 1) or 1),
                )
            )

        colors: list[str] = []
        types: list[str] = []
        if "Metadata/project_settings.config" in names:
            try:
                ps = json.loads(z.read("Metadata/project_settings.config"))
                colors = list(ps.get("filament_colour", []))
                types = list(ps.get("filament_type", []))
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
        elif "Metadata/Slic3r_PE.config" in names:
            txt = z.read("Metadata/Slic3r_PE.config").decode("utf-8", "replace")
            m = re.search(r"^; extruder_colour = (.*)$", txt, re.M)
            if m:
                colors = [c.strip().strip('"') for c in m.group(1).split(";")]

    return Project(path=path, objects=objects, filament_colors=colors, filament_types=types, application=app)
