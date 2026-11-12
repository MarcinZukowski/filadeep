"""Per-filament surface regions of a painted project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import paint
from .threemf import Object, Part, Project, apply_transform


@dataclass
class Surface:
    """A part's surface split into sub-triangles, each with a resolved filament."""

    object: Object
    part: Part
    tris: np.ndarray  # (N, 3, 3) world coordinates
    filament: np.ndarray  # (N,) 1-based filament per sub-triangle
    src_face: np.ndarray  # (N,) index into part.mesh.faces

    def areas(self) -> np.ndarray:
        e1 = self.tris[:, 1] - self.tris[:, 0]
        e2 = self.tris[:, 2] - self.tris[:, 0]
        return 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)

    def area_by_filament(self) -> dict[int, float]:
        a = self.areas()
        return {int(f): float(a[self.filament == f].sum()) for f in np.unique(self.filament)}


def surfaces(project: Project) -> list[Surface]:
    out = []
    for obj in project.objects:
        for part in obj.parts:
            m = part.mesh
            tris, states, src = paint.tessellate(m.vertices, m.faces, m.paint)
            world = part.transform @ obj.transform
            tris = apply_transform(tris.reshape(-1, 3), world).reshape(-1, 3, 3)
            fil = np.where(states == 0, obj.part_extruder(part), states)
            out.append(Surface(obj, part, tris, fil, src))
    return out


# ------------------------------------------------------------------ export


def write_stl(path: Path, tris: np.ndarray) -> None:
    """Binary STL."""
    tris = np.asarray(tris, dtype=np.float32).reshape(-1, 3, 3)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, ln, out=np.zeros_like(n), where=ln > 0)
    rec = np.zeros(len(tris), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    rec["n"] = n
    rec["v"] = tris
    with open(path, "wb") as f:
        f.write(b"filadeep".ljust(80, b"\0"))
        f.write(np.uint32(len(tris)).tobytes())
        f.write(rec.tobytes())


def hex_rgb(c: str) -> tuple[int, int, int]:
    c = c.lstrip("#")[:6]
    try:
        return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    except (ValueError, IndexError):
        return 128, 128, 128


def write_ply(path: Path, tris: np.ndarray, rgb: np.ndarray) -> None:
    """Binary PLY with per-face colors (unshared vertices)."""
    tris = np.asarray(tris, dtype=np.float32).reshape(-1, 3, 3)
    n = len(tris)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {3 * n}\nproperty float x\nproperty float y\nproperty float z\n"
        f"element face {n}\nproperty list uchar int vertex_indices\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    faces = np.zeros(n, dtype=[("k", "u1"), ("idx", "<i4", 3), ("rgb", "u1", 3)])
    faces["k"] = 3
    faces["idx"] = np.arange(3 * n, dtype=np.int32).reshape(-1, 3)
    faces["rgb"] = rgb
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(tris.reshape(-1, 3).tobytes())
        f.write(faces.tobytes())


def export(project: Project, outdir: Path) -> list[Path]:
    """Write one STL per filament plus a colored PLY preview of everything."""
    outdir.mkdir(parents=True, exist_ok=True)
    surfs = surfaces(project)
    all_tris = np.concatenate([s.tris for s in surfs])
    all_fil = np.concatenate([s.filament for s in surfs])
    written = []
    for f in np.unique(all_fil):
        p = outdir / f"filament_{int(f)}.stl"
        write_stl(p, all_tris[all_fil == f])
        written.append(p)
    palette = np.array([hex_rgb(project.filament_color(i)) for i in range(int(all_fil.max()) + 1)], dtype=np.uint8)
    p = outdir / "preview.ply"
    write_ply(p, all_tris, palette[all_fil])
    written.append(p)
    return written
