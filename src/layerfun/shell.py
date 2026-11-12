"""Outer shell: a layer of another filament wrapped around the model.

Use case: e.g. 1 mm of PVB around a print, smoothed with alcohol afterwards.
The shell is added as a new part of the same object.

Paint must be turned into solid parts first: Bambu/Orca only apply paint where
the painted surface is the object's outside, and under the shell it no longer
is (the paint is silently dropped when slicing). So painted parts go through
depth.process (per-color depth) before the shell is added.

    band  = { x : distance(x, surface) < t }      (both sides of the surface)
    outer = band ∪ body                            (body offset outward by t)
    shell = outer − body

Booleans only involve the band surface, which never coincides with the body
surface, so there are no coplanar cuts.
"""

from __future__ import annotations

import manifold3d as m3d
import numpy as np
from scipy.spatial import cKDTree

from .depth import _isosurface_blocks, from_manifold, log, sample_surface, to_manifold, unique_rows


def build_shell(
    bodies: list[tuple[np.ndarray, np.ndarray]],
    thickness: float,
    voxel: float = 0.25,
    block: int = 24,
    simplify: float = 0.3,
    trim_bottom: bool = True,
) -> tuple[np.ndarray, np.ndarray] | None:
    """bodies: [(verts, faces)] in world mm (parts of one object). Returns the shell mesh."""
    body = m3d.Manifold.batch_boolean([to_manifold(v, f) for v, f in bodies], m3d.OpType.Add)
    v, f = from_manifold(body)
    tris = v[f]
    log(f"body: {body.num_tri()} tris, volume {body.volume():.0f} mm³")
    pts, _ = sample_surface(tris, np.zeros(len(tris), np.int32), spacing=voxel * 0.4)
    tree = cKDTree(pts, balanced_tree=False)
    reach_mm = thickness + 2 * voxel
    margin = thickness + 3 * voxel
    lo = tris.reshape(-1, 3).min(0) - margin
    hi = tris.reshape(-1, 3).max(0) + margin
    shape = np.ceil((hi - lo) / voxel).astype(int) + 1
    band_cell = 4 * voxel

    def field(p):
        g = np.full(len(p), -voxel)
        cell = np.floor(p / band_cell).astype(np.int64)
        uc, inv = unique_rows(cell, return_inverse=True)
        dc, _ = tree.query((uc + 0.5) * band_cell, distance_upper_bound=reach_mm + band_cell * 0.8661, workers=-1)
        cand = np.isfinite(dc)[inv.ravel()]
        if cand.any():
            d, _ = tree.query(p[cand], distance_upper_bound=reach_mm, workers=-1)
            g[cand] = np.maximum(thickness - d, -voxel)
        return g

    reach = int(np.ceil(margin / (voxel * block)))
    bidx = unique_rows(((pts - lo) / (voxel * block)).astype(int))
    offs = np.stack(np.meshgrid(*[np.arange(-reach, reach + 1)] * 3, indexing="ij"), -1).reshape(-1, 3)
    cand = unique_rows((bidx[:, None, :] + offs[None]).reshape(-1, 3))
    nblk = np.ceil(shape / block).astype(int)
    cand = cand[np.all((cand >= 0) & (cand < nblk), axis=1)]
    log(f"shell {thickness:.2f} mm: {len(pts)} samples, {len(cand)} blocks")
    iso = _isosurface_blocks(field, lo, voxel, shape, [tuple(c) for c in cand], block)
    if iso is None:
        return None
    band = to_manifold(*iso)
    n0 = band.num_tri()
    band = band.simplify(simplify * voxel)
    log(f"band: {n0} -> {band.num_tri()} tris")
    shell = (band + body) - body
    if trim_bottom:
        # Keep the model's footprint: nothing below the original bottom.
        zmin = float(v[:, 2].min())
        shell = shell.trim_by_plane((0.0, 0.0, 1.0), zmin)
    if shell.is_empty():
        return None
    log(f"shell: {shell.num_tri()} tris, volume {shell.volume():.0f} mm³")
    return from_manifold(shell)


def process(project, filament: int, thickness: float, voxel: float = 0.25, trim_bottom: bool = True,
            object_name: str | None = None, depth_overrides: dict[int, float] | None = None):  # fmt: skip
    """Add a shell part to each selected object. Returns replacements for write3mf.replace_parts."""
    from . import depth, threemf
    from .depth import select_objects
    from .write3mf import NewPart

    # Painted parts -> solid color parts (the paint would be hidden from the slicer).
    replacements = depth.process(project, depth_overrides, voxel=voxel, object_name=object_name)
    for obj in select_objects(project, object_name):
        parts = [p for p in obj.parts if p.subtype == "normal_part"]
        if not parts:
            continue
        bodies = []
        for p in parts:
            W = p.transform @ obj.transform
            bodies.append((threemf.apply_transform(p.mesh.vertices, W), p.mesh.faces))
        log(f"object '{obj.name}': {len(parts)} part(s), shell filament {filament}")
        res = build_shell(bodies, thickness, voxel=voxel, trim_bottom=trim_bottom)
        if res is None:
            continue
        # Attach the shell next to the first part (same component transform),
        # after its color parts, or the unchanged part if it had no paint.
        first = parts[0]
        Winv = np.linalg.inv(first.transform @ obj.transform)
        keep = replacements.get((obj.id, first.id)) or [
            NewPart(first.name, obj.part_extruder(first), first.mesh.vertices, first.mesh.faces, paint=first.mesh.paint)
        ]
        shell = NewPart(f"{obj.name} shell", filament, threemf.apply_transform(res[0], Winv), res[1])
        replacements[(obj.id, first.id)] = keep + [shell]
    return replacements
