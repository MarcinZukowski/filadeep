"""Per-color depth: turn surface paint into solid color volumes of controlled depth.

Every interior point takes the color of its nearest painted surface point, but
only if it lies within that color's depth; deeper points belong to the core.

    plug_f = { x : D_f(x) < depth_f  and  D_f(x) < D_other(x) }

where D_f is the distance to surface painted f and D_other the distance to the
rest of the surface. We evaluate g_f = min(depth_f - D_f, D_other - D_f) on a
voxel grid near the surface (in blocks, so memory stays bounded), extract the
g_f = 0 isosurface with marching tetrahedra, and intersect it with the original
mesh so the outer surface is preserved exactly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import manifold3d as m3d
import numpy as np
from scipy.spatial import cKDTree


def log(msg: str, t0: list[float] = [time.perf_counter()]) -> None:
    print(f"[{time.perf_counter() - t0[0]:6.1f}s] {msg}", flush=True)


# ------------------------------------------------------------------- colors


def luminance(hex_color: str) -> float:
    """WCAG relative luminance of '#RRGGBB' in [0, 1]."""
    c = hex_color.lstrip("#")[:6]
    try:
        rgb = [int(c[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    except ValueError:
        return 0.5
    lin = [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def default_depth(hex_color: str, dmin: float = 0.45, dmax: float = 1.6) -> float:
    """Dark colors are opaque and bleed through light ones: keep them shallow.
    Light colors are translucent and need thickness to cover what's below."""
    return dmin + (dmax - dmin) * luminance(hex_color) ** 0.5


# ------------------------------------------------------------------ helpers


def to_manifold(verts: np.ndarray, faces: np.ndarray) -> m3d.Manifold:
    m = m3d.Manifold(m3d.Mesh(vert_properties=np.ascontiguousarray(verts, np.float32), tri_verts=np.ascontiguousarray(faces, np.uint32)))
    if m.status() != m3d.Error.NoError:
        raise ValueError(f"mesh is not manifold: {m.status()}")
    return m


def from_manifold(m: m3d.Manifold) -> tuple[np.ndarray, np.ndarray]:
    mesh = m.to_mesh()
    return np.asarray(mesh.vert_properties)[:, :3].astype(np.float64), np.asarray(mesh.tri_verts).astype(np.int64)


def sample_surface(tris: np.ndarray, labels: np.ndarray, spacing: float, seed: int = 0):
    """Points on the surface ~`spacing` apart, with the label of their triangle."""
    e1 = tris[:, 1] - tris[:, 0]
    e2 = tris[:, 2] - tris[:, 0]
    area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    n = np.maximum(1, np.ceil(2 * area / spacing**2).astype(np.int64))
    idx = np.repeat(np.arange(len(tris)), n)
    r = np.random.default_rng(seed).random((len(idx), 2))
    flip = r.sum(1) > 1
    r[flip] = 1 - r[flip]
    pts = tris[idx, 0] + e1[idx] * r[:, :1] + e2[idx] * r[:, 1:]
    # Always include vertices so tiny triangles and sharp corners are covered.
    pts = np.concatenate([pts, tris.reshape(-1, 3)])
    lab = np.concatenate([labels[idx], np.repeat(labels, 3)])
    return pts, lab


# --------------------------------------------------------------- isosurface


# Cube corners as bit offsets (x=1, y=2, z=4); Freudenthal split of each cube
# into 6 tetrahedra along the main diagonal. Every cube is split the same way,
# so face diagonals agree between neighbours and the tets form a conforming mesh.
_CORNER = np.array([[(c >> 0) & 1, (c >> 1) & 1, (c >> 2) & 1] for c in range(8)])
_TETS = np.array(
    [[0, 1 << a, (1 << a) | (1 << b), 7] for a, b in ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))]
)


def _marching_tets(g: np.ndarray, i0: np.ndarray, shape: np.ndarray):
    """Isosurface g = 0 of one block (g positive inside), oriented outward.

    Returns (keys (T,3,2), points (T,3,3)): for each triangle corner, the global
    grid edge it lies on (pair of linear grid ids) and its position in grid units.
    """
    nx, ny, nz = g.shape
    # Cubes that straddle the surface.
    s = g > 0
    c = np.zeros((nx - 1, ny - 1, nz - 1), dtype=np.int8)
    for k in range(8):
        dx, dy, dz = _CORNER[k]
        c += s[dx : nx - 1 + dx, dy : ny - 1 + dy, dz : nz - 1 + dz]
    ci = np.argwhere((c > 0) & (c < 8))  # (Q,3) local cube origins
    if len(ci) == 0:
        return None
    # Tets: (Q*6, 4) corner local coords.
    corners = ci[:, None, None, :] + _CORNER[_TETS][None]  # (Q,6,4,3)
    corners = corners.reshape(-1, 4, 3)
    val = g[corners[..., 0], corners[..., 1], corners[..., 2]]  # (T,4)
    pos = val > 0
    npos = pos.sum(1)
    keep = (npos > 0) & (npos < 4)
    corners, val, pos, npos = corners[keep], val[keep], pos[keep], npos[keep]
    # Order each tet's corners: positives first.
    order = np.argsort(~pos, axis=1, kind="stable")
    corners = np.take_along_axis(corners, order[..., None], 1)
    val = np.take_along_axis(val, order, 1)

    # Edge (a, b) lists for the triangles of each case, as corner slots 0..3.
    def tri_edges(mask, edges):
        return [(corners[mask][:, a], corners[mask][:, b], val[mask][:, a], val[mask][:, b]) for a, b in edges]

    groups = [
        tri_edges(npos == 1, [(0, 1), (0, 2), (0, 3)]),
        tri_edges(npos == 3, [(0, 3), (1, 3), (2, 3)]),
        tri_edges(npos == 2, [(0, 2), (0, 3), (1, 3)]),
        tri_edges(npos == 2, [(0, 2), (1, 3), (1, 2)]),
    ]
    # Per triangle: 3 vertices, each (A, B, gA, gB) with A positive, B negative.
    A = np.concatenate([np.stack([e[0] for e in gr], 1) for gr in groups])  # (T,3,3)
    B = np.concatenate([np.stack([e[1] for e in gr], 1) for gr in groups])
    gA = np.concatenate([np.stack([e[2] for e in gr], 1) for gr in groups])  # (T,3)
    gB = np.concatenate([np.stack([e[3] for e in gr], 1) for gr in groups])
    t = gA / (gA - gB)
    p = A + (B - A) * t[..., None]  # local index coords
    # Orient outward: normal should point from positive (A) toward negative (B).
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    out_dir = (B - A).sum(1).astype(np.float64)
    flip = (n * out_dir).sum(1) < 0
    A[flip] = A[flip][:, ::-1]
    B[flip] = B[flip][:, ::-1]
    p[flip] = p[flip][:, ::-1]
    # Global edge keys.
    gA_ = A + i0
    gB_ = B + i0
    la = (gA_[..., 0] * shape[1] + gA_[..., 1]) * shape[2] + gA_[..., 2]
    lb = (gB_[..., 0] * shape[1] + gB_[..., 1]) * shape[2] + gB_[..., 2]
    keys = np.stack([np.minimum(la, lb), np.maximum(la, lb)], -1)  # (T,3,2)
    return keys, p + i0


def _isosurface_blocks(field, lo: np.ndarray, voxel: float, shape: np.ndarray, active_blocks, block: int):
    """Marching tetrahedra over a set of blocks of a virtual grid.

    Vertices are identified by the global grid edge they lie on, so blocks
    stitch exactly and the result is watertight (manifold) by construction.
    """
    all_k, all_p = [], []
    for b in active_blocks:
        i0 = np.asarray(b) * block
        i1 = np.minimum(i0 + block + 1, shape)  # +1: blocks share a face of samples
        if np.any(i1 - i0 < 2):
            continue
        axes = [lo[k] + voxel * np.arange(i0[k], i1[k]) for k in range(3)]
        X, Y, Z = np.meshgrid(*axes, indexing="ij")
        g = field(np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)).reshape(X.shape)
        # Outermost grid layer must be "outside" so surfaces close.
        for k in range(3):
            if i0[k] == 0:
                g[(slice(None),) * k + (0,)] = -voxel
            if i1[k] == shape[k]:
                g[(slice(None),) * k + (-1,)] = -voxel
        # Keep values off the iso-level so vertices lie strictly inside edges.
        # (Before the emptiness check, so neighbouring blocks agree.)
        eps = 1e-3 * voxel
        small = np.abs(g) < eps
        g[small] = np.where(g[small] < 0, -eps, eps)
        if g.max() <= 0 or g.min() > 0:
            continue
        r = _marching_tets(g, i0, shape)
        if r is None:
            continue
        all_k.append(r[0].reshape(-1, 2))
        all_p.append(r[1].reshape(-1, 3))
    if not all_k:
        return None
    k = np.concatenate(all_k)
    p = np.concatenate(all_p)
    uk, first, inv = np.unique(k, axis=0, return_index=True, return_inverse=True)
    v = lo + p[first] * voxel
    f = inv.ravel().reshape(-1, 3)
    return v, f


# ------------------------------------------------------------------- driver


@dataclass
class DepthResult:
    core: tuple[np.ndarray, np.ndarray] | None
    plugs: dict[int, tuple[np.ndarray, np.ndarray]]  # filament -> (verts, faces)


def build(
    verts: np.ndarray,
    faces: np.ndarray,
    sub_tris: np.ndarray,
    sub_fil: np.ndarray,
    depths: dict[int, float],
    core_filament: int,
    voxel: float = 0.25,
    block: int = 24,
    simplify: float = 0.3,
) -> DepthResult:
    """Split a solid (verts/faces, world mm) into per-filament volumes.

    sub_tris/sub_fil: painted surface (world mm) and filament per sub-triangle.
    depths: filament -> depth in mm. Filaments missing from `depths` (and the
    core filament) get no volume of their own; they become core.
    """
    body = to_manifold(verts, faces)
    log(f"body: {body.num_tri()} tris, volume {body.volume():.0f} mm³")

    pts, lab = sample_surface(sub_tris, sub_fil, spacing=voxel * 0.4)
    log(f"surface samples: {len(pts)}")

    lo_all = sub_tris.reshape(-1, 3).min(0)
    hi_all = sub_tris.reshape(-1, 3).max(0)

    plugs: dict[int, m3d.Manifold] = {}
    taken = None
    for f in sorted(depths, key=lambda f: depths[f]):  # shallow first: dark wins ties
        d = depths[f]
        if f == core_filament or not np.any(lab == f):
            continue
        mine = pts[lab == f]
        tree_f = cKDTree(mine)
        tree_o = cKDTree(pts[lab != f]) if np.any(lab != f) else None
        margin = d + 3 * voxel
        lo = lo_all - margin
        shape = np.ceil((hi_all + margin - lo) / voxel).astype(int) + 1

        def field(p, tree_f=tree_f, tree_o=tree_o, d=d):
            df, _ = tree_f.query(p, distance_upper_bound=d + 2 * voxel, workers=-1)
            g = np.full(len(p), -voxel, dtype=np.float64)
            near = np.isfinite(df)
            if near.any():
                pn, dn = p[near], df[near]
                if tree_o is not None:
                    do, _ = tree_o.query(pn, distance_upper_bound=d + 2 * voxel, workers=-1)
                    do = np.minimum(do, d + 2 * voxel)
                else:
                    do = np.full(len(pn), np.inf)
                g[near] = np.minimum(d - dn, do - dn)
            return np.maximum(g, -voxel)  # no upper clip: equal values cause MC saddle ties

        # Blocks that can contain part of this plug.
        reach = int(np.ceil(margin / (voxel * block)))
        bidx = np.unique(((mine - lo) / (voxel * block)).astype(int), axis=0)
        offs = np.stack(np.meshgrid(*[np.arange(-reach, reach + 1)] * 3, indexing="ij"), -1).reshape(-1, 3)
        cand = np.unique((bidx[:, None, :] + offs[None]).reshape(-1, 3), axis=0)
        nblk = np.ceil(shape / block).astype(int)
        cand = cand[np.all((cand >= 0) & (cand < nblk), axis=1)]
        log(f"filament {f}: depth {d:.2f} mm, {len(mine)} samples, {len(cand)} blocks")

        iso = _isosurface_blocks(field, lo, voxel, shape, [tuple(c) for c in cand], block)
        if iso is None:
            continue
        shell = to_manifold(*iso)
        n0 = shell.num_tri()
        # Marching tets is very dense; flatten it before the booleans. Only the
        # inner (hidden) boundaries come from this mesh, so voxel-level
        # tolerance is plenty; the outer surface still comes from `body`.
        shell = shell.simplify(simplify * voxel)
        log(f"filament {f}: shell {n0} -> {shell.num_tri()} tris")
        plug = body ^ shell  # intersection
        if taken is not None:
            plug = plug - taken
        taken = plug if taken is None else taken + plug
        log(f"filament {f}: plug {plug.num_tri()} tris, volume {plug.volume():.0f} mm³")
        plugs[f] = plug

    core = body - taken if taken is not None else body
    log(f"core: {core.num_tri()} tris, volume {core.volume():.0f} mm³")
    return DepthResult(
        core=from_manifold(core) if not core.is_empty() else None,
        plugs={f: from_manifold(m) for f, m in plugs.items() if not m.is_empty()},
    )


# ------------------------------------------------------------------ project


def process(
    project,
    depth_overrides: dict[int, float] | None = None,
    core: int | None = None,
    voxel: float = 0.25,
    dmin: float = 0.45,
    dmax: float = 1.6,
):
    """Build per-filament parts for every painted part of `project`.

    Returns {(object id, part id): [NewPart, ...]} for write3mf.replace_parts.
    """
    from . import regions, threemf
    from .write3mf import NewPart

    replacements = {}
    for s in regions.surfaces(project):
        if s.part.mesh.painted_count == 0:
            continue
        obj, part = s.object, s.part
        core_f = core if core is not None else obj.part_extruder(part)
        depths = {int(f): default_depth(project.filament_color(int(f)), dmin, dmax) for f in np.unique(s.filament)}
        depths.update(depth_overrides or {})
        log(
            f"object '{obj.name}' part '{part.name}': core filament {core_f}, depths "
            + ", ".join(f"{f}({project.filament_color(f)})={'core' if f == core_f else f'{d:.2f}mm'}" for f, d in sorted(depths.items()))
        )
        W = part.transform @ obj.transform
        Winv = np.linalg.inv(W)
        V = threemf.apply_transform(part.mesh.vertices, W)
        res = build(V, part.mesh.faces, s.tris, s.filament, depths, core_f, voxel=voxel)
        new = []
        if res.core is not None:
            v, f = res.core
            new.append(NewPart(f"{part.name} core", core_f, threemf.apply_transform(v, Winv), f))
        for fil, (v, f) in sorted(res.plugs.items()):
            new.append(NewPart(f"{part.name} color {fil}", fil, threemf.apply_transform(v, Winv), f))
        replacements[(obj.id, part.id)] = new
    return replacements
