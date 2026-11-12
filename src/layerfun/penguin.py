"""A painted penguin: an organic example model that is ours to share.

Built from a few blended ellipsoids (a signed distance field, meshed with the
same marching tetrahedra as the depth step) and painted by position. It has the
cases that make real painted figures hard:
  - black pupils on white eye patches          dark on light: bleeds through if too deep
  - thin yellow eyebrows on the black head     light on dark: needs depth to stay bright
  - white belly, yellow chest on a black body  long boundaries on a curved surface
  - orange beak and feet                       a color on small protrusions
Slots as in the test plate, but 1 = orange and 2 = yellow (see RECOLOR).
"""

from __future__ import annotations

import numpy as np

from .depth import _isosurface_blocks

ORANGE, YELLOW, BLACK, WHITE = 1, 2, 3, 4
RECOLOR = {1: "#FF8A00", 2: "#FFD93B"}

# (center, radii) in mm at scale 1; front is -y, the base is z = 0.
BODY = ((0, 0, 14), (18, 16, 20))
HEAD = ((0, -1, 43), (13.5, 13.5, 13.5))
TAIL = ((0, 14, 3), (5, 7, 3.5))
FLIPPERS = [((s * 17.5, 1, 24), (3.5, 7.5, 13)) for s in (-1, 1)]
BEAK = ((0, -14, 41.5), (4.2, 6.5, 3.2))
FEET = [((s * 7.5, -11, 0), (5.5, 9, 3.2)) for s in (-1, 1)]


def _ellipsoid(p: np.ndarray, prim, scale: float) -> np.ndarray:
    """Approximate signed distance to an ellipsoid, negative inside."""
    c, r = np.asarray(prim[0], float) * scale, np.asarray(prim[1], float) * scale
    q = p - c
    k0 = np.linalg.norm(q / r, axis=1)
    k1 = np.linalg.norm(q / (r * r), axis=1)
    return k0 * (k0 - 1) / np.maximum(k1, 1e-12)


def _smin(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    """Smooth union of two distance fields; k = blend radius in mm."""
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b + (a - b) * h - k * h * (1 - h)


def _fields(p: np.ndarray, scale: float):
    """(black body with head, flippers and tail; beak; feet), each negative inside."""
    body = _smin(_ellipsoid(p, BODY, scale), _ellipsoid(p, HEAD, scale), 5 * scale)
    for prim in [*FLIPPERS, TAIL]:
        body = _smin(body, _ellipsoid(p, prim, scale), 1.5 * scale)
    beak = _ellipsoid(p, BEAK, scale)
    feet = np.minimum(*(_ellipsoid(p, f, scale) for f in FEET))
    return body, beak, feet


def mesh(scale: float = 1.0, voxel: float = 0.5):
    lo = np.array([-24, -23, 0], float) * scale - [0, 0, 1.5 * voxel]  # base between two grid layers
    hi = np.array([24, 23, 58], float) * scale
    shape = np.ceil((hi - lo) / voxel).astype(int) + 1
    block = 32

    def field(p):
        body, beak, feet = _fields(p, scale)
        d = _smin(_smin(body, beak, 0.6 * scale), feet, 0.8 * scale)
        return np.minimum(-d, p[:, 2])  # inside > 0, cut flat at z = 0

    blocks = [(i, j, k) for i in range(-(-shape[0] // block)) for j in range(-(-shape[1] // block)) for k in range(-(-shape[2] // block))]
    return _isosurface_blocks(field, lo, voxel, shape, blocks, block)


def paint(c: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Filament per triangle from its centroid."""
    body, beak, feet = _fields(c, scale)
    x, y, z = (c / scale).T
    lab = np.full(len(c), BLACK)
    front = y < 0
    on_flipper = np.minimum(*(_ellipsoid(c, f, scale) for f in FLIPPERS)) < 0.3 * scale
    belly = front & ~on_flipper & ((x / 11.5) ** 2 + ((z - 15) / 16.5) ** 2 < 1)
    lab[belly] = WHITE
    lab[belly & (z > 26.5)] = YELLOW
    for s in (-1, 1):
        ex, ez = x - s * 5.2, z - 46.5
        r = np.hypot(ex, ez)
        face = front & (y < -5)
        lab[face & (r < 4.6)] = WHITE
        lab[face & (np.hypot(x - s * 4.9, ez) < 2.0)] = BLACK
        ang = np.degrees(np.arctan2(ez, s * ex))  # 0 = outward, 90 = up
        lab[face & (np.abs(r - 6.5) < 0.6) & (ang > 35) & (ang < 138)] = YELLOW  # 1.2 mm eyebrow
    lab[(beak < body + 0.3 * scale) | (feet < body + 0.3 * scale)] = ORANGE
    return lab


def shapes(scale: float = 1.0, voxel: float = 0.5):
    v, f = mesh(scale, voxel)
    return [("penguin", v, f, BLACK, lambda c, n: paint(c, scale))]
