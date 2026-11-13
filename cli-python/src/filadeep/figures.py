"""More painted example figures, built like the penguin: blended ellipsoids
meshed with marching tetrahedra, painted by position. Front is -y, the base is z = 0.

Both have a light body, so a dark core dulls the colours unless a primer is used:
  frog   green body (the test palette as it is): light-blue belly, white eyes with
         black pupils, a thin black mouth, black spots with light-blue dots on the back
  duck   yellow body (slots 1 and 2 recoloured to orange and yellow): orange beak,
         black eyes with a white glint
"""

from __future__ import annotations

import numpy as np

from .depth import _isosurface_blocks
from .penguin import _ellipsoid, _smin

GREEN, BLUE, BLACK, WHITE = 1, 2, 3, 4  # frog: the test palette
ORANGE, YELLOW = 1, 2  # duck: see DUCK_RECOLOR
DUCK_RECOLOR = {1: "#FF8A00", 2: "#FFD400"}


def _mesh(field, lo, hi, scale: float, voxel: float):
    """Isosurface of field (negative inside), cut flat at z = 0."""
    lo = np.asarray(lo, float) * scale - [0, 0, 1.5 * voxel]  # base between two grid layers
    hi = np.asarray(hi, float) * scale
    shape = np.ceil((hi - lo) / voxel).astype(int) + 1
    block = 32
    blocks = [(i, j, k) for i in range(-(-shape[0] // block)) for j in range(-(-shape[1] // block)) for k in range(-(-shape[2] // block))]
    return _isosurface_blocks(lambda p: np.minimum(-field(p), p[:, 2]), lo, voxel, shape, blocks, block)


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _toward(c: np.ndarray, center, direction, scale: float) -> np.ndarray:
    """Cosine of the angle between (point - center) and direction."""
    q = c - np.asarray(center, float) * scale
    return (q @ _unit(direction)) / np.maximum(np.linalg.norm(q, axis=1), 1e-9)


# ------------------------------------------------------------------ frog

F_BODY = ((0, 3, 10), (19, 22, 14))
F_HEAD = ((0, -13, 19), (15, 13, 10))
F_EYES = [((s * 8, -14, 28), (5.5, 5.5, 5.5)) for s in (-1, 1)]
F_THIGHS = [((s * 17, 9, 6), (7, 13, 7.5)) for s in (-1, 1)]
F_ARMS = [((s * 13, -15, 8), (4, 5, 9)) for s in (-1, 1)]
F_HANDS = [((s * 13, -21, 0), (5.5, 7, 3.5)) for s in (-1, 1)]
F_SPOTS = [(-8, 6, 3.4), (7, 11, 3.8), (0, 20, 3.0), (-9, 19, 2.6), (9, -1, 2.8), (-2, 0, 2.4)]  # x, y, radius on the back


def _frog_parts(p, scale):
    body = _smin(_ellipsoid(p, F_BODY, scale), _ellipsoid(p, F_HEAD, scale), 6 * scale)
    eyes = np.minimum(*(_ellipsoid(p, e, scale) for e in F_EYES))
    limbs = np.minimum.reduce([_ellipsoid(p, e, scale) for e in F_THIGHS + F_ARMS + F_HANDS])
    return body, eyes, limbs


def frog_mesh(scale: float = 1.0, voxel: float = 0.5):
    def field(p):
        body, eyes, limbs = _frog_parts(p, scale)
        return _smin(_smin(body, eyes, 2 * scale), limbs, 2.5 * scale)

    return _mesh(field, (-27, -30, 0), (27, 27, 36), scale, voxel)


def frog_paint(c: np.ndarray, scale: float = 1.0) -> np.ndarray:
    body, eyes, limbs = _frog_parts(c, scale)
    x, y, z = (c / scale).T
    lab = np.full(len(c), GREEN)
    on_body = (body < limbs) & (body < eyes)
    # Belly and throat, seen from the front.
    lab[on_body & (y < -3) & ((x / 12.5) ** 2 + ((z - 7) / 9.5) ** 2 < 1)] = BLUE
    # Mouth: a thin black smile across the head (1.2 mm).
    lab[on_body & (y < -14) & (np.abs(x) < 12.5) & (np.abs(z - (17.2 + 0.016 * x * x)) < 0.6)] = BLACK
    for s in (-1, 1):
        lab[on_body & (np.hypot(x - s * 2.6, z - 22.5) < 0.9) & (y < -20)] = BLACK  # nostrils
    # Back: black spots, some with a light-blue dot (light on dark).
    top = on_body & (z > 15) & (y > -6)
    for k, (sx, sy, r) in enumerate(F_SPOTS):
        d = np.hypot(x - sx, y - sy)
        lab[top & (d < r)] = BLACK
        if k < 3:
            lab[top & (d < r * 0.4)] = BLUE
    # Eyes: white with a black pupil, looking forward and a little outward.
    for s, e in zip((-1, 1), F_EYES):
        t = _toward(c, e[0], (s * 0.45, -1, 0.3), scale)
        mine = (eyes < body) & (np.sign(x) == s)
        lab[mine & (t > 0.35)] = WHITE
        lab[mine & (t > 0.86)] = BLACK
    return lab


# ------------------------------------------------------------------ duck

D_BODY = ((0, 4, 11), (17, 23, 15))
D_TAIL = ((0, 23, 19), (7, 9, 6))
D_HEAD = ((0, -10, 33), (12.5, 12.5, 12.5))
D_BEAK = ((0, -23, 30.5), (6.5, 6.5, 2.8))
D_WINGS = [((s * 16, 7, 15), (3.5, 11, 7)) for s in (-1, 1)]


def _duck_parts(p, scale):
    body = _smin(_smin(_ellipsoid(p, D_BODY, scale), _ellipsoid(p, D_TAIL, scale), 5 * scale), _ellipsoid(p, D_HEAD, scale), 5 * scale)
    for w in D_WINGS:
        body = _smin(body, _ellipsoid(p, w, scale), 2 * scale)
    return body, _ellipsoid(p, D_BEAK, scale)


def duck_mesh(scale: float = 1.0, voxel: float = 0.5):
    def field(p):
        body, beak = _duck_parts(p, scale)
        return _smin(body, beak, 0.8 * scale)

    return _mesh(field, (-22, -32, 0), (22, 34, 48), scale, voxel)


def duck_paint(c: np.ndarray, scale: float = 1.0) -> np.ndarray:
    body, beak = _duck_parts(c, scale)
    x, y, z = (c / scale).T
    lab = np.full(len(c), YELLOW)
    lab[beak < body + 0.3 * scale] = ORANGE
    on_head = (beak >= body + 0.3 * scale) & (z > 24) & (y < 0)
    for s in (-1, 1):
        lab[on_head & (_toward(c, D_HEAD[0], (s * 0.6, -0.75, 0.3), scale) > 0.965)] = 3  # black eye, ~3.3 mm radius
        lab[on_head & (_toward(c, D_HEAD[0], (s * 0.66, -0.72, 0.42), scale) > 0.9935)] = 4  # white glint
    return lab


def shapes(model: str, scale: float = 1.0, voxel: float = 0.5):
    if model == "frog":
        v, f = frog_mesh(scale, voxel)
        return [("frog", v, f, GREEN, lambda c, n: frog_paint(c, scale))], {}
    v, f = duck_mesh(scale, voxel)
    return [("duck", v, f, YELLOW, lambda c, n: duck_paint(c, scale))], DUCK_RECOLOR
