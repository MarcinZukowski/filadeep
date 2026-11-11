"""Tiny dependency-free orthographic renderer (point splatting + z-buffer) for previews."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

# view name -> (right, up, toward-viewer) axes in world coords (Z up)
VIEWS = {
    "front": ((1, 0, 0), (0, 0, 1), (0, -1, 0)),
    "back": ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
    "left": ((0, -1, 0), (0, 0, 1), (-1, 0, 0)),
    "right": ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
    "top": ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    "iso": None,
}


def _basis(view: str) -> np.ndarray:
    if view == "iso":
        f = np.array([1.0, -1.0, 0.8])  # toward viewer: front-right-above
        f /= np.linalg.norm(f)
        r = np.cross((0, 0, 1), f)
        r /= np.linalg.norm(r)
        u = np.cross(f, r)
        return np.stack([r, u, f])
    return np.array(VIEWS[view], dtype=float)


def render(tris: np.ndarray, rgb: np.ndarray, view: str = "iso", size: int = 800, bg=(40, 40, 40)) -> np.ndarray:
    """Render triangles (N,3,3) with per-face colors (N,3 uint8) to an (H,W,3) uint8 image."""
    B = _basis(view)
    p = tris.reshape(-1, 3) @ B.T  # (x right, y up, z toward viewer)
    p = p.reshape(-1, 3, 3)
    lo = p.reshape(-1, 3).min(0)
    hi = p.reshape(-1, 3).max(0)
    scale = (size - 20) / max(hi[0] - lo[0], hi[1] - lo[1])
    w = int((hi[0] - lo[0]) * scale) + 20
    h = int((hi[1] - lo[1]) * scale) + 20
    px = (p[..., 0] - lo[0]) * scale + 10
    py = (hi[1] - p[..., 1]) * scale + 10
    pz = p[..., 2]

    # Flat shading: light from viewer, slightly above-left.
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    light = np.array([-0.3, 0.4, 1.0])
    light /= np.linalg.norm(light)
    shade = 0.35 + 0.65 * np.clip(n @ light, 0, 1)
    col = np.clip(rgb.astype(float) * shade[:, None], 0, 255).astype(np.uint8)

    # Barycentric samples per triangle, density from its pixel area.
    area_px = 0.5 * np.abs((px[:, 1] - px[:, 0]) * (py[:, 2] - py[:, 0]) - (px[:, 2] - px[:, 0]) * (py[:, 1] - py[:, 0]))
    k = np.clip(np.ceil(area_px * 6).astype(int) + 2, 3, 4000)
    idx = np.repeat(np.arange(len(p)), k)
    a = np.random.default_rng(0).random((len(idx), 2))
    flip = a.sum(1) > 1
    a[flip] = 1 - a[flip]
    b0, b1 = a[:, 0], a[:, 1]
    b2 = 1 - b0 - b1
    sx = px[idx, 0] * b2 + px[idx, 1] * b0 + px[idx, 2] * b1
    sy = py[idx, 0] * b2 + py[idx, 1] * b0 + py[idx, 2] * b1
    sz = pz[idx, 0] * b2 + pz[idx, 1] * b0 + pz[idx, 2] * b1
    ix = np.clip(sx.astype(int), 0, w - 1)
    iy = np.clip(sy.astype(int), 0, h - 1)
    lin = iy * w + ix

    # z-buffer: for each pixel keep the sample nearest the viewer (max z).
    order = np.lexsort((-sz, lin))
    lin_s = lin[order]
    first = np.ones(len(lin_s), bool)
    first[1:] = lin_s[1:] != lin_s[:-1]
    win = order[first]
    img = np.empty((h * w, 3), np.uint8)
    img[:] = bg
    img[lin[win]] = col[idx[win]]
    return img.reshape(h, w, 3)


def write_png(path: Path, img: np.ndarray) -> None:
    h, w, _ = img.shape
    raw = b"".join(b"\0" + img[y].tobytes() for y in range(h))

    def chunk(t: bytes, d: bytes) -> bytes:
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(raw, 6)))
        f.write(chunk(b"IEND", b""))
