"""Codec for slicer triangle painting (Bambu `paint_color`, Prusa `slic3rpe:mmu_segmentation`).

Each painted mesh triangle carries a hex string encoding a recursive subdivision
tree. Leaves hold a state: 0 = unpainted (use the part's default filament),
n >= 1 = filament n (1-based).

Bitstream layout (per node, depth-first):
  2 bits  number of split sides (0 = leaf, 1..3 = split)
  if split:
    2 bits  special side
    children, serialized in REVERSE order (last child first)
  if leaf:
    2 bits  state, if state < 3
    else    0b11, then (state - 3) as 4-bit nibbles, 0xF meaning "+15, continue"

Bits are packed 4 per hex digit, least-significant bit first, and the hex
string is written last-nibble-first, so we read it from the end.

Geometry of a split (vertices rotated so the special side's vertex is first):
  1 side split:  the edge opposite vertex 0 is halved      -> 2 children
  2 sides split: the edges adjacent to vertex 0 are halved -> 3 children
  3 sides split: all edges halved                          -> 4 children
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Node:
    state: int = 0  # meaningful only for leaves
    split_sides: int = 0
    special_side: int = 0
    children: list["Node"] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return self.split_sides == 0

    def states(self) -> set[int]:
        if self.is_leaf:
            return {self.state}
        out: set[int] = set()
        for c in self.children:
            out |= c.states()
        return out


# --------------------------------------------------------------------- decode


class _BitReader:
    __slots__ = ("bits", "pos")

    def __init__(self, s: str):
        bits: list[int] = []
        for ch in reversed(s.strip()):
            n = int(ch, 16)
            bits.extend(((n >> 0) & 1, (n >> 1) & 1, (n >> 2) & 1, (n >> 3) & 1))
        self.bits = bits
        self.pos = 0

    def read(self, nbits: int) -> int:
        v = 0
        for i in range(nbits):
            # Running off the end reads zeros (tolerates truncated padding).
            if self.pos < len(self.bits):
                v |= self.bits[self.pos] << i
            self.pos += 1
        return v


def _read_node(r: _BitReader) -> Node:
    split = r.read(2)
    if split:
        special = r.read(2)
        rev = [_read_node(r) for _ in range(split + 1)]
        return Node(split_sides=split, special_side=special, children=rev[::-1])
    state = r.read(2)
    if state == 3:
        state = 3
        while True:
            nib = r.read(4)
            state += nib
            if nib != 15:
                break
    return Node(state=state)


def decode(s: str) -> Node:
    """Decode a paint string into a subdivision tree."""
    if not s:
        return Node(state=0)
    return _read_node(_BitReader(s))


# --------------------------------------------------------------------- encode


def _write_node(n: Node, bits: list[int]) -> None:
    def put(v: int, nbits: int) -> None:
        bits.extend((v >> i) & 1 for i in range(nbits))

    put(n.split_sides, 2)
    if n.split_sides:
        put(n.special_side, 2)
        for c in reversed(n.children):
            _write_node(c, bits)
        return
    if n.state < 3:
        put(n.state, 2)
    else:
        put(3, 2)
        rest = n.state - 3
        while rest >= 15:
            put(15, 4)
            rest -= 15
        put(rest, 4)


def encode(n: Node) -> str:
    """Encode a subdivision tree as a paint string ('' for an unpainted leaf)."""
    if n.is_leaf and n.state == 0:
        return ""
    bits: list[int] = []
    _write_node(n, bits)
    while len(bits) % 4:
        bits.append(0)
    digits = []
    for i in range(0, len(bits), 4):
        v = bits[i] | bits[i + 1] << 1 | bits[i + 2] << 2 | bits[i + 3] << 3
        digits.append("0123456789ABCDEF"[v])
    return "".join(reversed(digits))


# ------------------------------------------------------------------- geometry


def _mid(a, b):
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5)


def _split(n: Node, v: tuple) -> list[tuple]:
    """Child triangles (as vertex triples) of a split node, in child order."""
    j = n.special_side
    a, b, c = v[j % 3], v[(j + 1) % 3], v[(j + 2) % 3]
    if n.split_sides == 1:
        m = _mid(b, c)
        return [(a, b, m), (m, c, a)]
    if n.split_sides == 2:
        mab, mac = _mid(a, b), _mid(a, c)
        return [(a, mab, mac), (mab, b, mac), (b, c, mac)]
    mab, mbc, mca = _mid(a, b), _mid(b, c), _mid(c, a)
    return [(a, mab, mca), (mab, b, mbc), (mbc, c, mca), (mab, mbc, mca)]


def leaves(n: Node, tri: tuple) -> list[tuple[tuple, int]]:
    """Flatten a tree over triangle `tri` into [(vertex triple, state), ...]."""
    out: list[tuple[tuple, int]] = []
    stack = [(n, tri)]
    while stack:
        node, t = stack.pop()
        if node.is_leaf:
            out.append((t, node.state))
        else:
            for child, ct in zip(node.children, _split(node, t)):
                stack.append((child, ct))
    return out


def tessellate(vertices: np.ndarray, faces: np.ndarray, paint: list[str | None]):
    """Expand painted faces into sub-triangles.

    Returns (tris, states, src) where tris is (N, 3, 3) float, states is (N,)
    int filament state per sub-triangle (0 = part default), and src is (N,)
    the original face index each sub-triangle came from.
    """
    tris: list = []
    states: list[int] = []
    src: list[int] = []
    for fi, s in enumerate(paint):
        tri = tuple(tuple(vertices[k]) for k in faces[fi])
        if not s:
            tris.append(tri)
            states.append(0)
            src.append(fi)
            continue
        for t, st in leaves(decode(s), tri):
            tris.append(t)
            states.append(st)
            src.append(fi)
    return (
        np.asarray(tris, dtype=np.float64).reshape(-1, 3, 3),
        np.asarray(states, dtype=np.int32),
        np.asarray(src, dtype=np.int64),
    )
