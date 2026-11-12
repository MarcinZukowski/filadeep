import random

import numpy as np
import pytest

from filadeep import paint
from filadeep.paint import Node


# Single-leaf strings as they appear in Bambu/Prusa files.
@pytest.mark.parametrize(
    "s,state",
    [("4", 1), ("8", 2), ("0C", 3), ("1C", 4), ("2C", 5), ("EC", 17), ("0FC", 18)],
)
def test_leaf_known_strings(s, state):
    n = paint.decode(s)
    assert n.is_leaf and n.state == state
    assert paint.encode(n) == s


def test_empty_is_unpainted():
    assert paint.decode("").state == 0
    assert paint.encode(Node(state=0)) == ""


def _random_tree(rng, depth=0):
    if depth > 4 or rng.random() < 0.4:
        return Node(state=rng.choice([0, 1, 2, 3, 4, 7, 20]))
    k = rng.randint(1, 3)
    return Node(
        split_sides=k,
        special_side=rng.randint(0, 2),
        children=[_random_tree(rng, depth + 1) for _ in range(k + 1)],
    )


def test_roundtrip_random_trees():
    rng = random.Random(1)
    for _ in range(500):
        t = _random_tree(rng)
        if t.is_leaf and t.state == 0:
            continue
        assert paint.decode(paint.encode(t)) == t


def _area(t):
    a, b, c = (np.array(p) for p in t)
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a))


def _normal(t):
    a, b, c = (np.array(p) for p in t)
    n = np.cross(b - a, c - a)
    return n / np.linalg.norm(n)


def test_split_geometry_preserves_area_and_orientation():
    tri = ((0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (1.0, 3.0, 0.0))
    rng = random.Random(2)
    for _ in range(200):
        t = _random_tree(rng)
        ls = paint.leaves(t, tri)
        assert sum(_area(x) for x, _ in ls) == pytest.approx(_area(tri))
        for x, _ in ls:
            assert np.allclose(_normal(x), _normal(tri))


def test_children_order_matches_reverse_serialization():
    # Split one side into two children: child 0 painted 1, child 1 painted 2.
    t = Node(split_sides=1, special_side=0, children=[Node(state=1), Node(state=2)])
    s = paint.encode(t)
    back = paint.decode(s)
    assert [c.state for c in back.children] == [1, 2]
