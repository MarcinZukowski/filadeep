import numpy as np
import pytest
from conftest import cube, make_3mf

from filadeep import depth, threemf, write3mf


@pytest.fixture
def top_painted_cube(tmp_path):
    """20 mm cube, filament 1 by default, top face painted filament 2 (black)."""
    v, f = cube(20.0)
    paint = [None] * 10 + ["8", "8"]
    return make_3mf(tmp_path / "cube.3mf", v, f, paint, ["#FFFFFF", "#000000"], build_tf="1 0 0 0 1 0 0 0 1 100 100 0")


def test_luminance_and_default_depth():
    assert depth.luminance("#000000") == 0
    assert depth.luminance("#FFFFFF") == pytest.approx(1)
    assert depth.default_depth("#000000") == pytest.approx(0.45)
    assert depth.default_depth("#FFFFFF") == pytest.approx(1.6)


def test_cube_top_plug(top_painted_cube, tmp_path):
    proj = threemf.load(top_painted_cube)
    repl = depth.process(proj, {2: 1.0}, voxel=0.25)
    ((key, parts),) = repl.items()
    assert key == ("2", "1")
    by_ext = {p.extruder: p for p in parts}
    assert set(by_ext) == {1, 2}

    def vol(p):
        return depth.to_manifold(p.vertices, p.faces).volume()

    # Top 1 mm slab, minus the 45° wedges along the 4 edges where the sides
    # (core filament) are nearer: 400 - 4 * (0.5 * 1 * 1 * 20) ≈ 360.
    assert vol(by_ext[2]) == pytest.approx(360, rel=0.05)
    assert vol(by_ext[1]) + vol(by_ext[2]) == pytest.approx(8000, rel=1e-3)
    # The plug sits at the top, in the part's own (pre-build-transform) coordinates.
    assert by_ext[2].vertices[:, 2].min() == pytest.approx(19.0, abs=0.15)
    assert by_ext[2].vertices[:, 2].max() == pytest.approx(20.0, abs=1e-4)

    out = tmp_path / "out.3mf"
    write3mf.replace_parts(proj, out, repl)
    back = threemf.load(out)
    (obj,) = back.objects
    assert sorted(obj.part_extruder(p) for p in obj.parts) == [1, 2]
    assert all(p.mesh.painted_count == 0 for p in obj.parts)
    assert np.allclose(obj.transform, proj.objects[0].transform)


def test_unique_rows_matches_numpy():
    rng = np.random.default_rng(1)
    a = rng.integers(-50, 50, size=(5000, 3))
    u, i, inv = np.unique(a, axis=0, return_index=True, return_inverse=True)
    u2, i2, inv2 = depth.unique_rows(a, return_index=True, return_inverse=True)
    assert np.array_equal(u, u2) and np.array_equal(u[inv.ravel()], a) and np.array_equal(u2[inv2], a)
    assert np.array_equal(a[i2], u2)
