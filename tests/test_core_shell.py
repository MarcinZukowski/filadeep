import math

import pytest
from conftest import cube, make_3mf

from layerfun import depth, shell, testmodel, threemf, write3mf

COLORS = ["#FFFFFF", "#000000", "#E8C060", "#2850E0"]


@pytest.fixture
def painted_cube(tmp_path):
    """20 mm cube, filament 1, top painted filament 2; filaments 3 and 4 available."""
    v, f = cube(20.0)
    return make_3mf(tmp_path / "cube.3mf", v, f, [None] * 10 + ["8", "8"], COLORS)


def _vol(p):
    return depth.to_manifold(p.vertices, p.faces).volume()


def test_core_uniform_skin(painted_cube, tmp_path):
    proj = threemf.load(painted_cube)
    repl = depth.process(proj, core=3, uniform=1.0, voxel=0.25)
    ((_, parts),) = repl.items()
    by = {p.extruder: p for p in parts}
    assert set(by) == {1, 2, 3}
    assert _vol(by[3]) == pytest.approx(18.0**3, rel=0.02)  # inward offset of a box is a box
    assert sum(_vol(p) for p in parts) == pytest.approx(8000, rel=1e-3)
    out = tmp_path / "out.3mf"
    write3mf.replace_parts(proj, out, repl)
    assert sorted(p.extruder or 0 for p in threemf.load(out).objects[0].parts) == [1, 2, 3]


def test_shell_bakes_paint_and_trims_bottom(painted_cube, tmp_path):
    proj = threemf.load(painted_cube)
    repl = shell.process(proj, filament=4, thickness=1.0, voxel=0.25)
    ((_, parts),) = repl.items()
    *colors, sh = parts
    # Paint under a shell is ignored by the slicer, so it must become solid parts.
    assert sorted(p.extruder for p in colors) == [1, 2]
    assert all(p.paint is None for p in colors)
    assert sum(_vol(p) for p in colors) == pytest.approx(8000, rel=1e-3)
    a, t = 20.0, 1.0
    # Steiner formula for a box offset by t, minus everything below the base.
    full = a**3 + 6 * a * a * t + 3 * math.pi * a * t * t + 4 / 3 * math.pi * t**3
    below = a * a * t + math.pi * a * t * t + 2 / 3 * math.pi * t**3  # base slab + 4 edge quarter-cylinders + 4 corner eighth-spheres
    assert _vol(sh) == pytest.approx(full - below - a**3, rel=0.03)
    assert sh.vertices[:, 2].min() == pytest.approx(0.0, abs=1e-6)
    out = tmp_path / "out.3mf"
    write3mf.replace_parts(proj, out, repl)
    back = threemf.load(out).objects[0]
    assert sorted(back.part_extruder(p) for p in back.parts) == [1, 2, 4]
    assert sum(p.mesh.painted_count for p in back.parts) == 0


def test_testmodel(tmp_path):
    p = testmodel.write(tmp_path / "t.3mf", scale=0.5)
    proj = threemf.load(p)
    assert [o.name for o in proj.objects] == ["cube", "pyramid", "sphere"]
    assert all(o.parts[0].mesh.painted_count > 0 for o in proj.objects)
    for o in proj.objects:
        depth.to_manifold(o.parts[0].mesh.vertices, o.parts[0].mesh.faces)  # watertight


def test_core_with_primer(painted_cube):
    proj = threemf.load(painted_cube)
    repl = depth.process(proj, core=4, uniform=1.0, primer=(3, 0.8), voxel=0.25)
    ((_, parts),) = repl.items()
    by = {p.extruder: p for p in parts}
    assert set(by) == {1, 2, 3, 4}
    assert "primer" in by[3].name
    assert _vol(by[4]) == pytest.approx(16.4**3, rel=0.02)  # core: 20 - 2 * (1.0 + 0.8)
    assert _vol(by[3]) == pytest.approx(18.0**3 - 16.4**3, rel=0.03)  # primer layer
    assert sum(_vol(p) for p in parts) == pytest.approx(8000, rel=1e-3)
