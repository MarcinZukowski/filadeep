"""Build a minimal Bambu-style painted 3MF and read it back."""

import json
import zipfile

import pytest

from layerfun import regions, threemf

ROOT = """<?xml version="1.0" encoding="UTF-8"?>
<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
 xmlns:p="http://schemas.microsoft.com/3dmanufacturing/production/2015/06" requiredextensions="p">
 <metadata name="Application">BambuStudio-02.03.00</metadata>
 <resources>
  <object id="2" type="model">
   <components>
    <component p:path="/3D/Objects/object_1.model" objectid="1" transform="1 0 0 0 1 0 0 0 1 0 0 5"/>
   </components>
  </object>
 </resources>
 <build>
  <item objectid="2" transform="1 0 0 0 1 0 0 0 1 100 100 0" printable="1"/>
 </build>
</model>"""

# Unit-ish tetrahedron: one face painted filament 2, one split half 3 / half unpainted.
OBJ = """<?xml version="1.0" encoding="UTF-8"?>
<model unit="millimeter" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">
 <resources>
  <object id="1" type="model">
   <mesh>
    <vertices>
     <vertex x="0" y="0" z="0"/><vertex x="10" y="0" z="0"/>
     <vertex x="0" y="10" z="0"/><vertex x="0" y="0" z="10"/>
    </vertices>
    <triangles>
     <triangle v1="0" v2="2" v3="1" paint_color="8"/>
     <triangle v1="0" v2="1" v3="3" paint_color="{split}"/>
     <triangle v1="1" v2="2" v3="3"/>
     <triangle v1="0" v2="3" v3="2"/>
    </triangles>
   </mesh>
  </object>
 </resources>
</model>"""

SETTINGS = """<?xml version="1.0" encoding="UTF-8"?>
<config>
  <object id="2">
    <metadata key="name" value="tetra"/>
    <metadata key="extruder" value="1"/>
    <part id="1" subtype="normal_part">
      <metadata key="name" value="tetra.stl"/>
    </part>
  </object>
</config>"""


@pytest.fixture
def tetra(tmp_path):
    from layerfun.paint import Node, encode

    split = encode(Node(split_sides=1, special_side=0, children=[Node(state=3), Node(state=0)]))
    p = tmp_path / "tetra.3mf"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr(
            "_rels/.rels",
            '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Target="/3D/3dmodel.model" Id="rel0" '
            'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>',
        )
        z.writestr("3D/3dmodel.model", ROOT)
        z.writestr("3D/Objects/object_1.model", OBJ.format(split=split))
        z.writestr("Metadata/model_settings.config", SETTINGS)
        z.writestr("Metadata/project_settings.config", json.dumps({"filament_colour": ["#FFFFFF", "#000000", "#00FF00"]}))
    return p


def test_load(tetra):
    proj = threemf.load(tetra)
    assert proj.application.startswith("BambuStudio")
    assert proj.filament_colors == ["#FFFFFF", "#000000", "#00FF00"]
    (obj,) = proj.objects
    assert obj.name == "tetra"
    (part,) = obj.parts
    assert part.mesh.faces.shape == (4, 3)
    assert part.mesh.painted_count == 2


def test_regions(tetra):
    proj = threemf.load(tetra)
    (s,) = regions.surfaces(proj)
    areas = s.area_by_filament()
    assert areas[2] == pytest.approx(50.0)  # bottom face
    assert areas[3] == pytest.approx(25.0)  # half of a side face
    total = sum(areas.values())
    assert total == pytest.approx(3 * 50 + 3**0.5 / 4 * 200)  # 3 right faces + equilateral (side 10√2)
    # Transforms: component z+5, build x/y+100.
    assert s.tris[:, :, 2].min() == pytest.approx(5.0)
    assert s.tris[:, :, 0].min() == pytest.approx(100.0)


def test_export(tetra, tmp_path):
    proj = threemf.load(tetra)
    files = regions.export(proj, tmp_path / "out")
    names = sorted(f.name for f in files)
    assert names == ["filament_1.stl", "filament_2.stl", "filament_3.stl", "preview.ply"]
