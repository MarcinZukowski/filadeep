import json
import zipfile

import numpy as np

RELS = (
    '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Target="/3D/3dmodel.model" Id="rel0" '
    'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>'
)

ROOT = """<?xml version="1.0" encoding="UTF-8"?>
<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
 xmlns:p="http://schemas.microsoft.com/3dmanufacturing/production/2015/06" requiredextensions="p">
 <metadata name="Application">BambuStudio-02.03.00</metadata>
 <resources>
  <object id="2" p:UUID="00000001-0000-0000-0000-000000000000" type="model">
   <components>
    <component p:path="/3D/Objects/object_1.model" objectid="1" p:UUID="00000002-0000-0000-0000-000000000000" transform="{ctf}"/>
   </components>
  </object>
 </resources>
 <build>
  <item objectid="2" transform="{btf}" printable="1"/>
 </build>
</model>"""

OBJ = """<?xml version="1.0" encoding="UTF-8"?>
<model unit="millimeter" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
 xmlns:p="http://schemas.microsoft.com/3dmanufacturing/production/2015/06">
 <resources>
  <object id="1" type="model">
   <mesh>
    <vertices>
{verts}
    </vertices>
    <triangles>
{tris}
    </triangles>
   </mesh>
  </object>
 </resources>
</model>"""

SETTINGS = """<?xml version="1.0" encoding="UTF-8"?>
<config>
  <object id="2">
    <metadata key="name" value="{name}"/>
    <metadata key="extruder" value="{extruder}"/>
    <metadata face_count="{nf}"/>
    <part id="1" subtype="normal_part" uuid="00000003-0000-0000-0000-000000000000">
      <metadata key="name" value="{name}.stl"/>
      <metadata key="matrix" value="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"/>
      <mesh_stat face_count="{nf}" edges_fixed="0" degenerate_facets="0" facets_removed="0" facets_reversed="0" backwards_edges="0"/>
    </part>
  </object>
</config>"""

IDENTITY = "1 0 0 0 1 0 0 0 1 0 0 0"


def make_3mf(path, vertices, faces, paint, colors, name="thing", extruder=1, component_tf=IDENTITY, build_tf=IDENTITY):
    verts = "\n".join(f'     <vertex x="{x}" y="{y}" z="{z}"/>' for x, y, z in np.asarray(vertices).tolist())
    tris = "\n".join(
        f'     <triangle v1="{a}" v2="{b}" v3="{c}"' + (f' paint_color="{p}"' if p else "") + "/>"
        for (a, b, c), p in zip(np.asarray(faces).tolist(), paint)
    )
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("_rels/.rels", RELS)
        z.writestr("3D/3dmodel.model", ROOT.format(ctf=component_tf, btf=build_tf))
        z.writestr("3D/Objects/object_1.model", OBJ.format(verts=verts, tris=tris))
        z.writestr("Metadata/model_settings.config", SETTINGS.format(name=name, extruder=extruder, nf=len(faces)))
        z.writestr("Metadata/project_settings.config", json.dumps({"filament_colour": colors}))
    return path


def cube(size=20.0):
    v = np.array([[x, y, z] for x in (0, size) for y in (0, size) for z in (0, size)], dtype=float)
    # outward-facing triangles, vertex index = 4x + 2y + z
    f = np.array(
        [
            [0, 1, 3], [0, 3, 2],  # x = 0
            [4, 6, 7], [4, 7, 5],  # x = size
            [0, 4, 5], [0, 5, 1],  # y = 0
            [2, 3, 7], [2, 7, 6],  # y = size
            [0, 2, 6], [0, 6, 4],  # z = 0
            [1, 5, 7], [1, 7, 3],  # z = size (top)
        ]
    )
    return v, f
