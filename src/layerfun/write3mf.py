"""Write a modified copy of a Bambu/Orca 3MF, replacing parts with new meshes.

We edit the small XML files (root model, model_settings.config) textually so
everything we don't touch stays byte-identical; Bambu's parser is picky about
namespace prefixes, which an XML library round-trip would rename.
"""

from __future__ import annotations

import re
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .threemf import NS_CORE, Project


@dataclass
class NewPart:
    name: str
    extruder: int
    vertices: np.ndarray  # (V,3) in the ORIGINAL part's mesh coordinates
    faces: np.ndarray  # (F,3)


def _mesh_xml(obj_id: int, part: NewPart) -> str:
    v = np.asarray(part.vertices, dtype=np.float64)
    f = np.asarray(part.faces, dtype=np.int64)
    vs = "\n".join(f'     <vertex x="{x:.6g}" y="{y:.6g}" z="{z:.6g}"/>' for x, y, z in v.tolist())
    ts = "\n".join(f'     <triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in f.tolist())
    return (
        f'  <object id="{obj_id}" p:UUID="{uuid.uuid4()}" type="model">\n'
        f"   <mesh>\n    <vertices>\n{vs}\n    </vertices>\n"
        f"    <triangles>\n{ts}\n    </triangles>\n   </mesh>\n  </object>\n"
    )


def _object_file(objects: list[tuple[int, NewPart]]) -> str:
    body = "".join(_mesh_xml(i, p) for i, p in objects)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<model unit="millimeter" xml:lang="en-US" xmlns="{NS_CORE}" '
        'xmlns:BambuStudio="http://schemas.bambulab.com/package/2021" '
        'xmlns:p="http://schemas.microsoft.com/3dmanufacturing/production/2015/06" requiredextensions="p">\n'
        ' <metadata name="BambuStudio:3mfVersion">1</metadata>\n'
        f" <resources>\n{body} </resources>\n <build/>\n</model>\n"
    )


def _attr(tag: str, name: str) -> str | None:
    m = re.search(rf'\s{re.escape(name)}="([^"]*)"', tag)
    return m.group(1) if m else None


def _find_object_block(xml: str, tag: str, obj_id: str) -> tuple[int, int]:
    m = re.search(rf'<{tag} id="{re.escape(obj_id)}"[^>]*>.*?</{tag}>', xml, re.S)
    if not m:
        raise ValueError(f"<{tag} id={obj_id}> not found")
    return m.span()


def replace_parts(project: Project, out_path: str | Path, replacements: dict[tuple[str, str], list[NewPart]]) -> None:
    """replacements: (object id, part id) -> new parts that replace that part."""
    src = project.path
    with zipfile.ZipFile(src) as zin:
        names = zin.namelist()
        files = {n: zin.read(n) for n in names}
        infos = {n: zin.getinfo(n) for n in names}

    root_path = "3D/3dmodel.model"
    root = files[root_path].decode("utf-8")
    settings = files.get("Metadata/model_settings.config", b"").decode("utf-8")

    # Fresh ids above anything used anywhere.
    used = [int(x) for blob in files.values() if isinstance(blob, bytes) for x in re.findall(rb'(?:objectid|id)="(\d+)"', blob)]
    next_id = max(used, default=0) + 1

    for (oid, pid), parts in replacements.items():
        # ---- root model: swap the component for one component per new part
        o0, o1 = _find_object_block(root, "object", oid)
        block = root[o0:o1]
        cm = re.search(rf'<component\b[^>]*\bobjectid="{re.escape(pid)}"[^>]*/>', block)
        if not cm:
            raise ValueError(f"component {pid} of object {oid} not found")
        comp = cm.group(0)
        path = _attr(comp, "p:path")
        if not path:
            raise NotImplementedError("parts stored inline in the root model are not supported yet")
        ids = list(range(next_id, next_id + len(parts)))
        next_id += len(parts)
        new_comps = []
        for i in ids:
            c = re.sub(r'objectid="\d+"', f'objectid="{i}"', comp)
            c = re.sub(r'p:UUID="[^"]*"', f'p:UUID="{uuid.uuid4()}"', c)
            new_comps.append(c)
        block = block.replace(comp, "\n    ".join(new_comps))
        root = root[:o0] + block + root[o1:]

        # ---- object file: rewrite with the new meshes, keeping other objects in it
        objfile = path.lstrip("/")
        old = files[objfile].decode("utf-8")
        keep = [m.group(0) for m in re.finditer(r"<object\b.*?</object>", old, re.S) if _attr(m.group(0).split(">", 1)[0], "id") != pid]
        xml = _object_file(list(zip(ids, parts)))
        if keep:
            xml = xml.replace(" </resources>", "".join("  " + k + "\n" for k in keep) + " </resources>")
        files[objfile] = xml.encode("utf-8")

        # ---- model_settings.config: one <part> per new part, metadata copied
        if settings:
            s0, s1 = _find_object_block(settings, "object", oid)
            sblock = settings[s0:s1]
            pm = re.search(rf'<part id="{re.escape(pid)}"[^>]*>.*?</part>', sblock, re.S)
            if pm:
                tmpl = pm.group(0)
                new_parts = []
                for i, p in zip(ids, parts):
                    t = re.sub(rf'<part id="{re.escape(pid)}"', f'<part id="{i}"', tmpl, count=1)
                    t = re.sub(r'uuid="[^"]*"', f'uuid="{uuid.uuid4()}"', t, count=1)
                    t = re.sub(r'<metadata key="name" value="[^"]*"/>', f'<metadata key="name" value="{p.name}"/>', t, count=1)
                    t = re.sub(r'\s*<metadata key="extruder" value="\d+"/>', "", t)
                    t = t.replace('<metadata key="name"', f'<metadata key="extruder" value="{p.extruder}"/>\n      <metadata key="name"', 1)
                    t = re.sub(r'(<mesh_stat face_count=")\d+(")', rf"\g<1>{len(p.faces)}\g<2>", t)
                    new_parts.append(t)
                sblock = sblock.replace(tmpl, "\n    ".join(new_parts))
                total = sum(len(p.faces) for p in parts)
                sblock = re.sub(r'<metadata face_count="\d+"/>', f'<metadata face_count="{total}"/>', sblock, count=1)
                settings = settings[:s0] + sblock + settings[s1:]

    files[root_path] = root.encode("utf-8")
    if settings:
        files["Metadata/model_settings.config"] = settings.encode("utf-8")

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zout:
        for n in names:
            info = zipfile.ZipInfo(n, date_time=infos[n].date_time)
            info.compress_type = zipfile.ZIP_DEFLATED
            zout.writestr(info, files[n])
