"""layerfun command line."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import regions, render, threemf


def cmd_info(args: argparse.Namespace) -> None:
    proj = threemf.load(args.file)
    print(f"{proj.path.name}  ({proj.application or 'unknown app'})")
    print(f"filaments: {', '.join(f'{i + 1}={c}' for i, c in enumerate(proj.filament_colors)) or '?'}")
    total: dict[int, float] = defaultdict(float)
    surfs = regions.surfaces(proj)
    by_part = {(s.object.id, s.part.id): s for s in surfs}
    for obj in proj.objects:
        print(f"\nobject {obj.id} '{obj.name}'  default filament {obj.extruder}")
        for part in obj.parts:
            m = part.mesh
            s = by_part[(obj.id, part.id)]
            print(
                f"  part {part.id} '{part.name}' [{part.subtype}]  filament {obj.part_extruder(part)}  "
                f"{len(m.vertices)} verts, {len(m.faces)} faces, {m.painted_count} painted "
                f"-> {len(s.tris)} sub-triangles"
            )
            for f, a in sorted(s.area_by_filament().items()):
                total[f] += a
                print(f"      filament {f} ({proj.filament_color(f)}): {a:10.1f} mm²")
    print("\nsurface area by filament:")
    whole = sum(total.values()) or 1.0
    for f, a in sorted(total.items()):
        print(f"  {f} {proj.filament_color(f)}  {a:10.1f} mm²  {100 * a / whole:5.1f}%")


def cmd_extract(args: argparse.Namespace) -> None:
    proj = threemf.load(args.file)
    out = Path(args.out) if args.out else Path(args.file).with_suffix("").with_name(Path(args.file).stem + "_regions")
    for p in regions.export(proj, out):
        print(p)


def cmd_render(args: argparse.Namespace) -> None:
    proj = threemf.load(args.file)
    surfs = [s for s in regions.surfaces(proj) if not args.object or args.object.lower() in s.object.name.lower()]
    if not surfs:
        raise SystemExit("no matching objects")
    tris = np.concatenate([s.tris for s in surfs])
    fil = np.concatenate([s.filament for s in surfs])
    palette = np.array([regions.hex_rgb(proj.filament_color(i)) for i in range(int(fil.max()) + 1)], dtype=np.uint8)
    stem = Path(args.file).stem
    for view in args.views.split(","):
        out = Path(args.out or ".") / f"{stem}_{view}.png"
        render.write_png(out, render.render(tris, palette[fil], view=view, size=args.size))
        print(out)


def _parse_depths(spec: str | None) -> dict[int, float]:
    out = {}
    for item in (spec or "").split(","):
        if item.strip():
            k, v = item.split("=")
            out[int(k)] = float(v)
    return out


def cmd_depth(args: argparse.Namespace) -> None:
    from . import depth, write3mf

    proj = threemf.load(args.file)
    out = Path(args.out) if args.out else Path(args.file).with_name(Path(args.file).stem + "-depth.3mf")
    repl = depth.process(proj, _parse_depths(args.depth), core=args.core, voxel=args.voxel, dmin=args.min_depth, dmax=args.max_depth)
    if not repl:
        raise SystemExit("no painted parts found")
    write3mf.replace_parts(proj, out, repl)
    depth.log(f"wrote {out}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="layerfun", description="Tools for painted multi-color 3MF files.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("info", help="summarize objects, parts and painted areas")
    p.add_argument("file")
    p.set_defaults(func=cmd_info)
    p = sub.add_parser("extract", help="export per-filament surface meshes (STL) and a colored preview (PLY)")
    p.add_argument("file")
    p.add_argument("-o", "--out", help="output directory (default: <file>_regions)")
    p.set_defaults(func=cmd_extract)
    p = sub.add_parser("render", help="render painted model to PNG for a quick visual check")
    p.add_argument("file")
    p.add_argument("--views", default="iso,front,back,top", help=f"comma list of: {', '.join(render.VIEWS)}")
    p.add_argument("--object", help="only objects whose name contains this")
    p.add_argument("--size", type=int, default=900)
    p.add_argument("-o", "--out", help="output directory")
    p.set_defaults(func=cmd_render)
    p = sub.add_parser("depth", help="turn paint into solid per-color parts with controlled depth")
    p.add_argument("file")
    p.add_argument("-o", "--out", help="output 3MF (default: <file>-depth.3mf)")
    p.add_argument("--depth", help="per-filament depth overrides in mm, e.g. 3=0.4,1=1.5")
    p.add_argument("--core", type=int, help="filament for the interior (default: the part's filament)")
    p.add_argument("--min-depth", type=float, default=0.45, help="depth for black (default 0.45)")
    p.add_argument("--max-depth", type=float, default=1.6, help="depth for white (default 1.6)")
    p.add_argument("--voxel", type=float, default=0.25, help="grid resolution in mm (default 0.25)")
    p.set_defaults(func=cmd_depth)
    args = ap.parse_args(argv)
    args.func(args)
