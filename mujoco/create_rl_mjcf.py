"""Generate a training MJCF with primitive collisions and mesh visuals."""

from __future__ import annotations

import argparse
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def stl_bounds(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = path.read_bytes()
    triangle_count = struct.unpack_from("<I", payload, 80)[0]
    expected_size = 84 + triangle_count * 50
    if len(payload) != expected_size:
        vertices = []
        for line in payload.decode("ascii", errors="ignore").splitlines():
            words = line.strip().split()
            if len(words) == 4 and words[0].lower() == "vertex":
                vertices.append([float(value) for value in words[1:]])
        points = np.asarray(vertices, dtype=float)
    else:
        triangle_dtype = np.dtype([
            ("normal", "<f4", 3),
            ("vertices", "<f4", (3, 3)),
            ("attribute", "<u2"),
        ])
        triangles = np.frombuffer(
            payload, dtype=triangle_dtype, offset=84, count=triangle_count
        )
        points = triangles["vertices"].reshape(-1, 3).astype(float)
    return points.min(axis=0), points.max(axis=0)


def build(source: Path, output: Path, meshes: Path) -> None:
    tree = ET.parse(source)
    root = tree.getroot()
    root.set("model", "hexapod_rl")
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("MJCF has no worldbody")

    replaced = 0
    for body in worldbody.findall(".//body"):
        body_name = body.get("name", "")
        collision_geoms = [
            geom for geom in body.findall("geom") if geom.get("contype") == "2"
        ]
        for geom in collision_geoms:
            for attribute in ("mesh", "fromto", "pos", "quat", "euler", "size"):
                geom.attrib.pop(attribute, None)
            if body_name == "base_link":
                geom.set("type", "box")
                geom.set("pos", "0 0 0.01")
                geom.set("size", "0.14 0.14 0.045")
            else:
                mesh_path = meshes / f"{body_name}.STL"
                lower, upper = stl_bounds(mesh_path)
                center = 0.5 * (lower + upper)
                span = upper - lower
                radius = max(0.018, min(0.035, 0.45 * max(span[1], span[2])))
                start_x = lower[0] + radius
                end_x = upper[0] - radius
                geom.set("type", "capsule")
                geom.set(
                    "fromto",
                    f"{start_x:.8g} {center[1]:.8g} {center[2]:.8g} "
                    f"{end_x:.8g} {center[1]:.8g} {center[2]:.8g}",
                )
                geom.set("size", f"{radius:.8g}")
            geom.set("rgba", "0 0 0 0")
            replaced += 1

    ET.indent(tree, space="  ")
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="utf-8", xml_declaration=True)
    print(f"Simplified collision geoms: {replaced}")
    print(f"Saved RL MJCF: {output}")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    workspace = script_dir.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=script_dir / "hexapod_dynamic.xml")
    parser.add_argument("--output", type=Path, default=script_dir / "hexapod_rl.xml")
    parser.add_argument("--meshes", type=Path, default=workspace / "meshes")
    args = parser.parse_args()
    build(args.source.resolve(), args.output.resolve(), args.meshes.resolve())


if __name__ == "__main__":
    main()
